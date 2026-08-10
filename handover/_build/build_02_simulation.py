"""Build 02_Loch_GCMC_Equilibration_and_Production.ipynb."""

from pathlib import Path

from nbtools import code, md, write_nb

OUT = Path(__file__).resolve().parents[1] / "02_Loch_GCMC_Equilibration_and_Production.ipynb"

cells = []

cells.append(md(r"""
# 02 · Loch GCMC — equilibration and production

## What this is

Ludovic's `GCMC_Grand/Equilibration.py` and `Production.py`, with **Loch**
replacing GRAND's water moves. The notebook drives the *real* pipeline helpers
(`scripts/loch_ludovic_common.py`) cell by cell rather than reimplementing them,
so there is exactly one implementation of the physics and this notebook cannot
drift away from what the cluster runs.

| Phase | Ensemble | GCMC | MD | Restraints | Saved boundary |
|---|---|---|---|---|---|
| UVT1 | NVT | 10,000 + 100 × 1,000 attempts | 100 × 5 steps = 1 ps | Cα | `CRY1AN139uvt1.{prmtop,rst7,pdb}` |
| NPT | NPT, 1 bar | none | 1,000,000 steps = 2 ns | **none** | `CRY1AN139npt.*` |
| UVT2 | NVT | 125 × 800 attempts | 125 × 2,000 steps = 0.5 ns | Cα | `CRY1AN139uvt2.*` |
| Production | NVT | 2,500 × 200 attempts | 2,500 × 2,000 steps = 10 ns | Cα | raw DCD + `-production-final.*` |

Total GCMC attempts: 110,000 + 100,000 + 500,000 = **710,000**.
Total dynamics: 1 ps + 2 ns + 0.5 ns + 10 ns = **12.501 ns**.

## The one thing to get right

Every GCMC → physical boundary must call `finalise_sampler_system()` and then
`save_physical_system()`. Skipping either one persists zero-interaction waters
into the next stage, which minimises cleanly and then destroys UVT2. That is the
July 2026 incident; notebook 07 has the evidence.

## Workflow

0. Configuration and run mode
1. Imports — the real pipeline helpers
2. Load and validate the prepared input
3. UVT1 — sphere emptying, minimisation, GCMC, MD
4. The UVT1 → NPT physical handoff
5. NPT — 2 ns, no ghosts, no restraints
6. UVT2 — fresh 45-water buffer
7. The UVT2 → production physical handoff
8. Production — 10 ns, MD-then-GCMC cycle order
9. Cluster submission
10. Timing and throughput actually observed
11. Recap
""".strip()))

cells.append(md(r"""
---
## 0 · Configuration and run mode

**Edit this cell.** `MODE` controls what happens:

| MODE | Behaviour | Requires |
|---|---|---|
| `describe` | print the exact call sequence and commands; run nothing | nothing |
| `smoke` | run every stage at reduced counts — **plumbing only** | CUDA GPU + `cry-loch-babel` |
| `full` | run the canonical Ludovic schedule (~2 h on an RTX 2080 Ti) | CUDA GPU + `cry-loch-babel` |

`smoke` is not a short scientific run. Its trajectory has 3 frames and its water
placement means nothing.
""".strip()))

cells.append(code(r'''
import sys
from pathlib import Path

MODE = "describe"          # "describe" | "smoke" | "full"
assert MODE in ("describe", "smoke", "full")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_1 = Path("/home/moshe/intern_projects/project_1")
SCRIPTS = PROJECT_1 / "scripts"

# Prepared AMBER pair from notebook 01.
PREP_DIR = PROJECT_1 / "preparation"
PRMTOP = PREP_DIR / "CRY1AN139_solvated.prmtop"
INPCRD = PREP_DIR / "CRY1AN139_solvated.inpcrd"

# Output roots for a notebook-driven run.
EQUIL_DIR = Path(f"nb02_{MODE}") / "equilibration"
PROD_DIR = Path(f"nb02_{MODE}") / "production"

PREFIX = "CRY1AN139"
SEED = 20260714            # base seed; stage seeds are SEED + {0,1,2,3}
BATCH_SIZE = 50            # Loch parallel trial width — NOT a move count

# ── Stage counts ──────────────────────────────────────────────────────────────
FULL = dict(
    uvt1_initial_attempts=10_000, uvt1_cycles=100, uvt1_attempts=1_000, uvt1_md_steps=5,
    npt_steps=1_000_000,
    uvt2_cycles=125, uvt2_attempts=800, uvt2_md_steps=2_000,
    prod_cycles=2_500, prod_attempts=200, prod_md_steps=2_000,
)
SMOKE = dict(
    uvt1_initial_attempts=10, uvt1_cycles=2, uvt1_attempts=5, uvt1_md_steps=5,
    npt_steps=500,
    uvt2_cycles=2, uvt2_attempts=5, uvt2_md_steps=100,
    prod_cycles=3, prod_attempts=5, prod_md_steps=100,
)
COUNTS = FULL if MODE in ("describe", "full") else SMOKE

# Report intervals are protocol, not tuning — they set the CSV/frame schedule.
REPORT = dict(uvt1=100, npt=2_500, uvt2=500, prod=500)
if MODE == "smoke":
    REPORT = dict(uvt1=5, npt=100, uvt2=100, prod=100)

print(f"MODE      : {MODE}")
print(f"prmtop    : {PRMTOP}  (exists={PRMTOP.is_file()})")
print(f"inpcrd    : {INPCRD}  (exists={INPCRD.is_file()})")
print(f"equil dir : {EQUIL_DIR}")
print(f"prod dir  : {PROD_DIR}")
print()
for k, v in COUNTS.items():
    print(f"  {k:<24} {v:,}")
'''.strip()))

cells.append(md(r"""
---
## 1 · Imports — the real pipeline helpers

`scripts/loch_ludovic_common.py` is put on `sys.path` and imported. It owns
sampler construction, dynamics construction, the Ludovic nonbonded
configuration, velocity assignment, the CSV writer, and — critically — the two
handoff helpers.

It imports `openmm`, `sire` and `loch` at module scope, so this cell needs the
full `cry-loch-babel` stack. In `describe` mode a failure here is reported and
the notebook continues.
""".strip()))

cells.append(code(r'''
sys.path.insert(0, str(SCRIPTS))

HELPERS = None
IMPORT_ERROR = None
try:
    import openmm
    from openmm import app, unit
    import sire as sr
    import loch_ludovic_common as H

    HELPERS = H
    print(f"sire  {sr.__version__}")
    print(f"openmm {openmm.__version__}")
    import loch
    print(f"loch  {loch.__version__}")
    print(f"\nloaded {H.__file__}")
except Exception as exc:                                      # noqa: BLE001
    IMPORT_ERROR = exc
    print(f"could not import the simulation stack: {type(exc).__name__}: {exc}")
    if MODE != "describe":
        raise
    print("continuing in describe mode — no stage will be executed")

# Everything the pipeline stages use, named once so the cells below read like
# the scripts they reproduce.
if HELPERS is not None:
    ca_restraints            = H.ca_restraints
    make_dynamics            = H.make_dynamics
    make_sampler             = H.make_sampler
    randomise_velocities     = H.randomise_velocities
    image_context            = H.image_context
    run_with_csv_reports     = H.run_with_csv_reports
    CsvStateWriter           = H.CsvStateWriter
    print_sampler            = H.print_sampler
    finalise_sampler_system  = H.finalise_sampler_system
    save_physical_system     = H.save_physical_system
    save_system              = H.save_system
    update_system_from_context = H.update_system_from_context
    validate_physical_water_topology = H.validate_physical_water_topology
    MIN_TOL = H.MINIMIZATION_TOLERANCE_KJ_MOL_NM
    print("\nhelpers bound:")
    for name in ("make_sampler", "make_dynamics", "finalise_sampler_system",
                 "save_physical_system", "save_system"):
        print(f"  {name}")
'''.strip()))

cells.append(md(r"""
### 1b · What `make_sampler` and `make_dynamics` actually set

Reproduced here from source so the numbers are visible without opening the file.

`GCMCSampler(...)`
: `reference="resname LIG"`, `radius="10 A"`, `cutoff_type="pme"`,
  `cutoff="12 A"`, `excess_chemical_potential="-6.09 kcal/mol"`,
  `standard_volume="30.345 A^3"`, `temperature="300 K"`,
  `num_ghost_waters=45`, **`bulk_sampling_probability=0.0`**,
  `platform="cuda"`, `nvcc=<env>/nvcc`, `overwrite=True`.

`system.dynamics(...)`
: `integrator="langevin_middle"`, `temperature="300 K"`, `cutoff_type="pme"`,
  `cutoff="12 A"`, `constraint="h_bonds"`, `timestep="2 fs"`,
  `com_reset_frequency=1`, `platform="cuda"`, `precision="mixed"`,
  `save_frequency=0`, and friction through the property map:
  `map={"friction": 1 / picosecond}`.

Then, on the live context, `configure_ludovic_nonbonded()` applies what GRAND set
in Ludovic's scripts and Sire does not: switching **on** at 1.0 nm, dispersion
correction **off**, Ewald tolerance 5e-4, followed by
`context.reinitialize(preserveState=True)`.

Two traps:

- `bulk_sampling_probability=0.0` — all GCMC moves target the sphere. Raising it
  changes the sampled ensemble, not just efficiency.
- `nvcc` must be the one next to the environment's Python. Loch compiles CUDA
  kernels at runtime; a visible `nvcc` wrapper is not enough if `cicc` is absent.
""".strip()))

cells.append(md("---\n## 2 · Load and validate the prepared input\n\n`validate_physical_water_topology` is called on the way *in*, not only on the way\nout. A stage refuses to start from a topology that already contains\nzero-interaction waters, so a contaminated boundary cannot quietly propagate."))

cells.append(code(r'''
system_in = None
if HELPERS is not None and PRMTOP.is_file() and INPCRD.is_file():
    system_in = sr.load(str(PRMTOP), str(INPCRD))
    validate_physical_water_topology(system_in, label="Loch equilibration input")
    waters = system_in["water"].molecules()
    print(f"  molecules      : {len(system_in.molecules()):,}")
    print(f"  atoms          : {system_in.num_atoms():,}")
    print(f"  water molecules: {len(waters):,}")
    print(f"  ligand atoms   : {system_in['resname LIG'].num_atoms()}")
    print(f"  C-alpha atoms  : {len(ca_restraints(system_in))}")
    print("\n  validate_physical_water_topology: PASSED "
          "(zero waters with simultaneously zero charge and zero LJ epsilon)")
else:
    print("input not loaded — describe mode or missing prepared files")
    print(f"  PRMTOP exists: {PRMTOP.is_file()}")
    print(f"  INPCRD exists: {INPCRD.is_file()}")
'''.strip()))

cells.append(md(r"""
---
## 3 · UVT1 — empty the sphere, minimise, then sample

Executed order, and it matters:

1. append 45 ghost waters (inside `make_sampler`)
2. `sampler.delete_waters(ctx)` — remove every water inside the 10 Å sphere
3. `LocalEnergyMinimizer.minimize(ctx, tol=10 kJ/mol/nm, maxIterations=0)`
4. **assign velocities** — after minimisation, not before
5. one standalone block of 10,000 GCMC attempts (advances no MD time)
6. 100 × (1,000 attempts → ghost report → 5 MD steps)

Step 4 is a deliberate departure from Ludovic. The raw AMBER coordinates violate
the h-bond constraints; assigning constrained velocities to that state produces
an enormous kinetic energy that the native-AMBER/Sire construction did not
survive. Assigning after minimisation fixes it and changes no stage length or
thermodynamic parameter.

Step 5's 10,000 attempts are what refill the emptied sphere. They are reported as
one Loch call; the diagnostic wrapper can chunk them 1,000 at a time for
localisation without changing the total.
""".strip()))

cells.append(code(r'''
# ── UVT1 ──────────────────────────────────────────────────────────────────────
UVT1_CALLS = f"""
uvt1 = make_sampler(original,
                    attempts={COUNTS['uvt1_attempts']},
                    batch_size={BATCH_SIZE},
                    seed={SEED},
                    log_file=<equil>/{PREFIX}_equilibration_uvt1.log,
                    ghost_file=<equil>/{PREFIX}_equilibration_uvt1_ghosts.txt)
system1 = uvt1.system()                       # 45 ghosts appended
dyn1    = make_dynamics(system1, restraints=ca_restraints(system1))
uvt1.bind_dynamics(dyn1)
ctx1    = dyn1.context()

uvt1.delete_waters(ctx1)                      # empty the 10 A sphere
openmm.LocalEnergyMinimizer.minimize(ctx1, 10.0 kJ/mol/nm, maxIterations=0)
randomise_velocities(ctx1, {SEED})            # AFTER minimisation

uvt1._num_attempts = {COUNTS['uvt1_initial_attempts']}   # standalone block
uvt1.move(ctx1)
uvt1._num_attempts = {COUNTS['uvt1_attempts']}

for cycle in range({COUNTS['uvt1_cycles']}):
    uvt1.move(ctx1)                           # {COUNTS['uvt1_attempts']} attempts
    uvt1.write_ghost_residues()               # one line per cycle
    run_with_csv_reports(dyn1, ctx1, {COUNTS['uvt1_md_steps']}, ...,
                         report_interval={REPORT['uvt1']}, csv1)
"""
print(UVT1_CALLS)

uvt1 = ctx1 = dyn1 = None
if MODE != "describe" and HELPERS is not None and system_in is not None:
    import time
    EQUIL_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    uvt1 = make_sampler(
        system_in,
        attempts=COUNTS["uvt1_attempts"],
        batch_size=BATCH_SIZE,
        seed=SEED,
        log_file=EQUIL_DIR / f"{PREFIX}_equilibration_uvt1.log",
        ghost_file=EQUIL_DIR / f"{PREFIX}_equilibration_uvt1_ghosts.txt",
    )
    system1 = uvt1.system()
    dyn1 = make_dynamics(system1, restraints=ca_restraints(system1))
    uvt1.bind_dynamics(dyn1)
    ctx1 = dyn1.context()

    uvt1.delete_waters(ctx1)
    openmm.LocalEnergyMinimizer.minimize(
        ctx1, MIN_TOL * unit.kilojoule_per_mole / unit.nanometer, 0)
    randomise_velocities(ctx1, SEED)

    if COUNTS["uvt1_initial_attempts"]:
        keep = uvt1._num_attempts
        uvt1._num_attempts = COUNTS["uvt1_initial_attempts"]
        uvt1.move(ctx1)
        uvt1._num_attempts = keep

    csv1 = CsvStateWriter(EQUIL_DIR / f"{PREFIX}_data_uvt1.csv", ctx1)
    done = 0
    for cycle in range(COUNTS["uvt1_cycles"]):
        uvt1.move(ctx1)
        uvt1.write_ghost_residues()
        done = run_with_csv_reports(dyn1, ctx1, COUNTS["uvt1_md_steps"],
                                    done, REPORT["uvt1"], csv1)
        print_sampler(f"UVT1 {cycle + 1}/{COUNTS['uvt1_cycles']}", uvt1)
    csv1.close()
    print(f"\nUVT1 elapsed: {time.time() - t0:.1f} s")
'''.strip()))

cells.append(md(r"""
---
## 4 · The UVT1 → NPT physical handoff

The two-call contract, in order. This is the single most consequential piece of
code in the pipeline.

```python
image_context(ctx1)                              # enforce periodic box first
no_ghosts1 = finalise_sampler_system(uvt1, ctx1)  # topology surgery
save_physical_system(no_ghosts1, <equil>/CRY1AN139uvt1)
```

`finalise_sampler_system` does five things and refuses the handoff if any fails:

1. copy live imaged positions and the live periodic box into a Sire clone;
2. remove every logically inactive water/ghost **by stable molecule identity** —
   captured before any deletion, because each removal renumbers Sire atoms;
3. require the remaining zero-interaction-water count to equal Loch's logically
   active appended-buffer count (this is the check the first implementation
   lacked, and why UVT1 job 61723 looked fine but wasn't);
4. give exactly those retained waters the physical TIP3P charge/σ/ε from
   `sampler._water_template`;
5. validate the committed values before returning.

`save_physical_system` then writes AMBER + PDB, **reloads the AMBER pair from
disk**, and requires the same physical-water count with zero all-zero
charge/LJ waters. In-memory validation alone is not sufficient — the failure mode
was specifically that the saved topology disagreed with the live context.
""".strip()))

cells.append(code(r'''
# ── UVT1 -> NPT handoff, plus the arithmetic that must hold ──────────────────
def handoff_arithmetic(input_waters, num_ghosts, final_state_zero, saved_waters):
    expected = input_waters + num_ghosts - final_state_zero
    return expected, saved_waters == expected


no_ghosts1 = None
if MODE != "describe" and uvt1 is not None:
    n_in = len(system_in["water"].molecules())
    state = uvt1.water_state()
    n_state_zero = sum(1 for s in state if s == 0)

    image_context(ctx1)
    no_ghosts1 = finalise_sampler_system(uvt1, ctx1)
    paths1 = save_physical_system(no_ghosts1, EQUIL_DIR / f"{PREFIX}uvt1")

    n_saved = len(no_ghosts1["water"].molecules())
    expected, ok = handoff_arithmetic(n_in, 45, n_state_zero, n_saved)
    print(f"  input physical waters : {n_in:,}")
    print(f"  ghost buffer          : 45")
    print(f"  final state-0 count   : {n_state_zero}")
    print(f"  expected saved waters : {expected:,}")
    print(f"  actual saved waters   : {n_saved:,}   -> {'OK' if ok else 'MISMATCH'}")
    print(f"  net change            : {n_saved - n_in:+d}")
    assert ok, "UVT1 handoff arithmetic failed"
    for k, v in paths1.items():
        print(f"  {k:<8} {v}")
else:
    print("""
  Reference values from the known-good July 2026 full run:

      input physical waters : 18,800
      final state-0 count   : 36
      expected/actual saved : 18,809      net +9
      all-zero waters       : 0

  Apply the same arithmetic INDEPENDENTLY to UVT2, because UVT2 opens a fresh
  45-water buffer. Checking only one of the two boundaries cannot distinguish a
  discarded accepted water from a leaked inactive ghost.
""")
'''.strip()))

cells.append(md(r"""
---
## 5 · NPT — 2 ns, no ghosts, no restraints

Rebuilt from the *saved* UVT1 handoff, not from the live sampler context. There
is no sampler and there are **no Cα restraints** — this is the stage that lets
the box relax to the correct density.

Ludovic's comment says 1 ns; the executed statement is 1,000,000 × 2 fs = **2 ns**.
Barostat every 25 steps at 1 bar; CSV every 2,500 steps → 400 records.

The NPT trace is the *only* place to judge physical density. The UVT1/UVT2 CSV
density column includes the mass of the 45 appended buffer waters while a sampler
is live, so it is systematically wrong for that purpose.
""".strip()))

cells.append(code(r'''
# ── NPT ───────────────────────────────────────────────────────────────────────
print(f"""
npt     = make_dynamics(no_ghosts1, pressure="1 bar", barostat_frequency=25)
npt_ctx = npt.context()
randomise_velocities(npt_ctx, {SEED} + 1)

completed = 0
while completed < {COUNTS['npt_steps']}:
    chunk = min({REPORT['npt']}, {COUNTS['npt_steps']} - completed)
    completed = run_with_csv_reports(npt, npt_ctx, chunk, completed,
                                     {REPORT['npt']}, npt_csv)

image_context(npt_ctx)
npt_system = npt.commit(return_as_system=True)
npt_system = update_system_from_context(npt_system, npt_ctx)   # imaged coords + live box
save_physical_system(npt_system, <equil>/{PREFIX}npt)
""")

npt_system = None
if MODE != "describe" and no_ghosts1 is not None:
    import time
    t0 = time.time()
    npt = make_dynamics(no_ghosts1, pressure="1 bar", barostat_frequency=25)
    npt_ctx = npt.context()
    randomise_velocities(npt_ctx, SEED + 1)
    npt_csv = CsvStateWriter(EQUIL_DIR / f"{PREFIX}_data_npt.csv", npt_ctx)
    completed = 0
    while completed < COUNTS["npt_steps"]:
        chunk = min(REPORT["npt"], COUNTS["npt_steps"] - completed)
        completed = run_with_csv_reports(npt, npt_ctx, chunk, completed,
                                         REPORT["npt"], npt_csv)
        print(f"NPT {completed:,}/{COUNTS['npt_steps']:,}")
    npt_csv.close()
    image_context(npt_ctx)
    npt_system = npt.commit(return_as_system=True)
    npt_system = update_system_from_context(npt_system, npt_ctx)
    save_physical_system(npt_system, EQUIL_DIR / f"{PREFIX}npt")
    print(f"\nNPT elapsed: {time.time() - t0:.1f} s")
    print(f"waters: {len(npt_system['water'].molecules()):,}  "
          "(must equal the UVT1 handoff count — NPT adds and removes nothing)")
else:
    print("""
  Reference NPT trace, known-good July 2026 full run:
      temperature  300.29 +/- 1.15 K
      density      1.0185 +/- 0.0024 g/mL
      volume       settled near 642 nm^3 (from 718 nm^3 at UVT1)
      records      400, at steps 2,500 .. 1,000,000
      waters       18,809 — identical to the UVT1 handoff
""")
'''.strip()))

cells.append(md(r"""
---
## 6 · UVT2 — a fresh 45-water buffer

A **new** sampler over the NPT system, which appends **another** 45 ghosts. The
UVT1 buffer is gone; its accepted members are now ordinary physical waters. This
matches `grand.utils.add_ghosts()` being called once per stage in Ludovic's
scripts.

New Cα restraint anchors are created from the relaxed NPT coordinates — not
carried over from UVT1, whose box was 12% larger.

125 × (800 attempts → ghost report → 2,000 MD steps) = 100,000 attempts and
0.5 ns. No standalone initial GCMC block, unlike UVT1.
""".strip()))

cells.append(code(r'''
# ── UVT2 ──────────────────────────────────────────────────────────────────────
print(f"""
uvt2 = make_sampler(npt_system,
                    attempts={COUNTS['uvt2_attempts']},
                    batch_size={BATCH_SIZE},
                    seed={SEED} + 2,
                    log_file=<equil>/{PREFIX}_equilibration_uvt2.log,
                    ghost_file=<equil>/{PREFIX}_equilibration_uvt2_ghosts.txt)
system2 = uvt2.system()                       # a FRESH 45 ghosts
dyn2    = make_dynamics(system2, restraints=ca_restraints(system2))  # fresh anchors
uvt2.bind_dynamics(dyn2)
ctx2    = dyn2.context()
randomise_velocities(ctx2, {SEED} + 2)

for cycle in range({COUNTS['uvt2_cycles']}):
    uvt2.move(ctx2)                           # {COUNTS['uvt2_attempts']} attempts
    uvt2.write_ghost_residues()
    run_with_csv_reports(dyn2, ctx2, {COUNTS['uvt2_md_steps']}, ...,
                         report_interval={REPORT['uvt2']}, csv2)
""")

uvt2 = ctx2 = equilibrated = None
if MODE != "describe" and npt_system is not None:
    import time
    t0 = time.time()
    uvt2 = make_sampler(
        npt_system,
        attempts=COUNTS["uvt2_attempts"],
        batch_size=BATCH_SIZE,
        seed=SEED + 2,
        log_file=EQUIL_DIR / f"{PREFIX}_equilibration_uvt2.log",
        ghost_file=EQUIL_DIR / f"{PREFIX}_equilibration_uvt2_ghosts.txt",
    )
    system2 = uvt2.system()
    dyn2 = make_dynamics(system2, restraints=ca_restraints(system2))
    uvt2.bind_dynamics(dyn2)
    ctx2 = dyn2.context()
    randomise_velocities(ctx2, SEED + 2)
    csv2 = CsvStateWriter(EQUIL_DIR / f"{PREFIX}_data_uvt2.csv", ctx2)
    done = 0
    for cycle in range(COUNTS["uvt2_cycles"]):
        uvt2.move(ctx2)
        uvt2.write_ghost_residues()
        done = run_with_csv_reports(dyn2, ctx2, COUNTS["uvt2_md_steps"],
                                    done, REPORT["uvt2"], csv2)
        print_sampler(f"UVT2 {cycle + 1}/{COUNTS['uvt2_cycles']}", uvt2)
    csv2.close()
    print(f"\nUVT2 elapsed: {time.time() - t0:.1f} s")
'''.strip()))

cells.append(md("---\n## 7 · The UVT2 → production physical handoff\n\nIdentical contract to §4, applied independently. This is the boundary whose\nfailure produced the UVT2 incident, because it is the first stage that *restores*\nnormal parameters to a water NPT had been propagating with none."))

cells.append(code(r'''
if MODE != "describe" and uvt2 is not None:
    n_in2 = len(npt_system["water"].molecules())
    state2 = uvt2.water_state()
    n_state_zero2 = sum(1 for s in state2 if s == 0)

    image_context(ctx2)
    equilibrated = finalise_sampler_system(uvt2, ctx2)
    paths2 = save_physical_system(equilibrated, EQUIL_DIR / f"{PREFIX}uvt2")

    n_saved2 = len(equilibrated["water"].molecules())
    expected2, ok2 = handoff_arithmetic(n_in2, 45, n_state_zero2, n_saved2)
    print(f"  input physical waters : {n_in2:,}")
    print(f"  final state-0 count   : {n_state_zero2}")
    print(f"  expected/actual saved : {expected2:,} / {n_saved2:,}  "
          f"-> {'OK' if ok2 else 'MISMATCH'}")
    print(f"  net change            : {n_saved2 - n_in2:+d}")
    assert ok2, "UVT2 handoff arithmetic failed"
    print("\n  production consumes ONLY this AMBER pair:")
    for k in ("prmtop", "rst7"):
        print(f"      {paths2[k]}")
else:
    print("""
  Reference values from the known-good July 2026 full run:

      atoms                 64,127
      input physical waters 18,809
      final state-0 count   48
      expected/actual saved 18,806      net -3
      all-zero waters       0
      UVT2 trace            300.24 +/- 1.22 K, finite through 250,000 MD steps

  Intermediate UVT1 and NPT AMBER files are DIAGNOSTIC stage boundaries.
  Production consumes only the UVT2 pair.
""")
'''.strip()))

cells.append(md(r"""
---
## 8 · Production — 10 ns, MD **then** GCMC

The cycle order follows Ludovic's code, not his comment:

```
for cycle in range(2500):
    run 2000 MD steps              (CSV every 500 steps)
    sampler.move(context)          200 GCMC attempts
    dcd.writeModel(state)          <- frame written AFTER the move
    sampler.write_ghost_residues() <- ghost line written AFTER the move
```

So DCD frame *i* and ghost line *i* describe the **same post-move state**. A
postprocessor that pairs them the other way round mislabels every frame's
inactive set, and nothing errors.

Two topologies are saved, and confusing them is a real hazard:

| Artefact | Contains | Use |
|---|---|---|
| `CRY1AN139-loch-ghosts.{prmtop,rst7,pdb}` | 45 inactive ghosts, via **`save_system`** | interpreting the raw DCD only |
| `CRY1AN139-production-final.*` | ghost-free, via `finalise` + `save_physical_system` | restarts, FEP bound frames |

Never feed the ghost-containing topology to a fresh sampler or to NPT.
""".strip()))

cells.append(code(r'''
# ── Production ────────────────────────────────────────────────────────────────
print(f"""
sampler = make_sampler(equilibrated,
                       attempts={COUNTS['prod_attempts']},
                       batch_size={BATCH_SIZE},
                       seed={SEED} + 3,
                       log_file=<prod>/{PREFIX}-gcmc.log,
                       ghost_file=<prod>/{PREFIX}-gcmc-ghosts.txt)
gcmc_system = sampler.system()                       # a FRESH 45 ghosts
save_system(gcmc_system, <prod>/{PREFIX}-loch-ghosts)   # GENERIC save, on purpose
topology = app.AmberPrmtopFile(...).topology

dynamics = make_dynamics(gcmc_system, restraints=ca_restraints(gcmc_system))
sampler.bind_dynamics(dynamics)
context = dynamics.context()
randomise_velocities(context, {SEED} + 3)

dcd = app.DCDFile(handle, topology, 2 fs, firstStep=0, interval={COUNTS['prod_md_steps']})
for cycle in range({COUNTS['prod_cycles']}):
    run_with_csv_reports(dynamics, context, {COUNTS['prod_md_steps']}, ...,
                         {REPORT['prod']}, csv)     # MD FIRST
    sampler.move(context)                            # then {COUNTS['prod_attempts']} attempts
    state = context.getState(getPositions=True, getEnergy=True, enforcePeriodicBox=True)
    dcd.writeModel(state.getPositions(), periodicBoxVectors=state.getPeriodicBoxVectors())
    sampler.write_ghost_residues()

final_system = finalise_sampler_system(sampler, context)
save_physical_system(final_system, <prod>/{PREFIX}-production-final)
""")

if MODE != "describe" and equilibrated is not None:
    import time
    PROD_DIR.mkdir(parents=True, exist_ok=True)
    validate_physical_water_topology(equilibrated, label="Loch production input")

    sampler = make_sampler(
        equilibrated,
        attempts=COUNTS["prod_attempts"],
        batch_size=BATCH_SIZE,
        seed=SEED + 3,
        log_file=PROD_DIR / f"{PREFIX}-gcmc.log",
        ghost_file=PROD_DIR / f"{PREFIX}-gcmc-ghosts.txt",
    )
    gcmc_system = sampler.system()
    topology_paths = save_system(gcmc_system, PROD_DIR / f"{PREFIX}-loch-ghosts")
    topology = app.AmberPrmtopFile(topology_paths["prmtop"]).topology

    dynamics = make_dynamics(gcmc_system, restraints=ca_restraints(gcmc_system))
    sampler.bind_dynamics(dynamics)
    context = dynamics.context()
    randomise_velocities(context, SEED + 3)
    csv = CsvStateWriter(PROD_DIR / f"{PREFIX}_data_prod.csv", context)

    handle = (PROD_DIR / f"{PREFIX}-raw.dcd").open("wb")
    dcd = app.DCDFile(handle, topology, 2.0 * unit.femtoseconds,
                      firstStep=0, interval=COUNTS["prod_md_steps"])
    t0, done = time.time(), 0
    for cycle in range(COUNTS["prod_cycles"]):
        md_t = time.time()
        done = run_with_csv_reports(dynamics, context, COUNTS["prod_md_steps"],
                                    done, REPORT["prod"], csv)
        md_s = time.time() - md_t
        mv_t = time.time()
        sampler.move(context)
        mv_s = time.time() - mv_t
        st = context.getState(getPositions=True, getEnergy=True, enforcePeriodicBox=True)
        dcd.writeModel(st.getPositions(), periodicBoxVectors=st.getPeriodicBoxVectors())
        sampler.write_ghost_residues()
        print_sampler(f"Production {cycle + 1}/{COUNTS['prod_cycles']}", sampler)
        print(f"MD={md_s:.2f}s Loch={mv_s:.2f}s")
    csv.close()
    handle.close()
    final_system = finalise_sampler_system(sampler, context)
    save_physical_system(final_system, PROD_DIR / f"{PREFIX}-production-final")
    print(f"\nProduction elapsed: {time.time() - t0:.1f} s")
'''.strip()))

cells.append(md("---\n## 9 · Cluster submission\n\nThis is how the work was actually run. The notebook path above exists so the\nbehaviour is legible and testable; the Slurm path is what produced every result\nin notebooks 03–05."))

cells.append(code(r'''
SUBMIT = r"""
# ── Single replica, end to end ────────────────────────────────────────────────
cd "$HOME/cry"
export SOURCE_PDB="$HOME/cry/ludovic-workflow-gcmc/Other/MD/CRY1AN139_HOLO.pdb"
sbatch --export=ALL,REPLICA=1,SEED=20260714 scripts/loch_full_pipeline.slurm

# ── N independent replicas, seed blocks spaced by 1000, no inter-job deps ──────
export SOURCE_PDB="$HOME/cry/ludovic-workflow-gcmc/Other/MD/CRY1AN139_HOLO.pdb"
./scripts/submit_loch_replicas.sh 12          # jobs cry-loch-r1 .. cry-loch-r12

# Defaults: project root $HOME/cry, results under $HOME/cry/cry-loch-full/repN,
# environment cry-loch-babel. Override the output root with RUN_ROOT.
# The wrapper REFUSES to overwrite an existing replica directory.

# ── Stage-by-stage, from already prepared AMBER files ─────────────────────────
sbatch scripts/loch_prepare.slurm             # preparation only
sbatch scripts/loch_replica.slurm             # equilibration + production only
sbatch scripts/loch_smoke.slurm               # one-cycle plumbing check

# ── Most defensive wrapper: separate prep / Loch / post environments ──────────
sbatch scripts/loch_multi_env_full.slurm      # cry-prep then cry-loch-babel

# ── Direct stage invocation (what the wrapper does) ───────────────────────────
python -u scripts/LochEquilibration.py \
    --prmtop <run>/preparation/CRY1AN139_solvated.prmtop \
    --rst7   <run>/preparation/CRY1AN139_solvated.inpcrd \
    --output-dir <run>/equilibration --seed 20260714 --batch-size 50

python -u scripts/LochProduction.py \
    --prmtop <run>/equilibration/CRY1AN139uvt2.prmtop \
    --rst7   <run>/equilibration/CRY1AN139uvt2.rst7 \
    --output-dir <run>/production --seed 20260717 --batch-size 50
"""
print(SUBMIT)

print("""
DO NOT SUBSTITUTE these — they are different experiments, not alternatives:

  scripts/Equilibration.py + scripts/Production.py
      a locally rewritten GRAND workflow over prepared/complex_ghosts.pdb,
      run in the cry-gcmc environment.
  scripts/ludovic_replica.slurm
      copies and runs Ludovic's ORIGINAL external GRAND scripts (cry-gcmc).
  loch_cry_smoke*.py, export_cry_for_loch.py
      pre-Ludovic development history over the old OpenFF/GRAND-prepared input
      family. They create no physical stage handoffs.
""")
'''.strip()))

cells.append(md("---\n## 10 · Timing and throughput actually observed\n\nMeasured on the completed CRY1 replicas retained in `cry-loch-multi/`\n(RTX 2080 Ti, one GPU, 16 CPU threads)."))

cells.append(code(r'''
# ── Wall times from the retained replica timing files ────────────────────────
import re

MULTI = PROJECT_1 / "cry-loch-multi"
rows = []
for d in sorted(MULTI.glob("rep*")):
    tf = list(d.glob("timing_replica*.txt"))
    if not tf:
        continue
    kv = dict(l.strip().split("=", 1) for l in tf[0].read_text().splitlines() if "=" in l)
    rows.append((
        d.name,
        int(kv.get("PREPARATION_WALL_SECONDS", 0)),
        int(kv.get("EQUILIBRATION_WALL_SECONDS", 0)),
        int(kv.get("PRODUCTION_WALL_SECONDS", 0)),
        int(kv.get("POSTPROCESSING_WALL_SECONDS", 0)),
        int(kv.get("TOTAL_WALL_SECONDS", 0)),
    ))

if rows:
    W = 84
    print("╔" + "═" * W + "╗")
    print("║ " + f"{'WALL TIME (seconds) — completed CRY1 replicas':^{W - 1}}"[:W - 1] + "║")
    print("╠" + "═" * W + "╣")
    hdr = f"{'replica':<9}{'prep':>9}{'equil':>9}{'production':>12}{'postproc':>10}{'total':>9}{'total h':>9}"
    print("║ " + hdr.ljust(W - 1) + "║")
    print("╟" + "─" * W + "╢")
    for name, p, e, pr, po, t in rows:
        line = f"{name:<9}{p:>9,}{e:>9,}{pr:>12,}{po:>10,}{t:>9,}{t / 3600:>9.2f}"
        print("║ " + line.ljust(W - 1) + "║")
    print("╟" + "─" * W + "╢")
    n = len(rows)
    means = [sum(r[i] for r in rows) / n for i in range(1, 6)]
    line = (f"{'mean':<9}{means[0]:>9,.0f}{means[1]:>9,.0f}{means[2]:>12,.0f}"
            f"{means[3]:>10,.0f}{means[4]:>9,.0f}{means[4] / 3600:>9.2f}")
    print("║ " + line.ljust(W - 1) + "║")
    print("╚" + "═" * W + "╝")
    print(f"\n  production throughput: 10 ns in ~{means[2] / 3600:.2f} h "
          f"= {10 * 86400 / means[2]:.0f} ns/day including all GCMC")
    print(f"  whole pipeline       : ~{means[4] / 3600:.2f} h per replica")
else:
    print(f"no timing files under {MULTI}")
'''.strip()))

cells.append(md(r"""
---
## 11 · Recap

**Per-stage artefacts.** Each replica directory holds `preparation/`,
`equilibration/` (UVT1, NPT, UVT2 boundaries), `production/` (raw DCD, ghost
history, state CSV, both final topologies), `postprocessing/`, and
`timing_replicaN.txt`. Console output goes to the Slurm `.out`/`.err`.

**Checkpointing.** Every stage writes a completion marker whose signature hashes
the implementation/protocol identity, the chemistry-tool and data provenance, and
the SHA-256 of *every* required output — including parameterization logs, the CSV
and the ghost history, not just the AMBER/DCD files. Hashes are relative to the
marker's own stage directory, so a copied marker cannot validate artefacts
sitting somewhere else. The old marker is invalidated *before* recomputation.

**Resume semantics.** Reissuing a job revalidates markers and restarts at the
earliest invalid stage. The aggregate audit/timing summary is cleared at the
start of every invocation and republished only after the final independent audit
passes, so a stale `completed` cannot survive a failed rerun.

**Five things that will bite.**

1. NPT built from a pre-fix (contaminated) UVT1 output is unusable and cannot be
   salvaged — zero-interaction waters may already have drifted into overlaps.
2. `save_system` is for the trajectory topology only. Everywhere else,
   `finalise_sampler_system` then `save_physical_system`.
3. Cα restraint anchors are re-created per stage from that stage's coordinates.
4. UVT1's CSV covers only 1 ps, so a sub-300 K temperature there is expected.
5. `profile=smoke` validates plumbing. Do not label a stride-subsampled
   clustering "full" even when the simulation schedule was.

**Next:** notebook 03 audits these boundaries against the retained reference run.
""".strip()))

write_nb(OUT, cells)
print(f"wrote {OUT} ({len(cells)} cells)")
