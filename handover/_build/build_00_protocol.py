"""Build 00_Protocol_and_Provenance.ipynb."""

from pathlib import Path

from nbtools import code, md, write_nb

OUT = Path(__file__).resolve().parents[1] / "00_Protocol_and_Provenance.ipynb"

cells = []

cells.append(md(r"""
# 00 · Executed protocol and provenance

## System overview

| Component | Identifier | Notes |
|---|---|---|
| Reference system | CRY1 · AN139 | mouse cryptochrome-1 holo complex, residue `LIG` |
| Ported system | EV-A71 2A protease · 32 pyrrolidine-thiopyrimidines | OpenBind release, one chain, 139 residues |
| Water sampling | **Loch** GCMC (2025.2) | replaces `grand` in Ludovic's original scripts |
| Dynamics | Sire 2025.4 / OpenMM 8.4, `LangevinMiddleIntegrator` | Ludovic used `openmmtools.BAOABIntegrator` |
| Environment | `cry-loch-babel` | CUDA 12.8-capped toolchain |

## What this notebook is

Every other notebook in this set depends on one thing: the **exact executed
protocol**. This notebook is that statement, and it is derived from source rather
than retyped — it parses the module-level constants out of
`scripts/loch_ludovic_common.py`, reconstructs the whole stage schedule from
them, and asserts the derived totals against the documented values.

If someone changes a constant in the pipeline, this notebook fails. That is the
point of it.

## Workflow

1. Configuration
2. Imports
3. Read protocol constants from source (AST — no GPU, no imports)
4. Derived stage schedule and totals
5. Ghost lifecycle and the physical-handoff contract
6. Ludovic ↔ Loch parity, and the differences that are unavoidable
7. Seeds and replica seed blocks
8. Environment fingerprint
9. Recap
""".strip()))

cells.append(md("---\n## 0 · Configuration\n\n**Edit the paths in this cell before running the notebook.**"))

cells.append(code(r'''
from pathlib import Path

# ── Repository roots ──────────────────────────────────────────────────────────
# PROJECT_1 holds the CRY1-AN139 reference pipeline; PROJECT_2 holds the EV71 port.
PROJECT_1 = Path("/home/moshe/intern_projects/project_1")
PROJECT_2 = Path("/home/moshe/intern_projects/project_2")

# Ludovic's original external workflow — the parity reference, never edited.
LUDOVIC_ROOT = Path("/home/moshe/intern_projects/workflow-GCMC-Ludovic (after online tutorials)")

# ── The single source of protocol truth ───────────────────────────────────────
COMMON_PY = PROJECT_1 / "scripts" / "loch_ludovic_common.py"

# EV71 equivalents (the same constants, re-declared for the ported system).
EV71_COMMON_PY = PROJECT_2 / "scripts" / "ev71_loch_common.py"

for p in (COMMON_PY,):
    if not p.is_file():
        raise FileNotFoundError(f"protocol source not found: {p}")

print(f"protocol source : {COMMON_PY}")
print(f"exists          : {COMMON_PY.is_file()}")
print(f"EV71 source     : {EV71_COMMON_PY}  (exists={EV71_COMMON_PY.is_file()})")
print(f"Ludovic root    : {LUDOVIC_ROOT}  (exists={LUDOVIC_ROOT.is_dir()})")
'''.strip()))

cells.append(md("---\n## 1 · Imports\n\nStdlib only. This notebook must run on a login node with no GPU and no\nsimulation stack installed."))

cells.append(code(r'''
import ast
import hashlib
import json
import platform
import sys
import textwrap
'''.strip()))

cells.append(md(r"""
---
## 2 · Read the protocol constants from source

Parsed with `ast`, not imported: `loch_ludovic_common` imports `openmm`, `sire`
and `loch` at module scope, so importing it would require the full GPU stack.
The AST walk reads the module-level assignments only, which is exactly what the
protocol is.
""".strip()))

cells.append(code(r'''
# ── AST extraction of module-level literal constants ─────────────────────────

def module_constants(path: Path) -> dict:
    """Return {NAME: value} for module-level assignments of literal values."""
    tree = ast.parse(path.read_text())
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            out[target.id] = ast.literal_eval(node.value)
        except ValueError:
            # f-strings such as TEMPERATURE = f"{TEMPERATURE_K:g} K" are derived,
            # not independent protocol values; they are reconstructed below.
            continue
    return out


K = module_constants(COMMON_PY)
SOURCE_SHA256 = hashlib.sha256(COMMON_PY.read_bytes()).hexdigest()

PHYSICAL = [
    ("TEMPERATURE_K",                 "K",              "thermostat set point"),
    ("FRICTION_PER_PS",               "ps^-1",          "Langevin friction (via Sire property map)"),
    ("TIMESTEP_FS",                   "fs",             "integration timestep"),
    ("CUTOFF",                        "",               "PME real-space cutoff"),
    ("SWITCH_DISTANCE_NM",            "nm",             "LJ switching distance"),
    ("EWALD_ERROR_TOLERANCE",         "",               "OpenMM Ewald tolerance"),
    ("SPHERE_RADIUS",                 "",               "ligand-centred GCMC sphere"),
    ("MU_EX",                         "",               "TIP3P excess chemical potential"),
    ("STANDARD_VOLUME",               "",               "TIP3P standard volume"),
    ("NUM_GHOSTS",                    "waters",         "GCMC buffer size, per stage"),
    ("PRESSURE",                      "",               "NPT barostat pressure"),
    ("BAROSTAT_FREQUENCY",            "steps",          "MonteCarloBarostat interval"),
    ("CA_RESTRAINT_K",                "kJ/mol/nm^2",    "C-alpha positional restraint"),
    ("COM_RESET_FREQUENCY",           "steps",          "centre-of-mass removal"),
    ("MINIMIZATION_TOLERANCE_KJ_MOL_NM", "kJ/mol/nm",   "minimizer tolerance"),
    ("DEFAULT_BATCH_SIZE",            "trials",         "Loch parallel width — NOT a move count"),
]

print("╔" + "═" * 78 + "╗")
print(f"║  {'PHYSICAL PROTOCOL — parsed from source':<76}║")
print("╠" + "═" * 78 + "╣")
for name, unit, note in PHYSICAL:
    value = K[name]
    shown = f"{value} {unit}".strip()
    print(f"║  {name:<34} {shown:<18} {note[:22]:<22}║")
print("╚" + "═" * 78 + "╝")
print(f"\nsource sha256 : {SOURCE_SHA256}")
'''.strip()))

cells.append(md(r"""
### 2b · The two constants that are routinely misread

`batch_size` is Loch's **parallel trial width**. It does not reduce, replace or
rescale any attempt count below. A run with `batch_size=50` performs exactly the
same number of GCMC attempts as one with `batch_size=1`.

`CA_RESTRAINT_K = 100` is the bare numerical coefficient in Ludovic's
particle-free restraint `E = k * periodicdistance(...)^2`, applied with
`kJ/mol/nm^2` units. It is active in UVT1, UVT2 and production, and **absent
during NPT** — see §5.
""".strip()))

cells.append(md("---\n## 3 · Derived stage schedule\n\nThe schedule is reconstructed arithmetically from the parsed constants, then\nchecked against the documented totals. Nothing here is hand-entered except the\nexpectations being tested."))

cells.append(code(r'''
# ── Reconstruct the executed schedule from the parsed constants ───────────────
fs_per_ns = 1_000_000.0
dt = K["TIMESTEP_FS"]


def ns(steps: int) -> float:
    return steps * dt / fs_per_ns


schedule = {}

# UVT1 --------------------------------------------------------------------------
schedule["UVT1"] = {
    "initial_attempts": K["UVT1_INITIAL_ATTEMPTS"],
    "cycles": K["UVT1_CYCLES"],
    "attempts_per_cycle": K["UVT1_ATTEMPTS"],
    "md_steps_per_cycle": K["UVT1_MD_STEPS"],
    "loop_attempts": K["UVT1_CYCLES"] * K["UVT1_ATTEMPTS"],
    "total_attempts": K["UVT1_INITIAL_ATTEMPTS"] + K["UVT1_CYCLES"] * K["UVT1_ATTEMPTS"],
    "md_steps": K["UVT1_CYCLES"] * K["UVT1_MD_STEPS"],
    "report_interval": K["UVT1_REPORT_INTERVAL"],
}

# NPT ---------------------------------------------------------------------------
schedule["NPT"] = {
    "initial_attempts": 0,
    "cycles": 1,
    "attempts_per_cycle": 0,
    "md_steps_per_cycle": K["NPT_STEPS"],
    "loop_attempts": 0,
    "total_attempts": 0,
    "md_steps": K["NPT_STEPS"],
    "report_interval": K["NPT_REPORT_INTERVAL"],
}

# UVT2 --------------------------------------------------------------------------
schedule["UVT2"] = {
    "initial_attempts": 0,
    "cycles": K["UVT2_CYCLES"],
    "attempts_per_cycle": K["UVT2_ATTEMPTS"],
    "md_steps_per_cycle": K["UVT2_MD_STEPS"],
    "loop_attempts": K["UVT2_CYCLES"] * K["UVT2_ATTEMPTS"],
    "total_attempts": K["UVT2_CYCLES"] * K["UVT2_ATTEMPTS"],
    "md_steps": K["UVT2_CYCLES"] * K["UVT2_MD_STEPS"],
    "report_interval": K["UVT2_REPORT_INTERVAL"],
}

# Production --------------------------------------------------------------------
schedule["PRODUCTION"] = {
    "initial_attempts": 0,
    "cycles": K["PRODUCTION_CYCLES"],
    "attempts_per_cycle": K["PRODUCTION_ATTEMPTS"],
    "md_steps_per_cycle": K["PRODUCTION_MD_STEPS"],
    "loop_attempts": K["PRODUCTION_CYCLES"] * K["PRODUCTION_ATTEMPTS"],
    "total_attempts": K["PRODUCTION_CYCLES"] * K["PRODUCTION_ATTEMPTS"],
    "md_steps": K["PRODUCTION_CYCLES"] * K["PRODUCTION_MD_STEPS"],
    "report_interval": K["PRODUCTION_REPORT_INTERVAL"],
}

for stage, s in schedule.items():
    s["ns"] = ns(s["md_steps"])
    s["csv_records"] = s["md_steps"] // s["report_interval"]
    # One ghost-state line per GCMC cycle; NPT has no sampler at all.
    s["ghost_lines"] = s["cycles"] if s["total_attempts"] else 0

W = 92
print("╔" + "═" * W + "╗")
print(f"║ {'EXECUTED SCHEDULE (derived)':^{W - 1}}║")
print("╠" + "═" * W + "╣")
hdr = f"{'stage':<11}{'cycles':>7}{'att/cyc':>9}{'MD/cyc':>8}{'attempts':>11}{'MD steps':>11}{'ns':>8}{'CSV':>7}{'ghost':>7}"
print(f"║ {hdr:<{W - 1}}║")
print("╟" + "─" * W + "╢")
for stage, s in schedule.items():
    row = (f"{stage:<11}{s['cycles']:>7}{s['attempts_per_cycle']:>9}"
           f"{s['md_steps_per_cycle']:>8}{s['total_attempts']:>11,}"
           f"{s['md_steps']:>11,}{s['ns']:>8.2f}{s['csv_records']:>7}{s['ghost_lines']:>7}")
    print(f"║ {row:<{W - 1}}║")
print("╚" + "═" * W + "╝")
print(f"\nUVT1 initial standalone attempts (no MD time advanced): "
      f"{schedule['UVT1']['initial_attempts']:,}")
'''.strip()))

cells.append(code(r'''
# ── Assert the derived totals against the documented protocol ────────────────
# These are the numbers quoted in README.md, ludovic-parity.md and diagnostics.md.
# A failure here means the pipeline constants and the documentation have diverged.

EXPECTED = {
    ("UVT1",       "total_attempts"): 110_000,
    ("UVT1",       "loop_attempts"):  100_000,
    ("UVT1",       "md_steps"):           500,
    ("UVT1",       "csv_records"):          5,
    ("UVT1",       "ghost_lines"):        100,
    ("NPT",        "md_steps"):     1_000_000,
    ("NPT",        "csv_records"):        400,
    ("NPT",        "ghost_lines"):          0,
    ("UVT2",       "total_attempts"): 100_000,
    ("UVT2",       "md_steps"):       250_000,
    ("UVT2",       "csv_records"):        500,
    ("UVT2",       "ghost_lines"):        125,
    ("PRODUCTION", "total_attempts"): 500_000,
    ("PRODUCTION", "md_steps"):     5_000_000,
    ("PRODUCTION", "csv_records"):     10_000,
    ("PRODUCTION", "ghost_lines"):      2_500,
}

failures = []
for (stage, key), want in EXPECTED.items():
    got = schedule[stage][key]
    if got != want:
        failures.append(f"{stage}.{key}: derived {got:,} != documented {want:,}")

assert schedule["NPT"]["ns"] == 2.0, schedule["NPT"]["ns"]
assert schedule["PRODUCTION"]["ns"] == 10.0, schedule["PRODUCTION"]["ns"]

if failures:
    raise AssertionError("protocol drift:\n  " + "\n  ".join(failures))

print(f"  {len(EXPECTED)} derived totals match the documented protocol")
print(f"  NPT        = {schedule['NPT']['ns']:.0f} ns")
print(f"  production = {schedule['PRODUCTION']['ns']:.0f} ns")
print(f"  UVT1 MD    = {schedule['UVT1']['ns'] * 1000:.0f} ps   (a 1 ps segment; see the caveat in §9)")
print(f"  UVT2 MD    = {schedule['UVT2']['ns'] * 1000:.0f} ps")
'''.strip()))

cells.append(md(r"""
### 3b · Where the comments in Ludovic's scripts disagree with the code

Two places, and in both the **executable statement is authoritative**:

| Ludovic's comment | What the code executes |
|---|---|
| NPT described as 1 ns | `1_000_000 × 2 fs` = **2 ns** |
| production loop reads as GCMC-then-MD | 2,000 MD steps **first**, then 200 attempts, then report + DCD frame |

The second one matters for reporter alignment: the DCD frame and the ghost line
are both written *after* the move, so frame *i* and ghost line *i* describe the
same post-move state. A postprocessor that assumes the opposite silently
mis-labels every frame's inactive set.
""".strip()))

cells.append(md("---\n## 4 · Ghost lifecycle and the physical-handoff contract\n\nThis is the part of the port that broke, and the part that had to be fixed for\nanything downstream to be meaningful. Notebook 07 has the diagnostic record."))

cells.append(code(r'''
# ── Ghost lifecycle across stage boundaries ──────────────────────────────────
lifecycle = [
    ("UVT1 construction",  f"append {K['NUM_GHOSTS']} ghost waters",         "sampler present"),
    ("UVT1 setup",         "delete waters inside the 10 A sphere, minimize", "sampler present"),
    ("UVT1 sampling",      f"{schedule['UVT1']['total_attempts']:,} attempts", "sampler present"),
    ("UVT1 -> NPT",        "finalise + save_physical: materialize accepted "
                           "buffers as TIP3P, remove every state-0 water",   "NO sampler"),
    ("NPT",                f"{schedule['NPT']['md_steps']:,} MD steps, 1 bar", "NO sampler"),
    ("NPT -> UVT2",        f"fresh sampler appends {K['NUM_GHOSTS']} ghosts", "sampler present"),
    ("UVT2 sampling",      f"{schedule['UVT2']['total_attempts']:,} attempts", "sampler present"),
    ("UVT2 -> production", "finalise + save_physical",                       "NO sampler"),
    ("Production",         f"fresh sampler appends {K['NUM_GHOSTS']} ghosts", "sampler present"),
    ("Production end",     "ghost-containing raw topology KEPT for the DCD, "
                           "plus a separate ghost-free physical restart",    "both saved"),
]

W = 100
print("╔" + "═" * W + "╗")
print(f"║ {'GHOST LIFECYCLE':^{W - 1}}║")
print("╠" + "═" * W + "╣")
for boundary, action, state in lifecycle:
    print(f"║ {boundary:<20} {action:<58} {state:<18}║")
print("╚" + "═" * W + "╝")

print(f"""
Every new GCMC stage creates a FRESH {K['NUM_GHOSTS']}-water buffer. It is not carried over.
This matches grand.utils.add_ghosts() being called once per stage in Ludovic's scripts.

The handoff arithmetic that must hold at each GCMC -> physical boundary:

    saved physical waters = input physical waters + {K['NUM_GHOSTS']} - final state-0 count

Apply it INDEPENDENTLY to UVT1 and UVT2, because UVT2 starts a new buffer.
It catches both discarded accepted waters and leaked inactive ghosts, and it
catches them even when MD and Slurm complete normally.
""")
'''.strip()))

cells.append(code(r'''
# ── The two-call contract, and why generic save_system() is not a substitute ──
contract = textwrap.dedent("""
    At EVERY saved GCMC-to-physical boundary, in this order:

      1. finalise_sampler_system()
           - copy live imaged positions + periodic box into a Sire clone
           - remove every logically inactive water/ghost by stable molecule identity
           - require remaining zero-interaction-water count == Loch's logically
             active appended-buffer count
           - give exactly those retained waters the physical TIP3P charge/LJ values
           - validate the committed values before returning

      2. save_physical_system()
           - validate the in-memory topology
           - write the AMBER pair + PDB
           - RELOAD the AMBER pair from disk
           - require identical physical-water count and ZERO waters with
             simultaneously all-zero charge and all-zero LJ epsilon

    Use the generic save_system() for exactly one artefact: the deliberately
    ghost-containing production trajectory topology. Never feed that topology to
    NPT, and never hand it to a fresh sampler as a physical restart.
""").strip()
print(contract)

print("""
WHY this exists — Loch 2025.2 mutates an accepted buffer water's charge and LJ
parameters only in the LIVE OpenMM NonbondedForce. The Sire topology still
carries the original ghost values. Saving the Sire topology therefore persists
zero-interaction waters that the sampler considers physical. NPT then propagates
them with no charge and no LJ until they drift into overlaps, and the next GCMC
stage restores their parameters and detonates.

Minimisation succeeding is not evidence of correctness. A ghost-parameter water
in bulk solvent minimises trivially.
""")
'''.strip()))

cells.append(md("---\n## 5 · Ludovic ↔ Loch parity, and the unavoidable differences"))

cells.append(code(r'''
# ── Parity: what is identical, and what cannot be ────────────────────────────
identical = [
    "Stage order: UVT1 -> NPT -> UVT2 -> production",
    "Every attempt count, cycle count and MD-step count in the table above",
    "Reporter frequencies and report/frame alignment",
    f"{K['NUM_GHOSTS']} ghosts created fresh per GCMC stage",
    f"Ligand-centred {K['SPHERE_RADIUS']} GCMC sphere",
    f"{K['TEMPERATURE_K']:g} K, {K['TIMESTEP_FS']:g} fs, friction {K['FRICTION_PER_PS']:g} ps^-1",
    f"PME {K['CUTOFF']} cutoff, {K['SWITCH_DISTANCE_NM']:g} nm switching, "
    f"dispersion correction OFF, Ewald tol {K['EWALD_ERROR_TOLERANCE']}",
    "h-bond constraints; COM removal every step",
    f"TIP3P mu_ex {K['MU_EX']}, standard volume {K['STANDARD_VOLUME']}",
    f"C-alpha restraint coefficient {K['CA_RESTRAINT_K']:g} in UVT1/UVT2/production, none in NPT",
    f"NPT barostat every {K['BAROSTAT_FREQUENCY']} steps at {K['PRESSURE']}",
]

unavoidable = [
    ("GCMC engine",   "GRAND: sequential PME moves",
                      "Loch: batched reaction-field preacceptance, then PME correction"),
    ("Integrator",    "openmmtools BAOABIntegrator",
                      "OpenMM LangevinMiddleIntegrator (same BAOAB-family middle splitting)"),
    ("Friction API",  "direct keyword",
                      'Sire 2025.4 needs map={"friction": 1/picosecond}; _dynamics() has no friction kwarg'),
    ("Seeds",         "not set",
                      "explicit deterministic seeds (reproducibility scaffolding only)"),
    ("UVT1 velocities", "assigned before minimization",
                      "assigned AFTER minimization — avoids constraint-inconsistent kinetic states"),
    ("Postprocessing", "rebuilt from PDB at the end of Production.py",
                      "separate entry point over saved AMBER boundaries"),
    ("Ghost shift order", "shift inactive ghosts BEFORE imaging",
                      "EV71 port images/aligns first, then shifts, and masks inactive IDs explicitly"),
]

print("IDENTICAL TO LUDOVIC")
print("─" * 78)
for line in identical:
    print(f"  * {line}")

print("\nNECESSARILY DIFFERENT (implementation, not protocol)")
print("─" * 78)
for what, grand, loch in unavoidable:
    print(f"  {what}")
    print(f"      Ludovic : {grand}")
    print(f"      Loch    : {loch}")

print("""
None of the above licenses a change to stage order, force-field settings, the
ghost lifecycle or reporting alignment. Exact trajectories and acceptance
histories cannot match, because the water-move engine is being replaced. Water
counts and acceptance histories are stochastic; topology invariants, report
counts and operation order are not.
""")
'''.strip()))

cells.append(md(r"""
### 5b · The postprocessing order decision, stated once

GRAND shifts inactive ghosts five box lengths away **before** molecular imaging.
Imaging can then wrap those ghosts back toward the ligand, where they can enter
the sphere and be clustered as if they were water.

- `LochPostprocess.py` (CRY1) **preserves GRAND's order deliberately**, because
  its job is bit-level parity with Ludovic's output. On the Loch smoke
  trajectory its processed coordinates, sphere centres, cluster coordinates and
  occupancies matched the real `grand.utils` functions exactly.
- `ev71_postprocess.py` (the port) **fixes the order**: image and align first,
  shift second, exclude the per-frame inactive residue IDs explicitly, then
  prove independently that they lie outside the sphere.

Both behaviours are correct for their own purpose. What is *not* acceptable is
an unlabelled mixture. If you touch either one, keep the label.
""".strip()))

cells.append(md("---\n## 6 · Seeds and replica seed blocks"))

cells.append(code(r'''
# ── Deterministic seed derivation ────────────────────────────────────────────
BASE_SEED = K["SEED"]
STAGE_OFFSET = {"UVT1": 0, "NPT velocities": 1, "UVT2": 2, "production": 3}
REPLICA_SEED_STRIDE = 1000   # submit_loch_replicas.sh / ev71 series


def replica_base_seed(replica: int, base: int = BASE_SEED) -> int:
    if replica < 1:
        raise ValueError("replica IDs are positive integers")
    return base + (replica - 1) * REPLICA_SEED_STRIDE


print(f"base seed                : {BASE_SEED}")
print(f"replica seed stride      : {REPLICA_SEED_STRIDE}")
print()
print(f"{'replica':>8}  " + "  ".join(f"{s:>16}" for s in STAGE_OFFSET))
for rep in (1, 2, 3, 7, 12):
    base = replica_base_seed(rep)
    print(f"{rep:>8}  " + "  ".join(f"{base + off:>16}" for off in STAGE_OFFSET.values()))

print("""
Replicas are independent: no dependencies between jobs, non-colliding seed
blocks, one directory each. The wrapper refuses to overwrite an existing replica
directory — choose another REPLICA or move the incomplete one.

Ludovic's original scripts set no seeds at all. These exist so a diagnostic can
change one variable at a time; they are not a claim that GCMC is deterministic.
""")
'''.strip()))

cells.append(md("---\n## 7 · Environment fingerprint\n\nThe validated endpoint stack. Recorded here because two of these pins are\nload-bearing and one of them cost a whole FEP network (notebook 06)."))

cells.append(code(r'''
# ── Validated environment ────────────────────────────────────────────────────
ENVIRONMENTS = {
    "cry-loch-babel": {
        "role": "endpoint MD/GCMC — CRY1 and EV71",
        "spec": "cry-loch-babel.yml",
        "pins": {
            "loch": "2025.2.0",
            "sire": "2025.4.0",
            "BioSimSpace": "2025.4.0",
            "openmm": "8.4.0",
            "cuda-version": "12.8  (capped to the cluster driver — LOAD-BEARING)",
        },
        "also": "Open Babel, PDBFixer, AmberTools, CUDA compiler (nvcc) for Loch",
    },
    "automated-fep": {
        "role": "relative FEP (SOMD2) — kept SEPARATE on purpose",
        "spec": "environment-fep.yml",
        "pins": {
            "SOMD2/OpenBioSim": "2026.1",
            "cuda-version": "12.8  (must be <= the NODE driver's CUDA)",
            "cuda-nvvm": "required whenever the bound leg uses Loch GCMC",
        },
        "also": "never install SOMD2 into cry-loch-babel; never upgrade the endpoint env in place",
    },
    "csbrt": {
        "role": "single-environment unified workflow (preprocess..analysis)",
        "spec": "project_2/csbrt/environment.yml",
        "pins": {
            "structure predictor": "OpenFold3 (Boltz-2/Chai-1 pin numpy<2 and break sire/somd2)",
            "cuda": "pinned WHOLE to 12.8, not just cuda-version",
        },
        "also": "OpenFold3 needs SM>=8.0; on Turing pass runner_turing.yml to disable Triton kernels",
    },
}

for name, env in ENVIRONMENTS.items():
    print("═" * 78)
    print(f"  {name}   —   {env['role']}")
    print(f"  spec: {env['spec']}")
    for pkg, ver in env["pins"].items():
        print(f"      {pkg:<22} {ver}")
    print(f"      note: {env['also']}")
print("═" * 78)

print(f"""
Mamba activation rules for every wrapper (all three are needed):
  * use "$HOME/miniforge3/bin/mamba" and its BASH SHELL HOOK
  * do NOT source conda.sh and do NOT call `conda activate`
  * run `set -eo pipefail`, then the hook and activation under `set +u`, then
    restore `set -u` — deactivation hooks can dereference unset CONDA_BACKUP_*
    variables when a submitted job inherits another active environment

This notebook's own interpreter: python {platform.python_version()} on {platform.system()}
""")
'''.strip()))

cells.append(md("---\n## 8 · Machine-readable protocol record\n\nWritten so downstream notebooks and audits can consume the same numbers without\nre-deriving them."))

cells.append(code(r'''
# ── Emit the protocol record ─────────────────────────────────────────────────
record = {
    "source": str(COMMON_PY),
    "source_sha256": SOURCE_SHA256,
    "physical": {name: K[name] for name, _, _ in PHYSICAL},
    "schedule": schedule,
    "seeds": {
        "base": BASE_SEED,
        "stage_offsets": STAGE_OFFSET,
        "replica_stride": REPLICA_SEED_STRIDE,
    },
    "environments": {k: v["pins"] for k, v in ENVIRONMENTS.items()},
}

OUT_JSON = Path("protocol_record.json")
OUT_JSON.write_text(json.dumps(record, indent=2) + "\n")
print(f"wrote {OUT_JSON.resolve()}  ({OUT_JSON.stat().st_size:,} bytes)")
print()
print(json.dumps({"schedule": {k: {"total_attempts": v["total_attempts"],
                                   "md_steps": v["md_steps"],
                                   "ns": v["ns"]}
                               for k, v in schedule.items()}}, indent=2))
'''.strip()))

cells.append(md(r"""
---
## 9 · Recap and the traps worth knowing before you touch anything

**The protocol.** 300 K, 2 fs, PME 12 Å with 10 Å switching, 45 ghosts per GCMC
stage, ligand-centred 10 Å sphere. UVT1 = 110,000 attempts + 1 ps MD.
NPT = 2 ns. UVT2 = 100,000 attempts + 0.5 ns MD. Production = 10 ns as
2,500 × (2,000 MD steps → 200 attempts → report).

**Six things that will mislead you.**

1. `profile=smoke` is plumbing validation *only*. It is not a short scientific
   run. A smoke trajectory has 3 frames. Several results elsewhere in this
   handover rest on smoke-scale water sampling and are flagged where they do.
2. **Exit status 0 does not mean anything was produced.** Verify by artefact:
   expected file present, non-zero frames, finite numbers. A SOMD2 window worker
   can die while the top-level command returns 0.
3. The UVT1/UVT2 CSV `density` column includes the masses of the 45 appended
   buffer waters while a sampler is live. It is **not** the physical density.
   Use NPT for density.
4. The UVT1 MD segment is only 1 ps, so its temperature legitimately sits below
   300 K (measured 208.6 K mean in the reference run). That is not a failure.
5. `batch_size` is not a move count.
6. Do not call endpoint MD/GCMC "FEP". Relative FEP needs a mapped perturbable
   pair, bound *and* free legs, lambda windows, and overlap analysis. Existing
   endpoint trajectories contain no hidden alchemical result.

**Where to go next.**

| Notebook | Subject |
|---|---|
| 01 | CRY1–AN139 system preparation (Ludovic's `System_Prep.ipynb`, as executed) |
| 02 | Loch UVT1 → NPT → UVT2 → 10 ns production |
| 03 | Boundary audit and the acceptance gate, run on retained reference data |
| 04 | Postprocessing and multi-replica water clusters |
| 05 | EV71 32 × 6 hydration-site series |
| 06 | Relative FEP over Loch-equilibrated waters — including its failure record |
| 07 | Stability diagnostics and energy parity |
| 08 | Whether GCMC waters help docking (they do not, against crystal) |
""".strip()))

write_nb(OUT, cells)
print(f"wrote {OUT} ({len(cells)} cells)")
