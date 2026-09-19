# MACE surrogate: decision register

Every judgement call made while building `implementation_plan.md`, in one
place, so each can be challenged on its own terms rather than discovered by
reading the diff.

Each entry says what the plan specified, what was done instead, why, where it
lives, how it is tested, and **how to reverse it**. Nothing here is locked in.

Decisions are grouped by how much they matter:

- **D1–D4** change what the work can be expected to deliver. Read these.
- **D5–D9** are engineering choices with real trade-offs.
- **D10–D16** are smaller calls, recorded for completeness.

---

## D1 — The ~50% speedup target is not achievable with this architecture

| | |
|---|---|
| **Plan said** | "reduce overall simulation time by roughly 50%" via a MACE surrogate |
| **Done** | Built exactly as specified, and added `csbrt-mace-benchmark` to measure the result rather than assume it |
| **Status** | **Needs a decision from you** |

**Why.** Hybrid ML/MM does not replace the classical force field, it adds to
it. The protein, all bulk solvent and all of PME stay classical; what moves to
MACE is a ~40-atom ligand's internal terms, well under 0.1% of the classical
work in a 40,000-atom box. On top of that goes a forward and backward pass
through an equivariant graph network at every timestep.

This is not a contested reading. `docs/wang2024_design_space_mm_mlff.pdf` —
cited in the brief as the grounding for the partitioning design — gives
100–1000 ns/day for classical MM against 10–50 for hybrid ML/MM. The brief
quotes that table and then states the opposite conclusion.

The plan's own reasoning is sound up to a point: a *pure* MLFF over the whole
complex is ~100× slower than classical MM, so partitioning is the right
architecture. Partitioning just does not get you below the classical cost; it
gets you from 100× slower to ~10× slower.

**Measured**, `csbrt-mace-benchmark`, built-in fixture, CPU, MACE-OFF23-small:

```
classical MM          :   69.26 ns/day
mixed, surrogate (l=1):    5.78 ns/day     speedup 0.084x
```

**What it does buy instead.** Quantum-quality energetics for the ligand's
internal degrees of freedom. MACE-OFF23 is fitted to
wB97M-D3(BJ)/def2-TZVPPD; GAFF2/AM1-BCC is fitted to neither, and its weakest
point — torsion profiles and intramolecular strain in the perturbable region —
is exactly what alchemical FEP is most sensitive to. This repository already
runs `torsion_diagnostics` on every edge because an unsampled rotatable bond
is a known failure mode here. Wang et al. show hybrid alchemical RBFE reaching
chemical accuracy with the perturbable region on a neural potential.

**Where a 50% reduction could actually come from**, all measurable against
checkpoints you already have: replica exchange (implemented, `--replica-exchange`),
λ-window count justified by the overlap diagnostics rather than a fixed
`num_lambda: 11`, runtime justified by convergence rather than the
"conservative starting point" `somd2_config.yaml` calls itself, and pruning the
edge network on cycle-closure residuals. Full argument in
[`mlff_throughput_expectations.md`](mlff_throughput_expectations.md) §4.

**To reverse:** nothing to reverse. The code is built and works; only the
expectation needs setting.

---

## D2 — The physics fallback buys correctness, not speed

| | |
|---|---|
| **Plan said** | "Zero-Overhead Dual-Hamiltonian Context Switching … sets w=1.0 instantaneously" |
| **Done** | Implemented exactly that — and documented that it does not reduce cost |

**Why.** The switch itself *is* zero-overhead: measured at **0.7 µs** against
the 200–500 ms a Context rebuild costs, which is the pitfall the plan
correctly identified. But openmm-ml implements the interpolation with a
`CustomCVForce`, and a `CustomCVForce` evaluates **every** collective variable
on every step regardless of its coefficient. At λ=0 the MACE forward pass
still runs; its result is multiplied by zero.

The benchmark shows this directly — the mixed system at λ=0 costs the same as
at λ=1, even though it computes the same energy as the classical system to
eight significant figures.

Consequence worth knowing: when the controller latches to classical mode
because the fallback rate got too high, the run keeps paying the full ML cost
for no benefit. It logs a warning saying so; the right response is to stop and
restart without the surrogate.

**The alternative** — separate force groups plus
`Integrator.setIntegrationForceGroups()`, which genuinely skips the ML
evaluation — would work, at the cost of hand-building the interpolation
openmm-ml already provides and verifying it ourselves. Not worth it while the
fallback is a few percent of steps. It would become worth it if the fallback
rate were high, and at that point the model is not fit for the system anyway.

**To reverse:** implement the force-group variant in
`mace_mixed_system.py`; `MACEConfig.interpolate` already exists as the switch
between designs.

---

## D3 — A committee cannot be assembled from different MACE-OFF sizes

| | |
|---|---|
| **Plan said** | "committee ensemble inference … M = 4", foundation models `mace-off23-small`, `mace-off23-medium`, `mace-omol-0` |
| **Done** | Committee support built; discovered the constraint; documented and enforced |

**Why.** `MACECalculator` requires every committee member to share the same
cutoff radius `r_max`. `mace-off23-small` and `-medium` do not, so the obvious
way to assemble a committee from the released models does not work. Worse, on
mace-torch 0.3.16 that check raises a `TypeError` while formatting its own
error message, so the real cause is invisible.

Found by actually running it, not by reading about it.

**What this means in practice.** A real committee has to come from independent
fine-tunes of **one** foundation model — different seeds, different data
splits. That is what `csbrt-mace-al finetune --seeds N` produces. Until a
campaign has produced its own committee, there is no force variance to measure
and the geometry guard (D4) is the only out-of-distribution detector.

**Also found:** a committee needs calibrating. A two-member committee built by
perturbing one model's weights disagreed by 0.26 eV/Å on a *relaxed* pose —
five times the 0.05 eV/Å threshold. A run with that committee pays the full
MACE cost on every step and then discards every one of those steps for the
classical Hamiltonian. `csbrt-mace-preflight` now **fails** on this rather than
warning, and names the two possible causes.

**Where:** `committee.py` (`check_compatibility`, the translated error),
`mace_preflight.py` (`check_committee`).
**Tested:** `test_committee.py` (including against real downloaded weights),
`test_pipeline_wiring.py`.

---

## D4 — Binding-site waters stay out of the ML region by default

| | |
|---|---|
| **Plan said** | "ML Region: the perturbable ligand and active-site hydrating waters"; `include_binding_site_waters: True` |
| **Done** | Default `false`, opt-in with a warning |

Three independent reasons, any one of which is sufficient.

**1. Loch exchanges waters; the ML atom list cannot follow.** The ML atom
indices are baked into the ML force when the Context is built. Loch's GCMC
inserts and deletes waters every cycle by toggling ghost molecules in the live
Context. A water in the ML region would stay in it after diffusing out of the
site, and could not be exchanged at all.

**2. A ghost water in the ML region would be computed twice, inconsistently.**
MACE would compute a real interaction energy for a molecule the MM side
believes does not exist. `partition_ml_atoms` now excludes any particle with
zero charge *and* zero LJ epsilon in every `NonbondedForce` for exactly this
reason.

**3. It breaks the exact classical limit** — this one was found by testing.
With the ligand alone, `λ=0` reproduces the untouched force field to
**4.5e-08 kJ/mol**. With the ligand plus four scattered waters it is out by
**0.014 kJ/mol**.

The cause is in openmm-ml: going back to the classical Hamiltonian it restores
the ML region's own nonbonded interactions with an explicit bonded term, and
that term has no cutoff. For electrostatics under PME that is exactly right —
the reciprocal sum covers every image, so removing an intra-region pair removes
exactly `q_i q_j / r` however far apart it is, which is why zeroing the charges
leaves the error untouched. For Lennard-Jones it is not: the force field
truncates beyond the cutoff and the restoring term does not, which is why
zeroing the LJ parameters removes almost all of it.

A compact ligand never crosses that line. A hydration shell does.
`check_region_compactness` measures the region's diameter against the cutoff at
attach time, warns (or fails under `strict`), and records the measurement in
the stage's checkpoint.

**A related finding:** openmm-ml's interpolation assumes PME. With a plain
cutoff or reaction-field `NonbondedForce` the classical limit is out by
**6.5 kJ/mol** for the ligand alone. Every stage here uses PME
(`cutoff_type="pme"` throughout `ev71_loch_common.py`), so this is documented
rather than worked around.

**Side benefit:** because no water's energy passes through MACE, the GCMC
acceptance criterion is untouched by the surrogate.

**To reverse:** set `mlff.include_binding_site_waters: true`. The path works
and is tested; it just is not exact.

---

## D5 — A geometry guard was added, alongside the committee

| | |
|---|---|
| **Plan said** | Committee force variance and energy variance |
| **Done** | Both, plus a cheap geometric plausibility check |

**Why.** A committee measures *disagreement*. Models trained on the same data
agree confidently about geometries none of them ever saw — which is precisely
the failure that puts a ligand through a protein wall and produces a
free energy that looks fine. The plan's own Stage 6 verification ("inject
synthetic OOD configurations … confirm 100% intercept rate") is not something
a committee can be relied on to pass.

The guard checks the ML region for atoms closer than 0.70 Å and bonds
stretched past 1.6× the sum of covalent radii. It is O(N²) over 40–250 atoms,
costs microseconds, and needs no second model — so it is also the only
detector available before active learning has produced a committee.

**Tested:** 20/20 interception on injected distortions, `test_uq_monitor.py`.
**To reverse:** `mlff.geometry_guard: false`.

---

## D6 — Fine-tuning refuses MM labels unless asked twice

| | |
|---|---|
| **Plan said** | "Computes ground truth energies and forces via Sire/SOMD2/AmberTools (or DFT single-point)" |
| **Done** | Level of theory recorded per frame; MM labels refused without `--allow-mm-labels` |

**Why.** MACE-OFF is fitted to wB97M-D3(BJ)/def2-TZVPPD. Fine-tuning it
against classical MM single-points would teach it the MM surface — replacing
the quantum accuracy that is the entire reason for the surrogate (see D1) with
the force field it exists to improve on.

The MM labeller is still shipped: it is the right reference for validating the
harvest → label → train path end to end, and for a run whose surrogate is
itself MM-level. It just cannot be used by accident. Mixed levels of theory in
one training set are refused outright.

**Consequence:** producing a committee needs a QM single-point wired into
`csbrt-mace-al label --labeller command`. That is open item 3 in the delivery
report.

**Where:** `active_learner.py` (`REFERENCE_LEVELS`, `MACEFineTuner.fine_tune`).
**To reverse:** pass `--allow-mm-labels`.

---

## D7 — SOMD2 gets a sidecar and a `sitecustomize`, not a config key

| | |
|---|---|
| **Previous cut did** | `config_payload["mlff"] = {...}` written into the SOMD2 YAML |
| **Done** | JSON sidecar + `PYTHONPATH` hook; the SOMD2 config is untouched |

**Why.** SOMD2 validates its own config keys and rejects unknown ones. An
`mlff:` block in that file would have failed **every** MACE-enabled FEP leg
before it started. This was a latent bug in the inherited code, not a design
choice.

SOMD2 owns its process, its Sire system and one OpenMM Context per λ window,
and exposes no hook for a third-party potential. The standard interpreter-level
entry point is what works: a `sitecustomize.py` on the subprocess's
`PYTHONPATH`, which wraps `sire.system.System.dynamics` so every Context comes
back mixed.

**Three things make it safe rather than clever:**

- **Ordering is verified, not assumed.** Sire's atom order is checked against
  the Context's particle order by comparing coordinates under the minimum image
  convention *before* anything attaches. An ML region built on a wrong mapping
  would apply quantum forces to arbitrary atoms and still produce
  plausible-looking numbers.
- **Failure is classical, not wrong.** Anything unexpected disables the
  surrogate for that window and logs it; the window runs the physics it would
  have run anyway. `--mace-strict` turns that into a hard failure.
- **It reports what it did.** Each window writes
  `mace_surrogate/window*.json`, and the leg's checkpoint records how much of
  it actually ran on the surrogate. A leg that requested the surrogate and got
  none says so.

**Not executed here** — Sire and SOMD2 are conda-only and not installable in
the build container. Treat the first MACE-enabled leg as a validation run.

**Where:** `mace_surrogate/somd2_hook.py`, `run_fep_leg.py`.

---

## D8 — The shared modules were left byte-identical

| | |
|---|---|
| **Previous cut did** | Added functions to `pipeline_utils.py` and `ev71_loch_common.py` |
| **Done** | Both restored byte-for-byte; everything moved to a new `mace_pipeline.py` |

**Why.** `pipeline_utils.py` is hashed into the `implementation_signature` of
**every** stage's `.complete.json` marker, and `ev71_loch_common.py` into
equilibration, production and the GCI windows. Adding a single function to
either changes its hash, which invalidates every completed checkpoint in a
running campaign — an equilibration is UVT1 + NPT + UVT2, roughly a million
steps — whether or not anyone ever enables the surrogate.

Paying that to gain an optional feature you are not using is the wrong trade.

**What still invalidates:** `ev71_production.py` and `run_fep_leg.py` changed,
so their markers do. Equilibration, density, postprocess, prepare and GCI
markers do not.

**How the MD loop avoids duplication.** `mace_pipeline.run_with_surrogate`
does not reimplement `run_with_csv_reports`. `md_chunks` subdivides a run at
the UQ interval while still landing exactly on the report boundaries, so each
chunk handed to the original function is one of its own iterations. 125
parametrised cases assert the CSV step schedule is identical, across every
combination of run length, starting offset and UQ interval.

**Enforced:** two tests fail if the word "mace" reappears in either shared
module, or if a surrogate-capable stage stops hashing `mace_pipeline.py`.

---

## D9 — Equilibration stays classical; production runs the surrogate

| | |
|---|---|
| **Plan said** | Did not specify |
| **Done** | Surrogate flags forwarded to production only |
| **Status** | **Open item — your call** |

**Why.** Loch's NPT block alone is a million steps, and is where a ~10×
slowdown (D1) would hurt most. Keeping equilibration classical also keeps every
existing equilibration checkpoint valid (D8).

**What it costs.** Production starts from a geometry relaxed under a different
Hamiltonian, so the ligand's internal coordinates relax from the GAFF2 minimum
to the MACE one at the start of the run. That is a picosecond-scale relaxation
of ~40 atoms inside a 10 ns production run; discarding the first few GCMC
cycles covers it.

**To reverse:** if you need the equilibrated ensemble itself to be a MACE
ensemble — for a published number rather than a screen — forward the flags to
the equilibration stage too, and accept the checkpoint invalidation.

---

## D10 — Fail-safe to classical physics, not fail-fast

Anything that stops the surrogate being built — openmm-ml missing, a model file
absent, an empty ML region — logs an **error** and continues on classical
physics rather than killing the stage. The sampling is then correct; only the
acceleration is missing.

The stage's checkpoint records that the surrogate was not active, so a degraded
run can never be mistaken for an accelerated one, and `run_fep_leg` says so
explicitly if no λ window reported attaching.

`mlff.strict: true` / `--mace-strict` inverts this for a run where silently
falling back would waste more than it saves.

---

## D11 — Tests run against real OpenMM with a stub ML potential

Rather than mocking OpenMM, the suite registers a cheap analytic potential
through openmm-ml's own `MLPotential.registerImplFactory` extension point. The
mixed-system construction, the in-place Context swap, the nonbonded exclusions
and the `lambda_interpolate` machinery therefore go through exactly the code
path MACE takes — without a GPU or a 100 MB checkpoint, in 11 seconds.

Four model-gated tests (`CSBRT_MACE_MODEL_TESTS=1`) repeat the critical
assertions against real downloaded MACE-OFF23 weights.

The stub supplies harmonic bonds *and* a repulsive term over the rest of the
region: a bonds-only stand-in left nothing holding the ML atoms apart once the
mechanical embedding zeroed their internal nonbonded interactions, so the
region collapsed under minimisation. That was the stub being unphysical, not
the machinery — but a fixture that cannot survive its own minimiser is not a
fixture.

---

## D12 — Two unit bugs fixed in the inherited code

1. **`eV/Å → kJ/(mol·nm)` was out by a factor of 10.** The constant was
   `9648.53`; it is `964.85` (1 eV = 96.485 kJ/mol, and 1/Å = 10/nm). This
   multiplied every force threshold expressed in OpenMM units by ten. The
   constants are now derived from one CODATA value rather than typed
   individually, so the set cannot drift apart again.
2. **`mace-omol-0` is not a name openmm-ml registers** — the registered name is
   `mace-omol-0-extra-large`. It is now an alias rather than a runtime failure.

---

## D13 — The whole `mlff:` block is forwarded, not just the flagged keys

Three argparse layers sit between `run.yaml` and the code that builds the mixed
system. Rendering only the settings that have a dedicated CLI flag meant
`precision`, `max_ml_atoms`, `fallback_abort_fraction`, the geometry thresholds
and the buffer directory were accepted in the config and then silently dropped
in transit.

The block now travels as one `--mace-config` JSON argument, with individual
flags overriding it. Every MACE flag defaults to `None` rather than to a value,
so "not given" is distinguishable from "given the default"; the defaults live
in `MACEConfig` and nowhere else.

**Tested:** a test walks all three layers with the real parsers.

---

## D14 — Model paths resolve against the working directory, buffers against the run

A config saying `committee_model_paths: models/gen2.model` means the path you
typed it at, not a directory several levels down inside a run tree the model
was never copied into. Model checkpoints resolve against the working directory;
the OOD buffer resolves against the stage's output directory, where it belongs.

---

## D15 — Tools are console scripts, not a `scripts/` directory

The plan put `preflight_mace.py` and `benchmark_mace_speedup.py` under
`csbrt/scripts/`. That directory is not installed by `pyproject.toml`, so the
tools would work from a checkout and vanish on `pip install`. They are
`csbrt-mace-preflight`, `csbrt-mace-benchmark` and `csbrt-mace-al` instead,
alongside the existing `csbrt-*` commands.

---

## D16 — The OOD buffer is saved from a `finally` block

A run that crashes is exactly the run whose out-of-distribution frames are
worth fine-tuning on. It was throwing them away. The save is guarded so a
failure writing statistics cannot replace the exception that caused the crash.

---

## Things deliberately **not** changed

- **`num_lambda`, `runtime`, the λ schedule, the edge network.** Out of scope,
  and the levers most likely to actually reduce campaign time (D1). Untouched.
- **The GCMC acceptance criterion.** Untouched, and provably so while waters
  stay out of the ML region (D4).
- **`physical_protocol_signature()`** and every fixed Ludovic physical setting.
  Untouched.
- **The classical code path.** With `mlff.enabled: false` — the default — the
  checkpoint signature is `None` and the behaviour is byte-for-byte what it was
  (D8, D13).
