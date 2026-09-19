# MACE surrogate: delivery report

**Branch:** `feature/euph1` · **Base:** `179a7ea` · **Commits:** 8
**Scope:** all six stages of [`implementation_plan.md`](../implementation_plan.md)
**Tests:** 330 pass without a GPU or any model weights (11 s); 334 with real
MACE-OFF23 weights downloaded (44 s)

This is the review entry point. Two companion documents go deeper:

- [`mlff_decisions.md`](mlff_decisions.md) — every judgement call, what the
  plan said, what was done instead, and how to reverse it. **Read this one if
  you only read one.**
- [`mlff_active_learning_architecture.md`](mlff_active_learning_architecture.md)
  — component-by-component specification of what exists.
- [`mlff_throughput_expectations.md`](mlff_throughput_expectations.md) — what
  the surrogate costs and what it buys.

---

## 1. The headline, first

The brief's stated objective is **a ~50% reduction in simulation time**. That
target does not follow from the architecture the same brief specifies, and the
paper it is grounded in says so directly.

A hybrid ML/MM system keeps the protein, all the bulk solvent and all of the
PME electrostatics on the classical force field, and adds a neural network on
top for the ligand. It removes a 40-atom ligand's internal terms — well under
0.1% of the classical work in a 40,000-atom box — and adds a forward and
backward pass through an equivariant graph network at every timestep.

Wang, Eastman, Tuckerman et al. (2024), already in `docs/`:

| regime | throughput |
|---|---|
| classical MM | 100–1000 ns/day |
| **hybrid ML/MM** | **10–50 ns/day** |
| pure MLFF on a solvated complex | 0.1–1 ns/day |

Measured with `csbrt-mace-benchmark` on the built-in fixture, CPU platform,
MACE-OFF23-small:

```
classical MM          :   69.26 ns/day
mixed, fallback (l=0) :    5.18 ns/day
mixed, surrogate (l=1):    5.78 ns/day
speedup vs classical  :    0.084x
```

The fixture exaggerates the effect — 26 particles have almost no MM work to
amortise the ML cost against, and it runs on a CPU. A real complex on an A100
will land closer to the paper's 10–50 ns/day. It will not land above the
classical rate.

**What the surrogate does buy is accuracy, and it is the kind of accuracy this
pipeline is short of.** MACE-OFF23 is fitted to wB97M-D3(BJ)/def2-TZVPPD;
GAFF2 with AM1-BCC charges is fitted to neither, and its weakest point is
exactly what alchemical FEP is most sensitive to — torsion profiles and
intramolecular strain in the perturbable region. This repository already runs
`torsion_diagnostics` on every edge because a rotatable bond that never
crosses its barrier is a known failure mode here.

So the work was built as specified, in full, and the throughput claim is
reported honestly rather than assumed. `docs/mlff_throughput_expectations.md`
sets out where a 50% reduction could actually come from in this pipeline —
replica exchange, λ-window count, runtime, network pruning — all of which are
measurable against the checkpoints you already have.

---

## 2. What was built

Against the six stages of the implementation plan.

### Stage 0 — environment and preflight

- `environment.yml`: `openmm-ml` and `openmm-torch` added, with a note that
  openmm-torch must be the build compiled against this OpenMM and this CUDA.
- `install.sh`: `mace-torch>=0.3.10` via pip (so it shares OpenFold3's cu126
  torch rather than pulling a second one), plus a hard check that OpenMM is
  ≥ 8.6.1 — older releases install fine and then fail at
  `createMixedSystem(interpolate=True)`, which is what the fallback is built
  on.
- **`csbrt-mace-preflight`** — new. Checks imports, that OpenMM and torch agree
  about the device, that the ML potential constructs, that a mixed system
  builds and its classical limit is exact, that the Hamiltonian switch is
  microseconds, and that the committee is compatible and calibrated. Runs in
  seconds on a login node. Exit 1 on any hard failure.

### Stage 1 — hybrid ML/MM construction (`mace_mixed_system.py`)

The previous cut called `MLPotential("mace", ...)` with an
`implementation="mace"` argument openmm-ml would have rejected, and never
built a system. Now:

- The potential is named by the model (`MLPotential('mace-off23-small')`);
  `MLPotential('mace', modelPath=...)` is only for a fine-tuned checkpoint.
- `partition_ml_atoms()` takes whole residues, refuses a missing or duplicated
  ligand residue rather than silently producing an empty ML region, and
  excludes any particle Loch has switched off as a ghost.
- `attach_mace_to_context()` applies the mixed system to a **live** Context and
  `reinitialize(preserveState=True)` — the same pattern `add_ca_restraints`
  already uses — because Sire, Loch and SOMD2 all build their own Context.
  Positions, velocities and box vectors survive; asserted by test.
- Two guards: `max_ml_atoms` (the pure-MLFF scalability trap) and
  `check_region_compactness` (see decision D4).

### Stage 2 — uncertainty quantification (`uq_monitor.py`, `committee.py`)

Two independent detectors; either firing trips the fallback.

- **Committee force variance**, the documented (M−1)-normalised formula, fed by
  `MACECalculator`'s own batched `forces_comm`/`energy_comm` rather than a
  reimplementation. Default trigger 0.05 eV/Å; `calibrate_thresholds()` resets
  it from a high percentile of an in-distribution trajectory.
- **Geometry guard** — new, not in the plan. A committee only measures
  disagreement, and models trained on the same data agree confidently about
  geometries none of them ever saw. This checks for atoms closer than 0.70 Å
  and bonds stretched past 1.6× the sum of covalent radii, costs microseconds,
  and needs no second model. It is therefore the only out-of-distribution
  detector available before active learning has produced a committee.

### Stage 3 — zero-overhead fallback (`fallback_controller.py`)

`createMixedSystem(..., interpolate=True)` gives the System a global parameter
`lambda_interpolate`; switching Hamiltonians is one `setParameter` call.
Measured at **0.7 µs** against the 200–500 ms a Context rebuild costs.

The state machine counts MD steps rather than UQ evaluations, captures one OOD
frame per *event* rather than per step, and latches to classical physics if the
surrogate spends more than `fallback_abort_fraction` of the run falling back —
at which point it is costing more than it saves and says so.

### Stage 4 — active learning (`active_learner.py`, `csbrt-mace-al`)

- One `.npz` per worker under `al_buffer/`, written atomically — 52 edges ×
  replicates write concurrently from separate Slurm tasks and a shared file
  would race. Saved from a `finally` block, so a crashed run keeps its frames.
- Greedy heavy-atom RMSD clustering with Kabsch alignment; frames of different
  molecules are never compared; the representative of a basin is its
  highest-uncertainty member.
- Labelling through any single-point callable, with the level of theory stamped
  on every frame and never mixed.
- Fine-tuning shells out to `mace_run_train --foundation_model`, MACE's own
  supported path. `--seeds N` trains N models from one foundation model, which
  is how a usable committee gets made.

### Stage 5 — pipeline and HPC integration

| Where | What |
|---|---|
| `mace_pipeline.py` (new) | The flags declared **once**, the settings block forwarded intact, the checkpoint signature, the Loch attach hook, the surrogate-aware MD loop |
| `config.example.yaml` | An `mlff:` block, off by default, fully commented |
| `cli.py` | `--enable-mace-surrogate` and friends overlay the config; a flag left unset does not touch it |
| `ev71_production.py` | Attaches before `sampler.bind_dynamics`, re-verifies Loch's ghost bookkeeping after, reports fallback statistics into the checkpoint |
| `run_fep_leg.py` + `somd2_hook.py` | Sidecar + `sitecustomize` injection, with Sire's atom order verified against the Context's before anything attaches |
| `fep_edge.slurm`, `submit_fep_edges.sh` | `--with-mace`, rendered once on the frontend so all 52 array tasks run identical physics |

### Stage 6 — verification

- **`csbrt-mace-benchmark`** — new. Times the same Context three ways
  (classical, mixed at λ=0, mixed at λ=1), adds the committee's inference cost,
  and reports the speedup factor. A factor below 1.0 is reported as a slowdown,
  in those words.
- Energy conservation, fallback interception, switch latency and ΔΔG-parity
  guidance: see §3 and §5.

---

## 3. What was verified here, with numbers

Everything below was executed in this session against **real OpenMM 8.6.1**,
and where marked, **real downloaded MACE-OFF23 weights**.

| Property | Result | Where |
|---|---|---|
| **Classical limit** — `λ=0` reproduces the untouched force field | **4.52e-08 kJ/mol** difference, with a real MACE-OFF23 model | `csbrt-mace-preflight`, `test_mace_mixed_system.py` |
| Classical *trajectory* reproduced, frame for frame | identical to 1e-6 nm over 100 steps | `test_energy_conservation.py` |
| **Hamiltonian switch latency** | **median 0.7 µs**, max 5.5 µs over 64 switches | `csbrt-mace-preflight`, `test_fallback_controller.py` |
| Positions and velocities across a switch | bit-identical | `test_energy_conservation.py` |
| **NVE drift**, mixed vs classical (stub potential) | **0.138 vs 0.273 kT/dof/ns** | `test_energy_conservation.py` |
| **NVE drift**, real MACE-OFF23 mixed system | **0.155 kT/dof/ns** | `test_energy_conservation.py` (model-gated) |
| **OOD interception** on injected distorted ligands | **20/20** | `test_uq_monitor.py` |
| Committee σ_F grows on a distorted geometry | confirmed with real MACE-OFF23 | `test_committee.py` (model-gated) |
| CSV report schedule unchanged by UQ interleaving | 125 parametrised cases | `test_runtime.py` |
| Active learning, end to end | 2 workers → separate buffers → merged → 1 basin → labelled → `mace_run_train` command | `test_active_learning_loop.py` |
| Whole `mlff:` block survives 3 argparse layers | asserted with the real parsers | `test_pipeline_wiring.py` |
| Shared modules byte-identical to pre-MACE | asserted | `test_pipeline_wiring.py` |
| Installed console scripts work | `csbrt-mace-preflight/-benchmark/-al` all run from a `pip install` | manual |

### The test suite

```
test_runtime.py                        141   (125 are the chunking invariant)
test_active_learner.py                  37
test_pipeline_wiring.py                 36
test_config.py                          32
test_mace_mixed_system.py               27
test_uq_monitor.py                      18
test_fallback_controller.py             16
test_active_learning_loop.py            10
test_committee.py                       10
test_energy_conservation.py              7
                                       ---
                                       334
```

They run against real OpenMM with a **registered stub ML potential** — a cheap
analytic force plugged into openmm-ml through its own
`registerImplFactory` extension point. That means the mixed-system
construction, the in-place Context swap, the nonbonded exclusions and the
`lambda_interpolate` machinery all go through exactly the code path MACE
takes, without needing a GPU or a 100 MB checkpoint. The four model-gated tests
(`CSBRT_MACE_MODEL_TESTS=1`) then repeat the critical ones against the real
thing.

---

## 4. What could **not** be verified here

Neither Sire, Loch nor SOMD2 is installable in this container — they are
conda-only, on OpenBioSim's channel. Two integration paths are therefore
written but unexecuted:

1. **The Loch GCMC attach** (`mace_pipeline.attach_mace_surrogate`). Written
   defensively: it attaches *before* `sampler.bind_dynamics`, then re-checks
   that Loch's ghost-water set is unchanged and that exactly one
   `NonbondedForce` remains, and aborts the run if either moved. A wrong GCMC
   acceptance is worse than no speedup.
2. **The SOMD2 FEP hook** (`mace_surrogate/somd2_hook.py`). Written
   defensively: it verifies Sire's atom order against the Context's particle
   order by comparing coordinates under the minimum image convention before
   attaching anything, because an ML region built on a wrong mapping would
   apply quantum forces to arbitrary atoms and still produce plausible-looking
   numbers. Anything unexpected disables the surrogate for that window and logs
   it, rather than running wrong physics.

**Treat the first MACE-enabled FEP leg as a validation run against its
classical twin**, not as a production result.

---

## 5. The small-scale test to run

On a GPU node, in the `csbrt` environment.

### Step 1 — preflight (seconds)

```bash
csbrt-mace-preflight --device cuda --json preflight.json
```

Expect `preflight PASSED`. It will `WARN` about having no committee
configured — that is correct and expected for a first run; the geometry guard
is the detector until active learning produces one. A `FAIL` names the cause
in a sentence.

### Step 2 — measure the cost on your hardware (minutes)

```bash
csbrt-mace-benchmark \
  --prmtop RUN/endpoint/LIG/rep1/production/LIG-production-final.prmtop \
  --rst7   RUN/endpoint/LIG/rep1/production/LIG-production-final.rst7 \
  --steps 2000 --json mace_benchmark.json
```

Do this **before** committing any campaign time. It prints ns/day for the
classical system and for the mixed system in both modes, and a speedup factor.
Expect a factor below 1.0. This is the number to bring to any discussion of
the 50% target.

### Step 3 — a smoke run through GCMC (~1 hour)

```bash
csbrt --from equilibrate --through gcmc \
      --config run.yaml --profile smoke --enable-mace-surrogate
```

`profile: smoke` reduces the step counts to a plumbing test, not a scientific
trajectory. What to check in the output:

- `MACE surrogate attached to Loch dynamics: N ML atoms, ghosts preserved` —
  the attach worked and Loch's bookkeeping survived it.
- `[MACE surrogate] X of Y MD steps ran on the surrogate (Z%)` — the fallback
  rate. Anything more than a few percent means the model does not cover this
  chemistry; harvest and fine-tune before trusting free energies from it.
- `production.complete.json` should contain a `mace_stats` block and a
  `mace_surrogate` signature.
- `al_buffer/ood_*.npz` should exist if anything was flagged.

### Step 4 — one FEP edge, both ways

```bash
# classical
python run_fep_leg.py --stream EDGE_bound.bss --config somd2_config.yaml \
    --output-dir edge/bound-classical --leg bound
# surrogate
python run_fep_leg.py --stream EDGE_bound.bss --config somd2_config.yaml \
    --output-dir edge/bound-mace --leg bound --enable-mace-surrogate
```

Compare the ΔG. The plan's bar is 0.3 kcal/mol; treat a larger deviation as a
finding about the surrogate, not about the classical reference. Check
`edge/bound-mace/mace_surrogate/window*.json` — if it is empty, the hook did
not attach and the leg ran classical physics (it will say so).

### Step 5 — close the loop

```bash
csbrt-mace-al report   --buffer-dir RUN/al_buffer
csbrt-mace-al harvest  --buffer-dir RUN/al_buffer --output frames.npz
csbrt-mace-al label    --frames frames.npz --labeller command \
                       --command "my_qm_singlepoint {xyz}" --output labelled.npz
csbrt-mace-al finetune --frames labelled.npz --generation 2 --seeds 2
```

`--seeds 2` is not optional if you want uncertainty quantification (see
decision D3).

---

## 6. Open items needing your decision

| # | Item | Why it needs a decision |
|---|---|---|
| 1 | **The 50% target** | Not achievable via hybrid ML/MM. Either re-frame the objective around accuracy, or pursue the throughput levers in `mlff_throughput_expectations.md` §4 instead. Everything is built either way. |
| 2 | **MACE-OFF23 licensing** | Academic Software Licence — **commercial use is not permitted**. openmm-ml warns on every load. Needs either a differently-licensed model or one trained from the active-learning loop. |
| 3 | **QM labeller** | `csbrt-mace-al label --labeller command` needs a single-point QM code wired in (Psi4, ORCA). The contract is one JSON object with `energy_ev` and `forces_ev_per_ang`. Until then there is no way to produce a committee, and no committee means geometry-guard-only detection. |
| 4 | **Whether to equilibrate under the surrogate** | Currently classical, to protect existing checkpoints and because the NPT block is a million steps. Costs a picosecond-scale ligand relaxation at the start of production. See decision D9. |
| 5 | **Checkpoint invalidation** | `ev71_production.py` and `run_fep_leg.py` changed, so their markers invalidate. Equilibration, density, postprocess and GCI markers **do not** — deliberately, see decision D8. |

---

## 7. Where everything lives

```
csbrt/src/csbrt/mace_surrogate/     the package (11 modules, ~4,100 lines)
  units.py            unit conversions, single source of truth
  config.py           MACEConfig: validation, aliases, checkpoint signature
  mace_mixed_system.py   partitioning, openmm-ml, live-Context attach
  committee.py        batched MACE inference over the ML region
  uq_monitor.py       committee variance + geometry guard
  fallback_controller.py  the lambda_interpolate switch and its state machine
  active_learner.py   OOD buffer, clustering, labelling, fine-tuning
  runtime.py          the assembled surrogate the stages talk to
  somd2_hook.py       attaching inside a SOMD2 FEP leg
  testsystems.py      the shared OpenMM fixture

csbrt/src/csbrt/mace_pipeline.py        stage-script plumbing (isolated on purpose)
csbrt/src/csbrt/mace_preflight.py       csbrt-mace-preflight
csbrt/src/csbrt/mace_benchmark.py       csbrt-mace-benchmark
csbrt/src/csbrt/mace_active_learning.py csbrt-mace-al
csbrt/tests/                            334 tests

docs/mlff_delivery_report.md            this file
docs/mlff_decisions.md                  the decision register
docs/mlff_active_learning_architecture.md   component specification
docs/mlff_throughput_expectations.md    what it costs and what it buys
```
