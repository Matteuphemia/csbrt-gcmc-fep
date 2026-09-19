# Architecture Specification: MACE Active Learning & Fallback Loop for `csbrt`

**Target Systems:** Loch GCMC (`ev71_production.py`) & SOMD2 Relative FEP (`run_fep_leg.py`)  
**Foundational References:** Duignan (2024, `docs/duignan2024_potential_nnps.pdf`) & Wang et al. (2024, `docs/wang2024_design_space_mm_mlff.pdf`)  
**Implementation Package:** `csbrt.mace_surrogate`

---

## 1. System Design & Algorithmic Loop

MACE (Multi-Atomic Cluster Expansion) evaluates the perturbable ligand inside
OpenMM while the protein and bulk solvent stay on Amber ff14SB / TIP3P, an
uncertainty check runs alongside it, and any frame the model is not confident
about is integrated on the original classical Hamiltonian instead and harvested
for fine-tuning.

> The brief's objective for this work is a ~50% reduction in simulation time.
> **That is not what a hybrid ML/MM surrogate does.** It adds a neural-network
> evaluation on top of nearly all of the classical cost, so it is slower, and
> the paper this design is grounded in says so directly. What it buys is
> quantum-quality energetics for the ligand's internal degrees of freedom —
> exactly the part alchemical FEP is most sensitive to. Section 6, and
> `docs/mlff_throughput_expectations.md` in full, set out the measurements and
> where a 50% reduction could actually come from.

```
                    +------------------------------------------+
                    |          MD Coordinate State X_t         |
                    +------------------------------------------+
                                         |
                                         v
                    +------------------------------------------+
                    |        MACE Ensemble / Committee         |
                    |   E_m(X_t), F_{i,m}(X_t) for m=1..M      |
                    +------------------------------------------+
                                         |
                                         v
                    +------------------------------------------+
                    |    Uncertainty Metric Evaluation         |
                    |  sigma_F = max_i std(F_{i,m})            |
                    |  sigma_E = std(E_m)                      |
                    +------------------------------------------+
                                         |
                       +-----------------+-----------------+
                       |                                   |
         [sigma_F <= sigma_thresh]               [sigma_F > sigma_thresh]
                       |                                   |
                       v                                   v
        +-----------------------------+    +-----------------------------+
        |        FAST PATH            |    |       PHYSICS FALLBACK      |
        | Apply MACE forces to        |    | Switch force evaluator to   |
        | OpenMM integrator.          |    | SOMD2 / Classical MM.       |
        | Advance time-step t -> t+dt |    | Log frame to OOD Buffer.    |
        +-----------------------------+    +-----------------------------+
                       |                                   |
                       +-----------------+-----------------+
                                         |
                                         v
                    +------------------------------------------+
                    |        Check Simulation Completion       |
                    +------------------------------------------+
                                         |
                        (Post-Simulation / Background)
                                         v
                    +------------------------------------------+
                    |     Active Learning Feedback Pipeline    |
                    | 1. Cluster & deduplicate OOD frames.     |
                    | 2. Compute ground-truth labels.          |
                    | 3. Fine-tune MACE foundation model.      |
                    | 4. Update surrogate weights for run.     |
                    +------------------------------------------+
```

---

## 2. Component Specifications

Implemented in `csbrt/src/csbrt/mace_surrogate/`. Every module below is covered
by the tests in `csbrt/tests/`, which run against real OpenMM on CPU in a few
seconds; the ones that need downloaded MACE weights are gated behind
`CSBRT_MACE_MODEL_TESTS=1`.

### 2.1 Hybrid ML/MM system generator (`mace_mixed_system.py`)

- **Library:** `openmmml.MLPotential`. The potential is named by the *model*
  (`MLPotential('mace-off23-small')`); `MLPotential('mace', modelPath=...)` is
  only for a locally trained or fine-tuned checkpoint.
- **Partitioning:** `partition_ml_atoms()` takes whole residues. It raises if
  the ligand residue is absent or appears more than once — a silently empty ML
  region is a surrogate that claims to be running and is not — and it excludes
  any particle with zero charge *and* zero LJ epsilon, which is how Loch
  represents its ghost-water buffer. MACE must never see a water the MM side
  believes does not exist.
- **Waters are out of the ML region by default.** The plan proposed including
  the GCMC sphere's waters. The ML atom list is baked into the ML force when
  the Context is built and cannot follow Loch's exchanges, and a fixed ML water
  set would both drift out of the site and break the ghost bookkeeping. The
  option remains (`include_binding_site_waters`) with a warning, for a system
  whose site waters are known to be buried. Keeping waters out also means the
  GCMC acceptance criterion is untouched by the surrogate, since no water's
  energy passes through MACE.
- **Applying to a live Context:** Sire, Loch and SOMD2 all build their own
  Context. `attach_mace_to_context()` replaces the Forces in the live System
  and calls `reinitialize(preserveState=True)` — the same pattern
  `ev71_loch_common.add_ca_restraints` already uses. Positions, velocities and
  box vectors survive; the test suite asserts it.
- **Size guard:** `max_ml_atoms` (250 by default) refuses a region large enough
  to hit the pure-MLFF scalability trap.

### 2.2 Uncertainty monitor (`uq_monitor.py`, `committee.py`)

Two independent detectors; either firing trips the fallback.

**Committee force variance**, the standard active-learning signal:

$$\sigma_{F,\max} = \max_i \Big[\tfrac{1}{M-1}\sum_m \lVert \mathbf{F}_{i,m} - \bar{\mathbf{F}}_i\rVert^2\Big]^{1/2}, \qquad \sigma_E = \Big[\tfrac{1}{M-1}\sum_m (E_m - \bar{E})^2\Big]^{1/2}$$

Default trigger 0.05 eV/Å (≈ 1.15 kcal/mol/Å); `calibrate_thresholds()` resets
it from a high percentile of an in-distribution trajectory, which is the
defensible way to set it per target. Inference goes through
`MACECalculator`'s own `forces_comm` / `energy_comm`, batched in one forward
pass, rather than a reimplementation.

> **A committee cannot be assembled from different foundation model sizes.**
> `MACECalculator` requires every member to share the same cutoff radius, and
> `mace-off23-small` and `-medium` do not. (On mace-torch 0.3.16 that check
> raises a `TypeError` while formatting its own error message, hiding the
> cause; `MACECommittee` catches it and explains.) A real committee comes from
> independent fine-tunes of **one** foundation model, which is what
> `csbrt-mace-al finetune --seeds N` produces.

**Geometry guard.** A committee only measures disagreement, and models trained
on the same data agree confidently on geometries none of them ever saw. The
guard checks the ML region for atoms closer than 0.70 Å and for covalent bonds
stretched past 1.6x the sum of covalent radii. It is O(N²) over 40–250 atoms,
costs microseconds, needs no second model, and is therefore the *only* OOD
detector available before active learning has produced a committee. A run with
no committee is not unguarded, but it is not fully guarded either, and the
runtime logs which of the two it is.

### 2.3 Physics fallback (`fallback_controller.py`)

`createMixedSystem(..., interpolate=True)` gives the System a global parameter
`lambda_interpolate`:

$$U(\lambda) = \lambda\, U_{\mathrm{MACE/MM}} + (1-\lambda)\, U_{\mathrm{MM}}$$

Switching is one `Context.setParameter` call — **measured at 0.7 µs** against
the 200–500 ms a Context rebuild costs. Positions and velocities are untouched,
so the trajectory is continuous through the transition.

The state machine: on a trigger, drop to λ = 0, record the frame to the OOD
buffer once per *event* (not per evaluation), run `fallback_steps` classical
steps, then retest. If more than `fallback_abort_fraction` of the run has gone
to the classical Hamiltonian the surrogate is not paying for itself, so the
controller latches classical and says so.

> **The fallback does not save time.** A `CustomCVForce` evaluates every
> collective variable whatever its coefficient, so MACE runs at λ = 0 too. The
> fallback buys correctness. See `docs/mlff_throughput_expectations.md`.

`lambda_interpolate = 0` is asserted by the test suite to reproduce the
untouched classical energy — to 4.5e-8 kJ/mol with a real MACE-OFF23 model —
and to reproduce the classical *trajectory*, frame for frame, not just one
energy. That equality is what the whole fallback rests on.

### 2.4 Active learning (`active_learner.py`, `csbrt-mace-al`)

- **Harvester.** One `.npz` per worker under `al_buffer/`, written atomically:
  52 edges x replicates write concurrently from separate Slurm tasks and a
  shared file would race. Only the ML region is stored, in Å. When the
  per-worker cap is hit the *least* uncertain frame is dropped, not the oldest,
  so the set does not bias toward the start of the trajectory.
- **Clustering.** Greedy heavy-atom RMSD with Kabsch alignment, 0.5 Å cutoff;
  frames of different molecules are never compared. The representative of a
  basin is its highest-uncertainty member.
- **Labelling.** `ReferenceLabeler` takes any single-point callable.
  `csbrt-mace-al label --labeller command` shells out to a QM code; the
  contract is one JSON object with `energy_ev` and `forces_ev_per_ang`. An
  MM labeller is shipped for validating the pipeline end to end.
- **Level of theory is recorded on every frame and never mixed.** Fine-tuning
  refuses MM-labelled data unless `--allow-mm-labels` is passed: MACE-OFF is
  fitted to wB97M-D3(BJ)/def2-TZVPPD, and training it on force-field labels
  would replace its quantum accuracy with the force field the surrogate exists
  to avoid.
- **Fine-tuning** shells out to `mace_run_train --foundation_model`, MACE's own
  supported path, with a conservative learning rate (1e-4) and early stopping.
  `--seeds N` trains N models from the same foundation model, which is how a
  usable committee gets made.

---

## 3. Integration points in `csbrt`

| Where | What happens |
|---|---|
| `pipeline_utils.add_mace_arguments` | The flags are declared **once**. `mace_command_arguments` renders them back into a command line, so `csbrt` → `run_ev71_pipeline` → `ev71_production` forwards exactly what it was given and a 52-edge network cannot run two versions of the physics. |
| `config.example.yaml` | The `mlff:` block. CLI flags override it; a flag left unset does not touch it. |
| `ev71_loch_common.attach_mace_surrogate` | Called **before** `sampler.bind_dynamics(dynamics)`. Attaching swaps the Forces in the live System, and Loch resolves the NonbondedForce it toggles ghost waters through when it binds; binding first would leave it mutating an orphaned Force. Afterwards the ghost set and the NonbondedForce count are re-checked and the run aborts if either moved. |
| `ev71_loch_common.run_with_csv_reports` | Takes an optional `surrogate` and subdivides the MD at its UQ interval. With none it is byte-for-byte what it was, so existing checkpoints still validate. |
| `run_fep_leg.py` + `mace_surrogate/somd2_hook.py` | SOMD2 validates its own config keys and rejects unknown ones, so the surrogate travels as a JSON sidecar plus a `sitecustomize` on the subprocess `PYTHONPATH`. The hook wraps `sire.system.System.dynamics` so each λ window's Context comes back mixed, and verifies Sire's atom order against the Context's particle order by comparing coordinates under the minimum image convention before attaching anything. |
| `pipeline_utils.mace_signature` | Returns `None` when the surrogate is off, so a classical run's checkpoint marker is unchanged by the existence of this package. When on, it records the model hash, thresholds and partition — everything that changes the sampled ensemble, and nothing that does not, so the same physics on a different GPU still matches its checkpoint. |
| `fep_edge.slurm`, `submit_fep_edges.sh` | `--with-mace` renders the flag list once on the frontend and exports it, so every array task runs the same physics. |

---

## 4. Running it

```bash
# 1. Check the node. Seconds, not thirty minutes into a 52-edge array.
csbrt-mace-preflight --model mace-off23-small --device cuda --json preflight.json

# 2. Measure what it costs on YOUR system before committing a campaign.
csbrt-mace-benchmark --prmtop complex.prmtop --rst7 complex.rst7 --steps 2000

# 3. Enable it.
csbrt --from equilibrate --through gcmc --config run.yaml --enable-mace-surrogate
./submit_fep_edges.sh --manifest fep_manifest.tsv --batch 24 --with-mace

# 4. Between generations, close the loop.
csbrt-mace-al report   --buffer-dir RUN/al_buffer
csbrt-mace-al harvest  --buffer-dir RUN/al_buffer --output frames.npz
csbrt-mace-al label    --frames frames.npz --labeller command \
                       --command "my_qm_singlepoint {xyz}" --output labelled.npz
csbrt-mace-al finetune --frames labelled.npz --generation 2 --seeds 2
# then add the two checkpoints to mlff.committee_model_paths
```

---

## 5. Validation before trusting a number

The surrogate changes the Hamiltonian. Nothing below is optional before a
production campaign.

1. **Preflight passes on the actual compute node**, with the actual model.
2. **Energy conservation.** `pytest csbrt/tests/test_energy_conservation.py`
   covers the mechanism; for a real complex, run NVE with the mixed system and
   confirm the drift per degree of freedom is comparable to the classical
   baseline, not merely small.
3. **ΔΔG parity on at least three benchmark edges.** Run them classically and
   with the surrogate, and compare. The plan's bar is 0.3 kcal/mol absolute
   deviation; treat a larger deviation as a finding about the surrogate, not
   about the classical reference.
4. **Fallback rate.** A run whose `fallback_fraction` is more than a few
   percent is telling you the model does not cover this chemistry. Harvest,
   fine-tune, and re-check before using its free energies.
5. **Licensing.** `mace-off23-*` is academic-use-only (ASL). See
   `docs/mlff_throughput_expectations.md` section 6.

---

## 6. Throughput

**The surrogate is slower than the classical force field, not faster.** The
architecture is sound and the implementation works; the ~50% time reduction in
the brief does not follow from it. `docs/mlff_throughput_expectations.md` sets
out why, what the surrogate buys instead, and where a 50% reduction could
actually come from in this pipeline.
