# Pipeline and script map

## Contents

- Active Loch workflow
- EV71 port in project_2
- General endpoint and FEP workflow
- Alternative launchers
- GRAND workflows
- Diagnostics and experiments
- Stage contracts

## Active Loch workflow

The authoritative from-source path documented in `README.md` is:

`scripts/loch_full_pipeline.slurm`
→ `scripts/prepare_ludovic_native.py`
→ `scripts/LochEquilibration.py`
→ `scripts/LochProduction.py`
→ `scripts/LochPostprocess.py`

`scripts/loch_ludovic_common.py` is shared by both active Loch simulation stages. It owns sampler/dynamics construction, protocol constants, velocity assignment, state CSV writing, Sire/OpenMM coordinate commits, and ghost-free handoff finalization.

`scripts/submit_loch_replicas.sh` submits independent instances of `loch_full_pipeline.slurm`, using seed blocks separated by 1000.

The default cluster project root is `$HOME/cry`; the default environment is `cry-loch-babel` from `cry-loch-babel.yml`.

## EV71 port in project_2

The reusable four-stage port lives in `/home/moshe/intern_projects/project_2`:

`scripts/ev71_full_pipeline.slurm`
→ `scripts/run_ev71_pipeline.py`
→ `scripts/extract_ligands.py`
→ `scripts/prepare_ev71_system.py`
→ `scripts/ev71_equilibrate.py`
→ `scripts/ev71_production.py`
→ `scripts/ev71_postprocess.py`

`scripts/audit_ev71_pipeline.py` runs after every external boundary. `scripts/ev71_loch_common.py` owns the matched physical protocol and Loch 2025.2 handoff materialization. The inputs actually present are the prepared receptor and 32-record Rowan docked SDF under `openbind_ev71_2a_pyrrolidine_benchmark_release`; the default validation ligand is exact record `x7259a`.

`scripts/ev71_density_sites.py` is the scalable Project 2 analysis companion.
It consumes the raw production topology, DCD, and synchronized ghost history;
aligns to a shared receptor reference; accumulates all physical-water oxygen
observations on a 3D grid; discovers or reuses a common site catalog; and
reports site occupancy, block uncertainty, ligand overlap, and geometric
protein-water-ligand bridges. Keep `ev71_postprocess.py` for exact Ludovic
clustering parity and crystallographic-water validation; do not describe its
O(N^2) hierarchy as the required chemical-series analysis.
`scripts/ev71_merge_site_catalogs.py` pools provisional catalogs in the common
receptor frame using complete linkage, records input-catalog support, and emits
the frozen catalog supplied back to every per-run density analysis.

The EV71 chemical-series launcher is:

`scripts/submit_ev71_density_series.sh`
→ `scripts/ev71_make_series_manifest.py`
→ Slurm array of `scripts/ev71_density_series_task.slurm`
→ `run_ev71_pipeline.py --through production`
→ `ev71_density_sites.py` (provisional sites)
→ dependent `scripts/ev71_finalize_density_series.slurm`
→ `scripts/ev71_finalize_density_series.py`
→ common catalog and common-catalog reanalysis of every task.

The manifest enumerates exact SDF titles, creates one row per ligand/replica,
and assigns deterministic seed blocks spaced by 1000. The submitter defaults to
six replicas and bounded GPU-array concurrency, but input folder, explicit
receptor/library, replica count, run root, resources, and scheduler account can
be changed without editing source.

Series manifests explicitly use LF line endings, and the worker strips a
trailing carriage return defensively. The finalizer also recognizes directories
created by the original CRLF manifest bug and atomically renames `repN\r` to
`repN` after the GPU array finishes but before common-catalog analysis. It fails
closed if both clean and legacy names exist.

The `full` profile passes every canonical count explicitly. The `smoke` profile is plumbing-only. Audit scope is `smoke_plumbing_only` for reduced counts, `full_ludovic_schedule` for canonical counts plus stride-1 clustering, and `full_simulation_approximate_postprocessing` when a full simulation uses `cluster_stride > 1`. Preparation, UVT1, NPT, UVT2, production, and postprocessing use source/protocol/runtime/output-hashed checkpoints. The postprocessor intentionally images before shifting and explicitly masks inactive ghosts, avoiding GRAND's inherited shift-before-image hazard. Read `porting.md` before adapting this implementation to another receptor or ligand library.

## Unified single-environment package (csbrt)

`project_2/csbrt/` packages the whole workflow behind one driver and, critically,
one conda environment:

`scripts/csbrt.py --all|--from|--through` over five stages:
`preprocess` (OpenFold3 loop modelling -> graft -> PDB2PQR/PROPKA protonation ->
explicit-hydrogen ligand prep) -> `equilibrate` -> `gcmc` (Loch production +
`ev71_density_sites.py`) -> `fep` -> `analysis`. Stage names in the EV71 driver
are `preparation/equilibration/production/postprocessing` -- there is no
`analysis` stage there, unlike `run_pipeline.py` in `automated_pipeline`.

Environment constraints, all verified on an RTX 2080 Ti (July 2026):

- **The structure predictor must be OpenFold3.** `boltz` requires `numpy<2.0` and
  `chai_lab` requires `numpy~=1.21`, while sire/somd2/loch/mdtraj are built
  against numpy 2.x; installing either downgrades numpy and breaks somd2 with
  `No module named 'numpy.core.multiarray'`. OpenFold3 leaves numpy unpinned and
  still supports explicit mmCIF templates (`template_cif_paths`, "CIF-direct")
  plus ColabFold MSAs, so it is the only one that fits in one environment.
- **Pin the whole CUDA toolchain**, not just `cuda-version`; then add a CUDA
  torch from pip (self-contained wheels) for OpenFold3. pip's cu126 wheels do not
  shadow the conda CUDA 12.8 libraries used by OpenMM/loch -- verified by running
  OpenMM CUDA, a Loch GCMC production and a SOMD2 leg in the same env.
- **OpenFold3 needs Ampere or newer (SM >= 8.0).** On Turing its Triton triangle
  kernels fail with `LLVM ERROR: Unsupported rounding mode for conversion`; pass a
  runner yaml setting `settings.memory.eval.use_triton_triangle_kernels: false`
  to fall back. Loch/SOMD2 have no such constraint.
- **Conda is mandatory; neither conda-forge nor pure PyPI can carry this stack.**
  The root cause is single: OpenBioSim ships `loch`, `somd2` and `biosimspace`
  through their own conda channel only, and AmberTools is conda-only too. That
  blocks conda-forge (which forbids depending on other channels or on pip) and
  equally blocks pure PyPI (`loch`, `somd2`, `biosimspace`, `ambertools` are all
  absent from PyPI; `sire`, `openmm`, `openfold3`, `rdkit`, `mdtraj`, `parmed`,
  `pdb2pqr` and `propka` are present, but the missing four are the engines).
  The workable shape is therefore: one conda environment for the scientific
  stack, plus a pure-Python CLI package that may be pip/PyPI-installable, plus a
  personal anaconda.org channel or pixi if a single install command is wanted.

Before trusting a swapped structure predictor, compare the modelled region
against the incumbent after superposing on the **non-modelled** scaffold, and
read the model's own per-residue confidence. Reporting the superposition RMSD of
the anchor atoms proves nothing about the modelled region.

## General endpoint and FEP workflow

The portable, explicitly bounded workflow is under
`scripts/automated_pipeline/`; read its `README.md` before use. Its endpoint
path is:

`run_pipeline.py`
→ `select_ligand.py`
→ `prepare_system.py`
→ `equilibrate.py`
→ `production.py`
→ `density_sites.py`.

`loch_common.py` preserves the Ludovic-order protocol and the Loch 2025.2/Sire
2025.4 physical handoff. `audit_pipeline.py` validates every endpoint boundary.

`production.py` accepts `--water-convergence {off,advisory,gated}` (default `off`,
which is byte-for-byte the prior behaviour). `advisory` writes
`PREFIX-water-convergence.json`; `gated` also stops early once the per-cycle
sphere-water count is stationary and records `completed_cycles` beside
`configured_cycles` so the checkpoint audit uses the actual cycle count.
`water_convergence.py` owns the metric: a stationarity/plateau test plus an
optional post-hoc Good-Turing unseen-state-mass signal computed from the
`occupied` array already written to `density_sites.py`'s
`PREFIX-frame-site-series.npz`. Two distinct observables feed the stationarity
test and the report records which one in its `observable` field: gating uses the
live `sampler.num_waters()` sphere-occupancy count (`sampler_num_waters`), while
the standalone CLI on a ghost file uses `45 - inactive` activated-buffer count
(`active_buffer_from_ghosts`). They usually correlate but can diverge (a smoke
run may leave every buffer ghost inactive while physical waters fill the sphere),
so do not compare reports across observables. It is heuristic; treat an early stop
as advisory.
`receptor_io.py` accepts prepared protein PDB or macromolecular CIF/mmCIF,
preserves the source, fingerprints Open Babel, and materializes the strict PDB
boundary consumed by preparation and alignment; ligand chemistry remains SDF.
`make_run_manifest.py`, `md_gcmc_task.slurm`, and
`submit_md_gcmc_series.sh` provide receptor/ligand-specific replica arrays. A
dependent `finalize_density_series.slurm` freezes one common site catalog in a
shared receptor frame and reanalyses every ligand/replica with
`finalize_density_series.py`.

For OpenBind releases, `prepare_openbind_dataset.py` discovers every event and
calls `merge_reference_pose.py`: reference SDF heavy coordinates are preserved
exactly while the prepared SDF supplies graph, charge, and hydrogens. Every
event retains its own receptor and ligand pose, so `resname LIG` centres Loch on
that event's pocket. This adapter does not establish that separate pockets form
one comparable thermodynamic network.

The relative-FEP path is separate from endpoint MD/GCMC:

`make_fep_manifest.py`
→ `prepare_fep.py` (BioSimSpace mapping/merge; bound and free `.bss` streams)
→ parallel `run_fep_leg.py` SOMD2 bound/free legs
→ `analyse_fep.py`
→ `aggregate_fep_network.py`.

`submit_fep_series.sh` and `fep_*.slurm` encode these dependencies. By default
`make_fep_manifest.py --bound-frame production` seeds each bound leg from state
A's water-equilibrated `*-production-final` restart and emits two extra manifest
columns (`state_a_bound_prmtop`, `state_a_bound_rst7`); `fep_prepare.slurm` then
passes them to `prepare_fep.py --bound-prmtop/--bound-rst7 --align-to-bound-pose`,
which RMSD-aligns the merged perturbable ligand onto that frame's ligand pose.
`--bound-frame preparation` restores the freshly-solvated frame and leaves the
columns empty. For a multi-replicate benchmark, `select_bound_frames.py` first
picks one medoid equilibrated frame per ligand (the replicate whose common-catalog
hydration-site occupancy is closest to the per-ligand mean; coordinate averaging
of indistinguishable waters is invalid), and `make_fep_manifest.py
--bound-frame-root` seeds every edge from those. `submit_fep_edges.sh` +
`fep_edge.slurm` then run the whole network as one throttled `gpu:1` array
(`--batch N` concurrency), each task doing prepare->bound->free->analyse, as an
alternative to `submit_fep_series.sh`'s per-stage dependency graph. Because those
waters are already placed, bound-leg GCMC is now **off by default**; pass
`--with-gcmc` to re-enable SOMD2's Loch GCMC with a ligand-centred sphere. The free leg never uses GCMC. FEP
runs in the separate `automated-fep` Mamba environment from `environment-fep.yml`
(SOMD2/OpenBioSim 2026.1 plus `cuda-nvvm`); never upgrade the validated
`cry-loch-babel` endpoint environment in place. FEP requires
reviewed same-site ligand edges, a substantial mapped core, bound and free
lambda schedules, overlap/convergence checks, and cycle/replicate agreement.
Existing endpoint trajectories contain no hidden FEP and need not be rerun just
because this pairwise branch was added.

## Alternative Loch launchers

| Wrapper | Purpose | Inputs | Notes |
| --- | --- | --- | --- |
| `loch_multi_env_full.slurm` | Full from-source run with separate preparation/Loch/post environments | Raw holo PDB | Most defensive environment and output checks; defaults to `cry-prep` then `cry-loch-babel`. |
| `loch_replica.slurm` | Equilibration and production from already prepared AMBER files | `prepared/ludovic_combined/CRY1AN139_solvated.{prmtop,inpcrd}` | Does not prepare or post-process. |
| `loch_prepare.slurm` | Preparation only | Raw holo PDB | Produces the prepared-input contract used by `loch_replica.slurm` and smoke jobs. |
| `loch_smoke.slurm` | One-cycle end-to-end Loch simulation smoke test | Prepared AMBER files | Uses reduced attempt/step counts; validates plumbing, not stochastic stability or scientific length. |

Older `loch_cry_smoke.py`, `loch_cry_smoke_corrected.py`, `loch_cry_smoke_ready.py`, `export_cry_for_loch.py`, and root `loch_cry_smoke.py` are explicitly labelled pre-Ludovic development history/benchmarks. They use the old OpenFF/GRAND-prepared input family, do not create physical stage handoffs, and are not protocol alternatives. Do not substitute them for the active pipeline.

## GRAND workflows

These are not the Loch pipeline:

| Path | Meaning |
| --- | --- |
| `equilibration.slurm` → `Equilibration.py`, then `production.slurm` → `Production.py` | Locally rewritten GRAND workflow using `prepared/complex_ghosts.pdb` and `prepared/initial_ghosts.txt` in `cry-gcmc`. |
| `ludovic_replica.slurm` | Copies and executes Ludovic's original external `GCMC_Grand/Equilibration.py` and `Production.py` in `cry-gcmc`. |

The helper/preparation scripts `common.py`, `build_complex.py`, `solvate.py`, `add_ghosts.py`, and `check_*` primarily support the local GRAND/OpenFF development path, not the native-AMBER Loch path.

## Diagnostics and experiments

| Script | Scope |
| --- | --- |
| `diagnose_direct_native_md.py` | Direct OpenMM native-AMBER MD control. |
| `diagnose_native_md.py` | Sire native-AMBER MD control. |
| `compare_native_constraints.py` | Direct OpenMM versus Sire constraint construction. |
| `diagnose_loch_handoff.py` | Short native → Loch → MD stability check. |
| `diagnose_uvt1_stability.py` / `.slurm` | Exact UVT1 stage tracing plus finalized/reloaded topology audit. |
| `diagnose_uvt2_stability.py` / `.slurm` | Five-way saved-NPT UVT2 failure isolation. |
| `compare_ludovic_loch_energy.py` | Identical-coordinate Ludovic/OpenMM versus Sire/Loch energy comparison. |
| `prepare_ludovic_loch.py` | Rebuilt Ludovic OpenMM-to-AMBER conversion used by parity tests and batch benchmarks; not the active native preparation. |
| `benchmark_loch_batch.py`, `run_loch_batch_sweep.sh` | Loch batch-size performance experiments. |
| `calibrate_tip3p_300.py`, `prepare_bulk_tip3p.py` | TIP3P calibration support. |
| `benchmark_grand_batches.py`, `profile_grand_move.py`, `test_barostat_transition.py`, `smoke_test_grand.py` | GRAND experiments; not Loch diagnostics. |

## Stage contracts

### Preparation

Input: combined experimental `CRY1AN139_HOLO.pdb` with residue `LIG`.

Required outputs for Loch equilibration:

- `CRY1AN139_solvated.prmtop`
- `CRY1AN139_solvated.inpcrd`

Do not use `an139_ligh.sdf` as the ligand chemistry source; its tautomer/bond assignment differs from the prepared Ludovic ligand.

### Loch equilibration

`LochEquilibration.py` runs:

1. UVT1: delete sphere waters, minimize, 10,000 initial attempts, then 100 × (1,000 attempts + 5 MD steps).
2. Handoff: materialize active accepted buffer waters, remove inactive waters/ghosts, save, reload, and reject any all-zero charge/LJ water.
3. NPT: 1,000,000 steps (2 ns) at 300 K/1 bar.
4. UVT2: add a fresh 45-water buffer, then 125 × (800 attempts + 2,000 MD steps).
5. Finalize, save, reload-audit, and expose `CRY1AN139uvt2.{prmtop,rst7,pdb}`.

Intermediate UVT1 and NPT AMBER files are diagnostic stage boundaries. Production consumes only the UVT2 AMBER pair.

### Loch production

`LochProduction.py` follows Ludovic's cycle order: 2,500 × (2,000 MD steps followed by 200 GCMC attempts), totaling 10 ns. It saves the raw DCD after each move, one matching inactive-ghost line per trajectory frame, state CSV, ghost-containing topology for trajectory interpretation, and a finalized physical restart.

### Post-processing

`LochPostprocess.py` consumes the ghost-containing production PDB, raw DCD, and per-frame ghost file. It shifts inactive ghosts, images, aligns, extracts the sphere, and clusters water oxygens in Ludovic/GRAND operation order.
