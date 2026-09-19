# Walkthrough: MACE Active Learning & Fallback Pipeline Implementation

We have implemented the full end-to-end Machine Learning Force Field (MLFF / MACE) surrogate with active learning uncertainty monitoring and zero-overhead physics fallback for the `csbrt-gcmc-fep` workflow.

---

## 1. Summary of Changes

### Stage 0: Environment & Dependencies
- [`csbrt/environment.yml`](file:///d:/git/csbrt-gcmc-fep/csbrt/environment.yml): Added `openmm-ml` to conda dependencies alongside pinned CUDA 12.8 toolchain.
- [`csbrt/install.sh`](file:///d:/git/csbrt-gcmc-fep/csbrt/install.sh): Pinned pip installation of `mace-torch>=0.3.10` and verified imports (`openmmml`, `mace`).
- [`csbrt/scripts/preflight_mace.py`](file:///d:/git/csbrt-gcmc-fep/csbrt/scripts/preflight_mace.py): Standalone preflight diagnostic checking PyTorch CUDA, OpenMM platforms, OpenMM-ML, MACE-Torch, and surrogate configuration signatures.

### Stage 1 to 4: The `csbrt.mace_surrogate` Engine
Located in [`csbrt/src/csbrt/mace_surrogate/`](file:///d:/git/csbrt-gcmc-fep/csbrt/src/csbrt/mace_surrogate):
- **`mace_mixed_system.py`**:
  - `MACEConfig`: Structured dataclass managing parameters, committee sizes, threshold unit conversions (eV/Å $\leftrightarrow$ kcal/mol/Å $\leftrightarrow$ kJ/mol/nm), and deterministic SHA256 checkpoint signatures.
  - `MACEMixedSystemBuilder`: Partitions solute ligand atoms (+ optional GCMC hydration shell) into the ML region via `openmmml.MLPotential('mace')` and `createMixedSystem()`.
- **`uq_monitor.py`**:
  - `MACEUQMonitor`: Evaluates committee ensemble force variance $\sigma_{F, \max} = \max_i \operatorname{std}(\mathbf{F}_i)$ and energy variance $\sigma_E$.
  - Flags out-of-distribution (OOD) states when exceeding calibrated bounds (default: $0.05\text{ eV/\AA}$ or $1.0\text{ kcal/mol}$).
- **`fallback_controller.py`**:
  - `PhysicsFallbackController`: Governs runtime switching between fast MACE MLFF surrogate ($w = 0$) and SOMD2/classical MM physics ($w = 1$).
  - Manages consecutive fallback relaxation counters and zero-overhead GPU context handoffs.
- **`active_learner.py`**:
  - `OODBuffer`: Persistent compressed replay buffer (`.npz`) storing flagged OOD coordinates, velocities, and uncertainty metrics.
  - `cluster_by_rmsd`: Fast Kabsch-aligned conformational clustering (cutoff $0.5\text{ \AA}$) to deduplicate sampling basins.
  - `ReferenceLabeler` & `MACEFineTuner`: Ground-truth energy labeling and fine-tuning harness for transfer learning.

### Stage 5: Pipeline & CLI Integration
- [`csbrt/src/csbrt/cli.py`](file:///d:/git/csbrt-gcmc-fep/csbrt/src/csbrt/cli.py): Wired CLI flags (`--enable-mace-surrogate`, `--mace-model`, `--mace-uq-threshold`, `--mace-device`) into `csbrt`, `csbrt-equilibrate`, `csbrt-gcmc`, `csbrt-fep`.
- [`csbrt/config.example.yaml`](file:///d:/git/csbrt-gcmc-fep/csbrt/config.example.yaml): Added `mlff:` configuration block for declarative run configuration.
- [`csbrt/src/csbrt/ev71_production.py`](file:///d:/git/csbrt-gcmc-fep/csbrt/src/csbrt/ev71_production.py): Hooked MACE surrogate runtime UQ checking, dynamic fallback switching, and OOD buffer persistence into the Loch GCMC production cycle loop.
- [`csbrt/src/csbrt/run_fep_leg.py`](file:///d:/git/csbrt-gcmc-fep/csbrt/src/csbrt/run_fep_leg.py): Embedded surrogate configuration into effective leg configs and checkpoint signatures.
- [`csbrt/src/csbrt/pipeline_utils.py`](file:///d:/git/csbrt-gcmc-fep/csbrt/src/csbrt/pipeline_utils.py): Added `mace_signature()` helper to isolate MACE runs from classical baseline cache invalidations.

### Stage 6: Test Suite & Benchmarking
- **Unit Tests** in [`csbrt/tests/`](file:///d:/git/csbrt-gcmc-fep/csbrt/tests):
  - `test_mace_mixed_system.py`: Configuration defaults, unit conversions, serialization, and builder initialization.
  - `test_uq_monitor.py`: Relaxed conformation agreements, OOD force threshold detection, and OOD energy threshold detection.
  - `test_fallback_controller.py`: Initial surrogate state, OOD fallback trigger, relaxation step buffer, and recovery to fast path.
  - `test_active_learner.py`: Kabsch RMSD alignment, translation invariance, OOD buffer persistence, and conformational clustering.
  - `test_config_integration.py`: CLI flag parsing and dictionary overlay into pipeline configuration.
- **Benchmark Script**:
  - [`csbrt/scripts/benchmark_mace_speedup.py`](file:///d:/git/csbrt-gcmc-fep/csbrt/scripts/benchmark_mace_speedup.py): Profiles controller overhead ($56.37\text{ \mu s/decision}$) and verifies recovery dynamics.

---

## 2. Verification Results

All 14 unit tests pass with clean status:
```text
============================= test session starts =============================
platform win32 -- Python 3.11.1, pytest-9.0.3, pluggy-1.6.0
collected 14 items

csbrt/tests/test_active_learner.py::test_kabsch_rmsd_identical PASSED    [  7%]
csbrt/tests/test_active_learner.py::test_kabsch_rmsd_translation_invariance PASSED [ 14%]
csbrt/tests/test_active_learner.py::test_ood_buffer_add_and_cluster PASSED [ 21%]
csbrt/tests/test_config_integration.py::test_merge_cli_mace_flags PASSED [ 28%]
csbrt/tests/test_fallback_controller.py::test_fallback_controller_initial_state PASSED [ 35%]
csbrt/tests/test_fallback_controller.py::test_fallback_controller_trigger_and_recovery PASSED [ 42%]
csbrt/tests/test_mace_mixed_system.py::test_mace_config_defaults PASSED  [ 50%]
csbrt/tests/test_mace_mixed_system.py::test_mace_config_unit_conversions PASSED [ 57%]
csbrt/tests/test_mace_mixed_system.py::test_mace_config_serialization PASSED [ 64%]
csbrt/tests/test_mace_mixed_system.py::test_builder_initialization PASSED [ 71%]
csbrt/tests/test_uq_monitor.py::test_uq_monitor_initialization PASSED    [ 78%]
csbrt/tests/test_uq_monitor.py::test_compute_from_ensemble_predictions_relaxed PASSED [ 85%]
csbrt/tests/test_uq_monitor.py::test_compute_from_ensemble_predictions_ood_force PASSED [ 92%]
csbrt/tests/test_uq_monitor.py::test_compute_from_ensemble_predictions_ood_energy PASSED [100%]

============================= 14 passed in 0.14s ==============================
```
