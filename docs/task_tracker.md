# MACE Active Learning Pipeline – Task Tracker

## Stage 0: Environment & Dependency Manifest [Sequential]
- [x] Update `csbrt/environment.yml` with `openmm-ml`, `openmm-torch`, `mace-torch`
- [x] Update `csbrt/install.sh` with pip install steps
- [x] Create `csbrt/scripts/preflight_mace.py` diagnostic

## Stage 1: Fast Path Engine – Hybrid ML/MM Construction [Track A]
- [x] Create `csbrt/src/csbrt/mace_surrogate/__init__.py`
- [x] Create `csbrt/src/csbrt/mace_surrogate/mace_mixed_system.py`

## Stage 2: Real-Time UQ Monitor [Track B]
- [x] Create `csbrt/src/csbrt/mace_surrogate/uq_monitor.py`

## Stage 3: Zero-Overhead Physics Fallback Engine [Sequential]
- [x] Create `csbrt/src/csbrt/mace_surrogate/fallback_controller.py`

## Stage 4: Active Learning Data Harvester & Fine-Tuning [Parallel w/ Stage 5]
- [x] Create `csbrt/src/csbrt/mace_surrogate/active_learner.py`

## Stage 5: csbrt Pipeline & HPC Integration [Sequential]
- [x] Modify `cli.py` – wire MACE surrogate CLI flags
- [x] Modify `ev71_loch_common.py` – hook MACE into `make_dynamics()` / `make_sampler()`
- [x] Modify `run_fep_leg.py` – support surrogate in SOMD2 legs
- [x] Modify `pipeline_utils.py` – include MACE in checkpoint signatures
- [x] Modify `config.example.yaml` – add `mlff:` config block

## Stage 6: Verification & Benchmark Suite [Final]
- [x] Create `csbrt/tests/test_mace_mixed_system.py`
- [x] Create `csbrt/tests/test_uq_monitor.py`
- [x] Create `csbrt/tests/test_fallback_controller.py`
- [x] Create `csbrt/tests/test_active_learner.py`
- [x] Create `csbrt/tests/test_config_integration.py`
- [x] Create `csbrt/scripts/benchmark_mace_speedup.py`
- [x] Execute unit tests (14/14 tests passing)
