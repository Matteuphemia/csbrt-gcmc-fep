# MACE Surrogate Integration — Status & Runbook

Status date: 2026-09-19 (branch `euph2`). This document describes what is
implemented, how to verify it, and exactly what remains before the surrogate can
drive production GCMC/FEP on the cluster.

## What is implemented

| Piece | File | Notes |
|---|---|---|
| Config schema | `csbrt/src/csbrt/mace_surrogate/mace_mixed_system.py` (`MACEConfig`) | canonical field names, CLI aliases, validation, SHA-256 signature |
| ML/MM partitioning | `mace_mixed_system.py` (`partition_ml_atoms`) | ligand + binding-site waters (residue-based, robust to atom order) |
| Mixed-system builder | `mace_mixed_system.py` (`MACEMixedSystemBuilder`) | `openmmml.MLPotential(model).createMixedSystem(..., interpolate=True)`; verified against the openmm-ml API: model names (`mace-off23-small` etc., **not** `mace-omol-0`), `modelPath` constructor kwarg, `device`/`precision` ('single'/'double') via `createMixedSystem` kwargs |
| Committee inference | `ensemble_evaluator.py` (`MACEEnsembleEvaluator`) | loads foundation model + fine-tuned generations via mace-torch; evaluates ML region + 6 Å neighbour shell; raises `MACEEnsembleUnavailable` when the stack is absent (never fabricates predictions) |
| UQ monitor | `uq_monitor.py` (`MACEUQMonitor`) | $\sigma_{F,\max} = \max_i \sqrt{\frac{1}{M-1}\sum_m \|F_{i,m}-\bar F_i\|^2}$, $\sigma_E$; unit-safe (eV/Å, kcal/mol/Å, kJ/mol/nm); statistics + threshold calibration |
| Fallback controller | `fallback_controller.py` (`PhysicsFallbackController`) | state machine (surrogate ↔ fallback), recovery buffer, OOD payload dispatch, `lambda_interpolate = 1 - w` mapping onto the openmm-ml dual-Hamiltonian parameter |
| Active learning | `active_learner.py` | OOD buffer (npz), Kabsch-RMSD dedup (0.5 Å), reference labeler, fine-tuner harness (JSON manifest when torch is absent) |
| CLI/config wiring | `cli.py`, `config.example.yaml` (`mlff:`), `run_ev71_pipeline.py`, `run_fep_leg.py`, `ev71_production.py`, `pipeline_utils.py` | `--enable-mace-surrogate`, `--mace-model`, `--mace-uq-threshold`, `--mace-device`; checkpoint signatures include the MACE config hash |
| Diagnostics | `scripts/preflight_mace.py`, `scripts/benchmark_mace_speedup.py` | dependency/CUDA/single-point checks; classical vs ML/MM ns/day benchmark |
| Tests | `csbrt/tests/` (43 tests) | `pytest` green on CPU-only dev machines; GPU-dependent paths degrade gracefully |

## Dual-Hamiltonian fallback (zero-overhead)

The mixed system returned by `createMixedSystem(..., interpolate=True)` carries
the global parameter `lambda_interpolate`:

* `lambda_interpolate = 1.0` → ML/MM fast path (MACE on the ML region)
* `lambda_interpolate = 0.0` → pure classical MM

`PhysicsFallbackController.apply_to_openmm_context()` sets it with a single
`setParameter` call — no `Context` reconstruction, no CUDA JIT recompile. This is
the mechanism Stage 3 of the implementation plan specified.

## How to verify

```bash
cd csbrt

# 1. Unit tests (no GPU, no scientific stack required)
python -m pytest tests -v

# 2. On the GPU node: dependency + inference preflight
python scripts/preflight_mace.py --model mace-off23-small --device cuda

# 3. Throughput smoke benchmark (classical vs ML/MM ns/day)
python scripts/benchmark_mace_speedup.py --smoke --device cuda

# 4. Pipeline plumbing with the surrogate flag (needs a full run.yaml)
csbrt --from equilibrate --through gcmc --config run.yaml \
      --enable-mace-surrogate --mace-model mace-off23-small --dry-run
```

## Remaining work (cluster-side, requires the scientific stack)

1. **Engine hookup for GCMC/FEP.** Loch (`GCMCSampler`) and SOMD2 both build and
   own their OpenMM contexts from Sire systems; a MACE mixed system cannot be
   swapped in without replacing that construction. `make_dynamics()` documents
   the seam; `ev71_production.py` currently runs committee UQ + fallback
   bookkeeping beside the classical dynamics (it never fabricates UQ — if the
   committee cannot load, UQ is disabled with a loud log). The actual fast-path
   context needs either an upstream Loch/SOMD2 hook or a csbrt-side OpenMM
   driver that replaces the Sire dynamics.
2. **FEP leg surrogate.** `run_fep_leg.py` records `mlff` in the effective config
   and checkpoint signature; the SOMD2 lambda-window dynamics still need the
   same engine-level hookup as (1).
3. **Stage-6 validation.** Energy-conservation (NVE drift), synthetic-OOD
   intercept rate, ΔΔG parity within 0.3 kcal/mol vs SOMD2, and the ≈50%
   throughput target on benchmark edges (7DLI / EV71).
4. **Committee formation.** σ_F requires M ≥ 2 models. With only the foundation
   model, `ensemble_size == 1`, σ_F = 0, and fallback is disabled by design.
   Committees form from fine-tuned generations (`mace_finetuned_gen{k}.pt`).
5. **License check.** MACE-OFF/OMAT/OMOL foundation models carry the ASL
   (restrictive, non-commercial) license; openmm-ml logs this when loading.
   `mace-mpa-0-medium` and the polar models are more permissive — confirm the
   model choice with the CSO before production use.
