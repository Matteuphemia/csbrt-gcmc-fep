# euph1/euph2/euph4 implementation comparison

This note compares the three implementation branches against `main`, against
each other, and against `feature/euph3` as the planning reference only.

## Scope reviewed

- `origin/main`
- `origin/feature/euph1`
- `origin/feature/euph2`
- `origin/feature/euph3` (plan only)
- `origin/feature/euph4`

The comparison was based on branch diffs, implementation/test file inventories,
branch-specific documentation, and the branch-local tests that could be run in
this environment.

## Executive summary

| Branch | Relative to plan | Strengths | Main gaps / risks | Verdict |
| --- | --- | --- | --- | --- |
| `feature/euph1` | Most expansive; goes well beyond the plan | Deepest implementation, most tests, strongest design separation, best delivery documentation | Very large delta from `main`; tests currently depend on `openmm` at collection time in this environment; biggest review burden | Strongest raw implementation, but highest merge/review cost |
| `feature/euph2` | Closest balance of plan coverage and manageable scope | Covers the core MACE surrogate path, adds preflight/benchmark scripts, has a practical integration runbook, passed targeted tests here | Less complete verification/integration coverage than `euph1`; still documents remaining engine-level work | Best merge-forward base |
| `feature/euph4` | Lightest implementation of the plan | Smallest implementation branch; targeted tests passed here | Thinnest docs, least verification, and fewer integration abstractions | Useful reference, not the preferred merge target |

## What changed versus `main`

### `feature/euph1`

- Approximate scope: `51` changed files, `11153` insertions.
- Adds the widest MACE surface area:
  - pipeline/wiring helpers: `csbrt/src/csbrt/mace_pipeline.py`
  - runtime/config split: `csbrt/src/csbrt/mace_surrogate/config.py`,
    `runtime.py`, `committee.py`, `somd2_hook.py`, `units.py`,
    `testsystems.py`
  - stage helpers: `csbrt/src/csbrt/mace_active_learning.py`,
    `mace_benchmark.py`, `mace_preflight.py`
  - broad test suite: `csbrt/tests/test_pipeline_wiring.py`,
    `test_runtime.py`, `test_committee.py`, `test_energy_conservation.py`,
    `test_active_learning_loop.py`, plus the shared core unit tests
  - strongest delivery docs: `docs/mlff_delivery_report.md`,
    `docs/mlff_decisions.md`, `docs/mlff_throughput_expectations.md`

### `feature/euph2`

- Approximate scope: `35` changed files, `3613` insertions.
- Keeps the core plan modules while staying much closer to the original layout:
  - core implementation: `csbrt/src/csbrt/mace_surrogate/active_learner.py`,
    `ensemble_evaluator.py`, `fallback_controller.py`,
    `mace_mixed_system.py`, `uq_monitor.py`
  - operational scripts: `csbrt/scripts/preflight_mace.py`,
    `csbrt/scripts/benchmark_mace_speedup.py`
  - focused integration docs: `docs/mace_surrogate_integration.md`
  - focused tests: `test_active_learner.py`, `test_config_integration.py`,
    `test_ensemble_evaluator.py`, `test_fallback_controller.py`,
    `test_mace_mixed_system.py`, `test_uq_monitor.py`

### `feature/euph4`

- Approximate scope: `32` changed files, `2551` insertions.
- Implements the core plan path with the smallest surface area:
  - core implementation: `csbrt/src/csbrt/mace_surrogate/active_learner.py`,
    `fallback_controller.py`, `mace_mixed_system.py`, `uq_monitor.py`
  - operational scripts: `csbrt/scripts/preflight_mace.py`,
    `csbrt/scripts/benchmark_mace_speedup.py`
  - lightweight docs: `docs/walkthrough.md`, `docs/task_tracker.md`
  - lightest test set: five test modules around config/core logic

## Comparison against the `feature/euph3` plan

The `feature/euph3` plan called for:

1. environment/install changes
2. mixed-system construction
3. UQ monitoring
4. fallback control
5. active-learning harvesting
6. pipeline/CLI integration
7. verification and benchmarking

### `feature/euph1`

- Exceeds the plan in modularization and supporting infrastructure.
- Adds dedicated runtime, committee, config, unit-conversion, and pipeline
  layers that are not explicitly split out in `euph3`.
- Best coverage of Stage 5/6 concerns through wiring, runtime, and benchmark
  oriented files.
- Diverges the most from the plan in shape and size, so reviewing and merging
  it would cost the most.

### `feature/euph2`

- Tracks the plan more directly than `euph1` while still adding the practical
  preflight and benchmark scripts that `euph4` also adds.
- The runbook in `docs/mace_surrogate_integration.md` is the clearest statement
  of what is implemented now versus what still needs upstream Loch/SOMD2
  engine-level hooks.
- Covers the plan's main deliverables without the broader refactor introduced
  in `euph1`.

### `feature/euph4`

- Closest to a minimal implementation of the plan.
- Still lacks the stronger verification/reporting story that appears in
  `euph1`, and the clearer operational runbook that appears in `euph2`.
- Feels more like a partial implementation plus walkthrough than the branch to
  merge directly.

## Branch-to-branch differences

### `feature/euph1` vs `feature/euph2`

- `euph1` adds many more abstractions (`config.py`, `runtime.py`,
  `committee.py`, `somd2_hook.py`, `mace_pipeline.py`) and substantially more
  tests/documentation.
- `euph2` is narrower and easier to reason about, but has less end-to-end
  wiring verification than `euph1`.

### `feature/euph2` vs `feature/euph4`

- Both share a similar core shape, but `euph2` is clearly more complete:
  - extra evaluator module: `ensemble_evaluator.py`
  - extra test coverage
  - stronger operational documentation in
    `docs/mace_surrogate_integration.md`

### `feature/euph1` vs `feature/euph4`

- `euph1` is effectively a larger productized delivery.
- `euph4` is substantially smaller, but it omits many of the design,
  verification, and maintenance aids that make `euph1` easier to trust long
  term.

## Test and verification observations

The following branch-local tests were exercised from temporary worktrees:

- `feature/euph2`: `43 passed` with
  `PYTHONPATH=src python -m pytest tests/test_config_integration.py tests/test_active_learner.py tests/test_fallback_controller.py tests/test_mace_mixed_system.py tests/test_uq_monitor.py tests/test_ensemble_evaluator.py -q`
- `feature/euph4`: `14 passed` with
  `PYTHONPATH=src python -m pytest tests/test_config_integration.py tests/test_active_learner.py tests/test_fallback_controller.py tests/test_mace_mixed_system.py tests/test_uq_monitor.py -q`
- `feature/euph1`: collection in this environment stops at
  `csbrt/tests/conftest.py` because it imports `openmm` with
  `pytest.importorskip(...)` at module import time. That means the branch has
  the richest test inventory, but its tests are less portable in a lightweight
  environment than the `euph2`/`euph4` suites.

## Code quality and correctness review

### `feature/euph1`

- **Pros:** richest separation of concerns, most evidence of deliberate design,
  strongest verification surface.
- **Cons:** much larger than the other implementations, so regression risk is
  concentrated in review/merge complexity rather than obvious missing pieces.
- **Notable caution:** test portability is weaker in this environment because of
  the eager `openmm` import in `tests/conftest.py`.

### `feature/euph2`

- **Pros:** best balance of completeness, clarity, and reviewability; includes
  practical operator-facing documentation about remaining gaps.
- **Cons:** not as comprehensively verified as `euph1`; still leaves the real
  engine-hook problem explicitly unresolved.
- **Overall:** strongest reliability/maintainability trade-off.

### `feature/euph4`

- **Pros:** concise implementation, low surface area.
- **Cons:** least evidence for correctness beyond core unit-level behavior, and
  the weakest documentation of production-readiness limits.
- **Overall:** acceptable prototype, weaker merge candidate.

## Recommended merge strategy

Recommended path:

1. **Use `feature/euph2` as the merge-forward base.**
   - It is substantially closer to the `euph3` plan than `euph1` in shape.
   - It passed the available targeted tests in this environment.
   - It includes the practical runbook documenting what remains unresolved.
2. **Cherry-pick selected review/docs/test ideas from `feature/euph1`.**
   - In particular, its stronger delivery documentation and broader wiring
     tests are worth preserving if this work continues.
3. **Do not merge `feature/euph4` forward as-is.**
   - Keep it as a comparative implementation reference only.

## Regressions or missing pieces relative to the plan

Across all three implementation branches, the same strategic gaps remain:

- no demonstrated end-to-end production hookup that fully replaces the underlying
  Loch/SOMD2 context ownership in real runs
- no completed Stage 6 scientific parity package showing benchmarked
  `ΔΔG` agreement and the claimed throughput outcome on production-like systems
- no single branch that simultaneously provides the narrow scope of `euph2` and
  the richer verification/reporting envelope of `euph1`

That makes `feature/euph2` the safest branch to advance first, with `euph1`
serving as the best source of follow-on hardening work.
