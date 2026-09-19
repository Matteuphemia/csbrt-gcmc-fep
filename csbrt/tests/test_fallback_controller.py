"""Unit tests for physics fallback controller."""

import pytest
import numpy as np

from csbrt.mace_surrogate import (
    EvaluatorMode,
    FallbackState,
    MACEConfig,
    MACEUQMonitor,
    PhysicsFallbackController,
    UQResult,
)


def test_fallback_controller_initial_state():
    cfg = MACEConfig(fallback_recovery_steps=5)
    uq = MACEUQMonitor(config=cfg)
    ctrl = PhysicsFallbackController(config=cfg, uq_monitor=uq)

    assert ctrl.state.current_mode == EvaluatorMode.MACE_SURROGATE
    assert ctrl.state.weight == 0.0
    assert ctrl.state.total_steps == 0


def test_fallback_controller_trigger_and_recovery():
    cfg = MACEConfig(fallback_recovery_steps=3)
    uq = MACEUQMonitor(config=cfg)
    recorded_ood = []

    ctrl = PhysicsFallbackController(
        config=cfg,
        uq_monitor=uq,
        on_ood_frame=lambda f: recorded_ood.append(f),
    )

    # Step 1: Confident frame
    uq_normal = UQResult(
        sigma_f_max_ev_per_ang=0.01,
        sigma_f_max_kcal_per_mol_ang=0.23,
        sigma_f_max_kj_per_mol_nm=96.48,
        sigma_e_ev=0.01,
        sigma_e_kcal_per_mol=0.23,
        sigma_e_kj_per_mol=0.96,
        is_ood=False,
    )
    mode, w = ctrl.decide_state(uq_normal, {"step": 1})
    assert mode == EvaluatorMode.MACE_SURROGATE
    assert w == 0.0
    assert len(recorded_ood) == 0

    # Step 2: OOD frame -> triggers fallback
    uq_ood = UQResult(
        sigma_f_max_ev_per_ang=0.08,
        sigma_f_max_kcal_per_mol_ang=1.84,
        sigma_f_max_kj_per_mol_nm=771.88,
        sigma_e_ev=0.1,
        sigma_e_kcal_per_mol=2.3,
        sigma_e_kj_per_mol=9.6,
        is_ood=True,
        trigger_reason="Force uncertainty: 0.0800 eV/A",
    )
    mode, w = ctrl.decide_state(uq_ood, {"step": 2, "positions": np.zeros((10, 3))})
    assert mode == EvaluatorMode.CLASSICAL_PHYSICS
    assert w == 1.0
    assert len(recorded_ood) == 1
    assert ctrl.state.total_fallback_events == 1

    # Steps 3-4: Recovery buffer (requires 3 steps of fallback before checking again)
    # s=3: consecutive=1 -> CLASSICAL
    # s=4: consecutive=2 -> CLASSICAL
    for s in (3, 4):
        mode, w = ctrl.decide_state(uq_normal, {"step": s})
        assert mode == EvaluatorMode.CLASSICAL_PHYSICS
        assert w == 1.0

    # Step 5: consecutive=3 >= recovery_steps(3) -> recovers to MACE_SURROGATE
    mode, w = ctrl.decide_state(uq_normal, {"step": 5})
    assert mode == EvaluatorMode.MACE_SURROGATE
    assert w == 0.0
