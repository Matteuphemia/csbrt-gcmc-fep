"""Tests for the zero-overhead physics fallback controller."""

from __future__ import annotations

import numpy as np
import pytest

from csbrt.mace_surrogate import (
    EvaluatorMode,
    MACEConfig,
    MACEUQMonitor,
    PhysicsFallbackController,
    UQResult,
)


def _uq(is_ood: bool) -> UQResult:
    return UQResult(
        sigma_f_max_ev_per_ang=0.2 if is_ood else 0.0,
        sigma_f_max_kcal_per_mol_ang=0.0,
        sigma_f_max_kj_per_mol_nm=0.0,
        sigma_e_ev=0.0,
        sigma_e_kcal_per_mol=5.0 if is_ood else 0.0,
        sigma_e_kj_per_mol=0.0,
        is_ood=is_ood,
        trigger_reason="force uncertainty high" if is_ood else None,
    )


class FakeContext:
    def __init__(self) -> None:
        self.params: dict[str, float] = {}

    def setParameter(self, name: str, value: float) -> None:
        self.params[name] = value


def test_initial_state_is_surrogate() -> None:
    ctrl = PhysicsFallbackController(MACEConfig())
    assert ctrl.state.current_mode == EvaluatorMode.MACE_SURROGATE
    assert ctrl.state.weight == 0.0


def test_fallback_triggered_on_ood() -> None:
    ctrl = PhysicsFallbackController(MACEConfig())
    mode, w = ctrl.decide_state(_uq(is_ood=True))
    assert mode == EvaluatorMode.CLASSICAL_PHYSICS
    assert w == 1.0
    assert ctrl.state.total_fallback_events == 1


def test_lambda_interpolate_mapping() -> None:
    ctrl = PhysicsFallbackController(MACEConfig())
    ctx = FakeContext()

    ctrl.state.weight = 0.0  # surrogate
    ctrl.apply_to_openmm_context(ctx)
    assert ctx.params["lambda_interpolate"] == pytest.approx(1.0)

    ctrl.state.weight = 1.0  # fallback
    ctrl.apply_to_openmm_context(ctx)
    assert ctx.params["lambda_interpolate"] == pytest.approx(0.0)


def test_recovery_after_recovery_steps() -> None:
    config = MACEConfig(fallback_recovery_steps=3)
    ctrl = PhysicsFallbackController(config)

    # Trigger fallback once.
    ctrl.decide_state(_uq(is_ood=True))
    assert ctrl.state.current_mode == EvaluatorMode.CLASSICAL_PHYSICS

    # First two in-distribution steps remain in fallback.
    for _ in range(2):
        mode, _ = ctrl.decide_state(_uq(is_ood=False))
        assert mode == EvaluatorMode.CLASSICAL_PHYSICS

    # Third in-distribution step recovers to surrogate.
    mode, w = ctrl.decide_state(_uq(is_ood=False))
    assert mode == EvaluatorMode.MACE_SURROGATE
    assert w == 0.0


def test_fallback_fraction_and_counts() -> None:
    config = MACEConfig(fallback_recovery_steps=1)
    ctrl = PhysicsFallbackController(config)
    # OOD -> fallback; next in-distribution step recovers.
    ctrl.decide_state(_uq(is_ood=True))
    ctrl.decide_state(_uq(is_ood=False))
    ctrl.decide_state(_uq(is_ood=False))  # surrogate step
    assert ctrl.state.total_fallback_steps == 2
    assert ctrl.state.total_surrogate_steps == 1
    assert ctrl.state.total_steps == 3
    assert ctrl.state.fallback_fraction == pytest.approx(2 / 3)


def test_ood_frame_callback_payload() -> None:
    captured: list[dict] = []
    ctrl = PhysicsFallbackController(
        MACEConfig(), on_ood_frame=lambda payload: captured.append(payload)
    )
    ctrl.decide_state(_uq(is_ood=True), frame_data={"positions": np.zeros((3, 3))})
    assert len(captured) == 1
    payload = captured[0]
    assert payload["reason"] == "force uncertainty high"
    assert payload["uq_result"]["is_ood"] is True
    assert "positions" in payload


def test_missing_lambda_parameter_warns_once() -> None:
    class StrictContext:
        def setParameter(self, name: str, value: float) -> None:
            raise RuntimeError("no such parameter")

    ctrl = PhysicsFallbackController(MACEConfig())
    ctx = StrictContext()
    ctrl.apply_to_openmm_context(ctx)
    assert ctrl._warned_missing_parameter is True
    ctrl.apply_to_openmm_context(ctx)  # second call: no exception, flag stays set
    assert ctrl._warned_missing_parameter is True
