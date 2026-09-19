"""Stage 3: the lambda_interpolate Hamiltonian switch and the state machine."""

from __future__ import annotations

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
unit = pytest.importorskip("openmm.unit")

from csbrt.mace_surrogate import (  # noqa: E402
    INTERPOLATION_PARAMETER,
    DualHamiltonianSwitch,
    EvaluatorMode,
    MACEConfig,
    MACESurrogateError,
    MACEUQMonitor,
    PhysicsFallbackController,
    UQResult,
    attach_mace_to_context,
)


def ood(reason: str = "synthetic") -> UQResult:
    result = UQResult.in_distribution()
    result.is_ood = True
    result.trigger_reason = reason
    result.sigma_f_max_ev_per_ang = 0.42
    return result


def clean() -> UQResult:
    return UQResult.in_distribution()


# --------------------------------------------------------------------------- switch


def test_switch_flips_a_real_context(built_system, stub_config):
    context = built_system.context()
    attach_mace_to_context(
        context, built_system.topology, stub_config, built_system.ligand_atoms
    )
    switch = DualHamiltonianSwitch(context)
    assert switch.available

    switch.set_mode(EvaluatorMode.CLASSICAL_PHYSICS)
    assert context.getParameter(INTERPOLATION_PARAMETER) == pytest.approx(0.0)
    switch.set_mode(EvaluatorMode.MACE_SURROGATE)
    assert context.getParameter(INTERPOLATION_PARAMETER) == pytest.approx(1.0)


def test_switch_is_sub_millisecond(built_system, stub_config):
    """The zero-overhead claim: a mode change must not cost a Context rebuild."""
    context = built_system.context()
    attach_mace_to_context(
        context, built_system.topology, stub_config, built_system.ligand_atoms
    )
    switch = DualHamiltonianSwitch(context)
    latency = switch.measure_latency(samples=64)
    assert latency["median_ms"] < 1.0
    assert latency["max_ms"] < 10.0  # generous: CI machines stall


def test_switch_changes_energy_but_not_state(built_system, stub_config):
    """Positions and velocities survive the transition -- trajectory continuity."""
    context = built_system.context()
    context.setVelocitiesToTemperature(300 * unit.kelvin, 7)
    attach_mace_to_context(
        context, built_system.topology, stub_config, built_system.ligand_atoms
    )
    switch = DualHamiltonianSwitch(context)

    def snapshot():
        state = context.getState(
            getPositions=True, getVelocities=True, getEnergy=True
        )
        return (
            state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
            state.getVelocities(asNumpy=True).value_in_unit(
                unit.nanometer / unit.picosecond
            ),
            state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
        )

    switch.set_mode(EvaluatorMode.MACE_SURROGATE)
    positions_ml, velocities_ml, energy_ml = snapshot()
    switch.set_mode(EvaluatorMode.CLASSICAL_PHYSICS)
    positions_mm, velocities_mm, energy_mm = snapshot()

    assert np.allclose(positions_ml, positions_mm)
    assert np.allclose(velocities_ml, velocities_mm)
    assert energy_ml != pytest.approx(energy_mm)


def test_switch_requires_the_parameter():
    class Bare:
        def getParameters(self):
            return {}

    with pytest.raises(MACESurrogateError, match="lambda_interpolate"):
        DualHamiltonianSwitch(Bare())


# --------------------------------------------------------------------------- state machine


def controller(**overrides):
    config = MACEConfig(enabled=True, fallback_steps=50, uq_interval_steps=10)
    for key, value in overrides.items():
        setattr(config, key, value)
    return PhysicsFallbackController(
        config=config, uq_monitor=MACEUQMonitor(config)
    )


def test_trigger_switches_to_classical():
    control = controller()
    assert control.observe(clean()) is EvaluatorMode.MACE_SURROGATE
    assert control.observe(ood("clash")) is EvaluatorMode.CLASSICAL_PHYSICS
    assert control.state.fallback_events == 1
    assert control.state.last_trigger_reason == "clash"


def test_relaxation_buffer_is_honoured_before_resuming():
    control = controller(fallback_steps=50, uq_interval_steps=10)
    control.observe(ood())
    # 50 classical steps must elapse before the surrogate is retried.
    for _ in range(4):
        control.advance(10)
        assert control.observe(clean()) is EvaluatorMode.CLASSICAL_PHYSICS
    control.advance(10)
    assert control.observe(clean()) is EvaluatorMode.MACE_SURROGATE
    assert control.state.classical_steps == 50


def test_repeated_triggers_extend_the_buffer_without_double_counting():
    control = controller(fallback_steps=20, uq_interval_steps=10)
    control.observe(ood())
    control.advance(10)
    control.observe(ood())  # still classical: not a new event
    control.advance(10)
    assert control.state.fallback_events == 1
    assert control.observe(clean()) is EvaluatorMode.CLASSICAL_PHYSICS
    control.advance(10)
    assert control.observe(clean()) is EvaluatorMode.MACE_SURROGATE
    assert control.state.fallback_events == 1


def test_step_accounting_splits_surrogate_and_classical():
    control = controller(fallback_steps=10, uq_interval_steps=10)
    control.observe(clean())
    control.advance(100)
    control.observe(ood())
    control.advance(10)
    control.observe(clean())
    control.advance(50)
    assert control.state.md_steps == 160
    assert control.state.surrogate_steps == 150
    assert control.state.classical_steps == 10
    assert control.state.fallback_fraction == pytest.approx(10 / 160)


def test_ood_frames_are_captured_once_per_event():
    captured = []
    config = MACEConfig(enabled=True, fallback_steps=10)
    control = PhysicsFallbackController(
        config=config,
        uq_monitor=MACEUQMonitor(config),
        on_ood_frame=captured.append,
    )
    frame = {"positions": np.zeros((3, 3))}
    control.observe(ood("first"), frame)
    control.advance(5)
    control.observe(ood("still bad"), frame)
    assert len(captured) == 1
    assert captured[0]["reason"] == "first"
    assert control.state.ood_frames_captured == 1


def test_harvester_failure_does_not_kill_the_run():
    config = MACEConfig(enabled=True)

    def explode(_payload):
        raise RuntimeError("disk full")

    control = PhysicsFallbackController(
        config=config, uq_monitor=MACEUQMonitor(config), on_ood_frame=explode
    )
    assert control.observe(ood(), {"positions": np.zeros((1, 3))}) is (
        EvaluatorMode.CLASSICAL_PHYSICS
    )
    assert control.state.ood_frames_captured == 0


def test_latches_classical_when_the_surrogate_stops_paying():
    control = controller(fallback_steps=100, fallback_abort_fraction=0.5)
    control.min_evaluations_before_abort = 5
    for _ in range(10):
        control.observe(ood())
        control.advance(20)
    assert control.state.latched_classical
    # Once latched, even a clean frame stays on classical physics.
    assert control.observe(clean()) is EvaluatorMode.CLASSICAL_PHYSICS
    assert any(t["event"] == "latched_classical" for t in control.state.transitions)


def test_abort_needs_enough_evaluations_first():
    control = controller(fallback_steps=100, fallback_abort_fraction=0.1)
    control.observe(ood())
    control.advance(1000)
    assert not control.state.latched_classical


def test_advance_rejects_negative_steps():
    with pytest.raises(ValueError, match="negative"):
        controller().advance(-1)


# --------------------------------------------------------------------------- driver


def test_segmented_dynamics_drives_a_real_context(built_system, stub_config):
    context = built_system.context()
    attach_mace_to_context(
        context, built_system.topology, stub_config, built_system.ligand_atoms
    )
    integrator = context.getIntegrator()
    switch = DualHamiltonianSwitch(context)
    config = MACEConfig(enabled=True, fallback_steps=10, uq_interval_steps=5)
    control = PhysicsFallbackController(
        config=config, uq_monitor=MACEUQMonitor(config), switch=switch
    )

    calls = {"n": 0}

    def evaluate():
        calls["n"] += 1
        return ood("injected") if calls["n"] == 2 else clean()

    state = control.run_segmented_dynamics(
        run_steps=integrator.step,
        total_steps=40,
        evaluate_uq=evaluate,
        collect_frame=lambda: {"positions": np.zeros((9, 3))},
    )
    assert state["md_steps"] == 40
    assert state["fallback_events"] == 1
    assert state["classical_steps"] >= 10
    assert switch.switch_count >= 2
    energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )
    assert np.isfinite(energy)


def test_segmented_dynamics_runs_every_requested_step():
    config = MACEConfig(enabled=True, uq_interval_steps=7)
    control = PhysicsFallbackController(config=config)
    run = []
    control.run_segmented_dynamics(run.append, total_steps=20, evaluate_uq=clean)
    assert sum(run) == 20
    assert run == [7, 7, 6]


def test_segmented_dynamics_rejects_a_zero_interval():
    with pytest.raises(ValueError, match="interval"):
        PhysicsFallbackController().run_segmented_dynamics(
            lambda n: None, total_steps=10, interval=0
        )
