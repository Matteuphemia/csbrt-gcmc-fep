"""The assembled surrogate the pipeline stages talk to."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
unit = pytest.importorskip("openmm.unit")

from csbrt.mace_surrogate import (  # noqa: E402
    INTERPOLATION_PARAMETER,
    EvaluatorMode,
    MACEConfig,
    MACERuntime,
    MACESurrogateError,
    ml_coordinates,
    ml_region_bonds,
    worker_tag,
)


def test_worker_tag_prefers_slurm(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "7")
    assert worker_tag() == "slurm123_7"
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID")
    assert worker_tag() == "slurm123"
    monkeypatch.delenv("SLURM_JOB_ID")
    assert "_" in worker_tag()


def test_ml_region_bonds_are_locally_indexed(built_system):
    bonds = ml_region_bonds(built_system.topology, built_system.ligand_atoms)
    assert len(bonds) == 8
    assert max(max(pair) for pair in bonds) < len(built_system.ligand_atoms)
    assert (0, 1) in bonds


def test_ml_coordinates_are_angstroms(built_system):
    context = built_system.context()
    coords, box = ml_coordinates(context, built_system.ligand_atoms)
    expected = built_system.positions_nm[built_system.ligand_atoms] * 10.0
    assert np.allclose(coords, expected)
    assert box is not None
    assert box.shape == (3, 3)
    assert box[0, 0] == pytest.approx(25.0)


def test_attach_returns_none_when_disabled(built_system, tmp_path):
    context = built_system.context()
    runtime = MACERuntime.attach(
        context, built_system.topology, MACEConfig(enabled=False),
        output_dir=tmp_path,
    )
    assert runtime is None


def test_attach_degrades_instead_of_crashing(built_system, tmp_path, caplog):
    """A missing ligand must not take the pipeline down when strict is off."""
    config = MACEConfig(enabled=True, model_name="csbrt-test-stub",
                        ligand_resname="NOPE", device="cpu")
    context = built_system.context()
    runtime = MACERuntime.attach(
        context, built_system.topology, config, output_dir=tmp_path
    )
    assert runtime is None
    assert INTERPOLATION_PARAMETER not in context.getParameters()


def test_strict_mode_raises(built_system, tmp_path):
    config = MACEConfig(enabled=True, model_name="csbrt-test-stub",
                        ligand_resname="NOPE", device="cpu", strict=True)
    with pytest.raises(MACESurrogateError):
        MACERuntime.attach(
            built_system.context(), built_system.topology, config,
            output_dir=tmp_path,
        )


def test_runtime_attaches_and_reports(built_system, stub_config, tmp_path):
    context = built_system.context()
    runtime = MACERuntime.attach(
        context, built_system.topology, stub_config,
        output_dir=tmp_path, source="edge01_bound",
    )
    assert runtime is not None
    assert context.getParameter(INTERPOLATION_PARAMETER) == pytest.approx(1.0)
    assert len(runtime.region) == 9
    assert runtime.mode is EvaluatorMode.MACE_SURROGATE

    assert runtime.check(context) is EvaluatorMode.MACE_SURROGATE
    runtime.advance(100)
    statistics = runtime.finish()
    assert statistics["fallback"]["md_steps"] == 100
    assert statistics["fallback"]["surrogate_steps"] == 100
    assert statistics["ml_region"]["ml_atom_count"] == 9
    assert statistics["mixed_system"]["backend"] == "openmm-ml"
    assert statistics["config_signature"]["enabled"] is True
    # Nothing was flagged, so there is no buffer to write.
    assert statistics["ood_buffer"]["frames"] == 0
    json.dumps(statistics)  # must be checkpoint-serialisable


def test_runtime_harvests_a_flagged_frame(built_system, stub_config, tmp_path):
    """Drive a real Context into a clash and check the whole loop reacts."""
    context = built_system.context()
    runtime = MACERuntime.attach(
        context, built_system.topology, stub_config,
        output_dir=tmp_path, source="clashtest",
    )
    assert runtime is not None

    # Collapse two ligand atoms onto each other in the live Context.
    positions = np.array(
        context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(
            unit.nanometer
        )
    )
    first, second = built_system.ligand_atoms[0], built_system.ligand_atoms[8]
    positions[second] = positions[first] + 0.005
    context.setPositions(positions * unit.nanometer)

    mode = runtime.check(context)
    assert mode is EvaluatorMode.CLASSICAL_PHYSICS
    assert context.getParameter(INTERPOLATION_PARAMETER) == pytest.approx(0.0)
    assert runtime.last_result.is_ood
    assert "steric clash" in runtime.last_result.trigger_reason

    runtime.advance(50)
    statistics = runtime.finish()
    assert statistics["fallback"]["fallback_events"] == 1
    assert statistics["fallback"]["classical_steps"] == 50
    assert statistics["ood_buffer"]["frames"] == 1

    buffer_path = Path(statistics["ood_buffer"]["path"])
    assert buffer_path.is_file()
    assert buffer_path.parent.name == "al_buffer"
    assert "clashtest" in buffer_path.name

    from csbrt.mace_surrogate import OODBuffer

    reloaded = OODBuffer(buffer_path)
    assert reloaded.load() == 1
    frame = reloaded.frames[0]
    assert frame.positions.shape == (9, 3)
    assert frame.atomic_numbers == runtime.region.atomic_numbers
    assert frame.ml_atoms == sorted(built_system.ligand_atoms)
    assert frame.box_vectors is not None


def test_runtime_recovers_to_the_surrogate(built_system, stub_config, tmp_path):
    context = built_system.context()
    runtime = MACERuntime.attach(
        context, built_system.topology, stub_config, output_dir=tmp_path
    )
    good = np.array(
        context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(
            unit.nanometer
        )
    )
    bad = good.copy()
    bad[built_system.ligand_atoms[8]] = bad[built_system.ligand_atoms[0]] + 0.005

    context.setPositions(bad * unit.nanometer)
    assert runtime.check(context) is EvaluatorMode.CLASSICAL_PHYSICS
    context.setPositions(good * unit.nanometer)
    runtime.advance(stub_config.fallback_steps)
    assert runtime.check(context) is EvaluatorMode.MACE_SURROGATE
    assert context.getParameter(INTERPOLATION_PARAMETER) == pytest.approx(1.0)


def test_runtime_resolves_relative_buffer_paths(built_system, tmp_path):
    config = MACEConfig(
        enabled=True, model_name="csbrt-test-stub", device="cpu",
        precision="double", ood_buffer_dir="al_buffer",
    )
    context = built_system.context()
    runtime = MACERuntime.attach(
        context, built_system.topology, config, output_dir=tmp_path, source="w0"
    )
    assert runtime.ood_buffer.buffer_file == tmp_path / "al_buffer" / "ood_w0.npz"


def test_runtime_survives_an_md_loop(built_system, stub_config, tmp_path):
    """Integrate for real under the surrogate and confirm nothing blows up."""
    context = built_system.context()
    runtime = MACERuntime.attach(
        context, built_system.topology, stub_config, output_dir=tmp_path
    )
    integrator = context.getIntegrator()
    state = runtime.controller.run_segmented_dynamics(
        run_steps=integrator.step,
        total_steps=50,
        evaluate_uq=lambda: runtime.controller.uq_monitor.evaluate(
            *ml_coordinates(context, runtime.region.atom_indices)[:1],
            runtime.region.atomic_numbers,
        ),
        interval=10,
    )
    assert state["md_steps"] == 50
    energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )
    assert np.isfinite(energy)


# --------------------------------------------------------------------------- chunking


def report_points(chunks, completed_steps, report_interval):
    """Steps at which the CSV writer would fire, given a chunk sequence."""
    points = []
    done = completed_steps
    for chunk in chunks:
        done += chunk
        if done % report_interval == 0:
            points.append(done)
    return points


@pytest.mark.parametrize("completed", [0, 1, 250, 499, 2000])
@pytest.mark.parametrize("num_steps", [1, 7, 500, 2000, 2001])
@pytest.mark.parametrize("uq_interval", [1, 7, 100, 500, 5000])
def test_uq_interleaving_never_moves_a_report_boundary(
    completed, num_steps, uq_interval
):
    """The CSV step schedule must be identical with and without the surrogate.

    Existing GCMC checkpoints validate the exact sequence of steps the CSV
    reports at. Subdividing the run for uncertainty checks must not perturb it,
    or every completed production run silently fails its own checkpoint.
    """
    from csbrt.mace_surrogate import md_chunks

    report_interval = 500
    classical = list(md_chunks(num_steps, completed, report_interval))
    with_uq = list(md_chunks(num_steps, completed, report_interval, uq_interval))

    assert sum(classical) == num_steps
    assert sum(with_uq) == num_steps
    assert report_points(classical, completed, report_interval) == report_points(
        with_uq, completed, report_interval
    )
    assert all(chunk <= uq_interval for chunk in with_uq)
    assert all(chunk > 0 for chunk in with_uq)


def test_chunking_of_a_zero_length_run():
    from csbrt.mace_surrogate import md_chunks

    assert list(md_chunks(0, 0, 500, 100)) == []


@pytest.mark.parametrize(
    "args,message",
    [
        ((-1, 0, 500), "negative"),
        ((10, -1, 500), "negative"),
        ((10, 0, 0), "Report interval"),
    ],
)
def test_chunking_rejects_nonsense(args, message):
    from csbrt.mace_surrogate import md_chunks

    with pytest.raises(ValueError, match=message):
        list(md_chunks(*args))


def test_chunking_rejects_a_zero_uq_interval():
    from csbrt.mace_surrogate import md_chunks

    with pytest.raises(ValueError, match="UQ interval"):
        list(md_chunks(10, 0, 500, 0))
