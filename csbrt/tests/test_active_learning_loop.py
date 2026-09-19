"""The active-learning loop end to end: run -> harvest -> label -> train set.

Uses the real OpenMM fixture and the real MM labeler, so every stage between a
fallback trigger and a `mace_run_train` invocation is exercised. Only the
training itself is a dry run, because that needs a GPU and an hour.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
unit = pytest.importorskip("openmm.unit")

from csbrt.mace_surrogate import (  # noqa: E402
    EvaluatorMode,
    MACEConfig,
    MACEFineTuner,
    MACERuntime,
    MMSubsystemLabeler,
    OODBuffer,
    ReferenceLabeler,
    harvest,
    write_training_set,
)
from csbrt.mace_surrogate.testsystems import build_reference_system  # noqa: E402
from csbrt.mace_active_learning import main as al_main  # noqa: E402


@pytest.fixture
def classical_reference():
    """A pristine copy of the fixture.

    ``attach_mace_to_context`` replaces the Forces of the System it is given,
    in place, so the System the campaign ran on is no longer classical. A
    reference labeler has to be built from a System the surrogate never touched
    -- which is exactly the mistake ``extract_subsystem`` now refuses.
    """
    return build_reference_system()


def clash_positions(built_system, context):
    positions = np.array(
        context.getState(getPositions=True)
        .getPositions(asNumpy=True)
        .value_in_unit(unit.nanometer)
    )
    first, last = built_system.ligand_atoms[0], built_system.ligand_atoms[8]
    positions[last] = positions[first] + 0.006
    return positions


@pytest.fixture
def campaign(built_system, stub_config, tmp_path):
    """Two 'workers' that each trip the fallback and write their own buffer."""
    buffer_dir = Path(stub_config.ood_buffer_dir)
    for worker in range(2):
        context = built_system.context()
        runtime = MACERuntime.attach(
            context,
            built_system.topology,
            stub_config,
            output_dir=tmp_path,
            source=f"worker{worker}",
        )
        assert runtime is not None
        positions = clash_positions(built_system, context)
        # Nudge each worker's clash so the two frames are genuinely different.
        positions[built_system.ligand_atoms[8]] += 0.001 * worker
        context.setPositions(positions * unit.nanometer)
        assert runtime.check(context) is EvaluatorMode.CLASSICAL_PHYSICS
        runtime.advance(100)
        runtime.finish()
    return buffer_dir


def test_workers_write_separate_buffers(campaign):
    files = sorted(campaign.glob("ood_*.npz"))
    assert len(files) == 2
    assert {f.name for f in files} == {"ood_worker0.npz", "ood_worker1.npz"}


def test_harvest_merges_and_clusters(campaign):
    merged = harvest(campaign)
    assert len(merged) == 2
    # Both workers clashed the same two atoms, so this is one basin.
    representatives = merged.cluster_and_deduplicate(rmsd_cutoff_angstrom=0.5)
    assert len(representatives) == 1
    assert representatives[0].positions.shape == (9, 3)


def test_labels_and_training_set(campaign, classical_reference, tmp_path):
    merged = harvest(campaign)
    frames = merged.cluster_and_deduplicate()
    labeler = ReferenceLabeler(
        MMSubsystemLabeler(
            classical_reference.system, classical_reference.ligand_atoms
        ),
        reference_level="mm",
    )
    labelled = labeler.label_all(frames)
    assert labelled and all(f.is_labelled for f in labelled)
    assert all(f.reference_level == "mm" for f in labelled)
    # A clashed geometry has a huge repulsive energy; that is the point.
    assert labelled[0].energy_label > 0.0

    dataset = tmp_path / "train.xyz"
    manifest = write_training_set(labelled, dataset)
    assert manifest["reference_level"] == "mm"

    ase_io = pytest.importorskip("ase.io")
    structures = ase_io.read(str(dataset), ":", format="extxyz")
    assert len(structures) == len(labelled)
    assert structures[0].arrays["REF_forces"].shape == (9, 3)


def test_finetune_refuses_the_mm_labels_it_was_handed(
    campaign, classical_reference, tmp_path
):
    merged = harvest(campaign)
    labeler = ReferenceLabeler(
        MMSubsystemLabeler(
            classical_reference.system, classical_reference.ligand_atoms
        ),
        "mm",
    )
    labelled = labeler.label_all(merged.cluster_and_deduplicate())
    tuner = MACEFineTuner(device="cpu", executable="/bin/true")
    result = tuner.fine_tune(labelled, tmp_path / "gen1.model")
    assert result["status"] == "refused"
    assert result["reason"] == "mm_labels_without_optin"


# --------------------------------------------------------------------------- CLI


def test_al_report_command(campaign, capsys):
    assert al_main(["report", "--buffer-dir", str(campaign)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["frames"] == 2
    assert payload["distinct_basins"] == 1
    assert payload["labelled"] == 0
    assert payload["reference_levels"] == ["unlabelled"]
    assert set(payload["sources"]) == {"worker0", "worker1"}


def test_al_harvest_command(campaign, tmp_path):
    target = tmp_path / "harvested.npz"
    assert al_main([
        "harvest", "--buffer-dir", str(campaign), "--output", str(target)
    ]) == 0
    buffer = OODBuffer(target)
    assert buffer.load() == 1


def test_al_harvest_on_an_empty_directory(tmp_path):
    assert al_main([
        "harvest", "--buffer-dir", str(tmp_path / "none"),
        "--output", str(tmp_path / "x.npz"),
    ]) == 1


def test_al_label_and_finetune_dry_run(campaign, tmp_path, monkeypatch):
    """The full aggregator path, with a scripted external labeller."""
    labeller = tmp_path / "fake_qm.py"
    labeller.write_text(
        "import json, sys\n"
        "lines = open(sys.argv[1]).read().splitlines()\n"
        "n = int(lines[0])\n"
        "print(json.dumps({'energy_ev': -3.25,\n"
        "                  'forces_ev_per_ang': [[0.01, 0.0, -0.02]] * n}))\n"
    )
    import sys as _sys

    labelled = tmp_path / "labelled.npz"
    assert al_main([
        "label",
        "--buffer-dir", str(campaign),
        "--labeller", "command",
        "--command", f"{_sys.executable} {labeller} {{xyz}}",
        "--output", str(labelled),
    ]) == 0

    buffer = OODBuffer(labelled)
    assert buffer.load() == 1
    assert buffer.frames[0].reference_level == "qm"
    assert buffer.frames[0].energy_label == pytest.approx(-3.25)

    models = tmp_path / "models"
    assert al_main([
        "finetune",
        "--frames", str(labelled),
        "--output-dir", str(models),
        "--generation", "2",
        "--seeds", "2",
        "--device", "cpu",
        "--dry-run",
    ]) == 0
    manifest = json.loads((models / "generation2.json").read_text())
    assert len(manifest) == 2
    assert {entry["seed"] for entry in manifest} == {20260714, 20260715}
    for entry in manifest:
        assert entry["status"] == "prepared"
        assert entry["reference_level"] == "qm"
        assert "--foundation_model" in entry["command"]
    assert (models / "train_gen2.xyz").is_file()


def test_al_label_reports_a_failing_labeller(campaign, tmp_path):
    assert al_main([
        "label",
        "--buffer-dir", str(campaign),
        "--labeller", "command",
        "--command", "exit 3",
        "--output", str(tmp_path / "out.npz"),
    ]) == 1


def test_al_label_mm_needs_a_topology(campaign, tmp_path):
    assert al_main([
        "label", "--buffer-dir", str(campaign), "--labeller", "mm",
        "--output", str(tmp_path / "out.npz"),
    ]) == 2
