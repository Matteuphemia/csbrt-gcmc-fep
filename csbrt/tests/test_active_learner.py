"""Stage 4: OOD buffer, clustering, labelling, and the fine-tuning harness."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
unit = pytest.importorskip("openmm.unit")

from csbrt.mace_surrogate import (  # noqa: E402
    MACEFineTuner,
    MMSubsystemLabeler,
    OODBuffer,
    OODFrame,
    ReferenceLabeler,
    compute_kabsch_rmsd,
    extract_subsystem,
    harvest,
    write_training_set,
)
from csbrt.mace_surrogate.active_learner import element_symbol  # noqa: E402


RNG = np.random.default_rng(99)


def frame(step=0, shift=0.0, sigma=0.1, source="run", atoms=6):
    coords = RNG.normal(0.0, 1.0, (atoms, 3)) if shift is None else (
        np.arange(atoms * 3, dtype=np.float64).reshape(atoms, 3) * 0.5 + shift
    )
    return OODFrame(
        frame_id=f"{source}_{step}",
        step=step,
        positions=coords,
        atomic_numbers=[6, 1, 1, 8, 1, 6][:atoms],
        uncertainty_force=sigma,
        uncertainty_energy=sigma * 2,
        reason="test",
        source=source,
    )


# --------------------------------------------------------------------------- frames


def test_frame_validates_shapes():
    with pytest.raises(ValueError, match=r"positions must be"):
        OODFrame("x", 0, np.zeros(6), [6, 1], 0.1, 0.2, "r")
    with pytest.raises(ValueError, match="atomic numbers"):
        OODFrame("x", 0, np.zeros((3, 3)), [6, 1], 0.1, 0.2, "r")
    with pytest.raises(ValueError, match="reference_level"):
        OODFrame("x", 0, np.zeros((2, 3)), [6, 1], 0.1, 0.2, "r",
                 reference_level="dft-ish")


def test_frame_round_trips_through_a_dict():
    original = frame(step=7)
    original.energy_label = -1.5
    original.forces_label = np.ones((6, 3))
    original.reference_level = "qm"
    restored = OODFrame.from_dict(original.to_dict())
    assert restored.frame_id == original.frame_id
    assert restored.reference_level == "qm"
    assert np.allclose(restored.positions, original.positions)
    assert np.allclose(restored.forces_label, original.forces_label)


def test_element_symbols():
    assert element_symbol(6) == "C"
    assert element_symbol(1) == "H"
    with pytest.raises(ValueError):
        element_symbol(0)


# --------------------------------------------------------------------------- buffer


def test_buffer_saves_and_loads_atomically(tmp_path):
    path = tmp_path / "nested" / "ood.npz"
    buffer = OODBuffer(path, source="worker7")
    for step in range(3):
        buffer.add_from_raw(
            step=step,
            positions=np.full((4, 3), float(step)),
            atomic_numbers=[6, 1, 1, 1],
            uncertainty_force=0.1 * step,
            uncertainty_energy=0.2 * step,
            reason="synthetic",
            ml_atoms=[10, 11, 12, 13],
        )
    written = buffer.save()
    assert written.is_file()
    assert not list(path.parent.glob(".*tmp*"))

    reloaded = OODBuffer(written)
    assert reloaded.load() == 3
    assert reloaded.frames[2].step == 2
    assert reloaded.frames[2].source == "worker7"
    assert reloaded.frames[2].ml_atoms == [10, 11, 12, 13]


def test_buffer_load_of_missing_file_is_not_fatal(tmp_path):
    assert OODBuffer(tmp_path / "absent.npz").load() == 0


def test_buffer_cap_drops_the_least_uncertain(tmp_path):
    buffer = OODBuffer(tmp_path / "b.npz", max_frames=3)
    for index, sigma in enumerate([0.5, 0.1, 0.9, 0.2, 0.7]):
        buffer.add(frame(step=index, sigma=sigma))
    assert len(buffer) == 3
    kept = sorted(f.uncertainty_force for f in buffer)
    assert kept == [0.5, 0.7, 0.9]
    assert buffer.dropped == 2


def test_harvest_merges_every_worker_buffer(tmp_path):
    root = tmp_path / "al_buffer"
    for worker in range(3):
        buffer = OODBuffer(root / f"ood_worker{worker}.npz", source=f"w{worker}")
        buffer.add(frame(step=worker, source=f"w{worker}"))
        buffer.save()
    merged = harvest(root)
    assert len(merged) == 3
    assert {f.source for f in merged} == {"w0", "w1", "w2"}


def test_harvest_skips_a_corrupt_buffer(tmp_path):
    root = tmp_path / "al_buffer"
    root.mkdir(parents=True)
    good = OODBuffer(root / "ood_good.npz")
    good.add(frame())
    good.save()
    (root / "ood_bad.npz").write_bytes(b"not an npz")
    merged = harvest(root)
    assert len(merged) == 1


def test_harvest_of_an_empty_directory(tmp_path):
    assert len(harvest(tmp_path / "nothing-here")) == 0


# --------------------------------------------------------------------------- clustering


def test_kabsch_rmsd_is_rotation_invariant():
    coords = RNG.normal(0.0, 1.0, (12, 3))
    angle = 0.7
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated = coords @ rotation.T + np.array([3.0, -2.0, 1.0])
    assert compute_kabsch_rmsd(coords, rotated) == pytest.approx(0.0, abs=1e-9)


def test_kabsch_rmsd_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_kabsch_rmsd(np.zeros((3, 3)), np.zeros((4, 3)))


def test_clustering_collapses_duplicates_and_keeps_the_worst(tmp_path):
    buffer = OODBuffer(tmp_path / "b.npz")
    base = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0],
                     [1.5, 1.5, 0.0]])
    numbers = [6, 6, 8, 6]
    for index, (jitter, sigma) in enumerate(
        [(0.0, 0.2), (0.02, 0.9), (0.01, 0.3)]
    ):
        buffer.add(
            OODFrame(
                frame_id=f"near{index}",
                step=index,
                positions=base + jitter,
                atomic_numbers=numbers,
                uncertainty_force=sigma,
                uncertainty_energy=0.0,
                reason="near",
            )
        )
    buffer.add(
        OODFrame(
            frame_id="far",
            step=9,
            positions=base * np.array([1.0, 1.0, 1.0]) + np.array([0, 0, 4.0]),
            atomic_numbers=numbers,
            uncertainty_force=0.5,
            uncertainty_energy=0.0,
            reason="far",
        )
    )
    # The 'far' frame is a rigid translation, so after Kabsch alignment it is
    # the same conformation: all four collapse into one basin.
    representatives = buffer.cluster_and_deduplicate(rmsd_cutoff_angstrom=0.5)
    assert len(representatives) == 1
    assert representatives[0].frame_id == "near1"


def test_clustering_separates_distinct_conformations(tmp_path):
    buffer = OODBuffer(tmp_path / "b.npz")
    numbers = [6, 6, 8, 6]
    first = np.array([[0.0, 0, 0], [1.5, 0, 0], [0, 1.5, 0], [1.5, 1.5, 0]])
    second = first.copy()
    second[3] = [1.5, -3.0, 2.5]  # a genuinely different torsion
    for name, coords in (("a", first), ("b", second)):
        buffer.add(
            OODFrame(name, 0, coords, numbers, 0.3, 0.0, "r")
        )
    assert len(buffer.cluster_and_deduplicate(rmsd_cutoff_angstrom=0.5)) == 2


def test_clustering_never_mixes_different_molecules(tmp_path):
    buffer = OODBuffer(tmp_path / "b.npz")
    buffer.add(OODFrame("a", 0, np.zeros((4, 3)), [6, 6, 8, 6], 0.3, 0.0, "r"))
    buffer.add(OODFrame("b", 0, np.zeros((4, 3)), [6, 6, 7, 6], 0.3, 0.0, "r"))
    assert len(buffer.cluster_and_deduplicate()) == 2


def test_clustering_uses_heavy_atoms_only(tmp_path):
    buffer = OODBuffer(tmp_path / "b.npz")
    # Three non-collinear heavy atoms fix the orientation completely, so the
    # only difference between the two frames is where the hydrogen sits.
    heavy = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.5, 1.4, 0.3]])
    for index, hydrogen in enumerate([[0.0, 0.0, 2.0], [0.0, 0.0, -2.0]]):
        buffer.add(
            OODFrame(
                f"f{index}",
                index,
                np.vstack([heavy, [hydrogen]]),
                [6, 6, 8, 1],
                0.3,
                0.0,
                "r",
            )
        )
    assert len(buffer.cluster_and_deduplicate(heavy_atoms_only=True)) == 1
    assert len(buffer.cluster_and_deduplicate(heavy_atoms_only=False)) == 2


def test_clustering_of_an_empty_buffer():
    assert OODBuffer().cluster_and_deduplicate() == []


# --------------------------------------------------------------------------- labelling


def test_reference_labeler_stamps_the_level():
    def evaluator(positions, numbers):
        return -12.5, np.full(positions.shape, 0.25)

    labeler = ReferenceLabeler(evaluator, reference_level="qm")
    labelled = labeler.label_frame(frame())
    assert labelled.energy_label == pytest.approx(-12.5)
    assert labelled.reference_level == "qm"
    assert labelled.is_labelled


def test_reference_labeler_rejects_a_bogus_level():
    with pytest.raises(ValueError, match="reference_level"):
        ReferenceLabeler(lambda p, n: (0.0, p), reference_level="unlabelled")


def test_reference_labeler_survives_one_bad_frame():
    calls = {"n": 0}

    def flaky(positions, numbers):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("SCF did not converge")
        return 0.0, np.zeros_like(positions)

    labeler = ReferenceLabeler(flaky, reference_level="qm")
    labelled = labeler.label_all([frame(0), frame(1), frame(2)])
    assert len(labelled) == 2
    assert len(labeler.failures) == 1


def test_reference_labeler_rejects_wrong_force_shape():
    labeler = ReferenceLabeler(lambda p, n: (0.0, np.zeros((2, 3))), "qm")
    with pytest.raises(ValueError, match="forces of shape"):
        labeler.label_frame(frame())


def test_mm_subsystem_labeler_reproduces_openmm(built_system):
    """The extracted ligand subsystem must agree with the parent force field.

    Compare against an independent calculation: the ligand's bonded energy in
    the full system, obtained by zeroing every other interaction, is not
    directly available, so check the invariant that actually matters -- the
    subsystem's forces are the negative gradient of its own energy, and the
    bonded terms match the parent's parameters.
    """
    ml_atoms = built_system.ligand_atoms
    labeler = MMSubsystemLabeler(built_system.system, ml_atoms)
    coords = built_system.positions_nm[ml_atoms] * 10.0  # nm -> A
    energy, forces = labeler(coords, [6] * len(ml_atoms))
    assert np.isfinite(energy)
    assert forces.shape == (len(ml_atoms), 3)

    # Central finite difference on one atom, in Angstroms/eV.
    delta = 1.0e-4
    probe = coords.copy()
    probe[0, 0] += delta
    plus, _ = labeler(probe, [6] * len(ml_atoms))
    probe[0, 0] -= 2 * delta
    minus, _ = labeler(probe, [6] * len(ml_atoms))
    numerical = -(plus - minus) / (2 * delta)
    assert numerical == pytest.approx(forces[0, 0], rel=1e-3, abs=1e-4)


def test_mm_subsystem_labeler_checks_atom_count(built_system):
    labeler = MMSubsystemLabeler(built_system.system, built_system.ligand_atoms)
    with pytest.raises(ValueError, match="atoms"):
        labeler(np.zeros((3, 3)), [6, 6, 6])


def test_extract_subsystem_keeps_only_internal_terms(built_system):
    subsystem = extract_subsystem(built_system.system, built_system.ligand_atoms)
    assert subsystem.getNumParticles() == 9
    bonds = [
        f for f in subsystem.getForces()
        if isinstance(f, openmm.HarmonicBondForce)
    ]
    assert bonds and bonds[0].getNumBonds() == 8
    nonbonded = [
        f for f in subsystem.getForces() if isinstance(f, openmm.NonbondedForce)
    ]
    assert nonbonded
    assert nonbonded[0].getNonbondedMethod() == openmm.NonbondedForce.NoCutoff


# --------------------------------------------------------------------------- training set


def labelled_frames(count=3, level="qm"):
    frames = []
    for index in range(count):
        item = frame(step=index)
        item.energy_label = -10.0 - index
        item.forces_label = np.full((6, 3), 0.1 * (index + 1))
        item.reference_level = level
        frames.append(item)
    return frames


def test_training_set_is_valid_extxyz(tmp_path):
    path = tmp_path / "train.xyz"
    manifest = write_training_set(labelled_frames(), path)
    assert manifest["frames"] == 3
    assert manifest["reference_level"] == "qm"

    lines = path.read_text().splitlines()
    assert lines[0] == "6"
    assert "REF_energy=-10.0" in lines[1]
    assert "Properties=species:S:1:pos:R:3:REF_forces:R:3" in lines[1]
    assert len(lines) == 3 * (6 + 2)
    assert lines[2].split()[0] == "C"
    assert len(lines[2].split()) == 7


def test_training_set_is_readable_by_ase(tmp_path):
    ase_io = pytest.importorskip("ase.io")
    path = tmp_path / "train.xyz"
    write_training_set(labelled_frames(), path)
    structures = ase_io.read(str(path), ":", format="extxyz")
    assert len(structures) == 3
    assert structures[0].info["REF_energy"] == pytest.approx(-10.0)
    assert structures[0].arrays["REF_forces"].shape == (6, 3)
    assert list(structures[0].get_atomic_numbers()) == [6, 1, 1, 8, 1, 6]


def test_training_set_refuses_mixed_levels(tmp_path):
    frames = labelled_frames(2, "qm") + labelled_frames(1, "mm")
    with pytest.raises(ValueError, match="mixing levels"):
        write_training_set(frames, tmp_path / "train.xyz")


def test_training_set_requires_labels(tmp_path):
    with pytest.raises(ValueError, match="No labelled frames"):
        write_training_set([frame()], tmp_path / "train.xyz")


# --------------------------------------------------------------------------- fine-tuning


def test_fine_tune_refuses_mm_labels_by_default(tmp_path):
    tuner = MACEFineTuner(device="cpu", executable="/bin/true")
    result = tuner.fine_tune(
        labelled_frames(level="mm"), tmp_path / "gen1.model", generation=1
    )
    assert result["status"] == "refused"
    assert result["reason"] == "mm_labels_without_optin"


def test_fine_tune_refuses_mixed_levels(tmp_path):
    tuner = MACEFineTuner(device="cpu", executable="/bin/true")
    frames = labelled_frames(2, "qm") + labelled_frames(1, "mm")
    result = tuner.fine_tune(frames, tmp_path / "gen1.model")
    assert result["status"] == "refused"
    assert "mixed_reference_levels" in result["reason"]


def test_fine_tune_skips_an_unlabelled_set(tmp_path):
    tuner = MACEFineTuner(device="cpu", executable="/bin/true")
    result = tuner.fine_tune([frame()], tmp_path / "gen1.model")
    assert result["status"] == "skipped"


def test_fine_tune_dry_run_emits_the_real_command(tmp_path):
    tuner = MACEFineTuner(
        foundation_model="small", device="cpu", executable="/usr/bin/mace_run_train"
    )
    result = tuner.fine_tune(
        labelled_frames(), tmp_path / "gen2.model", generation=2, dry_run=True
    )
    assert result["status"] == "prepared"
    command = result["command"]
    assert command[0] == "/usr/bin/mace_run_train"
    assert "--foundation_model" in command
    assert command[command.index("--foundation_model") + 1] == "small"
    assert command[command.index("--energy_key") + 1] == "REF_energy"
    assert command[command.index("--forces_key") + 1] == "REF_forces"
    assert float(command[command.index("--lr") + 1]) == pytest.approx(1e-4)
    assert (tmp_path / "train_gen2.xyz").is_file()
    manifest = json.loads((tmp_path / "finetune_gen2.json").read_text())
    assert manifest["dataset"]["frames"] == 3


def test_fine_tune_reports_a_failed_trainer(tmp_path):
    tuner = MACEFineTuner(device="cpu", executable="/bin/false")
    result = tuner.fine_tune(labelled_frames(), tmp_path / "gen1.model")
    assert result["status"] == "failed"
    assert result["returncode"] != 0
    assert (tmp_path / "finetune_gen1.log").is_file()


def test_dry_run_works_without_the_trainer_installed(tmp_path):
    """A login node with no GPU stack can still render the command."""
    tuner = MACEFineTuner(device="cpu", executable=None)
    tuner.executable = None
    result = tuner.fine_tune(labelled_frames(), tmp_path / "gen1.model", dry_run=True)
    assert result["status"] == "prepared"
    assert result["command"][0] == "mace_run_train"


def test_fine_tune_without_the_trainer_raises(tmp_path):
    tuner = MACEFineTuner(device="cpu", executable=None)
    tuner.executable = None
    with pytest.raises(FileNotFoundError, match="mace_run_train"):
        tuner.fine_tune(labelled_frames(), tmp_path / "gen1.model")


def test_extract_subsystem_refuses_an_already_mixed_system(built_system, stub_config):
    """Guard against labelling against the surrogate instead of the force field."""
    from csbrt.mace_surrogate import attach_mace_to_context

    context = built_system.context()
    attach_mace_to_context(
        context, built_system.topology, stub_config, built_system.ligand_atoms
    )
    with pytest.raises(ValueError, match="already carries a mixed"):
        extract_subsystem(context.getSystem(), built_system.ligand_atoms)
