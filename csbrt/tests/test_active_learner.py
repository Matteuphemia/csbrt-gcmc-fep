"""Tests for the active-learning OOD buffer, clustering, and fine-tuner harness."""

from __future__ import annotations

import json

import numpy as np
import pytest

from csbrt.mace_surrogate import (
    MACEFineTuner,
    OODBuffer,
    OODFrame,
    ReferenceLabeler,
    compute_kabsch_rmsd,
)


def _frame(positions: np.ndarray, uncertainty: float = 0.1) -> OODFrame:
    return OODFrame(
        frame_id="f",
        step=0,
        positions=np.asarray(positions, dtype=np.float32),
        atomic_numbers=[6, 8, 1],
        uncertainty_force=uncertainty,
        uncertainty_energy=0.5,
        reason="test",
        ml_atoms=[0, 1, 2],
    )


def test_kabsch_rmsd_identity_and_translation() -> None:
    p = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert compute_kabsch_rmsd(p, p) == pytest.approx(0.0, abs=1e-9)
    q = p + np.array([5.0, 5.0, 5.0])
    assert compute_kabsch_rmsd(p, q) == pytest.approx(0.0, abs=1e-9)


def test_kabsch_rmsd_rotation_and_difference() -> None:
    p = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    # 90-degree rotation about z permutes x/y coordinates.
    q = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    assert compute_kabsch_rmsd(p, q) == pytest.approx(0.0, abs=1e-6)

    far = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0], [0.0, 0.0, 1.0]])
    assert compute_kabsch_rmsd(p, far) > 1.0


def test_ood_buffer_save_load_roundtrip(tmp_path) -> None:
    buffer = OODBuffer()
    buffer.add(_frame(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])))
    buffer.add(_frame(np.array([[0.1, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])))
    target = buffer.save(tmp_path / "ood.npz")

    loaded = OODBuffer()
    count = loaded.load(target)
    assert count == 2
    assert len(loaded) == 2
    assert loaded.frames[0].atomic_numbers == [6, 8, 1]
    assert loaded.frames[1].positions.shape == (3, 3)


def test_cluster_and_deduplicate(tmp_path) -> None:
    buffer = OODBuffer()
    # Two near-identical frames + one distant frame -> 2 clusters.
    buffer.add(_frame(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), 0.1))
    buffer.add(_frame(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), 0.9))
    # Different internal geometry (scaled triangle) -> genuinely distinct cluster.
    buffer.add(_frame(np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]]), 0.3))
    reps = buffer.cluster_and_deduplicate(rmsd_cutoff_angstrom=0.5)
    assert len(reps) == 2
    # Representative of the duplicate cluster has the highest uncertainty.
    dup_rep = next(r for r in reps if r.uncertainty_force == 0.9)
    assert dup_rep is not None


def test_reference_labeler_synthetic() -> None:
    positions = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], dtype=np.float32)
    frame = _frame(positions)
    labeled = ReferenceLabeler().label_frame(frame)
    assert labeled.energy_label == pytest.approx(float(np.sum(positions**2) * 0.01))
    np.testing.assert_allclose(labeled.forces_label, -0.02 * positions)


def test_fine_tuner_without_torch_writes_manifest(tmp_path) -> None:
    tuner = MACEFineTuner(learning_rate=1e-4, max_epochs=2)
    out = tmp_path / "gen1.pt"
    result = tuner.fine_tune([_frame(np.zeros((3, 3)))], out, generation=1)
    assert result["status"] == "completed"
    assert out.is_file()
    manifest = json.loads(out.read_text())
    assert manifest["generation"] == 1
    assert manifest["num_samples"] == 1
