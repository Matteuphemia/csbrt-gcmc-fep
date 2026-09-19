"""Unit tests for active learning buffer and RMSD clustering."""

import tempfile
from pathlib import Path
import pytest
import numpy as np

from csbrt.mace_surrogate import (
    OODBuffer,
    OODFrame,
    compute_kabsch_rmsd,
)


def test_kabsch_rmsd_identical():
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    rmsd = compute_kabsch_rmsd(coords, coords)
    assert rmsd < 1e-6


def test_kabsch_rmsd_translation_invariance():
    coords1 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    coords2 = coords1 + np.array([10.0, -5.0, 2.0])
    rmsd = compute_kabsch_rmsd(coords1, coords2)
    assert rmsd < 1e-5


def test_ood_buffer_add_and_cluster():
    with tempfile.TemporaryDirectory() as tmpdir:
        buf_path = Path(tmpdir) / "test_ood.npz"
        buf = OODBuffer(buf_path)

        # Add 3 frames: frame 1 and 2 nearly identical, frame 3 has stretched bond (distinct conformation)
        p1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        p2 = np.array([[0.01, 0.0, 0.0], [1.01, 0.0, 0.0], [0.0, 1.01, 0.0], [0.0, 0.0, 1.01]])
        p3 = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # bond stretched 1.0 -> 3.0 A

        buf.add_from_raw(step=1, positions=p1, atomic_numbers=[6, 6, 6, 6], uncertainty_force=0.08, uncertainty_energy=1.2, reason="test1")
        buf.add_from_raw(step=2, positions=p2, atomic_numbers=[6, 6, 6, 6], uncertainty_force=0.09, uncertainty_energy=1.3, reason="test2")
        buf.add_from_raw(step=3, positions=p3, atomic_numbers=[6, 6, 6, 6], uncertainty_force=0.15, uncertainty_energy=2.0, reason="test3")

        assert len(buf) == 3
        # Cluster with 0.5 Angstrom cutoff -> p1 and p2 should cluster together, resulting in 2 clusters
        clusters = buf.cluster_by_rmsd(cutoff_angstrom=0.5)
        assert len(clusters) == 2

        # Save and reload
        buf.save()
        assert buf_path.is_file()

        buf_reloaded = OODBuffer(buf_path)
        assert len(buf_reloaded) == 3
