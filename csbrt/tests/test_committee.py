"""Committee inference: periodic unwrapping, loading, and real MACE models.

The MACE-backed test is opt-in (``CSBRT_MACE_MODEL_TESTS=1``) because it
downloads foundation-model weights. Everything that does not need a model runs
everywhere.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from csbrt.mace_surrogate import (
    CommitteeUnavailable,
    MACECommittee,
    MACEConfig,
    MACEUQMonitor,
    make_whole,
)

MODEL_TESTS = os.environ.get("CSBRT_MACE_MODEL_TESTS") == "1"


# --------------------------------------------------------------------------- imaging


def test_make_whole_reassembles_a_split_molecule():
    """An ML region straddling a box face must not look like a torn molecule."""
    box = np.eye(3) * 20.0
    intact = np.array([[19.0, 1.0, 1.0], [19.8, 1.0, 1.0], [20.6, 1.0, 1.0]])
    wrapped = intact.copy()
    wrapped[2, 0] -= 20.0  # OpenMM wraps the third atom into the box
    restored = make_whole(wrapped, box)
    assert np.allclose(restored, intact)


def test_make_whole_is_a_noop_without_a_box():
    coords = np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]])
    assert np.allclose(make_whole(coords, None), coords)


def test_make_whole_handles_a_triclinic_box():
    box = np.array([[20.0, 0.0, 0.0], [3.0, 19.0, 0.0], [2.0, 1.0, 18.0]])
    intact = np.array([[1.0, 1.0, 1.0], [2.0, 1.5, 1.2]])
    wrapped = intact - box[0]  # image the second atom by a full box vector
    wrapped[0] = intact[0]
    restored = make_whole(wrapped, box)
    assert np.allclose(restored, intact)


def test_make_whole_leaves_a_single_atom_alone():
    assert np.allclose(make_whole(np.zeros((1, 3)), np.eye(3) * 10), np.zeros((1, 3)))


# --------------------------------------------------------------------------- loading


def test_committee_needs_two_models(tmp_path):
    model = tmp_path / "one.model"
    model.write_bytes(b"x")
    with pytest.raises(CommitteeUnavailable, match="at least two"):
        MACECommittee([model])


def test_committee_reports_missing_files(tmp_path):
    present = tmp_path / "a.model"
    present.write_bytes(b"x")
    with pytest.raises(CommitteeUnavailable, match="not found"):
        MACECommittee([present, tmp_path / "absent.model"])


def test_config_knows_it_has_no_committee():
    assert not MACEConfig().has_committee
    assert MACEConfig(committee_model_paths=("a", "b")).has_committee
    assert MACEConfig(committee_model_paths="a").committee_size == 1


# --------------------------------------------------------------------------- with MACE


@pytest.fixture(scope="session")
def mace_off_committee(tmp_path_factory):
    """Two architecturally identical MACE models that disagree.

    A committee cannot be built from different foundation model *sizes*:
    MACECalculator requires every member to share the same cutoff radius, and
    mace-off23-small and -medium do not. A real committee comes from
    independent fine-tunes of one foundation model. This fixture stands in for
    that by perturbing a copy of the weights, which keeps the architecture
    identical and makes the members predict differently.
    """
    if not MODEL_TESTS:
        pytest.skip("set CSBRT_MACE_MODEL_TESTS=1 to download MACE-OFF weights")
    pytest.importorskip("mace")
    import copy

    import torch
    from mace.calculators.foundations_models import mace_off

    directory = tmp_path_factory.mktemp("mace-models")
    base = mace_off(model="small", device="cpu", return_raw_model=True)
    first = directory / "member0.model"
    torch.save(base, first)

    jittered = copy.deepcopy(base)
    generator = torch.Generator().manual_seed(20260714)
    with torch.no_grad():
        for parameter in jittered.parameters():
            noise = torch.randn(
                parameter.shape, generator=generator, dtype=parameter.dtype
            )
            parameter.add_(0.02 * parameter.std().clamp(min=1e-6) * noise)
    second = directory / "member1.model"
    torch.save(jittered, second)
    return [first, second]


@pytest.mark.skipif(not MODEL_TESTS, reason="needs MACE weights")
def test_incompatible_committee_is_reported_clearly(tmp_path_factory):
    """Pairing two foundation model sizes must fail with an actionable error."""
    import torch
    from mace.calculators.foundations_models import mace_off

    directory = tmp_path_factory.mktemp("mace-mismatch")
    paths = []
    for size in ("small", "medium"):
        path = directory / f"{size}.model"
        torch.save(mace_off(model=size, device="cpu", return_raw_model=True), path)
        paths.append(path)

    committee = MACECommittee(paths, device="cpu", precision="double")
    report = committee.check_compatibility()
    assert not report["compatible"]
    assert len(report["r_max_values"]) == 2

    with pytest.raises(CommitteeUnavailable, match="fine-tunes of ONE"):
        committee.predict(np.zeros((2, 3)), [6, 6])


@pytest.mark.skipif(not MODEL_TESTS, reason="needs MACE weights")
def test_real_committee_predicts_and_disagrees(mace_off_committee):
    from conftest import LIGAND_ATOMS

    numbers = {"C": 6, "O": 8, "H": 1}
    atomic_numbers = [numbers[atom[1]] for atom in LIGAND_ATOMS]
    coords = np.asarray([atom[2] for atom in LIGAND_ATOMS], dtype=np.float64)

    committee = MACECommittee(mace_off_committee, device="cpu", precision="double")
    assert committee.check_compatibility()["compatible"]
    prediction = committee.predict(coords, atomic_numbers)
    assert prediction.size == 2
    assert prediction.forces_ev_per_ang.shape == (2, len(atomic_numbers), 3)
    assert np.all(np.isfinite(prediction.forces_ev_per_ang))
    assert committee.statistics()["evaluations"] == 1

    monitor = MACEUQMonitor(MACEConfig(), committee=committee)
    result = monitor.evaluate(coords, atomic_numbers)
    assert result.ensemble_size == 2
    assert result.sigma_f_max_ev_per_ang >= 0.0


@pytest.mark.skipif(not MODEL_TESTS, reason="needs MACE weights")
def test_real_committee_flags_a_distorted_ligand(mace_off_committee):
    """Committee disagreement must grow when the geometry leaves the manifold."""
    from conftest import LIGAND_ATOMS

    numbers = {"C": 6, "O": 8, "H": 1}
    atomic_numbers = [numbers[atom[1]] for atom in LIGAND_ATOMS]
    relaxed = np.asarray([atom[2] for atom in LIGAND_ATOMS], dtype=np.float64)
    distorted = relaxed.copy()
    distorted[2] += np.array([0.9, -0.9, 0.5])  # wrench the C-O bond

    committee = MACECommittee(mace_off_committee, device="cpu", precision="double")
    monitor = MACEUQMonitor(MACEConfig(), committee=committee)
    calm = monitor.evaluate(relaxed, atomic_numbers)
    wild = monitor.evaluate(distorted, atomic_numbers)
    assert wild.sigma_f_max_ev_per_ang > calm.sigma_f_max_ev_per_ang
