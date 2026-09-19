"""Stage 2: committee force variance and the geometry guard."""

from __future__ import annotations

import numpy as np
import pytest

from csbrt.mace_surrogate import (
    EV_PER_ANGSTROM_TO_KCAL_PER_MOL_ANG,
    EV_PER_ANGSTROM_TO_KJ_PER_MOL_NM,
    GeometryGuard,
    MACEConfig,
    MACEUQMonitor,
)
from csbrt.mace_surrogate.units import (
    EV_TO_KJ_PER_MOL,
    energy_to_ev,
    force_to_ev_per_angstrom,
)


RNG = np.random.default_rng(20260714)


def committee(mean_force: np.ndarray, spread: float, members: int = 4):
    """M force sets scattered around ``mean_force`` by ``spread`` eV/A."""
    return [mean_force + RNG.normal(0.0, spread, mean_force.shape)
            for _ in range(members)]


# --------------------------------------------------------------------------- units


def test_unit_constants_are_self_consistent():
    # 1 eV/A is 96.485 kJ/mol per Angstrom, i.e. 964.85 kJ/mol/nm. Getting the
    # A/nm factor wrong here silently scales every force threshold by ten.
    assert EV_PER_ANGSTROM_TO_KJ_PER_MOL_NM == pytest.approx(
        EV_TO_KJ_PER_MOL * 10.0
    )
    assert EV_PER_ANGSTROM_TO_KCAL_PER_MOL_ANG == pytest.approx(23.0605, abs=1e-3)
    assert MACEConfig().uq_force_threshold_kj_per_mol_nm == pytest.approx(
        48.24, abs=0.01
    )
    assert MACEConfig().uq_force_threshold_kcal_per_mol_ang == pytest.approx(
        1.153, abs=0.01
    )


def test_unit_helpers_round_trip():
    assert force_to_ev_per_angstrom(
        EV_PER_ANGSTROM_TO_KJ_PER_MOL_NM, "kj_per_mol_nm"
    ) == pytest.approx(1.0)
    assert energy_to_ev(EV_TO_KJ_PER_MOL, "kj_per_mol") == pytest.approx(1.0)
    with pytest.raises(ValueError, match="Unknown force unit"):
        force_to_ev_per_angstrom(1.0, "furlongs")


# --------------------------------------------------------------------------- variance


def test_relaxed_pose_is_in_distribution():
    """Tight committee agreement stays below the 0.02 eV/A pass criterion."""
    monitor = MACEUQMonitor(MACEConfig())
    forces = committee(RNG.normal(0.0, 1.0, (30, 3)), spread=0.002)
    result = monitor.compute_from_ensemble_predictions(
        forces, [1.0, 1.0002, 0.9999, 1.0001]
    )
    assert not result.is_ood
    assert result.sigma_f_max_ev_per_ang < 0.02
    assert result.ensemble_size == 4


def test_perturbed_pose_is_flagged():
    """Committee disagreement above 0.10 eV/A must trip the fallback."""
    monitor = MACEUQMonitor(MACEConfig())
    forces = committee(RNG.normal(0.0, 1.0, (30, 3)), spread=0.25)
    result = monitor.compute_from_ensemble_predictions(
        forces, [1.0, 1.2, 0.7, 1.4]
    )
    assert result.is_ood
    assert result.sigma_f_max_ev_per_ang > 0.10
    assert "sigma_F" in result.trigger_reason


def test_sigma_matches_the_documented_formula():
    """sigma_F is sqrt of the (M-1)-normalised mean squared force deviation."""
    forces = np.zeros((3, 2, 3))
    forces[0, 1] = [0.3, 0.0, 0.0]
    forces[1, 1] = [-0.3, 0.0, 0.0]
    forces[2, 1] = [0.0, 0.0, 0.0]
    monitor = MACEUQMonitor(MACEConfig())
    result = monitor.compute_from_ensemble_predictions(forces, [0.0, 0.0, 0.0])
    # mean = 0; sum ||dev||^2 = 0.09 + 0.09 = 0.18; /(M-1)=2 -> 0.09; sqrt=0.3
    assert result.sigma_f_max_ev_per_ang == pytest.approx(0.3)
    assert result.max_force_atom_idx == 1
    assert result.sigma_f_max_kcal_per_mol_ang == pytest.approx(
        0.3 * EV_PER_ANGSTROM_TO_KCAL_PER_MOL_ANG
    )


def test_energy_only_trigger():
    monitor = MACEUQMonitor(MACEConfig(uq_energy_threshold_kcal_per_mol=0.5))
    forces = committee(np.zeros((5, 3)), spread=0.0005)
    # 0.05 eV spread is ~1.15 kcal/mol, above the 0.5 kcal/mol threshold.
    result = monitor.compute_from_ensemble_predictions(
        forces, [0.00, 0.05, 0.10, 0.15]
    )
    assert result.is_ood
    assert "sigma_E" in result.trigger_reason


def test_single_model_reports_no_variance():
    monitor = MACEUQMonitor(MACEConfig())
    result = monitor.compute_from_ensemble_predictions(
        [RNG.normal(0.0, 1.0, (10, 3))], [1.0]
    )
    assert result.ensemble_size == 1
    assert result.sigma_f_max_ev_per_ang == 0.0
    assert not result.is_ood


def test_mismatched_ensemble_lengths_raise():
    monitor = MACEUQMonitor(MACEConfig())
    with pytest.raises(ValueError, match="force sets"):
        monitor.compute_from_ensemble_predictions(
            [np.zeros((4, 3)), np.zeros((4, 3))], [1.0]
        )


def test_input_units_are_honoured():
    monitor = MACEUQMonitor(MACEConfig())
    forces_ev = np.zeros((2, 2, 3))
    forces_ev[0, 0, 0] = 0.1
    in_ev = monitor.compute_from_ensemble_predictions(forces_ev, [0.0, 0.0])
    in_kj = monitor.compute_from_ensemble_predictions(
        forces_ev * EV_PER_ANGSTROM_TO_KJ_PER_MOL_NM,
        [0.0, 0.0],
        units_forces="kj_per_mol_nm",
    )
    assert in_kj.sigma_f_max_ev_per_ang == pytest.approx(
        in_ev.sigma_f_max_ev_per_ang
    )


def test_ml_atom_subset_restricts_sigma():
    forces = np.zeros((2, 4, 3))
    forces[0, 3] = [5.0, 0.0, 0.0]  # a huge disagreement outside the ML region
    monitor = MACEUQMonitor(MACEConfig())
    whole = monitor.compute_from_ensemble_predictions(forces, [0.0, 0.0])
    subset = monitor.compute_from_ensemble_predictions(
        forces, [0.0, 0.0], ml_atom_indices=[0, 1, 2]
    )
    assert whole.is_ood
    assert not subset.is_ood


# --------------------------------------------------------------------------- geometry


def ethanol_coords():
    from conftest import LIGAND_ATOMS

    return np.asarray([atom[2] for atom in LIGAND_ATOMS], dtype=np.float64)


def ethanol_guard(**kwargs):
    from conftest import LIGAND_ATOMS, LIGAND_BONDS

    numbers = {"C": 6, "O": 8, "H": 1}
    return GeometryGuard(
        [numbers[atom[1]] for atom in LIGAND_ATOMS],
        bonds=LIGAND_BONDS,
        **kwargs,
    )


def test_geometry_guard_passes_a_relaxed_pose():
    ok, reason = ethanol_guard().check(ethanol_coords())
    assert ok and reason is None


def test_geometry_guard_catches_a_steric_clash():
    coords = ethanol_coords()
    coords[8] = coords[3] + 0.2  # drop the hydroxyl H onto a methyl H
    ok, reason = ethanol_guard().check(coords)
    assert not ok
    assert "steric clash" in reason


def test_geometry_guard_catches_a_broken_bond():
    coords = ethanol_coords()
    coords[2] = coords[2] + np.array([6.0, 0.0, 0.0])  # tear the C-O bond open
    ok, reason = ethanol_guard().check(coords)
    assert not ok
    assert "broken bond" in reason


def test_geometry_guard_catches_nan():
    coords = ethanol_coords()
    coords[0, 0] = np.nan
    ok, reason = ethanol_guard().check(coords)
    assert not ok
    assert "non-finite" in reason


def test_geometry_guard_fires_without_a_committee():
    """Intercept rate with a single model: the guard is the whole detector."""
    monitor = MACEUQMonitor(MACEConfig(), geometry_guard=ethanol_guard())
    good = monitor.evaluate(ethanol_coords())
    assert not good.is_ood

    distorted = ethanol_coords()
    distorted[8] = distorted[4] + 0.05
    bad = monitor.evaluate(distorted)
    assert bad.is_ood
    assert not bad.geometry_ok
    assert monitor.geometry_triggers == 1


def test_geometry_guard_intercepts_every_injected_ood_frame():
    """Stage 6 pass criterion: 100% intercept on injected OOD configurations."""
    guard = ethanol_guard()
    monitor = MACEUQMonitor(MACEConfig(), geometry_guard=guard)
    base = ethanol_coords()
    injected = []
    for index in range(20):
        coords = base.copy()
        # Collapse a random pair of atoms onto each other.
        first, second = RNG.choice(len(coords), size=2, replace=False)
        coords[first] = coords[second] + RNG.normal(0.0, 0.05, 3)
        injected.append(coords)
    flagged = sum(monitor.evaluate(coords).is_ood for coords in injected)
    assert flagged == len(injected)


# --------------------------------------------------------------------------- stats


def test_statistics_and_calibration():
    monitor = MACEUQMonitor(MACEConfig())
    baseline = []
    for _ in range(50):
        forces = committee(np.zeros((6, 3)), spread=0.003)
        result = monitor.compute_from_ensemble_predictions(
            forces, [0.0, 0.001, 0.002, 0.0015]
        )
        baseline.append(result.sigma_f_max_ev_per_ang)
    stats = monitor.get_statistics()
    assert stats["total_evaluations"] == 50
    assert stats["ood_triggers"] == 0
    assert stats["ood_rate"] == 0.0

    calibrated = monitor.calibrate_thresholds(baseline, [0.01] * 50)
    # The baseline spread is far below the floor, so the floor wins.
    assert calibrated["calibrated_force_threshold_ev_per_ang"] == pytest.approx(0.02)
    assert calibrated["calibrated_energy_threshold_kcal"] == pytest.approx(0.5)


def test_calibration_raises_on_empty_baseline():
    with pytest.raises(ValueError, match="non-empty"):
        MACEUQMonitor(MACEConfig()).calibrate_thresholds([], [])
