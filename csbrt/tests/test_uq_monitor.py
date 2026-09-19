"""Tests for real-time uncertainty quantification (UQ) math and thresholds."""

from __future__ import annotations

import numpy as np
import pytest

from csbrt.mace_surrogate import (
    EV_PER_ANGSTROM_TO_KCAL_PER_MOL_ANG,
    EV_PER_ANGSTROM_TO_KJ_PER_MOL_NM,
    EV_TO_KCAL_PER_MOL,
    MACEConfig,
    MACEUQMonitor,
)


def _monitor(force_threshold_ev: float = 0.05, energy_threshold_kcal: float = 1.0):
    return MACEUQMonitor(
        MACEConfig(
            uq_force_threshold_ev_per_ang=force_threshold_ev,
            uq_energy_threshold_kcal_per_mol=energy_threshold_kcal,
        )
    )


def test_single_model_returns_in_distribution_zero() -> None:
    mon = _monitor()
    res = mon.compute_from_ensemble_predictions(
        [np.array([[1.0, 0.0, 0.0]])], [0.0]
    )
    assert res.ensemble_size == 1
    assert res.sigma_f_max_ev_per_ang == 0.0
    assert res.is_ood is False


def test_sigma_f_math_two_models() -> None:
    mon = _monitor(force_threshold_ev=10.0, energy_threshold_kcal=1e6)
    f0 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    f1 = np.array([[3.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    res = mon.compute_from_ensemble_predictions([f0, f1], [0.0, 0.0])

    # per-atom std = sqrt(2) eV/Å for atom 0, 0 for atom 1
    assert res.sigma_f_max_ev_per_ang == pytest.approx(np.sqrt(2.0))
    assert res.max_force_atom_idx == 0
    assert res.mean_sigma_f_ev_per_ang == pytest.approx(np.sqrt(2.0) / 2.0)
    assert res.is_ood is False  # threshold 10 eV/Å


def test_energy_uncertainty_and_ood_trigger() -> None:
    mon = _monitor(force_threshold_ev=10.0, energy_threshold_kcal=1.0)
    f0 = np.array([[0.0, 0.0, 0.0]])
    f1 = np.array([[0.0, 0.0, 0.0]])
    # energies 0 eV and 2 eV -> sigma_e = sqrt(2) eV ~ 32.6 kcal/mol
    res = mon.compute_from_ensemble_predictions([f0, f1], [0.0, 2.0])
    assert res.sigma_e_kcal_per_mol == pytest.approx(np.sqrt(2.0) * EV_TO_KCAL_PER_MOL)
    assert res.is_ood is True
    assert res.trigger_reason is not None
    assert "energy" in res.trigger_reason.lower()


def test_force_ood_trigger() -> None:
    mon = _monitor(force_threshold_ev=0.05, energy_threshold_kcal=1e6)
    f0 = np.array([[0.0, 0.0, 0.0]])
    f1 = np.array([[0.2, 0.0, 0.0]])
    res = mon.compute_from_ensemble_predictions([f0, f1], [0.0, 0.0])
    # std = |0.2|/sqrt(2)*?  For M=2: f_mean=[0.1,0,0]; diffs +-0.1;
    # per-atom sigma = sqrt(2*0.01) = sqrt(0.02) ~ 0.1414 eV/Å > 0.05
    assert res.sigma_f_max_ev_per_ang > 0.05
    assert res.is_ood is True


def test_force_unit_conversion_consistency() -> None:
    cfg = MACEConfig(uq_force_threshold_ev_per_ang=1e6, uq_energy_threshold_kcal_per_mol=1e6)
    mon_ev = MACEUQMonitor(cfg)
    f_ev = np.array([[1.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
    res_ev = mon_ev.compute_from_ensemble_predictions(
        [f_ev, f_ev * 1.5], [0.0, 0.0], units_forces="ev_per_ang"
    )

    f_kcal = f_ev * EV_PER_ANGSTROM_TO_KCAL_PER_MOL_ANG
    res_kcal = MACEUQMonitor(cfg).compute_from_ensemble_predictions(
        [f_kcal, f_kcal * 1.5], [0.0, 0.0], units_forces="kcal_per_mol_ang"
    )

    f_kj = f_ev * EV_PER_ANGSTROM_TO_KJ_PER_MOL_NM
    res_kj = MACEUQMonitor(cfg).compute_from_ensemble_predictions(
        [f_kj, f_kj * 1.5], [0.0, 0.0], units_forces="kj_per_mol_nm"
    )

    assert res_ev.sigma_f_max_ev_per_ang == pytest.approx(
        res_kcal.sigma_f_max_ev_per_ang, rel=1e-9
    )
    assert res_ev.sigma_f_max_ev_per_ang == pytest.approx(
        res_kj.sigma_f_max_ev_per_ang, rel=1e-9
    )


def test_energy_unit_conversion_consistency() -> None:
    cfg = MACEConfig(uq_force_threshold_ev_per_ang=1e6, uq_energy_threshold_kcal_per_mol=1e6)
    f = np.array([[0.0, 0.0, 0.0]])
    res_ev = MACEUQMonitor(cfg).compute_from_ensemble_predictions(
        [f, f], [0.0, 2.0], units_energy="ev"
    )
    res_kcal = MACEUQMonitor(cfg).compute_from_ensemble_predictions(
        [f, f], [0.0, 2.0 * EV_TO_KCAL_PER_MOL], units_energy="kcal_per_mol"
    )
    assert res_ev.sigma_e_kcal_per_mol == pytest.approx(
        res_kcal.sigma_e_kcal_per_mol, rel=1e-9
    )


def test_statistics_tracking() -> None:
    mon = _monitor(force_threshold_ev=0.05, energy_threshold_kcal=1.0)
    f0 = np.array([[0.0, 0.0, 0.0]])
    f1 = np.array([[0.0, 0.0, 0.0]])
    mon.compute_from_ensemble_predictions([f0, f1], [0.0, 0.0])      # in-dist
    mon.compute_from_ensemble_predictions([f0, f1], [0.0, 2.0])      # OOD energy
    stats = mon.get_statistics()
    assert stats["total_evaluations"] == 2
    assert stats["ood_triggers"] == 1
    assert stats["fallback_rate"] == pytest.approx(0.5)
    assert stats["max_sigma_e_kcal_per_mol"] > 0.0


def test_calibrate_thresholds() -> None:
    mon = _monitor()
    out = mon.calibrate_thresholds([0.01, 0.02, 0.5], [0.1, 0.2, 2.0], percentile=99.0)
    expected = float(np.percentile([0.01, 0.02, 0.5], 99.0))
    assert out["calibrated_force_threshold_ev_per_ang"] == pytest.approx(expected)
    assert mon.force_threshold_ev_per_ang == pytest.approx(expected)
    assert mon.energy_threshold_kcal == pytest.approx(max(float(np.percentile([0.1, 0.2, 2.0], 99.0)), 0.5))
