"""Unit tests for MACE uncertainty quantification monitor."""

import pytest
import numpy as np

from csbrt.mace_surrogate import (
    MACEConfig,
    MACEUQMonitor,
    UQResult,
)


def test_uq_monitor_initialization():
    cfg = MACEConfig(uq_force_threshold_ev_per_ang=0.05, uq_energy_threshold_kcal_per_mol=1.0)
    mon = MACEUQMonitor(config=cfg)
    assert mon.config == cfg
    assert mon.force_threshold_ev == 0.05


def test_compute_from_ensemble_predictions_relaxed():
    cfg = MACEConfig(uq_force_threshold_ev_per_ang=0.05, uq_energy_threshold_kcal_per_mol=1.0)
    mon = MACEUQMonitor(config=cfg)

    # 4 models agreeing very closely (low variance)
    n_atoms = 20
    base_forces = np.random.randn(n_atoms, 3) * 100.0  # in kJ/(mol*nm)
    forces_ensemble = [base_forces + np.random.randn(n_atoms, 3) * 0.001 for _ in range(4)]
    energies_ensemble = [-500.0, -500.0001, -499.9999, -500.0]  # kJ/mol

    res = mon.compute_from_ensemble_predictions(
        forces_ensemble,
        energies_ensemble,
        units_forces="kj_per_mol_nm",
        units_energy="kj_per_mol",
    )
    assert isinstance(res, UQResult)
    assert res.is_ood is False
    assert res.trigger_reason is None
    assert res.ensemble_size == 4
    assert res.sigma_f_max_ev_per_ang < 0.05


def test_compute_from_ensemble_predictions_ood_force():
    cfg = MACEConfig(uq_force_threshold_ev_per_ang=0.05, uq_energy_threshold_kcal_per_mol=1.0)
    mon = MACEUQMonitor(config=cfg)

    n_atoms = 10
    # Model 1 and Model 2 disagree widely on atom 3
    f1 = np.zeros((n_atoms, 3))
    f2 = np.zeros((n_atoms, 3))
    # 0.1 eV/A difference = ~965 kJ/(mol*nm)
    f2[3, 0] = 1000.0

    res = mon.compute_from_ensemble_predictions([f1, f2], [0.0, 0.0])
    assert res.is_ood is True
    assert "Force uncertainty" in res.trigger_reason
    assert res.max_force_atom_idx == 3


def test_compute_from_ensemble_predictions_ood_energy():
    cfg = MACEConfig(uq_force_threshold_ev_per_ang=0.10, uq_energy_threshold_kcal_per_mol=1.0)
    mon = MACEUQMonitor(config=cfg)

    n_atoms = 5
    f1 = np.zeros((n_atoms, 3))
    f2 = np.zeros((n_atoms, 3))
    # Disagreement of 10 kcal/mol (~41.84 kJ/mol)
    res = mon.compute_from_ensemble_predictions([f1, f2], [0.0, 41.84])
    assert res.is_ood is True
    assert "Energy uncertainty" in res.trigger_reason
