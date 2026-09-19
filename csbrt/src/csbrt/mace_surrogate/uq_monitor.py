"""Real-time Uncertainty Quantification (UQ) Runtime Monitor for MACE.

Computes prediction variance across a committee of MACE models or ensemble heads:
$$\\sigma_{F, \\max} = \\max_{i \\in \\text{ML region}} \\left[ \\frac{1}{M-1} \\sum_{m=1}^M \\left\\| \\mathbf{F}_{i, m} - \\bar{\\mathbf{F}}_i \\right\\|^2 \\right]^{1/2}$$
$$\\sigma_E = \\left[ \\frac{1}{M-1} \\sum_{m=1}^M (E_m - \\bar{E})^2 \\right]^{1/2}$$

When uncertainty exceeds calibrated thresholds (default: 0.05 eV/Å force or 1.0 kcal/mol energy),
the frame is marked out-of-distribution (OOD) to trigger physics fallback and active learning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from typing import Any, Sequence

import numpy as np

from .mace_mixed_system import (
    EV_PER_ANGSTROM_TO_KCAL_PER_MOL_ANG,
    EV_PER_ANGSTROM_TO_KJ_PER_MOL_NM,
    EV_TO_KCAL_PER_MOL,
    EV_TO_KJ_PER_MOL,
    KCAL_TO_KJ,
    MACEConfig,
)

logger = logging.getLogger("csbrt.mace_surrogate.uq")


@dataclass
class UQResult:
    """Evaluation result for uncertainty quantification."""

    sigma_f_max_ev_per_ang: float
    sigma_f_max_kcal_per_mol_ang: float
    sigma_f_max_kj_per_mol_nm: float
    sigma_e_ev: float
    sigma_e_kcal_per_mol: float
    sigma_e_kj_per_mol: float
    is_ood: bool
    trigger_reason: str | None = None
    max_force_atom_idx: int | None = None
    mean_sigma_f_ev_per_ang: float = 0.0
    ensemble_size: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sigma_f_max_ev_per_ang": self.sigma_f_max_ev_per_ang,
            "sigma_f_max_kcal_per_mol_ang": self.sigma_f_max_kcal_per_mol_ang,
            "sigma_e_kcal_per_mol": self.sigma_e_kcal_per_mol,
            "is_ood": self.is_ood,
            "trigger_reason": self.trigger_reason,
            "max_force_atom_idx": self.max_force_atom_idx,
            "mean_sigma_f_ev_per_ang": self.mean_sigma_f_ev_per_ang,
            "ensemble_size": self.ensemble_size,
        }


class MACEUQMonitor:
    """Runtime monitor tracking MACE uncertainty across dynamic simulation steps."""

    def __init__(
        self,
        config: MACEConfig | None = None,
        models: Sequence[Any] | None = None,
    ) -> None:
        self.config = config or MACEConfig()
        self.models = list(models) if models else []
        self.force_threshold_ev_per_ang = self.config.uq_force_threshold_ev_per_ang
        self.energy_threshold_kcal = self.config.uq_energy_threshold_kcal_per_mol

        # Tracking statistics
        self.total_evaluations: int = 0
        self.ood_triggers: int = 0
        self.max_observed_sigma_f_ev: float = 0.0
        self.max_observed_sigma_e_kcal: float = 0.0
        self._history_sigma_f: list[float] = []
        self._history_sigma_e: list[float] = []

    def compute_from_ensemble_predictions(
        self,
        forces_list: Sequence[np.ndarray],
        energies_list: Sequence[float],
        ml_atom_indices: Sequence[int] | None = None,
        units_forces: str = "ev_per_ang",
        units_energy: str = "ev",
    ) -> UQResult:
        """Compute UQ metrics from an ensemble of force and energy predictions.

        Parameters
        ----------
        forces_list:
            List of M arrays, each of shape (N, 3).
        energies_list:
            List or array of M scalar energies.
        ml_atom_indices:
            Optional subset of atom indices (e.g. ML region) over which to evaluate sigma_F.
        units_forces:
            Input force unit: 'ev_per_ang', 'kcal_per_mol_ang', or 'kj_per_mol_nm'.
        units_energy:
            Input energy unit: 'ev', 'kcal_per_mol', or 'kj_per_mol'.
        """
        M = len(forces_list)
        if M < 2:
            # Single model: UQ cannot be computed from variance; return zero / in-distribution
            return UQResult(
                sigma_f_max_ev_per_ang=0.0,
                sigma_f_max_kcal_per_mol_ang=0.0,
                sigma_f_max_kj_per_mol_nm=0.0,
                sigma_e_ev=0.0,
                sigma_e_kcal_per_mol=0.0,
                sigma_e_kj_per_mol=0.0,
                is_ood=False,
                ensemble_size=M,
            )

        forces_arr = np.stack(forces_list, axis=0)  # Shape: (M, N, 3)
        energies_arr = np.array(energies_list, dtype=np.float64)  # Shape: (M,)

        # Convert forces to eV/Å for uniform threshold comparison
        if units_forces == "kcal_per_mol_ang":
            forces_ev = forces_arr / EV_PER_ANGSTROM_TO_KCAL_PER_MOL_ANG
        elif units_forces == "kj_per_mol_nm":
            forces_ev = forces_arr / EV_PER_ANGSTROM_TO_KJ_PER_MOL_NM
        else:
            forces_ev = forces_arr

        # Convert energies to kcal/mol and eV
        if units_energy == "ev":
            energies_ev = energies_arr
            energies_kcal = energies_arr * EV_TO_KCAL_PER_MOL
        elif units_energy == "kj_per_mol":
            energies_ev = energies_arr / EV_TO_KJ_PER_MOL
            energies_kcal = energies_arr / KCAL_TO_KJ
        else:  # kcal_per_mol
            energies_ev = energies_arr / EV_TO_KCAL_PER_MOL
            energies_kcal = energies_arr

        # Filter by ML atom indices if provided
        if ml_atom_indices is not None and len(ml_atom_indices) > 0:
            active_forces = forces_ev[:, ml_atom_indices, :]
            index_mapping = list(ml_atom_indices)
        else:
            active_forces = forces_ev
            index_mapping = list(range(forces_ev.shape[1]))

        # Calculate force variance across ensemble:
        # F_mean: (N_active, 3)
        f_mean = np.mean(active_forces, axis=0)
        # Residuals: (M, N_active, 3)
        diff = active_forces - f_mean[np.newaxis, :, :]
        # Squared norm per model per atom: (M, N_active)
        sq_norm = np.sum(diff ** 2, axis=-1)
        # Sample variance per atom: (N_active,)
        force_var_per_atom = np.sum(sq_norm, axis=0) / (M - 1)
        # Standard deviation per atom: (N_active,)
        force_std_per_atom = np.sqrt(np.maximum(force_var_per_atom, 0.0))

        sigma_f_max_ev = float(np.max(force_std_per_atom))
        mean_sigma_f_ev = float(np.mean(force_std_per_atom))
        max_idx_local = int(np.argmax(force_std_per_atom))
        max_atom_idx = index_mapping[max_idx_local] if index_mapping else max_idx_local

        # Calculate energy standard deviation
        e_mean = np.mean(energies_kcal)
        e_var = np.sum((energies_kcal - e_mean) ** 2) / (M - 1)
        sigma_e_kcal = float(np.sqrt(max(e_var, 0.0)))
        sigma_e_ev = float(np.sqrt(max(np.sum((energies_ev - np.mean(energies_ev)) ** 2) / (M - 1), 0.0)))

        # Unit conversions
        sigma_f_max_kcal = sigma_f_max_ev * EV_PER_ANGSTROM_TO_KCAL_PER_MOL_ANG
        sigma_f_max_kj = sigma_f_max_ev * EV_PER_ANGSTROM_TO_KJ_PER_MOL_NM
        sigma_e_kj = sigma_e_kcal * KCAL_TO_KJ

        # Check OOD thresholds
        is_ood_force = sigma_f_max_ev > self.force_threshold_ev_per_ang
        is_ood_energy = sigma_e_kcal > self.energy_threshold_kcal
        is_ood = is_ood_force or is_ood_energy

        reason = None
        if is_ood_force and is_ood_energy:
            reason = f"Both force ({sigma_f_max_ev:.4f} eV/Å) and energy ({sigma_e_kcal:.3f} kcal/mol) exceed thresholds"
        elif is_ood_force:
            reason = f"Force uncertainty {sigma_f_max_ev:.4f} eV/Å > threshold {self.force_threshold_ev_per_ang:.4f} eV/Å"
        elif is_ood_energy:
            reason = f"Energy uncertainty {sigma_e_kcal:.3f} kcal/mol > threshold {self.energy_threshold_kcal:.3f} kcal/mol"

        result = UQResult(
            sigma_f_max_ev_per_ang=sigma_f_max_ev,
            sigma_f_max_kcal_per_mol_ang=sigma_f_max_kcal,
            sigma_f_max_kj_per_mol_nm=sigma_f_max_kj,
            sigma_e_ev=sigma_e_ev,
            sigma_e_kcal_per_mol=sigma_e_kcal,
            sigma_e_kj_per_mol=sigma_e_kj,
            is_ood=is_ood,
            trigger_reason=reason,
            max_force_atom_idx=max_atom_idx,
            mean_sigma_f_ev_per_ang=mean_sigma_f_ev,
            ensemble_size=M,
        )

        self.record_result(result)
        return result

    def record_result(self, result: UQResult) -> None:
        """Update tracker statistics."""
        self.total_evaluations += 1
        if result.is_ood:
            self.ood_triggers += 1
        if result.sigma_f_max_ev_per_ang > self.max_observed_sigma_f_ev:
            self.max_observed_sigma_f_ev = result.sigma_f_max_ev_per_ang
        if result.sigma_e_kcal_per_mol > self.max_observed_sigma_e_kcal:
            self.max_observed_sigma_e_kcal = result.sigma_e_kcal_per_mol

        self._history_sigma_f.append(result.sigma_f_max_ev_per_ang)
        self._history_sigma_e.append(result.sigma_e_kcal_per_mol)

    def get_statistics(self) -> dict[str, Any]:
        """Summary of UQ metrics and fallback rate."""
        rate = (self.ood_triggers / self.total_evaluations) if self.total_evaluations > 0 else 0.0
        mean_f = float(np.mean(self._history_sigma_f)) if self._history_sigma_f else 0.0
        mean_e = float(np.mean(self._history_sigma_e)) if self._history_sigma_e else 0.0
        return {
            "total_evaluations": self.total_evaluations,
            "ood_triggers": self.ood_triggers,
            "fallback_rate": rate,
            "mean_sigma_f_ev_per_ang": mean_f,
            "mean_sigma_e_kcal_per_mol": mean_e,
            "max_sigma_f_ev_per_ang": self.max_observed_sigma_f_ev,
            "max_sigma_e_kcal_per_mol": self.max_observed_sigma_e_kcal,
            "force_threshold_ev_per_ang": self.force_threshold_ev_per_ang,
            "energy_threshold_kcal_per_mol": self.energy_threshold_kcal,
        }

    def calibrate_thresholds(
        self,
        baseline_sigma_f: Sequence[float],
        baseline_sigma_e: Sequence[float],
        percentile: float = 99.0,
    ) -> dict[str, float]:
        """Calibrate thresholds against baseline in-distribution trajectory."""
        calibrated_f = float(np.percentile(baseline_sigma_f, percentile))
        calibrated_e = float(np.percentile(baseline_sigma_e, percentile))
        self.force_threshold_ev_per_ang = max(calibrated_f, 0.02)
        self.energy_threshold_kcal = max(calibrated_e, 0.5)
        logger.info(
            f"Calibrated UQ thresholds at {percentile}th percentile: "
            f"force={self.force_threshold_ev_per_ang:.4f} eV/Å, energy={self.energy_threshold_kcal:.3f} kcal/mol"
        )
        return {
            "calibrated_force_threshold_ev_per_ang": self.force_threshold_ev_per_ang,
            "calibrated_energy_threshold_kcal": self.energy_threshold_kcal,
        }
