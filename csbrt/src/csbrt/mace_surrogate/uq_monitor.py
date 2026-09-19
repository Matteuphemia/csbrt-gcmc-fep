r"""Real-time uncertainty quantification for the MACE surrogate.

Two independent detectors decide whether the current frame is safe for the
surrogate. Either one firing trips the physics fallback.

**Committee force variance** -- the standard active-learning signal
(Zhang 2019; Kulichenko 2023; Duignan 2024). For a committee of :math:`M`
models over the ML region,

.. math::

    \sigma_{F,\max} = \max_{i} \Big[ \tfrac{1}{M-1}
        \sum_{m=1}^{M} \lVert \mathbf{F}_{i,m} - \bar{\mathbf{F}}_i \rVert^2
        \Big]^{1/2},
    \qquad
    \sigma_E = \Big[ \tfrac{1}{M-1}
        \sum_{m=1}^{M} (E_m - \bar{E})^2 \Big]^{1/2}

with the default trigger at :math:`\sigma_{F,\max} > 0.05\ \mathrm{eV/\AA}`
(:math:`\approx 1.15\ \mathrm{kcal/mol/\AA}`).

**Geometry guard** -- a committee only measures disagreement. Models trained on
the same data agree confidently on configurations none of them ever saw, which
is exactly the failure that puts a ligand through a protein wall. This check is
O(N^2) over 40-250 atoms, costs microseconds, needs no second model, and catches
the unphysical geometries: atoms closer than ``geometry_min_distance_ang``, or a
covalent bond stretched past ``geometry_max_bond_scale`` times the sum of
covalent radii.

Thresholds default to literature values and can be recalibrated against an
in-distribution trajectory with :meth:`MACEUQMonitor.calibrate_thresholds`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Sequence

import numpy as np

from .config import MACEConfig
from .units import (
    EV_PER_ANGSTROM_TO_KCAL_PER_MOL_ANG,
    EV_PER_ANGSTROM_TO_KJ_PER_MOL_NM,
    EV_TO_KCAL_PER_MOL,
    KCAL_TO_KJ,
    energy_to_ev,
    force_to_ev_per_angstrom,
)

logger = logging.getLogger("csbrt.mace_surrogate.uq")

#: Cordero et al. (2008) covalent radii in Angstroms, for the elements MACE-OFF
#: covers plus the common ions that can appear in a prepared system.
COVALENT_RADII_ANG = {
    1: 0.31,
    3: 1.28,
    6: 0.76,
    7: 0.71,
    8: 0.66,
    9: 0.57,
    11: 1.66,
    12: 1.41,
    15: 1.07,
    16: 1.05,
    17: 1.02,
    19: 2.03,
    20: 1.76,
    35: 1.20,
    53: 1.39,
}
DEFAULT_COVALENT_RADIUS_ANG = 0.80


@dataclass
class UQResult:
    """One uncertainty evaluation."""

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
    geometry_ok: bool = True
    geometry_reason: str | None = None
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
            "geometry_ok": self.geometry_ok,
            "geometry_reason": self.geometry_reason,
        }

    @classmethod
    def in_distribution(cls, ensemble_size: int = 1) -> "UQResult":
        """A zero-uncertainty result, for when there is nothing to measure."""
        return cls(
            sigma_f_max_ev_per_ang=0.0,
            sigma_f_max_kcal_per_mol_ang=0.0,
            sigma_f_max_kj_per_mol_nm=0.0,
            sigma_e_ev=0.0,
            sigma_e_kcal_per_mol=0.0,
            sigma_e_kj_per_mol=0.0,
            is_ood=False,
            ensemble_size=ensemble_size,
        )


class GeometryGuard:
    """Cheap physical-plausibility check over the ML region.

    Works with a single model, so it is the only uncertainty signal available
    before active learning has produced a committee.
    """

    def __init__(
        self,
        atomic_numbers: Sequence[int],
        *,
        min_distance_ang: float = 0.70,
        max_bond_scale: float = 1.60,
        bonds: Sequence[tuple[int, int]] = (),
    ) -> None:
        self.atomic_numbers = np.asarray(list(atomic_numbers), dtype=np.int64)
        self.min_distance_ang = float(min_distance_ang)
        self.max_bond_scale = float(max_bond_scale)
        self.bonds = [(int(a), int(b)) for a, b in bonds]
        radii = np.array(
            [
                COVALENT_RADII_ANG.get(int(z), DEFAULT_COVALENT_RADIUS_ANG)
                for z in self.atomic_numbers
            ],
            dtype=np.float64,
        )
        self.radii = radii
        if self.bonds:
            pairs = np.asarray(self.bonds, dtype=np.int64)
            self._bond_pairs = pairs
            self._bond_limits = self.max_bond_scale * (
                radii[pairs[:, 0]] + radii[pairs[:, 1]]
            )
        else:
            self._bond_pairs = np.empty((0, 2), dtype=np.int64)
            self._bond_limits = np.empty((0,), dtype=np.float64)

    def check(self, coords_ang: np.ndarray) -> tuple[bool, str | None]:
        """Return ``(ok, reason)`` for one ML-region configuration."""
        coords = np.asarray(coords_ang, dtype=np.float64)
        if not np.all(np.isfinite(coords)):
            return False, "ML region contains non-finite coordinates"

        count = coords.shape[0]
        if count > 1:
            deltas = coords[:, None, :] - coords[None, :, :]
            distances = np.sqrt(np.einsum("ijk,ijk->ij", deltas, deltas))
            np.fill_diagonal(distances, np.inf)
            closest = float(distances.min())
            if closest < self.min_distance_ang:
                i, j = np.unravel_index(int(distances.argmin()), distances.shape)
                return False, (
                    f"steric clash: atoms {int(i)} and {int(j)} are "
                    f"{closest:.2f} A apart (< {self.min_distance_ang:.2f} A)"
                )

        if len(self._bond_pairs):
            bonded = coords[self._bond_pairs[:, 0]] - coords[self._bond_pairs[:, 1]]
            lengths = np.sqrt(np.einsum("ij,ij->i", bonded, bonded))
            overshoot = lengths - self._bond_limits
            worst = int(np.argmax(overshoot))
            if overshoot[worst] > 0.0:
                a, b = self._bond_pairs[worst]
                return False, (
                    f"broken bond: atoms {int(a)}-{int(b)} at "
                    f"{lengths[worst]:.2f} A (limit {self._bond_limits[worst]:.2f} A)"
                )
        return True, None


class MACEUQMonitor:
    """Runtime uncertainty monitor: committee variance plus geometry guard."""

    def __init__(
        self,
        config: MACEConfig | None = None,
        committee: Any | None = None,
        geometry_guard: GeometryGuard | None = None,
    ) -> None:
        self.config = config or MACEConfig()
        self.committee = committee
        self.geometry_guard = geometry_guard
        self.force_threshold_ev_per_ang = self.config.uq_force_threshold_ev_per_ang
        self.energy_threshold_kcal = self.config.uq_energy_threshold_kcal_per_mol

        self.total_evaluations = 0
        self.ood_triggers = 0
        self.geometry_triggers = 0
        self.committee_triggers = 0
        self.max_observed_sigma_f_ev = 0.0
        self.max_observed_sigma_e_kcal = 0.0
        self._history_sigma_f: list[float] = []
        self._history_sigma_e: list[float] = []

    # --- primary entry point -------------------------------------------------

    def evaluate(
        self,
        coords_ang: np.ndarray,
        atomic_numbers: Sequence[int] | None = None,
        box_ang: np.ndarray | None = None,
    ) -> UQResult:
        """Evaluate the current ML-region configuration.

        Runs the geometry guard first -- it is orders of magnitude cheaper than
        a forward pass, and a clashed frame needs no committee to be rejected.
        """
        coords = np.asarray(coords_ang, dtype=np.float64)

        geometry_ok, geometry_reason = True, None
        if self.geometry_guard is not None:
            geometry_ok, geometry_reason = self.geometry_guard.check(coords)

        if not geometry_ok:
            result = UQResult.in_distribution()
            result.is_ood = True
            result.geometry_ok = False
            result.geometry_reason = geometry_reason
            result.trigger_reason = f"geometry guard: {geometry_reason}"
            self.geometry_triggers += 1
            self.record_result(result)
            return result

        if self.committee is None:
            result = UQResult.in_distribution()
            result.geometry_ok = True
            self.record_result(result)
            return result

        if atomic_numbers is None:
            raise ValueError("atomic_numbers are required for committee inference")
        prediction = self.committee.predict(coords, atomic_numbers, box_ang=box_ang)
        result = self.compute_from_ensemble_predictions(
            prediction.forces_ev_per_ang,
            prediction.energies_ev,
            record=False,
        )
        result.geometry_ok = True
        result.metadata["committee_seconds"] = prediction.wall_seconds
        if result.is_ood:
            self.committee_triggers += 1
        self.record_result(result)
        return result

    # --- variance arithmetic -------------------------------------------------

    def compute_from_ensemble_predictions(
        self,
        forces_list: Sequence[np.ndarray],
        energies_list: Sequence[float],
        ml_atom_indices: Sequence[int] | None = None,
        units_forces: str = "ev_per_ang",
        units_energy: str = "ev",
        record: bool = True,
    ) -> UQResult:
        """Committee variance from raw per-model predictions.

        ``forces_list`` is M arrays of shape (N, 3); ``energies_list`` is M
        scalars. ``ml_atom_indices`` optionally restricts sigma_F to a subset of
        the N rows (use it when the predictions cover the whole system rather
        than the ML region alone) and is interpreted as positions into those
        rows.
        """
        forces = np.asarray(forces_list, dtype=np.float64)
        energies = np.asarray(energies_list, dtype=np.float64).reshape(-1)
        members = forces.shape[0]
        if energies.shape[0] != members:
            raise ValueError(
                f"Got {members} force sets but {energies.shape[0]} energies"
            )
        if members < 2:
            # No variance to measure. Report zero rather than inventing an
            # uncertainty: the geometry guard is the only live detector.
            result = UQResult.in_distribution(ensemble_size=members)
            if record:
                self.record_result(result)
            return result

        forces_ev = force_to_ev_per_angstrom(forces, units_forces)
        energies_ev = energy_to_ev(energies, units_energy)

        if ml_atom_indices is not None and len(ml_atom_indices) > 0:
            selection = list(int(i) for i in ml_atom_indices)
            active = forces_ev[:, selection, :]
        else:
            selection = list(range(forces_ev.shape[1]))
            active = forces_ev

        mean_force = active.mean(axis=0)
        deviation = active - mean_force[np.newaxis, :, :]
        # Per atom: (1/(M-1)) * sum_m ||F_im - Fbar_i||^2, then sqrt.
        variance_per_atom = np.einsum("mij,mij->i", deviation, deviation) / (
            members - 1
        )
        sigma_per_atom = np.sqrt(np.maximum(variance_per_atom, 0.0))

        sigma_f_max_ev = float(sigma_per_atom.max())
        mean_sigma_f_ev = float(sigma_per_atom.mean())
        local_max = int(sigma_per_atom.argmax())
        max_atom_idx = selection[local_max] if selection else local_max

        energy_mean = float(energies_ev.mean())
        energy_variance = float(
            np.sum((energies_ev - energy_mean) ** 2) / (members - 1)
        )
        sigma_e_ev = float(np.sqrt(max(energy_variance, 0.0)))
        sigma_e_kcal = sigma_e_ev * EV_TO_KCAL_PER_MOL

        force_ood = sigma_f_max_ev > self.force_threshold_ev_per_ang
        energy_ood = sigma_e_kcal > self.energy_threshold_kcal
        reason = None
        if force_ood and energy_ood:
            reason = (
                f"committee sigma_F {sigma_f_max_ev:.4f} eV/A > "
                f"{self.force_threshold_ev_per_ang:.4f} and sigma_E "
                f"{sigma_e_kcal:.3f} kcal/mol > {self.energy_threshold_kcal:.3f}"
            )
        elif force_ood:
            reason = (
                f"committee sigma_F {sigma_f_max_ev:.4f} eV/A > threshold "
                f"{self.force_threshold_ev_per_ang:.4f} eV/A (atom {max_atom_idx})"
            )
        elif energy_ood:
            reason = (
                f"committee sigma_E {sigma_e_kcal:.3f} kcal/mol > threshold "
                f"{self.energy_threshold_kcal:.3f} kcal/mol"
            )

        result = UQResult(
            sigma_f_max_ev_per_ang=sigma_f_max_ev,
            sigma_f_max_kcal_per_mol_ang=sigma_f_max_ev
            * EV_PER_ANGSTROM_TO_KCAL_PER_MOL_ANG,
            sigma_f_max_kj_per_mol_nm=sigma_f_max_ev
            * EV_PER_ANGSTROM_TO_KJ_PER_MOL_NM,
            sigma_e_ev=sigma_e_ev,
            sigma_e_kcal_per_mol=sigma_e_kcal,
            sigma_e_kj_per_mol=sigma_e_kcal * KCAL_TO_KJ,
            is_ood=bool(force_ood or energy_ood),
            trigger_reason=reason,
            max_force_atom_idx=int(max_atom_idx),
            mean_sigma_f_ev_per_ang=mean_sigma_f_ev,
            ensemble_size=members,
        )
        if record:
            self.record_result(result)
        return result

    # --- bookkeeping ---------------------------------------------------------

    def record_result(self, result: UQResult) -> None:
        self.total_evaluations += 1
        if result.is_ood:
            self.ood_triggers += 1
        self.max_observed_sigma_f_ev = max(
            self.max_observed_sigma_f_ev, result.sigma_f_max_ev_per_ang
        )
        self.max_observed_sigma_e_kcal = max(
            self.max_observed_sigma_e_kcal, result.sigma_e_kcal_per_mol
        )
        self._history_sigma_f.append(result.sigma_f_max_ev_per_ang)
        self._history_sigma_e.append(result.sigma_e_kcal_per_mol)

    @property
    def history_sigma_f(self) -> list[float]:
        return list(self._history_sigma_f)

    @property
    def history_sigma_e(self) -> list[float]:
        return list(self._history_sigma_e)

    def get_statistics(self) -> dict[str, Any]:
        rate = (
            self.ood_triggers / self.total_evaluations
            if self.total_evaluations
            else 0.0
        )
        return {
            "total_evaluations": self.total_evaluations,
            "ood_triggers": self.ood_triggers,
            "committee_triggers": self.committee_triggers,
            "geometry_triggers": self.geometry_triggers,
            "ood_rate": rate,
            "mean_sigma_f_ev_per_ang": (
                float(np.mean(self._history_sigma_f)) if self._history_sigma_f else 0.0
            ),
            "mean_sigma_e_kcal_per_mol": (
                float(np.mean(self._history_sigma_e)) if self._history_sigma_e else 0.0
            ),
            "max_sigma_f_ev_per_ang": self.max_observed_sigma_f_ev,
            "max_sigma_e_kcal_per_mol": self.max_observed_sigma_e_kcal,
            "force_threshold_ev_per_ang": self.force_threshold_ev_per_ang,
            "energy_threshold_kcal_per_mol": self.energy_threshold_kcal,
            "committee": (
                self.committee.statistics() if self.committee is not None else None
            ),
        }

    def calibrate_thresholds(
        self,
        baseline_sigma_f: Sequence[float],
        baseline_sigma_e: Sequence[float],
        percentile: float = 99.0,
        *,
        floor_sigma_f_ev_per_ang: float = 0.02,
        floor_sigma_e_kcal_per_mol: float = 0.5,
    ) -> dict[str, float]:
        """Set thresholds from an in-distribution trajectory.

        The literature default (0.05 eV/A) is a starting point, not a property
        of a particular committee. Running the committee over a converged
        classical trajectory of the same complex and taking a high percentile of
        the observed sigma gives a threshold that fires on genuine novelty
        rather than on the committee's baseline spread. Floors stop a
        suspiciously tight baseline from producing a threshold that fires
        constantly.
        """
        if len(baseline_sigma_f) == 0 or len(baseline_sigma_e) == 0:
            raise ValueError("Calibration needs a non-empty baseline")
        self.force_threshold_ev_per_ang = max(
            float(np.percentile(baseline_sigma_f, percentile)),
            floor_sigma_f_ev_per_ang,
        )
        self.energy_threshold_kcal = max(
            float(np.percentile(baseline_sigma_e, percentile)),
            floor_sigma_e_kcal_per_mol,
        )
        logger.info(
            "Calibrated UQ thresholds at the %.1fth percentile of %d baseline "
            "frames: sigma_F=%.4f eV/A, sigma_E=%.3f kcal/mol",
            percentile,
            len(baseline_sigma_f),
            self.force_threshold_ev_per_ang,
            self.energy_threshold_kcal,
        )
        return {
            "calibrated_force_threshold_ev_per_ang": self.force_threshold_ev_per_ang,
            "calibrated_energy_threshold_kcal": self.energy_threshold_kcal,
        }
