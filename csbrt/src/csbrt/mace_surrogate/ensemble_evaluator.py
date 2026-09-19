"""Committee MACE inference for runtime uncertainty quantification.

The OpenMM-ML mixed system used for the fast path embeds a *single* MACE model.
Uncertainty quantification therefore needs a small committee evaluated separately
on the ML region (plus a neighbour shell so the ML-region forces see their
chemical environment).  This module loads that committee -- foundation model
plus any fine-tuned generations -- via mace-torch and returns the per-model
energy/force predictions consumed by :class:`csbrt.mace_surrogate.MACEUQMonitor`.

Degradation contract (important for testability):

* ``mace``/``ase``/``torch`` absent  -> ``MACEEnsembleUnavailable`` is raised so
  callers can skip UQ instead of silently fabricating predictions.
* Only one model available            -> ``ensemble_size == 1``, ``sigma_F = 0``,
  which disables fallback (documented limitation of the single-model case).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .mace_mixed_system import (
    KNOWN_FOUNDATION_MODELS,
    MACEConfig,
)

logger = logging.getLogger("csbrt.mace_surrogate.ensemble")

# Map openmm-ml MACE model names to (mace calculator function, model key).
_FOUNDATION_CALC = {
    "mace-off23-small": ("mace_off", "small"),
    "mace-off23-medium": ("mace_off", "medium"),
    "mace-off23-large": ("mace_off", "large"),
    "mace-mpa-0-medium": ("mace_mp", "medium-mpa-0"),
    "mace-omat-0-small": ("mace_mp", "small-omat-0"),
    "mace-omat-0-medium": ("mace_mp", "medium-omat-0"),
    "mace-omol-0-extra-large": ("mace_omol", "extra_large"),
}

# Default model cutoff used for the environment shell (Å).
DEFAULT_SHELL_RADIUS_ANG = 6.0


class MACEEnsembleUnavailable(RuntimeError):
    """Raised when the MACE/torch/ASE stack required for committee UQ is missing."""


def _select_ml_shell(
    positions: np.ndarray,
    ml_atoms: Sequence[int],
    shell_radius_ang: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (local_atom_indices, local_ml_indices) for the ML region + shell."""
    positions = np.asarray(positions, dtype=np.float64)
    ml_atoms = list(ml_atoms)
    if not ml_atoms:
        raise ValueError("ml_atoms must be non-empty")

    ml_pos = positions[ml_atoms]
    # Distance from every atom to every ML atom (vectorised; systems are small here).
    d2 = np.sum((positions[:, None, :] - ml_pos[None, :, :]) ** 2, axis=-1)
    within = d2 <= (shell_radius_ang * shell_radius_ang)
    shell_mask = np.any(within, axis=1)
    local_indices = np.flatnonzero(shell_mask)
    local_to_global = {int(g): i for i, g in enumerate(local_indices)}
    local_ml_indices = np.array(
        [local_to_global[int(a)] for a in ml_atoms if int(a) in local_to_global],
        dtype=np.int64,
    )
    return local_indices, local_ml_indices


class MACEEnsembleEvaluator:
    """Evaluates a committee of MACE models over the ML region + shell."""

    def __init__(
        self,
        config: MACEConfig | None = None,
        model_paths: Sequence[str] | None = None,
        shell_radius_ang: float = DEFAULT_SHELL_RADIUS_ANG,
    ) -> None:
        self.config = config or MACEConfig()
        # Committee: explicit checkpoints take precedence; otherwise the
        # foundation model name alone forms a single-member committee.
        self.model_paths = [str(p) for p in model_paths] if model_paths else []
        self.shell_radius_ang = shell_radius_ang
        self._calculators: list[Any] = []
        self._loaded = False

    @property
    def committee_size(self) -> int:
        if self.model_paths:
            return len(self.model_paths)
        return 1

    def _build_calculator(self, model_ref: str) -> Any:
        """Construct one ASE calculator for a foundation name or checkpoint path."""
        try:
            from mace.calculators import (  # type: ignore[import-not-found]
                MACECalculator,
                mace_mp,
                mace_off,
                mace_omol,
                mace_polar,
            )
        except ImportError as err:
            raise MACEEnsembleUnavailable(
                "mace-torch is not installed; cannot evaluate MACE committee. "
                "Install with `pip install 'mace-torch>=0.3.10'`."
            ) from err

        device = self.config.device
        default_dtype = "float64" if self.config.precision == "float64" else "float32"

        if model_ref in _FOUNDATION_CALC:
            fn_name, model_key = _FOUNDATION_CALC[model_ref]
            fn = {"mace_off": mace_off, "mace_mp": mace_mp, "mace_omol": mace_omol,
                  "mace_polar": mace_polar}[fn_name]
            return fn(model=model_key, device=device, default_dtype=default_dtype)

        # Local checkpoint path.
        path = Path(model_ref)
        if not path.is_file():
            raise ValueError(
                f"Unknown MACE model {model_ref!r} (not a foundation model and no "
                f"such checkpoint file). Known foundation models: "
                f"{sorted(_FOUNDATION_CALC)}."
            )
        return MACECalculator(
            model_paths=[str(path)], device=device, default_dtype=default_dtype
        )

    def ensure_loaded(self) -> int:
        """Load (lazily) the committee and return its size."""
        if self._loaded:
            return len(self._calculators)

        try:
            import ase  # noqa: F401  (only to surface a clean error)
            import torch  # noqa: F401
        except ImportError as err:
            raise MACEEnsembleUnavailable(
                "ASE and/or PyTorch are not installed; cannot evaluate MACE committee."
            ) from err

        if self.model_paths:
            refs = self.model_paths
        else:
            refs = [self.config.model_name]

        calculators = []
        for ref in refs:
            calculators.append(self._build_calculator(ref))
        self._calculators = calculators
        self._loaded = True
        logger.info(f"Loaded MACE committee of {len(calculators)} model(s).")
        return len(calculators)

    def evaluate(
        self,
        positions: np.ndarray,
        atomic_numbers: Sequence[int],
        ml_atoms: Sequence[int] | None = None,
        cell: np.ndarray | None = None,
        pbc: bool = False,
        shell_radius_ang: float | None = None,
    ) -> tuple[list[np.ndarray], list[float]]:
        """Evaluate the committee.

        Parameters
        ----------
        positions:
            (N, 3) coordinates in Ångström for the full system.
        atomic_numbers:
            Element atomic numbers for the full system.
        ml_atoms:
            Indices (into ``positions``) of the ML region.  If omitted, all atoms
            are treated as the ML region.
        cell:
            Optional (3, 3) periodic box vectors in Å.
        pbc:
            Whether periodic boundary conditions apply.
        shell_radius_ang:
            Environment shell radius (Å); defaults to ``DEFAULT_SHELL_RADIUS_ANG``.

        Returns
        -------
        (forces_list, energies_list):
            ``forces_list`` is a list of arrays of shape (n_ml, 3) in eV/Å, one
            per committee member; ``energies_list`` is the per-member energy in eV
            for the evaluated sub-system.
        """
        self.ensure_loaded()

        positions = np.asarray(positions, dtype=np.float64)
        atomic_numbers = list(atomic_numbers)
        if ml_atoms is None:
            ml_atoms = list(range(len(positions)))

        radius = self.shell_radius_ang if shell_radius_ang is None else shell_radius_ang
        local_indices, local_ml_indices = _select_ml_shell(positions, ml_atoms, radius)

        try:
            from ase import Atoms
        except ImportError as err:
            raise MACEEnsembleUnavailable(
                "ASE is not installed; cannot evaluate MACE committee."
            ) from err

        sub_positions = positions[local_indices]
        sub_numbers = [atomic_numbers[int(i)] for i in local_indices]
        cell_ase = cell if cell is not None else np.identity(3) * 0.0
        atoms = Atoms(
            numbers=sub_numbers,
            positions=sub_positions,
            cell=cell_ase,
            pbc=[pbc, pbc, pbc],
        )

        forces_list: list[np.ndarray] = []
        energies_list: list[float] = []
        for calc in self._calculators:
            atoms.calc = calc
            forces = np.asarray(calc.get_forces(atoms), dtype=np.float64)  # eV/Å
            energy = float(calc.get_potential_energy(atoms))  # eV
            forces_list.append(forces[local_ml_indices])
            energies_list.append(energy)

        return forces_list, energies_list
