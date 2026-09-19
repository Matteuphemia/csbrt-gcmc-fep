"""Committee inference for MACE: the measurement behind the uncertainty.

``MACECalculator`` already evaluates a list of models in one batched forward
pass and publishes the per-model predictions as ``forces_comm`` (M, N, 3) and
``energy_comm`` (M,), both in MACE's native eV / eV-A. That is the wheel; this
module does not rebuild it. What it adds is the part specific to running inside
an MD loop:

* the ML region is pulled out of a periodic OpenMM Context, so it has to be
  made whole across the periodic boundary before it looks like a molecule;
* the evaluation is over the ML subset alone, which is exactly the set the
  mixed-system ML force sees under mechanical embedding, so the uncertainty
  measured here is the uncertainty of the force actually being applied;
* the models are loaded once and pinned to the device, because the point of the
  surrogate is throughput.

A committee needs at least two independently trained models. The MACE-OFF
foundation releases ship one model per size, so a real committee comes from
active learning: ``active_learner.MACEFineTuner`` writes
``mace_finetuned_gen{k}_seed{s}.model`` files that belong here.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger("csbrt.mace_surrogate.committee")


class CommitteeUnavailable(RuntimeError):
    """The committee could not be loaded (missing mace-torch, missing models)."""


@dataclass
class CommitteePrediction:
    """Per-model energies and forces over the ML region, in MACE units."""

    energies_ev: np.ndarray  # (M,)
    forces_ev_per_ang: np.ndarray  # (M, N, 3)
    wall_seconds: float = 0.0

    @property
    def size(self) -> int:
        return int(self.energies_ev.shape[0])

    @property
    def mean_forces(self) -> np.ndarray:
        return self.forces_ev_per_ang.mean(axis=0)


def make_whole(
    coords_ang: np.ndarray, box_ang: np.ndarray | None
) -> np.ndarray:
    """Undo periodic wrapping within one connected region.

    OpenMM stores positions wrapped into the central box, so an ML region that
    straddles a face comes out as two halves a box-length apart. Feeding that to
    MACE produces a stretched molecule and a spurious uncertainty spike. Each
    atom is shifted to the periodic image nearest the region's first atom.

    ``box_ang`` is the (3, 3) periodic box in Angstroms, rows being the box
    vectors; ``None`` means non-periodic and the coordinates pass through.
    """
    if box_ang is None:
        return np.asarray(coords_ang, dtype=np.float64)
    coords = np.asarray(coords_ang, dtype=np.float64)
    box = np.asarray(box_ang, dtype=np.float64)
    if coords.shape[0] < 2:
        return coords
    # Triclinic-safe: express the offset from the anchor in fractional
    # coordinates, round to the nearest image, subtract.
    inverse = np.linalg.inv(box.T)
    delta = coords - coords[0]
    fractional = delta @ inverse.T
    shifts = np.round(fractional)
    return coords - shifts @ box


class MACECommittee:
    """A loaded committee of MACE models, evaluated on the ML region."""

    def __init__(
        self,
        model_paths: Sequence[str | Path],
        *,
        device: str = "cpu",
        precision: str = "single",
        periodic: bool = True,
    ) -> None:
        paths = [Path(p) for p in model_paths]
        if len(paths) < 2:
            raise CommitteeUnavailable(
                f"A force-variance committee needs at least two models; got "
                f"{len(paths)}. Set mlff.committee_model_paths, or disable UQ "
                "and rely on the geometry guard."
            )
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise CommitteeUnavailable(
                f"Committee model file(s) not found: {', '.join(missing)}"
            )
        self.model_paths = [str(p) for p in paths]
        self.device = device
        self.default_dtype = "float64" if precision == "double" else "float32"
        self.periodic = periodic
        self._calculator: Any | None = None
        self._atoms: Any | None = None
        self._atomic_numbers: tuple[int, ...] | None = None
        self.evaluations = 0
        self.total_wall_seconds = 0.0

    # --- loading -------------------------------------------------------------

    def _load(self) -> Any:
        if self._calculator is not None:
            return self._calculator
        try:
            from mace.calculators import MACECalculator
        except ImportError as error:  # pragma: no cover - environment dependent
            raise CommitteeUnavailable(
                "mace-torch is not installed; `pip install 'mace-torch>=0.3.10'` "
                "(see csbrt/install.sh)"
            ) from error
        started = time.perf_counter()
        try:
            self._calculator = MACECalculator(
                model_paths=self.model_paths,
                device=self.device,
                default_dtype=self.default_dtype,
            )
        except (ValueError, TypeError) as error:
            # MACECalculator requires every committee member to share the same
            # cutoff radius, so mace-off23-small and -medium cannot be paired.
            # (On mace-torch 0.3.16 that check raises TypeError while trying to
            # format its own error message, which hides the cause entirely.)
            raise CommitteeUnavailable(
                "MACE refused this committee: "
                f"{error!r}. Committee members must be architecturally "
                "compatible -- same cutoff r_max, same element table. Build "
                "the committee from independent fine-tunes of ONE foundation "
                "model (different seeds / data splits), not from different "
                f"foundation model sizes. Models: {', '.join(self.model_paths)}"
            ) from error
        logger.info(
            "Loaded MACE committee of %d model(s) on %s in %.1f s",
            len(self.model_paths),
            self.device,
            time.perf_counter() - started,
        )
        return self._calculator

    def warm_up(self, atomic_numbers: Sequence[int], coords_ang: np.ndarray) -> None:
        """Pay the model-load and kernel-compile cost before the timing loop."""
        self.predict(coords_ang, atomic_numbers, box_ang=None)
        self.evaluations = 0
        self.total_wall_seconds = 0.0

    # --- inference -----------------------------------------------------------

    def predict(
        self,
        coords_ang: np.ndarray,
        atomic_numbers: Sequence[int],
        box_ang: np.ndarray | None = None,
    ) -> CommitteePrediction:
        """Evaluate every committee member on one ML-region configuration.

        ``coords_ang`` is (N, 3) in Angstroms for the ML atoms *only*, in the
        same order as ``atomic_numbers``.
        """
        from ase import Atoms

        calculator = self._load()
        numbers = tuple(int(z) for z in atomic_numbers)
        coords = make_whole(coords_ang, box_ang)
        if coords.shape != (len(numbers), 3):
            raise ValueError(
                f"coords_ang has shape {coords.shape}; expected "
                f"({len(numbers)}, 3) to match atomic_numbers"
            )

        if self._atoms is None or self._atomic_numbers != numbers:
            # The ML region is fixed for the lifetime of the Context, so the
            # Atoms object is built once and only its positions move.
            self._atoms = Atoms(numbers=list(numbers), positions=coords)
            self._atoms.calc = calculator
            self._atomic_numbers = numbers
        else:
            self._atoms.set_positions(coords)
            self._atoms.calc.results.clear()

        started = time.perf_counter()
        self._atoms.get_potential_energy()
        elapsed = time.perf_counter() - started
        self.evaluations += 1
        self.total_wall_seconds += elapsed

        results = calculator.results
        if "forces_comm" in results and "energy_comm" in results:
            forces = np.asarray(results["forces_comm"], dtype=np.float64)
            energies = np.asarray(results["energy_comm"], dtype=np.float64).reshape(-1)
        else:  # pragma: no cover - only reachable with a one-model calculator
            forces = np.asarray(results["forces"], dtype=np.float64)[None, ...]
            energies = np.asarray([float(results["energy"])], dtype=np.float64)
        return CommitteePrediction(
            energies_ev=energies,
            forces_ev_per_ang=forces,
            wall_seconds=elapsed,
        )

    # --- reporting -----------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self.model_paths)

    def describe_models(self) -> list[dict[str, Any]]:
        """Load each member just far enough to report its cutoff and elements.

        Used by the preflight check: mismatched models fail deep inside
        ``MACECalculator`` with an unhelpful error, so name the mismatch here.
        """
        import torch

        described: list[dict[str, Any]] = []
        for path in self.model_paths:
            model = torch.load(path, map_location="cpu", weights_only=False)
            r_max = getattr(model, "r_max", None)
            table = getattr(model, "atomic_numbers", None)
            described.append(
                {
                    "path": path,
                    "r_max": float(r_max) if r_max is not None else None,
                    "atomic_numbers": (
                        [int(z) for z in table] if table is not None else None
                    ),
                }
            )
            del model
        return described

    def check_compatibility(self) -> dict[str, Any]:
        """Verify the members can actually form a committee."""
        described = self.describe_models()
        cutoffs = {entry["r_max"] for entry in described}
        tables = {
            tuple(entry["atomic_numbers"] or ()) for entry in described
        }
        compatible = len(cutoffs) == 1 and len(tables) == 1
        return {
            "compatible": compatible,
            "r_max_values": sorted(c for c in cutoffs if c is not None),
            "element_tables": len(tables),
            "models": described,
        }

    def statistics(self) -> dict[str, Any]:
        mean = (
            self.total_wall_seconds / self.evaluations if self.evaluations else 0.0
        )
        return {
            "committee_size": self.size,
            "model_paths": list(self.model_paths),
            "device": self.device,
            "dtype": self.default_dtype,
            "evaluations": self.evaluations,
            "total_wall_seconds": self.total_wall_seconds,
            "mean_seconds_per_evaluation": mean,
        }
