"""Hybrid ML/MM potential integration using MACE and OpenMM-ML.

Grounding:
- Duignan (2024): The Potential of Neural Network Potentials (ACS Phys. Chem. Au)
- Wang, Eastman, Tuckerman et al. (2024): On the Design Space Between Molecular Mechanics
  and Machine Learning Force Fields (arXiv:2409.02861)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

logger = logging.getLogger("csbrt.mace_surrogate")

# Unit conversions (sources of truth, do not hand-tune)
# Force: 1 eV/Å = 9648.533 kJ/(mol·nm) = 23.06054887 kcal/(mol·Å)
EV_PER_ANGSTROM_TO_KJ_PER_MOL_NM = 9648.53321233
EV_PER_ANGSTROM_TO_KCAL_PER_MOL_ANG = 23.06054887
# Energy: 1 eV = 96.485332 kJ/mol = 23.06054887 kcal/mol
EV_TO_KJ_PER_MOL = 96.48533212
EV_TO_KCAL_PER_MOL = 23.06054887
KCAL_TO_KJ = 4.184
ANGSTROM_TO_NM = 0.1
NM_TO_ANGSTROM = 10.0

# Exact MACE model names registered by openmm-ml (verified against
# openmmml/models/macepotential.py).  "mace-omol-0" is NOT a valid name:
# the registered foundation model is "mace-omol-0-extra-large".
KNOWN_FOUNDATION_MODELS = {
    "mace-off23-small",
    "mace-off23-medium",
    "mace-off23-large",
    "mace-off24-medium",
    "mace-mpa-0-medium",
    "mace-omat-0-small",
    "mace-omat-0-medium",
    "mace-omol-0-extra-large",
    "mace-les-off-small",
    "mace-polar-1-small",
    "mace-polar-1-medium",
    "mace-polar-1-large",
}

DEFAULT_FOUNDATION_MODELS = (
    "mace-off23-small",
    "mace-off23-medium",
    "mace-omol-0-extra-large",
)

# openmm-ml accepts 'single'/'double' precision keywords, not 'float32'.
_OPENMMML_PRECISION = {
    "float32": "single",
    "float64": "double",
    "single": "single",
    "double": "double",
}

_WATER_RESNAMES = {"HOH", "WAT", "TIP3", "TIP3P", "H2O", "SOL", "SPC", "SPCE"}


@dataclass
class MACEConfig:
    """Configuration for MACE surrogate potential and active learning loop."""

    enabled: bool = True
    model_name: str = "mace-off23-small"
    model_path: str | None = None
    device: str = "cuda"
    precision: str = "float32"
    embedding: str = "mechanical"  # 'mechanical' or 'electrostatic'
    uq_force_threshold_ev_per_ang: float = 0.05
    uq_energy_threshold_kcal_per_mol: float = 1.0
    include_binding_site_waters: bool = True
    binding_site_radius_nm: float = 1.0  # 10 Angstroms
    ligand_resname: str = "LIG"
    fallback_recovery_steps: int = 10
    ood_buffer_path: str | None = None
    active_learning: bool = True
    committee_size: int = 4
    custom_params: dict[str, Any] = field(default_factory=dict)

    # Aliases accepted by from_dict for CLI/backwards compatibility.
    _FIELD_ALIASES = {
        "model": "model_name",
        "uq_threshold": "uq_force_threshold_ev_per_ang",
        "uq_force_threshold": "uq_force_threshold_ev_per_ang",
        "energy_threshold": "uq_energy_threshold_kcal_per_mol",
        "fallback_steps": "fallback_recovery_steps",
    }

    @property
    def uq_force_threshold_kj_per_mol_nm(self) -> float:
        return self.uq_force_threshold_ev_per_ang * EV_PER_ANGSTROM_TO_KJ_PER_MOL_NM

    @property
    def uq_force_threshold_kcal_per_mol_ang(self) -> float:
        return self.uq_force_threshold_ev_per_ang * EV_PER_ANGSTROM_TO_KCAL_PER_MOL_ANG

    @property
    def uq_energy_threshold_kj_per_mol(self) -> float:
        return self.uq_energy_threshold_kcal_per_mol * KCAL_TO_KJ

    @property
    def openmmml_precision(self) -> str | None:
        """openmm-ml precision keyword ('single'/'double') or None for model default."""
        return _OPENMMML_PRECISION.get(self.precision)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> MACEConfig:
        if not data:
            return cls()
        fields = {f for f in cls.__dataclass_fields__ if not f.startswith("_")}
        normalized: dict[str, Any] = {}
        for key, value in data.items():
            canonical = cls._FIELD_ALIASES.get(key, key)
            if canonical in fields:
                normalized[canonical] = value
        return cls(**normalized)

    def validate(self) -> None:
        """Raise a clear error early on unsupported configuration."""
        if self.model_path:
            if not Path(self.model_path).is_file():
                raise FileNotFoundError(f"MACE model checkpoint not found: {self.model_path}")
        elif self.model_name not in KNOWN_FOUNDATION_MODELS:
            raise ValueError(
                f"Unknown MACE foundation model {self.model_name!r}. "
                f"Known: {sorted(KNOWN_FOUNDATION_MODELS)}. "
                "For a local checkpoint set model_name='mace' and model_path=<file>."
            )
        if self.embedding not in ("mechanical", "electrostatic"):
            raise ValueError(f"Unsupported embedding {self.embedding!r}")
        if self.uq_force_threshold_ev_per_ang < 0 or self.uq_energy_threshold_kcal_per_mol < 0:
            raise ValueError("UQ thresholds must be non-negative")
        if self.committee_size < 1:
            raise ValueError("committee_size must be >= 1")
        if self.fallback_recovery_steps < 1:
            raise ValueError("fallback_recovery_steps must be >= 1")

    def compute_signature(self) -> dict[str, Any]:
        """Generate deterministic signature for checkpointing and reproducibility."""
        model_hash = "built-in"
        if self.model_path and Path(self.model_path).is_file():
            h = hashlib.sha256()
            h.update(Path(self.model_path).read_bytes())
            model_hash = h.hexdigest()
        return {
            "enabled": self.enabled,
            "model_name": self.model_name,
            "model_hash": model_hash,
            "device": self.device,
            "precision": self.precision,
            "embedding": self.embedding,
            "uq_force_threshold_ev_per_ang": self.uq_force_threshold_ev_per_ang,
            "uq_energy_threshold_kcal_per_mol": self.uq_energy_threshold_kcal_per_mol,
            "include_binding_site_waters": self.include_binding_site_waters,
            "binding_site_radius_nm": self.binding_site_radius_nm,
            "ligand_resname": self.ligand_resname,
            "committee_size": self.committee_size,
        }


def _extract_atom_coords(positions: Any, index: int) -> tuple[float, float, float]:
    """Helper to extract (x, y, z) in nanometers from OpenMM or numpy positions."""
    pos = positions[index]
    if hasattr(pos, "value_in_unit"):
        try:
            import openmm.unit as unit
            val = pos.value_in_unit(unit.nanometer)
            return (float(val[0]), float(val[1]), float(val[2]))
        except Exception:
            pass
    if hasattr(pos, "x") and hasattr(pos, "y") and hasattr(pos, "z"):
        return (float(pos.x), float(pos.y), float(pos.z))
    return (float(pos[0]), float(pos[1]), float(pos[2]))


def partition_ml_atoms(
    topology: Any,
    positions: Any = None,
    *,
    ligand_resname: str = "LIG",
    include_waters: bool = False,
    sphere_center: tuple[float, float, float] | None = None,
    sphere_radius_nm: float = 1.0,
) -> list[int]:
    """Partition system atoms into ML and MM regions.

    Parameters
    ----------
    topology:
        OpenMM Topology, or any object exposing ``.residues()`` / ``.atoms()``.
    positions:
        Optional coordinates (in nm) used to select binding-site waters.
    ligand_resname:
        Residue name identifying the ligand molecule.
    include_waters:
        Whether to include hydration waters within the binding sphere.  Requires
        ``positions`` so oxygen positions can be measured; otherwise the ML
        region is ligand-only.
    sphere_center:
        Optional (x, y, z) in nm for the water sphere; defaults to the ligand centroid.
    sphere_radius_nm:
        Cutoff radius for waters around the center (default: 1.0 nm = 10 Å).

    Returns
    -------
    list[int]:
        Sorted unique 0-based indices of atoms assigned to the ML potential region.
    """
    ml_atoms: set[int] = set()
    ligand_coords: list[tuple[float, float, float]] = []

    def _select_waters(water_residue_atoms: list[list[int]]) -> None:
        if not (include_waters and positions is not None and water_residue_atoms):
            return
        center = sphere_center
        if center is None and ligand_coords:
            center = (
                sum(c[0] for c in ligand_coords) / len(ligand_coords),
                sum(c[1] for c in ligand_coords) / len(ligand_coords),
                sum(c[2] for c in ligand_coords) / len(ligand_coords),
            )
        if center is None:
            return
        r_sq = sphere_radius_nm * sphere_radius_nm
        for atom_indices in water_residue_atoms:
            if not atom_indices:
                continue
            ox_coord = _extract_atom_coords(positions, atom_indices[0])
            dist_sq = (
                (ox_coord[0] - center[0]) ** 2
                + (ox_coord[1] - center[1]) ** 2
                + (ox_coord[2] - center[2]) ** 2
            )
            if dist_sq <= r_sq:
                ml_atoms.update(atom_indices)

    # Preferred path: iterate residues directly (robust regardless of atom order).
    if hasattr(topology, "residues") and callable(topology.residues):
        water_residue_atoms: list[list[int]] = []
        for residue in topology.residues():
            res_name = getattr(residue, "name", "")
            atom_indices = [int(atom.index) for atom in residue.atoms()]
            if res_name == ligand_resname:
                ml_atoms.update(atom_indices)
                if positions is not None:
                    for idx in atom_indices:
                        ligand_coords.append(_extract_atom_coords(positions, idx))
            elif include_waters and res_name in _WATER_RESNAMES:
                if atom_indices:
                    water_residue_atoms.append(atom_indices)
        _select_waters(water_residue_atoms)

    # Fallback: atom-walk over objects exposing .atoms() (OpenMM/MDTraj-style).
    elif hasattr(topology, "atoms") and callable(topology.atoms):
        water_residue_atoms: list[list[int]] = []
        current_res_idx = None
        current_water_atoms: list[int] = []
        for atom in topology.atoms():
            residue = getattr(atom, "residue", None)
            res_name = getattr(residue, "name", "")
            atom_idx = int(atom.index)
            if res_name == ligand_resname:
                ml_atoms.add(atom_idx)
                if positions is not None:
                    ligand_coords.append(_extract_atom_coords(positions, atom_idx))
            elif include_waters and res_name in _WATER_RESNAMES:
                res_idx = getattr(residue, "index", None)
                if res_idx is not None and res_idx != current_res_idx:
                    if current_water_atoms:
                        water_residue_atoms.append(current_water_atoms)
                    current_water_atoms = [atom_idx]
                    current_res_idx = res_idx
                else:
                    current_water_atoms.append(atom_idx)
        if current_water_atoms:
            water_residue_atoms.append(current_water_atoms)
        _select_waters(water_residue_atoms)

    # Fallback: object supporting selection syntax (e.g. Sire System).
    elif hasattr(topology, "select") or hasattr(topology, "__getitem__"):
        try:
            ligand_sel = topology[f"resname {ligand_resname}"]
            for atom in ligand_sel.atoms():
                idx = atom.index()
                ml_atoms.add(int(idx.value() if hasattr(idx, "value") else idx))
        except Exception:
            pass

    return sorted(ml_atoms)


class MACEMixedSystemBuilder:
    """Builder for hybrid ML/MM OpenMM systems with MACE surrogate potentials.

    Uses the OpenMM-ML ``createMixedSystem(..., interpolate=True)`` contract so
    the returned system can toggle between classical MM and ML/MM via the global
    parameter ``lambda_interpolate`` (see :class:`MACEConfig` module docstring).
    """

    def __init__(self, config: MACEConfig | None = None) -> None:
        self.config = config or MACEConfig()
        self.last_mixed_system: Any = None
        self.last_ml_atoms: list[int] = []

    def build_mixed_system(
        self,
        system: Any,
        topology: Any,
        ml_atoms: Sequence[int] | None = None,
        positions: Any = None,
        interpolate: bool = True,
    ) -> Any:
        """Create a mixed ML/MM OpenMM ``System``.

        When ``interpolate`` is True the returned system carries a global
        parameter ``lambda_interpolate`` (0.0 = classical, 1.0 = ML/MM) enabling
        zero-overhead physics fallback.  If OpenMM-ML is unavailable the
        classical ``system`` is returned unchanged so callers degrade gracefully.
        """
        if ml_atoms is None:
            ml_atoms = partition_ml_atoms(
                topology,
                positions=positions,
                ligand_resname=self.config.ligand_resname,
                include_waters=self.config.include_binding_site_waters,
                sphere_radius_nm=self.config.binding_site_radius_nm,
            )
        self.last_ml_atoms = list(ml_atoms)

        if not ml_atoms:
            logger.warning(
                "No ML atoms identified for MACE surrogate; returning classical system."
            )
            return system

        logger.info(
            f"Building MACE ML/MM mixed system: {len(ml_atoms)} ML atoms, "
            f"model={self.config.model_name}, embedding={self.config.embedding}, "
            f"interpolate={interpolate}"
        )

        try:
            import openmmml
        except ImportError:
            logger.warning(
                "openmmml not installed; MACE mixed system unavailable. "
                "Returning classical system (no fast path)."
            )
            return system

        # modelPath is a constructor argument of MLPotential; device/precision
        # are forwarded through createMixedSystem(**args) to the implementation.
        ctor_kwargs: dict[str, Any] = {}
        if self.config.model_path:
            model_name = "mace"
            ctor_kwargs["modelPath"] = str(self.config.model_path)
        else:
            model_name = self.config.model_name

        try:
            potential = openmmml.MLPotential(model_name, **ctor_kwargs)
        except Exception as err:
            logger.warning(f"Could not create MLPotential({model_name!r}): {err}")
            return system

        create_kwargs: dict[str, Any] = {
            "interpolate": interpolate,
            "device": self.config.device,
        }
        precision = self.config.openmmml_precision
        if precision is not None:
            create_kwargs["precision"] = precision

        # Validate the requested embedding against what the potential supports.
        try:
            supported = set(potential.getSupportedEmbeddings())
        except Exception:
            supported = {"mechanical"}
        embedding = self.config.embedding
        if embedding not in supported:
            logger.warning(
                f"Embedding {embedding!r} not supported (supported={sorted(supported)}); "
                "falling back to 'mechanical'."
            )
            embedding = "mechanical"
        create_kwargs["embedding"] = embedding

        try:
            mixed_system = potential.createMixedSystem(
                topology, system, list(ml_atoms), **create_kwargs
            )
        except Exception as err:
            logger.warning(
                f"OpenMM-ML createMixedSystem encountered: {err}; "
                "returning classical system (no fast path)."
            )
            return system

        self.last_mixed_system = mixed_system
        return mixed_system


def create_mace_mixed_system(
    system: Any,
    topology: Any,
    ml_atoms: Sequence[int] | None = None,
    config: MACEConfig | None = None,
    positions: Any = None,
    interpolate: bool = True,
) -> Any:
    """Convenience function to generate a MACE mixed system."""
    builder = MACEMixedSystemBuilder(config)
    return builder.build_mixed_system(
        system, topology, ml_atoms=ml_atoms, positions=positions, interpolate=interpolate
    )
