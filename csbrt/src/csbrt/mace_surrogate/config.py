"""Configuration object for the MACE hybrid ML/MM surrogate.

One dataclass carries every knob the surrogate needs, is serialisable into the
``.complete.json`` checkpoint markers the rest of ``csbrt`` uses, and hashes to
a stable signature so a MACE-accelerated run can never be mistaken for the
classical baseline that produced the same filenames.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import hashlib
import logging
from pathlib import Path
from typing import Any

from .units import (
    EV_PER_ANGSTROM_TO_KCAL_PER_MOL_ANG,
    EV_PER_ANGSTROM_TO_KJ_PER_MOL_NM,
    EV_TO_KCAL_PER_MOL,
    KCAL_TO_KJ,
)

logger = logging.getLogger("csbrt.mace_surrogate.config")

#: Names openmm-ml registers for pre-trained MACE foundation models. Passed
#: straight to ``MLPotential(name)``; ``'mace'`` means "load modelPath".
#: openmm-ml is the authority here, so an unknown name is a warning rather
#: than an error: a newer openmm-ml may register models this list predates.
KNOWN_FOUNDATION_MODELS = (
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
)

#: Shorthands the brief and the implementation plan use, mapped onto the names
#: openmm-ml actually registers.
MODEL_ALIASES = {
    "mace-off23": "mace-off23-small",
    "mace-off": "mace-off23-small",
    "mace-omol-0": "mace-omol-0-extra-large",
    "mace-omol": "mace-omol-0-extra-large",
    "mace-mp": "mace-mpa-0-medium",
}

#: Models trained on organic molecules only (H, C, N, O, F, P, S, Cl, Br, I).
#: MACE-OFF is the right family for a drug-like ligand in water; the materials
#: models (mace-mp / mace-omat) cover the periodic table but are not fitted to
#: molecular conformational energies.
ORGANIC_MODELS = frozenset(
    {
        "mace-off23-small",
        "mace-off23-medium",
        "mace-off23-large",
        "mace-off24-medium",
        "mace-les-off-small",
        "mace-omol-0-extra-large",
    }
)

#: Renamed config keys, so a config written against the first cut of this
#: package still loads.
LEGACY_KEYS = {
    "fallback_recovery_steps": "fallback_steps",
    "uq_force_threshold": "uq_force_threshold_ev_per_ang",
    "ood_buffer_path": "ood_buffer_dir",
}

#: Force group reserved for the ML potential. OpenMM allows 0..31; Loch, Sire
#: and SOMD2 all build their forces from group 0 upwards, so take one from the
#: top to avoid colliding with a barostat or a restraint force added later.
DEFAULT_ML_FORCE_GROUP = 30


@dataclass
class MACEConfig:
    """Every setting the MACE surrogate, its UQ monitor and its fallback need."""

    # --- surrogate potential -------------------------------------------------
    enabled: bool = False
    model_name: str = "mace-off23-small"
    #: Path to a locally trained / fine-tuned ``.model``. When set, the openmm-ml
    #: potential name becomes ``'mace'`` and this file is loaded instead of a
    #: foundation model. Active-learning generations land here.
    model_path: str | None = None
    device: str = "cuda"
    #: openmm-ml accepts 'single' or 'double'. 'single' is the MACE
    #: recommendation for MD; 'double' for geometry optimisation.
    precision: str = "single"
    embedding: str = "mechanical"
    ml_force_group: int = DEFAULT_ML_FORCE_GROUP
    #: Build the mixed system with openmm-ml's ``interpolate=True`` so the
    #: 'lambda_interpolate' global parameter exists and the fallback is a
    #: parameter write rather than a Context rebuild.
    interpolate: bool = True
    remove_constraints: bool = True
    #: 'interaction_energy' (default) subtracts MACE's atomic self-energies,
    #: which is what a mixed ML/MM system wants; 'energy' returns the total.
    return_energy_type: str = "interaction_energy"

    # --- ML/MM partition -----------------------------------------------------
    ligand_resname: str = "LIG"
    #: Loch swaps water identities every GCMC cycle and toggles ghost waters
    #: between interacting and non-interacting in the live Context. The ML atom
    #: list is baked into the ML force at build time and cannot follow that, so
    #: putting waters in the ML region is off by default. See
    #: docs/mlff_active_learning_architecture.md.
    include_binding_site_waters: bool = False
    binding_site_radius_nm: float = 1.0
    #: Guard rail against the pure-MLFF scalability trap (Wang et al. 2024): an
    #: ML region this large is slower than the classical system it replaced.
    max_ml_atoms: int = 250

    # --- uncertainty quantification -----------------------------------------
    uq_enabled: bool = True
    #: Committee members. Two or more ``.model`` files are required for a
    #: force-variance estimate; a single foundation model has no variance to
    #: measure and leaves only the geometry guard.
    committee_model_paths: tuple[str, ...] = ()
    uq_force_threshold_ev_per_ang: float = 0.05
    uq_energy_threshold_kcal_per_mol: float = 1.0
    #: MD steps between UQ evaluations. Every step would cost more than the
    #: surrogate saves; 100 steps at 2 fs is one check per 0.2 ps.
    uq_interval_steps: int = 100
    #: Cheap always-on geometry sanity check. Catches the failure mode a
    #: committee of similarly-trained models can agree on: a distorted or
    #: clashing ligand that is off the training manifold entirely.
    geometry_guard: bool = True
    #: Minimum tolerated distance between two ML atoms.
    geometry_min_distance_ang: float = 0.70
    #: Maximum tolerated bond length as a multiple of the sum of covalent radii.
    geometry_max_bond_scale: float = 1.60

    # --- physics fallback ----------------------------------------------------
    #: MD steps to run on the classical Hamiltonian after a trigger before the
    #: surrogate is re-tested.
    fallback_steps: int = 50
    #: If more than this fraction of steps run classically the surrogate is not
    #: paying for itself; the run latches to classical physics and says so.
    fallback_abort_fraction: float = 0.5
    #: Raise instead of degrading to pure classical physics when the surrogate
    #: cannot be built (missing openmm-ml, missing model, empty ML region).
    strict: bool = False

    # --- active learning -----------------------------------------------------
    active_learning: bool = True
    #: Directory for per-worker OOD buffers. Relative paths resolve against the
    #: stage output directory.
    ood_buffer_dir: str | None = "al_buffer"
    #: Cap on frames held in memory per worker before the oldest low-uncertainty
    #: frames are dropped.
    ood_max_frames: int = 2000
    #: Heavy-atom RMSD below which two OOD frames are the same basin.
    ood_rmsd_cutoff_ang: float = 0.5

    custom_params: dict[str, Any] = field(default_factory=dict)

    # --- derived -------------------------------------------------------------

    def __post_init__(self) -> None:
        self.model_name = MODEL_ALIASES.get(self.model_name, self.model_name)
        if isinstance(self.committee_model_paths, (str, Path)):
            self.committee_model_paths = (str(self.committee_model_paths),)
        else:
            self.committee_model_paths = tuple(
                str(p) for p in (self.committee_model_paths or ())
            )
        if self.precision in ("float32", "single", "mixed"):
            self.precision = "single"
        elif self.precision in ("float64", "double"):
            self.precision = "double"
        else:
            raise ValueError(
                f"precision must be single/double (got {self.precision!r})"
            )
        if self.embedding not in ("mechanical", "electrostatic"):
            raise ValueError(
                f"embedding must be 'mechanical' or 'electrostatic' "
                f"(got {self.embedding!r})"
            )
        if self.return_energy_type not in ("interaction_energy", "energy"):
            raise ValueError(
                "return_energy_type must be 'interaction_energy' or 'energy'"
            )
        if not 0 <= self.ml_force_group <= 31:
            raise ValueError("ml_force_group must be an OpenMM force group, 0..31")
        if self.uq_force_threshold_ev_per_ang <= 0:
            raise ValueError("uq_force_threshold_ev_per_ang must be positive")
        if self.uq_energy_threshold_kcal_per_mol <= 0:
            raise ValueError("uq_energy_threshold_kcal_per_mol must be positive")
        if self.uq_interval_steps < 1:
            raise ValueError("uq_interval_steps must be at least 1")
        if self.fallback_steps < 1:
            raise ValueError("fallback_steps must be at least 1")
        if not 0.0 < self.fallback_abort_fraction <= 1.0:
            raise ValueError("fallback_abort_fraction must be in (0, 1]")
        if self.max_ml_atoms < 1:
            raise ValueError("max_ml_atoms must be positive")
        if self.binding_site_radius_nm <= 0:
            raise ValueError("binding_site_radius_nm must be positive")
        if self.ood_rmsd_cutoff_ang <= 0:
            raise ValueError("ood_rmsd_cutoff_ang must be positive")
        if (
            self.enabled
            and self.model_path is None
            and self.model_name not in KNOWN_FOUNDATION_MODELS
        ):
            logger.warning(
                "MACE model %r is not one of the foundation models this build "
                "knows (%s). Passing it to openmm-ml unchanged; set model_path "
                "instead for a locally trained model.",
                self.model_name,
                ", ".join(KNOWN_FOUNDATION_MODELS),
            )
        if (
            self.enabled
            and self.model_path is None
            and self.model_name not in ORGANIC_MODELS
        ):
            logger.warning(
                "MACE model %r is not fitted to organic molecules; MACE-OFF is "
                "the right family for a drug-like ligand in TIP3P water.",
                self.model_name,
            )

    @property
    def potential_name(self) -> str:
        """The name to hand ``openmmml.MLPotential``."""
        return "mace" if self.model_path else self.model_name

    @property
    def committee_size(self) -> int:
        return len(self.committee_model_paths)

    @property
    def has_committee(self) -> bool:
        """A force-variance estimate needs at least two independent models."""
        return self.committee_size >= 2

    @property
    def uq_force_threshold_kj_per_mol_nm(self) -> float:
        return self.uq_force_threshold_ev_per_ang * EV_PER_ANGSTROM_TO_KJ_PER_MOL_NM

    @property
    def uq_force_threshold_kcal_per_mol_ang(self) -> float:
        return (
            self.uq_force_threshold_ev_per_ang * EV_PER_ANGSTROM_TO_KCAL_PER_MOL_ANG
        )

    @property
    def uq_energy_threshold_ev(self) -> float:
        return self.uq_energy_threshold_kcal_per_mol / EV_TO_KCAL_PER_MOL

    @property
    def uq_energy_threshold_kj_per_mol(self) -> float:
        return self.uq_energy_threshold_kcal_per_mol * KCAL_TO_KJ

    def resolve_paths(self, root: Path) -> "MACEConfig":
        """Return a copy with relative model/buffer paths resolved under ``root``."""
        data = self.to_dict()
        if self.ood_buffer_dir and not Path(self.ood_buffer_dir).is_absolute():
            data["ood_buffer_dir"] = str(root / self.ood_buffer_dir)
        if self.model_path and not Path(self.model_path).is_absolute():
            data["model_path"] = str(root / self.model_path)
        data["committee_model_paths"] = tuple(
            str(p) if Path(p).is_absolute() else str(root / p)
            for p in self.committee_model_paths
        )
        return MACEConfig.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["committee_model_paths"] = list(self.committee_model_paths)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MACEConfig":
        """Build from a config mapping, tolerating unknown and legacy keys."""
        if not data:
            return cls()
        known = {f.name for f in fields(cls)}
        renamed = {LEGACY_KEYS.get(k, k): v for k, v in data.items()}
        unknown = sorted(set(renamed) - known)
        if unknown:
            logger.warning(
                "Ignoring unrecognised mlff config key(s): %s", ", ".join(unknown)
            )
        return cls(**{k: v for k, v in renamed.items() if k in known})

    # --- reproducibility -----------------------------------------------------

    @staticmethod
    def _hash_model(path: str | None) -> str:
        if not path:
            return "none"
        target = Path(path)
        if not target.is_file():
            return "missing"
        digest = hashlib.sha256()
        with target.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def compute_signature(self) -> dict[str, Any]:
        """Deterministic fingerprint for the ``.complete.json`` markers.

        Everything that changes the sampled ensemble goes in. Runtime-only
        settings (device, buffer directory, active-learning bookkeeping) stay
        out, so re-running the same physics on a different GPU still matches
        its checkpoint.
        """
        return {
            "enabled": self.enabled,
            "potential_name": self.potential_name,
            "model_name": self.model_name,
            "model_sha256": self._hash_model(self.model_path),
            "committee_sha256": [
                self._hash_model(p) for p in self.committee_model_paths
            ],
            "precision": self.precision,
            "embedding": self.embedding,
            "interpolate": self.interpolate,
            "remove_constraints": self.remove_constraints,
            "return_energy_type": self.return_energy_type,
            "ml_force_group": self.ml_force_group,
            "ligand_resname": self.ligand_resname,
            "include_binding_site_waters": self.include_binding_site_waters,
            "binding_site_radius_nm": self.binding_site_radius_nm,
            "max_ml_atoms": self.max_ml_atoms,
            "uq_enabled": self.uq_enabled,
            "uq_force_threshold_ev_per_ang": self.uq_force_threshold_ev_per_ang,
            "uq_energy_threshold_kcal_per_mol": self.uq_energy_threshold_kcal_per_mol,
            "uq_interval_steps": self.uq_interval_steps,
            "geometry_guard": self.geometry_guard,
            "geometry_min_distance_ang": self.geometry_min_distance_ang,
            "geometry_max_bond_scale": self.geometry_max_bond_scale,
            "fallback_steps": self.fallback_steps,
            "fallback_abort_fraction": self.fallback_abort_fraction,
        }
