"""Hybrid ML/MM system construction: the surrogate fast path.

A pure MLFF over a 30k-60k atom solvated complex is ~100x *slower* than
GPU-accelerated classical MM (Wang et al. 2024, arXiv:2409.02861), so the
speedup comes from partitioning: 40-250 atoms of ligand go to MACE, the
protein and bulk solvent stay on Amber ff14SB / TIP3P. openmm-ml's
``MLPotential.createMixedSystem`` does the partitioning, the nonbonded
exclusions and the PME bookkeeping; this module decides *which* atoms, applies
the result to a live OpenMM Context without rebuilding it, and records what it
did.

Two entry points matter:

``create_mace_mixed_system``
    Classical System in, mixed ML/MM System out. Use when you own the System
    before the Context exists.

``attach_mace_to_context``
    Mutate the System behind a live Context and ``reinitialize(preserveState=
    True)``. Use for Sire/Loch and SOMD2, which build their own Context. This
    is the same pattern ``ev71_loch_common.add_ca_restraints`` already uses.

Both build with ``interpolate=True``, which gives the System a global parameter
``lambda_interpolate``: 1.0 evaluates the mixed ML/MM Hamiltonian, 0.0 the
original classical one. That parameter is the zero-overhead fallback switch --
see ``fallback_controller``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import logging
from typing import Any, Iterable, Sequence

import numpy as np

from .config import (  # re-exported for callers that import from here
    DEFAULT_ML_FORCE_GROUP,
    KNOWN_FOUNDATION_MODELS,
    MACEConfig,
)
from .units import (
    ANGSTROM_TO_NM,
    EV_PER_ANGSTROM_TO_KCAL_PER_MOL_ANG,
    EV_PER_ANGSTROM_TO_KJ_PER_MOL_NM,
    KCAL_TO_KJ,
    NM_TO_ANGSTROM,
)

logger = logging.getLogger("csbrt.mace_surrogate.mixed_system")

#: Kept for backwards compatibility with the first cut of this package, which
#: exported the model tuple under this name.
DEFAULT_FOUNDATION_MODELS = KNOWN_FOUNDATION_MODELS

#: Residue names Amber/OpenMM use for water across the tools in this pipeline.
WATER_RESNAMES = frozenset({"HOH", "WAT", "TIP3", "TIP3P", "T3P", "SOL", "H2O"})

#: The global parameter openmm-ml creates when ``interpolate=True``.
INTERPOLATION_PARAMETER = "lambda_interpolate"

#: Elements MACE-OFF covers. An ML region containing anything else is outside
#: the model's training set no matter what the committee says.
MACE_OFF_ELEMENTS = frozenset({1, 6, 7, 8, 9, 15, 16, 17, 35, 53})


class MACESurrogateError(RuntimeError):
    """The surrogate could not be built or attached."""


@dataclass
class MLRegion:
    """The atoms handed to the ML potential, and why."""

    atom_indices: list[int]
    atomic_numbers: list[int]
    residues: dict[str, int] = field(default_factory=dict)
    ligand_atom_count: int = 0
    water_atom_count: int = 0

    def __len__(self) -> int:
        return len(self.atom_indices)

    @property
    def unsupported_elements(self) -> list[int]:
        return sorted(set(self.atomic_numbers) - MACE_OFF_ELEMENTS)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ml_atom_count": len(self.atom_indices),
            "ligand_atom_count": self.ligand_atom_count,
            "water_atom_count": self.water_atom_count,
            "residues": dict(self.residues),
            "unsupported_elements": self.unsupported_elements,
        }


@dataclass
class MixedSystemHandle:
    """What ``attach_mace_to_context`` did, for logging and checkpointing."""

    region: MLRegion
    parameter_name: str
    surrogate_value: float
    classical_value: float
    force_group: int
    potential_name: str
    interpolating: bool
    backend: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "potential_name": self.potential_name,
            "parameter_name": self.parameter_name,
            "interpolating": self.interpolating,
            "force_group": self.force_group,
            **self.region.to_dict(),
        }


# --------------------------------------------------------------------------- positions


def positions_in_nm(positions: Any) -> np.ndarray:
    """Return an ``(N, 3)`` array in nanometres from whatever OpenMM handed us.

    Accepts an OpenMM ``Quantity`` (array or list of ``Vec3``), a plain numpy
    array already in nm, or a sequence of triples.
    """
    if positions is None:
        raise ValueError("positions are required")
    if hasattr(positions, "value_in_unit"):
        import openmm.unit as unit

        return np.asarray(positions.value_in_unit(unit.nanometer), dtype=np.float64)
    array = np.asarray(
        [[float(p[0]), float(p[1]), float(p[2])] for p in positions],
        dtype=np.float64,
    )
    return array


# --------------------------------------------------------------------------- partition


def _iter_residues(topology: Any):
    if hasattr(topology, "residues") and callable(topology.residues):
        return topology.residues()
    raise TypeError(
        f"{type(topology).__name__} is not an OpenMM Topology; "
        "pass app.AmberPrmtopFile(...).topology or an equivalent"
    )


def noninteracting_particles(system: Any, tolerance: float = 1.0e-8) -> set[int]:
    """Particles with zero charge *and* zero LJ epsilon in every NonbondedForce.

    Loch represents its ghost-water buffer this way: the molecules are present
    in the topology but switched off in the live Context, and Loch toggles them
    as GCMC accepts insertions and deletions. Such a particle must never enter
    the ML region -- MACE would compute a real interaction energy for a water
    the MM side believes does not exist.
    """
    import openmm

    ghosts: set[int] | None = None
    for force in system.getForces():
        if not isinstance(force, openmm.NonbondedForce):
            continue
        current = set()
        for index in range(force.getNumParticles()):
            charge, _sigma, epsilon = force.getParticleParameters(index)
            if (
                abs(charge.value_in_unit(charge.unit)) < tolerance
                and abs(epsilon.value_in_unit(epsilon.unit)) < tolerance
            ):
                current.add(index)
        ghosts = current if ghosts is None else (ghosts & current)
    return ghosts or set()


def partition_ml_atoms(
    topology: Any,
    positions: Any = None,
    *,
    ligand_resname: str = "LIG",
    include_waters: bool = False,
    sphere_center: Sequence[float] | None = None,
    sphere_radius_nm: float = 1.0,
    exclude: Iterable[int] = (),
) -> list[int]:
    """Choose the atoms the ML potential will own.

    The ligand residue always goes in, whole. Waters are optional and are
    selected whole-residue by the distance of their oxygen from ``sphere_center``
    (the ligand centroid when not given), which requires ``positions``.

    Returns a sorted list of 0-based OpenMM particle indices. Raises if the
    ligand residue is absent or appears more than once, because a silently
    empty ML region turns the surrogate into a no-op that still claims to be
    running.
    """
    excluded = set(int(i) for i in exclude)
    ligand_atoms: list[int] = []
    ligand_residues = 0
    water_residues: list[list[int]] = []

    for residue in _iter_residues(topology):
        name = str(getattr(residue, "name", "")).strip().upper()
        indices = [int(atom.index) for atom in residue.atoms()]
        if name == ligand_resname.strip().upper():
            ligand_residues += 1
            ligand_atoms.extend(indices)
        elif include_waters and name in WATER_RESNAMES:
            water_residues.append(indices)

    if ligand_residues == 0:
        raise MACESurrogateError(
            f"No residue named {ligand_resname!r} in the topology; the ML region "
            "would be empty. Check --ligand-resname."
        )
    if ligand_residues > 1:
        raise MACESurrogateError(
            f"Found {ligand_residues} residues named {ligand_resname!r}. The ML "
            "region must be one molecule; the GCMC sphere reference already "
            "requires a single ligand."
        )
    if excluded & set(ligand_atoms):
        raise MACESurrogateError(
            "Ligand atoms are in the excluded set (non-interacting in the live "
            "Context). Refusing to build an ML region around a switched-off "
            "molecule."
        )

    selected = set(ligand_atoms)

    if include_waters and water_residues:
        if positions is None:
            raise MACESurrogateError(
                "include_binding_site_waters needs positions to select the "
                "sphere; pass the Context's positions."
            )
        coords = positions_in_nm(positions)
        if sphere_center is None:
            center = coords[ligand_atoms].mean(axis=0)
        else:
            center = np.asarray(sphere_center, dtype=np.float64)
        radius_sq = float(sphere_radius_nm) ** 2
        for indices in water_residues:
            if excluded & set(indices):
                # A Loch ghost. Skip it rather than failing the run: the
                # buffer waters are meant to come and go.
                continue
            oxygen = indices[0]
            delta = coords[oxygen] - center
            if float(delta @ delta) <= radius_sq:
                selected.update(indices)

    return sorted(selected)


def describe_ml_region(topology: Any, ml_atoms: Sequence[int]) -> MLRegion:
    """Summarise an ML atom list: elements, residue composition, counts."""
    wanted = set(int(i) for i in ml_atoms)
    atomic_numbers: dict[int, int] = {}
    residues: dict[str, int] = {}
    ligand_atoms = 0
    water_atoms = 0
    for residue in _iter_residues(topology):
        name = str(getattr(residue, "name", "")).strip().upper()
        hits = 0
        for atom in residue.atoms():
            index = int(atom.index)
            if index not in wanted:
                continue
            hits += 1
            element = getattr(atom, "element", None)
            if element is None or getattr(element, "atomic_number", None) is None:
                raise MACESurrogateError(
                    f"Atom {index} has no element; MACE needs atomic numbers. "
                    "The topology is probably missing element assignments."
                )
            atomic_numbers[index] = int(element.atomic_number)
        if hits:
            residues[name] = residues.get(name, 0) + 1
            if name in WATER_RESNAMES:
                water_atoms += hits
            else:
                ligand_atoms += hits
    missing = wanted - set(atomic_numbers)
    if missing:
        raise MACESurrogateError(
            f"{len(missing)} ML atom index/indices are not in the topology: "
            f"{sorted(missing)[:8]}"
        )
    ordered = sorted(wanted)
    return MLRegion(
        atom_indices=ordered,
        atomic_numbers=[atomic_numbers[i] for i in ordered],
        residues=residues,
        ligand_atom_count=ligand_atoms,
        water_atom_count=water_atoms,
    )


def validate_ml_region(region: MLRegion, config: MACEConfig) -> None:
    """Reject an ML region that would be wrong or slower than plain MM."""
    if not region.atom_indices:
        raise MACESurrogateError("ML region is empty")
    if len(region) > config.max_ml_atoms:
        raise MACESurrogateError(
            f"ML region has {len(region)} atoms, above max_ml_atoms="
            f"{config.max_ml_atoms}. Beyond a few hundred atoms the MLFF costs "
            "more than the classical forces it replaces (Wang et al. 2024); "
            "shrink the region or raise the cap deliberately."
        )
    unsupported = region.unsupported_elements
    if unsupported and config.model_name in (
        "mace-off23-small",
        "mace-off23-medium",
        "mace-off23-large",
        "mace-off24-medium",
    ):
        raise MACESurrogateError(
            f"ML region contains element(s) {unsupported} outside MACE-OFF's "
            "training set (H C N O F P S Cl Br I). Choose a model that covers "
            "them or exclude those atoms."
        )


# --------------------------------------------------------------------------- build


def build_ml_potential(config: MACEConfig):
    """Instantiate the openmm-ml ``MLPotential`` this config asks for."""
    try:
        from openmmml import MLPotential
    except ImportError as error:  # pragma: no cover - environment dependent
        raise MACESurrogateError(
            "openmm-ml is not installed; `mamba install -c conda-forge openmm-ml` "
            "(see csbrt/environment.yml)"
        ) from error

    if config.model_path:
        return MLPotential("mace", modelPath=config.model_path)
    return MLPotential(config.model_name)


def _mace_force_kwargs(config: MACEConfig) -> dict[str, Any]:
    """Keyword arguments openmm-ml passes through to the MACE implementation."""
    return {
        "precision": config.precision,
        "returnEnergyType": config.return_energy_type,
        "device": config.device,
    }


def create_mace_mixed_system(
    system: Any,
    topology: Any,
    ml_atoms: Sequence[int] | None = None,
    config: MACEConfig | None = None,
    positions: Any = None,
) -> Any:
    """Return a new mixed ML/MM ``openmm.System``.

    The input System is not modified. When ``config.interpolate`` is set the
    returned System carries the ``lambda_interpolate`` global parameter.
    """
    config = config or MACEConfig(enabled=True)
    if ml_atoms is None:
        ml_atoms = partition_ml_atoms(
            topology,
            positions=positions,
            ligand_resname=config.ligand_resname,
            include_waters=config.include_binding_site_waters,
            sphere_radius_nm=config.binding_site_radius_nm,
            exclude=noninteracting_particles(system),
        )
    region = describe_ml_region(topology, ml_atoms)
    validate_ml_region(region, config)

    potential = build_ml_potential(config)
    logger.info(
        "Building MACE mixed system: %d ML atoms (%d ligand + %d water), "
        "potential=%s embedding=%s interpolate=%s force_group=%d",
        len(region),
        region.ligand_atom_count,
        region.water_atom_count,
        config.potential_name,
        config.embedding,
        config.interpolate,
        config.ml_force_group,
    )
    return potential.createMixedSystem(
        topology,
        system,
        region.atom_indices,
        removeConstraints=config.remove_constraints,
        forceGroup=config.ml_force_group,
        interpolate=config.interpolate,
        embedding=config.embedding,
        **_mace_force_kwargs(config),
    )


def apply_system_in_place(live: Any, replacement: Any) -> None:
    """Make ``live`` carry ``replacement``'s forces and constraints.

    OpenMM gives no way to swap the System behind a Context, but a Context does
    pick up changes to its own System after ``reinitialize(preserveState=True)``.
    So the mixed System's forces are deep-copied across (an OpenMM Force belongs
    to exactly one System; ``copy.deepcopy`` is how openmm-ml itself moves them)
    and the constraint list is resynchronised, because ``createMixedSystem``
    drops the constraints inside the ML region.

    Particle masses, virtual sites and the default box are left alone: the
    mechanical embedding does not change them, and this asserts as much.
    """
    if live.getNumParticles() != replacement.getNumParticles():
        raise MACESurrogateError(
            f"Mixed system has {replacement.getNumParticles()} particles but the "
            f"live system has {live.getNumParticles()}; the embedding added or "
            "removed particles and cannot be applied to a live Context."
        )
    for index in reversed(range(live.getNumForces())):
        live.removeForce(index)
    for force in replacement.getForces():
        live.addForce(copy.deepcopy(force))
    for index in reversed(range(live.getNumConstraints())):
        live.removeConstraint(index)
    for index in range(replacement.getNumConstraints()):
        particle1, particle2, distance = replacement.getConstraintParameters(index)
        live.addConstraint(particle1, particle2, distance)


def attach_mace_to_context(
    context: Any,
    topology: Any,
    config: MACEConfig,
    ml_atoms: Sequence[int] | None = None,
) -> MixedSystemHandle:
    """Turn a live classical Context into a dual-Hamiltonian ML/MM Context.

    The Context keeps its positions, velocities and box vectors. Afterwards
    ``context.setParameter('lambda_interpolate', 1.0)`` runs the surrogate and
    ``0.0`` runs the original classical Hamiltonian, with no rebuild in between.
    """
    live = context.getSystem()
    state = context.getState(getPositions=True)
    if ml_atoms is None:
        ml_atoms = partition_ml_atoms(
            topology,
            positions=state.getPositions(asNumpy=True),
            ligand_resname=config.ligand_resname,
            include_waters=config.include_binding_site_waters,
            sphere_radius_nm=config.binding_site_radius_nm,
            exclude=noninteracting_particles(live),
        )
    region = describe_ml_region(topology, ml_atoms)
    validate_ml_region(region, config)

    mixed = create_mace_mixed_system(
        live, topology, region.atom_indices, config=config
    )
    apply_system_in_place(live, mixed)
    context.reinitialize(preserveState=True)

    handle = MixedSystemHandle(
        region=region,
        parameter_name=INTERPOLATION_PARAMETER,
        surrogate_value=1.0,
        classical_value=0.0,
        force_group=config.ml_force_group,
        potential_name=config.potential_name,
        interpolating=bool(config.interpolate),
        backend="openmm-ml",
    )
    if config.interpolate:
        _require_parameter(context, INTERPOLATION_PARAMETER)
        context.setParameter(INTERPOLATION_PARAMETER, handle.surrogate_value)
    else:
        logger.warning(
            "MACE mixed system built with interpolate=False: there is no "
            "lambda_interpolate parameter, so the physics fallback would need a "
            "Context rebuild (200-500 ms each). Leave interpolate on."
        )
    logger.info(
        "MACE surrogate attached to live Context: %s", handle.to_dict()
    )
    return handle


def _require_parameter(context: Any, name: str) -> None:
    parameters = context.getParameters()
    if name not in parameters:
        raise MACESurrogateError(
            f"Mixed system has no global parameter {name!r}; openmm-ml was asked "
            "for an interpolating system but did not produce one. Check the "
            "openmm-ml version (>= 1.1 supports interpolate=True)."
        )
