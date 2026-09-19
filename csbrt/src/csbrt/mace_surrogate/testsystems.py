"""A small self-contained OpenMM system for preflight checks and benchmarks.

Ethanol in a periodic box with a few waters, one of them switched off the way
Loch represents a ghost, plus a short peptide-like residue. Hand-built rather
than loaded from a force field so it needs no input files and no Amber
installation: the preflight check has to run on a login node, and a benchmark
has to be reproducible.

It is a plumbing fixture, not a scientific system. Nothing here is fitted to
anything; it exists so the surrogate's construction, switching and inference
paths can be exercised end to end in a second.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

BOX_NM = 2.5

#: Ethanol, Angstroms. Every bond and contact is physically sane, so the
#: geometry guard passes on this pose and only fires on distorted ones.
LIGAND_ATOMS = (
    ("C1", "C", (0.00, 0.00, 0.00)),
    ("C2", "C", (1.52, 0.00, 0.00)),
    ("O1", "O", (2.10, 1.20, 0.00)),
    ("H1", "H", (-0.36, 1.02, 0.00)),
    ("H2", "H", (-0.36, -0.51, 0.88)),
    ("H3", "H", (-0.36, -0.51, -0.88)),
    ("H4", "H", (1.88, -0.51, 0.88)),
    ("H5", "H", (1.88, -0.51, -0.88)),
    ("H6", "H", (3.06, 1.16, 0.00)),
)
LIGAND_BONDS = ((0, 1), (1, 2), (0, 3), (0, 4), (0, 5), (1, 6), (1, 7), (2, 8))
LIGAND_ATOMIC_NUMBERS = (6, 6, 8, 1, 1, 1, 1, 1, 1)

#: TIP3P internal geometry, Angstroms.
WATER_ATOMS = (
    ("O", "O", (0.0000, 0.0000, 0.0)),
    ("H1", "H", (0.9572, 0.0000, 0.0)),
    ("H2", "H", (-0.2400, 0.9266, 0.0)),
)
PROTEIN_ATOMS = (
    ("N", "N", (0.00, 0.00, 0.00)),
    ("CA", "C", (1.45, 0.00, 0.00)),
    ("CB", "C", (1.95, 1.42, 0.00)),
    ("C", "C", (2.00, -0.75, 1.20)),
    ("O", "O", (1.60, -1.90, 1.40)),
)

_MASSES = {"C": 12.011, "O": 15.999, "H": 1.008, "N": 14.007}
_CHARGES = {"C": -0.1, "O": -0.4, "H": 0.1, "N": -0.3}
_SIGMAS = {"C": 0.34, "O": 0.315, "H": 0.106, "N": 0.325}
_EPSILONS = {"C": 0.36, "O": 0.636, "H": 0.066, "N": 0.71}


@dataclass
class ReferenceSystem:
    """A periodic OpenMM system with the residue layout the pipeline expects."""

    topology: Any
    system: Any
    positions_nm: np.ndarray
    ligand_atoms: list[int]
    ghost_atoms: list[int]

    def positions_quantity(self):
        import openmm.unit as unit

        return self.positions_nm * unit.nanometer

    def context(self, system: Any = None, platform: str = "Reference"):
        import openmm
        import openmm.unit as unit

        integrator = openmm.VerletIntegrator(0.0005 * unit.picoseconds)
        context = openmm.Context(
            system if system is not None else self.system,
            integrator,
            openmm.Platform.getPlatformByName(platform),
        )
        context.setPositions(self.positions_quantity())
        context.setPeriodicBoxVectors(*self.system.getDefaultPeriodicBoxVectors())
        # OpenMM does not own the Python Integrator; a garbage-collected
        # integrator takes its Context down with it.
        context._csbrt_integrator = integrator
        return context

    def potential_energy(self, system: Any = None, parameters: dict | None = None):
        import openmm.unit as unit

        context = self.context(system=system)
        for name, value in (parameters or {}).items():
            context.setParameter(name, value)
        return (
            context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoule_per_mole)
        )


def build_reference_system(
    num_waters: int = 4, ghost_water_index: int | None = 3
) -> ReferenceSystem:
    """Build the fixture. ``ghost_water_index`` switches one water off."""
    import openmm
    import openmm.app as app
    import openmm.unit as unit

    topology = app.Topology()
    box = (
        openmm.Vec3(BOX_NM, 0, 0),
        openmm.Vec3(0, BOX_NM, 0),
        openmm.Vec3(0, 0, BOX_NM),
    )
    topology.setPeriodicBoxVectors(box * unit.nanometer)
    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(*[v * unit.nanometer for v in box])

    elements = {
        "C": app.element.carbon,
        "O": app.element.oxygen,
        "H": app.element.hydrogen,
        "N": app.element.nitrogen,
    }
    positions: list[tuple[float, float, float]] = []
    ligand_atoms: list[int] = []
    ghost_atoms: list[int] = []

    nonbonded = openmm.NonbondedForce()
    nonbonded.setNonbondedMethod(openmm.NonbondedForce.PME)
    nonbonded.setCutoffDistance(0.9 * unit.nanometer)
    nonbonded.setEwaldErrorTolerance(5.0e-4)
    bonds = openmm.HarmonicBondForce()
    angles = openmm.HarmonicAngleForce()
    torsions = openmm.PeriodicTorsionForce()

    chain = topology.addChain()

    def add_residue(name, template, origin_nm, ghost=False):
        residue = topology.addResidue(name, chain)
        indices = []
        for atom_name, symbol, xyz_ang in template:
            atom = topology.addAtom(atom_name, elements[symbol], residue)
            index = system.addParticle(_MASSES[symbol] * unit.dalton)
            assert index == atom.index
            positions.append(
                tuple(origin_nm[axis] + xyz_ang[axis] * 0.1 for axis in range(3))
            )
            if ghost:
                nonbonded.addParticle(0.0, _SIGMAS[symbol], 0.0)
                ghost_atoms.append(index)
            else:
                nonbonded.addParticle(
                    _CHARGES[symbol], _SIGMAS[symbol], _EPSILONS[symbol]
                )
            indices.append(index)
        return indices

    # Protein first, so the ligand does not start at particle 0 and an
    # off-by-one in the partitioner cannot pass by accident.
    protein = add_residue("ALA", PROTEIN_ATOMS, (0.4, 0.4, 0.4))
    atoms = list(topology.atoms())
    for first, second in ((0, 1), (1, 2), (1, 3), (3, 4)):
        topology.addBond(atoms[protein[first]], atoms[protein[second]])
        bonds.addBond(protein[first], protein[second], 0.15, 250000.0)

    ligand = add_residue("LIG", LIGAND_ATOMS, (1.25, 1.25, 1.25))
    ligand_atoms.extend(ligand)
    atoms = list(topology.atoms())
    for first, second in LIGAND_BONDS:
        topology.addBond(atoms[ligand[first]], atoms[ligand[second]])
        bonds.addBond(ligand[first], ligand[second], 0.145, 260000.0)
    angles.addAngle(ligand[0], ligand[1], ligand[2], 1.911, 400.0)
    angles.addAngle(ligand[1], ligand[2], ligand[8], 1.824, 400.0)
    torsions.addTorsion(ligand[0], ligand[1], ligand[2], ligand[8], 3, 0.0, 1.5)

    for index in range(num_waters):
        offset = (0.6 + 0.35 * index, 1.9, 0.5 + 0.3 * index)
        water = add_residue(
            "HOH", WATER_ATOMS, offset, ghost=(index == ghost_water_index)
        )
        atoms = list(topology.atoms())
        for first, second in ((0, 1), (0, 2)):
            topology.addBond(atoms[water[first]], atoms[water[second]])
        bonds.addBond(water[0], water[1], 0.09572, 462750.0)
        bonds.addBond(water[0], water[2], 0.09572, 462750.0)
        angles.addAngle(water[1], water[0], water[2], 1.8242, 836.8)

    # 1-2 exceptions, so the nonbonded force does not blow up on bonded pairs.
    bond_pairs = [
        (bonds.getBondParameters(i)[0], bonds.getBondParameters(i)[1])
        for i in range(bonds.getNumBonds())
    ]
    nonbonded.createExceptionsFromBonds(bond_pairs, 0.8333, 0.5)

    for force in (bonds, angles, torsions, nonbonded):
        system.addForce(force)

    return ReferenceSystem(
        topology=topology,
        system=system,
        positions_nm=np.asarray(positions, dtype=np.float64),
        ligand_atoms=ligand_atoms,
        ghost_atoms=ghost_atoms,
    )


def ligand_coordinates_angstrom() -> np.ndarray:
    """The reference ligand pose, in Angstroms, for UQ and committee checks."""
    return np.asarray([atom[2] for atom in LIGAND_ATOMS], dtype=np.float64)


def distorted_ligand_coordinates(kind: str = "clash") -> np.ndarray:
    """A deliberately out-of-distribution ligand pose.

    ``clash`` drops the hydroxyl hydrogen onto a methyl hydrogen; ``stretch``
    tears the C-O bond open; ``torsion`` rotates the hydroxyl out of its
    minimum without breaking anything, which only a committee can catch.
    """
    coords = ligand_coordinates_angstrom()
    if kind == "clash":
        coords[8] = coords[3] + 0.2
    elif kind == "stretch":
        coords[2] = coords[2] + np.array([6.0, 0.0, 0.0])
        coords[8] = coords[8] + np.array([6.0, 0.0, 0.0])
    elif kind == "torsion":
        # Rotate H6 about the C2-O1 axis by 90 degrees.
        axis = coords[2] - coords[1]
        axis = axis / np.linalg.norm(axis)
        offset = coords[8] - coords[2]
        parallel = np.dot(offset, axis) * axis
        perpendicular = offset - parallel
        binormal = np.cross(axis, perpendicular)
        coords[8] = coords[2] + parallel + binormal
    else:
        raise ValueError(f"Unknown distortion {kind!r}")
    return coords
