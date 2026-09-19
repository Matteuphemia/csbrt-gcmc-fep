"""Fixtures for the MACE surrogate tests.

Everything here is real OpenMM: a hand-built periodic System with a ligand, a
few waters (one of them a Loch-style non-interacting ghost) and a protein
residue. The only thing faked is the ML model itself -- a registered openmm-ml
potential that applies a cheap analytic force. That keeps the tests honest
about the parts this package owns (partitioning, the in-place mixed-system
swap, the lambda_interpolate switch, the UQ and fallback logic) without needing
a GPU or a 100 MB checkpoint to run them.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

openmm = pytest.importorskip("openmm")
app = pytest.importorskip("openmm.app")
unit = pytest.importorskip("openmm.unit")

BOX_NM = 2.5
STUB_POTENTIAL_NAME = "csbrt-test-stub"

# Ethanol, Angstroms. Chosen so every bond and contact is physically sane, so
# the geometry guard passes on the reference pose and only fires on the
# deliberately distorted ones.
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
LIGAND_BONDS = (
    (0, 1), (1, 2), (0, 3), (0, 4), (0, 5), (1, 6), (1, 7), (2, 8),
)
# TIP3P internal geometry, Angstroms.
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


class BuiltSystem:
    """A periodic OpenMM System with the residue layout the pipeline expects."""

    def __init__(self, topology, system, positions_nm, ligand_atoms, ghost_atoms):
        self.topology = topology
        self.system = system
        self.positions_nm = positions_nm
        self.ligand_atoms = ligand_atoms
        self.ghost_atoms = ghost_atoms

    def positions_quantity(self):
        return self.positions_nm * unit.nanometer

    def context(self, system=None, platform="Reference"):
        integrator = openmm.VerletIntegrator(0.0005 * unit.picoseconds)
        context = openmm.Context(
            system if system is not None else self.system,
            integrator,
            openmm.Platform.getPlatformByName(platform),
        )
        context.setPositions(self.positions_quantity())
        context.setPeriodicBoxVectors(*self.system.getDefaultPeriodicBoxVectors())
        # Keep a reference: OpenMM does not own the Python Integrator, and a
        # garbage-collected integrator takes the Context down with it.
        context._csbrt_integrator = integrator
        return context

    def potential_energy(self, system=None, parameters=None):
        context = self.context(system=system)
        for name, value in (parameters or {}).items():
            context.setParameter(name, value)
        state = context.getState(getEnergy=True)
        return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def _build(num_waters: int = 4, ghost_water_index: int | None = 3) -> BuiltSystem:
    topology = app.Topology()
    topology.setPeriodicBoxVectors(
        (
            openmm.Vec3(BOX_NM, 0, 0),
            openmm.Vec3(0, BOX_NM, 0),
            openmm.Vec3(0, 0, BOX_NM),
        )
        * unit.nanometer
    )
    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(BOX_NM, 0, 0) * unit.nanometer,
        openmm.Vec3(0, BOX_NM, 0) * unit.nanometer,
        openmm.Vec3(0, 0, BOX_NM) * unit.nanometer,
    )

    positions: list[tuple[float, float, float]] = []
    elements = {"C": app.element.carbon, "O": app.element.oxygen,
                "H": app.element.hydrogen, "N": app.element.nitrogen}
    masses = {"C": 12.011, "O": 15.999, "H": 1.008, "N": 14.007}
    charges = {"C": -0.1, "O": -0.4, "H": 0.1, "N": -0.3}
    sigmas = {"C": 0.34, "O": 0.315, "H": 0.106, "N": 0.325}
    epsilons = {"C": 0.36, "O": 0.636, "H": 0.066, "N": 0.71}

    nonbonded = openmm.NonbondedForce()
    nonbonded.setNonbondedMethod(openmm.NonbondedForce.PME)
    nonbonded.setCutoffDistance(0.9 * unit.nanometer)
    nonbonded.setEwaldErrorTolerance(5.0e-4)
    bonds = openmm.HarmonicBondForce()
    angles = openmm.HarmonicAngleForce()
    torsions = openmm.PeriodicTorsionForce()

    ligand_atoms: list[int] = []
    ghost_atoms: list[int] = []

    def add_residue(chain, name, template, origin_nm, ghost=False):
        residue = topology.addResidue(name, chain)
        indices = []
        for atom_name, symbol, xyz_ang in template:
            atom = topology.addAtom(atom_name, elements[symbol], residue)
            index = system.addParticle(masses[symbol] * unit.dalton)
            assert index == atom.index
            positions.append(
                tuple(origin_nm[axis] + xyz_ang[axis] * 0.1 for axis in range(3))
            )
            if ghost:
                nonbonded.addParticle(0.0, sigmas[symbol], 0.0)
                ghost_atoms.append(index)
            else:
                nonbonded.addParticle(
                    charges[symbol], sigmas[symbol], epsilons[symbol]
                )
            indices.append(index)
        return residue, indices

    chain = topology.addChain()

    # Protein first, so the ligand does not start at index 0 and an off-by-one
    # in the partitioner cannot pass by accident.
    _protein, protein_indices = add_residue(
        chain, "ALA", PROTEIN_ATOMS, (0.4, 0.4, 0.4)
    )
    for first, second in ((0, 1), (1, 2), (1, 3), (3, 4)):
        topology.addBond(
            list(topology.atoms())[protein_indices[first]],
            list(topology.atoms())[protein_indices[second]],
        )
        bonds.addBond(
            protein_indices[first], protein_indices[second], 0.15, 250000.0
        )

    _ligand, ligand_indices = add_residue(
        chain, "LIG", LIGAND_ATOMS, (1.25, 1.25, 1.25)
    )
    ligand_atoms.extend(ligand_indices)
    atoms = list(topology.atoms())
    for first, second in LIGAND_BONDS:
        topology.addBond(atoms[ligand_indices[first]], atoms[ligand_indices[second]])
        bonds.addBond(ligand_indices[first], ligand_indices[second], 0.145, 260000.0)
    angles.addAngle(ligand_indices[0], ligand_indices[1], ligand_indices[2],
                    1.911, 400.0)
    angles.addAngle(ligand_indices[1], ligand_indices[2], ligand_indices[8],
                    1.824, 400.0)
    torsions.addTorsion(
        ligand_indices[0], ligand_indices[1], ligand_indices[2],
        ligand_indices[8], 3, 0.0, 1.5,
    )

    for index in range(num_waters):
        offset = (0.6 + 0.35 * index, 1.9, 0.5 + 0.3 * index)
        _water, water_indices = add_residue(
            chain, "HOH", WATER_ATOMS, offset, ghost=(index == ghost_water_index)
        )
        water_atoms = list(topology.atoms())
        for first, second in ((0, 1), (0, 2)):
            topology.addBond(
                water_atoms[water_indices[first]], water_atoms[water_indices[second]]
            )
        bonds.addBond(water_indices[0], water_indices[1], 0.09572, 462750.0)
        bonds.addBond(water_indices[0], water_indices[2], 0.09572, 462750.0)
        angles.addAngle(
            water_indices[1], water_indices[0], water_indices[2], 1.8242, 836.8
        )

    # 1-2 exceptions, so the nonbonded force does not blow up on bonded pairs.
    bond_pairs = [
        (bonds.getBondParameters(i)[0], bonds.getBondParameters(i)[1])
        for i in range(bonds.getNumBonds())
    ]
    nonbonded.createExceptionsFromBonds(bond_pairs, 0.8333, 0.5)

    for force in (bonds, angles, torsions, nonbonded):
        system.addForce(force)

    return BuiltSystem(
        topology=topology,
        system=system,
        positions_nm=np.asarray(positions, dtype=np.float64),
        ligand_atoms=ligand_atoms,
        ghost_atoms=ghost_atoms,
    )


@pytest.fixture
def built_system() -> BuiltSystem:
    return _build()


@pytest.fixture
def built_system_no_ghost() -> BuiltSystem:
    return _build(ghost_water_index=None)


# --------------------------------------------------------------------------- stub ML


def register_stub_potential(name: str = STUB_POTENTIAL_NAME, strength: float = 1000.0):
    """Register a trivial openmm-ml potential so mixed systems can be built.

    openmm-ml's own ``registerImplFactory`` hook is the supported extension
    point, so the mixed system produced here goes through exactly the code path
    MACE would: the same ML-ML bonded/nonbonded removal, the same exceptions,
    the same ``lambda_interpolate`` CustomCVForce.
    """
    openmmml = pytest.importorskip("openmmml")

    class StubImpl(openmmml.mlpotential.MLPotentialImpl):
        def __init__(self, name):
            self.name = name

        def addForces(self, topology, system, atoms, forceGroup, **args):
            # A harmonic tether on each ML atom: cheap, analytic, and non-zero
            # so a test can tell the ML and MM Hamiltonians apart by energy.
            force = openmm.CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
            force.addGlobalParameter("k", strength)
            for axis in ("x0", "y0", "z0"):
                force.addPerParticleParameter(axis)
            selection = (
                range(system.getNumParticles()) if atoms is None else atoms
            )
            for index in selection:
                force.addParticle(int(index), [0.0, 0.0, 0.0])
            force.setForceGroup(forceGroup)
            system.addForce(force)

        def getMLLongRange(self):
            # MACE-OFF is short-range; say the same so the periodic mechanical
            # embedding takes the same branch it takes for the real model.
            return False

    class StubFactory(openmmml.mlpotential.MLPotentialImplFactory):
        def createImpl(self, name, **args):
            return StubImpl(name)

    openmmml.MLPotential.registerImplFactory(name, StubFactory())
    return name


@pytest.fixture(scope="session")
def stub_potential() -> str:
    return register_stub_potential()


@pytest.fixture
def stub_config(stub_potential, tmp_path):
    from csbrt.mace_surrogate import MACEConfig

    return MACEConfig(
        enabled=True,
        model_name=stub_potential,
        device="cpu",
        precision="double",
        ligand_resname="LIG",
        include_binding_site_waters=False,
        ood_buffer_dir=str(tmp_path / "al_buffer"),
        uq_interval_steps=5,
        fallback_steps=10,
    )
