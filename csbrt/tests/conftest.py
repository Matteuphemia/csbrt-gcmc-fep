"""Fixtures for the MACE surrogate tests.

Everything here is real OpenMM: the periodic fixture from
``mace_surrogate.testsystems`` -- a ligand, a few waters (one of them a
Loch-style non-interacting ghost) and a protein residue. The only thing faked
is the ML model itself: a registered openmm-ml potential that applies a cheap
analytic force. That keeps the tests honest about the parts this package owns
(partitioning, the in-place mixed-system swap, the lambda_interpolate switch,
the UQ and fallback logic) without needing a GPU or a 100 MB checkpoint.

The fixture lives in the package rather than here because the preflight check
and the throughput benchmark exercise the same construction path, and a
separate copy would drift.
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

from csbrt.mace_surrogate.testsystems import (  # noqa: E402
    BOX_NM,
    LIGAND_ATOMS,
    LIGAND_BONDS,
    build_reference_system,
)

STUB_POTENTIAL_NAME = "csbrt-test-stub"


@pytest.fixture
def built_system():
    return build_reference_system()


@pytest.fixture
def built_system_no_ghost():
    return build_reference_system(ghost_water_index=None)


# --------------------------------------------------------------------------- stub ML


def register_stub_potential(name: str = STUB_POTENTIAL_NAME, strength: float = 200000.0):
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
            # Stand in for what an MLFF supplies: the ML region's whole
            # intramolecular energy. The mechanical embedding removes the MM
            # bonded terms inside the region *and* zeroes its internal
            # nonbonded interactions, so a stand-in that only replaces the
            # bonds leaves nothing holding the atoms apart and the region
            # collapses under minimisation. Harmonic bonds plus a purely
            # repulsive term over every other in-region pair is the smallest
            # thing that behaves like a molecule.
            #
            # Both are CustomBondForce: openmm-ml's interpolation refuses a
            # CustomNonbondedForce, and over 40-250 atoms an explicit pair list
            # is cheap anyway.
            selection = sorted(
                set(range(system.getNumParticles()))
                if atoms is None
                else {int(i) for i in atoms}
            )
            in_region = set(selection)

            bonded = openmm.CustomBondForce("0.5*kml*(r-r0ml)^2")
            bonded.addGlobalParameter("kml", strength)
            bonded.addGlobalParameter("r0ml", 0.145)
            neighbours: dict[int, set[int]] = {i: set() for i in selection}
            for bond in topology.bonds():
                first, second = int(bond[0].index), int(bond[1].index)
                if first in in_region and second in in_region:
                    bonded.addBond(first, second, [])
                    neighbours[first].add(second)
                    neighbours[second].add(first)
            bonded.setForceGroup(forceGroup)
            system.addForce(bonded)

            repulsive = openmm.CustomBondForce("epsml*(sigml/r)^12")
            repulsive.addGlobalParameter("epsml", 1.0)
            repulsive.addGlobalParameter("sigml", 0.24)
            for index, first in enumerate(selection):
                one_three = {
                    third
                    for second in neighbours[first]
                    for third in neighbours[second]
                }
                excluded = neighbours[first] | one_three | {first}
                for second in selection[:index]:
                    if second not in excluded:
                        repulsive.addBond(first, second, [])
            repulsive.setForceGroup(forceGroup)
            system.addForce(repulsive)

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
