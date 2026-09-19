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
