"""Tests for MACE mixed-system construction and configuration."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from csbrt.mace_surrogate import (
    MACEConfig,
    MACEMixedSystemBuilder,
    create_mace_mixed_system,
    partition_ml_atoms,
)


# --------------------------------------------------------------------------- config


def test_config_defaults() -> None:
    cfg = MACEConfig()
    assert cfg.model_name == "mace-off23-small"
    assert cfg.uq_force_threshold_ev_per_ang == pytest.approx(0.05)
    assert cfg.openmmml_precision == "single"


def test_config_from_dict_canonical_and_aliases() -> None:
    cfg = MACEConfig.from_dict(
        {"enabled": True, "model": "mace-off23-medium", "uq_threshold": 0.07}
    )
    assert cfg.model_name == "mace-off23-medium"
    assert cfg.uq_force_threshold_ev_per_ang == pytest.approx(0.07)

    cfg2 = MACEConfig.from_dict(
        {"uq_force_threshold": 0.09, "fallback_steps": 25}
    )
    assert cfg2.uq_force_threshold_ev_per_ang == pytest.approx(0.09)
    assert cfg2.fallback_recovery_steps == 25


def test_config_validate_rejects_unknown_model() -> None:
    with pytest.raises(ValueError):
        MACEConfig(model_name="mace-omol-0").validate()


def test_config_validate_rejects_missing_checkpoint() -> None:
    with pytest.raises(FileNotFoundError):
        MACEConfig(model_name="mace", model_path="no-such-file.pt").validate()


def test_config_validate_rejects_bad_threshold() -> None:
    with pytest.raises(ValueError):
        MACEConfig(uq_force_threshold_ev_per_ang=-1.0).validate()


def test_config_precision_mapping() -> None:
    assert MACEConfig(precision="float32").openmmml_precision == "single"
    assert MACEConfig(precision="float64").openmmml_precision == "double"
    assert MACEConfig(precision="single").openmmml_precision == "single"
    assert MACEConfig(precision="double").openmmml_precision == "double"


def test_config_signature_deterministic() -> None:
    a = MACEConfig(model_name="mace-off23-small", uq_force_threshold_ev_per_ang=0.05)
    b = MACEConfig(model_name="mace-off23-small", uq_force_threshold_ev_per_ang=0.05)
    assert a.compute_signature() == b.compute_signature()
    assert a.compute_signature()["model_name"] == "mace-off23-small"


# --------------------------------------------------------------------------- partition


class FakeAtom:
    def __init__(self, index: int) -> None:
        self.index = index


class FakeResidue:
    def __init__(self, name: str, index: int, atoms: list[FakeAtom]) -> None:
        self.name = name
        self.index = index
        self._atoms = atoms

    def atoms(self):
        return self._atoms


class FakeTopology:
    """OpenMM-style topology exposing .residues() and .atoms()."""

    def __init__(self, residues: list[FakeResidue]) -> None:
        self._residues = residues

    def residues(self):
        return self._residues

    def atoms(self):
        for res in self._residues:
            yield from res.atoms()


def _make_topology() -> FakeTopology:
    atoms: list[FakeAtom] = []
    residues: list[FakeResidue] = []

    def add_residue(name: str) -> list[int]:
        idxs = []
        for _ in range(3):
            atom = FakeAtom(len(atoms))
            atoms.append(atom)
            idxs.append(atom.index)
        residues.append(FakeResidue(name, len(residues), [FakeAtom(i) for i in idxs]))
        return idxs

    lig = add_residue("LIG")       # atoms 0,1,2 near origin
    add_residue("PRO")             # atoms 3,4,5
    near_water = add_residue("HOH")  # atoms 6,7,8  (O at index 6)
    far_water = add_residue("HOH")   # atoms 9,10,11 (O at index 9)
    assert lig == [0, 1, 2]
    assert near_water == [6, 7, 8]
    assert far_water == [9, 10, 11]
    return FakeTopology(residues)


def test_partition_ligand_only_without_positions() -> None:
    topo = _make_topology()
    ml = partition_ml_atoms(topo, ligand_resname="LIG", include_waters=True)
    assert ml == [0, 1, 2]


def test_partition_includes_binding_site_waters_with_positions() -> None:
    topo = _make_topology()
    positions = np.zeros((12, 3), dtype=np.float64)
    positions[0] = (0.0, 0.0, 0.0)
    positions[1] = (0.1, 0.0, 0.0)
    positions[2] = (0.0, 0.1, 0.0)
    positions[6] = (0.5, 0.0, 0.0)   # near water oxygen -> within 1.0 nm
    positions[9] = (5.0, 0.0, 0.0)   # far water oxygen -> excluded
    ml = partition_ml_atoms(
        topo, positions=positions, ligand_resname="LIG", include_waters=True
    )
    assert ml == [0, 1, 2, 6, 7, 8]


def test_partition_requires_positions_for_waters() -> None:
    topo = _make_topology()
    ml = partition_ml_atoms(topo, ligand_resname="LIG", include_waters=False)
    assert ml == [0, 1, 2]


# --------------------------------------------------------------------------- builder


def test_builder_returns_classical_when_openmmml_missing(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "openmmml", raising=False)
    builder = MACEMixedSystemBuilder(MACEConfig())
    result = builder.build_mixed_system("system", "topology", ml_atoms=[0, 1])
    assert result == "system"


def test_builder_returns_classical_when_no_ml_atoms() -> None:
    builder = MACEMixedSystemBuilder(MACEConfig())
    result = builder.build_mixed_system("system", "topology", ml_atoms=[])
    assert result == "system"


def test_builder_wires_openmmml_correctly(monkeypatch) -> None:
    calls: dict = {}

    class FakePotential:
        def __init__(self, name: str, **kwargs) -> None:
            self.name = name
            self.ctor_kwargs = kwargs
            calls["ctor"] = (name, kwargs)

        def getSupportedEmbeddings(self):
            return {"mechanical", "electrostatic"}

        def createMixedSystem(self, topology, system, atoms, **kwargs):
            calls["create"] = (topology, system, list(atoms), kwargs)
            return "MIXED-SYSTEM"

    fake_module = types.ModuleType("openmmml")
    fake_module.MLPotential = FakePotential
    monkeypatch.setitem(sys.modules, "openmmml", fake_module)

    builder = MACEMixedSystemBuilder(
        MACEConfig(model_name="mace-off23-small", device="cuda", precision="float32")
    )
    result = builder.build_mixed_system("sys", "topo", ml_atoms=[0, 1, 2])
    assert result == "MIXED-SYSTEM"
    assert calls["ctor"] == ("mace-off23-small", {})
    topology, system, atoms, kwargs = calls["create"]
    assert topology == "topo"
    assert system == "sys"
    assert atoms == [0, 1, 2]
    assert kwargs["interpolate"] is True
    assert kwargs["device"] == "cuda"
    assert kwargs["precision"] == "single"
    assert kwargs["embedding"] == "mechanical"


def test_builder_local_checkpoint_uses_mace_and_model_path(monkeypatch) -> None:
    calls: dict = {}

    class FakePotential:
        def __init__(self, name: str, **kwargs) -> None:
            calls["ctor"] = (name, kwargs)

        def getSupportedEmbeddings(self):
            return {"mechanical"}

        def createMixedSystem(self, topology, system, atoms, **kwargs):
            return "MIXED"

    fake_module = types.ModuleType("openmmml")
    fake_module.MLPotential = FakePotential
    monkeypatch.setitem(sys.modules, "openmmml", fake_module)

    builder = MACEMixedSystemBuilder(
        MACEConfig(model_name="mace", model_path="custom.model", device="cuda")
    )
    builder.build_mixed_system("sys", "topo", ml_atoms=[0])
    assert calls["ctor"] == ("mace", {"modelPath": "custom.model"})


def test_create_mace_mixed_system_convenience(monkeypatch) -> None:
    class FakePotential:
        def __init__(self, name: str, **kwargs) -> None:
            pass

        def getSupportedEmbeddings(self):
            return {"mechanical"}

        def createMixedSystem(self, topology, system, atoms, **kwargs):
            return ("MIXED", list(atoms))

    fake_module = types.ModuleType("openmmml")
    fake_module.MLPotential = FakePotential
    monkeypatch.setitem(sys.modules, "openmmml", fake_module)

    result = create_mace_mixed_system(
        "sys", "topo", ml_atoms=[1, 2], config=MACEConfig(model_name="mace-off23-small")
    )
    assert result == ("MIXED", [1, 2])
