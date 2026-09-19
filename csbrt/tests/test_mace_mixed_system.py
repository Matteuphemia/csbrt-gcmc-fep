"""Unit tests for MACE mixed system creation and configuration."""

import pytest
import numpy as np

from csbrt.mace_surrogate import (
    MACEConfig,
    MACEMixedSystemBuilder,
    partition_ml_atoms,
    DEFAULT_FOUNDATION_MODELS,
)


def test_mace_config_defaults():
    cfg = MACEConfig()
    assert cfg.enabled is True
    assert cfg.model_name in DEFAULT_FOUNDATION_MODELS
    assert cfg.uq_force_threshold_ev_per_ang == 0.05
    assert cfg.uq_energy_threshold_kcal_per_mol == 1.0
    assert cfg.binding_site_radius_nm == 1.0
    assert cfg.embedding == "mechanical"


def test_mace_config_unit_conversions():
    cfg = MACEConfig(uq_force_threshold_ev_per_ang=0.05, uq_energy_threshold_kcal_per_mol=1.0)
    # 0.05 eV/A * 9648.53321233
    assert abs(cfg.uq_force_threshold_kj_per_mol_nm - 482.426) < 0.1
    # 1.0 kcal/mol * 4.184
    assert abs(cfg.uq_energy_threshold_kj_per_mol - 4.184) < 1e-4


def test_mace_config_serialization():
    cfg = MACEConfig(model_name="mace-off23-medium", committee_size=6)
    d = cfg.to_dict()
    assert d["model_name"] == "mace-off23-medium"
    assert d["committee_size"] == 6

    cfg2 = MACEConfig.from_dict(d)
    assert cfg2.model_name == "mace-off23-medium"
    assert cfg2.committee_size == 6

    sig = cfg.compute_signature()
    assert sig["model_name"] == "mace-off23-medium"
    assert "model_hash" in sig


def test_builder_initialization():
    cfg = MACEConfig(enabled=True, model_name="mace-off23-small")
    builder = MACEMixedSystemBuilder(cfg)
    assert builder.config == cfg

