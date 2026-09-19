"""Integration tests: config schema and CLI wiring for the MACE surrogate."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from csbrt.mace_surrogate import MACEConfig


def test_config_example_mlff_block_roundtrips() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config.example.yaml"
    cfg = yaml.safe_load(config_path.read_text())
    assert "mlff" in cfg, "config.example.yaml must document the mlff block"
    mace = MACEConfig.from_dict(cfg["mlff"])
    assert mace.enabled is False
    assert mace.model_name == "mace-off23-small"
    assert mace.uq_force_threshold_ev_per_ang == 0.05
    mace.validate()


def test_cli_merge_mace_uses_canonical_keys() -> None:
    from csbrt import cli

    cfg: dict = {}
    opt = argparse.Namespace(
        enable_mace_surrogate=True,
        mace_model="mace-off23-medium",
        mace_uq_threshold=0.12,
        mace_device="cpu",
    )
    cli._merge_cli_mace(cfg, opt)
    assert cfg["mlff"]["enabled"] is True
    assert cfg["mlff"]["model_name"] == "mace-off23-medium"
    # canonical key, so MACEConfig.from_dict picks the threshold up
    assert cfg["mlff"]["uq_force_threshold_ev_per_ang"] == 0.12
    mace = MACEConfig.from_dict(cfg["mlff"])
    assert mace.uq_force_threshold_ev_per_ang == 0.12


def test_cli_merge_mace_no_flags_leaves_config() -> None:
    from csbrt import cli

    cfg: dict = {"mlff": {"enabled": False}}
    opt = argparse.Namespace(
        enable_mace_surrogate=False,
        mace_model=None,
        mace_uq_threshold=None,
        mace_device=None,
    )
    cli._merge_cli_mace(cfg, opt)
    assert cfg == {"mlff": {"enabled": False}}
