"""Integration test for CLI MACE argument parsing and config overlay."""

import pytest
import argparse
from pathlib import Path

from csbrt.cli import _merge_cli_mace, load_config


def test_merge_cli_mace_flags():
    cfg = {"receptor": "rec.pdb", "output_dir": "runs"}
    
    # Simulate CLI arguments passed
    opt = argparse.Namespace(
        enable_mace_surrogate=True,
        mace_model="mace-off23-medium",
        mace_uq_threshold=0.08,
        mace_device="cuda",
    )
    
    _merge_cli_mace(cfg, opt)
    
    assert "mlff" in cfg
    assert cfg["mlff"]["enabled"] is True
    assert cfg["mlff"]["model_name"] == "mace-off23-medium"
    assert cfg["mlff"]["uq_force_threshold"] == 0.08
    assert cfg["mlff"]["device"] == "cuda"

