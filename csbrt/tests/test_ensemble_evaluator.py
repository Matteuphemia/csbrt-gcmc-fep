"""Tests for the MACE committee ensemble evaluator (pure-Python parts)."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from csbrt.mace_surrogate import (
    MACEConfig,
    MACEEnsembleEvaluator,
    MACEEnsembleUnavailable,
    _select_ml_shell,
)


def test_select_ml_shell_geometry() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.5, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    local_indices, local_ml = _select_ml_shell(positions, ml_atoms=[1], shell_radius_ang=3.0)
    assert list(local_indices) == [0, 1, 2]
    assert list(local_ml) == [1]  # atom 1 maps to local index 1


def test_select_ml_shell_requires_ml_atoms() -> None:
    with pytest.raises(ValueError):
        _select_ml_shell(np.zeros((3, 3)), ml_atoms=[], shell_radius_ang=3.0)


def test_committee_size() -> None:
    assert MACEEnsembleEvaluator(MACEConfig()).committee_size == 1
    assert MACEEnsembleEvaluator(MACEConfig(), model_paths=["a", "b"]).committee_size == 2


def test_ensure_loaded_raises_when_stack_unavailable(monkeypatch) -> None:
    # Insert fake ase/torch so the availability gate passes, then force the
    # calculator builder to report the stack missing.
    monkeypatch.setitem(sys.modules, "ase", types.ModuleType("ase"))
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))

    evaluator = MACEEnsembleEvaluator(MACEConfig())

    def boom(model_ref: str):
        raise MACEEnsembleUnavailable("mace-torch missing (test)")

    monkeypatch.setattr(evaluator, "_build_calculator", boom)
    with pytest.raises(MACEEnsembleUnavailable):
        evaluator.ensure_loaded()
