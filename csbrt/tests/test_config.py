"""MACEConfig: validation, aliases, path resolution and checkpoint signatures."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from csbrt.mace_surrogate import KNOWN_FOUNDATION_MODELS, MACEConfig


def test_defaults_are_off_and_safe():
    config = MACEConfig()
    assert config.enabled is False
    # Loch toggles ghost waters in the live Context; a fixed ML water set
    # cannot follow that, so waters stay out unless asked for.
    assert config.include_binding_site_waters is False
    assert config.interpolate is True
    assert config.geometry_guard is True
    assert config.strict is False
    assert config.uq_force_threshold_ev_per_ang == pytest.approx(0.05)


def test_model_aliases_resolve():
    assert MACEConfig(model_name="mace-omol-0").model_name == (
        "mace-omol-0-extra-large"
    )
    assert MACEConfig(model_name="mace-off23").model_name == "mace-off23-small"
    for name in KNOWN_FOUNDATION_MODELS:
        assert MACEConfig(model_name=name).model_name == name


def test_potential_name_switches_for_a_local_model(tmp_path):
    assert MACEConfig().potential_name == "mace-off23-small"
    local = MACEConfig(model_path=str(tmp_path / "gen3.model"))
    assert local.potential_name == "mace"


def test_precision_is_normalised():
    assert MACEConfig(precision="float32").precision == "single"
    assert MACEConfig(precision="mixed").precision == "single"
    assert MACEConfig(precision="float64").precision == "double"
    with pytest.raises(ValueError, match="precision"):
        MACEConfig(precision="bfloat16")


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"uq_force_threshold_ev_per_ang": 0.0}, "uq_force_threshold"),
        ({"uq_energy_threshold_kcal_per_mol": -1}, "uq_energy_threshold"),
        ({"uq_interval_steps": 0}, "uq_interval_steps"),
        ({"fallback_steps": 0}, "fallback_steps"),
        ({"fallback_abort_fraction": 0.0}, "fallback_abort_fraction"),
        ({"fallback_abort_fraction": 1.5}, "fallback_abort_fraction"),
        ({"max_ml_atoms": 0}, "max_ml_atoms"),
        ({"ml_force_group": 32}, "force group"),
        ({"binding_site_radius_nm": 0}, "binding_site_radius_nm"),
        ({"embedding": "qm/mm"}, "embedding"),
        ({"return_energy_type": "total"}, "return_energy_type"),
    ],
)
def test_invalid_settings_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        MACEConfig(**kwargs)


def test_from_dict_ignores_unknown_and_renames_legacy(caplog):
    with caplog.at_level(logging.WARNING):
        config = MACEConfig.from_dict(
            {
                "enabled": True,
                "fallback_recovery_steps": 25,
                "uq_force_threshold": 0.08,
                "ood_buffer_path": "buffers",
                "nonsense": 1,
            }
        )
    assert config.fallback_steps == 25
    assert config.uq_force_threshold_ev_per_ang == pytest.approx(0.08)
    assert config.ood_buffer_dir == "buffers"
    assert "nonsense" in caplog.text


def test_from_dict_round_trips():
    original = MACEConfig(
        enabled=True, committee_model_paths=["a.model", "b.model"],
        uq_interval_steps=50,
    )
    restored = MACEConfig.from_dict(original.to_dict())
    assert restored == original
    assert restored.committee_size == 2


def test_resolve_paths_is_relative_to_the_stage_directory(tmp_path):
    config = MACEConfig(
        enabled=True,
        ood_buffer_dir="al_buffer",
        model_path="models/gen2.model",
        committee_model_paths=["models/a.model", "/abs/b.model"],
    )
    resolved = config.resolve_paths(tmp_path)
    assert resolved.ood_buffer_dir == str(tmp_path / "al_buffer")
    assert resolved.model_path == str(tmp_path / "models/gen2.model")
    assert resolved.committee_model_paths == (
        str(tmp_path / "models/a.model"),
        "/abs/b.model",
    )


def test_threshold_conversions():
    config = MACEConfig(
        uq_force_threshold_ev_per_ang=0.05,
        uq_energy_threshold_kcal_per_mol=1.0,
    )
    assert config.uq_force_threshold_kcal_per_mol_ang == pytest.approx(1.153, abs=1e-3)
    assert config.uq_force_threshold_kj_per_mol_nm == pytest.approx(48.24, abs=1e-2)
    assert config.uq_energy_threshold_kj_per_mol == pytest.approx(4.184)
    assert config.uq_energy_threshold_ev == pytest.approx(0.04336, abs=1e-5)


# --------------------------------------------------------------------------- signature


def test_signature_hashes_a_local_model(tmp_path):
    model = tmp_path / "gen1.model"
    model.write_bytes(b"weights")
    signature = MACEConfig(enabled=True, model_path=str(model)).compute_signature()
    assert signature["model_sha256"] != "none"
    assert len(signature["model_sha256"]) == 64

    model.write_bytes(b"different weights")
    assert (
        MACEConfig(enabled=True, model_path=str(model)).compute_signature()[
            "model_sha256"
        ]
        != signature["model_sha256"]
    )


def test_signature_marks_a_missing_model(tmp_path):
    signature = MACEConfig(
        enabled=True, model_path=str(tmp_path / "absent.model")
    ).compute_signature()
    assert signature["model_sha256"] == "missing"


def test_signature_ignores_runtime_only_settings():
    """The same physics on a different GPU must still match its checkpoint."""
    first = MACEConfig(enabled=True, device="cuda", ood_buffer_dir="a")
    second = MACEConfig(enabled=True, device="cpu", ood_buffer_dir="b")
    assert first.compute_signature() == second.compute_signature()


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_name", "mace-off23-medium"),
        ("uq_force_threshold_ev_per_ang", 0.08),
        ("fallback_steps", 25),
        ("include_binding_site_waters", True),
        ("ligand_resname", "MOL"),
        ("embedding", "electrostatic"),
        ("precision", "double"),
        ("geometry_guard", False),
    ],
)
def test_signature_changes_with_the_sampled_ensemble(field, value):
    """Anything that changes the physics must invalidate the checkpoint."""
    baseline = MACEConfig(enabled=True).compute_signature()
    changed = MACEConfig(enabled=True, **{field: value}).compute_signature()
    assert changed != baseline


def test_disabled_and_enabled_signatures_differ():
    assert (
        MACEConfig(enabled=False).compute_signature()
        != MACEConfig(enabled=True).compute_signature()
    )
