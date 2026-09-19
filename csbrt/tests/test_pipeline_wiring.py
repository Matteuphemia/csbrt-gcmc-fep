"""Stage 5: the surrogate's wiring into csbrt's CLI, stages and Slurm scripts."""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "csbrt"
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))  # stage scripts import each other flat

import pipeline_utils as pu  # noqa: E402

from csbrt import cli  # noqa: E402
from csbrt.mace_surrogate import MACEConfig  # noqa: E402
from csbrt.mace_surrogate import somd2_hook  # noqa: E402


# --------------------------------------------------------------------------- flags


def parse(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    pu.add_mace_arguments(parser)
    return parser.parse_args(list(argv))


def test_flags_default_to_off():
    options = parse()
    payload = pu.mace_options_dict(options)
    assert payload["enabled"] is False
    assert pu.mace_command_arguments(payload) == []
    assert pu.mace_signature(payload) is None


def test_flags_round_trip_through_the_command_line():
    """A stage that forwards its flags must reproduce them exactly."""
    original = parse(
        "--enable-mace-surrogate",
        "--mace-model", "mace-off23-medium",
        "--mace-committee", "a.model",
        "--mace-committee", "b.model",
        "--mace-uq-threshold", "0.08",
        "--mace-uq-interval", "250",
        "--mace-fallback-steps", "20",
        "--mace-device", "cpu",
        "--mace-ml-waters",
        "--mace-strict",
    )
    forwarded = pu.mace_command_arguments(pu.mace_options_dict(original))
    restored = parse(*forwarded)
    assert pu.mace_options_dict(restored) == pu.mace_options_dict(original)


def test_legacy_surrogate_alias_still_parses():
    assert parse("--mace-surrogate").enable_mace_surrogate is True


def test_config_is_built_with_the_stage_ligand_name():
    options = parse("--enable-mace-surrogate", "--mace-uq-interval", "37")
    config = pu.mace_config_from_options(options, ligand_resname="MOL")
    assert isinstance(config, MACEConfig)
    assert config.ligand_resname == "MOL"
    assert config.uq_interval_steps == 37


def test_signature_accepts_a_config_or_a_mapping():
    config = pu.mace_config_from_options(parse("--enable-mace-surrogate"))
    assert pu.mace_signature(config) == config.compute_signature()
    assert pu.mace_signature(None) is None
    assert pu.mace_signature({"enabled": False, "model_name": "x"}) is None


def test_classical_runs_keep_a_null_signature():
    """A classical run's marker must not change just because this code exists."""
    assert pu.mace_signature(pu.mace_options_dict(parse())) is None


# --------------------------------------------------------------------------- cli


def test_cli_overrides_only_what_was_given():
    config = {"mlff": {"enabled": True, "model_name": "mace-off23-large"}}
    options = argparse.Namespace(
        enable_mace_surrogate=False, mace_strict=False, mace_model=None,
        mace_model_path=None, mace_committee=None, mace_uq_threshold=None,
        mace_device=None,
    )
    cli._merge_cli_mace(config, options)
    assert config["mlff"] == {"enabled": True, "model_name": "mace-off23-large"}


def test_cli_flags_win_over_the_config():
    config = {"mlff": {"enabled": False, "model_name": "mace-off23-small"}}
    options = argparse.Namespace(
        enable_mace_surrogate=True, mace_strict=True, mace_model="mace-off23-medium",
        mace_model_path=None, mace_committee=["a.model", "b.model"],
        mace_uq_threshold=0.09, mace_device="cpu",
    )
    cli._merge_cli_mace(config, options)
    assert config["mlff"]["enabled"] is True
    assert config["mlff"]["strict"] is True
    assert config["mlff"]["model_name"] == "mace-off23-medium"
    assert config["mlff"]["committee_model_paths"] == ["a.model", "b.model"]
    assert config["mlff"]["uq_force_threshold_ev_per_ang"] == 0.09
    assert MACEConfig.from_dict(config["mlff"]).committee_size == 2


def test_cli_leaves_a_plain_config_untouched():
    config = {"output_dir": "run"}
    options = argparse.Namespace(
        enable_mace_surrogate=False, mace_strict=False, mace_model=None,
        mace_model_path=None, mace_committee=None, mace_uq_threshold=None,
        mace_device=None,
    )
    cli._merge_cli_mace(config, options)
    assert "mlff" not in config


def test_example_config_parses_into_a_valid_config():
    import yaml

    path = Path(__file__).resolve().parents[1] / "config.example.yaml"
    payload = yaml.safe_load(path.read_text())
    assert "mlff" in payload
    config = MACEConfig.from_dict(payload["mlff"])
    assert config.enabled is False
    assert config.include_binding_site_waters is False


def test_stage_scripts_share_one_flag_definition():
    """Every stage must accept the same flags, or a network runs mixed physics."""
    scripts = ("ev71_production.py", "run_ev71_pipeline.py", "run_fep_leg.py")
    for name in scripts:
        source = (PACKAGE / name).read_text()
        assert "add_mace_arguments(parser)" in source, name
        assert "--enable-mace-surrogate\"" not in source.replace(
            "add_mace_arguments", ""
        ), f"{name} redefines the flags instead of using add_mace_arguments"


def test_somd2_config_is_never_polluted_with_mlff():
    """SOMD2 validates its own config keys and rejects unknown ones."""
    source = (PACKAGE / "run_fep_leg.py").read_text()
    assert 'config_payload["mlff"]' not in source
    assert "prepare_leg_environment" in source


# --------------------------------------------------------------------------- somd2 hook


def test_leg_environment_is_wired(tmp_path):
    environment = {"PYTHONPATH": "/existing/path"}
    config = MACEConfig(enabled=True, model_name="mace-off23-small")
    manifest = somd2_hook.prepare_leg_environment(
        config, tmp_path, environment, source="bound-edge01"
    )

    sidecar = Path(manifest["sidecar"])
    assert sidecar.is_file()
    payload = json.loads(sidecar.read_text())
    assert payload["source"] == "bound-edge01"
    assert MACEConfig.from_dict(payload["mlff"]).enabled

    hook = Path(manifest["sitecustomize"])
    assert hook.name == "sitecustomize.py"
    assert str(hook.parent) in environment["PYTHONPATH"].split(os.pathsep)
    assert "/existing/path" in environment["PYTHONPATH"].split(os.pathsep)
    assert environment[somd2_hook.CONFIG_ENVIRONMENT_VARIABLE] == str(sidecar)
    assert Path(manifest["stats_dir"]).is_dir()


def test_sitecustomize_is_valid_python_and_cannot_raise(tmp_path):
    """It runs at interpreter startup; a traceback there kills SOMD2."""
    environment: dict[str, str] = {}
    manifest = somd2_hook.prepare_leg_environment(
        MACEConfig(enabled=True), tmp_path, environment
    )
    source = Path(manifest["sitecustomize"]).read_text()
    ast.parse(source)

    # Import it for real, with nothing installed, and confirm the interpreter
    # survives and says why rather than crashing.
    completed = subprocess.run(
        [sys.executable, "-c", "import sitecustomize; print('alive')"],
        capture_output=True,
        text=True,
        env={
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(Path(manifest["sitecustomize"]).parent),
        },
    )
    assert completed.returncode == 0
    assert "alive" in completed.stdout


def test_sidecar_round_trips(tmp_path):
    environment: dict[str, str] = {}
    config = MACEConfig(
        enabled=True, model_name="mace-off23-medium", uq_interval_steps=250
    )
    manifest = somd2_hook.prepare_leg_environment(config, tmp_path, environment)
    loaded, payload = somd2_hook.load_sidecar(manifest["sidecar"])
    assert loaded == config
    assert payload["output_dir"] == str(tmp_path)


def test_sidecar_absent_is_not_an_error(monkeypatch, tmp_path):
    monkeypatch.delenv(somd2_hook.CONFIG_ENVIRONMENT_VARIABLE, raising=False)
    assert somd2_hook.load_sidecar() is None
    monkeypatch.setenv(
        somd2_hook.CONFIG_ENVIRONMENT_VARIABLE, str(tmp_path / "gone.json")
    )
    assert somd2_hook.load_sidecar() is None


def test_install_without_sire_is_a_no_op(tmp_path, monkeypatch):
    monkeypatch.setattr(somd2_hook, "_INSTALLED", False)
    assert somd2_hook.install(MACEConfig(enabled=False)) is False


def test_window_statistics_are_aggregated(tmp_path):
    stats = tmp_path / "mace_surrogate"
    stats.mkdir()
    for index, (steps, classical) in enumerate([(1000, 100), (1000, 300)]):
        (stats / f"window{index:03d}.json").write_text(
            json.dumps(
                {
                    "fallback": {
                        "md_steps": steps,
                        "classical_steps": classical,
                        "fallback_events": index + 1,
                    },
                    "ood_buffer": {"frames": index * 2},
                }
            )
        )
    summary = somd2_hook.collect_window_statistics(stats)
    assert summary["attached"] is True
    assert summary["windows"] == 2
    assert summary["md_steps"] == 2000
    assert summary["fallback_fraction"] == pytest.approx(0.2)
    assert summary["fallback_events"] == 3
    assert summary["ood_frames"] == 2


def test_window_statistics_of_a_leg_that_never_attached(tmp_path):
    summary = somd2_hook.collect_window_statistics(tmp_path / "nothing")
    assert summary == {"windows": 0, "attached": False}


# --------------------------------------------------------------------------- slurm


def test_fep_slurm_passes_the_surrogate_flags_to_both_legs():
    script = (PACKAGE / "fep_edge.slurm").read_text()
    assert 'FEP_MACE="${FEP_MACE:-0}"' in script
    assert script.count('${mace_args[@]+"${mace_args[@]}"}') == 2


def test_fep_submitter_renders_the_flags_once():
    script = (PACKAGE / "submit_fep_edges.sh").read_text()
    assert "--with-mace" in script
    assert "export FEP_MACE MACE_ARGS" in script


# --------------------------------------------------------------------------- preflight


def test_preflight_environment_only_pass():
    from csbrt import mace_preflight

    assert mace_preflight.main(["--skip-model", "--device", "cpu"]) == 0


def test_preflight_flags_a_cuda_request_without_cuda_torch():
    from csbrt import mace_preflight

    report = mace_preflight.Report()
    mace_preflight.check_compute(report, "cuda")
    import torch

    if not torch.cuda.is_available():
        assert any(
            c["check"] == "device agreement" and c["status"] == "fail"
            for c in report.checks
        )


def test_preflight_writes_a_report(tmp_path):
    from csbrt import mace_preflight

    target = tmp_path / "preflight.json"
    mace_preflight.main(["--skip-model", "--device", "cpu", "--json", str(target)])
    payload = json.loads(target.read_text())
    assert payload["ok"] is True
    assert payload["config"]["enabled"] is True
    assert any(c["check"] == "geometry guard" for c in payload["checks"])
