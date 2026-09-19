"""Small checkpoint and validation helpers for the EV71 Loch pipeline."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
from pathlib import Path
from typing import Any


def require_file(path: Path) -> Path:
    path = path.resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return path


def resolve_scripts_dir(project: Path, sentinel: str) -> Path:
    """Locate the directory holding the stage scripts.

    In a source checkout they live in ``<project>/scripts/``. Once the package is
    pip-installed they sit beside this module in ``site-packages/csbrt/`` and
    ``<project>/scripts/`` does not exist -- ``Path(__file__).parents[1]`` then
    points at ``site-packages`` itself, so the naive ``project / "scripts"``
    resolves to a directory that was never installed.

    ``sentinel`` is a stage script the caller actually invokes, so an unrelated
    ``scripts/`` directory cannot satisfy the check.
    """
    candidate = project / "scripts"
    if (candidate / sentinel).is_file():
        return candidate
    packaged = Path(__file__).resolve().parent
    if (packaged / sentinel).is_file():
        return packaged
    raise FileNotFoundError(
        f"cannot locate {sentinel}: looked in {candidate} and {packaged}"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with require_file(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(paths: dict[str, Path]) -> dict[str, str]:
    """Hash versioned pipeline sources without baking checkout paths into markers."""
    return {name: sha256(path.resolve()) for name, path in sorted(paths.items())}


def implementation_signature(
    *,
    sources: dict[str, Path],
    distributions: tuple[str, ...] = (),
    modules: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Fingerprint pipeline code and the third-party implementation it executes.

    Module hashes matter for local validation overlays: a patched Loch module can
    retain the same package version, so version strings alone are insufficient.
    """
    versions: dict[str, str] = {}
    for name in sorted(set(distributions)):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"

    module_hashes: dict[str, str] = {}
    module_versions: dict[str, str] = {}
    for name in sorted(set(modules)):
        spec = importlib.util.find_spec(name)
        origin = None if spec is None else spec.origin
        if origin is None or not Path(origin).is_file():
            module_hashes[name] = "no-python-source"
        else:
            module_hashes[name] = sha256(Path(origin))
        root_name = name.split(".", 1)[0]
        if root_name not in module_versions:
            module = importlib.import_module(root_name)
            module_versions[root_name] = str(getattr(module, "__version__", "unknown"))

    return {
        "python": platform.python_version(),
        "source_sha256": source_hashes(sources),
        "dependency_versions": versions,
        "module_versions": module_versions,
        "module_sha256": module_hashes,
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(require_file(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def checkpoint_output_path(marker: Path, output: Path) -> tuple[str, Path]:
    """Return a marker-relative key and its resolved, contained output path.

    Checkpoints are intended to move with their stage directory.  Absolute
    output names silently made a copied run validate files in the original run,
    so every recorded artifact is now required to live beneath the marker.
    """
    root = marker.resolve().parent
    resolved = output.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"Checkpoint output {resolved} is outside marker directory {root}"
        ) from error
    if not relative.parts or ".." in relative.parts:
        raise ValueError(f"Unsafe checkpoint output path: {relative}")
    return relative.as_posix(), resolved


def output_hashes(marker: Path, outputs: list[Path]) -> dict[str, str]:
    return {
        key: sha256(resolved)
        for key, resolved in (checkpoint_output_path(marker, path) for path in outputs)
    }


def complete_checkpoint(
    marker: Path,
    *,
    signature: dict[str, Any],
    outputs: list[Path],
    details: dict[str, Any] | None = None,
) -> None:
    """Atomically publish a completed marker only after hashing every output."""
    payload: dict[str, Any] = {
        "status": "completed",
        "signature": signature,
        "output_sha256": output_hashes(marker, outputs),
    }
    if details:
        payload.update(details)
    write_json_atomic(marker, payload)


def invalidate_checkpoint(marker: Path) -> None:
    """Ensure a failed forced/rebuild attempt cannot leave an old valid marker."""
    marker.unlink(missing_ok=True)


def checkpoint_matches(
    marker: Path,
    *,
    signature: dict[str, Any],
    outputs: list[Path],
) -> bool:
    if not marker.is_file():
        return False
    try:
        payload = read_json(marker)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if payload.get("status") != "completed" or payload.get("signature") != signature:
        return False
    recorded = payload.get("output_sha256")
    if not isinstance(recorded, dict):
        return False
    expected_keys: set[str] = set()
    for path in outputs:
        try:
            key, resolved = checkpoint_output_path(marker, path)
        except ValueError:
            return False
        expected_keys.add(key)
        if not resolved.is_file() or resolved.stat().st_size == 0:
            return False
        if recorded.get(key) != sha256(resolved):
            return False
    return set(recorded) == expected_keys


def validate_recorded_outputs(marker: Path) -> dict[str, Any]:
    """Independently verify status and every output hash recorded by a marker."""
    payload = read_json(marker)
    if payload.get("status") != "completed":
        raise ValueError(f"Checkpoint is not completed: {marker}")
    recorded = payload.get("output_sha256")
    if not isinstance(recorded, dict) or not recorded:
        raise ValueError(f"Checkpoint has no output hashes: {marker}")
    root = marker.resolve().parent
    for filename, expected in recorded.items():
        relative = Path(filename)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError(f"Unsafe checkpoint output path in {marker}: {filename}")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"Checkpoint output escapes marker directory: {filename}"
            ) from error
        path = require_file(path)
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Checkpoint output hash mismatch: {path}")
    return payload


def finite_csv(
    path: Path,
    *,
    total_steps: int | None = None,
    report_interval: int | None = None,
) -> dict[str, Any]:
    import csv

    with require_file(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No data rows in {path}")
    for row_number, row in enumerate(rows, start=2):
        for key, value in row.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Non-numeric {key} at {path}:{row_number}") from error
            if not math.isfinite(numeric):
                raise ValueError(f"Non-finite {key} at {path}:{row_number}")
    steps = [int(row["step"]) for row in rows]
    if steps != sorted(steps) or len(steps) != len(set(steps)):
        raise ValueError(f"Steps are not strictly increasing in {path}")
    if (total_steps is None) != (report_interval is None):
        raise ValueError("total_steps and report_interval must be supplied together")
    if total_steps is not None and report_interval is not None:
        expected = list(range(report_interval, total_steps + 1, report_interval))
        if steps != expected:
            raise ValueError(
                f"CSV step schedule in {path} is {steps[:5]}...{steps[-5:]}; "
                f"expected {expected[:5]}...{expected[-5:]}"
            )
    digest = hashlib.sha256(",".join(str(step) for step in steps).encode()).hexdigest()
    return {
        "rows": len(rows),
        "first_step": steps[0],
        "last_step": steps[-1],
        "step_sequence_sha256": digest,
    }


def read_ghost_records(path: Path) -> list[list[int]]:
    records: list[list[int]] = []
    for line_number, line in enumerate(require_file(path).read_text().splitlines(), start=1):
        try:
            values = [int(value.strip()) for value in line.split(",") if value.strip()]
        except ValueError as error:
            raise ValueError(f"Invalid ghost index at {path}:{line_number}") from error
        if any(value < 0 for value in values):
            raise ValueError(f"Negative ghost index at {path}:{line_number}")
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate ghost index at {path}:{line_number}")
        records.append(values)
    if not records:
        raise ValueError(f"No ghost records in {path}")
    return records


def ghost_history(path: Path) -> dict[str, Any]:
    records = read_ghost_records(path)
    counts = [len(record) for record in records]
    return {
        "lines": len(records),
        "minimum_state_zero": min(counts),
        "maximum_state_zero": max(counts),
        "final_state_zero": counts[-1],
    }


# --------------------------------------------------------------------------- MLFF

# The MACE surrogate flags are identical across every stage script, and a stage
# whose flags drift from its neighbours silently runs different physics. They
# are declared once here and parsed once here.

MACE_ARGUMENT_NAMES = (
    "enable_mace_surrogate",
    "mace_model",
    "mace_model_path",
    "mace_committee",
    "mace_uq_threshold",
    "mace_uq_interval",
    "mace_fallback_steps",
    "mace_device",
    "mace_ml_waters",
    "mace_strict",
)


def add_mace_arguments(parser) -> None:
    """Register the MACE surrogate flags on a stage script's parser."""
    group = parser.add_argument_group("MACE ML/MM surrogate")
    group.add_argument(
        "--enable-mace-surrogate", "--mace-surrogate",
        action="store_true", dest="enable_mace_surrogate",
        help="Run the perturbable ligand on a MACE ML/MM surrogate with "
             "uncertainty-triggered fallback to classical physics",
    )
    group.add_argument(
        "--mace-model", default="mace-off23-small",
        help="openmm-ml foundation model name (default: mace-off23-small)",
    )
    group.add_argument(
        "--mace-model-path", default=None,
        help="Locally trained/fine-tuned .model file; overrides --mace-model",
    )
    group.add_argument(
        "--mace-committee", action="append", default=[], metavar="MODEL",
        help="Committee member for uncertainty quantification; repeat for "
             "each member. Two or more are needed for a force variance, and "
             "they must be fine-tunes of the same foundation model.",
    )
    group.add_argument(
        "--mace-uq-threshold", type=float, default=0.05,
        help="Committee force-variance trigger in eV/A (default: 0.05)",
    )
    group.add_argument(
        "--mace-uq-interval", type=int, default=100,
        help="MD steps between uncertainty evaluations (default: 100)",
    )
    group.add_argument(
        "--mace-fallback-steps", type=int, default=50,
        help="Classical steps to run after a trigger before retrying the "
             "surrogate (default: 50)",
    )
    group.add_argument("--mace-device", default="cuda",
                       help="Device for MACE inference (default: cuda)")
    group.add_argument(
        "--mace-ml-waters", action="store_true",
        help="Put binding-site waters in the ML region as well as the ligand. "
             "Off by default: the ML atom list is fixed for the life of the "
             "Context, so Loch cannot exchange an ML water.",
    )
    group.add_argument(
        "--mace-strict", action="store_true",
        help="Fail the stage if the surrogate cannot be built, instead of "
             "continuing on classical physics",
    )


def mace_options_dict(options) -> dict[str, Any]:
    """Collect the MACE flags off a parsed namespace as an mlff config mapping."""
    return {
        "enabled": bool(getattr(options, "enable_mace_surrogate", False)),
        "model_name": getattr(options, "mace_model", "mace-off23-small"),
        "model_path": getattr(options, "mace_model_path", None),
        "committee_model_paths": list(getattr(options, "mace_committee", []) or []),
        "uq_force_threshold_ev_per_ang": getattr(options, "mace_uq_threshold", 0.05),
        "uq_interval_steps": getattr(options, "mace_uq_interval", 100),
        "fallback_steps": getattr(options, "mace_fallback_steps", 50),
        "device": getattr(options, "mace_device", "cuda"),
        "include_binding_site_waters": bool(getattr(options, "mace_ml_waters", False)),
        "strict": bool(getattr(options, "mace_strict", False)),
    }


def mace_config_from_options(options, *, ligand_resname: str | None = None):
    """Build a ``MACEConfig`` from a stage script's parsed arguments."""
    from csbrt.mace_surrogate import MACEConfig

    payload = mace_options_dict(options)
    if ligand_resname is not None:
        payload["ligand_resname"] = ligand_resname
    return MACEConfig.from_dict(payload)


def mace_command_arguments(mlff: dict[str, Any] | None) -> list[str]:
    """Render an ``mlff:`` config block back into stage-script flags."""
    if not mlff or not mlff.get("enabled"):
        return []
    arguments: list[str] = ["--enable-mace-surrogate"]
    simple = {
        "model_name": "--mace-model",
        "model_path": "--mace-model-path",
        "uq_force_threshold_ev_per_ang": "--mace-uq-threshold",
        "uq_interval_steps": "--mace-uq-interval",
        "fallback_steps": "--mace-fallback-steps",
        "device": "--mace-device",
    }
    for key, flag in simple.items():
        value = mlff.get(key)
        if value is not None:
            arguments.extend([flag, str(value)])
    for member in mlff.get("committee_model_paths") or []:
        arguments.extend(["--mace-committee", str(member)])
    if mlff.get("include_binding_site_waters"):
        arguments.append("--mace-ml-waters")
    if mlff.get("strict"):
        arguments.append("--mace-strict")
    return arguments


def mace_signature(config: "dict[str, Any] | Any | None") -> dict[str, Any] | None:
    """Fingerprint the MACE surrogate settings for a checkpoint marker.

    Accepts either an ``mlff`` mapping or a ``MACEConfig``. Returns ``None``
    when the surrogate is off, so a classical run's marker is byte-identical to
    one produced before this package existed and existing checkpoints stay
    valid.
    """
    from csbrt.mace_surrogate import MACEConfig

    if config is None:
        return None
    if isinstance(config, MACEConfig):
        resolved = config
    else:
        if not config.get("enabled", False):
            return None
        resolved = MACEConfig.from_dict(config)
    if not resolved.enabled:
        return None
    return resolved.compute_signature()

