"""MACE surrogate plumbing for the csbrt stage scripts.

Deliberately a module of its own rather than additions to ``pipeline_utils``
or ``ev71_loch_common``. Those two are hashed into the
``implementation_signature`` of every stage's ``.complete.json`` marker --
``pipeline_utils`` into all of them -- so adding a function to either
invalidates every completed equilibration, production, density and GCI
checkpoint in an existing campaign, whether or not the surrogate is ever
enabled. A campaign's finished work should not have to be repeated to gain an
optional feature it is not using.

The cost of that isolation is one indirection: the MD loop here delegates to
``ev71_loch_common.run_with_csv_reports`` rather than reimplementing it, so
there is still exactly one piece of code that decides when the CSV reporter
fires.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- MLFF

# The MACE surrogate flags are identical across every stage script, and a stage
# whose flags drift from its neighbours silently runs different physics. They
# are declared once here and parsed once here.

def add_mace_arguments(parser) -> None:
    """Register the MACE surrogate flags on a stage script's parser.

    Every option defaults to ``None`` rather than to a value. The defaults live
    in ``MACEConfig`` and nowhere else: a default repeated in argparse is a
    default that drifts, and the whole point of declaring these once is that a
    52-edge network cannot end up running two versions of the physics.
    """
    group = parser.add_argument_group("MACE ML/MM surrogate")
    group.add_argument(
        "--enable-mace-surrogate", "--mace-surrogate",
        action="store_true", dest="enable_mace_surrogate",
        help="Run the perturbable ligand on a MACE ML/MM surrogate with "
             "uncertainty-triggered fallback to classical physics",
    )
    group.add_argument(
        "--mace-config", default=None, metavar="JSON_OR_PATH",
        help="The whole mlff settings block as JSON, or a path to a JSON file. "
             "This is how the csbrt driver forwards a config's mlff: block "
             "intact; the individual flags below override whatever it carries.",
    )
    group.add_argument(
        "--mace-model", default=None,
        help="openmm-ml foundation model name (default: mace-off23-small)",
    )
    group.add_argument(
        "--mace-model-path", default=None,
        help="Locally trained/fine-tuned .model file; overrides --mace-model",
    )
    group.add_argument(
        "--mace-committee", action="append", default=None, metavar="MODEL",
        help="Committee member for uncertainty quantification; repeat for "
             "each member. Two or more are needed for a force variance, and "
             "they must be fine-tunes of the same foundation model.",
    )
    group.add_argument(
        "--mace-uq-threshold", type=float, default=None,
        help="Committee force-variance trigger in eV/A (default: 0.05)",
    )
    group.add_argument(
        "--mace-uq-interval", type=int, default=None,
        help="MD steps between uncertainty evaluations (default: 100)",
    )
    group.add_argument(
        "--mace-fallback-steps", type=int, default=None,
        help="Classical steps to run after a trigger before retrying the "
             "surrogate (default: 50)",
    )
    group.add_argument("--mace-device", default=None,
                       help="Device for MACE inference (default: cuda)")
    group.add_argument(
        "--mace-ml-waters", action="store_true", default=None,
        help="Put binding-site waters in the ML region as well as the ligand. "
             "Off by default: the ML atom list is fixed for the life of the "
             "Context, so Loch cannot exchange an ML water.",
    )
    group.add_argument(
        "--mace-strict", action="store_true", default=None,
        help="Fail the stage if the surrogate cannot be built, instead of "
             "continuing on classical physics",
    )


#: CLI attribute -> key in the mlff settings block.
MACE_FLAG_KEYS = {
    "mace_model": "model_name",
    "mace_model_path": "model_path",
    "mace_committee": "committee_model_paths",
    "mace_uq_threshold": "uq_force_threshold_ev_per_ang",
    "mace_uq_interval": "uq_interval_steps",
    "mace_fallback_steps": "fallback_steps",
    "mace_device": "device",
    "mace_ml_waters": "include_binding_site_waters",
    "mace_strict": "strict",
}


def _load_mace_config_blob(value: str | None) -> dict[str, Any]:
    """Parse ``--mace-config``: inline JSON, or a path to a JSON file.

    A serialised settings block is longer than any filesystem's name limit, so
    ``Path(value).is_file()`` on one raises rather than returning False. Decide
    on the leading character first, and treat any remaining filesystem error as
    "not a path" rather than letting it reach the user as an ENAMETOOLONG.
    """
    if not value:
        return {}
    text = value
    if not value.lstrip().startswith(("{", "[")):
        try:
            candidate = Path(value)
            if candidate.is_file():
                text = candidate.read_text()
        except OSError:
            pass
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"--mace-config is neither a readable file nor valid JSON: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("--mace-config must be a JSON object")
    return payload


def mace_options_dict(options) -> dict[str, Any]:
    """Collect the MACE flags off a parsed namespace as an mlff settings block.

    The ``--mace-config`` blob is the base and the individual flags override
    it, so a config file's ``mlff:`` block survives being forwarded through a
    stage script even for settings that have no dedicated flag.
    """
    payload = _load_mace_config_blob(getattr(options, "mace_config", None))
    payload["enabled"] = bool(
        getattr(options, "enable_mace_surrogate", False) or payload.get("enabled")
    )
    for attribute, key in MACE_FLAG_KEYS.items():
        value = getattr(options, attribute, None)
        if value is not None:
            payload[key] = value
    return payload


def mace_config_from_options(options, *, ligand_resname: str | None = None):
    """Build a ``MACEConfig`` from a stage script's parsed arguments."""
    from csbrt.mace_surrogate import MACEConfig

    payload = mace_options_dict(options)
    if ligand_resname is not None:
        payload["ligand_resname"] = ligand_resname
    return MACEConfig.from_dict(payload)


def mace_command_arguments(mlff: dict[str, Any] | None) -> list[str]:
    """Render an ``mlff:`` settings block into stage-script flags.

    The whole block goes through as one JSON argument rather than as a flag per
    key. Settings such as ``precision``, ``max_ml_atoms`` or
    ``fallback_abort_fraction`` have no dedicated flag, and rendering only the
    ones that do would silently drop them somewhere between the driver and the
    stage that runs the physics.
    """
    if not mlff or not mlff.get("enabled"):
        return []
    payload = dict(mlff)
    payload["enabled"] = True
    return ["--enable-mace-surrogate", "--mace-config", json.dumps(payload, sort_keys=True)]


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


# --------------------------------------------------------------------------- Loch


def attach_mace_surrogate(
    dynamics,
    topology,
    mace_config,
    *,
    output_dir,
    source: str = "",
):
    """Turn a Loch MD context into a hybrid MACE ML/MM context.

    Must be called *before* ``sampler.bind_dynamics(dynamics)``. Attaching
    replaces the Forces in the live System (see
    ``mace_surrogate.attach_mace_to_context``), and Loch resolves the
    NonbondedForce it toggles ghost waters through when it binds. Binding first
    would leave it holding a Force that is no longer in the System, and the
    ghost bookkeeping would break silently.

    Two properties are checked afterwards and the run aborts if either fails,
    because a wrong GCMC acceptance is worse than no speedup:

    * every particle Loch sees as a ghost (zero charge and epsilon) is still a
      ghost, and no new ones appeared;
    * exactly one NonbondedForce remains, which is what Loch expects.

    Returns a ``MACERuntime`` or ``None``; ``None`` means the run continues on
    classical physics.
    """
    if mace_config is None or not getattr(mace_config, "enabled", False):
        return None

    import openmm

    from csbrt.mace_surrogate import MACERuntime, noninteracting_particles

    context = dynamics.context()
    system = context.getSystem()
    ghosts_before = noninteracting_particles(system)
    nonbonded_before = sum(
        1 for force in system.getForces() if isinstance(force, openmm.NonbondedForce)
    )

    runtime = MACERuntime.attach(
        context, topology, mace_config, output_dir=output_dir, source=source
    )
    if runtime is None:
        return None

    ghosts_after = noninteracting_particles(system)
    nonbonded_after = sum(
        1 for force in system.getForces() if isinstance(force, openmm.NonbondedForce)
    )
    problems = []
    if ghosts_after != ghosts_before:
        problems.append(
            f"ghost-water set changed across the mixed-system swap: "
            f"{len(ghosts_before)} -> {len(ghosts_after)} non-interacting particles"
        )
    if nonbonded_after != nonbonded_before or nonbonded_after != 1:
        problems.append(
            f"NonbondedForce count is {nonbonded_after} (was {nonbonded_before}); "
            "Loch toggles ghost waters through exactly one"
        )
    if problems:
        raise RuntimeError(
            "MACE surrogate would break Loch's GCMC bookkeeping: "
            + "; ".join(problems)
        )

    print(
        f"MACE surrogate attached to Loch dynamics: "
        f"{len(runtime.region)} ML atoms, ghosts preserved "
        f"({len(ghosts_after)} non-interacting particles)",
        flush=True,
    )
    return runtime


def run_with_surrogate(
    dynamics,
    context,
    num_steps: int,
    completed_steps: int,
    report_interval: int,
    writer,
    surrogate,
) -> int:
    """Run MD with uncertainty checks interleaved, reporting as usual.

    ``md_chunks`` subdivides the run at the UQ interval while still landing
    exactly on the report boundaries, so each chunk handed to
    ``run_with_csv_reports`` is one of that function's own iterations and the
    CSV step schedule is identical to a classical run's. Existing production
    checkpoints validate against that schedule.
    """
    from csbrt.mace_surrogate import md_chunks

    from ev71_loch_common import run_with_csv_reports

    if surrogate is None:
        return run_with_csv_reports(
            dynamics, context, num_steps, completed_steps, report_interval, writer
        )
    for chunk in md_chunks(
        num_steps, completed_steps, report_interval,
        surrogate.config.uq_interval_steps,
    ):
        surrogate.check(context)
        completed_steps = run_with_csv_reports(
            dynamics, context, chunk, completed_steps, report_interval, writer
        )
        surrogate.advance(chunk)
    return completed_steps
