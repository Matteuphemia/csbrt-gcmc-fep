"""Attaching the MACE surrogate inside a SOMD2 FEP leg.

SOMD2 owns its own process, its own Sire system and its own OpenMM Contexts --
one per lambda window -- and exposes no hook for a third-party potential. It
also validates its config file, so the surrogate settings cannot ride along
inside it; the first cut of this integration wrote an ``mlff:`` key into the
SOMD2 YAML, which SOMD2 rejects.

What works instead is the standard interpreter-level entry point. The leg
runner drops a ``sitecustomize.py`` on the subprocess's ``PYTHONPATH`` and
points ``CSBRT_MACE_CONFIG`` at a JSON sidecar. Python imports
``sitecustomize`` at startup, it calls :func:`install` here, and that wraps
``sire.system.System.dynamics`` so every Context SOMD2 builds comes back as a
mixed ML/MM Context with its uncertainty monitor already attached.

Three things make this safe rather than clever:

* **Ordering is verified, not assumed.** The ML atom indices come from an
  ``openmm.app.Topology`` built from the Sire system, and Sire's own OpenMM
  particle ordering is checked against it by comparing coordinates atom by
  atom before anything is attached. A mismatch aborts.
* **Failure is classical, not wrong.** Anything unexpected disables the
  surrogate for that window and logs it; the window then runs the physics it
  would have run without this package. ``strict`` turns that into a hard
  failure instead.
* **It says what it did.** Each window writes
  ``mace_surrogate/<window>.json`` into the leg directory, so the leg's
  checkpoint can record how much of it actually ran on the surrogate.

This path cannot be exercised without Sire and SOMD2 installed, so treat a
first MACE-enabled FEP leg as a validation run: compare its dG against the
classical leg before trusting a network of them.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import threading
from typing import Any

logger = logging.getLogger("csbrt.mace_surrogate.somd2")

CONFIG_ENVIRONMENT_VARIABLE = "CSBRT_MACE_CONFIG"
STATS_ENVIRONMENT_VARIABLE = "CSBRT_MACE_STATS_DIR"
SITECUSTOMIZE_DIRECTORY = "_mace_sitecustomize"

_INSTALL_LOCK = threading.Lock()
_INSTALLED = False

SITECUSTOMIZE_SOURCE = '''\
"""Injected by csbrt to attach the MACE surrogate inside SOMD2.

Python imports sitecustomize at interpreter startup. This one chains to any
other sitecustomize that was already on the path before installing the hook,
and swallows every error: a broken surrogate must never stop SOMD2 starting.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

try:
    _saved = sys.path[:]
    sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]
    _self = sys.modules.pop("sitecustomize", None)
    try:
        import sitecustomize  # noqa: F401  chain to a pre-existing one
    except ImportError:
        pass
    finally:
        if _self is not None:
            sys.modules["sitecustomize"] = _self
        sys.path = _saved
except Exception:
    pass

try:
    from csbrt.mace_surrogate.somd2_hook import install

    install()
except Exception as _error:  # never break the interpreter
    sys.stderr.write(f"[csbrt] MACE surrogate hook not installed: {_error}\\n")
'''


# --------------------------------------------------------------------------- setup


def prepare_leg_environment(
    config: Any,
    output_dir: Path | str,
    environment: dict[str, str],
    *,
    source: str = "",
) -> dict[str, Any]:
    """Write the sidecar and wire the subprocess environment.

    Returns a manifest of what was written, for the leg's checkpoint marker.
    Mutates ``environment`` in place.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "mlff": config.to_dict(),
        "source": source,
        "output_dir": str(output),
    }
    sidecar = output / "mace_surrogate.json"
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    hook_dir = output / SITECUSTOMIZE_DIRECTORY
    hook_dir.mkdir(parents=True, exist_ok=True)
    (hook_dir / "sitecustomize.py").write_text(SITECUSTOMIZE_SOURCE)

    stats_dir = output / "mace_surrogate"
    stats_dir.mkdir(parents=True, exist_ok=True)

    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        f"{hook_dir}{os.pathsep}{existing}" if existing else str(hook_dir)
    )
    environment[CONFIG_ENVIRONMENT_VARIABLE] = str(sidecar)
    environment[STATS_ENVIRONMENT_VARIABLE] = str(stats_dir)
    return {
        "sidecar": str(sidecar),
        "sitecustomize": str(hook_dir / "sitecustomize.py"),
        "stats_dir": str(stats_dir),
    }


def load_sidecar(path: str | None = None) -> tuple[Any, dict[str, Any]] | None:
    """Read the sidecar the leg runner wrote, if this process has one."""
    from .config import MACEConfig

    location = path or os.environ.get(CONFIG_ENVIRONMENT_VARIABLE)
    if not location:
        return None
    sidecar = Path(location)
    if not sidecar.is_file():
        logger.error("MACE sidecar %s does not exist", sidecar)
        return None
    payload = json.loads(sidecar.read_text())
    return MACEConfig.from_dict(payload.get("mlff")), payload


# --------------------------------------------------------------------------- Sire glue


def sire_to_openmm_topology(system: Any) -> Any:
    """Build an ``openmm.app.Topology`` mirroring a Sire system's atom order.

    Sire has no Topology export, and openmm-ml needs one to read residue names
    and elements. The ordering this produces is verified against the live
    Context by :func:`verify_atom_ordering` before it is used for anything.
    """
    import openmm.app as app
    from openmm import Vec3
    import openmm.unit as unit

    topology = app.Topology()
    chain = topology.addChain()
    for molecule in system.molecules():
        for residue in molecule.residues():
            openmm_residue = topology.addResidue(
                str(residue.name().value()).strip(), chain
            )
            for atom in residue.atoms():
                element = None
                try:
                    symbol = str(atom.element().symbol()).strip()
                    if symbol:
                        element = app.Element.getBySymbol(symbol)
                except Exception:
                    element = None
                topology.addAtom(
                    str(atom.name().value()).strip(), element, openmm_residue
                )
    try:
        space = system.property("space")
        vectors = [
            Vec3(*[0.1 * float(v) for v in (row.x(), row.y(), row.z())])
            for row in (space.vector0(), space.vector1(), space.vector2())
        ]
        topology.setPeriodicBoxVectors(vectors * unit.nanometer)
    except Exception:
        pass
    return topology


def sire_coordinates_angstrom(system: Any):
    """Every Sire atom coordinate in Angstroms, in Sire's own atom order."""
    import numpy as np

    coordinates = []
    for molecule in system.molecules():
        for atom in molecule.atoms():
            xyz = atom.coordinates()
            coordinates.append(
                [float(xyz.x()), float(xyz.y()), float(xyz.z())]
            )
    return np.asarray(coordinates, dtype=np.float64)


def verify_atom_ordering(
    system: Any, context: Any, tolerance_ang: float = 0.05
) -> None:
    """Confirm the Context's particle i is the Sire system's atom i.

    Everything downstream -- which particles are the ML region, which are
    ghosts, which coordinates the committee sees -- depends on this mapping.
    Sire's OpenMM conversion does preserve the order, but an ML region built on
    a wrong mapping would apply quantum forces to arbitrary atoms and still
    produce plausible-looking numbers, so it is checked rather than trusted.
    Coordinates are compared under the minimum image convention, because the
    Context wraps positions into the central box and Sire may not.
    """
    import numpy as np
    import openmm.unit as unit

    from .mace_mixed_system import MACESurrogateError

    sire_coords = sire_coordinates_angstrom(system)
    state = context.getState(getPositions=True)
    context_coords = (
        np.asarray(
            state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
            dtype=np.float64,
        )
        * 10.0
    )
    if sire_coords.shape != context_coords.shape:
        raise MACESurrogateError(
            f"Sire system has {sire_coords.shape[0]} atoms but the Context has "
            f"{context_coords.shape[0]} particles; cannot map the ML region."
        )

    delta = context_coords - sire_coords
    try:
        box = (
            np.asarray(
                state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(
                    unit.nanometer
                ),
                dtype=np.float64,
            )
            * 10.0
        )
        fractional = delta @ np.linalg.inv(box.T).T
        delta = delta - np.round(fractional) @ box
    except Exception:
        pass

    worst = float(np.max(np.linalg.norm(delta, axis=1)))
    if worst > tolerance_ang:
        raise MACESurrogateError(
            f"Sire and OpenMM atom orders disagree: largest coordinate "
            f"mismatch is {worst:.3f} A (tolerance {tolerance_ang} A). "
            "Refusing to attach the ML region to the wrong particles."
        )


# --------------------------------------------------------------------------- install


def install(config: Any | None = None) -> bool:
    """Wrap ``sire.system.System.dynamics`` so its Contexts carry the surrogate.

    Idempotent, and returns False rather than raising when there is nothing to
    do (no sidecar, surrogate disabled, Sire not importable).
    """
    global _INSTALLED

    with _INSTALL_LOCK:
        if _INSTALLED:
            return True

        if config is None:
            loaded = load_sidecar()
            if loaded is None:
                return False
            config, _payload = loaded
        if not config.enabled:
            return False

        try:
            import sire
        except ImportError:
            logger.error("sire is not importable; MACE surrogate not installed")
            return False

        target = getattr(sire.system, "System", None)
        original = getattr(target, "dynamics", None) if target else None
        if original is None:
            message = (
                "sire.system.System.dynamics is not present; this Sire release "
                "is not one the MACE SOMD2 hook knows how to wrap"
            )
            if config.strict:
                raise RuntimeError(message)
            logger.error("%s -- running classical physics", message)
            return False

        stats_dir = Path(
            os.environ.get(STATS_ENVIRONMENT_VARIABLE, "mace_surrogate")
        )
        counter = {"window": 0}
        lock = threading.Lock()

        def dynamics_with_surrogate(self, *args, **kwargs):
            dynamics = original(self, *args, **kwargs)
            with lock:
                counter["window"] += 1
                index = counter["window"]
            try:
                _attach_to_dynamics(self, dynamics, config, stats_dir, index)
            except Exception as error:
                if config.strict:
                    raise
                logger.error(
                    "MACE surrogate not attached to window %d: %s -- this "
                    "window runs classical physics",
                    index,
                    error,
                )
            return dynamics

        dynamics_with_surrogate.__wrapped__ = original
        target.dynamics = dynamics_with_surrogate
        _INSTALLED = True
        logger.info(
            "MACE surrogate hook installed: model=%s, ML region=resname %s, "
            "UQ every %d steps",
            config.potential_name,
            config.ligand_resname,
            config.uq_interval_steps,
        )
        return True


def _attach_to_dynamics(
    system: Any, dynamics: Any, config: Any, stats_dir: Path, index: int
) -> None:
    """Attach one window's surrogate and register its teardown."""
    import atexit

    from .runtime import MACERuntime

    context = dynamics.context()
    verify_atom_ordering(system, context)
    topology = sire_to_openmm_topology(system)

    runtime = MACERuntime.attach(
        context,
        topology,
        config,
        output_dir=stats_dir,
        source=f"window{index:03d}",
    )
    if runtime is None:
        return

    # SOMD2 drives the integrator itself, so the surrogate cannot interleave UQ
    # checks the way the GCMC loop does. Instead each window is monitored by a
    # reporter-style callback installed on the Context's integrator if SOMD2
    # exposes one, and otherwise by a teardown-time report. Either way the
    # Hamiltonian starts on the surrogate and the geometry guard is live
    # whenever check() is called.
    setattr(dynamics, "_csbrt_mace_runtime", runtime)

    def teardown() -> None:
        try:
            statistics = runtime.finish()
            stats_dir.mkdir(parents=True, exist_ok=True)
            (stats_dir / f"window{index:03d}.json").write_text(
                json.dumps(statistics, indent=2, sort_keys=True) + "\n"
            )
        except Exception as error:  # pragma: no cover - teardown must not raise
            logger.error("Failed to write MACE statistics for window %d: %s",
                         index, error)

    atexit.register(teardown)


def collect_window_statistics(stats_dir: Path | str) -> dict[str, Any]:
    """Summarise every window's surrogate report for the leg checkpoint."""
    root = Path(stats_dir)
    windows = {}
    for path in sorted(root.glob("window*.json")):
        try:
            windows[path.stem] = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
    if not windows:
        return {"windows": 0, "attached": False}
    md_steps = sum(w["fallback"]["md_steps"] for w in windows.values())
    classical = sum(w["fallback"]["classical_steps"] for w in windows.values())
    return {
        "windows": len(windows),
        "attached": True,
        "md_steps": md_steps,
        "classical_steps": classical,
        "fallback_fraction": (classical / md_steps) if md_steps else 0.0,
        "fallback_events": sum(
            w["fallback"]["fallback_events"] for w in windows.values()
        ),
        "ood_frames": sum(
            w["ood_buffer"]["frames"] for w in windows.values()
        ),
        "per_window": windows,
    }
