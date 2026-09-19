"""One object that wires the surrogate into a running simulation.

The GCMC driver and the FEP leg hook both want the same five things -- a mixed
ML/MM Context, a loaded committee, a UQ monitor, a Hamiltonian switch and an
OOD buffer -- built in the right order and torn down with their statistics
written out. :class:`MACERuntime` is that assembly, so neither caller has to
know how the pieces fit.

The contract with the callers is deliberately narrow:

    runtime = MACERuntime.attach(context, topology, config, ...)   # or None
    ...
    runtime.check(context)        # evaluate UQ, switch Hamiltonian if needed
    runtime.advance(n_steps)      # account for MD that has been run
    runtime.finish()              # persist the OOD buffer, return statistics

``attach`` returns ``None`` rather than raising when the surrogate cannot be
built and ``config.strict`` is off. That is the safe direction: the run
continues on classical physics, which is the answer the pipeline was producing
before this package existed. It is logged as an error, not a warning, and the
stage's checkpoint records that the surrogate was not active, so a degraded run
can never be mistaken for an accelerated one.
"""

from __future__ import annotations

import logging
from pathlib import Path
import socket
import os
from typing import Any, Sequence

import numpy as np

from .active_learner import OODBuffer
from .committee import CommitteeUnavailable, MACECommittee
from .config import MACEConfig
from .fallback_controller import (
    DualHamiltonianSwitch,
    EvaluatorMode,
    PhysicsFallbackController,
)
from .mace_mixed_system import (
    MACESurrogateError,
    MLRegion,
    attach_mace_to_context,
    describe_ml_region,
    noninteracting_particles,
    partition_ml_atoms,
)
from .uq_monitor import GeometryGuard, MACEUQMonitor, UQResult

logger = logging.getLogger("csbrt.mace_surrogate.runtime")

NM_TO_ANGSTROM = 10.0


def worker_tag() -> str:
    """A label unique to this process, for naming per-worker OOD buffers."""
    array = os.environ.get("SLURM_ARRAY_TASK_ID")
    job = os.environ.get("SLURM_JOB_ID")
    if job and array:
        return f"slurm{job}_{array}"
    if job:
        return f"slurm{job}"
    return f"{socket.gethostname().split('.')[0]}_{os.getpid()}"


def md_chunks(
    num_steps: int,
    completed_steps: int,
    report_interval: int,
    uq_interval: int | None = None,
):
    """Split an MD run into chunks that land exactly on report boundaries.

    Yields the step counts to hand the integrator. With ``uq_interval`` set the
    chunks are further subdivided so uncertainty can be evaluated between them,
    but the points at which ``completed_steps`` becomes a multiple of
    ``report_interval`` are unchanged. That invariant is what lets the GCMC
    driver interleave UQ checks without altering the CSV step schedule its
    checkpoints validate against.
    """
    if num_steps < 0 or completed_steps < 0:
        raise ValueError("MD step counts cannot be negative")
    if report_interval < 1:
        raise ValueError("Report interval must be positive")
    if uq_interval is not None and uq_interval < 1:
        raise ValueError("UQ interval must be positive")

    remaining = num_steps
    done = completed_steps
    while remaining:
        until_report = report_interval - (done % report_interval)
        chunk = min(remaining, until_report)
        if uq_interval is not None:
            chunk = min(chunk, uq_interval)
        yield chunk
        done += chunk
        remaining -= chunk


def ml_region_bonds(topology: Any, ml_atoms: Sequence[int]) -> list[tuple[int, int]]:
    """Bonds wholly inside the ML region, re-indexed to ML-local positions."""
    local = {int(atom): index for index, atom in enumerate(sorted(ml_atoms))}
    bonds: list[tuple[int, int]] = []
    for bond in topology.bonds():
        first = int(bond[0].index)
        second = int(bond[1].index)
        if first in local and second in local:
            bonds.append((local[first], local[second]))
    return bonds


def ml_coordinates(
    context: Any, ml_atoms: Sequence[int]
) -> tuple[np.ndarray, np.ndarray | None]:
    """ML-region positions in Angstroms plus the periodic box, from a Context."""
    import openmm.unit as unit

    state = context.getState(getPositions=True)
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    coords = np.asarray(positions, dtype=np.float64)[list(ml_atoms)] * NM_TO_ANGSTROM
    box = None
    try:
        vectors = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(
            unit.nanometer
        )
        box = np.asarray(vectors, dtype=np.float64) * NM_TO_ANGSTROM
    except Exception:  # non-periodic Context
        box = None
    return coords, box


class MACERuntime:
    """Live MACE surrogate bound to one OpenMM Context."""

    def __init__(
        self,
        config: MACEConfig,
        region: MLRegion,
        controller: PhysicsFallbackController,
        ood_buffer: OODBuffer | None,
        handle: Any,
        source: str,
    ) -> None:
        self.config = config
        self.region = region
        self.controller = controller
        self.ood_buffer = ood_buffer
        self.handle = handle
        self.source = source
        self._last_result: UQResult | None = None

    # --- construction --------------------------------------------------------

    @classmethod
    def attach(
        cls,
        context: Any,
        topology: Any,
        config: MACEConfig,
        *,
        output_dir: Path | str,
        source: str = "",
        ml_atoms: Sequence[int] | None = None,
    ) -> "MACERuntime | None":
        """Build and attach the surrogate, or return ``None`` and run classical."""
        if not config.enabled:
            return None
        root = Path(output_dir)
        config = config.resolve_paths(root)
        try:
            return cls._attach(
                context, topology, config, root, source, ml_atoms
            )
        except (MACESurrogateError, CommitteeUnavailable, ImportError) as error:
            if config.strict:
                raise
            logger.error(
                "MACE surrogate disabled for this run: %s. Continuing on "
                "classical physics -- the sampling is correct, the speedup is "
                "not being realised.",
                error,
            )
            return None

    @classmethod
    def _attach(
        cls,
        context: Any,
        topology: Any,
        config: MACEConfig,
        root: Path,
        source: str,
        ml_atoms: Sequence[int] | None,
    ) -> "MACERuntime":
        live = context.getSystem()
        if ml_atoms is None:
            state = context.getState(getPositions=True)
            ml_atoms = partition_ml_atoms(
                topology,
                positions=state.getPositions(asNumpy=True),
                ligand_resname=config.ligand_resname,
                include_waters=config.include_binding_site_waters,
                sphere_radius_nm=config.binding_site_radius_nm,
                exclude=noninteracting_particles(live),
            )
        region = describe_ml_region(topology, ml_atoms)
        if config.include_binding_site_waters and region.water_atom_count:
            logger.warning(
                "%d water atoms are in the ML region. The ML atom list is fixed "
                "for the life of the Context, so these specific waters stay in "
                "the ML region even after they diffuse out of the binding site, "
                "and Loch cannot exchange them. Only do this for a run where "
                "the site waters are known to be buried.",
                region.water_atom_count,
            )

        handle = attach_mace_to_context(context, topology, config, region.atom_indices)

        committee = None
        if config.uq_enabled and config.has_committee:
            committee = MACECommittee(
                config.committee_model_paths,
                device=config.device,
                precision=config.precision,
            )
        elif config.uq_enabled:
            logger.warning(
                "No MACE committee configured (mlff.committee_model_paths has "
                "%d entry/entries, 2 are needed for a force variance). Running "
                "with the geometry guard as the only OOD detector; fine-tune a "
                "committee with csbrt-mace-al to get the full signal.",
                config.committee_size,
            )

        guard = None
        if config.geometry_guard:
            guard = GeometryGuard(
                region.atomic_numbers,
                min_distance_ang=config.geometry_min_distance_ang,
                max_bond_scale=config.geometry_max_bond_scale,
                bonds=ml_region_bonds(topology, region.atom_indices),
            )

        monitor = MACEUQMonitor(config, committee=committee, geometry_guard=guard)
        switch = DualHamiltonianSwitch(
            context,
            parameter_name=handle.parameter_name,
            surrogate_value=handle.surrogate_value,
            classical_value=handle.classical_value,
            require_parameter=config.interpolate,
        )

        buffer = None
        if config.active_learning and config.ood_buffer_dir:
            tag = source or worker_tag()
            buffer = OODBuffer(
                Path(config.ood_buffer_dir) / f"ood_{tag}.npz",
                max_frames=config.ood_max_frames,
                source=tag,
            )

        runtime = cls(
            config=config,
            region=region,
            controller=PhysicsFallbackController(
                config=config, uq_monitor=monitor, switch=switch
            ),
            ood_buffer=buffer,
            handle=handle,
            source=source or worker_tag(),
        )
        runtime.controller.on_ood_frame = runtime._store_ood_frame
        logger.info(
            "MACE surrogate live: %d ML atoms, committee=%d, geometry_guard=%s, "
            "UQ every %d steps, fallback %d steps",
            len(region),
            committee.size if committee else 0,
            bool(guard),
            config.uq_interval_steps,
            config.fallback_steps,
        )
        return runtime

    # --- runtime -------------------------------------------------------------

    def check(self, context: Any) -> EvaluatorMode:
        """Evaluate uncertainty on the Context's current frame and switch."""
        coords, box = ml_coordinates(context, self.region.atom_indices)
        result = self.controller.uq_monitor.evaluate(
            coords, self.region.atomic_numbers, box_ang=box
        )
        self._last_result = result
        frame = {"positions": coords, "box_vectors": box}
        return self.controller.observe(result, frame)

    def advance(self, steps: int) -> None:
        self.controller.advance(steps)

    @property
    def mode(self) -> EvaluatorMode:
        return self.controller.state.current_mode

    @property
    def last_result(self) -> UQResult | None:
        return self._last_result

    def _store_ood_frame(self, payload: dict[str, Any]) -> None:
        if self.ood_buffer is None:
            return
        uq = payload.get("uq_result", {})
        self.ood_buffer.add_from_raw(
            step=int(payload.get("md_step", 0)),
            positions=payload["positions"],
            atomic_numbers=self.region.atomic_numbers,
            uncertainty_force=float(uq.get("sigma_f_max_ev_per_ang", 0.0)),
            uncertainty_energy=float(uq.get("sigma_e_kcal_per_mol", 0.0)),
            reason=str(payload.get("reason") or "out of distribution"),
            box_vectors=payload.get("box_vectors"),
            ml_atoms=self.region.atom_indices,
            source=self.source,
        )

    # --- teardown ------------------------------------------------------------

    def finish(self) -> dict[str, Any]:
        """Persist the OOD buffer and return everything worth checkpointing."""
        buffer_path = None
        if self.ood_buffer is not None and len(self.ood_buffer):
            buffer_path = str(self.ood_buffer.save())
        statistics = self.controller.statistics()
        statistics["ml_region"] = self.region.to_dict()
        statistics["mixed_system"] = self.handle.to_dict()
        statistics["ood_buffer"] = {
            "path": buffer_path,
            "frames": len(self.ood_buffer) if self.ood_buffer else 0,
            "dropped": self.ood_buffer.dropped if self.ood_buffer else 0,
        }
        statistics["config_signature"] = self.config.compute_signature()
        return statistics
