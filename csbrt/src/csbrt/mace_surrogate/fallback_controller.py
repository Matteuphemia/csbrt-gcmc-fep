"""Physics Fallback and Context Handoff Engine.

Governs runtime switching between fast MACE MLFF surrogate and SOMD2/classical MM physics:
$$U_{\\text{effective}} = (1 - w) U_{\\text{MACE/MM}} + w U_{\\text{SOMD2/MM}}$$
where $w = 0$ is the fast surrogate path and $w = 1$ is the classical physics fallback.

Maintains trajectory continuity, coordinate positions, velocities, and box vectors
without reconstructing the OpenMM context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from typing import Any, Callable, Sequence

import numpy as np

from .mace_mixed_system import MACEConfig
from .uq_monitor import MACEUQMonitor, UQResult

logger = logging.getLogger("csbrt.mace_surrogate.fallback")


class EvaluatorMode(str, Enum):
    MACE_SURROGATE = "mace_surrogate"
    CLASSICAL_PHYSICS = "classical_physics"


@dataclass
class FallbackState:
    """Live state tracking for the fallback controller."""

    current_mode: EvaluatorMode = EvaluatorMode.MACE_SURROGATE
    weight: float = 0.0  # 0.0 = MACE, 1.0 = Classical
    consecutive_fallback_steps: int = 0
    total_steps: int = 0
    total_surrogate_steps: int = 0
    total_fallback_steps: int = 0
    total_fallback_events: int = 0
    last_trigger_reason: str | None = None
    last_trigger_time: float | None = None
    history_transitions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def fallback_fraction(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.total_fallback_steps / self.total_steps

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_mode": self.current_mode.value,
            "weight": self.weight,
            "consecutive_fallback_steps": self.consecutive_fallback_steps,
            "total_steps": self.total_steps,
            "total_surrogate_steps": self.total_surrogate_steps,
            "total_fallback_steps": self.total_fallback_steps,
            "total_fallback_events": self.total_fallback_events,
            "fallback_fraction": self.fallback_fraction,
            "last_trigger_reason": self.last_trigger_reason,
            "last_trigger_time": self.last_trigger_time,
        }


class PhysicsFallbackController:
    """Manages runtime switching between MACE surrogate and classical physics engine."""

    def __init__(
        self,
        config: MACEConfig | None = None,
        uq_monitor: MACEUQMonitor | None = None,
        on_ood_frame: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config or MACEConfig()
        self.uq_monitor = uq_monitor or MACEUQMonitor(self.config)
        self.on_ood_frame = on_ood_frame
        self.recovery_steps = max(1, self.config.fallback_recovery_steps)
        self.state = FallbackState()

    def decide_state(
        self,
        uq_result: UQResult,
        frame_data: dict[str, Any] | None = None,
    ) -> tuple[EvaluatorMode, float]:
        """Determine next simulation evaluator mode based on UQ result."""
        self.state.total_steps += 1

        if uq_result.is_ood:
            # Out-of-distribution state detected
            if self.state.current_mode != EvaluatorMode.CLASSICAL_PHYSICS:
                self.state.total_fallback_events += 1
                self.state.last_trigger_reason = uq_result.trigger_reason
                self.state.last_trigger_time = time.time()
                self.state.history_transitions.append({
                    "step": self.state.total_steps,
                    "event": "fallback_triggered",
                    "reason": uq_result.trigger_reason,
                    "sigma_f_max": uq_result.sigma_f_max_ev_per_ang,
                    "sigma_e": uq_result.sigma_e_kcal_per_mol,
                })
                logger.warning(
                    f"[FALLBACK TRIGGERED] Step {self.state.total_steps}: {uq_result.trigger_reason}"
                )

                # Dispatch frame to OOD buffer for active learning
                if self.on_ood_frame and frame_data is not None:
                    payload = {
                        "step": self.state.total_steps,
                        "timestamp": time.time(),
                        "reason": uq_result.trigger_reason,
                        "uq_result": uq_result.to_dict(),
                        **frame_data,
                    }
                    try:
                        self.on_ood_frame(payload)
                    except Exception as err:
                        logger.error(f"Error harvesting OOD frame: {err}")

            self.state.current_mode = EvaluatorMode.CLASSICAL_PHYSICS
            self.state.weight = 1.0
            self.state.consecutive_fallback_steps = 0
            self.state.total_fallback_steps += 1
            return self.state.current_mode, self.state.weight

        # In-distribution state
        if self.state.current_mode == EvaluatorMode.CLASSICAL_PHYSICS:
            # Undergoing classical relaxation buffer
            self.state.consecutive_fallback_steps += 1
            self.state.total_fallback_steps += 1
            if self.state.consecutive_fallback_steps >= self.recovery_steps:
                # Recover to MACE surrogate
                self.state.current_mode = EvaluatorMode.MACE_SURROGATE
                self.state.weight = 0.0
                self.state.consecutive_fallback_steps = 0
                self.state.history_transitions.append({
                    "step": self.state.total_steps,
                    "event": "surrogate_resumed",
                    "recovery_steps": self.recovery_steps,
                })
                logger.info(
                    f"[SURROGATE RESUMED] Step {self.state.total_steps}: completed "
                    f"{self.recovery_steps} relaxation steps, returning to MACE fast path."
                )
            return self.state.current_mode, self.state.weight

        # Normal fast path operation
        self.state.total_surrogate_steps += 1
        return self.state.current_mode, self.state.weight

    def apply_to_openmm_context(self, context: Any, weight: float | None = None) -> None:
        """Apply the fallback weight parameter to an OpenMM Context without reinitialization."""
        w = self.state.weight if weight is None else weight
        if hasattr(context, "setParameter"):
            try:
                context.setParameter("fallback_weight", float(w))
            except Exception:
                pass

    def run_stepped_simulation(
        self,
        context: Any,
        integrator: Any,
        total_steps: int,
        eval_interval: int = 10,
        evaluate_uq_fn: Callable[[Any], UQResult] | None = None,
    ) -> dict[str, Any]:
        """Drive dynamics with periodic uncertainty checks and automatic fallback.

        Parameters
        ----------
        context:
            OpenMM Context.
        integrator:
            OpenMM Integrator.
        total_steps:
            Total simulation steps to advance.
        eval_interval:
            Frequency of UQ evaluation in steps.
        evaluate_uq_fn:
            Callable returning UQResult for the current context coordinates.
        """
        step = 0
        while step < total_steps:
            chunk = min(eval_interval, total_steps - step)

            # Evaluate UQ if function provided
            if evaluate_uq_fn is not None:
                uq_res = evaluate_uq_fn(context)
                frame_data = {}
                if hasattr(context, "getState"):
                    st = context.getState(getPositions=True)
                    frame_data["positions"] = np.array(st.getPositions(asNumpy=True))
                mode, w = self.decide_state(uq_res, frame_data)
                self.apply_to_openmm_context(context, w)

            # Integrate forward
            if hasattr(integrator, "step"):
                integrator.step(chunk)
            step += chunk

        return self.state.to_dict()
