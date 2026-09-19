r"""Zero-overhead physics fallback: the safety net under the surrogate.

Rebuilding an OpenMM ``Context`` costs 200-500 ms of CUDA JIT and allocation.
Doing that every time the surrogate loses confidence would cost more than the
surrogate saves. Instead the mixed system is built once with openmm-ml's
``interpolate=True``, which adds a global parameter:

.. math::

    U(\lambda) = \lambda\, U_{\mathrm{MACE/MM}} + (1 - \lambda)\, U_{\mathrm{MM}}

Switching Hamiltonians is then a single ``Context.setParameter`` call --
microseconds, no reinitialisation, positions and velocities untouched, so the
trajectory stays continuous through the transition.

The controller is a two-state machine over that parameter:

* **surrogate** -- run on MACE/MM. A UQ trigger drops to classical.
* **classical** -- run on pure MM for ``fallback_steps``, then retest. While
  here the flagged frame is written to the OOD buffer for active learning.

If the surrogate spends more than ``fallback_abort_fraction`` of the run in
classical mode it is not paying for itself, so the controller latches to
classical and says so rather than paying the UQ cost for no benefit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from typing import Any, Callable

import numpy as np

from .config import MACEConfig
from .mace_mixed_system import INTERPOLATION_PARAMETER, MACESurrogateError
from .uq_monitor import MACEUQMonitor, UQResult

logger = logging.getLogger("csbrt.mace_surrogate.fallback")


class EvaluatorMode(str, Enum):
    MACE_SURROGATE = "mace_surrogate"
    CLASSICAL_PHYSICS = "classical_physics"


class DualHamiltonianSwitch:
    """Flips a live Context between the mixed ML/MM and classical Hamiltonians.

    Wraps one global parameter write. Kept as a class so the latency can be
    measured and so a Context that never got an interpolating mixed system
    fails loudly instead of silently running classical physics while the log
    claims a surrogate.
    """

    def __init__(
        self,
        context: Any,
        *,
        parameter_name: str = INTERPOLATION_PARAMETER,
        surrogate_value: float = 1.0,
        classical_value: float = 0.0,
        require_parameter: bool = True,
    ) -> None:
        self.context = context
        self.parameter_name = parameter_name
        self.surrogate_value = float(surrogate_value)
        self.classical_value = float(classical_value)
        self.available = self._parameter_present(context, parameter_name)
        if require_parameter and not self.available:
            raise MACESurrogateError(
                f"Context has no global parameter {parameter_name!r}. Build the "
                "mixed system with interpolate=True (MACEConfig.interpolate), "
                "otherwise the fallback would need a Context rebuild."
            )
        self.switch_count = 0
        self.total_switch_seconds = 0.0
        self.max_switch_seconds = 0.0

    @staticmethod
    def _parameter_present(context: Any, name: str) -> bool:
        getter = getattr(context, "getParameters", None)
        if getter is None:
            return False
        try:
            return name in getter()
        except Exception:  # pragma: no cover - defensive against odd mocks
            return False

    def set_mode(self, mode: EvaluatorMode) -> float:
        """Apply ``mode`` to the Context. Returns the write latency in seconds."""
        value = (
            self.surrogate_value
            if mode is EvaluatorMode.MACE_SURROGATE
            else self.classical_value
        )
        return self.set_value(value)

    def set_value(self, value: float) -> float:
        if not self.available:
            return 0.0
        started = time.perf_counter()
        self.context.setParameter(self.parameter_name, float(value))
        elapsed = time.perf_counter() - started
        self.switch_count += 1
        self.total_switch_seconds += elapsed
        self.max_switch_seconds = max(self.max_switch_seconds, elapsed)
        return elapsed

    def current_value(self) -> float | None:
        if not self.available:
            return None
        return float(self.context.getParameter(self.parameter_name))

    def measure_latency(self, samples: int = 32) -> dict[str, float]:
        """Time the Hamiltonian switch, for the zero-overhead claim in the plan."""
        if not self.available:
            raise MACESurrogateError("No interpolation parameter to measure")
        original = self.current_value()
        timings = []
        for index in range(samples):
            mode = (
                EvaluatorMode.CLASSICAL_PHYSICS
                if index % 2
                else EvaluatorMode.MACE_SURROGATE
            )
            timings.append(self.set_mode(mode))
        self.set_value(original if original is not None else self.surrogate_value)
        array = np.asarray(timings, dtype=np.float64)
        return {
            "samples": int(array.size),
            "mean_ms": float(array.mean() * 1e3),
            "median_ms": float(np.median(array) * 1e3),
            "max_ms": float(array.max() * 1e3),
        }

    def statistics(self) -> dict[str, Any]:
        mean = (
            self.total_switch_seconds / self.switch_count if self.switch_count else 0.0
        )
        return {
            "parameter_name": self.parameter_name,
            "available": self.available,
            "switch_count": self.switch_count,
            "mean_switch_ms": mean * 1e3,
            "max_switch_ms": self.max_switch_seconds * 1e3,
        }


@dataclass
class FallbackState:
    """Live accounting for one run."""

    current_mode: EvaluatorMode = EvaluatorMode.MACE_SURROGATE
    classical_steps_remaining: int = 0
    evaluations: int = 0
    md_steps: int = 0
    surrogate_steps: int = 0
    classical_steps: int = 0
    fallback_events: int = 0
    ood_frames_captured: int = 0
    latched_classical: bool = False
    last_trigger_reason: str | None = None
    last_trigger_time: float | None = None
    transitions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def weight(self) -> float:
        """The classical weight ``w``: 0.0 on the surrogate, 1.0 on physics."""
        return 0.0 if self.current_mode is EvaluatorMode.MACE_SURROGATE else 1.0

    @property
    def fallback_fraction(self) -> float:
        if self.md_steps == 0:
            return 0.0
        return self.classical_steps / self.md_steps

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_mode": self.current_mode.value,
            "weight": self.weight,
            "evaluations": self.evaluations,
            "md_steps": self.md_steps,
            "surrogate_steps": self.surrogate_steps,
            "classical_steps": self.classical_steps,
            "fallback_events": self.fallback_events,
            "fallback_fraction": self.fallback_fraction,
            "ood_frames_captured": self.ood_frames_captured,
            "latched_classical": self.latched_classical,
            "last_trigger_reason": self.last_trigger_reason,
            "last_trigger_time": self.last_trigger_time,
            "transitions": list(self.transitions),
        }


class PhysicsFallbackController:
    """Decides which Hamiltonian runs next, and applies it."""

    def __init__(
        self,
        config: MACEConfig | None = None,
        uq_monitor: MACEUQMonitor | None = None,
        switch: DualHamiltonianSwitch | None = None,
        on_ood_frame: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config or MACEConfig()
        self.uq_monitor = uq_monitor or MACEUQMonitor(self.config)
        self.switch = switch
        self.on_ood_frame = on_ood_frame
        self.fallback_steps = max(1, int(self.config.fallback_steps))
        self.state = FallbackState()
        #: Below this many evaluations the abort check is meaningless, so a
        #: single early trigger cannot latch the whole run to classical.
        self.min_evaluations_before_abort = 20

    # --- decision ------------------------------------------------------------

    def observe(
        self,
        uq_result: UQResult,
        frame_data: dict[str, Any] | None = None,
    ) -> EvaluatorMode:
        """Fold one UQ evaluation into the state machine and apply the result."""
        self.state.evaluations += 1

        if self.state.latched_classical:
            self._apply(EvaluatorMode.CLASSICAL_PHYSICS)
            return self.state.current_mode

        if uq_result.is_ood:
            first_entry = (
                self.state.current_mode is EvaluatorMode.MACE_SURROGATE
            )
            self.state.classical_steps_remaining = self.fallback_steps
            self.state.last_trigger_reason = uq_result.trigger_reason
            self.state.last_trigger_time = time.time()
            if first_entry:
                self.state.fallback_events += 1
                self.state.transitions.append(
                    {
                        "evaluation": self.state.evaluations,
                        "md_step": self.state.md_steps,
                        "event": "fallback_triggered",
                        "reason": uq_result.trigger_reason,
                        "sigma_f_max_ev_per_ang": uq_result.sigma_f_max_ev_per_ang,
                        "sigma_e_kcal_per_mol": uq_result.sigma_e_kcal_per_mol,
                    }
                )
                logger.warning(
                    "[FALLBACK] step %d: %s",
                    self.state.md_steps,
                    uq_result.trigger_reason,
                )
                self._capture(uq_result, frame_data)
            self._apply(EvaluatorMode.CLASSICAL_PHYSICS)
            return self.state.current_mode

        if self.state.current_mode is EvaluatorMode.CLASSICAL_PHYSICS:
            if self.state.classical_steps_remaining > 0:
                # Still inside the relaxation buffer.
                return self.state.current_mode
            self.state.transitions.append(
                {
                    "evaluation": self.state.evaluations,
                    "md_step": self.state.md_steps,
                    "event": "surrogate_resumed",
                }
            )
            logger.info(
                "[SURROGATE RESUMED] step %d after %d classical steps",
                self.state.md_steps,
                self.fallback_steps,
            )
            self._apply(EvaluatorMode.MACE_SURROGATE)
            return self.state.current_mode

        self._apply(EvaluatorMode.MACE_SURROGATE)
        return self.state.current_mode

    def advance(self, steps: int) -> None:
        """Record ``steps`` of MD run under the current mode."""
        if steps < 0:
            raise ValueError("steps must not be negative")
        self.state.md_steps += steps
        if self.state.current_mode is EvaluatorMode.CLASSICAL_PHYSICS:
            self.state.classical_steps += steps
            self.state.classical_steps_remaining = max(
                0, self.state.classical_steps_remaining - steps
            )
        else:
            self.state.surrogate_steps += steps
        self._check_abort()

    def _apply(self, mode: EvaluatorMode) -> None:
        if self.state.current_mode is not mode:
            self.state.current_mode = mode
        if self.switch is not None:
            self.switch.set_mode(mode)

    def _check_abort(self) -> None:
        if self.state.latched_classical:
            return
        if self.state.evaluations < self.min_evaluations_before_abort:
            return
        if self.state.fallback_fraction <= self.config.fallback_abort_fraction:
            return
        self.state.latched_classical = True
        self.state.transitions.append(
            {
                "evaluation": self.state.evaluations,
                "md_step": self.state.md_steps,
                "event": "latched_classical",
                "fallback_fraction": self.state.fallback_fraction,
            }
        )
        logger.warning(
            "MACE surrogate spent %.0f%% of %d steps on the classical "
            "Hamiltonian, above the %.0f%% abort threshold. Latching to "
            "classical physics for the rest of this run; the sampling is "
            "unaffected, the speedup is not being realised. The OOD buffer "
            "holds the frames to fine-tune on.",
            100.0 * self.state.fallback_fraction,
            self.state.md_steps,
            100.0 * self.config.fallback_abort_fraction,
        )
        self._apply(EvaluatorMode.CLASSICAL_PHYSICS)

    def _capture(
        self, uq_result: UQResult, frame_data: dict[str, Any] | None
    ) -> None:
        if self.on_ood_frame is None or frame_data is None:
            return
        payload = {
            "md_step": self.state.md_steps,
            "timestamp": time.time(),
            "reason": uq_result.trigger_reason,
            "uq_result": uq_result.to_dict(),
            **frame_data,
        }
        try:
            self.on_ood_frame(payload)
        except Exception as error:  # active learning must never kill a run
            logger.error("Failed to harvest OOD frame: %s", error)
        else:
            self.state.ood_frames_captured += 1

    # --- driving -------------------------------------------------------------

    def run_segmented_dynamics(
        self,
        run_steps: Callable[[int], None],
        total_steps: int,
        evaluate_uq: Callable[[], UQResult] | None = None,
        collect_frame: Callable[[], dict[str, Any]] | None = None,
        interval: int | None = None,
    ) -> dict[str, Any]:
        """Advance ``total_steps`` of MD, checking uncertainty every ``interval``.

        ``run_steps(n)`` advances the integrator; ``evaluate_uq()`` returns the
        UQ result for the Context's current coordinates; ``collect_frame()``
        supplies the coordinates and metadata stored when a frame is flagged.
        Kept callback-shaped so this drives Sire/Loch dynamics, a bare OpenMM
        integrator, or a test double without knowing which.
        """
        if total_steps < 0:
            raise ValueError("total_steps must not be negative")
        stride = int(
            self.config.uq_interval_steps if interval is None else interval
        )
        if stride < 1:
            raise ValueError("interval must be at least 1")

        done = 0
        while done < total_steps:
            if evaluate_uq is not None:
                result = evaluate_uq()
                frame = collect_frame() if collect_frame is not None else None
                self.observe(result, frame)
            chunk = min(stride, total_steps - done)
            run_steps(chunk)
            self.advance(chunk)
            done += chunk
        return self.state.to_dict()

    # --- reporting -----------------------------------------------------------

    def statistics(self) -> dict[str, Any]:
        return {
            "fallback": self.state.to_dict(),
            "uq": self.uq_monitor.get_statistics(),
            "switch": self.switch.statistics() if self.switch else None,
            "fallback_steps": self.fallback_steps,
            "uq_interval_steps": self.config.uq_interval_steps,
        }
