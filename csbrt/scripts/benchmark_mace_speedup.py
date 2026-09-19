#!/usr/bin/env python3
"""Benchmark throughput and speedup of MACE surrogate vs classical MM.

Measures:
1. Pure classical MM time per step / ns per day.
2. MACE hybrid ML/MM time per step / ns per day.
3. Fallback frequency and overhead under synthetic perturbation.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from csbrt.mace_surrogate import (
    MACEConfig,
    MACEUQMonitor,
    PhysicsFallbackController,
)


def benchmark_mock_dynamics(
    num_atoms: int = 50,
    steps: int = 1000,
    ood_rate: float = 0.02,
) -> dict[str, float]:
    """Simulate runtime loop with mock energies to benchmark controller overhead."""
    cfg = MACEConfig(enabled=True, uq_force_threshold_ev_per_ang=0.05)
    uq = MACEUQMonitor(config=cfg)
    ctrl = PhysicsFallbackController(config=cfg, uq_monitor=uq)

    # 1. Benchmark pure MM mock
    start_mm = time.perf_counter()
    for _ in range(steps):
        # mock MM step: compute forces, update coords
        _ = np.random.randn(num_atoms, 3) * 0.1
    time_mm = time.perf_counter() - start_mm

    # 2. Benchmark surrogate + UQ check loop
    start_surrogate = time.perf_counter()
    rng = np.random.default_rng(42)
    for step in range(steps):
        is_ood_step = (rng.random() < ood_rate)
        force_std = 0.08 if is_ood_step else 0.01  # eV/A
        
        # Committee forces
        f1 = np.zeros((num_atoms, 3))
        f2 = np.ones((num_atoms, 3)) * force_std * 9648.53
        uq_res = uq.compute_from_ensemble_predictions([f1, f2], [0.0, 0.0])
        mode, w = ctrl.decide_state(uq_res, {"step": step})
    time_surrogate = time.perf_counter() - start_surrogate

    stats = ctrl.state.to_dict()
    return {
        "steps": steps,
        "time_mm_s": time_mm,
        "time_surrogate_overhead_s": time_surrogate,
        "us_per_decision": (time_surrogate / steps) * 1e6,
        "fallback_rate": stats["fallback_fraction"],
        "fallback_events": stats["total_fallback_events"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--ood-rate", type=float, default=0.05)
    args = parser.parse_args()

    print(f"Running MACE controller benchmark ({args.steps} steps, target OOD rate {args.ood_rate*100:.1f}%)...")
    res = benchmark_mock_dynamics(steps=args.steps, ood_rate=args.ood_rate)
    print(json.dumps(res, indent=2))
    print(f"Controller decision overhead: {res['us_per_decision']:.2f} µs/step")
    print(f"Observed fallback fraction: {res['fallback_rate']*100:.2f}%")


if __name__ == "__main__":
    main()

