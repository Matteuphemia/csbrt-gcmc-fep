"""Angle 3: Real-time uncertainty quantification and fallback interception.

Every uncertainty here is a real committee measurement. A two-member MACE-OFF23
committee (medium + large -- they share r_max and element table, so
``MACECalculator`` accepts them) evaluates a matrix of ligand geometries:
relaxed / thermally jittered poses (in-distribution) and deliberately distorted
poses (steric clashes, stretched bonds). Each geometry passes through the real
``GeometryGuard``, ``MACEUQMonitor`` and ``PhysicsFallbackController``. The
sensitivity, specificity and per-case sigma_F are computed from what those
components actually returned -- nothing is simulated.

Caveat: two foundation models of different sizes are correlated, so their force
variance underestimates a true active-learning committee's. This measures that
the interception *machinery* works on real forces, not a production fallback
rate.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    import torch
    from csbrt.mace_surrogate.uq_monitor import GeometryGuard, MACEUQMonitor
    from csbrt.mace_surrogate.fallback_controller import (
        EvaluatorMode,
        PhysicsFallbackController,
    )
    from csbrt.mace_surrogate.committee import MACECommittee
    from csbrt.mace_surrogate.config import MACEConfig
    from csbrt.mace_surrogate.testsystems import (
        LIGAND_ATOMIC_NUMBERS,
        LIGAND_ATOMS,
        LIGAND_BONDS,
    )
    from csbrt.compare_tests._common import COMMITTEE_MODELS, ensure_mace_off_cached

    HAS_DEPS = True
    _IMPORT_ERROR = None
except Exception as error:  # pragma: no cover - environment dependent
    HAS_DEPS = False
    _IMPORT_ERROR = repr(error)


def run_uq_interception_benchmark() -> dict[str, Any]:
    if not HAS_DEPS:
        return {
            "summary": {
                "test_name": "UQ & Geometry Guard Interception",
                "status": "not_run",
                "reason": f"Required dependency missing: {_IMPORT_ERROR}",
            },
            "details": [],
        }

    model_paths = [str(ensure_mace_off_cached(name)) for name in COMMITTEE_MODELS]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    committee = MACECommittee(model_paths, device=device, precision="single")

    base = np.array([atom[2] for atom in LIGAND_ATOMS], dtype=np.float64)
    atomic_numbers = list(LIGAND_ATOMIC_NUMBERS)

    committee.warm_up(atomic_numbers, base)

    threshold_ev = 0.05
    config = MACEConfig(
        enabled=True, uq_force_threshold_ev_per_ang=threshold_ev
    )
    guard = GeometryGuard(
        atomic_numbers=atomic_numbers,
        bonds=LIGAND_BONDS,
        min_distance_ang=config.geometry_min_distance_ang,
        max_bond_scale=config.geometry_max_bond_scale,
    )
    monitor = MACEUQMonitor(config, committee=committee, geometry_guard=guard)
    controller = PhysicsFallbackController(config=config, uq_monitor=monitor)

    cases: list[dict[str, Any]] = []
    rng = np.random.default_rng(42)
    for i in range(12):
        cases.append(
            {
                "type": f"thermal_frame_{i + 1}",
                "category": "in_distribution",
                "coords": base + rng.normal(0.0, 0.04, base.shape),
                "expected_flag": False,
            }
        )
    for clash in (0.40, 0.50, 0.60):
        coords = base.copy()
        coords[3] = coords[0] + np.array([clash, 0.0, 0.0])
        cases.append(
            {
                "type": f"steric_clash_{clash:.2f}A",
                "category": "out_of_distribution",
                "coords": coords,
                "expected_flag": True,
            }
        )
    for stretch in (1.8, 2.4, 3.0):
        coords = base.copy()
        coords[1] = coords[0] + np.array([stretch, 0.0, 0.0])
        cases.append(
            {
                "type": f"bond_stretch_{stretch:.1f}A",
                "category": "out_of_distribution",
                "coords": coords,
                "expected_flag": True,
            }
        )

    tp = fp = tn = fn = 0
    details: list[dict[str, Any]] = []
    for case in cases:
        t0 = time.perf_counter_ns()
        result = monitor.evaluate(case["coords"], atomic_numbers)
        mode = controller.observe(result, {"positions": case["coords"]})
        latency_us = (time.perf_counter_ns() - t0) / 1000.0

        flagged = bool(result.is_ood)
        expected = case["expected_flag"]
        if expected and flagged:
            tp += 1
        elif expected and not flagged:
            fn += 1
        elif not expected and not flagged:
            tn += 1
        else:
            fp += 1

        details.append(
            {
                "type": case["type"],
                "category": case["category"],
                "expected_flag": expected,
                "actual_flag": flagged,
                "reason": result.trigger_reason,
                "sigma_f_max_ev_per_ang": result.sigma_f_max_ev_per_ang,
                "geometry_ok": result.geometry_ok,
                "mode": mode.value,
                "latency_us": latency_us,
            }
        )

    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    in_dist_sigma = [
        d["sigma_f_max_ev_per_ang"]
        for d in details
        if d["category"] == "in_distribution"
    ]

    summary = {
        "test_name": "UQ & Geometry Guard Interception",
        "status": "passed" if fn == 0 else "failed",
        "measured": True,
        "committee_models": list(COMMITTEE_MODELS),
        "device": device,
        "force_threshold_ev_per_ang": threshold_ev,
        "total_cases_evaluated": len(cases),
        "in_distribution_cases": len(in_dist_sigma),
        "out_of_distribution_cases": len(cases) - len(in_dist_sigma),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "intercept_rate_pct": float(sensitivity * 100.0),
        "false_negative_rate_pct": float(fn / max(tp + fn, 1) * 100.0),
        "in_distribution_sigma_f_mean_ev_per_ang": (
            float(np.mean(in_dist_sigma)) if in_dist_sigma else None
        ),
        "in_distribution_sigma_f_max_ev_per_ang": (
            float(np.max(in_dist_sigma)) if in_dist_sigma else None
        ),
        "committee_note": (
            "Two different-size foundation models are correlated, so this "
            "force variance is a lower bound on a true fine-tune committee's; "
            "the geometry guard carries the distorted-pose interception."
        ),
    }
    return {"summary": summary, "details": details}


if __name__ == "__main__":
    import json

    print(json.dumps(run_uq_interception_benchmark()["summary"], indent=2))
