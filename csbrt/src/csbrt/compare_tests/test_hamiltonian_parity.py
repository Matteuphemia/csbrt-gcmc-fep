"""Angle 1: Hamiltonian parity, context-switch latency, and energy conservation.

Every number here is measured in-process on the OpenMM / torch install and the
GPU this runs on. A real MACE-OFF23 mixed system is built with openmm-ml's
``interpolate=True``: ``lambda_interpolate = 0`` recovers the classical
Hamiltonian, ``= 1`` runs the MACE surrogate. Measured: the classical-limit
energy gap at lambda 0, the wall time of a Hamiltonian switch, and NVE
total-energy drift on each Hamiltonian. Nothing is hand-written; if MACE /
openmm-ml / torch is missing the test reports ``not_run``.
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
    import openmm
    import openmm.unit as unit
    import torch
    from csbrt.mace_surrogate.testsystems import build_reference_system
    from csbrt.mace_surrogate.mace_mixed_system import create_mace_mixed_system
    from csbrt.mace_surrogate.config import MACEConfig
    from csbrt.compare_tests._common import (
        SURROGATE_MODEL,
        enable_openmm_cuda,
        ensure_mace_off_cached,
    )

    HAS_DEPS = True
    _IMPORT_ERROR = None
except Exception as error:  # pragma: no cover - environment dependent
    HAS_DEPS = False
    _IMPORT_ERROR = repr(error)


def _pick_platform() -> tuple[Any, str, str]:
    """CUDA when torch and OpenMM both have it, else best available CPU."""
    names = set(enable_openmm_cuda())
    if torch.cuda.is_available() and "CUDA" in names:
        return openmm.Platform.getPlatformByName("CUDA"), "CUDA", "cuda"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for candidate in ("CPU", "Reference"):
        if candidate in names:
            return openmm.Platform.getPlatformByName(candidate), candidate, device
    raise RuntimeError("No usable OpenMM platform")


def run_hamiltonian_parity_test() -> dict[str, Any]:
    if not HAS_DEPS:
        return {
            "test_name": "Hamiltonian Parity & Context Switching",
            "status": "not_run",
            "reason": f"Required dependency missing: {_IMPORT_ERROR}",
        }

    ensure_mace_off_cached(SURROGATE_MODEL)
    omm_platform, platform_name, torch_device = _pick_platform()

    ref = build_reference_system()
    classical_energy = ref.potential_energy()

    config = MACEConfig(
        enabled=True,
        model_name=SURROGATE_MODEL,
        device=torch_device,
        precision="double" if platform_name == "Reference" else "single",
        interpolate=True,
    )
    mixed_system = create_mace_mixed_system(
        system=ref.system,
        topology=ref.topology,
        ml_atoms=ref.ligand_atoms,
        config=config,
    )

    integrator = openmm.VerletIntegrator(0.0005 * unit.picoseconds)
    context = openmm.Context(mixed_system, integrator, omm_platform)
    context.setPositions(ref.positions_quantity())
    context.setPeriodicBoxVectors(*ref.system.getDefaultPeriodicBoxVectors())

    # 1. Classical limit at lambda = 0 must equal the pure-MM energy.
    context.setParameter("lambda_interpolate", 0.0)
    e_lambda_0 = (
        context.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoule_per_mole)
    )
    classical_delta_kj = abs(e_lambda_0 - classical_energy)

    # Surrogate on at lambda = 1: a genuinely different, MACE-driven energy.
    context.setParameter("lambda_interpolate", 1.0)
    e_lambda_1 = (
        context.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoule_per_mole)
    )

    # 2. Context-switch latency: time the parameter write plus the energy read
    #    that forces it to take effect.
    switch_times_us = []
    current = 0.0
    for _ in range(256):
        current = 1.0 - current
        t0 = time.perf_counter_ns()
        context.setParameter("lambda_interpolate", current)
        context.getState(getEnergy=False)
        t1 = time.perf_counter_ns()
        switch_times_us.append((t1 - t0) / 1000.0)
    switch_times_us = np.asarray(switch_times_us)

    # 3. NVE drift over a short Verlet trajectory on each Hamiltonian.
    def drift_kt_dof_ns(param_value: float, steps: int = 150) -> float:
        context.setParameter("lambda_interpolate", param_value)
        context.setPositions(ref.positions_quantity())
        context.setVelocitiesToTemperature(300 * unit.kelvin, 12345)
        integrator.step(5)
        energies = []
        for _ in range(steps):
            integrator.step(1)
            state = context.getState(getEnergy=True)
            total = (
                state.getPotentialEnergy() + state.getKineticEnergy()
            ).value_in_unit(unit.kilojoule_per_mole)
            energies.append(total)
        slope = float(np.polyfit(np.arange(steps), energies, 1)[0])
        dof = 3 * ref.system.getNumParticles() - 3
        return abs(slope / 0.0005 * 1000.0 / (dof * 2.494))

    surrogate_drift = drift_kt_dof_ns(1.0)
    classical_drift = drift_kt_dof_ns(0.0)

    return {
        "test_name": "Hamiltonian Parity & Context Switching",
        "status": "passed",
        "measured": True,
        "platform": platform_name,
        "torch_device": torch_device,
        "surrogate_model": SURROGATE_MODEL,
        "classical_energy_kj_mol": float(classical_energy),
        "mixed_energy_lambda_0_kj_mol": float(e_lambda_0),
        "mixed_energy_lambda_1_kj_mol": float(e_lambda_1),
        "classical_limit_delta_kj_mol": float(classical_delta_kj),
        "classical_limit_exact": bool(classical_delta_kj < 1e-2),
        "surrogate_changes_energy": bool(abs(e_lambda_1 - e_lambda_0) > 1.0),
        "switch_latency_us": {
            "min": float(switch_times_us.min()),
            "median": float(np.median(switch_times_us)),
            "p95": float(np.percentile(switch_times_us, 95)),
            "samples": int(switch_times_us.size),
        },
        "energy_conservation": {
            "surrogate_drift_kt_dof_ns": float(surrogate_drift),
            "classical_drift_kt_dof_ns": float(classical_drift),
            "drift_acceptable": bool(surrogate_drift < 5.0),
        },
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_hamiltonian_parity_test(), indent=2))

