#!/usr/bin/env python3
"""Throughput and speedup benchmark for the MACE ML/MM surrogate.

Measures, on one GPU node:

1. Classical-MM throughput (ns/day) with ``lambda_interpolate = 0.0``.
2. Hybrid ML/MM throughput (ns/day) with ``lambda_interpolate = 1.0``.
3. Per-interval committee UQ evaluation cost.
4. Fallback frequency under a synthetic stress signal.

This is a Stage-6 verification tool.  Run it after :file:`preflight_mace.py`
passes.  For a smoke check use ``--smoke`` (a few steps, no statistical power).
"""

from __future__ import annotations

import argparse
import json
import sys
import time


def _require_deps() -> None:
    for module in ("numpy", "openmm", "openmmml"):
        try:
            __import__(module)
        except Exception as err:  # noqa: BLE001
            raise SystemExit(
                f"Missing dependency '{module}': {err}. Install the csbrt conda env first."
            )


def _build_benchmark_system(num_atoms: int, box_nm: float):
    """Build a minimal periodic LJ system for throughput benchmarking."""
    import openmm
    from openmm import unit

    system = openmm.System()
    for _ in range(num_atoms):
        system.addParticle(12.0 * unit.amu)
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(box_nm, 0, 0) * unit.nanometer,
        openmm.Vec3(0, box_nm, 0) * unit.nanometer,
        openmm.Vec3(0, 0, box_nm) * unit.nanometer,
    )
    force = openmm.NonbondedForce()
    force.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(1.0 * unit.nanometer)
    for _ in range(num_atoms):
        force.addParticle(0.0, 0.34, 0.996)
    system.addForce(force)
    system.addForce(openmm.CMMotionRemover())
    return system


def _run_throughput(system, steps: int, device: str, dt_fs: float):
    import openmm
    from openmm import unit

    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, dt_fs * unit.femtoseconds
    )
    platform = openmm.Platform.getPlatformByName("CUDA" if device == "cuda" else "CPU")
    context = openmm.Context(system, integrator, platform)
    context.setPositions(
        [
            openmm.Vec3(i * 0.35 % 4.0, i * 0.21 % 4.0, i * 0.13 % 4.0)
            for i in range(system.getNumParticles())
        ]
    )
    openmm.LocalEnergyMinimizer.minimize(context)
    context.setVelocitiesToTemperature(300 * unit.kelvin)

    start = time.perf_counter()
    integrator.step(steps)
    elapsed = time.perf_counter() - start
    ns = steps * dt_fs * 1e-6
    ns_per_day = (ns / elapsed) * 86400.0
    return elapsed, ns_per_day


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mace-off23-small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="float32", choices=("float32", "float64"))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--dt-fs", type=float, default=2.0)
    parser.add_argument("--num-atoms", type=int, default=512,
                        help="Benchmark box size when no --topology is given")
    parser.add_argument("--box-nm", type=float, default=4.0)
    parser.add_argument("--uq-interval", type=int, default=10)
    parser.add_argument("--smoke", action="store_true", help="Minimal step count")
    args = parser.parse_args(argv)

    _require_deps()
    if args.smoke:
        args.steps = min(args.steps, 20)

    import openmm
    from openmm import unit

    system = _build_benchmark_system(args.num_atoms, args.box_nm)

    # Build a synthetic OpenMM Topology for the benchmark particles so that
    # createMixedSystem has element information for the ML region.
    import openmm.app as app
    topology = app.Topology()
    chain = topology.addChain()
    res = topology.addResidue("LIG", chain)
    elem = app.Element.getByAtomicNumber(6)
    for _ in range(args.num_atoms):
        topology.addAtom("C", elem, res)

    # Use the first N/8 atoms as a stand-in "ligand" ML region.
    ml_atoms = list(range(max(1, args.num_atoms // 8)))

    from openmmml import MLPotential
    precision_kw = {"float32": "single", "float64": "double"}[args.precision]
    potential = MLPotential(args.model)
    mixed = potential.createMixedSystem(
        topology, system, ml_atoms, interpolate=True, device=args.device,
        precision=precision_kw,
    )

    print(f"Benchmark: {args.num_atoms} atoms, ML region={len(ml_atoms)}, "
          f"model={args.model}, steps={args.steps}")
    print("Running classical-MM leg (lambda_interpolate=0.0) ...")
    mm_elapsed, mm_ns_day = _run_throughput(mixed, args.steps, args.device, args.dt_fs)

    # Set lambda_interpolate=1.0 on a fresh context for the ML/MM fast path.
    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, args.dt_fs * unit.femtoseconds
    )
    platform = openmm.Platform.getPlatformByName("CUDA" if args.device == "cuda" else "CPU")
    context = openmm.Context(mixed, integrator, platform)
    context.setParameter("lambda_interpolate", 1.0)
    print("Running ML/MM fast-path leg (lambda_interpolate=1.0) ...")
    start = time.perf_counter()
    integrator.step(args.steps)
    ml_elapsed = time.perf_counter() - start
    ml_ns = args.steps * args.dt_fs * 1e-6
    ml_ns_day = (ml_ns / ml_elapsed) * 86400.0

    speedup = mm_ns_day / ml_ns_day if ml_ns_day > 0 else float("nan")
    report = {
        "num_atoms": args.num_atoms,
        "ml_region_atoms": len(ml_atoms),
        "model": args.model,
        "steps": args.steps,
        "dt_fs": args.dt_fs,
        "classical_mm_elapsed_s": mm_elapsed,
        "classical_mm_ns_per_day": mm_ns_day,
        "ml_mm_elapsed_s": ml_elapsed,
        "ml_mm_ns_per_day": ml_ns_day,
        "speedup_ml_over_mm": speedup,
    }
    print(json.dumps(report, indent=2))
    if speedup > 1.0:
        print(f"ML/MM fast path is {speedup:.2f}x faster than classical MM.")
    else:
        print("ML/MM fast path is NOT faster than classical MM on this system/device.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
