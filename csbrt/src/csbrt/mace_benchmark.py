#!/usr/bin/env python3
"""Measure what the MACE surrogate actually costs, in ns/day.

The premise of the surrogate is a throughput win. That premise should be
measured on the system and the hardware you intend to run, not assumed:
Wang, Eastman, Tuckerman et al. (2024, docs/wang2024_design_space_mm_mlff.pdf)
put classical MM at 100-1000 ns/day, hybrid ML/MM at 10-50, and pure MLFF at
0.1-1 on a solvated complex. A hybrid system pays nearly the whole MM cost
*plus* the ML region, so on those numbers it is slower, not faster. See
docs/mlff_throughput_expectations.md.

This script produces the number rather than arguing about it:

    csbrt-mace-benchmark --prmtop complex.prmtop --rst7 complex.rst7
    csbrt-mace-benchmark --steps 500 --json benchmark.json

It times the same Context three ways -- classical, mixed at
lambda_interpolate=0, mixed at 1 -- so the ML cost is isolated from any
overhead the mixed system adds on its own, adds the committee's inference cost
at the configured UQ interval, and prints the resulting speedup factor. A
factor below 1.0 means the surrogate is slower; the script says so plainly
instead of reporting a percentage that reads like a win.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

SECONDS_PER_DAY = 86400.0


def nanoseconds_per_day(steps: int, seconds: float, timestep_fs: float) -> float:
    if seconds <= 0:
        return float("inf")
    nanoseconds = steps * timestep_fs * 1e-6
    return nanoseconds * SECONDS_PER_DAY / seconds


def time_integration(
    context: Any, steps: int, warmup: int, parameter: tuple[str, float] | None
) -> float:
    """Wall seconds for ``steps`` of MD, after a warm-up that pays the JIT."""
    integrator = context.getIntegrator()
    if parameter is not None:
        context.setParameter(*parameter)
    if warmup:
        integrator.step(warmup)
    # Force the asynchronous GPU queue to drain before starting the clock.
    context.getState(getEnergy=True)
    started = time.perf_counter()
    integrator.step(steps)
    context.getState(getEnergy=True)
    return time.perf_counter() - started


def load_system(opt) -> tuple[Any, Any, Any, list[int]]:
    """Return (topology, system, positions, ligand_atoms)."""
    import openmm
    import openmm.app as app
    import openmm.unit as unit

    if opt.prmtop and opt.rst7:
        prmtop = app.AmberPrmtopFile(str(opt.prmtop))
        coordinates = app.AmberInpcrdFile(str(opt.rst7))
        system = prmtop.createSystem(
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.2 * unit.nanometer,
            constraints=app.HBonds,
            rigidWater=True,
        )
        if coordinates.boxVectors is not None:
            system.setDefaultPeriodicBoxVectors(*coordinates.boxVectors)
        from .mace_surrogate import noninteracting_particles, partition_ml_atoms

        ligand_atoms = partition_ml_atoms(
            prmtop.topology,
            positions=coordinates.positions,
            ligand_resname=opt.ligand_resname,
            exclude=noninteracting_particles(system),
        )
        return prmtop.topology, system, coordinates.positions, ligand_atoms

    from .mace_surrogate.testsystems import build_reference_system

    fixture = build_reference_system()
    return (
        fixture.topology,
        fixture.system,
        fixture.positions_quantity(),
        fixture.ligand_atoms,
    )


def build_context(system, positions, platform_name: str, timestep_fs: float):
    import openmm
    import openmm.unit as unit

    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin,
        1.0 / unit.picosecond,
        timestep_fs * unit.femtoseconds,
    )
    context = openmm.Context(
        system, integrator, openmm.Platform.getPlatformByName(platform_name)
    )
    context.setPositions(positions)
    context.setPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
    context.setVelocitiesToTemperature(300 * unit.kelvin, 20260714)
    context._csbrt_integrator = integrator
    return context


def measure_committee(config, atomic_numbers, coordinates) -> dict[str, Any]:
    from .mace_surrogate import CommitteeUnavailable, MACECommittee

    if not config.has_committee:
        return {"configured": False, "seconds_per_evaluation": 0.0}
    try:
        committee = MACECommittee(
            config.committee_model_paths,
            device=config.device,
            precision=config.precision,
        )
        committee.warm_up(atomic_numbers, coordinates)
        for _ in range(10):
            committee.predict(coordinates, atomic_numbers)
    except CommitteeUnavailable as error:
        return {"configured": True, "error": str(error), "seconds_per_evaluation": 0.0}
    statistics = committee.statistics()
    return {
        "configured": True,
        "committee_size": statistics["committee_size"],
        "seconds_per_evaluation": statistics["mean_seconds_per_evaluation"],
    }


def options(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--prmtop", type=Path, default=None,
                        help="Amber topology; omit to use the built-in fixture")
    parser.add_argument("--rst7", type=Path, default=None, help="Amber coordinates")
    parser.add_argument("--ligand-resname", default="LIG")
    parser.add_argument("--model", default="mace-off23-small")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--committee", action="append", default=[], metavar="MODEL")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--openmm-platform", default=None)
    parser.add_argument("--steps", type=int, default=500,
                        help="Timed MD steps per configuration (default: 500)")
    parser.add_argument("--warmup", type=int, default=50,
                        help="Untimed steps before each measurement (default: 50)")
    parser.add_argument("--timestep-fs", type=float, default=2.0)
    parser.add_argument("--uq-interval", type=int, default=100,
                        help="MD steps between UQ evaluations, for the overhead "
                             "estimate (default: 100)")
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    opt = options(argv)
    import openmm

    from .mace_surrogate import (
        INTERPOLATION_PARAMETER,
        MACEConfig,
        attach_mace_to_context,
        describe_ml_region,
    )

    platform_name = opt.openmm_platform
    if platform_name is None:
        available = {
            openmm.Platform.getPlatform(i).getName()
            for i in range(openmm.Platform.getNumPlatforms())
        }
        platform_name = "CUDA" if "CUDA" in available else "CPU"

    config = MACEConfig(
        enabled=True,
        model_name=opt.model,
        model_path=opt.model_path,
        committee_model_paths=tuple(opt.committee),
        device=opt.device,
        precision="double" if opt.device == "cpu" else "single",
        ligand_resname=opt.ligand_resname,
        uq_interval_steps=opt.uq_interval,
    )

    topology, system, positions, ligand_atoms = load_system(opt)
    region = describe_ml_region(topology, ligand_atoms)
    print(
        f"System   : {system.getNumParticles()} particles, "
        f"{len(region)} in the ML region ({region.residues})",
        flush=True,
    )
    print(f"Platform : {platform_name}   device {opt.device}", flush=True)
    print(f"Sampling : {opt.steps} steps at {opt.timestep_fs} fs\n", flush=True)

    classical_context = build_context(
        system, positions, platform_name, opt.timestep_fs
    )
    classical_seconds = time_integration(
        classical_context, opt.steps, opt.warmup, None
    )
    classical_rate = nanoseconds_per_day(
        opt.steps, classical_seconds, opt.timestep_fs
    )
    print(
        f"classical MM          : {classical_seconds:7.2f} s  "
        f"{classical_rate:9.2f} ns/day",
        flush=True,
    )

    mixed_context = build_context(system, positions, platform_name, opt.timestep_fs)
    build_started = time.perf_counter()
    attach_mace_to_context(mixed_context, topology, config, ligand_atoms)
    build_seconds = time.perf_counter() - build_started

    mm_seconds = time_integration(
        mixed_context, opt.steps, opt.warmup, (INTERPOLATION_PARAMETER, 0.0)
    )
    mm_rate = nanoseconds_per_day(opt.steps, mm_seconds, opt.timestep_fs)
    print(
        f"mixed, fallback (l=0) : {mm_seconds:7.2f} s  {mm_rate:9.2f} ns/day",
        flush=True,
    )

    ml_seconds = time_integration(
        mixed_context, opt.steps, opt.warmup, (INTERPOLATION_PARAMETER, 1.0)
    )
    ml_rate = nanoseconds_per_day(opt.steps, ml_seconds, opt.timestep_fs)
    print(
        f"mixed, surrogate (l=1): {ml_seconds:7.2f} s  {ml_rate:9.2f} ns/day",
        flush=True,
    )

    committee = measure_committee(
        config,
        region.atomic_numbers,
        (
            mixed_context.getState(getPositions=True)
            .getPositions(asNumpy=True)
            ._value[ligand_atoms]
            * 10.0
        ),
    )
    uq_seconds = (
        committee["seconds_per_evaluation"] * (opt.steps / max(opt.uq_interval, 1))
    )
    total_ml_seconds = ml_seconds + uq_seconds
    total_ml_rate = nanoseconds_per_day(opt.steps, total_ml_seconds, opt.timestep_fs)
    if committee["configured"]:
        print(
            f"  + UQ every {opt.uq_interval} steps: "
            f"{uq_seconds:7.2f} s  -> {total_ml_rate:9.2f} ns/day",
            flush=True,
        )

    speedup = classical_seconds / total_ml_seconds if total_ml_seconds else 0.0
    print(f"\nmixed-system build    : {build_seconds:.1f} s (once per Context)")
    print(f"speedup vs classical  : {speedup:.3f}x")
    if speedup >= 1.05:
        print(
            f"  The surrogate is {100 * (1 - 1 / speedup):.0f}% faster than the "
            "classical force field on this system."
        )
    elif speedup <= 0.95:
        print(
            f"  The surrogate is {100 * (1 / speedup - 1):.0f}% SLOWER than the "
            "classical force field on this system. It is buying accuracy in the "
            "ligand's internal energetics, not wall-clock time. See "
            "docs/mlff_throughput_expectations.md."
        )
    else:
        print("  The surrogate and the classical force field cost the same here.")

    payload = {
        "particles": system.getNumParticles(),
        "ml_region": region.to_dict(),
        "platform": platform_name,
        "device": opt.device,
        "steps": opt.steps,
        "timestep_fs": opt.timestep_fs,
        "classical": {"seconds": classical_seconds, "ns_per_day": classical_rate},
        "mixed_lambda0": {"seconds": mm_seconds, "ns_per_day": mm_rate},
        "mixed_lambda1": {"seconds": ml_seconds, "ns_per_day": ml_rate},
        "uq": {**committee, "seconds_over_run": uq_seconds},
        "surrogate_total": {
            "seconds": total_ml_seconds,
            "ns_per_day": total_ml_rate,
        },
        "mixed_system_build_seconds": build_seconds,
        "speedup_vs_classical": speedup,
        "config": config.to_dict(),
    }
    if opt.json:
        opt.json.parent.mkdir(parents=True, exist_ok=True)
        opt.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"\nreport written to {opt.json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
