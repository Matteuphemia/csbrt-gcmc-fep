#!/usr/bin/env python3
"""Diagnose a node's readiness to run the MACE ML/MM surrogate.

Run this once per cluster image, and again whenever the environment changes.
It is cheap, it needs no input files, and it fails in the same places a real
run would -- but in seconds, on a login node, with an error you can read,
instead of thirty minutes into a 52-edge array.

    csbrt-mace-preflight
    csbrt-mace-preflight --model mace-off23-small --device cuda
    csbrt-mace-preflight --committee gen1_seed0.model --committee gen1_seed1.model

Checks, in the order a run hits them:

1. imports and versions -- openmm, openmm-ml, torch, mace-torch;
2. compute -- which OpenMM platforms exist, whether torch sees CUDA, and
   whether the two agree (a CUDA OpenMM with a CPU-only torch will run, very
   slowly, and nothing will say so);
3. the ML potential -- MLPotential(name) constructs and downloads its weights;
4. a mixed system -- built on a small periodic fixture, with the classical
   limit asserted to reproduce the untouched MM energy;
5. the Hamiltonian switch -- timed, because the whole design rests on it being
   microseconds rather than a Context rebuild;
6. the committee -- loaded, checked for compatible cutoffs, and timed.

Exit status is 0 when everything required passed, 1 otherwise. ``--json``
writes the full report for a checkpoint or a ticket.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

STATUS_SYMBOL = {"ok": "PASS", "warn": "WARN", "fail": "FAIL", "skip": "SKIP"}


class Report:
    """Ordered check results, with an exit status."""

    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(
        self, name: str, status: str, detail: str, **extra: Any
    ) -> dict[str, Any]:
        entry = {"check": name, "status": status, "detail": detail, **extra}
        self.checks.append(entry)
        print(f"[{STATUS_SYMBOL[status]}] {name}: {detail}", flush=True)
        return entry

    @property
    def failed(self) -> list[dict[str, Any]]:
        return [c for c in self.checks if c["status"] == "fail"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "checks": self.checks,
            "failures": len(self.failed),
            "ok": not self.failed,
        }


def check_imports(report: Report) -> tuple[dict[str, Any], set[str]]:
    """Return the module versions and the set of modules that imported.

    Availability is tracked separately from the version string: openmm-ml
    exposes no ``__version__``, so "did it import" cannot be inferred from
    "does it have a version".
    """
    versions: dict[str, Any] = {}
    available: set[str] = set()
    for module, required in (
        ("openmm", True),
        ("openmmml", True),
        ("torch", True),
        ("mace", True),
        ("numpy", True),
        ("ase", False),
    ):
        try:
            imported = __import__(module)
        except ImportError as error:
            versions[module] = None
            report.add(
                f"import {module}",
                "fail" if required else "warn",
                str(error),
            )
            continue
        version = getattr(imported, "__version__", None)
        if module == "openmm":
            version = imported.version.version
        versions[module] = version
        available.add(module)
        report.add(
            f"import {module}",
            "ok",
            f"version {version}" if version else "present (no version string)",
        )
    return versions, available


def check_compute(report: Report, device: str) -> None:
    try:
        import openmm
    except ImportError:
        return
    platforms = [
        openmm.Platform.getPlatform(i).getName()
        for i in range(openmm.Platform.getNumPlatforms())
    ]
    has_cuda_platform = "CUDA" in platforms
    report.add(
        "openmm platforms",
        "ok" if has_cuda_platform else "warn",
        ", ".join(platforms)
        + ("" if has_cuda_platform else " (no CUDA platform: MD will be slow)"),
        platforms=platforms,
    )

    try:
        import torch
    except ImportError:
        return
    torch_cuda = bool(torch.cuda.is_available())
    detail = f"torch.cuda.is_available()={torch_cuda}"
    if torch_cuda:
        detail += f", {torch.cuda.device_count()} device(s), {torch.version.cuda}"
    report.add("torch cuda", "ok" if torch_cuda else "warn", detail)

    if device.startswith("cuda") and not torch_cuda:
        report.add(
            "device agreement",
            "fail",
            f"--device {device} was requested but torch reports no CUDA device; "
            "MACE would run on the CPU and the surrogate would be far slower "
            "than the classical force field it replaces",
        )
    elif has_cuda_platform != torch_cuda:
        report.add(
            "device agreement",
            "warn",
            f"OpenMM CUDA platform present={has_cuda_platform} but torch CUDA="
            f"{torch_cuda}; the ML and MM halves would run on different devices",
        )
    else:
        report.add("device agreement", "ok", f"OpenMM and torch agree on {device}")


def check_potential(report: Report, config) -> bool:
    from .mace_surrogate import MACESurrogateError, build_ml_potential

    started = time.perf_counter()
    try:
        build_ml_potential(config)
    except (MACESurrogateError, ImportError, KeyError, ValueError) as error:
        report.add(
            "ml potential",
            "fail",
            f"{config.potential_name}: {error}",
        )
        return False
    report.add(
        "ml potential",
        "ok",
        f"{config.potential_name} constructed in "
        f"{time.perf_counter() - started:.1f} s",
    )
    return True


def check_mixed_system(report: Report, config, platform_name: str) -> Any:
    import openmm.unit as unit

    from .mace_surrogate import (
        INTERPOLATION_PARAMETER,
        MACESurrogateError,
        attach_mace_to_context,
    )
    from .mace_surrogate.testsystems import build_reference_system

    fixture = build_reference_system()
    classical = fixture.potential_energy()
    context = fixture.context(platform=platform_name)
    started = time.perf_counter()
    try:
        handle = attach_mace_to_context(
            context, fixture.topology, config, fixture.ligand_atoms
        )
    except (MACESurrogateError, Exception) as error:  # noqa: BLE001
        report.add("mixed system", "fail", f"{type(error).__name__}: {error}")
        return None
    elapsed = time.perf_counter() - started
    report.add(
        "mixed system",
        "ok",
        f"{len(handle.region)} ML atoms, built and attached in {elapsed:.1f} s",
        ml_atoms=len(handle.region),
        build_seconds=elapsed,
    )

    context.setParameter(INTERPOLATION_PARAMETER, 0.0)
    at_zero = (
        context.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoule_per_mole)
    )
    context.setParameter(INTERPOLATION_PARAMETER, 1.0)
    at_one = (
        context.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoule_per_mole)
    )
    difference = abs(at_zero - classical)
    tolerance = 1.0e-3 + 1.0e-6 * abs(classical)
    report.add(
        "classical limit",
        "ok" if difference <= tolerance else "fail",
        f"lambda_interpolate=0 gives {at_zero:.6f} kJ/mol against the "
        f"untouched force field's {classical:.6f} (difference {difference:.2e}); "
        "the fallback is only correct if these agree",
        classical_kj_per_mol=classical,
        interpolated_kj_per_mol=at_zero,
    )
    report.add(
        "surrogate energy",
        "ok",
        f"lambda_interpolate=1 gives {at_one:.3f} kJ/mol",
        surrogate_kj_per_mol=at_one,
    )
    return context


def check_switch(report: Report, context) -> None:
    from .mace_surrogate import DualHamiltonianSwitch

    switch = DualHamiltonianSwitch(context)
    latency = switch.measure_latency(samples=64)
    report.add(
        "hamiltonian switch",
        "ok" if latency["median_ms"] < 1.0 else "warn",
        f"median {latency['median_ms']:.4f} ms, max {latency['max_ms']:.4f} ms "
        f"over {latency['samples']} switches (a Context rebuild is 200-500 ms)",
        **latency,
    )


def check_committee(report: Report, config) -> None:
    from .mace_surrogate import CommitteeUnavailable, MACECommittee, MACEUQMonitor
    from .mace_surrogate.testsystems import (
        LIGAND_ATOMIC_NUMBERS,
        distorted_ligand_coordinates,
        ligand_coordinates_angstrom,
    )

    if not config.has_committee:
        report.add(
            "committee",
            "warn",
            f"{config.committee_size} model(s) configured; a force-variance "
            "estimate needs two or more. The geometry guard will be the only "
            "out-of-distribution detector. Produce a committee with csbrt-mace-al.",
        )
        return

    try:
        committee = MACECommittee(
            config.committee_model_paths,
            device=config.device,
            precision=config.precision,
        )
    except CommitteeUnavailable as error:
        report.add("committee", "fail", str(error))
        return

    compatibility = committee.check_compatibility()
    if not compatibility["compatible"]:
        report.add(
            "committee compatibility",
            "fail",
            "members do not share a cutoff radius or element table "
            f"(r_max values {compatibility['r_max_values']}); MACE requires "
            "them to, so build the committee from fine-tunes of one foundation "
            "model",
            **{k: v for k, v in compatibility.items() if k != "models"},
        )
        return
    report.add(
        "committee compatibility",
        "ok",
        f"{committee.size} members share r_max="
        f"{compatibility['r_max_values'][0]}",
    )

    numbers = list(LIGAND_ATOMIC_NUMBERS)
    relaxed = ligand_coordinates_angstrom()
    try:
        committee.warm_up(numbers, relaxed)
        monitor = MACEUQMonitor(config, committee=committee)
        calm = monitor.evaluate(relaxed, numbers)
        wild = monitor.evaluate(distorted_ligand_coordinates("stretch"), numbers)
    except Exception as error:  # noqa: BLE001
        report.add("committee inference", "fail", f"{type(error).__name__}: {error}")
        return

    statistics = committee.statistics()
    report.add(
        "committee inference",
        "ok",
        f"{statistics['mean_seconds_per_evaluation'] * 1e3:.1f} ms per "
        f"evaluation over {statistics['evaluations']} call(s)",
        **{k: statistics[k] for k in ("evaluations", "mean_seconds_per_evaluation")},
    )
    report.add(
        "committee sensitivity",
        "ok" if wild.sigma_f_max_ev_per_ang > calm.sigma_f_max_ev_per_ang else "warn",
        f"sigma_F is {calm.sigma_f_max_ev_per_ang:.4f} eV/A on the relaxed pose "
        f"and {wild.sigma_f_max_ev_per_ang:.4f} on a torn one "
        f"(threshold {config.uq_force_threshold_ev_per_ang:.4f})",
        relaxed_sigma_f=calm.sigma_f_max_ev_per_ang,
        distorted_sigma_f=wild.sigma_f_max_ev_per_ang,
    )


def check_geometry_guard(report: Report, config) -> None:
    from .mace_surrogate import GeometryGuard, MACEUQMonitor
    from .mace_surrogate.testsystems import (
        LIGAND_ATOMIC_NUMBERS,
        LIGAND_BONDS,
        distorted_ligand_coordinates,
        ligand_coordinates_angstrom,
    )

    guard = GeometryGuard(
        LIGAND_ATOMIC_NUMBERS,
        min_distance_ang=config.geometry_min_distance_ang,
        max_bond_scale=config.geometry_max_bond_scale,
        bonds=LIGAND_BONDS,
    )
    monitor = MACEUQMonitor(config, geometry_guard=guard)
    relaxed = monitor.evaluate(ligand_coordinates_angstrom())
    clash = monitor.evaluate(distorted_ligand_coordinates("clash"))
    stretch = monitor.evaluate(distorted_ligand_coordinates("stretch"))
    passed = (not relaxed.is_ood) and clash.is_ood and stretch.is_ood
    report.add(
        "geometry guard",
        "ok" if passed else "fail",
        "relaxed pose accepted, clashed and torn poses both intercepted"
        if passed
        else f"relaxed={relaxed.is_ood} clash={clash.is_ood} stretch={stretch.is_ood}",
    )


def options(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="mace-off23-small",
                        help="openmm-ml foundation model to test")
    parser.add_argument("--model-path", default=None,
                        help="Locally trained .model to test instead")
    parser.add_argument("--committee", action="append", default=[], metavar="MODEL",
                        help="Committee member; repeat for each")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--openmm-platform", default=None,
                        help="OpenMM platform for the mixed-system check "
                             "(default: CUDA when available, else Reference)")
    parser.add_argument("--json", type=Path, default=None,
                        help="Write the full report here")
    parser.add_argument("--skip-model", action="store_true",
                        help="Skip everything that needs model weights; check "
                             "only the environment and the pure-python paths")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    opt = options(argv)
    report = Report()
    print("csbrt MACE surrogate preflight\n" + "=" * 46, flush=True)

    versions, available = check_imports(report)
    check_compute(report, opt.device)

    from .mace_surrogate import MACEConfig

    config = MACEConfig(
        enabled=True,
        model_name=opt.model,
        model_path=opt.model_path,
        committee_model_paths=tuple(opt.committee),
        device=opt.device,
        precision="double" if opt.device == "cpu" else "single",
    )

    check_geometry_guard(report, config)

    if opt.skip_model:
        report.add("ml potential", "skip", "--skip-model")
    elif {"openmmml", "openmm"} <= available:
        platform_name = opt.openmm_platform
        if platform_name is None:
            import openmm

            available = {
                openmm.Platform.getPlatform(i).getName()
                for i in range(openmm.Platform.getNumPlatforms())
            }
            platform_name = "CUDA" if "CUDA" in available else "Reference"
        if check_potential(report, config):
            context = check_mixed_system(report, config, platform_name)
            if context is not None:
                check_switch(report, context)
        check_committee(report, config)
    else:
        report.add("ml potential", "skip", "openmm/openmm-ml unavailable")

    payload = report.to_dict()
    payload["config"] = config.to_dict()
    if opt.json:
        opt.json.parent.mkdir(parents=True, exist_ok=True)
        opt.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"\nreport written to {opt.json}", flush=True)

    print("=" * 46, flush=True)
    if payload["ok"]:
        print("preflight PASSED", flush=True)
        return 0
    print(
        "preflight FAILED: "
        + "; ".join(f"{c['check']} ({c['detail']})" for c in report.failed),
        flush=True,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
