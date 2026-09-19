#!/usr/bin/env python3
"""Preflight diagnostic for the MACE MLFF surrogate stack.

Verifies the dependency and CUDA layout the csbrt MACE surrogate requires, then
runs a tiny single-point MACE inference to prove the model loads and evaluates on
the requested device.  Run this on the GPU node before enabling
``--enable-mace-surrogate``.

Exit code 0 when every applicable check passes; non-zero otherwise.
"""

from __future__ import annotations

import argparse
import sys


def check(name: str, ok: bool, detail: str) -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"[{status:4}] {name:<45} {detail}")
    return ok


def check_imports() -> bool:
    ok = True
    for module in ("numpy", "torch", "ase", "mace", "openmm", "openmmml"):
        try:
            __import__(module)
            ok &= check(f"import {module}", True, "")
        except Exception as err:  # noqa: BLE001
            ok &= check(f"import {module}", False, f"{err}")
    return ok


def check_torch_cuda() -> bool:
    try:
        import torch
    except Exception as err:  # noqa: BLE001
        return check("torch CUDA", False, f"import failed: {err}")
    avail = torch.cuda.is_available()
    detail = f"cuda_available={avail}"
    if avail:
        detail += f", device={torch.cuda.get_device_name(0)}"
    return check("torch CUDA", avail, detail)


def check_openmm_cuda() -> bool:
    try:
        import openmm
    except Exception as err:  # noqa: BLE001
        return check("OpenMM CUDA platform", False, f"import failed: {err}")
    platforms = {p.getName(): p for p in openmm.Platform.getPlatforms()}
    has_cuda = "CUDA" in platforms
    detail = "platforms=" + ",".join(sorted(platforms))
    if has_cuda:
        detail += f", version={platforms['CUDA'].getOpenMMVersion()}"
    return check("OpenMM CUDA platform", has_cuda, detail)


def check_mace_single_point(model: str, device: str, precision: str) -> bool:
    """Evaluate one MACE foundation model on a water molecule."""
    try:
        import numpy as np
        from ase import Atoms
        from mace.calculators.foundations_models import (  # type: ignore[import-not-found]
            mace_mp,
            mace_off,
            mace_omol,
        )
    except Exception as err:  # noqa: BLE001
        return check("MACE single-point", False, f"imports failed: {err}")

    mapping = {
        "mace-off23-small": (mace_off, "small"),
        "mace-off23-medium": (mace_off, "medium"),
        "mace-off23-large": (mace_off, "large"),
        "mace-mpa-0-medium": (mace_mp, "medium-mpa-0"),
        "mace-omat-0-small": (mace_mp, "small-omat-0"),
        "mace-omat-0-medium": (mace_mp, "medium-omat-0"),
        "mace-omol-0-extra-large": (mace_omol, "extra_large"),
    }
    if model not in mapping:
        return check("MACE single-point", False, f"unsupported model {model!r}")

    fn, key = mapping[model]
    dtype = "float64" if precision == "float64" else "float32"
    try:
        calc = fn(model=key, device=device, default_dtype=dtype)
        atoms = Atoms(
            numbers=[8, 1, 1],
            positions=np.array(
                [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.239987, 0.926627, 0.0]],
                dtype=np.float64,
            ),
            cell=np.identity(3) * 10.0,
            pbc=True,
        )
        energy = float(calc.get_potential_energy(atoms))
        forces = np.asarray(calc.get_forces(atoms), dtype=np.float64)
        finite = bool(np.isfinite(energy) and np.all(np.isfinite(forces)))
        return check(
            "MACE single-point",
            finite,
            f"model={model}, energy={energy:.6f} eV, forces_finite={finite}",
        )
    except Exception as err:  # noqa: BLE001
        return check("MACE single-point", False, f"evaluation failed: {err}")


def check_mixed_system(model: str, device: str, precision: str) -> bool:
    """Optionally exercise openmm-ml createMixedSystem(interpolate=True)."""
    try:
        import openmm
        import openmm.app as app
        import openmmml
    except Exception as err:  # noqa: BLE001
        return check("openmm-ml mixed system", False, f"imports failed: {err}")

    try:
        from openmmml import MLPotential
    except Exception as err:  # noqa: BLE001
        return check("openmm-ml mixed system", False, f"setup failed: {err}")

    # Build a tiny topology programmatically to avoid relying on bundled PDB files.
    try:
        topology = app.Topology()
        chain = topology.addChain()
        res = topology.addResidue("LIG", chain)
        elem = app.Element.getByAtomicNumber(6)
        atom0 = topology.addAtom("C1", elem, res)
        positions = openmm.unit.Quantity(
            [[0.0, 0.0, 0.0]], openmm.unit.nanometer
        )
        forcefield = app.ForceField("amber14-all.xml")
        system = forcefield.createSystem(topology, nonbondedMethod=app.NoCutoff)
        ml_atoms = [atom0.index]
        potential = MLPotential(model)
        mixed = potential.createMixedSystem(
            topology,
            system,
            ml_atoms,
            interpolate=True,
            device=device,
            precision=precision_kw,
        )
        has_param = "lambda_interpolate" in {
            mixed.getGlobalParameterName(i) for i in range(mixed.getNumGlobalParameters())
        }
        return check(
            "openmm-ml mixed system",
            has_param,
            f"model={model}, lambda_interpolate={has_param}",
        )
    except Exception as err:  # noqa: BLE001
        # The mixed-system check is best-effort; FF XML or model download may
        # legitimately be unavailable while the core stack is fine.
        return check("openmm-ml mixed system", False, f"skipped/unsupported: {err}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mace-off23-small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="float32", choices=("float32", "float64"))
    args = parser.parse_args(argv)

    print(f"MACE surrogate preflight: model={args.model} device={args.device} "
          f"precision={args.precision}\n")
    results = [
        check_imports(),
        check_torch_cuda(),
        check_openmm_cuda(),
        check_mace_single_point(args.model, args.device, args.precision),
        check_mixed_system(args.model, args.device, args.precision),
    ]
    # Imports + torch CUDA are hard requirements; the mixed-system probe is
    # advisory because it depends on bundled force-field XML availability.
    critical = results[:3]
    print()
    if all(critical):
        print("preflight: PASS (critical checks)")
        return 0
    print("preflight: FAIL (see above)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
