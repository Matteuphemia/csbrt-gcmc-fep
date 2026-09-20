"""Quantum Mechanical (QM) calculation engine for reference calculations.

Computes real reference single-point energies and analytical gradients at the
MACE-OFF23 reference level of theory: wB97M-D3(BJ) / def2-TZVPPD.

Executes via QCEngine/Psi4 or PySCF directly or via the configured WSL2
environment, caching results to disk to ensure reproducibility and provenance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

import numpy as np

# Conversion constants
HARTREE_TO_KCAL_MOL = 627.5094740631
HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM

_DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[4] / "demo" / "data" / "qm_cache"


class QMEngine:
    """Interface to QM reference calculations (wB97M-D3(BJ) / def2-TZVPPD)."""

    def __init__(
        self,
        method: str = "wb97m-d3bj",
        basis: str = "def2-tzvppd",
        cache_dir: Path | None = None,
        backend: str = "auto",  # 'auto', 'psi4', 'pyscf'
    ):
        self.method = method.lower()
        self.basis = basis.lower()
        self.cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend

    def _cache_key(
        self,
        coords: np.ndarray,
        atomic_numbers: Sequence[int],
        driver: str,
    ) -> str:
        h = hashlib.sha256()
        h.update(self.method.encode())
        h.update(self.basis.encode())
        h.update(driver.encode())
        h.update(np.asarray(atomic_numbers, dtype=np.int32).tobytes())
        # Round coords to 6 decimal places to prevent float serialization jitter
        h.update(np.round(np.asarray(coords, dtype=np.float64), 6).tobytes())
        return h.hexdigest()

    def compute_energy(
        self,
        coords: np.ndarray,
        atomic_numbers: Sequence[int],
        charge: int = 0,
        spin: int = 0,
    ) -> float:
        """Compute electronic energy in Hartree."""
        coords = np.asarray(coords, dtype=np.float64)
        key = self._cache_key(coords, atomic_numbers, "energy")
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                if "energy_hartree" in data:
                    return float(data["energy_hartree"])
            except Exception:
                pass

        res = self._run_qm_calculation(
            coords=coords,
            atomic_numbers=atomic_numbers,
            driver="energy",
            charge=charge,
            spin=spin,
        )
        energy_hartree = float(res["energy_hartree"])
        res["key"] = key
        cache_file.write_text(json.dumps(res, indent=2), encoding="utf-8")
        return energy_hartree

    def compute_energy_and_forces(
        self,
        coords: np.ndarray,
        atomic_numbers: Sequence[int],
        charge: int = 0,
        spin: int = 0,
    ) -> tuple[float, np.ndarray]:
        """Compute energy (Hartree) and forces (eV/Angstrom).

        Note: forces = -grad(E).
        """
        coords = np.asarray(coords, dtype=np.float64)
        key = self._cache_key(coords, atomic_numbers, "gradient")
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                if "energy_hartree" in data and "forces_ev_angstrom" in data:
                    return (
                        float(data["energy_hartree"]),
                        np.array(data["forces_ev_angstrom"], dtype=np.float64),
                    )
            except Exception:
                pass

        res = self._run_qm_calculation(
            coords=coords,
            atomic_numbers=atomic_numbers,
            driver="gradient",
            charge=charge,
            spin=spin,
        )
        energy_hartree = float(res["energy_hartree"])
        forces_ev_ang = np.array(res["forces_ev_angstrom"], dtype=np.float64)
        res["key"] = key
        cache_file.write_text(json.dumps(res, indent=2), encoding="utf-8")
        return energy_hartree, forces_ev_ang

    def compute_batch_energies(
        self,
        coords_list: Sequence[np.ndarray],
        atomic_numbers: Sequence[int],
        charge: int = 0,
        spin: int = 0,
    ) -> list[float]:
        """Compute energies for multiple geometries efficiently."""
        energies: list[float | None] = [None] * len(coords_list)
        missing_indices: list[int] = []

        # Check cache first
        for idx, coords in enumerate(coords_list):
            c = np.asarray(coords, dtype=np.float64)
            key = self._cache_key(c, atomic_numbers, "energy")
            cache_file = self.cache_dir / f"{key}.json"
            if cache_file.exists():
                try:
                    data = json.loads(cache_file.read_text(encoding="utf-8"))
                    if "energy_hartree" in data:
                        energies[idx] = float(data["energy_hartree"])
                        continue
                except Exception:
                    pass
            missing_indices.append(idx)

        if not missing_indices:
            return [float(e) for e in energies if e is not None]

        # Compute missing geometries in batch via PySCF script
        missing_coords = [coords_list[i].tolist() for i in missing_indices]
        batch_input = {
            "method": self.method,
            "basis": self.basis,
            "charge": charge,
            "spin": spin,
            "atomic_numbers": list(atomic_numbers),
            "geometries": missing_coords,
        }

        batch_output = self._run_wsl_pyscf_batch(batch_input)
        for i, idx in enumerate(missing_indices):
            e_hartree = float(batch_output["energies_hartree"][i])
            energies[idx] = e_hartree
            # Save to cache
            key = self._cache_key(np.asarray(coords_list[idx]), atomic_numbers, "energy")
            record = {
                "key": key,
                "method": self.method,
                "basis": self.basis,
                "charge": charge,
                "spin": spin,
                "atomic_numbers": list(atomic_numbers),
                "coords": coords_list[idx].tolist(),
                "energy_hartree": e_hartree,
                "energy_ev": e_hartree * HARTREE_TO_EV,
                "energy_kcal_mol": e_hartree * HARTREE_TO_KCAL_MOL,
                "engine": batch_output.get("engine", "pyscf"),
            }
            (self.cache_dir / f"{key}.json").write_text(
                json.dumps(record, indent=2), encoding="utf-8"
            )

        return [float(e) for e in energies if e is not None]

    def _run_qm_calculation(
        self,
        coords: np.ndarray,
        atomic_numbers: Sequence[int],
        driver: str,
        charge: int,
        spin: int,
    ) -> dict[str, Any]:
        """Execute single point QM calculation via WSL Psi4 or PySCF."""
        payload = {
            "method": self.method,
            "basis": self.basis,
            "driver": driver,
            "charge": charge,
            "spin": spin,
            "atomic_numbers": list(atomic_numbers),
            "coords": coords.tolist(),
        }

        # Try Psi4 via micromamba in WSL first if requested
        if self.backend in ("auto", "psi4"):
            try:
                res = self._run_wsl_psi4_single(payload)
                if res.get("success"):
                    return res
            except Exception:
                if self.backend == "psi4":
                    raise

        # Fallback to PySCF in WSL
        return self._run_wsl_pyscf_single(payload)

    def _run_wsl_psi4_single(self, payload: dict[str, Any]) -> dict[str, Any]:
        py_code = """
import json, sys
import qcengine as qcng
import qcelemental as qcel
import numpy as np

req = json.loads(sys.stdin.read())
elements = [qcel.periodictable.to_E(z) for z in req['atomic_numbers']]
coords = req['coords']
lines = [f"{req['charge']} {req['spin'] + 1}"]
for el, (x, y, z) in zip(elements, coords):
    lines.append(f"{el} {x:.8f} {y:.8f} {z:.8f}")
mol_str = "\\n".join(lines)

mol = qcel.models.Molecule.from_data(mol_str)
driver = req['driver']
inp = qcel.models.AtomicInput(
    molecule=mol,
    driver=driver,
    model={'method': req['method'], 'basis': req['basis']},
)
ret = qcng.compute(inp, 'psi4')
if not ret.success:
    print(json.dumps({'success': False, 'error': ret.error.error_message}))
    sys.exit(0)

res = {'success': True, 'engine': 'psi4', 'driver': driver}
if driver == 'energy':
    e = float(ret.return_result)
    res['energy_hartree'] = e
    res['energy_ev'] = e * 27.211386245988
    res['energy_kcal_mol'] = e * 627.5094740631
elif driver == 'gradient':
    e = float(ret.properties.return_energy)
    grad = np.array(ret.return_result, dtype=np.float64) # Hartree / Bohr
    forces_ev_ang = -grad * (27.211386245988 / 0.529177210903)
    res['energy_hartree'] = e
    res['energy_ev'] = e * 27.211386245988
    res['energy_kcal_mol'] = e * 627.5094740631
    res['forces_ev_angstrom'] = forces_ev_ang.tolist()

print(json.dumps(res))
"""
        cmd = [
            "wsl",
            "-d",
            "Ubuntu",
            "-u",
            "root",
            "micromamba",
            "run",
            "-n",
            "qm",
            "python",
            "-c",
            py_code,
        ]
        proc = subprocess.run(
            cmd,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=True,
        )
        out = proc.stdout.strip()
        last_line = [ln for ln in out.splitlines() if ln.startswith("{")][-1]
        return json.loads(last_line)

    def _run_wsl_pyscf_single(self, payload: dict[str, Any]) -> dict[str, Any]:
        py_code = """
import json, sys
import numpy as np
from pyscf import gto, dft

req = json.loads(sys.stdin.read())
atomic_numbers = req['atomic_numbers']
coords = req['coords']
z_symbols = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}
lines = []
for z, (x, y, zc) in zip(atomic_numbers, coords):
    lines.append(f"{z_symbols.get(z, 'C')} {x:.8f} {y:.8f} {zc:.8f}")
mol_str = "; ".join(lines)

mol = gto.M(
    atom=mol_str,
    basis=req['basis'],
    charge=req['charge'],
    spin=req['spin'],
)
mf = dft.RKS(mol).density_fit()
mf.xc = req['method']
e = mf.kernel()

res = {'success': True, 'engine': 'pyscf', 'driver': req['driver']}
res['energy_hartree'] = float(e)
res['energy_ev'] = float(e) * 27.211386245988
res['energy_kcal_mol'] = float(e) * 627.5094740631

if req['driver'] == 'gradient':
    g = mf.nuc_grad_method().kernel() # Hartree / Bohr
    forces_ev_ang = -g * (27.211386245988 / 0.529177210903)
    res['forces_ev_angstrom'] = forces_ev_ang.tolist()

print(json.dumps(res))
"""
        cmd = ["wsl", "-d", "Ubuntu", "-u", "root", "python3", "-c", py_code]
        proc = subprocess.run(
            cmd,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=True,
        )
        out = proc.stdout.strip()
        last_line = [ln for ln in out.splitlines() if ln.startswith("{")][-1]
        return json.loads(last_line)

    def _run_wsl_pyscf_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        py_code = """
import json, sys
import numpy as np
from pyscf import gto, dft

req = json.loads(sys.stdin.read())
atomic_numbers = req['atomic_numbers']
geometries = req['geometries']
z_symbols = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}

energies = []
for coords in geometries:
    lines = []
    for z, (x, y, zc) in zip(atomic_numbers, coords):
        lines.append(f"{z_symbols.get(z, 'C')} {x:.8f} {y:.8f} {zc:.8f}")
    mol_str = "; ".join(lines)
    mol = gto.M(
        atom=mol_str,
        basis=req['basis'],
        charge=req['charge'],
        spin=req['spin'],
    )
    mf = dft.RKS(mol).density_fit()
    mf.xc = req['method']
    e = mf.kernel()
    energies.append(float(e))

print(json.dumps({'success': True, 'engine': 'pyscf_batch', 'energies_hartree': energies}))
"""
        cmd = ["wsl", "-d", "Ubuntu", "-u", "root", "python3", "-c", py_code]
        proc = subprocess.run(
            cmd,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=True,
        )
        out = proc.stdout.strip()
        last_line = [ln for ln in out.splitlines() if ln.startswith("{")][-1]
        return json.loads(last_line)

