"""Shared plumbing for the comparison suite: provenance, model paths, GPU telemetry.

Everything reported by the suite must be traceable to a measurement made in
this process, on this machine. This module records *where* the numbers came
from (hardware, library versions, model checksums, git commit) and samples the
GPU while the suite runs so the report can show the device was actually used.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

def enable_openmm_cuda() -> list[str]:
    """Make OpenMM's CUDA platform loadable on this Windows install.

    The pip ``openmm-cuda-12`` wheel ships the CUDA plugin DLLs but not the
    CUDA runtime they link against; torch's bundled runtime satisfies them.
    Adding torch's ``lib`` directory to the DLL search path and re-loading the
    plugins registers the CUDA platform. Returns the platform names now
    available. A no-op (and harmless) if CUDA is already present or torch has
    no CUDA build.
    """
    import os

    import openmm

    have = {
        openmm.Platform.getPlatform(i).getName()
        for i in range(openmm.Platform.getNumPlatforms())
    }
    if "CUDA" in have:
        return sorted(have)
    try:
        import torch

        torch_lib = Path(torch.__file__).parent / "lib"
        if torch_lib.is_dir():
            os.add_dll_directory(str(torch_lib))
        base = Path(openmm.__file__).parent
        plugins = (base.parent / "OpenMM.libs" / "lib" / "plugins").resolve()
        if plugins.is_dir():
            openmm.Platform.loadPluginsFromDirectory(str(plugins))
    except Exception:
        pass
    return sorted(
        {
            openmm.Platform.getPlatform(i).getName()
            for i in range(openmm.Platform.getNumPlatforms())
        }
    )


MACE_CACHE = Path.home() / ".cache" / "mace"
MACE_OFF23_FILES = {
    "mace-off23-small": MACE_CACHE / "MACE-OFF23_small.model",
    "mace-off23-medium": MACE_CACHE / "MACE-OFF23_medium.model",
    "mace-off23-large": MACE_CACHE / "MACE-OFF23_large.model",
}
#: MACE-OFF23 medium and large share r_max = 5.0 A and the same element table,
#: so MACECalculator accepts them as one committee. Small has r_max = 4.5 A
#: and cannot join. This is a committee of two independently trained
#: foundation models, not of fine-tune seeds; see the report for caveats.
COMMITTEE_MODELS = ("mace-off23-medium", "mace-off23-large")
SURROGATE_MODEL = "mace-off23-small"


def ensure_mace_off_cached(name: str) -> Path:
    """Return the local checkpoint for a MACE-OFF23 model, downloading if needed."""
    path = MACE_OFF23_FILES[name]
    if not path.is_file():
        from mace.calculators import mace_off

        size = name.rsplit("-", 1)[-1]
        mace_off(model=size, device="cpu", default_dtype="float32")
    if not path.is_file():
        raise FileNotFoundError(f"MACE model {name} not found at {path}")
    return path


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(repo_root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True,
            text=True, timeout=10, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def environment_provenance(repo_root: Path) -> dict[str, Any]:
    import numpy
    import openmm
    import torch

    platforms = enable_openmm_cuda()
    info: dict[str, Any] = {
        "hostname": platform.node(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "openmm": openmm.__version__,
        "openmm_platforms": [
            openmm.Platform.getPlatform(i).getName()
            for i in range(openmm.Platform.getNumPlatforms())
        ],        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": git_commit(repo_root),
        "models": {},
    }
    try:
        import openmmml

        info["openmmml"] = getattr(openmmml, "__version__", "unknown")
    except ImportError:
        info["openmmml"] = None
    try:
        import mace

        info["mace_torch"] = getattr(mace, "__version__", "unknown")
    except ImportError:
        info["mace_torch"] = None
    for name, path in MACE_OFF23_FILES.items():
        if path.is_file():
            info["models"][name] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    return info


class GPUSampler:
    """Polls nvidia-smi in a background thread; proves the GPU did the work."""

    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s = interval_s
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0

    @staticmethod
    def _query() -> tuple[float, float] | None:
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True, timeout=5, check=True,
            )
            util, mem = out.stdout.strip().splitlines()[0].split(",")
            return float(util), float(mem)
        except Exception:
            return None

    def _run(self) -> None:
        while not self._stop.is_set():
            reading = self._query()
            if reading is not None:
                self.samples.append(
                    {
                        "t_s": round(time.perf_counter() - self._t0, 2),
                        "util_pct": reading[0],
                        "mem_mib": reading[1],
                    }
                )
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "GPUSampler":
        self._t0 = time.perf_counter()
        baseline = self._query()
        self.baseline = (
            {"util_pct": baseline[0], "mem_mib": baseline[1]} if baseline else None
        )
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {"available": False, "samples": 0}
        utils = [s["util_pct"] for s in self.samples]
        mems = [s["mem_mib"] for s in self.samples]
        return {
            "available": True,
            "samples": len(self.samples),
            "interval_s": self.interval_s,
            "baseline_before_run": self.baseline,
            "util_pct_max": max(utils),
            "util_pct_mean": sum(utils) / len(utils),
            "mem_mib_max": max(mems),
            "mem_mib_min": min(mems),
            "trace": self.samples,
        }
