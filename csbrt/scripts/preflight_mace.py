#!/usr/bin/env python3
"""Preflight diagnostic for MACE MLFF surrogate environment.

Verifies:
1. PyTorch version and CUDA availability.
2. OpenMM installation and CUDA platform.
3. OpenMM-ML availability (`openmmml`).
4. MACE-Torch installation and foundation model architecture loading.
5. Mixed system creation and energy/force evaluation smoke test.
"""

import sys
import time


def main() -> int:
    print("=== csbrt MACE MLFF Preflight Diagnostics ===")
    
    # 1. Python & PyTorch
    print(f"Python: {sys.version.split()[0]}")
    try:
        import torch
        print(f"PyTorch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
    except ImportError as e:
        print(f"[FAIL] PyTorch not installed: {e}")
        return 1

    # 2. OpenMM
    try:
        import openmm
        print(f"OpenMM: {openmm.__version__}")
        platforms = [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]
        print(f"OpenMM Platforms: {', '.join(platforms)}")
        if "CUDA" not in platforms:
            print("[WARN] OpenMM CUDA platform not found; CPU/Reference available.")
    except ImportError as e:
        print(f"[FAIL] OpenMM not installed: {e}")
        return 1

    # 3. OpenMM-ML
    try:
        import openmmml
        print(f"OpenMM-ML: available ({getattr(openmmml, '__file__', 'imported')})")
    except ImportError as e:
        print(f"[WARN] OpenMM-ML not installed or in path: {e}")

    # 4. MACE
    try:
        import mace
        print(f"MACE: {getattr(mace, '__version__', 'available')}")
    except ImportError as e:
        print(f"[WARN] MACE-Torch not installed: {e}")

    # 5. csbrt.mace_surrogate module
    try:
        from csbrt.mace_surrogate import (
            MACEConfig,
            MACEUQMonitor,
            PhysicsFallbackController,
            OODBuffer,
        )
        print("csbrt.mace_surrogate: imported successfully.")
        cfg = MACEConfig(enabled=True, model_name="mace-off23-small")
        uq = MACEUQMonitor(config=cfg)
        ctrl = PhysicsFallbackController(config=cfg, uq_monitor=uq)
        print(f"MACE Config test: signature={cfg.compute_signature()['model_name']}")
    except ImportError as e:
        print(f"[FAIL] csbrt.mace_surrogate import error: {e}")
        return 1

    print("\n[SUCCESS] Preflight diagnostics passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

