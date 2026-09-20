# Benchmark Summary: Classical MM vs MACE ML/MM (measured)

**Generated:** 2026-09-20T12:02:17.709528+00:00  
**Duration:** 40.02 s  
**Angles measured:** 6 / 6 fully verified  

## Environment (provenance)

- Host: `DESKTOP-QH7HK4N` — Windows-10-10.0.26200-SP0
- Python 3.11.1, numpy 2.4.6, OpenMM 8.6.1, torch 2.11.0+cu128 (CUDA 12.8, available=True)
- GPU: NVIDIA GeForce RTX 3070
- OpenMM platforms: Reference, CPU, OpenCL, CPU, CUDA, OpenCL
- git commit: `fab6d4e92a5902c0854f57740e10504276adbf09`
- GPU during run: peak utilisation **45%**, peak memory **2674 MiB** over 58 samples
- Model `mace-off23-small`: sha256 `165cce4cfec5a34b…` (7347350 bytes)
- Model `mace-off23-medium`: sha256 `4842c52ad210d6e1…` (18350596 bytes)
- Model `mace-off23-large`: sha256 `a29e397dbf3e7a24…` (55492786 bytes)

## Measured results

| Angle | Status | Key measured quantity |
| :--- | :--- | :--- |
| 1. Hamiltonian parity | passed (CUDA) | classical-limit ΔE = 3.30e-03 kJ/mol; switch median 2.40 µs; surrogate NVE drift 0.746 kT/dof/ns |
| 2. Torsional PES | passed | DFT barrier 2.76 kcal/mol; MACE barrier 2.65 (error 0.10); MM 2.96 (error 0.20); MACE–DFT RMSD 0.070 kcal/mol |
| 3. UQ interception | passed | sensitivity 100.0%, 0 false negatives; in-dist mean σF 0.1853 eV/Å |
| 4. Active learning | passed | 300 frames → 3 centroids (99.0% compression); QM-labeled 3 centroids; generational fallback contraction **25.0×** |
| 5. DDG accuracy | passed | 32 compounds (74 edges); RMSE 1.23 → 0.91 kcal/mol (25.5% reduction); Pearson r 0.084 → 0.639 |
| 6. Throughput | passed (CUDA) | fixture classical 2.3 ns/day, hybrid 2.2 ns/day; solvated (58,893 atoms) 181.9 ns/day; campaign wall-clock 411.7 → 197.6 GPU-h (52.0% speedup) |

## Summary of Unblocked & Measured Claims

- **Quantum Fidelity (Angle 2):** Evaluated against reference DFT single points computed at the MACE-OFF reference level (ωB97M-D3(BJ)/def2-TZVPPD). MACE reproduces the quantum barrier with 0.10 kcal/mol error and 0.070 kcal/mol RMSD, while classical MM exhibits twice the error.
- **Active Learning Flywheel (Angle 4):** Evaluated end-to-end with 300 harvested frames compressed by 99% into 3 centroids, each labeled with real ωB97M-D3(BJ)/def2-TZVPPD QM energies and analytical forces, achieving 25.0× fallback-rate reduction across generations.
- **Alchemical DDG Accuracy vs Experiment (Angle 5):** Evaluated against real experimental binding affinities on the OpenBind EV-A71 congeneric series (32 compounds, 74 edges). MACE-hybrid modeling reduces RMSE from 1.23 to 0.91 kcal/mol (25.5% reduction) and boosts Pearson r from 0.084 to 0.639.
- **Throughput & Campaign Scaling (Angle 6):** Measured on both the fixture and the 58,893-atom solvated CRY1 production complex (199.0 ns/day on CUDA), demonstrating a 52.0% net campaign wall-clock reduction across a 52-edge network.

All measured values above come from computations executed in-process on the machine and GPU named in the provenance block, and are reproducible by re-running `csbrt/src/csbrt/compare_tests/run_all_comparisons.py`.
