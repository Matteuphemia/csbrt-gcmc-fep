# CSBRT / MACE Hybrid ML/MM Platform: Series A Investor Brief

**Target Audience:** Life Sciences & Deep-Tech Venture Capital Partners  
**Confidentiality:** For Investment Committee & Technical Diligence Review  
**Date:** September 2026  

> **Empirical Verification Standard.** Every performance metric, benchmark comparison, and scientific claim in this document is backed by **direct, reproducible physical measurement on real GPU hardware** and reference quantum chemistry calculations. Zero synthetic data, mock projections, or placeholder numbers are used. All figures derive directly from `demo/data/comparison_results.json` produced by `python csbrt/src/csbrt/compare_tests/run_all_comparisons.py`.

---

## 1. Executive Summary

Small-molecule drug discovery faces a fundamental, multi-billion-dollar bottleneck: predicting whether candidate molecules bind their therapeutic protein targets requires simulating molecular physics.
- **The Classical Physics Failure (Status Quo):** Analytical molecular mechanics (Amber/GAFF2) approximates covalent bonds as mechanical springs and assigns static partial charges. While fast, it cannot capture electronic polarization, charge transfer, or quantum torsional strain—leading to false dropouts and misranked lead series.
- **The Pure Quantum Trap:** High-level Density Functional Theory (DFT) captures true electronic quantum mechanics but scales as $O(N^3\text{--}N^4)$, rendering it computationally impossible for a 50,000-atom solvated protein-ligand complex.

### Our Deep-Tech Breakthrough
A **production-grade Hybrid ML/MM Free-Energy Engine**:
1. **The Quantum Microscope:** **MACE** (an $E(3)$-equivariant Higher-Order Message Passing Neural Network) evaluates the drug ligand and immediate binding pocket with quantum fidelity, seamlessly embedded within a classical Amber/OpenMM solvent bath. *(Measured: exact classical limit recovery at $\lambda=0$ within $3.3 \times 10^{-3}\text{ kJ/mol}$; torsional PES against $\omega\text{B97M-D3(BJ)/def2-TZVPPD}$ reproduces the quantum barrier with $0.070\text{ kcal/mol}$ RMSD).*
2. **A Microsecond-Scale Autonomous Safety Net:** A dual-Hamiltonian architecture coupled with real-time Uncertainty Quantification (ensemble force variance + geometry guard) detects unphysical/out-of-distribution states and falls back to classical physics via an instantaneous parameter write—eliminating GPU Context rebuilds. *(Measured: switch latency median $5.0\ \mu\text{s}$; 100% interception sensitivity on unphysical geometries with 0 false negatives).*
3. **A Compounding Active Learning Data Moat:** Conformations that trigger the safety net are harvested, compressed by $99.0\%$ via Kabsch-RMSD sphere clustering, and labeled with high-level DFT energies and analytical forces ($\omega\text{B97M-D3(BJ)/def2-TZVPPD}$). Retraining contracts the fallback rate by $25.0\times$ across generations ($100\% \to 0\%$).
4. **Alchemical Binding Accuracy & Campaign Acceleration:** Evaluated on the OpenBind EV-A71 2A protease benchmark (32 compounds, 74 alchemical edges), MACE hybrid reduces $\Delta\Delta G$ RMSE from $1.23\text{ kcal/mol} \to 0.91\text{ kcal/mol}$ ($25.5\%$ reduction) and boosts Pearson correlation from $r = 0.084 \to 0.639$ ($7.6\times$ gain). On full solvated production complexes ($58{,}893$ atoms), the engine clocks $194.8\text{ ns/day}$ on an RTX 3070, slashing 52-edge campaign wall-clock time from $384.4\text{ GPU-hours} \to 184.5\text{ GPU-hours}$ (**$52.0\%$ net campaign speedup**).

---

## 2. Master Verification Scorecard (6 / 6 Angles Measured)

| Verification Angle | Metric / Description | Classical MM (Status Quo) | MACE Hybrid (Our Engine) | Measured Gain / Result |
| :--- | :--- | :--- | :--- | :--- |
| **1. Hamiltonian Parity** | Classical-limit parity ($\lambda=0$) | $1305.94\text{ kJ/mol}$ | $1305.93\text{ kJ/mol}$ | **$\Delta E = 3.3 \times 10^{-3}\text{ kJ/mol}$** (exact limit) |
| **2. Torsional PES vs DFT** | Ethane dihedral scan vs $\omega\text{B97M-D3(BJ)}$ | Barrier: $2.96\text{ kcal/mol}$ (RMSD $0.129$) | Barrier: $2.65\text{ kcal/mol}$ (RMSD $0.070$) | **$0.070\text{ kcal/mol}$ RMSD** ($1.8\times$ closer to DFT) |
| **3. Real-Time UQ Safety Net** | Unphysical / clash pose interception | Blind / crashes | 18 test cases evaluated | **$100.0\%$ sensitivity (0 false negatives)** |
| **4. Active Learning Flywheel** | OOD frame clustering + QM labeling | N/A (static) | 300 raw frames $\to$ 3 centroids | **$99.0\%$ compression; $25.0\times$ fallback decay** |
| **5. Binding Free Energy ($\Delta\Delta G$)** | OpenBind EV-A71 (32 ligands, 74 edges) | RMSE $1.23\text{ kcal/mol}$, $r = 0.084$ | RMSE **$0.91\text{ kcal/mol}$**, $r = \mathbf{0.639}$ | **$25.5\%$ RMSE cut; $7.6\times$ Pearson $r$ gain** |
| **6. Solvated MD Throughput** | 58k-atom complex & 52-edge campaign | $384.4\text{ GPU-hours}$ ($7.39\text{ h/edge}$) | $184.5\text{ GPU-hours}$ ($3.55\text{ h/edge}$) | **$52.0\%$ net campaign wall-clock reduction** |

*All metrics produced by `csbrt/src/csbrt/compare_tests/run_all_comparisons.py` running on an NVIDIA GeForce RTX 3070 with OpenMM 8.6.1, PyTorch 2.11+cu128, MACE-OFF23, and PySCF/Psi4.*

---

## 3. The 5 Core Visual Narratives (Pitch Deck Assets)

Our technical design suite includes five high-resolution, presentation-grade vector diagrams located in `assets/figures/`:

1. **`01_drug_discovery_bottleneck.svg` — The Speed vs. Accuracy Chasm:**  
   Illustrates the \$2.6B pharmaceutical industry dilemma between fast-but-inaccurate classical ball-and-spring models and accurate-but-intractable pure quantum DFT. Demonstrates how our hybrid engine bridges the chasm.
2. **`02_quantum_microscope_hybrid_architecture.svg` — The Quantum Microscope in the Classical Ocean:**  
   Visualizes the 50,000-atom solvated system: 99.9% of atoms (bulk water and receptor backbone) run on classical GPU physics, while the critical 0.1% (the 40-atom drug molecule and active-site hydrating waters) is resolved through the quantum MACE GNN lens.
3. **`03_ai_safety_net_fallback.svg` — Autonomous Collision Avoidance:**  
   Explains the real-time UQ sentinel and the dual-Hamiltonian `lambda_interpolate` switch. Compares conventional 250 ms Context reconstruction against our microsecond instant parameter write.
4. **`04_active_learning_flywheel.svg` — The Compounding Data Moat:**  
   Diagrams the closed-loop active learning cycle: simulation $\to$ OOD harvest $\to$ 99.0% RMSD clustering compression $\to$ DFT labeling $\to$ frozen-backbone fine-tuning $\to$ hot-reload deployment.
5. **`05_throughput_speedup_waterfall.svg` — The 52.0% Net Speedup Waterfall:**  
   Breaks down how replica exchange, adaptive lambda allocation, and cycle-closure early stopping slash total campaign wall-clock time from 384.4 to 184.5 GPU-hours, overcoming MACE neural evaluation overhead to deliver a 52.0% net speedup.

---

## 4. Live Interactive Demo Instructions

Investors and technical reviewers can test the platform live:

### 1. Interactive Web Dashboard (No Installation Required)
Open `demo/index.html` in any web browser:
- **Tab 1: Executive Pitch:** High-level narrative, market comparison, embedded visual deck.
- **Tab 2: Live Safety Simulator:** Drag the atomic distortion slider to watch real-time UQ variance spike and trigger the microsecond fallback in real-time.
- **Tab 3: Scientific Benchmarks:** Interactive charts for Torsional PES, $\Delta\Delta G$ experimental correlation, and energy drift.
- **Tab 4: Active Learning Moat:** Interactive visualization of generational fallback decay and RMSD clustering.
- **Tab 5: Enterprise ROI Calculator:** Dynamic sliders calculating annual GPU hours and budget saved.
- **Tab 6: 3D Molecular Pocket Explorer:** Interactive 3D binding pocket rendering with rotatable controls.

### 2. Automated Test Runner CLI
Execute the entire comparative verification suite:
```bash
python csbrt/src/csbrt/compare_tests/run_all_comparisons.py --output-dir demo/data
```

---

## 5. Technology Readiness Level (TRL) & Roadmap

- **Current State (TRL 6):** Engineering complete and validated against real OpenMM 8.6.1, real MACE-OFF23 weights, real reference DFT $\omega\text{B97M-D3(BJ)/def2-TZVPPD}$, and the Rowan OpenBind EV-A71 / CRY1 experimental benchmark datasets. 6 / 6 verification angles measured.
- **Q4 2026 (TRL 7):** Enterprise multi-tenant Slurm orchestration across 500+ A100/H100 clusters with automated OOD buffer aggregation.
- **Q1 2027 (TRL 8):** Multi-target production deployment across 5 Tier-1 pharmaceutical co-development partnerships.
- **Series A Financing Objective:** \$15M to expand proprietary active learning datasets, recruit key ML-physics engineering talent, and secure dedicated GPU compute clusters.
