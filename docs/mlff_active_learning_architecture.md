# Architecture Specification: MACE Active Learning & Fallback Loop for `csbrt`

**Target Systems:** Loch GCMC (`ev71_production.py`) & SOMD2 Relative FEP (`run_fep_leg.py`)  
**Foundational References:** Duignan (2024, `docs/duignan2024_potential_nnps.pdf`) & Wang et al. (2024, `docs/wang2024_design_space_mm_mlff.pdf`)  
**Implementation Package:** `csbrt.mace_surrogate`

---

## 1. System Design & Algorithmic Loop

The primary objective is to cut total simulation time by $\sim 50\%$ by utilizing MACE (Multi-Atomic Cluster Expansion) as a fast surrogate evaluator within OpenMM, backed by an active learning runtime uncertainty check that seamlessly falls back to SOMD2 / classical MM whenever the surrogate confidence drops.

```
                    +------------------------------------------+
                    |          MD Coordinate State X_t         |
                    +------------------------------------------+
                                         |
                                         v
                    +------------------------------------------+
                    |        MACE Ensemble / Committee         |
                    |   E_m(X_t), F_{i,m}(X_t) for m=1..M      |
                    +------------------------------------------+
                                         |
                                         v
                    +------------------------------------------+
                    |    Uncertainty Metric Evaluation         |
                    |  sigma_F = max_i std(F_{i,m})            |
                    |  sigma_E = std(E_m)                      |
                    +------------------------------------------+
                                         |
                       +-----------------+-----------------+
                       |                                   |
         [sigma_F <= sigma_thresh]               [sigma_F > sigma_thresh]
                       |                                   |
                       v                                   v
        +-----------------------------+    +-----------------------------+
        |        FAST PATH            |    |       PHYSICS FALLBACK      |
        | Apply MACE forces to        |    | Switch force evaluator to   |
        | OpenMM integrator.          |    | SOMD2 / Classical MM.       |
        | Advance time-step t -> t+dt |    | Log frame to OOD Buffer.    |
        +-----------------------------+    +-----------------------------+
                       |                                   |
                       +-----------------+-----------------+
                                         |
                                         v
                    +------------------------------------------+
                    |        Check Simulation Completion       |
                    +------------------------------------------+
                                         |
                        (Post-Simulation / Background)
                                         v
                    +------------------------------------------+
                    |     Active Learning Feedback Pipeline    |
                    | 1. Cluster & deduplicate OOD frames.     |
                    | 2. Compute ground-truth labels.          |
                    | 3. Fine-tune MACE foundation model.      |
                    | 4. Update surrogate weights for run.     |
                    +------------------------------------------+
```

---

## 2. Component Specifications

### 2.1 Component 1: Hybrid ML/MM System Generator (`mace_mixed_system.py`)
- **Library:** `openmmml` (`openmmml.MLPotential`).
- **Partitioning Strategy:**
  - **Loch GCMC:** The ligand molecule (`resname LIG`) and water molecules currently within the GCMC sphere ($R = 10\text{ \AA}$) are assigned to the ML potential region. Protein residues and bulk waters are assigned to Amber ff14SB and TIP3P.
  - **SOMD2 FEP:** The perturbable ligand core ($A \to B$ alchemical morph) is assigned to the ML potential region; the receptor frame and bulk solvent remain classical.
- **Embedding Scheme:** Mechanical embedding via `potential.createMixedSystem(system, topology, ml_atoms)`.
- **Foundation Models:** `mace-off23-small`, `mace-off23-medium`, or `mace-omol-0`.

### 2.2 Component 2: Uncertainty Quantification Runtime Monitor (`uq_monitor.py`)
- **Metric Formulation:**
  For a committee of $M$ models (typically $M = 4$):
  $$\sigma_{F, \max} = \max_{i \in \text{ML region}} \left[ \frac{1}{M-1} \sum_{m=1}^M \left\| \mathbf{F}_{i, m} - \bar{\mathbf{F}}_i \right\|^2 \right]^{1/2}$$
  $$\sigma_E = \left[ \frac{1}{M-1} \sum_{m=1}^M (E_m - \bar{E})^2 \right]^{1/2}$$
- **Threshold Calibration:**
  - Default Force Threshold: $\sigma_{\text{threshold}} = 0.05\text{ eV/\AA} \approx 1.15\text{ kcal/mol/\AA}$.
  - Default Energy Threshold: $\sigma_{E, \text{threshold}} = 1.0\text{ kcal/mol}$.
- **Performance Requirement:** Tensor operations must be executed directly on the GPU without round-trip CPU memory transfers during dynamics.

### 2.3 Component 3: Physics Fallback Interceptor (`fallback_controller.py`)
- **OpenMM Context Management:**
  Instead of destroying and reconstructing the OpenMM `Context` (which incurs massive CUDA re-initialization and PTX JIT overhead), the system uses a **Dual-Hamiltonian Context** or a switchable `CustomCVForce` weight parameter $w \in \{0, 1\}$:
  $$U_{\text{effective}} = (1 - w) U_{\text{MACE/MM}} + w U_{\text{SOMD2/MM}}$$
- **Fallback Trigger Action:**
  1. Set $w = 1.0$ (instantaneous zero-overhead switch to full classical physics).
  2. Log the out-of-distribution frame coordinates, box vectors, and uncertainty scores to `ood_frames.npz`.
  3. Integrate step using classical forces.
  4. Attempt return to $w = 0.0$ on the subsequent step once coordinates return to a sampled basin, or execute a relaxation buffer of $N_{\text{fallback}}$ steps.

### 2.4 Component 4: Active Learning Data Buffer & Fine-Tuning (`active_learner.py`)
- **OOD Frame Harvester:**
  - Aggregates flagged frames from all concurrent replica exchange or array tasks.
  - Applies pairwise RMSD clustering (cutoff $0.5\text{ \AA}$) to eliminate redundant conformations.
- **Reference Labeling:**
  - Computes single-point energies and forces using SOMD2/AmberTools (or ORCA/DFT reference).
- **Fine-Tuning:**
  - Loads pre-trained `mace-off23`.
  - Runs fine-tuning with conservative learning rate ($\eta = 10^{-4}$ with early stopping) and frozen lower equivariant layers to prevent catastrophic forgetting.
  - Emits versioned checkpoint: `mace_checkpoint_gen{k}.pt`.

---

## 3. Integration Points in `csbrt`

1. **CLI & Config:**
   - Add MACE flags to `csbrt/src/csbrt/cli.py`:
     `--mace-surrogate`, `--mace-model`, `--mace-uq-threshold`, `--mace-device`.
   - Update `config.example.yaml` with the `mlff:` section.
2. **Loch GCMC Hook:**
   - In `csbrt/src/csbrt/ev71_loch_common.py`, intercept `make_dynamics()` and `make_sampler()` to inject the mixed system potential when enabled.
3. **SOMD2 FEP Hook:**
   - In `csbrt/src/csbrt/run_fep_leg.py`, pass the surrogate config options to `somd2_config.yaml` or inject the hybrid potential wrapper into the leg runner.
4. **Checkpointing & Signatures:**
   - Update `pipeline_utils.py` implementation signatures to include the MACE model hash and UQ configuration, ensuring strict reproducibility and no stale checkpoint clashes.

