# MD work handover — CRY1/AN139 and EV-A71 2A, Loch GCMC and relative FEP

Prepared 2026-08-10. Nine reproducible notebooks, in Ludovic's style, covering
every piece of molecular-dynamics work in `intern_projects/project_1` and
`intern_projects/project_2`.

Every notebook has been **executed end to end** in `cry-loch-babel` with zero
errors, and its outputs are saved in the file. Analysis notebooks re-derive their
numbers from retained artefacts, so re-running them is a regression test, not a
re-typing exercise.

```bash
mamba activate cry-loch-babel        # has jupyter, sire 2025.4, loch 2025.2, openmm 8.4
jupyter lab                          # open in numeric order
```

---

## The notebooks

| # | Notebook | Reproduces / needs |
|---|---|---|
| 00 | `00_Protocol_and_Provenance.ipynb` | the exact executed protocol, parsed from `loch_ludovic_common.py` by AST and asserted against the documented totals. **Stdlib only** — runs on a login node. Start here. |
| 01 | `01_CRY1AN139_System_Preparation.ipynb` | Ludovic's `System_Prep.ipynb` as executed. Audits the retained reference preparation by default (`RUN_PREPARATION = False`); set `True` to run the real toolchain. |
| 02 | `02_Loch_GCMC_Equilibration_and_Production.ipynb` | UVT1 → NPT → UVT2 → 10 ns production. Drives the *real* pipeline helpers, so it cannot drift from the cluster. `MODE = "describe" \| "smoke" \| "full"`. |
| 03 | `03_Boundary_Audit_and_Acceptance.ipynb` | the four-level acceptance gate, run on `preparation/` + `equilibration/`. Reproduces the known-good July 2026 benchmark exactly. |
| 04 | `04_Postprocessing_and_MultiReplica_Water_Clusters.ipynb` | `LochPostprocess.py` parity, plus **Loch vs GRAND** water clusters over 4 Loch + 12 Ludovic replicas and 9 crystal waters. All local. |
| 05 | `05_EV71_Hydration_Site_Series.ipynb` | the 32 × 6 = 192-run hydration-site consistency analysis. All local. |
| 06 | `06_Relative_FEP_over_Loch_Waters.ipynb` | the 52-edge FEP network, its frame defect, and its environment failure. All local. |
| 07 | `07_Stability_Diagnostics_and_Energy_Parity.ipynb` | energy parity and the UVT1/UVT2 incident, from retained diagnostic JSON. All local. |
| 08 | `08_Do_GCMC_Waters_Help_Docking.ipynb` | the hydration-vs-docking measurements. All local. |

`_build/` holds the generator scripts (`nbtools.py` + one per notebook). Rebuild
with `cd _build && python3 build_NN_*.py`. `protocol_record.json` and
`nb04_outputs/` are notebook outputs.

---

## The protocol, in one table

Derived and asserted in notebook 00. Constants live in
`scripts/loch_ludovic_common.py`; the EV71 port re-declares them in
`ev71_loch_common.py`.

| Stage | Order | GCMC attempts | MD | CSV records | Ghost lines |
|---|---|---|---|---|---|
| UVT1 | 45 ghosts → delete sphere waters → minimise → **velocities** → 10,000 attempts → 100 × (1,000 attempts → report → 5 MD steps) | 110,000 | 500 steps = 1 ps | 5 | 100 |
| NPT | rebuild ghost-free, no restraints, 1 bar | 0 | 1,000,000 = **2 ns** | 400 | 0 |
| UVT2 | **fresh** 45 ghosts → 125 × (800 attempts → report → 2,000 MD steps) | 100,000 | 250,000 = 0.5 ns | 500 | 125 |
| Production | **fresh** 45 ghosts → 2,500 × (2,000 MD steps → 200 attempts → report + DCD frame) | 500,000 | 5,000,000 = **10 ns** | 10,000 | 2,500 |

300 K · 2 fs · friction 1 ps⁻¹ · PME 12 Å cutoff with 10 Å switching ·
dispersion correction **off** · Ewald tol 5e-4 · h-bond constraints · COM removal
every step · ligand-centred 10 Å sphere · TIP3P µ_ex −6.09 kcal/mol, standard
volume 30.345 Å³ · Cα restraint coefficient 100 in UVT1/UVT2/production, **none**
in NPT · barostat every 25 steps · mixed CUDA precision · seeds
`20260714 + {0,1,2,3}`, replica blocks spaced 1,000.

Two places where Ludovic's **comments** disagree with his **code** — the code wins:
NPT is 2 ns (not 1), and production runs MD *before* GCMC, writing the DCD frame
and ghost line *after* the move.

`batch_size=50` is Loch's parallel trial width. It changes no attempt count.

### Reference run (July 15 2026) — a regression benchmark, not an acceptance target

| Boundary | Observed |
|---|---|
| prepared AMBER | 64,109 atoms · 18,800 waters · 59 ligand atoms · 476 Cα · 0 all-zero waters |
| UVT1 handoff | 18,809 waters · state-0 = 36 · net **+9** |
| NPT handoff | 18,809 waters · 300.28 ± 1.15 K · 1.0185 ± 0.0024 g/mL · volume → 642 nm³ |
| UVT2 handoff | 64,127 atoms · 18,806 waters · state-0 = 48 · net **−3** |
| production (4 replicas) | 2,500 frames · 58k–73k water observations · 277–299 clusters |
| wall time | prep 122 s · equil 1,362 s · production 4,714 s · post 838 s · **total ~1.96 h** (RTX 2080 Ti) |

Water counts and acceptance histories are stochastic. Topology invariants, report
counts and operation order are not.

---

## How to run it

```bash
# CRY1-AN139, one replica end to end
cd "$HOME/cry"
export SOURCE_PDB="$HOME/cry/ludovic-workflow-gcmc/Other/MD/CRY1AN139_HOLO.pdb"
sbatch --export=ALL,REPLICA=1,SEED=20260714 scripts/loch_full_pipeline.slurm

# 12 independent replicas
./scripts/submit_loch_replicas.sh 12

# EV71 chemical series, 32 ligands x 6 replicas (dry-run first)
cd "$HOME/cry/project_2"
scripts/submit_ev71_density_series.sh \
    --input-folder openbind_ev71_2a_pyrrolidine_benchmark_release \
    --replicates 6 --max-concurrent 8 --dry-run

# relative FEP: gate on a real GPU node FIRST
srun --gres=gpu:1 --pty scripts/preflight_fep.sh
scripts/submit_fep_edges.sh --manifest fep_manifest.tsv --batch 12 \
    --rowan-edges fep_edges/rowan_xtal_edges_full.tsv
```

Call path: `loch_full_pipeline.slurm` → `prepare_ludovic_native.py` →
`LochEquilibration.py` → `LochProduction.py` → `LochPostprocess.py`, with
`loch_ludovic_common.py` shared by both simulation stages.

**Do not substitute** `scripts/Equilibration.py` / `Production.py` (a separate
rewritten GRAND workflow, `cry-gcmc` env), `ludovic_replica.slurm` (Ludovic's
original external GRAND scripts), or `loch_cry_smoke*.py` / `export_cry_for_loch.py`
(pre-Ludovic development history over the abandoned OpenFF-prepared input family —
its constraint count is 41,429 against 60,230 for the native-AMBER path, i.e. a
different system).

Environments: `cry-loch-babel` for endpoints, `automated-fep` for SOMD2 (never
merge them), `csbrt` for the unified single-environment workflow. Activate with
`$HOME/miniforge3/bin/mamba` and its bash shell hook under `set +u`; never
`source conda.sh`.

---

## Results

### 1 · Loch reproduces GRAND's water placement (notebooks 04, 07)

Energy parity at identical coordinates: **−0.00098 kJ/mol** on a total of
8.892 × 10⁸ — a relative difference of −1.1 × 10⁻¹². Same force field, cutoffs, switching, dispersion,
Ewald tolerance and constraints.

Water clusters, mapped into the crystal frame by Cα Kabsch:

| Comparison | Median nearest-neighbour distance |
|---|---|
| GRAND → GRAND (12 replicas) | 0.989 Å |
| Loch → Loch (4 replicas) | 1.056 Å |
| **Loch → GRAND** | **1.012 Å** |
| random → GRAND (null) | 1.453 Å |

Crystal-water recall at 1.5 Å: **1.000** for all four Loch replicas, 0.963 mean for
GRAND's twelve (Mann–Whitney p = 0.46). Cross-engine agreement is
indistinguishable from within-engine reproducibility.

This is consistency at the available resolution, not proof of equivalence: n = 4
Loch replicas, 9 crystal waters, and the within-engine spread is the ceiling.

### 2 · EV71 hydration is regionally ligand-sensitive, not a fingerprint (notebook 05)

192/192 runs pass the canonical schedule and topology audits; zero deviations;
prepared poses within 0.00055 Å of their supplied SDF records.

| Claim | Supported? |
|---|---|
| specific pocket regions are ligand-sensitive | **yes** — NG16, NG15, NG08, NG21, NG03 |
| a conserved structural-water candidate exists | **yes** — HS009, occupancy 0.967, bridge fraction 0.468, nearest GLU A:85 |
| averaging 6 replicas suppresses replica noise | **yes** — mean-profile separation 0.094 vs single-run 0.182 |
| each ligand has a reproducible whole-pocket signature | **no** — P(between > within) = 0.547, leave-one-out accuracy 12.5 % (chance 3.1 %), median true rank 12/32 |
| hydration profiles predict affinity | **no** — one exploratory neighbourhood (NG05), derived from the same 32 compounds |

**The methodological point that generalises:** site labels closer than the 1.4 Å
assignment radius can exchange a physical water, so an individual-site "effect" may
be a coordinate-label shift. Always repeat the test on non-overlapping
neighbourhoods and report *that*. HS012 has ICC 0.47 alone and **0.04** once
combined with its 1.02 Å neighbour HS030.

Sphere water counts run 46.6–66.5 per run (instantaneous 36–83). That is normal and
is **not** capped by the 45-ghost buffer — ghosts are trial capacity, the sphere
count includes ordinary physical waters.

### 3 · GCMC waters do not improve docking against crystal (notebook 08)

Paired, 32 EV71 ligands, each in its own medoid production frame:

| Reference | rank-1 mean Δ | p |
|---|---|---|
| vs **medoid** (the MD frame the waters came from) | **−0.31 Å** | **0.028** |
| vs **crystal** (independent) | +0.04 Å | 0.56 |

A direct measurement of **circular improvement**. The medoid itself sits 1.39 Å
from crystal, so hydrated docking chases the MD pose. Restricting to
series-conserved waters — the most promising variant a priori — trends slightly
*worse*.

Also measured: **re-docking cannot create FEP edges** (both gates are pure
chemistry: charge equality, mapped-heavy fraction ≥ 0.50); the residual pose error
is **internal conformation** (~1.2–1.4 Å after removing translation and rotation),
not rigid-body placement; failures are **bimodal** (~10 % catastrophic flips, r = 0.86
between rotation and RMSD); **CNNscore margin is worse than chance** as a confidence
signal; and on the plastic CRY pocket, cross-docking collapses (self 1.45 Å → cross
4.09 Å, p = 0.040) while EV71 barely notices, driven by pocket side chains
(0.96 Å vs 0.19 Å RMSD) that **flexible docking does not fix** — its oracle sub-2 Å
count falls 11/13 → 6/13 at ~15× the runtime.

What *is* worth building: a **scaffold-consensus gate**. Leave-one-out scaffold
deviation, AUC 0.818, catches 7 of 11 flips with **zero** false positives out of 117
good poses — including all six severe inversions — and needs no crystal reference.
It removes meaningless edges; it cannot add edges.

### 4 · Relative FEP: prepared, never run (notebook 06)

| Stage | Status |
|---|---|
| medoid bound-frame selection | done |
| 52-edge reference network (connected, 21 cycles) | done |
| mapping + merge, all 52 edges | done — mapped-heavy 0.833–1.000, no charge changes |
| 52 bound legs × 11 λ | **FAILED** 2026-07-24, `CUDA_ERROR_UNSUPPORTED_PTX_VERSION (222)` on every window |
| free legs, network fit | never reached |
| frame handling | **fixed 2026-08-10** — see open item 1; pre-fix streams and markers are invalid |

**There is no ΔΔG from this branch.** One blocker remains (the `cuda-version` pin);
the frame defect is closed and guarded.

---

## Open items, in priority order

### 1 · The FEP frame defect — **FIXED 2026-08-10**

Kept here because the pre-fix artefacts are invalid and because one of the two
offsets involved was never documented.

Three coordinate frames are in play, and **neither leg keeps the parameterisation
frame**:

| Object | Frame | Offset from the ligand input frame |
|---|---|---|
| `ligand.prmtop`/`.rst7`/`.mol2`, `ligand_input.sdf` | ligand input | 0 |
| bound system — prepared complex **or** production restart | receptor | **28.97 Å** |
| free system after `solvateOct` | tLEaP's re-centred box | **12.30 Å** |

The 29 Å bound offset was known. The **12.3 Å free offset was not**: `solvateOct`
translates the solute when it builds the box, so the free leg is not in the mol2
frame either. Placing the "correct" unaligned merge there put **47 of 62 ligand
atoms within 2.0 Å of a water oxygen** (closest contact 0.73 Å vs 3.22 Å correct),
in a 36 Å box. The free legs of the 2026-07-24 network were sterically broken too.

**The fix.** `build_leg_merge()` builds **one merge per leg, RMSD-aligned onto the
ligand it replaces in that leg** — one rule, no flags, all three frames. Both merges
are built after both boxes exist, because the free box's ligand position is tLEaP's,
not the mol2's. Measured residual alignment error ~2 × 10⁻¹⁴ Å.
`--align-to-bound-pose` is deprecated and ignored, still accepted so
`fep_edge.slurm` and `fep_prepare.slurm` work unchanged.

**The guard.** `verify_leg_frame()` runs on **both** legs — checking one cannot
distinguish a real fix from alignment disabled everywhere — and records
`frame_check.*` offsets in `fep_preparation.complete.json`. Tolerance
`--max-leg-centroid-offset`, default 1.0 Å; **do not raise it to make a run pass**.
`bound_to_free_separation_angstrom` is *expected* to be ~24 Å: separate boxes.

Verified in notebook 06 §3b–3c, which runs the real script on a null perturbation
and then asserts the guard rejects all three historical mistakes (28.97 Å, 12.30 Å,
23.64 Å) and accepts both correct merges. Both produced streams have zero
ligand–water clashes (closest contact 3.30 Å bound, 3.16 Å free).

**Why it survived its own self-test (§3d).** The 2026-07-31 change was validated with
a null perturbation, which is *mathematically insensitive* to where the ligand sits —
with A ≡ B any environment error cancels between λ=0 and λ=1. That run returned
ΔΔG = −0.0001 kcal/mol, 11/11 windows, zero minimisation failures, **through a free
leg with 33 of 57 atoms overlapping water**. Keep null perturbations for the run
path; use `frame_check` for the geometry.

**What pre-fix runs actually contained**, measured from retained streams:

| Built | Code | Free-leg ligand | Closest water O |
|---|---|---|---|
| ≤ 2026-07-30 | single merge, `align=true` | receptor frame — outside the water box | **31.08 Å** (in vacuum) |
| 2026-07-31 → 08-10 | two merges | mol2 frame, box elsewhere | **0.56–1.85 Å** (overlapping) |

Locally the only numeric FEP results are a 2-window smoke (ΔΔG −1.4×10⁵ kcal/mol,
zero overlap — meaningless by config regardless of frame) and that null
perturbation. Real-edge runs lived on `/home/moshe/cry`, which is not on this
machine and not in the home backups, so **no real ΔΔG can be re-examined here**.

Applied to all seven copies of `prepare_fep.py` — `project_2/scripts`,
`project_2/csbrt/{scripts,src/csbrt}`, `project_2/scripts/fep_for_aldo`,
`project_2/loch_fep_pipeline/scripts`, `project_1/scripts/automated_pipeline`,
`csbrt-gcmc-fep/csbrt/src/csbrt` — with stale `__pycache__` bytecode removed. The
`csbrt-gcmc-fep` copy is modified in its working tree and **not committed**.

**Consequence:** every existing `fep_preparation.complete.json` and `.bss` stream is
invalid and will recompute (the source hash in `implementation_signature` forces
it). That is intended.

### 2 · The FEP environment pin — still open

Pin `cuda-version` ≤ the cluster node driver's CUDA (12.8 here), and pin the
**whole** toolchain — an unpinned `cuda-nvcc` resolves to 12.9. Then run
`preflight_fep.sh` under `srun --gres=gpu:1` **on the target node**; a smoke test on
a box with a newer driver gives a false pass. Keep `cuda-nvvm` if the bound leg
uses Loch GCMC.

### 3 · Repeat the CRY hydration arm with full-profile hosts — still open

**Not stated in the existing write-ups.** The three CRY1 hosts under
`project_2/csbrt/csbrt-run/endpoint/{6kx5,7d0m,7dli}` were run with
`PROFILE=smoke`: UVT1 got 10 MD steps, NPT 500, UVT2 200, production **3 frames**,
and their density analyses saw 2 / 16 / 17 physical water observations total.

The paired dry-vs-hydrated contrast is still valid (both arms share the receptor),
so "hydration did not help CRY" stands for *those* receptors. But the structural
explanation — "the CRY pocket holds only 9–16 waters against ~98 for EV71" — is
**confounded**, because EV71's ~98 came from full 10 ns runs. Part of that gap is
undersampling, not pocket chemistry. Rerun with `PROFILE=full` before relying on it.

The EV71 experiments are unaffected: those medoid frames come from the audited
`full_ludovic_schedule` matrix.

### 4 · Smaller items

- The 32 medoid frames are **not on this machine**. `select_bound_frames.py`
  symlinks by default, so `medoids.tgz` (2.3 KB) archives dangling links. Use
  `tar czhf` or `--copy`; the real payload is ~115 MB.
- GNINA pose-exploration code is only on `feature/pose-exploration` of
  `/home/moshe/csbrt-gcmc-fep` — not on `main`, not in `project_2`. Stale `.pyc`
  files are the only trace on `main`. Use `git show feature/pose-exploration:<path>`.
- The `gcmc_pose` and `endpoint_preparation` Snakemake rules and the DAG end to end
  have **never been executed** (dry-run only). A dry-run is not a smoke test.
- `screen_fep_edges.py` records `pose_rmsd_angstrom` and
  `centroid_distance_angstrom` but does not gate on them. That is the natural place
  to make conserved binding position load-bearing.

---

## Ten things that will mislead you

Each cost real time. The notebook that documents each is named.

1. **Exit status 0 means nothing.** Verify by artefact: expected files present,
   non-zero frames, finite numbers. A SOMD2 window worker can die while the
   top-level command returns 0. (03, 06)
2. **A stage completing is not evidence its output is correct.** NPT ran 2 ns and
   reported a perfect 300.29 K / 1.0185 g/mL trace while carrying waters with zero
   charge and zero LJ. Only the reload-and-audit found it. (07)
3. **Minimisation succeeding is not correctness.** A non-interacting water, or a
   ligand parked in bulk solvent, both minimise beautifully. **The loud failure is
   safer than the quiet one.** (06, 07)
4. **A rejected GCMC move can still destroy the system.** At the failure cycle:
   0 insertions, 0 deletions, and a 1.8 × 10⁷ kJ/mol/nm force — rejection *restored*
   an overlapping water's real parameters. (07)
5. **`profile=smoke` is plumbing validation only.** 3 frames. Never a short
   scientific run — and it silently underlies the CRY hydration result. (02, 08)
6. **The topology PDB and the processed trajectory are in different frames.**
   `LochPostprocess.py` copies the raw production topology verbatim; Ludovic's
   equivalent is post-processing. Using the wrong one dropped apparent Loch
   crystal-water recall from 1.000 to 0.472 with no error raised. Check
   `gcmc_sphere.pdb`'s `MODEL 0` against `MODEL 1`. (04)
7. **UVT1/UVT2 CSV density is not physical density** — the writer includes the 45
   buffer waters while a sampler is live. Use NPT. And UVT1 covers 1 ps, so its
   208 K mean is expected. (03)
8. **Occupancy is not residence time.** GCMC destroys water identity by design.
   Residence time needs a separate fixed-water-number MD analysis. (04, 05)
9. **Measure RMSD by graph matching, never by atom index.** Index-wise reads 2.58 Å
   where graph-matched is 0.74 Å. And if symmetry correction appears to do nothing,
   verify the mapping applied — a silent `IndexError` skip once produced a bogus
   "the error is orientational" conclusion. (08)
10. **`Overall performance: N ns day⁻¹` is not throughput** — it is one window's
    runtime over the whole-leg wall time. 11 × 5 ns at ~700 ns/day is ~2 h, not
    55 ns/day. And SOMD2 **appends** to `log.txt` across runs, so grepping for
    errors returns stale failures; check timestamps and `sacct`. (06)

Two more worth keeping: **weigh base rates before accepting a systemic hypothesis**
(if a theory predicts every edge fails and 51/52 succeeded, it is refuted), and
**the free leg contains no receptor**, so receptor-side explanations cannot explain
a free-leg failure.

---

## Where things live

| What | Where |
|---|---|
| CRY1 pipeline, diagnostics, skills | `project_1/scripts/`, `project_1/skills/loch-ludovic/` |
| CRY1 reference prep + equilibration | `project_1/preparation/`, `project_1/equilibration/` |
| CRY1 completed replicas (4 with postprocessing) | `project_1/cry-loch-multi/rep{6,7,8,9}/` |
| Diagnostic runs | `project_1/uvt1-stability-6172{3,4}/`, `project_1/uvt2-stability-61716/` |
| Energy parity | `project_1/output/*energy_comparison*.json` |
| Ludovic's original workflow + 12 GRAND replicas | `intern_projects/workflow-GCMC-Ludovic (after online tutorials)/` |
| EV71 port, FEP toolchain, csbrt package | `project_2/scripts/`, `project_2/csbrt/` |
| EV71 192-run analysis | `project_1/analysis_outputs/ev71_consistency_20260721/` |
| Docking / hydration experiments | `project_1/analysis_outputs/gnina_hydration_test_20260803/`, `cry_ligand_structures_20260804/` |
| Shareable pipeline bundles | `project_1/pipeline_share_packages/` (+ `ARCHIVE_SHA256SUMS`) |
| Reference docs | `project_1/skills/loch-ludovic/references/{pipeline-map,ludovic-parity,diagnostics,fep,porting}.md` |

`skills/loch-ludovic/references/pipeline-map.md` is the authoritative script map —
read it before editing any Python, and update it when adding or retiring a script.

Not MD, and out of scope here: the FoldGuard MMseqs2/Foldseek screening cascade
(`scripts/foldguard_screen/`, `skills/foldguard-screen/`) and the
`tools/` protein-design checkouts (ProteinMPNN, EvoDiff).
