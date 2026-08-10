# Relative FEP over Loch-equilibrated waters

Operating guide for the relative binding free energy (RBFE) branch that reuses the
Loch endpoint's placed waters. Read this before running, porting, or debugging
FEP. It is separate from endpoint MD/GCMC: a trajectory is not FEP just because it
has a ligand and GCMC waters. FEP needs a mapped perturbable pair, bound and free
legs, lambda windows, and overlap/convergence analysis.

## Environment

- FEP runs in its own Mamba env `automated-fep` from `environment-fep.yml`
  (SOMD2/OpenBioSim 2026.1 plus `cuda-nvvm`). Never install SOMD2 into
  `cry-loch-babel`, and never upgrade the validated endpoint env in place.
- `cuda-nvvm` is required whenever the bound leg uses Loch GCMC (runtime CUDA
  kernel compilation); a bare `nvcc` wrapper alone fails when `cicc` is absent.
  The default FEP path runs GCMC off, so the free-leg-style path does not need it.
- The endpoint prep and the averaging step run in `cry-loch-babel`.

## The bridge: seed FEP from equilibrated waters

The whole endpoint pipeline exists to place the pocket waters, so FEP should start
from that settled state, not a fresh solvation.

- `make_fep_manifest.py --bound-frame production` (default) seeds each edge's bound
  leg from state A's `*-production-final` restart and adds two manifest columns
  (`state_a_bound_prmtop/rst7`). `--bound-frame-root DIR` uses precomputed frames
  instead. `--bound-frame preparation` restores the legacy fresh-solvation frame.
- `prepare_fep.py` builds **one merge per leg, each RMSD-aligned onto the ligand it
  replaces in that leg**, and verifies both. `--align-to-bound-pose` is deprecated
  and ignored; it is still accepted so existing wrappers and manifests keep working.
  See "Three frames, one rule" below — this changed on 2026-08-10 and older
  `fep_preparation.complete.json` markers are invalid.
- Bound-leg GCMC is **off by default** (`submit_fep_edges.sh`/`submit_fep_series.sh`);
  the waters are already placed. `--with-gcmc` re-enables SOMD2 Loch GCMC. The free
  leg never uses GCMC.

## Three frames, one rule

Three coordinate frames are in play, and **neither leg keeps the parameterisation
frame**. Measured on `runs/x7259a-*` and the retained 52-edge setups:

| Object | Frame | Offset from the ligand input frame |
| --- | --- | --- |
| `ligand.prmtop`/`.rst7`/`.mol2`, `ligand_input.sdf` | ligand input | 0 |
| bound system (prepared complex **or** production restart) | receptor | **28.97 A** |
| free system after `solvateOct` | tLEaP's re-centred box | **12.5 A** |

`solvateOct` translates the solute when it builds the octahedral box, so the free
leg is *not* in the mol2 frame either. That second offset was missed originally.
Placing the unaligned merge in the free box put **47 of 62 ligand atoms within
2.0 A of a water oxygen** (closest contact 0.73 A, against 3.22 A for the correctly
placed ligand) — the free legs of the 2026-07-24 network were sterically broken as
well as the bound legs.

**The rule: each leg's merge is RMSD-aligned onto that leg's own ligand.** One rule,
no flags, all three frames handled. Both alignments are rigid-body moves of the same
conformer, so they reproduce each leg's translation exactly (measured residual
~3e-14 A).

`verify_leg_frame()` enforces it on **both** legs — checking one cannot distinguish a
real fix from alignment disabled everywhere — and records
`frame_check.{bound,free}_centroid_offset_angstrom` in the checkpoint. Tolerance is
`--max-leg-centroid-offset`, default 1.0 A. **Do not raise it to make a run pass.**
`frame_check.bound_to_free_separation_angstrom` is *expected* to be large (~24 A):
the legs live in different boxes.

Why this needs a guard rather than care: neither mistake raises on its own. A
bound-leg ligand parked in bulk solvent minimises trivially, and a displaced
free-leg ligand surfaces only later as "could not minimise while simultaneously
satisfying the constraints" — which the section below will otherwise send you to
blame on the perturbable pair.

### A null perturbation CANNOT detect a frame error

This is why the defect survived its own self-test. A null perturbation (A → A) is
the standard minimal check, and it is **mathematically insensitive to where the
ligand sits**: with state A identical to state B, any environment error — vacuum,
overlapping water, wrong pocket — contributes identically at λ=0 and λ=1 and
cancels exactly in ΔΔG.

Measured on the 2026-07-31 `fixtest` self-test (x7259a → x7259a, identity mapping
over all 57 atoms), which was used to validate the two-merge change:

| | value |
| --- | --- |
| free-leg ligand vs its own tLEaP box | **0.56 A** closest water O, **33 of 57 atoms inside 2.0 A** |
| free-leg minimisation failures | **0** |
| windows completed | 11/11 bound, 11/11 free |
| ΔΔG measured | **−0.0001 kcal/mol** |
| ΔΔG true | 0.000 |

It returned the right answer through a severely clashing free leg, so it read as a
pass. For a **real** edge the cancellation fails: dummy atoms appear and disappear
and the perturbed atoms' interactions with their environment change, so a wrong
environment gives a wrong ΔG_free and hence a wrong ΔΔG.

Consequences for how to test this branch:

- Keep the null perturbation for the **run path** (does a leg complete, are all
  Parquets written) — that is all it is good for.
- Use `frame_check` / `verify_leg_frame` for the **geometry**. It does not depend on
  the perturbation being non-trivial, which is exactly why it catches what the null
  test cannot.
- Add a **closest solute–solvent contact** check to any new leg-level smoke test.
  Completion and a plausible ΔΔG are both compatible with a broken free leg.

## Averaging across replicates

When each ligand has several endpoint replicates, precompute one representative
bound frame per ligand. Do **not** average water Cartesian coordinates: waters are
indistinguishable and permute between replicates, so coordinate averaging is
meaningless (a Cartesian mean structure is also non-physical without minimization).
Two legitimate options, both emitting the same `LIGAND/production-final.{prmtop,rst7}`
layout that `--bound-frame-root` consumes:

- `select_bound_frames.py` — the **medoid replicate**: the run whose common-catalog
  hydration-site occupancy is closest to the per-ligand mean. Returns one real
  replicate endpoint (uses all replicates to choose, one run's frame as output).
  Simple and robust; inspect `selection_report.csv` (chosen replicate + spread).
- `build_consensus_frames.py` — **density-based consensus placement** (cf.
  GIST/3D-RISM/WATsite): keep the medoid's protein/ligand/bulk, snap the in-sphere
  pocket waters onto the all-replicate consensus hydration sites (optimal
  assignment), then minimize. Topology is unchanged from the medoid, so it inherits
  the ghost-free guarantee; no prmtop surgery. Differs from the medoid *only* in
  pocket-water placement, which is the clean way to test the effect. In practice the
  displacements are small (well-agreed replicates sit near the consensus peaks
  already), so consensus and medoid are usually close.

## Submission

- `submit_fep_edges.sh --manifest M --batch N` runs the network as one throttled
  `gpu:1` array (`0-(E-1)%N`); each task does prepare→bound→free→analyse. Dependent
  jobs then fit the network (`fep_aggregate.slurm`) and, if `--rowan-edges` is
  given, compare against the benchmark (`fep_compare.slurm`). Chain:
  edge array → aggregate → compare.
- `--batch N` is the concurrency knob and the way to share a cluster: two users
  with `--batch 12` each (and separate `--run-root`) coexist on 24 GPUs. `%N` caps
  one submission; Slurm fair-share arbitrates across users, so batch sizes that sum
  to the GPU count keep it predictable. `--partition/--account/--qos` pass through.
- `submit_fep_series.sh` is the older per-stage dependency-graph submitter
  (separate prep/leg/analyse jobs); equivalent, finer-grained.
- `fep_benchmark.py` wraps averaging → manifest → FEP (and optionally endpoint
  production) as one entry point — the future package console-script.

## Benchmarking and edge selection

- Do not run all-to-all (496 pairs for 32 ligands). Reuse a published reference
  network. The Rowan OpenBind EV-A71 XTAL graph (32 ligands, 52 edges, connected,
  21 cycles) is provided as `fep_edges/rowan_xtal_edges_full.tsv` (+ a 32-edge
  spanning and a 7-edge validation subset). Every edge is then 1:1 comparable to a
  published DDG.
- `compare_to_rowan.py` joins the fitted network against
  `rowan_results_per_edge_wide.csv` (edge DDG) and a per-compound experimental CSV
  (ligand dG), reporting Pearson/Spearman/MUE/RMSE at edge and ligand level.
- To shrink a network by optimal design: `aggregate_fep_network.py` already
  computes the network covariance `pinv(Wᵀ W)`; its diagonal is each ligand's dG
  variance. Greedily add the edge that most reduces `trace(covariance)` (A-optimal;
  Bayesian A-optimality under a Gaussian prior). Xu 2019 JCIM is the reference.
  Prefer small same-scaffold perturbations (reliable); charge/ring-break edges are
  rejected by default.

## Smoke-testing FEP

- A leg is only complete when there is one `energy_traj_*.parquet` per lambda
  window; `run_fep_leg.py` enforces this because a window worker can die while the
  top-level `somd2` exits 0. It is resumable (re-run short-circuits on a valid
  marker; `--restart` from a checkpoint, `--overwrite` only for an uncheckpointed
  attempt).
- A cheap real-edge smoke: build a small perturbable stream (RDKit → antechamber →
  tleap → BSS `matchAtoms`/`merge`), run both legs with a 2-window/4-ps config,
  then `analyse_fep.py`. This exercises mapping/merge on different molecules and
  the analysis path. A **null perturbation** (same molecule A→A) is the guaranteed-
  to-merge minimal case for testing the run path alone.
- A tiny smoke config gives near-zero adjacent-window overlap (~1e-80) and an
  astronomically uncertain DDG. That is expected and is the overlap diagnostic
  working, not a bug. `somd2_config.yaml` (11 windows, 5 ns/leg) is a conservative
  starting schedule; before a full network, run one edge and check per-window
  `ns day⁻¹` and adjacent-window overlap.
- **Smoke on an actual cluster GPU node — a GPU smoke on a machine with a newer
  driver gives a false pass.** The CUDA build must be ≤ the *node* driver's CUDA
  version, or every window fails at context creation with
  `CUDA_ERROR_UNSUPPORTED_PTX_VERSION (222)`, produces zero energy Parquets, and
  the network dies in seconds. This is invisible to a smoke on a box whose driver
  is newer than the cluster's (e.g. a dev box on CUDA 13.x will happily run a 12.9
  build that a 12.8 cluster node rejects) — the failure is driver skew, not
  CPU-vs-GPU. Durable fix: pin `cuda-version` in `environment-fep.yml` to the
  cluster driver's max (12.8 here, matching `cry-loch-babel`). Verification:
  `preflight_fep.sh` under `srun --gres=gpu:1` gates on an OpenMM CUDA context
  (catches PTX 222 in ~2 s) then a tiny real leg; run it on the target node after
  any env change, before submitting a network.

## Operational behaviour that misleads during debugging

These cost real time in July 2026; check them before theorising.

- **SOMD2 appends to `log.txt` across runs.** A failed attempt and a later
  successful one live in the same file, so `grep`-ing for errors happily returns
  stale failures from days earlier. Always check timestamps and cross-check
  `sacct` job IDs/elapsed times before concluding a run failed.
- **`Overall performance: N ns day-1` is not total throughput.** It is one
  window's runtime divided by the whole-leg wall time. Real GPU throughput is the
  per-lambda `complete, speed = ...` line. A leg is `num_lambda x runtime` of
  sampling run serially, so 11 x 5 ns at ~700 ns/day is ~2 h, not 55 ns/day.
- **Checkpoint extension is version-dependent**: `.s3` on older SOMD2, `.npz` on
  2026.1. `run_fep_leg.py` globs both; matching only one silently falls through to
  `--overwrite` and re-runs every completed lambda window.
- **Checkpoints invalidate on source-hash change.** `implementation_signature`
  hashes the pipeline sources, so editing `run_fep_leg.py`/`prepare_fep.py`
  invalidates completed stage markers. Completed *analysis* outputs are unaffected
  (`aggregate_fep_network.py` reads `analysis.json`, not leg markers). When
  retrying one failed array task after a code fix, submit only that array index —
  resubmitting the whole array re-runs everything.
- **One failed array task blocks the whole network.** `aggregate_fep_network.py`
  requires every manifest edge, so a single missing `analysis.json` stops it. Drop
  the edge from a copy of the manifest to analyse the rest; the network usually
  stays connected and only loses a little cycle redundancy.
- **The free leg contains no receptor** (ligand + water only). Receptor-side
  explanations - chain duplication, pocket waters, binding-site geometry - cannot
  explain a free-leg failure. Free-leg minimisation failures
  ("could not minimise while simultaneously satisfying the constraints") point at
  either the **free-leg frame** (check `frame_check` in the marker first — it is one
  number and it is decisive) or the perturbable pair itself: low mapped-heavy
  fraction, many dummy atoms, or constrained bonds on perturbed heavy atoms.
- **A rigid translation of the solute in a periodic box is a symmetry operation —
  but only into *equivalent* solvent.** PBC makes the absolute position irrelevant;
  it does **not** repair an overlap. A merged ligand displaced from the cavity tLEaP
  carved lands *inside* water: the 12.5 A free-leg offset put 47 of 62 atoms within
  2.0 A of a water oxygen. So measure ligand-to-own-periodic-image separation (want
  >8-10 A) **and closest solute-solvent contact** (want >~2.5 A) before calling a
  frame offset benign — the second one is what distinguishes a harmless translation
  from a steric clash.
- **Weigh base rates before accepting a systemic hypothesis.** If a theory
  predicts every edge should fail and 51/52 succeeded, the theory is refuted -
  do not patch it with "they mostly recovered".

## Prep compatibility and deployment

- The EV71 endpoint prep (`prepare_ev71_system.py`) writes exactly what
  `prepare_fep.py` reads: `ligand.{mol2,frcmod,prmtop,rst7}`,
  `{prefix}_solvated.{prmtop,inpcrd}`, and `preparation.complete.json` whose
  signature carries `ligand_charge/ligand_sha256/ligand_id`. Existing
  `rep*/preparation/` dirs are consumed directly — no re-prep.
- The FEP toolchain is deployed in `project_2/scripts` alongside the EV71 endpoint
  scripts; it shares that directory's `pipeline_utils.py` and `ev71_loch_common.py`
  (`select_bound_frames.py`'s ghost-free check imports whichever is present).
