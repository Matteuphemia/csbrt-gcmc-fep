---
name: loch-ludovic
description: Develop, port, diagnose, and operate Ludovic-order workflows that replace GRAND water moves with Loch GCMC, including CRY1-AN139 and new receptor/ligand systems. Use for Loch equilibration/production/postprocessing, Slurm wrappers, UVT1/NPT/UVT2 handoffs, ghost-water state, RF-to-PME acceptance, stability traces, resumable checkpoints, replica submission, comparisons with original Ludovic GRAND scripts, hydration-convergence gating, or the relative-FEP (SOMD2) branch that reuses equilibrated Loch waters — replicate-averaged bound frames, edge-network submission, and benchmark comparison.
---

# Loch/Ludovic workflow

## Start from the submitted wrapper

Identify the wrapper the user actually submitted before editing Python. Read [pipeline-map.md](references/pipeline-map.md) for the complete script-family map, stage contracts, environments, and authoritative paths. Never infer the execution path from a similarly named file.

Before claiming equivalence with Ludovic's GRAND workflow or changing stage/cycle order, read [ludovic-parity.md](references/ludovic-parity.md) and compare against the original external scripts named there.

Before adapting the pipeline to a new receptor or ligand library, read [porting.md](references/porting.md). Inventory the files actually present, preserve supplied protonation/poses unless the user requests chemistry changes, verify that preservation with executable chemistry checks, and make every boundary independently auditable.

Before running, porting, or debugging the relative-FEP branch — SOMD2 legs seeded from equilibrated Loch waters, replicate-averaged bound frames, edge-network submission, benchmark comparison, and hydration-convergence gating — read [fep.md](references/fep.md).

For the current end-to-end Loch pipeline, trace calls in this order:

1. `scripts/loch_full_pipeline.slurm`
2. `scripts/prepare_ludovic_native.py`
3. `scripts/LochEquilibration.py`
4. `scripts/loch_ludovic_common.py`
5. `scripts/LochProduction.py`
6. `scripts/LochPostprocess.py`

Treat `scripts/Equilibration.py` and `scripts/Production.py` as the separate rewritten GRAND workflow. Treat `scripts/ludovic_replica.slurm` as the launcher for Ludovic's original external GRAND scripts.

## Check Ludovic parity before debugging internals

Make an original-script parity audit the first diagnostic step for every Loch failure. Open the matching stage in Ludovic's original external `GCMC_Grand/Equilibration.py` or `Production.py`, then compare it line by line with the code actually reached from the submitted Loch wrapper. Follow executable statements rather than comments.

Check stage construction and teardown, operation order, ghost creation/removal and reporting, restraints, force configuration, move/MD counts, reporter alignment, and saved handoffs. Treat an omitted, reordered, or numerically different Ludovic operation as the leading failure hypothesis. Correct and test that mismatch before starting stochastic traces, changing physical parameters, or investigating Loch/OpenMM internals. Proceed to deeper diagnostics only when the parity audit finds no explanatory mismatch.

## Diagnose by stage boundary

After the parity gate, read [diagnostics.md](references/diagnostics.md) before interpreting GCMC instability. Preserve raw output folders and inspect JSON before proposing physical-protocol changes.

When reviewing completed preparation or equilibration folders, apply the completed-output acceptance gate in `diagnostics.md`. Check protocol-length/report parity, finite thermodynamic traces, ghost-state arithmetic, and independently reloaded AMBER topology integrity before declaring the system production-ready.

Use the narrowest existing diagnostic:

- Native AMBER stability: `diagnose_direct_native_md.py`, `diagnose_native_md.py`, `compare_native_constraints.py`.
- Native-to-Loch smoke handoff: `diagnose_loch_handoff.py`.
- UVT1 and UVT1-to-NPT handoff: `diagnose_uvt1_stability.py` with `diagnose_uvt1_stability.slurm`.
- Saved-NPT/UVT2 stochastic failure: `diagnose_uvt2_stability.py` with its five-way Slurm array.
- Energy-construction differences: `compare_ludovic_loch_energy.py`.

Prefer a fixed seed, one changed variable, and JSON state snapshots at move/MD boundaries. Record atom identity, logical ghost state, force magnitude, constraint error, RF preacceptances, final PME moves, and coordinate changes. Stop before MD when the post-move force is already unsafe.

## Preserve protocol and compatibility invariants

- Keep the production protocol values centralized in `loch_ludovic_common.py`: 300 K, 12 A PME cutoff, 10 A switching, 10 A ligand-centered sphere, 45 Loch ghosts, 2 fs, TIP3P chemical potential/standard volume, and mixed CUDA precision.
- Keep `batch_size` distinct from total attempts. It controls parallel Loch trials, not protocol length.
- Keep C-alpha restraints and post-construction OpenMM nonbonded configuration consistent across stages.
- With Sire 2025.4, pass Langevin friction through the dynamics property map (`map={"friction": 1 / picosecond}`), not as a direct `friction=` keyword; `_dynamics()` does not accept that keyword even though its OpenMM converter consumes the mapped value.
- At every saved GCMC-to-physical boundary, call `finalise_sampler_system()` and then `save_physical_system()`. Loch 2025.2 changes accepted buffer-water parameters only in the live OpenMM force; finalization materializes accepted buffer waters in the Sire topology and removes inactive ghosts, while the physical saver validates both the in-memory and AMBER-reloaded topology.
- Require identical in-memory/reloaded physical-water counts and zero all-zero charge/LJ waters before NPT or another new sampler. Use generic `save_system()` only for an intentionally ghost-containing trajectory topology.
- Treat `profile=smoke` as plumbing validation only. Require the canonical move/step counts and exact report schedules before labeling a run full; label stride-subsampled clustering as approximate even when the simulation schedule is full.
- Include implementation/protocol, chemistry-tool/data provenance, and input hashes in checkpoint signatures. Hash every required output relative to the marker's stage-directory trust root, reject paths that escape it, and invalidate the old marker before recomputation. A marker must cover parameterization logs, CSV and ghost histories as well as AMBER/DCD files.
- Invalidate stale aggregate audit/timing success at the start of an invocation, record `running`/`completed` state and reused versus recomputed stages, and publish aggregate completion only after the final independent audit. Preserve valid stage markers for resume.
- Give replicas deterministic noncolliding seed blocks, validate positive replica IDs, and record resolved stage seeds and explicit overrides.
- In new postprocessors, image and align before shifting inactive ghosts, then explicitly exclude the per-frame inactive residue IDs from clustering. GRAND's historical shift-before-image order can wrap ghosts back near the ligand. Keep a strict-parity workflow unchanged only when it is clearly labeled, and audit the resulting risk.
- Do not reuse NPT output produced from a faulty UVT1 topology; zero-interaction waters may already have drifted into overlaps.
- Keep the cluster environment at `cry-loch-babel` (Loch 2025.2.0, Sire/BioSimSpace 2025.4.0, OpenMM 8.4.0, CUDA compiler compatible with CUDA 12.8) unless the user explicitly requests an environment migration.
- Use `$HOME/miniforge3/bin/mamba` and its Bash shell hook for every cluster environment activation; do not source `conda.sh` or call `conda activate` in Loch wrappers.
- Keep Bash nounset disabled around the Mamba shell hook and every activation. Its underlying deactivation hooks can reference unset `CONDA_BACKUP_*` variables when a submitted job inherits another active environment. Use `set -eo pipefail`, run the shell hook and activation under `set +u`, then restore `set -u` after activation.
- Do not call endpoint MD/GCMC "FEP". Relative FEP requires a mapped perturbable ligand pair, bound and free legs, lambda windows, PMF/overlap analysis, and network consistency checks. Existing endpoint trajectories remain useful but contain no hidden alchemical result.
- Seed FEP bound legs from equilibrated Loch frames, not fresh solvation, and keep bound-leg GCMC off by default because those waters are already placed. Across replicates, select the medoid frame by hydration-site-occupancy consensus; never average indistinguishable water coordinates. See [fep.md](references/fep.md).
- Validate a FEP leg by requiring one energy Parquet per lambda window and non-vanishing adjacent-window overlap. A small smoke config yields near-zero overlap and an astronomically uncertain DDG by design — that is the overlap diagnostic working, not a result. Run one edge and check per-window `ns day-1` and overlap before a full network.
- Reuse a published reference edge network rather than all-to-all, so every DDG is directly comparable; for a smaller network use optimal-design selection on `aggregate_fep_network.py`'s covariance, not an arbitrary subset.
- Keep SOMD2 FEP in a separate Mamba environment matched to its OpenBioSim release. For SOMD2+Loch GCMC, include `cuda-nvvm`; a visible `nvcc` wrapper alone can still fail when `cicc` is absent. Do not trust process exit status alone: require the expected energy Parquet file for every lambda window before publishing completion, because a window worker failure can leave the top-level command with a zero exit code.
- Benchmark FEP timing with at least about 100 ps of dynamics per sampled lambda on the target GPU; 4 ps smoke windows are dominated by sampler construction, minimization, serialization, and checkpoint startup. Report both sustained per-window `ns day-1` and whole-leg wall time, then include the configured per-window equilibration and sampler startup when projecting production. Do not equate a SOMD2 GCMC event with a Loch endpoint batched call without verifying their attempt counts and frequency.

## Never hardcode a system, path, or dataset

Every script must work on a receptor and ligand series it has never seen. Treat
any of the following as a defect, not a convenience:

- **Dataset-specific default arguments.** `--ligand-id` defaulting to `x7259a`,
  `--prefix-template` defaulting to `ev71_2a_{ligand_id}`, or a `--receptor`
  that silently falls back to one benchmark's PDB. A wrong default does not fail
  loudly — it simulates the wrong system and reports success. Make the argument
  **required** instead.
- **Dataset names baked into output filenames.** Prefixes must derive from the
  ligand or system actually being run.
- **Absolute paths to anyone's machine.** Including in vendored dependencies:
  `slow-rotations` was unusable because its `__init__` wrote to
  `/Users/megosato/Desktop/`.
- **Assumptions about chains, residue numbering, or protonation** that hold only
  for the current target. Where a genuine constraint exists (single protein
  chain; consecutive residue numbering; no insertion codes), state it in the
  error message and the README as a documented limitation rather than leaving it
  implicit.

Before declaring a pipeline reusable, grep for the current target's identifiers
(`ev71`, `x7`, the receptor PDB code, the ligand series name) across executable
code, and check every `default=` in every `add_argument`. Names in *filenames*
are cosmetic; names in *defaults and fallbacks* are bugs.

## Never add a capability without an executed path to it

A dependency, diagnostic, or tool that is installed and documented but never
called is not a feature — it is a claim. Treat the following as one unit of work,
not three:

1. Add the dependency.
2. Add code that **calls** it on a real input.
3. Run that code once, on real data, and check the output is sensible.

If step 3 has not happened, the capability does not exist regardless of what the
README says. Two failures of this rule have already cost real time here:
`slow-rotations` was installed and advertised in `README.md` and the `analysis`
stage's printed output while nothing in the codebase ever imported it — and when
finally exercised it turned out it could not even be constructed (an unused
`MDAnalysis.tests.datafiles` import, plus a hardcoded write to
`/Users/megosato/Desktop/` in `LigandTorsionFinder.__init__`). Its replacement is
`csbrt/torsion_diagnostics.py`, called from `stage_analysis`.

Corollaries:

- Prefer implementing a small computation over depending on a package you have
  not executed. Before adding one, import it and run its main entry point.
- A printed suggestion ("for X, use tool Y") is not an integration. Either wire
  it in or delete the sentence.
- Vendored fixes that require hand-editing installed source do not survive a
  reinstall. If a dependency needs patching to function, either patch it in the
  installer or do not depend on it.
- The same applies to config options that are declared but never read, and to
  documented CLI flags with no code path.

## Verify by output artefact, never by exit status

A stage that exits 0 has not necessarily produced anything. Require the expected
files and check their contents: one energy Parquet per lambda window, non-zero
frame counts, finite numbers. A SOMD2 window worker can die while the top-level
command returns 0; a leg missing interior lambda windows is severed into
disconnected segments and yields no DDG at all, even though every file present
looks valid. Smoke tests must assert on artefacts for the same reason.

## Change safely

Search callers before changing shared helpers. Keep diagnostic outputs in new, non-overwriting directories. Compile all touched Python entry points. Run CPU-safe parsing/unit checks locally; submit GPU/Slurm work only on the cluster.

When adding or retiring scripts, update [pipeline-map.md](references/pipeline-map.md). When a diagnosis changes the established failure model, update [diagnostics.md](references/diagnostics.md). When the FEP workflow, environment, or benchmark changes, update [fep.md](references/fep.md). Keep these references factual and concise.
