# Porting Ludovic-order Loch to a new system

## Contents

- Inventory and chemistry contract
- Parameterization provenance
- Four-stage architecture
- Checkpoint contract
- Aggregate-run contract
- Boundary audit contract
- Postprocessing and resources
- Validation sequence

## Inventory and chemistry contract

Start from files that are actually present. Enumerate receptor PDBs and every ligand structure library; do not infer missing crystallographic poses, metals, cofactors, or alternate releases from a README. Select a ligand by exact SDF title or `ligand_name`, never by record position, and prove the selected record is unique.

Audit before parameterization:

- receptor chains, residue range/gaps, alternate locations, waters/heteroatoms, atom count, C-alpha count, explicit hydrogens, termini, disulfides, and net charge;
- supplied histidine proton locations and residue templates, especially catalytic residues;
- ligand validity, 3D coordinates, explicit hydrogens, formal charge, multiplicity, title/alias, and consistency with any compound table.

Preserve the supplied receptor coordinates, explicit protonation, and ligand pose unless the user asks for a scientific change. Normalize only names required by the selected force field. Record every normalization in the preparation marker. Do not introduce or reconstruct missing zinc or other chemistry when the task explicitly scopes it out.

Accept a prepared macromolecular `.cif`/`.mmcif` receptor only through an explicit, checkpointed conversion boundary. Preserve and hash the original CIF, fingerprint the converter, materialize a PDB for the existing strict receptor audit, and prove atom identity/count and coordinates before parameterization. Use Open Babel's `mmcif` reader rather than its small-molecule `cif` reader. Keep ligands on the separately validated SDF path; receptor CIF support does not imply arbitrary ligand-CIF support.

For a separate receptor PDB plus SDF ligand, do not reuse a preparation script that assumes a combined holo PDB, a hard-coded residue number, or a positional ligand record. Parameterize the exact selected ligand and then combine it with the normalized receptor.

Make ambiguity fail closed. Define an explicit policy for alternate locations, multiple chains, heteroatoms, termini, `TER` records, and residue-name normalization instead of silently dropping or merging them. The policy may differ by system, but the preparation marker must make it auditable.

Test chemistry preservation as executable invariants, not prose. Compare the extracted SDF and AMBER ligand for atom/element order, explicit-hydrogen count, connectivity edge set, ligand charge, and rigid-aligned pose. Resolve multiplicity from explicit SDF metadata or a required user input, then pass and record it in the charge-tool command. Rigidly align the ligand heavy atoms before checking pose RMSD because solvation may translate or rotate the complex; reject an unexplained conformational change. Independently compare source and prepared receptor chain/residue/C-alpha identity, coordinates, and protonation-defining hydrogens after applying only the declared name mappings.

## Receptor conformance checklist (strict EV71-style preparation)

`prepare_ev71_system.py` enforces a narrow input contract inherited from the
OpenBind EV71 release, and `audit_ev71_pipeline.py` independently re-checks much
of it. A receptor from any other provenance -- crystal chain B, a modelled loop,
PDB2PQR/PROPKA protonation -- will fail these one at a time, because the audit
stops at the first violation. Conform the input in an adapter step rather than
loosening the audit; each item below was hit in the CRY1/7DLI port (July 2026).

| enforced requirement | typical violation | fix |
| --- | --- | --- |
| single chain named `A` | crystal protein is chain B | relabel after cleaning |
| no HETATM | crystallographic waters | drop them; tleap solvates and GCMC re-places pocket waters |
| no insertion codes | loop grafted as `395A-395F` | renumber contiguously, write an old->new residue map |
| no fragmentary residues | stray single-residue chains (a free lysine) | drop chains whose only non-water residue is one amino acid |
| standard/AMBER residue names | PROPKA emits `HID/HIE/HIP/CYX/CYM/ASH/GLH/LYN` | accept them: ff14SB parameterises all of these |
| TER identifies the final residue (chain col 22, resid 23-26) | PDB2PQR writes a bare `TER` | rebuild the record from the preceding ATOM |
| one N-terminal `H` renamed to `H1` | `--ffout=AMBER` already yields `H1/H2/H3` | accept the already-normalised form |
| histidine templates recorded == audit's inference | preparation only scanned residues named `HIS`, skipping `HID/HIE/HIP` | scan the whole `HIS/HID/HIE/HIP` set so both sides agree |
| ligand: explicit hydrogens | docked/crystal SDFs are heavy-atom only | add H with coordinates, then relax **only** H with heavy atoms fixed, and assert the heavy-atom pose did not move |
| ligand: `charge`/`multiplicity` SDF metadata | absent, and `charge` is cross-checked against the molecular graph | write formal charge and `radical_electrons + 1` |

Two of these are judgement calls rather than plumbing, and should be surfaced to
the project owner rather than made silently: dropping crystallographic waters
(correct for GCMC, which re-samples them, but a divergence from a workflow that
kept them), and accepting AMBER protonation variants (collapsing `HID/HIE/HIP`
back to `HIS` would discard the pH-dependent assignment the protonation step
exists to produce). Before accepting the second, re-derive each template from
HD1/HE2 occupancy and confirm it reproduces the supplied names exactly.

Prefer writing a single conformance pre-check that reports every violation at
once over discovering them serially; the audit's fail-fast behaviour otherwise
costs one full run per gap.

## Parameterization provenance

Treat AmberTools and its data files as implementation inputs. Record and hash the resolved `antechamber`, `parmchk2`, and `tleap` executables; the installed package record; and every loaded `leaprc`, residue library, parameter, water, and ion file. Record the exact commands, requested charge/multiplicity, and tool versions.

Make generated command files and complete logs required checkpoint outputs. Before publishing preparation success, require SQM's completion marker, tLEaP `Errors = 0`, a ligand partial-charge sum consistent with the requested integral formal charge, and an independently reloadable final AMBER pair. A process exit code alone is not sufficient evidence of a valid parameterization.

## Four-stage architecture

Keep four resumable external stages:

1. Preparation: normalize names, parameterize ligand, combine, solvate, neutralize, save AMBER, reload-audit.
2. Equilibration: UVT1 → physical handoff → unrestrained NPT → fresh-buffer UVT2 → physical handoff. Keep UVT1, NPT, and UVT2 as internal checkpoints.
3. Production: fresh 45-water buffer, matched MD→move→frame/ghost reporting, raw ghost topology/trajectory, and a finalized physical restart.
4. Postprocessing: image/align, move inactive ghosts away, explicitly exclude them, extract the sphere, and cluster physical water observations.

Use the canonical full schedule in `ludovic-parity.md`. Pass every count explicitly from an orchestrator even when stage defaults agree, so a changed default cannot masquerade as a full run. Determine the top-level validation scope from both simulation counts and analysis settings: label reduced counts `smoke_plumbing_only`, canonical counts with stride-1 clustering `full_ludovic_schedule`, and canonical counts with `cluster_stride > 1` `full_simulation_approximate_postprocessing`.

Map replica IDs to deterministic, noncolliding seed blocks such as `base_seed + (replica - 1) * 1000`, leaving room for the stage-specific offsets. Require a positive integer replica, record the resolved base and stage seeds, and make an explicit seed override visible in the signature rather than silently combining it with the replica offset.

Keep GCMC and MD platform arguments separate. Stock Loch 2025.2 requires CUDA or a GPU OpenCL device for supported GCMC execution. A temporary PoCL/CPU adapter can validate plumbing only; fingerprint its patched Loch modules and never present that run as a stock-environment scientific validation.

## Checkpoint contract

For each stage marker, include:

- SHA-256 of every upstream input;
- all user-visible numerical counts and the complete fixed physical protocol;
- SHA-256 of the stage source, shared helper sources, and relevant installed Loch platform/sampler modules;
- Python and dependency versions;
- SHA-256 of every required output.

Required outputs include CSV and ghost histories, not only AMBER/DCD/PDB files. Validate hashes and semantic content on the fast path. Delete the old completion marker before recomputation; publish a new marker atomically only after all outputs pass. This prevents a failed forced rerun from exposing a stale success marker.

Treat the marker's directory as the trust root for its outputs. Store output-hash keys as normalized paths relative to that directory; reject absolute paths, `..` traversal, and resolved paths outside the trust root. Never preserve absolute output paths in a copied marker, because the copy could otherwise validate files in the original run instead of its own artifacts.

Re-extract or hash-validate a cached ligand against the current source library and requested identifier on every invocation. Store source-library and extracted-record hashes in the manifest.

## Aggregate-run contract

Stage markers are durable resume evidence; aggregate timing, audit, and state files are invocation summaries. At the start of every orchestration attempt, atomically mark the run `running` and archive or invalidate prior aggregate success/timing outputs. Record the invocation identity and whether each stage was reused or recomputed. Publish aggregate `completed` only after the final independent audit succeeds. If execution fails or is interrupted, do not leave a stale completed summary that can be mistaken for the current attempt; preserve valid stage markers so the next attempt resumes at the earliest invalid boundary.

## Boundary audit contract

Run an independent audit after every external boundary. Require:

- one ligand and the expected receptor C-alpha count;
- finite coordinates and thermodynamic CSV fields;
- the exact expected CSV step sequence, cycle count, DCD frame count, and ghost-line count for the declared profile;
- duplicate-free, nonnegative ghost residue IDs that map to valid three-site waters for that sampler topology;
- exact TIP3P charge/LJ/mass/name values for every interacting water and finite coordinates/box vectors in every streamed DCD chunk;
- zero all-zero charge/LJ waters in preparation, UVT1, NPT, UVT2, and production-final;
- exactly 45 zero-interaction buffer waters in the raw production topology;
- NPT water count equal to UVT1 water count;
- `saved physical waters = input physical waters + 45 - final state-0 count` independently at UVT1, UVT2, and production;
- unchanged solute force field across every physical boundary and the raw production topology.

The solute comparison should cover particle mass/charge/LJ, bonds, angles, nonzero torsions, and nonbonded exclusions/1-4 exceptions. Ignore only zero-force torsion records that Sire legitimately omits during AMBER serialization, and normalize negligible serialization roundoff. An atom-name-only hash is insufficient.

Check a failure's upstream handoff as well as the stage where it surfaced. A deleted physical water can be latent during NPT and become catastrophic only when a later sampler restores its interactions.

## Postprocessing and resources

GRAND historically shifts inactive ghosts and then images molecules. Imaging can wrap those ghosts back toward the ligand. For a new correctness-oriented pipeline:

1. image molecules and align on protein C-alpha atoms;
2. shift each frame's inactive residues by five box lengths;
3. explicitly mask the same inactive residue IDs out of sphere selection/clustering;
4. independently prove no inactive oxygen lies inside the sphere in the saved processed DCD.

This deliberately differs from the historical operation order only for inactive coordinates; physical atom coordinates and the clustering definition remain unchanged. Document the difference instead of silently claiming byte-for-byte GRAND parity.

Average-linkage clustering uses an O(N^2) condensed distance array. Estimate observations from a smoke run before full postprocessing. Roughly 30 waters/frame across 2,500 frames requires about 22 GB for distances alone, before SciPy linkage overhead. Preserve stride 1 for exact sampling and request a node with substantial headroom (96 GB is a reasonable starting point); label `cluster_stride > 1` as an approximation.

For a ligand-series project, use exact average-linkage clustering as a parity
or crystal-water benchmark, not as the only scientific representation. Build a
protein-aligned oxygen-density grid from all physical waters, extract candidate
sites, freeze a receptor-coordinate site catalog, and reassign every ligand and
replica to that same catalog. Report occupancy with block/replica uncertainty,
ligand overlap/displacement, and water-mediated contacts. Occupancy is not
water residence time: GCMC insertion/deletion breaks molecular identity, so
measure residence with a separate fixed-N MD analysis if required.

For a large ligand-by-replica launch, generate and preserve a manifest before
submission. It must record the exact SDF title, replica, seed block, prefix, and
run directory for every array index. Bound array concurrency explicitly. Run
the canonical checkpointed simulation through production independently for
each row, then perform provisional density analysis. Only after all tasks pass
their production and density markers should a dependent CPU job freeze the
common catalog and reassign every trajectory to that catalog. Publish series
completion only after all common-catalog analyses validate.

## Validation sequence

1. Compile every Python entry point and syntax-check the Slurm wrapper.
2. Run preparation alone and audit it.
3. Run UVT1, NPT, and UVT2 with smoke counts; audit every handoff and inspect water arithmetic.
4. Run production smoke; verify post-move DCD/ghost synchronization and raw/final topology contracts.
5. Run postprocessing and independently reload its DCD/PDB outputs.
6. Run the four stages once from a new directory with no checkpoints.
7. Reissue the same command and prove all semantic/hash checkpoints resume cleanly.
8. Copy the completed run directory, remove or alter one artifact, and prove marker-relative trust-root validation rejects the copy or invalidates the owning stage.
9. Run two replica dry-runs and prove their resolved seed blocks differ without changing protocol counts.
10. Run the full schedule on a supported GPU host; do not promote smoke or stride-subsampled results to exact full validation.
