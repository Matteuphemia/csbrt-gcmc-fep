# Stability diagnosis

## Contents

- First diagnostic gate: original-script parity
- Established UVT2 incident
- Current compatibility fix
- Completed preparation/equilibration acceptance gate
- Cross-system checkpoint and postprocessing gate
- UVT1 diagnostic workflow
- Result interpretation

## First diagnostic gate: original-script parity

Before running a dedicated stability diagnostic, locate the failing stage in the submitted wrapper's real call path and compare it against Ludovic's original external script identified in `ludovic-parity.md`. Do not use the local rewritten GRAND pair as the reference, and do not trust comments when the executable order differs.

Audit these items in execution order:

1. Input topology and stage construction.
2. Ghost-buffer creation, state reporting, removal, and replacement between stages.
3. Minimization, velocity assignment, GCMC moves, MD segments, and frame/report writes.
4. Restraints, nonbonded settings, integrator, thermostat, barostat, timestep, and platform precision.
5. Cycle counts, move counts, report intervals, and physical saved-handoff behavior.

If the Loch path skips, reorders, or changes an original Ludovic operation, test that mismatch first. Use force/constraint/RF tracing only after this comparison finds no mismatch capable of explaining the failure.

## Established UVT2 incident

The July 2026 five-way diagnostic in `uvt2-stability-61716/` found:

- continued NPT, restrained NVT without ghosts, and NVT with fresh ghosts but no GCMC all completed 500 ps;
- both GCMC variants failed immediately after RF-preaccepted trials yielded no final PME move;
- the force was already enormous after `sampler.move()` and before MD;
- the largest-force atoms were physical waters near the end of the NPT topology, not the 45 fresh inactive ghosts.

The implicated waters were UVT1 buffer waters that had become logically active. Loch 2025.2 updated their live OpenMM charge/LJ parameters, but the old handoff saved the original Sire ghost topology. NPT therefore propagated them with zero charge/LJ until they drifted into overlaps. UVT2 deletion-rejection restored normal parameters and exposed the overlaps as a force spike. The first MD step only turned the existing bad force into NaNs.

This evidence rules out NPT, the 2 fs timestep, minimization, C-alpha restraints, and ordinary constraint drift as the primary cause of that incident.

## Current compatibility fix

`loch_ludovic_common.finalise_sampler_system()` ports the needed topology behavior for Loch 2025.2:

1. Copy live imaged positions and periodic box into a Sire clone.
2. Remove every logically inactive water/ghost using stable molecule identities.
3. Require the remaining zero-interaction-water count to equal Loch's logically active appended-buffer count.
4. Give exactly those retained waters the physical water-template charge/LJ values in Sire.
5. Validate the committed values before returning the stage handoff.

The first compatibility implementation attempted to activate buffer waters before ghost removal. UVT1 diagnostic job 61723 showed that this did not survive into the returned topology: UVT1 itself remained finite through 135,000 attempts, but all six accepted buffer waters were still zero-interaction before and after save/reload. The revised helper enforces the final topology invariant directly after removal and refuses the handoff if the logical and stored counts differ.

UVT1 diagnostic job 61724 confirmed the revised helper with the exact production call structure: 10,000 initial attempts followed by 100 × 1,000 attempts and five MD steps. All states stayed finite, nine accepted appended waters were materialized, 41 inactive waters were removed, and both the finalized and AMBER-reloaded topologies contained 18,804 physical waters with zero zero-interaction waters. This closes the established UVT1-to-NPT topology failure mechanism.

Both `LochEquilibration.py` and `LochProduction.py` call this helper before saving a physical handoff/restart, then use `save_physical_system()` to validate the in-memory topology and the AMBER reload. They also reject all-zero physical waters at their external inputs. NPT made from the pre-fix UVT1 output is contaminated and must not be reused.

## Completed preparation/equilibration acceptance gate

Audit a copied full-pipeline output at four levels before allowing production:

1. **Preparation completion.** Require the SQM `Calculation Completed` marker, a near-integral requested ligand charge, and final complex tLEaP `Errors = 0`. For the CRY1-AN139 reference input, the protonated ligand has 59 atoms and net charge zero within MOL2 rounding. tLEaP adds 18,801 solvent residues, then replaces one water with the neutralizing chloride, leaving 18,800 physical waters in the prepared AMBER system. The `addIonsRand` same-sign warning for the attempted `Na+ 0` command is expected because the unsolvated complex charge is +1; the following `Cl- 0` command neutralizes it.
2. **Executed protocol length.** Excluding headers, require 5 UVT1 CSV records at steps 100 through 500, 400 NPT records at steps 2,500 through 1,000,000, and 500 UVT2 records at steps 500 through 250,000. Require exactly 100 UVT1 and 125 UVT2 ghost-state lines. An empty UVT2 Loch log is not by itself a failure; the CSV, ghost record, stdout, and saved handoff are authoritative.
3. **Thermodynamic behavior.** Require every CSV field to be finite and inspect the trace rather than only its final row. The short 1 ps UVT1 segment can remain below 300 K; NPT should rapidly reach the target temperature and a stable volume/density without later drift or excursions. UVT2 should remain finite near 300 K. Do not interpret the UVT1/UVT2 CSV density as the ghost-free physical density: `CsvStateWriter` includes the masses of the 45 appended buffer waters while a sampler is live. Use NPT for physical density assessment.
4. **Physical topology integrity.** Reload every AMBER stage pair independently and require zero three-site waters with simultaneously all-zero charge and LJ epsilon. NPT must have the same physical-water count as UVT1. For a GCMC handoff, verify the arithmetic

   `saved physical waters = input physical waters + 45 - final state-0 count`.

   Apply it independently to UVT1 and UVT2 because UVT2 creates a fresh buffer. This catches both discarded accepted waters and leaked inactive ghosts even when MD and Slurm complete normally.

Use the July 15, 2026 full run as a known-good regression benchmark, not as a deterministic acceptance target:

| Boundary or trace | Observed result |
| --- | --- |
| Prepared AMBER | 64,109 atoms; 18,800 waters; zero all-zero waters |
| UVT1 handoff | 18,809 waters; final state-0 count 36; net +9 waters; zero all-zero waters |
| NPT handoff | 18,809 waters; zero all-zero waters |
| UVT2 handoff | 64,127 atoms; 18,806 waters; final state-0 count 48; net -3 waters; zero all-zero waters |
| NPT trace | 300.29 +/- 1.15 K; 1.0185 +/- 0.0024 g/mL; volume settled near 642 nm^3 |
| UVT2 trace | 300.24 +/- 1.22 K; finite throughout 250,000 MD steps |

Water counts and acceptance histories are stochastic; topology invariants, report counts, and operation order are not.

The reference preparation intentionally retains the two long C--N bonds reported by tLEaP at 10.563 A and 17.240 A because Ludovic's original system contains them. In the known-good Loch run they relaxed to 1.301 A and 2.327 A by the UVT2 handoff. The second distance is unusual but matches Ludovic's 12 deposited production replicas, which span 2.308--2.372 A. Treat this as an inherited Ludovic structural artifact, not a Loch failure. Do not silently add `TER`, delete the bond, or otherwise repair it in the parity workflow; make any scientific correction a separately named protocol variant.

## Cross-system checkpoint and postprocessing gate

For a ported system, distinguish `smoke_plumbing_only` from the canonical full schedule in every audit JSON. Do not validate requested marker counts against themselves: compare marker signatures with an independent profile table and verify every expected CSV step, ghost line, and DCD frame.

Require completion markers to hash source/protocol/runtime identity and every required output. Missing or changed CSV/ghost diagnostics must invalidate the owning stage instead of letting it skip and fail repeatedly in the downstream audit. Invalidate the old marker before any forced/rebuild attempt.

Strengthen topology comparisons beyond residue/atom names. Compare mass, charge, LJ, bonds, angles, nonzero torsions, and exclusions/1-4 exceptions across preparation, UVT1, NPT, UVT2, raw production, and production-final. Sire may omit zero-force torsion records; this does not change the potential. Negligible serialization roundoff is also acceptable, but a changed nonzero parameter is not.

For postprocessing, validate the copied topology hash, processed DCD frame/atom counts and finite coordinates, sphere model count, cluster-record count, and marker/metrics agreement. Reject duplicate, negative, out-of-range, or non-water ghost IDs. GRAND's historical shift-before-image order may wrap inactive waters back toward the ligand; a correctness-oriented port should image/align first, shift second, mask inactive IDs explicitly, and prove they remain outside the sphere.

## UVT1 diagnostic workflow

Submit from the cluster project directory:

```bash
sbatch scripts/diagnose_uvt1_stability.slurm
```

By default the wrapper reads `$HOME/cry/cry-loch-full/rep1/preparation`, matching `loch_full_pipeline.slurm`. Select another full-pipeline replica with `REPLICA`, another run root with `SOURCE_RUN_ROOT`, or any prepared input directory with `INPUT_DIR`.

Useful overrides:

```bash
sbatch --export=ALL,REPLICA=2,CYCLES=100,SEED=20260714 scripts/diagnose_uvt1_stability.slurm
sbatch --export=ALL,INPUT_DIR=/path/to/preparation,CYCLES=100,SEED=20260714 scripts/diagnose_uvt1_stability.slurm
```

The wrapper defaults to 1,000-attempt chunks for the initial 10,000 attempts. This improves localization while retaining the same total work. Set `INITIAL_CHUNK_SIZE=10000` to reproduce the single production call exactly. Set `CYCLES=0` for an initial-move-only check. `MAX_FORCE` defaults to 100,000 kJ mol^-1 nm^-1 and stops before MD once exceeded; set it to `0` only when deliberately tracing through the spike. `BATCH_SIZE`, `INITIAL_ATTEMPTS`, `ATTEMPTS`, and `MD_STEPS` are also available as Slurm environment overrides, but their defaults match `LochEquilibration.py`.

The diagnostic records after every stage:

- finite positions, velocities, forces, energies;
- maximum-force and maximum-speed atom identity and logical water state;
- constraint errors;
- RF preacceptance count and final PME moves;
- coordinates changed by a move;
- logical water-state counts and active appended buffer identities;
- mismatches between logical state and mutable OpenMM charge/LJ parameters.

After successful UVT1 it finalizes, saves, reloads, and audits the topology. `result.json` is atomically replaced after every record so partial jobs remain useful.
Unsafe states and invalid handoffs exit nonzero after recording their terminal status, so Slurm completion alone cannot mask a diagnostic failure.

## Result interpretation

| Pattern | Most likely layer |
| --- | --- |
| Unsafe at `sampler_bound` | Input topology or Sire/OpenMM construction. |
| Unsafe after `sphere_deleted` or minimization | Initial sphere deletion, geometry, or minimization setup. |
| Large force directly after a move, no MD yet | GCMC trial/PME correction or rollback. Inspect atom ghost state and coordinate changes. |
| Parameter-state mismatch | Loch logical state and OpenMM `NonbondedForce` are inconsistent. |
| Inactive ghost has large force with changed coordinates | Rejected insertion geometry or bonded ghost terms. |
| Active/physical water has large force, especially after rejected deletion | Overlap or restored nonbonded coupling. |
| Constraint error grows only during MD | Geometry/constraint integration issue rather than move-state rollback. |
| `handoff_contains_zero_interaction_waters` | Sire topology materialization/save-reload failure; do not run NPT. |
| UVT1 and handoff complete cleanly, later NPT control is stable | Move diagnosis to UVT2 or production using the saved boundary inputs. |
