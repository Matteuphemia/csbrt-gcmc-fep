# Ludovic/Loch protocol parity

## Authoritative comparison sources

Use Ludovic's original external scripts, not the rewritten GRAND scripts in this repository:

- `../workflow-GCMC-Ludovic (after online tutorials)/GCMC_Grand/Equilibration.py`
- `../workflow-GCMC-Ludovic (after online tutorials)/GCMC_Grand/Production.py`

The cluster copy is normally `$HOME/cry/ludovic-workflow-gcmc/GCMC_Grand`.

## Stage and ghost lifecycle

| Boundary | Ludovic GRAND | Loch replacement |
| --- | --- | --- |
| UVT1 construction | Add 45 ghosts. | `make_sampler()` appends 45 ghosts. |
| UVT1 sampling | Delete sphere waters; minimize; 10,000 moves; 100 × (1,000 moves then 5 MD steps). | Same stage order and counts. |
| UVT1 → NPT | Query state-0 waters, remove them, and write a physical PDB. Accepted buffer waters remain and acquire TIP3P parameters when the force field is rebuilt. | Remove every state-0 water and explicitly materialize retained accepted buffers as TIP3P before saving. |
| NPT | No ghosts; 1,000,000 × 2 fs steps at 300 K and 1 bar, barostat every 25 steps. | Same. The original comment says 1 ns, but the executed protocol is 2 ns. |
| NPT → UVT2 | Add a fresh set of 45 ghosts. | A fresh sampler appends 45 ghosts. |
| UVT2 | 125 × (800 moves then 2,000 MD steps). | Same stage order and counts. |
| UVT2 → production | Remove state-0 waters, retaining accepted physical waters. | Same physical handoff, with explicit topology parameter validation. |
| Production construction | Add a fresh set of 45 ghosts. | A fresh sampler appends 45 ghosts and saves a ghost-containing trajectory topology. |
| Production cycle | 2,000 MD steps, then 200 GCMC moves, then report/write the ghost state and DCD frame. | Same order; the DCD and ghost line are written after the move. |
| Production end | Preserve ghost-containing raw trajectory/topology for processing. | Preserve those files and additionally save a ghost-free physical restart. |

Inactive ghosts are required inside each live GCMC stage but must not enter the intervening physical NPT stage. Every new GCMC stage creates a fresh 45-water buffer, matching `grand.utils.add_ghosts()` in Ludovic's scripts.

## Exact executed schedule

Follow the executable statements in Ludovic's scripts when their comments disagree with the code.

| Stage | Per-stage ordering | Totals and report alignment |
| --- | --- | --- |
| UVT1 setup | Add 45 ghosts → delete waters in the 10 A sphere → minimize → 10,000 GCMC attempts. | The initial attempts do not advance MD time. |
| UVT1 loop | 100 × (1,000 GCMC attempts → GCMC/ghost report → 5 MD steps). | 100,000 loop attempts; 110,000 UVT1 attempts total; 500 MD steps = 1 ps. State CSV every 100 MD steps. |
| UVT1 handoff | Image → remove state-0 waters → retain accepted waters as physical TIP3P. | The saved/reloaded physical topology must contain zero all-zero charge/LJ waters. |
| NPT | Rebuild without ghosts or C-alpha restraints → assign 300 K velocities → 1,000,000 MD steps at 1 bar. | 2 ns at 2 fs; barostat every 25 steps; state CSV every 2,500 steps. |
| UVT2 setup | Add a fresh 45-water buffer → create fresh C-alpha restraint anchors → assign velocities. | No initial standalone GCMC block. |
| UVT2 loop | 125 × (800 GCMC attempts → GCMC/ghost report → 2,000 MD steps). | 100,000 attempts; 250,000 MD steps = 0.5 ns; state CSV every 500 MD steps. |
| UVT2 handoff | Image → remove state-0 waters → retain accepted waters as physical TIP3P. | Save and reload-audit the physical AMBER pair before production. |
| Production setup | Add a fresh 45-water buffer → create fresh C-alpha restraint anchors → assign velocities. | Save a separate ghost-containing topology for the raw trajectory. |
| Production loop | 2,500 × (2,000 MD steps → 200 GCMC attempts → GCMC/ghost report and raw DCD frame). | 5,000,000 MD steps = 10 ns; 500,000 attempts; state CSV every 500 MD steps; 2,500 matched DCD frames and ghost lines. |

The authoritative defaults are constants in `scripts/loch_ludovic_common.py`. `batch_size=50` is Loch's parallel execution width, not a Ludovic move count; changing it must not change any attempt total above.

## Matched numerical protocol

- 300 K; 2 fs; friction 1 ps^-1; mixed CUDA precision.
- PME with 12 A cutoff, 10 A switching, dispersion correction disabled, and OpenMM's 5e-4 Ewald tolerance.
- Hydrogen-bond constraints and center-of-mass removal every step.
- Ligand-centered 10 A GCMC sphere.
- TIP3P excess chemical potential -6.09 kcal/mol and standard volume 30.345 A^3.
- C-alpha periodic positional restraints with numerical coefficient 100 in UVT1, UVT2, and production; no restraints during NPT.
- UVT1, NPT, UVT2, and production ordering, reporter frequencies, and total move/MD counts exactly as tabulated above.

The Loch seed sequence is deterministic: base seed 20260714 for UVT1, base + 1 for NPT velocities, base + 2 for UVT2, and base + 3 for production. This is reproducibility scaffolding; Ludovic's scripts do not set explicit seeds.

## Physical handoff contract

Every GCMC-to-physical boundary must call `finalise_sampler_system()` and then `save_physical_system()`. The first helper removes only logical state-0 waters and gives retained accepted buffer waters the physical TIP3P charge/LJ parameters that Loch 2025.2 changed only in the live OpenMM context. The second helper validates in memory, writes AMBER/PDB, reloads the AMBER pair, and validates the same water count with zero all-zero charge/LJ waters.

Use generic `save_system()` only for deliberately ghost-containing artifacts, specifically the production trajectory topology. Never feed that topology to NPT or to a fresh sampler as a physical restart.

The native-AMBER and rebuilt OpenMM energy constructions were checked at identical minimized coordinates. The adjusted total-energy difference was approximately 0.001 kJ/mol in `output/ludovic_combined_energy_comparison_min20.json`.

## Necessary engine differences

Exact trajectories and acceptance histories cannot match because GRAND is being replaced:

- GRAND performs its own sequential PME GCMC moves; Loch uses batched RF preacceptance followed by PME correction.
- Ludovic uses `openmmtools.BAOABIntegrator`; Sire uses OpenMM's `LangevinMiddleIntegrator`, with the same temperature, friction, and timestep and the same BAOAB-family middle splitting intent.
- Sire 2025.4 receives the matched 1 ps^-1 friction through its dynamics property map because `_dynamics()` does not expose a direct `friction` keyword.
- The Loch workflow uses explicit random seeds; Ludovic's scripts do not.
- UVT1 velocity assignment occurs after minimization in the Loch compatibility path. This avoids constraint-inconsistent kinetic states observed with the native-AMBER/Sire construction; it does not change the stage lengths or thermodynamic parameters.
- Loch saves AMBER stage boundaries and performs post-processing in a separate entry point; Ludovic rebuilds from PDB and post-processes at the end of `Production.py`.
- GRAND's historical postprocessing shifts inactive ghosts before molecular imaging. Imaging can wrap those ghosts back toward the ligand. Preserve that order only for a clearly labeled strict-parity comparison. A correctness-oriented new-system port should image/align first, shift second, and explicitly exclude the per-frame inactive IDs; physical-atom coordinates and the average-linkage clustering definition are otherwise unchanged.

Treat these as implementation differences, not permission to change the stage order, force-field settings, ghost lifecycle, or reporting alignment.
