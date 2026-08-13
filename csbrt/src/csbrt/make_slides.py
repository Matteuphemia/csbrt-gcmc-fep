#!/usr/bin/env python3
"""Build the CRY1/KL101 update deck as PPTX (and PDF for quick preview).

    python make_slides.py                 # both formats
    python make_slides.py --format pptx   # just the editable deck

Slide content is defined once as data and rendered by two back-ends, so the
PPTX and the PDF can never drift apart. Numbers come from analysis/combined/*.json.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
ACC = "1F4E79"
MUT = "5A5A5A"
GOODBG = "EFF7EF"
BADBG = "FDF2EE"
NEUTBG = "F2F6FA"


# ---------------------------------------------------------------------------
# content
# ---------------------------------------------------------------------------
def build(prefix="CRY1KL101"):
    A = HERE / "analysis" / "combined"
    comb = json.loads((A / f"{prefix}_combined_summary.json").read_text())
    water = json.loads((A / f"{prefix}_water_sites.json").read_text())
    net = json.loads((A / f"{prefix}_water_network.json").read_text())
    tasks = json.loads((HERE / "tasks.json").read_text())
    pooled = comb["pooled_curve_integrated"]
    bs = pooled["bootstrap"]
    df, lo, hi = pooled["df_bind_star_kcal_per_mol"], bs["ci95_lo"], bs["ci95_hi"]
    st = net["summary_table"]
    cry = net["crystal"]
    prof = {p["n"]: p for p in pooled["profile"]}
    ncons = len(net["consensus_sites"])
    nmatch = sum(1 for c in net["consensus_sites"] if c["matched_xray"])

    S = []

    S.append({"kind": "title", "title": "CRY1 / KL101",
              "sub": "Grand Canonical Integration campaign — status update",
              "line": f"216 GCMC/MD windows · 6 replicas × 36 Adams values · "
                      f"164.5 GPU-hours · complete",
              "foot": f"thira.r-ccs27.riken.jp · {date.today().isoformat()}",
              "panel": f"Headline:  ΔF_bind(N*) = {df:+.2f} kcal/mol  "
                       f"[95% CI {lo:+.2f}, {hi:+.2f}]  at N* = {pooled['n_star']} waters.  "
                       f"All 216 windows verified.  Result is preliminary and carries a "
                       f"systematic caveat — slide 18."})

    S.append({"title": "Bottom line", "kicker": "what you need to know in 30 seconds",
              "big": f"ΔF_bind(N*) = {df:+.2f} kcal/mol",
              "bigsub": f"95% CI [{lo:+.2f}, {hi:+.2f}]   ·   N* = {pooled['n_star']} waters",
              "bullets": [
                  "All 216 windows completed with verified checkpoints. No failed tasks in the "
                  "final run.",
                  "The campaign as shipped could not run here. Four defects fixed before "
                  "submission; a fifth forced a restart at 96/216.",
                  f"Hydration network recovers {st['Recall']['mean']:.0%} of the crystallographic "
                  f"waters of 6KX6 chain A — consistent with the AN139 study under the same protocol.",
                  "Two honest caveats: the curve needs an isotonic fit (plateau noise), and the "
                  "absolute value carries an unquantified diffusion bias.",
              ]})

    S.append({"title": "What was run", "kicker": "protocol identical to the published one",
              "table": [["Parameter", "Value", "Cluster", "Value"],
                        ["System", "CRY1 + KL101, 22,246 waters", "Partition", "all — 22 GPUs / 11 nodes"],
                        ["Sphere", f"r = {tasks['radius_angstrom']:g} Å", "Cards",
                         "4090 / 3090 / RTX 4000 Ada / 2080 Ti"],
                        ["Ladder", "36 Adams values, −30 → −4", "Job", "1 GPU, 4 cores, 32 GB, 8 h"],
                        ["Per window", "625 × (400 GCMC + 4000 MD) = 5 ns", "Concurrency",
                         "--array=0-215%12"],
                        ["Replicas", f"{tasks['replicas']}", "Measured",
                         "median 0.70 h, max 0.96 h/window"],
                        ["Ensemble", "NVT, 300 K, PME 12 Å, ff14SB/TIP3P", "Total",
                         "164.5 GPU-hours, 15 GB"]],
              "widths": [0.13, 0.32, 0.13, 0.42],
              "note": "Wall limit set from measurement, not guesswork: 25 production cycles timed "
                      "on the slowest and reference cards (2080 Ti 1.38 h/window, 3090 0.97 h). The "
                      "3090 figure reproduces the documented ~50 min, so the slow-card number can "
                      "be trusted."})

    S.append({"title": "How it went", "kicker": "two submissions, one deliberate restart",
              "table": [["Stage", "What happened"],
                        ["Env build", "CUDA 12.9 as shipped → OpenMM CUDA platform dead. Rebuilt on 12.8."],
                        ["Smoke test", "Caught 2 further blockers before any production GPU time was spent."],
                        ["Run 1 (66398)", "216 windows launched. At 96/216 a window died: ghost buffer exhausted."],
                        ["Decision", "Cancelled. A partial fix was impossible — the analysis refuses to "
                                     "pool windows with different buffer sizes."],
                        ["Run 2 (66510)", "NUM_GHOSTS 27 → 100 after measuring true demand. 216/216 clean."],
                        ["Analysis", "Per replica + pooled, water network, crystal comparison."]],
              "widths": [0.16, 0.84],
              "note": "Cost of the restart: ~60 GPU-hours discarded. The archived run is preserved "
                      "in gci_kl101_6rep_ghosts27_discarded/."})

    S.append({"title": "Getting it to run", "kicker": "four defects, three of which fail silently",
              "table": [["#", "Symptom", "Cause", "Impact if missed"],
                        ["1", "Every job exits in ~2 s",
                         "conda activate under set -u; the cuda-nvcc hook expands an unset variable",
                         "All 216 jobs dead. No checkpoint → resubmitting fails identically, forever"],
                        ["2", "OpenMM CUDA platform fails",
                         "CUDA 12.9 toolchain vs a 570.x (CUDA 12.8) driver — PTX the driver cannot load",
                         "Every window fails at the first GCMC move"],
                        ["3", "Smoke test runs a full window",
                         "run_window.sh sources config.sh, which overwrites the SMOKE variable",
                         "3-minute check silently becomes a 1-hour run"],
                        ["4", "Smoke path cannot run at all",
                         "stride 25 does not divide the 10 smoke cycles",
                         "No way to validate before committing GPU-weeks"]],
              "widths": [0.03, 0.19, 0.42, 0.36],
              "note": "A PyCUDA kernel test passes on the broken 12.9 toolchain — it compiles to "
                      "arch-specific cubin, while OpenMM JITs PTX. Verify with "
                      "openmm.testInstallation, not a hand-rolled kernel."})

    S.append({"title": "The one that mattered: ghost-water exhaustion",
              "kicker": "a single crashed job that was not a single crashed job",
              "bullets": [
                  "One task died at cycle 469/625: “Cannot insert any more waters”. The obvious "
                  "response — resubmit that task — would have been wrong.",
                  "NUM_GHOSTS=27 is 3× the instantaneous bulk-equivalent occupancy. But the buffer "
                  "drains cumulatively: an inserted water can diffuse out and stays physical, never "
                  "returning to the pool.",
                  "The danger: at the same B another replica completed with a valid checkpoint, "
                  "having bottomed out at 3 ghosts left. Its high-B tail was clipped — and it would "
                  "have entered the analysis as a normal result.",
                  "Re-measured with real headroom: true demand is 34 of 100 — it exceeds the old "
                  "pool of 27 outright. The 27-ghost run understated its own demand because it was "
                  "clipped.",
              ],
              "table": [["Minimum pool remaining", "B ≤ −11.5", "−10.0", "−8.5", "−7.0", "−5.5", "−4.0"],
                        ["27-ghost run", "27", "26", "25", "26", "16", "3 / exhausted"],
                        ["100-ghost run", "100", "100", "100", "92–94", "87–91", "66–75"]],
              "widths": [0.28] + [0.12] * 6,
              "note": "Caught by the minimum_ghost_pool field recorded in every checkpoint. Without "
                      "it, the campaign would have produced 216 green checkmarks over a biased curve."})

    S.append({"title": "Result: titration curve",
              "kicker": "grey = individual replicas, blue = mean ± between-replica SEM",
              "image": str(A / f"{prefix}_combined_titration_curve.png"), "image_frac": 0.55,
              "bullets": [
                  "Clean sigmoid from ⟨N⟩ ≈ 0.2 to saturation at ~5 waters.",
                  "Using 6 replicas was the right call: the between-replica SEM exceeds the "
                  "within-replica error at 28 of 36 rungs, median 1.6×, peaking at 5.2×.",
                  "A single-replica analysis would have quoted an error bar ~5× too small in the "
                  "steep region, where it matters most.",
              ]})

    fe = [["N", "ΔF_bind", "step"]]
    prev = 0.0
    for n in sorted(prof):
        v = prof[n]["df"]
        fe.append([str(n), f"{v:+.2f}", "—" if n == 0 else f"{v - prev:+.2f}"])
        prev = v
    S.append({"title": "Result: binding free energy", "kicker": "minimum at N* = 5",
              "image": str(A / f"{prefix}_combined_free_energy.png"), "image_frac": 0.50,
              "table": fe, "widths": [0.25, 0.4, 0.35],
              "note": "Increments shrink monotonically — first water −10.19, fifth only −3.14. "
                      "N = 6 is extrapolated by construction, so the upturn is not itself measured; "
                      f"N* = 5 is independently supported by ⟨N⟩ = {pooled['n_at_equilibrium_b']:.2f} "
                      "at equilibrium."})

    S.append({"title": "How much do I trust it?",
              "kicker": "two estimators, two resampling methods, one sensitivity scan",
              "table": [["Check", "Result", "Reading"],
                        ["Pooled curve integrated", f"{df:+.2f} kcal/mol", "primary estimator"],
                        ["Mean of 6 per-replica ΔF",
                         f"{comb['df_bind_star_each_replica_own_n_star']['mean']:+.2f} kcal/mol",
                         "agrees to 0.15 — nonlinearity negligible"],
                        ["Bootstrap SD / jackknife SE",
                         f"{bs['sd']:.3f} / {pooled['jackknife']['se']:.3f}",
                         "two independent methods agree"],
                        ["N* across 2000 resamples", "5 in 2000/2000", "no ambiguity in N*"],
                        ["Equilibration cutoff 75/50/25/0%",
                         "−33.32 / −33.53 / −33.72 / −33.38", "spread 0.40, inside the CI"]],
              "widths": [0.28, 0.27, 0.45],
              "panel": "Why it is still labelled preliminary.  All six replicas fail the default "
                       "monotonicity gate, so an isotonic fit is required. But: pooled drift over "
                       "all 216 windows is +0.034 ± 0.031 waters (t = 1.12) — no equilibration "
                       "bias; 7 of 8 residual inversions are smaller than the SEM; and the isotonic "
                       "correction is at most 0.26 waters against a 0.29 SEM at that point. "
                       "This is plateau noise, not a convergence failure.",
              "panel_tone": "neutral"})

    S.append({"title": "Hydration inside the GCI sphere",
              "kicker": "ghost-masked, 150 frames (6 replicas × 25)",
              "table": [["Site", "Occupancy", "Spread (Å)", "From centre (Å)"]] +
                       [[str(s["index"]), f"{s['occupancy']:.2f}", f"{s['spread_angstrom']:.2f}",
                         f"{s['distance_from_sphere_centre_angstrom']:.2f}"]
                        for s in water["sites"][:6]],
              "widths": [0.2, 0.28, 0.26, 0.26], "table_frac": 0.38,
              "bullets": [
                  f"Ensemble occupancy {water['ensemble_mean_occupancy']:.2f} waters — independently "
                  "reproduces the titration ⟨N⟩ = 4.67 at this rung by a separate route. Good check "
                  "on the ghost masking.",
                  f"{len(water['sites'])} sites above 10% occupancy, 6 above 20%. Only two are well "
                  "defined (0.75, 0.63); the rest are diffuse.",
                  "All 4 waters in the equilibrated input are recovered, every one within 0.50 Å.",
                  "Signature of a partly solvent-exposed pocket, not buried ordered waters.",
              ]})

    S.append({"title": "Water-network analysis", "kicker": "reproducing the CRY1/AN139 method",
              "bullets": [
                  f"Region: {net['region_radius_angstrom']:g} Å around the sphere centre, pooling "
                  f"the three rungs bracketing B_equil (B = "
                  f"{', '.join(f'{b:g}' for b in net['target_b_pooled'])}) — 75 frames per replica.",
                  f"Per replica: average-linkage clustering of water oxygens at "
                  f"{net['parameters']['cluster_cutoff']:g} Å; occupancy = fraction of frames the "
                  f"site is filled. Ghosts excluded frame-by-frame.",
                  f"Matched to crystallographic waters within "
                  f"{net['parameters']['match_cutoff']:g} Å, minimum occupancy "
                  f"{net['parameters']['min_occupancy']:g} — the same parameters as the AN139 study.",
                  f"Consensus sites: second complete-linkage pass at "
                  f"{net['parameters']['consensus_cutoff']:g} Å, retained if present in ≥ "
                  f"{net['parameters']['consensus_fraction']:.0%} of replicas.",
              ],
              "panel": "Scope limit, stated up front. Grand-canonical sampling applied only inside "
                       "the 4 Å GCI sphere. Beyond it water is sampled by ordinary MD, so those "
                       "occupancies are ordinary MD occupancies — a site MD cannot fill will look "
                       "empty. Every site is flagged inside/outside the sphere. The AN139 study used "
                       "a 10 Å GCMC region throughout.",
              "panel_tone": "bad"})

    S.append({"title": "Agreement with crystallographic waters",
              "kicker": "6KX6 chain A — the copy the system was built from",
              "table": [["", "Cai — AN139", "Cai — Apo", "Cai — AN174", "This work — KL101"],
                        ["N_Xray", "9.0", "9.0", "5.0", f"{cry['waters_in_region']}"],
                        ["N_clusters", "288.1 ± 11.8", "397.5 ± 20.3", "279.8 ± 18.0",
                         f"{st['N_clusters']['mean']:.1f} ± {st['N_clusters']['sd']:.1f}"],
                        ["N_clusters (occ ≥ 0.2)", "36.8 ± 4.6", "67.9 ± 4.6", "34.5 ± 3.2",
                         f"{st['N_clusters(occ>=0.2)']['mean']:.1f} ± "
                         f"{st['N_clusters(occ>=0.2)']['sd']:.1f}"],
                        ["Matched", "7.8 ± 1.2", "7.8 ± 0.9", "3.7 ± 1.2",
                         f"{st['Matched']['mean']:.1f} ± {st['Matched']['sd']:.1f}"],
                        ["Recall", "0.870 ± 0.133", "0.861 ± 0.096", "0.733 ± 0.246",
                         f"{st['Recall']['mean']:.3f} ± {st['Recall']['sd']:.3f}"],
                        ["Precision", "0.211 ± 0.041", "0.114 ± 0.014", "0.104 ± 0.034",
                         f"{st['Precision']['mean']:.3f} ± {st['Precision']['sd']:.3f}"],
                        ["Tanimoto", "0.205 ± 0.044", "0.112 ± 0.015", "0.101 ± 0.035",
                         f"{st['Tanimoto']['mean']:.3f} ± {st['Tanimoto']['sd']:.3f}"],
                        ["Replicas", "12", "12", "12", f"{tasks['replicas']}"]],
              "widths": [0.22, 0.19, 0.19, 0.19, 0.21],
              "note": f"Recall {st['Recall']['mean']:.3f} sits just below AN139 (0.870) and above "
                      f"AN174 (0.733): the method recovers this pocket's hydration about as well "
                      f"for KL101 as for the other ligands. Low precision is expected in every "
                      f"column — a crystal model does not resolve transient waters."})

    S.append({"title": "Occupancy and matching vs B-factor",
              "kicker": "reproducing the AN139 study's Figure 5(b,c) for KL101",
              "image": str(A / f"{prefix}_bfactor_occupancy.png"), "image_frac": 0.60,
              "bullets": [
                  "Same trend as the AN139 study: high-B crystallographic waters match "
                  "low-occupancy simulated sites. The two most ordered (B = 14.6, 14.8 Å²) are "
                  "occupied 91% and 93%.",
                  "The converse fails — HOH 635, at a modest B = 23.8 Å², has no matching site at "
                  "all. A low B-factor does not guarantee a stable simulated site.",
                  "3 of 14 waters unmatched (HOH 635, 753, 708) — that is what limits recall to 0.774.",
                  "Matching distance shows no B-factor trend, supporting a single cutoff for all "
                  "waters — again as reported for AN139.",
              ]})

    S.append({"title": "Consensus hydration sites",
              "kicker": f"present in ≥75% of replicas — {ncons} sites, {nmatch} matched",
              "image": str(A / f"{prefix}_consensus_sites.png"), "image_frac": 0.62,
              "bullets": [
                  "Left: one bar per crystallographic water, ordered and shaded by B-factor. An "
                  "absent bar means no consensus site was found.",
                  "Right: the strongest consensus sites with no crystallographic counterpart — "
                  "several above 0.7 occupancy.",
                  f"{ncons - nmatch} of {ncons} consensus sites are reproducible across replicas yet "
                  "absent from the deposited model.",
                  "As the AN139 study observed: hydration that is thermodynamically favourable but "
                  "not crystallographically resolved.",
              ]})

    S.append({"title": "Crystal structure comparison",
              "kicker": "6KX6 — mouse CRY1 + KL101 (DYU), 2.00 Å, chain A",
              "table": [["Copy", "Cα superposed", "RMSD (Å)", "DYU vs simulated LIG",
                         "Waters in 4 Å sphere"]] +
                       [[f"chain {c['chain']}" + (" (reference)" if c["chain"] == "A" else ""),
                         str(c["superposed_atoms"]), f"{c['ca_rmsd_angstrom']:.2f}",
                         f"{c['ligand_centroid_offset_angstrom']:.2f} Å",
                         f"{c['waters_inside_sphere']} of {c['deposited_waters']}"]
                        for c in (water.get("crystal_comparison") or {"copies": []})["copies"]],
              "widths": [0.24, 0.19, 0.15, 0.22, 0.20],
              "bullets": [
                  "The mapping validates itself: superposing on backbone Cα alone puts the "
                  "crystallographic ligand 0.68–0.69 Å from the simulated one. The simulation "
                  "clearly derives from this structure.",
                  "Only one deposited water sits inside the 4 Å GCI sphere — and the two copies "
                  "disagree on which. Chain A has HOH 804, chain B has HOH 634. That disagreement "
                  "is itself evidence the region is weakly ordered.",
                  "Chain A is used throughout as the reference, matching how the system was built.",
              ]})

    S.append({"title": "Cross-check: the protocols are the same",
              "kicker": "L. Cai, M1 report 2026 — same pocket, different ligand",
              "table": [["", "CRY1/AN139 (Cai)", "CRY1/KL101 (this work)"],
                        ["Adams ladder", "36 values, −30 → −4, variable spacing", "identical"],
                        ["Schedule", "625 × (400 GCMC + 8 ps MD) = 5 ns", "identical"],
                        ["μ′_sol / V°", "−6.09 kcal/mol / 30.345 Å³", "identical"],
                        ["B_equil", "−8.038", f"{comb['equilibrium_b']:.4f}"],
                        ["Force field", "ff14SB / TIP3P / GAFF", "identical"],
                        ["GCMC engine", "grand (Samways)", "loch 2026.1.0"],
                        ["Replicas", "12", f"{tasks['replicas']}"]],
              "widths": [0.18, 0.42, 0.40],
              "panel": "B_equil agrees to four decimal places. Two independent GCMC codes deriving "
                       "the same equilibrium Adams value from the same calibration constants — a "
                       "meaningful cross-validation that our setup is doing the same physics.",
              "panel_tone": "good"})

    S.append({"title": "Cross-check: free energies", "kicker": "the comparison that raises a flag",
              "table": [["Region", "Radius", "X-ray waters", "N*", "ΔF_bind(N*)", "Per water",
                         "Occupancy vs bulk"],
                        ["Cai, Sphere 2", "4 Å", "2", "2", "−9.40", "−4.70", "0.23"],
                        ["Cai, Sphere 1 (rejected)", "7 Å", "7", "25 (expected 7)", "−157.17",
                         "−6.29", "0.53"],
                        ["This work", "4 Å", "1", "5", f"{df:.2f}", "−6.71", "0.57"]],
              "widths": [0.22, 0.09, 0.13, 0.16, 0.14, 0.11, 0.15],
              "bullets": [
                  "Cai reports Sphere 1 as not meaningful: N* = 25 where 7 was expected, "
                  "ΔF = −157 kcal/mol. Diagnosis: diffusive exchange.",
                  "In GCMC/MD the region boundary is permeable — water enters and leaves during MD. "
                  "When that flux rivals GCMC exchange, the GCI assumptions break and the curve is "
                  "biased. The effect scales with region volume.",
              ]})

    S.append({"title": "Does that caveat apply to us?",
              "kicker": "partly — and it deserves attention",
              "twopanel": [
                  ("Reassuring", "good", [
                      "Our region is 4 Å — the size Cai validated, and one eighth the volume of the "
                      "one that failed.",
                      "His diagnostic is whether occupancy falls to zero at unfavourable B. Ours "
                      "reaches ⟨N⟩ = 0.217 ± 0.008 at B = −30 and descends smoothly.",
                      "So GCMC, not diffusion, is controlling the occupancy here.",
                  ]),
                  ("Cautionary", "bad", [
                      "Our site is far more solvent-accessible: 0.57 of bulk density at N*, vs 0.23 "
                      "for his validated sphere — and 0.53 for the one he rejected.",
                      "Per water, our −6.71 sits closer to his biased case (−6.29) than to his "
                      "trusted one (−4.70).",
                      "A more open site permits more diffusive exchange at the same radius.",
                  ])],
              "panel": f"How I would read it.  Treat the absolute ΔF_bind as carrying an "
                       f"unquantified systematic component beyond the ±0.88 kcal/mol statistical "
                       f"CI. Cai makes the same argument for his own systems, noting that relative "
                       f"comparisons survive such a shift far better than absolute values. "
                       f"Comparing this pocket across ligands, or against the apo form, would be on "
                       f"much firmer ground than quoting {df:.2f} kcal/mol on its own.",
              "panel_tone": "neutral"})

    S.append({"title": "Open items and what I would do next", "kicker": "in priority order",
              "table": [["", "Item", "Why it matters", "Cost"],
                        ["1", "Confirm the sphere centre",
                         "Still the shipped placeholder — the README calls it “a working "
                         "placeholder, not a chosen site”. It decides what these 164 GPU-hours "
                         "measured.", "none — a decision"],
                        ["2", "Ligand-scale comparison",
                         "The step least affected by the diffusion caveat, and the natural "
                         "continuation of the AN139 work: KL101 vs AN139 vs AN174 vs apo.",
                         "~165 GPU-h each"],
                        ["3", "Quantify the diffusion bias",
                         "Pure GCMC with an impermeable boundary would test the absolute value "
                         "directly.", "new calculation"],
                        ["4", "Wider water-mapping region",
                         "A 10 Å GCMC region would make recall directly comparable to AN139 and "
                         "cover ~30 crystallographic waters instead of 14.", "GCMC/MD run"],
                        ["5", "Tighter error bar (optional)",
                         "Adding replicas is verified incremental — existing windows are skipped. "
                         "24 replicas halves the CI to ±0.44.", "~493 GPU-h"]],
              "widths": [0.03, 0.20, 0.62, 0.15]})

    S.append({"title": "Backup: reproducing this", "kicker": "everything is in the campaign directory",
              "code": "export CONDA_OVERRIDE_CUDA=\"12.8\"        # login node has no GPU\n"
                      "conda env create -f environment.yml       # nvcc MUST land inside the env\n"
                      "python -m openmm.testInstallation         # all four platforms must pass\n\n"
                      "SMOKE=1 ./run_window.sh 0                 # ~2.5 min sanity check\n"
                      "sbatch submit_gci.slurm                   # --array=0-215%12\n\n"
                      "python combine_replicas.py                # per replica, then pooled\n"
                      "python water_sites.py --crystal inputs/6kx6.pdb\n"
                      "python water_network.py --chain A         # Cai-style network analysis\n"
                      "python make_report.py && python make_slides.py",
              "bullets": [
                  "scripts/ is unmodified. The four new scripts import compute_df_bind, "
                  "isotonic_non_decreasing, monotonicity_violations and kabsch from the campaign's "
                  "own modules rather than reimplementing the physics.",
                  "Re-submitting is safe: finished windows carry verified checkpoints and are skipped.",
                  "Provenance: input SHA-256s and tasks.tsv hash in tasks.json; 11 outputs hashed "
                  "per window.",
                  "Full detail in the 11-page report, analysis/CRY1KL101_GCI_report.pdf.",
              ]})
    return S


# ---------------------------------------------------------------------------
# PPTX renderer
# ---------------------------------------------------------------------------
def render_pptx(slides, out):
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.enum.text import PP_ALIGN

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    SW, SH = 13.333, 7.5
    M = 0.55

    def rgb(h):
        return RGBColor.from_string(h)

    def textbox(sl, x, y, w, h, blocks, size=13, color="222222", space=6, bullet=False):
        tb = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame
        tf.word_wrap = True
        for i, t in enumerate(blocks):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = ("•  " + t) if bullet else t
            p.space_after = Pt(space)
            for r in p.runs:
                r.font.size = Pt(size)
                r.font.color.rgb = rgb(color)
                r.font.name = "Calibri"
        return tb

    def header(sl, title, kicker):
        bar = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.13))
        bar.fill.solid()
        bar.fill.fore_color.rgb = rgb(ACC)
        bar.line.fill.background()
        tb = sl.shapes.add_textbox(Inches(M), Inches(0.28), Inches(SW - 2 * M), Inches(0.6))
        p = tb.text_frame.paragraphs[0]
        p.text = title
        p.runs[0].font.size = Pt(28)
        p.runs[0].font.bold = True
        p.runs[0].font.color.rgb = rgb(ACC)
        y = 0.95
        if kicker:
            k = sl.shapes.add_textbox(Inches(M), Inches(0.92), Inches(SW - 2 * M), Inches(0.32))
            kp = k.text_frame.paragraphs[0]
            kp.text = kicker
            kp.runs[0].font.size = Pt(13)
            kp.runs[0].font.color.rgb = rgb(MUT)
            y = 1.32
        ln = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(M), Inches(y),
                                 Inches(SW - 2 * M), Inches(0.012))
        ln.fill.solid()
        ln.fill.fore_color.rgb = rgb("C9D6E2")
        ln.line.fill.background()
        return y + 0.18

    def add_table(sl, rows, widths, x, y, w, size=11):
        nr, nc = len(rows), len(rows[0])
        h = 0.34 * nr
        shp = sl.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(h))
        t = shp.table
        for j, frac in enumerate(widths[:nc]):
            t.columns[j].width = Inches(w * frac)
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                cell = t.cell(i, j)
                cell.text = str(val)
                cell.margin_left = Inches(0.06)
                cell.margin_right = Inches(0.06)
                cell.margin_top = Inches(0.02)
                cell.margin_bottom = Inches(0.02)
                pr = cell.text_frame.paragraphs[0]
                for r in pr.runs:
                    r.font.size = Pt(size)
                    r.font.name = "Calibri"
                    if i == 0:
                        r.font.bold = True
                        r.font.color.rgb = rgb("FFFFFF")
                cell.fill.solid()
                cell.fill.fore_color.rgb = rgb(ACC if i == 0 else
                                               ("FFFFFF" if i % 2 else "EDF2F7"))
        return y + h + 0.12

    def est_height(text, w_in, size, pad=0.22):
        """Height in inches for `text` wrapped into a box `w_in` wide at `size` pt.

        Average glyph advance is about 0.5 em, so a line holds w * 144 / size
        characters; line pitch is 1.25 em.
        """
        cpl = max(10.0, w_in * 144.0 / size)
        lines = 0
        for para in str(text).split("\n"):
            lines += max(1, -(-len(para) // int(cpl)))
        return pad + lines * (size * 1.25 / 72.0)

    def add_panel(sl, text, x, y, w, tone="neutral", size=12.5):
        bg = {"good": GOODBG, "bad": BADBG, "neutral": NEUTBG}[tone]
        est = max(0.5, est_height(text, w - 0.3, size, 0.26))
        est = min(est, SH - 0.45 - y)
        box = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
                                  Inches(w), Inches(est))
        box.fill.solid()
        box.fill.fore_color.rgb = rgb(bg)
        box.line.color.rgb = rgb("C9D6E2")
        box.line.width = Pt(0.75)
        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.14)
        tf.margin_right = Inches(0.14)
        tf.margin_top = Inches(0.08)
        p = tf.paragraphs[0]
        p.text = text
        p.alignment = PP_ALIGN.LEFT
        for r in p.runs:
            r.font.size = Pt(size)
            r.font.color.rgb = rgb("222222")
            r.font.name = "Calibri"
        return y + est + 0.12

    for i, spec in enumerate(slides):
        sl = prs.slides.add_slide(prs.slide_layouts[6])
        if spec.get("kind") == "title":
            bar = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.22))
            bar.fill.solid()
            bar.fill.fore_color.rgb = rgb(ACC)
            bar.line.fill.background()
            tb = sl.shapes.add_textbox(Inches(M), Inches(1.5), Inches(SW - 2 * M), Inches(1.2))
            p = tb.text_frame.paragraphs[0]
            p.text = spec["title"]
            p.runs[0].font.size = Pt(46)
            p.runs[0].font.bold = True
            p.runs[0].font.color.rgb = rgb(ACC)
            textbox(sl, M, 2.6, SW - 2 * M, 0.5, [spec["sub"]], size=18, color=MUT)
            textbox(sl, M, 3.25, SW - 2 * M, 0.5, [spec["line"]], size=14)
            textbox(sl, M, 3.8, SW - 2 * M, 0.4, [spec["foot"]], size=11, color=MUT)
            add_panel(sl, spec["panel"], M, 4.6, SW - 2 * M, "neutral", 14)
        else:
            y = header(sl, spec["title"], spec.get("kicker"))
            full = SW - 2 * M
            if "big" in spec:
                textbox(sl, M, y, full, 0.7, [spec["big"]], size=30, color=ACC)
                textbox(sl, M, y + 0.75, full, 0.4, [spec["bigsub"]], size=15, color=MUT)
                y += 1.35
            if "image" in spec and Path(spec["image"]).exists():
                iw = full * spec.get("image_frac", 0.55)
                sl.shapes.add_picture(spec["image"], Inches(M), Inches(y), width=Inches(iw))
                rx, rw = M + iw + 0.25, full - iw - 0.25
                if spec.get("bullets"):
                    textbox(sl, rx, y, rw, 4.5, spec["bullets"], size=12, bullet=True, space=9)
                if spec.get("table"):
                    add_table(sl, spec["table"], spec.get("widths", [1 / len(spec["table"][0])] *
                                                          len(spec["table"][0])), rx, y, rw, 10.5)
                if spec.get("note"):
                    textbox(sl, M, SH - 1.35, full, 1.0, [spec["note"]], size=10.5, color=MUT)
                continue
            if spec.get("twopanel"):
                cw = (full - 0.3) / 2
                for j, (ttl, tone, items) in enumerate(spec["twopanel"]):
                    x = M + j * (cw + 0.3)
                    bg = {"good": GOODBG, "bad": BADBG}[tone]
                    box = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
                                              Inches(cw), Inches(2.75))
                    box.fill.solid()
                    box.fill.fore_color.rgb = rgb(bg)
                    box.line.color.rgb = rgb("9CC79C" if tone == "good" else "D8A08A")
                    tf = box.text_frame
                    tf.word_wrap = True
                    tf.margin_left = Inches(0.15)
                    tf.margin_top = Inches(0.1)
                    p = tf.paragraphs[0]
                    p.text = ttl
                    p.runs[0].font.size = Pt(16)
                    p.runs[0].font.bold = True
                    p.runs[0].font.color.rgb = rgb("1E6B34" if tone == "good" else "8B2500")
                    for it in items:
                        q = tf.add_paragraph()
                        q.text = "•  " + it
                        q.space_before = Pt(5)
                        for r in q.runs:
                            r.font.size = Pt(11.5)
                            r.font.name = "Calibri"
                y += 2.95
            if spec.get("table") and "image" not in spec:
                tf_ = spec.get("table_frac")
                tw = full * tf_ if tf_ else full
                yy = add_table(sl, spec["table"], spec.get("widths",
                               [1 / len(spec["table"][0])] * len(spec["table"][0])),
                               M, y, tw, 10.5 if len(spec["table"][0]) > 4 else 11.5)
                if tf_ and spec.get("bullets"):
                    textbox(sl, M + tw + 0.3, y, full - tw - 0.3, 4.2, spec["bullets"],
                            size=12, bullet=True, space=9)
                    spec = {**spec, "bullets": None}
                y = yy
            if spec.get("code"):
                box = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(M), Inches(y),
                                          Inches(full), Inches(2.5))
                box.fill.solid()
                box.fill.fore_color.rgb = rgb("EDF2F7")
                box.line.color.rgb = rgb("C9D6E2")
                tfx = box.text_frame
                tfx.word_wrap = True
                tfx.margin_left = Inches(0.15)
                for j, line in enumerate(spec["code"].split("\n")):
                    p = tfx.paragraphs[0] if j == 0 else tfx.add_paragraph()
                    p.text = line
                    for r in p.runs:
                        r.font.size = Pt(11)
                        r.font.name = "Consolas"
                y += 2.65
            if spec.get("bullets"):
                bh = sum(est_height(b, full - 0.3, 13, 0.0) for b in spec["bullets"]) \
                    + 0.14 * len(spec["bullets"])
                textbox(sl, M, y, full, max(0.5, bh), spec["bullets"], size=13,
                        bullet=True, space=10)
                y += bh + 0.18
            if spec.get("panel"):
                y = add_panel(sl, spec["panel"], M, min(y, SH - 2.0), full,
                              spec.get("panel_tone", "neutral"))
            if spec.get("note"):
                textbox(sl, M, min(y + 0.05, SH - 1.15), full, 1.0, [spec["note"]],
                        size=10.5, color=MUT)
        # slide number
        n = sl.shapes.add_textbox(Inches(SW - 1.0), Inches(SH - 0.42), Inches(0.6), Inches(0.3))
        pn = n.text_frame.paragraphs[0]
        pn.text = str(i + 1)
        pn.runs[0].font.size = Pt(10)
        pn.runs[0].font.color.rgb = rgb(MUT)

    prs.save(str(out))
    return len(slides)


# ---------------------------------------------------------------------------
def render_pdf(slides, out):
    """Simple PDF mirror of the same content, for inline preview."""
    import matplotlib
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas as pdfcanvas
    from reportlab.platypus import Frame, Image, Paragraph, Spacer, Table, TableStyle

    F = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    for nm, fn in (("DJ", "DejaVuSans.ttf"), ("DJ-B", "DejaVuSans-Bold.ttf"),
                   ("DJ-M", "DejaVuSansMono.ttf")):
        try:
            pdfmetrics.registerFont(TTFont(nm, F / fn))
        except Exception:
            pass
    W, H = 13.333 * inch, 7.5 * inch
    acc = colors.HexColor("#" + ACC)
    mut = colors.HexColor("#" + MUT)
    S = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=S["BodyText"], fontName="DJ", fontSize=12.5, leading=17)
    bull = ParagraphStyle("bu", parent=body, leftIndent=14, bulletIndent=2, spaceAfter=8)
    small = ParagraphStyle("s", parent=body, fontSize=10, leading=13.5, textColor=mut)
    cell = ParagraphStyle("c", parent=body, fontSize=10, leading=13)
    cellh = ParagraphStyle("ch", parent=cell, fontName="DJ-B", textColor=colors.white)
    c = pdfcanvas.Canvas(str(out), pagesize=(W, H))
    M = 0.55 * inch

    def flow(items, x, ytop, w, h):
        Frame(x, ytop - h, w, h, showBoundary=0, leftPadding=0, rightPadding=0,
              topPadding=0, bottomPadding=0).addFromList(list(items), c)

    for i, sp in enumerate(slides):
        if sp.get("kind") == "title":
            c.setFillColor(acc)
            c.rect(0, H - 0.3 * inch, W, 0.3 * inch, stroke=0, fill=1)
            flow([Paragraph(sp["title"], ParagraphStyle("t", parent=body, fontName="DJ-B",
                                                        fontSize=40, leading=46, textColor=acc)),
                  Spacer(1, 10), Paragraph(sp["sub"], ParagraphStyle("x", parent=body,
                                                                     fontSize=17, textColor=mut)),
                  Spacer(1, 14), Paragraph(sp["line"], body), Spacer(1, 6),
                  Paragraph(sp["foot"], small), Spacer(1, 18),
                  Paragraph(sp["panel"], body)],
                 M, H - 1.6 * inch, W - 2 * M, H - 2.4 * inch)
        else:
            c.setFillColor(acc)
            c.rect(0, H - 0.16 * inch, W, 0.16 * inch, stroke=0, fill=1)
            c.setFont("DJ-B", 22)
            c.drawString(M, H - 0.78 * inch, sp["title"])
            y = H - 1.02 * inch
            if sp.get("kicker"):
                c.setFillColor(mut)
                c.setFont("DJ", 11.5)
                c.drawString(M, H - 1.06 * inch, sp["kicker"])
                y = H - 1.3 * inch
            c.setStrokeColor(colors.HexColor("#C9D6E2"))
            c.line(M, y, W - M, y)
            items = []
            if "big" in sp:
                items += [Paragraph(sp["big"], ParagraphStyle("bg", parent=body, fontName="DJ-B",
                                                              fontSize=26, leading=31,
                                                              textColor=acc)),
                          Paragraph(sp["bigsub"], ParagraphStyle("bs", parent=body, fontSize=14,
                                                                 textColor=mut)), Spacer(1, 12)]
            if sp.get("image") and Path(sp["image"]).exists():
                # Image left, text right -- mirroring the PPTX layout. Stacking them
                # in one frame silently drops the text when the picture is tall.
                from PIL import Image as PIL
                iw0, ih0 = PIL.open(sp["image"]).size
                full = W - 2 * M
                iw = full * sp.get("image_frac", 0.55)
                avail = y - 0.75 * inch
                ih = min(iw * ih0 / iw0, avail)
                flow([Image(sp["image"], width=ih * iw0 / ih0 if ih < iw * ih0 / iw0 else iw,
                            height=ih)], M, y - 0.12 * inch, iw, avail)
                right = [Paragraph(x, bull, bulletText="•") for x in sp.get("bullets") or []]
                if sp.get("table"):
                    rows = sp["table"]
                    nc = len(rows[0])
                    fr = sp.get("widths", [1 / nc] * nc)
                    data = [[Paragraph(str(v), cellh if r == 0 else cell) for v in row]
                            for r, row in enumerate(rows)]
                    t = Table(data, colWidths=[(full - iw - 0.25 * inch) * f for f in fr[:nc]])
                    t.setStyle(TableStyle([
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#B8C4D0")),
                        ("BACKGROUND", (0, 0), (-1, 0), acc),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                         [colors.white, colors.HexColor("#EDF2F7")]),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 4),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
                    right = [t, Spacer(1, 8)] + right
                flow(right, M + iw + 0.25 * inch, y - 0.12 * inch,
                     full - iw - 0.25 * inch, avail)
                if sp.get("note"):
                    flow([Paragraph(sp["note"], small)], M, 1.05 * inch, full, 0.75 * inch)
                c.setFillColor(mut)
                c.setFont("DJ", 9)
                c.drawRightString(W - 0.5 * inch, 0.3 * inch, str(i + 1))
                c.showPage()
                continue
            for key in ("table",):
                if sp.get(key):
                    rows = sp[key]
                    nc = len(rows[0])
                    fr = sp.get("widths", [1 / nc] * nc)
                    data = [[Paragraph(str(x), cellh if r == 0 else cell) for x in row]
                            for r, row in enumerate(rows)]
                    t = Table(data, colWidths=[(W - 2 * M) * f for f in fr[:nc]])
                    t.setStyle(TableStyle([
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#B8C4D0")),
                        ("BACKGROUND", (0, 0), (-1, 0), acc),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                         [colors.white, colors.HexColor("#EDF2F7")]),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 5),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
                    items += [t, Spacer(1, 10)]
            if sp.get("twopanel"):
                for ttl, tone, its in sp["twopanel"]:
                    items += [Paragraph(f"<b>{ttl}</b>", body)]
                    items += [Paragraph(x, bull, bulletText="•") for x in its]
                    items += [Spacer(1, 6)]
            if sp.get("code"):
                items += [Paragraph(sp["code"].replace("\n", "<br/>"),
                                    ParagraphStyle("cd", parent=body, fontName="DJ-M", fontSize=9.5,
                                                   leading=13,
                                                   backColor=colors.HexColor("#EDF2F7"),
                                                   borderPadding=6)), Spacer(1, 10)]
            if sp.get("bullets"):
                items += [Paragraph(x, bull, bulletText="•") for x in sp["bullets"]]
            if sp.get("panel"):
                items += [Spacer(1, 6), Paragraph(sp["panel"], body)]
            if sp.get("note"):
                items += [Spacer(1, 6), Paragraph(sp["note"], small)]
            flow(items, M, y - 0.12 * inch, W - 2 * M, y - 0.55 * inch)
        c.setFillColor(mut)
        c.setFont("DJ", 9)
        c.drawRightString(W - 0.5 * inch, 0.3 * inch, str(i + 1))
        c.showPage()
    c.save()
    return len(slides)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=("pptx", "pdf", "both"), default="both")
    ap.add_argument("--prefix", default="CRY1KL101")
    ap.add_argument("--stem", type=Path,
                    default=HERE / "analysis" / "CRY1KL101_GCI_slides")
    o = ap.parse_args()
    slides = build(o.prefix)
    o.stem.parent.mkdir(parents=True, exist_ok=True)
    if o.format in ("pptx", "both"):
        n = render_pptx(slides, o.stem.with_suffix(".pptx"))
        print(f"wrote {o.stem.with_suffix('.pptx')}  ({n} slides)")
    if o.format in ("pdf", "both"):
        n = render_pdf(slides, o.stem.with_suffix(".pdf"))
        print(f"wrote {o.stem.with_suffix('.pdf')}  ({n} slides)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
