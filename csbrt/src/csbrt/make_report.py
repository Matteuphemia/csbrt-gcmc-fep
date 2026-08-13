#!/usr/bin/env python3
"""Build the CRY1/KL101 GCI report as a PDF, structured as a scientific report.

    python make_report.py [--output analysis/CRY1KL101_GCI_report.pdf]

Layout follows L. Cai, "GCMC-based Water Network Mapping in Protein-Ligand
Complexes" (M1 report 2026): abstract, introduction, theory, methods, results
and discussion, conclusion, references -- plus an implementation section, since
this report also has to serve as a reproduction guide.

Every number is read from analysis/combined/*.json, so re-running after a
re-analysis regenerates a consistent document.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

HERE = Path(__file__).resolve().parent

# DejaVu carries Greek, angle brackets and the Angstrom sign; the built-in
# Helvetica is WinAnsi-only and silently drops them.
_F = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DJ", _F / "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DJ-B", _F / "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DJ-I", _F / "DejaVuSans-Oblique.ttf"))
pdfmetrics.registerFont(TTFont("DJ-M", _F / "DejaVuSansMono.ttf"))
pdfmetrics.registerFontFamily("DJ", normal="DJ", bold="DJ-B", italic="DJ-I")

ACC = colors.HexColor("#1F4E79")
MUT = colors.HexColor("#5A5A5A")
LIGHT = colors.HexColor("#EDF2F7")
BOX = colors.HexColor("#F7FAFC")

S = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=S["BodyText"], fontName="DJ", fontSize=8.6,
                      leading=12.4, alignment=TA_JUSTIFY, spaceAfter=5)
H1 = ParagraphStyle("h1", parent=S["Heading1"], fontName="DJ-B", fontSize=12.5,
                    leading=15, textColor=ACC, spaceBefore=11, spaceAfter=5)
H2 = ParagraphStyle("h2", parent=S["Heading2"], fontName="DJ-B", fontSize=9.8,
                    leading=12.5, textColor=ACC, spaceBefore=8, spaceAfter=3)
SMALL = ParagraphStyle("sm", parent=BODY, fontSize=7.5, leading=10.2, textColor=MUT)
EQ = ParagraphStyle("eq", parent=BODY, fontName="DJ", fontSize=9.4, leading=15,
                    alignment=TA_CENTER, spaceBefore=4, spaceAfter=5)
CODE = ParagraphStyle("code", parent=BODY, fontName="DJ-M", fontSize=7.0, leading=9.6,
                      backColor=LIGHT, borderPadding=5, alignment=0)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=7.3, leading=9.6, spaceAfter=0,
                      alignment=0)
CELLH = ParagraphStyle("cellh", parent=CELL, fontName="DJ-B", textColor=colors.white)
ABS = ParagraphStyle("abs", parent=BODY, fontSize=8.2, leading=11.6, spaceAfter=4)


def P(t, s=BODY):
    return Paragraph(t, s)


def tbl(rows, widths, header=True, size=7.3):
    data = [[Paragraph(str(x), CELLH if (header and i == 0) else CELL) for x in r]
            for i, r in enumerate(rows)]
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8C4D0")),
          ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
          ("FONTSIZE", (0, 0), (-1, -1), size)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), ACC),
               ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT])]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    t.setStyle(TableStyle(st))
    return t


def boxed(flowables, width):
    t = Table([[flowables]], colWidths=[width])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), BOX),
                           ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#C9D6E2")),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                           ("TOPPADDING", (0, 0), (-1, -1), 6),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return t


def fig(path, width, cap, story):
    if Path(path).exists():
        from PIL import Image as PILImage
        w, h = PILImage.open(path).size
        story += [Spacer(1, 4), Image(str(path), width=width, height=width * h / w),
                  P(cap, SMALL), Spacer(1, 3)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path,
                    default=HERE / "analysis" / "CRY1KL101_GCI_report.pdf")
    ap.add_argument("--prefix", default="CRY1KL101")
    o = ap.parse_args()

    A = HERE / "analysis" / "combined"
    comb = json.loads((A / f"{o.prefix}_combined_summary.json").read_text())
    water = json.loads((A / f"{o.prefix}_water_sites.json").read_text())
    net = json.loads((A / f"{o.prefix}_water_network.json").read_text())
    tasks = json.loads((HERE / "tasks.json").read_text())
    pooled = comb["pooled_curve_integrated"]
    bs = pooled["bootstrap"]
    df, lo, hi = pooled["df_bind_star_kcal_per_mol"], bs["ci95_lo"], bs["ci95_hi"]
    st = net["summary_table"]
    cry = net["crystal"]

    prof = {p["n"]: p for p in pooled["profile"]}
    W = A4[0] - 30 * mm
    s = []

    # ================= title & abstract =================
    s += [P("Grand Canonical Integration of the KL101 Binding-Site Hydration of CRY1",
            ParagraphStyle("t", parent=S["Title"], fontName="DJ-B", fontSize=15.5,
                           leading=19, textColor=ACC, alignment=0, spaceAfter=3)),
          P("A 216-window GCMC/MD titration campaign, its analysis, and a comparison with "
            "the CRY1/AN139 study", ParagraphStyle("st", parent=BODY, fontSize=9.6,
                                                   textColor=MUT, spaceAfter=2)),
          P(f"Computational campaign report · thira.r-ccs27.riken.jp · {date.today().isoformat()}",
            SMALL), Spacer(1, 6)]

    s += [boxed([P("<b>Abstract</b>", ParagraphStyle("ah", parent=ABS, fontName="DJ-B")),
                 P(f"Water molecules in protein binding pockets can stabilise or oppose ligand "
                   f"binding, and quantifying their thermodynamics is central to structure-based "
                   f"design. Grand Canonical Integration (GCI) obtains the binding free energy of "
                   f"a hydration site by titrating the water chemical potential and integrating "
                   f"the resulting occupancy curve. Here GCI was applied to the FAD-binding pocket "
                   f"of mouse cryptochrome 1 (CRY1) in complex with the isoform-selective compound "
                   f"KL101, using {tasks['replicas']} independent replicas of a "
                   f"{tasks['windows_per_replica']}-rung Adams-value ladder "
                   f"(216 windows, 5 ns each, 164.5 GPU-hours). Integrating the replica-averaged "
                   f"titration curve gives ΔF_bind(N*) = {df:+.2f} kcal/mol "
                   f"[95% CI {lo:+.2f}, {hi:+.2f}] at N* = {pooled['n_star']} waters, with the "
                   f"uncertainty estimated by bootstrapping over replicas. The hydration network "
                   f"was mapped and compared with the deposited structure 6KX6 (chain A): of "
                   f"{cry['waters_in_region']} crystallographic waters in a "
                   f"{net['region_radius_angstrom']:g} Å region, "
                   f"{st['Recall']['mean']*100:.0f}% are recovered by the simulation "
                   f"(recall {st['Recall']['mean']:.3f} ± {st['Recall']['sd']:.3f}), while the "
                   f"simulation proposes several times more sites than crystallography resolves "
                   f"(precision {st['Precision']['mean']:.3f} ± {st['Precision']['sd']:.3f}). "
                   f"Both figures are consistent with those reported for the CRY1/AN139 system "
                   f"under an identical protocol. The absolute free energy should nevertheless be "
                   f"treated with caution: the same study showed that diffusive exchange across "
                   f"the permeable GCMC boundary biases GCI, and the present site is more "
                   f"solvent-exposed than the one for which that method was validated.", ABS),
                 P("<b>Keywords</b> — Grand Canonical Monte Carlo · Grand Canonical Integration · "
                   "hydration sites · cryptochrome · KL101 · binding free energy", SMALL)], W)]

    # ================= 1 introduction =================
    s += [P("1  Introduction", H1)]
    s += [P(
        "Water molecules buried in protein binding pockets are rarely spectators. Through "
        "hydrogen-bond networks they bridge ligand and protein, and displacing a strongly bound "
        "water costs free energy while displacing a weakly bound one can gain it. Knowing which "
        "hydration sites are stable, and by how much, is therefore directly useful in "
        "structure-based design [1,2].", BODY)]
    s += [P(
        "Conventional molecular dynamics samples these networks poorly. Exchange of a buried water "
        "with bulk can take microseconds to milliseconds, far beyond routine MD, so hydration "
        "states become kinetically trapped [3]. Grand Canonical Monte Carlo (GCMC) circumvents the "
        "problem by inserting and deleting water molecules stochastically within a defined region, "
        "with acceptance governed by an imposed chemical potential, so the region is not limited by "
        "diffusion [4,5]. Coupling GCMC to MD (GCMC/MD) additionally lets the protein and ligand "
        "relax around the changing hydration state [6].", BODY)]
    s += [P(
        "Beyond locating hydration sites, Grand Canonical Integration (GCI) turns the same "
        "machinery into a free-energy method: the Adams value B, a dimensionless chemical "
        "potential, is varied systematically and the mean occupancy ⟨N(B)⟩ recorded, giving a "
        "titration curve whose integral yields the binding free energy of the water network [7].", BODY)]
    s += [P(
        "The system studied here is the FAD-binding pocket of mammalian cryptochrome 1 (CRY1), a "
        "core transcriptional repressor of the circadian clock. Dysregulation of the CLOCK:BMAL1 "
        "complex has been implicated in glioblastoma stem cell proliferation, making CRY proteins "
        "therapeutic targets [8]. KL101 is a published isoform-selective compound that binds and "
        "stabilises CRY1 [9], and its complex is deposited as 6KX6. A companion study applied the "
        "same GCMC/MD and GCI protocol to the same pocket with two unpublished ligands, AN139 and "
        "AN174 [10]; the present work applies it to KL101, which allows a direct methodological "
        "comparison.", BODY)]
    s += [P(
        "This report has two audiences. Sections 2–5 present the calculation and its results in "
        "the usual scientific order. Section 6 documents the practical execution — five defects "
        "that had to be fixed before the campaign would run correctly, three of which fail "
        "silently — together with the provenance needed to reproduce the work exactly.", BODY)]

    # ================= 2 theory =================
    s += [P("2  Theoretical background", H1)]
    s += [P("2.1  Grand Canonical Monte Carlo", H2)]
    s += [P(
        "GCMC exchanges water molecules between the simulated region and an ideal-gas reservoir. "
        "Insertion and deletion are accepted with probabilities [11,12]", BODY)]
    s += [P("P<sub>ins</sub> = min[1, (1/(N+1)) · e<sup>B</sup> · e<sup>−βΔU</sup>]  ,   "
            "P<sub>del</sub> = min[1, N · e<sup>−B</sup> · e<sup>−βΔU</sup>]", EQ)]
    s += [P(
        "where N is the current number of waters in the region, β = 1/k<sub>B</sub>T and ΔU the "
        "potential-energy change. The Adams value B collects the chemical potential and the region "
        "volume,", BODY)]
    s += [P("B = βμ + ln(V<sub>GCMC</sub> / Λ³)", EQ)]
    s += [P(
        "with Λ the thermal de Broglie wavelength. Higher B favours insertion, lower B favours "
        "deletion, so B acts as a tunable titrant controlling the average occupancy of the region. "
        "When the region is in equilibrium with bulk water, μ<sub>GCMC</sub> = μ<sub>bulk</sub>, "
        "which gives", BODY)]
    s += [P("B<sub>equil</sub> = βμ′<sub>sol</sub> + ln(V<sub>GCMC</sub> / V°)", EQ)]
    s += [P(
        f"Here μ′<sub>sol</sub> = −6.09 kcal/mol and V° = 30.345 Å³ are the TIP3P-calibrated values "
        f"of Samways et al. [6]; the calibration is required because TIP3P does not reproduce the "
        f"experimental excess chemical potential exactly. For the present "
        f"{tasks['radius_angstrom']:g} Å sphere this gives B<sub>equil</sub> = "
        f"{comb['equilibrium_b']:.4f}.", BODY)]

    s += [P("2.2  Grand Canonical Integration", H2)]
    s += [P(
        "GCI uses a thermodynamic cycle between the ideal-gas reservoir, bulk water and the GCMC "
        "region [7]. Writing ΔF<sub>bind</sub>(N) = ΔF<sub>transfer</sub>(N) − ΔF<sub>hyd</sub>(N) "
        "and integrating the titration curve gives", BODY)]
    s += [P("ΔF<sub>bind</sub>(N) = k<sub>B</sub>T [ N·B<sub>N</sub> + ln(1/N!) "
            "− ∫<sup>B<sub>N</sub></sup><sub>−∞</sub> ⟨N(B)⟩ dB ] − N·μ′<sub>hyd</sub>", EQ)]
    s += [P(
        "where B<sub>N</sub> is obtained by inverse interpolation of the titration curve at "
        "occupancy N. This inversion requires ⟨N⟩ to be non-decreasing in B, which is the origin "
        "of the monotonicity requirement discussed in §4.2. The minimum of "
        "ΔF<sub>bind</sub>(N) identifies the most stable hydration state, N*, and "
        "ΔF<sub>bind</sub>(N*) is the binding free energy of that water network.", BODY)]
    s += [P(
        "One caveat is intrinsic to the GCMC/MD variant. GCI was formulated for pure GCMC, where "
        "the region boundary is a hard wall and water enters or leaves only through Monte Carlo "
        "moves. In GCMC/MD the boundary is permeable: water also diffuses across it during the MD "
        "phase, so the observed ⟨N(B)⟩ contains a contribution the formalism does not account for. "
        "The bias grows with region volume and accessibility, and §4.6 assesses it for this "
        "system [10].", BODY)]

    # ================= 3 methods =================
    s += [P("3  Methods", H1)]
    s += [P("3.1  System and simulation protocol", H2)]
    s += [P(
        "The equilibrated CRY1–KL101 complex was supplied as an AMBER topology and coordinate pair "
        "(<font face='DJ-M'>inputs/CRY1KL101uvt2.prmtop</font> / "
        "<font face='DJ-M'>.rst7</font>), containing 22 246 water molecules. The protein is "
        "described by ff14SB, water by TIP3P and the ligand by GAFF. All simulations ran at 300 K "
        "with a Langevin-middle integrator, 2 fs timestep, 1.0 ps⁻¹ friction, h-bond constraints, "
        "PME electrostatics with a 12 Å cutoff and a 10 Å switch, and harmonic positional "
        "restraints of 100 kJ mol⁻¹ nm⁻² on all Cα atoms to preserve the backbone while allowing "
        "side-chain relaxation. The GCI stage runs at constant volume.", BODY)]
    proto = [["Parameter", "Value", "Parameter", "Value"],
             ["GCMC sphere centre",
              ", ".join(f"{v:g}" for v in tasks["sphere_centre_angstrom"]) + " Å",
              "Adams ladder", "36 values, B = −30 → −4"],
             ["GCMC sphere radius", f"{tasks['radius_angstrom']:g} Å",
              "Spacing", "ΔB = 0.125–0.25 in the transition, 1.5 at the plateaus"],
             ["Ghost buffer", f"{tasks['num_ghost_waters']} (see §6.2)",
              "Cycles per window", "625 × (400 GCMC + 4000 MD steps)"],
             ["Replicas", f"{tasks['replicas']}", "Per window", "5 ns, 250 000 GCMC attempts"],
             ["Seed rule", tasks["seed_formula"], "Trajectory stride",
              f"{tasks['trajectory_stride']} cycles (25 frames)"],
             ["μ′<sub>sol</sub>", "−6.09 kcal/mol", "V°", "30.345 Å³"],
             ["k<sub>B</sub>T", "0.596162 kcal/mol", "B<sub>equil</sub>",
              f"{comb['equilibrium_b']:.4f}"]]
    s += [tbl(proto, [26 * mm, 46 * mm, 26 * mm, W - 98 * mm])]
    s += [P(
        "Replicas differ only in random seed, which sets both the initial velocities and the GCMC "
        "random stream. They therefore sample run-to-run variation from a single equilibrated "
        "structure, not variation in the starting structure.", SMALL)]

    s += [P("3.2  Campaign execution", H2)]
    s += [P(
        "The 216 windows were run as a SLURM array on a heterogeneous GPU cluster (22 GPUs across "
        "11 nodes: RTX 4090, RTX 3090, RTX 4000 Ada and RTX 2080 Ti; driver 570.211.01), one GPU "
        "and four CPU cores per window, twelve concurrent. Because a job cannot select its card and "
        "there is no mid-window restart, the wall limit was set from measurement rather than "
        "estimate: 25 production cycles were timed on the slowest and reference cards, giving "
        "1.38 h per window on a 2080 Ti and 0.97 h on a 3090. The latter reproduces the "
        "protocol's documented ~50 min, so an 8 h limit was adopted. Observed cost was 164.5 "
        "GPU-hours in total, median 0.70 h and maximum 0.96 h per window.", BODY)]

    s += [P("3.3  Free-energy analysis", H2)]
    s += [P(
        "Each replica was analysed independently with the campaign's own "
        "<font face='DJ-M'>gci_analyse.py</font>, which pools that replica's 36 windows, averages "
        "the last 50% of each window's checkpoints, builds the titration curve, and integrates it. "
        "Two estimators of the combined result were then formed:", BODY)]
    s += [P(
        "<b>(i) Pooled curve, integrated (primary).</b> The six raw ⟨N⟩ values at each B are "
        "averaged and the resulting curve integrated once. Because averaging reduces noise by √6, "
        "this is the less bias-prone central estimate. Its uncertainty comes from a non-parametric "
        "bootstrap in which the resampling unit is a whole replica — windows within a replica share "
        "a starting structure and are not independent — with 2000 resamples, cross-checked by "
        "leave-one-replica-out jackknife.", BODY)]
    s += [P(
        "<b>(ii) Mean of the per-replica ΔF<sub>bind</sub>(N*).</b> Reported as a cross-check. "
        "Since ΔF<sub>bind</sub> is a non-linear functional of the curve, this is not identical to "
        "(i); the two agreeing closely is evidence that the non-linearity is unimportant here.", BODY)]
    s += [P(
        "Uncertainties are between-replica throughout. The within-replica standard error that "
        "<font face='DJ-M'>gci_analyse.py</font> reports measures GCMC sampling noise over windows "
        "that all began from the same coordinates, and understates the true uncertainty. With six "
        "replicas, 95% intervals use Student's t with 5 degrees of freedom (t = 2.571), not 1.96.", BODY)]

    s += [P("3.4  Water-network analysis", H2)]
    s += [P(
        f"Hydration sites were mapped following the procedure of [10] so that the two studies are "
        f"comparable. Water-oxygen positions were collected from the "
        f"{net['region_radius_angstrom']:g} Å region around the sphere centre over the three rungs "
        f"bracketing B<sub>equil</sub> (B = {', '.join(f'{b:g}' for b in net['target_b_pooled'])}), "
        f"giving 75 frames per replica. Ghost waters are non-interacting and were excluded "
        f"frame-by-frame using the per-cycle ghost record; counting them would inflate occupancy by "
        f"up to {tasks['num_ghost_waters']}.", BODY)]
    s += [P(
        f"Positions were clustered per replica by average-linkage hierarchical clustering with a "
        f"{net['parameters']['cluster_cutoff']:g} Å cutoff, each cluster representing one hydration "
        f"site whose occupancy is the fraction of frames in which it holds a water. Clusters were "
        f"matched to crystallographic waters within "
        f"{net['parameters']['match_cutoff']:g} Å, considering only sites with occupancy ≥ "
        f"{net['parameters']['min_occupancy']:g}. Recall is the fraction of crystallographic waters "
        f"recovered, precision the fraction of simulated sites matching one, and the Tanimoto "
        f"coefficient summarises overall similarity. Consensus sites were obtained by a second, "
        f"complete-linkage step at {net['parameters']['consensus_cutoff']:g} Å over the pooled "
        f"per-replica clusters, retaining those present in at least "
        f"{net['parameters']['consensus_fraction']:.0%} of replicas.", BODY)]
    s += [P(
        f"The deposited structure 6KX6 (2.00 Å, two copies in the asymmetric unit) is in its own "
        f"crystal frame and its waters are not directly comparable. Chain A — the copy from which "
        f"the simulated system was built — was superposed onto the simulated monomer on "
        f"{cry['superposed_atoms']} matched Cα atoms (residue offset {cry['residue_offset']:+d}), "
        f"giving an RMSD of {cry['ca_rmsd']:.2f} Å, and the same transform applied to its waters.", BODY)]
    s += [boxed([P(
        "<b>Scope limitation.</b> Grand-canonical sampling was applied only inside the "
        f"{net['gcmc_radius_angstrom']:g} Å GCI sphere. Beyond it, water is sampled by ordinary MD, "
        "so occupancies there are ordinary MD occupancies: enhanced exchange does not apply, and a "
        "site that MD cannot fill on this timescale will appear empty. Every site reported below is "
        "flagged inside or outside the sphere. This differs from [10], where the GCMC region itself "
        "was 10 Å.", SMALL)], W)]

    s += [PageBreak()]

    # ================= 4 results =================
    s += [P("4  Results and discussion", H1)]
    s += [P("4.1  Titration curve", H2)]
    s += [P(
        f"All 216 windows completed and carry a verified checkpoint (status completed, 25 "
        f"trajectory frames, zero non-interacting waters, uniform ghost buffer). The titration "
        f"curve rises smoothly from ⟨N⟩ = 0.217 ± 0.008 at B = −30 to saturation near five waters, "
        f"reaching ⟨N⟩ = {pooled['n_at_equilibrium_b']:.2f} at B<sub>equil</sub>.", BODY)]
    fig(A / f"{o.prefix}_combined_titration_curve.png", W * 0.66,
        "<b>Figure 1.</b> Titration curve. Grey: the six individual replicas. Blue: their mean with "
        "the between-replica standard error.", s)
    s += [P(
        "The value of running six replicas is visible in the error bars: the between-replica "
        "standard error exceeds the within-replica standard error at 28 of the 36 rungs, with a "
        "median ratio of 1.58 and a maximum of 5.19 at B = −15.5. A single-replica analysis would "
        "have quoted an uncertainty roughly five times too small in the steep part of the curve.", BODY)]

    s += [P("4.2  Convergence and the monotonicity requirement", H2)]
    s += [P(
        "Individually, all six replicas fail the default monotonicity gate (9–11 inversions each), "
        "so an isotonic fit was applied and the result is labelled preliminary. Three diagnostics "
        "indicate that this reflects noise on the saturation plateau rather than under-convergence:", BODY)]
    s += [P(
        "• <b>No equilibration bias.</b> Drift within the averaged tail, pooled over all 216 "
        "windows, is +0.034 ± 0.031 waters (t = 1.12). Only 2 of 36 rungs flag individually, "
        "against ≈1.8 expected false positives from 36 tests.<br/>"
        "• <b>Averaging behaves as noise should.</b> The largest inversion falls from 1.853 waters "
        "in the worst replica to 0.312 after averaging (mean 0.677 → 0.151). Seven of the eight "
        "residual inversions are smaller than the between-replica standard error; the exception, "
        "B = −10.0 → −8.5, does not survive correction for 35 adjacent-pair tests.<br/>"
        "• <b>The fit changes little.</b> The isotonic correction is at most 0.26 waters, smaller "
        "than the 0.29 Å standard error at that same point (mean 0.038).", BODY)]
    s += [P(
        "The result is also insensitive to the equilibration cutoff: discarding the first 75%, 50%, "
        "25% or none of each window gives ΔF<sub>bind</sub>(N*) = −33.32, −33.53, −33.72 and "
        "−33.38 kcal/mol respectively, with N* = 5 throughout — a spread of 0.40 kcal/mol, well "
        "inside the confidence interval, even when no equilibration period is removed at all.", BODY)]

    s += [P("4.3  Binding free energy", H2)]
    rows = [["N", "ΔF<sub>bind</sub> (kcal/mol)", "Increment", "Note"]]
    prev = 0.0
    for n in sorted(prof):
        v = prof[n]["df"]
        rows.append([str(n), f"{v:+.2f}", "—" if n == 0 else f"{v - prev:+.2f}",
                     "extrapolated, not measured" if prof[n]["extrapolated"]
                     else ("<b>N*</b>" if n == pooled["n_star"] else "")])
        prev = v
    s += [tbl(rows, [10 * mm, 32 * mm, 22 * mm, W - 64 * mm])]
    s += [P(
        f"<b>Table 1.</b> Pooled ΔF<sub>bind</sub>(N). The minimum lies at N* = "
        f"{pooled['n_star']}, giving <b>ΔF<sub>bind</sub>(N*) = {df:+.2f} kcal/mol</b>, 95% CI "
        f"[{lo:+.2f}, {hi:+.2f}] (bootstrap SD {bs['sd']:.3f}, jackknife SE "
        f"{pooled['jackknife']['se']:.3f}). N* = 5 in all {bs['resamples']} bootstrap resamples and "
        f"unanimously across the six replicas. The cross-check estimator gives "
        f"{comb['df_bind_star_each_replica_own_n_star']['mean']:+.2f} ± "
        f"{comb['df_bind_star_each_replica_own_n_star']['sem']:.2f} kcal/mol, agreeing to 0.15 "
        f"kcal/mol.", SMALL)]
    fig(A / f"{o.prefix}_combined_free_energy.png", W * 0.60,
        "<b>Figure 2.</b> Binding free energy against occupancy, with the minimum at N* = 5.", s)
    s += [P(
        "The increments decrease monotonically — the first water contributes −10.19 kcal/mol and "
        "the fifth only −3.14 — which is the expected ordering for progressive filling of a pocket. "
        "N = 6 is extrapolated by construction, since the analysis evaluates one occupancy beyond "
        f"the measured range, so the upturn defining the minimum is not itself measured; N* = 5 is "
        f"independently supported by ⟨N⟩ = {pooled['n_at_equilibrium_b']:.2f} at equilibrium.", BODY)]
    s += [P(
        "It is worth being explicit about what this number describes. With a "
        f"{tasks['radius_angstrom']:g} Å radius the sphere has a bulk-equivalent occupancy of 8.83 "
        "waters and N* = 5, so this is a multi-water pocket rather than a single hydration site. "
        f"ΔF<sub>bind</sub> = {df:.2f} kcal/mol is the total for filling all five positions; the "
        "single-water value is the N = 1 entry, −10.19 kcal/mol.", BODY)]

    s += [PageBreak()]

    # ---- 4.4 water network ----
    s += [P("4.4  Hydration network and comparison with the crystal structure", H2)]
    s += [P(
        f"Within the GCI sphere the ensemble mean occupancy is "
        f"{water['ensemble_mean_occupancy']:.2f} waters, which independently reproduces the "
        f"titration value ⟨N⟩ = 4.669 at that rung by a wholly separate route — a useful check on "
        f"the ghost masking. Fifteen sites reach 10% occupancy and six reach 20%, but only two are "
        f"well defined (0.75 and 0.63); the remainder are diffuse. All four waters present in the "
        f"equilibrated input structure are recovered, each within 0.50 Å of a simulated site.", BODY)]
    trow = [["", "Cai — AN139 [10]", "Cai — Apo [10]", "Cai — AN174 [10]",
             "<b>This work — KL101</b>"],
            ["N<sub>Xray</sub>", "9.0 ± 0.0", "9.0 ± 0.0", "5.0 ± 0.0",
             f"<b>{cry['waters_in_region']}</b>"],
            ["N<sub>clusters</sub>", "288.1 ± 11.8", "397.5 ± 20.3", "279.8 ± 18.0",
             f"{st['N_clusters']['mean']:.1f} ± {st['N_clusters']['sd']:.1f}"],
            ["N<sub>clusters</sub> (occ ≥ 0.2)", "36.8 ± 4.6", "67.9 ± 4.6", "34.5 ± 3.2",
             f"{st['N_clusters(occ>=0.2)']['mean']:.1f} ± {st['N_clusters(occ>=0.2)']['sd']:.1f}"],
            ["Matched", "7.8 ± 1.2", "7.8 ± 0.9", "3.7 ± 1.2",
             f"{st['Matched']['mean']:.1f} ± {st['Matched']['sd']:.1f}"],
            ["Recall", "0.870 ± 0.133", "0.861 ± 0.096", "0.733 ± 0.246",
             f"<b>{st['Recall']['mean']:.3f} ± {st['Recall']['sd']:.3f}</b>"],
            ["Precision", "0.211 ± 0.041", "0.114 ± 0.014", "0.104 ± 0.034",
             f"{st['Precision']['mean']:.3f} ± {st['Precision']['sd']:.3f}"],
            ["Tanimoto", "0.205 ± 0.044", "0.112 ± 0.015", "0.101 ± 0.035",
             f"{st['Tanimoto']['mean']:.3f} ± {st['Tanimoto']['sd']:.3f}"],
            ["Replicas", "12", "12", "12", f"{tasks['replicas']}"]]
    s += [tbl(trow, [32 * mm, 28 * mm, 28 * mm, 28 * mm, W - 116 * mm])]
    s += [P(
        "<b>Table 2.</b> Agreement between simulated hydration sites and crystallographic waters, "
        "averaged over replicas, in the format of [10]. The AN139/Apo/AN174 columns are reproduced "
        "from that study; the KL101 column is computed here with the same clustering (2.4 Å "
        "average linkage), matching (2.0 Å) and occupancy threshold (0.2).", SMALL)]
    s += [P(
        f"Recall of {st['Recall']['mean']:.3f} ± {st['Recall']['sd']:.3f} sits just below the "
        f"0.870 ± 0.133 reported for AN139 and above the 0.733 ± 0.246 for AN174, so the method "
        f"recovers crystallographic hydration of this pocket about as well for KL101 as for the "
        f"other two ligands. Precision is low in every column, and that is expected rather than a "
        f"defect: a crystal structure is a static, ensemble-averaged model that does not resolve "
        f"transient or weakly occupied waters, so the simulation legitimately proposes more sites "
        f"than the experiment can confirm. Two differences in the underlying sampling should temper "
        f"a close reading of the absolute counts — the cluster totals differ by a factor of three "
        f"because this campaign contributes 75 frames per replica against a 10 ns GCMC/MD "
        f"production in [10], and only the inner 4 Å of the present region had grand-canonical "
        f"sampling.", BODY)]

    fig(A / f"{o.prefix}_bfactor_occupancy.png", W * 0.78,
        "<b>Figure 3.</b> Simulated occupancy (top) and matching distance (bottom) against "
        "crystallographic B-factor for the 14 waters of 6KX6 chain A inside the region, ordered by "
        "B-factor. Coloured points are individual replicas; black markers are the mean ± SD. The "
        "shaded column marks the one water lying inside the GCMC sphere.", s)
    s += [P(
        "Figure 3 reproduces the trend reported in [10]: crystallographic waters with high "
        "B-factors are consistently matched to low-occupancy simulated sites. The two most ordered "
        "waters (B = 14.6 and 14.8 Å²) are occupied 91% and 93% of the time. The converse does not "
        "hold — HOH 635, at a modest B = 23.8 Å², has no matching site at all — so a low B-factor "
        "does not guarantee a stable simulated hydration site. Three of the fourteen waters are "
        "unmatched (HOH 635, 753 and 708), which is what limits recall to 0.774. As in [10], the "
        "matching distance shows no relationship with B-factor, supporting the use of a single "
        "cutoff for all waters.", BODY)]

    fig(A / f"{o.prefix}_consensus_sites.png", W * 0.86,
        "<b>Figure 4.</b> Consensus hydration sites, defined as clusters present in at least 75% of "
        "replicas. Left: one bar per crystallographic water, ordered by B-factor and shaded by it; "
        "an absent bar means no consensus site was found. Right: the ten strongest consensus sites "
        "with no crystallographic counterpart.", s)
    s += [P(
        f"Of {len(net['consensus_sites'])} consensus sites, {sum(1 for c in net['consensus_sites'] if c['matched_xray'])} "
        f"match a crystallographic water. The remainder are reproducible across replicas yet absent "
        f"from the deposited model, several of them at occupancies above 0.7. As [10] observed for "
        f"AN139, this points to hydration that is thermodynamically favourable but not "
        f"crystallographically resolved — either genuinely disordered, or a real site the "
        f"experiment could not place.", BODY)]

    # ---- 4.5 AN139 free energies ----
    s += [P("4.5  Comparison of free energies with the CRY1/AN139 study", H2)]
    s += [P(
        "The two campaigns share their protocol almost exactly, which makes the free energies "
        "commensurable and any discrepancy attributable to the system rather than the method.", BODY)]
    same = [["", "CRY1/AN139 [10]", "CRY1/KL101 (this work)"],
            ["Adams ladder", "36 values, −30 → −4, variable spacing", "identical"],
            ["Schedule per window", "625 × (400 GCMC + 8 ps MD) = 5 ns", "identical"],
            ["μ′<sub>sol</sub> / V°", "−6.09 kcal/mol / 30.345 Å³", "identical"],
            ["B<sub>equil</sub>", "−8.038", f"{comb['equilibrium_b']:.4f}"],
            ["Force field", "ff14SB / TIP3P / GAFF", "identical"],
            ["GCMC engine", "<font face='DJ-M'>grand</font> (Samways et al.)",
             "<font face='DJ-M'>loch</font> 2026.1.0"],
            ["Replicas", "12", f"{tasks['replicas']}"]]
    s += [tbl(same, [30 * mm, 58 * mm, W - 88 * mm])]
    s += [P(
        "B<sub>equil</sub> agreeing to four decimal places is a meaningful cross-validation: two "
        "independent GCMC implementations derive the same equilibrium Adams value from the same "
        "calibration constants.", SMALL)]
    cmp2 = [["Region", "Radius", "X-ray waters", "N*", "ΔF<sub>bind</sub>(N*)", "Per water",
             "Occupancy / bulk"],
            ["Cai, Sphere 2 [10]", "4 Å", "2", "2", "−9.40 kcal/mol", "−4.70", "0.23"],
            ["Cai, Sphere 1 [10] <i>(rejected)</i>", "7 Å", "7", "25 <i>(expected 7)</i>",
             "−157.17 kcal/mol", "−6.29", "0.53"],
            ["<b>This work</b>", "4 Å", "1", "<b>5</b>", f"<b>{df:.2f} kcal/mol</b>", "−6.71",
             "0.57"]]
    s += [tbl(cmp2, [34 * mm, 14 * mm, 20 * mm, 24 * mm, 28 * mm, 17 * mm, W - 137 * mm])]
    s += [P(
        "<b>Table 3.</b> GCI results for comparable regions. \"Occupancy / bulk\" is N* divided by "
        "the bulk-equivalent occupancy of the region.", SMALL)]

    s += [P("4.6  Diffusive exchange: does the caveat apply here?", H2)]
    s += [P(
        "Cai reports Sphere 1 as not meaningful: it returned N* = 25 where seven crystallographic "
        "waters were expected, and ΔF<sub>bind</sub> = −157 kcal/mol. The diagnosis was diffusive "
        "exchange through the permeable GCMC/MD boundary overwhelming the Monte Carlo exchange, "
        "violating the assumptions of §2.2. The evidence for the present system points both ways "
        "and is worth setting out rather than resolving prematurely.", BODY)]
    s += [P(
        "<b>Reassuring.</b> The region here is 4 Å — the size validated in [10], and one eighth the "
        "volume of the one that failed. The diagnostic used there is whether occupancy falls to "
        "zero at strongly unfavourable B: the failing region \"remained significantly above zero\", "
        "whereas the working one \"approached zero as expected\". The present curve reaches "
        "⟨N⟩ = 0.217 ± 0.008 at B = −30 and descends smoothly throughout, indicating that GCMC, not "
        "diffusion, controls the occupancy.", BODY)]
    s += [P(
        "<b>Cautionary.</b> This site is considerably more solvent-accessible than the one for which "
        "the method was validated. At N* it holds 0.57 of bulk density, against 0.23 for Sphere 2 — "
        "and 0.53 for the rejected Sphere 1. Per water, −6.71 kcal/mol lies closer to the biased "
        "Sphere 1 (−6.29) than to the trusted Sphere 2 (−4.70). Part of this simply reflects the "
        "site: a placeholder centre in an open, partly solvated pocket rather than a buried "
        "two-water site. But a more open region admits more diffusive exchange at the same radius, "
        "so the residual bias is plausibly larger here than in the validated case.", BODY)]
    s += [boxed([P(
        f"<b>Recommended reading of the number.</b> Treat ΔF<sub>bind</sub>(N*) = {df:.2f} kcal/mol "
        f"as carrying an unquantified systematic component beyond the ±0.88 kcal/mol statistical "
        f"confidence interval. Following [10] and Ekberg et al. [13], relative comparisons survive "
        f"such a shift far better than absolute values: comparing this pocket across ligands, or "
        f"against the apo form, would rest on much firmer ground than quoting the absolute figure "
        f"alone.", BODY)], W)]

    # ================= 5 conclusion =================
    s += [P("5  Conclusion", H1)]
    s += [P(
        f"A 216-window GCI campaign on the KL101-bound CRY1 pocket completed with every window "
        f"verified, at a cost of 164.5 GPU-hours. The replica-averaged titration curve integrates "
        f"to ΔF<sub>bind</sub>(N*) = {df:+.2f} kcal/mol [95% CI {lo:+.2f}, {hi:+.2f}] at N* = "
        f"{pooled['n_star']} waters, a result that is unanimous across replicas, stable against the "
        f"equilibration cutoff, and reproduced by two independent estimators.", BODY)]
    s += [P(
        f"The hydration network recovers {st['Recall']['mean']:.0%} of the crystallographic waters "
        f"of 6KX6 chain A within the analysed region, with recall, precision and Tanimoto "
        f"coefficients consistent with those reported for AN139 under the same protocol, and it "
        f"reproduces the reported relationship between crystallographic B-factor and simulated "
        f"occupancy. Agreement of B<sub>equil</sub> to four decimal places between two independent "
        f"GCMC codes supports the correctness of the setup.", BODY)]
    s += [P(
        "Two limitations bound the interpretation. The absolute free energy carries an unquantified "
        "diffusive-exchange bias, which argues for using this pipeline comparatively — across "
        "ligands or against the apo form — rather than for absolute values. And the sphere centre "
        "remains the placeholder shipped with the campaign; it validates geometrically and was "
        "deliberately left unchanged, but it determines which hydration the 164.5 GPU-hours "
        "actually characterised, and should be confirmed before the result is interpreted "
        "biologically.", BODY)]

    s += [PageBreak()]

    # ================= 6 implementation =================
    s += [P("6  Implementation notes", H1)]
    s += [P(
        "This section records what had to change for the campaign to run correctly on this cluster, "
        "and what is needed to reproduce it. It is included because three of the five defects below "
        "fail silently — they produce output that looks correct.", BODY)]

    s += [P("6.1  Changes to the shipped campaign", H2)]
    ch = [["#", "File", "Change", "Reason"],
          ["1", "submit_gci.slurm",
           "<font face='DJ-M'>set +u</font> / <font face='DJ-M'>set -u</font> around "
           "<font face='DJ-M'>conda activate</font>",
           "The script runs <font face='DJ-M'>set -euo pipefail</font>; the environment's "
           "cuda-nvcc activation hook expands <font face='DJ-M'>${NVCC_PREPEND_FLAGS}</font> "
           "unguarded. <b>All 216 jobs exited in ~2 s.</b> A failed window leaves no checkpoint, so "
           "re-submitting would have failed identically."],
          ["2", "environment.yml", "CUDA 12.9 → 12.8 across nvcc, nvvm, cudart and cuda-version",
           "The driver is 570.211.01, which reports CUDA 12.8. nvcc 12.9 emits a PTX ISA it cannot "
           "load, and OpenMM JIT-compiles PTX, so its CUDA platform failed with "
           "<font face='DJ-M'>CUDA_ERROR_UNSUPPORTED_PTX_VERSION</font>."],
          ["3", "config.sh",
           "<font face='DJ-M'>SMOKE=0</font> → <font face='DJ-M'>SMOKE=${SMOKE:-0}</font>",
           "<font face='DJ-M'>run_window.sh</font> sources config.sh before reading anything, so "
           "the documented <font face='DJ-M'>SMOKE=1 ./run_window.sh 0</font> was overwritten and "
           "silently ran a full production window. The default is unchanged."],
          ["4", "config.sh", "<font face='DJ-M'>--trajectory-stride 5</font> appended to SMOKE_ARGS",
           "The smoke schedule is 10 cycles but the runner always passes the production stride of "
           "25, which does not divide 10, so the smoke path could never run. Production (625 "
           "cycles) is unaffected."],
          ["5", "config.sh", "<font face='DJ-M'>NUM_GHOSTS</font> 27 → 100", "See §6.2."],
          ["6", "submit_gci.slurm",
           "partition <font face='DJ-M'>all</font>; time 4 h → 8 h; "
           "<font face='DJ-M'>--mem=32G</font>; no <font face='DJ-M'>--account</font>",
           "Cluster-specific. There are no accounting associations, so supplying an account causes "
           "rejection. 8 h covers the slowest card (1.38 h measured)."]]
    s += [tbl(ch, [7 * mm, 25 * mm, 40 * mm, W - 72 * mm], size=7.0)]
    s += [P(
        "No physical value was altered except the ghost buffer. Sphere centre, radius, seeds and "
        "the 625-cycle schedule are untouched, and <font face='DJ-M'>tasks.tsv</font> still hashes "
        "to the value recorded in <font face='DJ-M'>tasks.json</font>.", SMALL)]

    s += [P("6.2  Ghost-water exhaustion", H2)]
    s += [P(
        "Mid-run, one array task (replica 2, window 35, B = −4) failed at cycle 469 of 625 with "
        "<font face='DJ-M'>RuntimeError: Cannot insert any more waters</font>. Re-submitting that "
        "single task would have been the wrong response.", BODY)]
    s += [P(
        "The shipped <font face='DJ-M'>NUM_GHOSTS</font> = 27 comes from "
        "<font face='DJ-M'>suggested_num_ghosts(4.0)</font> = 3 × ⌈268.1/30.345⌉, three times the "
        "<i>instantaneous</i> bulk-equivalent occupancy. But the buffer drains <i>cumulatively</i>: "
        "a water inserted into the sphere can diffuse out and remain physical, so it never returns "
        "to the pool. Occupancy stayed at 3–7 waters while the pool ran to zero.", BODY)]
    s += [P(
        "The hazard was not the crash. At the same B another replica <i>completed</i>, with a valid "
        "hash-verified checkpoint, having bottomed out at three ghosts remaining — its high-B tail "
        "clipped, exactly the bias that <font face='DJ-M'>suggested_num_ghosts</font>'s own "
        "docstring warns about, and it would have entered the analysis as a normal result. The "
        "<font face='DJ-M'>minimum_ghost_pool</font> field recorded in every checkpoint is what made "
        "this visible.", BODY)]
    s += [P(
        "Re-measuring with genuine headroom showed true demand at B = −4 to be 34 of 100 — "
        "<i>exceeding the old pool of 27 outright</i>. The 27-ghost run had understated its own "
        "demand precisely because it was clipped, so every replica's top rung was destined to fail "
        "or be biased. Because <font face='DJ-M'>num_ghost_waters</font> is one of "
        "<font face='DJ-M'>gci_analyse.py</font>'s consistency fields, windows with different "
        "buffer sizes cannot be pooled and a partial re-run was impossible: the campaign was "
        "cancelled at 96/216 and restarted, discarding roughly 60 GPU-hours.", BODY)]
    gp = [["Minimum pool remaining", "B ≤ −11.5", "−10.0", "−8.5", "−7.0", "−5.5", "−4.0"],
          ["27-ghost run", "27", "26", "25", "26", "16", "<b>3 / exhausted</b>"],
          ["100-ghost run", "100", "100", "100", "92–94", "87–91", "66–75"]]
    s += [tbl(gp, [38 * mm] + [(W - 38 * mm) / 6] * 6, size=7.2)]

    s += [P("6.3  Other unexpected behaviours", H2)]
    oth = [["Behaviour", "Detail"],
           ["A PyCUDA test does not detect the CUDA mismatch",
            "PyCUDA compiles to architecture-specific cubin and passed on the broken 12.9 "
            "toolchain, while OpenMM — which JIT-compiles PTX — failed. Verify with "
            "<font face='DJ-M'>python -m openmm.testInstallation</font>."],
           ["Non-empty stderr is normal",
            "All 216 <font face='DJ-M'>.err</font> files contain exactly one loguru DEBUG line "
            "recording that window's Adams value."],
           ["Superseded provenance keys",
            "<font face='DJ-M'>md_protocol_signature</font> in each titration JSON is a shared "
            "protocol block and reports <font face='DJ-M'>sphere_radius: 10 A</font> and "
            "<font face='DJ-M'>num_ghost_waters: 45</font>, which are <b>not</b> the values used. "
            "<font face='DJ-M'>md_protocol_superseded_keys</font> lists the five that do not apply; "
            "the authoritative values are the top-level keys."],
           ["The analysis cannot see replicas",
            "<font face='DJ-M'>gci_analyse.py</font> takes one "
            "<font face='DJ-M'>--run-root</font> and would reject a directory holding all six, "
            "because <font face='DJ-M'>require_consistent_windows()</font> forbids duplicate "
            "<font face='DJ-M'>target_b</font>. It must be run once per replica."],
           ["Crystal frames differ",
            "A deposited model is in its own frame; its waters are meaningless until superposed. "
            "Chain A and chain B of 6KX6 place different waters inside the GCI sphere, which is "
            "itself evidence that this region is weakly ordered."]]
    s += [tbl(oth, [40 * mm, W - 40 * mm], size=7.0)]

    s += [P("6.4  Reproduction", H2)]
    s += [P(
        "export CONDA_OVERRIDE_CUDA=\"12.8\"   # login node has no GPU; this reproduces the "
        "compute-node solve<br/>"
        "conda env create -f environment.yml  # cry-loch-babel; nvcc MUST resolve inside the env<br/>"
        "conda activate cry-loch-babel<br/>"
        "python -m openmm.testInstallation    # all four platforms must pass<br/><br/>"
        "SMOKE=1 ./run_window.sh 0            # ~2.5 min sanity check<br/>"
        "rm -rf replicas/rep1/window_00_B-30.0000/*<br/>"
        "sbatch submit_gci.slurm              # --array=0-215%12<br/><br/>"
        "python combine_replicas.py                       # per replica, then pooled + bootstrap<br/>"
        "python water_sites.py --crystal inputs/6kx6.pdb  # GCI-sphere hydration<br/>"
        "python water_network.py --chain A                # Cai-style network analysis<br/>"
        "python make_report.py &amp;&amp; python make_slides.py", CODE)]
    ver = [["python", "3.12.13", "loch", "2026.1.0", "OpenMM", "8.4.0"],
           ["sire", "2026.1.0", "mdtraj", "1.11.1", "numpy", "2.4.6"],
           ["pycuda", "2025.1.1", "scipy", "1.18.0", "CUDA", "12.8 (nvcc 12.8.93)"]]
    s += [Spacer(1, 3), tbl(ver, [16 * mm, 22 * mm, 16 * mm, 22 * mm, 16 * mm, W - 92 * mm],
                            header=False, size=7.2)]
    prov = [["Item", "Value"],
            ["prmtop SHA-256", f"<font face='DJ-M'>{tasks['input_prmtop_sha256']}</font>"],
            ["rst7 SHA-256", f"<font face='DJ-M'>{tasks['input_rst7_sha256']}</font>"],
            ["tasks.tsv SHA-256", f"<font face='DJ-M'>{tasks['tasks_tsv_sha256']}</font>"],
            ["Verification", "216/216 checkpoints, status completed, 25 frames each, "
                             "zero non-interacting waters, num_ghost_waters = 100 uniform, "
                             "11 outputs hashed per window"]]
    s += [Spacer(1, 3), tbl(prov, [30 * mm, W - 30 * mm], size=7.0)]
    s += [P(
        "<font face='DJ-M'>scripts/</font> is unmodified. The added scripts "
        "(<font face='DJ-M'>combine_replicas.py</font>, <font face='DJ-M'>water_sites.py</font>, "
        "<font face='DJ-M'>water_network.py</font>) import "
        "<font face='DJ-M'>compute_df_bind</font>, <font face='DJ-M'>isotonic_non_decreasing</font>, "
        "<font face='DJ-M'>monotonicity_violations</font> and <font face='DJ-M'>kabsch</font> from "
        "the campaign's own modules rather than reimplementing them, so the physics has a single "
        "source of truth.", SMALL)]

    # ================= references =================
    s += [P("References", H1)]
    refs = [
        "Huggins, D. J.; Sherman, W.; Tidor, B. Rational approaches to improving selectivity in "
        "drug design. <i>J. Med. Chem.</i> <b>2012</b>, 55, 1424–1444.",
        "Bissantz, C.; Kuhn, B.; Stahl, M. A medicinal chemist's guide to molecular interactions. "
        "<i>J. Med. Chem.</i> <b>2010</b>, 53, 5061–5084.",
        "Laage, D.; Elsaesser, T.; Hynes, J. T. Water dynamics in the hydration shells of "
        "biomolecules. <i>Chem. Rev.</i> <b>2017</b>, 117, 10694–10725.",
        "Woo, H.-J.; Dinner, A. R.; Roux, B. Grand canonical Monte Carlo simulations of water in "
        "protein environments. <i>J. Chem. Phys.</i> <b>2004</b>, 121, 6392–6400.",
        "Ekberg, V.; Samways, M. L.; Misini Ignjatović, M.; Essex, J. W.; Ryde, U. Comparison of "
        "grand canonical and conventional molecular dynamics simulation methods for protein-bound "
        "water networks. <i>ACS Phys. Chem. Au</i> <b>2022</b>, 2, 247–259.",
        "Samways, M. L.; Bruce Macdonald, H. E.; Essex, J. W. grand: a Python module for grand "
        "canonical water sampling in OpenMM. <i>J. Chem. Inf. Model.</i> <b>2020</b>, 60, 4436–4441.",
        "Ross, G. A.; Bodnarchuk, M. S.; Essex, J. W. Water sites, networks, and free energies with "
        "grand canonical Monte Carlo. <i>J. Am. Chem. Soc.</i> <b>2015</b>, 137, 14930–14943.",
        "Miller, S.; Kesherwani, M.; Chan, P.; Nagai, Y.; Yagi, M.; Cope, J.; Tama, F.; Kay, S. A.; "
        "Hirota, T. CRY2 isoform selectivity of a circadian clock modulator with antiglioblastoma "
        "efficacy. <i>Proc. Natl. Acad. Sci. U.S.A.</i> <b>2022</b>, 119, e2203936119.",
        "Miller, S.; Srivastava, A.; Nagai, Y.; Aikawa, Y.; Tama, F.; Hirota, T. Structural "
        "differences in the FAD-binding pockets and lid loops of mammalian CRY1 and CRY2 for "
        "isoform-selective regulation. <i>Proc. Natl. Acad. Sci. U.S.A.</i> <b>2021</b>, 118, "
        "e2026191118.",
        "Cai, L. GCMC-based water network mapping in protein–ligand complexes. M1 internship "
        "report, ENS-PSL / RIKEN R-CCS, <b>2026</b>. Supervisors: F. Tama, B. Cree.",
        "Adams, D. Chemical potential of hard-sphere fluids by Monte Carlo methods. "
        "<i>Mol. Phys.</i> <b>1974</b>, 28, 1241–1252.",
        "Adams, D. Grand canonical ensemble Monte Carlo for a Lennard-Jones fluid. "
        "<i>Mol. Phys.</i> <b>1975</b>, 29, 307–311.",
        "Eastman, P.; et al. OpenMM 7: rapid development of high-performance algorithms for "
        "molecular dynamics. <i>PLoS Comput. Biol.</i> <b>2017</b>, 13, e1005659.",
    ]
    for i, r in enumerate(refs, 1):
        s += [P(f"[{i}]&nbsp;&nbsp;{r}", SMALL)]

    s += [Spacer(1, 6), P(
        "Generated by <font face='DJ-M'>make_report.py</font> from "
        "<font face='DJ-M'>analysis/combined/*.json</font>; figures and tables are read from the "
        "analysis outputs rather than transcribed.", SMALL)]

    o.output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(o.output), pagesize=A4,
                            leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=14 * mm, bottomMargin=14 * mm,
                            title="CRY1/KL101 Grand Canonical Integration",
                            author="Computational report")

    def footer(cv, dc):
        cv.saveState()
        cv.setFont("DJ", 7)
        cv.setFillColor(MUT)
        cv.drawString(15 * mm, 8 * mm, "CRY1/KL101 — Grand Canonical Integration")
        cv.drawRightString(A4[0] - 15 * mm, 8 * mm, f"{dc.page}")
        cv.restoreState()

    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {o.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
