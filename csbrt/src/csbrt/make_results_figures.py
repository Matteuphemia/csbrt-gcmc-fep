#!/usr/bin/env python3
"""Generate presentation-ready figures for the refined AN139 -> KL101 FEP pilot."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile

import BioSimSpace as BSS
import MDAnalysis as mda
from MDAnalysis.analysis import align, rms
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

COLORS = {
    "navy": "#17324D",
    "blue": "#2E6F9E",
    "cyan": "#5AA9C8",
    "orange": "#D97941",
    "gold": "#E5AE38",
    "green": "#428A73",
    "red": "#B84A4A",
    "gray": "#6E7781",
    "light": "#E8EEF2",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.png", dpi=300)
    fig.savefig(FIGURES / f"{stem}.pdf")
    plt.close(fig)


def parse_energy(value: str) -> float:
    return float(value.split()[0])


def load_inputs():
    combined = json.loads((ROOT / "combined_summary.json").read_text())
    analyses = {}
    for rep in (1, 2):
        edge = ROOT / f"rep{rep}" / f"AN139_to_KL101_rep{rep}"
        analyses[rep] = json.loads((edge / "analysis.json").read_text())
    return combined, analyses


def time_slice(edge_root: Path, end_fraction: float):
    results = {}
    with tempfile.TemporaryDirectory(prefix="fep-plot-slice-") as tmp:
        tmp = Path(tmp)
        for leg in ("bound", "free"):
            out = tmp / leg
            out.mkdir()
            for source in sorted((edge_root / leg).glob("energy_traj_*.parquet")):
                table = pq.read_table(source)
                end = round(len(table) * end_fraction)
                pq.write_table(table.slice(0, end), out / source.name)
            pmf, _ = BSS.FreeEnergy.Relative.analyse(str(out))
            results[leg] = pmf
        value, error = BSS.FreeEnergy.Relative.difference(results["bound"], results["free"])
        return parse_energy(str(value)), parse_energy(str(error))


def convergence_data():
    cache = ROOT / "convergence_summary.json"
    if cache.exists():
        return json.loads(cache.read_text())
    payload = {}
    for rep in (1, 2):
        edge = ROOT / f"rep{rep}" / f"AN139_to_KL101_rep{rep}"
        rows = []
        for ns in (1, 2, 3, 4, 5):
            value, error = time_slice(edge, ns / 5)
            rows.append({"ns_per_window": ns, "ddg": value, "formal_error": error})
        payload[f"replica_{rep}"] = rows
    cache.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def common_core_indices(edge: Path) -> list[int]:
    details = json.loads((edge / "setup/fep_preparation.complete.json").read_text())
    prep = Path("/home/aldo/cry1_fep_handoff_20260805/data/preparations/KL101")
    ligand_b = BSS.IO.readMolecules(
        [str(prep / "ligand.prmtop"), str(prep / "ligand.rst7")]
    ).getMolecule(0)
    atoms_b = ligand_b.getAtoms()
    return [
        row["state_a_atom"]
        for row in details["mapping"]
        if not atoms_b[row["state_b_atom"]].element().lower().startswith("hydrogen")
    ]


def endpoint_rmsd_data():
    cache = ROOT / "endpoint_rmsd_summary.json"
    if cache.exists():
        return json.loads(cache.read_text())
    payload = {}
    for rep in (1, 2):
        edge = ROOT / f"rep{rep}" / f"AN139_to_KL101_rep{rep}"
        core_local = common_core_indices(edge)
        payload[f"replica_{rep}"] = {}
        for label, top, traj in (
            ("lambda_0_AN139", "system0.prm7", "traj_0.00000.dcd"),
            ("lambda_1_KL101", "system1.prm7", "traj_1.00000.dcd"),
        ):
            top = edge / "bound" / top
            traj = edge / "bound" / traj
            mobile = mda.Universe(str(top), str(traj), topology_format="PARM7")
            reference = mda.Universe(str(top), str(traj), topology_format="PARM7")
            reference.trajectory[0]
            lig = mobile.select_atoms("resname LIG")
            ref_lig = reference.select_atoms("resname LIG")
            core = mobile.atoms[lig.indices[core_local]]
            ref_core = reference.atoms[ref_lig.indices[core_local]]
            values = []
            for _ in mobile.trajectory:
                align.alignto(mobile, reference, select="protein and name CA", weights="mass")
                values.append(
                    float(
                        rms.rmsd(
                            core.positions,
                            ref_core.positions,
                            center=False,
                            superposition=False,
                        )
                    )
                )
            payload[f"replica_{rep}"][label] = values
    cache.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def figure_headline(combined):
    values = np.asarray(combined["cross_combinations_kcal_mol"])
    labels = [f"B{b}–F{f}" for b in (1, 2) for f in (1, 2, 3, 4)]
    colors = [COLORS["blue"]] * 4 + [COLORS["orange"]] * 4
    mean = combined["combined_ddg_kcal_mol"]
    sd = combined["cross_combination_sd_kcal_mol"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    y = np.arange(len(values))
    ax.axvline(0, color=COLORS["gray"], lw=1.2)
    ax.axvspan(mean - sd, mean + sd, color=COLORS["gold"], alpha=0.22, label="Empirical ±1 SD")
    ax.axvline(mean, color=COLORS["navy"], lw=2.2, label=f"Mean = {mean:.2f} kcal/mol")
    ax.scatter(values, y, c=colors, s=58, edgecolor="white", linewidth=0.8, zorder=3)
    for value, yy in zip(values, y):
        ax.text(value + 0.035, yy, f"{value:.2f}", va="center", fontsize=9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(-0.15, max(values) + 0.38)
    ax.set_xlabel("ΔΔG binding, AN139 → KL101 (kcal/mol)")
    ax.set_title(f"All replicate pairings predict weaker binding for KL101\nMean = {mean:.2f} ± {sd:.2f} kcal/mol (≈22× higher Kd at 300 K)")
    ax.legend(loc="upper left")
    save(fig, "01_final_ddg_replicate_pairings")


def figure_components(combined):
    bound = np.asarray([x[0] for x in combined["bound_replicates_kcal_mol"]])
    free = np.asarray([x[0] for x in combined["free_replicates_kcal_mol"]])
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    rng = np.random.default_rng(7)
    for xpos, values, color, name in (
        (0, bound, COLORS["blue"], "Bound"),
        (1, free, COLORS["orange"], "Free"),
    ):
        jitter = rng.uniform(-0.08, 0.08, len(values))
        ax.scatter(np.full(len(values), xpos) + jitter, values, s=75, color=color, edgecolor="white", zorder=3)
        mean, sd = values.mean(), values.std(ddof=1)
        ax.errorbar(xpos, mean, yerr=sd, fmt="D", ms=7, color=COLORS["navy"], capsize=7, lw=2)
        for idx, (xx, value) in enumerate(zip(np.full(len(values), xpos) + jitter, values), 1):
            ax.annotate(f"R{idx}: {value:.3f}", xy=(xx, value), xytext=(6, 10 if idx % 2 else -12), textcoords="offset points", va="center", fontsize=8.5)
    ax.set_xticks([0, 1], ["Bound (n=2)", "Free (n=4)"])
    ax.set_ylabel("Endpoint alchemical free energy (kcal/mol)")
    ax.set_xlim(-0.4, 1.45)
    ax.set_title("Free-leg variability dominates the uncertainty")
    ax.text(0.02, 0.04, f"Bound SD = {bound.std(ddof=1):.3f}\nFree SD = {free.std(ddof=1):.3f} kcal/mol", transform=ax.transAxes)
    save(fig, "02_bound_free_replicate_decomposition")


def figure_overlap(analyses):
    cmap = LinearSegmentedColormap.from_list("overlap", ["#F7FAFC", "#87BDD4", "#17324D"])
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
    lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    image = None
    for row, rep in enumerate((1, 2)):
        for col, leg in enumerate(("bound", "free")):
            ax = axes[row, col]
            matrix = np.asarray(analyses[rep][f"{leg}_overlap_matrix"])
            image = ax.imshow(matrix, vmin=0, vmax=0.45, cmap=cmap, origin="lower")
            minimum = analyses[rep][f"{leg}_adjacent_overlap_minimum"]
            ax.set_title(f"Replica {rep} · {leg}\nmin adjacent = {minimum:.3f}", fontsize=11)
            ax.set_xticks(range(12), [f"{x:g}" for x in lambdas], rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(12), [f"{x:g}" for x in lambdas], fontsize=8)
            for i in range(11):
                ax.plot([i, i + 1], [i, i + 1], color=COLORS["gold"], lw=1.8, alpha=0.9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Sampled λ")
    for ax in axes[-1, :]:
        ax.set_xlabel("Evaluated λ")
    fig.suptitle("Refined λ-overlap matrices: λ=0.95 repairs the endpoint bottleneck", fontsize=14, fontweight="bold")
    fig.colorbar(image, ax=axes, shrink=0.82, label="MBAR overlap probability")
    save(fig, "03_lambda_overlap_heatmaps")


def figure_convergence(convergence, combined):
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for rep, color in ((1, COLORS["blue"]), (2, COLORS["orange"])):
        rows = convergence[f"replica_{rep}"]
        x = np.asarray([r["ns_per_window"] for r in rows])
        y = np.asarray([r["ddg"] for r in rows])
        e = np.asarray([r["formal_error"] for r in rows])
        ax.errorbar(x, y, yerr=e, marker="o", color=color, lw=2, capsize=3, label=f"Replica {rep}")
    mean = combined["combined_ddg_kcal_mol"]
    sd = combined["cross_combination_sd_kcal_mol"]
    ax.axhspan(mean - sd, mean + sd, color=COLORS["gold"], alpha=0.2, label="Final empirical ±1 SD")
    ax.axhline(mean, color=COLORS["navy"], ls="--", lw=1.8)
    ax.axhline(0, color=COLORS["gray"], lw=1)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Cumulative sampling per λ window (ns)")
    ax.set_ylabel("Cumulative ΔΔG binding (kcal/mol)")
    ax.set_title("Cumulative convergence: direction agrees, magnitude remains replica-dependent")
    ax.legend(loc="upper left", ncol=2)
    save(fig, "04_cumulative_convergence")


def figure_pmf(analyses):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, leg in zip(axes, ("bound", "free")):
        for rep, color in ((1, COLORS["blue"]), (2, COLORS["orange"])):
            pmf = analyses[rep][f"{leg}_pmf"]
            x = np.asarray([row[0] for row in pmf])
            y = np.asarray([parse_energy(row[1]) for row in pmf])
            e = np.asarray([parse_energy(row[2]) for row in pmf])
            ax.errorbar(x, y, yerr=e, marker="o", ms=4, color=color, lw=2, capsize=2, label=f"Replica {rep}")
        ax.set_title(f"{leg.capitalize()} leg")
        ax.set_xlabel("λ")
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes[0].set_ylabel("Cumulative alchemical free energy (kcal/mol)")
    axes[0].legend()
    fig.suptitle("Alchemical PMF profiles", fontsize=14, fontweight="bold")
    save(fig, "05_bound_free_pmf_profiles")


def figure_rmsd(rmsd_data):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    time = np.linspace(0.1, 5.0, 50)
    for rep, ax in zip((1, 2), axes):
        for key, color, label in (
            ("lambda_0_AN139", COLORS["blue"], "λ=0, AN139"),
            ("lambda_1_KL101", COLORS["orange"], "λ=1, KL101"),
        ):
            values = np.asarray(rmsd_data[f"replica_{rep}"][key])
            ax.plot(time, values, color=color, lw=1.8, label=label)
        ax.axhline(2.0, color=COLORS["red"], ls="--", lw=1.2, label="2 Å reference" if rep == 1 else None)
        ax.set_title(f"Bound starting frame replica {rep}")
        ax.set_xlabel("Production time (ns)")
        ax.set_ylim(0, 2.2)
    axes[0].set_ylabel("Protein-aligned ligand common-core RMSD (Å)")
    axes[0].legend(loc="upper left")
    fig.suptitle("Bound endpoint poses remain stable", fontsize=14, fontweight="bold")
    save(fig, "06_bound_endpoint_pose_stability")


def figure_dashboard(combined, analyses, convergence):
    fig = plt.figure(figsize=(13, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    mean = combined["combined_ddg_kcal_mol"]
    sd = combined["cross_combination_sd_kcal_mol"]
    values = np.asarray(combined["cross_combinations_kcal_mol"])
    ax0.axvline(0, color=COLORS["gray"], lw=1)
    ax0.axvspan(mean - sd, mean + sd, color=COLORS["gold"], alpha=0.25)
    ax0.scatter(values, np.arange(8), c=[COLORS["blue"]] * 4 + [COLORS["orange"]] * 4, s=45)
    ax0.axvline(mean, color=COLORS["navy"], lw=2)
    ax0.set_yticks([])
    ax0.set_xlabel("ΔΔG (kcal/mol)")
    ax0.set_title(f"A  Final prediction: +{mean:.2f} ± {sd:.2f} kcal/mol", loc="left")
    ax0.text(0.98, 0.08, "KL101 ≈22× weaker", transform=ax0.transAxes, ha="right", color=COLORS["navy"], fontweight="bold")

    bound = np.asarray([x[0] for x in combined["bound_replicates_kcal_mol"]])
    free = np.asarray([x[0] for x in combined["free_replicates_kcal_mol"]])
    ax1.scatter(np.zeros(len(bound)), bound, color=COLORS["blue"], s=55)
    ax1.scatter(np.ones(len(free)), free, color=COLORS["orange"], s=55)
    ax1.errorbar([0, 1], [bound.mean(), free.mean()], [bound.std(ddof=1), free.std(ddof=1)], fmt="D", color=COLORS["navy"], capsize=6)
    ax1.set_xticks([0, 1], ["Bound (n=2)", "Free (n=4)"])
    ax1.set_ylabel("Endpoint ΔG (kcal/mol)")
    ax1.set_title("B  Replicate decomposition", loc="left")

    for rep, color in ((1, COLORS["blue"]), (2, COLORS["orange"])):
        rows = convergence[f"replica_{rep}"]
        ax2.plot([r["ns_per_window"] for r in rows], [r["ddg"] for r in rows], marker="o", color=color, lw=2, label=f"Replica {rep}")
    ax2.axhspan(mean - sd, mean + sd, color=COLORS["gold"], alpha=0.2)
    ax2.axhline(0, color=COLORS["gray"], lw=1)
    ax2.set_xlabel("Cumulative ns/window")
    ax2.set_ylabel("ΔΔG (kcal/mol)")
    ax2.set_title("C  Time convergence", loc="left")
    ax2.legend()

    labels, mins = [], []
    for rep in (1, 2):
        for leg in ("bound", "free"):
            labels.append(f"R{rep} {leg[0].upper()}")
            mins.append(analyses[rep][f"{leg}_adjacent_overlap_minimum"])
    bars = ax3.bar(labels, mins, color=[COLORS["blue"], COLORS["cyan"], COLORS["orange"], COLORS["gold"]])
    ax3.axhline(0.03, color=COLORS["red"], ls="--", lw=1.2, label="0.03 reference")
    ax3.set_ylim(0, 0.12)
    ax3.set_ylabel("Minimum adjacent overlap")
    ax3.set_title("D  Refined endpoint overlap", loc="left")
    for bar, value in zip(bars, mins):
        ax3.text(bar.get_x() + bar.get_width() / 2, value + 0.004, f"{value:.3f}", ha="center", fontsize=9)
    ax3.legend(loc="upper right")

    fig.suptitle("AN139 → KL101 refined relative binding free-energy pilot", fontsize=17, fontweight="bold")
    save(fig, "00_presentation_summary")


def captions() -> None:
    text = """# Figure index

1. `00_presentation_summary`: recommended single-slide overview.
2. `01_final_ddg_replicate_pairings`: all eight bound/free pairings and the empirical result.
3. `02_bound_free_replicate_decomposition`: shows that free-leg sampling dominates uncertainty.
4. `03_lambda_overlap_heatmaps`: four MBAR overlap matrices; λ=0.95 repairs the original endpoint bottleneck.
5. `04_cumulative_convergence`: cumulative ΔΔG from 1–5 ns/window for both full replicas.
6. `05_bound_free_pmf_profiles`: bound and free alchemical PMFs for both full replicas.
7. `06_bound_endpoint_pose_stability`: protein-aligned common-core RMSD for physical endpoints.

Use `00`, `01`, `02`, and `04` in the main presentation. Keep overlap, PMF, and RMSD figures as methodological backup. Error bars on individual MBAR curves are formal within-run errors; the headline ±0.39 kcal/mol is the empirical SD across all bound/free replicate pairings.
"""
    (FIGURES / "README.md").write_text(text)


def main() -> None:
    style()
    combined, analyses = load_inputs()
    convergence = convergence_data()
    rmsd_data = endpoint_rmsd_data()
    figure_headline(combined)
    figure_components(combined)
    figure_overlap(analyses)
    figure_convergence(convergence, combined)
    figure_pmf(analyses)
    figure_rmsd(rmsd_data)
    figure_dashboard(combined, analyses, convergence)
    captions()
    print(f"FIGURES={FIGURES}")


if __name__ == "__main__":
    main()
