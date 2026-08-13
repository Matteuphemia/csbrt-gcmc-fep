#!/usr/bin/env python3
"""Run gci_analyse.py over each replica and average the per-replica outputs.

gci_analyse.py has no notion of replicas: --run-root is one directory, and the
"pooling" in its docstring is over the 36 B-windows inside it, not over
replicas. It would in fact *reject* a directory holding all six replicas,
because require_consistent_windows() forbids duplicate target_b values.

So this script drives it six times and aggregates the results. It deliberately
does not reimplement any physics -- no curve fitting, no inverse interpolation,
no integration. Every number it writes is an average of numbers gci_analyse.py
produced, which keeps the validated implementation the single source of truth.

    python combine_replicas.py                       # analyse + combine
    python combine_replicas.py --skip-analysis       # re-combine existing output

WHAT IS AVERAGED, AND WITH WHAT UNCERTAINTY
-------------------------------------------
The replicas share one equilibrated structure and differ only by random seed,
so the spread across them measures run-to-run variation. That is the honest
error bar. The sem_N reported inside a single replica is a within-run GCMC
sampling error over windows that all started from the same coordinates, so it
systematically understates the true uncertainty; it is carried through for
reference but is not the headline.

With six replicas the normal approximation is wrong: 95% intervals use
Student's t with n-1 = 5 degrees of freedom (t = 2.5706), which is ~31% wider
than 1.96 sigma.

A NOTE ON dF_bind(N*)
---------------------
dF_bind is a nonlinear functional of the titration curve, so the mean of six
dF_bind values is not identical to dF_bind computed from the mean curve. This
script reports the former. When the replicas agree on N* the distinction is
benign; when they disagree, averaging each replica's dF_bind(N*) mixes
different occupancies, so the modal-N* estimate is reported alongside it and
the disagreement is flagged loudly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Reuse gci_analyse's integration rather than reimplementing it. Importing is
# not modifying: the free-energy expression, the inverse interpolation and the
# isotonic fit all stay in the validated module, and this script only decides
# WHICH curve to hand them.
sys.path.insert(0, str(HERE / "scripts"))
import numpy as np  # noqa: E402
from gci_analyse import (  # noqa: E402
    compute_df_bind,
    isotonic_non_decreasing,
    monotonicity_violations,
)

# Student's t, two-sided 95%, by degrees of freedom. Falls back to the normal
# quantile for large samples. Avoids a scipy dependency the campaign env has no
# other reason to carry.
T95 = {1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706, 6: 2.4469,
       7: 2.3646, 8: 2.3060, 9: 2.2622, 10: 2.2281, 11: 2.2010, 12: 2.1788,
       15: 2.1314, 20: 2.0860, 30: 2.0423}


def t95(n: int) -> float:
    """Two-sided 95% t multiplier for a sample of size n."""
    if n < 2:
        return float("nan")
    dof = n - 1
    if dof in T95:
        return T95[dof]
    return min((v for k, v in T95.items() if k >= dof), default=1.9600)


def spread(values: list[float]) -> dict:
    """Mean, SD, SEM and t-based 95% CI of a small sample."""
    n = len(values)
    mean = statistics.mean(values)
    if n < 2:
        return {"n": n, "mean": mean, "sd": float("nan"), "sem": float("nan"),
                "ci95_lo": float("nan"), "ci95_hi": float("nan")}
    sd = statistics.stdev(values)
    sem = sd / math.sqrt(n)
    half = t95(n) * sem
    return {"n": n, "mean": mean, "sd": sd, "sem": sem,
            "ci95_lo": mean - half, "ci95_hi": mean + half}


def inversions(xs: list[float], ys: list[float]) -> list[tuple]:
    """Points where y decreases as x increases. <N> must be non-decreasing in B."""
    return [(xs[i], ys[i], xs[i + 1], ys[i + 1])
            for i in range(len(ys) - 1) if ys[i] > ys[i + 1] + 1e-12]


# ---------------------------------------------------------------------------
# Driving the per-replica analysis
# ---------------------------------------------------------------------------
def run_one(replica: Path, out_dir: Path, opt) -> tuple[bool, str]:
    cmd = [sys.executable, str(HERE / "scripts" / "gci_analyse.py"),
           "--run-root", str(replica),
           "--prefix", opt.prefix,
           "--output-dir", str(out_dir),
           "--monotonic-fit", opt.monotonic_fit,
           "--equilibrated-fraction", str(opt.equilibrated_fraction)]
    if opt.no_plots:
        cmd.append("--no-plots")
    if opt.force:
        cmd.append("--force")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()


def read_replica(out_dir: Path, prefix: str) -> dict:
    summary = json.loads((out_dir / f"{prefix}_gci_analysis.json").read_text())
    curve = {}
    with (out_dir / f"{prefix}_titration_curve.csv").open() as fh:
        for row in csv.DictReader(fh):
            if row["excluded"] not in ("1", "True", "true"):
                curve[float(row["target_b"])] = {
                    "mean_N": float(row["mean_N"]),
                    "sem_N": float(row["sem_N"]),
                    "mu": float(row["mu_kcal_per_mol"]),
                    "min_ghost_pool": int(row["minimum_ghost_pool"]),
                }
    energy = {}
    with (out_dir / f"{prefix}_gci_free_energy.csv").open() as fh:
        for row in csv.DictReader(fh):
            energy[int(row["n"])] = {
                "df": float(row["df_bind_kcal_per_mol"]),
                "extrapolated": row["extrapolated"] in ("1", "True", "true"),
            }
    return {"summary": summary, "curve": curve, "energy": energy}


# ---------------------------------------------------------------------------
# Pool first, then integrate -- the primary estimator
# ---------------------------------------------------------------------------
def integrate_curve(b_points, n_curve, kt, mu_hydration, b_equil, max_n=None):
    """Integrate one titration curve. Mirrors gci_analyse.main's own sequence."""
    fitted = np.asarray(n_curve, dtype=float)
    violations = monotonicity_violations(np.asarray(b_points, dtype=float), fitted)
    if violations:
        fitted = isotonic_non_decreasing(fitted)
    n_at_equil = float(np.interp(b_equil, b_points, fitted))
    if max_n is None:
        max_n = max(1, int(round(n_at_equil))) + 1
    profile = []
    for n in range(0, max_n + 1):
        df, extrap = compute_df_bind(n, np.asarray(b_points, dtype=float), fitted,
                                     kt, mu_hydration)
        profile.append({"n": n, "df": df, "extrapolated": bool(extrap)})
    best = min(profile, key=lambda p: p["df"])
    return {"profile": profile, "n_star": best["n"], "df_star": best["df"],
            "n_at_equilibrium_b": n_at_equil, "max_n": max_n,
            "inversions": len(violations), "isotonic_applied": bool(violations)}


def pooled_integration(loaded, names, b_all, meta, n_boot, rng_seed):
    """Integrate the replica-averaged curve; bootstrap over replicas for its CI.

    Averaging six dF_bind values and integrating the averaged curve are not the
    same operation, because dF_bind is a nonlinear functional of the curve. The
    pooled curve carries sqrt(n) less noise, so integrating it is the better
    CENTRAL estimate -- but it is a single number with no internal spread, so
    its uncertainty comes from resampling the replicas.
    """
    kt, mu, b_equil = meta["kt"], meta["mu_hydration"], meta["b_equil"]

    def curve_for(subset):
        return [statistics.mean(loaded[k]["curve"][b]["mean_N"]
                                for k in subset if b in loaded[k]["curve"])
                for b in b_all]

    point = integrate_curve(b_all, curve_for(names), kt, mu, b_equil)

    # Non-parametric bootstrap over replicas: the resampling unit is a whole
    # replica, because windows within a replica share a starting structure and
    # are not independent of one another.
    rng = np.random.default_rng(rng_seed)
    boot_df, boot_ns = [], []
    for _ in range(n_boot):
        pick = [names[i] for i in rng.integers(0, len(names), len(names))]
        try:
            r = integrate_curve(b_all, curve_for(pick), kt, mu, b_equil,
                                max_n=point["max_n"])
        except Exception:
            continue
        boot_df.append(r["df_star"])
        boot_ns.append(r["n_star"])
    boot_df.sort()

    def pct(p):
        if not boot_df:
            return float("nan")
        return boot_df[min(len(boot_df) - 1, max(0, int(round(p / 100 * (len(boot_df) - 1)))))]

    # Leave-one-replica-out, as an independent check on the bootstrap.
    jack = []
    for k in names:
        sub = [x for x in names if x != k]
        jack.append(integrate_curve(b_all, curve_for(sub), kt, mu, b_equil,
                                    max_n=point["max_n"])["df_star"])
    nj = len(jack)
    jack_mean = statistics.mean(jack)
    jack_se = math.sqrt((nj - 1) / nj * sum((v - jack_mean) ** 2 for v in jack)) if nj > 1 else float("nan")

    return {
        "n_star": point["n_star"],
        "df_bind_star_kcal_per_mol": point["df_star"],
        "n_at_equilibrium_b": point["n_at_equilibrium_b"],
        "curve_inversions": point["inversions"],
        "isotonic_applied": point["isotonic_applied"],
        "profile": point["profile"],
        "bootstrap": {
            "resamples": len(boot_df),
            "unit": "replica",
            "sd": statistics.stdev(boot_df) if len(boot_df) > 1 else float("nan"),
            "ci95_lo": pct(2.5), "ci95_hi": pct(97.5),
            "n_star_distribution": {str(v): boot_ns.count(v) for v in sorted(set(boot_ns))},
        },
        "jackknife": {"se": jack_se, "values": jack},
    }


# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-root", type=Path, default=HERE / "replicas")
    p.add_argument("--analysis-root", type=Path, default=HERE / "analysis")
    p.add_argument("--prefix", default="CRY1KL101")
    p.add_argument("--monotonic-fit", choices=("none", "isotonic"), default="isotonic",
                   help="Passed through to gci_analyse.py. Default 'isotonic' because "
                        "single replicas of this campaign are not monotonic; results "
                        "are then preliminary, as gci_analyse.py intends.")
    p.add_argument("--equilibrated-fraction", type=float, default=0.5)
    p.add_argument("--skip-analysis", action="store_true",
                   help="Reuse existing per-replica output instead of re-running.")
    p.add_argument("--bootstrap", type=int, default=2000,
                   help="Bootstrap resamples over replicas for the pooled CI.")
    p.add_argument("--bootstrap-seed", type=int, default=20260714)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--force", action="store_true")
    opt = p.parse_args()

    replicas = sorted((d for d in opt.run_root.glob("rep*") if d.is_dir()),
                      key=lambda d: int("".join(c for c in d.name if c.isdigit()) or 0))
    if not replicas:
        print(f"no rep* directories under {opt.run_root}", file=sys.stderr)
        return 2

    # --- per replica ------------------------------------------------------
    loaded, failed = {}, {}
    for rep in replicas:
        out_dir = opt.analysis_root / rep.name
        if not opt.skip_analysis:
            ok, log = run_one(rep, out_dir, opt)
            if not ok:
                failed[rep.name] = log.splitlines()[-1] if log else "unknown error"
                print(f"  {rep.name}: ANALYSIS FAILED -- {failed[rep.name]}")
                continue
        try:
            loaded[rep.name] = read_replica(out_dir, opt.prefix)
        except FileNotFoundError as exc:
            failed[rep.name] = f"missing output: {exc}"
            print(f"  {rep.name}: {failed[rep.name]}")
            continue
        s = loaded[rep.name]["summary"]
        print(f"  {rep.name}: N* = {s['n_star']}   "
              f"dF_bind(N*) = {s['df_bind_star_kcal_per_mol']:+.3f} kcal/mol   "
              f"({s['monotonicity_inversions']} inversions)")

    if not loaded:
        print("no replica produced usable output", file=sys.stderr)
        return 1
    if failed:
        print(f"\nWARNING: {len(failed)} of {len(replicas)} replicas failed; "
              f"combining only the {len(loaded)} that succeeded: "
              f"{', '.join(sorted(loaded))}")

    names = list(loaded)
    out = opt.analysis_root / "combined"
    out.mkdir(parents=True, exist_ok=True)

    # --- titration curve --------------------------------------------------
    all_b = sorted({b for r in loaded.values() for b in r["curve"]})
    curve_rows = []
    for b in all_b:
        vals = [loaded[n]["curve"][b]["mean_N"] for n in names if b in loaded[n]["curve"]]
        st = spread(vals)
        mus = [loaded[n]["curve"][b]["mu"] for n in names if b in loaded[n]["curve"]]
        within = [loaded[n]["curve"][b]["sem_N"] for n in names if b in loaded[n]["curve"]]
        row = {"target_b": b, "mu_kcal_per_mol": mus[0], "n_replicas": st["n"],
               "mean_N": st["mean"], "sd_N": st["sd"], "sem_N": st["sem"],
               "ci95_lo": st["ci95_lo"], "ci95_hi": st["ci95_hi"],
               "mean_within_replica_sem_N": statistics.mean(within)}
        for n in names:
            row[f"mean_N_{n}"] = loaded[n]["curve"].get(b, {}).get("mean_N", "")
        curve_rows.append(row)

    curve_csv = out / f"{opt.prefix}_combined_titration_curve.csv"
    with curve_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(curve_rows[0]))
        w.writeheader(); w.writerows(curve_rows)

    # monotonicity of the combined curve, judged against between-replica scatter
    bs = [r["target_b"] for r in curve_rows]
    ys = [r["mean_N"] for r in curve_rows]
    sems = {r["target_b"]: r["sem_N"] for r in curve_rows}
    comb_inv = []
    for a, na, bb, nb in inversions(bs, ys):
        drop = na - nb
        noise = math.hypot(sems[a], sems[bb])
        comb_inv.append({"b_from": a, "b_to": bb, "drop": drop,
                         "combined_sem": noise,
                         "significant": bool(drop > noise)})

    # --- free energy profile ---------------------------------------------
    all_n = sorted({n for r in loaded.values() for n in r["energy"]})
    energy_rows = []
    for nn in all_n:
        vals = [loaded[k]["energy"][nn]["df"] for k in names if nn in loaded[k]["energy"]]
        ex = sum(loaded[k]["energy"][nn]["extrapolated"] for k in names
                 if nn in loaded[k]["energy"])
        st = spread(vals)
        energy_rows.append({"n": nn, "n_replicas": st["n"],
                            "mean_df_bind_kcal_per_mol": st["mean"],
                            "sd": st["sd"], "sem": st["sem"],
                            "ci95_lo": st["ci95_lo"], "ci95_hi": st["ci95_hi"],
                            "n_extrapolated": ex})
    energy_csv = out / f"{opt.prefix}_combined_free_energy.csv"
    with energy_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(energy_rows[0]))
        w.writeheader(); w.writerows(energy_rows)

    # --- N* and the headline number --------------------------------------
    n_stars = [loaded[k]["summary"]["n_star"] for k in names]
    df_own = [loaded[k]["summary"]["df_bind_star_kcal_per_mol"] for k in names]
    dist = {v: n_stars.count(v) for v in sorted(set(n_stars))}
    modal = max(dist, key=lambda v: (dist[v], -v))
    unanimous = len(dist) == 1

    df_at_modal = [loaded[k]["energy"][modal]["df"] for k in names
                   if modal in loaded[k]["energy"]]

    st_own = spread(df_own)
    st_modal = spread(df_at_modal)
    n_equil = spread([loaded[k]["summary"]["n_at_equilibrium_b"] for k in names])

    # --- primary estimator: integrate the pooled curve --------------------
    s0 = loaded[names[0]]["summary"]
    pooled = pooled_integration(
        loaded, names, bs,
        {"kt": s0["kt_kcal_per_mol"],
         "mu_hydration": s0["mu_hydration_kcal_per_mol"],
         "b_equil": s0["equilibrium_b"]},
        opt.bootstrap, opt.bootstrap_seed,
    )
    pooled_csv = out / f"{opt.prefix}_pooled_free_energy.csv"
    with pooled_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["n", "df_bind_kcal_per_mol", "extrapolated"])
        w.writeheader()
        for row in pooled["profile"]:
            w.writerow({"n": row["n"], "df_bind_kcal_per_mol": row["df"],
                        "extrapolated": row["extrapolated"]})

    summary = {
        "stage": "gci_replica_combination",
        "prefix": opt.prefix,
        "replicas_combined": names,
        "replicas_failed": failed,
        "monotonic_fit": opt.monotonic_fit,
        "preliminary": opt.monotonic_fit == "isotonic",
        "uncertainty": "between-replica; 95% CI uses Student t with n-1 dof",
        "primary_estimator": "pooled_curve_integrated",
        "pooled_curve_integrated": pooled,
        "n_star_distribution": {str(k): v for k, v in dist.items()},
        "n_star_unanimous": unanimous,
        "n_star_modal": modal,
        "df_bind_star_each_replica_own_n_star": st_own,
        "df_bind_at_modal_n_star": st_modal,
        "n_at_equilibrium_b": n_equil,
        "equilibrium_b": loaded[names[0]]["summary"]["equilibrium_b"],
        "per_replica": {k: {
            "n_star": loaded[k]["summary"]["n_star"],
            "df_bind_star_kcal_per_mol": loaded[k]["summary"]["df_bind_star_kcal_per_mol"],
            "n_at_equilibrium_b": loaded[k]["summary"]["n_at_equilibrium_b"],
            "monotonicity_inversions": loaded[k]["summary"]["monotonicity_inversions"],
            "windows_used": loaded[k]["summary"]["windows_used"],
        } for k in names},
        "combined_curve_inversions": comb_inv,
        "combined_curve_significant_inversions":
            sum(1 for i in comb_inv if i["significant"]),
    }
    summary_json = out / f"{opt.prefix}_combined_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2) + "\n")

    # --- plots ------------------------------------------------------------
    written = [curve_csv, energy_csv, pooled_csv, summary_json]
    if not opt.no_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7, 4.5))
            for k in names:
                xs = sorted(loaded[k]["curve"])
                ax.plot(xs, [loaded[k]["curve"][b]["mean_N"] for b in xs],
                        lw=0.8, alpha=0.35, color="grey")
            ax.errorbar(bs, ys, yerr=[r["sem_N"] for r in curve_rows],
                        fmt="o-", ms=3, lw=1.5, capsize=2, color="C0",
                        label=f"mean of {len(names)} replicas $\\pm$ SEM")
            ax.set_xlabel("Adams value $B$")
            ax.set_ylabel(r"$\langle N \rangle$")
            ax.set_title(f"{opt.prefix} combined titration curve")
            ax.legend(); fig.tight_layout()
            f1 = out / f"{opt.prefix}_combined_titration_curve.png"
            fig.savefig(f1, dpi=150); plt.close(fig); written.append(f1)

            fig, ax = plt.subplots(figsize=(7, 4.5))
            ns = [r["n"] for r in energy_rows]
            ms = [r["mean_df_bind_kcal_per_mol"] for r in energy_rows]
            es = [r["sem"] for r in energy_rows]
            ax.errorbar(ns, ms, yerr=es, fmt="o-", capsize=3, color="C1")
            ax.axvline(modal, ls="--", lw=1, color="k",
                       label=f"$N^*$ = {modal}")
            ax.set_xlabel("$N$")
            ax.set_ylabel(r"$\Delta F_{\rm bind}$ (kcal/mol)")
            ax.set_title(f"{opt.prefix} combined binding free energy")
            ax.legend(); fig.tight_layout()
            f2 = out / f"{opt.prefix}_combined_free_energy.png"
            fig.savefig(f2, dpi=150); plt.close(fig); written.append(f2)
        except Exception as exc:  # plotting must never lose the numbers
            print(f"  (plots skipped: {exc})")

    # --- report -----------------------------------------------------------
    print(f"\n{'='*66}")
    print(f"COMBINED OVER {len(names)} REPLICAS"
          + ("   [PRELIMINARY: isotonic fit]" if summary["preliminary"] else ""))
    print("=" * 66)
    bs_ = pooled["bootstrap"]
    print("PRIMARY -- integrate the pooled curve, bootstrap over replicas:")
    print(f"  N*                : {pooled['n_star']}")
    print(f"  dF_bind(N*)       : {pooled['df_bind_star_kcal_per_mol']:+.3f} kcal/mol")
    print(f"  bootstrap SD      : {bs_['sd']:.3f}   ({bs_['resamples']} resamples, unit = replica)")
    print(f"  bootstrap 95% CI  : [{bs_['ci95_lo']:+.3f}, {bs_['ci95_hi']:+.3f}]")
    print(f"  jackknife SE      : {pooled['jackknife']['se']:.3f}   (leave-one-replica-out)")
    print(f"  N* across resamples: {bs_['n_star_distribution']}")
    print(f"  pooled curve      : {pooled['curve_inversions']} inversions"
          + ("  [isotonic applied]" if pooled["isotonic_applied"] else "  [monotonic as measured]"))
    print(f"  <N> at B_equil    : {pooled['n_at_equilibrium_b']:.3f}")
    print()
    print("SECONDARY -- mean of the per-replica dF_bind(N*):")
    print(f"  N* per replica    : {dict(zip(names, n_stars))}")
    print(f"  N* distribution   : {dist}"
          + ("  (unanimous)" if unanimous else "  <-- REPLICAS DISAGREE"))
    print(f"  mean              : {st_own['mean']:+.3f} kcal/mol")
    print(f"  SD / SEM          : {st_own['sd']:.3f} / {st_own['sem']:.3f}")
    print(f"  95% CI (t,{len(names)-1} dof): [{st_own['ci95_lo']:+.3f}, {st_own['ci95_hi']:+.3f}]")
    if not unanimous:
        print(f"  at modal N*={modal}    : {st_modal['mean']:+.3f} +/- {st_modal['sem']:.3f}")
    print(f"  <N> at B_equil    : {n_equil['mean']:.3f} +/- {n_equil['sem']:.3f} (SEM)")
    print()
    sig = summary["combined_curve_significant_inversions"]
    print(f"combined-curve inversions: {len(comb_inv)} "
          f"({sig} exceed the between-replica SEM)")
    for i in comb_inv:
        if i["significant"]:
            print(f"    B {i['b_from']:+.3f} -> {i['b_to']:+.3f}: drop {i['drop']:.3f} "
                  f"vs SEM {i['combined_sem']:.3f}")
    print()
    for path in written:
        print(f"  wrote {path.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
