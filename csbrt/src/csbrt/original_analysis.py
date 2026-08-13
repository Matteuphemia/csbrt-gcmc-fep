#!/usr/bin/env python3
"""The original GCI analysis, extracted verbatim from the notebook.

Source: Data/GCI_Analysis_discrete.ipynb ("GCI Analysis - Discrete approach,
Ross et al. 2015"), 23 cells, no stored outputs. The parsing regex,
``integrate_titration``, ``compute_dF_bind``, the tail estimator and the
automatic N_MAX_EVAL rule are reproduced unchanged so that this and the new
pipeline can be compared on identical data.

Two constants are exposed on the command line rather than hardcoded, because
the notebook's shipped values disagree with the data it was pointed at:

  --radius     notebook ships 7 A; the archived runs used 4 A (V_sphere = 268.1)
  --checkpoint notebook ships 200 moves; the records are 400 moves apart

Running with the shipped values reproduces the original result including those
errors; running with the corrected values isolates their effect.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from math import lgamma

import numpy as np

# ---- notebook cell 4: physical constants (kT truncated, as in the original) --
KT = 0.5961          # kcal/mol at 300 K
MU_SOL = -6.09       # kcal/mol, mu'_sol TIP3P/OpenMM
STANDARD_VOL = 30.345


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--pattern", default="titration_*.out")
    parser.add_argument("--radius", type=float, default=7.0,
                        help="RADIUS_ANG (notebook ships 7.0)")
    parser.add_argument("--checkpoint", type=int, default=200,
                        help="CHECKPOINT_INT (notebook ships 200)")
    parser.add_argument("--last-moves", type=int, default=25000,
                        help="N_LAST_MOVES (notebook ships 25000)")
    parser.add_argument("--max-n", type=int, default=None)
    parser.add_argument("--output", default=None, help="Write a JSON summary here")
    return parser.parse_args()


# ---- notebook cell 6: parsing ----------------------------------------------
def parse_titration_file(filepath, n_last_moves, checkpoint_int):
    mu, B = None, None
    moves, avg_N, current_N, acceptance = [], [], [], []

    re_mu = re.compile(r"mu\s*=\s*([\-\d\.]+)\s*kcal")
    re_B = re.compile(r"B value\s*:\s*([\-\d\.]+)")
    re_gcmc = re.compile(
        r"(\d+)\s+move\(s\) completed\s+\((\d+)\s+accepted\s+\(([\d\.]+)\s*%\)\)\."
        r"\s+Current N\s*=\s*(\d+).*Average N\s*=\s*([\d\.]+)"
    )

    with open(filepath, errors="replace") as fh:
        for line in fh:
            if mu is None:
                m = re_mu.search(line)
                if m:
                    mu = float(m.group(1))
            if B is None:
                m = re_B.search(line)
                if m:
                    B = float(m.group(1))
            m = re_gcmc.search(line)
            if m:
                moves.append(int(m.group(1)))
                acceptance.append(float(m.group(3)))
                current_N.append(int(m.group(4)))
                avg_N.append(float(m.group(5)))

    current_N = np.array(current_N)
    n_last = n_last_moves // checkpoint_int
    tail = current_N[-n_last:] if len(current_N) >= n_last else current_N
    mean_N = float(np.mean(tail))
    std_N = float(np.std(tail))
    return {
        "file": os.path.basename(filepath),
        "mu": mu,
        "B": B,
        "current_N": current_N,
        "final_N": avg_N[-1] if avg_N else None,
        "mean_N_last": mean_N,
        "std_N_last": std_N,
        "sem_N_last": std_N / np.sqrt(len(tail)),
        "n_tail": int(len(tail)),
    }


# ---- notebook cell 15: integration and free energy --------------------------
def integrate_titration(B_pts, N_pts, B_upper):
    B_min_data = B_pts[0]
    B_max_data = B_pts[-1]
    extrapolated = False
    if B_upper <= B_min_data:
        return 0.0, extrapolated
    if B_upper <= B_max_data:
        N_at_upper = float(np.interp(B_upper, B_pts, N_pts))
        mask = B_pts < B_upper
        B_int = np.concatenate([B_pts[mask], [B_upper]])
        N_int = np.concatenate([N_pts[mask], [N_at_upper]])
    else:
        extrapolated = True
        slope = ((N_pts[-1] - N_pts[-2]) / (B_pts[-1] - B_pts[-2])
                 if len(B_pts) >= 2 else 0.0)
        N_at_upper = max(N_pts[-1] + slope * (B_upper - B_max_data), 0.0)
        B_int = np.concatenate([B_pts, [B_upper]])
        N_int = np.concatenate([N_pts, [N_at_upper]])
    N_int = np.clip(N_int, 0, None)
    return float(np.trapezoid(N_int, B_int)), extrapolated


def compute_dF_bind(N, B_pts, N_pts, kT, mu_hyd):
    if N == 0:
        return 0.0, False
    N_max_data = N_pts[-1]
    N_min_data = N_pts[0]
    extrapolated = False
    if N_min_data <= N <= N_max_data:
        B_N = float(np.interp(float(N), N_pts, B_pts))
    elif N > N_max_data:
        extrapolated = True
        slope_BN = ((B_pts[-1] - B_pts[-2]) / (N_pts[-1] - N_pts[-2])
                    if len(B_pts) >= 2 else 0.0)
        B_N = B_pts[-1] + slope_BN * (N - N_max_data)
    else:
        extrapolated = True
        slope_BN = ((B_pts[1] - B_pts[0]) / (N_pts[1] - N_pts[0])
                    if len(B_pts) >= 2 else 0.0)
        B_N = B_pts[0] + slope_BN * (N - N_min_data)
    integral, extrap_int = integrate_titration(B_pts, N_pts, B_N)
    dF = kT * (N * B_N + (-lgamma(N + 1)) - integral) - N * mu_hyd
    return float(dF), bool(extrapolated or extrap_int)


def main() -> None:
    opt = options()
    v_sys = (4 / 3) * np.pi * opt.radius**3
    b_equil = (MU_SOL / KT) + np.log(v_sys / STANDARD_VOL)

    files = sorted(glob.glob(os.path.join(opt.input_dir, opt.pattern)))
    if not files:
        raise SystemExit(f"No files matching {opt.pattern} in {opt.input_dir}")
    replicas = [parse_titration_file(f, opt.last_moves, opt.checkpoint) for f in files]
    replicas.sort(key=lambda r: r["B"])

    print(f"RADIUS_ANG     = {opt.radius:g}   ->  V_sys = {v_sys:.1f} A^3")
    print(f"CHECKPOINT_INT = {opt.checkpoint}   N_LAST_MOVES = {opt.last_moves}"
          f"  ->  tail = {replicas[0]['n_tail']} checkpoints")
    print(f"B_equil        = {b_equil:.4f}")
    print(f"{len(replicas)} files parsed")

    B_use = np.array([r["B"] for r in replicas])
    N_use = np.array([r["mean_N_last"] for r in replicas])

    # cell 14: automatic N_MAX_EVAL from <N> at B_equil
    mask_le = B_use <= b_equil
    if mask_le.sum() == 0:
        raise SystemExit("No point with B <= B_equil")
    n_at_equil = float(np.interp(b_equil, B_use[mask_le], N_use[mask_le]))
    max_n = opt.max_n if opt.max_n is not None else max(1, int(np.round(n_at_equil))) + 1

    order = np.argsort(B_use)
    B_sorted, N_sorted = B_use[order], N_use[order]

    print(f"<N> at B_equil = {n_at_equil:.3f}   N_MAX_EVAL = {max_n}")
    print(f"{'N':>4s}  {'dF_bind (kcal/mol)':>20s}")
    rows = []
    for N in range(0, max_n + 1):
        dF, ext = compute_dF_bind(N, B_sorted, N_sorted, KT, MU_SOL)
        rows.append({"n": N, "df_bind_kcal_per_mol": dF, "extrapolated": ext})
        print(f"  {N:2d}   {dF:>+18.3f}{'  (extrapolated)' if ext else ''}")

    values = [r["df_bind_kcal_per_mol"] for r in rows]
    best = int(np.argmin(values))
    n_star, df_star = rows[best]["n"], values[best]
    print(f"\n  N* = {n_star}    dF_bind(N*) = {df_star:+.3f} kcal/mol")

    inversions = int(sum(1 for i in range(len(N_sorted) - 1)
                         if N_sorted[i + 1] < N_sorted[i] - 1e-12))
    print(f"  monotonicity inversions in the curve: {inversions} "
          f"(the notebook does not check this)")

    if opt.output:
        with open(opt.output, "w") as handle:
            json.dump(
                {
                    "analysis": "original notebook (extracted)",
                    "radius_angstrom": opt.radius,
                    "checkpoint_interval": opt.checkpoint,
                    "last_moves": opt.last_moves,
                    "tail_checkpoints": replicas[0]["n_tail"],
                    "kt_kcal_per_mol": KT,
                    "equilibrium_b": float(b_equil),
                    "n_at_equilibrium_b": n_at_equil,
                    "n_star": n_star,
                    "df_bind_star_kcal_per_mol": df_star,
                    "monotonicity_inversions": inversions,
                    "curve": [
                        {"B": float(r["B"]), "mu": r["mu"],
                         "mean_N": r["mean_N_last"], "sem_N": r["sem_N_last"]}
                        for r in replicas
                    ],
                    "free_energy": rows,
                },
                handle,
                indent=2,
            )
        print(f"  wrote {opt.output}")


if __name__ == "__main__":
    main()
