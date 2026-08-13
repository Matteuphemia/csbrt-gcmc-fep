#!/usr/bin/env python3
"""Water-network analysis in the style of Cai (CRY1/AN139, M1 report 2026).

Reproduces that study's analysis for the CRY1/KL101 campaign so the two are
directly comparable:

  * per-replica average-linkage clustering of water-oxygen positions (2.4 A),
  * matching to crystallographic waters (2.0 A, minimum occupancy 0.2),
  * recall / precision / Tanimoto per replica, reported as mean +/- SD,
  * consensus hydration sites (complete-linkage 2.0 A, present in >= 75% of
    replicas),
  * occupancy and matching distance against crystallographic B-factor.

    python water_network.py --crystal inputs/6kx6.pdb --chain A

TWO THINGS TO BE HONEST ABOUT
-----------------------------
1. Only the inner GCMC sphere (radius 4 A) had grand-canonical sampling. Beyond
   it, water is sampled by ordinary MD, so occupancies out there are ordinary MD
   occupancies -- they are not enhanced, and a buried site that MD cannot fill
   will stay empty. Every site is flagged inside/outside the sphere.
2. Cai's clustering forbids two waters from the same frame merging into one
   cluster. Plain average linkage cannot express that constraint, so clusters
   here are split afterwards if they hold more than one water in the same frame
   too often; `max_same_frame_occupancy` records how well that worked.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import mdtraj as md
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from water_sites import AA3, ghost_sets, parse_pdb

HERE = Path(__file__).resolve().parent


def load_region(wd, prefix, centre, radius, stride):
    """Ghost-masked water-oxygen positions inside the region, per frame."""
    top = md.load_prmtop(str(wd / f"{prefix}-loch-ghosts.prmtop"))
    traj = md.load_dcd(str(wd / f"{prefix}-raw.dcd"), top=top)
    gs = ghost_sets(wd / f"{prefix}-gcmc-ghosts.txt")
    o_idx, o_res = [], []
    for res in top.residues:
        if res.name == "HOH":
            for a in res.atoms:
                if a.element.symbol == "O":
                    o_idx.append(a.index)
                    o_res.append(res.index)
    o_idx, o_res = np.array(o_idx), np.array(o_res)
    out = []
    for k in range(traj.n_frames):
        cycle = (k + 1) * stride
        ghosts = gs[cycle - 1] if cycle - 1 < len(gs) else set()
        live = ~np.isin(o_res, list(ghosts)) if ghosts else np.ones(len(o_res), bool)
        xyz = traj.xyz[k, o_idx[live], :] * 10.0
        out.append(xyz[np.linalg.norm(xyz - centre, axis=1) < radius])
    return out


def cluster_frames(frames, cutoff, max_same_frame=0.25):
    """Average-linkage clustering of pooled positions, with a same-frame split.

    Returns clusters as dicts: centre, occupancy (fraction of frames holding at
    least one member), n_members, same_frame_rate.
    """
    pts = np.vstack([f for f in frames if len(f)]) if any(len(f) for f in frames) \
        else np.empty((0, 3))
    if len(pts) < 2:
        return []
    owner = np.concatenate([[i] * len(f) for i, f in enumerate(frames) if len(f)])
    Z = linkage(pdist(pts), method="average")
    labels = fcluster(Z, t=cutoff, criterion="distance")

    # Cai's constraint: two waters coexisting in one frame are distinct sites.
    # Split any cluster that violates this too often by re-cutting it tighter.
    final = []
    for lab in np.unique(labels):
        m = labels == lab
        sub, subown = pts[m], owner[m]
        _, counts = np.unique(subown, return_counts=True)
        rate = float((counts > 1).mean()) if len(counts) else 0.0
        if rate > max_same_frame and m.sum() > 2:
            Zs = linkage(pdist(sub), method="average")
            for t in (cutoff * 0.6, cutoff * 0.4, cutoff * 0.25):
                sl = fcluster(Zs, t=t, criterion="distance")
                ok = True
                for s in np.unique(sl):
                    _, cc = np.unique(subown[sl == s], return_counts=True)
                    if len(cc) and (cc > 1).mean() > max_same_frame:
                        ok = False
                        break
                if ok:
                    break
            for s in np.unique(sl):
                final.append((sub[sl == s], subown[sl == s]))
        else:
            final.append((sub, subown))

    n_frames = len(frames)
    out = []
    for sub, subown in final:
        u, counts = np.unique(subown, return_counts=True)
        out.append({"centre": sub.mean(axis=0),
                    "occupancy": len(u) / n_frames,
                    "n_members": int(len(sub)),
                    "same_frame_rate": float((counts > 1).mean()) if len(counts) else 0.0,
                    "spread": float(np.linalg.norm(sub - sub.mean(axis=0), axis=1).mean())})
    out.sort(key=lambda c: -c["occupancy"])
    return out


def match(clusters, xray, cutoff, min_occ):
    """Greedy one-to-one matching of clusters to crystallographic waters."""
    cand = [c for c in clusters if c["occupancy"] >= min_occ]
    if not cand or not len(xray):
        return {}, cand
    D = np.linalg.norm(np.array([c["centre"] for c in cand])[:, None, :]
                       - xray[None, :, :], axis=-1)
    pairs = {}
    used_c, used_x = set(), set()
    order = np.dstack(np.unravel_index(np.argsort(D, axis=None), D.shape))[0]
    for ci, xi in order:
        if D[ci, xi] > cutoff:
            break
        if ci in used_c or xi in used_x:
            continue
        pairs[int(xi)] = {"cluster": int(ci), "distance": float(D[ci, xi]),
                          "occupancy": cand[int(ci)]["occupancy"]}
        used_c.add(ci)
        used_x.add(xi)
    return pairs, cand


def superpose_chain(crystal, chain):
    """Map a deposited chain's waters into the simulation frame."""
    import sys as _sys
    _sys.path.insert(0, str(HERE / "scripts"))
    from gci_map_centre import kabsch
    top = md.load_prmtop(str(HERE / "inputs" / "CRY1KL101uvt2.prmtop"))
    sim = md.load(str(HERE / "inputs" / "CRY1KL101uvt2.rst7"), top=top)
    sim_ca = {}
    for r in top.residues:
        if r.is_protein:
            for a in r.atoms:
                if a.name == "CA":
                    sim_ca[r.resSeq] = {"name": AA3.get(r.name, r.name),
                                        "xyz": sim.xyz[0, a.index, :] * 10.0}
    ca, wat, _ = parse_pdb(crystal)
    atoms = ca[chain]
    off, n = max(((o, sum(1 for a in atoms if a["resSeq"] + o in sim_ca
                          and sim_ca[a["resSeq"] + o]["name"] == a["name"]))
                  for o in range(-30, 31)), key=lambda t: t[1])
    pairs = [(a["xyz"], sim_ca[a["resSeq"] + off]["xyz"]) for a in atoms
             if a["resSeq"] + off in sim_ca and sim_ca[a["resSeq"] + off]["name"] == a["name"]]
    R, t, rmsd = kabsch(np.array([p for p, _ in pairs]), np.array([q for _, q in pairs]))
    w = wat[chain]
    xyz = (R @ np.array([x["xyz"] for x in w]).T).T + t
    return xyz, w, rmsd, off, len(pairs)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-root", type=Path, default=HERE / "replicas")
    p.add_argument("--prefix", default="CRY1KL101")
    p.add_argument("--crystal", type=Path, default=HERE / "inputs" / "6kx6.pdb")
    p.add_argument("--chain", default="A", help="Deposited chain the system was built from.")
    p.add_argument("--region-radius", type=float, default=10.0)
    p.add_argument("--gcmc-radius", type=float, default=4.0)
    p.add_argument("--cluster-cutoff", type=float, default=2.4)
    p.add_argument("--consensus-cutoff", type=float, default=2.0)
    p.add_argument("--match-cutoff", type=float, default=2.0)
    p.add_argument("--min-occupancy", type=float, default=0.2)
    p.add_argument("--consensus-fraction", type=float, default=0.75)
    p.add_argument("--windows", default="31,32,33",
                   help="Window indices to pool (default: the three rungs bracketing B_equil).")
    p.add_argument("--output", type=Path, default=HERE / "analysis" / "combined")
    opt = p.parse_args()

    reps = sorted((d for d in opt.run_root.glob("rep*") if d.is_dir()),
                  key=lambda d: int("".join(c for c in d.name if c.isdigit()) or 0))
    widx = [int(x) for x in opt.windows.split(",")]
    wdirs = {}
    for rep in reps:
        wdirs[rep.name] = [next(rep.glob(f"window_{i:02d}_B*")) for i in widx]
    meta = json.loads((wdirs[reps[0].name][0] / f"{opt.prefix}_titration.json").read_text())
    centre = np.array(meta["sphere_centre_angstrom"], dtype=float)
    stride = meta["trajectory_stride"]
    bvals = [json.loads((w / f"{opt.prefix}_titration.json").read_text())["target_b"]
             for w in wdirs[reps[0].name]]

    print(f"region: {opt.region_radius:g} A around {centre} "
          f"(GCMC sphere {opt.gcmc_radius:g} A)")
    print(f"windows pooled: {widx} -> B = {bvals}  (B_equil = {meta['equilibrium_b']:.4f})")

    xw, xmeta, rmsd, off, npairs = superpose_chain(opt.crystal, opt.chain)
    inreg = np.linalg.norm(xw - centre, axis=1) < opt.region_radius
    xray = xw[inreg]
    xinfo = [xmeta[i] for i in np.where(inreg)[0]]
    print(f"crystal chain {opt.chain}: superposed {npairs} CA (offset {off:+d}), RMSD {rmsd:.2f} A")
    print(f"  {len(xray)} crystallographic waters inside the region "
          f"({int((np.linalg.norm(xray - centre, axis=1) < opt.gcmc_radius).sum())} inside the GCMC sphere)")

    per_rep, stats = {}, []
    for rep in reps:
        frames = []
        for wd in wdirs[rep.name]:
            frames += load_region(wd, opt.prefix, centre, opt.region_radius, stride)
        cl = cluster_frames(frames, opt.cluster_cutoff)
        pairs, cand = match(cl, xray, opt.match_cutoff, opt.min_occupancy)
        nx, ns, m = len(xray), len(cand), len(pairs)
        stats.append({"replica": rep.name, "frames": len(frames), "n_clusters": len(cl),
                      "n_clusters_occ": ns, "matched": m,
                      "recall": m / nx if nx else float("nan"),
                      "precision": m / ns if ns else float("nan"),
                      "tanimoto": m / (ns + nx - m) if (ns + nx - m) else float("nan")})
        per_rep[rep.name] = {"clusters": cl, "pairs": pairs}
        s = stats[-1]
        print(f"  {rep.name}: {len(frames)} frames, {len(cl)} clusters "
              f"({ns} with occ>={opt.min_occupancy:g}), matched {m}/{nx}  "
              f"recall {s['recall']:.3f} precision {s['precision']:.3f} tanimoto {s['tanimoto']:.3f}")

    def ms(key):
        v = [s[key] for s in stats]
        return float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0

    print("\nTABLE (mean +/- SD over replicas, Cai Table 2 format)")
    hdr = ("N_Xray", "N_clusters", "N_clusters(occ>=0.2)", "Matched", "Recall",
           "Precision", "Tanimoto")
    vals = [(len(xray), 0.0), ms("n_clusters"), ms("n_clusters_occ"), ms("matched"),
            ms("recall"), ms("precision"), ms("tanimoto")]
    for h, (a, b) in zip(hdr, vals):
        print(f"  {h:<22} {a:7.3f} +/- {b:.3f}")

    # ---- consensus sites -------------------------------------------------
    pool, owner = [], []
    for rep in reps:
        for c in per_rep[rep.name]["clusters"]:
            if c["occupancy"] >= opt.min_occupancy:
                pool.append(c["centre"])
                owner.append(rep.name)
    consensus = []
    if len(pool) > 1:
        Z = linkage(pdist(np.array(pool)), method="complete")
        lab = fcluster(Z, t=opt.consensus_cutoff, criterion="distance")
        need = opt.consensus_fraction * len(reps)
        for L in np.unique(lab):
            m = lab == L
            reps_here = {owner[i] for i in np.where(m)[0]}
            if len(reps_here) >= need:
                cen = np.array(pool)[m].mean(axis=0)
                occ = []
                for rep in reps:
                    best = 0.0
                    for c in per_rep[rep.name]["clusters"]:
                        if (c["occupancy"] >= opt.min_occupancy
                                and np.linalg.norm(c["centre"] - cen) <= opt.consensus_cutoff):
                            best = max(best, c["occupancy"])
                    occ.append(best)
                consensus.append({"centre": cen, "replicas": len(reps_here),
                                  "occupancy_mean": float(np.mean(occ)),
                                  "occupancy_sd": float(np.std(occ, ddof=1)),
                                  "occupancy_per_replica": occ,
                                  "inside_gcmc_sphere":
                                      bool(np.linalg.norm(cen - centre) < opt.gcmc_radius),
                                  "distance_from_centre": float(np.linalg.norm(cen - centre))})
    consensus.sort(key=lambda c: -c["occupancy_mean"])
    print(f"\nconsensus sites (>= {opt.consensus_fraction:.0%} of replicas): {len(consensus)}")

    # match consensus sites to crystal waters
    cmatch = {}
    if consensus and len(xray):
        D = np.linalg.norm(np.array([c["centre"] for c in consensus])[:, None, :]
                           - xray[None, :, :], axis=-1)
        uc, ux = set(), set()
        for ci, xi in np.dstack(np.unravel_index(np.argsort(D, axis=None), D.shape))[0]:
            if D[ci, xi] > opt.match_cutoff:
                break
            if ci in uc or xi in ux:
                continue
            cmatch[int(ci)] = {"xray_index": int(xi), "distance": float(D[ci, xi])}
            uc.add(ci)
            ux.add(xi)
    print(f"  matched to crystallographic waters: {len(cmatch)}/{len(consensus)}")

    # ---- per-crystal-water occupancy across replicas ---------------------
    waters = []
    for xi in range(len(xray)):
        occ, dist = [], []
        for rep in reps:
            pr = per_rep[rep.name]["pairs"].get(xi)
            occ.append(pr["occupancy"] if pr else 0.0)
            dist.append(pr["distance"] if pr else np.nan)
        waters.append({
            "index": xi, "resSeq": xinfo[xi]["resSeq"], "b_factor": xinfo[xi]["b"],
            "xray_occupancy": xinfo[xi]["occ"],
            "distance_from_sphere_centre": float(np.linalg.norm(xray[xi] - centre)),
            "inside_gcmc_sphere": bool(np.linalg.norm(xray[xi] - centre) < opt.gcmc_radius),
            "occupancy_per_replica": occ,
            "occupancy_mean": float(np.mean(occ)),
            "occupancy_sd": float(np.std(occ, ddof=1)),
            "matched_replicas": int(sum(1 for o in occ if o > 0)),
            "match_distance_mean": float(np.nanmean(dist)) if np.any(~np.isnan(dist)) else None,
            "match_distance_sd": float(np.nanstd(dist, ddof=1))
            if np.sum(~np.isnan(dist)) > 1 else None,
        })
    waters.sort(key=lambda w: w["b_factor"])

    opt.output.mkdir(parents=True, exist_ok=True)
    out = {
        "region_radius_angstrom": opt.region_radius,
        "gcmc_radius_angstrom": opt.gcmc_radius,
        "sphere_centre_angstrom": centre.tolist(),
        "windows_pooled": widx, "target_b_pooled": bvals,
        "equilibrium_b": meta["equilibrium_b"],
        "crystal": {"file": str(opt.crystal), "chain": opt.chain, "ca_rmsd": rmsd,
                    "residue_offset": off, "superposed_atoms": npairs,
                    "waters_in_region": len(xray)},
        "parameters": {"cluster_cutoff": opt.cluster_cutoff,
                       "consensus_cutoff": opt.consensus_cutoff,
                       "match_cutoff": opt.match_cutoff,
                       "min_occupancy": opt.min_occupancy,
                       "consensus_fraction": opt.consensus_fraction},
        "per_replica": stats,
        "summary_table": {h: {"mean": a, "sd": b} for h, (a, b) in zip(hdr, vals)},
        "consensus_sites": [{**{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                for k, v in c.items()},
                             "matched_xray": cmatch.get(i)} for i, c in enumerate(consensus)],
        "crystal_waters": waters,
    }
    dest = opt.output / f"{opt.prefix}_water_network.json"
    dest.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"\nwrote {dest.relative_to(HERE)}")

    with (opt.output / f"{opt.prefix}_water_network_table.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(stats[0]))
        w.writeheader()
        w.writerows(stats)

    make_figures(out, opt.output, opt.prefix)
    return 0


def make_figures(out, outdir, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    waters = out["crystal_waters"]
    nrep = len(out["per_replica"])
    labels = [f"HOH{w['resSeq']}\n{w['b_factor']:.1f}" for w in waters]
    x = np.arange(len(waters))

    # --- occupancy and matching distance vs B-factor (Cai Fig 5b,c) -------
    fig, ax = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
    cmap = colormaps["tab10"]
    for w, xi in zip(waters, x):
        for r, o in enumerate(w["occupancy_per_replica"]):
            ax[0].scatter(xi + (r - nrep / 2) * 0.045, o, s=16, alpha=0.55,
                          color=cmap(r % 10), zorder=2)
    ax[0].errorbar(x, [w["occupancy_mean"] for w in waters],
                   yerr=[w["occupancy_sd"] for w in waters], fmt="o", color="k",
                   ms=5, capsize=3, lw=1.2, zorder=3, label="mean $\\pm$ SD")
    ax[0].set_ylabel("Occupancy of matched cluster")
    ax[0].set_ylim(-0.05, 1.05)
    ax[0].legend(fontsize=8, loc="upper right")
    ax[0].grid(alpha=0.25, ls=":")
    for w, xi in zip(waters, x):
        d = [v for v in [w["match_distance_mean"]] if v is not None]
        if d:
            ax[1].errorbar([xi], d, yerr=[w["match_distance_sd"] or 0.0], fmt="s",
                           color="C3", ms=5, capsize=3, lw=1.2)
    ax[1].axhline(out["parameters"]["match_cutoff"], ls="--", lw=1, color="grey",
                  label=f"{out['parameters']['match_cutoff']:g} $\\AA$ cutoff")
    ax[1].set_ylabel("Matched distance ($\\AA$)")
    ax[1].set_xlabel("Crystallographic water / B-factor ($\\AA^2$)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.25, ls=":")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels, fontsize=7)
    for a in ax:
        for w, xi in zip(waters, x):
            if w["inside_gcmc_sphere"]:
                a.axvspan(xi - 0.45, xi + 0.45, color="#DCE9F5", zorder=0)
    fig.suptitle("GCMC/MD hydration vs crystallographic waters (6KX6 chain A)\n"
                 "shaded = inside the 4 $\\AA$ GCMC sphere", fontsize=10)
    fig.tight_layout()
    f1 = outdir / f"{prefix}_bfactor_occupancy.png"
    fig.savefig(f1, dpi=160)
    plt.close(fig)

    # --- consensus site occupancies (Cai Fig 7) ---------------------------
    # One bar per crystallographic water (sorted by B-factor, absent bar = no
    # consensus site found), then the strongest unmatched consensus sites.
    cons = out["consensus_sites"]
    n_show = 10
    by_xray = {c["matched_xray"]["xray_index"]: c for c in cons if c["matched_xray"]}
    unmatched = sorted([c for c in cons if not c["matched_xray"]],
                       key=lambda c: -c["occupancy_mean"])[:n_show]

    fig, ax = plt.subplots(figsize=(11, 5.0))
    gr = colormaps["YlGn"]
    bfs = [w["b_factor"] for w in waters]
    vmin, vmax = min(bfs), max(bfs)
    xs, vals, errs, cols, edges, labs = [], [], [], [], [], []
    for i, w in enumerate(waters):
        c = by_xray.get(w["index"])
        xs.append(i)
        vals.append(c["occupancy_mean"] if c else 0.0)
        errs.append(c["occupancy_sd"] if c else 0.0)
        shade = 0.25 + 0.7 * (1 - (w["b_factor"] - vmin) / (vmax - vmin + 1e-9))
        cols.append(gr(shade) if c else "#FFFFFF")
        edges.append("#33691E" if c else "#B0B0B0")
        labs.append(f"HOH{w['resSeq']}\n{w['b_factor']:.1f}")
    off = len(waters) + 1
    for j, c in enumerate(unmatched):
        xs.append(off + j)
        vals.append(c["occupancy_mean"])
        errs.append(c["occupancy_sd"])
        cols.append("#C0504D")
        edges.append("#7F2B28")
        labs.append(f"S{cons.index(c)+1}\nno X-ray")
    ax.bar(xs, vals, yerr=errs, color=cols, edgecolor=edges, capsize=2.5, lw=0.8)
    for i, w in enumerate(waters):
        if w["inside_gcmc_sphere"]:
            ax.axvspan(i - 0.45, i + 0.45, color="#DCE9F5", zorder=0)
    ax.axvline(off - 1, color="grey", lw=1, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels([l.replace("\n", " / ") for l in labs], fontsize=7.5,
                       rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Occupancy (mean over replicas)")
    ax.set_xlabel("Crystallographic water / B-factor ($\\AA^2$)"
                  "                     strongest unmatched consensus sites")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.25, ls=":")
    ax.set_title(f"Consensus hydration sites (present in $\\geq$75% of "
                 f"{len(out['per_replica'])} replicas; {len(cons)} in total, "
                 f"{len(by_xray)} matched)\n"
                 "green = matched (darker = lower B-factor); empty = no consensus site; "
                 "red = site with no crystallographic water; shaded = inside the GCMC sphere",
                 fontsize=9)
    fig.tight_layout()
    f2 = outdir / f"{prefix}_consensus_sites.png"
    fig.savefig(f2, dpi=160)
    plt.close(fig)

    # --- recall / precision per replica -----------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    st = out["per_replica"]
    xr = np.arange(len(st))
    for key, col, mk in (("recall", "C0", "o"), ("precision", "C1", "s"),
                         ("tanimoto", "C2", "^")):
        ax.plot(xr, [s[key] for s in st], mk + "-", color=col, label=key, ms=5, lw=1.1)
    ax.set_xticks(xr)
    ax.set_xticklabels([s["replica"] for s in st], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("value")
    ax.grid(alpha=0.25, ls=":")
    ax.legend(fontsize=8)
    ax.set_title("Agreement with crystallographic waters, per replica", fontsize=10)
    fig.tight_layout()
    f3 = outdir / f"{prefix}_recall_precision.png"
    fig.savefig(f3, dpi=160)
    plt.close(fig)

    for f in (f1, f2, f3):
        print(f"  wrote {f.name}")


if __name__ == "__main__":
    raise SystemExit(main())
