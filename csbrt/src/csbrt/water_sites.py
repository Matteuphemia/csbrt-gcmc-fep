#!/usr/bin/env python3
"""Locate the hydration sites the GCMC found, and compare them to the input structure.

Runs on the window whose Adams value is closest to B_equil, i.e. the rung that
represents equilibrium with bulk water -- that is the only rung whose occupancy
is physically comparable to a structure solved under ordinary conditions.

Ghost waters are non-interacting and must not be counted. CRY1KL101-gcmc-ghosts.txt
records, per cycle, which residues are currently ghosts, so each trajectory frame
is masked with the ghost set belonging to its own cycle. Counting the raw DCD
without that mask would inflate occupancy by up to NUM_GHOSTS.

    python water_sites.py [--radius 4.0] [--cluster-cutoff 1.0]

NOTE ON SCOPE: this compares GCMC hydration against the equilibrated MD input
(inputs/CRY1KL101uvt2.rst7). It is NOT a crystallographic comparison -- no
crystal structure is present in this directory. To compare against one, supply
the deposited model with --crystal <file.pdb>; its waters must already be in the
frame of inputs/CRY1KL101uvt2.rst7 (use scripts/gci_map_centre.py to check).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import mdtraj as md

HERE = Path(__file__).resolve().parent


def ghost_sets(path: Path) -> list[set[int]]:
    """Per-cycle sets of ghost residue indices."""
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        out.append({int(t) for t in line.split(",") if t.strip()} if line else set())
    return out


def sphere_waters(traj, top, centre_nm, radius_nm, ghosts_per_cycle, stride):
    """Water-oxygen positions inside the sphere, per frame, ghosts excluded."""
    o_idx, o_res = [], []
    for res in top.residues:
        if res.name == "HOH":
            for a in res.atoms:
                if a.element.symbol == "O":
                    o_idx.append(a.index)
                    o_res.append(res.index)
    o_idx = np.array(o_idx)
    o_res = np.array(o_res)

    counts, positions = [], []
    for k in range(traj.n_frames):
        cycle = (k + 1) * stride           # frame k was written after this cycle
        ghosts = ghosts_per_cycle[cycle - 1] if cycle - 1 < len(ghosts_per_cycle) else set()
        live = ~np.isin(o_res, list(ghosts)) if ghosts else np.ones(len(o_res), bool)
        xyz = traj.xyz[k, o_idx[live], :]
        d = np.linalg.norm(xyz - centre_nm, axis=1)
        inside = d < radius_nm
        counts.append(int(inside.sum()))
        positions.append(xyz[inside] * 10.0)   # nm -> Angstrom
    return counts, positions


def cluster(points: np.ndarray, cutoff: float) -> list[dict]:
    """Greedy population-ranked clustering; cutoff in Angstrom."""
    remaining = points.copy()
    sites = []
    while len(remaining):
        # seed at the point with the most neighbours
        d = np.linalg.norm(remaining[:, None, :] - remaining[None, :, :], axis=-1)
        seed = int((d < cutoff).sum(axis=1).argmax())
        members = d[seed] < cutoff
        sites.append({"centre": remaining[members].mean(axis=0),
                      "n": int(members.sum()),
                      "spread": float(np.linalg.norm(
                          remaining[members] - remaining[members].mean(axis=0), axis=1).mean())})
        remaining = remaining[~members]
    return sites


AA3 = {"HIE": "HIS", "HID": "HIS", "HIP": "HIS", "CYX": "CYS", "CYM": "CYS",
       "ASH": "ASP", "GLH": "GLU", "LYN": "LYS"}


def parse_pdb(path: Path):
    """Cα atoms, waters (with B-factors) and ligand copies, keyed by PDB chain ID."""
    ca, wat, lig = {}, {}, {}
    for ln in Path(path).read_text().splitlines():
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        name, res, chain = ln[12:16].strip(), ln[17:20].strip(), ln[21]
        try:
            xyz = [float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]
        except ValueError:
            continue
        if res in ("HOH", "WAT", "DOD") and ln[76:78].strip() in ("O", ""):
            if name in ("O", "OW", "OH2"):
                b = float(ln[60:66]) if ln[60:66].strip() else float("nan")
                occ = float(ln[54:60]) if ln[54:60].strip() else 1.0
                wat.setdefault(chain, []).append(
                    {"resSeq": int(ln[22:26]), "xyz": xyz, "b": b, "occ": occ})
        elif res in ("HOH", "WAT", "DOD"):
            continue
        elif ln.startswith("ATOM") and name == "CA":
            ca.setdefault(chain, []).append({"resSeq": int(ln[22:26]),
                                             "name": AA3.get(res, res), "xyz": xyz})
        elif ln.startswith("HETATM"):
            lig.setdefault(chain, {}).setdefault(res, []).append(xyz)
    return ca, wat, lig


def crystal_comparison(path, centre, radius, sites, n_frames, match_cutoff, shell):
    """Superpose each crystal copy onto the simulated monomer and compare waters.

    A deposited structure is in its own crystal frame, so its waters mean nothing
    until they are mapped into the frame of inputs/*.rst7. The mapping reuses
    gci_map_centre.kabsch, whose convention is rotation @ p + translation.
    """
    import sys as _sys
    _sys.path.insert(0, str(HERE / "scripts"))
    from gci_map_centre import kabsch

    sim_top = md.load_prmtop(str(HERE / "inputs" / "CRY1KL101uvt2.prmtop"))
    sim = md.load(str(HERE / "inputs" / "CRY1KL101uvt2.rst7"), top=sim_top)
    sim_ca = {}
    for r in sim_top.residues:
        if r.is_protein:
            for a in r.atoms:
                if a.name == "CA":
                    sim_ca[r.resSeq] = {"name": AA3.get(r.name, r.name),
                                        "xyz": sim.xyz[0, a.index, :] * 10.0}
    sim_lig = np.array([sim.xyz[0, a.index, :] * 10.0 for r in sim_top.residues
                        if r.name == "LIG" for a in r.atoms])

    ca, wat, lig = parse_pdb(path)
    print(f"\nCRYSTAL COMPARISON  {Path(path).name}")
    print(f"  simulated monomer: {len(sim_ca)} C-alpha; deposited chains: "
          f"{', '.join(f'{c}({len(v)})' for c, v in ca.items())}")

    copies = []
    for chain, atoms in ca.items():
        # Residue numbering conventions differ; find the constant offset that makes
        # the residue names agree, and refuse the mapping if they do not.
        best = (0, -1)
        for off in range(-30, 31):
            m = sum(1 for a in atoms
                    if a["resSeq"] + off in sim_ca
                    and sim_ca[a["resSeq"] + off]["name"] == a["name"])
            if m > best[1]:
                best = (off, m)
        off, matched = best
        if matched < 0.8 * len(atoms):
            print(f"  chain {chain}: only {matched}/{len(atoms)} residue names agree "
                  f"(offset {off:+d}) -- refusing to map through an ambiguous alignment")
            continue
        pairs = [(a["xyz"], sim_ca[a["resSeq"] + off]["xyz"]) for a in atoms
                 if a["resSeq"] + off in sim_ca
                 and sim_ca[a["resSeq"] + off]["name"] == a["name"]]
        R, t, rmsd = kabsch(np.array([p for p, _ in pairs]),
                            np.array([q for _, q in pairs]))
        xf = lambda p: (R @ np.asarray(p, dtype=float).T).T + t

        ligd = None
        for resn, xyzs in lig.get(chain, {}).items():
            d = float(np.linalg.norm(xf(xyzs).mean(axis=0) - sim_lig.mean(axis=0)))
            ligd = (resn, d) if ligd is None or d < ligd[1] else ligd
        w = wat.get(chain, [])
        wx = xf([x["xyz"] for x in w]) if w else np.empty((0, 3))
        dist = np.linalg.norm(wx - centre, axis=1) if len(wx) else np.array([])
        inside = dist < radius
        near = dist < shell
        print(f"  chain {chain}: offset {off:+d}, {len(pairs)} C-alpha superposed, "
              f"RMSD {rmsd:.2f} A"
              + (f"; ligand {ligd[0]} lands {ligd[1]:.2f} A from the simulated LIG"
                 if ligd else ""))
        print(f"    {len(w)} deposited waters -> {int(inside.sum())} inside the "
              f"{radius:g} A sphere, {int(near.sum())} within {shell:g} A")

        rows = []
        for k in np.argsort(dist)[:int(near.sum())]:
            wq, dc = w[int(k)], float(dist[int(k)])
            ds = [float(np.linalg.norm(wx[int(k)] - s["centre"])) for s in sites]
            j = int(np.argmin(ds)) if ds else -1
            rows.append({"resSeq": wq["resSeq"], "b_factor": wq["b"],
                         "occupancy_xray": wq["occ"],
                         "distance_from_sphere_centre_A": dc,
                         "inside_sphere": bool(dc < radius),
                         "nearest_gcmc_site": (j + 1) if j >= 0 else None,
                         "distance_to_site_A": ds[j] if j >= 0 else None,
                         "site_occupancy": sites[j]["n"] / n_frames if j >= 0 else None,
                         "matched": bool(j >= 0 and ds[j] <= match_cutoff)})
            tag = "inside" if dc < radius else f"{dc:.1f} A out"
            print(f"      HOH {wq['resSeq']:>4} (B={wq['b']:.1f}, {tag}): nearest GCMC site "
                  f"{j+1} at {ds[j]:.2f} A, occupancy {sites[j]['n']/n_frames:.2f}"
                  f"{'  MATCHED' if ds[j] <= match_cutoff else ''}")
        ins = [r for r in rows if r["inside_sphere"]]
        rec = (sum(r["matched"] for r in ins) / len(ins)) if ins else float("nan")
        print(f"    recall inside sphere (<= {match_cutoff:g} A): "
              f"{sum(r['matched'] for r in ins)}/{len(ins)}")
        copies.append({"chain": chain, "residue_offset": off,
                       "superposed_atoms": len(pairs), "ca_rmsd_angstrom": rmsd,
                       "ligand_resname": ligd[0] if ligd else None,
                       "ligand_centroid_offset_angstrom": ligd[1] if ligd else None,
                       "deposited_waters": len(w),
                       "waters_inside_sphere": int(inside.sum()),
                       f"waters_within_{shell:g}A": int(near.sum()),
                       "recall_inside_sphere": rec, "waters": rows})
    return {"file": str(path), "match_cutoff_angstrom": match_cutoff,
            "shell_angstrom": shell, "copies": copies}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-root", type=Path, default=HERE / "replicas")
    p.add_argument("--prefix", default="CRY1KL101")
    p.add_argument("--radius", type=float, default=4.0)
    p.add_argument("--cluster-cutoff", type=float, default=1.0)
    p.add_argument("--crystal", type=Path, default=None,
                   help="Deposited structure (PDB). It is superposed onto the simulated "
                        "monomer automatically; it need NOT already be in the input frame.")
    p.add_argument("--match-cutoff", type=float, default=2.0,
                   help="Crystal-water to GCMC-site matching distance (default 2.0 A, "
                        "as used in the CRY1/AN139 study).")
    p.add_argument("--shell", type=float, default=6.0,
                   help="Also report deposited waters out to this radius, for context.")
    p.add_argument("--output", type=Path, default=HERE / "analysis" / "combined")
    opt = p.parse_args()

    reps = sorted((d for d in opt.run_root.glob("rep*") if d.is_dir()),
                  key=lambda d: int("".join(c for c in d.name if c.isdigit()) or 0))

    # pick the rung closest to B_equil
    probe = sorted(reps[0].glob("window_*"))[0] / f"{opt.prefix}_titration.json"
    b_equil = json.loads(probe.read_text())["equilibrium_b"]
    best, best_d = None, 1e9
    for w in sorted(reps[0].glob("window_*")):
        b = json.loads((w / f"{opt.prefix}_titration.json").read_text())["target_b"]
        if abs(b - b_equil) < best_d:
            best, best_d = w.name, abs(b - b_equil)
    meta = json.loads((reps[0] / best / f"{opt.prefix}_titration.json").read_text())
    centre = np.array(meta["sphere_centre_angstrom"], dtype=float)
    stride = meta["trajectory_stride"]

    print(f"B_equil = {b_equil:.4f}; nearest rung = {best} (B = {meta['target_b']})")
    print(f"sphere centre {centre} A, radius {opt.radius} A, "
          f"centre drift {meta['sphere_centre_drift_angstrom']:.2e} A\n")

    all_pos, per_rep = [], {}
    for rep in reps:
        wd = rep / best
        top = md.load_prmtop(str(wd / f"{opt.prefix}-loch-ghosts.prmtop"))
        traj = md.load_dcd(str(wd / f"{opt.prefix}-raw.dcd"), top=top)
        gs = ghost_sets(wd / f"{opt.prefix}-gcmc-ghosts.txt")
        counts, pos = sphere_waters(traj, top, centre / 10.0, opt.radius / 10.0, gs, stride)
        per_rep[rep.name] = counts
        all_pos.append(np.vstack(pos) if any(len(x) for x in pos) else np.empty((0, 3)))
        print(f"  {rep.name}: {traj.n_frames} frames, "
              f"sphere occupancy mean {np.mean(counts):.2f} (range {min(counts)}-{max(counts)})")

    pooled = np.vstack(all_pos)
    n_frames = sum(len(v) for v in per_rep.values())
    print(f"\npooled: {len(pooled)} water positions over {n_frames} frames "
          f"({len(reps)} replicas x {len(per_rep[reps[0].name])} frames)")
    print(f"ensemble mean occupancy: "
          f"{np.mean([c for v in per_rep.values() for c in v]):.2f}")

    sites = [s for s in cluster(pooled, opt.cluster_cutoff) if s["n"] >= 0.10 * n_frames]
    print(f"\nHYDRATION SITES (>=10% occupancy, {opt.cluster_cutoff} A clustering)")
    print(f"{'site':>4} {'occupancy':>10} {'spread_A':>9}   position (A)")
    for i, s in enumerate(sites, 1):
        c = s["centre"]
        print(f"{i:>4} {s['n']/n_frames:>9.2f}  {s['spread']:>8.2f}   "
              f"[{c[0]:7.3f} {c[1]:7.3f} {c[2]:7.3f}]  {np.linalg.norm(c-centre):5.2f} A from centre")

    # --- reference: the equilibrated input structure ----------------------
    ref_top = md.load_prmtop(str(HERE / "inputs" / f"{opt.prefix}uvt2.prmtop"))
    ref = md.load(str(HERE / "inputs" / f"{opt.prefix}uvt2.rst7"), top=ref_top)
    ro = np.array([a.index for r in ref_top.residues if r.name == "HOH"
                   for a in r.atoms if a.element.symbol == "O"])
    rxyz = ref.xyz[0, ro, :] * 10.0
    inside = np.linalg.norm(rxyz - centre, axis=1) < opt.radius
    ref_w = rxyz[inside]
    print(f"\nINPUT EQUILIBRATED STRUCTURE: {len(ref_w)} waters inside the sphere")
    rows = []
    for i, w in enumerate(ref_w, 1):
        if sites:
            d = [np.linalg.norm(w - s["centre"]) for s in sites]
            j = int(np.argmin(d))
            rows.append({"ref_water": i, "nearest_site": j + 1,
                         "distance_A": d[j], "site_occupancy": sites[j]["n"] / n_frames})
            print(f"  water {i}: nearest GCMC site {j+1} at {d[j]:.2f} A "
                  f"(site occupancy {sites[j]['n']/n_frames:.2f})")
    recovered = sum(1 for r in rows if r["distance_A"] <= 1.5)
    print(f"  recovered within 1.5 A: {recovered}/{len(ref_w)}")

    crystal_out = None
    if opt.crystal:
        crystal_out = crystal_comparison(opt.crystal, centre, opt.radius, sites,
                                         n_frames, opt.match_cutoff, opt.shell)

    out = {
        "window": best, "target_b": meta["target_b"], "equilibrium_b": b_equil,
        "sphere_centre_angstrom": centre.tolist(), "radius_angstrom": opt.radius,
        "cluster_cutoff_angstrom": opt.cluster_cutoff,
        "frames_total": n_frames, "replicas": [r.name for r in reps],
        "occupancy_per_replica": {k: {"mean": float(np.mean(v)), "min": min(v), "max": max(v)}
                                  for k, v in per_rep.items()},
        "ensemble_mean_occupancy": float(np.mean([c for v in per_rep.values() for c in v])),
        "sites": [{"index": i + 1, "occupancy": s["n"] / n_frames,
                   "centre_angstrom": s["centre"].tolist(),
                   "spread_angstrom": s["spread"],
                   "distance_from_sphere_centre_angstrom": float(np.linalg.norm(s["centre"] - centre))}
                  for i, s in enumerate(sites)],
        "input_structure_waters_in_sphere": int(len(ref_w)),
        "input_water_to_site": rows,
        "input_waters_recovered_within_1.5A": recovered,
        "crystal_structure_supplied": bool(opt.crystal),
        "crystal_comparison": crystal_out,
    }
    opt.output.mkdir(parents=True, exist_ok=True)
    dest = opt.output / f"{opt.prefix}_water_sites.json"
    # numpy float32 is not JSON-serialisable; coerce on the way out.
    dest.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"\nwrote {dest.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
