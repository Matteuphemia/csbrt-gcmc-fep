#!/usr/bin/env python3
"""Convert grand GCI SLURM output into the Loch pipeline's window format.

The original GCI implementation never wrote an occupancy file. ``grand`` only
logged prose lines such as

    400 move(s) completed (3 accepted (0.7500 %)). Current N = 0. Average N = 0.445

and the analysis notebook recovered the titration observable by regular
expression over scheduler stdout. To compare the two implementations fairly the
same analysis code has to run on both datasets, so this reads those ``.out``
files and materialises, per window, the three artifacts ``gci_analyse.py``
expects: a titration CSV, a metadata JSON, and a verified checkpoint.

It is read-only with respect to the historical data.

Fields grand did not record are written as -1 (not measured) rather than as a
plausible value: the ghost-pool occupancy, and the insertion/deletion split.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

from pipeline_utils import complete_checkpoint, require_file, sha256, write_json_atomic

# Physical constants of the original study.
MU_HYDRATION = -6.09          # kcal/mol, excess chemical potential of bulk TIP3P
STANDARD_VOLUME = 30.345      # A^3
TEMPERATURE_K = 300.0
# Exact kT, as Loch computes it. The original script used a truncated 0.5961
# when it printed its B value, which is why the header B and the B recomputed
# here differ in the fourth decimal.
KT_KCAL_PER_MOL = 0.5961619502868069
TIMESTEP_FS = 2.0
NOT_MEASURED = -1

RE_MU = re.compile(r"mu\s*\(excess\)\s*:\s*([-\d.]+)\s*kcal")
RE_B = re.compile(r"B value\s*:\s*([-\d.]+)")
RE_CENTRE = re.compile(r"Sphere center\s*:\s*\[([-\d.,\s]+)\]")
RE_VOLUME = re.compile(r"V_sphere\s*:\s*([\d.]+)")
RE_GHOSTS = re.compile(r"N_ghost\s*:\s*(\d+)")
RE_NANOSECONDS = re.compile(r"Total simulation time\s*:\s*([\d.]+)\s*ns")
RE_TOTAL_MOVES = re.compile(r"Total GCMC moves\s*:\s*(\d+)")
RE_RECORD = re.compile(
    r"(\d+)\s+move\(s\) completed\s+\((\d+)\s+accepted\s+\(([\d.]+)\s*%\)\)\."
    r"\s+Current N\s*=\s*(\d+).*Average N\s*=\s*([\d.]+)"
)


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="CRY1AN139")
    parser.add_argument("--pattern", default="titration_*.out")
    return parser.parse_args()


def parse_out_file(path: Path) -> dict:
    text = require_file(path).read_text(errors="replace")

    def one(pattern: re.Pattern, name: str, cast=float):
        match = pattern.search(text)
        if match is None:
            raise ValueError(f"{path.name}: could not find {name}")
        return cast(match.group(1))

    centre_match = RE_CENTRE.search(text)
    if centre_match is None:
        raise ValueError(f"{path.name}: could not find the sphere centre")
    centre = [float(value) for value in centre_match.group(1).split(",")]

    records = [
        {
            "move": int(m.group(1)),
            "accepted": int(m.group(2)),
            "acceptance_percent": float(m.group(3)),
            "current_n": int(m.group(4)),
            "average_n": float(m.group(5)),
        }
        for m in RE_RECORD.finditer(text)
    ]
    if not records:
        raise ValueError(f"{path.name}: no GCMC records found")

    volume = one(RE_VOLUME, "V_sphere")
    radius = (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0)
    total_moves = one(RE_TOTAL_MOVES, "total GCMC moves", int)
    nanoseconds = one(RE_NANOSECONDS, "total simulation time")

    intervals = {
        records[i + 1]["move"] - records[i]["move"] for i in range(len(records) - 1)
    }
    if len(intervals) != 1:
        raise ValueError(f"{path.name}: irregular checkpoint spacing {sorted(intervals)}")
    attempts_per_cycle = intervals.pop()
    cycles = len(records)
    if cycles * attempts_per_cycle != total_moves:
        raise ValueError(
            f"{path.name}: {cycles} records x {attempts_per_cycle} != {total_moves} moves"
        )
    md_steps_per_cycle = int(round(nanoseconds * 1.0e6 / TIMESTEP_FS / cycles))

    mu = one(RE_MU, "mu")
    return {
        "mu": mu,
        "b_header": one(RE_B, "B value"),
        "centre": centre,
        "radius": radius,
        "sphere_volume": volume,
        "num_ghosts": one(RE_GHOSTS, "N_ghost", int),
        "cycles": cycles,
        "attempts_per_cycle": attempts_per_cycle,
        "md_steps_per_cycle": md_steps_per_cycle,
        "total_moves": total_moves,
        "nanoseconds": nanoseconds,
        "records": records,
        "source": path,
    }


def main() -> None:
    opt = options()
    output = opt.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    sources = sorted(opt.input_dir.resolve().glob(opt.pattern))
    if not sources:
        raise ValueError(f"No files matching {opt.pattern} in {opt.input_dir}")
    parsed = [parse_out_file(path) for path in sources]

    # Recompute B from mu with the exact kT and the radius implied by the logged
    # sphere volume, so the converted dataset is internally consistent with the
    # analysis. The header value is kept alongside for reference.
    for window in parsed:
        window["target_b"] = (
            window["mu"] / KT_KCAL_PER_MOL
            + math.log(window["sphere_volume"] / STANDARD_VOLUME)
        )
    parsed.sort(key=lambda window: window["target_b"])

    print(f"{len(parsed)} windows from {opt.input_dir}")
    first = parsed[0]
    print(
        f"  radius {first['radius']:.3f} A | {first['cycles']} cycles x "
        f"{first['attempts_per_cycle']} attempts | "
        f"{first['md_steps_per_cycle']} MD steps/cycle | "
        f"{first['nanoseconds']:g} ns"
    )
    drift = max(abs(w["target_b"] - w["b_header"]) for w in parsed)
    print(f"  max |B_recomputed - B_header| = {drift:.4f} (truncated kT in the original)")

    for index, window in enumerate(parsed):
        directory = output / f"window_{index:02d}_B{window['target_b']:+.4f}"
        directory.mkdir(parents=True, exist_ok=True)
        csv_path = directory / f"{opt.prefix}_titration.csv"
        json_path = directory / f"{opt.prefix}_titration.json"

        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "step",
                    "cycle",
                    "md_steps_completed",
                    "sphere_waters",
                    "accepted_moves",
                    "accepted_attempts",
                    "insertions",
                    "deletions",
                    "ghost_pool",
                ]
            )
            for cycle, record in enumerate(window["records"], start=1):
                writer.writerow(
                    [
                        record["move"],
                        cycle,
                        cycle * window["md_steps_per_cycle"],
                        record["current_n"],
                        record["accepted"],
                        NOT_MEASURED,
                        NOT_MEASURED,
                        NOT_MEASURED,
                        NOT_MEASURED,
                    ]
                )

        metadata = {
            "stage": "gci_window",
            "provenance": "converted from grand SLURM output",
            "source_file": window["source"].name,
            "source_sha256": sha256(window["source"]),
            "window_index": index,
            "prefix": opt.prefix,
            "target_b": window["target_b"],
            "target_b_from_grand_header": window["b_header"],
            "adams_value_from_loch": None,
            "mu_kcal_per_mol": window["mu"],
            "adams_shift": 0.0,
            "radius_angstrom": window["radius"],
            "sphere_volume_angstrom3": window["sphere_volume"],
            "standard_volume_angstrom3": STANDARD_VOLUME,
            "bulk_sampling_probability": 0.0,
            "temperature_K": TEMPERATURE_K,
            "kt_kcal_per_mol": KT_KCAL_PER_MOL,
            "mu_hydration_kcal_per_mol": MU_HYDRATION,
            "sphere_centre_angstrom": window["centre"],
            "reference": "resname DUM and atomname SPH",
            "num_ghost_waters": window["num_ghosts"],
            "minimum_ghost_pool": NOT_MEASURED,
            "cycles": window["cycles"],
            "attempts_per_cycle": window["attempts_per_cycle"],
            "md_steps_per_cycle": window["md_steps_per_cycle"],
            "total_gcmc_attempts": window["total_moves"],
            "total_md_ps": window["nanoseconds"] * 1000.0,
            # grand built its system from OpenMM XML force fields, so there is no
            # AMBER topology to hash. A constant keeps the consistency check
            # meaningful across these windows without inventing a file hash.
            "input_prmtop_sha256": "grand-openmm-xml-no-prmtop",
            "input_rst7_sha256": "grand-openmm-xml-no-rst7",
        }
        write_json_atomic(json_path, metadata)

        marker = directory / "gci_window.complete.json"
        complete_checkpoint(
            marker,
            signature={
                "stage": "gci_window",
                "provenance": "converted from grand SLURM output",
                "source_sha256": metadata["source_sha256"],
                "window_index": index,
                "target_b": window["target_b"],
            },
            outputs=[csv_path, json_path],
            details={"converted_from": window["source"].name},
        )

    print(f"Wrote {len(parsed)} windows to {output}")


if __name__ == "__main__":
    main()
