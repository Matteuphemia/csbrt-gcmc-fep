#!/usr/bin/env python3
"""Create a movable, absolute-path manifest for the AN139 -> KL101 pilot."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replica", type=int, choices=(1, 2), default=1)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    opt = options()
    root = Path(__file__).resolve().parent
    scripts = root / "pipeline" / "csbrt" / "src" / "csbrt"
    sys.path.insert(0, str(scripts))
    from pipeline_utils import require_file, validate_recorded_outputs

    preparation_a = root / "data" / "preparations" / "AN139"
    preparation_b = root / "data" / "preparations" / "KL101"
    validate_recorded_outputs(preparation_a / "preparation.complete.json")
    validate_recorded_outputs(preparation_b / "preparation.complete.json")

    bound = root / "data" / "bound_frames" / "AN139" / f"rep{opt.replica}"
    bound_prmtop = require_file(bound / "production-final.prmtop")
    bound_rst7 = require_file(bound / "production-final.rst7")
    output = (opt.output or root / "config" / f"fep_manifest_rep{opt.replica}.tsv").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "edge_index": 0,
        "edge_id": f"AN139_to_KL101_rep{opt.replica}",
        "state_a": "AN139",
        "state_b": "KL101",
        "state_a_preparation": str(preparation_a.resolve()),
        "state_b_preparation": str(preparation_b.resolve()),
        "state_a_bound_prmtop": str(bound_prmtop),
        "state_a_bound_rst7": str(bound_rst7),
    }
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(row), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
    print(f"FEP_MANIFEST={output}")
    print(f"PIPELINE_DIR={scripts.resolve()}")
    print(f"FEP_CONFIG={(scripts / 'somd2_config.yaml').resolve()}")


if __name__ == "__main__":
    main()
