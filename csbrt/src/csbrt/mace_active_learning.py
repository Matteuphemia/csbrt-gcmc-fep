#!/usr/bin/env python3
"""Aggregate the campaign's out-of-distribution frames and fine-tune on them.

Worker nodes write flagged frames to their own ``al_buffer/ood_*.npz`` during a
run -- never a shared file, because 52 edges times replicates write
concurrently and online fine-tuning on a worker would race across GPUs. This is
the other half: an aggregator run between generations that merges the buffers,
throws away the redundant conformations, labels what is left, and produces the
next model.

    csbrt-mace-al report   --buffer-dir RUN/al_buffer
    csbrt-mace-al harvest  --buffer-dir RUN/al_buffer --output frames.npz
    csbrt-mace-al label    --buffer-dir RUN/al_buffer --labeller mm \\
                           --prmtop complex.prmtop --output labelled.npz
    csbrt-mace-al finetune --frames labelled.npz --generation 2 --seeds 2

``finetune --seeds N`` trains N models from the same foundation model with
different seeds. That is how a usable committee gets made: MACE refuses a
committee whose members have different cutoff radii, so the foundation releases
cannot be paired with each other, and until a campaign has produced its own
committee the geometry guard is the only detector the runtime has.

**Labelling.** ``--labeller mm`` computes classical single-points on the ML
region, which is the right reference for validating this pipeline end to end
and the wrong one for training: MACE-OFF is fitted to wB97M-D3(BJ)/def2-TZVPPD,
and fine-tuning it on force-field labels would replace its quantum accuracy
with the force field the surrogate exists to avoid. ``finetune`` refuses
MM-labelled data unless ``--allow-mm-labels`` is passed. For a real generation
use ``--labeller command`` with a QM single-point; the contract is in
:func:`command_labeller`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Sequence

import numpy as np


# --------------------------------------------------------------------------- labellers


def command_labeller(template: str, reference_level: str = "qm"):
    """Label frames by shelling out to an external single-point calculation.

    ``template`` is a shell command containing ``{xyz}``, replaced with the
    path to a plain XYZ file of the ML region in Angstroms. The command must
    print one JSON object on stdout:

        {"energy_ev": <float>, "forces_ev_per_ang": [[fx, fy, fz], ...]}

    Forces are in eV/A in the same atom order as the input file. Anything else
    -- a non-zero exit, unparsable output, a wrong force count -- fails that
    frame, which is reported and skipped rather than taking the harvest down.
    """
    from .mace_surrogate.active_learner import element_symbol

    def labeller(positions: np.ndarray, atomic_numbers: Sequence[int]):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.xyz"
            lines = [str(len(atomic_numbers)), "csbrt active-learning frame"]
            for number, xyz in zip(atomic_numbers, positions):
                lines.append(
                    f"{element_symbol(number):<2s} "
                    f"{xyz[0]:14.8f} {xyz[1]:14.8f} {xyz[2]:14.8f}"
                )
            path.write_text("\n".join(lines) + "\n")
            completed = subprocess.run(
                template.format(xyz=str(path)),
                shell=True,
                capture_output=True,
                text=True,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"labeller exited {completed.returncode}: "
                f"{completed.stderr.strip()[-500:]}"
            )
        payload = json.loads(completed.stdout)
        forces = np.asarray(payload["forces_ev_per_ang"], dtype=np.float64)
        return float(payload["energy_ev"]), forces

    labeller.reference_level = reference_level
    return labeller


def mm_labeller(prmtop: Path, ligand_resname: str, ml_atoms: Sequence[int] | None):
    """Classical single-points on the ML region, from an Amber topology."""
    import openmm.app as app
    import openmm.unit as unit

    from .mace_surrogate import (
        MMSubsystemLabeler,
        noninteracting_particles,
        partition_ml_atoms,
    )

    prmtop_file = app.AmberPrmtopFile(str(prmtop))
    system = prmtop_file.createSystem(
        nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False
    )
    if ml_atoms is None:
        ml_atoms = partition_ml_atoms(
            prmtop_file.topology,
            ligand_resname=ligand_resname,
            exclude=noninteracting_particles(system),
        )
    labeller = MMSubsystemLabeler(system, ml_atoms)
    labeller.reference_level = "mm"
    return labeller


# --------------------------------------------------------------------------- commands


def summarise(frames: Sequence[Any]) -> dict[str, Any]:
    if not frames:
        return {"frames": 0}
    sigma = np.asarray([f.uncertainty_force for f in frames], dtype=np.float64)
    sources: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for frame in frames:
        sources[frame.source or "unknown"] = sources.get(frame.source or "unknown", 0) + 1
        key = (frame.reason or "unknown").split(":")[0]
        reasons[key] = reasons.get(key, 0) + 1
    return {
        "frames": len(frames),
        "labelled": sum(1 for f in frames if f.is_labelled),
        "reference_levels": sorted({f.reference_level for f in frames}),
        "sigma_f_ev_per_ang": {
            "min": float(sigma.min()),
            "median": float(np.median(sigma)),
            "max": float(sigma.max()),
        },
        "atom_counts": sorted({int(f.positions.shape[0]) for f in frames}),
        "sources": dict(sorted(sources.items())),
        "trigger_reasons": dict(sorted(reasons.items())),
    }


def command_report(opt) -> int:
    from .mace_surrogate import harvest

    buffer = harvest(opt.buffer_dir)
    summary = summarise(buffer.frames)
    representatives = buffer.cluster_and_deduplicate(opt.rmsd_cutoff)
    summary["distinct_basins"] = len(representatives)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if opt.json:
        Path(opt.json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


def command_harvest(opt) -> int:
    from .mace_surrogate import OODBuffer, harvest

    buffer = harvest(opt.buffer_dir)
    if not len(buffer):
        print(f"No OOD frames found under {opt.buffer_dir}", flush=True)
        return 1
    representatives = buffer.cluster_and_deduplicate(opt.rmsd_cutoff)
    merged = OODBuffer(opt.output, max_frames=len(representatives) or 1)
    merged.extend(representatives)
    merged.save()
    print(
        f"{len(buffer)} frame(s) -> {len(representatives)} distinct basin(s) "
        f"at {opt.rmsd_cutoff} A -> {opt.output}",
        flush=True,
    )
    return 0


def command_label(opt) -> int:
    from .mace_surrogate import OODBuffer, ReferenceLabeler, harvest

    if opt.frames:
        buffer = OODBuffer(opt.frames)
        buffer.load()
        frames = buffer.frames
    else:
        buffer = harvest(opt.buffer_dir)
        frames = buffer.cluster_and_deduplicate(opt.rmsd_cutoff)
    if not frames:
        print("Nothing to label", flush=True)
        return 1

    if opt.labeller == "mm":
        if not opt.prmtop:
            print("--labeller mm needs --prmtop", file=sys.stderr)
            return 2
        evaluator = mm_labeller(opt.prmtop, opt.ligand_resname, None)
    else:
        if not opt.command:
            print("--labeller command needs --command", file=sys.stderr)
            return 2
        evaluator = command_labeller(opt.command)

    labeler = ReferenceLabeler(evaluator, reference_level=evaluator.reference_level)
    labelled = labeler.label_all(frames)
    if not labelled:
        print("Every frame failed to label", file=sys.stderr)
        return 1

    output = OODBuffer(opt.output, max_frames=len(labelled))
    output.extend(labelled)
    output.save()
    print(
        f"Labelled {len(labelled)} of {len(frames)} frame(s) at the "
        f"{evaluator.reference_level} level -> {opt.output}",
        flush=True,
    )
    for frame_id, error in labeler.failures:
        print(f"  failed: {frame_id}: {error}", flush=True)
    return 0


def command_finetune(opt) -> int:
    from .mace_surrogate import MACEFineTuner, OODBuffer

    buffer = OODBuffer(opt.frames)
    if not buffer.load():
        print(f"No frames in {opt.frames}", file=sys.stderr)
        return 1

    work_dir = Path(opt.output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in range(opt.seeds):
        tuner = MACEFineTuner(
            foundation_model=opt.foundation_model,
            learning_rate=opt.learning_rate,
            max_epochs=opt.max_epochs,
            batch_size=opt.batch_size,
            device=opt.device,
            extra_args=["--seed", str(opt.seed_base + seed)],
        )
        checkpoint = work_dir / f"mace_finetuned_gen{opt.generation}_seed{seed}.model"
        result = tuner.fine_tune(
            buffer.frames,
            checkpoint,
            generation=opt.generation,
            allow_mm_labels=opt.allow_mm_labels,
            dry_run=opt.dry_run,
        )
        result["seed"] = opt.seed_base + seed
        results.append(result)
        print(
            f"seed {opt.seed_base + seed}: {result['status']}"
            + (f" ({result.get('reason', '')})" if result["status"] != "completed" else ""),
            flush=True,
        )
        if result["status"] in ("refused", "skipped"):
            break

    manifest = work_dir / f"generation{opt.generation}.json"
    manifest.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    completed = [r for r in results if r["status"] in ("completed", "prepared")]
    if len(completed) >= 2:
        members = [r["checkpoint_path"] for r in completed]
        print(
            "\nCommittee ready. Add to the run config:\n"
            "  mlff:\n"
            f"    model_path: {members[0]}\n"
            "    committee_model_paths:\n"
            + "".join(f"      - {m}\n" for m in members),
            flush=True,
        )
    elif len(completed) == 1:
        print(
            "\nOne model produced. Re-run with --seeds 2 or more to get a "
            "committee; a single model has no force variance to measure.",
            flush=True,
        )
    return 0 if completed else 1


# --------------------------------------------------------------------------- CLI


def options(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_buffer_arguments(target, required=True):
        target.add_argument("--buffer-dir", type=Path, required=required,
                            help="Directory of per-worker ood_*.npz buffers")
        target.add_argument("--rmsd-cutoff", type=float, default=0.5,
                            help="Heavy-atom RMSD for one basin (default: 0.5 A)")

    report = sub.add_parser("report", help="Summarise a campaign's OOD buffers")
    add_buffer_arguments(report)
    report.add_argument("--json", type=Path, default=None)
    report.set_defaults(handler=command_report)

    harvest_parser = sub.add_parser("harvest", help="Merge and deduplicate buffers")
    add_buffer_arguments(harvest_parser)
    harvest_parser.add_argument("--output", type=Path, required=True)
    harvest_parser.set_defaults(handler=command_harvest)

    label = sub.add_parser("label", help="Attach reference energies and forces")
    add_buffer_arguments(label, required=False)
    label.add_argument("--frames", type=Path, default=None,
                       help="A harvested .npz instead of --buffer-dir")
    label.add_argument("--labeller", choices=("mm", "command"), default="command")
    label.add_argument("--command", default=None,
                       help="Shell command with {xyz}; see command_labeller")
    label.add_argument("--prmtop", type=Path, default=None,
                       help="Amber topology for --labeller mm")
    label.add_argument("--ligand-resname", default="LIG")
    label.add_argument("--output", type=Path, required=True)
    label.set_defaults(handler=command_label)

    finetune = sub.add_parser("finetune", help="Fine-tune MACE on labelled frames")
    finetune.add_argument("--frames", type=Path, required=True)
    finetune.add_argument("--output-dir", type=Path, default=Path("mace_models"))
    finetune.add_argument("--generation", type=int, default=1)
    finetune.add_argument("--seeds", type=int, default=2,
                          help="Models to train; 2+ gives a committee (default: 2)")
    finetune.add_argument("--seed-base", type=int, default=20260714)
    finetune.add_argument("--foundation-model", default="small")
    finetune.add_argument("--learning-rate", type=float, default=1.0e-4)
    finetune.add_argument("--max-epochs", type=int, default=20)
    finetune.add_argument("--batch-size", type=int, default=4)
    finetune.add_argument("--device", default="cuda")
    finetune.add_argument("--allow-mm-labels", action="store_true",
                          help="Train on force-field labels. This degrades a "
                               "QM-fitted foundation model; only for validating "
                               "the pipeline or for an MM-level surrogate.")
    finetune.add_argument("--dry-run", action="store_true",
                          help="Write the dataset and the command without running it")
    finetune.set_defaults(handler=command_finetune)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    opt = options(argv)
    return opt.handler(opt)


if __name__ == "__main__":
    sys.exit(main())
