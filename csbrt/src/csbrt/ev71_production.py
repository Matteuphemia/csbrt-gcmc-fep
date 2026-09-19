#!/usr/bin/env python3
"""Run resumable Ludovic-order Loch production for one prepared EV71 complex."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import mdtraj as md
from openmm import app, unit
import sire as sr

from ev71_loch_common import (
    CsvStateWriter,
    DEFAULT_BATCH_SIZE,
    NUM_GHOSTS,
    PRODUCTION_ATTEMPTS,
    PRODUCTION_CYCLES,
    PRODUCTION_MD_STEPS,
    PRODUCTION_REPORT_INTERVAL,
    SEED,
    TIMESTEP_FS,
    ca_restraints,
    finalise_sampler_system,
    make_dynamics,
    make_sampler,
    physical_water_audit,
    physical_protocol_signature,
    print_sampler,
    randomise_velocities,
    run_with_csv_reports,
    save_physical_system,
    save_system,
    validate_gcmc_handoff,
    validate_physical_water_topology,
    validate_single_ligand,
)
from pipeline_utils import (
    checkpoint_matches,
    complete_checkpoint,
    finite_csv,
    ghost_history,
    implementation_signature,
    invalidate_checkpoint,
    mace_signature,
    sha256,
)


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prmtop", type=Path, required=True)
    parser.add_argument("--rst7", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--ligand-resname", default="LIG")
    parser.add_argument("--gcmc-platform", default="cuda", choices=("cuda", "opencl"))
    parser.add_argument("--md-platform", default="cuda", choices=("cuda", "opencl", "cpu"))
    parser.add_argument("--precision", default="mixed")
    parser.add_argument("--seed", type=int, default=SEED + 3)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--cycles", type=int, default=PRODUCTION_CYCLES)
    parser.add_argument("--md-steps", type=int, default=PRODUCTION_MD_STEPS)
    parser.add_argument("--attempts", type=int, default=PRODUCTION_ATTEMPTS)
    parser.add_argument("--report-interval", type=int, default=PRODUCTION_REPORT_INTERVAL)
    parser.add_argument("--force", action="store_true")
    # MACE surrogate options
    parser.add_argument("--enable-mace-surrogate", "--mace-surrogate", action="store_true")
    parser.add_argument("--mace-model", default="mace-off23-small")
    parser.add_argument("--mace-uq-threshold", type=float, default=0.05)
    parser.add_argument("--mace-device", default="cuda")
    return parser.parse_args()


def dcd_frames(path: Path) -> int:
    with md.open(str(path)) as handle:
        return len(handle)


def main() -> None:
    total_started = time.time()
    opt = options()
    if min(opt.batch_size, opt.cycles, opt.md_steps, opt.attempts, opt.report_interval) < 1:
        raise ValueError("Production counts and intervals must be positive")
    output = opt.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    input_top = opt.prmtop.resolve()
    input_rst = opt.rst7.resolve()
    raw_prefix = output / f"{opt.prefix}-loch-ghosts"
    raw_pdb = raw_prefix.with_suffix(".pdb")
    raw_dcd = output / f"{opt.prefix}-raw.dcd"
    ghost_file = output / f"{opt.prefix}-gcmc-ghosts.txt"
    csv_file = output / f"{opt.prefix}_data_prod.csv"
    final_prefix = output / f"{opt.prefix}-production-final"
    final_top = final_prefix.with_suffix(".prmtop")
    final_rst = final_prefix.with_suffix(".rst7")
    ood_file = output / f"{opt.prefix}-ood_frames.npz"
    outputs = [
        raw_prefix.with_suffix(".prmtop"),
        raw_prefix.with_suffix(".rst7"),
        raw_pdb,
        raw_dcd,
        ghost_file,
        csv_file,
        final_top,
        final_rst,
        final_prefix.with_suffix(".pdb"),
    ]
    if opt.enable_mace_surrogate:
        outputs.append(ood_file)
    signature = {
        "input_prmtop_sha256": sha256(input_top),
        "input_rst7_sha256": sha256(input_rst),
        "prefix": opt.prefix,
        "ligand_resname": opt.ligand_resname,
        "gcmc_platform": opt.gcmc_platform,
        "md_platform": opt.md_platform,
        "precision": opt.precision,
        "seed": opt.seed,
        "batch_size": opt.batch_size,
        "cycles": opt.cycles,
        "md_steps_per_cycle": opt.md_steps,
        "attempts_per_cycle": opt.attempts,
        "report_interval": opt.report_interval,
        "physical_protocol": physical_protocol_signature(),
        "mace_surrogate": (
            mace_signature({
                "enabled": opt.enable_mace_surrogate,
                "model_name": opt.mace_model,
                "uq_force_threshold_ev_per_ang": opt.mace_uq_threshold,
                "device": opt.mace_device,
                "ligand_resname": opt.ligand_resname,
            })
            if opt.enable_mace_surrogate
            else None
        ),
        "implementation": implementation_signature(
            sources={
                "ev71_production.py": Path(__file__),
                "ev71_loch_common.py": Path(__file__).with_name("ev71_loch_common.py"),
                "pipeline_utils.py": Path(__file__).with_name("pipeline_utils.py"),
            },
            distributions=("loch", "sire", "OpenMM", "mdtraj"),
            modules=(
                "sire",
                "openmm",
                "loch._sampler",
                f"loch._platforms._{opt.gcmc_platform.lower()}",
            ),
        ),
    }
    marker = output / "production.complete.json"
    if not opt.force and checkpoint_matches(marker, signature=signature, outputs=outputs):
        equilibrated = sr.load(str(input_top), str(input_rst))
        input_audit = validate_physical_water_topology(
            equilibrated, label="Production checkpoint input"
        )
        input_waters = int(input_audit["water_molecules"])
        raw = sr.load(str(raw_prefix.with_suffix(".prmtop")), str(raw_prefix.with_suffix(".rst7")))
        raw_audit = physical_water_audit(raw)
        if int(raw_audit["water_molecules"]) != input_waters + NUM_GHOSTS:
            raise ValueError("Production checkpoint raw topology is not input + 45 waters")
        if int(raw_audit["zero_interaction_water_count"]) != NUM_GHOSTS:
            raise ValueError("Production checkpoint raw topology does not have 45 ghosts")
        final = sr.load(str(final_top), str(final_rst))
        final_audit = validate_physical_water_topology(
            final, label="Production checkpoint final handoff"
        )
        validate_single_ligand(final, opt.ligand_resname)
        ghosts = ghost_history(ghost_file)
        frames = dcd_frames(raw_dcd)
        if ghosts["lines"] != opt.cycles or frames != opt.cycles:
            raise ValueError("Production checkpoint frame/ghost counts do not match cycles")
        expected_final = input_waters + NUM_GHOSTS - int(ghosts["final_state_zero"])
        if int(final_audit["water_molecules"]) != expected_final:
            raise ValueError("Production checkpoint physical-water arithmetic failed")
        finite_csv(
            csv_file,
            total_steps=opt.cycles * opt.md_steps,
            report_interval=opt.report_interval,
        )
        print(f"Production checkpoint is valid: {marker}", flush=True)
        return
    invalidate_checkpoint(marker)

    equilibrated = sr.load(str(input_top), str(input_rst))
    input_audit = validate_physical_water_topology(
        equilibrated, label="Loch production input"
    )
    validate_single_ligand(equilibrated, opt.ligand_resname)
    input_waters = int(input_audit["water_molecules"])
    sampler = make_sampler(
        equilibrated,
        attempts=opt.attempts,
        batch_size=opt.batch_size,
        seed=opt.seed,
        log_file=output / f"{opt.prefix}-gcmc.log",
        ghost_file=ghost_file,
        ligand_resname=opt.ligand_resname,
        platform=opt.gcmc_platform,
    )
    gcmc_system = sampler.system()
    raw_audit = physical_water_audit(gcmc_system)
    if int(raw_audit["water_molecules"]) != input_waters + NUM_GHOSTS:
        raise ValueError("Production sampler did not append exactly 45 waters")
    if int(raw_audit["zero_interaction_water_count"]) != NUM_GHOSTS:
        raise ValueError("Fresh production topology does not contain exactly 45 inactive ghosts")
    topology_paths = save_system(gcmc_system, raw_prefix)
    topology = app.AmberPrmtopFile(topology_paths["prmtop"]).topology

    dynamics = make_dynamics(
        gcmc_system,
        restraints=ca_restraints(gcmc_system),
        platform=opt.md_platform,
        precision=opt.precision,
    )
    sampler.bind_dynamics(dynamics)
    context = dynamics.context()
    randomise_velocities(context, opt.seed)
    csv = CsvStateWriter(csv_file, context)
    dcd_handle = raw_dcd.open("wb")
    dcd = app.DCDFile(
        dcd_handle,
        topology,
        TIMESTEP_FS * unit.femtoseconds,
        firstStep=0,
        interval=opt.md_steps,
    )
    started = time.time()
    completed = 0

    # MACE surrogate & active learning runtime controllers
    ood_buffer = None
    fallback_ctrl = None
    ensemble_evaluator = None
    atomic_numbers = None
    if opt.enable_mace_surrogate:
        import numpy as np
        from csbrt.mace_surrogate import (
            MACEConfig,
            MACEEnsembleEvaluator,
            MACEUQMonitor,
            PhysicsFallbackController,
            OODBuffer,
            partition_ml_atoms,
        )

        mace_cfg = MACEConfig(
            enabled=True,
            model_name=opt.mace_model,
            uq_force_threshold_ev_per_ang=opt.mace_uq_threshold,
            device=opt.mace_device,
            ligand_resname=opt.ligand_resname,
        )
        try:
            mace_cfg.validate()
        except Exception as err:
            print(f"[MACE surrogate] invalid configuration, disabling: {err}", flush=True)
            mace_cfg = None

        atomic_numbers = [int(atom.element.atomic_number) for atom in topology.atoms()]
        ood_buffer = OODBuffer(ood_file)
        uq_mon = MACEUQMonitor(config=mace_cfg) if mace_cfg is not None else None

        if mace_cfg is not None:
            try:
                ensemble_evaluator = MACEEnsembleEvaluator(mace_cfg)
                ensemble_evaluator.ensure_loaded()
            except Exception as err:
                print(f"[MACE surrogate] UQ disabled at runtime: {err}", flush=True)

        if ensemble_evaluator is not None:
            def _on_ood(f_data: dict) -> None:
                positions_ang = np.asarray(
                    f_data.get("positions", np.zeros((len(atomic_numbers), 3))),
                    dtype=np.float32,
                )
                ood_buffer.add_from_raw(
                    step=f_data.get("step", 0),
                    positions=positions_ang,
                    atomic_numbers=atomic_numbers,
                    uncertainty_force=f_data.get("uq_result", {}).get(
                        "sigma_f_max_ev_per_ang", 0.0
                    ),
                    uncertainty_energy=f_data.get("uq_result", {}).get(
                        "sigma_e_kcal_per_mol", 0.0
                    ),
                    reason=f_data.get("reason", "OOD"),
                    ml_atoms=f_data.get("ml_atoms"),
                )

            fallback_ctrl = PhysicsFallbackController(
                config=mace_cfg,
                uq_monitor=uq_mon,
                on_ood_frame=_on_ood,
            )

    try:
        for cycle in range(opt.cycles):
            md_started = time.time()
            completed = run_with_csv_reports(
                dynamics,
                context,
                opt.md_steps,
                completed,
                opt.report_interval,
                csv,
            )
            md_seconds = time.time() - md_started

            # MACE UQ evaluation and physics fallback monitoring
            if fallback_ctrl is not None and ensemble_evaluator is not None:
                curr_state = context.getState(getPositions=True)
                pos_nm = curr_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                ml_atoms = partition_ml_atoms(
                    topology,
                    positions=pos_nm,
                    ligand_resname=opt.ligand_resname,
                    include_waters=True,
                )
                if ml_atoms:
                    pos_ang = pos_nm * 10.0  # nm -> Angstrom for MACE
                    box_ang = None
                    try:
                        box_ang = (
                            curr_state.getPeriodicBoxVectors(asNumpy=True)
                            .value_in_unit(unit.nanometer)
                            * 10.0
                        )
                    except Exception:
                        box_ang = None
                    try:
                        forces_list, energies_list = ensemble_evaluator.evaluate(
                            pos_ang,
                            atomic_numbers,
                            ml_atoms=ml_atoms,
                            cell=box_ang,
                            pbc=box_ang is not None,
                        )
                    except Exception as err:
                        print(
                            f"[MACE surrogate] committee evaluation failed: {err}",
                            flush=True,
                        )
                    else:
                        uq_res = fallback_ctrl.uq_monitor.compute_from_ensemble_predictions(
                            forces_list,
                            energies_list,
                            ml_atom_indices=list(range(len(ml_atoms))),
                        )
                        mode, w = fallback_ctrl.decide_state(
                            uq_res, {"positions": pos_ang, "ml_atoms": ml_atoms}
                        )
                        fallback_ctrl.apply_to_openmm_context(context, w)

            move_started = time.time()
            sampler.move(context)
            move_seconds = time.time() - move_started
            state = context.getState(
                getPositions=True, getEnergy=True, enforcePeriodicBox=True
            )
            dcd.writeModel(
                state.getPositions(), periodicBoxVectors=state.getPeriodicBoxVectors()
            )
            sampler.write_ghost_residues()
            print_sampler(f"Production {cycle + 1}/{opt.cycles}", sampler)
            print(f"MD={md_seconds:.2f}s Loch={move_seconds:.2f}s", flush=True)
    finally:
        csv.close()
        dcd_handle.close()

    if ood_buffer is not None:
        ood_buffer.save()
        logger_stats = fallback_ctrl.state.to_dict() if fallback_ctrl else {}
        print(f"[MACE surrogate] Active learning OOD buffer saved -> {ood_file}", flush=True)
        print(f"[MACE surrogate] Runtime fallback statistics: {logger_stats}", flush=True)

    final_system = finalise_sampler_system(sampler, context)
    handoff = validate_gcmc_handoff(
        final_system,
        sampler,
        input_water_count=input_waters,
        label="Production finalized handoff",
    )
    save_physical_system(
        final_system,
        final_prefix,
        expected_water_count=handoff["expected_physical_waters"],
    )
    ghosts = ghost_history(ghost_file)
    frames = dcd_frames(raw_dcd)
    csv_audit = finite_csv(
        csv_file,
        total_steps=opt.cycles * opt.md_steps,
        report_interval=opt.report_interval,
    )
    if ghosts["lines"] != opt.cycles or frames != opt.cycles:
        raise ValueError(
            f"Production emitted {frames} frames and {ghosts['lines']} ghost lines "
            f"for {opt.cycles} cycles"
        )
    complete_checkpoint(
        marker,
        signature=signature,
        outputs=outputs,
        details={
            "raw_topology": {
                "physical_plus_buffer_waters": int(raw_audit["water_molecules"]),
                "zero_interaction_waters": int(raw_audit["zero_interaction_water_count"]),
            },
            "handoff": handoff,
            "trajectory_frames": frames,
            "ghost_history": ghosts,
            "csv": csv_audit,
            "wall_seconds": time.time() - started,
            "mace_stats": fallback_ctrl.state.to_dict() if fallback_ctrl else None,
        },
    )
    print(json.dumps(handoff, indent=2), flush=True)
    print(f"Production elapsed: {time.time() - started:.1f} s", flush=True)
    print(f"PRODUCTION_TOTAL_WALL_SECONDS={time.time() - total_started:.3f}", flush=True)


if __name__ == "__main__":
    main()
