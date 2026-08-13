#!/usr/bin/env python3
"""Reproduce Ludovic's raw-complex-to-solvated Amber preparation.

This is a headless version of System_Prep.ipynb.  It deliberately retains the
two long peptide bonds present in Ludovic's system and preserves the genuine
LYS/ALA chain boundary encoded by terminal hydrogens in the starting PDB.
Run this script in the combined ``cry-loch-babel`` environment.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import random
import subprocess

from openmm.app import Modeller, PDBFile
from pdbfixer import PDBFixer


SOLVENT = {"HOH", "WAT", "TIP", "SOL", "TIP3"}
SKIP_MERGE_RECORDS = {"END", "CONECT", "MASTER"}


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-pdb",
        type=Path,
        default=Path("CRY1AN139_HOLO.pdb"),
        help="Combined raw holo complex containing the 34-atom LIG residue",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("prepared/ludovic_exact"))
    parser.add_argument("--ph", type=float, default=7.4)
    parser.add_argument("--ligand-charge", type=int, default=0)
    parser.add_argument(
        "--pdbfixer-seed",
        type=int,
        default=0,
        help="Seed for reproducible placement/minimisation of missing protein atoms",
    )
    parser.add_argument(
        "--reuse-ligand-parameters",
        action="store_true",
        help="Reuse AN139_protonated.pdb/.mol2/.prepi already present in output-dir",
    )
    return parser.parse_args()


def run(command: list[str], cwd: Path) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def extract(source: Path, ligand_out: Path, protein_out: Path) -> None:
    ligand, protein = [], []
    retained_records = {
        "TER", "END", "REMARK", "HEADER", "TITLE", "CRYST1", "SEQRES",
        "DBREF", "SSBOND",
    }
    for line in source.read_text().splitlines(keepends=True):
        record = line[:6].strip()
        resname = line[17:20].strip() if len(line) > 20 else ""
        if record in {"ATOM", "HETATM"}:
            if resname == "LIG":
                ligand.append(line)
            elif resname not in SOLVENT:
                protein.append(line)
        elif record in retained_records:
            protein.append(line)
    if not ligand:
        raise ValueError(
            f"No LIG residue found in {source}. Ludovic's workflow requires the "
            "combined CRY1AN139_HOLO.pdb; do not infer ligand chemistry from SDF."
        )
    ligand_out.write_text("".join(ligand) + "END\n")
    protein_out.write_text("".join(protein) + "END\n")
    print(f"Extracted {len(ligand)} ligand and {sum(x.startswith(('ATOM', 'HETATM')) for x in protein)} protein atoms")


def merge_pdbs(paths: list[Path], output: Path) -> None:
    lines: list[str] = []
    for path in paths:
        for line in path.read_text().splitlines(keepends=True):
            if line[:6].strip() not in SKIP_MERGE_RECORDS:
                lines.append(line)
        if lines and not lines[-1].startswith("TER"):
            lines.append("TER\n")
    lines.append("END\n")
    output.write_text("".join(lines))


def unique_atom_names(input_pdb: Path, output_pdb: Path) -> None:
    """Make duplicated ligand atom names unique for reliable tLeap mapping."""
    lines = input_pdb.read_text().splitlines(keepends=True)
    names = [
        line[12:16].strip() for line in lines
        if line.startswith(("ATOM", "HETATM"))
    ]
    totals = Counter(names)
    seen: Counter[str] = Counter()
    result: list[str] = []
    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            name = line[12:16].strip()
            if totals[name] > 1:
                seen[name] += 1
                name = f"{name}{seen[name]}"
                if len(name) > 4:
                    raise ValueError(f"Cannot represent unique PDB atom name {name}")
                line = line[:12] + f"{name:>4}" + line[16:]
        result.append(line)
    output_pdb.write_text("".join(result))
    duplicates = {name: count for name, count in totals.items() if count > 1}
    if duplicates:
        print(f"Made duplicated ligand atom names unique for tLeap: {duplicates}")


def residue_atom_names(path: Path) -> tuple[list[tuple[str, str, str]], dict[tuple[str, str, str], list[str]]]:
    order: list[tuple[str, str, str]] = []
    atoms: dict[tuple[str, str, str], list[str]] = {}
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        key = (line[21], line[22:26].strip(), line[26])
        if key not in atoms:
            order.append(key)
            atoms[key] = []
        atoms[key].append(line[12:16].strip())
    return order, atoms


def terminal_boundaries(protonated_protein: Path) -> list[int]:
    """Locate chain boundaries encoded by adjacent C-/N-terminal hydrogens.

    In this input, PDBFixer retains two backbone H atoms on LYS 220 and three
    N-terminal H atoms on ALA 221.  PDBFixer/Open Babel do not reliably retain
    a TER record for that boundary, so it must be restored after H deletion.
    """
    order, atoms = residue_atom_names(protonated_protein)
    boundaries: list[int] = []
    for index, (previous, current) in enumerate(zip(order, order[1:]), start=1):
        previous_names = Counter(atoms[previous])
        current_names = Counter(atoms[current])
        previous_has_terminal_h = previous_names["H"] >= 2
        current_has_n_terminal_h = (
            sum(current_names[name] for name in ("H", "H1", "H2", "H3")) >= 3
            and current_names["H2"] >= 1
            and current_names["H3"] >= 1
        )
        if previous_has_terminal_h and current_has_n_terminal_h:
            boundaries.append(index)
            print(f"Detected terminal boundary after protein residue {index}: {previous} -> {current}")
    return boundaries


def pdb_ter_boundaries(path: Path) -> list[int]:
    """Return internal TER locations as preceding residue indices."""
    residue_index = 0
    last_key: tuple[str, str, str] | None = None
    ter_after: list[int] = []
    for line in path.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            key = (line[21], line[22:26].strip(), line[26])
            if key != last_key:
                residue_index += 1
                last_key = key
        elif line.startswith("TER") and residue_index:
            ter_after.append(residue_index)
    return [index for index in ter_after if index < residue_index]


def restore_boundaries(input_pdb: Path, output_pdb: Path, after_residue_indices: list[int]) -> None:
    boundaries = set(after_residue_indices)
    lines = input_pdb.read_text().splitlines(keepends=True)
    result: list[str] = []
    last_key: tuple[str, str, str] | None = None
    residue_index = 0
    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            key = (line[21], line[22:26].strip(), line[26])
            if key != last_key:
                if (
                    last_key is not None
                    and residue_index in boundaries
                    and (not result or not result[-1].startswith("TER"))
                ):
                    result.append("TER\n")
                residue_index += 1
                last_key = key
        result.append(line)
    output_pdb.write_text("".join(result))


def strip_protein_hydrogens(input_pdb: Path, output_pdb: Path) -> None:
    """Delete every protein hydrogen while preserving OpenMM chain topology."""
    pdb = PDBFile(str(input_pdb))
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.delete(
        atom for atom in modeller.topology.atoms()
        if atom.element is not None and atom.element.symbol == "H"
    )
    with output_pdb.open("w") as handle:
        PDBFile.writeFile(modeller.topology, modeller.positions, handle, keepIds=True)


def fix_mol2_names(mol2: Path, ligand_pdb: Path, output: Path) -> None:
    pdb_names = [
        line[12:16].strip() for line in ligand_pdb.read_text().splitlines()
        if line.startswith(("ATOM", "HETATM"))
    ]
    result, atom_index, in_atoms = [], 0, False
    for line in mol2.read_text().splitlines():
        if line == "@<TRIPOS>ATOM":
            in_atoms = True
            result.append(line)
            continue
        if in_atoms and line.startswith("@<TRIPOS>"):
            in_atoms = False
        if in_atoms and line.strip():
            parts = line.split()
            if atom_index >= len(pdb_names):
                raise RuntimeError("MOL2 has more atoms than the ligand PDB")
            line = line.replace(parts[1], pdb_names[atom_index], 1)
            atom_index += 1
        result.append(line)
    if atom_index != len(pdb_names):
        raise RuntimeError(f"Renamed {atom_index} MOL2 atoms for {len(pdb_names)} PDB atoms")
    output.write_text("\n".join(result) + "\n")


def main() -> None:
    opt = options()
    source = opt.input_pdb.resolve()
    output = opt.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not source.is_file():
        raise FileNotFoundError(source)

    ligand_raw = output / "AN139_raw.pdb"
    protein_raw = output / "CRY1_raw.pdb"
    extract(source, ligand_raw, protein_raw)

    ligand_pdb = output / "AN139_protonated.pdb"
    ligand_pdb_not_unique = output / "AN139_protonated_ludovic_names.pdb"
    ligand_sdf = output / "AN139_protonated.sdf"
    ligand_mol2 = output / "AN139_protonated.mol2"
    ligand_prepi = output / "AN139_protonated.prepi"
    ligand_frcmod = output / "AN139_protonated.frcmod"
    if opt.reuse_ligand_parameters:
        for path in (ligand_pdb, ligand_mol2, ligand_prepi):
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Cannot reuse missing ligand file: {path}")
        print("Reusing completed ligand PDB/MOL2 parameterization")
    else:
        run(["obabel", ligand_raw.name, "-O", ligand_pdb_not_unique.name, "-p", str(opt.ph), "--partialcharge", "gasteiger"], output)
        unique_atom_names(ligand_pdb_not_unique, ligand_pdb)
        run(["obabel", ligand_pdb.name, "-O", ligand_sdf.name], output)
        common = ["-i", ligand_sdf.name, "-fi", "sdf", "-c", "bcc", "-nc", str(opt.ligand_charge), "-rn", "LIG", "-at", "gaff2"]
        run(["antechamber", *common, "-o", ligand_mol2.name, "-fo", "mol2"], output)
        # System_Prep.ipynb performs a second AM1-BCC calculation to retain a
        # PREPI for GRAND's ligand-XML route. Loch does not consume it, but a
        # complete notebook reproduction should still create it.
        run(["antechamber", *common, "-o", ligand_prepi.name, "-fo", "prepi"], output)
    run(["parmchk2", "-i", ligand_mol2.name, "-f", "mol2", "-o", ligand_frcmod.name], output)

    # Retain the standalone ligand topology produced by prepare_ligand.sh in
    # Ludovic's notebook. The full solvated Loch system does not consume it.
    ligand_leap = output / "ligand.in"
    ligand_leap.write_text(
        "source leaprc.gaff2\n"
        "LIG = loadmol2 AN139_protonated.mol2\n"
        "loadamberparams AN139_protonated.frcmod\n"
        "saveamberparm LIG AN139_protonated.prmtop AN139_protonated.inpcrd\n"
        "quit\n"
    )
    run(["tleap", "-f", ligand_leap.name], output)
    for path in (ligand_mol2, ligand_prepi, output / "AN139_protonated.prmtop"):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Ligand preparation did not create {path}")

    fixer = PDBFixer(filename=str(protein_raw))
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms(seed=opt.pdbfixer_seed)
    # OpenMM Modeller uses Python's random module for initial H placement.
    random.seed(opt.pdbfixer_seed)
    fixer.addMissingHydrogens(opt.ph)
    protein_fixed = output / "CRY1_fixed.pdb"
    with protein_fixed.open("w") as handle:
        PDBFile.writeFile(fixer.topology, fixer.positions, handle)
    boundaries = pdb_ter_boundaries(protein_fixed)
    if boundaries:
        print(f"PDBFixer preserved internal TER boundary/boundaries after residues {boundaries}")
    else:
        boundaries = terminal_boundaries(protein_fixed)
        if not boundaries:
            raise RuntimeError("Neither an internal TER nor a terminal-hydrogen boundary was detected")

    # This is the visualisation/checkpoint complex written by the notebook
    # before it removes protein hydrogens for the final tLeap reconstruction.
    merge_pdbs(
        [protein_fixed, ligand_pdb],
        output / "CRY1AN139_ready.pdb",
    )

    # The notebook calls Open Babel here, but it can leave unbound H atoms
    # behind depending on its bond perception. OpenMM deletion is chemically
    # equivalent and guarantees that every protein hydrogen is removed.
    protein_noh_openmm = output / "CRY1_fixed_nohyd_openmm.pdb"
    strip_protein_hydrogens(protein_fixed, protein_noh_openmm)
    protein_noh = output / "CRY1_fixed_nohyd.pdb"
    restore_boundaries(protein_noh_openmm, protein_noh, boundaries)

    complex_noh = output / "CRY1AN139_ready_nohyd.pdb"
    merge_pdbs([protein_noh, ligand_pdb], complex_noh)
    fixed_mol2 = output / "AN139_protonated_fixed.mol2"
    fix_mol2_names(ligand_mol2, ligand_pdb, fixed_mol2)

    leap = output / "solvate.in"
    leap.write_text(
        "source leaprc.protein.ff14SB\n"
        "source leaprc.gaff2\n"
        "source leaprc.water.tip3p\n"
        "loadamberparams AN139_protonated.frcmod\n"
        "LIG = loadmol2 AN139_protonated_fixed.mol2\n"
        "complex = loadpdb CRY1AN139_ready_nohyd.pdb\n"
        "solvateOct complex TIP3PBOX 10.0\n"
        "addionsrand complex Na+ 0\n"
        "addionsrand complex Cl- 0\n"
        "saveamberparm complex CRY1AN139_solvated.prmtop CRY1AN139_solvated.inpcrd\n"
        "savepdb complex CRY1AN139_solvated.pdb\n"
        "quit\n"
    )
    run(["tleap", "-f", leap.name], output)
    for name in ("CRY1AN139_solvated.prmtop", "CRY1AN139_solvated.inpcrd", "CRY1AN139_solvated.pdb"):
        path = output / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Preparation did not create {path}")
        print(f"Created {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
