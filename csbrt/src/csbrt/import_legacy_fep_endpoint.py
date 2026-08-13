#!/usr/bin/env python3
"""Import and audit a legacy AMBER endpoint for the current FEP pipeline.

Older CRY1 preparations predate ``preparation.complete.json`` but already contain
the parameterised ligand and solvated complex needed by ``prepare_fep.py``.  This
utility copies those files into the current endpoint layout only after checking
that the standalone ligand and the complex LIG residue have identical indexed
elements, force-field parameters, and bonded topology.  Atom *names* may differ;
legacy tLEaP/Sire workflows commonly renamed them while preserving atom order.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil

import parmed as pmd
from rdkit import Chem

from pipeline_utils import (
    checkpoint_matches,
    complete_checkpoint,
    implementation_signature,
    require_file,
    sha256,
)


SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ligand-id", required=True)
    parser.add_argument("--ligand-sdf", type=Path, required=True)
    parser.add_argument("--ligand-prmtop", type=Path, required=True)
    parser.add_argument("--ligand-coordinates", type=Path, required=True)
    parser.add_argument("--ligand-mol2", type=Path, required=True)
    parser.add_argument("--ligand-frcmod", type=Path, required=True)
    parser.add_argument("--solvated-prmtop", type=Path, required=True)
    parser.add_argument("--solvated-coordinates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def one_sdf(path: Path) -> Chem.Mol:
    records = [
        molecule
        for molecule in Chem.SDMolSupplier(
            str(require_file(path)), removeHs=False, sanitize=True, strictParsing=True
        )
        if molecule is not None
    ]
    if len(records) != 1:
        raise ValueError(f"Expected one valid SDF record in {path}; found {len(records)}")
    molecule = records[0]
    if molecule.GetNumConformers() != 1 or not molecule.GetConformer().Is3D():
        raise ValueError("Legacy ligand must contain one three-dimensional conformer")
    if not any(atom.GetAtomicNum() == 1 for atom in molecule.GetAtoms()):
        raise ValueError("Legacy ligand has no explicit hydrogens")
    if any(atom.GetAtomicNum() <= 0 for atom in molecule.GetAtoms()):
        raise ValueError("Legacy ligand contains a dummy or unknown element")
    return molecule


def atom_signature(atom) -> tuple[object, ...]:
    return (
        atom.element_name,
        atom.type,
        round(float(atom.charge), 8),
        round(float(atom.mass), 5),
        round(float(atom.epsilon), 8),
        round(float(atom.rmin), 8),
    )


def bond_signature(structure, *, offset: int = 0, residue=None) -> list[tuple[object, ...]]:
    rows = []
    for bond in structure.bonds:
        if residue is not None and not (
            bond.atom1.residue is residue and bond.atom2.residue is residue
        ):
            continue
        rows.append(
            (
                min(bond.atom1.idx, bond.atom2.idx) - offset,
                max(bond.atom1.idx, bond.atom2.idx) - offset,
                round(float(bond.type.k), 6),
                round(float(bond.type.req), 6),
            )
        )
    return sorted(rows)


def audit_embedded_ligand(standalone, complex_system, *, label: str):
    candidates = [residue for residue in complex_system.residues if residue.name == "LIG"]
    if len(candidates) != 1:
        raise ValueError(f"{label} must contain exactly one LIG residue; found {len(candidates)}")
    embedded = candidates[0]
    if len(embedded.atoms) != len(standalone.atoms):
        raise ValueError(
            f"{label} LIG has {len(embedded.atoms)} atoms; standalone has "
            f"{len(standalone.atoms)}"
        )
    if [atom_signature(atom) for atom in embedded.atoms] != [
        atom_signature(atom) for atom in standalone.atoms
    ]:
        raise ValueError(f"{label} LIG indexed atom parameters differ from standalone ligand")
    offset = embedded.atoms[0].idx
    if bond_signature(complex_system, offset=offset, residue=embedded) != bond_signature(
        standalone
    ):
        raise ValueError(f"{label} LIG bonded topology differs from standalone ligand")
    return embedded


def main() -> None:
    opt = options()
    if not SAFE_ID.fullmatch(opt.ligand_id):
        raise ValueError("--ligand-id contains unsafe filename characters")

    sources = {
        "sdf": require_file(opt.ligand_sdf),
        "prmtop": require_file(opt.ligand_prmtop),
        "coordinates": require_file(opt.ligand_coordinates),
        "mol2": require_file(opt.ligand_mol2),
        "frcmod": require_file(opt.ligand_frcmod),
        "solvated_prmtop": require_file(opt.solvated_prmtop),
        "solvated_coordinates": require_file(opt.solvated_coordinates),
    }
    molecule = one_sdf(sources["sdf"])
    standalone = pmd.load_file(str(sources["prmtop"]), xyz=str(sources["coordinates"]))
    if len(standalone.residues) != 1 or standalone.residues[0].name != "LIG":
        raise ValueError("Standalone topology must contain exactly one LIG residue")
    if len(standalone.atoms) != molecule.GetNumAtoms():
        raise ValueError("SDF and standalone topology atom counts differ")
    sdf_elements = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    topology_elements = [atom.element_name for atom in standalone.atoms]
    if sdf_elements != topology_elements:
        raise ValueError("SDF and standalone topology element order differs")

    formal_charge = int(Chem.GetFormalCharge(molecule))
    topology_charge = float(sum(atom.charge for atom in standalone.atoms))
    if abs(topology_charge - formal_charge) > 1.0e-2:
        raise ValueError(
            f"Topology charge {topology_charge:.6f} disagrees with SDF formal "
            f"charge {formal_charge:+d}"
        )
    radicals = sum(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms())
    multiplicity = radicals + 1

    solvated = pmd.load_file(
        str(sources["solvated_prmtop"]), xyz=str(sources["solvated_coordinates"])
    )
    embedded = audit_embedded_ligand(standalone, solvated, label="solvated complex")

    output = opt.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    destinations = {
        "sdf": output / "ligand.sdf",
        "prmtop": output / "ligand.prmtop",
        "coordinates": output / "ligand.rst7",
        "mol2": output / "ligand.mol2",
        "frcmod": output / "ligand.frcmod",
        "solvated_prmtop": output / f"{opt.ligand_id}_solvated.prmtop",
        "solvated_coordinates": output / f"{opt.ligand_id}_solvated.inpcrd",
    }
    marker = output / "preparation.complete.json"
    signature = {
        "import_type": "validated_legacy_amber_endpoint",
        "ligand_id": opt.ligand_id,
        "ligand_charge": formal_charge,
        "ligand_multiplicity": multiplicity,
        "ligand_sha256": sha256(sources["sdf"]),
        "source_sha256": {name: sha256(path) for name, path in sorted(sources.items())},
        "implementation": implementation_signature(
            sources={
                "import_legacy_fep_endpoint.py": Path(__file__),
                "pipeline_utils.py": Path(__file__).with_name("pipeline_utils.py"),
            },
            distributions=("ParmEd", "rdkit"),
        ),
    }
    outputs = list(destinations.values())
    if not opt.force and checkpoint_matches(marker, signature=signature, outputs=outputs):
        print(f"Legacy endpoint checkpoint is valid: {marker}", flush=True)
        return

    marker.unlink(missing_ok=True)
    for name, destination in destinations.items():
        shutil.copy2(sources[name], destination)
    complete_checkpoint(
        marker,
        signature=signature,
        outputs=outputs,
        details={
            "atoms": len(standalone.atoms),
            "heavy_atoms": sum(atom.GetAtomicNum() > 1 for atom in molecule.GetAtoms()),
            "formal_charge": formal_charge,
            "multiplicity": multiplicity,
            "solvated_atoms": len(solvated.atoms),
            "solvated_ligand_residue_index": int(embedded.idx),
            "atom_names_may_differ_but_indexed_parameters_match": True,
        },
    )
    print(f"LEGACY_FEP_ENDPOINT={marker}", flush=True)


if __name__ == "__main__":
    main()
