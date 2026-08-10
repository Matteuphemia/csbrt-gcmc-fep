"""Cell bodies for notebook 06 sections 3b/3c.

Kept in a real file rather than inline in the builder so that backslash escapes
survive verbatim. `build_06_fep.py` imports MD_3B / CODE_3B / CODE_3B2 / MD_3C /
CODE_3C from here.
"""

MD_3B = r"""
### 3b · End-to-end check — run the fixed `prepare_fep.py` on a null perturbation

A null perturbation (state A → state A) is the guaranteed-to-merge minimal case, so
it isolates the frame handling from mapping difficulty. This runs the **real**
script on the retained local endpoint preparation, then checks the geometry of both
streams it produced.

Needs `tleap` and BioSimSpace in the running kernel and takes about a minute; it
reports and skips otherwise. Output goes to a temporary directory.
""".strip()


CODE_3B = r'''
# ── Run the real prepare_fep.py on a null edge ───────────────────────────────
import os
import sys
import shutil
import subprocess
import tempfile

NULL_EDGE = None
prep_dir = ENDPOINT_RUN / "preparation"
script = FEP_SCRIPTS / "prepare_fep.py"

# Resolve tleap the same way prepare_fep.py does: PATH first, then the sibling of
# the running interpreter. A Jupyter kernel usually inherits the launching shell's
# PATH rather than its own environment's bin/, so the sibling lookup is the one that
# actually finds it here.
env_bin = Path(sys.executable).resolve().parent
tleap_path = shutil.which("tleap") or (
    str(env_bin / "tleap") if (env_bin / "tleap").is_file() else None
)

if not (script.is_file() and prep_dir.is_dir()):
    print(f"skipped: need {script} and {prep_dir}")
elif tleap_path is None:
    print(f"skipped: tleap not found on PATH or beside {sys.executable}")
else:
    print(f"tleap: {tleap_path}")
    NULL_EDGE = Path(tempfile.mkdtemp(prefix="nb06_null_")) / "null"
    cmd = [
        sys.executable, "-u", str(script),
        "--state-a-preparation", str(prep_dir),
        "--state-b-preparation", str(prep_dir),
        "--output-dir", str(NULL_EDGE),
        "--edge-id", "nb06_null",
        # Passed deliberately: it must be accepted and IGNORED, so the existing
        # wrappers keep working without silently changing behaviour.
        "--align-to-bound-pose",
    ]
    print(" ".join(cmd[1:]))
    print()
    # Put the interpreter's own bin/ on PATH so the child's tleap lookup succeeds.
    child_env = dict(os.environ)
    child_env["PATH"] = f"{env_bin}{os.pathsep}{child_env.get('PATH', '')}"
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(FEP_SCRIPTS), env=child_env
    )
    for line in result.stdout.splitlines():
        if line.startswith("[prepare_fep]") or line.startswith("FEP_PREPARATION="):
            print(line)
    if result.returncode != 0:
        print(f"FAILED (exit {result.returncode})")
        print(result.stdout[-1200:])
        print(result.stderr[-1200:])
        NULL_EDGE = None
    else:
        marker = json.loads((NULL_EDGE / "fep_preparation.complete.json").read_text())
        fc = marker["frame_check"]
        print()
        print("recorded frame_check:")
        print(f"    {'tolerance_angstrom':<38} {fc['tolerance_angstrom']}")
        print(f"    {'bound_centroid_offset_angstrom':<38} "
              f"{fc['bound_centroid_offset_angstrom']:.3e}")
        print(f"    {'free_centroid_offset_angstrom':<38} "
              f"{fc['free_centroid_offset_angstrom']:.3e}")
        print(f"    {'bound_to_free_separation_angstrom':<38} "
              f"{fc['bound_to_free_separation_angstrom']:.2f}   "
              "(expected to be large: separate boxes)")
        print(f"    {'bound_alignment':<38} {marker['bound_alignment']}")
        print(f"    {'align_to_bound_pose_flag_ignored':<38} "
              f"{marker['align_to_bound_pose_flag_ignored']}")
        assert fc["bound_centroid_offset_angstrom"] < 1e-6
        assert fc["free_centroid_offset_angstrom"] < 1e-6
        print()
        print("  both legs aligned to machine precision; the deprecated flag was ignored")
'''.strip()


CODE_3B2 = r'''
# ── The physically meaningful check: no steric clash in either stream ────────
# This is what distinguishes a harmless rigid translation from a real overlap, and
# it is the measurement that exposed the free-leg defect: 0.73 A closest contact
# with 47 of 62 atoms inside 2.0 A, before the fix.
if NULL_EDGE is None:
    print("skipped: no null-edge run available")
else:
    import BioSimSpace as BSS

    def coordinates(molecule):
        if molecule.isPerturbable():
            molecule = molecule._toRegularMolecule(is_lambda1=False)
        out = []
        for atom in molecule.getAtoms():
            c = atom.coordinates()
            out.append([c.x().angstroms().value(), c.y().angstroms().value(),
                        c.z().angstroms().value()])
        return np.asarray(out)

    print(f"{'leg':<7}{'molecules':>11}{'waters':>8}{'lig atoms':>11}"
          f"{'min lig-water O':>17}{'atoms < 2.0 A':>15}")
    print("-" * 70)
    for leg in ("bound", "free"):
        system = BSS.Stream.load(str(NULL_EDGE / f"nb06_null_{leg}.bss"))
        perturbable = [i for i, m in enumerate(system.getMolecules()) if m.isPerturbable()]
        assert len(perturbable) == 1, f"{leg} leg lost its perturbable molecule"
        ligand = coordinates(system.getMolecule(perturbable[0]))
        oxygens = np.asarray([
            coordinates(m)[0] for m in system.getMolecules()
            if not m.isPerturbable() and m.nAtoms() == 3
            and m.getResidues()[0].name() in ("WAT", "HOH")
        ])
        distances = np.linalg.norm(ligand[:, None, :] - oxygens[None, :, :], axis=2)
        closest = float(distances.min())
        clashes = int((distances.min(axis=1) < 2.0).sum())
        print(f"{leg:<7}{system.nMolecules():>11,}{len(oxygens):>8,}{len(ligand):>11}"
              f"{closest:>17.2f}{clashes:>15}")
        assert clashes == 0, f"{leg} leg ligand overlaps water"
        assert closest > 2.5, f"{leg} leg closest contact {closest:.2f} A is too short"

    print()
    print("  Both legs sit in their own cavity with no overlap. Before the fix the free")
    print("  leg read 0.73 A closest contact with 47 of 62 atoms inside 2.0 A -- a hard")
    print("  clash that surfaces as a minimisation failure and gets blamed on the mapping.")
'''.strip()


MD_3C = r"""
### 3c · The guard must reject all three historical mistakes

A guard that never fires is not a guard. Each case below is a frame mistake that has
actually occurred in this pipeline.
""".strip()


CODE_3C = r'''
# ── Negative and positive tests of verify_leg_frame ──────────────────────────
import math

if NULL_EDGE is None:
    print("skipped: no null-edge run available")
else:
    sys.path.insert(0, str(FEP_SCRIPTS))
    import BioSimSpace as BSS
    import prepare_fep as P

    lig_a = P.ligand_from_preparation(prep_dir)
    lig_b = P.ligand_from_preparation(prep_dir)
    ident = {i: i for i in range(lig_a.nAtoms())}

    bound_sys = BSS.IO.readMolecules([
        str(prep_dir / f"{PREFIX}_solvated.prmtop"),
        str(prep_dir / f"{PREFIX}_solvated.inpcrd"),
    ])
    bi = P.ligand_index(bound_sys)
    bound_target = P.centroid(bound_sys.getMolecule(bi))

    # The free box built by the run in 3b: tLEaP's re-centred frame, same ligand.
    free_sys = BSS.IO.readMolecules([
        str(NULL_EDGE / "state_a_free.prmtop"), str(NULL_EDGE / "state_a_free.rst7")
    ])
    fi = P.ligand_index(free_sys)
    free_target = P.centroid(free_sys.getMolecule(fi))
    input_centroid = P.centroid(lig_a)
    print(f"ligand input centroid {np.round(input_centroid, 2)}")
    print(f"bound target centroid {np.round(bound_target, 2)}   "
          f"({math.dist(bound_target, input_centroid):.2f} A from input)")
    print(f"free  target centroid {np.round(free_target, 2)}   "
          f"({math.dist(free_target, input_centroid):.2f} A from input -- tLEaP re-centred)")

    unaligned = BSS.Align.merge(lig_a, BSS.Align.rmsdAlign(lig_b, lig_a, ident), ident)
    bound_merge = P.build_leg_merge("bound", bound_sys, bi, lig_a, lig_b, ident, ident,
                                    allow_ring_breaking=False, allow_ring_size_change=False)
    cases = [
        ("bound", unaligned, bound_target,
         "unaligned merge on the BOUND leg (old flag off: outside the pocket)"),
        ("free", unaligned, free_target,
         "unaligned merge on the FREE leg  (the offset that was missed)"),
        ("free", bound_merge, free_target,
         "bound merge on the FREE leg      (old flag on: outside the water cavity)"),
    ]

    print()
    print("NEGATIVE TESTS")
    print("-" * 76)
    not_rejected = 0
    for leg, merged, target, label in cases:
        try:
            offset = P.verify_leg_frame(leg, merged, target, 1.0)
            print(f"  !! NOT REJECTED at {offset:.2f} A - {label}")
            not_rejected += 1
        except RuntimeError as exc:
            measured = str(exc).split(" from the")[0].split("is ")[-1]
            print(f"  ok rejected at {measured:<8} - {label}")

    print()
    print("POSITIVE CONTROL")
    print("-" * 76)
    for leg, system, index, target in (("bound", bound_sys, bi, bound_target),
                                       ("free", free_sys, fi, free_target)):
        merged = P.build_leg_merge(leg, system, index, lig_a, lig_b, ident, ident,
                                   allow_ring_breaking=False, allow_ring_size_change=False)
        offset = P.verify_leg_frame(leg, merged, target, 1.0)
        print(f"  ok {leg:<6} correct per-leg merge accepted at {offset:.2e} A")

    assert not_rejected == 0, "the frame guard failed to reject a known-bad frame"
    print()
    print(f"  {len(cases)} negative + 2 positive checks passed")
'''.strip()
