"""Active learning: harvest what the surrogate could not model, then fix it.

Every fallback trigger is a labelled failure of the surrogate, and the loop
that closes on those failures is what makes the fast path get faster over a
campaign (Duignan 2024).

Four stages, each usable on its own:

1. :class:`OODBuffer` -- a worker-local, crash-safe ``.npz`` of flagged frames.
   One file per worker under ``al_buffer/``: 52 FEP edges x replicates write
   concurrently from separate Slurm tasks, and a shared file would race.
2. :func:`harvest` + :meth:`OODBuffer.cluster_and_deduplicate` -- merge the
   per-worker buffers and collapse redundant conformations by heavy-atom RMSD
   (0.5 A default), so the fine-tune set is diverse rather than large.
3. :class:`ReferenceLabeler` -- attach ground-truth energies and forces. The
   level of theory is recorded with every frame and never mixed.
4. :class:`MACEFineTuner` -- shell out to ``mace_run_train`` with
   ``--foundation_model``, which is MACE's own supported fine-tuning path.
   Writing a training loop here would be reinventing it, badly.

A note on label quality. MACE-OFF is fitted to wB97M-D3(BJ)/def2-TZVPPD. Fine-
tuning it against classical MM single-points would teach it the MM surface and
destroy the accuracy the surrogate exists to provide, so :class:`MACEFineTuner`
refuses MM-labelled data unless the caller sets ``allow_mm_labels=True``
deliberately. The MM labeler is still shipped: it is the right reference for
validating the harvesting machinery end to end, and for a run whose surrogate
is itself MM-level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .units import EV_TO_KCAL_PER_MOL, KCAL_TO_KJ

logger = logging.getLogger("csbrt.mace_surrogate.active_learning")

#: Levels of theory a frame's labels can carry. Fine-tuning refuses to mix them.
REFERENCE_LEVELS = ("unlabelled", "mm", "qm")

#: Element symbols, indexed by atomic number, for extended-XYZ output.
_SYMBOLS = (
    "X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co "
    "Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te "
    "I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir "
    "Pt Au Hg Tl Pb Bi Po At Rn"
).split()


def element_symbol(atomic_number: int) -> str:
    index = int(atomic_number)
    if 0 < index < len(_SYMBOLS):
        return _SYMBOLS[index]
    raise ValueError(f"No element symbol for atomic number {atomic_number}")


@dataclass
class OODFrame:
    """One flagged configuration of the ML region.

    Only the ML region is stored, in Angstroms: it is what MACE is trained on
    and what the committee evaluates, and keeping the 30k-atom environment would
    make the buffers unusable.
    """

    frame_id: str
    step: int
    positions: np.ndarray  # (N, 3) Angstroms, ML region only
    atomic_numbers: list[int]
    uncertainty_force: float  # eV/A
    uncertainty_energy: float  # kcal/mol
    reason: str
    box_vectors: np.ndarray | None = None  # (3, 3) Angstroms
    ml_atoms: list[int] | None = None  # indices in the parent system
    source: str = ""  # run/leg/replicate this came from
    energy_label: float | None = None  # eV
    forces_label: np.ndarray | None = None  # (N, 3) eV/A
    reference_level: str = "unlabelled"
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=np.float32)
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError(
                f"positions must be (N, 3); got {self.positions.shape}"
            )
        self.atomic_numbers = [int(z) for z in self.atomic_numbers]
        if len(self.atomic_numbers) != self.positions.shape[0]:
            raise ValueError(
                f"{len(self.atomic_numbers)} atomic numbers for "
                f"{self.positions.shape[0]} positions"
            )
        if self.box_vectors is not None:
            self.box_vectors = np.asarray(self.box_vectors, dtype=np.float32)
        if self.forces_label is not None:
            self.forces_label = np.asarray(self.forces_label, dtype=np.float32)
        if self.reference_level not in REFERENCE_LEVELS:
            raise ValueError(
                f"reference_level must be one of {REFERENCE_LEVELS}; got "
                f"{self.reference_level!r}"
            )

    @property
    def heavy_atom_mask(self) -> np.ndarray:
        return np.asarray(self.atomic_numbers, dtype=np.int64) > 1

    @property
    def is_labelled(self) -> bool:
        return self.energy_label is not None and self.forces_label is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "step": self.step,
            "positions": self.positions.tolist(),
            "atomic_numbers": list(self.atomic_numbers),
            "uncertainty_force": float(self.uncertainty_force),
            "uncertainty_energy": float(self.uncertainty_energy),
            "reason": self.reason,
            "box_vectors": (
                self.box_vectors.tolist() if self.box_vectors is not None else None
            ),
            "ml_atoms": list(self.ml_atoms) if self.ml_atoms is not None else None,
            "source": self.source,
            "energy_label": (
                float(self.energy_label) if self.energy_label is not None else None
            ),
            "forces_label": (
                self.forces_label.tolist() if self.forces_label is not None else None
            ),
            "reference_level": self.reference_level,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OODFrame":
        return cls(
            frame_id=data["frame_id"],
            step=int(data["step"]),
            positions=np.asarray(data["positions"], dtype=np.float32),
            atomic_numbers=data["atomic_numbers"],
            uncertainty_force=float(data["uncertainty_force"]),
            uncertainty_energy=float(data["uncertainty_energy"]),
            reason=data.get("reason", ""),
            box_vectors=(
                np.asarray(data["box_vectors"], dtype=np.float32)
                if data.get("box_vectors") is not None
                else None
            ),
            ml_atoms=data.get("ml_atoms"),
            source=data.get("source", ""),
            energy_label=data.get("energy_label"),
            forces_label=(
                np.asarray(data["forces_label"], dtype=np.float32)
                if data.get("forces_label") is not None
                else None
            ),
            reference_level=data.get("reference_level", "unlabelled"),
            timestamp=float(data.get("timestamp", time.time())),
        )

    def to_extxyz(self) -> str:
        """Extended-XYZ block with ``REF_energy`` / ``REF_forces`` labels.

        Written here rather than via ``ase.io`` so the key names, column order
        and precision are fixed by this package: ``mace_run_train`` is told the
        same key names, and a silent rename between ase versions would train on
        zeros.
        """
        if not self.is_labelled:
            raise ValueError(f"Frame {self.frame_id} has no labels to write")
        lines = [str(self.positions.shape[0])]
        comment = [
            'Properties=species:S:1:pos:R:3:REF_forces:R:3',
            f"REF_energy={float(self.energy_label):.10f}",
            f"reference_level={self.reference_level}",
            f"frame_id={self.frame_id}",
            f"sigma_f={float(self.uncertainty_force):.6f}",
        ]
        if self.box_vectors is not None:
            flat = " ".join(f"{v:.8f}" for v in self.box_vectors.reshape(-1))
            comment.insert(0, f'Lattice="{flat}"')
            comment.append("pbc=\"T T T\"")
        else:
            comment.append("pbc=\"F F F\"")
        lines.append(" ".join(comment))
        for symbol_index, position, force in zip(
            self.atomic_numbers, self.positions, self.forces_label
        ):
            lines.append(
                f"{element_symbol(symbol_index):<2s} "
                f"{position[0]:14.8f} {position[1]:14.8f} {position[2]:14.8f} "
                f"{force[0]:14.8f} {force[1]:14.8f} {force[2]:14.8f}"
            )
        return "\n".join(lines)


def compute_kabsch_rmsd(coords_p: np.ndarray, coords_q: np.ndarray) -> float:
    """Minimum RMSD between two equal-length coordinate sets (Kabsch 1976)."""
    p = np.asarray(coords_p, dtype=np.float64)
    q = np.asarray(coords_q, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError(f"Shape mismatch: {p.shape} vs {q.shape}")
    if p.shape[0] == 0:
        raise ValueError("Cannot align empty coordinate sets")
    p = p - p.mean(axis=0)
    q = q - q.mean(axis=0)
    covariance = p.T @ q
    v, _s, w = np.linalg.svd(covariance)
    if (np.linalg.det(v) * np.linalg.det(w)) < 0.0:
        v[:, -1] = -v[:, -1]
    rotated = p @ (v @ w)
    return float(np.sqrt(np.mean(np.sum((rotated - q) ** 2, axis=-1))))


class OODBuffer:
    """Worker-local store of flagged frames, written atomically."""

    def __init__(
        self,
        buffer_file: Path | str | None = None,
        *,
        max_frames: int = 2000,
        source: str = "",
    ) -> None:
        self.buffer_file = Path(buffer_file) if buffer_file else None
        self.max_frames = int(max_frames)
        self.source = source
        self.frames: list[OODFrame] = []
        self.dropped = 0

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self):
        return iter(self.frames)

    def add(self, frame: OODFrame) -> OODFrame:
        self.frames.append(frame)
        self._enforce_cap()
        return frame

    def add_from_raw(
        self,
        step: int,
        positions: np.ndarray,
        atomic_numbers: Sequence[int],
        uncertainty_force: float,
        uncertainty_energy: float,
        reason: str,
        box_vectors: np.ndarray | None = None,
        ml_atoms: Sequence[int] | None = None,
        source: str | None = None,
    ) -> OODFrame:
        origin = self.source if source is None else source
        identifier = f"{origin or 'run'}_ood{len(self.frames) + 1:06d}_step{step}"
        return self.add(
            OODFrame(
                frame_id=identifier,
                step=int(step),
                positions=positions,
                atomic_numbers=list(atomic_numbers),
                uncertainty_force=float(uncertainty_force),
                uncertainty_energy=float(uncertainty_energy),
                reason=reason,
                box_vectors=box_vectors,
                ml_atoms=list(ml_atoms) if ml_atoms is not None else None,
                source=origin,
            )
        )

    def _enforce_cap(self) -> None:
        """Keep the most informative frames when the cap is hit.

        Dropping the oldest would bias the set toward the start of the
        trajectory, so the lowest-uncertainty frame goes instead: it is the one
        the model is closest to handling already.
        """
        while len(self.frames) > self.max_frames:
            weakest = min(
                range(len(self.frames)),
                key=lambda i: self.frames[i].uncertainty_force,
            )
            self.frames.pop(weakest)
            self.dropped += 1

    # --- persistence ---------------------------------------------------------

    def save(self, file_path: Path | str | None = None) -> Path:
        """Write the buffer. Atomic: a killed worker cannot leave a torn file."""
        target = Path(file_path or self.buffer_file or "ood_frames.npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp.npz")
        metadata = json.dumps([frame.to_dict() for frame in self.frames])
        np.savez_compressed(
            str(temporary),
            steps=np.array([f.step for f in self.frames], dtype=np.int64),
            uncertainty_force=np.array(
                [f.uncertainty_force for f in self.frames], dtype=np.float32
            ),
            uncertainty_energy=np.array(
                [f.uncertainty_energy for f in self.frames], dtype=np.float32
            ),
            metadata_json=np.array(metadata),
            dropped=np.array(self.dropped, dtype=np.int64),
        )
        temporary.replace(target)
        logger.info("Saved %d OOD frame(s) to %s", len(self.frames), target)
        return target

    def load(self, file_path: Path | str | None = None) -> int:
        source = Path(file_path or self.buffer_file or "ood_frames.npz")
        if not source.is_file():
            logger.warning("OOD buffer %s does not exist", source)
            return 0
        with np.load(str(source), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
            self.dropped = int(data["dropped"]) if "dropped" in data else 0
        self.frames = [OODFrame.from_dict(entry) for entry in metadata]
        logger.info("Loaded %d OOD frame(s) from %s", len(self.frames), source)
        return len(self.frames)

    def extend(self, frames: Iterable[OODFrame]) -> None:
        for frame in frames:
            self.frames.append(frame)
        self._enforce_cap()

    # --- selection -----------------------------------------------------------

    def cluster_and_deduplicate(
        self,
        rmsd_cutoff_angstrom: float = 0.5,
        heavy_atoms_only: bool = True,
    ) -> list[OODFrame]:
        """Greedy RMSD clustering; returns one representative per basin.

        The representative is the highest-uncertainty member: within a basin the
        frame the surrogate handles worst is the one worth labelling.
        """
        if not self.frames:
            return []

        def selected(frame: OODFrame) -> np.ndarray:
            if heavy_atoms_only:
                mask = frame.heavy_atom_mask
                if mask.any():
                    return frame.positions[mask]
            return frame.positions

        clusters: list[list[OODFrame]] = []
        signatures: list[tuple[tuple[int, ...], np.ndarray]] = []
        for frame in self.frames:
            coords = selected(frame)
            key = tuple(frame.atomic_numbers)
            placed = False
            for index, (cluster_key, reference) in enumerate(signatures):
                if cluster_key != key or reference.shape != coords.shape:
                    # Different molecule (a different FEP edge, say): never the
                    # same basin, and RMSD between them is meaningless.
                    continue
                if compute_kabsch_rmsd(coords, reference) <= rmsd_cutoff_angstrom:
                    clusters[index].append(frame)
                    placed = True
                    break
            if not placed:
                clusters.append([frame])
                signatures.append((key, coords))

        representatives = [
            max(cluster, key=lambda f: f.uncertainty_force) for cluster in clusters
        ]
        logger.info(
            "Deduplicated %d OOD frame(s) -> %d basin(s) at %.2f A RMSD",
            len(self.frames),
            len(representatives),
            rmsd_cutoff_angstrom,
        )
        return representatives


def harvest(
    buffer_dir: Path | str, pattern: str = "*.npz", max_frames: int = 100_000
) -> OODBuffer:
    """Merge every per-worker buffer under ``buffer_dir`` into one."""
    root = Path(buffer_dir)
    merged = OODBuffer(max_frames=max_frames)
    files = sorted(root.glob(pattern)) if root.is_dir() else []
    for path in files:
        worker = OODBuffer(path)
        try:
            worker.load()
        except (OSError, ValueError, KeyError) as error:
            logger.error("Skipping unreadable OOD buffer %s: %s", path, error)
            continue
        merged.extend(worker.frames)
    logger.info(
        "Harvested %d OOD frame(s) from %d buffer file(s) under %s",
        len(merged),
        len(files),
        root,
    )
    return merged


# --------------------------------------------------------------------------- labels


class ReferenceLabeler:
    """Attaches ground-truth energies and forces to OOD frames.

    ``evaluator_fn(positions_ang, atomic_numbers) -> (energy_eV, forces_eV_per_A)``
    is the hook: point it at a QM single-point (Psi4, ORCA) for a real
    fine-tuning set, or at :class:`MMSubsystemLabeler` for an MM-level
    reference. ``reference_level`` is stamped on every frame it touches and is
    what stops two levels of theory being trained on together.
    """

    def __init__(
        self,
        evaluator_fn: Callable[[np.ndarray, Sequence[int]], tuple[float, np.ndarray]],
        reference_level: str = "qm",
    ) -> None:
        if reference_level not in REFERENCE_LEVELS or reference_level == "unlabelled":
            raise ValueError(
                f"reference_level must be 'mm' or 'qm'; got {reference_level!r}"
            )
        self.evaluator_fn = evaluator_fn
        self.reference_level = reference_level
        self.failures: list[tuple[str, str]] = []

    def label_frame(self, frame: OODFrame) -> OODFrame:
        energy, forces = self.evaluator_fn(frame.positions, frame.atomic_numbers)
        forces = np.asarray(forces, dtype=np.float32)
        if forces.shape != frame.positions.shape:
            raise ValueError(
                f"Labeler returned forces of shape {forces.shape} for "
                f"{frame.positions.shape} positions"
            )
        frame.energy_label = float(energy)
        frame.forces_label = forces
        frame.reference_level = self.reference_level
        return frame

    def label_all(self, frames: Sequence[OODFrame]) -> list[OODFrame]:
        """Label every frame, skipping (and reporting) the ones that fail.

        One pathological geometry must not throw away a whole harvest.
        """
        labelled: list[OODFrame] = []
        for frame in frames:
            try:
                labelled.append(self.label_frame(frame))
            except Exception as error:
                self.failures.append((frame.frame_id, str(error)))
                logger.error("Labelling failed for %s: %s", frame.frame_id, error)
        if self.failures:
            logger.warning(
                "%d of %d frame(s) could not be labelled",
                len(self.failures),
                len(frames),
            )
        return labelled


class MMSubsystemLabeler:
    """Classical single-points on the ML region alone, via OpenMM.

    Builds a standalone ``openmm.System`` containing only the ML atoms and the
    bonded/nonbonded terms entirely within them -- the intramolecular Hamiltonian
    the mixed system hands to MACE. Useful for validating the harvest-label-train
    path end to end and as the reference for an MM-level surrogate. It is *not*
    an appropriate target for fine-tuning a QM-fitted foundation model; see the
    module docstring.
    """

    reference_level = "mm"

    def __init__(self, system: Any, ml_atoms: Sequence[int], platform: str = "Reference"):
        import openmm

        self.ml_atoms = [int(i) for i in ml_atoms]
        self.subsystem = extract_subsystem(system, self.ml_atoms)
        integrator = openmm.VerletIntegrator(0.001)
        self.context = openmm.Context(
            self.subsystem, integrator, openmm.Platform.getPlatformByName(platform)
        )

    def __call__(
        self, positions_ang: np.ndarray, atomic_numbers: Sequence[int]
    ) -> tuple[float, np.ndarray]:
        import openmm.unit as unit

        from .units import KJ_PER_MOL_TO_EV

        coords = np.asarray(positions_ang, dtype=np.float64)
        if coords.shape[0] != len(self.ml_atoms):
            raise ValueError(
                f"Frame has {coords.shape[0]} atoms; this labeler was built for "
                f"{len(self.ml_atoms)}"
            )
        self.context.setPositions(coords * 0.1)  # A -> nm
        state = self.context.getState(getEnergy=True, getForces=True)
        energy_kj = state.getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole
        )
        forces_kj_nm = state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        )
        # kJ/(mol nm) -> eV/A
        forces_ev_ang = np.asarray(forces_kj_nm) * KJ_PER_MOL_TO_EV * 0.1
        return energy_kj * KJ_PER_MOL_TO_EV, forces_ev_ang


def extract_subsystem(system: Any, atom_indices: Sequence[int]) -> Any:
    """Build a standalone System from the terms wholly inside ``atom_indices``.

    Handles the force types an Amber ff14SB/GAFF ligand actually uses:
    harmonic bonds and angles, periodic torsions, and the nonbonded force with
    its exceptions. Any other force is dropped, with a warning, rather than
    being silently mis-translated.
    """
    import openmm

    wanted = [int(i) for i in atom_indices]
    remap = {old: new for new, old in enumerate(wanted)}
    subsystem = openmm.System()
    for index in wanted:
        subsystem.addParticle(system.getParticleMass(index))

    for index in range(system.getNumConstraints()):
        p1, p2, distance = system.getConstraintParameters(index)
        if p1 in remap and p2 in remap:
            subsystem.addConstraint(remap[p1], remap[p2], distance)

    dropped: set[str] = set()
    for force in system.getForces():
        if isinstance(force, openmm.HarmonicBondForce):
            new = openmm.HarmonicBondForce()
            for index in range(force.getNumBonds()):
                p1, p2, length, k = force.getBondParameters(index)
                if p1 in remap and p2 in remap:
                    new.addBond(remap[p1], remap[p2], length, k)
            if new.getNumBonds():
                subsystem.addForce(new)
        elif isinstance(force, openmm.HarmonicAngleForce):
            new = openmm.HarmonicAngleForce()
            for index in range(force.getNumAngles()):
                p1, p2, p3, angle, k = force.getAngleParameters(index)
                if p1 in remap and p2 in remap and p3 in remap:
                    new.addAngle(remap[p1], remap[p2], remap[p3], angle, k)
            if new.getNumAngles():
                subsystem.addForce(new)
        elif isinstance(force, openmm.PeriodicTorsionForce):
            new = openmm.PeriodicTorsionForce()
            for index in range(force.getNumTorsions()):
                p1, p2, p3, p4, periodicity, phase, k = force.getTorsionParameters(
                    index
                )
                if all(p in remap for p in (p1, p2, p3, p4)):
                    new.addTorsion(
                        remap[p1],
                        remap[p2],
                        remap[p3],
                        remap[p4],
                        periodicity,
                        phase,
                        k,
                    )
            if new.getNumTorsions():
                subsystem.addForce(new)
        elif isinstance(force, openmm.NonbondedForce):
            new = openmm.NonbondedForce()
            new.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
            for index in wanted:
                charge, sigma, epsilon = force.getParticleParameters(index)
                new.addParticle(charge, sigma, epsilon)
            for index in range(force.getNumExceptions()):
                p1, p2, charge, sigma, epsilon = force.getExceptionParameters(index)
                if p1 in remap and p2 in remap:
                    new.addException(remap[p1], remap[p2], charge, sigma, epsilon)
            subsystem.addForce(new)
        elif isinstance(force, (openmm.CMMotionRemover,)):
            continue
        else:
            dropped.add(type(force).__name__)
    if dropped:
        logger.warning(
            "extract_subsystem dropped unsupported force type(s): %s",
            ", ".join(sorted(dropped)),
        )
    return subsystem


# --------------------------------------------------------------------------- training


def write_training_set(
    frames: Sequence[OODFrame], path: Path | str
) -> dict[str, Any]:
    """Write labelled frames as extended XYZ for ``mace_run_train``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    labelled = [frame for frame in frames if frame.is_labelled]
    if not labelled:
        raise ValueError("No labelled frames to write")
    levels = {frame.reference_level for frame in labelled}
    if len(levels) > 1:
        raise ValueError(
            f"Refusing to write a training set mixing levels of theory: {sorted(levels)}"
        )
    target.write_text("\n".join(frame.to_extxyz() for frame in labelled) + "\n")
    return {
        "path": str(target),
        "frames": len(labelled),
        "skipped_unlabelled": len(frames) - len(labelled),
        "reference_level": levels.pop(),
    }


class MACEFineTuner:
    """Fine-tunes a MACE foundation model on harvested OOD frames.

    Delegates the training loop to ``mace_run_train``, which mace-torch installs
    and which implements the supported fine-tuning path: ``--foundation_model``
    starts from the released weights, ``--num_interactions``/head replay and a
    low learning rate keep the fitted chemistry intact. The knobs exposed here
    are the ones that matter for not destroying the foundation model:
    a conservative learning rate (1e-4), few epochs, and early stopping.
    """

    def __init__(
        self,
        foundation_model: str = "small",
        *,
        learning_rate: float = 1.0e-4,
        max_epochs: int = 20,
        batch_size: int = 4,
        patience: int = 5,
        valid_fraction: float = 0.2,
        device: str = "cuda",
        default_dtype: str = "float64",
        extra_args: Sequence[str] = (),
        executable: str | None = None,
    ) -> None:
        self.foundation_model = foundation_model
        self.learning_rate = float(learning_rate)
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self.patience = int(patience)
        self.valid_fraction = float(valid_fraction)
        self.device = device
        self.default_dtype = default_dtype
        self.extra_args = list(extra_args)
        self.executable = executable or shutil.which("mace_run_train")

    def command(self, dataset: Path, work_dir: Path, name: str) -> list[str]:
        """The exact ``mace_run_train`` invocation, so it can be inspected/logged."""
        if self.executable is None:
            raise FileNotFoundError(
                "mace_run_train is not on PATH; install mace-torch "
                "(`pip install 'mace-torch>=0.3.10'`)"
            )
        return [
            self.executable,
            "--name", name,
            "--foundation_model", self.foundation_model,
            "--train_file", str(dataset),
            "--valid_fraction", str(self.valid_fraction),
            "--energy_key", "REF_energy",
            "--forces_key", "REF_forces",
            "--lr", str(self.learning_rate),
            "--max_num_epochs", str(self.max_epochs),
            "--batch_size", str(self.batch_size),
            "--patience", str(self.patience),
            "--default_dtype", self.default_dtype,
            "--device", self.device,
            "--model_dir", str(work_dir),
            "--checkpoints_dir", str(work_dir / "checkpoints"),
            "--log_dir", str(work_dir / "logs"),
            "--results_dir", str(work_dir / "results"),
            *self.extra_args,
        ]

    def fine_tune(
        self,
        training_frames: Sequence[OODFrame],
        output_checkpoint_path: Path | str,
        generation: int = 1,
        *,
        allow_mm_labels: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Run one fine-tuning generation and return a manifest of what happened."""
        output = Path(output_checkpoint_path)
        work_dir = output.parent
        work_dir.mkdir(parents=True, exist_ok=True)

        labelled = [frame for frame in training_frames if frame.is_labelled]
        if not labelled:
            logger.warning("No labelled frames; skipping fine-tuning")
            return {"status": "skipped", "reason": "no_labelled_frames"}

        levels = {frame.reference_level for frame in labelled}
        if len(levels) > 1:
            return {
                "status": "refused",
                "reason": f"mixed_reference_levels:{sorted(levels)}",
            }
        level = levels.pop()
        if level == "mm" and not allow_mm_labels:
            logger.error(
                "Refusing to fine-tune a QM-fitted MACE model on %d MM-labelled "
                "frame(s): it would replace the model's quantum accuracy with the "
                "force field the surrogate was meant to replace. Label with a QM "
                "single-point, or pass allow_mm_labels=True deliberately.",
                len(labelled),
            )
            return {"status": "refused", "reason": "mm_labels_without_optin"}

        dataset = work_dir / f"train_gen{generation}.xyz"
        manifest = write_training_set(labelled, dataset)
        name = f"mace_finetuned_gen{generation}"
        command = self.command(dataset, work_dir, name)

        record: dict[str, Any] = {
            "status": "prepared",
            "generation": generation,
            "reference_level": level,
            "dataset": manifest,
            "command": command,
            "checkpoint_path": str(output),
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "timestamp": time.time(),
        }
        if dry_run:
            (work_dir / f"finetune_gen{generation}.json").write_text(
                json.dumps(record, indent=2)
            )
            return record

        logger.info("Fine-tuning MACE generation %d: %s", generation, " ".join(command))
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False
        )
        (work_dir / f"finetune_gen{generation}.log").write_text(
            completed.stdout + "\n" + completed.stderr
        )
        if completed.returncode != 0:
            record["status"] = "failed"
            record["returncode"] = completed.returncode
            record["stderr_tail"] = completed.stderr[-4000:]
            logger.error(
                "mace_run_train exited %d; see %s",
                completed.returncode,
                work_dir / f"finetune_gen{generation}.log",
            )
            return record

        produced = work_dir / f"{name}.model"
        if produced.is_file():
            if produced.resolve() != output.resolve():
                shutil.copy2(produced, output)
            record["status"] = "completed"
        else:
            record["status"] = "failed"
            record["reason"] = f"mace_run_train produced no {produced.name}"
        (work_dir / f"finetune_gen{generation}.json").write_text(
            json.dumps(record, indent=2)
        )
        return record
