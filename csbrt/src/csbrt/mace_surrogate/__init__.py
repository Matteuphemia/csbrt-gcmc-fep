"""csbrt.mace_surrogate -- hybrid MACE ML/MM surrogate with active learning.

A MACE foundation model replaces the classical force field for the perturbable
ligand (40-250 atoms) while the protein and bulk solvent stay on Amber ff14SB /
TIP3P, a committee watches the surrogate's uncertainty in real time, and any
frame it is not confident about is integrated on the original classical
Hamiltonian instead and harvested for fine-tuning.

    mace_mixed_system   ML/MM partitioning and openmm-ml mixed-system building
    committee           batched MACE inference over the ML region
    uq_monitor          committee force variance and the geometry guard
    fallback_controller the lambda_interpolate Hamiltonian switch and its logic
    active_learner      OOD buffer, clustering, labelling and fine-tuning
    runtime             the assembled surrogate the pipeline stages talk to
    somd2_hook          attaching the surrogate inside a SOMD2 FEP leg

Grounding:
- Duignan (2024), The Potential of Neural Network Potentials, ACS Phys. Chem.
  Au 4, 232-241 -- uncertainty-driven active learning for NNPs.
- Wang, Eastman, Tuckerman et al. (2024), On the Design Space Between Molecular
  Mechanics and Machine Learning Force Fields, arXiv:2409.02861 -- why the ML
  region has to stay small, and the createMixedSystem partitioning used here.
"""

from __future__ import annotations

from .config import (
    DEFAULT_ML_FORCE_GROUP,
    KNOWN_FOUNDATION_MODELS,
    MODEL_ALIASES,
    MACEConfig,
)
from .units import (
    EV_PER_ANGSTROM_TO_KCAL_PER_MOL_ANG,
    EV_PER_ANGSTROM_TO_KJ_PER_MOL_NM,
    EV_TO_KCAL_PER_MOL,
    EV_TO_KJ_PER_MOL,
    KCAL_TO_KJ,
)
from .mace_mixed_system import (
    DEFAULT_FOUNDATION_MODELS,
    INTERPOLATION_PARAMETER,
    MACESurrogateError,
    MixedSystemHandle,
    MLRegion,
    apply_system_in_place,
    attach_mace_to_context,
    build_ml_potential,
    check_region_compactness,
    create_mace_mixed_system,
    describe_ml_region,
    noninteracting_particles,
    nonbonded_cutoff_nm,
    partition_ml_atoms,
    positions_in_nm,
    region_diameter_nm,
    validate_ml_region,
)
from .committee import (
    CommitteePrediction,
    CommitteeUnavailable,
    MACECommittee,
    make_whole,
)
from .uq_monitor import (
    COVALENT_RADII_ANG,
    GeometryGuard,
    MACEUQMonitor,
    UQResult,
)
from .fallback_controller import (
    DualHamiltonianSwitch,
    EvaluatorMode,
    FallbackState,
    PhysicsFallbackController,
)
from .active_learner import (
    MACEFineTuner,
    MMSubsystemLabeler,
    OODBuffer,
    OODFrame,
    ReferenceLabeler,
    compute_kabsch_rmsd,
    extract_subsystem,
    harvest,
    write_training_set,
)
from .runtime import (
    MACERuntime,
    md_chunks,
    ml_coordinates,
    ml_region_bonds,
    worker_tag,
)

__all__ = [
    # config / units
    "MACEConfig",
    "KNOWN_FOUNDATION_MODELS",
    "MODEL_ALIASES",
    "DEFAULT_FOUNDATION_MODELS",
    "DEFAULT_ML_FORCE_GROUP",
    "EV_TO_KJ_PER_MOL",
    "EV_TO_KCAL_PER_MOL",
    "EV_PER_ANGSTROM_TO_KCAL_PER_MOL_ANG",
    "EV_PER_ANGSTROM_TO_KJ_PER_MOL_NM",
    "KCAL_TO_KJ",
    # mixed system
    "INTERPOLATION_PARAMETER",
    "MACESurrogateError",
    "MLRegion",
    "MixedSystemHandle",
    "partition_ml_atoms",
    "describe_ml_region",
    "validate_ml_region",
    "noninteracting_particles",
    "positions_in_nm",
    "build_ml_potential",
    "check_region_compactness",
    "nonbonded_cutoff_nm",
    "region_diameter_nm",
    "create_mace_mixed_system",
    "apply_system_in_place",
    "attach_mace_to_context",
    # committee / uq
    "MACECommittee",
    "CommitteePrediction",
    "CommitteeUnavailable",
    "make_whole",
    "MACEUQMonitor",
    "UQResult",
    "GeometryGuard",
    "COVALENT_RADII_ANG",
    # fallback
    "DualHamiltonianSwitch",
    "EvaluatorMode",
    "FallbackState",
    "PhysicsFallbackController",
    # active learning
    "OODBuffer",
    "OODFrame",
    "ReferenceLabeler",
    "MMSubsystemLabeler",
    "MACEFineTuner",
    "compute_kabsch_rmsd",
    "extract_subsystem",
    "harvest",
    "write_training_set",
    # runtime
    "MACERuntime",
    "md_chunks",
    "ml_coordinates",
    "ml_region_bonds",
    "worker_tag",
]
