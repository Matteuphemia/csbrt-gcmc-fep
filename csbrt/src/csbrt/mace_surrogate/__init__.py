"""csbrt.mace_surrogate

Hybrid Machine Learning Force Field (MLFF / MACE) surrogate with active learning
runtime uncertainty monitoring and automated physics fallback for csbrt GCMC and FEP.

Grounding:
- Duignan (2024): The Potential of Neural Network Potentials (ACS Phys. Chem. Au 2024)
- Wang, Eastman, Tuckerman et al. (2024): On the Design Space Between Molecular Mechanics
  and Machine Learning Force Fields (arXiv:2409.02861)
"""

from __future__ import annotations

from .mace_mixed_system import (
    MACEConfig,
    MACEMixedSystemBuilder,
    create_mace_mixed_system,
    partition_ml_atoms,
    DEFAULT_FOUNDATION_MODELS,
    KNOWN_FOUNDATION_MODELS,
    EV_PER_ANGSTROM_TO_KCAL_PER_MOL_ANG,
    EV_PER_ANGSTROM_TO_KJ_PER_MOL_NM,
    EV_TO_KCAL_PER_MOL,
    EV_TO_KJ_PER_MOL,
    KCAL_TO_KJ,
)
from .uq_monitor import (
    MACEUQMonitor,
    UQResult,
)
from .ensemble_evaluator import (
    MACEEnsembleEvaluator,
    MACEEnsembleUnavailable,
    DEFAULT_SHELL_RADIUS_ANG,
    _select_ml_shell,
)
from .fallback_controller import (
    EvaluatorMode,
    FallbackState,
    PhysicsFallbackController,
)
from .active_learner import (
    MACEFineTuner,
    OODBuffer,
    OODFrame,
    ReferenceLabeler,
    compute_kabsch_rmsd,
)

__all__ = [
    "MACEConfig",
    "MACEMixedSystemBuilder",
    "create_mace_mixed_system",
    "partition_ml_atoms",
    "DEFAULT_FOUNDATION_MODELS",
    "KNOWN_FOUNDATION_MODELS",
    "EV_PER_ANGSTROM_TO_KCAL_PER_MOL_ANG",
    "EV_PER_ANGSTROM_TO_KJ_PER_MOL_NM",
    "EV_TO_KCAL_PER_MOL",
    "EV_TO_KJ_PER_MOL",
    "KCAL_TO_KJ",
    "MACEUQMonitor",
    "UQResult",
    "MACEEnsembleEvaluator",
    "MACEEnsembleUnavailable",
    "DEFAULT_SHELL_RADIUS_ANG",
    "_select_ml_shell",
    "EvaluatorMode",
    "FallbackState",
    "PhysicsFallbackController",
    "MACEFineTuner",
    "OODBuffer",
    "OODFrame",
    "ReferenceLabeler",
    "compute_kabsch_rmsd",
]
