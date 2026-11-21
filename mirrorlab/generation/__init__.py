"""
Procedural Scene and Visual Task Factory for MIRRORLAB.
"""

from mirrorlab.generation.counterfactual_factory import CounterfactualFactory
from mirrorlab.generation.obstruction_factory import ObstructionTaskFactory
from mirrorlab.generation.repair_factory import RepairTaskFactory
from mirrorlab.generation.scene_generator import (
    DIFFICULTY_PRESETS,
    DifficultyParams,
    GenerationError,
    SceneGenerator,
)
from mirrorlab.generation.task_factory import VisibilityTaskFactory

__all__ = [
    "DIFFICULTY_PRESETS",
    "CounterfactualFactory",
    "DifficultyParams",
    "GenerationError",
    "ObstructionTaskFactory",
    "RepairTaskFactory",
    "SceneGenerator",
    "VisibilityTaskFactory",
]
