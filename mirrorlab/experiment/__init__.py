"""
MIRRORLAB Experiment and Task Discovery Subsystem.
"""

from mirrorlab.experiment.difficulty_calibrator import (
    DifficultyCalibrator,
    DifficultyMetrics,
    DifficultyTier,
)
from mirrorlab.experiment.task_explorer import (
    ClutterRegime,
    ExperimentManifest,
    GeometryRegime,
    OcclusionRegime,
    ReasoningRegime,
    TaskCondition,
    TaskExplorer,
    TaskGenerationResult,
)

__all__ = [
    "ClutterRegime",
    "DifficultyCalibrator",
    "DifficultyMetrics",
    "DifficultyTier",
    "ExperimentManifest",
    "GeometryRegime",
    "OcclusionRegime",
    "ReasoningRegime",
    "TaskCondition",
    "TaskExplorer",
    "TaskGenerationResult",
]
