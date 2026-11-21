"""
Schema definitions for MIRRORLAB.
"""

from mirrorlab.schema.common import (
    ObstructionReason,
    Point2D,
    TaskType,
    Vector2D,
    VisibilityState,
)
from mirrorlab.schema.scene import (
    CameraSpec,
    DifficultyLevel,
    InspectionScene,
    MirrorSpec,
    ObstacleSpec,
    SceneCategory,
    TargetSpec,
)

__all__ = [
    "CameraSpec",
    "DifficultyLevel",
    "InspectionScene",
    "MirrorSpec",
    "ObstacleReason",
    "ObstacleSpec",
    "ObstructionReason",
    "Point2D",
    "SceneCategory",
    "TargetSpec",
    "TaskType",
    "Vector2D",
    "VisibilityState",
]
