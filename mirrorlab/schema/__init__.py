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
from mirrorlab.schema.task import (
    OBSTRUCTION_TO_REASON_CODE,
    ProofGeometry,
    VisibilityGroundTruth,
    VisibilityQuestion,
    VisibilityReasonCode,
    VisibilityTaskBundle,
)

__all__ = [
    "OBSTRUCTION_TO_REASON_CODE",
    "CameraSpec",
    "DifficultyLevel",
    "InspectionScene",
    "MirrorSpec",
    "ObstacleSpec",
    "ObstructionReason",
    "Point2D",
    "ProofGeometry",
    "SceneCategory",
    "TargetSpec",
    "TaskType",
    "Vector2D",
    "VisibilityGroundTruth",
    "VisibilityQuestion",
    "VisibilityReasonCode",
    "VisibilityState",
    "VisibilityTaskBundle",
]
