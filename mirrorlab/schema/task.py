"""
Pydantic Schemas for Visibility Benchmark Tasks in MIRRORLAB.

Enforces strict separation between public blind evaluation items and
private mathematical ground-truth answer keys.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from mirrorlab.schema.common import ObstructionReason, Point2D, TaskType
from mirrorlab.schema.scene import DifficultyLevel, InspectionScene, SceneCategory


class VisibilityReasonCode(str, Enum):
    """
    Standardized reason codes for reflected visibility determination.
    """

    CLEAR = "CLEAR"
    MIRROR_MISS = "MIRROR_MISS"
    OUTSIDE_FOV = "OUTSIDE_FOV"
    WRONG_SIDE = "WRONG_SIDE"
    BLOCKED_CAMERA_TO_MIRROR = "BLOCKED_CAMERA_TO_MIRROR"
    BLOCKED_MIRROR_TO_TARGET = "BLOCKED_MIRROR_TO_TARGET"


# Mapping from internal geometry engine obstruction reason to standard benchmark code
OBSTRUCTION_TO_REASON_CODE: dict[ObstructionReason, VisibilityReasonCode] = {
    ObstructionReason.NONE: VisibilityReasonCode.CLEAR,
    ObstructionReason.MISSED_MIRROR: VisibilityReasonCode.MIRROR_MISS,
    ObstructionReason.OUT_OF_FOV: VisibilityReasonCode.OUTSIDE_FOV,
    ObstructionReason.BACK_OF_MIRROR: VisibilityReasonCode.WRONG_SIDE,
    ObstructionReason.BLOCKED_INCIDENT: VisibilityReasonCode.BLOCKED_CAMERA_TO_MIRROR,
    ObstructionReason.BLOCKED_REFLECTED: VisibilityReasonCode.BLOCKED_MIRROR_TO_TARGET,
}


class VisibilityQuestion(BaseModel):
    """
    Public task record given to Vision-Language Models for blind evaluation.
    STRICT INVARIANT: Contains zero answers, zero ray coordinates, zero ground-truth keys.
    """

    model_config = ConfigDict(frozen=True)

    task_id: str = Field(..., description="Unique benchmark task identifier")
    task_type: TaskType = Field(
        default=TaskType.TASK_A_VISIBILITY, description="Benchmark task category"
    )
    question: str = Field(
        default="Can the camera see the center of the red X through the mirror?",
        description="The primary visibility question",
    )
    prompt: str = Field(
        ..., description="Complete multi-modal instructional prompt for the VLM"
    )
    image_filename: str = Field(
        ..., description="Filename of the blind inspection diagram PNG"
    )
    binary_options: list[str] = Field(
        default=["YES", "NO"], description="Available binary answer choices"
    )
    reason_options: list[str] = Field(
        default=[code.value for code in VisibilityReasonCode],
        description="Available physical reason codes",
    )


class ProofGeometry(BaseModel):
    """Exact computational geometry proof coordinates for private ground truth."""

    model_config = ConfigDict(frozen=True)

    reflection_point: Point2D | None = None
    virtual_target: Point2D | None = None
    incident_angle_deg: float | None = None
    reflected_angle_deg: float | None = None
    total_optical_distance: float | None = None
    optical_path: list[Point2D] = Field(default_factory=list)


class VisibilityGroundTruth(BaseModel):
    """
    Private answer-key and proof record for benchmark grading.
    STRICT INVARIANT: Must NEVER be passed to evaluated models.
    """

    model_config = ConfigDict(frozen=True)

    task_id: str = Field(..., description="Matches VisibilityQuestion.task_id")
    can_see: bool = Field(
        ..., description="True if red-X scratch is visible via specular reflection"
    )
    answer: str = Field(..., description="'YES' if visible, 'NO' if blocked or missed")
    reason_code: VisibilityReasonCode = Field(
        ..., description="Standardized physical reason code"
    )
    detailed_explanation: str = Field(
        ..., description="Deterministic step-by-step mathematical proof"
    )
    proof_geometry: ProofGeometry = Field(
        ..., description="Geometric coordinates and angles"
    )


class VisibilityTaskBundle(BaseModel):
    """
    Complete bundle containing all 5 generated artifacts for a single task:
    1. Scene JSON (SceneSpec)
    2. Blind Diagram PNG (path)
    3. Question JSON (VisibilityQuestion)
    4. Ground-Truth Answer JSON (VisibilityGroundTruth)
    5. Explanation Overlay PNG (path)
    """

    task_id: str
    seed: int
    category: SceneCategory
    difficulty: DifficultyLevel
    theme: str
    scene: InspectionScene
    question: VisibilityQuestion
    ground_truth: VisibilityGroundTruth
    blind_image_path: str
    overlay_image_path: str
