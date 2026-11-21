"""
Pydantic Schemas for Visibility Benchmark Tasks in MIRRORLAB.

Enforces strict separation between public blind evaluation items and
private mathematical ground-truth answer keys.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mirrorlab.schema.common import ObstructionReason, Point2D, TaskType
from mirrorlab.schema.scene import (
    DifficultyLevel,
    InspectionScene,
    MirrorSpec,
    SceneCategory,
)


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


class BlockedSegment(str, Enum):
    """Path segment identified in obstruction reasoning tasks."""

    NONE = "NONE"
    CAMERA_TO_MIRROR = "CAMERA_TO_MIRROR"
    MIRROR_TO_TARGET = "MIRROR_TO_TARGET"


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


class ObstructionQuestion(BaseModel):
    """
    Public task record for Task B: Obstruction Identification.
    STRICT INVARIANT: Contains zero answers, zero ray coordinates, zero proof coordinates.
    """

    model_config = ConfigDict(frozen=True)

    task_id: str = Field(..., description="Unique benchmark task identifier")
    task_type: TaskType = Field(
        default=TaskType.TASK_B_OBSTRUCTION, description="Benchmark task category"
    )
    question: str = Field(
        default="Is the camera's view of the red X blocked by an obstacle? If so, which obstacle blocks which segment of the path?",
        description="The primary obstruction question",
    )
    prompt: str = Field(
        ..., description="Complete multi-modal instructional prompt for the VLM"
    )
    image_filename: str = Field(
        ..., description="Filename of the blind inspection diagram PNG"
    )
    obstacle_labels: list[str] = Field(
        ...,
        description="List of obstacle labels present in diagram (e.g. ['O1', 'O2', 'O3'])",
    )
    segment_options: list[str] = Field(
        default=[s.value for s in BlockedSegment],
        description="Available path segment choices",
    )


class ObstructionProof(BaseModel):
    """Exact computational geometry proof coordinates for obstruction ground truth."""

    model_config = ConfigDict(frozen=True)

    reflection_point: Point2D
    virtual_target: Point2D
    incident_angle_deg: float
    reflected_angle_deg: float
    blocked_segment: BlockedSegment
    blocking_obstacle_label: str | None = None
    blocking_obstacle_id: str | None = None
    intersection_point: Point2D | None = None
    penetration_depth_mm: float | None = None
    min_distractor_clearance_mm: float
    optical_path: list[Point2D] = Field(default_factory=list)


class ObstructionGroundTruth(BaseModel):
    """
    Private answer-key and proof record for Task B: Obstruction Identification.
    """

    model_config = ConfigDict(frozen=True)

    task_id: str = Field(..., description="Matches ObstructionQuestion.task_id")
    is_blocked: bool = Field(
        ..., description="True if any obstacle blocks the specular reflection path"
    )
    blocking_obstacle_label: str | None = Field(
        None, description="Label of blocking obstacle (e.g. 'O2') or None if unblocked"
    )
    blocked_segment: BlockedSegment = Field(
        ...,
        description="Which path segment is blocked: CAMERA_TO_MIRROR, MIRROR_TO_TARGET, or NONE",
    )
    detailed_explanation: str = Field(
        ..., description="Deterministic step-by-step mathematical proof"
    )
    proof: ObstructionProof = Field(
        ..., description="Geometric coordinates and clearances"
    )


class ObstructionTaskBundle(BaseModel):
    """
    Complete bundle containing all 5 generated artifacts for Task B:
    1. Scene JSON (InspectionScene)
    2. Blind Diagram PNG (path)
    3. Question JSON (ObstructionQuestion)
    4. Ground-Truth Answer JSON (ObstructionGroundTruth)
    5. Explanation Overlay PNG (path)
    """

    task_id: str
    seed: int
    difficulty: DifficultyLevel
    theme: str
    scene: InspectionScene
    question: ObstructionQuestion
    ground_truth: ObstructionGroundTruth
    blind_image_path: str
    overlay_image_path: str


class RepairCandidateEvaluation(BaseModel):
    """
    Detailed independent evaluation of a single candidate mirror configuration.
    """

    model_config = ConfigDict(frozen=True)

    candidate_id: str = Field(
        ..., description="Candidate option identifier ('A', 'B', or 'C')"
    )
    mirror: MirrorSpec = Field(
        ..., description="Mirror geometry specification for this candidate"
    )
    is_visible: bool = Field(
        ..., description="Whether this candidate restores visibility"
    )
    reason_code: VisibilityReasonCode = Field(
        ..., description="Standardized physical reason code"
    )
    diagnostic_details: str = Field(..., description="Oracle diagnostic details")
    reflection_point: Point2D | None = None
    incident_angle_deg: float | None = None
    reflected_angle_deg: float | None = None


class RepairQuestion(BaseModel):
    """
    Public task record for Task C: Mirror-Setting Repair.
    STRICT INVARIANT: Contains zero answers, zero ray coordinates, zero proof coordinates.
    """

    model_config = ConfigDict(frozen=True)

    task_id: str = Field(..., description="Unique benchmark task identifier")
    task_type: TaskType = Field(
        default=TaskType.TASK_C_REPAIR, description="Benchmark task category"
    )
    question: str = Field(
        default="Which candidate mirror configuration (A, B, or C) successfully restores the camera's view of the red X target scratch?",
        description="The primary repair question",
    )
    prompt: str = Field(
        ..., description="Complete multi-modal instructional prompt for the VLM"
    )
    image_filename: str = Field(
        ..., description="Filename of the blind inspection diagram PNG"
    )
    options: list[str] = Field(
        default=["A", "B", "C"], description="Candidate mirror choices"
    )


class RepairGroundTruth(BaseModel):
    """
    Private answer-key and proof record for Task C: Mirror-Setting Repair.
    Contains independent evaluations for all three candidates (A, B, and C).
    """

    model_config = ConfigDict(frozen=True)

    task_id: str = Field(..., description="Matches RepairQuestion.task_id")
    correct_candidate: str = Field(
        ...,
        description="The unique candidate option that restores visibility ('A', 'B', or 'C')",
    )
    detailed_explanation: str = Field(
        ..., description="Deterministic step-by-step mathematical proof"
    )
    candidates: dict[str, RepairCandidateEvaluation] = Field(
        ..., description="Evaluations of candidates A, B, and C"
    )


class RepairTaskBundle(BaseModel):
    """
    Complete bundle containing all 5 generated artifacts for Task C:
    1. Scene JSON (InspectionScene)
    2. Blind Diagram PNG (path)
    3. Question JSON (RepairQuestion)
    4. Ground-Truth Answer JSON (RepairGroundTruth)
    5. Explanation Overlay PNG (path)
    """

    task_id: str
    seed: int
    difficulty: DifficultyLevel
    theme: str
    scene: InspectionScene
    candidate_mirrors: dict[str, MirrorSpec]
    question: RepairQuestion
    ground_truth: RepairGroundTruth
    blind_image_path: str
    overlay_image_path: str


class CounterfactualPairType(str, Enum):
    """Types of counterfactual and invariant paired scenes."""

    MIRROR_ANGLE_FLIP = "MIRROR_ANGLE_FLIP"
    COSMETIC_CONTROL = "COSMETIC_CONTROL"
    OBSTACLE_INSERTION = "OBSTACLE_INSERTION"
    IRRELEVANT_OBSTACLE = "IRRELEVANT_OBSTACLE"


class ExpectedPairRelationship(str, Enum):
    """Expected relationship between ground truth outcomes of child scenes in a pair."""

    FLIPPED_OUTCOME = "FLIPPED_OUTCOME"
    INVARIANT_OUTCOME = "INVARIANT_OUTCOME"


class CounterfactualSceneOutcome(BaseModel):
    """Ground truth summary of an individual scene within a counterfactual pair."""

    model_config = ConfigDict(frozen=True)

    scene_id: str
    is_visible: bool
    reason_code: VisibilityReasonCode
    diagnostic_details: str
    image_filename: str


class CounterfactualPairRecord(BaseModel):
    """
    Master metadata record for a counterfactual pair.
    Stores parent reference, child scene IDs, parameter deltas, and verified relationship.
    """

    model_config = ConfigDict(frozen=True)

    pair_id: str = Field(..., description="Unique counterfactual pair identifier")
    pair_type: CounterfactualPairType = Field(..., description="Specific type of counterfactual pair")
    parent_scene_id: str = Field(..., description="Base reference scene identifier")
    child_scene_ids: list[str] = Field(..., description="List of [scene_a_id, scene_b_id]")
    changed_parameter: str = Field(..., description="Name of the parameter modified between scenes")
    original_values: dict[str, Any] = Field(..., description="Original parameter values in Scene A")
    new_values: dict[str, Any] = Field(..., description="Modified parameter values in Scene B")
    ground_truth_outcomes: dict[str, CounterfactualSceneOutcome] = Field(
        ..., description="Ground truth outcome records for each child scene"
    )
    expected_relationship: ExpectedPairRelationship = Field(
        ..., description="Expected relationship: FLIPPED_OUTCOME or INVARIANT_OUTCOME"
    )
    relationship_verified: bool = Field(
        default=True, description="Whether the oracle verified the expected relationship"
    )
    difference_summary: str = Field(
        ..., description="Human-readable summary of what changed and its physical optical effect"
    )


class CounterfactualPairBundle(BaseModel):
    """
    Complete bundle for a counterfactual pair containing all artifacts for both scenes.
    """

    pair_id: str
    pair_type: CounterfactualPairType
    pair_record: CounterfactualPairRecord
    scene_a_bundle: VisibilityTaskBundle
    scene_b_bundle: VisibilityTaskBundle

