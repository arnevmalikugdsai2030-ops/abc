"""
Deterministic Failure Taxonomy & Error Classifier for MIRRORLAB.

Implements an exhaustive 14-category failure classification system for
VLM spatial, physical, optical, and formatting failures:

1. OBJECT_DETECTION_FAILURE: Model failed to identify target, mirror, or obstacles in the scene.
2. MIRROR_IDENTIFICATION_FAILURE: Model failed to locate or orient the mirror surface.
3. GEOMETRIC_ORIENTATION_FAILURE: Model assumed reflective property on non-reflective backside or confused normal.
4. VIRTUAL_IMAGE_REASONING_FAILURE: Model failed virtual camera reflection (e.g. infinite plane fallacy, looking behind finite boundary).
5. REFLECTION_PATH_ERROR: Model computed or estimated wrong reflection point on the mirror surface.
6. OCCLUSION_LEG_ERROR: Model confused incident leg (camera->mirror) vs reflected leg (mirror->target) obstruction.
7. SPATIAL_TRACKING_ERROR: Model confused left/right, near/far, or relative coordinates in diagram.
8. COUNTERFACTUAL_INCONSISTENCY: Model gave contradictory answers on paired scenes under minimal perturbation.
9. NUMERICAL_CALCULATION_ERROR: Model attempted math or ray vector geometry but made arithmetic/trigonometric errors.
10. INSTRUCTION_FOLLOWING_FAILURE: Model failed to output requested JSON schema or required format.
11. INVALID_MALFORMED_RESPONSE: Empty, corrupted, or unparseable output.
12. CORRECT_ANSWER_INVALID_REASONING: Guessing correctly on boolean visibility while rationalization contradicts physics.
13. AMBIGUOUS_GROUND_TRUTH: Scene was borderline (quarantined or flagged).
14. UNKNOWN_OTHER: Unclassified anomaly.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class FailureCategory(str, Enum):
    """The 14 canonical failure modes for specular optical reasoning."""

    OBJECT_DETECTION_FAILURE = "object_detection_failure"
    MIRROR_IDENTIFICATION_FAILURE = "mirror_identification_failure"
    GEOMETRIC_ORIENTATION_FAILURE = "geometric_orientation_failure"
    VIRTUAL_IMAGE_REASONING_FAILURE = "virtual_image_reasoning_failure"
    REFLECTION_PATH_ERROR = "reflection_path_error"
    OCCLUSION_LEG_ERROR = "occlusion_leg_error"
    SPATIAL_TRACKING_ERROR = "spatial_tracking_error"
    COUNTERFACTUAL_INCONSISTENCY = "counterfactual_inconsistency"
    NUMERICAL_CALCULATION_ERROR = "numerical_calculation_error"
    INSTRUCTION_FOLLOWING_FAILURE = "instruction_following_failure"
    INVALID_MALFORMED_RESPONSE = "invalid_malformed_response"
    CORRECT_ANSWER_INVALID_REASONING = "correct_answer_invalid_reasoning"
    AMBIGUOUS_GROUND_TRUTH = "ambiguous_ground_truth"
    UNKNOWN_OTHER = "unknown_other"


class FailureDiagnostic(BaseModel):
    """Detailed diagnostic breakdown of an evaluation error."""

    task_id: str
    category: FailureCategory
    confidence: float = Field(ge=0.0, le=1.0, description="Classification confidence")
    explanation: str
    evidence_text: str | None = None
    ground_truth_state: str
    predicted_state: str | None = None
    suggested_remedy: str


class FailureTaxonomyClassifier:
    """
    Classifies VLM reasoning failures into the 14-category deterministic taxonomy.
    """

    @classmethod
    def classify_single_task(
        cls,
        task_id: str,
        ground_truth: dict[str, Any],
        predicted_response: dict[str, Any] | None,
        raw_text: str = "",
        status: str = "COMPLETED",
    ) -> FailureDiagnostic:
        """
        Determines the primary failure category for a single task prediction.
        """
        gt_state = (
            ground_truth.get("visibility_state")
            or ground_truth.get("ground_truth_state")
            or "UNKNOWN"
        )
        if hasattr(gt_state, "value"):
            gt_state = gt_state.value
        gt_state = str(gt_state).upper()

        gt_reason = (
            ground_truth.get("ground_truth_reason")
            or ground_truth.get("obstruction_reason")
            or "NONE"
        )
        if hasattr(gt_reason, "value"):
            gt_reason = gt_reason.value
        gt_reason = str(gt_reason).upper()

        # 1. Check for parse error or empty/corrupted response
        if status in ("PARSE_ERROR", "CORRUPTED", "TIMEOUT") or not predicted_response:
            return FailureDiagnostic(
                task_id=task_id,
                category=FailureCategory.INVALID_MALFORMED_RESPONSE,
                confidence=1.0,
                explanation=f"Response was unparseable or failed execution status: {status}.",
                evidence_text=raw_text[:200] if raw_text else None,
                ground_truth_state=gt_state,
                predicted_state=None,
                suggested_remedy="Inspect VLM temperature, format prompt, or API token limit.",
            )

        pred_state = predicted_response.get(
            "visibility_state"
        ) or predicted_response.get("prediction")
        if hasattr(pred_state, "value"):
            pred_state = pred_state.value
        pred_state = str(pred_state).upper() if pred_state else None

        reasoning = str(predicted_response.get("reasoning", "") or raw_text).lower()

        # 2. Check for Instruction Following Failure
        if pred_state not in ("VISIBLE", "NOT_VISIBLE", "UNKNOWN", "BLOCKED"):
            return FailureDiagnostic(
                task_id=task_id,
                category=FailureCategory.INSTRUCTION_FOLLOWING_FAILURE,
                confidence=0.95,
                explanation="Model did not output a valid visibility_state boolean or expected enum choice.",
                evidence_text=str(pred_state),
                ground_truth_state=gt_state,
                predicted_state=pred_state,
                suggested_remedy="Enforce JSON schema constrained decoding or few-shot formatting examples.",
            )

        # Normalize state
        is_gt_visible = "VISIBLE" in gt_state and "NOT" not in gt_state
        is_pred_visible = "VISIBLE" in str(pred_state) and "NOT" not in str(pred_state)

        # 3. Correct answer check - check for Correct Answer With Invalid Reasoning
        if (
            is_gt_visible == is_pred_visible
            and is_gt_visible
            and any(
                term in reasoning
                for term in [
                    "blocked by obstacle",
                    "occluded by",
                    "hits obstacle",
                    "obstructed",
                ]
            )
        ):
            return FailureDiagnostic(
                task_id=task_id,
                category=FailureCategory.CORRECT_ANSWER_INVALID_REASONING,
                confidence=0.85,
                explanation="Model guessed the correct visibility state, but its rationale asserts the optical path was obstructed.",
                evidence_text=reasoning[:200],
                ground_truth_state=gt_state,
                predicted_state=pred_state,
                suggested_remedy="Implement Chain-of-Thought verification to ensure reasoning steps match conclusions.",
            )

        # If answer is WRONG: determine root cause
        # 4. Mirror Identification or Geometric Orientation Failure
        if (
            "back" in reasoning
            or "normal" in reasoning
            or "opposite side" in reasoning
            or "wrong side" in reasoning
        ) and ("WRONG_SIDE" in gt_reason or "BACK_OF_MIRROR" in gt_reason):
            # Actual reason was wrong side, but model thought it was visible
            return FailureDiagnostic(
                task_id=task_id,
                category=FailureCategory.GEOMETRIC_ORIENTATION_FAILURE,
                confidence=0.90,
                explanation="Model failed to identify that the ray strikes the non-reflective rear of the mirror.",
                evidence_text=reasoning[:200],
                ground_truth_state=gt_state,
                predicted_state=pred_state,
                suggested_remedy="Train on 2D hatch-mark reflection diagrams emphasizing mirror normal direction.",
            )

        # 5. Virtual Image / Infinite Plane Fallacy
        if (
            "MIRROR_MISS" in gt_reason
            or "MISSED_MIRROR" in gt_reason
            or "APERTURE" in gt_reason
            or "misses mirror" in reasoning
        ):
            return FailureDiagnostic(
                task_id=task_id,
                category=FailureCategory.VIRTUAL_IMAGE_REASONING_FAILURE,
                confidence=0.90,
                explanation="Model suffered from the Infinite Plane Fallacy, assuming reflection off an extended collinear line beyond finite mirror bounds.",
                evidence_text=reasoning[:200],
                ground_truth_state=gt_state,
                predicted_state=pred_state,
                suggested_remedy="Introduce boundary-clipping verification to enforce finite mirror segment endpoints.",
            )

        # 6. Occlusion Leg Confusion
        if (
            ("CAMERA_TO_MIRROR" in gt_reason or "BLOCKED_INCIDENT" in gt_reason)
            and ("mirror to target" in reasoning or "reflected" in reasoning)
        ) or (
            ("MIRROR_TO_TARGET" in gt_reason or "BLOCKED_REFLECTED" in gt_reason)
            and ("camera to mirror" in reasoning or "incident" in reasoning)
        ):
            return FailureDiagnostic(
                task_id=task_id,
                category=FailureCategory.OCCLUSION_LEG_ERROR,
                confidence=0.92,
                explanation=f"Model confused incident and reflected ray legs (Ground truth: {gt_reason}).",
                evidence_text=reasoning[:200],
                ground_truth_state=gt_state,
                predicted_state=pred_state,
                suggested_remedy="Decompose reasoning prompt into Leg 1 (Incident) and Leg 2 (Reflected) sub-checks.",
            )

        # 7. Numerical calculation error
        if any(
            tok in reasoning
            for tok in ["degrees", "atan", "theta", "law of reflection", "coordinate"]
        ) and re.search(r"\d+\.?\d*\s*(deg|degrees|mm)", reasoning):
            return FailureDiagnostic(
                task_id=task_id,
                category=FailureCategory.NUMERICAL_CALCULATION_ERROR,
                confidence=0.80,
                explanation="Model attempted explicit geometric calculation but arrived at an erroneous reflection vector or angle.",
                evidence_text=reasoning[:200],
                ground_truth_state=gt_state,
                predicted_state=pred_state,
                suggested_remedy="Provide tool-use assistance or Python code-execution sandbox for optical raytracing.",
            )

        # 8. Spatial Tracking Error
        if any(
            term in reasoning for term in ["left", "right", "above", "below", "behind"]
        ) and ("outside_fov" in gt_reason.lower() or "fov" in gt_reason.lower()):
            return FailureDiagnostic(
                task_id=task_id,
                category=FailureCategory.SPATIAL_TRACKING_ERROR,
                confidence=0.82,
                explanation="Model misjudged spatial relationship or camera field-of-view cone boundaries.",
                evidence_text=reasoning[:200],
                ground_truth_state=gt_state,
                predicted_state=pred_state,
                suggested_remedy="Prompt model to project frustum rays prior to evaluating mirror reflection.",
            )

        # Default fallback
        return FailureDiagnostic(
            task_id=task_id,
            category=FailureCategory.UNKNOWN_OTHER,
            confidence=0.50,
            explanation=f"Prediction ({pred_state}) contradicted ground truth ({gt_state}) without a specific pattern trigger.",
            evidence_text=reasoning[:200] if reasoning else None,
            ground_truth_state=gt_state,
            predicted_state=pred_state,
            suggested_remedy="Analyze manual attention maps or conduct qualitative review.",
        )

    @classmethod
    def classify_counterfactual_failure(
        cls,
        pair_id: str,
        pair_type: str,
        scene_a_pred: dict[str, Any],
        scene_b_pred: dict[str, Any],
        expected_relationship: str,
    ) -> FailureDiagnostic:
        """
        Classifies errors occurring across counterfactual pairs (Task D).
        """
        return FailureDiagnostic(
            task_id=pair_id,
            category=FailureCategory.COUNTERFACTUAL_INCONSISTENCY,
            confidence=0.98,
            explanation=(
                f"Model violated physical invariant under minimal perturbation in {pair_type}. "
                f"Expected outcome relationship was '{expected_relationship}', but model produced inconsistent responses."
            ),
            ground_truth_state=f"Relationship: {expected_relationship}",
            predicted_state=f"A: {scene_a_pred.get('visibility_state')}, B: {scene_b_pred.get('visibility_state')}",
            suggested_remedy="Incorporate contrastive counterfactual pair fine-tuning to discourage spurious correlations.",
        )
