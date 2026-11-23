"""
Unit Tests for Failure Taxonomy & Error Classification.

Verifies:
1. All 14 canonical failure categories are recognized.
2. Accurate classification of instruction-following, malformed, orientation, and leg errors.
3. Accurate identification of correct answers with invalid reasoning.
4. Robust counterfactual pair discrepancy classification.
"""

from mirrorlab.evaluation.failure_taxonomy import (
    FailureCategory,
    FailureTaxonomyClassifier,
)


class TestFailureTaxonomyClassifier:
    """Verifies deterministic classification of VLM failures."""

    def test_all_14_categories_present(self):
        categories = list(FailureCategory)
        assert len(categories) == 14
        assert FailureCategory.OBJECT_DETECTION_FAILURE in categories
        assert FailureCategory.MIRROR_IDENTIFICATION_FAILURE in categories
        assert FailureCategory.GEOMETRIC_ORIENTATION_FAILURE in categories
        assert FailureCategory.VIRTUAL_IMAGE_REASONING_FAILURE in categories
        assert FailureCategory.REFLECTION_PATH_ERROR in categories
        assert FailureCategory.OCCLUSION_LEG_ERROR in categories
        assert FailureCategory.SPATIAL_TRACKING_ERROR in categories
        assert FailureCategory.COUNTERFACTUAL_INCONSISTENCY in categories
        assert FailureCategory.NUMERICAL_CALCULATION_ERROR in categories
        assert FailureCategory.INSTRUCTION_FOLLOWING_FAILURE in categories
        assert FailureCategory.INVALID_MALFORMED_RESPONSE in categories
        assert FailureCategory.CORRECT_ANSWER_INVALID_REASONING in categories
        assert FailureCategory.AMBIGUOUS_GROUND_TRUTH in categories
        assert FailureCategory.UNKNOWN_OTHER in categories

    def test_classify_malformed_response(self):
        diag = FailureTaxonomyClassifier.classify_single_task(
            task_id="T001",
            ground_truth={"visibility_state": "VISIBLE"},
            predicted_response=None,
            status="PARSE_ERROR",
        )
        assert diag.category == FailureCategory.INVALID_MALFORMED_RESPONSE
        assert diag.confidence == 1.0

    def test_classify_instruction_following_failure(self):
        diag = FailureTaxonomyClassifier.classify_single_task(
            task_id="T002",
            ground_truth={"visibility_state": "VISIBLE"},
            predicted_response={"visibility_state": "MAYBE_OR_PERHAPS"},
        )
        assert diag.category == FailureCategory.INSTRUCTION_FOLLOWING_FAILURE
        assert diag.confidence >= 0.90

    def test_classify_correct_answer_invalid_reasoning(self):
        diag = FailureTaxonomyClassifier.classify_single_task(
            task_id="T003",
            ground_truth={"visibility_state": "VISIBLE", "ground_truth_reason": "NONE"},
            predicted_response={
                "visibility_state": "VISIBLE",
                "reasoning": "The target is visible because the ray hits obstacle O1 and is occluded by it.",
            },
        )
        assert diag.category == FailureCategory.CORRECT_ANSWER_INVALID_REASONING

    def test_classify_infinite_plane_fallacy(self):
        diag = FailureTaxonomyClassifier.classify_single_task(
            task_id="T004",
            ground_truth={
                "visibility_state": "NOT_VISIBLE",
                "ground_truth_reason": "MIRROR_MISS",
            },
            predicted_response={
                "visibility_state": "VISIBLE",
                "reasoning": "The ray reflects off the infinite mirror line even though it misses mirror endpoint.",
            },
        )
        assert diag.category == FailureCategory.VIRTUAL_IMAGE_REASONING_FAILURE

    def test_classify_occlusion_leg_error(self):
        diag = FailureTaxonomyClassifier.classify_single_task(
            task_id="T005",
            ground_truth={
                "visibility_state": "NOT_VISIBLE",
                "ground_truth_reason": "CAMERA_TO_MIRROR",
            },
            predicted_response={
                "visibility_state": "NOT_VISIBLE",
                "reasoning": "The ray is blocked on the reflected path from mirror to target.",
            },
        )
        assert diag.category == FailureCategory.OCCLUSION_LEG_ERROR

    def test_classify_counterfactual_pair_failure(self):
        diag = FailureTaxonomyClassifier.classify_counterfactual_failure(
            pair_id="PAIR_001",
            pair_type="cosmetic_control",
            scene_a_pred={"visibility_state": "VISIBLE"},
            scene_b_pred={"visibility_state": "NOT_VISIBLE"},
            expected_relationship="PRESERVED",
        )
        assert diag.category == FailureCategory.COUNTERFACTUAL_INCONSISTENCY
        assert "PRESERVED" in diag.explanation
