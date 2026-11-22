"""
Unit and Integration Tests for MIRRORLAB Strict Response Parser.

Exhaustively verifies every failure case:
1. Valid JSON.
2. JSON inside Markdown code fences.
3. Missing required keys.
4. Invalid data types.
5. Unknown obstacle labels.
6. Invalid repair settings.
7. Empty responses.
8. API failures.
9. Malformed JSON.
10. Invariant: Never silently convert an invalid answer into a correct one.
11. Full storage of: raw response, parsed response, validation errors, final score, error classification.
"""

from __future__ import annotations

from mirrorlab.evaluation.response_parser import (
    ErrorCategory,
    ParsedEvaluationRecord,
    ResponseParser,
)
from mirrorlab.schema.common import TaskType


class TestResponseParserValidInputs:
    """Verifies parsing and grading of valid model outputs."""

    def test_valid_json_visibility_correct(self):
        raw = '{"visible": true, "answer": "YES", "reason": "CLEAR", "explanation": "Path is clear"}'
        gt = {"answer": "YES", "reason_code": "CLEAR"}
        record = ResponseParser.parse_and_grade(
            raw_response=raw,
            task_type=TaskType.TASK_A_VISIBILITY,
            task_id="t1",
            ground_truth=gt,
        )

        assert record.is_valid_format is True
        assert record.final_score == 1.0
        assert record.error_classification == ErrorCategory.NONE
        assert record.validation_errors == []
        assert record.raw_response == raw
        assert isinstance(record.parsed_response, dict)
        assert record.parsed_response["visible"] is True

    def test_valid_json_visibility_wrong_answer(self):
        raw = '{"visible": false, "answer": "NO", "reason": "OUTSIDE_FOV", "explanation": "Target out of FOV"}'
        gt = {"answer": "YES", "reason_code": "CLEAR"}
        record = ResponseParser.parse_and_grade(
            raw_response=raw,
            task_type=TaskType.TASK_A_VISIBILITY,
            task_id="t2",
            ground_truth=gt,
        )

        assert record.is_valid_format is True
        assert record.final_score == 0.0
        assert record.error_classification == ErrorCategory.WRONG_ANSWER
        assert record.validation_errors == []

    def test_json_inside_markdown_code_fences(self):
        raw = (
            "Here is the machine-vision analysis result:\n\n"
            "```json\n"
            "{\n"
            '  "blocked": true,\n'
            '  "answer": "YES",\n'
            '  "obstacle_id": "O2",\n'
            '  "blocked_leg": "CAMERA_TO_MIRROR",\n'
            '  "explanation": "Blocked by barrier O2"\n'
            "}\n"
            "```\n"
            "End of report."
        )
        gt = {
            "is_blocked": True,
            "blocking_obstacle_label": "O2",
            "blocked_segment": "CAMERA_TO_MIRROR",
        }
        record = ResponseParser.parse_and_grade(
            raw_response=raw,
            task_type=TaskType.TASK_B_OBSTRUCTION,
            task_id="t3",
            ground_truth=gt,
        )

        assert record.is_valid_format is True
        assert record.final_score == 1.0
        assert record.error_classification == ErrorCategory.NONE
        assert record.parsed_response["obstacle_id"] == "O2"


class TestResponseParserFailureCases:
    """Verifies that all 8 distinct failure modes are caught and classified."""

    def test_empty_response(self):
        for empty_val in ["", "   ", "\n\t\n", None]:
            record = ResponseParser.parse_and_grade(
                raw_response=empty_val,
                task_type=TaskType.TASK_A_VISIBILITY,
                task_id="fail_empty",
            )
            assert record.is_valid_format is False
            assert record.final_score == 0.0
            assert record.error_classification == ErrorCategory.EMPTY_RESPONSE
            assert len(record.validation_errors) == 1
            assert "empty" in record.validation_errors[0].lower()
            assert record.parsed_response is None

    def test_api_failure(self):
        record = ResponseParser.parse_and_grade(
            raw_response="",
            task_type=TaskType.TASK_A_VISIBILITY,
            task_id="fail_api",
            api_error="503 Service Unavailable: Model overloaded",
        )
        assert record.is_valid_format is False
        assert record.final_score == 0.0
        assert record.error_classification == ErrorCategory.API_FAILURE
        assert "503" in record.validation_errors[0]

    def test_malformed_json_syntax(self):
        broken_json = '{"visible": true, "answer": "YES", reason: broken'
        record = ResponseParser.parse_and_grade(
            raw_response=broken_json,
            task_type=TaskType.TASK_A_VISIBILITY,
            task_id="fail_malformed",
        )
        assert record.is_valid_format is False
        assert record.final_score == 0.0
        assert record.error_classification == ErrorCategory.MALFORMED_JSON
        assert len(record.validation_errors) == 1
        assert "Malformed JSON" in record.validation_errors[0]

    def test_missing_required_keys_task_a(self):
        # Missing 'visible' or 'answer'
        raw = '{"reason": "CLEAR", "explanation": "Only explanation and reason"}'
        record = ResponseParser.parse_and_grade(
            raw_response=raw,
            task_type=TaskType.TASK_A_VISIBILITY,
            task_id="fail_missing_a",
        )
        assert record.is_valid_format is False
        assert record.final_score == 0.0
        assert record.error_classification == ErrorCategory.MISSING_REQUIRED_KEY
        assert any("Missing required key" in err for err in record.validation_errors)

    def test_missing_required_keys_task_b(self):
        # Missing 'obstacle_id'
        raw = '{"blocked": true, "blocked_leg": "CAMERA_TO_MIRROR"}'
        record = ResponseParser.parse_and_grade(
            raw_response=raw,
            task_type=TaskType.TASK_B_OBSTRUCTION,
            task_id="fail_missing_b",
        )
        assert record.is_valid_format is False
        assert record.final_score == 0.0
        assert record.error_classification == ErrorCategory.MISSING_REQUIRED_KEY
        assert any("obstacle_id" in err for err in record.validation_errors)

    def test_invalid_data_types_boolean(self):
        # 'visible' passed as string "true" instead of boolean true
        raw = '{"visible": "true", "answer": "YES", "reason": "CLEAR"}'
        record = ResponseParser.parse_and_grade(
            raw_response=raw,
            task_type=TaskType.TASK_A_VISIBILITY,
            task_id="fail_type_bool",
        )
        assert record.is_valid_format is False
        assert record.final_score == 0.0
        assert record.error_classification == ErrorCategory.INVALID_DATA_TYPE
        assert any("boolean" in err for err in record.validation_errors)

    def test_invalid_data_types_non_string_setting(self):
        # 'setting' passed as integer 1 instead of string "A"
        raw = '{"setting": 1, "explanation": "Setting 1"}'
        record = ResponseParser.parse_and_grade(
            raw_response=raw,
            task_type=TaskType.TASK_C_REPAIR,
            task_id="fail_type_setting",
        )
        assert record.is_valid_format is False
        assert record.final_score == 0.0
        assert record.error_classification == ErrorCategory.INVALID_DATA_TYPE
        assert any("string" in err for err in record.validation_errors)

    def test_unknown_obstacle_label(self):
        # Obstacle label O9 is not in {O1, O2, O3, NONE}
        raw = '{"blocked": true, "obstacle_id": "O9", "blocked_leg": "CAMERA_TO_MIRROR"}'
        record = ResponseParser.parse_and_grade(
            raw_response=raw,
            task_type=TaskType.TASK_B_OBSTRUCTION,
            task_id="fail_unknown_obs",
        )
        assert record.is_valid_format is False
        assert record.final_score == 0.0
        assert record.error_classification == ErrorCategory.UNKNOWN_OBSTACLE_LABEL
        assert any("Unknown obstacle label 'O9'" in err for err in record.validation_errors)

    def test_invalid_repair_setting(self):
        # Candidate setting D does not exist in {A, B, C}
        raw = '{"setting": "D", "explanation": "Candidate D restores sight"}'
        record = ResponseParser.parse_and_grade(
            raw_response=raw,
            task_type=TaskType.TASK_C_REPAIR,
            task_id="fail_invalid_setting",
        )
        assert record.is_valid_format is False
        assert record.final_score == 0.0
        assert record.error_classification == ErrorCategory.INVALID_REPAIR_SETTING
        assert any("Invalid repair setting 'D'" in err for err in record.validation_errors)

    def test_invalid_reason_code(self):
        # Invalid reason code
        raw = '{"visible": true, "answer": "YES", "reason": "BECAUSE_IT_BOUNCED"}'
        record = ResponseParser.parse_and_grade(
            raw_response=raw,
            task_type=TaskType.TASK_A_VISIBILITY,
            task_id="fail_reason",
        )
        assert record.is_valid_format is False
        assert record.final_score == 0.0
        assert record.error_classification == ErrorCategory.INVALID_REASON_CODE


class TestNoSilentConversionInvariant:
    """Verifies that invalid or malformed outputs are NEVER silently converted to correct answers."""

    def test_invalid_setting_with_matching_answer_is_never_converted(self):
        # Ground truth is "A", model provides setting: "Z", answer: "A"
        raw = '{"setting": "Z", "answer": "A"}'
        gt = {"correct_candidate": "A"}
        record = ResponseParser.parse_and_grade(
            raw_response=raw,
            task_type=TaskType.TASK_C_REPAIR,
            task_id="no_silent_conv",
            ground_truth=gt,
        )
        # MUST NOT award 1.0 point! Setting Z is invalid!
        assert record.final_score == 0.0
        assert record.is_valid_format is False
        assert record.error_classification == ErrorCategory.INVALID_REPAIR_SETTING

    def test_unknown_obstacle_is_never_silently_converted(self):
        # Ground truth is O2, model outputs O99
        raw = '{"blocked": true, "obstacle_id": "O99", "blocked_leg": "CAMERA_TO_MIRROR"}'
        gt = {
            "is_blocked": True,
            "blocking_obstacle_label": "O2",
            "blocked_segment": "CAMERA_TO_MIRROR",
        }
        record = ResponseParser.parse_and_grade(
            raw_response=raw,
            task_type=TaskType.TASK_B_OBSTRUCTION,
            task_id="no_silent_conv_obs",
            ground_truth=gt,
        )
        assert record.final_score == 0.0
        assert record.is_valid_format is False
        assert record.error_classification == ErrorCategory.UNKNOWN_OBSTACLE_LABEL


class TestRecordContainerCompleteness:
    """Verifies that all 5 required elements are strictly stored in ParsedEvaluationRecord."""

    def test_all_five_fields_stored(self):
        raw = '{"visible": true, "answer": "YES", "reason": "CLEAR"}'
        record: ParsedEvaluationRecord = ResponseParser.parse_and_grade(
            raw_response=raw,
            task_type=TaskType.TASK_A_VISIBILITY,
            task_id="t_store",
            ground_truth={"answer": "YES"},
        )

        # 1. Raw response
        assert record.raw_response == raw
        # 2. Parsed response
        assert isinstance(record.parsed_response, dict)
        # 3. Validation errors
        assert isinstance(record.validation_errors, list)
        # 4. Final score
        assert record.final_score in (0.0, 1.0)
        # 5. Error classification
        assert isinstance(record.error_classification, ErrorCategory)
