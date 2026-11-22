"""
MIRRORLAB Strict Response Parser & Evaluator.

Handles:
- Valid JSON.
- JSON inside Markdown code fences (```json ... ``` and ``` ... ```).
- Missing required keys.
- Invalid data types (e.g. non-boolean for visible/blocked).
- Unknown obstacle labels (not in O1, O2, O3, NONE).
- Invalid repair settings (not in A, B, C).
- Empty responses.
- API failures.
- Malformed JSON.

Strict Scientific Invariant:
Never silently convert an invalid answer into a correct one.

Stores:
1. Raw response
2. Parsed response
3. Validation errors
4. Final score
5. Error classification
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from mirrorlab.schema.common import TaskType


class ErrorCategory(str, Enum):
    """Categorization of parsing, validation, and evaluation failures."""

    NONE = "NONE"  # Valid format and correct answer
    WRONG_ANSWER = "WRONG_ANSWER"  # Valid format, but incorrect geometric answer
    API_FAILURE = "API_FAILURE"  # Upstream API threw error, timeout, or rate-limit
    EMPTY_RESPONSE = "EMPTY_RESPONSE"  # Empty string or whitespace returned by model
    MALFORMED_JSON = "MALFORMED_JSON"  # JSON syntax error / non-JSON output
    MISSING_REQUIRED_KEY = "MISSING_REQUIRED_KEY"  # Missing required schema key
    INVALID_DATA_TYPE = "INVALID_DATA_TYPE"  # Key present but wrong type (e.g. string for boolean)
    UNKNOWN_OBSTACLE_LABEL = "UNKNOWN_OBSTACLE_LABEL"  # Obstacle label not in {O1, O2, O3, NONE}
    INVALID_REPAIR_SETTING = "INVALID_REPAIR_SETTING"  # Setting not in {A, B, C}
    INVALID_REASON_CODE = "INVALID_REASON_CODE"  # Reason code not in allowed enum


class ParsedEvaluationRecord(BaseModel):
    """
    Standardized evaluation record storing all 5 required elements:
    1. Raw response
    2. Parsed response
    3. Validation errors
    4. Final score (1.0 for correct, 0.0 for incorrect/invalid)
    5. Error classification
    """

    raw_response: str
    parsed_response: dict[str, Any] | None = None
    validation_errors: list[str] = Field(default_factory=list)
    final_score: float = 0.0
    error_classification: ErrorCategory = ErrorCategory.NONE

    task_id: str = "unknown"
    task_type: TaskType = TaskType.TASK_A_VISIBILITY
    is_valid_format: bool = False
    extracted_prediction: Any = None
    detailed_explanation: str | None = None


class ResponseParser:
    """
    Strict response parser enforcing exact schema constraints and zero silent conversions.
    """

    VALID_REASON_CODES: ClassVar[set[str]] = {
        "CLEAR",
        "MIRROR_MISS",
        "OUTSIDE_FOV",
        "WRONG_SIDE",
        "BLOCKED_CAMERA_TO_MIRROR",
        "BLOCKED_MIRROR_TO_TARGET",
    }
    VALID_OBSTACLE_LABELS: ClassVar[set[str]] = {"O1", "O2", "O3", "NONE"}
    VALID_BLOCKED_LEGS: ClassVar[set[str]] = {"CAMERA_TO_MIRROR", "MIRROR_TO_TARGET", "NONE"}
    VALID_REPAIR_SETTINGS: ClassVar[set[str]] = {"A", "B", "C"}

    @classmethod
    def parse_and_grade(
        cls,
        raw_response: str | None,
        task_type: TaskType | str = TaskType.TASK_A_VISIBILITY,
        task_id: str = "unknown",
        ground_truth: dict[str, Any] | None = None,
        api_error: str | None = None,
    ) -> ParsedEvaluationRecord:
        """
        Parses raw model output, validates schema rules, and grades against ground truth.

        Parameters
        ----------
        raw_response:
            Raw text string returned by the VLM.
        task_type:
            Type of benchmark task (Visibility, Obstruction, Repair).
        task_id:
            Unique identifier of the task.
        ground_truth:
            Optional ground-truth dictionary for automated grading.
        api_error:
            Optional upstream API error message if call failed.

        Returns
        -------
        ParsedEvaluationRecord:
            Container with raw_response, parsed_response, validation_errors,
            final_score, and error_classification.
        """
        tt = TaskType(task_type) if isinstance(task_type, str) else task_type
        raw_str = raw_response or ""

        # 1. Check for upstream API failures
        if api_error:
            return ParsedEvaluationRecord(
                raw_response=raw_str,
                parsed_response=None,
                validation_errors=[f"API Failure: {api_error}"],
                final_score=0.0,
                error_classification=ErrorCategory.API_FAILURE,
                task_id=task_id,
                task_type=tt,
                is_valid_format=False,
            )

        # 2. Check for empty responses
        if not raw_str or not raw_str.strip():
            return ParsedEvaluationRecord(
                raw_response=raw_str,
                parsed_response=None,
                validation_errors=["Model returned empty or whitespace response."],
                final_score=0.0,
                error_classification=ErrorCategory.EMPTY_RESPONSE,
                task_id=task_id,
                task_type=tt,
                is_valid_format=False,
            )

        # 3. Extract JSON object (direct, markdown code block, or curly braces)
        parsed_dict, json_err = cls._extract_json_dict(raw_str.strip())
        if parsed_dict is None:
            return ParsedEvaluationRecord(
                raw_response=raw_str,
                parsed_response=None,
                validation_errors=[f"Malformed JSON: {json_err}"],
                final_score=0.0,
                error_classification=ErrorCategory.MALFORMED_JSON,
                task_id=task_id,
                task_type=tt,
                is_valid_format=False,
                detailed_explanation=raw_str[:300],
            )

        # 4. Task-Specific Schema & Type Validation
        errors: list[str] = []
        err_cat: ErrorCategory = ErrorCategory.NONE
        extracted_pred: Any = None
        explanation: str | None = None

        if tt in (TaskType.TASK_A_VISIBILITY, TaskType.TASK_D_COUNTERFACTUAL):
            err_cat, extracted_pred, explanation = cls._validate_visibility_schema(
                parsed_dict, errors
            )
        elif tt == TaskType.TASK_B_OBSTRUCTION:
            err_cat, extracted_pred, explanation = cls._validate_obstruction_schema(
                parsed_dict, errors
            )
        elif tt == TaskType.TASK_C_REPAIR:
            err_cat, extracted_pred, explanation = cls._validate_repair_schema(
                parsed_dict, errors
            )

        # If any validation errors occurred, reject immediately (NO SILENT CONVERSIONS)
        if errors:
            return ParsedEvaluationRecord(
                raw_response=raw_str,
                parsed_response=parsed_dict,
                validation_errors=errors,
                final_score=0.0,
                error_classification=err_cat,
                task_id=task_id,
                task_type=tt,
                is_valid_format=False,
                extracted_prediction=extracted_pred,
                detailed_explanation=explanation,
            )

        # 5. Grade against Ground Truth (if provided)
        score = 0.0
        final_cat = ErrorCategory.NONE

        if ground_truth is not None:
            is_correct = cls._grade_prediction(tt, extracted_pred, ground_truth)
            if is_correct:
                score = 1.0
                final_cat = ErrorCategory.NONE
            else:
                score = 0.0
                final_cat = ErrorCategory.WRONG_ANSWER

        return ParsedEvaluationRecord(
            raw_response=raw_str,
            parsed_response=parsed_dict,
            validation_errors=[],
            final_score=score,
            error_classification=final_cat,
            task_id=task_id,
            task_type=tt,
            is_valid_format=True,
            extracted_prediction=extracted_pred,
            detailed_explanation=explanation,
        )

    @classmethod
    def _extract_json_dict(cls, text: str) -> tuple[dict[str, Any] | None, str | None]:
        """Tolerantly extracts a JSON dictionary from raw text or markdown code fences."""
        # Attempt 1: Strict direct parse
        try:
            val = json.loads(text)
            if isinstance(val, dict):
                return val, None
            return None, f"JSON root must be an object/dict, got {type(val).__name__}"
        except json.JSONDecodeError as e:
            direct_err = str(e)

        # Attempt 2: Markdown code fence ```json ... ```
        fence_matches = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        for block in fence_matches:
            try:
                val = json.loads(block.strip())
                if isinstance(val, dict):
                    return val, None
            except json.JSONDecodeError:
                continue

        # Attempt 3: Outermost balanced curly braces
        brace_candidate = cls._find_outermost_json_object(text)
        if brace_candidate:
            try:
                val = json.loads(brace_candidate)
                if isinstance(val, dict):
                    return val, None
            except json.JSONDecodeError:
                pass

        return None, direct_err

    @staticmethod
    def _find_outermost_json_object(text: str) -> str | None:
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False

        for idx in range(start, len(text)):
            c = text[idx]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if not in_string:
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : idx + 1]
        return None

    @classmethod
    def _validate_visibility_schema(
        cls, data: dict[str, Any], errors: list[str]
    ) -> tuple[ErrorCategory, Any, str | None]:
        """Validates Task A: Visibility prompt requirements."""
        norm_keys = {k.lower(): (k, v) for k, v in data.items()}

        # 1. Check required visible key
        visible_entry = norm_keys.get("visible") or norm_keys.get("can_see")
        if not visible_entry:
            # Fall back to answer if present, but MUST be boolean or YES/NO
            ans_entry = norm_keys.get("answer")
            if not ans_entry:
                errors.append("Missing required key: 'visible' (or 'answer')")
                return ErrorCategory.MISSING_REQUIRED_KEY, None, None
            orig_k, ans_val = ans_entry
            if isinstance(ans_val, bool):
                vis_bool = ans_val
            elif isinstance(ans_val, str) and ans_val.upper() in ("YES", "NO"):
                vis_bool = ans_val.upper() == "YES"
            else:
                errors.append(f"Field '{orig_k}' must be 'YES' or 'NO', got {ans_val!r}")
                return ErrorCategory.INVALID_DATA_TYPE, None, None
        else:
            orig_k, vis_val = visible_entry
            if not isinstance(vis_val, bool):
                errors.append(f"Field '{orig_k}' must be a boolean (true or false), got {type(vis_val).__name__} ({vis_val!r})")
                return ErrorCategory.INVALID_DATA_TYPE, None, None
            vis_bool = vis_val

        # 2. Check reason code
        reason_entry = norm_keys.get("reason") or norm_keys.get("reason_code")
        reason_code = None
        if reason_entry:
            orig_rk, r_val = reason_entry
            if not isinstance(r_val, str):
                errors.append(f"Field '{orig_rk}' must be a string, got {type(r_val).__name__}")
                return ErrorCategory.INVALID_DATA_TYPE, None, None
            u_reason = r_val.strip().upper()
            if u_reason not in cls.VALID_REASON_CODES:
                errors.append(f"Invalid reason code '{r_val}'. Allowed: {sorted(cls.VALID_REASON_CODES)}")
                return ErrorCategory.INVALID_REASON_CODE, None, None
            reason_code = u_reason

        explanation = str(norm_keys.get("explanation", (None, ""))[1])
        prediction = {
            "visible": vis_bool,
            "answer": "YES" if vis_bool else "NO",
            "reason": reason_code,
        }
        return ErrorCategory.NONE, prediction, explanation

    @classmethod
    def _validate_obstruction_schema(
        cls, data: dict[str, Any], errors: list[str]
    ) -> tuple[ErrorCategory, Any, str | None]:
        """Validates Task B: Obstruction prompt requirements."""
        norm_keys = {k.lower(): (k, v) for k, v in data.items()}

        # 1. Check required blocked key
        blocked_entry = norm_keys.get("blocked") or norm_keys.get("is_blocked")
        if not blocked_entry:
            ans_entry = norm_keys.get("answer")
            if not ans_entry:
                errors.append("Missing required key: 'blocked' (or 'answer')")
                return ErrorCategory.MISSING_REQUIRED_KEY, None, None
            orig_k, ans_val = ans_entry
            if isinstance(ans_val, bool):
                blocked_bool = ans_val
            elif isinstance(ans_val, str) and ans_val.upper() in ("YES", "NO"):
                blocked_bool = ans_val.upper() == "YES"
            else:
                errors.append(f"Field '{orig_k}' must be 'YES' or 'NO', got {ans_val!r}")
                return ErrorCategory.INVALID_DATA_TYPE, None, None
        else:
            orig_k, blk_val = blocked_entry
            if not isinstance(blk_val, bool):
                errors.append(f"Field '{orig_k}' must be a boolean (true or false), got {type(blk_val).__name__} ({blk_val!r})")
                return ErrorCategory.INVALID_DATA_TYPE, None, None
            blocked_bool = blk_val

        # 2. Check obstacle_id
        obs_entry = norm_keys.get("obstacle_id") or norm_keys.get("blocker_label") or norm_keys.get("blocker")
        if not obs_entry:
            errors.append("Missing required key: 'obstacle_id'")
            return ErrorCategory.MISSING_REQUIRED_KEY, None, None
        orig_ok, obs_val = obs_entry
        if not isinstance(obs_val, str):
            errors.append(f"Field '{orig_ok}' must be a string label, got {type(obs_val).__name__}")
            return ErrorCategory.INVALID_DATA_TYPE, None, None
        u_obs = obs_val.strip().upper()
        if u_obs not in cls.VALID_OBSTACLE_LABELS:
            errors.append(f"Unknown obstacle label '{obs_val}'. Allowed: {sorted(cls.VALID_OBSTACLE_LABELS)}")
            return ErrorCategory.UNKNOWN_OBSTACLE_LABEL, None, None

        # 3. Check blocked_leg
        leg_entry = norm_keys.get("blocked_leg") or norm_keys.get("blocked_segment") or norm_keys.get("segment")
        if not leg_entry:
            errors.append("Missing required key: 'blocked_leg'")
            return ErrorCategory.MISSING_REQUIRED_KEY, None, None
        orig_lk, leg_val = leg_entry
        if not isinstance(leg_val, str):
            errors.append(f"Field '{orig_lk}' must be a string, got {type(leg_val).__name__}")
            return ErrorCategory.INVALID_DATA_TYPE, None, None
        u_leg = leg_val.strip().upper()
        if u_leg not in cls.VALID_BLOCKED_LEGS:
            errors.append(f"Invalid blocked_leg '{leg_val}'. Allowed: {sorted(cls.VALID_BLOCKED_LEGS)}")
            return ErrorCategory.INVALID_DATA_TYPE, None, None

        explanation = str(norm_keys.get("explanation", (None, ""))[1])
        prediction = {
            "blocked": blocked_bool,
            "answer": "YES" if blocked_bool else "NO",
            "obstacle_id": u_obs,
            "blocked_leg": u_leg,
        }
        return ErrorCategory.NONE, prediction, explanation

    @classmethod
    def _validate_repair_schema(
        cls, data: dict[str, Any], errors: list[str]
    ) -> tuple[ErrorCategory, Any, str | None]:
        """Validates Task C: Repair prompt requirements."""
        norm_keys = {k.lower(): (k, v) for k, v in data.items()}

        # Check setting key
        setting_entry = (
            norm_keys.get("setting")
            or norm_keys.get("answer")
            or norm_keys.get("selected_candidate")
        )
        if not setting_entry:
            errors.append("Missing required key: 'setting' (or 'answer')")
            return ErrorCategory.MISSING_REQUIRED_KEY, None, None

        orig_sk, s_val = setting_entry
        if not isinstance(s_val, str):
            errors.append(f"Field '{orig_sk}' must be a string candidate letter ('A', 'B', or 'C'), got {type(s_val).__name__}")
            return ErrorCategory.INVALID_DATA_TYPE, None, None

        u_setting = s_val.strip().upper()
        if u_setting not in cls.VALID_REPAIR_SETTINGS:
            errors.append(f"Invalid repair setting '{s_val}'. Allowed: {sorted(cls.VALID_REPAIR_SETTINGS)}")
            return ErrorCategory.INVALID_REPAIR_SETTING, None, None

        explanation = str(norm_keys.get("explanation", (None, ""))[1])
        prediction = {
            "setting": u_setting,
            "answer": u_setting,
        }
        return ErrorCategory.NONE, prediction, explanation

    @classmethod
    def _grade_prediction(
        cls, task_type: TaskType, prediction: dict[str, Any], ground_truth: dict[str, Any]
    ) -> bool:
        """Compares validated prediction dictionary against mathematical ground truth."""
        if task_type in (TaskType.TASK_A_VISIBILITY, TaskType.TASK_D_COUNTERFACTUAL):
            gt_ans = str(ground_truth.get("answer", "")).upper()
            p_ans = str(prediction.get("answer", "")).upper()
            return p_ans == gt_ans

        elif task_type == TaskType.TASK_B_OBSTRUCTION:
            gt_blocked = ground_truth.get("is_blocked")
            p_blocked = prediction.get("blocked")
            if gt_blocked != p_blocked:
                return False

            raw_gt_blocker = ground_truth.get("blocking_obstacle_label") or ground_truth.get("blocker_label")
            gt_blocker = str(raw_gt_blocker).upper() if raw_gt_blocker else "NONE"
            p_blocker = str(prediction.get("obstacle_id", "")).upper()

            gt_leg = str(ground_truth.get("blocked_segment") or "NONE").upper()
            p_leg = str(prediction.get("blocked_leg", "")).upper()

            return (gt_blocker == p_blocker) and (gt_leg == p_leg)

        elif task_type == TaskType.TASK_C_REPAIR:
            gt_winner = str(ground_truth.get("correct_candidate", "")).upper()
            p_setting = str(prediction.get("setting", "")).upper()
            return p_setting == gt_winner

        return False
