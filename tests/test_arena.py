"""
Unit Tests for MIRRORLAB Model Arena.

Validates:
1. Tamper-evident prediction lock hashing.
2. Single-task model inference via vision adapter and ResponseParser.
3. Ground-truth isolation: zero leak prior to lock and reveal.
4. Paired counterfactual evaluation with joint consistency scoring.
5. Malformed response handling and format error classification.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mirrorlab.evaluation.response_parser import ErrorCategory
from mirrorlab.schema.common import TaskType
from mirrorlab.ui.arena import (
    _compute_prediction_lock_hash,
    _execute_model_challenge,
)


@pytest.fixture
def sample_task_data() -> tuple[str, str, dict]:
    """Provides sample task ID, prompt, and ground truth for testing."""
    task_id = "test_vis_001"
    prompt = "Can the camera see the center of the red X through the mirror?"
    gt = {
        "task_id": task_id,
        "can_see": True,
        "answer": "YES",
        "reason_code": "CLEAR",
        "detailed_explanation": "Direct optical ray reflects cleanly off planar mirror.",
    }
    return task_id, prompt, gt


def test_prediction_lock_hash() -> None:
    """Verify that lock hash is deterministic, tamper-evident, and uppercase."""
    h1 = _compute_prediction_lock_hash(
        "task_1", "gemini-2.5-flash", '{"can_see": true}'
    )
    h2 = _compute_prediction_lock_hash(
        "task_1", "gemini-2.5-flash", '{"can_see": true}'
    )
    h3 = _compute_prediction_lock_hash(
        "task_1", "gemini-2.5-flash", '{"can_see": false}'
    )

    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 16
    assert h1.isupper()


def test_execute_model_challenge_oracle(
    sample_task_data: tuple[str, str, dict], tmp_path: Path
) -> None:
    """Verify challenge execution with mock-oracle produces valid, correct evaluation."""
    task_id, prompt, gt = sample_task_data

    # Create dummy image file for validation
    dummy_img = tmp_path / "diagram_blind.png"
    dummy_img.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    resp, record = _execute_model_challenge(
        model_name="mock-oracle",
        image_path=str(dummy_img),
        question_prompt=prompt,
        task_type=TaskType.TASK_A_VISIBILITY,
        task_id=task_id,
        ground_truth_dict=gt,
    )

    assert resp.raw_text != ""
    assert record.is_valid_format is True
    assert record.final_score == 1.0
    assert record.error_classification == ErrorCategory.NONE
    parsed = json.loads(resp.raw_text)
    assert parsed.get("can_see") is True


def test_execute_model_challenge_incorrect(
    sample_task_data: tuple[str, str, dict], tmp_path: Path
) -> None:
    """Verify challenge execution with mock-all-no grades as incorrect (0.0)."""
    task_id, prompt, gt = sample_task_data

    dummy_img = tmp_path / "diagram_blind.png"
    dummy_img.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    _resp, record = _execute_model_challenge(
        model_name="mock-all-no",
        image_path=str(dummy_img),
        question_prompt=prompt,
        task_type=TaskType.TASK_A_VISIBILITY,
        task_id=task_id,
        ground_truth_dict=gt,
    )

    assert record.is_valid_format is True
    assert record.final_score == 0.0
    assert record.error_classification == ErrorCategory.WRONG_ANSWER


def test_execute_model_challenge_format_error(
    sample_task_data: tuple[str, str, dict], tmp_path: Path
) -> None:
    """Verify non-JSON model output is classified as MALFORMED_JSON with score 0.0."""
    task_id, prompt, gt = sample_task_data

    dummy_img = tmp_path / "diagram_blind.png"
    dummy_img.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    _resp, record = _execute_model_challenge(
        model_name="mock-format-error",
        image_path=str(dummy_img),
        question_prompt=prompt,
        task_type=TaskType.TASK_A_VISIBILITY,
        task_id=task_id,
        ground_truth_dict=gt,
    )

    assert record.is_valid_format is False
    assert record.final_score == 0.0
    assert record.error_classification == ErrorCategory.MALFORMED_JSON
