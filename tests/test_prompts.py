"""
Unit and Integration Tests for MIRRORLAB Versioned Prompt Library.

Tests:
1. Verification of separate prompts for Visibility, Obstruction, and Repair.
2. Correct required fields in prompt schemas:
   - Visibility: 'visible' (bool), 'reason' (reason code)
   - Obstruction: 'blocked' (bool), 'obstacle_id', 'blocked_leg'
   - Repair: 'setting' (A/B/C)
3. Strict JSON output instructions and schema definitions.
4. Prompt neutrality and zero-leakage verification:
   - No answer JSON keywords, no geometry overlays, no oracle derivations,
     no test-set labels or hidden coordinates.
5. Version tracking & SHA-256 fingerprint verification:
   - Version hashing changes if template text is altered.
   - Registry retrieves by version and resolves latest.
6. Schema validation for example responses.
7. StructuredParser compatibility with all prompt example responses.
"""

from __future__ import annotations

import json
from typing import ClassVar

import pytest

from mirrorlab.evaluation.structured_parser import StructuredParser
from mirrorlab.prompts import (
    PROMPT_REGISTRY,
    PromptMetadata,
    get_prompt,
    get_prompt_schema,
    list_all_prompts,
    render_prompt,
)
from mirrorlab.schema.common import TaskType


class TestVersionedPromptLibraryStructure:
    """Verifies that all three benchmark task prompts exist and have correct metadata."""

    def test_three_separate_prompts_exist_in_registry(self):
        prompts = list_all_prompts()
        assert len(prompts) >= 3

        task_types = {p["task_type"] for p in prompts}
        assert TaskType.TASK_A_VISIBILITY.value in task_types
        assert TaskType.TASK_B_OBSTRUCTION.value in task_types
        assert TaskType.TASK_C_REPAIR.value in task_types

    def test_visibility_prompt_v1_metadata(self):
        prompt = get_prompt(TaskType.TASK_A_VISIBILITY)
        assert prompt.metadata.prompt_id == "prompt_vis_v1"
        assert prompt.metadata.version == "1.0.0"
        assert "visible" in prompt.question.lower() or "see" in prompt.question.lower()
        assert prompt.version_hash is not None
        assert len(prompt.version_hash) == 64  # SHA-256 hex string

    def test_obstruction_prompt_v1_metadata(self):
        prompt = get_prompt(TaskType.TASK_B_OBSTRUCTION)
        assert prompt.metadata.prompt_id == "prompt_obs_v1"
        assert prompt.metadata.version == "1.0.0"
        assert "blocked" in prompt.question.lower() or "obstacle" in prompt.question.lower()

    def test_repair_prompt_v1_metadata(self):
        prompt = get_prompt(TaskType.TASK_C_REPAIR)
        assert prompt.metadata.prompt_id == "prompt_repair_v1"
        assert prompt.metadata.version == "1.0.0"
        assert "setting" in prompt.question.lower() or "mirror" in prompt.question.lower()


class TestPromptSchemaRequirements:
    """Verifies that each prompt asks for the exact required fields."""

    def test_visibility_prompt_schema_fields(self):
        schema = get_prompt_schema(TaskType.TASK_A_VISIBILITY)
        assert "visible" in schema["required"]
        assert "reason" in schema["required"]
        assert schema["properties"]["visible"]["type"] == "boolean"

        # Check allowed reason codes
        allowed_reasons = set(schema["properties"]["reason"]["enum"])
        expected_reasons = {
            "CLEAR",
            "MIRROR_MISS",
            "OUTSIDE_FOV",
            "WRONG_SIDE",
            "BLOCKED_CAMERA_TO_MIRROR",
            "BLOCKED_MIRROR_TO_TARGET",
        }
        assert allowed_reasons == expected_reasons

    def test_obstruction_prompt_schema_fields(self):
        schema = get_prompt_schema(TaskType.TASK_B_OBSTRUCTION)
        assert "blocked" in schema["required"]
        assert "obstacle_id" in schema["required"]
        assert "blocked_leg" in schema["required"]
        assert schema["properties"]["blocked"]["type"] == "boolean"

        allowed_blockers = set(schema["properties"]["obstacle_id"]["enum"])
        assert {"O1", "O2", "O3", "NONE"}.issubset(allowed_blockers)

        allowed_legs = set(schema["properties"]["blocked_leg"]["enum"])
        assert {"CAMERA_TO_MIRROR", "MIRROR_TO_TARGET", "NONE"}.issubset(allowed_legs)

    def test_repair_prompt_schema_fields(self):
        schema = get_prompt_schema(TaskType.TASK_C_REPAIR)
        assert "setting" in schema["required"]
        allowed_settings = set(schema["properties"]["setting"]["enum"])
        assert allowed_settings == {"A", "B", "C"}


class TestPromptNeutralityAndZeroLeakage:
    """Guarantees prompts are neutral and do not leak ground truth, overlays, or derivations."""

    LEAKAGE_FORBIDDEN_TERMS: ClassVar[list[str]] = [
        "the correct answer is",
        "ground truth is",
        "oracle derivation",
        "diagram_overlay.png",
        "proof_geometry",
        "t_mirror =",
        "is_boundary = true",
        "winner is candidate",
        "secret",
        "test set label",
    ]

    @pytest.mark.parametrize(
        "task_type",
        [
            TaskType.TASK_A_VISIBILITY,
            TaskType.TASK_B_OBSTRUCTION,
            TaskType.TASK_C_REPAIR,
        ],
    )
    def test_prompt_has_no_leakage(self, task_type: TaskType):
        rendered = render_prompt(task_type).lower()
        for forbidden in self.LEAKAGE_FORBIDDEN_TERMS:
            assert forbidden not in rendered, f"Forbidden leaked term '{forbidden}' found in {task_type.value} prompt"

    def test_questions_are_concise_and_neutral(self):
        for task_type in [TaskType.TASK_A_VISIBILITY, TaskType.TASK_B_OBSTRUCTION, TaskType.TASK_C_REPAIR]:
            p = get_prompt(task_type)
            # Question must end with question mark
            assert p.question.strip().endswith("?")
            # Question must be under 150 characters
            assert len(p.question) < 150


class TestVersionTrackingAndHashing:
    """Verifies cryptographic version tracking and hash sensitivity."""

    def test_version_hash_is_deterministic(self):
        p1 = get_prompt(TaskType.TASK_A_VISIBILITY)
        p2 = get_prompt(TaskType.TASK_A_VISIBILITY)
        assert p1.version_hash == p2.version_hash

    def test_modifying_prompt_text_changes_hash(self):
        p = get_prompt(TaskType.TASK_A_VISIBILITY)
        original_hash = p.version_hash

        # Create modified clone
        modified = p.model_copy(deep=True)
        modified.question = "Can the camera clearly see the red defect scratch?"
        new_hash = modified.version_hash

        assert original_hash != new_hash

    def test_registry_supports_multiple_versions(self):
        v1 = get_prompt(TaskType.TASK_A_VISIBILITY, version="1.0.0")

        # Register v1.1.0 with updated description
        v1_1 = v1.model_copy(deep=True)
        v1_1.metadata = PromptMetadata(
            prompt_id="prompt_vis_v1_1",
            task_type=TaskType.TASK_A_VISIBILITY,
            version="1.1.0",
            created_date="2026-11-23",
            description="Task A: Visibility with enhanced CAD legend.",
        )
        PROMPT_REGISTRY.register(v1_1)

        # Retrieve explicit v1.0.0
        retrieved_v1 = PROMPT_REGISTRY.get(TaskType.TASK_A_VISIBILITY, version="1.0.0")
        assert retrieved_v1.metadata.version == "1.0.0"

        # Retrieve latest (should be 1.1.0)
        retrieved_latest = PROMPT_REGISTRY.get(TaskType.TASK_A_VISIBILITY)
        assert retrieved_latest.metadata.version == "1.1.0"

        # Restore registry state
        PROMPT_REGISTRY.register(v1)


class TestExampleResponsesAndParserCompatibility:
    """Verifies that example responses validate against their schema and parse in StructuredParser."""

    def test_visibility_example_response_validates(self):
        p = get_prompt(TaskType.TASK_A_VISIBILITY, version="1.0.0")
        ex = p.example_response
        valid, err = p.validate_response(ex)
        assert valid is True, f"Visibility example failed validation: {err}"

        # Test with StructuredParser
        json_str = json.dumps(ex)
        parsed = StructuredParser.parse(json_str, task_type="TASK_A_VISIBILITY")
        assert parsed.is_valid_format is True
        assert parsed.can_see is True
        assert parsed.prediction == "YES"
        assert parsed.reason_code == "CLEAR"

    def test_obstruction_example_response_validates(self):
        p = get_prompt(TaskType.TASK_B_OBSTRUCTION, version="1.0.0")
        ex = p.example_response
        valid, err = p.validate_response(ex)
        assert valid is True, f"Obstruction example failed validation: {err}"

        # Test with StructuredParser
        json_str = json.dumps(ex)
        parsed = StructuredParser.parse(json_str, task_type="TASK_B_OBSTRUCTION")
        assert parsed.is_valid_format is True
        assert parsed.is_blocked is True
        assert parsed.prediction == "YES"
        assert parsed.blocker_label == "O2"
        assert parsed.blocked_segment == "CAMERA_TO_MIRROR"

    def test_repair_example_response_validates(self):
        p = get_prompt(TaskType.TASK_C_REPAIR, version="1.0.0")
        ex = p.example_response
        valid, err = p.validate_response(ex)
        assert valid is True, f"Repair example failed validation: {err}"

        # Test with StructuredParser
        json_str = json.dumps(ex)
        parsed = StructuredParser.parse(json_str, task_type="TASK_C_REPAIR")
        assert parsed.is_valid_format is True
        assert parsed.selected_candidate == "B"
        assert parsed.prediction == "B"

    def test_invalid_schema_is_rejected(self):
        p = get_prompt(TaskType.TASK_A_VISIBILITY, version="1.0.0")
        bad_response = {
            "visible": True,
            "answer": "YES",
            "reason": "INVALID_REASON_CODE",  # Invalid enum value
            "explanation": "Test explanation",
        }
        valid, err = p.validate_response(bad_response)
        assert valid is False
        assert "not in allowed enum" in err
