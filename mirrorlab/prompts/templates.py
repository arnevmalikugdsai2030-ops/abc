"""
Versioned Prompt Templates for MIRRORLAB Machine-Vision Benchmark.

Defines neutral, unambiguous, zero-leakage instructional prompts for:
1. Task A: Visibility Detection
2. Task B: Obstruction Identification
3. Task C: Mirror Configuration Repair

Guarantees:
- Strict JSON output format.
- Questions are short, neutral, and unambiguous.
- Zero access to ground-truth answers, overlays, oracle derivations, or hidden labels.
- Cryptographic version tracking with SHA-256 template hashing.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel

from mirrorlab.schema.common import TaskType


class PromptMetadata(BaseModel):
    """Metadata and version tracking for an individual prompt template."""

    prompt_id: str
    task_type: TaskType
    version: str
    created_date: str
    description: str
    author: str = "MIRRORLAB Benchmark Team"


class PromptTemplate(BaseModel):
    """
    Structured, versioned prompt template with schema validation and example response.
    """

    metadata: PromptMetadata
    role_instruction: str
    legend: str
    physics_principle: str
    question: str
    checklist: list[str]
    json_schema_spec: dict[str, Any]
    example_response: dict[str, Any]

    def render(self) -> str:
        """Renders the complete instructional prompt text presented to the VLM."""
        sections = [
            self.role_instruction.strip(),
            "",
            "Apparatus Legend:",
            self.legend.strip(),
            "",
            "Physical Principle:",
            self.physics_principle.strip(),
            "",
            f"Question:\n{self.question.strip()}",
            "",
            "Analysis Checklist:",
            "\n".join(f"{idx}. {item}" for idx, item in enumerate(self.checklist, 1)),
            "",
            "Response Instructions:",
            "Respond in strict JSON format matching the schema below. Do not include markdown code fences or conversational text outside the JSON object.",
            json.dumps(self.json_schema_spec, indent=2),
        ]
        return "\n".join(sections).strip() + "\n"

    @property
    def version_hash(self) -> str:
        """SHA-256 fingerprint of the rendered prompt text for tamper-evident audit."""
        rendered = self.render().encode("utf-8")
        return hashlib.sha256(rendered).hexdigest()

    def validate_response(
        self, response_data: dict[str, Any]
    ) -> tuple[bool, str | None]:
        """Validates that a response dictionary conforms to the required keys and types."""
        req_keys = self.json_schema_spec.get("required", [])
        for k in req_keys:
            if k not in response_data:
                return False, f"Missing required key: '{k}'"

        # Check enums
        properties = self.json_schema_spec.get("properties", {})
        for prop_name, prop_spec in properties.items():
            if prop_name in response_data and "enum" in prop_spec:
                val = response_data[prop_name]
                if val not in prop_spec["enum"]:
                    return (
                        False,
                        f"Value '{val}' for '{prop_name}' not in allowed enum: {prop_spec['enum']}",
                    )

        return True, None


# ------------------------------------------------------------------------------
# 1. TASK A: VISIBILITY PROMPT (v1.0.0)
# ------------------------------------------------------------------------------
VISIBILITY_SCHEMA_V1: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["visible", "answer", "reason", "explanation"],
    "properties": {
        "visible": {
            "type": "boolean",
            "description": "True if the camera sees the center of the red X through the mirror, false otherwise",
        },
        "answer": {
            "type": "string",
            "enum": ["YES", "NO"],
            "description": "YES if visible, NO otherwise",
        },
        "reason": {
            "type": "string",
            "enum": [
                "CLEAR",
                "MIRROR_MISS",
                "OUTSIDE_FOV",
                "WRONG_SIDE",
                "BLOCKED_CAMERA_TO_MIRROR",
                "BLOCKED_MIRROR_TO_TARGET",
            ],
            "description": "Optical reason code explaining visibility state",
        },
        "explanation": {
            "type": "string",
            "description": "Step-by-step geometric reasoning",
        },
    },
    "additionalProperties": False,
}

VISIBILITY_EXAMPLE_V1: dict[str, Any] = {
    "visible": True,
    "answer": "YES",
    "reason": "CLEAR",
    "explanation": "Specular reflection ray strikes within mirror segment AB, falls within camera FOV, faces the reflective front, and is not obstructed by barriers.",
}

PROMPT_VISIBILITY_V1 = PromptTemplate(
    metadata=PromptMetadata(
        prompt_id="prompt_vis_v1",
        task_type=TaskType.TASK_A_VISIBILITY,
        version="1.0.0",
        created_date="2026-11-22",
        description="Task A: Finite specular mirror visibility and failure mode attribution.",
    ),
    role_instruction="You are an industrial machine-vision inspector analyzing a 2D optical inspection diagram.",
    legend=(
        "- CAMERA: Indicated by the camera icon with directional lens pointing along its optical axis. The translucent cone indicates its field of view (FOV).\n"
        "- MIRROR: Finite flat mirror segment between endpoints A and B. The reflective front face is marked with a vibrant cyan line. The non-reflective back face is marked with diagonal hatch lines.\n"
        "- TARGET: A red 'X' scratch defect in the cavity.\n"
        "- OBSTACLES: Opaque polygonal barriers labeled O1, O2, etc. Direct line-of-sight between camera and scratch is blocked by barrier O1, so inspection is only possible via specular reflection."
    ),
    physics_principle=(
        "Light travels in straight lines. Specular reflection occurs at the mirror surface where angle of incidence equals angle of reflection (theta_i = theta_r) relative to the mirror surface normal."
    ),
    question="Can the camera see the center of the red X through the mirror?",
    checklist=[
        "Does the specular reflection ray from camera to target strike within finite mirror segment AB?",
        "Is the reflection point on the mirror within the camera's field of view cone?",
        "Does the mirror's reflective face (cyan) face towards both camera and target?",
        "Is the optical sightline blocked by any opaque obstacle along the incident leg (camera to mirror) or reflected leg (mirror to target)?",
    ],
    json_schema_spec=VISIBILITY_SCHEMA_V1,
    example_response=VISIBILITY_EXAMPLE_V1,
)


# ------------------------------------------------------------------------------
# 2. TASK B: OBSTRUCTION PROMPT (v1.0.0)
# ------------------------------------------------------------------------------
OBSTRUCTION_SCHEMA_V1: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["blocked", "answer", "obstacle_id", "blocked_leg", "explanation"],
    "properties": {
        "blocked": {
            "type": "boolean",
            "description": "True if an obstacle physically blocks the optical path, false if completely clear",
        },
        "answer": {
            "type": "string",
            "enum": ["YES", "NO"],
            "description": "YES if blocked, NO if clear",
        },
        "obstacle_id": {
            "type": "string",
            "enum": ["O1", "O2", "O3", "NONE"],
            "description": "Label of the blocking obstacle, or NONE if unobstructed",
        },
        "blocked_leg": {
            "type": "string",
            "enum": ["CAMERA_TO_MIRROR", "MIRROR_TO_TARGET", "NONE"],
            "description": "Which segment of the optical path is blocked, or NONE if clear",
        },
        "explanation": {
            "type": "string",
            "description": "Physical explanation identifying ray-polygon intersection",
        },
    },
    "additionalProperties": False,
}

OBSTRUCTION_EXAMPLE_V1: dict[str, Any] = {
    "blocked": True,
    "answer": "YES",
    "obstacle_id": "O2",
    "blocked_leg": "CAMERA_TO_MIRROR",
    "explanation": "Barrier O2 physically intersects the incident sightline between the camera lens and mirror reflection point.",
}

PROMPT_OBSTRUCTION_V1 = PromptTemplate(
    metadata=PromptMetadata(
        prompt_id="prompt_obs_v1",
        task_type=TaskType.TASK_B_OBSTRUCTION,
        version="1.0.0",
        created_date="2026-11-22",
        description="Task B: Obstacle collision detection, blocker identification, and path segment attribution.",
    ),
    role_instruction="You are an industrial machine-vision inspector analyzing a 2D optical inspection diagram.",
    legend=(
        "- CAMERA: Indicated by the camera icon pointing along its optical axis.\n"
        "- MIRROR: Finite flat mirror segment between endpoints A and B with cyan reflective face.\n"
        "- TARGET: A red 'X' scratch defect in the cavity.\n"
        "- OBSTACLES: Opaque polygonal barriers labeled O1, O2, O3. One obstacle may block the specular reflection path, while other obstacles act as irrelevant distractors with clear margins."
    ),
    physics_principle=(
        "An optical path consists of two straight-line segments: the incident leg (camera to reflection point) and the reflected leg (reflection point to target). An obstacle blocks inspection if and only if its physical boundary intersects either path segment."
    ),
    question="Is the optical reflection path between the camera and the scratch blocked by an obstacle?",
    checklist=[
        "Identify the specular reflection point on mirror segment AB where theta_i = theta_r.",
        "Check whether any barrier (O1, O2, O3) intersects the incident leg (camera to mirror).",
        "Check whether any barrier (O1, O2, O3) intersects the reflected leg (mirror to scratch).",
        "If blocked, identify the exact obstacle label and which leg is obstructed. If unblocked, report NONE.",
    ],
    json_schema_spec=OBSTRUCTION_SCHEMA_V1,
    example_response=OBSTRUCTION_EXAMPLE_V1,
)


# ------------------------------------------------------------------------------
# 3. TASK C: REPAIR PROMPT (v1.0.0)
# ------------------------------------------------------------------------------
REPAIR_SCHEMA_V1: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["setting", "answer", "explanation"],
    "properties": {
        "setting": {
            "type": "string",
            "enum": ["A", "B", "C"],
            "description": "Selected candidate mirror setting letter (A, B, or C)",
        },
        "answer": {
            "type": "string",
            "enum": ["A", "B", "C"],
            "description": "Selected candidate mirror setting letter (A, B, or C)",
        },
        "explanation": {
            "type": "string",
            "description": "Geometric reasoning comparing candidate mirror configurations",
        },
    },
    "additionalProperties": False,
}

REPAIR_EXAMPLE_V1: dict[str, Any] = {
    "setting": "B",
    "answer": "B",
    "explanation": "Candidate setting B tilts the mirror such that specular reflection strikes within mirror bounds inside the camera FOV, while candidates A and C miss the mirror or are blocked by barrier O1.",
}

PROMPT_REPAIR_V1 = PromptTemplate(
    metadata=PromptMetadata(
        prompt_id="prompt_repair_v1",
        task_type=TaskType.TASK_C_REPAIR,
        version="1.0.0",
        created_date="2026-11-22",
        description="Task C: Mirror configuration repair and single-winner candidate selection.",
    ),
    role_instruction="You are an industrial machine-vision inspector analyzing a 2D optical inspection diagram.",
    legend=(
        "- FIXED APPARATUS: Camera, red 'X' scratch defect, and obstacles are fixed in place.\n"
        "- CANDIDATE MIRROR SETTINGS: Three alternative mirror placements are shown, clearly labeled A, B, and C in distinct colors.\n"
        "- EXACTLY ONE WINNER: Exactly one candidate setting successfully restores unobstructed visibility from camera to scratch defect. The other two settings fail (e.g. mirror miss, out of FOV, wrong reflective side, or blocked by obstacle)."
    ),
    physics_principle=(
        "Evaluate specular reflection independently for each candidate mirror setting. The valid setting must satisfy: (1) reflection ray strikes within mirror segment, (2) reflection point is within camera FOV, (3) reflective face points towards camera and target, and (4) neither path leg is blocked."
    ),
    question="Which candidate mirror setting (A, B, or C) restores unobstructed visibility from the camera to the red X scratch?",
    checklist=[
        "Evaluate candidate mirror A: does specular reflection succeed and remain unblocked?",
        "Evaluate candidate mirror B: does specular reflection succeed and remain unblocked?",
        "Evaluate candidate mirror C: does specular reflection succeed and remain unblocked?",
        "Select the single candidate setting that successfully restores sightline to the scratch.",
    ],
    json_schema_spec=REPAIR_SCHEMA_V1,
    example_response=REPAIR_EXAMPLE_V1,
)
