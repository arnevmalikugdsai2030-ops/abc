"""
Unit Tests for Supervised Fine-Tuning (SFT) Exporter.

Verifies:
1. Exporting InspectionScene to SFT format creates valid 7-step formal derivation.
2. Formats correctly for both HuggingFace and OpenAI formats.
3. Batch export produces valid JSONL file.
"""

import json
from pathlib import Path

import pytest

from mirrorlab.export.sft_exporter import SFTDatasetExporter
from mirrorlab.schema.common import (
    ObstructionReason,
    Point2D,
    Vector2D,
    VisibilityState,
)
from mirrorlab.schema.scene import (
    CameraSpec,
    DifficultyLevel,
    InspectionScene,
    MirrorSpec,
    SceneCategory,
    TargetSpec,
)


@pytest.fixture
def sample_sft_scene() -> InspectionScene:
    cam = CameraSpec(
        position=Point2D(x=200.0, y=200.0), heading_deg=65.0, fov_degrees=65.0
    )
    mirror = MirrorSpec(
        endpoint_a=Point2D(x=150.0, y=700.0),
        endpoint_b=Point2D(x=650.0, y=700.0),
        reflective_normal=Vector2D(dx=0.0, dy=-1.0),
    )
    target = TargetSpec(center=Point2D(x=500.0, y=250.0), radius_mm=8.0)
    return InspectionScene(
        scene_id="sft_test_scene_01",
        seed=101,
        category=SceneCategory.VALID_VISIBLE,
        difficulty=DifficultyLevel.STANDARD,
        ground_truth_state=VisibilityState.VISIBLE,
        ground_truth_reason=ObstructionReason.NONE,
        diagnostic_details="Clean optical path",
        camera=cam,
        mirror=mirror,
        target=target,
        obstacles=[],
        reflection_point=Point2D(x=328.57, y=700.0),
        virtual_target=Point2D(x=500.0, y=1150.0),
    )


class TestSFTExporter:
    """Tests SFT dataset generation."""

    def test_export_single_scene_reasoning_trace(
        self, sample_sft_scene: InspectionScene
    ):
        example = SFTDatasetExporter.export_scene(sample_sft_scene)
        assert example.id == "sft_test_scene_01"
        assert len(example.messages) == 3

        assistant_msg = next(m for m in example.messages if m.role == "assistant")
        content = assistant_msg.content

        # Verify all 7 steps are present
        assert "Step 1: Coordinate Extraction" in content
        assert "Step 2: Virtual Target Position Construction" in content
        assert "Step 3: Ray-Mirror Intersection" in content
        assert "Step 4: Surface Normal" in content
        assert "Step 5: Camera Field-of-View Cone Check" in content
        assert "Step 6: Two-Leg Occlusion Analysis" in content
        assert "Step 7: Conclusion & Verdict" in content
        assert '"visibility_state": "VISIBLE"' in content

    def test_huggingface_format(self, sample_sft_scene: InspectionScene):
        example = SFTDatasetExporter.export_scene(sample_sft_scene)
        hf_dict = example.to_huggingface_dict()
        assert "conversations" in hf_dict
        assert len(hf_dict["conversations"]) == 3
        assert hf_dict["conversations"][0]["from"] == "system"
        assert hf_dict["conversations"][1]["from"] == "user"
        assert hf_dict["conversations"][2]["from"] == "gpt"

    def test_openai_format(self, sample_sft_scene: InspectionScene):
        example = SFTDatasetExporter.export_scene(sample_sft_scene)
        oa_dict = example.to_openai_dict()
        assert "messages" in oa_dict
        assert len(oa_dict["messages"]) == 3
        assert oa_dict["messages"][0]["role"] == "system"

    def test_export_batch_jsonl(
        self, sample_sft_scene: InspectionScene, tmp_path: Path
    ):
        out_file = tmp_path / "sft_train.jsonl"
        count = SFTDatasetExporter.export_to_jsonl([sample_sft_scene], out_file)
        assert count == 1
        assert out_file.exists()

        with open(out_file, encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["id"] == "sft_test_scene_01"

    def test_sft_round_trip_and_non_leakage(
        self, sample_sft_scene: InspectionScene, tmp_path: Path
    ):
        """Verifies round-trip fidelity and strict non-leakage in user prompt."""
        example = SFTDatasetExporter.export_scene(sample_sft_scene)

        # 1. User message non-leakage
        user_msg = next(m for m in example.messages if m.role == "user")
        assert (
            "visible" not in user_msg.content.lower()
            or "determine whether" in user_msg.content.lower()
        )
        assert "step 1" not in user_msg.content.lower()
        assert "reflection point" not in user_msg.content.lower()

        # 2. Assistant message Step 7 JSON validity
        assistant_msg = next(m for m in example.messages if m.role == "assistant")
        assert "```json" in assistant_msg.content
        json_str = assistant_msg.content.split("```json")[1].split("```")[0].strip()
        parsed_verdict = json.loads(json_str)
        assert parsed_verdict["visibility_state"] == "VISIBLE"
        assert parsed_verdict["obstruction_reason"] == "NONE"
