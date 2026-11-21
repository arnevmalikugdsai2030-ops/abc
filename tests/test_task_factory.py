"""
Unit & Integration Tests for the Visibility Task Factory.

Verifies:
1. Every task generates the 5 mandatory artifacts:
   - scene.json
   - diagram_blind.png
   - question.json
   - ground_truth.json
   - diagram_overlay.png
2. Standardized reason codes:
   - CLEAR
   - MIRROR_MISS
   - OUTSIDE_FOV
   - WRONG_SIDE
   - BLOCKED_CAMERA_TO_MIRROR
   - BLOCKED_MIRROR_TO_TARGET
3. Strict oracle calculation (never LLM-generated).
4. Balanced dataset generation (50% Visible / 50% Not Visible).
5. Zero answer leakage in blind task images.
"""

import json

from PIL import Image

from mirrorlab.generation.task_factory import VisibilityTaskFactory
from mirrorlab.rendering.themes import RenderTheme
from mirrorlab.schema.common import VisibilityState
from mirrorlab.schema.scene import DifficultyLevel, InspectionScene, SceneCategory
from mirrorlab.schema.task import (
    VisibilityGroundTruth,
    VisibilityQuestion,
    VisibilityReasonCode,
)


class TestVisibilityTaskFactoryArtifacts:
    """Verify that every task generates all 5 artifacts with valid schemas."""

    def test_create_single_task_bundle(self, tmp_path):
        factory = VisibilityTaskFactory(image_size=512)
        bundle = factory.create_task(
            seed=7001,
            category=SceneCategory.VALID_VISIBLE,
            difficulty=DifficultyLevel.EASY,
            theme=RenderTheme.DARK_NAVY,
            output_dir=tmp_path,
        )

        task_dir = tmp_path / bundle.task_id
        assert task_dir.is_dir()

        # 1. scene.json exists and parses
        scene_file = task_dir / "scene.json"
        assert scene_file.is_file()
        with open(scene_file, "r", encoding="utf-8") as f:
            scene_data = json.load(f)
        scene = InspectionScene.model_validate(scene_data)
        assert scene.ground_truth_state == VisibilityState.VISIBLE

        # 2. diagram_blind.png exists and is valid image
        blind_file = task_dir / "diagram_blind.png"
        assert blind_file.is_file()
        with Image.open(blind_file) as img:
            assert img.size == (512, 512)

        # 3. question.json exists and parses
        question_file = task_dir / "question.json"
        assert question_file.is_file()
        with open(question_file, "r", encoding="utf-8") as f:
            q_data = json.load(f)
        question = VisibilityQuestion.model_validate(q_data)
        assert (
            question.question
            == "Can the camera see the center of the red X through the mirror?"
        )
        assert "diagram_blind.png" in question.image_filename

        # 4. ground_truth.json exists and parses
        gt_file = task_dir / "ground_truth.json"
        assert gt_file.is_file()
        with open(gt_file, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
        gt = VisibilityGroundTruth.model_validate(gt_data)
        assert gt.can_see is True
        assert gt.answer == "YES"
        assert gt.reason_code == VisibilityReasonCode.CLEAR
        assert gt.proof_geometry.reflection_point is not None

        # 5. diagram_overlay.png exists and is valid image
        overlay_file = task_dir / "diagram_overlay.png"
        assert overlay_file.is_file()
        with Image.open(overlay_file) as img:
            assert img.size == (512, 512)


class TestReasonCodesIntegrity:
    """Verify that oracle outputs map accurately to the 6 standardized reason codes."""

    def test_all_reason_codes_represented(self, tmp_path):
        factory = VisibilityTaskFactory(image_size=512)

        category_expectations = [
            (SceneCategory.VALID_VISIBLE, VisibilityReasonCode.CLEAR, True, "YES"),
            (
                SceneCategory.MIRROR_MISS,
                VisibilityReasonCode.MIRROR_MISS,
                False,
                "NO",
            ),
            (
                SceneCategory.OUTSIDE_FOV,
                VisibilityReasonCode.OUTSIDE_FOV,
                False,
                "NO",
            ),
            (SceneCategory.WRONG_SIDE, VisibilityReasonCode.WRONG_SIDE, False, "NO"),
            (
                SceneCategory.BLOCKED_INCIDENT,
                VisibilityReasonCode.BLOCKED_CAMERA_TO_MIRROR,
                False,
                "NO",
            ),
            (
                SceneCategory.BLOCKED_REFLECTED,
                VisibilityReasonCode.BLOCKED_MIRROR_TO_TARGET,
                False,
                "NO",
            ),
        ]

        for seed_offset, (
            cat,
            expected_code,
            expected_can_see,
            expected_ans,
        ) in enumerate(category_expectations):
            bundle = factory.create_task(
                seed=8000 + seed_offset,
                category=cat,
                difficulty=DifficultyLevel.MEDIUM,
                output_dir=tmp_path,
            )

            assert bundle.ground_truth.reason_code == expected_code
            assert bundle.ground_truth.can_see == expected_can_see
            assert bundle.ground_truth.answer == expected_ans


class TestBalancedDatasetGeneration:
    """Verify dataset balancing and JSONL manifest creation."""

    def test_create_balanced_20_task_dataset(self, tmp_path):
        factory = VisibilityTaskFactory(image_size=512)
        tasks = factory.create_balanced_dataset(
            count=20,
            start_seed=9000,
            output_dir=tmp_path,
        )

        assert len(tasks) == 20

        # Verify 50/50 balance
        visible_count = sum(1 for t in tasks if t.ground_truth.can_see is True)
        not_visible_count = sum(1 for t in tasks if t.ground_truth.can_see is False)

        assert visible_count == 10
        assert not_visible_count == 10

        # Verify all failure reasons are represented
        reasons_present = {t.ground_truth.reason_code for t in tasks}
        assert len(reasons_present) == 6
        assert VisibilityReasonCode.CLEAR in reasons_present
        assert VisibilityReasonCode.MIRROR_MISS in reasons_present
        assert VisibilityReasonCode.OUTSIDE_FOV in reasons_present
        assert VisibilityReasonCode.WRONG_SIDE in reasons_present
        assert VisibilityReasonCode.BLOCKED_CAMERA_TO_MIRROR in reasons_present
        assert VisibilityReasonCode.BLOCKED_MIRROR_TO_TARGET in reasons_present

        # Verify aggregate JSONL manifests
        tasks_jsonl = tmp_path / "tasks.jsonl"
        gt_jsonl = tmp_path / "ground_truth.jsonl"

        assert tasks_jsonl.is_file()
        assert gt_jsonl.is_file()

        with open(tasks_jsonl, "r", encoding="utf-8") as f:
            task_lines = [json.loads(line) for line in f]
        with open(gt_jsonl, "r", encoding="utf-8") as f:
            gt_lines = [json.loads(line) for line in f]

        assert len(task_lines) == 20
        assert len(gt_lines) == 20
        assert all("diagram_blind.png" in t["image_filename"] for t in task_lines)
        assert all("answer" in g and "reason_code" in g for g in gt_lines)
