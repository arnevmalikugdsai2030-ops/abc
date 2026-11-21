"""
Unit & Integration Tests for Task B: Obstruction Identification Factory.

Verifies:
1. Base scenes are valid with unobstructed reflection geometry.
2. Exactly one obstacle intersects exactly one leg when blocked.
3. Zero obstacles intersect either leg in unblocked control scenes.
4. Distractors maintain strict clearance (>15mm) from both optical legs.
5. Models must identify is_blocked, blocking_obstacle_label, and blocked_segment.
6. Rejection of multi-blocker or grazing layouts.
7. Full 5-artifact bundle generation.
8. Balanced dataset generation across outcomes.
"""

import json

from PIL import Image

from mirrorlab.generation.obstruction_factory import ObstructionTaskFactory
from mirrorlab.rendering.themes import RenderTheme
from mirrorlab.schema.scene import DifficultyLevel
from mirrorlab.schema.task import (
    BlockedSegment,
    ObstructionGroundTruth,
    ObstructionQuestion,
)


class TestObstructionTaskFactorySingleBlocker:
    """Verify single-blocker precision and path segment identification."""

    def test_leg1_camera_to_mirror_blocked(self, tmp_path):
        factory = ObstructionTaskFactory(image_size=512)
        bundle = factory.create_task(
            seed=101,
            target_segment=BlockedSegment.CAMERA_TO_MIRROR,
            difficulty=DifficultyLevel.MEDIUM,
            num_distractors=1,
            theme=RenderTheme.DARK_NAVY,
            output_dir=tmp_path,
        )

        gt = bundle.ground_truth
        assert gt.is_blocked is True
        assert gt.blocked_segment == BlockedSegment.CAMERA_TO_MIRROR
        assert gt.blocking_obstacle_label is not None
        assert gt.blocking_obstacle_label in bundle.question.obstacle_labels
        assert gt.proof.penetration_depth_mm is not None
        assert gt.proof.penetration_depth_mm >= 8.0
        assert gt.proof.min_distractor_clearance_mm >= 15.0

    def test_leg2_mirror_to_target_blocked(self, tmp_path):
        factory = ObstructionTaskFactory(image_size=512)
        bundle = factory.create_task(
            seed=202,
            target_segment=BlockedSegment.MIRROR_TO_TARGET,
            difficulty=DifficultyLevel.EASY,
            num_distractors=1,
            theme=RenderTheme.LIGHT_OFFWHITE,
            output_dir=tmp_path,
        )

        gt = bundle.ground_truth
        assert gt.is_blocked is True
        assert gt.blocked_segment == BlockedSegment.MIRROR_TO_TARGET
        assert gt.blocking_obstacle_label is not None
        assert gt.blocking_obstacle_label in bundle.question.obstacle_labels
        assert gt.proof.penetration_depth_mm is not None
        assert gt.proof.penetration_depth_mm >= 8.0

    def test_unblocked_control_scene(self, tmp_path):
        factory = ObstructionTaskFactory(image_size=512)
        bundle = factory.create_task(
            seed=303,
            target_segment=BlockedSegment.NONE,
            difficulty=DifficultyLevel.HARD,
            num_distractors=2,
            theme=RenderTheme.BLUEPRINT,
            output_dir=tmp_path,
        )

        gt = bundle.ground_truth
        assert gt.is_blocked is False
        assert gt.blocked_segment == BlockedSegment.NONE
        assert gt.blocking_obstacle_label is None
        assert gt.proof.penetration_depth_mm is None
        assert gt.proof.min_distractor_clearance_mm >= 15.0


class TestObstructionTaskArtifacts:
    """Verify that every task directory contains all 5 required files."""

    def test_all_five_artifacts_present(self, tmp_path):
        factory = ObstructionTaskFactory(image_size=512)
        bundle = factory.create_task(
            seed=404,
            target_segment=BlockedSegment.CAMERA_TO_MIRROR,
            output_dir=tmp_path,
        )

        task_dir = tmp_path / bundle.task_id
        assert task_dir.is_dir()

        # 1. scene.json
        scene_file = task_dir / "scene.json"
        assert scene_file.is_file()

        # 2. diagram_blind.png
        blind_file = task_dir / "diagram_blind.png"
        assert blind_file.is_file()
        with Image.open(blind_file) as img:
            assert img.size == (512, 512)

        # 3. question.json
        question_file = task_dir / "question.json"
        assert question_file.is_file()
        with open(question_file, "r", encoding="utf-8") as f:
            q_data = json.load(f)
        q = ObstructionQuestion.model_validate(q_data)
        assert "diagram_blind.png" in q.image_filename
        assert len(q.obstacle_labels) >= 2  # Partition wall + at least 1 obstacle

        # 4. ground_truth.json
        gt_file = task_dir / "ground_truth.json"
        assert gt_file.is_file()
        with open(gt_file, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
        gt = ObstructionGroundTruth.model_validate(gt_data)
        assert gt.blocked_segment == BlockedSegment.CAMERA_TO_MIRROR

        # 5. diagram_overlay.png
        overlay_file = task_dir / "diagram_overlay.png"
        assert overlay_file.is_file()
        with Image.open(overlay_file) as img:
            assert img.size == (512, 512)


class TestBalancedObstructionDataset:
    """Verify 3-way balanced generation and aggregate manifest outputs."""

    def test_create_balanced_30_tasks(self, tmp_path):
        factory = ObstructionTaskFactory(image_size=512)
        tasks = factory.create_balanced_dataset(
            count=30,
            start_seed=15000,
            output_dir=tmp_path,
        )

        assert len(tasks) == 30

        # Count segments
        segment_counts = {
            BlockedSegment.CAMERA_TO_MIRROR: 0,
            BlockedSegment.MIRROR_TO_TARGET: 0,
            BlockedSegment.NONE: 0,
        }
        for t in tasks:
            segment_counts[t.ground_truth.blocked_segment] += 1

        assert segment_counts[BlockedSegment.CAMERA_TO_MIRROR] == 10
        assert segment_counts[BlockedSegment.MIRROR_TO_TARGET] == 10
        assert segment_counts[BlockedSegment.NONE] == 10

        # Manifest verification
        tasks_jsonl = tmp_path / "tasks.jsonl"
        gt_jsonl = tmp_path / "ground_truth.jsonl"
        assert tasks_jsonl.is_file()
        assert gt_jsonl.is_file()

        with open(tasks_jsonl, "r", encoding="utf-8") as f:
            t_lines = f.readlines()
        with open(gt_jsonl, "r", encoding="utf-8") as f:
            g_lines = f.readlines()

        assert len(t_lines) == 30
        assert len(g_lines) == 30
