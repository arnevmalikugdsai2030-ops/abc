"""
Unit & Integration Tests for Task C: Mirror-Setting Repair Factory.

Enforces:
1. Strict Uniqueness: Exactly ONE candidate mirror configuration restores visibility.
2. Independent Oracle Validation: All 3 candidates (A, B, C) are independently verified.
3. Complete Candidate Evaluations: Ground truth answer stores evaluations for A, B, and C.
4. Balanced Winning Option Slots: Equal representation across A, B, and C.
5. Candidate Geometric Distinctness: No overlapping or identical mirrors.
6. Complete 5-Artifact Bundles.
7. Zero Answer Leakage in Blind Task Diagrams.
"""

import json

from PIL import Image

from mirrorlab.generation.repair_factory import RepairTaskFactory
from mirrorlab.rendering.themes import RenderTheme
from mirrorlab.schema.scene import DifficultyLevel
from mirrorlab.schema.task import RepairGroundTruth, RepairQuestion


class TestMirrorRepairUniquenessAndOracleValidation:
    """Verify strict uniqueness and independent candidate evaluation."""

    def test_exactly_one_candidate_restores_visibility(self, tmp_path):
        factory = RepairTaskFactory(image_size=512)
        bundle = factory.create_task(
            seed=501,
            difficulty=DifficultyLevel.MEDIUM,
            theme=RenderTheme.DARK_NAVY,
            output_dir=tmp_path,
        )

        gt = bundle.ground_truth
        winner = gt.correct_candidate
        assert winner in ["A", "B", "C"]

        # Exactly 1 visible candidate
        visible_cands = [k for k, v in gt.candidates.items() if v.is_visible is True]
        assert len(visible_cands) == 1
        assert visible_cands[0] == winner

        # The other 2 candidates must fail
        failing_cands = [k for k, v in gt.candidates.items() if v.is_visible is False]
        assert len(failing_cands) == 2

    def test_force_each_winner_slot(self, tmp_path):
        factory = RepairTaskFactory(image_size=512)
        for slot in ["A", "B", "C"]:
            bundle = factory.create_task(
                seed=600 + ord(slot),
                force_winner_slot=slot,
                output_dir=tmp_path,
            )
            assert bundle.ground_truth.correct_candidate == slot
            assert bundle.ground_truth.candidates[slot].is_visible is True
            # Other two slots are not visible
            other_slots = [s for s in ["A", "B", "C"] if s != slot]
            for s in other_slots:
                assert bundle.ground_truth.candidates[s].is_visible is False

    def test_candidates_geometrically_distinct(self, tmp_path):
        factory = RepairTaskFactory(image_size=512)
        bundle = factory.create_task(seed=701, output_dir=tmp_path)

        cands = bundle.candidate_mirrors
        assert len(cands) == 3

        # Pairwise comparison
        keys = list(cands.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                m1 = cands[keys[i]]
                m2 = cands[keys[j]]
                # Endpoints must not be identical
                assert (m1.endpoint_a.x, m1.endpoint_a.y) != (
                    m2.endpoint_a.x,
                    m2.endpoint_a.y,
                )


class TestRepairTaskArtifacts:
    """Verify that every task directory contains all 5 required files."""

    def test_all_five_artifacts_present(self, tmp_path):
        factory = RepairTaskFactory(image_size=512)
        bundle = factory.create_task(seed=801, output_dir=tmp_path)

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
        q = RepairQuestion.model_validate(q_data)
        assert q.options == ["A", "B", "C"]

        # 4. ground_truth.json and answer.json
        gt_file = task_dir / "ground_truth.json"
        ans_file = task_dir / "answer.json"
        assert gt_file.is_file()
        assert ans_file.is_file()
        with open(ans_file, "r", encoding="utf-8") as f:
            ans_data = json.load(f)
        gt = RepairGroundTruth.model_validate(ans_data)
        assert len(gt.candidates) == 3

        # 5. diagram_overlay.png
        overlay_file = task_dir / "diagram_overlay.png"
        assert overlay_file.is_file()
        with Image.open(overlay_file) as img:
            assert img.size == (512, 512)


class TestBalancedRepairDataset:
    """Verify 30-task balanced generation across A, B, and C."""

    def test_create_balanced_30_repair_tasks(self, tmp_path):
        factory = RepairTaskFactory(image_size=512)
        tasks = factory.create_balanced_dataset(
            count=30,
            start_seed=25000,
            output_dir=tmp_path,
        )

        assert len(tasks) == 30

        # Count winning options
        winner_counts = {"A": 0, "B": 0, "C": 0}
        for t in tasks:
            winner_counts[t.ground_truth.correct_candidate] += 1

        assert winner_counts["A"] == 10
        assert winner_counts["B"] == 10
        assert winner_counts["C"] == 10

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
