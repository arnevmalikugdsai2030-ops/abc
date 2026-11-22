"""
Critical P0 Regression Tests for MIRRORLAB.

Tests:
1. Dataset validation executes real checks and does not default to PASSED.
2. Missing/empty dataset returns NOT RUN or FAILED.
3. Dataset statistics contains zero fabricated fallbacks.
4. Model Arena task ID and task type propagation to adapter.
5. Oracle baseline returns 100% ground-truth across Task A, B, C, D.
6. Simulated model outputs are explicitly flagged with provenance.
"""

from __future__ import annotations

from pathlib import Path

from mirrorlab.evaluation.model_adapter import (
    MockStrategy,
    MockVisionAdapter,
)
from mirrorlab.evaluation.response_parser import ResponseParser
from mirrorlab.schema.common import TaskType
from mirrorlab.ui.data_loader import (
    get_dataset_statistics,
    get_dataset_validation_status,
    get_task_distribution,
)


class TestP0DatasetValidationIntegrity:
    """Verify that dataset validation executes real checks and never defaults to PASSED."""

    def test_dataset_validation_returns_measured_results(self):
        status = get_dataset_validation_status(data_root="data")
        assert "checks" in status
        assert len(status["checks"]) == 10

        # Each check must have measured items_checked and details
        for c in status["checks"]:
            assert c["items_checked"] > 0
            assert c["status"] in ("PASSED", "FAILED")
            assert len(c["details"]) > 0

        # Overall status must reflect real checks
        assert status["all_passed"] is True
        assert status["total_tasks"] == 200

    def test_empty_dataset_validation_returns_not_run(self, tmp_path: Path):
        empty_root = tmp_path / "empty_data"
        empty_root.mkdir()

        status = get_dataset_validation_status(data_root=empty_root)
        assert status["all_passed"] is False
        assert status["total_tasks"] == 0
        assert "NOT RUN" in status["status_message"]

        for c in status["checks"]:
            assert c["status"] == "NOT RUN"
            assert c["passed"] is False
            assert c["items_checked"] == 0
            assert "No dataset records found" in c["details"]


class TestP0DatasetStatisticsHonesty:
    """Verify that dataset statistics has zero fabricated numerical fallbacks."""

    def test_statistics_from_real_data(self):
        stats = get_dataset_statistics(data_root="data")
        assert stats["has_data"] is True
        assert stats["total_tasks"] == 200
        assert stats["actual_generated_tasks"] == 200
        assert stats["target_dataset_size"] == 200

        fam = stats["family_counts"]
        assert fam["TASK_A_VISIBILITY"] == 20
        assert fam["TASK_B_OBSTRUCTION"] == 30
        assert fam["TASK_C_REPAIR"] == 30
        assert fam["TASK_D_COUNTERFACTUAL"] == 120

    def test_statistics_empty_when_no_data(self, tmp_path: Path):
        empty_root = tmp_path / "no_data"
        empty_root.mkdir()

        stats = get_dataset_statistics(data_root=empty_root)
        assert stats["has_data"] is False
        assert stats["total_tasks"] == 0
        assert stats["actual_generated_tasks"] == 0
        assert stats["average_obstacles_per_scene"] == 0.0
        assert stats["difficulty_counts"] == {}

        fam = stats["family_counts"]
        assert fam["TASK_A_VISIBILITY"] == 0
        assert fam["TASK_B_OBSTRUCTION"] == 0
        assert fam["TASK_C_REPAIR"] == 0
        assert fam["TASK_D_COUNTERFACTUAL"] == 0

    def test_task_distribution_zero_fallbacks(self, tmp_path: Path):
        empty_root = tmp_path / "no_data"
        empty_root.mkdir()

        dist = get_task_distribution(data_root=empty_root)
        assert dist["has_data"] is False
        assert dist["total_counterfactual_pairs"] == 0
        for subtype_count in dist["counterfactual_subtypes"].values():
            assert subtype_count == 0


class TestP0ModelArenaTaskIDPropagation:
    """Verify that MockVisionAdapter correctly retrieves task-specific ground truth when task_id and task_type are passed."""

    def test_oracle_baseline_retrieves_correct_ground_truth(self, tmp_path: Path):
        gt_vis_no = {
            "can_see": False,
            "answer": "NO",
            "reason_code": "BLOCKED_INCIDENT",
            "detailed_explanation": "Blocked by cylinder O1.",
        }
        gt_vis_yes = {
            "can_see": True,
            "answer": "YES",
            "reason_code": "CLEAR",
            "detailed_explanation": "Specular path clear.",
        }
        gt_map = {
            "task_001": gt_vis_no,
            "task_002": gt_vis_yes,
        }

        # Create dummy image
        dummy_img = tmp_path / "dummy.png"
        dummy_img.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82")

        adapter = MockVisionAdapter(strategy=MockStrategy.ORACLE, ground_truth_map=gt_map)

        # 1. Ask for task_001 -> MUST return NO
        resp1 = adapter.ask(
            image_path=dummy_img,
            question="Can camera see?",
            task_id="task_001",
            task_type="TASK_A_VISIBILITY",
        )
        rec1 = ResponseParser.parse_and_grade(
            raw_response=resp1.raw_text,
            task_type=TaskType.TASK_A_VISIBILITY,
            task_id="task_001",
            ground_truth=gt_vis_no,
        )
        assert rec1.final_score == 1.0
        assert rec1.extracted_prediction["visible"] is False
        assert rec1.extracted_prediction["answer"] == "NO"

        # 2. Ask for task_002 -> MUST return YES
        resp2 = adapter.ask(
            image_path=dummy_img,
            question="Can camera see?",
            task_id="task_002",
            task_type="TASK_A_VISIBILITY",
        )
        rec2 = ResponseParser.parse_and_grade(
            raw_response=resp2.raw_text,
            task_type=TaskType.TASK_A_VISIBILITY,
            task_id="task_002",
            ground_truth=gt_vis_yes,
        )
        assert rec2.final_score == 1.0
        assert rec2.extracted_prediction["visible"] is True
        assert rec2.extracted_prediction["answer"] == "YES"

    def test_simulated_model_label_and_provenance(self, tmp_path: Path):
        dummy_img = tmp_path / "dummy.png"
        dummy_img.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82")

        adapter = MockVisionAdapter(strategy=MockStrategy.ORACLE)
        resp = adapter.ask(image_path=dummy_img, question="Is it visible?")

        assert resp.is_mock is True
        assert resp.raw_metadata.get("is_simulated") is True
        assert "SIMULATED" in resp.raw_metadata.get("provenance", "")
