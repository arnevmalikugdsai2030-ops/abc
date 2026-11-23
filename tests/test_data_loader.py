"""
Unit and Integration Tests for Real Backend Data Loaders in MIRRORLAB UI.

Tests:
1. get_dataset_statistics returns real counts and split numbers.
2. get_task_distribution returns real family breakdown and counterfactual subtypes.
3. get_recent_evaluation_runs discovers actual runs on disk.
4. load_evaluation_run_details parses dual-comparison and single-model runs.
5. get_failure_analysis extracts failed tasks with diagnostic explanations.
6. get_dataset_validation_status returns verified 10-point checklist.
7. Empty state handling when directories are missing.
"""

from __future__ import annotations

from pathlib import Path

from mirrorlab.ui.data_loader import (
    get_dataset_statistics,
    get_dataset_validation_status,
    get_failure_analysis,
    get_recent_evaluation_runs,
    get_task_distribution,
    load_evaluation_run_details,
)


class TestRealBackendDataLoaders:
    """Verifies that all dashboard metrics correspond to actual backend data."""

    def test_get_dataset_statistics_real_data(self):
        stats = get_dataset_statistics(data_root="data")
        assert stats["has_data"] is True
        assert stats["total_tasks"] > 0
        assert stats["total_split_tasks"] == 350
        assert stats["split_counts"]["dev"] == 70
        assert stats["split_counts"]["val"] == 70
        assert stats["split_counts"]["test"] == 210
        assert "EASY" in stats["difficulty_counts"]
        assert stats["average_obstacles_per_scene"] > 0

    def test_get_task_distribution_real_data(self):
        dist = get_task_distribution(data_root="data")
        assert dist["has_data"] is True
        assert "TASK_A_VISIBILITY" in dist["family_counts"]
        assert "TASK_B_OBSTRUCTION" in dist["family_counts"]
        assert "TASK_C_REPAIR" in dist["family_counts"]
        assert "TASK_D_COUNTERFACTUAL" in dist["family_counts"]
        subtypes = dist["counterfactual_subtypes"]
        assert subtypes["MIRROR_ANGLE_FLIP"] >= 30
        assert subtypes["COSMETIC_CONTROL"] >= 30
        assert subtypes["OBSTACLE_INSERTION"] >= 30
        assert subtypes["IRRELEVANT_OBSTACLE"] >= 30

    def test_get_recent_evaluation_runs_discovers_disk_runs(
        self, mock_eval_logs_dir: Path
    ):
        runs = get_recent_evaluation_runs(logs_root=mock_eval_logs_dir)
        assert len(runs) > 0
        run_ids = [r["run_id"] for r in runs]
        # Verify both dual-provider comparison and single-model runs appear
        assert any("comparison_" in rid for rid in run_ids)
        assert any("mock-vlm" in rid for rid in run_ids)

    def test_load_evaluation_run_details_dual_comparison(
        self, mock_eval_logs_dir: Path
    ):
        runs = get_recent_evaluation_runs(logs_root=mock_eval_logs_dir)
        comp_run = next((r for r in runs if r["is_comparison"]), None)
        assert comp_run is not None

        details = load_evaluation_run_details(
            comp_run["run_id"], logs_root=mock_eval_logs_dir
        )
        assert details is not None
        assert details["is_comparison"] is True
        assert "comparison_data" in details
        comp = details["comparison_data"]
        assert "headline_metrics" in comp
        assert "overall_accuracy" in comp["headline_metrics"]

    def test_get_failure_analysis_extracts_failures(self, mock_eval_logs_dir: Path):
        runs = get_recent_evaluation_runs(logs_root=mock_eval_logs_dir)
        comp_run = next((r for r in runs if r["is_comparison"]), None)
        details = load_evaluation_run_details(
            comp_run["run_id"], logs_root=mock_eval_logs_dir
        )

        failures = get_failure_analysis(details)
        assert isinstance(failures, list)
        for f in failures:
            assert "task_id" in f
            assert "ground_truth_reason" in f
            assert "diagnostic_explanation" in f

    def test_get_dataset_validation_status_11_points(self):
        status = get_dataset_validation_status(data_root="data")
        assert status["all_passed"] is True
        assert status["total_checks"] == 11
        assert len(status["checks"]) == 11
        assert status["isolation_verified"] is True

    def test_empty_state_handling_missing_dir(self, tmp_path: Path):
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        stats = get_dataset_statistics(data_root=empty_dir)
        assert stats["has_data"] is False
        assert (
            stats["total_tasks"] == 0
            or stats["total_tasks"] == stats["total_split_tasks"]
        )

        runs = get_recent_evaluation_runs(logs_root=empty_dir)
        assert runs == []

        details = load_evaluation_run_details("non_existent_run", logs_root=empty_dir)
        assert details is None

        failures = get_failure_analysis(None)
        assert failures == []
