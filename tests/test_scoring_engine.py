"""
Unit and Integration Tests for the Complete MIRRORLAB Scoring Engine.

Tests:
1. Overall accuracy with exact numerators and denominators.
2. Visibility accuracy (binary and reason code).
3. Obstruction accuracy (separated blocker ID, blocked leg, blocked status, and joint).
4. Repair accuracy (exact unique candidate setting A/B/C).
5. Counterfactual pair consistency (joint correctness rule: both A and B must be correct).
6. Mirror-angle flip consistency, cosmetic invariance, and obstacle insertion.
7. Invalid response rate (API failures and malformed format never scored correct).
8. Stratified difficulty and reason-code accuracy.
9. Latency profiling: median and P90 latency.
10. Denominators reported for every single metric.
11. Zero fabrication on empty or missing data.
12. JSON and CSV export verification.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mirrorlab.evaluation.scoring_engine import (
    MetricFraction,
    ObstructionMetrics,
    ScoringEngine,
)


class TestMetricFraction:
    """Tests the fundamental MetricFraction container."""

    def test_fraction_creation_and_properties(self):
        frac = MetricFraction.create(correct=7, total=10)
        assert frac.correct == 7
        assert frac.total == 10
        assert frac.numerator == 7
        assert frac.denominator == 10
        assert frac.rate == 0.7
        assert frac.percentage == 70.0

    def test_zero_denominator_safe(self):
        frac = MetricFraction.create(correct=0, total=0)
        assert frac.correct == 0
        assert frac.total == 0
        assert frac.rate == 0.0
        assert frac.percentage == 0.0

    def test_bounds_clamp(self):
        frac = MetricFraction.create(correct=15, total=10)
        assert frac.rate == 1.0  # Clamped to 1.0 max


class TestScoringEngineOverallAccuracy:
    """Tests overall accuracy calculation and reporting."""

    def test_overall_accuracy_manually_verified(self):
        # 5 samples: 3 correct, 2 incorrect
        records = [
            {
                "task_id": "t1",
                "task_type": "TASK_A_VISIBILITY",
                "ground_truth": {"answer": "YES"},
                "parsed_answer": {"answer": "YES", "is_valid_format": True},
                "is_valid_format": True,
            },
            {
                "task_id": "t2",
                "task_type": "TASK_A_VISIBILITY",
                "ground_truth": {"answer": "NO"},
                "parsed_answer": {"answer": "NO", "is_valid_format": True},
                "is_valid_format": True,
            },
            {
                "task_id": "t3",
                "task_type": "TASK_A_VISIBILITY",
                "ground_truth": {"answer": "YES"},
                "parsed_answer": {"answer": "NO", "is_valid_format": True},
                "is_valid_format": True,
            },
            {
                "task_id": "t4",
                "task_type": "TASK_C_REPAIR",
                "ground_truth": {"correct_candidate": "B"},
                "parsed_answer": {"selected_candidate": "B", "is_valid_format": True},
                "is_valid_format": True,
            },
            {
                "task_id": "t5",
                "task_type": "TASK_C_REPAIR",
                "ground_truth": {"correct_candidate": "A"},
                "parsed_answer": {"selected_candidate": "C", "is_valid_format": True},
                "is_valid_format": True,
            },
        ]

        report = ScoringEngine.evaluate(records)
        assert report.total_tasks_evaluated == 5
        assert report.overall_accuracy.correct == 3
        assert report.overall_accuracy.total == 5
        assert report.overall_accuracy.rate == 0.6
        assert report.overall_accuracy.percentage == 60.0


class TestScoringEngineVisibility:
    """Tests Task A Visibility accuracy and reason code scoring."""

    def test_visibility_binary_and_reason_code(self):
        records = [
            {
                "task_id": "v1",
                "task_type": "TASK_A_VISIBILITY",
                "ground_truth": {"answer": "YES", "reason_code": "CLEAR"},
                "parsed_answer": {
                    "answer": "YES",
                    "reason_code": "CLEAR",
                    "is_valid_format": True,
                },
            },
            {
                "task_id": "v2",
                "task_type": "TASK_A_VISIBILITY",
                "ground_truth": {"answer": "NO", "reason_code": "MIRROR_MISS"},
                "parsed_answer": {
                    "answer": "NO",
                    "reason_code": "OUTSIDE_FOV",
                    "is_valid_format": True,
                },
            },
            {
                "task_id": "v3",
                "task_type": "TASK_A_VISIBILITY",
                "ground_truth": {
                    "answer": "NO",
                    "reason_code": "BLOCKED_CAMERA_TO_MIRROR",
                },
                "parsed_answer": {
                    "answer": "YES",
                    "reason_code": "CLEAR",
                    "is_valid_format": True,
                },
            },
        ]

        report = ScoringEngine.evaluate(records)
        # Binary visibility: 2 out of 3 correct (v1, v2)
        assert report.visibility_accuracy.correct == 2
        assert report.visibility_accuracy.total == 3
        assert report.visibility_accuracy.rate == pytest.approx(2 / 3, 0.001)

        # Reason code: 1 out of 3 correct (only v1)
        assert report.visibility_reason_accuracy.correct == 1
        assert report.visibility_reason_accuracy.total == 3
        assert report.visibility_reason_accuracy.rate == pytest.approx(1 / 3, 0.001)


class TestScoringEngineObstruction:
    """Tests Task B Obstruction scoring: blocker ID, blocked leg, and joint."""

    def test_obstruction_separated_and_joint_scoring(self):
        records = [
            # Item 1: Perfect match (all 3 correct) -> Status OK, Blocker OK, Leg OK, Joint OK
            {
                "task_id": "b1",
                "task_type": "TASK_B_OBSTRUCTION",
                "ground_truth": {
                    "is_blocked": True,
                    "blocker_label": "O1",
                    "blocked_segment": "CAMERA_TO_MIRROR",
                },
                "parsed_answer": {
                    "is_blocked": True,
                    "blocker_label": "O1",
                    "blocked_segment": "CAMERA_TO_MIRROR",
                    "is_valid_format": True,
                },
            },
            # Item 2: Status OK, Leg OK, but Blocker WRONG ("O2" vs "O3") -> Joint FAIL
            {
                "task_id": "b2",
                "task_type": "TASK_B_OBSTRUCTION",
                "ground_truth": {
                    "is_blocked": True,
                    "blocker_label": "O3",
                    "blocked_segment": "MIRROR_TO_TARGET",
                },
                "parsed_answer": {
                    "is_blocked": True,
                    "blocker_label": "O2",
                    "blocked_segment": "MIRROR_TO_TARGET",
                    "is_valid_format": True,
                },
            },
            # Item 3: Status OK, Blocker OK ("O1"), but Leg WRONG (MIRROR_TO_TARGET vs CAMERA_TO_MIRROR) -> Joint FAIL
            {
                "task_id": "b3",
                "task_type": "TASK_B_OBSTRUCTION",
                "ground_truth": {
                    "is_blocked": True,
                    "blocker_label": "O1",
                    "blocked_segment": "CAMERA_TO_MIRROR",
                },
                "parsed_answer": {
                    "is_blocked": True,
                    "blocker_label": "O1",
                    "blocked_segment": "MIRROR_TO_TARGET",
                    "is_valid_format": True,
                },
            },
            # Item 4: Unblocked scene, all match ("NONE") -> Status OK, Blocker OK, Leg OK, Joint OK
            {
                "task_id": "b4",
                "task_type": "TASK_B_OBSTRUCTION",
                "ground_truth": {
                    "is_blocked": False,
                    "blocker_label": "NONE",
                    "blocked_segment": "NONE",
                },
                "parsed_answer": {
                    "is_blocked": False,
                    "blocker_label": "NONE",
                    "blocked_segment": "NONE",
                    "is_valid_format": True,
                },
            },
        ]

        report = ScoringEngine.evaluate(records)
        obs: ObstructionMetrics = report.obstruction_accuracy

        # Blocked Status: 4/4 = 1.0 (all 4 correctly identified blocked or unblocked)
        assert obs.blocked_status.correct == 4
        assert obs.blocked_status.total == 4
        assert obs.blocked_status.rate == 1.0

        # Blocker ID: 3/4 = 0.75 (b1, b3, b4 correct; b2 guessed O2 instead of O3)
        assert obs.blocker_id.correct == 3
        assert obs.blocker_id.total == 4
        assert obs.blocker_id.rate == 0.75

        # Blocked Leg: 3/4 = 0.75 (b1, b2, b4 correct; b3 guessed MIRROR_TO_TARGET instead of CAMERA_TO_MIRROR)
        assert obs.blocked_leg.correct == 3
        assert obs.blocked_leg.total == 4
        assert obs.blocked_leg.rate == 0.75

        # Joint Match: 2/4 = 0.5 (only b1 and b4 got all 3 correct simultaneously)
        assert obs.joint.correct == 2
        assert obs.joint.total == 4
        assert obs.joint.rate == 0.5


class TestScoringEngineRepair:
    """Tests Task C Repair accuracy with exact unique candidate requirement."""

    def test_repair_exact_unique_setting(self):
        records = [
            # Correct candidate B
            {
                "task_id": "r1",
                "task_type": "TASK_C_REPAIR",
                "ground_truth": {"correct_candidate": "B"},
                "parsed_answer": {"selected_candidate": "B", "is_valid_format": True},
            },
            # Wrong candidate A instead of C
            {
                "task_id": "r2",
                "task_type": "TASK_C_REPAIR",
                "ground_truth": {"correct_candidate": "C"},
                "parsed_answer": {"selected_candidate": "A", "is_valid_format": True},
            },
            # Invalid candidate D (not in A/B/C) -> must be rejected as score 0
            {
                "task_id": "r3",
                "task_type": "TASK_C_REPAIR",
                "ground_truth": {"correct_candidate": "A"},
                "parsed_answer": {"selected_candidate": "D", "is_valid_format": True},
            },
            # Correct candidate A with prefix string ("Setting A") normalized
            {
                "task_id": "r4",
                "task_type": "TASK_C_REPAIR",
                "ground_truth": {"correct_candidate": "A"},
                "parsed_answer": {
                    "selected_candidate": "SETTING A",
                    "is_valid_format": True,
                },
            },
        ]

        report = ScoringEngine.evaluate(records)
        # 2 out of 4 correct (r1, r4)
        assert report.repair_accuracy.correct == 2
        assert report.repair_accuracy.total == 4
        assert report.repair_accuracy.rate == 0.5


class TestScoringEngineCounterfactualPairs:
    """Tests Counterfactual Pair Consistency and specific pair type metrics."""

    def test_counterfactual_pair_joint_correctness_and_types(self):
        records = [
            # Flip Pair 1: Base A=YES, Perturbed B=NO
            {
                "task_id": "flip_1_a",
                "task_type": "TASK_A_VISIBILITY",
                "ground_truth": {"answer": "YES"},
                "parsed_answer": {"answer": "YES", "is_valid_format": True},
            },
            {
                "task_id": "flip_1_b",
                "task_type": "TASK_A_VISIBILITY",
                "ground_truth": {"answer": "NO"},
                "parsed_answer": {"answer": "NO", "is_valid_format": True},
            },
            # Flip Pair 2: Base A=YES, Perturbed B=NO. Model answers YES to BOTH (guessing bias)!
            {
                "task_id": "flip_2_a",
                "task_type": "TASK_A_VISIBILITY",
                "ground_truth": {"answer": "YES"},
                "parsed_answer": {"answer": "YES", "is_valid_format": True},
            },
            {
                "task_id": "flip_2_b",
                "task_type": "TASK_A_VISIBILITY",
                "ground_truth": {"answer": "NO"},
                "parsed_answer": {"answer": "YES", "is_valid_format": True},  # WRONG
            },
            # Cosmetic Pair 1: Base A=YES, Perturbed B=YES. Model answers YES to BOTH.
            {
                "task_id": "cosm_1_a",
                "task_type": "TASK_A_VISIBILITY",
                "ground_truth": {"answer": "YES"},
                "parsed_answer": {"answer": "YES", "is_valid_format": True},
            },
            {
                "task_id": "cosm_1_b",
                "task_type": "TASK_A_VISIBILITY",
                "ground_truth": {"answer": "YES"},
                "parsed_answer": {"answer": "YES", "is_valid_format": True},
            },
            # Obstacle Insertion Pair 1: Base A=YES, Perturbed B=NO. Model answers YES, NO.
            {
                "task_id": "obs_ins_1_a",
                "task_type": "TASK_A_VISIBILITY",
                "ground_truth": {"answer": "YES"},
                "parsed_answer": {"answer": "YES", "is_valid_format": True},
            },
            {
                "task_id": "obs_ins_1_b",
                "task_type": "TASK_A_VISIBILITY",
                "ground_truth": {"answer": "NO"},
                "parsed_answer": {"answer": "NO", "is_valid_format": True},
            },
        ]

        pairs = [
            {
                "pair_id": "p_flip_1",
                "pair_type": "MIRROR_ANGLE_FLIP",
                "child_scene_ids": ["flip_1_a", "flip_1_b"],
            },
            {
                "pair_id": "p_flip_2",
                "pair_type": "MIRROR_ANGLE_FLIP",
                "child_scene_ids": ["flip_2_a", "flip_2_b"],
            },
            {
                "pair_id": "p_cosm_1",
                "pair_type": "COSMETIC_CONTROL",
                "child_scene_ids": ["cosm_1_a", "cosm_1_b"],
            },
            {
                "pair_id": "p_obs_ins_1",
                "pair_type": "OBSTACLE_INSERTION",
                "child_scene_ids": ["obs_ins_1_a", "obs_ins_1_b"],
            },
        ]

        report = ScoringEngine.evaluate(eval_records=records, pairs_records=pairs)

        # Overall pairs: 4 pairs total. p_flip_1 (OK), p_flip_2 (FAIL), p_cosm_1 (OK), p_obs_ins_1 (OK)
        # 3 out of 4 correct -> rate = 0.75
        assert report.pair_accuracy.total == 4
        assert report.pair_accuracy.correct == 3
        assert report.pair_accuracy.rate == 0.75

        # Mirror Angle Flip: 1 out of 2 correct (p_flip_1 correct, p_flip_2 failed joint rule)
        assert report.mirror_angle_flip_consistency.total == 2
        assert report.mirror_angle_flip_consistency.correct == 1
        assert report.mirror_angle_flip_consistency.rate == 0.5

        # Cosmetic Invariance: 1 out of 1 correct
        assert report.cosmetic_invariance.total == 1
        assert report.cosmetic_invariance.correct == 1
        assert report.cosmetic_invariance.rate == 1.0

        # Obstacle Insertion: 1 out of 1 correct
        assert report.obstacle_insertion_consistency.total == 1
        assert report.obstacle_insertion_consistency.correct == 1
        assert report.obstacle_insertion_consistency.rate == 1.0


class TestScoringEngineSafetyAndInvariants:
    """Verifies that API errors, format errors, and missing data are never fabricated or marked correct."""

    def test_api_errors_never_marked_correct(self):
        records = [
            # Record with is_api_error=True even though parsed text coincidentally says YES
            {
                "task_id": "err1",
                "task_type": "TASK_A_VISIBILITY",
                "ground_truth": {"answer": "YES"},
                "parsed_answer": {"answer": "YES"},
                "is_valid_format": True,
                "is_api_error": True,
                "error": "RateLimitError: 429",
            },
            # Record with format failure (empty / malformed)
            {
                "task_id": "err2",
                "task_type": "TASK_A_VISIBILITY",
                "ground_truth": {"answer": "YES"},
                "parsed_answer": None,
                "is_valid_format": False,
            },
            # Normal correct record
            {
                "task_id": "ok1",
                "task_type": "TASK_A_VISIBILITY",
                "ground_truth": {"answer": "YES"},
                "parsed_answer": {"answer": "YES"},
                "is_valid_format": True,
            },
        ]

        report = ScoringEngine.evaluate(records)
        assert report.total_tasks_evaluated == 3
        # Only ok1 should be correct
        assert report.overall_accuracy.correct == 1
        assert report.overall_accuracy.total == 3
        assert report.overall_accuracy.rate == pytest.approx(1 / 3, 0.001)

        # Invalid response count: 2 out of 3
        assert report.invalid_response_rate.correct == 2
        assert report.invalid_response_rate.total == 3
        assert report.invalid_response_rate.rate == pytest.approx(2 / 3, 0.001)

    def test_empty_dataset_safe_zero_fabrication(self):
        report = ScoringEngine.evaluate([])
        assert report.total_tasks_evaluated == 0
        assert report.overall_accuracy.total == 0
        assert report.overall_accuracy.correct == 0
        assert report.overall_accuracy.rate == 0.0
        assert report.latency.count == 0


class TestScoringEngineStratificationAndLatency:
    """Tests per-difficulty, per-reason, and latency calculations."""

    def test_stratifications_and_latency_quantiles(self):
        records = [
            {
                "task_id": "d1",
                "task_type": "TASK_A_VISIBILITY",
                "difficulty_tier": "EASY",
                "ground_truth": {"answer": "YES", "reason_code": "CLEAR"},
                "parsed_answer": {"answer": "YES"},
                "is_valid_format": True,
                "latency_ms": 100.0,
            },
            {
                "task_id": "d2",
                "task_type": "TASK_A_VISIBILITY",
                "difficulty_tier": "EASY",
                "ground_truth": {"answer": "YES", "reason_code": "CLEAR"},
                "parsed_answer": {"answer": "NO"},
                "is_valid_format": True,
                "latency_ms": 200.0,
            },
            {
                "task_id": "d3",
                "task_type": "TASK_A_VISIBILITY",
                "difficulty_tier": "HARD",
                "ground_truth": {"answer": "NO", "reason_code": "MIRROR_MISS"},
                "parsed_answer": {"answer": "NO"},
                "is_valid_format": True,
                "latency_ms": 300.0,
            },
            {
                "task_id": "d4",
                "task_type": "TASK_A_VISIBILITY",
                "difficulty_tier": "HARD",
                "ground_truth": {"answer": "NO", "reason_code": "MIRROR_MISS"},
                "parsed_answer": {"answer": "NO"},
                "is_valid_format": True,
                "latency_ms": 400.0,
            },
            {
                "task_id": "d5",
                "task_type": "TASK_A_VISIBILITY",
                "difficulty_tier": "BOUNDARY",
                "ground_truth": {"answer": "NO", "reason_code": "OUTSIDE_FOV"},
                "parsed_answer": {"answer": "YES"},
                "is_valid_format": True,
                "latency_ms": 500.0,
            },
        ]

        report = ScoringEngine.evaluate(records)

        # Per-Difficulty
        assert "EASY" in report.per_difficulty_accuracy
        assert report.per_difficulty_accuracy["EASY"].correct == 1
        assert report.per_difficulty_accuracy["EASY"].total == 2
        assert report.per_difficulty_accuracy["EASY"].rate == 0.5

        assert "HARD" in report.per_difficulty_accuracy
        assert report.per_difficulty_accuracy["HARD"].correct == 2
        assert report.per_difficulty_accuracy["HARD"].total == 2
        assert report.per_difficulty_accuracy["HARD"].rate == 1.0

        assert "BOUNDARY" in report.per_difficulty_accuracy
        assert report.per_difficulty_accuracy["BOUNDARY"].correct == 0
        assert report.per_difficulty_accuracy["BOUNDARY"].total == 1
        assert report.per_difficulty_accuracy["BOUNDARY"].rate == 0.0

        # Per-Reason Code
        assert "CLEAR" in report.per_reason_code_accuracy
        assert report.per_reason_code_accuracy["CLEAR"].correct == 1
        assert report.per_reason_code_accuracy["CLEAR"].total == 2

        assert "MIRROR_MISS" in report.per_reason_code_accuracy
        assert report.per_reason_code_accuracy["MIRROR_MISS"].correct == 2
        assert report.per_reason_code_accuracy["MIRROR_MISS"].total == 2

        # Latency: [100, 200, 300, 400, 500]
        # Median is 300.0 ms
        assert report.latency.median_ms == 300.0
        # P90 of [100, 200, 300, 400, 500]
        assert report.latency.p90_ms == 460.0
        assert report.latency.mean_ms == 300.0
        assert report.latency.min_ms == 100.0
        assert report.latency.max_ms == 500.0


class TestScoringEngineExports:
    """Tests JSON and CSV file export features."""

    def test_json_and_csv_export(self, tmp_path: Path):
        records = [
            {
                "task_id": "task_exp_1",
                "task_type": "TASK_A_VISIBILITY",
                "difficulty_tier": "STANDARD",
                "ground_truth": {"answer": "YES", "reason_code": "CLEAR"},
                "parsed_answer": {"answer": "YES"},
                "is_valid_format": True,
                "latency_ms": 120.0,
            },
            {
                "task_id": "task_exp_2",
                "task_type": "TASK_B_OBSTRUCTION",
                "difficulty_tier": "STANDARD",
                "ground_truth": {
                    "is_blocked": True,
                    "blocker_label": "O1",
                    "blocked_segment": "CAMERA_TO_MIRROR",
                },
                "parsed_answer": {
                    "is_blocked": True,
                    "blocker_label": "O1",
                    "blocked_segment": "CAMERA_TO_MIRROR",
                },
                "is_valid_format": True,
                "latency_ms": 250.0,
            },
        ]

        report = ScoringEngine.evaluate(records)

        # JSON Export
        json_file = tmp_path / "metrics.json"
        saved_json_path = report.export_json(json_file)
        assert saved_json_path.exists()

        with open(saved_json_path, encoding="utf-8") as fp:
            loaded_data = json.load(fp)
        assert loaded_data["total_tasks_evaluated"] == 2
        assert loaded_data["overall_accuracy"]["correct"] == 2
        assert loaded_data["overall_accuracy"]["total"] == 2
        assert loaded_data["overall_accuracy"]["rate"] == 1.0

        # CSV Export
        summary_csv = tmp_path / "metrics_summary.csv"
        per_task_csv = tmp_path / "per_task_details.csv"
        s_path, t_path = report.export_csv(
            summary_path=summary_csv, per_task_path=per_task_csv
        )

        assert s_path.exists()
        assert t_path is not None and t_path.exists()

        # Read summary CSV
        with open(s_path, encoding="utf-8") as fp:
            s_content = fp.read()
        assert "Overall Accuracy" in s_content
        assert "Blocked Status Accuracy" in s_content
        assert "Mirror Repair Accuracy (Unique Setting)" in s_content
        assert "Overall Pair Accuracy (Joint)" in s_content

        # Read per-task CSV
        with open(t_path, encoding="utf-8") as fp:
            t_content = fp.read()
        assert "task_exp_1" in t_content
        assert "task_exp_2" in t_content
        assert "TASK_B_OBSTRUCTION" in t_content
