"""
Unit and Integration Tests for VLM Evaluation Harness and Metrics Engine.

Tests:
1. StructuredParser multi-pass extraction (clean JSON, markdown blocks, embedded braces, conversational heuristics).
2. ModelAdapter factory and MockVLMAdapter simulation strategies.
3. Counterfactual Pair Consistency: Joint Correctness verification (0% for ALL_YES on flips).
4. Full end-to-end BenchmarkEvaluator execution on dev split with artifact verification.
"""

from __future__ import annotations

import json
from pathlib import Path

from mirrorlab.evaluation import (
    BenchmarkEvaluator,
    BenchmarkMetricsCalculator,
    MockStrategy,
    MockVLMAdapter,
    StructuredParser,
    get_model_adapter,
)


class TestStructuredParser:
    """Tests multi-pass extraction on various response formats."""

    def test_pass_1_strict_json(self):
        text = '{"answer": "YES", "reason": "CLEAR", "explanation": "Path is clear."}'
        parsed = StructuredParser.parse(text, task_type="TASK_A_VISIBILITY", task_id="t1")
        assert parsed.is_valid_format is True
        assert parsed.extraction_pass == 1
        assert parsed.prediction == "YES"
        assert parsed.can_see is True
        assert parsed.reason_code == "CLEAR"

    def test_pass_2_markdown_fenced_json(self):
        text = (
            "Based on the 2D schematic diagram:\n\n"
            "```json\n"
            '{\n  "can_see": false,\n  "answer": "NO",\n  "reason": "MIRROR_MISS"\n}\n'
            "```\n"
            "This concludes the ray-tracing inspection."
        )
        parsed = StructuredParser.parse(text, task_type="TASK_A_VISIBILITY", task_id="t2")
        assert parsed.is_valid_format is True
        assert parsed.extraction_pass == 2
        assert parsed.prediction == "NO"
        assert parsed.can_see is False
        assert parsed.reason_code == "MIRROR_MISS"

    def test_pass_3_embedded_braces(self):
        text = (
            "Here is my inspection result: "
            '{"is_blocked": true, "answer": "YES", "blocker_label": "O2", "blocked_segment": "CAMERA_TO_MIRROR"} '
            "Hope this helps."
        )
        parsed = StructuredParser.parse(text, task_type="TASK_B_OBSTRUCTION", task_id="t3")
        assert parsed.is_valid_format is True
        assert parsed.extraction_pass == 3
        assert parsed.prediction == "YES"
        assert parsed.is_blocked is True
        assert parsed.blocker_label == "O2"
        assert parsed.blocked_segment == "CAMERA_TO_MIRROR"

    def test_pass_4_conversational_heuristics_task_a(self):
        text = (
            "After calculating the reflection angle, I determined that the mirror reflective face "
            "faces the camera and target. The answer is YES, the scratch is CLEAR and visible."
        )
        parsed = StructuredParser.parse(text, task_type="TASK_A_VISIBILITY", task_id="t4")
        assert parsed.is_valid_format is True
        assert parsed.extraction_pass == 4
        assert parsed.prediction == "YES"
        assert parsed.can_see is True
        assert parsed.reason_code == "CLEAR"

    def test_pass_4_task_c_repair(self):
        text = "I evaluated mirrors A, B, and C. Only Candidate B redirects the light correctly."
        parsed = StructuredParser.parse(text, task_type="TASK_C_REPAIR", task_id="t5")
        assert parsed.is_valid_format is True
        assert parsed.extraction_pass == 4
        assert parsed.prediction == "B"
        assert parsed.selected_candidate == "B"

    def test_unparseable_gibberish_returns_format_error(self):
        text = "The diagram is blurry and I am not sure about optical geometry."
        parsed = StructuredParser.parse(text, task_type="TASK_A_VISIBILITY", task_id="t6")
        assert parsed.is_valid_format is False
        assert parsed.extraction_pass == 0
        assert parsed.prediction is None
        assert parsed.format_error is not None


class TestModelAdapter:
    """Tests model adapter instantiation and mock generation."""

    def test_mock_adapter_oracle_strategy(self, tmp_path: Path):
        img_p = tmp_path / "test.png"
        img_p.write_bytes(b"\x89PNG\r\n\x1a\n")
        gt_map = {
            "task_001": {"can_see": True, "answer": "YES", "reason_code": "CLEAR"}
        }
        adapter = get_model_adapter("mock-oracle", ground_truth_map=gt_map)
        res = adapter.generate(
            prompt="Is it visible?",
            image_path=img_p,
            task_id="task_001",
            task_type="TASK_A_VISIBILITY",
        )
        assert res.is_mock is True
        assert "YES" in res.raw_text

    def test_mock_adapter_all_yes_strategy(self, tmp_path: Path):
        img_p = tmp_path / "test.png"
        img_p.write_bytes(b"\x89PNG\r\n\x1a\n")
        adapter = get_model_adapter("mock-all-yes")
        res = adapter.generate(
            prompt="Is it visible?",
            image_path=img_p,
            task_id="task_002",
            task_type="TASK_A_VISIBILITY",
        )
        assert '"answer": "YES"' in res.raw_text


class TestCounterfactualPairScoring:
    """Verifies joint correctness scoring logic for counterfactual pairs."""

    def test_joint_correctness_eliminates_all_yes_bias(self):
        """
        In a flip pair, Scene A is YES and Scene B is NO.
        An ALL_YES guesser predicts YES to both.
        Joint correctness MUST score 0.0 points!
        """
        pair_rec = {
            "pair_id": "pair_001",
            "pair_type": "MIRROR_ANGLE_FLIP",
            "child_scene_ids": ["scene_a", "scene_b"],
        }
        eval_records = [
            {
                "task_id": "scene_a",
                "task_type": "TASK_A_VISIBILITY",
                "ground_truth": {"answer": "YES", "reason_code": "CLEAR"},
                "parsed_answer": {"prediction": "YES", "is_valid_format": True},
            },
            {
                "task_id": "scene_b",
                "task_type": "TASK_A_VISIBILITY",
                "ground_truth": {"answer": "NO", "reason_code": "MIRROR_MISS"},
                "parsed_answer": {"prediction": "YES", "is_valid_format": True},  # Guessed YES -> wrong!
            },
        ]

        metrics = BenchmarkMetricsCalculator.compute(eval_records, pairs_records=[pair_rec])
        # Individual accuracy on the 2 tasks is 1/2 = 50%
        assert metrics.overall_accuracy == 0.50
        # But counterfactual pair consistency MUST be 0.0%!
        assert metrics.pair_consistency_score == 0.0
        assert metrics.pair_consistency_by_type.get("MIRROR_ANGLE_FLIP") == 0.0

    def test_joint_correctness_scores_1_when_both_correct(self):
        pair_rec = {
            "pair_id": "pair_002",
            "pair_type": "MIRROR_ANGLE_FLIP",
            "child_scene_ids": ["scene_a", "scene_b"],
        }
        eval_records = [
            {
                "task_id": "scene_a",
                "task_type": "TASK_A_VISIBILITY",
                "ground_truth": {"answer": "YES", "reason_code": "CLEAR"},
                "parsed_answer": {"prediction": "YES", "is_valid_format": True},
            },
            {
                "task_id": "scene_b",
                "task_type": "TASK_A_VISIBILITY",
                "ground_truth": {"answer": "NO", "reason_code": "MIRROR_MISS"},
                "parsed_answer": {"prediction": "NO", "is_valid_format": True},
            },
        ]

        metrics = BenchmarkMetricsCalculator.compute(eval_records, pairs_records=[pair_rec])
        assert metrics.overall_accuracy == 1.0
        assert metrics.pair_consistency_score == 1.0


class TestBenchmarkEvaluatorIntegration:
    """Tests end-to-end evaluation runner against the dev split."""

    def test_evaluator_run_oracle_mock_on_dev(self, tmp_path: Path):
        # Load dev ground truth for mock oracle
        gt_path = Path("data/benchmark_splits/dev/ground_truth.jsonl")
        gt_map = {}
        if gt_path.is_file():
            with open(gt_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        gt_map[obj["task_id"]] = obj

        adapter = MockVLMAdapter(strategy=MockStrategy.ORACLE, ground_truth_map=gt_map)
        evaluator = BenchmarkEvaluator(
            adapter=adapter,
            split_name="dev",
            splits_root=Path("data/benchmark_splits"),
            output_dir=tmp_path / "eval_logs",
            limit=10,  # Evaluate first 10 tasks for speed
        )

        metrics, report_path = evaluator.run()

        assert metrics.total_tasks_evaluated == 10
        assert metrics.format_compliance_rate == 1.0
        assert metrics.overall_accuracy == 1.0
        assert report_path.is_file()

        # Check raw responses and predictions logs exist
        run_dir = evaluator.run_dir
        assert (run_dir / "raw_responses.jsonl").is_file()
        assert (run_dir / "predictions.jsonl").is_file()
        assert (run_dir / "metrics.json").is_file()
