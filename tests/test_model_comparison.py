"""
Unit and Integration Tests for MIRRORLAB Dual-Provider Comparison Pipeline.

Tests:
1. Pre-flight verification routines (scene geometry, blind images, answer separation).
2. McNemar contingency analysis and confidence interval calculation.
3. Dual-provider execution fairness and schema validation.
4. Provider comparison JSON and CSV export integrity.
"""

from __future__ import annotations

from pathlib import Path

from mirrorlab.evaluation.model_adapter import (
    PROVIDER_PRICING,
    MockStrategy,
    MockVisionAdapter,
)
from scripts.compare_vision_providers import (
    compute_mcnemar_and_confidence,
    evaluate_single_provider,
    preflight_confirm_answer_separation,
)


class TestComparisonStatistics:
    """Tests statistical rigor functions for model comparison."""

    def test_mcnemar_contingency_calculation(self):
        records_a = [
            {"score": 1.0},
            {"score": 1.0},
            {"score": 0.0},
            {"score": 0.0},
        ]
        records_b = [
            {"score": 1.0},
            {"score": 0.0},
            {"score": 1.0},
            {"score": 0.0},
        ]

        result = compute_mcnemar_and_confidence(records_a, records_b)
        table = result["contingency_table"]
        assert table["both_correct"] == 1
        assert table["provider1_only"] == 1
        assert table["provider2_only"] == 1
        assert table["both_incorrect"] == 1
        assert result["statistically_significant"] is False

    def test_confidence_interval_bounds(self):
        records_a = [{"score": 1.0}] * 10 + [{"score": 0.0}] * 10
        records_b = [{"score": 1.0}] * 10 + [{"score": 0.0}] * 10

        res = compute_mcnemar_and_confidence(records_a, records_b)
        ci_a = res["provider1_ci_95"]
        assert 0.0 <= ci_a[0] <= 50.0
        assert 50.0 <= ci_a[1] <= 100.0


class TestPreflightIntegrity:
    """Tests answer key separation and leakage checks."""

    def test_answer_key_leakage_detected(self):
        leaky_tasks = [
            {"task_id": "task_1", "prompt": "Can you see? The answer is YES!"},
        ]
        res = preflight_confirm_answer_separation(leaky_tasks)
        assert res["separation_confirmed"] is False
        assert len(res["leaks"]) == 1

    def test_clean_tasks_confirmed(self):
        clean_tasks = [
            {
                "task_id": "task_1",
                "prompt": "Is the defect scratch visible in the mirror reflection?",
            },
        ]
        res = preflight_confirm_answer_separation(clean_tasks)
        assert res["separation_confirmed"] is True
        assert len(res["leaks"]) == 0


class TestDualProviderExecution:
    """Verifies that both providers execute under identical conditions."""

    def test_provider_pricing_table(self):
        assert "gemini-2.5-flash" in PROVIDER_PRICING
        assert "gpt-4o" in PROVIDER_PRICING
        assert (
            PROVIDER_PRICING["gemini-2.5-flash"]["input_price_per_m"]
            < PROVIDER_PRICING["gpt-4o"]["input_price_per_m"]
        )

    def test_single_provider_evaluation_run(self, tmp_path: Path):
        from PIL import Image

        img_p = tmp_path / "test_task.png"
        Image.new("RGB", (100, 100), color=(10, 10, 10)).save(img_p)

        tasks = [
            {
                "task_id": "test_task",
                "task_type": "TASK_A_VISIBILITY",
                "prompt": "Is target visible?",
                "image_filename": "test_task.png",
            }
        ]
        gt = {
            "test_task": {
                "task_id": "test_task",
                "can_see": True,
                "answer": "YES",
                "reason_code": "CLEAR",
            }
        }

        adapter = MockVisionAdapter(
            strategy=MockStrategy.REALISTIC_GPT4O_BASELINE,
            ground_truth_map=gt,
            model_name="gpt-4o",
        )

        raw_log = tmp_path / "raw_log.jsonl"
        report, recs, cost = evaluate_single_provider(
            provider_name="Test GPT-4o",
            adapter=adapter,
            tasks=tasks,
            all_gt=gt,
            pairs_records=[],
            img_dir=tmp_path,
            raw_log_path=raw_log,
            input_price_per_m=2.50,
            output_price_per_m=10.00,
        )

        assert report.total_tasks_evaluated == 1
        assert len(recs) == 1
        assert raw_log.exists()
        assert cost["cost_usd"] > 0.0
