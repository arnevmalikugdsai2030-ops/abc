"""
Unit and Integration Tests for Shortcut Resistance and Visual Necessity Evaluator.
"""

from pathlib import Path

from PIL import Image

from mirrorlab.evaluation.model_adapter import MockStrategy, MockVLMAdapter
from mirrorlab.evaluation.shortcut_evaluator import (
    AblationCondition,
    ShortcutEvaluator,
    compute_wilson_ci,
)


class TestWilsonCI:
    def test_wilson_ci_bounds(self):
        low, high = compute_wilson_ci(50, 100)
        assert 0.39 < low < 0.41
        assert 0.59 < high < 0.61

    def test_wilson_ci_extremes(self):
        low, high = compute_wilson_ci(0, 50)
        assert low == 0.0
        assert high < 0.10

        low, high = compute_wilson_ci(50, 50)
        assert low > 0.90
        assert high == 1.0

        low, high = compute_wilson_ci(0, 0)
        assert low == 0.0
        assert high == 0.0


class TestAblationImagePreparation:
    def test_all_8_conditions_generate_valid_images(self, tmp_path: Path):
        evaluator = ShortcutEvaluator(cache_dir=tmp_path / "cache")
        # Create a test base image
        orig_img_path = tmp_path / "test_orig.png"
        img = Image.new("RGB", (800, 800), color=(255, 255, 255))
        img.save(orig_img_path)

        for cond in AblationCondition:
            ablated_path = evaluator.prepare_ablated_image(
                original_image_path=orig_img_path,
                condition=cond,
                task_id="test_task_001",
            )
            assert ablated_path.is_file()
            with Image.open(ablated_path) as ab_img:
                assert ab_img.size == (800, 800)


class TestShortcutEvaluatorIntegration:
    def test_run_evaluation_with_oracle_mock_on_dev(self, tmp_path: Path):
        evaluator = ShortcutEvaluator(
            splits_root="data/benchmark_splits",
            cache_dir=tmp_path / "cache",
        )
        adapter = MockVLMAdapter(strategy=MockStrategy.ORACLE)
        output_json = tmp_path / "shortcut_results.json"

        # Run on small slice of dev split
        report = evaluator.run_evaluation(
            adapter=adapter,
            split_name="dev",
            limit=10,
            output_json=output_json,
        )

        assert report.sample_count == 10
        assert len(report.conditions) == 8
        assert output_json.is_file()

        # Oracle on Image+Q should have high accuracy
        base_acc = report.conditions[
            AblationCondition.IMAGE_PLUS_QUESTION.value
        ].accuracy
        assert base_acc > 0.80

        # Filename only should have no signal
        assert report.filename_leakage_detected is False

        # Render report
        md = evaluator.render_markdown_report(report)
        assert "# MIRRORLAB: Visual Necessity & Shortcut-Resistance Audit Report" in md
        assert "Ablation Conditions" in md
        assert "Proof of Visual Necessity" in md
