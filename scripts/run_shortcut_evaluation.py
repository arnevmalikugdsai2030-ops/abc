"""
Executable script to run the 8-condition Shortcut Resistance & Visual Necessity Evaluation.
Generates data/shortcut_eval_results.json and docs/bias_and_shortcut_audit.md.
"""

from pathlib import Path

from mirrorlab.evaluation.model_adapter import MockStrategy, MockVLMAdapter
from mirrorlab.evaluation.shortcut_evaluator import ShortcutEvaluator


def main() -> None:
    print(
        "Executing 8-Condition Shortcut Resistance & Visual Necessity Evaluation on 'dev' split..."
    )
    evaluator = ShortcutEvaluator(
        splits_root="data/benchmark_splits",
        cache_dir="data/ablation_cache",
    )
    # Using calibrated realistic GPT-4o simulation baseline for offline benchmarking
    adapter = MockVLMAdapter(strategy=MockStrategy.REALISTIC_GPT4O_BASELINE)

    json_path = Path("data/shortcut_eval_results.json")
    report = evaluator.run_evaluation(
        adapter=adapter,
        split_name="dev",
        limit=None,
        output_json=json_path,
    )
    print(f"Evaluated {report.sample_count} tasks across all 8 ablation conditions.")
    print(f"Visual Necessity Ratio (VNR): {report.visual_necessity_ratio:+.2%}")
    print(f"Language Prior Bias (LPB): {report.language_prior_bias:+.2%}")
    print(f"Cosmetic Invariance Score (CIS): {report.cosmetic_invariance_score:.2%}")
    print(f"Geometric Sensitivity (GS): {report.geometric_sensitivity:+.2%}")
    print(f"Results saved to: {json_path}")

    # Generate markdown report
    md_content = evaluator.render_markdown_report(report)
    md_path = Path("docs/bias_and_shortcut_audit.md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Markdown report written to: {md_path}")


if __name__ == "__main__":
    main()
