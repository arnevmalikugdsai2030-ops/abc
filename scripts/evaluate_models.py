"""
Vision-Language Model Evaluation CLI Script for MIRRORLAB.

Executes blind evaluation runs on benchmark splits (dev, val, test)
and generates scientific accuracy and counterfactual consistency reports.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from mirrorlab.evaluation import (
    BenchmarkEvaluator,
    get_model_adapter,
)


def load_private_ground_truth(splits_dir: Path, split: str) -> dict[str, dict]:
    """Helper to load ground-truth answers for mock oracle evaluation."""
    gt_map = {}
    if split == "test":
        gt_path = splits_dir / "test" / "private" / "ground_truth.jsonl"
    else:
        gt_path = splits_dir / split / "ground_truth.jsonl"

    if gt_path.is_file():
        with open(gt_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    gt_map[item["task_id"]] = item
    return gt_map


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="MIRRORLAB Vision-Language Model Benchmark Evaluator"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mock-oracle",
        help="Model identifier (e.g. gemini-2.5-flash, gpt-4o, claude-3-7-sonnet, mock-oracle, mock-all-yes, mock-noisy, mock-random)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["dev", "val", "test"],
        help="Benchmark split to evaluate (default: test)",
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        default="data/benchmark_splits",
        help="Path to benchmark splits directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_logs",
        help="Directory to store evaluation logs and reports",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of tasks to evaluate",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Explicit strategy override for mock runner (oracle, all_yes, all_no, random, noisy_wrapper, format_error)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key override (defaults to environment variables)",
    )

    args = parser.parse_args()
    splits_dir = Path(args.splits_dir)

    print("=" * 70)
    print("  MIRRORLAB: Benchmark Vision-Language Model Evaluation Runner")
    print("=" * 70)
    print(f"Model:      {args.model}")
    print(f"Split:      {args.split}")
    print(f"Splits Dir: {splits_dir}")
    print(f"Output Dir: {args.output_dir}")
    if args.limit:
        print(f"Limit:      {args.limit} tasks")
    print("-" * 70)

    # Prepare ground truth map if mock oracle is selected
    gt_map = {}
    if "mock" in args.model.lower():
        gt_map = load_private_ground_truth(splits_dir, args.split)

    try:
        adapter = get_model_adapter(
            model_name=args.model,
            api_key=args.api_key,
            temperature=0.0,
            ground_truth_map=gt_map,
            mock_strategy=args.strategy,
        )
    except Exception as e:  # noqa: BLE001
        print(f"\n[ERROR] Failed to initialize model adapter: {e}")
        return 1

    evaluator = BenchmarkEvaluator(
        adapter=adapter,
        split_name=args.split,
        splits_root=splits_dir,
        output_dir=args.output_dir,
        limit=args.limit,
    )

    print(f"\nRunning blind evaluation on split '{args.split}'...")
    metrics, report_path = evaluator.run()

    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total Tasks Evaluated:       {metrics.total_tasks_evaluated}")
    print(f"Overall Task Accuracy:       {metrics.overall_accuracy * 100:.1f}%")
    print(f"Task A (Visibility):         {metrics.task_a_accuracy * 100:.1f}%")
    print(f"Task B (Obstruction 3-Field):{metrics.task_b_full_accuracy * 100:.1f}%")
    print(f"Task C (Mirror Repair):      {metrics.task_c_accuracy * 100:.1f}%")
    print(
        f"Counterfactual Pair Consistency:{metrics.pair_consistency_score * 100:.1f}%"
    )
    print(f"Cosmetic Theme Invariance:   {metrics.cosmetic_invariance_rate * 100:.1f}%")
    print(f"Format Compliance Rate:      {metrics.format_compliance_rate * 100:.1f}%")
    print("-" * 70)
    print(f"Full Report Saved:           {report_path}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
