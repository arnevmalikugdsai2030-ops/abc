"""
Benchmark Dataset Generation CLI Script for MIRRORLAB.

Generates complete, multimodal machine-vision inspection task benchmarks:
- Scene JSON
- Blind diagram PNG
- Question JSON
- Ground-truth answer JSON
- Explanation overlay PNG
- Aggregate tasks.jsonl & ground_truth.jsonl
"""

import argparse
import sys
from pathlib import Path

from mirrorlab.generation.task_factory import VisibilityTaskFactory
from mirrorlab.rendering.themes import RenderTheme
from mirrorlab.schema.scene import DifficultyLevel


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic MIRRORLAB visual task benchmark."
    )
    parser.add_argument(
        "--count", type=int, default=20, help="Number of benchmark tasks to generate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10000,
        help="Starting random seed for reproducibility",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/benchmark_tasks",
        help="Output directory for generated dataset",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="all",
        choices=["easy", "medium", "hard", "all"],
        help="Difficulty preset tier",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="cycle",
        choices=["dark_navy", "light_offwhite", "blueprint", "monochrome_cad", "cycle"],
        help="Rendering theme palette",
    )
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("MIRRORLAB VISIBILITY TASK FACTORY (BENCHMARK GENERATOR)")
    print(f"Count: {args.count} | Seed: {args.seed} | Target: {out_path.resolve()}")
    print("=" * 75)

    difficulties = (
        [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD]
        if args.difficulty == "all"
        else [DifficultyLevel(args.difficulty.upper())]
    )

    theme_map = {
        "dark_navy": [RenderTheme.DARK_NAVY],
        "light_offwhite": [RenderTheme.LIGHT_OFFWHITE],
        "blueprint": [RenderTheme.BLUEPRINT],
        "monochrome_cad": [RenderTheme.MONOCHROME_CAD],
        "cycle": [
            RenderTheme.DARK_NAVY,
            RenderTheme.LIGHT_OFFWHITE,
            RenderTheme.BLUEPRINT,
            RenderTheme.MONOCHROME_CAD,
        ],
    }
    themes = theme_map[args.theme]

    factory = VisibilityTaskFactory(image_size=1024)
    tasks = factory.create_balanced_dataset(
        count=args.count,
        start_seed=args.seed,
        output_dir=out_path,
        difficulties=difficulties,
        themes=themes,
    )

    visible_count = sum(1 for t in tasks if t.ground_truth.can_see)
    not_visible_count = len(tasks) - visible_count

    print("-" * 75)
    for idx, t in enumerate(tasks):
        gt = t.ground_truth
        print(
            f"[{idx + 1:02d}/{len(tasks)}] ID: {t.task_id:<32} | "
            f"Theme: {t.theme:<15} | Ans: {gt.answer:<3} | Reason: {gt.reason_code.value}"
        )

    print("=" * 75)
    print(
        f"Generated {len(tasks)} tasks: {visible_count} VISIBLE (50%), "
        f"{not_visible_count} NOT VISIBLE (50%)"
    )
    print(f"Public questions -> {out_path / 'tasks.jsonl'}")
    print(f"Private ground truth -> {out_path / 'ground_truth.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
