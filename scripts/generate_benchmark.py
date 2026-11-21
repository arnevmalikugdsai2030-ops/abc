"""
Benchmark Dataset Generation CLI Script for MIRRORLAB.

Generates constraint-verified machine-vision inspection scenes and benchmarks.
"""

import argparse
import json
import sys
from pathlib import Path

from mirrorlab.generation.scene_generator import SceneGenerator
from mirrorlab.schema.scene import DifficultyLevel, SceneCategory


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic MIRRORLAB visual task benchmark."
    )
    parser.add_argument(
        "--count", type=int, default=20, help="Number of benchmark scenes to generate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=5000,
        help="Starting random seed for reproducibility",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/sample_scenes",
        help="Output directory for generated dataset",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="all",
        choices=["easy", "medium", "hard", "all"],
        help="Difficulty preset tier",
    )
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"MIRRORLAB Seeded Scene Generator: seed={args.seed}, count={args.count}")
    print(f"Target Directory: {out_path.resolve()}")
    print("=" * 70)

    gen = SceneGenerator()
    difficulties = (
        [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD]
        if args.difficulty == "all"
        else [DifficultyLevel(args.difficulty.upper())]
    )
    cats = list(SceneCategory)

    scenes_generated = 0
    for idx in range(args.count):
        seed = args.seed + idx
        diff = difficulties[idx % len(difficulties)]
        cat = cats[idx % len(cats)]

        scene = gen.generate_scene(
            seed=seed,
            category=cat,
            difficulty=diff,
            scene_id=f"scene_{seed:04d}_{cat.value.lower()}",
        )

        file_path = out_path / f"{scene.scene_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(scene.model_dump(mode="json"), f, indent=2)

        print(
            f"[{idx + 1:02d}/{args.count}] ID: {scene.scene_id:<30} | "
            f"Cat: {cat.value:<18} | Diff: {diff.value:<6} | State: {scene.ground_truth_state.value}"
        )
        scenes_generated += 1

    print("=" * 70)
    print(f"Successfully generated {scenes_generated} scenes to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
