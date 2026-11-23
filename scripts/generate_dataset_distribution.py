"""
Generate a diverse multi-tier benchmark dataset and compute dataset coverage statistics.
"""

from pathlib import Path

from mirrorlab.generation.scene_generator import SceneGenerator
from mirrorlab.reporting.dataset_stats import (
    compute_dataset_statistics,
    render_distribution_markdown,
)
from mirrorlab.schema.scene import DifficultyLevel


def main():
    print("=================================================================")
    print("MIRRORLAB Dataset Generator & Statistical Coverage Analyzer")
    print("=================================================================")

    gen = SceneGenerator(canvas_width=1000.0, canvas_height=1000.0)

    # 1. Generate diverse scenes across all 4 difficulty tiers
    scenes = []

    # EASY: 30 scenes (seeds 10000 - 10029)
    print(
        "Generating EASY tier scenes (Large mirrors, broad FOV, obvious obstacles)..."
    )
    easy_scenes = gen.generate_batch(
        count=30, start_seed=10000, difficulty=DifficultyLevel.EASY
    )
    scenes.extend(easy_scenes)

    # STANDARD: 50 scenes (seeds 20000 - 20049)
    print(
        "Generating STANDARD tier scenes (Moderate mirror, moderate FOV, distractors)..."
    )
    std_scenes = gen.generate_batch(
        count=50, start_seed=20000, difficulty=DifficultyLevel.STANDARD
    )
    scenes.extend(std_scenes)

    # HARD: 30 scenes (seeds 30000 - 30029)
    print(
        "Generating HARD tier scenes (Small mirror, narrow FOV, multiple obstacles)..."
    )
    hard_scenes = gen.generate_batch(
        count=30, start_seed=30000, difficulty=DifficultyLevel.HARD
    )
    scenes.extend(hard_scenes)

    # BOUNDARY: 20 scenes (seeds 40000 - 40019)
    print("Generating BOUNDARY tier scenes (Endpoint grazing, FOV boundaries)...")
    boundary_scenes = gen.generate_batch(
        count=20, start_seed=40000, difficulty=DifficultyLevel.BOUNDARY
    )
    scenes.extend(boundary_scenes)

    print(f"\nTotal scenes generated: {len(scenes)}")

    # 2. Compute statistics
    report = compute_dataset_statistics(scenes)

    # 3. Export Markdown Report
    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)
    report_file = docs_dir / "dataset_coverage_report.md"
    md_content = render_distribution_markdown(report)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Coverage report exported to: {report_file.resolve()}")

    # 4. Print Distribution Tables
    print("\n" + "=" * 65)
    print("DATASET DISTRIBUTION REPORT")
    print("=" * 65)

    print("\n1. DATASET PARTITION SEPARATION:")
    print(
        f"   - Headline Benchmark:   {report.headline_count:3d} scenes ({report.headline_pct:5.1f}%)"
    )
    print(
        f"   - Boundary Stress Test: {report.boundary_count:3d} scenes ({report.boundary_pct:5.1f}%)"
    )
    print(
        f"   - Zero Duplicate Check: {report.duplicate_count} duplicates ({report.unique_fingerprints} unique fingerprints)"
    )

    print("\n2. DIFFICULTY TIER DISTRIBUTION:")
    for diff, count in report.difficulty_counts.items():
        pct = report.difficulty_percentages[diff]
        print(f"   - {diff:<10}: {count:3d} scenes ({pct:5.1f}%)")

    print("\n3. PHYSICAL CATEGORY BREAKDOWN:")
    for cat, count in report.category_counts.items():
        pct = report.category_percentages[cat]
        print(f"   - {cat:<20}: {count:3d} scenes ({pct:5.1f}%)")

    print("\n4. GROUND-TRUTH VISIBILITY OUTCOME:")
    for state, count in report.state_counts.items():
        pct = report.state_percentages[state]
        print(f"   - {state:<12}: {count:3d} scenes ({pct:5.1f}%)")

    print("\n5. GEOMETRIC METRICS DISTRIBUTION:")
    print(
        f"   {'Metric':<25} | {'Min':>8} | {'Mean':>8} | {'Median':>8} | {'Max':>8} | {'StdDev':>8}"
    )
    print("   " + "-" * 73)
    metrics = [
        ("Mirror Length (mm)", report.mirror_length_stats),
        ("Camera FOV Angle (°)", report.fov_stats),
        ("Camera Heading (°)", report.camera_heading_stats),
        ("Incident Angle θ_i (°)", report.incident_angle_stats),
        ("Total Optical Path (mm)", report.optical_distance_stats),
        ("Obstacles per Scene", report.obstacle_count_stats),
    ]
    for label, s in metrics:
        print(
            f"   {label:<25} | {s.min_val:8.1f} | {s.mean_val:8.1f} | {s.median_val:8.1f} | {s.max_val:8.1f} | {s.std_val:8.1f}"
        )
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
