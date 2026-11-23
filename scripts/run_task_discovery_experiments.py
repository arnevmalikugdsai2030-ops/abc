"""
Task Discovery & Difficulty Exploration Experiment Runner.
Executes systematic condition sweeps and writes manifests to experiments/.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mirrorlab.experiment.task_explorer import (
    ClutterRegime,
    GeometryRegime,
    OcclusionRegime,
    ReasoningRegime,
    TaskCondition,
    TaskExplorer,
)


def main():
    print("=" * 60)
    print("MIRRORLAB TASK DISCOVERY & CONDITION EXPLORATION")
    print("=" * 60)

    explorer = TaskExplorer(output_dir="experiments")

    test_conditions = [
        TaskCondition(
            condition_id="planar_standard",
            geometry=GeometryRegime.SINGLE_PLANAR,
            occlusion=OcclusionRegime.CORRIDOR_PARTITION,
            clutter=ClutterRegime.CLEAN,
            reasoning=ReasoningRegime.TASK_A_VISIBILITY,
            sample_size=10,
        ),
        TaskCondition(
            condition_id="narrow_aperture_tight",
            geometry=GeometryRegime.NARROW_APERTURE,
            occlusion=OcclusionRegime.CORRIDOR_PARTITION,
            clutter=ClutterRegime.CLEAN,
            reasoning=ReasoningRegime.TASK_A_VISIBILITY,
            sample_size=10,
        ),
        TaskCondition(
            condition_id="grazing_incidence_oblique",
            geometry=GeometryRegime.GRAZING_INCIDENCE,
            occlusion=OcclusionRegime.CORRIDOR_PARTITION,
            clutter=ClutterRegime.CLEAN,
            reasoning=ReasoningRegime.TASK_A_VISIBILITY,
            sample_size=10,
        ),
        TaskCondition(
            condition_id="near_mirror_occlusion",
            geometry=GeometryRegime.SINGLE_PLANAR,
            occlusion=OcclusionRegime.NEAR_MIRROR_BLOCKER,
            clutter=ClutterRegime.CLEAN,
            reasoning=ReasoningRegime.TASK_B_OBSTRUCTION,
            sample_size=10,
        ),
        TaskCondition(
            condition_id="high_clutter_distractors",
            geometry=GeometryRegime.SINGLE_PLANAR,
            occlusion=OcclusionRegime.MULTIPLE_OBSTACLES,
            clutter=ClutterRegime.HIGH_DISTRACTORS,
            reasoning=ReasoningRegime.TASK_A_VISIBILITY,
            sample_size=10,
        ),
    ]

    manifests = []
    for cond in test_conditions:
        print(f"\nRunning exploration for condition: '{cond.condition_id}'...")
        manifest = explorer.run_condition_experiment(cond, base_seed=100)
        manifests.append(manifest)
        print(
            f"  -> Generated: {manifest.total_valid}/{manifest.total_attempted} valid"
        )
        print(f"  -> Rejections: {manifest.rejection_summary}")
        print(f"  -> Difficulty Distribution: {manifest.difficulty_distribution}")
        print(f"  -> Runtime: {manifest.runtime_seconds}s")

    print("\n" + "=" * 60)
    print(
        f"EXPLORATION SUITE COMPLETE: {len(manifests)} condition manifests written to experiments/"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
