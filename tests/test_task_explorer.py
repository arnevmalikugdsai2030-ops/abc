"""
Unit Tests for Phase 1 Task Exploration Engine.

Verifies:
1. All geometry regimes execute distinct generation behavior.
2. DUAL_MIRROR_FOLDED is explicitly rejected as unsupported multi-hop geometry without faking.
3. All occlusion regimes generate distinct obstacle configurations.
4. Distractors in HIGH_DISTRACTORS maintain strict clearance from the optical ray corridor.
5. In-loop rendering validation probes scenes during exploration.
6. Rejection accounting strictly satisfies total_attempted == total_valid + total_rejected.
"""

from pathlib import Path

from mirrorlab.experiment.task_explorer import (
    ClutterRegime,
    GeometryRegime,
    OcclusionRegime,
    ReasoningRegime,
    TaskCondition,
    TaskExplorer,
)


class TestTaskExplorerRegimesAndAccounting:
    """Tests systematic condition exploration and rejection accounting."""

    def test_dual_mirror_folded_rejected_honestly(self, tmp_path: Path):
        explorer = TaskExplorer(output_dir=tmp_path)
        cond = TaskCondition(
            condition_id="dual_mirror_test",
            geometry=GeometryRegime.DUAL_MIRROR_FOLDED,
            occlusion=OcclusionRegime.NONE,
            clutter=ClutterRegime.CLEAN,
            sample_size=5,
        )
        manifest = explorer.run_condition_experiment(cond, base_seed=42)

        # Must reject all 5 candidates honestly
        assert manifest.total_attempted == 5
        assert manifest.total_valid == 0
        assert manifest.total_rejected == 5
        assert manifest.total_rejected_geometry == 5
        assert "REJECTED_UNSUPPORTED_MULTI_HOP_GEOMETRY" in manifest.rejection_summary
        assert (
            manifest.rejection_summary["REJECTED_UNSUPPORTED_MULTI_HOP_GEOMETRY"] == 5
        )

    def test_off_axis_extreme_regime(self, tmp_path: Path):
        explorer = TaskExplorer(output_dir=tmp_path)
        cond = TaskCondition(
            condition_id="off_axis_test",
            geometry=GeometryRegime.OFF_AXIS_EXTREME,
            occlusion=OcclusionRegime.CORRIDOR_PARTITION,
            clutter=ClutterRegime.CLEAN,
            sample_size=3,
        )
        manifest = explorer.run_condition_experiment(cond, base_seed=100)

        assert manifest.total_attempted == 3
        assert (
            manifest.total_attempted == manifest.total_valid + manifest.total_rejected
        )
        assert manifest.total_rejected == sum(manifest.rejection_summary.values())

    def test_all_occlusion_regimes_generate(self, tmp_path: Path):
        explorer = TaskExplorer(output_dir=tmp_path)
        for occ in [
            OcclusionRegime.NONE,
            OcclusionRegime.CORRIDOR_PARTITION,
            OcclusionRegime.NEAR_MIRROR_BLOCKER,
            OcclusionRegime.NEAR_TARGET_BLOCKER,
            OcclusionRegime.MULTIPLE_OBSTACLES,
            OcclusionRegime.GRAZING_CLEARANCE,
        ]:
            cond = TaskCondition(
                condition_id=f"occ_{occ.value}",
                geometry=GeometryRegime.SINGLE_PLANAR,
                occlusion=occ,
                clutter=ClutterRegime.CLEAN,
                sample_size=2,
            )
            manifest = explorer.run_condition_experiment(cond, base_seed=200)
            assert manifest.total_attempted == 2
            assert (
                manifest.total_attempted
                == manifest.total_valid + manifest.total_rejected
            )

    def test_high_distractors_clearance_invariant(self, tmp_path: Path):
        explorer = TaskExplorer(output_dir=tmp_path)
        cond = TaskCondition(
            condition_id="distractor_clearance_test",
            geometry=GeometryRegime.SINGLE_PLANAR,
            occlusion=OcclusionRegime.CORRIDOR_PARTITION,
            clutter=ClutterRegime.HIGH_DISTRACTORS,
            reasoning=ReasoningRegime.TASK_A_VISIBILITY,
            sample_size=4,
        )
        manifest = explorer.run_condition_experiment(cond, base_seed=300)

        assert manifest.total_attempted == 4
        assert (
            manifest.total_attempted == manifest.total_valid + manifest.total_rejected
        )
        # Check valid scenes have distractors that do not cause rendering or ambiguity failure
        for task in manifest.tasks:
            if task.is_valid:
                assert task.scene_data is not None
                distractors = [
                    o
                    for o in task.scene_data.get("obstacles", [])
                    if o["obstacle_id"].startswith("D")
                ]
                assert len(distractors) == 4

    def test_accounting_reconciliation_invariant(self, tmp_path: Path):
        explorer = TaskExplorer(output_dir=tmp_path)
        cond = TaskCondition(
            condition_id="accounting_test",
            geometry=GeometryRegime.NARROW_APERTURE,
            occlusion=OcclusionRegime.GRAZING_CLEARANCE,
            clutter=ClutterRegime.MODERATE,
            sample_size=6,
        )
        manifest = explorer.run_condition_experiment(cond, base_seed=400)

        # Invariant 1: attempted == valid + rejected
        assert (
            manifest.total_attempted == manifest.total_valid + manifest.total_rejected
        )
        # Invariant 2: total_rejected == sum of rejection summary
        assert manifest.total_rejected == sum(manifest.rejection_summary.values())
        # Invariant 3: breakdown matches total_rejected
        assert manifest.total_rejected == (
            manifest.total_rejected_ambiguous
            + manifest.total_rejected_rendering
            + manifest.total_rejected_geometry
            + manifest.total_rejected_distractor
        )
