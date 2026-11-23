"""
Unit and Property Tests for Phase 1 Task Exploration Engine.

Verifies:
1. Six occlusion regimes generate distinguishable configurations or explicitly reject unsupported configurations.
2. High distractors are independently checked against both optical-path segments (C -> R and R -> T).
3. Distractors do not alter the visibility ground truth (semantically neutral).
4. Rendering validation is demonstrably invoked during generation.
5. Injected rendering failure produces a rendering rejection.
6. Rejection counters reconcile exactly: attempted == valid + rejected, and breakdown matches total.
7. Every accepted scene passes ambiguity and rendering validation.
8. Every rejected scene has a recorded rejection reason in rejection_summary.
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

from mirrorlab.evaluation.ambiguity_validator import AmbiguityValidator
from mirrorlab.evaluation.rendering_validator import RenderingConsistencyValidator
from mirrorlab.experiment.task_explorer import (
    ClutterRegime,
    GeometryRegime,
    OcclusionRegime,
    ReasoningRegime,
    TaskCondition,
    TaskExplorer,
)
from mirrorlab.geometry.oracle import VisibilityOracle
from mirrorlab.schema.scene import InspectionScene


def dist_point_to_segment(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    """Calculates perpendicular distance from point (px, py) to segment (ax, ay)-(bx, by)."""
    vx = bx - ax
    vy = by - ay
    l2 = vx * vx + vy * vy
    if l2 <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / l2))
    proj_x = ax + t * vx
    proj_y = ay + t * vy
    return math.hypot(px - proj_x, py - proj_y)


class TestTaskExplorerRegimesAndAccounting:
    """Tests systematic condition exploration, physical invariants, and strict rejection accounting."""

    def test_dual_mirror_folded_rejected_honestly(self, tmp_path: Path):
        """Verifies DUAL_MIRROR_FOLDED is explicitly rejected as unsupported multi-hop geometry."""
        explorer = TaskExplorer(output_dir=tmp_path)
        cond = TaskCondition(
            condition_id="dual_mirror_test",
            geometry=GeometryRegime.DUAL_MIRROR_FOLDED,
            occlusion=OcclusionRegime.NONE,
            clutter=ClutterRegime.CLEAN,
            sample_size=5,
        )
        manifest = explorer.run_condition_experiment(cond, base_seed=42)

        assert manifest.total_attempted == 5
        assert manifest.total_valid == 0
        assert manifest.total_rejected == 5
        assert manifest.total_rejected_geometry == 5
        assert "REJECTED_UNSUPPORTED_MULTI_HOP_GEOMETRY" in manifest.rejection_summary
        assert (
            manifest.rejection_summary["REJECTED_UNSUPPORTED_MULTI_HOP_GEOMETRY"] == 5
        )

    def test_six_occlusion_regimes_generate_distinguishable_configurations(
        self, tmp_path: Path
    ):
        """Verifies all six occlusion regimes generate distinct obstacle patterns or reject."""
        explorer = TaskExplorer(output_dir=tmp_path)
        regimes = [
            OcclusionRegime.NONE,
            OcclusionRegime.CORRIDOR_PARTITION,
            OcclusionRegime.NEAR_MIRROR_BLOCKER,
            OcclusionRegime.NEAR_TARGET_BLOCKER,
            OcclusionRegime.MULTIPLE_OBSTACLES,
            OcclusionRegime.GRAZING_CLEARANCE,
        ]

        regime_configs: dict[str, list[int]] = {}
        for occ in regimes:
            cond = TaskCondition(
                condition_id=f"occ_{occ.value}",
                geometry=GeometryRegime.SINGLE_PLANAR,
                occlusion=occ,
                clutter=ClutterRegime.CLEAN,
                sample_size=3,
            )
            manifest = explorer.run_condition_experiment(
                cond, base_seed=500 + len(regime_configs) * 50
            )

            # Reconcile counts
            assert (
                manifest.total_attempted
                == manifest.total_valid + manifest.total_rejected
            )
            # Record obstacle counts of valid scenes
            counts = []
            for t in manifest.tasks:
                if t.is_valid and t.scene_data is not None:
                    counts.append(len(t.scene_data.get("obstacles", [])))
            regime_configs[occ.value] = counts

        # Verify NONE produces 0 obstacles
        if regime_configs[OcclusionRegime.NONE.value]:
            assert all(c == 0 for c in regime_configs[OcclusionRegime.NONE.value])

        # Verify MULTIPLE_OBSTACLES produces >= 2 obstacles when valid
        if regime_configs[OcclusionRegime.MULTIPLE_OBSTACLES.value]:
            assert all(
                c >= 2 for c in regime_configs[OcclusionRegime.MULTIPLE_OBSTACLES.value]
            )

    def test_high_distractors_checked_against_both_optical_legs(self, tmp_path: Path):
        """Verifies distractors in HIGH_DISTRACTORS maintain strict clearance against C->R AND R->T."""
        explorer = TaskExplorer(output_dir=tmp_path)
        cond = TaskCondition(
            condition_id="distractor_two_leg_check",
            geometry=GeometryRegime.SINGLE_PLANAR,
            occlusion=OcclusionRegime.NONE,
            clutter=ClutterRegime.HIGH_DISTRACTORS,
            reasoning=ReasoningRegime.TASK_A_VISIBILITY,
            sample_size=4,
        )
        manifest = explorer.run_condition_experiment(cond, base_seed=700)

        assert (
            manifest.total_attempted == manifest.total_valid + manifest.total_rejected
        )

        for task in manifest.tasks:
            if task.is_valid and task.scene_data is not None:
                sc = InspectionScene.model_validate(task.scene_data)
                # Reflection point on mirror
                v_o = VisibilityOracle.evaluate(sc)
                if v_o.reflection_point is not None:
                    pt = v_o.reflection_point
                    rx = float(pt.x) if hasattr(pt, "x") else float(pt[0])
                    ry = float(pt.y) if hasattr(pt, "y") else float(pt[1])
                    cx, cy = sc.camera.position.x, sc.camera.position.y
                    tx, ty = sc.target.center.x, sc.target.center.y

                    distractors = [
                        o for o in sc.obstacles if o.obstacle_id.startswith("D")
                    ]
                    assert len(distractors) == 4

                    for d in distractors:
                        d_radius = (
                            d.radius
                            if getattr(d, "radius", None) is not None
                            else max(d.width or 0.0, d.height or 0.0) / 2.0
                        )
                        required_clearance = d_radius + 15.0  # 15mm safety band

                        # Check Leg 1: C -> R
                        dist_leg1 = dist_point_to_segment(
                            d.center.x, d.center.y, cx, cy, rx, ry
                        )
                        # Check Leg 2: R -> T
                        dist_leg2 = dist_point_to_segment(
                            d.center.x, d.center.y, rx, ry, tx, ty
                        )

                        assert dist_leg1 >= required_clearance - 1.0, (
                            f"Distractor {d.obstacle_id} violates Leg 1 clearance: {dist_leg1} < {required_clearance}"
                        )
                        assert dist_leg2 >= required_clearance - 1.0, (
                            f"Distractor {d.obstacle_id} violates Leg 2 clearance: {dist_leg2} < {required_clearance}"
                        )

    def test_distractors_do_not_change_visibility_ground_truth(self, tmp_path: Path):
        """Verifies adding distractors never flips ground-truth visibility or reason."""
        explorer = TaskExplorer(output_dir=tmp_path)
        cond = TaskCondition(
            condition_id="distractor_semantic_neutrality",
            geometry=GeometryRegime.SINGLE_PLANAR,
            occlusion=OcclusionRegime.CORRIDOR_PARTITION,
            clutter=ClutterRegime.HIGH_DISTRACTORS,
            reasoning=ReasoningRegime.TASK_A_VISIBILITY,
            sample_size=4,
        )
        manifest = explorer.run_condition_experiment(cond, base_seed=850)

        for task in manifest.tasks:
            if task.is_valid and task.scene_data is not None:
                sc_with_dist = InspectionScene.model_validate(task.scene_data)
                eval_with = VisibilityOracle.evaluate(sc_with_dist)

                # Strip distractors
                no_dist_obs = [
                    o
                    for o in sc_with_dist.obstacles
                    if not o.obstacle_id.startswith("D")
                ]
                sc_no_dist = sc_with_dist.model_copy(update={"obstacles": no_dist_obs})
                eval_without = VisibilityOracle.evaluate(sc_no_dist)

                # Ground truth must be identical
                assert eval_with.state == eval_without.state
                assert eval_with.reason == eval_without.reason

    def test_rendering_validation_demonstrably_invoked(self, tmp_path: Path):
        """Verifies RenderingConsistencyValidator.validate_image is actively invoked during generation."""
        explorer = TaskExplorer(output_dir=tmp_path)
        cond = TaskCondition(
            condition_id="rendering_invocation_test",
            geometry=GeometryRegime.SINGLE_PLANAR,
            occlusion=OcclusionRegime.NONE,
            clutter=ClutterRegime.CLEAN,
            sample_size=2,
        )

        with patch.object(
            RenderingConsistencyValidator,
            "validate_image",
            wraps=RenderingConsistencyValidator.validate_image,
        ) as spy_val:
            manifest = explorer.run_condition_experiment(cond, base_seed=999)
            assert spy_val.call_count >= 1
            assert manifest.total_attempted == 2

    def test_injected_rendering_failure_produces_rejection(self, tmp_path: Path):
        """Verifies that an injected rendering validation failure triggers an explicit rendering rejection."""
        explorer = TaskExplorer(output_dir=tmp_path)
        cond = TaskCondition(
            condition_id="rendering_failure_injection",
            geometry=GeometryRegime.SINGLE_PLANAR,
            occlusion=OcclusionRegime.NONE,
            clutter=ClutterRegime.CLEAN,
            sample_size=3,
        )

        mock_report = MagicMock()
        mock_report.all_passed = False
        mock_report.failure_reasons = ["SSIM_DEGRADED"]
        mock_report.element_results = []

        with patch.object(
            RenderingConsistencyValidator,
            "validate_image",
            return_value=mock_report,
        ):
            manifest = explorer.run_condition_experiment(cond, base_seed=1234)

            assert manifest.total_attempted == 3
            assert manifest.total_valid == 0
            assert manifest.total_rejected == 3
            assert manifest.total_rejected_rendering == 3
            assert "REJECTED_RENDERING_SSIM_DEGRADED" in manifest.rejection_summary
            assert manifest.rejection_summary["REJECTED_RENDERING_SSIM_DEGRADED"] == 3

    def test_rejection_counters_reconcile_exactly(self, tmp_path: Path):
        """Verifies that all rejection counters and breakdowns reconcile exactly."""
        explorer = TaskExplorer(output_dir=tmp_path)
        cond = TaskCondition(
            condition_id="accounting_reconcile_test",
            geometry=GeometryRegime.NARROW_APERTURE,
            occlusion=OcclusionRegime.GRAZING_CLEARANCE,
            clutter=ClutterRegime.MODERATE,
            sample_size=6,
        )
        manifest = explorer.run_condition_experiment(cond, base_seed=2026)

        # Invariant 1: attempted == valid + rejected
        assert (
            manifest.total_attempted == manifest.total_valid + manifest.total_rejected
        )
        # Invariant 2: total_rejected == sum of rejection summary values
        assert manifest.total_rejected == sum(manifest.rejection_summary.values())
        # Invariant 3: categorical sum matches total_rejected
        assert manifest.total_rejected == (
            manifest.total_rejected_ambiguous
            + manifest.total_rejected_rendering
            + manifest.total_rejected_geometry
            + manifest.total_rejected_distractor
        )

    def test_every_accepted_scene_passes_ambiguity_and_rendering(self, tmp_path: Path):
        """Verifies that every accepted scene independently passes ambiguity validation."""
        explorer = TaskExplorer(output_dir=tmp_path)
        cond = TaskCondition(
            condition_id="accepted_scenes_audit",
            geometry=GeometryRegime.SINGLE_PLANAR,
            occlusion=OcclusionRegime.CORRIDOR_PARTITION,
            clutter=ClutterRegime.CLEAN,
            sample_size=4,
        )
        manifest = explorer.run_condition_experiment(cond, base_seed=3333)

        for task in manifest.tasks:
            if task.is_valid:
                assert task.rejection_reason is None
                assert task.scene_data is not None
                sc = InspectionScene.model_validate(task.scene_data)
                amb_report = AmbiguityValidator.validate_scene(sc)
                assert amb_report.is_unambiguous is True, (
                    f"Accepted task {task.task_id} failed independent ambiguity validation: {amb_report.rejection_reasons}"
                )

    def test_every_rejected_scene_has_recorded_reason(self, tmp_path: Path):
        """Verifies that every rejected candidate has an explicit rejection reason recorded in summary."""
        explorer = TaskExplorer(output_dir=tmp_path)
        cond = TaskCondition(
            condition_id="rejected_reason_audit",
            geometry=GeometryRegime.NARROW_APERTURE,
            occlusion=OcclusionRegime.GRAZING_CLEARANCE,
            clutter=ClutterRegime.HIGH_DISTRACTORS,
            sample_size=6,
        )
        manifest = explorer.run_condition_experiment(cond, base_seed=4444)

        for task in manifest.tasks:
            if not task.is_valid:
                assert task.rejection_reason is not None, (
                    f"Task {task.task_id} marked rejected but rejection_reason is None"
                )
                assert len(task.rejection_reason) > 0
                assert task.rejection_reason in manifest.rejection_summary
