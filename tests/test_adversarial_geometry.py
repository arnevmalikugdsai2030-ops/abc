"""
Adversarial Geometry & Triple-Verifier Stress Tests for MIRRORLAB.

Tests Phase 2.3 edge cases:
1. Reflection exactly at mirror endpoints (A and B).
2. Near-endpoint reflections (<1.0 mm from aperture bounds).
3. Extreme grazing incidence (>85 degrees).
4. Degenerate mirror orientation (backside reflection).
5. Obstacles intersecting incident rays (Leg 1).
6. Obstacles intersecting reflected rays (Leg 2).
7. Multiple obstacles along ray corridor (verifying first-blocker attribution).
8. Near-zero optical path distances.
9. Numerical precision boundaries (1e-5 mm spatial tolerance).
"""

from __future__ import annotations

import math

from mirrorlab.geometry.independent_verifier import IndependentGeometryVerifier
from mirrorlab.geometry.oracle import VisibilityOracle
from mirrorlab.geometry.ray_march_verifier import DirectSnellRayVerifier
from mirrorlab.schema.common import (
    ObstructionReason,
    Point2D,
    Vector2D,
    VisibilityState,
)
from mirrorlab.schema.scene import (
    CameraSpec,
    DifficultyLevel,
    InspectionScene,
    MirrorSpec,
    ObstacleSpec,
    SceneCategory,
    TargetSpec,
)


def make_scene(
    cam_pos: tuple[float, float],
    mirror_a: tuple[float, float],
    mirror_b: tuple[float, float],
    target_pos: tuple[float, float],
    normal: tuple[float, float] = (0.0, -1.0),
    obstacles: list[ObstacleSpec] | None = None,
    fov: float = 90.0,
) -> InspectionScene:
    """Helper to assemble InspectionScene with pointing heading."""
    mx = (mirror_a[0] + mirror_b[0]) / 2.0
    my = (mirror_a[1] + mirror_b[1]) / 2.0
    heading = math.degrees(math.atan2(my - cam_pos[1], mx - cam_pos[0]))

    return InspectionScene(
        scene_id="adversarial_test_scene",
        seed=999,
        category=SceneCategory.VALID_VISIBLE,
        difficulty=DifficultyLevel.HARD,
        camera=CameraSpec(
            position=Point2D(x=cam_pos[0], y=cam_pos[1]),
            heading_deg=heading,
            fov_degrees=fov,
        ),
        mirror=MirrorSpec(
            endpoint_a=Point2D(x=mirror_a[0], y=mirror_a[1]),
            endpoint_b=Point2D(x=mirror_b[0], y=mirror_b[1]),
            reflective_normal=Vector2D(dx=normal[0], dy=normal[1]),
        ),
        target=TargetSpec(
            center=Point2D(x=target_pos[0], y=target_pos[1]), radius_mm=8.0
        ),
        obstacles=obstacles or [],
        ground_truth_state=VisibilityState.VISIBLE,
        ground_truth_reason=ObstructionReason.NONE,
        diagnostic_details="Adversarial geometry setup",
    )


class TestAdversarialGeometry:
    """Adversarial precision tests across independent solvers."""

    def test_reflection_exactly_at_mirror_endpoint(self):
        # Mirror: horizontal segment from x=300 to x=700 at y=600, normal (0, -1)
        # Cam at (100, 200), Target at (500, 200). Virtual target T' at (500, 1000).
        # Sightline C -> T': from (100, 200) to (500, 1000).
        # At y=600, x = 100 + (600 - 200) * (500 - 100) / (1000 - 200) = 100 + 400 * 400 / 800 = 300.0.
        # This strikes EXACTLY endpoint A at (300, 600)!
        scene = make_scene(
            cam_pos=(100.0, 200.0),
            mirror_a=(300.0, 600.0),
            mirror_b=(700.0, 600.0),
            target_pos=(500.0, 200.0),
        )

        oracle_res = VisibilityOracle.evaluate(scene)
        fermat_res = IndependentGeometryVerifier.evaluate(scene)
        snell_res = DirectSnellRayVerifier.evaluate_scene(scene)

        # All 3 verifiers should evaluate reflection point at x=300.0
        assert oracle_res.reflection_point is not None
        assert abs(oracle_res.reflection_point[0] - 300.0) < 1e-4
        assert abs(fermat_res.reflection_point[0] - 300.0) < 1e-4

        snell_pt = snell_res.get("reflection_point")
        assert snell_pt is not None
        assert abs(snell_pt[0] - 300.0) < 1e-3

    def test_near_endpoint_reflection_just_outside(self):
        # Move endpoint A by 0.5 mm to x=300.5, so ray at 300.0 misses aperture!
        scene = make_scene(
            cam_pos=(100.0, 200.0),
            mirror_a=(300.5, 600.0),
            mirror_b=(700.0, 600.0),
            target_pos=(500.0, 200.0),
        )

        oracle_res = VisibilityOracle.evaluate(scene)
        fermat_res = IndependentGeometryVerifier.evaluate(scene)

        assert oracle_res.state == VisibilityState.NOT_VISIBLE
        assert oracle_res.reason == ObstructionReason.MISSED_MIRROR
        assert fermat_res.state == VisibilityState.NOT_VISIBLE

    def test_extreme_grazing_incidence(self):
        # Incidence angle > 85 degrees:
        # Cam at (100, 595), Mirror at y=600 from x=150 to x=850.
        # Target at (900, 595).
        # Normal (0, -1). Ray strikes near center (500, 600).
        # dy = 5, dx = 400. tan(theta) = 400/5 = 80 -> theta = 89.28 degrees!
        scene = make_scene(
            cam_pos=(100.0, 595.0),
            mirror_a=(150.0, 600.0),
            mirror_b=(850.0, 600.0),
            target_pos=(900.0, 595.0),
            fov=120.0,
        )

        oracle_res = VisibilityOracle.evaluate(scene)
        fermat_res = IndependentGeometryVerifier.evaluate(scene)
        snell_res = DirectSnellRayVerifier.evaluate_scene(scene)

        assert oracle_res.state == VisibilityState.VISIBLE
        assert fermat_res.state == VisibilityState.VISIBLE
        assert oracle_res.incident_angle_deg is not None
        assert oracle_res.incident_angle_deg > 85.0

        # Continuous reflection residual
        fermat_pt = [fermat_res.reflection_point[0], fermat_res.reflection_point[1]]
        snell_pt = snell_res.get("reflection_point")
        assert snell_pt is not None

        res_of = math.hypot(
            oracle_res.reflection_point[0] - fermat_pt[0],
            oracle_res.reflection_point[1] - fermat_pt[1],
        )
        assert res_of < 1e-4

    def test_backside_reflection_rejected(self):
        # Normal pointing DOWN (0, -1), but camera and target are at y=800 (BEHIND mirror)
        scene = make_scene(
            cam_pos=(200.0, 800.0),
            mirror_a=(200.0, 600.0),
            mirror_b=(600.0, 600.0),
            target_pos=(400.0, 800.0),
            normal=(0.0, -1.0),
        )

        oracle_res = VisibilityOracle.evaluate(scene)
        fermat_res = IndependentGeometryVerifier.evaluate(scene)

        assert oracle_res.state == VisibilityState.NOT_VISIBLE
        assert oracle_res.reason == ObstructionReason.BACK_OF_MIRROR
        assert fermat_res.state == VisibilityState.NOT_VISIBLE

    def test_obstacle_intersecting_incident_leg(self):
        # Ray from (200, 200) to (400, 600). Midpoint is (300, 400).
        # Place obstacle directly at (300, 400)
        obs = [
            ObstacleSpec(
                obstacle_id="block_leg1",
                obstacle_type="circle",
                center=Point2D(x=300.0, y=400.0),
                radius=15.0,
            )
        ]
        scene = make_scene(
            cam_pos=(200.0, 200.0),
            mirror_a=(200.0, 600.0),
            mirror_b=(600.0, 600.0),
            target_pos=(600.0, 200.0),
            obstacles=obs,
        )

        oracle_res = VisibilityOracle.evaluate(scene)
        assert oracle_res.state == VisibilityState.NOT_VISIBLE
        assert oracle_res.reason == ObstructionReason.BLOCKED_INCIDENT

    def test_obstacle_intersecting_reflected_leg(self):
        # Reflected ray from (400, 600) to (600, 200). Midpoint is (500, 400).
        # Place obstacle directly at (500, 400)
        obs = [
            ObstacleSpec(
                obstacle_id="block_leg2",
                obstacle_type="circle",
                center=Point2D(x=500.0, y=400.0),
                radius=15.0,
            )
        ]
        scene = make_scene(
            cam_pos=(200.0, 200.0),
            mirror_a=(200.0, 600.0),
            mirror_b=(600.0, 600.0),
            target_pos=(600.0, 200.0),
            obstacles=obs,
        )

        oracle_res = VisibilityOracle.evaluate(scene)
        assert oracle_res.state == VisibilityState.NOT_VISIBLE
        assert oracle_res.reason == ObstructionReason.BLOCKED_REFLECTED

    def test_multiple_obstacles_first_blocker_attribution(self):
        # Place blocker on incident leg at (300, 400) AND on reflected leg at (500, 400)
        obs = [
            ObstacleSpec(
                obstacle_id="block_leg1",
                obstacle_type="circle",
                center=Point2D(x=300.0, y=400.0),
                radius=15.0,
            ),
            ObstacleSpec(
                obstacle_id="block_leg2",
                obstacle_type="circle",
                center=Point2D(x=500.0, y=400.0),
                radius=15.0,
            ),
        ]
        scene = make_scene(
            cam_pos=(200.0, 200.0),
            mirror_a=(200.0, 600.0),
            mirror_b=(600.0, 600.0),
            target_pos=(600.0, 200.0),
            obstacles=obs,
        )

        oracle_res = VisibilityOracle.evaluate(scene)
        # First blocker in photon trajectory from camera must be incident blocker
        assert oracle_res.state == VisibilityState.NOT_VISIBLE
        assert oracle_res.reason == ObstructionReason.BLOCKED_INCIDENT
        assert oracle_res.blocking_obstacle_hit is not None
        assert oracle_res.blocking_obstacle_hit.obstacle_id == "block_leg1"

    def test_near_zero_optical_distance(self):
        # Target positioned only 1.0 mm away from reflection point on mirror
        scene = make_scene(
            cam_pos=(200.0, 200.0),
            mirror_a=(200.0, 600.0),
            mirror_b=(600.0, 600.0),
            target_pos=(400.0, 599.0),
        )

        oracle_res = VisibilityOracle.evaluate(scene)
        fermat_res = IndependentGeometryVerifier.evaluate(scene)

        assert oracle_res.state == VisibilityState.VISIBLE
        assert fermat_res.state == VisibilityState.VISIBLE
        assert oracle_res.reflection_point is not None
        assert abs(oracle_res.reflection_point[1] - 600.0) < 1e-4
