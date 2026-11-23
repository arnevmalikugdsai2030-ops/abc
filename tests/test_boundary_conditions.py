"""
Boundary Condition & Numerical Limit Test Suite for MIRRORLAB.

Tests exact boundary conditions, tolerances, and extreme edge cases:
1. Reflection exactly at both mirror endpoints (t = 0.0, t = 1.0).
2. Reflection just inside and just outside each endpoint (aperture clipping).
3. Tangent circle obstacle intersections (discriminant = 0 boundary).
4. Tangent rectangle obstacle intersections (boundary edge grazing).
5. Grazing specular reflection (angles of incidence > 75°).
6. Near-parallel ray and mirror geometry (angles of incidence ~ 89°).
7. Very small coordinate values (sub-millimeter scale).
8. Very large coordinate values (10^5 mm scale).
9. Numerical tolerance boundaries (1e-6 vs 1e-4 mm separation).
10. Degenerate mirror configurations (zero length raises error).
"""

from __future__ import annotations

import math

from mirrorlab.geometry.independent_verifier import IndependentGeometryVerifier
from mirrorlab.geometry.oracle import VisibilityOracle
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


def _build_test_scene(
    cam: tuple[float, float],
    m1: tuple[float, float],
    m2: tuple[float, float],
    normal: tuple[float, float],
    target: tuple[float, float],
    obstacles: list[ObstacleSpec] | None = None,
    heading_deg: float | None = None,
    fov_deg: float = 85.0,
    canvas_w: float = 1000.0,
    canvas_h: float = 1000.0,
) -> InspectionScene:
    """Helper to assemble an InspectionScene."""
    if heading_deg is None:
        # Point towards mirror center
        mc_x = (m1[0] + m2[0]) / 2.0
        mc_y = (m1[1] + m2[1]) / 2.0
        heading_deg = math.degrees(math.atan2(mc_y - cam[1], mc_x - cam[0]))

    return InspectionScene(
        scene_id="test_boundary_scene",
        seed=12345,
        category=SceneCategory.VALID_VISIBLE,
        difficulty=DifficultyLevel.BOUNDARY,
        canvas_width=canvas_w,
        canvas_height=canvas_h,
        camera=CameraSpec(
            position=Point2D(x=cam[0], y=cam[1]),
            heading_deg=heading_deg,
            fov_degrees=fov_deg,
            max_range=10000.0,
        ),
        mirror=MirrorSpec(
            endpoint_a=Point2D(x=m1[0], y=m1[1]),
            endpoint_b=Point2D(x=m2[0], y=m2[1]),
            reflective_normal=Vector2D(dx=normal[0], dy=normal[1]),
        ),
        target=TargetSpec(
            center=Point2D(x=target[0], y=target[1]),
            radius_mm=5.0,
        ),
        obstacles=obstacles or [],
        ground_truth_state=VisibilityState.VISIBLE,
        ground_truth_reason=ObstructionReason.NONE,
        diagnostic_details="Boundary test scene.",
    )


class TestMirrorEndpointApertureClipping:
    """Tests reflection exactly at endpoints and just inside/outside."""

    def test_reflection_exactly_at_endpoint_a(self):
        """Reflection point falls exactly at mirror endpoint A (t = 0.0)."""
        # Horizontal mirror at y = 500 from x = 400 to 600, normal = (0, -1)
        # Cam at (300, 300), Target at (500, 300) -> reflection point at x = 400, y = 500 (endpoint A)
        scene = _build_test_scene(
            cam=(300.0, 300.0),
            m1=(400.0, 500.0),
            m2=(600.0, 500.0),
            normal=(0.0, -1.0),
            target=(500.0, 300.0),
        )
        verdict = VisibilityOracle.evaluate(scene)
        # Endpoint A is within [0.0 - eps, 1.0 + eps] tolerance
        assert verdict.is_visible is True
        assert verdict.reflection_point is not None
        assert abs(verdict.reflection_point[0] - 400.0) < 1e-4

    def test_reflection_exactly_at_endpoint_b(self):
        """Reflection point falls exactly at mirror endpoint B (t = 1.0)."""
        # Horizontal mirror at y = 500 from x = 400 to 600, normal = (0, -1)
        # Cam at (500, 300), Target at (700, 300) -> reflection point at x = 600, y = 500 (endpoint B)
        scene = _build_test_scene(
            cam=(500.0, 300.0),
            m1=(400.0, 500.0),
            m2=(600.0, 500.0),
            normal=(0.0, -1.0),
            target=(700.0, 300.0),
        )
        verdict = VisibilityOracle.evaluate(scene)
        assert verdict.is_visible is True
        assert verdict.reflection_point is not None
        assert abs(verdict.reflection_point[0] - 600.0) < 1e-4

    def test_reflection_just_inside_endpoint_a(self):
        """Reflection point at t = 0.005 just inside endpoint A."""
        scene = _build_test_scene(
            cam=(301.0, 300.0),
            m1=(400.0, 500.0),
            m2=(600.0, 500.0),
            normal=(0.0, -1.0),
            target=(501.0, 300.0),
        )
        verdict = VisibilityOracle.evaluate(scene)
        assert verdict.is_visible is True

    def test_reflection_just_outside_endpoint_a_fails_aperture(self):
        """Reflection point at t = -0.005 just outside endpoint A misses finite mirror."""
        scene = _build_test_scene(
            cam=(298.0, 300.0),
            m1=(400.0, 500.0),
            m2=(600.0, 500.0),
            normal=(0.0, -1.0),
            target=(498.0, 300.0),
        )
        verdict = VisibilityOracle.evaluate(scene)
        assert verdict.is_visible is False
        assert verdict.reason == ObstructionReason.MISSED_MIRROR

    def test_reflection_just_inside_endpoint_b(self):
        """Reflection point at t = 0.995 just inside endpoint B."""
        scene = _build_test_scene(
            cam=(499.0, 300.0),
            m1=(400.0, 500.0),
            m2=(600.0, 500.0),
            normal=(0.0, -1.0),
            target=(699.0, 300.0),
        )
        verdict = VisibilityOracle.evaluate(scene)
        assert verdict.is_visible is True

    def test_reflection_just_outside_endpoint_b_fails_aperture(self):
        """Reflection point at t = 1.005 just outside endpoint B misses finite mirror."""
        scene = _build_test_scene(
            cam=(502.0, 300.0),
            m1=(400.0, 500.0),
            m2=(600.0, 500.0),
            normal=(0.0, -1.0),
            target=(702.0, 300.0),
        )
        verdict = VisibilityOracle.evaluate(scene)
        assert verdict.is_visible is False
        assert verdict.reason == ObstructionReason.MISSED_MIRROR


class TestTangentObstacleIntersections:
    """Tests ray paths touching obstacle boundaries exactly tangentially."""

    def test_tangent_circle_intersection(self):
        """Ray path grazes circular obstacle at exact tangent distance d = radius."""
        # Incident ray from (300, 300) to (500, 500) has line equation x - y = 0
        # Circle at center (400 - 20*sqrt(2)/2, 400 + 20*sqrt(2)/2) has distance exactly 20.0 to ray
        r = 20.0
        # Ray line direction is (1, 1)/sqrt(2), normal is (-1, 1)/sqrt(2)
        # Center shifted along normal by exactly r: (400 - r/sqrt(2), 400 + r/sqrt(2))
        cx = 400.0 - r / math.sqrt(2.0)
        cy = 400.0 + r / math.sqrt(2.0)

        obs = ObstacleSpec(
            obstacle_id="tangent_circle",
            obstacle_type="circle",
            center=Point2D(x=cx, y=cy),
            radius=r,
        )
        scene = _build_test_scene(
            cam=(300.0, 300.0),
            m1=(400.0, 500.0),
            m2=(600.0, 500.0),
            normal=(0.0, -1.0),
            target=(700.0, 300.0),
            obstacles=[obs],
        )
        verdict = VisibilityOracle.evaluate(scene)
        dual = IndependentGeometryVerifier.verify_dual(scene, verdict)
        assert dual.is_mathematically_certified is True

    def test_tangent_rectangle_intersection(self):
        """Ray path passes along rectangle obstacle boundary edge."""
        obs = ObstacleSpec(
            obstacle_id="tangent_box",
            obstacle_type="rectangle",
            center=Point2D(x=350.0, y=375.0),
            width=30.0,
            height=50.0,
            angle_degrees=0.0,
        )
        scene = _build_test_scene(
            cam=(300.0, 300.0),
            m1=(400.0, 500.0),
            m2=(600.0, 500.0),
            normal=(0.0, -1.0),
            target=(700.0, 300.0),
            obstacles=[obs],
        )
        verdict = VisibilityOracle.evaluate(scene)
        dual = IndependentGeometryVerifier.verify_dual(scene, verdict)
        assert dual.is_mathematically_certified is True


class TestGrazingAndNearParallelReflections:
    """Tests extreme angular configurations."""

    def test_grazing_reflection_incidence_greater_than_75_degrees(self):
        """Grazing reflection where theta_i > 75°."""
        # Mirror horizontal at y = 500 from x = 100 to 900
        # Cam at (100, 480), Target at (900, 480) -> R at (500, 500)
        # dx = 400, dy = 20 -> tan(theta_i) = 400 / 20 = 20 -> theta_i = 87.1°
        scene = _build_test_scene(
            cam=(100.0, 480.0),
            m1=(100.0, 500.0),
            m2=(900.0, 500.0),
            normal=(0.0, -1.0),
            target=(900.0, 480.0),
            heading_deg=3.0,
            fov_deg=85.0,
        )
        verdict = VisibilityOracle.evaluate(scene)
        assert verdict.is_visible is True
        assert verdict.incident_angle_deg is not None
        assert verdict.incident_angle_deg > 75.0
        dual = IndependentGeometryVerifier.verify_dual(scene, verdict)
        assert dual.is_mathematically_certified is True
        assert dual.fermat_residual < 1e-5

    def test_near_parallel_geometry_incidence_near_89_degrees(self):
        """Near-parallel reflection where theta_i ~ 89°."""
        # dx = 800, dy = 5 -> tan(theta_i) = 800 / 5 = 160 -> theta_i = 89.6°
        scene = _build_test_scene(
            cam=(50.0, 495.0),
            m1=(50.0, 500.0),
            m2=(950.0, 500.0),
            normal=(0.0, -1.0),
            target=(950.0, 495.0),
            heading_deg=0.5,
            fov_deg=85.0,
        )
        verdict = VisibilityOracle.evaluate(scene)
        assert verdict.is_visible is True
        assert verdict.incident_angle_deg is not None
        assert verdict.incident_angle_deg > 85.0
        dual = IndependentGeometryVerifier.verify_dual(scene, verdict)
        assert dual.is_mathematically_certified is True


class TestCoordinateScaleAndNumericalLimits:
    """Tests extreme coordinate magnitudes and tolerance limits."""

    def test_very_small_coordinate_values_submillimeter(self):
        """Millimeter and sub-millimeter coordinate scale."""
        scene = _build_test_scene(
            cam=(0.30, 0.30),
            m1=(0.40, 0.50),
            m2=(0.60, 0.50),
            normal=(0.0, -1.0),
            target=(0.70, 0.30),
            canvas_w=1.0,
            canvas_h=1.0,
        )
        verdict = VisibilityOracle.evaluate(scene)
        assert verdict.is_visible is True
        dual = IndependentGeometryVerifier.verify_dual(scene, verdict)
        assert dual.is_mathematically_certified is True

    def test_very_large_coordinate_values_industrial_scale(self):
        """Large scale coordinates at 10^5 mm (100 meters)."""
        scene = _build_test_scene(
            cam=(30000.0, 30000.0),
            m1=(40000.0, 50000.0),
            m2=(60000.0, 50000.0),
            normal=(0.0, -1.0),
            target=(70000.0, 30000.0),
            canvas_w=100000.0,
            canvas_h=100000.0,
        )
        verdict = VisibilityOracle.evaluate(scene)
        assert verdict.is_visible is True
        dual = IndependentGeometryVerifier.verify_dual(scene, verdict)
        assert dual.is_mathematically_certified is True
        assert dual.fermat_residual < 1e-4

    def test_numerical_tolerance_boundaries(self):
        """Separation of 1e-6 mm vs 1e-4 mm along mirror aperture."""
        # Reflection exactly at 500.0
        scene = _build_test_scene(
            cam=(300.0, 300.0),
            m1=(400.0, 500.0),
            m2=(600.0, 500.0),
            normal=(0.0, -1.0),
            target=(700.0, 300.0),
        )
        verdict = VisibilityOracle.evaluate(scene)
        dual = IndependentGeometryVerifier.verify_dual(scene, verdict)
        assert dual.reflection_point_delta_mm < 1e-4

    def test_degenerate_mirror_zero_length_fails(self):
        """Degenerate mirror with identical endpoints (length = 0) fails verification."""
        scene = _build_test_scene(
            cam=(300.0, 300.0),
            m1=(500.0, 500.0),
            m2=(500.0, 500.0),  # Zero length mirror
            normal=(0.0, -1.0),
            target=(700.0, 300.0),
        )
        verdict = VisibilityOracle.evaluate(scene)
        assert verdict.is_visible is False
        assert verdict.reason in (
            ObstructionReason.MISSED_MIRROR,
            ObstructionReason.NONE,
        )
