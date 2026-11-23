"""
Property-Based and Consensus Tests for Triple-Redundant Geometric Verification:
1. Method 1: VisibilityOracle (Virtual Target C -> T')
2. Method 2: IndependentGeometryVerifier (Virtual Camera C' -> T + Fermat stationary path)
3. Method 3: DirectSnellRayVerifier (Householder Snell-Vector collinearity root-finding + SAT)
"""

import math

import numpy as np
import pytest

from mirrorlab.geometry.independent_verifier import IndependentGeometryVerifier
from mirrorlab.geometry.oracle import VisibilityOracle
from mirrorlab.geometry.ray_march_verifier import (
    DirectSnellRayVerifier,
)
from mirrorlab.schema.common import ObstructionReason, VisibilityState
from mirrorlab.schema.scene import (
    CameraSpec,
    DifficultyLevel,
    InspectionScene,
    MirrorSpec,
    ObstacleSpec,
    Point2D,
    SceneCategory,
    TargetSpec,
    Vector2D,
)


def create_simple_test_scene(
    cam_pos=(100.0, 100.0),
    tgt_pos=(300.0, 100.0),
    mirror_a=(50.0, 300.0),
    mirror_b=(350.0, 300.0),
    mirror_normal=(0.0, -1.0),
    obstacles=None,
    scene_id="test_triple_001",
) -> InspectionScene:
    return InspectionScene(
        scene_id=scene_id,
        seed=42,
        category=SceneCategory.VALID_VISIBLE,
        difficulty=DifficultyLevel.STANDARD,
        ground_truth_state=VisibilityState.VISIBLE,
        ground_truth_reason=ObstructionReason.NONE,
        diagnostic_details="Synthetic test scene for triple verification consensus.",
        camera=CameraSpec(
            position=Point2D(x=cam_pos[0], y=cam_pos[1]),
            heading_deg=63.435,  # points directly at reflection point (200, 300)
            fov_degrees=90.0,
        ),
        target=TargetSpec(
            center=Point2D(x=tgt_pos[0], y=tgt_pos[1]),
        ),
        mirror=MirrorSpec(
            endpoint_a=Point2D(x=mirror_a[0], y=mirror_a[1]),
            endpoint_b=Point2D(x=mirror_b[0], y=mirror_b[1]),
            reflective_normal=Vector2D(dx=mirror_normal[0], dy=mirror_normal[1]),
        ),
        obstacles=obstacles or [],
    )


class TestDirectSnellRayVerifier:
    def test_solve_reflection_point_symmetric(self):
        cam = np.array([100.0, 100.0])
        target = np.array([300.0, 100.0])
        m_a = np.array([0.0, 300.0])
        m_b = np.array([400.0, 300.0])
        normal = np.array([0.0, -1.0])

        res = DirectSnellRayVerifier.solve_reflection_point(
            cam, target, m_a, m_b, normal
        )
        assert res.is_valid is True
        assert res.reflection_point is not None
        assert math.isclose(res.reflection_point[0], 200.0, abs_tol=1e-4)
        assert math.isclose(res.reflection_point[1], 300.0, abs_tol=1e-4)
        assert res.collinearity_residual < 1e-6
        assert res.angular_disparity_deg < 1e-3

    def test_solve_reflection_point_mirror_miss(self):
        cam = np.array([100.0, 100.0])
        target = np.array([300.0, 100.0])
        # Mirror is shifted far to the left [0, 100], while reflection requires X=200
        m_a = np.array([0.0, 300.0])
        m_b = np.array([100.0, 300.0])
        normal = np.array([0.0, -1.0])

        res = DirectSnellRayVerifier.solve_reflection_point(
            cam, target, m_a, m_b, normal
        )
        assert res.is_valid is False
        assert "MISSED_FINITE_MIRROR" in res.failure_reason

    def test_solve_reflection_point_wrong_side(self):
        cam = np.array([100.0, 400.0])  # Behind mirror (mirror at Y=300 facing -Y)
        target = np.array([300.0, 400.0])
        m_a = np.array([0.0, 300.0])
        m_b = np.array([400.0, 300.0])
        normal = np.array([0.0, -1.0])

        res = DirectSnellRayVerifier.solve_reflection_point(
            cam, target, m_a, m_b, normal
        )
        assert res.is_valid is False
        assert "WRONG_SIDE" in res.failure_reason


class TestTripleVerificationConsensus:
    def test_visible_unobstructed_scene(self):
        scene = create_simple_test_scene()

        oracle_res = VisibilityOracle.evaluate(scene)
        fermat_res = IndependentGeometryVerifier.evaluate(scene)
        triple_rep = DirectSnellRayVerifier.verify_triple(scene, oracle_res, fermat_res)

        assert triple_rep.all_three_agree is True
        assert triple_rep.method1_oracle_visible is True
        assert triple_rep.method2_fermat_visible is True
        assert triple_rep.method3_snell_visible is True
        assert triple_rep.max_reflection_disparity_mm < 1e-3

    def test_incident_occlusion_consensus(self):
        # Blocker on camera to mirror segment (X=150, Y=200, radius=20)
        obs = [
            ObstacleSpec(
                obstacle_id="O_inc",
                obstacle_type="circle",
                center=Point2D(x=150.0, y=200.0),
                radius=25.0,
            )
        ]
        scene = create_simple_test_scene(obstacles=obs)

        oracle_res = VisibilityOracle.evaluate(scene)
        fermat_res = IndependentGeometryVerifier.evaluate(scene)
        triple_rep = DirectSnellRayVerifier.verify_triple(scene, oracle_res, fermat_res)

        assert triple_rep.all_three_agree is True
        assert triple_rep.method1_oracle_visible is False
        assert triple_rep.method2_fermat_visible is False
        assert triple_rep.method3_snell_visible is False

    def test_reflected_occlusion_consensus(self):
        # Blocker on mirror to target segment (X=250, Y=200, radius=25)
        obs = [
            ObstacleSpec(
                obstacle_id="O_refl",
                obstacle_type="circle",
                center=Point2D(x=250.0, y=200.0),
                radius=25.0,
            )
        ]
        scene = create_simple_test_scene(obstacles=obs)

        oracle_res = VisibilityOracle.evaluate(scene)
        fermat_res = IndependentGeometryVerifier.evaluate(scene)
        triple_rep = DirectSnellRayVerifier.verify_triple(scene, oracle_res, fermat_res)

        assert triple_rep.all_three_agree is True
        assert triple_rep.method1_oracle_visible is False
        assert triple_rep.method2_fermat_visible is False
        assert triple_rep.method3_snell_visible is False

    @pytest.mark.parametrize("seed", [10, 42, 99, 1337])
    def test_parameterized_agreement_across_rotations(self, seed):
        rng = np.random.default_rng(seed)
        # Small mirror tilt in radians
        angle = rng.uniform(-0.10, 0.10)
        cos_a, sin_a = math.cos(angle), math.sin(angle)

        # Mirror center at (200, 300), half length 200
        center_m = np.array([200.0, 300.0])
        tangent = np.array([cos_a, sin_a])
        # Inward-pointing normal (-90 deg from tangent: [sin_a, -cos_a])
        normal = np.array([sin_a, -cos_a])

        m_a = center_m - 200.0 * tangent
        m_b = center_m + 200.0 * tangent

        cam_x = rng.uniform(80.0, 140.0)
        tgt_x = rng.uniform(260.0, 320.0)

        # Compute mid point for camera heading
        mid_x = (cam_x + tgt_x) / 2.0
        heading = math.degrees(math.atan2(300.0 - 100.0, mid_x - cam_x))

        scene = create_simple_test_scene(
            cam_pos=(cam_x, 100.0),
            tgt_pos=(tgt_x, 100.0),
            mirror_a=(m_a[0], m_a[1]),
            mirror_b=(m_b[0], m_b[1]),
            mirror_normal=(normal[0], normal[1]),
        )
        # Adjust heading to point toward mirror
        scene = InspectionScene(
            scene_id=scene.scene_id,
            seed=seed,
            category=SceneCategory.VALID_VISIBLE,
            difficulty=DifficultyLevel.STANDARD,
            ground_truth_state=VisibilityState.VISIBLE,
            ground_truth_reason=ObstructionReason.NONE,
            diagnostic_details="Rotated test scene",
            camera=CameraSpec(
                position=scene.camera.position,
                heading_deg=heading,
                fov_degrees=120.0,
            ),
            target=scene.target,
            mirror=scene.mirror,
            obstacles=scene.obstacles,
        )

        oracle_res = VisibilityOracle.evaluate(scene)
        fermat_res = IndependentGeometryVerifier.evaluate(scene)
        triple_rep = DirectSnellRayVerifier.verify_triple(scene, oracle_res, fermat_res)

        assert triple_rep.all_three_agree is True
        if triple_rep.method1_oracle_visible:
            assert triple_rep.max_reflection_disparity_mm < 1e-3
