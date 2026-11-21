"""
Comprehensive Integration Tests for Obstacle Occlusion and the Visibility Oracle.

Tests:
1. Circular opaque obstacles.
2. Rectangular opaque obstacles (axis-aligned and rotated).
3. Irrelevant obstacles (outside the optical path).
4. Obstacle blocking Leg 1 (Camera to Mirror).
5. Obstacle blocking Leg 2 (Mirror to Target).
6. Multiple blockers (verifying first blocker is reported).
7. Boundary contact and near-endpoint tolerance trimming (no false positive self-occlusion).
8. Full Visibility Oracle integration:
   - VISIBLE (clean reflected path).
   - BLOCKED_INCIDENT (obstacle on Leg 1).
   - BLOCKED_REFLECTED (obstacle on Leg 2).
   - OUT_OF_FOV (camera misaligned).
   - MISSED_MIRROR (ray misses mirror bounds).
   - BACK_OF_MIRROR (camera or target behind mirror).
"""

import numpy as np
import pytest

from mirrorlab.geometry.obstacles import (
    CircleObstacle,
    PathLeg,
    RectangleObstacle,
    check_segment_occlusion,
    check_two_leg_path_occlusion,
)
from mirrorlab.geometry.oracle import evaluate_visibility_oracle
from mirrorlab.schema.common import ObstructionReason, VisibilityState


class TestObstacleIntersectionBasics:
    """Tests for individual obstacle collision detection."""

    def test_circle_obstacle_direct_hit(self):
        """Ray passes directly through center of circle."""
        circle = CircleObstacle("obs_circ", center=[50.0, 50.0], radius=20.0)
        p1 = [0.0, 50.0]
        p2 = [100.0, 50.0]

        hit = check_segment_occlusion(
            p1, p2, [circle], leg=PathLeg.LEG_1_CAMERA_TO_MIRROR
        )

        assert hit is not None
        assert hit.obstacle_id == "obs_circ"
        assert hit.leg == PathLeg.LEG_1_CAMERA_TO_MIRROR
        # Circle starts at x = 50 - 20 = 30
        assert hit.distance_along_leg == pytest.approx(30.0, abs=0.2)
        assert hit.hit_point[1] == pytest.approx(50.0, abs=0.2)

    def test_rectangle_obstacle_direct_hit(self):
        """Ray passes through an axis-aligned rectangular block."""
        rect = RectangleObstacle("obs_rect", bounds=(40.0, 30.0, 60.0, 70.0))
        p1 = [0.0, 50.0]
        p2 = [100.0, 50.0]

        hit = check_segment_occlusion(
            p1, p2, [rect], leg=PathLeg.LEG_2_MIRROR_TO_TARGET
        )

        assert hit is not None
        assert hit.obstacle_id == "obs_rect"
        # Hits left face at x = 40.0
        assert hit.hit_point[0] == pytest.approx(40.0, abs=1e-3)
        assert hit.hit_point[1] == pytest.approx(50.0, abs=1e-3)

    def test_irrelevant_obstacle_miss(self):
        """Obstacle is far away from the ray path."""
        circle = CircleObstacle("obs_far", center=[50.0, 500.0], radius=20.0)
        rect = RectangleObstacle("obs_rect_far", bounds=(200.0, 200.0, 250.0, 250.0))
        p1 = [0.0, 50.0]
        p2 = [100.0, 50.0]

        hit = check_segment_occlusion(
            p1, p2, [circle, rect], leg=PathLeg.LEG_1_CAMERA_TO_MIRROR
        )
        assert hit is None

    def test_rotated_rectangle_obstacle(self):
        """Rotated rectangle blocking a diagonal ray."""
        # Rectangle centered at (50, 50) rotated 45 degrees
        rect = RectangleObstacle(
            "obs_rot", center=[50.0, 50.0], width=20.0, height=20.0, angle_degrees=45.0
        )
        p1 = [0.0, 0.0]
        p2 = [100.0, 100.0]

        hit = check_segment_occlusion(
            p1, p2, [rect], leg=PathLeg.LEG_1_CAMERA_TO_MIRROR
        )
        assert hit is not None
        assert hit.obstacle_id == "obs_rot"

    def test_obstacle_near_endpoints_tolerance_trimming(self):
        """Obstacle placed right up against endpoint R without blocking the ray body."""
        # An obstacle that touches mirror point (100, 100) exactly at its corner
        # but does not overlap the interior of the ray segment from (0, 0) to (100, 100)
        rect = RectangleObstacle("obs_adjacent", bounds=(100.0, 100.0, 150.0, 150.0))
        p1 = [0.0, 0.0]
        p2 = [100.0, 100.0]

        # With endpoint trimming, endpoint touch is not flagged as a blocker
        hit = check_segment_occlusion(
            p1, p2, [rect], leg=PathLeg.LEG_1_CAMERA_TO_MIRROR, trim_epsilon=1e-3
        )
        assert hit is None


class TestTwoLegPathOcclusion:
    """Tests for the complete two-leg optical path."""

    def test_obstacle_on_leg_1_only(self):
        """Obstacle placed between Camera (100, 200) and Mirror (300, 400)."""
        camera = [100.0, 200.0]
        reflection_point = [300.0, 400.0]
        target = [500.0, 200.0]

        # Block at midpoint of Leg 1: (200, 300)
        blocker = CircleObstacle("blocker_leg1", center=[200.0, 300.0], radius=15.0)

        res = check_two_leg_path_occlusion(camera, reflection_point, target, [blocker])

        assert res.is_path_blocked is True
        assert res.leg_1_hit is not None
        assert res.leg_1_hit.obstacle_id == "blocker_leg1"
        assert res.leg_2_hit is None
        assert res.first_blocker == res.leg_1_hit

    def test_obstacle_on_leg_2_only(self):
        """Obstacle placed between Mirror (300, 400) and Target (500, 200)."""
        camera = [100.0, 200.0]
        reflection_point = [300.0, 400.0]
        target = [500.0, 200.0]

        # Block at midpoint of Leg 2: (400, 300)
        blocker = RectangleObstacle("blocker_leg2", bounds=(380.0, 280.0, 420.0, 320.0))

        res = check_two_leg_path_occlusion(camera, reflection_point, target, [blocker])

        assert res.is_path_blocked is True
        assert res.leg_1_hit is None
        assert res.leg_2_hit is not None
        assert res.leg_2_hit.obstacle_id == "blocker_leg2"
        assert res.first_blocker == res.leg_2_hit

    def test_multiple_blockers_reports_first_blocker(self):
        """Obstacles on both Leg 1 and Leg 2. First blocker must be Leg 1."""
        camera = [100.0, 200.0]
        reflection_point = [300.0, 400.0]
        target = [500.0, 200.0]

        obs1 = CircleObstacle("blocker_leg1", center=[200.0, 300.0], radius=10.0)
        obs2 = CircleObstacle("blocker_leg2", center=[400.0, 300.0], radius=10.0)

        res = check_two_leg_path_occlusion(
            camera, reflection_point, target, [obs1, obs2]
        )

        assert res.is_path_blocked is True
        assert res.first_blocker is not None
        assert res.first_blocker.obstacle_id == "blocker_leg1"


class TestVisibilityOracleIntegration:
    """Full integration tests combining reflection, FOV, and obstacle occlusion."""

    @pytest.fixture
    def base_scene(self):
        """Standard valid reflection scene: Camera(100, 200), Target(500, 200), Mirror at y=400."""
        return {
            "camera_pos": [100.0, 200.0],
            "camera_forward": [
                1.0,
                1.0,
            ],  # Looking 45 deg North-East directly towards mirror point (300, 400)
            "camera_fov_degrees": 60.0,
            "target_pos": [500.0, 200.0],
            "mirror_a": [500.0, 400.0],
            "mirror_b": [100.0, 400.0],
            "reflective_normal": [0.0, -1.0],
        }

    def test_oracle_clean_visible_scene(self, base_scene):
        """Scene with no obstacles in the optical path (must be VISIBLE)."""
        # Obstacle blocks direct sight between (100, 200) and (500, 200)
        cavity_wall = RectangleObstacle(
            "cavity_partition", bounds=(280.0, 150.0, 320.0, 250.0)
        )

        res = evaluate_visibility_oracle(obstacles=[cavity_wall], **base_scene)

        assert res.state == VisibilityState.VISIBLE
        assert res.is_visible is True
        assert res.reason == ObstructionReason.NONE
        assert (
            res.is_direct_sight_blocked is True
        )  # Direct sight is blocked by partition!
        assert len(res.path_points) == 3  # [C, R, T]
        np.testing.assert_allclose(res.reflection_point, [300.0, 400.0])

    def test_oracle_blocked_incident_leg(self, base_scene):
        """Obstacle on Leg 1 (Camera to Mirror) -> BLOCKED_INCIDENT."""
        cavity_wall = RectangleObstacle(
            "partition", bounds=(280.0, 150.0, 320.0, 250.0)
        )
        leg1_obs = CircleObstacle("leg1_block", center=[200.0, 300.0], radius=15.0)

        res = evaluate_visibility_oracle(
            obstacles=[cavity_wall, leg1_obs], **base_scene
        )

        assert res.state == VisibilityState.NOT_VISIBLE
        assert res.is_visible is False
        assert res.reason == ObstructionReason.BLOCKED_INCIDENT
        assert res.blocking_obstacle_hit is not None
        assert res.blocking_obstacle_hit.obstacle_id == "leg1_block"

    def test_oracle_blocked_reflected_leg(self, base_scene):
        """Obstacle on Leg 2 (Mirror to Target) -> BLOCKED_REFLECTED."""
        cavity_wall = RectangleObstacle(
            "partition", bounds=(280.0, 150.0, 320.0, 250.0)
        )
        leg2_obs = RectangleObstacle("leg2_block", bounds=(380.0, 280.0, 420.0, 320.0))

        res = evaluate_visibility_oracle(
            obstacles=[cavity_wall, leg2_obs], **base_scene
        )

        assert res.state == VisibilityState.NOT_VISIBLE
        assert res.is_visible is False
        assert res.reason == ObstructionReason.BLOCKED_REFLECTED
        assert res.blocking_obstacle_hit is not None
        assert res.blocking_obstacle_hit.obstacle_id == "leg2_block"

    def test_oracle_out_of_camera_fov(self, base_scene):
        """Camera turned away from mirror -> OUT_OF_FOV."""
        scene = dict(base_scene)
        scene["camera_forward"] = [
            -1.0,
            0.0,
        ]  # Camera pointing West (away from mirror at East)

        res = evaluate_visibility_oracle(**scene)

        assert res.state == VisibilityState.NOT_VISIBLE
        assert res.is_visible is False
        assert res.reason == ObstructionReason.OUT_OF_FOV

    def test_oracle_missed_mirror(self, base_scene):
        """Mirror is too short; reflection ray misses the glass -> MISSED_MIRROR."""
        scene = dict(base_scene)
        # Mirror is at x=200 to x=100 (reflection point at x=300 is off the mirror!)
        scene["mirror_a"] = [200.0, 400.0]
        scene["mirror_b"] = [100.0, 400.0]

        res = evaluate_visibility_oracle(**scene)

        assert res.state == VisibilityState.NOT_VISIBLE
        assert res.is_visible is False
        assert res.reason == ObstructionReason.MISSED_MIRROR

    def test_oracle_back_of_mirror(self, base_scene):
        """Target placed behind the mirror backing -> BACK_OF_MIRROR."""
        scene = dict(base_scene)
        scene["target_pos"] = [500.0, 500.0]  # Behind mirror at y=400

        res = evaluate_visibility_oracle(**scene)

        assert res.state == VisibilityState.NOT_VISIBLE
        assert res.is_visible is False
        assert res.reason == ObstructionReason.BACK_OF_MIRROR
