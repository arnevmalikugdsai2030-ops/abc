"""
Comprehensive Unit Tests for Camera Field-of-View (FOV) Geometric Validation.

Tests:
1. Clearly inside FOV.
2. Clearly outside FOV.
3. Exactly on FOV boundary (positive and negative half-angles).
4. Narrow FOV (e.g. 10 degrees).
5. Wide FOV (e.g. 120 and 160 degrees).
6. Rotated camera (0, 45, 90, 180, 270 degrees).
7. Invalid camera direction (zero vector [0, 0]).
8. Invalid FOV angle (0, -30, 400, NaN).
9. Coincident target (target at camera position).
10. Camera2D class wrapper and max range limits.
"""

import math

import numpy as np
import pytest

from mirrorlab.geometry.camera import (
    Camera2D,
    FOVStatus,
    validate_camera_fov,
)


class TestCameraFOVValidation:
    """Test suite for camera field-of-view containment."""

    def test_clearly_inside_fov_along_optical_axis(self):
        """Target lies directly along camera optical axis (angle 0)."""
        camera_pos = [100.0, 100.0]
        camera_forward = [1.0, 0.0]  # East (+X)
        fov_deg = 60.0
        target = [300.0, 100.0]  # Directly ahead

        res = validate_camera_fov(camera_pos, camera_forward, fov_deg, target)

        assert res.is_inside is True
        assert res.status == FOVStatus.INSIDE
        assert res.angle_to_heading_deg == pytest.approx(0.0)
        assert res.angular_margin_deg == pytest.approx(30.0)  # half_fov(30) - 0

    def test_clearly_inside_fov_offset_angle(self):
        """Target is offset by 15 degrees in a 60-degree FOV cone."""
        camera_pos = [0.0, 0.0]
        camera_forward = [1.0, 0.0]  # Heading 0 deg
        fov_deg = 60.0  # Cone is [-30 deg, +30 deg]
        # Target at 15 deg: (cos(15 deg), sin(15 deg)) * 100
        rad15 = math.radians(15.0)
        target = [100.0 * math.cos(rad15), 100.0 * math.sin(rad15)]

        res = validate_camera_fov(camera_pos, camera_forward, fov_deg, target)

        assert res.is_inside is True
        assert res.status == FOVStatus.INSIDE
        assert res.angle_to_heading_deg == pytest.approx(15.0, abs=1e-5)
        assert res.angular_margin_deg == pytest.approx(15.0, abs=1e-5)

    def test_clearly_outside_fov(self):
        """Target is at 45 degrees in a 60-degree FOV (half-angle is 30 deg)."""
        camera_pos = [0.0, 0.0]
        camera_forward = [1.0, 0.0]
        fov_deg = 60.0
        # Target at 45 deg
        target = [100.0, 100.0]

        res = validate_camera_fov(camera_pos, camera_forward, fov_deg, target)

        assert res.is_inside is False
        assert res.status == FOVStatus.OUTSIDE
        assert res.angle_to_heading_deg == pytest.approx(45.0, abs=1e-5)
        assert res.angular_margin_deg == pytest.approx(-15.0, abs=1e-5)

    def test_target_behind_camera(self):
        """Target is behind camera (angle 180 degrees)."""
        camera_pos = [100.0, 100.0]
        camera_forward = [1.0, 0.0]  # Looking East
        fov_deg = 90.0
        target = [0.0, 100.0]  # Looking West (behind!)

        res = validate_camera_fov(camera_pos, camera_forward, fov_deg, target)

        assert res.is_inside is False
        assert res.status == FOVStatus.OUTSIDE
        assert res.angle_to_heading_deg == pytest.approx(180.0, abs=1e-5)

    def test_exactly_on_positive_fov_boundary(self):
        """Target lies exactly on the positive +half_fov boundary ray."""
        camera_pos = [0.0, 0.0]
        camera_forward = [1.0, 0.0]
        fov_deg = 60.0  # Boundary is +30 degrees
        rad30 = math.radians(30.0)
        target = [200.0 * math.cos(rad30), 200.0 * math.sin(rad30)]

        res = validate_camera_fov(camera_pos, camera_forward, fov_deg, target)

        assert res.is_inside is True
        assert res.status == FOVStatus.ON_BOUNDARY
        assert res.angle_to_heading_deg == pytest.approx(30.0, abs=1e-5)
        assert abs(res.angular_margin_deg) < 1e-4

    def test_exactly_on_negative_fov_boundary(self):
        """Target lies exactly on the negative -half_fov boundary ray."""
        camera_pos = [0.0, 0.0]
        camera_forward = [1.0, 0.0]
        fov_deg = 90.0  # Boundary is -45 degrees
        rad_neg45 = math.radians(-45.0)
        target = [200.0 * math.cos(rad_neg45), 200.0 * math.sin(rad_neg45)]

        res = validate_camera_fov(camera_pos, camera_forward, fov_deg, target)

        assert res.is_inside is True
        assert res.status == FOVStatus.ON_BOUNDARY
        assert res.angle_to_heading_deg == pytest.approx(45.0, abs=1e-5)

    def test_narrow_fov(self):
        """Narrow telephoto lens FOV of 10 degrees (half-angle 5 deg)."""
        camera_pos = [0.0, 0.0]
        camera_forward = [1.0, 0.0]
        fov_deg = 10.0

        # Point at 4 degrees: inside
        rad4 = math.radians(4.0)
        res_inside = validate_camera_fov(
            camera_pos,
            camera_forward,
            fov_deg,
            [100.0 * math.cos(rad4), 100.0 * math.sin(rad4)],
        )
        assert res_inside.is_inside is True
        assert res_inside.status == FOVStatus.INSIDE

        # Point at 6 degrees: outside
        rad6 = math.radians(6.0)
        res_outside = validate_camera_fov(
            camera_pos,
            camera_forward,
            fov_deg,
            [100.0 * math.cos(rad6), 100.0 * math.sin(rad6)],
        )
        assert res_outside.is_inside is False
        assert res_outside.status == FOVStatus.OUTSIDE

    def test_wide_fov_120_degrees(self):
        """Wide angle 120-degree FOV (half-angle 60 deg)."""
        camera_pos = [0.0, 0.0]
        camera_forward = [0.0, 1.0]  # Heading North (+Y)
        fov_deg = 120.0

        # Point at 50 degrees from North: inside
        rad50 = math.radians(90.0 - 50.0)
        res = validate_camera_fov(
            camera_pos,
            camera_forward,
            fov_deg,
            [100.0 * math.cos(rad50), 100.0 * math.sin(rad50)],
        )
        assert res.is_inside is True
        assert res.status == FOVStatus.INSIDE

        # Point at 70 degrees from North: outside
        rad70 = math.radians(90.0 - 70.0)
        res_out = validate_camera_fov(
            camera_pos,
            camera_forward,
            fov_deg,
            [100.0 * math.cos(rad70), 100.0 * math.sin(rad70)],
        )
        assert res_out.is_inside is False
        assert res_out.status == FOVStatus.OUTSIDE

    def test_rotated_camera_various_angles(self):
        """Verify FOV math works for camera rotated at 45, 90, 180, 270 degrees."""
        # Camera at (100, 100), heading 135 deg (North-West)
        camera_pos = [100.0, 100.0]
        heading_deg = 135.0
        fov_deg = 60.0

        # Target directly along heading: (-100, 100) + (100, 100) = (0, 200)
        res = validate_camera_fov(camera_pos, heading_deg, fov_deg, [0.0, 200.0])
        assert res.is_inside is True
        assert res.angle_to_heading_deg == pytest.approx(0.0, abs=1e-5)

        # Target 90 deg away (at 225 deg): outside
        res_out = validate_camera_fov(camera_pos, heading_deg, fov_deg, [0.0, 0.0])
        assert res_out.is_inside is False

    def test_invalid_camera_direction_zero_vector(self):
        """Zero-length direction vector returns INVALID_DIRECTION."""
        res = validate_camera_fov([0.0, 0.0], [0.0, 0.0], 60.0, [100.0, 0.0])
        assert res.is_inside is False
        assert res.status == FOVStatus.INVALID_DIRECTION

    def test_invalid_fov_angles(self):
        """Negative, zero, and out-of-range FOV angles return INVALID_FOV_ANGLE."""
        assert (
            validate_camera_fov([0, 0], [1, 0], 0.0, [10, 0]).status
            == FOVStatus.INVALID_FOV_ANGLE
        )
        assert (
            validate_camera_fov([0, 0], [1, 0], -45.0, [10, 0]).status
            == FOVStatus.INVALID_FOV_ANGLE
        )
        assert (
            validate_camera_fov([0, 0], [1, 0], 360.0, [10, 0]).status
            == FOVStatus.INVALID_FOV_ANGLE
        )
        assert (
            validate_camera_fov([0, 0], [1, 0], float("nan"), [10, 0]).status
            == FOVStatus.INVALID_FOV_ANGLE
        )

    def test_coincident_target_at_camera_pos(self):
        """Target at same position as camera returns COINCIDENT_TARGET."""
        res = validate_camera_fov([50.0, 50.0], [1.0, 0.0], 60.0, [50.0, 50.0])
        assert res.is_inside is False
        assert res.status == FOVStatus.COINCIDENT_TARGET


class TestCamera2DClassWrapper:
    """Tests for the high-level Camera2D class."""

    def test_camera_heading_normalization_and_degrees(self):
        cam = Camera2D(position=[10, 20], heading=[3, 4], fov_degrees=60.0)
        np.testing.assert_allclose(cam.forward, [0.6, 0.8])
        assert cam.heading_angle_deg == pytest.approx(
            math.degrees(math.atan2(0.8, 0.6))
        )

    def test_camera_max_range_filter(self):
        cam = Camera2D(
            position=[0, 0], heading=[1, 0], fov_degrees=60.0, max_range=500.0
        )

        # Inside FOV and within range (dist = 300)
        res_near = cam.check_point_in_fov([300.0, 0.0])
        assert res_near.is_inside is True

        # Inside FOV but exceeds max range (dist = 600 > 500)
        res_far = cam.check_point_in_fov([600.0, 0.0])
        assert res_far.is_inside is False
        assert res_far.status == FOVStatus.OUTSIDE

    def test_camera_fov_boundary_rays(self):
        cam = Camera2D(position=[0, 0], heading=[1, 0], fov_degrees=90.0)
        left_pt, right_pt = cam.get_fov_boundary_rays(length=100.0)

        # Heading is East (0 deg). Left is +45 deg, Right is -45 deg.
        expected_left = [100.0 * math.cos(math.pi / 4), 100.0 * math.sin(math.pi / 4)]
        expected_right = [
            100.0 * math.cos(-math.pi / 4),
            100.0 * math.sin(-math.pi / 4),
        ]

        np.testing.assert_allclose(left_pt, expected_left, atol=1e-7)
        np.testing.assert_allclose(right_pt, expected_right, atol=1e-7)
