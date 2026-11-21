"""
Comprehensive Unit Tests for Single Finite Flat Mirror Reflection Engine.

Tests:
1. Midpoint mirror hits (symmetric and asymmetric).
2. Exact endpoint hits (t = 0.0 and t = 1.0).
3. Misses beyond either endpoint (t < 0.0 past A, t > 1.0 past B).
4. Parallel cases.
5. Wrong-side cases (camera behind, target behind, both behind).
6. Degenerate geometry (A == B).
7. Strict Law of Reflection validation (theta_i == theta_r).
8. Path length equality: ||R - C|| + ||T - R|| == ||T' - C|| (Heron's theorem).
"""

import math

import numpy as np
import pytest

from mirrorlab.geometry.reflection import (
    ReflectionFailureReason,
    calculate_mirror_reflection,
)


class TestFiniteMirrorReflection:
    """Test suite for finite planar mirror specular reflection solver."""

    def test_midpoint_symmetric_hit(self):
        """
        Symmetric V-reflection:
        Camera at (100, 200), Target at (500, 200).
        Mirror at y = 400 from (500, 400) to (100, 400), reflective normal down (0, -1).
        By symmetry, reflection point must be exact midpoint at (300, 400).
        """
        camera = [100.0, 200.0]
        target = [500.0, 200.0]
        mirror_a = [500.0, 400.0]
        mirror_b = [100.0, 400.0]
        normal = [0.0, -1.0]

        res = calculate_mirror_reflection(
            camera, target, mirror_a, mirror_b, reflective_normal=normal
        )

        assert res.is_valid is True
        assert res.failure_reason is None
        assert res.is_camera_on_reflective_side is True
        assert res.is_target_on_reflective_side is True
        assert res.is_in_front_of_camera is True
        assert res.is_on_finite_segment is True

        # Virtual target T' at (500, 600)
        np.testing.assert_allclose(res.virtual_target, [500.0, 600.0])

        # Intersection point R at (300, 400)
        np.testing.assert_allclose(res.intersection_point, [300.0, 400.0])

        # Parameter t on mirror segment [500, 400] -> [100, 400]:
        # t = (300 - 500) / (100 - 500) = -200 / -400 = 0.5 (exact midpoint!)
        assert res.mirror_t == pytest.approx(0.5)

        # Law of reflection: theta_i == theta_r == 45 degrees
        assert res.incident_angle_rad == pytest.approx(math.radians(45.0), abs=1e-6)
        assert res.reflected_angle_rad == pytest.approx(math.radians(45.0), abs=1e-6)
        assert res.angle_difference_rad < 1e-6

        # Heron's path length equality: ||R - C|| + ||T - R|| == ||T' - C||
        expected_path_len = math.hypot(500.0 - 100.0, 600.0 - 200.0)
        assert res.total_path_length == pytest.approx(expected_path_len)

    def test_asymmetric_midsegment_hit(self):
        """
        Asymmetric reflection:
        Camera at (0, 100), Target at (300, 200).
        Mirror at y = 300 from (0, 300) to (500, 300), normal down (0, -1).
        d_C = 200, d_T = 100.
        Virtual target T' at (300, 400).
        alpha = d_C / (d_C + d_T) = 200 / 300 = 2/3.
        R = (0, 100) + 2/3 * (300, 300) = (200, 300).
        """
        camera = [0.0, 100.0]
        target = [300.0, 200.0]
        mirror_a = [0.0, 300.0]
        mirror_b = [500.0, 300.0]
        normal = [0.0, -1.0]

        res = calculate_mirror_reflection(
            camera, target, mirror_a, mirror_b, reflective_normal=normal
        )

        assert res.is_valid is True
        np.testing.assert_allclose(res.virtual_target, [300.0, 400.0])
        np.testing.assert_allclose(res.intersection_point, [200.0, 300.0])

        # t on segment from x=0 to x=500 is 200 / 500 = 0.4
        assert res.mirror_t == pytest.approx(0.4)

        # Angle of incidence == angle of reflection
        assert res.angle_difference_rad < 1e-6

    def test_hit_at_exact_endpoint_a(self):
        """Reflection hits exactly at mirror endpoint A (t = 0.0)."""
        camera = [100.0, 200.0]
        target = [500.0, 200.0]
        # Mirror starts at x=300 (which is the reflection point)
        mirror_a = [300.0, 400.0]
        mirror_b = [100.0, 400.0]
        normal = [0.0, -1.0]

        res = calculate_mirror_reflection(
            camera, target, mirror_a, mirror_b, reflective_normal=normal
        )

        assert res.is_valid is True
        np.testing.assert_allclose(res.intersection_point, [300.0, 400.0])
        assert res.mirror_t == pytest.approx(0.0)

    def test_hit_at_exact_endpoint_b(self):
        """Reflection hits exactly at mirror endpoint B (t = 1.0)."""
        camera = [100.0, 200.0]
        target = [500.0, 200.0]
        # Mirror ends at x=300
        mirror_a = [500.0, 400.0]
        mirror_b = [300.0, 400.0]
        normal = [0.0, -1.0]

        res = calculate_mirror_reflection(
            camera, target, mirror_a, mirror_b, reflective_normal=normal
        )

        assert res.is_valid is True
        np.testing.assert_allclose(res.intersection_point, [300.0, 400.0])
        assert res.mirror_t == pytest.approx(1.0)

    def test_miss_past_endpoint_a(self):
        """Ray reflects on the infinite line before endpoint A (t < 0)."""
        camera = [100.0, 200.0]
        target = [500.0, 200.0]
        # Mirror is positioned from x=200 to x=100 (reflection point at x=300 is past A)
        mirror_a = [200.0, 400.0]
        mirror_b = [100.0, 400.0]
        normal = [0.0, -1.0]

        res = calculate_mirror_reflection(
            camera, target, mirror_a, mirror_b, reflective_normal=normal
        )

        assert res.is_valid is False
        assert res.failure_reason == ReflectionFailureReason.MISSED_MIRROR_PAST_A
        assert res.mirror_t < 0.0

    def test_miss_past_endpoint_b(self):
        """Ray reflects on the infinite line beyond endpoint B (t > 1)."""
        camera = [100.0, 200.0]
        target = [500.0, 200.0]
        # Mirror is positioned from x=500 to x=400 (reflection point at x=300 is past B)
        mirror_a = [500.0, 400.0]
        mirror_b = [400.0, 400.0]
        normal = [0.0, -1.0]

        res = calculate_mirror_reflection(
            camera, target, mirror_a, mirror_b, reflective_normal=normal
        )

        assert res.is_valid is False
        assert res.failure_reason == ReflectionFailureReason.MISSED_MIRROR_PAST_B
        assert res.mirror_t > 1.0


class TestEdgeCasesAndPhysicalFailures:
    """Tests for physical violations, wrong-side geometry, and degenerate cases."""

    def test_camera_on_wrong_side(self):
        """Camera is behind the mirror's reflective coating (d_C < 0)."""
        # Mirror normal points down (0, -1), mirror at y=400
        # Camera is placed at y=500 (behind mirror!)
        camera = [300.0, 500.0]
        target = [500.0, 200.0]
        mirror_a = [500.0, 400.0]
        mirror_b = [100.0, 400.0]
        normal = [0.0, -1.0]

        res = calculate_mirror_reflection(
            camera, target, mirror_a, mirror_b, reflective_normal=normal
        )

        assert res.is_valid is False
        assert res.failure_reason == ReflectionFailureReason.WRONG_SIDE_CAMERA
        assert res.is_camera_on_reflective_side is False

    def test_target_on_wrong_side(self):
        """Target is behind the mirror's reflective coating (d_T < 0)."""
        camera = [100.0, 200.0]
        target = [300.0, 500.0]  # Behind mirror at y=400
        mirror_a = [500.0, 400.0]
        mirror_b = [100.0, 400.0]
        normal = [0.0, -1.0]

        res = calculate_mirror_reflection(
            camera, target, mirror_a, mirror_b, reflective_normal=normal
        )

        assert res.is_valid is False
        assert res.failure_reason == ReflectionFailureReason.WRONG_SIDE_TARGET
        assert res.is_target_on_reflective_side is False

    def test_both_on_wrong_side(self):
        """Both camera and target are behind the mirror."""
        camera = [100.0, 500.0]
        target = [300.0, 500.0]
        mirror_a = [500.0, 400.0]
        mirror_b = [100.0, 400.0]
        normal = [0.0, -1.0]

        res = calculate_mirror_reflection(
            camera, target, mirror_a, mirror_b, reflective_normal=normal
        )

        assert res.is_valid is False
        assert res.failure_reason == ReflectionFailureReason.WRONG_SIDE_BOTH

    def test_degenerate_mirror_coincident_endpoints(self):
        """Mirror has length zero (endpoints coincide)."""
        camera = [100.0, 200.0]
        target = [500.0, 200.0]
        mirror_a = [300.0, 400.0]
        mirror_b = [300.0, 400.0]

        res = calculate_mirror_reflection(camera, target, mirror_a, mirror_b)

        assert res.is_valid is False
        assert res.failure_reason == ReflectionFailureReason.DEGENERATE_MIRROR

    def test_slanted_mirror_45_degrees(self):
        """
        Reflection across a 45-degree mirror.
        Mirror line: x + y = 200 (segment from (0, 200) to (200, 0)).
        Reflective normal points up-right (1, 1)/sqrt(2).
        Camera at (0, 400), Target at (400, 0) - both on front side (x + y > 200).
        By symmetry across the line x + y = 200, reflection occurs at midpoint (100, 100).
        """
        camera = [0.0, 400.0]
        target = [400.0, 0.0]
        mirror_a = [0.0, 200.0]
        mirror_b = [200.0, 0.0]
        # Normal points up-right (1, 1)
        normal = [1.0, 1.0]

        res = calculate_mirror_reflection(
            camera, target, mirror_a, mirror_b, reflective_normal=normal
        )

        assert res.is_valid is True
        np.testing.assert_allclose(res.intersection_point, [100.0, 100.0], atol=1e-7)
        assert res.mirror_t == pytest.approx(0.5)
        assert res.angle_difference_rad < 1e-6
