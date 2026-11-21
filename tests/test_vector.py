"""
Comprehensive Unit Tests for 2D Vector Mathematics Module.
"""

import math

import numpy as np
import pytest

from mirrorlab.geometry.vector import (
    DegenerateLineError,
    ZeroVectorError,
    cross_product_2d,
    dot_product,
    line_tangent_and_normal,
    point_to_line_distance,
    point_to_line_signed_distance,
    reflect_point_across_line,
    scalar_mul,
    vector_add,
    vector_magnitude,
    vector_normalize,
    vector_rotate,
    vector_sub,
)


class TestVectorArithmetic:
    """Tests for fundamental vector arithmetic."""

    def test_vector_addition_known_values(self):
        v1 = [3.0, 4.0]
        v2 = [1.0, 2.0]
        result = vector_add(v1, v2)
        np.testing.assert_allclose(result, [4.0, 6.0])

    def test_vector_addition_with_negative_values(self):
        v1 = [-5.0, 10.0]
        v2 = [5.0, -3.0]
        result = vector_add(v1, v2)
        np.testing.assert_allclose(result, [0.0, 7.0])

    def test_vector_subtraction_known_values(self):
        v1 = [5.0, 7.0]
        v2 = [2.0, 3.0]
        result = vector_sub(v1, v2)
        np.testing.assert_allclose(result, [3.0, 4.0])

    def test_scalar_multiplication(self):
        v = [3.0, -2.0]
        result = scalar_mul(v, 2.5)
        np.testing.assert_allclose(result, [7.5, -5.0])

    def test_scalar_multiplication_zero(self):
        v = [10.0, -50.0]
        result = scalar_mul(v, 0.0)
        np.testing.assert_allclose(result, [0.0, 0.0])

    def test_invalid_shape_raises_value_error(self):
        with pytest.raises(ValueError, match="Expected 2D"):
            vector_add([1.0, 2.0, 3.0], [4.0, 5.0])


class TestVectorProductsAndNorms:
    """Tests for dot product, cross product, magnitude, and normalization."""

    def test_dot_product_orthogonal(self):
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        assert dot_product(v1, v2) == pytest.approx(0.0)

    def test_dot_product_parallel(self):
        v1 = [3.0, 4.0]
        assert dot_product(v1, v1) == pytest.approx(25.0)

    def test_cross_product_counter_clockwise(self):
        # (1, 0) x (0, 1) = +1.0
        assert cross_product_2d([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)

    def test_cross_product_clockwise(self):
        # (0, 1) x (1, 0) = -1.0
        assert cross_product_2d([0.0, 1.0], [1.0, 0.0]) == pytest.approx(-1.0)

    def test_cross_product_collinear(self):
        # Parallel vectors have cross product 0.0
        assert cross_product_2d([2.0, 3.0], [4.0, 6.0]) == pytest.approx(0.0)

    def test_vector_magnitude(self):
        assert vector_magnitude([3.0, 4.0]) == pytest.approx(5.0)
        assert vector_magnitude([5.0, 12.0]) == pytest.approx(13.0)
        assert vector_magnitude([0.0, 0.0]) == pytest.approx(0.0)

    def test_vector_normalization(self):
        norm = vector_normalize([3.0, 4.0])
        np.testing.assert_allclose(norm, [0.6, 0.8])
        assert vector_magnitude(norm) == pytest.approx(1.0)

    def test_vector_normalization_zero_raises(self):
        with pytest.raises(ZeroVectorError, match="Cannot normalize zero or near-zero"):
            vector_normalize([0.0, 0.0])

    def test_vector_normalization_near_zero_raises(self):
        with pytest.raises(ZeroVectorError):
            vector_normalize([1e-9, 1e-9], eps=1e-7)


class TestVectorRotation:
    """Tests for 2D counter-clockwise rotations."""

    def test_rotate_90_degrees(self):
        v = [1.0, 0.0]
        rotated = vector_rotate(v, math.pi / 2.0)
        np.testing.assert_allclose(rotated, [0.0, 1.0], atol=1e-12)

    def test_rotate_180_degrees(self):
        v = [1.0, 2.0]
        rotated = vector_rotate(v, math.pi)
        np.testing.assert_allclose(rotated, [-1.0, -2.0], atol=1e-12)

    def test_rotate_45_degrees(self):
        v = [1.0, 0.0]
        rotated = vector_rotate(v, math.pi / 4.0)
        expected = [math.sqrt(2) / 2.0, math.sqrt(2) / 2.0]
        np.testing.assert_allclose(rotated, expected, atol=1e-12)

    def test_rotate_full_360_degrees(self):
        v = [3.5, -7.2]
        rotated = vector_rotate(v, 2.0 * math.pi)
        np.testing.assert_allclose(rotated, v, atol=1e-12)


class TestLineAndReflectionMath:
    """Tests for tangents, normals, distances, and point reflections."""

    def test_line_tangent_and_normal_horizontal(self):
        p1 = [0.0, 0.0]
        p2 = [10.0, 0.0]
        tangent, normal, length = line_tangent_and_normal(p1, p2)
        np.testing.assert_allclose(tangent, [1.0, 0.0])
        np.testing.assert_allclose(normal, [0.0, 1.0])
        assert length == pytest.approx(10.0)
        # Verify orthogonality
        assert dot_product(tangent, normal) == pytest.approx(0.0)

    def test_line_tangent_and_normal_vertical(self):
        p1 = [0.0, 0.0]
        p2 = [0.0, 5.0]
        tangent, normal, length = line_tangent_and_normal(p1, p2)
        np.testing.assert_allclose(tangent, [0.0, 1.0])
        np.testing.assert_allclose(normal, [-1.0, 0.0])
        assert length == pytest.approx(5.0)

    def test_degenerate_line_raises_error(self):
        with pytest.raises(DegenerateLineError, match="endpoints are coincident"):
            line_tangent_and_normal([2.0, 2.0], [2.0, 2.0])

    def test_point_to_line_signed_distance(self):
        line_p1 = [0.0, 0.0]
        line_p2 = [10.0, 0.0]  # Normal points up (0, 1)

        # Point above line: positive distance
        assert point_to_line_signed_distance(
            [5.0, 4.0], line_p1, line_p2
        ) == pytest.approx(4.0)

        # Point below line: negative distance
        assert point_to_line_signed_distance(
            [5.0, -3.0], line_p1, line_p2
        ) == pytest.approx(-3.0)

        # Point on line: zero distance
        assert point_to_line_signed_distance(
            [7.0, 0.0], line_p1, line_p2
        ) == pytest.approx(0.0)

    def test_point_to_line_distance_absolute(self):
        line_p1 = [0.0, 0.0]
        line_p2 = [10.0, 0.0]
        assert point_to_line_distance([5.0, -3.0], line_p1, line_p2) == pytest.approx(
            3.0
        )

    def test_reflect_across_horizontal_line(self):
        # Reflect point (2, 4) across line y = 1 (from (0, 1) to (10, 1))
        # Normal is (0, 1). Distance is 3. Reflected point should be (2, 1 - 3) = (2, -2)
        pt = [2.0, 4.0]
        p1 = [0.0, 1.0]
        p2 = [10.0, 1.0]
        reflected = reflect_point_across_line(pt, p1, p2)
        np.testing.assert_allclose(reflected, [2.0, -2.0])

    def test_reflect_across_vertical_line(self):
        # Reflect point (3, 5) across line x = 1 (from (1, 0) to (1, 10))
        # Tangent is (0, 1), normal is (-1, 0) (points left).
        # pt = (3, 5) is to the right of line, so normal distance is (3 - 1) * (-1) = -2.
        # Reflected point = (3, 5) - 2 * (-2) * (-1, 0) = (3, 5) + 4 * (-1, 0) = (-1, 5)
        pt = [3.0, 5.0]
        p1 = [1.0, 0.0]
        p2 = [1.0, 10.0]
        reflected = reflect_point_across_line(pt, p1, p2)
        np.testing.assert_allclose(reflected, [-1.0, 5.0])

    def test_reflect_across_diagonal_line(self):
        # Reflect point (0, 1) across diagonal line y = x (from (0, 0) to (1, 1))
        # Should swap coordinates to (1, 0)
        pt = [0.0, 1.0]
        p1 = [0.0, 0.0]
        p2 = [1.0, 1.0]
        reflected = reflect_point_across_line(pt, p1, p2)
        np.testing.assert_allclose(reflected, [1.0, 0.0], atol=1e-12)

    def test_reflect_point_on_line_is_invariant(self):
        # A point lying directly on the mirror line reflects to itself
        p1 = [0.0, 0.0]
        p2 = [10.0, 10.0]
        pt = [5.0, 5.0]
        reflected = reflect_point_across_line(pt, p1, p2)
        np.testing.assert_allclose(reflected, pt, atol=1e-12)

    def test_double_reflection_recovers_original_point(self):
        # Reflecting a point twice across any line must return the original point
        pt = [42.5, -17.3]
        p1 = [-10.0, 25.0]
        p2 = [80.0, 95.0]
        once = reflect_point_across_line(pt, p1, p2)
        twice = reflect_point_across_line(once, p1, p2)
        np.testing.assert_allclose(twice, pt, atol=1e-12)
