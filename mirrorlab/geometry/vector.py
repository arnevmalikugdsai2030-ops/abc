"""
Vector Mathematics & 2D Euclidean Geometry Primitives for MIRRORLAB.

This module provides deterministic, high-precision vector arithmetic,
dot/cross products, 2D rotations, line-distance calculations, and
point reflections using NumPy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

# Global numerical tolerance for floating point comparisons and degeneracy guards
EPSILON: float = 1e-7

ArrayLike = Sequence[float] | np.ndarray


class GeometryError(Exception):
    """Base exception for geometric computation errors."""


class ZeroVectorError(GeometryError):
    """Raised when an operation cannot be performed on a zero-magnitude vector."""


class DegenerateLineError(GeometryError):
    """Raised when a line segment has coincident endpoints (length zero)."""


def _to_vec2(v: ArrayLike, name: str = "vector") -> np.ndarray:
    """
    Validate and convert an input sequence or array to a 1D float64 NumPy array of shape (2,).

    Args:
        v: Array-like input representing a 2D vector or point [x, y].
        name: Name of the variable for descriptive error messages.

    Returns:
        np.ndarray: 1D array of shape (2,) and dtype float64.

    Raises:
        ValueError: If input does not have exactly 2 numerical elements.
    """
    arr = np.asarray(v, dtype=np.float64)
    if arr.ndim != 1 or arr.size != 2:
        raise ValueError(
            f"Expected 2D {name} with 2 components [x, y], got shape {arr.shape}"
        )
    return arr


def vector_add(v1: ArrayLike, v2: ArrayLike) -> np.ndarray:
    """
    Compute vector sum: v1 + v2.

    Args:
        v1: First 2D vector [x1, y1].
        v2: Second 2D vector [x2, y2].

    Returns:
        np.ndarray: Resulting vector [x1 + x2, y1 + y2].
    """
    a = _to_vec2(v1, "v1")
    b = _to_vec2(v2, "v2")
    return a + b


def vector_sub(v1: ArrayLike, v2: ArrayLike) -> np.ndarray:
    """
    Compute vector difference: v1 - v2.

    Args:
        v1: First 2D vector [x1, y1].
        v2: Second 2D vector [x2, y2].

    Returns:
        np.ndarray: Resulting vector [x1 - x2, y1 - y2].
    """
    a = _to_vec2(v1, "v1")
    b = _to_vec2(v2, "v2")
    return a - b


def scalar_mul(v: ArrayLike, s: float) -> np.ndarray:
    """
    Multiply a 2D vector by a scalar: s * v.

    Args:
        v: 2D vector [x, y].
        s: Scalar multiplier.

    Returns:
        np.ndarray: Scaled vector [s * x, s * y].
    """
    a = _to_vec2(v, "vector")
    return a * float(s)


def dot_product(v1: ArrayLike, v2: ArrayLike) -> float:
    """
    Compute the standard Euclidean scalar dot product: v1 . v2 = x1*x2 + y1*y2.

    Args:
        v1: First 2D vector.
        v2: Second 2D vector.

    Returns:
        float: Dot product scalar.
    """
    a = _to_vec2(v1, "v1")
    b = _to_vec2(v2, "v2")
    return float(np.dot(a, b))


def cross_product_2d(v1: ArrayLike, v2: ArrayLike) -> float:
    """
    Compute the 2D cross product (determinant / perpendicular dot product):
    v1 x v2 = x1*y2 - y1*x2.

    Positive if v2 is counter-clockwise from v1.
    Negative if v2 is clockwise from v1.
    Zero if v1 and v2 are collinear (parallel).

    Args:
        v1: First 2D vector.
        v2: Second 2D vector.

    Returns:
        float: Scalar 2D cross product value.
    """
    a = _to_vec2(v1, "v1")
    b = _to_vec2(v2, "v2")
    return float(a[0] * b[1] - a[1] * b[0])


def vector_magnitude(v: ArrayLike) -> float:
    """
    Compute the Euclidean magnitude (length) of a 2D vector: sqrt(x^2 + y^2).

    Args:
        v: 2D vector [x, y].

    Returns:
        float: Vector length.
    """
    a = _to_vec2(v, "vector")
    return float(np.hypot(a[0], a[1]))


def vector_normalize(v: ArrayLike, eps: float = EPSILON) -> np.ndarray:
    """
    Normalize a 2D vector to have unit length (magnitude = 1.0).

    Args:
        v: 2D vector [x, y].
        eps: Tolerance threshold below which vector is considered zero.

    Returns:
        np.ndarray: Unit vector pointing in the same direction.

    Raises:
        ZeroVectorError: If vector magnitude is smaller than eps.
    """
    a = _to_vec2(v, "vector")
    mag = float(np.hypot(a[0], a[1]))
    if mag < eps:
        raise ZeroVectorError(
            f"Cannot normalize zero or near-zero vector with magnitude {mag:.2e} < {eps:.2e}"
        )
    return a / mag


def vector_rotate(v: ArrayLike, angle_radians: float) -> np.ndarray:
    """
    Rotate a 2D vector counter-clockwise by a given angle in radians:
    x' = x * cos(theta) - y * sin(theta)
    y' = x * sin(theta) + y * cos(theta)

    Args:
        v: 2D vector [x, y].
        angle_radians: Rotation angle in radians (counter-clockwise).

    Returns:
        np.ndarray: Rotated 2D vector.
    """
    a = _to_vec2(v, "vector")
    cos_theta = math.cos(angle_radians)
    sin_theta = math.sin(angle_radians)
    rot_matrix = np.array(
        [[cos_theta, -sin_theta], [sin_theta, cos_theta]], dtype=np.float64
    )
    return rot_matrix @ a


def line_tangent_and_normal(
    line_p1: ArrayLike, line_p2: ArrayLike, eps: float = EPSILON
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute the unit tangent, unit normal, and length for a directed line segment from p1 to p2.

    By convention:
    - Tangent t = (p2 - p1) / length
    - Normal n = (-t_y, t_x) (counter-clockwise 90-degree rotation of tangent, points to the left of the line)

    Args:
        line_p1: Start point of line segment.
        line_p2: End point of line segment.
        eps: Tolerance threshold for degenerate line detection.

    Returns:
        tuple[np.ndarray, np.ndarray, float]: (unit_tangent, unit_normal, length).

    Raises:
        DegenerateLineError: If distance between line_p1 and line_p2 is less than eps.
    """
    p1 = _to_vec2(line_p1, "line_p1")
    p2 = _to_vec2(line_p2, "line_p2")
    delta = p2 - p1
    length = float(np.hypot(delta[0], delta[1]))

    if length < eps:
        raise DegenerateLineError(
            f"Line endpoints are coincident or degenerate (length={length:.2e} < {eps:.2e})"
        )

    t_hat = delta / length
    n_hat = np.array([-t_hat[1], t_hat[0]], dtype=np.float64)
    return t_hat, n_hat, length


def point_to_line_signed_distance(
    point: ArrayLike, line_p1: ArrayLike, line_p2: ArrayLike, eps: float = EPSILON
) -> float:
    """
    Compute the signed perpendicular distance from a point to an infinite line passing through line_p1 and line_p2:
    d_signed = (point - line_p1) . n_hat

    Positive if point lies on the front/normal side (to the left of p1 -> p2).
    Negative if point lies on the back/opposite side.
    Zero if point lies exactly on the line.

    Args:
        point: Query point [x, y].
        line_p1: Line start point.
        line_p2: Line end point.
        eps: Tolerance for line degeneracy check.

    Returns:
        float: Signed distance in millimeters.

    Raises:
        DegenerateLineError: If line_p1 and line_p2 are coincident.
    """
    pt = _to_vec2(point, "point")
    p1 = _to_vec2(line_p1, "line_p1")
    _, n_hat, _ = line_tangent_and_normal(p1, line_p2, eps=eps)
    return float(np.dot(pt - p1, n_hat))


def point_to_line_distance(
    point: ArrayLike, line_p1: ArrayLike, line_p2: ArrayLike, eps: float = EPSILON
) -> float:
    """
    Compute the absolute perpendicular Euclidean distance from a point to an infinite line:
    d = |d_signed|

    Args:
        point: Query point [x, y].
        line_p1: Line start point.
        line_p2: Line end point.
        eps: Tolerance for line degeneracy check.

    Returns:
        float: Non-negative Euclidean distance.
    """
    return abs(point_to_line_signed_distance(point, line_p1, line_p2, eps=eps))


def reflect_point_across_line(
    point: ArrayLike, line_p1: ArrayLike, line_p2: ArrayLike, eps: float = EPSILON
) -> np.ndarray:
    """
    Reflect a 2D point across an infinite line defined by line_p1 and line_p2:
    P' = P - 2 * d_signed * n_hat

    Derived from first principles:
    1. Projection onto line: P_proj = P - d_signed * n_hat
    2. Mirror image behind line: P' = P_proj - d_signed * n_hat = P - 2 * d_signed * n_hat

    Args:
        point: Point to reflect [x, y].
        line_p1: Line start point.
        line_p2: Line end point.
        eps: Tolerance for line degeneracy check.

    Returns:
        np.ndarray: Reflected point coordinates [x', y'].

    Raises:
        DegenerateLineError: If line_p1 and line_p2 are coincident.
    """
    pt = _to_vec2(point, "point")
    p1 = _to_vec2(line_p1, "line_p1")
    _, n_hat, _ = line_tangent_and_normal(p1, line_p2, eps=eps)
    d_signed = float(np.dot(pt - p1, n_hat))
    return pt - 2.0 * d_signed * n_hat
