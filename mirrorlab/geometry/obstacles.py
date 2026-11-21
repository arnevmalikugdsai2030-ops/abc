"""
Opaque Obstacle Models & Ray-Obstacle Occlusion Detection for MIRRORLAB.

Supports:
1. Circular opaque obstacles (center, radius).
2. Rectangular opaque obstacles (axis-aligned or arbitrarily oriented).
3. General polygon obstacles.

Tests optical ray segments (C -> R, R -> T, and direct sightline C -> T)
with endpoint tolerance trimming to prevent false-positive self-occlusion at mirror
and target contact points.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points

from mirrorlab.geometry.vector import (
    EPSILON,
    _to_vec2,
    vector_magnitude,
    vector_rotate,
)

ArrayLike = Sequence[float] | np.ndarray


class PathLeg(str, Enum):
    """Identifier for the path segment being tested for occlusion."""

    LEG_1_CAMERA_TO_MIRROR = "LEG_1_CAMERA_TO_MIRROR"
    LEG_2_MIRROR_TO_TARGET = "LEG_2_MIRROR_TO_TARGET"
    DIRECT_CAMERA_TO_TARGET = "DIRECT_CAMERA_TO_TARGET"


@dataclass(frozen=True)
class ObstacleHit:
    """
    Detailed information about a ray-obstacle collision.
    """

    obstacle_id: str
    leg: PathLeg
    hit_point: np.ndarray
    distance_along_leg: float
    total_optical_distance: float


@dataclass
class ObstacleBase:
    """Base class for all opaque obstacles."""

    obstacle_id: str

    def to_shapely_polygon(self) -> Polygon:
        """Convert obstacle to a Shapely Polygon."""
        raise NotImplementedError

    def contains_point(self, point: ArrayLike) -> bool:
        """Check if point lies strictly inside or on the obstacle boundary."""
        pt = _to_vec2(point, "point")
        poly = self.to_shapely_polygon()
        return poly.contains(Point(pt[0], pt[1])) or poly.touches(Point(pt[0], pt[1]))


@dataclass
class CircleObstacle(ObstacleBase):
    """
    Circular opaque obstacle.
    """

    center: np.ndarray
    radius: float

    def __init__(self, obstacle_id: str, center: ArrayLike, radius: float):
        super().__init__(obstacle_id=obstacle_id)
        self.center = _to_vec2(center, "center")
        self.radius = float(radius)
        if self.radius <= 0.0:
            raise ValueError(
                f"CircleObstacle radius must be positive, got {self.radius}"
            )
        self._cached_poly: Polygon | None = None

    def to_shapely_polygon(self, resolution: int = 64) -> Polygon:
        """Generate high-resolution polygonal approximation for Shapely queries."""
        if self._cached_poly is None:
            self._cached_poly = Point(self.center[0], self.center[1]).buffer(
                self.radius, quad_segs=resolution
            )
        return self._cached_poly

    def intersect_segment_analytic(
        self, p1: np.ndarray, p2: np.ndarray, eps: float = EPSILON
    ) -> float | None:
        """
        Exact analytic ray-circle intersection for segment P(t) = p1 + t*(p2 - p1), t in [0, 1].

        Returns:
            float | None: Smallest t in [0, 1] where ray enters circle, or None if no intersection.
        """
        d = p2 - p1
        f = p1 - self.center

        a = float(np.dot(d, d))
        if a < eps:
            return None

        b = 2.0 * float(np.dot(f, d))
        c = float(np.dot(f, f)) - self.radius * self.radius

        discriminant = b * b - 4.0 * a * c
        if discriminant < -eps:
            return None  # Ray misses sphere completely

        if abs(discriminant) <= eps:
            # Grazing tangent ray
            t = -b / (2.0 * a)
            return t if (0.0 <= t <= 1.0) else None

        sqrt_disc = math.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)

        # We want the entry point (smallest t >= 0)
        valid_ts = [t for t in (t1, t2) if 0.0 <= t <= 1.0]
        return min(valid_ts) if valid_ts else None


@dataclass
class RectangleObstacle(ObstacleBase):
    """
    Rectangular opaque obstacle, optionally rotated by angle_degrees.
    """

    center: np.ndarray
    width: float
    height: float
    angle_degrees: float

    def __init__(
        self,
        obstacle_id: str,
        center: ArrayLike | None = None,
        width: float = 0.0,
        height: float = 0.0,
        angle_degrees: float = 0.0,
        bounds: tuple[float, float, float, float] | None = None,
    ):
        super().__init__(obstacle_id=obstacle_id)
        if bounds is not None:
            min_x, min_y, max_x, max_y = bounds
            self.width = float(max_x - min_x)
            self.height = float(max_y - min_y)
            self.center = np.array(
                [(min_x + max_x) / 2.0, (min_y + max_y) / 2.0], dtype=np.float64
            )
            self.angle_degrees = 0.0
        else:
            if center is None:
                raise ValueError("Must provide either center or bounds")
            self.center = _to_vec2(center, "center")
            self.width = float(width)
            self.height = float(height)
            self.angle_degrees = float(angle_degrees)

        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError(
                f"RectangleObstacle width and height must be positive, got {self.width}x{self.height}"
            )

        self._cached_poly: Polygon | None = None

    def get_corners(self) -> list[np.ndarray]:
        """Compute the 4 corner points of the oriented rectangle."""
        hw = self.width / 2.0
        hh = self.height / 2.0
        local_corners = [
            np.array([-hw, -hh]),
            np.array([hw, -hh]),
            np.array([hw, hh]),
            np.array([-hw, hh]),
        ]
        rad = math.radians(self.angle_degrees)
        return [self.center + vector_rotate(c, rad) for c in local_corners]

    def to_shapely_polygon(self) -> Polygon:
        if self._cached_poly is None:
            corners = self.get_corners()
            self._cached_poly = Polygon([(c[0], c[1]) for c in corners])
        return self._cached_poly


def check_segment_occlusion(
    p1: ArrayLike,
    p2: ArrayLike,
    obstacles: Sequence[ObstacleBase],
    leg: PathLeg,
    optical_start_distance: float = 0.0,
    trim_epsilon: float = 1e-4,
    eps: float = EPSILON,
) -> ObstacleHit | None:
    """
    Test whether the directed segment from p1 to p2 is occluded by any obstacle in the list.

    If multiple obstacles block the ray, returns the FIRST obstacle struck along
    the photon direction (smallest distance from p1).

    Applies endpoint tolerance trimming:
    p1' = p1 + trim_epsilon * dir
    p2' = p2 - trim_epsilon * dir
    This prevents false-positive self-occlusion if an obstacle touches the mirror reflection
    point R or scratch target center T.

    Args:
        p1: Start point of ray segment [x, y].
        p2: End point of ray segment [x, y].
        obstacles: List of circular or rectangular obstacles.
        leg: Identification of the path leg being tested.
        optical_start_distance: Cumulative distance along the photon path before this leg.
        trim_epsilon: Distance in mm to trim from endpoints.
        eps: Numerical guard.

    Returns:
        ObstacleHit | None: Detailed collision record for the first blocker, or None if clear.
    """
    v1 = _to_vec2(p1, "p1")
    v2 = _to_vec2(p2, "p2")
    delta = v2 - v1
    length = vector_magnitude(delta)

    if length < 2.0 * trim_epsilon + eps:
        # Segment too short to be blocked
        return None

    dir_u = delta / length

    # Trimmed endpoints
    p1_trim = v1 + trim_epsilon * dir_u
    p2_trim = v2 - trim_epsilon * dir_u

    line_geom = LineString([(p1_trim[0], p1_trim[1]), (p2_trim[0], p2_trim[1])])

    closest_hit: ObstacleHit | None = None
    min_dist_along_leg = float("inf")

    for obs in obstacles:
        # Optimization: Fast bounding-box check
        poly = obs.to_shapely_polygon()
        if not line_geom.intersects(poly):
            continue

        # Extract intersection geometry
        inter = line_geom.intersection(poly)
        if inter.is_empty:
            continue

        # Find closest point on intersection geometry to p1_trim
        p1_point = Point(p1_trim[0], p1_trim[1])
        nearest_on_inter, _ = nearest_points(inter, p1_point)

        hit_x = float(nearest_on_inter.x)
        hit_y = float(nearest_on_inter.y)
        hit_pt = np.array([hit_x, hit_y], dtype=np.float64)

        # Distance from original p1
        dist_from_p1 = vector_magnitude(hit_pt - v1)

        if dist_from_p1 < min_dist_along_leg:
            min_dist_along_leg = dist_from_p1
            closest_hit = ObstacleHit(
                obstacle_id=obs.obstacle_id,
                leg=leg,
                hit_point=hit_pt,
                distance_along_leg=dist_from_p1,
                total_optical_distance=optical_start_distance + dist_from_p1,
            )

    return closest_hit


@dataclass(frozen=True)
class PathOcclusionResult:
    """
    Complete occlusion audit for an inspection scenario:
    - Leg 1: Camera -> Mirror Reflection Point R
    - Leg 2: Reflection Point R -> Target Scratch T
    - Direct Sight: Camera -> Target Scratch T
    """

    is_path_blocked: bool
    first_blocker: ObstacleHit | None = None
    leg_1_hit: ObstacleHit | None = None
    leg_2_hit: ObstacleHit | None = None
    direct_sight_hit: ObstacleHit | None = None
    is_direct_sight_blocked: bool = False


def check_two_leg_path_occlusion(
    camera_pos: ArrayLike,
    reflection_point: ArrayLike,
    target_pos: ArrayLike,
    obstacles: Sequence[ObstacleBase],
    trim_epsilon: float = 1e-4,
) -> PathOcclusionResult:
    """
    Perform a complete two-leg and direct-sight occlusion audit.

    1. Checks Leg 1: C -> R
    2. Checks Leg 2: R -> T
    3. Checks Direct: C -> T

    Returns:
        PathOcclusionResult: Structured audit identifying exact blockers and legs.
    """
    c = _to_vec2(camera_pos, "camera_pos")
    r = _to_vec2(reflection_point, "reflection_point")
    t = _to_vec2(target_pos, "target_pos")

    d_leg1 = vector_magnitude(r - c)

    # 1. Test Leg 1: Camera to Mirror
    hit1 = check_segment_occlusion(
        c,
        r,
        obstacles,
        leg=PathLeg.LEG_1_CAMERA_TO_MIRROR,
        optical_start_distance=0.0,
        trim_epsilon=trim_epsilon,
    )

    # 2. Test Leg 2: Mirror to Target
    hit2 = check_segment_occlusion(
        r,
        t,
        obstacles,
        leg=PathLeg.LEG_2_MIRROR_TO_TARGET,
        optical_start_distance=d_leg1,
        trim_epsilon=trim_epsilon,
    )

    # 3. Test Direct Sightline: Camera directly to Target
    hit_direct = check_segment_occlusion(
        c,
        t,
        obstacles,
        leg=PathLeg.DIRECT_CAMERA_TO_TARGET,
        optical_start_distance=0.0,
        trim_epsilon=trim_epsilon,
    )

    # First blocker along photon trajectory (Leg 1 precedes Leg 2)
    first_blocker = hit1 if hit1 is not None else hit2
    is_blocked = first_blocker is not None
    is_direct_blocked = hit_direct is not None

    return PathOcclusionResult(
        is_path_blocked=is_blocked,
        first_blocker=first_blocker,
        leg_1_hit=hit1,
        leg_2_hit=hit2,
        direct_sight_hit=hit_direct,
        is_direct_sight_blocked=is_direct_blocked,
    )
