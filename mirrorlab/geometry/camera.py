"""
Camera Model & Field-of-View (FOV) Geometric Validation for MIRRORLAB.

This module validates whether an outgoing ray from a 2D camera to a mirror point
falls within the camera's angular Field of View (FOV) cone.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

from mirrorlab.geometry.vector import (
    EPSILON,
    _to_vec2,
    dot_product,
    vector_magnitude,
    vector_normalize,
    vector_rotate,
)

ArrayLike = Sequence[float] | np.ndarray


class FOVStatus(str, Enum):
    """Diagnostic status codes for camera field-of-view testing."""

    INSIDE = "INSIDE"
    OUTSIDE = "OUTSIDE"
    ON_BOUNDARY = "ON_BOUNDARY"
    INVALID_DIRECTION = "INVALID_DIRECTION"
    INVALID_FOV_ANGLE = "INVALID_FOV_ANGLE"
    COINCIDENT_TARGET = "COINCIDENT_TARGET"


@dataclass(frozen=True)
class CameraFOVResult:
    """
    Structured outcome of the camera FOV containment test.
    """

    is_inside: bool
    status: FOVStatus

    # Outgoing ray metrics
    ray_vector: np.ndarray | None = None
    ray_distance: float | None = None
    ray_unit_direction: np.ndarray | None = None

    # Camera heading
    camera_heading_unit: np.ndarray | None = None

    # Angular measurements
    angle_to_heading_rad: float | None = None
    angle_to_heading_deg: float | None = None
    fov_half_angle_rad: float = 0.0
    fov_half_angle_deg: float = 0.0

    # Cosine comparisons
    cos_angle: float | None = None
    cos_threshold: float = 0.0

    # Angular margin: positive if inside, negative if outside, ~0 on boundary
    angular_margin_deg: float | None = None


def validate_camera_fov(
    camera_pos: ArrayLike,
    camera_forward: ArrayLike | float,
    fov_degrees: float,
    target_point: ArrayLike,
    eps: float = EPSILON,
) -> CameraFOVResult:
    """
    Determine whether the outgoing line-of-sight ray from camera_pos to target_point
    lies within the camera's angular Field of View cone.

    The ray vector points from the camera to the target:
        v_ray = target_point - camera_pos

    The angle alpha between v_ray and the camera forward direction h_hat is tested:
        cos(alpha) = (v_ray . h_hat) / ||v_ray||
        threshold = cos(fov_degrees / 2)
        is_inside = cos(alpha) >= threshold

    Args:
        camera_pos: 2D coordinates of camera [x, y].
        camera_forward: Either a 2D vector [dx, dy] or an angle in degrees/radians.
            If a 2D vector, it represents the optical axis direction.
            If a scalar float, it represents heading angle in degrees (0 = +X, 90 = +Y).
        fov_degrees: Total angular field of view in degrees (e.g. 60.0). Must be in (0, 360).
        target_point: Point on mirror or object to test [x, y].
        eps: Numerical tolerance for boundary detection and zero guards.

    Returns:
        CameraFOVResult: Structured evaluation with angles and diagnostic status.
    """
    c = _to_vec2(camera_pos, "camera_pos")
    tgt = _to_vec2(target_point, "target_point")

    # Validate FOV angle range
    if fov_degrees <= 0.0 or fov_degrees >= 360.0 or math.isnan(fov_degrees):
        return CameraFOVResult(
            is_inside=False,
            status=FOVStatus.INVALID_FOV_ANGLE,
            fov_half_angle_deg=fov_degrees / 2.0
            if not math.isnan(fov_degrees)
            else 0.0,
            fov_half_angle_rad=math.radians(fov_degrees / 2.0)
            if not math.isnan(fov_degrees)
            else 0.0,
        )

    half_fov_deg = fov_degrees / 2.0
    half_fov_rad = math.radians(half_fov_deg)
    cos_threshold = math.cos(half_fov_rad)

    # Parse and validate camera forward heading vector
    if isinstance(camera_forward, (int, float)):
        # Angle in degrees
        rad = math.radians(float(camera_forward))
        h_hat = np.array([math.cos(rad), math.sin(rad)], dtype=np.float64)
    else:
        try:
            h_raw = _to_vec2(camera_forward, "camera_forward")
            mag_h = vector_magnitude(h_raw)
            if mag_h < eps:
                return CameraFOVResult(
                    is_inside=False,
                    status=FOVStatus.INVALID_DIRECTION,
                    fov_half_angle_deg=half_fov_deg,
                    fov_half_angle_rad=half_fov_rad,
                    cos_threshold=cos_threshold,
                )
            h_hat = h_raw / mag_h
        except ValueError:
            return CameraFOVResult(
                is_inside=False,
                status=FOVStatus.INVALID_DIRECTION,
                fov_half_angle_deg=half_fov_deg,
                fov_half_angle_rad=half_fov_rad,
                cos_threshold=cos_threshold,
            )

    # Compute ray vector from camera to target
    ray_vec = tgt - c
    dist = vector_magnitude(ray_vec)

    if dist < eps:
        return CameraFOVResult(
            is_inside=False,
            status=FOVStatus.COINCIDENT_TARGET,
            ray_vector=ray_vec,
            ray_distance=dist,
            camera_heading_unit=h_hat,
            fov_half_angle_deg=half_fov_deg,
            fov_half_angle_rad=half_fov_rad,
            cos_threshold=cos_threshold,
        )

    ray_u = ray_vec / dist

    # Compute dot product and angle
    cos_alpha = float(np.clip(dot_product(ray_u, h_hat), -1.0, 1.0))
    alpha_rad = math.acos(cos_alpha)
    alpha_deg = math.degrees(alpha_rad)

    angular_margin = half_fov_deg - alpha_deg

    # Check boundary tolerance (within 1e-5 radians ~ 0.0005 degrees)
    is_boundary = abs(alpha_rad - half_fov_rad) < 1e-5

    if is_boundary:
        return CameraFOVResult(
            is_inside=True,  # Points on boundary are considered visible in Euclidean math
            status=FOVStatus.ON_BOUNDARY,
            ray_vector=ray_vec,
            ray_distance=dist,
            ray_unit_direction=ray_u,
            camera_heading_unit=h_hat,
            angle_to_heading_rad=alpha_rad,
            angle_to_heading_deg=alpha_deg,
            fov_half_angle_rad=half_fov_rad,
            fov_half_angle_deg=half_fov_deg,
            cos_angle=cos_alpha,
            cos_threshold=cos_threshold,
            angular_margin_deg=angular_margin,
        )

    is_inside = cos_alpha >= cos_threshold
    status = FOVStatus.INSIDE if is_inside else FOVStatus.OUTSIDE

    return CameraFOVResult(
        is_inside=is_inside,
        status=status,
        ray_vector=ray_vec,
        ray_distance=dist,
        ray_unit_direction=ray_u,
        camera_heading_unit=h_hat,
        angle_to_heading_rad=alpha_rad,
        angle_to_heading_deg=alpha_deg,
        fov_half_angle_rad=half_fov_rad,
        fov_half_angle_deg=half_fov_deg,
        cos_angle=cos_alpha,
        cos_threshold=cos_threshold,
        angular_margin_deg=angular_margin,
    )


class Camera2D:
    """
    2D Industrial Camera Model with position, heading direction, and FOV cone.
    """

    def __init__(
        self,
        position: ArrayLike,
        heading: ArrayLike | float,
        fov_degrees: float = 60.0,
        max_range: float | None = None,
    ):
        self.position = _to_vec2(position, "position")
        self.fov_degrees = float(fov_degrees)
        self.max_range = float(max_range) if max_range is not None else None

        if isinstance(heading, (int, float)):
            self.heading_angle_deg = float(heading)
            rad = math.radians(self.heading_angle_deg)
            self.forward = np.array([math.cos(rad), math.sin(rad)], dtype=np.float64)
        else:
            self.forward = vector_normalize(_to_vec2(heading, "heading"))
            self.heading_angle_deg = math.degrees(
                math.atan2(self.forward[1], self.forward[0])
            )

    def check_point_in_fov(
        self, target_point: ArrayLike, eps: float = EPSILON
    ) -> CameraFOVResult:
        """Test whether target_point falls inside this camera's FOV."""
        res = validate_camera_fov(
            self.position,
            self.forward,
            self.fov_degrees,
            target_point,
            eps=eps,
        )
        if (
            self.max_range is not None
            and res.ray_distance is not None
            and res.ray_distance > self.max_range
        ):
            return CameraFOVResult(
                is_inside=False,
                status=FOVStatus.OUTSIDE,
                ray_vector=res.ray_vector,
                ray_distance=res.ray_distance,
                ray_unit_direction=res.ray_unit_direction,
                camera_heading_unit=res.camera_heading_unit,
                angle_to_heading_rad=res.angle_to_heading_rad,
                angle_to_heading_deg=res.angle_to_heading_deg,
                fov_half_angle_rad=res.fov_half_angle_rad,
                fov_half_angle_deg=res.fov_half_angle_deg,
                cos_angle=res.cos_angle,
                cos_threshold=res.cos_threshold,
                angular_margin_deg=res.angular_margin_deg,
            )
        return res

    def get_fov_boundary_rays(
        self, length: float = 500.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the left and right boundary vectors defining the FOV wedge.

        Returns:
            tuple[np.ndarray, np.ndarray]: (left_boundary_point, right_boundary_point).
        """
        half_fov_rad = math.radians(self.fov_degrees / 2.0)
        # Left boundary: rotate forward CCW by +half_fov
        left_dir = vector_rotate(self.forward, half_fov_rad)
        # Right boundary: rotate forward CW by -half_fov
        right_dir = vector_rotate(self.forward, -half_fov_rad)

        left_pt = self.position + length * left_dir
        right_pt = self.position + length * right_dir
        return left_pt, right_pt
