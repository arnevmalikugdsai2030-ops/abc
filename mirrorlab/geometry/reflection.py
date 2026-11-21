"""
Single Finite Flat Mirror Specular Reflection Engine for MIRRORLAB.

This module implements the exact analytic Virtual-Target Reflection Method.
Given a camera position, target center, and finite mirror segment (endpoints A and B)
with a reflective-face normal, it determines:
1. Virtual target position T'
2. Line C -> T'
3. Intersection point R with the infinite mirror line
4. Finite segment clipping (t in [0, 1])
5. Reflective front-face validation
6. Law of reflection validation (theta_i == theta_r)
7. Full physical optical path C -> R -> T
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from mirrorlab.geometry.vector import (
    EPSILON,
    DegenerateLineError,
    _to_vec2,
    dot_product,
    line_tangent_and_normal,
    vector_magnitude,
    vector_normalize,
)

ArrayLike = Sequence[float] | np.ndarray


class ReflectionFailureReason(str, Enum):
    """Enumeration of physical and geometric reasons why specular reflection fails."""

    DEGENERATE_MIRROR = "DEGENERATE_MIRROR"
    DEGENERATE_TARGET = "DEGENERATE_TARGET"
    WRONG_SIDE_CAMERA = "WRONG_SIDE_CAMERA"
    WRONG_SIDE_TARGET = "WRONG_SIDE_TARGET"
    WRONG_SIDE_BOTH = "WRONG_SIDE_BOTH"
    PARALLEL_TO_MIRROR = "PARALLEL_TO_MIRROR"
    BEHIND_CAMERA = "BEHIND_CAMERA"
    MISSED_MIRROR_PAST_A = "MISSED_MIRROR_PAST_A"
    MISSED_MIRROR_PAST_B = "MISSED_MIRROR_PAST_B"
    LAW_OF_REFLECTION_VIOLATION = "LAW_OF_REFLECTION_VIOLATION"


@dataclass(frozen=True)
class MirrorReflectionResult:
    """
    Structured outcome of the specular reflection calculation containing
    all intermediate geometric variables and diagnostic flags.
    """

    is_valid: bool
    failure_reason: ReflectionFailureReason | None = None

    # Mirror properties
    mirror_length: float = 0.0
    mirror_unit_tangent: np.ndarray = field(default_factory=lambda: np.zeros(2))
    mirror_unit_normal: np.ndarray = field(default_factory=lambda: np.zeros(2))

    # Signed distances to infinite mirror line (positive = reflective front side)
    camera_signed_dist: float = 0.0
    target_signed_dist: float = 0.0
    is_camera_on_reflective_side: bool = False
    is_target_on_reflective_side: bool = False

    # Virtual target position (T')
    virtual_target: np.ndarray | None = None

    # Intersection point R on infinite mirror line
    intersection_point: np.ndarray | None = None
    ray_alpha: float | None = None  # Parameter on line C -> T': R = C + alpha*(T' - C)
    is_in_front_of_camera: bool = False

    # Finite mirror segment parameter: R = A + t*(B - A)
    mirror_t: float | None = None
    is_on_finite_segment: bool = False

    # Physical optical path metrics
    incident_ray_vector: np.ndarray | None = None  # R - C
    reflected_ray_vector: np.ndarray | None = None  # T - R
    incident_angle_rad: float | None = None  # Angle with normal (theta_i)
    reflected_angle_rad: float | None = None  # Angle with normal (theta_r)
    angle_difference_rad: float | None = None  # |theta_i - theta_r|
    total_path_length: float | None = None  # ||R - C|| + ||T - R||
    path_points: list[np.ndarray] = field(default_factory=list)  # [C, R, T]


def calculate_mirror_reflection(
    camera_pos: ArrayLike,
    target_center: ArrayLike,
    mirror_a: ArrayLike,
    mirror_b: ArrayLike,
    reflective_normal: ArrayLike | None = None,
    eps: float = EPSILON,
) -> MirrorReflectionResult:
    """
    Compute specular reflection from a camera to a target via a finite flat mirror
    using the exact analytic Virtual-Target Method.

    Steps:
    1. Validate mirror endpoints A and B. Compute tangent and reflective normal.
    2. Check that both Camera and Target lie on the reflective side (d_signed > eps).
    3. Calculate the virtual target position T' behind the mirror line:
       T' = T - 2 * d_T * n_hat
    4. Cast virtual sightline from Camera C to Virtual Target T'.
       Find intersection R with the infinite mirror line:
       alpha = d_C / (d_C + d_T),  R = C + alpha * (T' - C)
    5. Test if R is in front of camera (alpha > eps).
    6. Test if R lies strictly within the finite mirror segment [A, B] (0.0 <= t <= 1.0).
    7. Form optical path C -> R -> T and independently verify the Law of Reflection
       (theta_i == theta_r).

    Args:
        camera_pos: 2D coordinates of camera [x, y].
        target_center: 2D coordinates of target scratch center [x, y].
        mirror_a: First endpoint of mirror segment [x, y].
        mirror_b: Second endpoint of mirror segment [x, y].
        reflective_normal: Optional user-specified normal vector indicating which
            side is reflective. If provided, the computed normal is oriented to have
            positive dot product with this vector.
        eps: Numerical tolerance threshold.

    Returns:
        MirrorReflectionResult: Comprehensive structured evaluation.
    """
    c = _to_vec2(camera_pos, "camera_pos")
    target = _to_vec2(target_center, "target_center")
    a = _to_vec2(mirror_a, "mirror_a")
    b = _to_vec2(mirror_b, "mirror_b")

    # Step 1: Compute mirror tangent, normal, and length
    try:
        t_hat, n_hat, mirror_len = line_tangent_and_normal(a, b, eps=eps)
    except DegenerateLineError:
        return MirrorReflectionResult(
            is_valid=False,
            failure_reason=ReflectionFailureReason.DEGENERATE_MIRROR,
        )

    # If user provided a normal orientation hint, ensure n_hat aligns with it
    if reflective_normal is not None:
        user_norm = _to_vec2(reflective_normal, "reflective_normal")
        mag_u = vector_magnitude(user_norm)
        if mag_u > eps and dot_product(n_hat, user_norm) < 0.0:
            n_hat = -n_hat

    # Step 2: Signed distances to infinite mirror line
    # d = (P - A) . n_hat
    d_c = dot_product(c - a, n_hat)
    d_t = dot_product(target - a, n_hat)

    cam_front = d_c > eps
    tgt_front = d_t > eps

    # Check reflective side validity
    if not cam_front and not tgt_front:
        return MirrorReflectionResult(
            is_valid=False,
            failure_reason=ReflectionFailureReason.WRONG_SIDE_BOTH,
            mirror_length=mirror_len,
            mirror_unit_tangent=t_hat,
            mirror_unit_normal=n_hat,
            camera_signed_dist=d_c,
            target_signed_dist=d_t,
            is_camera_on_reflective_side=False,
            is_target_on_reflective_side=False,
        )
    if not cam_front:
        return MirrorReflectionResult(
            is_valid=False,
            failure_reason=ReflectionFailureReason.WRONG_SIDE_CAMERA,
            mirror_length=mirror_len,
            mirror_unit_tangent=t_hat,
            mirror_unit_normal=n_hat,
            camera_signed_dist=d_c,
            target_signed_dist=d_t,
            is_camera_on_reflective_side=False,
            is_target_on_reflective_side=tgt_front,
        )
    if not tgt_front:
        return MirrorReflectionResult(
            is_valid=False,
            failure_reason=ReflectionFailureReason.WRONG_SIDE_TARGET,
            mirror_length=mirror_len,
            mirror_unit_tangent=t_hat,
            mirror_unit_normal=n_hat,
            camera_signed_dist=d_c,
            target_signed_dist=d_t,
            is_camera_on_reflective_side=cam_front,
            is_target_on_reflective_side=False,
        )

    # Step 3: Calculate Virtual Target T' behind mirror line
    # T' = T - 2 * d_t * n_hat
    virtual_t = target - 2.0 * d_t * n_hat

    # Step 4: Line C -> T' and intersection R with mirror line
    # Denominator for intersection: d_C + d_T
    denom = d_c + d_t
    if abs(denom) < eps:
        # Line from C to T' is parallel to the mirror line
        return MirrorReflectionResult(
            is_valid=False,
            failure_reason=ReflectionFailureReason.PARALLEL_TO_MIRROR,
            mirror_length=mirror_len,
            mirror_unit_tangent=t_hat,
            mirror_unit_normal=n_hat,
            camera_signed_dist=d_c,
            target_signed_dist=d_t,
            is_camera_on_reflective_side=True,
            is_target_on_reflective_side=True,
            virtual_target=virtual_t,
        )

    alpha = d_c / denom
    r = c + alpha * (virtual_t - c)

    # Step 5: Check if intersection is in front of the camera along the ray
    is_in_front = alpha > eps
    if not is_in_front:
        return MirrorReflectionResult(
            is_valid=False,
            failure_reason=ReflectionFailureReason.BEHIND_CAMERA,
            mirror_length=mirror_len,
            mirror_unit_tangent=t_hat,
            mirror_unit_normal=n_hat,
            camera_signed_dist=d_c,
            target_signed_dist=d_t,
            is_camera_on_reflective_side=True,
            is_target_on_reflective_side=True,
            virtual_target=virtual_t,
            intersection_point=r,
            ray_alpha=alpha,
            is_in_front_of_camera=False,
        )

    # Step 6: Test finite mirror segment bounds
    # R = A + t_mirror * (B - A)
    # t_mirror = ((R - A) . (B - A)) / ||B - A||^2
    t_mirror = dot_product(r - a, t_hat) / mirror_len

    # Clamp tiny epsilon overshoots onto [0.0, 1.0]
    if abs(t_mirror) < eps:
        t_mirror = 0.0
    elif abs(t_mirror - 1.0) < eps:
        t_mirror = 1.0

    if t_mirror < 0.0:
        return MirrorReflectionResult(
            is_valid=False,
            failure_reason=ReflectionFailureReason.MISSED_MIRROR_PAST_A,
            mirror_length=mirror_len,
            mirror_unit_tangent=t_hat,
            mirror_unit_normal=n_hat,
            camera_signed_dist=d_c,
            target_signed_dist=d_t,
            is_camera_on_reflective_side=True,
            is_target_on_reflective_side=True,
            virtual_target=virtual_t,
            intersection_point=r,
            ray_alpha=alpha,
            is_in_front_of_camera=True,
            mirror_t=t_mirror,
            is_on_finite_segment=False,
        )

    if t_mirror > 1.0:
        return MirrorReflectionResult(
            is_valid=False,
            failure_reason=ReflectionFailureReason.MISSED_MIRROR_PAST_B,
            mirror_length=mirror_len,
            mirror_unit_tangent=t_hat,
            mirror_unit_normal=n_hat,
            camera_signed_dist=d_c,
            target_signed_dist=d_t,
            is_camera_on_reflective_side=True,
            is_target_on_reflective_side=True,
            virtual_target=virtual_t,
            intersection_point=r,
            ray_alpha=alpha,
            is_in_front_of_camera=True,
            mirror_t=t_mirror,
            is_on_finite_segment=False,
        )

    # Step 7: Form physical ray vectors and test Law of Reflection
    inc_vec = r - c
    refl_vec = target - r
    d_inc = vector_magnitude(inc_vec)
    d_refl = vector_magnitude(refl_vec)

    # If Camera or Target coincides with reflection point
    if d_inc < eps or d_refl < eps:
        return MirrorReflectionResult(
            is_valid=False,
            failure_reason=ReflectionFailureReason.DEGENERATE_TARGET,
            mirror_length=mirror_len,
            mirror_unit_tangent=t_hat,
            mirror_unit_normal=n_hat,
            camera_signed_dist=d_c,
            target_signed_dist=d_t,
            is_camera_on_reflective_side=True,
            is_target_on_reflective_side=True,
            virtual_target=virtual_t,
            intersection_point=r,
            ray_alpha=alpha,
            is_in_front_of_camera=True,
            mirror_t=t_mirror,
            is_on_finite_segment=True,
        )

    u_inc = vector_normalize(inc_vec, eps=eps)
    u_refl = vector_normalize(refl_vec, eps=eps)

    # Angle of incidence: angle of -u_inc with normal n_hat
    # cos(theta_i) = (-u_inc) . n_hat
    cos_theta_i = float(np.clip(dot_product(-u_inc, n_hat), -1.0, 1.0))
    theta_i = math.acos(cos_theta_i)

    # Angle of reflection: angle of u_refl with normal n_hat
    # cos(theta_r) = u_refl . n_hat
    cos_theta_r = float(np.clip(dot_product(u_refl, n_hat), -1.0, 1.0))
    theta_r = math.acos(cos_theta_r)

    angle_diff = abs(theta_i - theta_r)

    # Validate the Law of Reflection within numerical tolerance
    if angle_diff > 1e-5:
        return MirrorReflectionResult(
            is_valid=False,
            failure_reason=ReflectionFailureReason.LAW_OF_REFLECTION_VIOLATION,
            mirror_length=mirror_len,
            mirror_unit_tangent=t_hat,
            mirror_unit_normal=n_hat,
            camera_signed_dist=d_c,
            target_signed_dist=d_t,
            is_camera_on_reflective_side=True,
            is_target_on_reflective_side=True,
            virtual_target=virtual_t,
            intersection_point=r,
            ray_alpha=alpha,
            is_in_front_of_camera=True,
            mirror_t=t_mirror,
            is_on_finite_segment=True,
            incident_ray_vector=inc_vec,
            reflected_ray_vector=refl_vec,
            incident_angle_rad=theta_i,
            reflected_angle_rad=theta_r,
            angle_difference_rad=angle_diff,
        )

    total_len = d_inc + d_refl

    return MirrorReflectionResult(
        is_valid=True,
        failure_reason=None,
        mirror_length=mirror_len,
        mirror_unit_tangent=t_hat,
        mirror_unit_normal=n_hat,
        camera_signed_dist=d_c,
        target_signed_dist=d_t,
        is_camera_on_reflective_side=True,
        is_target_on_reflective_side=True,
        virtual_target=virtual_t,
        intersection_point=r,
        ray_alpha=alpha,
        is_in_front_of_camera=True,
        mirror_t=t_mirror,
        is_on_finite_segment=True,
        incident_ray_vector=inc_vec,
        reflected_ray_vector=refl_vec,
        incident_angle_rad=theta_i,
        reflected_angle_rad=theta_r,
        angle_difference_rad=angle_diff,
        total_path_length=total_len,
        path_points=[c, r, target],
    )
