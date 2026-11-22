"""
Independent Ground-Truth Geometry Verification Engine for MIRRORLAB.

Provides a completely independent mathematical and computational implementation
of specular reflection and obstacle occlusion to audit and cross-verify the primary
VisibilityOracle:

1. Method Disparity:
   - Primary Oracle: Virtual-Target reflection method (reflects Target T -> T', casts C -> T').
   - Independent Verifier: Virtual-Camera reflection method (reflects Camera C -> C', casts C' -> T').
2. Physical Principle Invariant:
   - Evaluates Fermat's Principle of Stationary Optical Path Length:
     L(t) = ||R(t) - C|| + ||T - R(t)||.
     Verifies that dL/dt = 0 at the reflection point R with residual < 1e-5.
3. Independent Collision Primitives:
   - Independent quadratic ray-circle intersection.
   - Independent Cyrus-Beck slab clipping for oriented rectangular obstacles.
4. Dual Verification:
   - Recomputes ground-truth verdict without reusing any primary oracle data.
   - Asserts delta_R < 1e-4 mm and exact state agreement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from mirrorlab.geometry.oracle import VisibilityOracle, VisibilityOracleResult
from mirrorlab.schema.common import ObstructionReason, VisibilityState
from mirrorlab.schema.scene import (
    InspectionScene,
)


@dataclass(frozen=True)
class IndependentReflectionResult:
    """Outcome of independent virtual-camera specular reflection."""

    is_valid: bool
    failure_reason: str | None = None
    virtual_camera: np.ndarray | None = None
    reflection_point: np.ndarray | None = None
    mirror_t: float | None = None
    fermat_residual: float = 0.0
    incident_angle_deg: float | None = None
    reflected_angle_deg: float | None = None
    total_path_distance: float | None = None


@dataclass(frozen=True)
class IndependentObstacleHit:
    """Record of independent obstacle intersection."""

    obstacle_id: str
    leg: str  # "LEG_1_CAMERA_TO_MIRROR" or "LEG_2_MIRROR_TO_TARGET"
    hit_point: np.ndarray
    hit_distance_along_leg: float


@dataclass(frozen=True)
class IndependentVisibilityVerdict:
    """Complete ground-truth verdict computed by the independent verifier."""

    is_visible: bool
    state: VisibilityState
    reason: ObstructionReason
    reflection_point: np.ndarray | None
    total_optical_distance: float | None
    fermat_residual: float
    blocking_hit: IndependentObstacleHit | None
    diagnostic_proof: str


@dataclass(frozen=True)
class DualVerificationResult:
    """Cross-validation report between primary VisibilityOracle and IndependentVerifier."""

    agreed: bool
    oracle_visible: bool
    independent_visible: bool
    oracle_reason: str
    independent_reason: str
    reflection_point_delta_mm: float
    fermat_residual: float
    is_mathematically_certified: bool
    proof_summary: str


class IndependentGeometryVerifier:
    """
    Independent computational geometry verifier using Virtual-Camera reflection,
    Fermat stationary path validation, and independent obstacle intersection.
    """

    EPSILON = 1e-7

    @classmethod
    def calculate_virtual_camera_reflection(
        cls,
        cam_pos: np.ndarray,
        target_pos: np.ndarray,
        mirror_a: np.ndarray,
        mirror_b: np.ndarray,
        reflective_normal: np.ndarray,
    ) -> IndependentReflectionResult:
        """
        Solves specular reflection by reflecting the camera across the infinite mirror line.
        """
        W = mirror_b - mirror_a
        mirror_len = float(np.linalg.norm(W))
        if mirror_len < cls.EPSILON:
            return IndependentReflectionResult(is_valid=False, failure_reason="DEGENERATE_MIRROR")

        u = W / mirror_len  # Unit tangent along mirror
        # Normal pointing into cavity
        n = np.array([-u[1], u[0]], dtype=np.float64)
        if np.dot(n, reflective_normal) < 0:
            n = -n

        # Check reflective side: signed distances to infinite mirror line
        d_C = float(np.dot(cam_pos - mirror_a, n))
        d_T = float(np.dot(target_pos - mirror_a, n))

        if d_C <= cls.EPSILON and d_T <= cls.EPSILON:
            return IndependentReflectionResult(is_valid=False, failure_reason="WRONG_SIDE_BOTH")
        if d_C <= cls.EPSILON:
            return IndependentReflectionResult(is_valid=False, failure_reason="WRONG_SIDE_CAMERA")
        if d_T <= cls.EPSILON:
            return IndependentReflectionResult(is_valid=False, failure_reason="WRONG_SIDE_TARGET")

        # Virtual camera position: C' = C - 2 * d_C * n
        C_prime = cam_pos - 2.0 * d_C * n

        # Ray from Virtual Camera C' to Target T: P(s) = C' + s * (T - C')
        V = target_pos - C_prime
        det = W[0] * V[1] - W[1] * V[0]

        if abs(det) < 1e-12:
            return IndependentReflectionResult(is_valid=False, failure_reason="PARALLEL_TO_MIRROR")

        # Intersection parameter t on mirror line: R = A + t * W
        diff = mirror_a - C_prime
        s = (diff[1] * W[0] - diff[0] * W[1]) / det
        t = (V[0] * diff[1] - V[1] * diff[0]) / det

        # Must be in front of virtual camera towards target: s in (0, 1)
        if s <= cls.EPSILON or s >= 1.0 - cls.EPSILON:
            return IndependentReflectionResult(is_valid=False, failure_reason="BEHIND_CAMERA")

        # Finite mirror segment bounds: t in [0, 1]
        is_on_segment = (0.0 - cls.EPSILON) <= t <= (1.0 + cls.EPSILON)
        if not is_on_segment:
            reason = "MISSED_MIRROR_PAST_A" if t < 0.0 else "MISSED_MIRROR_PAST_B"
            return IndependentReflectionResult(is_valid=False, failure_reason=reason, mirror_t=t)

        # Exact reflection point
        t_clamped = max(0.0, min(1.0, float(t)))
        R = mirror_a + t_clamped * W

        # Incident and Reflected Vectors
        v_inc = R - cam_pos
        v_refl = target_pos - R
        len_inc = float(np.linalg.norm(v_inc))
        len_refl = float(np.linalg.norm(v_refl))

        if len_inc < cls.EPSILON or len_refl < cls.EPSILON:
            return IndependentReflectionResult(is_valid=False, failure_reason="DEGENERATE_PATH")

        u_inc = v_inc / len_inc
        u_refl = v_refl / len_refl

        # Fermat Principle Verification:
        # Stationary path requires projection of incident ray onto mirror tangent
        # to equal projection of reflected ray onto mirror tangent.
        proj_inc = float(np.dot(u_inc, u))
        proj_refl = float(np.dot(u_refl, u))
        fermat_res = abs(proj_inc - proj_refl)

        # Law of Reflection angles with normal
        cos_i = max(-1.0, min(1.0, float(np.dot(-u_inc, n))))
        cos_r = max(-1.0, min(1.0, float(np.dot(u_refl, n))))
        ang_i_deg = math.degrees(math.acos(cos_i))
        ang_r_deg = math.degrees(math.acos(cos_r))

        return IndependentReflectionResult(
            is_valid=True,
            virtual_camera=C_prime,
            reflection_point=R,
            mirror_t=t_clamped,
            fermat_residual=fermat_res,
            incident_angle_deg=ang_i_deg,
            reflected_angle_deg=ang_r_deg,
            total_path_distance=len_inc + len_refl,
        )

    @classmethod
    def check_segment_circle_occlusion(
        cls,
        p1: np.ndarray,
        p2: np.ndarray,
        center: np.ndarray,
        radius: float,
        trim_eps: float = 1e-4,
    ) -> tuple[bool, float, np.ndarray | None]:
        """Independent quadratic line-segment vs circle intersection."""
        D = p2 - p1
        leg_len = float(np.linalg.norm(D))
        if leg_len < cls.EPSILON:
            return False, 0.0, None

        d_unit = D / leg_len
        f = p1 - center

        a = float(np.dot(d_unit, d_unit))
        b = 2.0 * float(np.dot(f, d_unit))
        c = float(np.dot(f, f)) - radius * radius

        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:
            return False, 0.0, None

        sqrt_d = math.sqrt(discriminant)
        t1 = (-b - sqrt_d) / (2.0 * a)
        t2 = (-b + sqrt_d) / (2.0 * a)

        # Check intersection with trimmed segment [trim_eps, leg_len - trim_eps]
        min_dist = trim_eps
        max_dist = leg_len - trim_eps

        if t2 < min_dist or t1 > max_dist:
            return False, 0.0, None

        hit_t = max(t1, min_dist)
        hit_pt = p1 + hit_t * d_unit
        return True, hit_t, hit_pt

    @classmethod
    def check_segment_obb_occlusion(
        cls,
        p1: np.ndarray,
        p2: np.ndarray,
        center: np.ndarray,
        width: float,
        height: float,
        rotation_deg: float,
        trim_eps: float = 1e-4,
    ) -> tuple[bool, float, np.ndarray | None]:
        """Independent Cyrus-Beck clipping for segment vs rotated rectangle."""
        D = p2 - p1
        leg_len = float(np.linalg.norm(D))
        if leg_len < cls.EPSILON:
            return False, 0.0, None

        # Transform segment endpoints into local box coordinate frame
        rad = math.radians(-rotation_deg)
        cos_r = math.cos(rad)
        sin_r = math.sin(rad)
        rot_mat = np.array([[cos_r, -sin_r], [sin_r, cos_r]], dtype=np.float64)

        p1_local = rot_mat @ (p1 - center)
        p2_local = rot_mat @ (p2 - center)
        D_local = p2_local - p1_local

        hw = width / 2.0
        hh = height / 2.0

        t_min = trim_eps / leg_len
        t_max = 1.0 - (trim_eps / leg_len)

        for i, bound in [(0, hw), (1, hh)]:
            if abs(D_local[i]) < 1e-9:
                if abs(p1_local[i]) > bound:
                    return False, 0.0, None
            else:
                t1 = (-bound - p1_local[i]) / D_local[i]
                t2 = (bound - p1_local[i]) / D_local[i]
                if t1 > t2:
                    t1, t2 = t2, t1
                t_min = max(t_min, t1)
                t_max = min(t_max, t2)
                if t_min > t_max:
                    return False, 0.0, None

        if t_min <= t_max and t_max >= (trim_eps / leg_len) and t_min <= (1.0 - trim_eps / leg_len):
            hit_dist = t_min * leg_len
            hit_pt = p1 + (t_min * D)
            return True, hit_dist, hit_pt

        return False, 0.0, None

    @classmethod
    def evaluate(cls, scene: InspectionScene) -> IndependentVisibilityVerdict:
        """
        Calculates ground-truth visibility using the independent virtual-camera method.
        """
        cam = np.array([scene.camera.position.x, scene.camera.position.y], dtype=np.float64)
        target = np.array([scene.target.center.x, scene.target.center.y], dtype=np.float64)
        m_a = np.array([scene.mirror.endpoint_a.x, scene.mirror.endpoint_a.y], dtype=np.float64)
        m_b = np.array([scene.mirror.endpoint_b.x, scene.mirror.endpoint_b.y], dtype=np.float64)
        norm = np.array([scene.mirror.reflective_normal.dx, scene.mirror.reflective_normal.dy], dtype=np.float64)

        # 1. Specular Reflection Check via Virtual Camera
        refl = cls.calculate_virtual_camera_reflection(cam, target, m_a, m_b, norm)

        if not refl.is_valid:
            reason_map = {
                "WRONG_SIDE_BOTH": ObstructionReason.BACK_OF_MIRROR,
                "WRONG_SIDE_CAMERA": ObstructionReason.BACK_OF_MIRROR,
                "WRONG_SIDE_TARGET": ObstructionReason.BACK_OF_MIRROR,
                "MISSED_MIRROR_PAST_A": ObstructionReason.MISSED_MIRROR,
                "MISSED_MIRROR_PAST_B": ObstructionReason.MISSED_MIRROR,
                "PARALLEL_TO_MIRROR": ObstructionReason.MISSED_MIRROR,
                "BEHIND_CAMERA": ObstructionReason.MISSED_MIRROR,
            }
            obs_reason = reason_map.get(refl.failure_reason or "", ObstructionReason.MISSED_MIRROR)
            return IndependentVisibilityVerdict(
                is_visible=False,
                state=VisibilityState.NOT_VISIBLE,
                reason=obs_reason,
                reflection_point=None,
                total_optical_distance=None,
                fermat_residual=refl.fermat_residual,
                blocking_hit=None,
                diagnostic_proof=f"Independent reflection failed: {refl.failure_reason}",
            )

        R = refl.reflection_point
        assert R is not None

        # 2. Camera Angular FOV Cone Check
        forward_rad = math.radians(scene.camera.heading_deg)
        fwd_vec = np.array([math.cos(forward_rad), math.sin(forward_rad)], dtype=np.float64)
        v_ray = R - cam
        d_ray = float(np.linalg.norm(v_ray))
        if d_ray < cls.EPSILON:
            return IndependentVisibilityVerdict(
                is_visible=False,
                state=VisibilityState.NOT_VISIBLE,
                reason=ObstructionReason.OUT_OF_FOV,
                reflection_point=R,
                total_optical_distance=None,
                fermat_residual=refl.fermat_residual,
                blocking_hit=None,
                diagnostic_proof="Camera at reflection point.",
            )

        u_ray = v_ray / d_ray
        cos_fov = float(np.dot(fwd_vec, u_ray))
        half_fov_cos = math.cos(math.radians(scene.camera.fov_degrees / 2.0))

        if cos_fov < half_fov_cos - cls.EPSILON:
            return IndependentVisibilityVerdict(
                is_visible=False,
                state=VisibilityState.NOT_VISIBLE,
                reason=ObstructionReason.OUT_OF_FOV,
                reflection_point=R,
                total_optical_distance=refl.total_path_distance,
                fermat_residual=refl.fermat_residual,
                blocking_hit=None,
                diagnostic_proof=f"Incident ray angle ({math.degrees(math.acos(max(-1.0, min(1.0, cos_fov)))):.1f} deg) outside FOV ({scene.camera.fov_degrees/2.0:.1f} deg).",
            )

        # 3. Two-Leg Obstacle Occlusion Check
        closest_hit: IndependentObstacleHit | None = None
        closest_leg: str | None = None

        legs = [("LEG_1_CAMERA_TO_MIRROR", cam, R), ("LEG_2_MIRROR_TO_TARGET", R, target)]

        for leg_name, p_start, p_end in legs:
            for obs in scene.obstacles:
                c_obs = np.array([obs.center.x, obs.center.y], dtype=np.float64)
                hit = False
                dist = 0.0
                hit_pt = None

                if obs.obstacle_type == "circle":
                    radius = obs.radius if obs.radius is not None else 0.0
                    hit, dist, hit_pt = cls.check_segment_circle_occlusion(p_start, p_end, c_obs, radius)
                elif obs.obstacle_type == "rectangle":
                    width = obs.width if obs.width is not None else 0.0
                    height = obs.height if obs.height is not None else 0.0
                    hit, dist, hit_pt = cls.check_segment_obb_occlusion(
                        p_start, p_end, c_obs, width, height, obs.angle_degrees
                    )

                if hit and hit_pt is not None:
                    if closest_hit is None or (leg_name == "LEG_1_CAMERA_TO_MIRROR" and closest_leg == "LEG_2_MIRROR_TO_TARGET"):
                        closest_hit = IndependentObstacleHit(obs.obstacle_id, leg_name, hit_pt, dist)
                        closest_leg = leg_name
                    elif leg_name == closest_leg and dist < closest_hit.hit_distance_along_leg:
                        closest_hit = IndependentObstacleHit(obs.obstacle_id, leg_name, hit_pt, dist)

            # If leg 1 is blocked, no need to check leg 2 as leg 1 blocks sight first
            if closest_hit and closest_leg == "LEG_1_CAMERA_TO_MIRROR":
                break

        if closest_hit is not None:
            reason = (
                ObstructionReason.BLOCKED_INCIDENT
                if closest_leg == "LEG_1_CAMERA_TO_MIRROR"
                else ObstructionReason.BLOCKED_REFLECTED
            )
            return IndependentVisibilityVerdict(
                is_visible=False,
                state=VisibilityState.NOT_VISIBLE,
                reason=reason,
                reflection_point=R,
                total_optical_distance=refl.total_path_distance,
                fermat_residual=refl.fermat_residual,
                blocking_hit=closest_hit,
                diagnostic_proof=f"Ray blocked by {closest_hit.obstacle_id} on {closest_leg}.",
            )

        # Clear, Unobstructed Specular Sightline
        proof = (
            f"Verified visible via virtual camera C'=[{refl.virtual_camera[0]:.2f}, {refl.virtual_camera[1]:.2f}], "
            f"reflection point R=[{R[0]:.2f}, {R[1]:.2f}] (t={refl.mirror_t:.3f}), "
            f"theta_i={refl.incident_angle_deg:.2f} deg, theta_r={refl.reflected_angle_deg:.2f} deg, "
            f"Fermat residual={refl.fermat_residual:.2e}."
        )
        return IndependentVisibilityVerdict(
            is_visible=True,
            state=VisibilityState.VISIBLE,
            reason=ObstructionReason.NONE,
            reflection_point=R,
            total_optical_distance=refl.total_path_distance,
            fermat_residual=refl.fermat_residual,
            blocking_hit=None,
            diagnostic_proof=proof,
        )

    @classmethod
    def verify_dual(
        cls,
        scene: InspectionScene,
        oracle_result: VisibilityOracleResult | None = None,
    ) -> DualVerificationResult:
        """
        Directly compares VisibilityOracle output with IndependentGeometryVerifier.
        """
        if oracle_result is None:
            oracle_result = VisibilityOracle.evaluate(scene)

        indep = cls.evaluate(scene)

        # Check visibility agreement
        vis_match = oracle_result.is_visible == indep.is_visible
        reason_match = oracle_result.reason == indep.reason

        # Reflection point delta
        delta_r = 0.0
        if oracle_result.reflection_point is not None and indep.reflection_point is not None:
            delta_r = float(np.linalg.norm(oracle_result.reflection_point - indep.reflection_point))

        # Mathematical certification requires:
        # 1. State agreement
        # 2. Reason code agreement
        # 3. Position error < 1e-4 mm
        # 4. Fermat residual < 1e-5
        certified = (
            vis_match
            and reason_match
            and (delta_r < 1e-4)
            and (indep.fermat_residual < 1e-5)
        )

        proof = (
            f"Dual Verification Agreement: Oracle={oracle_result.is_visible} ({oracle_result.reason}) "
            f"vs Independent={indep.is_visible} ({indep.reason}). "
            f"Delta_R={delta_r:.4e} mm | Fermat_Residual={indep.fermat_residual:.4e}."
        )

        return DualVerificationResult(
            agreed=vis_match and reason_match,
            oracle_visible=oracle_result.is_visible,
            independent_visible=indep.is_visible,
            oracle_reason=str(oracle_result.reason.value if hasattr(oracle_result.reason, "value") else oracle_result.reason),
            independent_reason=str(indep.reason.value if hasattr(indep.reason, "value") else indep.reason),
            reflection_point_delta_mm=delta_r,
            fermat_residual=indep.fermat_residual,
            is_mathematically_certified=certified,
            proof_summary=proof,
        )
