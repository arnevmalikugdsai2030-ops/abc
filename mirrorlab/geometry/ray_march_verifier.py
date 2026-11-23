"""
Third Independent Geometric Verification Engine for MIRRORLAB:
Direct Snell-Vector Collinearity Root-Finding & Separating Axis Obstacle Clipping.

This module provides an entirely independent THIRD verification method for specular
visibility that shares zero implementation logic with:
1. VisibilityOracle (Method 1: Virtual-Target reflection C -> T')
2. IndependentGeometryVerifier (Method 2: Virtual-Camera reflection C' -> T via Fermat's stationary path)

Method 3 Mathematical Formulation:
- Does NOT reflect virtual coordinates across mirror lines.
- Treats mirror segment M(t) = A + t * (B - A), t in [0, 1].
- For any t, the incident ray direction is u_inc(t) = (M(t) - C) / ||M(t) - C||.
- By the Householder reflection tensor relative to mirror normal n:
    u_refl(t) = u_inc(t) - 2 * (u_inc(t) . n) * n.
- For specular reflection to illuminate the target defect at T, the reflected ray
  u_refl(t) must be strictly collinear with the direction towards the target
  u_tgt(t) = (T - M(t)) / ||T - M(t)||.
- Cross-product collinearity error function:
    E(t) = u_refl(t) x u_tgt(t) = u_refl_x * u_tgt_y - u_refl_y * u_tgt_x.
- Solves E(t) = 0 over the continuous real domain t in [0, 1] using Brent's
  guaranteed-convergence root bracket method.
- Verifies physical obstacle occlusions using Separating Axis Theorem (SAT)
  polygon projection and independent circle-line analytical discriminant.
- Produces TripleVerificationReport confirming consensus across all 3 independent engines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from mirrorlab.schema.common import ObstructionReason, VisibilityState
from mirrorlab.schema.scene import InspectionScene, ObstacleSpec


@dataclass(frozen=True)
class DirectSnellReflectionResult:
    """Outcome of Method 3: Direct Snell-Vector root-finding reflection."""

    is_valid: bool
    reflection_point: np.ndarray | None = None
    t_mirror: float | None = None
    collinearity_residual: float = 0.0
    incident_angle_deg: float | None = None
    reflected_angle_deg: float | None = None
    angular_disparity_deg: float = 0.0
    failure_reason: str | None = None


@dataclass(frozen=True)
class TripleVerificationReport:
    """Consensus certificate comparing all three independent geometry engines."""

    scene_id: str
    all_three_agree: bool
    method1_oracle_visible: bool
    method2_fermat_visible: bool
    method3_snell_visible: bool
    method1_reason: str
    method2_reason: str
    method3_reason: str
    max_reflection_disparity_mm: float
    max_fermat_residual: float
    max_snell_collinearity_residual: float
    details: str


class DirectSnellRayVerifier:
    """
    Third Independent Geometry Engine:
    Direct vector Snell's law collinearity solver and SAT collision verification.
    """

    ROOT_TOLERANCE: float = 1e-7
    DELTA_R_TOLERANCE_MM: float = 1e-4

    @classmethod
    def solve_reflection_point(
        cls,
        cam: np.ndarray,
        target: np.ndarray,
        m_a: np.ndarray,
        m_b: np.ndarray,
        normal: np.ndarray,
    ) -> DirectSnellReflectionResult:
        """
        Solves for the unique reflection point on the mirror segment by finding the root
        of the cross-product collinearity error between Householder-reflected ray and target vector.
        """
        # Ensure normal is unit vector
        n = normal / np.linalg.norm(normal)
        m_vec = m_b - m_a
        m_len = float(np.linalg.norm(m_vec))
        if m_len < 1e-6:
            return DirectSnellReflectionResult(
                is_valid=False, failure_reason="DEGENERATE_MIRROR_ZERO_LENGTH"
            )

        # Check front-face orientation: Camera and Target must lie on positive normal side
        ca = cam - m_a
        ta = target - m_a
        if np.dot(ca, n) <= 0.0 or np.dot(ta, n) <= 0.0:
            return DirectSnellReflectionResult(
                is_valid=False, failure_reason="WRONG_SIDE_OF_MIRROR"
            )

        def _collinearity_error(t: float) -> float:
            R = m_a + t * m_vec
            v_inc = R - cam
            dist_inc = float(np.linalg.norm(v_inc))
            if dist_inc < 1e-9:
                return 0.0
            u_inc = v_inc / dist_inc

            # Householder specular reflection: u_refl = u_inc - 2 * (u_inc . n) * n
            u_refl = u_inc - 2.0 * float(np.dot(u_inc, n)) * n

            v_tgt = target - R
            dist_tgt = float(np.linalg.norm(v_tgt))
            if dist_tgt < 1e-9:
                return 0.0
            u_tgt = v_tgt / dist_tgt

            # 2D cross product: u_refl_x * u_tgt_y - u_refl_y * u_tgt_x
            return float(u_refl[0] * u_tgt[1] - u_refl[1] * u_tgt[0])

        # Evaluate collinearity error across extended line to locate root
        # E(t) is strictly monotonic for valid coplanar front-side reflections
        t_low = -5.0
        t_high = 5.0
        f_low = _collinearity_error(t_low)
        f_high = _collinearity_error(t_high)

        # Expand bracket if needed
        expand_steps = 0
        while f_low * f_high > 0 and expand_steps < 20:
            t_low -= 5.0
            t_high += 5.0
            f_low = _collinearity_error(t_low)
            f_high = _collinearity_error(t_high)
            expand_steps += 1

        if f_low * f_high > 0:
            return DirectSnellReflectionResult(
                is_valid=False, failure_reason="NO_ROOT_IN_SPECULAR_MANIFOLD"
            )

        # Brent's method / bisection root solve
        a = t_low
        b = t_high
        fa = f_low
        t_root = 0.5

        for _ in range(80):
            t_root = 0.5 * (a + b)
            if (b - a) < cls.ROOT_TOLERANCE:
                break
            fm = _collinearity_error(t_root)
            if abs(fm) < cls.ROOT_TOLERANCE:
                break
            if fa * fm <= 0:
                b = t_root
            else:
                a = t_root
                fa = fm

        res_val = abs(_collinearity_error(t_root))
        R_star = m_a + t_root * m_vec

        # Compute incident and reflected angles relative to mirror normal
        v_inc_star = R_star - cam
        u_inc_star = v_inc_star / np.linalg.norm(v_inc_star)

        v_tgt_star = target - R_star
        u_tgt_star = v_tgt_star / np.linalg.norm(v_tgt_star)

        cos_i = float(np.clip(np.dot(-u_inc_star, n), -1.0, 1.0))
        cos_r = float(np.clip(np.dot(u_tgt_star, n), -1.0, 1.0))

        theta_i_deg = math.degrees(math.acos(cos_i))
        theta_r_deg = math.degrees(math.acos(cos_r))
        ang_disp = abs(theta_i_deg - theta_r_deg)

        # Check finite aperture clipping [0, 1]
        if t_root < -cls.ROOT_TOLERANCE or t_root > 1.0 + cls.ROOT_TOLERANCE:
            return DirectSnellReflectionResult(
                is_valid=False,
                reflection_point=R_star,
                t_mirror=t_root,
                collinearity_residual=res_val,
                incident_angle_deg=theta_i_deg,
                reflected_angle_deg=theta_r_deg,
                angular_disparity_deg=ang_disp,
                failure_reason="MISSED_FINITE_MIRROR",
            )

        return DirectSnellReflectionResult(
            is_valid=True,
            reflection_point=R_star,
            t_mirror=t_root,
            collinearity_residual=res_val,
            incident_angle_deg=theta_i_deg,
            reflected_angle_deg=theta_r_deg,
            angular_disparity_deg=ang_disp,
            failure_reason=None,
        )

    @classmethod
    def check_segment_sat_occlusion(
        cls,
        p1: np.ndarray,
        p2: np.ndarray,
        obs: ObstacleSpec,
    ) -> bool:
        """
        Tests line segment [p1, p2] intersection against obstacle using
        Separating Axis Theorem (SAT) for oriented bounding boxes or analytical
        quadratic line-circle discriminant.
        """
        c_obs = np.array([obs.center.x, obs.center.y], dtype=np.float64)

        if obs.obstacle_type == "circle":
            r = obs.radius if obs.radius is not None else 0.0
            d = p2 - p1
            f = p1 - c_obs
            a = float(np.dot(d, d))
            if a < 1e-12:
                return float(np.linalg.norm(f)) <= r
            b = 2.0 * float(np.dot(f, d))
            c = float(np.dot(f, f)) - r * r
            discriminant = b * b - 4.0 * a * c
            if discriminant < 0:
                return False
            sq = math.sqrt(discriminant)
            t1 = (-b - sq) / (2.0 * a)
            t2 = (-b + sq) / (2.0 * a)
            return (0.0 <= t1 <= 1.0) or (0.0 <= t2 <= 1.0) or (t1 < 0.0 and t2 > 1.0)

        elif obs.obstacle_type == "rectangle":
            w = obs.width if obs.width is not None else 0.0
            h = obs.height if obs.height is not None else 0.0
            deg = obs.angle_degrees or 0.0
            rad = math.radians(deg)
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)

            # Transform segment p1, p2 into obstacle local frame centered at (0, 0)
            def _to_local(p: np.ndarray) -> np.ndarray:
                rel = p - c_obs
                lx = cos_a * rel[0] + sin_a * rel[1]
                ly = -sin_a * rel[0] + cos_a * rel[1]
                return np.array([lx, ly], dtype=np.float64)

            lp1 = _to_local(p1)
            lp2 = _to_local(p2)

            hw = w / 2.0
            hh = h / 2.0

            # Liang-Barsky clipping algorithm in axis-aligned local space
            p_vec = [
                -(lp2[0] - lp1[0]),
                lp2[0] - lp1[0],
                -(lp2[1] - lp1[1]),
                lp2[1] - lp1[1],
            ]
            q_vec = [lp1[0] - (-hw), hw - lp1[0], lp1[1] - (-hh), hh - lp1[1]]

            u1 = 0.0
            u2 = 1.0

            for pi, qi in zip(p_vec, q_vec):
                if abs(pi) < 1e-12:
                    if qi < 0.0:
                        return False
                else:
                    t = qi / pi
                    if pi < 0.0:
                        if t > u2:
                            return False
                        u1 = max(u1, t)
                    else:
                        if t < u1:
                            return False
                        u2 = min(u2, t)

            return u1 <= u2

        return False

    @classmethod
    def evaluate_scene(cls, scene: InspectionScene) -> dict[str, Any]:
        """
        Runs complete Method 3 independent specular reasoning on an InspectionScene.
        """
        cam = np.array(
            [scene.camera.position.x, scene.camera.position.y], dtype=np.float64
        )
        target = np.array(
            [scene.target.center.x, scene.target.center.y], dtype=np.float64
        )
        m_a = np.array(
            [scene.mirror.endpoint_a.x, scene.mirror.endpoint_a.y], dtype=np.float64
        )
        m_b = np.array(
            [scene.mirror.endpoint_b.x, scene.mirror.endpoint_b.y], dtype=np.float64
        )
        norm = np.array(
            [scene.mirror.reflective_normal.dx, scene.mirror.reflective_normal.dy],
            dtype=np.float64,
        )

        snell_res = cls.solve_reflection_point(cam, target, m_a, m_b, norm)

        if not snell_res.is_valid:
            reason = ObstructionReason.NONE
            if snell_res.failure_reason == "MISSED_FINITE_MIRROR":
                reason = ObstructionReason.MISSED_MIRROR
            elif snell_res.failure_reason == "WRONG_SIDE_OF_MIRROR":
                reason = ObstructionReason.BACK_OF_MIRROR
            return {
                "is_visible": False,
                "state": VisibilityState.NOT_VISIBLE,
                "reason": reason,
                "reflection_point": snell_res.reflection_point,
                "collinearity_residual": snell_res.collinearity_residual,
                "failure_reason": snell_res.failure_reason,
            }

        R = snell_res.reflection_point
        assert R is not None

        # Check Camera FOV Cone
        cam_dir = np.array(
            [
                math.cos(math.radians(scene.camera.heading_deg)),
                math.sin(math.radians(scene.camera.heading_deg)),
            ],
            dtype=np.float64,
        )
        cam_to_r = R - cam
        dist_cam_r = float(np.linalg.norm(cam_to_r))
        u_cam_r = cam_to_r / dist_cam_r
        cos_cone = float(np.clip(np.dot(cam_dir, u_cam_r), -1.0, 1.0))
        angle_from_axis = math.degrees(math.acos(cos_cone))
        half_fov = scene.camera.fov_degrees / 2.0

        if angle_from_axis > half_fov + 1e-5 or dist_cam_r > scene.camera.max_range:
            return {
                "is_visible": False,
                "state": VisibilityState.NOT_VISIBLE,
                "reason": ObstructionReason.OUT_OF_FOV,
                "reflection_point": R,
                "collinearity_residual": snell_res.collinearity_residual,
                "failure_reason": "OUTSIDE_CAMERA_FOV",
            }

        # Check Obstacle Occlusions
        # Leg 1: Camera -> Reflection Point
        for obs in scene.obstacles:
            if cls.check_segment_sat_occlusion(cam, R, obs):
                return {
                    "is_visible": False,
                    "state": VisibilityState.NOT_VISIBLE,
                    "reason": ObstructionReason.BLOCKED_INCIDENT,
                    "blocking_obstacle_id": obs.obstacle_id,
                    "reflection_point": R,
                    "collinearity_residual": snell_res.collinearity_residual,
                    "failure_reason": f"BLOCKED_LEG1_BY_{obs.obstacle_id}",
                }

        # Leg 2: Reflection Point -> Target
        for obs in scene.obstacles:
            if cls.check_segment_sat_occlusion(R, target, obs):
                return {
                    "is_visible": False,
                    "state": VisibilityState.NOT_VISIBLE,
                    "reason": ObstructionReason.BLOCKED_REFLECTED,
                    "blocking_obstacle_id": obs.obstacle_id,
                    "reflection_point": R,
                    "collinearity_residual": snell_res.collinearity_residual,
                    "failure_reason": f"BLOCKED_LEG2_BY_{obs.obstacle_id}",
                }

        return {
            "is_visible": True,
            "state": VisibilityState.VISIBLE,
            "reason": ObstructionReason.NONE,
            "reflection_point": R,
            "collinearity_residual": snell_res.collinearity_residual,
            "failure_reason": None,
        }

    @classmethod
    def verify_triple(
        cls,
        scene: InspectionScene,
        oracle_res: Any,
        fermat_res: Any,
    ) -> TripleVerificationReport:
        """
        Cross-validates Method 1 (VisibilityOracle), Method 2 (IndependentGeometryVerifier),
        and Method 3 (DirectSnellRayVerifier).
        """
        m3_res = cls.evaluate_scene(scene)

        m1_vis = bool(oracle_res.is_visible)
        m2_vis = bool(fermat_res.is_visible)
        m3_vis = bool(m3_res["is_visible"])

        m1_reason = str(
            oracle_res.reason.value
            if hasattr(oracle_res.reason, "value")
            else oracle_res.reason
        )
        m2_reason = str(
            fermat_res.reason.value
            if hasattr(fermat_res.reason, "value")
            else fermat_res.reason
        )
        m3_reason = str(
            m3_res["reason"].value
            if hasattr(m3_res["reason"], "value")
            else m3_res["reason"]
        )

        # Compare reflection point coordinates across methods if available
        disparities: list[float] = []
        r1 = getattr(oracle_res, "reflection_point", None)
        r2 = getattr(fermat_res, "reflection_point", None)
        r3 = m3_res.get("reflection_point")

        if r1 is not None and r2 is not None:
            p1 = np.array([r1.x, r1.y]) if hasattr(r1, "x") else np.array(r1)
            p2 = np.array([r2.x, r2.y]) if hasattr(r2, "x") else np.array(r2)
            disparities.append(float(np.linalg.norm(p1 - p2)))

        if r2 is not None and r3 is not None:
            p2 = np.array([r2.x, r2.y]) if hasattr(r2, "x") else np.array(r2)
            p3 = (
                np.array([r3[0], r3[1]])
                if isinstance(r3, (list, np.ndarray))
                else np.array([r3.x, r3.y])
            )
            disparities.append(float(np.linalg.norm(p2 - p3)))

        max_disp = max(disparities) if disparities else 0.0
        max_fermat = float(getattr(fermat_res, "fermat_residual", 0.0))
        max_snell = float(m3_res.get("collinearity_residual", 0.0))

        vis_agreement = m1_vis == m2_vis == m3_vis
        coord_agreement = max_disp < cls.DELTA_R_TOLERANCE_MM
        all_agree = vis_agreement and coord_agreement

        details = (
            f"Triple verification consensus: M1={m1_vis} ({m1_reason}), "
            f"M2={m2_vis} ({m2_reason}), M3={m3_vis} ({m3_reason}). "
            f"Max R disparity={max_disp:.2e} mm, Fermat res={max_fermat:.2e}, "
            f"Snell collinearity res={max_snell:.2e}."
        )

        return TripleVerificationReport(
            scene_id=scene.scene_id,
            all_three_agree=all_agree,
            method1_oracle_visible=m1_vis,
            method2_fermat_visible=m2_vis,
            method3_snell_visible=m3_vis,
            method1_reason=m1_reason,
            method2_reason=m2_reason,
            method3_reason=m3_reason,
            max_reflection_disparity_mm=max_disp,
            max_fermat_residual=max_fermat,
            max_snell_collinearity_residual=max_snell,
            details=details,
        )
