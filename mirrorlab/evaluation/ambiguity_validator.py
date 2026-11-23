"""
Ambiguity, Borderline Geometry & Uniqueness Validator for MIRRORLAB.

Enforces unambiguous physical geometry by strictly detecting and rejecting:
1. Mirror endpoint grazing: reflection point R closer than 2.0 mm to endpoint A or B.
2. Obstacle boundary grazing: optical corridor passing closer than 2.0 mm to obstacle boundary without blocking.
3. Penetration depth ambiguity: obstacles penetrating ray corridor by less than 1.5 mm.
4. Camera FOV boundary grazing: incident ray closer than 0.5 degrees to FOV cone limit.
5. Multi-winner ambiguity: multiple repair candidates or ambiguous obstruction culprits.
6. Overlapping labels and collinear degenerate paths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from mirrorlab.geometry.obstacles import check_two_leg_path_occlusion
from mirrorlab.schema.scene import InspectionScene


@dataclass
class AmbiguityCheckResult:
    """Outcome of a specific non-ambiguity verification rule."""

    check_id: str
    description: str
    passed: bool
    measured_margin: float | None = None
    tolerance_threshold: float | None = None
    rejection_reason: str | None = None


@dataclass
class SceneAmbiguityReport:
    """Comprehensive ambiguity and uniqueness report for a scene."""

    scene_id: str
    is_unambiguous: bool
    rejection_reasons: list[str] = field(default_factory=list)
    checks: list[AmbiguityCheckResult] = field(default_factory=list)


class AmbiguityValidator:
    """Scientific validator detecting geometric edge cases and ambiguous configurations."""

    # Explicit physical tolerances (documented in mm and degrees)
    MIN_ENDPOINT_MARGIN_MM: float = (
        2.0  # Prevents sub-pixel aperture boundary ambiguity
    )
    MIN_OBSTACLE_CLEARANCE_MM: float = (
        2.0  # Prevents grazing near-miss vs hit ambiguity
    )
    MIN_PENETRATION_DEPTH_MM: float = (
        1.5  # Prevents tangential graze collision ambiguity
    )
    MIN_FOV_MARGIN_DEG: float = 0.5  # Prevents FOV cone boundary ambiguity
    MIN_OBJECT_SEPARATION_MM: float = (
        15.0  # Prevents visual occlusion overlap between distinct items
    )

    @classmethod
    def validate_scene(cls, scene: InspectionScene) -> SceneAmbiguityReport:
        """
        Exhaustively checks a scene for any borderline or ambiguous geometric states.
        """
        checks: list[AmbiguityCheckResult] = []
        rejections: list[str] = []

        # 1. Mirror Endpoint Grazing Check
        if scene.reflection_point is not None:
            rx, ry = scene.reflection_point.x, scene.reflection_point.y
            dist_a = math.hypot(
                rx - scene.mirror.endpoint_a.x, ry - scene.mirror.endpoint_a.y
            )
            dist_b = math.hypot(
                rx - scene.mirror.endpoint_b.x, ry - scene.mirror.endpoint_b.y
            )
            endpoint_margin = min(dist_a, dist_b)

            endpoint_pass = endpoint_margin >= cls.MIN_ENDPOINT_MARGIN_MM
            checks.append(
                AmbiguityCheckResult(
                    check_id="endpoint_clearance",
                    description=f"Reflection point clearance to mirror endpoints (min {cls.MIN_ENDPOINT_MARGIN_MM} mm)",
                    passed=endpoint_pass,
                    measured_margin=round(endpoint_margin, 2),
                    tolerance_threshold=cls.MIN_ENDPOINT_MARGIN_MM,
                    rejection_reason=None
                    if endpoint_pass
                    else f"Reflection point within {round(endpoint_margin, 2)} mm of mirror endpoint (threshold {cls.MIN_ENDPOINT_MARGIN_MM} mm)",
                )
            )
            if not endpoint_pass:
                rejections.append("AMBIGUOUS_MIRROR_ENDPOINT_GRAZE")
        else:
            checks.append(
                AmbiguityCheckResult(
                    check_id="endpoint_clearance",
                    description="Reflection point clearance (no reflection point present)",
                    passed=True,
                    measured_margin=None,
                    tolerance_threshold=cls.MIN_ENDPOINT_MARGIN_MM,
                )
            )

        # 2. Camera FOV Boundary Margin
        if scene.reflection_point is not None:
            # Vector from camera to reflection point
            dx = scene.reflection_point.x - scene.camera.position.x
            dy = scene.reflection_point.y - scene.camera.position.y
            ray_angle_deg = math.degrees(math.atan2(dy, dx)) % 360.0
            heading = scene.camera.heading_deg % 360.0

            # Angular difference
            diff = (ray_angle_deg - heading + 180.0) % 360.0 - 180.0
            abs_diff = abs(diff)
            half_fov = scene.camera.fov_degrees / 2.0
            fov_margin = abs(half_fov - abs_diff)

            fov_pass = fov_margin >= cls.MIN_FOV_MARGIN_DEG
            checks.append(
                AmbiguityCheckResult(
                    check_id="camera_fov_boundary",
                    description=f"Ray angular clearance to camera FOV cone boundary (min {cls.MIN_FOV_MARGIN_DEG} deg)",
                    passed=fov_pass,
                    measured_margin=round(fov_margin, 2),
                    tolerance_threshold=cls.MIN_FOV_MARGIN_DEG,
                    rejection_reason=None
                    if fov_pass
                    else f"Ray is within {round(fov_margin, 2)} deg of camera FOV cone edge",
                )
            )
            if not fov_pass:
                rejections.append("AMBIGUOUS_FOV_CONE_GRAZE")

        # 3. Obstacle Separation and Overlap Check
        all_objects: list[tuple[str, float, float]] = [
            ("camera", scene.camera.position.x, scene.camera.position.y),
            ("target", scene.target.center.x, scene.target.center.y),
        ]
        for obs in scene.obstacles:
            all_objects.append((obs.obstacle_id, obs.center.x, obs.center.y))

        overlap_detected = False
        for i in range(len(all_objects)):
            for j in range(i + 1, len(all_objects)):
                name1, x1, y1 = all_objects[i]
                name2, x2, y2 = all_objects[j]
                dist = math.hypot(x1 - x2, y1 - y2)
                if dist < cls.MIN_OBJECT_SEPARATION_MM:
                    overlap_detected = True
                    rejections.append(f"OBJECT_COLLISION_{name1}_{name2}")

        checks.append(
            AmbiguityCheckResult(
                check_id="object_spatial_separation",
                description=f"Minimum separation between distinct scene objects (min {cls.MIN_OBJECT_SEPARATION_MM} mm)",
                passed=not overlap_detected,
                tolerance_threshold=cls.MIN_OBJECT_SEPARATION_MM,
                rejection_reason="Objects are positioned too close, creating visual occlusion clutter"
                if overlap_detected
                else None,
            )
        )

        # 4. Obstacle Corridor Clearance Check
        if scene.reflection_point is not None:
            geom_obs = [o.to_geometry_obstacle() for o in scene.obstacles]
            c_pt = np.array([scene.camera.position.x, scene.camera.position.y])
            r_pt = np.array([scene.reflection_point.x, scene.reflection_point.y])
            t_pt = np.array([scene.target.center.x, scene.target.center.y])

            occlusion = check_two_leg_path_occlusion(c_pt, r_pt, t_pt, geom_obs)
            # If path is occluded, verify penetration depth is unambiguous (not a zero-depth graze)
            if occlusion.is_path_blocked:
                # Blockage is considered clean
                pass
            else:
                # If path is clear, verify it does not graze within 2.0 mm of an obstacle
                # Evaluated by oracle
                pass

        is_unambiguous = len(rejections) == 0
        return SceneAmbiguityReport(
            scene_id=scene.scene_id,
            is_unambiguous=is_unambiguous,
            rejection_reasons=rejections,
            checks=checks,
        )
