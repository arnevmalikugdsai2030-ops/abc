"""
Seeded, Constraint-Based Machine-Vision Inspection Scene Generator for MIRRORLAB.

Generates mathematically verified, non-ambiguous inspection scenes across the 6 controlled categories:
1. VALID_VISIBLE
2. MIRROR_MISS
3. OUTSIDE_FOV
4. WRONG_SIDE
5. BLOCKED_INCIDENT
6. BLOCKED_REFLECTED

Supports controlled difficulty tiers:
- EASY: Large mirrors (220-300 mm), broad FOV (70-90°), obvious obstacles, well-centered reflection.
- STANDARD: Moderate mirror (150-220 mm), moderate FOV (55-70°), 1-2 distractor obstacles.
- HARD: Small mirror (75-130 mm), narrow FOV (35-48°), multiple obstacles, two-leg reasoning.
- BOUNDARY: Near mirror endpoint, near FOV cone boundary, near obstacle intersection.
  (Strictly flagged as is_boundary=True to keep separate from headline benchmark).

Supports visual variations:
- Varied camera orientations and spatial setups (bottom-left, bottom-right, side-wall, angled).
- Varied scene positions and asymmetric camera/target distances.
- Diverse mirror angles and lengths.
- Obstacle arrangements with distractors.
- Anti-duplicate geometry fingerprinting.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from mirrorlab.geometry.obstacles import CircleObstacle, ObstacleBase, RectangleObstacle
from mirrorlab.geometry.oracle import VisibilityOracleResult, evaluate_visibility_oracle
from mirrorlab.geometry.vector import (
    vector_normalize,
    vector_rotate,
)
from mirrorlab.schema.common import (
    ObstructionReason,
    Point2D,
    Vector2D,
)
from mirrorlab.schema.scene import (
    CameraSpec,
    DifficultyLevel,
    InspectionScene,
    MirrorSpec,
    ObstacleSpec,
    SceneCategory,
    TargetSpec,
)


class GenerationError(Exception):
    """Raised when scene generation cannot satisfy constraints within max attempts."""


@dataclass(frozen=True)
class DifficultyParams:
    """Numerical parameters dictating geometric tightness for each difficulty tier."""

    mirror_length_range: tuple[float, float]
    fov_range_deg: tuple[float, float]
    min_clearance_mm: float
    min_edge_margin_t: float
    camera_offset_range_deg: tuple[float, float]
    num_distractors_range: tuple[int, int] = (0, 0)
    is_boundary: bool = False


DIFFICULTY_PRESETS: dict[DifficultyLevel, DifficultyParams] = {
    DifficultyLevel.EASY: DifficultyParams(
        mirror_length_range=(220.0, 300.0),
        fov_range_deg=(70.0, 90.0),
        min_clearance_mm=25.0,
        min_edge_margin_t=0.20,
        camera_offset_range_deg=(-3.0, 3.0),
        num_distractors_range=(0, 0),
        is_boundary=False,
    ),
    DifficultyLevel.STANDARD: DifficultyParams(
        mirror_length_range=(150.0, 220.0),
        fov_range_deg=(55.0, 70.0),
        min_clearance_mm=15.0,
        min_edge_margin_t=0.08,
        camera_offset_range_deg=(-8.0, 8.0),
        num_distractors_range=(1, 2),
        is_boundary=False,
    ),
    DifficultyLevel.MEDIUM: DifficultyParams(  # Alias for STANDARD
        mirror_length_range=(150.0, 220.0),
        fov_range_deg=(55.0, 70.0),
        min_clearance_mm=15.0,
        min_edge_margin_t=0.08,
        camera_offset_range_deg=(-8.0, 8.0),
        num_distractors_range=(1, 2),
        is_boundary=False,
    ),
    DifficultyLevel.HARD: DifficultyParams(
        mirror_length_range=(75.0, 130.0),
        fov_range_deg=(35.0, 48.0),
        min_clearance_mm=10.0,
        min_edge_margin_t=0.03,
        camera_offset_range_deg=(-15.0, 15.0),
        num_distractors_range=(2, 3),
        is_boundary=False,
    ),
    DifficultyLevel.BOUNDARY: DifficultyParams(
        mirror_length_range=(120.0, 180.0),
        fov_range_deg=(45.0, 60.0),
        min_clearance_mm=3.0,
        min_edge_margin_t=0.02,
        camera_offset_range_deg=(-12.0, 12.0),
        num_distractors_range=(1, 2),
        is_boundary=True,
    ),
}


def compute_geometry_fingerprint(
    camera_pos: Sequence[float],
    camera_heading: float,
    mirror_a: Sequence[float],
    mirror_b: Sequence[float],
    target_pos: Sequence[float],
    obstacles: Sequence[ObstacleBase],
    decimals: int = 1,
) -> str:
    """
    Generate a deterministic string fingerprint of rounded geometry to detect duplicates.
    """
    cam_str = f"C:({round(float(camera_pos[0]), decimals)},{round(float(camera_pos[1]), decimals)},{round(float(camera_heading), decimals)})"
    mir_str = f"M:({round(float(mirror_a[0]), decimals)},{round(float(mirror_a[1]), decimals)})-({round(float(mirror_b[0]), decimals)},{round(float(mirror_b[1]), decimals)})"
    tgt_str = f"T:({round(float(target_pos[0]), decimals)},{round(float(target_pos[1]), decimals)})"

    obs_parts = []
    for o in sorted(obstacles, key=lambda x: x.obstacle_id):
        if isinstance(o, CircleObstacle):
            obs_parts.append(
                f"C_{o.obstacle_id}:({round(float(o.center[0]), decimals)},{round(float(o.center[1]), decimals)},r={round(float(o.radius), decimals)})"
            )
        elif isinstance(o, RectangleObstacle):
            obs_parts.append(
                f"R_{o.obstacle_id}:({round(float(o.center[0]), decimals)},{round(float(o.center[1]), decimals)},w={round(float(o.width), decimals)},h={round(float(o.height), decimals)})"
            )

    return "|".join([cam_str, mir_str, tgt_str, ",".join(obs_parts)])


class SceneGenerator:
    """
    Constraint-based procedural generator for machine-vision reflection diagrams.
    """

    def __init__(
        self,
        canvas_width: float = 1000.0,
        canvas_height: float = 1000.0,
        max_attempts: int = 150,
    ):
        self.canvas_width = float(canvas_width)
        self.canvas_height = float(canvas_height)
        self.max_attempts = int(max_attempts)
        self.rejection_log: list[str] = []
        self.seen_fingerprints: set[str] = set()

    def reset_fingerprints(self) -> None:
        """Clear duplicate tracking cache."""
        self.seen_fingerprints.clear()

    def generate_scene(
        self,
        seed: int,
        category: SceneCategory = SceneCategory.VALID_VISIBLE,
        difficulty: DifficultyLevel = DifficultyLevel.STANDARD,
        scene_id: str | None = None,
    ) -> InspectionScene:
        """
        Generate a single constraint-verified inspection scene for the requested category.
        Guarantees zero geometric duplicates and independent oracle verification.
        """
        rng = np.random.default_rng(seed)
        params = DIFFICULTY_PRESETS[difficulty]
        sid = (
            scene_id
            or f"scene_{difficulty.value.lower()}_{category.value.lower()}_{seed}"
        )

        for attempt in range(1, self.max_attempts + 1):
            candidate = self._generate_candidate(rng, category, difficulty, params)
            if candidate is None:
                self.rejection_log.append(
                    f"Seed {seed} attempt {attempt}: Candidate construction failed"
                )
                continue

            # Duplicate geometry check
            fingerprint = compute_geometry_fingerprint(
                camera_pos=candidate["camera_pos"],
                camera_heading=candidate["camera_heading_deg"],
                mirror_a=candidate["mirror_a"],
                mirror_b=candidate["mirror_b"],
                target_pos=candidate["target_pos"],
                obstacles=candidate["obstacles"],
            )

            if fingerprint in self.seen_fingerprints:
                self.rejection_log.append(
                    f"Seed {seed} attempt {attempt}: Duplicate geometry detected"
                )
                continue

            # Run independent oracle verification
            oracle_verdict = self._verify_with_oracle(candidate)

            # Check if oracle verdict matches desired category
            if self._matches_category(oracle_verdict, category):
                self.seen_fingerprints.add(fingerprint)
                return self._build_inspection_scene(
                    sid,
                    seed,
                    category,
                    difficulty,
                    candidate,
                    oracle_verdict,
                    params.is_boundary,
                )
            else:
                self.rejection_log.append(
                    f"Seed {seed} attempt {attempt}: Oracle outcome mismatch "
                    f"(expected={category.value}, got state={oracle_verdict.state.value}, reason={oracle_verdict.reason.value})"
                )

        raise GenerationError(
            f"Failed to generate {difficulty.value} {category.value} scene after {self.max_attempts} attempts for seed {seed}."
        )

    def _generate_candidate(
        self,
        rng: np.random.Generator,
        category: SceneCategory,
        difficulty: DifficultyLevel,
        params: DifficultyParams,
    ) -> dict | None:
        """Procedurally construct candidate geometry with high visual and spatial variation."""
        # 1. Mirror geometry: support multiple center positions and angles
        m_len = float(rng.uniform(*params.mirror_length_range))
        m_cx = float(rng.uniform(380.0, 620.0))
        m_cy = float(rng.uniform(660.0, 780.0))

        # Diverse mirror tilt angles
        if difficulty == DifficultyLevel.EASY:
            m_tilt_deg = 0.0
        elif difficulty == DifficultyLevel.BOUNDARY:
            m_tilt_deg = float(rng.uniform(-10.0, 10.0))
        else:
            m_tilt_deg = float(rng.uniform(-20.0, 20.0))

        rad_tilt = math.radians(m_tilt_deg)
        t_unit = np.array([math.cos(rad_tilt), math.sin(rad_tilt)], dtype=np.float64)
        # Normal pointing into inspection cavity (-Y)
        n_unit = np.array([t_unit[1], -t_unit[0]], dtype=np.float64)

        half_len = m_len / 2.0
        m_a = np.array([m_cx, m_cy]) - half_len * t_unit
        m_b = np.array([m_cx, m_cy]) + half_len * t_unit

        # 2. Pick intended reflection point R on mirror
        if (
            difficulty == DifficultyLevel.BOUNDARY
            and category == SceneCategory.VALID_VISIBLE
        ):
            # Place reflection point extremely close to mirror endpoint (within 2% to 4%)
            t_hit = (
                float(rng.uniform(0.02, 0.045))
                if rng.random() > 0.5
                else float(rng.uniform(0.955, 0.98))
            )
        else:
            t_hit = float(
                rng.uniform(params.min_edge_margin_t, 1.0 - params.min_edge_margin_t)
            )

        r_pt = m_a + t_hit * (m_b - m_a)

        # 3. Choose incidence angle and optical distances (diverse setup variations)
        theta_i_deg = float(rng.uniform(22.0, 52.0))
        theta_i_rad = math.radians(theta_i_deg)

        # Diverse target and camera distances
        d_cam = float(rng.uniform(250.0, 420.0))
        d_tgt = float(rng.uniform(250.0, 420.0))

        # Support left-to-right or right-to-left optical paths
        side = 1.0 if rng.random() > 0.5 else -1.0
        c_dir = vector_rotate(n_unit, side * theta_i_rad)
        t_dir = vector_rotate(n_unit, -side * theta_i_rad)

        cam_pos = r_pt + d_cam * c_dir
        tgt_pos = r_pt + d_tgt * t_dir

        # Bounds check: Camera and Target must lie inside canvas with margin
        margin = 45.0
        if not (
            margin < cam_pos[0] < self.canvas_width - margin
            and margin < cam_pos[1] < self.canvas_height - margin
        ):
            return None
        if not (
            margin < tgt_pos[0] < self.canvas_width - margin
            and margin < tgt_pos[1] < self.canvas_height - margin
        ):
            return None

        # 4. Camera FOV and heading
        fov_deg = float(rng.uniform(*params.fov_range_deg))
        ray_dir = vector_normalize(r_pt - cam_pos)
        cam_aim_deg = math.degrees(math.atan2(ray_dir[1], ray_dir[0]))

        if (
            difficulty == DifficultyLevel.BOUNDARY
            and category == SceneCategory.VALID_VISIBLE
        ):
            # Place ray near FOV boundary (within 1.0° - 1.8° of cone edge)
            boundary_offset = (fov_deg / 2.0) - float(rng.uniform(0.8, 1.8))
            offset_dir = 1.0 if rng.random() > 0.5 else -1.0
            cam_heading_deg = cam_aim_deg + (boundary_offset * offset_dir)
        else:
            offset_deg = float(rng.uniform(*params.camera_offset_range_deg))
            cam_heading_deg = cam_aim_deg + offset_deg

        # 5. Partition wall to block direct sightline C -> T
        direct_mid = (cam_pos + tgt_pos) / 2.0
        partition_w = float(rng.uniform(30.0, 48.0))
        partition_h = float(rng.uniform(65.0, 105.0))
        partition = RectangleObstacle(
            "O1",
            center=direct_mid,
            width=partition_w,
            height=partition_h,
            angle_degrees=0.0,
        )

        obstacles: list[ObstacleBase] = [partition]

        # 6. Distractor obstacles (for STANDARD, HARD, and BOUNDARY)
        num_distractors = int(
            rng.integers(
                params.num_distractors_range[0], params.num_distractors_range[1] + 1
            )
        )
        if num_distractors > 0:
            # Distractor zones away from optical rays
            distractor_corners = [
                (100.0, 240.0, 680.0, 840.0),  # Upper-left
                (760.0, 900.0, 680.0, 840.0),  # Upper-right
                (760.0, 900.0, 120.0, 300.0),  # Lower-right
                (100.0, 240.0, 120.0, 300.0),  # Lower-left
            ]
            for d_idx in range(num_distractors):
                zone = distractor_corners[
                    (d_idx + int(rng.integers(0, len(distractor_corners))))
                    % len(distractor_corners)
                ]
                cx = float(rng.uniform(zone[0], zone[1]))
                cy = float(rng.uniform(zone[2], zone[3]))
                obs_id = f"O{len(obstacles) + 1}"

                # Ensure distractor does not overlap camera, target, mirror
                d_c = math.hypot(cx - cam_pos[0], cy - cam_pos[1])
                d_t = math.hypot(cx - tgt_pos[0], cy - tgt_pos[1])
                d_m = min(
                    math.hypot(cx - m_a[0], cy - m_a[1]),
                    math.hypot(cx - m_b[0], cy - m_b[1]),
                )

                if d_c > 60.0 and d_t > 60.0 and d_m > 50.0:
                    if rng.random() > 0.5:
                        obstacles.append(
                            CircleObstacle(
                                obs_id,
                                center=[cx, cy],
                                radius=float(rng.uniform(18.0, 28.0)),
                            )
                        )
                    else:
                        obstacles.append(
                            RectangleObstacle(
                                obs_id,
                                center=[cx, cy],
                                width=float(rng.uniform(35.0, 50.0)),
                                height=float(rng.uniform(35.0, 50.0)),
                                angle_degrees=float(rng.uniform(0.0, 45.0)),
                            )
                        )

        # 7. Apply Category-Specific Perturbations
        if category == SceneCategory.VALID_VISIBLE:
            pass  # Valid layout

        elif category == SceneCategory.MIRROR_MISS:
            if difficulty == DifficultyLevel.BOUNDARY:
                # Boundary miss: reflection point is just 3 - 8 mm outside mirror endpoint
                shift_dir = 1.0 if rng.random() > 0.5 else -1.0
                shift_amount = (half_len + float(rng.uniform(3.0, 8.0))) * shift_dir
            else:
                shift_dir = 1.0 if rng.random() > 0.5 else -1.0
                shift_amount = (half_len + float(rng.uniform(20.0, 50.0))) * shift_dir
            m_a = m_a + shift_amount * t_unit
            m_b = m_b + shift_amount * t_unit

        elif category == SceneCategory.OUTSIDE_FOV:
            if difficulty == DifficultyLevel.BOUNDARY:
                # Boundary out-of-FOV: ray is only 0.8° to 2.2° outside FOV limit
                turn_deg = (fov_deg / 2.0) + float(rng.uniform(0.8, 2.2))
            else:
                turn_deg = (fov_deg / 2.0) + float(rng.uniform(15.0, 45.0))
            turn_dir = 1.0 if rng.random() > 0.5 else -1.0
            cam_heading_deg = cam_aim_deg + (turn_deg * turn_dir)

        elif category == SceneCategory.WRONG_SIDE:
            n_unit = -n_unit

        elif category == SceneCategory.BLOCKED_INCIDENT:
            t_block = float(rng.uniform(0.35, 0.65))
            block_pt = cam_pos + t_block * (r_pt - cam_pos)
            radius = float(rng.uniform(16.0, 26.0))
            obs_id = f"O{len(obstacles) + 1}"
            obstacles.append(CircleObstacle(obs_id, center=block_pt, radius=radius))

        elif category == SceneCategory.BLOCKED_REFLECTED:
            t_block = float(rng.uniform(0.35, 0.65))
            block_pt = r_pt + t_block * (tgt_pos - r_pt)
            radius = float(rng.uniform(16.0, 26.0))
            obs_id = f"O{len(obstacles) + 1}"
            obstacles.append(CircleObstacle(obs_id, center=block_pt, radius=radius))

        return {
            "camera_pos": cam_pos,
            "camera_heading_deg": cam_heading_deg,
            "camera_fov_degrees": fov_deg,
            "target_pos": tgt_pos,
            "mirror_a": m_a,
            "mirror_b": m_b,
            "reflective_normal": n_unit,
            "obstacles": obstacles,
        }

    def _verify_with_oracle(self, candidate: dict) -> VisibilityOracleResult:
        """Run candidate geometry through the independent visibility oracle."""
        return evaluate_visibility_oracle(
            camera_pos=candidate["camera_pos"],
            camera_forward=candidate["camera_heading_deg"],
            camera_fov_degrees=candidate["camera_fov_degrees"],
            target_pos=candidate["target_pos"],
            mirror_a=candidate["mirror_a"],
            mirror_b=candidate["mirror_b"],
            obstacles=candidate["obstacles"],
            reflective_normal=candidate["reflective_normal"],
        )

    def _matches_category(
        self, res: VisibilityOracleResult, cat: SceneCategory
    ) -> bool:
        """Check if oracle verdict satisfies the exact category requirements."""
        if cat == SceneCategory.VALID_VISIBLE:
            return (
                res.is_visible is True
                and res.reason == ObstructionReason.NONE
                and res.is_direct_sight_blocked is True
            )
        elif cat == SceneCategory.MIRROR_MISS:
            return (
                res.is_visible is False
                and res.reason == ObstructionReason.MISSED_MIRROR
            )
        elif cat == SceneCategory.OUTSIDE_FOV:
            return (
                res.is_visible is False and res.reason == ObstructionReason.OUT_OF_FOV
            )
        elif cat == SceneCategory.WRONG_SIDE:
            return (
                res.is_visible is False
                and res.reason == ObstructionReason.BACK_OF_MIRROR
            )
        elif cat == SceneCategory.BLOCKED_INCIDENT:
            return (
                res.is_visible is False
                and res.reason == ObstructionReason.BLOCKED_INCIDENT
            )
        elif cat == SceneCategory.BLOCKED_REFLECTED:
            return (
                res.is_visible is False
                and res.reason == ObstructionReason.BLOCKED_REFLECTED
            )
        return False

    def _build_inspection_scene(
        self,
        scene_id: str,
        seed: int,
        category: SceneCategory,
        difficulty: DifficultyLevel,
        c: dict,
        oracle: VisibilityOracleResult,
        is_boundary: bool = False,
    ) -> InspectionScene:
        """Assemble fully typed Pydantic InspectionScene."""
        cam_spec = CameraSpec(
            position=Point2D(x=float(c["camera_pos"][0]), y=float(c["camera_pos"][1])),
            heading_deg=float(c["camera_heading_deg"]),
            fov_degrees=float(c["camera_fov_degrees"]),
        )
        mirror_spec = MirrorSpec(
            endpoint_a=Point2D(x=float(c["mirror_a"][0]), y=float(c["mirror_a"][1])),
            endpoint_b=Point2D(x=float(c["mirror_b"][0]), y=float(c["mirror_b"][1])),
            reflective_normal=Vector2D(
                dx=float(c["reflective_normal"][0]), dy=float(c["reflective_normal"][1])
            ),
        )
        target_spec = TargetSpec(
            center=Point2D(x=float(c["target_pos"][0]), y=float(c["target_pos"][1])),
        )

        obstacle_specs: list[ObstacleSpec] = []
        for obs in c["obstacles"]:
            if isinstance(obs, CircleObstacle):
                obstacle_specs.append(
                    ObstacleSpec(
                        obstacle_id=obs.obstacle_id,
                        obstacle_type="circle",
                        center=Point2D(x=float(obs.center[0]), y=float(obs.center[1])),
                        radius=float(obs.radius),
                    )
                )
            elif isinstance(obs, RectangleObstacle):
                obstacle_specs.append(
                    ObstacleSpec(
                        obstacle_id=obs.obstacle_id,
                        obstacle_type="rectangle",
                        center=Point2D(x=float(obs.center[0]), y=float(obs.center[1])),
                        width=float(obs.width),
                        height=float(obs.height),
                        angle_degrees=float(obs.angle_degrees),
                    )
                )

        refl_pt = (
            Point2D(
                x=float(oracle.reflection_point[0]), y=float(oracle.reflection_point[1])
            )
            if oracle.reflection_point is not None
            else None
        )
        vt_pt = (
            Point2D(
                x=float(oracle.virtual_target[0]), y=float(oracle.virtual_target[1])
            )
            if oracle.virtual_target is not None
            else None
        )
        path = [Point2D(x=float(p[0]), y=float(p[1])) for p in oracle.path_points]

        return InspectionScene(
            scene_id=scene_id,
            seed=seed,
            category=category,
            difficulty=difficulty,
            canvas_width=self.canvas_width,
            canvas_height=self.canvas_height,
            camera=cam_spec,
            mirror=mirror_spec,
            target=target_spec,
            obstacles=obstacle_specs,
            ground_truth_state=oracle.state,
            ground_truth_reason=oracle.reason,
            diagnostic_details=oracle.diagnostic_details,
            reflection_point=refl_pt,
            virtual_target=vt_pt,
            optical_path=path,
            incident_angle_deg=oracle.incident_angle_deg,
            reflected_angle_deg=oracle.reflected_angle_deg,
            total_optical_distance=oracle.total_optical_distance,
            is_direct_sight_blocked=oracle.is_direct_sight_blocked,
            is_boundary=is_boundary,
        )

    def generate_batch(
        self,
        count: int,
        start_seed: int = 1000,
        difficulty: DifficultyLevel = DifficultyLevel.STANDARD,
        categories: list[SceneCategory] | None = None,
    ) -> list[InspectionScene]:
        """
        Generate a balanced batch of verified scenes across requested categories.
        """
        cats = categories or list(SceneCategory)
        scenes: list[InspectionScene] = []

        for i in range(count):
            cat = cats[i % len(cats)]
            seed = start_seed + i
            scene = self.generate_scene(seed=seed, category=cat, difficulty=difficulty)
            scenes.append(scene)

        return scenes
