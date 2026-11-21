"""
Seeded, Constraint-Based Machine-Vision Inspection Scene Generator for MIRRORLAB.

Generates mathematically verified, non-ambiguous inspection scenes across the 6 controlled categories:
1. VALID_VISIBLE
2. MIRROR_MISS
3. OUTSIDE_FOV
4. WRONG_SIDE
5. BLOCKED_INCIDENT
6. BLOCKED_REFLECTED

Enforces:
- Deterministic reproducibility given integer seeds.
- Rejection of ambiguous, grazing, or degenerate cases.
- Independent oracle validation on every candidate.
- Configurable difficulty settings (EASY, MEDIUM, HARD).
"""

from __future__ import annotations

import math
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


DIFFICULTY_PRESETS: dict[DifficultyLevel, DifficultyParams] = {
    DifficultyLevel.EASY: DifficultyParams(
        mirror_length_range=(200.0, 260.0),
        fov_range_deg=(70.0, 85.0),
        min_clearance_mm=25.0,
        min_edge_margin_t=0.10,
        camera_offset_range_deg=(-5.0, 5.0),
    ),
    DifficultyLevel.MEDIUM: DifficultyParams(
        mirror_length_range=(140.0, 200.0),
        fov_range_deg=(55.0, 70.0),
        min_clearance_mm=15.0,
        min_edge_margin_t=0.06,
        camera_offset_range_deg=(-10.0, 10.0),
    ),
    DifficultyLevel.HARD: DifficultyParams(
        mirror_length_range=(90.0, 140.0),
        fov_range_deg=(40.0, 52.0),
        min_clearance_mm=10.0,
        min_edge_margin_t=0.03,
        camera_offset_range_deg=(-15.0, 15.0),
    ),
}


class SceneGenerator:
    """
    Constraint-based procedural generator for machine-vision reflection diagrams.
    """

    def __init__(
        self,
        canvas_width: float = 1000.0,
        canvas_height: float = 1000.0,
        max_attempts: int = 100,
    ):
        self.canvas_width = float(canvas_width)
        self.canvas_height = float(canvas_height)
        self.max_attempts = int(max_attempts)
        self.rejection_log: list[str] = []

    def generate_scene(
        self,
        seed: int,
        category: SceneCategory = SceneCategory.VALID_VISIBLE,
        difficulty: DifficultyLevel = DifficultyLevel.MEDIUM,
        scene_id: str | None = None,
    ) -> InspectionScene:
        """
        Generate a single constraint-verified inspection scene for the requested category.

        Args:
            seed: Integer random seed for deterministic reproduction.
            category: Desired category outcome.
            difficulty: EASY, MEDIUM, or HARD.
            scene_id: Optional string identifier.

        Returns:
            InspectionScene: Verified scene with full ground-truth oracle proof.

        Raises:
            GenerationError: If constraints cannot be met within max_attempts.
        """
        rng = np.random.default_rng(seed)
        params = DIFFICULTY_PRESETS[difficulty]
        sid = scene_id or f"scene_{category.value.lower()}_{seed}"

        for attempt in range(1, self.max_attempts + 1):
            candidate = self._generate_candidate(rng, category, difficulty, params)
            if candidate is None:
                self.rejection_log.append(
                    f"Seed {seed} attempt {attempt}: Candidate construction failed"
                )
                continue

            # Run independent oracle verification
            oracle_verdict = self._verify_with_oracle(candidate)

            # Check if oracle verdict matches desired category
            if self._matches_category(oracle_verdict, category):
                return self._build_inspection_scene(
                    sid, seed, category, difficulty, candidate, oracle_verdict
                )
            else:
                self.rejection_log.append(
                    f"Seed {seed} attempt {attempt}: Oracle outcome mismatch "
                    f"(expected={category.value}, got state={oracle_verdict.state.value}, reason={oracle_verdict.reason.value})"
                )

        raise GenerationError(
            f"Failed to generate {category.value} scene after {self.max_attempts} attempts for seed {seed}."
        )

    def _generate_candidate(
        self,
        rng: np.random.Generator,
        category: SceneCategory,
        difficulty: DifficultyLevel,
        params: DifficultyParams,
    ) -> dict | None:
        """Procedurally construct candidate geometry based on inverse optical reflection."""
        # 1. Mirror geometry: Centered in upper canvas (y in [650, 750])
        m_len = rng.uniform(*params.mirror_length_range)
        m_cx = rng.uniform(400.0, 600.0)
        m_cy = rng.uniform(650.0, 750.0)
        m_tilt_deg = (
            rng.uniform(-15.0, 15.0) if difficulty != DifficultyLevel.EASY else 0.0
        )

        rad_tilt = math.radians(m_tilt_deg)
        t_unit = np.array([math.cos(rad_tilt), math.sin(rad_tilt)], dtype=np.float64)
        # Normal pointing down towards lower inspection stage (-Y)
        n_unit = np.array([t_unit[1], -t_unit[0]], dtype=np.float64)

        half_len = m_len / 2.0
        m_a = np.array([m_cx, m_cy]) - half_len * t_unit
        m_b = np.array([m_cx, m_cy]) + half_len * t_unit

        # 2. Pick intended reflection point R on mirror
        t_hit = rng.uniform(params.min_edge_margin_t, 1.0 - params.min_edge_margin_t)
        r_pt = m_a + t_hit * (m_b - m_a)

        # 3. Choose incidence angle and optical distances
        theta_i_deg = rng.uniform(25.0, 50.0)
        theta_i_rad = math.radians(theta_i_deg)

        d_cam = rng.uniform(260.0, 400.0)
        d_tgt = rng.uniform(260.0, 400.0)

        # In mirror frame, reflective normal n_unit points into front half-space (-Y).
        # Camera and Target both lie in front of the mirror (in direction of n_unit).
        # Using Law of Reflection:
        # Camera is on one side (+theta_i), Target is on opposite side (-theta_i).
        side = 1.0 if rng.random() > 0.5 else -1.0
        c_dir = vector_rotate(n_unit, side * theta_i_rad)
        t_dir = vector_rotate(n_unit, -side * theta_i_rad)

        cam_pos = r_pt + d_cam * c_dir
        tgt_pos = r_pt + d_tgt * t_dir

        # Bounds check: Camera and Target must lie inside canvas with margin
        margin = 50.0
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
        fov_deg = rng.uniform(*params.fov_range_deg)
        # Camera aims towards R, optionally with slight angular offset
        ray_dir = vector_normalize(r_pt - cam_pos)
        cam_aim_deg = math.degrees(math.atan2(ray_dir[1], ray_dir[0]))

        offset_deg = rng.uniform(*params.camera_offset_range_deg)
        cam_heading_deg = cam_aim_deg + offset_deg

        # 5. Partition wall to block direct sightline C -> T
        direct_mid = (cam_pos + tgt_pos) / 2.0
        partition_w = rng.uniform(30.0, 45.0)
        partition_h = rng.uniform(60.0, 95.0)
        partition = RectangleObstacle(
            "partition_wall",
            center=direct_mid,
            width=partition_w,
            height=partition_h,
            angle_degrees=0.0,
        )

        obstacles: list[ObstacleBase] = [partition]

        # 6. Apply Category-Specific Perturbations
        if category == SceneCategory.VALID_VISIBLE:
            pass  # Keep valid layout

        elif category == SceneCategory.MIRROR_MISS:
            # Shift mirror segment so R misses it
            shift_dir = 1.0 if rng.random() > 0.5 else -1.0
            shift_amount = (half_len + rng.uniform(20.0, 50.0)) * shift_dir
            m_a = m_a + shift_amount * t_unit
            m_b = m_b + shift_amount * t_unit

        elif category == SceneCategory.OUTSIDE_FOV:
            # Turn camera away so R is outside FOV cone
            turn_deg = (fov_deg / 2.0) + rng.uniform(15.0, 45.0)
            turn_dir = 1.0 if rng.random() > 0.5 else -1.0
            cam_heading_deg = cam_aim_deg + (turn_deg * turn_dir)

        elif category == SceneCategory.WRONG_SIDE:
            # Invert mirror normal so reflective side points away from camera
            n_unit = -n_unit

        elif category == SceneCategory.BLOCKED_INCIDENT:
            # Place blocker on Leg 1 (Camera to Mirror)
            t_block = rng.uniform(0.35, 0.65)
            block_pt = cam_pos + t_block * (r_pt - cam_pos)
            radius = rng.uniform(15.0, 25.0)
            obstacles.append(
                CircleObstacle("blocker_incident", center=block_pt, radius=radius)
            )

        elif category == SceneCategory.BLOCKED_REFLECTED:
            # Place blocker on Leg 2 (Mirror to Target)
            t_block = rng.uniform(0.35, 0.65)
            block_pt = r_pt + t_block * (tgt_pos - r_pt)
            radius = rng.uniform(15.0, 25.0)
            obstacles.append(
                CircleObstacle("blocker_reflected", center=block_pt, radius=radius)
            )

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
        )

    def generate_batch(
        self,
        count: int,
        start_seed: int = 1000,
        difficulty: DifficultyLevel = DifficultyLevel.MEDIUM,
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
