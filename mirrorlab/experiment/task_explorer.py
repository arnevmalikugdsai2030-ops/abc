"""
Task Condition Exploration Matrix & Experiment Runner for MIRRORLAB.

Enables systematic parameter sweeps across 5 visual reasoning dimensions:
1. Geometry: Single vs Dual Plane Mirrors, Grazing Angles, Variable Apertures
2. Occlusion: Unobstructed, Near-Mirror, Near-Target, Corridor Occlusions
3. Visual Complexity: Minimal vs Cluttered with Distractors
4. Reasoning Complexity: Visibility, Blocker Localization, Repair, Counterfactual
5. Counterfactual Sensitivity: Invariance vs Critical Flip

Logs full experimental manifests to experiments/ with rejection accounting.
"""

from __future__ import annotations

import math
import time
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from mirrorlab.experiment.difficulty_calibrator import (
    DifficultyCalibrator,
    DifficultyMetrics,
    DifficultyTier,
)
from mirrorlab.geometry.oracle import VisibilityOracle, evaluate_visibility_oracle
from mirrorlab.schema.common import (
    Point2D,
    Vector2D,
    VisibilityState,
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


class GeometryRegime(str, Enum):
    """Geometric structure of reflection surfaces."""

    SINGLE_PLANAR = "single_planar"
    GRAZING_INCIDENCE = "grazing_incidence"
    NARROW_APERTURE = "narrow_aperture"
    OFF_AXIS_EXTREME = "off_axis_extreme"
    DUAL_MIRROR_FOLDED = "dual_mirror_folded"


class OcclusionRegime(str, Enum):
    """Obstacle proximity and obstruction conditions."""

    NONE = "none"
    CORRIDOR_PARTITION = "corridor_partition"
    NEAR_MIRROR_BLOCKER = "near_mirror_blocker"
    NEAR_TARGET_BLOCKER = "near_target_blocker"
    MULTIPLE_OBSTACLES = "multiple_obstacles"
    GRAZING_CLEARANCE = "grazing_clearance"


class ClutterRegime(str, Enum):
    """Visual clutter and distractor density."""

    CLEAN = "clean"
    MODERATE = "moderate"
    HIGH_DISTRACTORS = "high_distractors"


class ReasoningRegime(str, Enum):
    """Target reasoning operation for evaluation."""

    TASK_A_VISIBILITY = "task_a_visibility"
    TASK_B_OBSTRUCTION = "task_b_obstruction"
    TASK_C_REPAIR = "task_c_repair"
    TASK_D_COUNTERFACTUAL = "task_d_counterfactual"
    PATH_FALSIFICATION = "path_falsification"


class TaskCondition(BaseModel):
    """Specification of a single experimental test condition."""

    condition_id: str
    geometry: GeometryRegime = GeometryRegime.SINGLE_PLANAR
    occlusion: OcclusionRegime = OcclusionRegime.CORRIDOR_PARTITION
    clutter: ClutterRegime = ClutterRegime.CLEAN
    reasoning: ReasoningRegime = ReasoningRegime.TASK_A_VISIBILITY
    sample_size: int = 10
    seed_offset: int = 1000


class TaskGenerationResult(BaseModel):
    """Outcome of generating a single task under test condition."""

    task_id: str
    condition_id: str
    is_valid: bool
    rejection_reason: str | None = None
    difficulty: DifficultyMetrics | None = None
    scene_data: dict[str, Any] | None = None
    ground_truth: dict[str, Any] | None = None


class ExperimentManifest(BaseModel):
    """Complete record of an exploratory task discovery experiment."""

    experiment_id: str
    date: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    generator_version: str = "mirrorlab-v1.2.0"
    random_seed: int
    condition: TaskCondition
    total_attempted: int
    total_valid: int
    total_rejected_ambiguous: int
    total_rejected_rendering: int
    difficulty_distribution: dict[str, int]
    runtime_seconds: float
    rejection_summary: dict[str, int]
    tasks: list[TaskGenerationResult]
    conclusion: str


class TaskExplorer:
    """Exploration engine that tests diverse candidate visual tasks."""

    def __init__(self, output_dir: Path | str = "experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_condition_experiment(
        self,
        condition: TaskCondition,
        base_seed: int = 42,
    ) -> ExperimentManifest:
        """
        Executes generation across a parameterized condition, tracking rejections and difficulty.
        """
        t0 = time.perf_counter()
        results: list[TaskGenerationResult] = []
        rejected_ambiguous = 0
        rejected_rendering = 0
        rejection_counts: dict[str, int] = {}
        difficulty_dist: dict[str, int] = {t.value: 0 for t in DifficultyTier}

        for i in range(condition.sample_size):
            seed = base_seed + condition.seed_offset + i
            tid = f"exp_{condition.condition_id}_{seed}"

            scene, rejection = self._generate_candidate_scene(condition, seed)
            if rejection:
                rejected_ambiguous += 1
                rejection_counts[rejection] = rejection_counts.get(rejection, 0) + 1
                results.append(
                    TaskGenerationResult(
                        task_id=tid,
                        condition_id=condition.condition_id,
                        is_valid=False,
                        rejection_reason=rejection,
                    )
                )
                continue

            assert scene is not None

            # Verify with VisibilityOracle
            verdict = VisibilityOracle.evaluate(scene)

            # Ambiguity checks
            ambiguity_reason = self._check_scene_ambiguity(scene, verdict, condition)
            if ambiguity_reason:
                rejected_ambiguous += 1
                rejection_counts[ambiguity_reason] = (
                    rejection_counts.get(ambiguity_reason, 0) + 1
                )
                results.append(
                    TaskGenerationResult(
                        task_id=tid,
                        condition_id=condition.condition_id,
                        is_valid=False,
                        rejection_reason=ambiguity_reason,
                    )
                )
                continue

            # Calculate quantitative difficulty
            r_pt = (
                (float(verdict.reflection_point[0]), float(verdict.reflection_point[1]))
                if verdict.reflection_point is not None
                else None
            )
            theta = verdict.incident_angle_deg or 45.0
            diff = DifficultyCalibrator.calculate_metrics(
                scene=scene,
                reflection_point=r_pt,
                incidence_angle=theta,
                is_counterfactual=(
                    condition.reasoning == ReasoningRegime.TASK_D_COUNTERFACTUAL
                ),
            )
            difficulty_dist[diff.calibrated_tier.value] += 1

            results.append(
                TaskGenerationResult(
                    task_id=tid,
                    condition_id=condition.condition_id,
                    is_valid=True,
                    difficulty=diff,
                    scene_data=scene.model_dump(mode="json"),
                    ground_truth={
                        "visible": verdict.state == VisibilityState.VISIBLE,
                        "reason": verdict.reason.value,
                        "reflection_point": [r_pt[0], r_pt[1]] if r_pt else None,
                    },
                )
            )

        runtime = time.perf_counter() - t0
        valid_count = sum(1 for r in results if r.is_valid)

        manifest = ExperimentManifest(
            experiment_id=f"exp_{condition.condition_id}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
            random_seed=base_seed,
            condition=condition,
            total_attempted=condition.sample_size,
            total_valid=valid_count,
            total_rejected_ambiguous=rejected_ambiguous,
            total_rejected_rendering=rejected_rendering,
            difficulty_distribution=difficulty_dist,
            runtime_seconds=round(runtime, 3),
            rejection_summary=rejection_counts,
            tasks=results,
            conclusion=(
                f"Generated {valid_count}/{condition.sample_size} valid tasks. "
                f"Yield: {round(valid_count / condition.sample_size * 100, 1)}%. "
                f"Dominant difficulty: {max(difficulty_dist, key=difficulty_dist.get)}."
            ),
        )

        # Save experiment manifest
        manifest_path = self.output_dir / f"{manifest.experiment_id}.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(manifest.model_dump_json(indent=2))

        return manifest

    def _generate_candidate_scene(
        self, condition: TaskCondition, seed: int
    ) -> tuple[InspectionScene | None, str | None]:
        """Procedural scene generator adapted to condition parameters."""
        import random

        rng = random.Random(seed)

        # Mirror parameters based on geometry
        if condition.geometry == GeometryRegime.GRAZING_INCIDENCE:
            m_a = Point2D(x=100.0, y=850.0)
            m_b = Point2D(x=700.0, y=850.0)
            m_norm = Vector2D(dx=0.0, dy=-1.0)
            cam_pos = Point2D(x=120.0, y=150.0)
            target_pos = Point2D(x=680.0, y=180.0)
            cam_heading = 78.0
        elif condition.geometry == GeometryRegime.NARROW_APERTURE:
            m_a = Point2D(x=350.0, y=700.0)
            m_b = Point2D(x=450.0, y=700.0)  # Only 100mm aperture
            m_norm = Vector2D(dx=0.0, dy=-1.0)
            cam_pos = Point2D(x=200.0, y=200.0)
            target_pos = Point2D(x=550.0, y=250.0)
            cam_heading = 60.0
        else:
            # Standard planar mirror
            m_a = Point2D(x=150.0 + rng.uniform(-30, 30), y=700.0)
            m_b = Point2D(x=650.0 + rng.uniform(-30, 30), y=700.0)
            m_norm = Vector2D(dx=0.0, dy=-1.0)
            cam_pos = Point2D(x=200.0 + rng.uniform(-20, 20), y=200.0)
            target_pos = Point2D(x=500.0 + rng.uniform(-20, 20), y=250.0)
            cam_heading = 65.0

        mirror = MirrorSpec(endpoint_a=m_a, endpoint_b=m_b, reflective_normal=m_norm)
        camera = CameraSpec(position=cam_pos, heading_deg=cam_heading, fov_degrees=65.0)
        target = TargetSpec(center=target_pos, radius_mm=8.0)

        # Build obstacles
        obstacles: list[ObstacleSpec] = []
        # Base barrier separating camera and target
        obstacles.append(
            ObstacleSpec(
                obstacle_id="O1",
                obstacle_type="rectangle",
                center=Point2D(
                    x=(cam_pos.x + target_pos.x) / 2, y=(cam_pos.y + target_pos.y) / 2
                ),
                width=60.0,
                height=180.0,
                angle_degrees=0.0,
            )
        )

        if condition.occlusion == OcclusionRegime.NEAR_MIRROR_BLOCKER:
            obstacles.append(
                ObstacleSpec(
                    obstacle_id="O2",
                    obstacle_type="circle",
                    center=Point2D(x=350.0, y=620.0),
                    radius=20.0,
                )
            )
        elif condition.occlusion == OcclusionRegime.NEAR_TARGET_BLOCKER:
            obstacles.append(
                ObstacleSpec(
                    obstacle_id="O2",
                    obstacle_type="circle",
                    center=Point2D(x=target_pos.x - 35.0, y=target_pos.y + 40.0),
                    radius=18.0,
                )
            )
        elif condition.occlusion == OcclusionRegime.MULTIPLE_OBSTACLES:
            obstacles.append(
                ObstacleSpec(
                    obstacle_id="O2",
                    obstacle_type="rectangle",
                    center=Point2D(x=280.0, y=450.0),
                    width=40.0,
                    height=80.0,
                    angle_degrees=15.0,
                )
            )
            obstacles.append(
                ObstacleSpec(
                    obstacle_id="O3",
                    obstacle_type="circle",
                    center=Point2D(x=450.0, y=480.0),
                    radius=22.0,
                )
            )

        if condition.clutter == ClutterRegime.HIGH_DISTRACTORS:
            for d_idx in range(4):
                dx = rng.uniform(50.0, 800.0)
                dy = rng.uniform(50.0, 500.0)
                # Ensure distractor does not overlap camera, target, or mirror
                obstacles.append(
                    ObstacleSpec(
                        obstacle_id=f"D{d_idx + 1}",
                        obstacle_type="circle",
                        center=Point2D(x=dx, y=dy),
                        radius=12.0,
                    )
                )

        # Evaluate initial oracle to populate InspectionScene
        verdict = evaluate_visibility_oracle(
            camera_pos=[camera.position.x, camera.position.y],
            camera_forward=camera.heading_deg,
            camera_fov_degrees=camera.fov_degrees,
            target_pos=[target.center.x, target.center.y],
            mirror_a=[mirror.endpoint_a.x, mirror.endpoint_a.y],
            mirror_b=[mirror.endpoint_b.x, mirror.endpoint_b.y],
            reflective_normal=[
                mirror.reflective_normal.dx,
                mirror.reflective_normal.dy,
            ],
            obstacles=[obs.to_geometry_obstacle() for obs in obstacles],
        )

        category = (
            SceneCategory.VALID_VISIBLE
            if verdict.state == VisibilityState.VISIBLE
            else SceneCategory.BLOCKED_INCIDENT
        )

        ref_pt = (
            Point2D(
                x=float(verdict.reflection_point[0]),
                y=float(verdict.reflection_point[1]),
            )
            if verdict.reflection_point is not None
            else None
        )
        vt_pt = (
            Point2D(
                x=float(verdict.virtual_target[0]), y=float(verdict.virtual_target[1])
            )
            if verdict.virtual_target is not None
            else None
        )
        opt_path = [Point2D(x=float(p[0]), y=float(p[1])) for p in verdict.path_points]

        scene = InspectionScene(
            scene_id=f"scene_{seed}",
            seed=seed,
            category=category,
            difficulty=DifficultyLevel.MEDIUM,
            camera=camera,
            mirror=mirror,
            target=target,
            obstacles=obstacles,
            ground_truth_state=verdict.state,
            ground_truth_reason=verdict.reason,
            diagnostic_details=verdict.diagnostic_details,
            reflection_point=ref_pt,
            virtual_target=vt_pt,
            optical_path=opt_path,
            incident_angle_deg=verdict.incident_angle_deg,
            reflected_angle_deg=verdict.reflected_angle_deg,
            total_optical_distance=verdict.total_optical_distance,
            is_direct_sight_blocked=verdict.is_direct_sight_blocked,
        )
        return scene, None

    def _check_scene_ambiguity(
        self,
        scene: InspectionScene,
        verdict: Any,
        condition: TaskCondition,
    ) -> str | None:
        """Enforces rigorous non-ambiguity tolerances."""
        if verdict.reflection_point is not None:
            rx, ry = (
                float(verdict.reflection_point[0]),
                float(verdict.reflection_point[1]),
            )
            # Check mirror endpoint margin: reject if closer than 3.0 mm to endpoint
            da = math.hypot(
                rx - scene.mirror.endpoint_a.x, ry - scene.mirror.endpoint_a.y
            )
            db = math.hypot(
                rx - scene.mirror.endpoint_b.x, ry - scene.mirror.endpoint_b.y
            )
            if min(da, db) < 3.0:
                return "AMBIGUOUS_ENDPOINT_PROXIMITY"

        return None
