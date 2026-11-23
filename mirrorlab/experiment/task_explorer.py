"""
Task Condition Exploration Matrix & Experiment Runner for MIRRORLAB.

Enables systematic parameter sweeps across 5 visual reasoning dimensions:
1. Geometry: Single Planar, Grazing Incidence, Narrow Aperture, Off-Axis Extreme, Dual-Mirror Folded
2. Occlusion: None, Corridor Partition, Near-Mirror Blocker, Near-Target Blocker, Multiple Obstacles, Grazing Clearance
3. Visual Complexity: Clean, Moderate, High Distractors (with strict ray-clearance enforcement)
4. Reasoning Complexity: Task A Visibility, Task B Obstruction, Task C Repair, Task D Counterfactual
5. Counterfactual Sensitivity: Invariance vs Critical Flip

Logs full experimental manifests to experiments/ with rigorous rejection accounting and rendering validation.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from mirrorlab.evaluation.ambiguity_validator import AmbiguityValidator
from mirrorlab.evaluation.rendering_validator import RenderingConsistencyValidator
from mirrorlab.experiment.difficulty_calibrator import (
    DifficultyCalibrator,
    DifficultyMetrics,
    DifficultyTier,
)
from mirrorlab.geometry.oracle import VisibilityOracle, evaluate_visibility_oracle
from mirrorlab.rendering.pillow_renderer import PillowDiagramRenderer
from mirrorlab.rendering.themes import RenderMode, RenderTheme
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
    generator_version: str = "mirrorlab-v1.3.0"
    random_seed: int
    condition: TaskCondition
    total_attempted: int
    total_valid: int
    total_rejected: int
    total_rejected_ambiguous: int
    total_rejected_rendering: int
    total_rejected_geometry: int
    total_rejected_distractor: int
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
        self._renderer = PillowDiagramRenderer(
            theme=RenderTheme.DARK_NAVY, image_width=512, image_height=512
        )

    def run_condition_experiment(
        self,
        condition: TaskCondition,
        base_seed: int = 42,
    ) -> ExperimentManifest:
        """
        Executes generation across a parameterized condition, tracking rejections,
        rendering validation, and difficulty calibration.
        """
        t0 = time.perf_counter()
        results: list[TaskGenerationResult] = []
        rejected_ambiguous = 0
        rejected_rendering = 0
        rejected_geometry = 0
        rejected_distractor = 0
        rejection_counts: dict[str, int] = {}
        difficulty_dist: dict[str, int] = {t.value: 0 for t in DifficultyTier}

        for i in range(condition.sample_size):
            seed = base_seed + condition.seed_offset + i
            tid = f"exp_{condition.condition_id}_{seed}"

            # 1. Candidate generation
            scene, rejection = self._generate_candidate_scene(condition, seed)
            if rejection:
                if "MULTI_HOP" in rejection or "GEOMETRY" in rejection:
                    rejected_geometry += 1
                elif "DISTRACTOR" in rejection:
                    rejected_distractor += 1
                else:
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

            # 2. Comprehensive Ambiguity Validation (Layer 3)
            ambiguity_report = AmbiguityValidator.validate_scene(scene)
            if not ambiguity_report.is_unambiguous:
                rejected_ambiguous += 1
                primary_rej = ambiguity_report.rejection_reasons[0]
                rejection_counts[primary_rej] = rejection_counts.get(primary_rej, 0) + 1
                results.append(
                    TaskGenerationResult(
                        task_id=tid,
                        condition_id=condition.condition_id,
                        is_valid=False,
                        rejection_reason=primary_rej,
                    )
                )
                continue

            # 3. Real Rendering Validation (Layer 4)
            rendered_img = self._renderer.render(scene, mode=RenderMode.BLIND)
            rendering_report = RenderingConsistencyValidator.validate_image(
                scene=scene,
                pil_image=rendered_img,
                image_name=f"{tid}_blind.png",
                is_blind_image=True,
            )
            if not rendering_report.all_passed:
                rejected_rendering += 1
                failed_probes = [
                    r.element_name
                    for r in rendering_report.element_results
                    if not r.passed
                ]
                rej_key = f"REJECTED_RENDERING_PROBE_{'_'.join(failed_probes)}"
                rejection_counts[rej_key] = rejection_counts.get(rej_key, 0) + 1
                results.append(
                    TaskGenerationResult(
                        task_id=tid,
                        condition_id=condition.condition_id,
                        is_valid=False,
                        rejection_reason=rej_key,
                    )
                )
                continue

            # 4. Calculate quantitative difficulty
            verdict = VisibilityOracle.evaluate(scene)
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
        total_rejected = sum(1 for r in results if not r.is_valid)

        # Enforce exact accounting invariant
        assert condition.sample_size == valid_count + total_rejected

        manifest = ExperimentManifest(
            experiment_id=f"exp_{condition.condition_id}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
            random_seed=base_seed,
            condition=condition,
            total_attempted=condition.sample_size,
            total_valid=valid_count,
            total_rejected=total_rejected,
            total_rejected_ambiguous=rejected_ambiguous,
            total_rejected_rendering=rejected_rendering,
            total_rejected_geometry=rejected_geometry,
            total_rejected_distractor=rejected_distractor,
            difficulty_distribution=difficulty_dist,
            runtime_seconds=round(runtime, 3),
            rejection_summary=rejection_counts,
            tasks=results,
            conclusion=(
                f"Generated {valid_count}/{condition.sample_size} valid tasks. "
                f"Yield: {round(valid_count / condition.sample_size * 100, 1)}%. "
                f"Rejections: {total_rejected} (Ambiguity: {rejected_ambiguous}, "
                f"Rendering: {rejected_rendering}, Geometry: {rejected_geometry}, Distractor: {rejected_distractor})."
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

        # Explicit handling of all 5 Geometry regimes
        if condition.geometry == GeometryRegime.DUAL_MIRROR_FOLDED:
            # Multi-hop reflection is unsupported by single-mirror InspectionScene schema.
            # As required by scientific integrity rules: do not fake multi-mirror reflections.
            return None, "REJECTED_UNSUPPORTED_MULTI_HOP_GEOMETRY"

        elif condition.geometry == GeometryRegime.GRAZING_INCIDENCE:
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

        elif condition.geometry == GeometryRegime.OFF_AXIS_EXTREME:
            m_a = Point2D(x=200.0, y=750.0)
            m_b = Point2D(x=700.0, y=750.0)
            m_norm = Vector2D(dx=0.0, dy=-1.0)
            cam_pos = Point2D(x=100.0, y=150.0)
            target_pos = Point2D(x=600.0, y=200.0)
            cam_heading = 45.0  # Off-axis heading close to boundary of FOV

        else:  # SINGLE_PLANAR
            m_a = Point2D(x=150.0 + rng.uniform(-20, 20), y=700.0)
            m_b = Point2D(x=650.0 + rng.uniform(-20, 20), y=700.0)
            m_norm = Vector2D(dx=0.0, dy=-1.0)
            cam_pos = Point2D(x=200.0 + rng.uniform(-15, 15), y=200.0)
            target_pos = Point2D(x=500.0 + rng.uniform(-15, 15), y=250.0)
            cam_heading = 65.0

        mirror = MirrorSpec(endpoint_a=m_a, endpoint_b=m_b, reflective_normal=m_norm)
        camera = CameraSpec(position=cam_pos, heading_deg=cam_heading, fov_degrees=65.0)
        target = TargetSpec(center=target_pos, radius_mm=8.0)

        # Build obstacles based on OcclusionRegime
        obstacles: list[ObstacleSpec] = []

        if condition.occlusion == OcclusionRegime.NONE:
            # Unobstructed direct and reflected path
            pass

        elif condition.occlusion == OcclusionRegime.CORRIDOR_PARTITION:
            # Base barrier separating camera and target direct sightline
            obstacles.append(
                ObstacleSpec(
                    obstacle_id="O1",
                    obstacle_type="rectangle",
                    center=Point2D(
                        x=(cam_pos.x + target_pos.x) / 2,
                        y=(cam_pos.y + target_pos.y) / 2,
                    ),
                    width=60.0,
                    height=180.0,
                    angle_degrees=0.0,
                )
            )

        elif condition.occlusion == OcclusionRegime.NEAR_MIRROR_BLOCKER:
            obstacles.append(
                ObstacleSpec(
                    obstacle_id="O1_partition",
                    obstacle_type="rectangle",
                    center=Point2D(
                        x=(cam_pos.x + target_pos.x) / 2,
                        y=(cam_pos.y + target_pos.y) / 2,
                    ),
                    width=50.0,
                    height=150.0,
                    angle_degrees=0.0,
                )
            )
            obstacles.append(
                ObstacleSpec(
                    obstacle_id="O2_near_mirror",
                    obstacle_type="circle",
                    center=Point2D(x=350.0, y=620.0),
                    radius=20.0,
                )
            )

        elif condition.occlusion == OcclusionRegime.NEAR_TARGET_BLOCKER:
            obstacles.append(
                ObstacleSpec(
                    obstacle_id="O1_partition",
                    obstacle_type="rectangle",
                    center=Point2D(
                        x=(cam_pos.x + target_pos.x) / 2,
                        y=(cam_pos.y + target_pos.y) / 2,
                    ),
                    width=50.0,
                    height=150.0,
                    angle_degrees=0.0,
                )
            )
            obstacles.append(
                ObstacleSpec(
                    obstacle_id="O2_near_target",
                    obstacle_type="circle",
                    center=Point2D(x=target_pos.x - 35.0, y=target_pos.y + 40.0),
                    radius=18.0,
                )
            )

        elif condition.occlusion == OcclusionRegime.MULTIPLE_OBSTACLES:
            obstacles.append(
                ObstacleSpec(
                    obstacle_id="O1_partition",
                    obstacle_type="rectangle",
                    center=Point2D(
                        x=(cam_pos.x + target_pos.x) / 2,
                        y=(cam_pos.y + target_pos.y) / 2,
                    ),
                    width=50.0,
                    height=150.0,
                    angle_degrees=0.0,
                )
            )
            obstacles.append(
                ObstacleSpec(
                    obstacle_id="O2_side",
                    obstacle_type="rectangle",
                    center=Point2D(x=280.0, y=450.0),
                    width=40.0,
                    height=80.0,
                    angle_degrees=15.0,
                )
            )
            obstacles.append(
                ObstacleSpec(
                    obstacle_id="O3_baffle",
                    obstacle_type="circle",
                    center=Point2D(x=450.0, y=480.0),
                    radius=22.0,
                )
            )

        elif condition.occlusion == OcclusionRegime.GRAZING_CLEARANCE:
            # Place obstacle right near the corridor with tight 4.0 mm clearance
            obstacles.append(
                ObstacleSpec(
                    obstacle_id="O1_graze",
                    obstacle_type="circle",
                    center=Point2D(x=260.0, y=410.0),
                    radius=15.0,
                )
            )

        # Baseline oracle evaluation to identify optical corridor before placing distractors
        base_verdict = evaluate_visibility_oracle(
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

        # Distractor placement with strict ray-corridor clearance
        if condition.clutter in (
            ClutterRegime.MODERATE,
            ClutterRegime.HIGH_DISTRACTORS,
        ):
            num_distractors = 2 if condition.clutter == ClutterRegime.MODERATE else 4

            # Extract reflection point if it exists
            r_coords = base_verdict.reflection_point
            c_pt = np.array([camera.position.x, camera.position.y])
            t_pt = np.array([target.center.x, target.center.y])

            for d_idx in range(num_distractors):
                placed = False
                d_radius = 12.0

                for _attempt in range(25):
                    dx = rng.uniform(80.0, 820.0)
                    dy = rng.uniform(80.0, 600.0)
                    distractor_pt = np.array([dx, dy])

                    # 1. Check distance to camera, target, mirror endpoints
                    if np.linalg.norm(distractor_pt - c_pt) < (d_radius + 30.0):
                        continue
                    if np.linalg.norm(distractor_pt - t_pt) < (d_radius + 30.0):
                        continue
                    if np.linalg.norm(
                        distractor_pt
                        - np.array([mirror.endpoint_a.x, mirror.endpoint_a.y])
                    ) < (d_radius + 20.0):
                        continue
                    if np.linalg.norm(
                        distractor_pt
                        - np.array([mirror.endpoint_b.x, mirror.endpoint_b.y])
                    ) < (d_radius + 20.0):
                        continue

                    # 2. Check collision with existing obstacles
                    collision = False
                    for existing_obs in obstacles:
                        e_pt = np.array([existing_obs.center.x, existing_obs.center.y])
                        if np.linalg.norm(distractor_pt - e_pt) < (d_radius + 25.0):
                            collision = True
                            break
                    if collision:
                        continue

                    # 3. Ray corridor clearance: ensure distractor does NOT intersect optical path
                    if r_coords is not None:
                        r_pt = np.array([r_coords[0], r_coords[1]])
                        # Perpendicular distance to Leg 1 (C -> R)
                        d_leg1 = self._point_to_segment_distance(
                            distractor_pt, c_pt, r_pt
                        )
                        # Perpendicular distance to Leg 2 (R -> T)
                        d_leg2 = self._point_to_segment_distance(
                            distractor_pt, r_pt, t_pt
                        )

                        # Must maintain at least distractor_radius + 15 mm corridor clearance
                        if d_leg1 < (d_radius + 15.0) or d_leg2 < (d_radius + 15.0):
                            continue

                    # Placed successfully
                    obstacles.append(
                        ObstacleSpec(
                            obstacle_id=f"D{d_idx + 1}",
                            obstacle_type="circle",
                            center=Point2D(x=dx, y=dy),
                            radius=d_radius,
                        )
                    )
                    placed = True
                    break

                if not placed:
                    return None, "REJECTED_DISTRACTOR_INTERFERENCE"

        # Evaluate final oracle with all obstacles and distractors
        final_verdict = evaluate_visibility_oracle(
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
            if final_verdict.state == VisibilityState.VISIBLE
            else SceneCategory.BLOCKED_INCIDENT
        )

        ref_pt = (
            Point2D(
                x=float(final_verdict.reflection_point[0]),
                y=float(final_verdict.reflection_point[1]),
            )
            if final_verdict.reflection_point is not None
            else None
        )
        vt_pt = (
            Point2D(
                x=float(final_verdict.virtual_target[0]),
                y=float(final_verdict.virtual_target[1]),
            )
            if final_verdict.virtual_target is not None
            else None
        )
        opt_path = [
            Point2D(x=float(p[0]), y=float(p[1])) for p in final_verdict.path_points
        ]

        scene = InspectionScene(
            scene_id=f"scene_{seed}",
            seed=seed,
            category=category,
            difficulty=DifficultyLevel.MEDIUM,
            camera=camera,
            mirror=mirror,
            target=target,
            obstacles=obstacles,
            ground_truth_state=final_verdict.state,
            ground_truth_reason=final_verdict.reason,
            diagnostic_details=final_verdict.diagnostic_details,
            reflection_point=ref_pt,
            virtual_target=vt_pt,
            optical_path=opt_path,
            incident_angle_deg=final_verdict.incident_angle_deg,
            reflected_angle_deg=final_verdict.reflected_angle_deg,
            total_optical_distance=final_verdict.total_optical_distance,
            is_direct_sight_blocked=final_verdict.is_direct_sight_blocked,
        )
        return scene, None

    @staticmethod
    def _point_to_segment_distance(
        pt: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray
    ) -> float:
        """Computes minimum distance from a point to a finite line segment."""
        ab = seg_b - seg_a
        ab_len2 = float(np.dot(ab, ab))
        if ab_len2 < 1e-8:
            return float(np.linalg.norm(pt - seg_a))
        t = max(0.0, min(1.0, float(np.dot(pt - seg_a, ab)) / ab_len2))
        proj = seg_a + t * ab
        return float(np.linalg.norm(pt - proj))
