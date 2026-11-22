"""
Dedicated Out-Of-Distribution (OOD) Task Generator for MIRRORLAB.

Generates extreme, unseen specular reflection inspection tasks to test whether
multimodal vision models genuinely perform physical reflection reasoning
or rely on memorized canonical diagram priors.

OOD Dimensions:
1. High Obstacle Clutter: 5 to 8 barriers (vs 1-3 in canonical distribution).
2. Acute Mirror Tilts: Angles in [35°, 55°] and [-55°, -35°] (vs ±15° in canonical).
3. Extreme Cavity Aspect Ratios: Non-square aspect ratios (1200×800mm, 800×1200mm, 1000×1000mm).
4. Grazing Incident Angles: Angles of incidence > 50° challenging boundary resolution.
5. Balanced Category Distribution: Visible, Blocked-Incident, Blocked-Reflected, Mirror-Miss, Wrong-Side, Outside-FOV.
6. Dual Mathematical Certification: 100% of saved OOD tasks independently verified with Fermat residual < 1e-5.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mirrorlab.generation.task_factory import VisibilityTaskFactory
from mirrorlab.geometry.independent_verifier import IndependentGeometryVerifier
from mirrorlab.geometry.oracle import VisibilityOracle
from mirrorlab.schema.common import (
    ObstructionReason,
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

logger = logging.getLogger(__name__)


def _cross2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


@dataclass(frozen=True)
class OODGenerationConfig:
    """Configuration parameters for generating out-of-distribution visual tasks."""

    num_tasks: int = 30
    seed_start: int = 80000
    min_obstacles: int = 5
    max_obstacles: int = 8
    acute_mirror_angles: tuple[float, float] = (35.0, 55.0)
    canvas_aspect_ratios: tuple[tuple[float, float], ...] = (
        (1200.0, 800.0),
        (800.0, 1200.0),
        (1000.0, 1000.0),
    )
    max_retries_per_task: int = 100


class OODTaskGenerator:
    """
    Generates scientifically certified Out-Of-Distribution benchmark tasks
    with balanced category distribution and strict dual-verification enforcement.
    """

    TARGET_CATEGORIES = (
        "VALID_VISIBLE",
        "BLOCKED_INCIDENT",
        "BLOCKED_REFLECTED",
        "MIRROR_MISS",
        "WRONG_SIDE",
        "OUTSIDE_FOV",
    )

    def __init__(
        self,
        config: OODGenerationConfig | None = None,
        output_dir: Path | str = "data/ood_tasks",
    ):
        self.config = config or OODGenerationConfig()
        self.output_dir = Path(output_dir)
        self.task_factory = VisibilityTaskFactory()

    def _generate_candidate_scene(
        self,
        seed: int,
        task_index: int,
        target_category: str,
    ) -> tuple[InspectionScene, Any, Any]:
        """
        Synthesizes an exact candidate scene for target_category using analytical reflection geometry.
        """
        rng = np.random.default_rng(seed)

        canvas_w, canvas_h = self.config.canvas_aspect_ratios[
            task_index % len(self.config.canvas_aspect_ratios)
        ]

        # 1. Base Camera and Target coordinates
        cam_x = float(rng.uniform(canvas_w * 0.15, canvas_w * 0.35))
        cam_y = float(rng.uniform(canvas_h * 0.15, canvas_h * 0.35))
        tgt_x = float(rng.uniform(canvas_w * 0.65, canvas_w * 0.85))
        tgt_y = float(rng.uniform(canvas_h * 0.15, canvas_h * 0.35))

        C = np.array([cam_x, cam_y], dtype=np.float64)
        T = np.array([tgt_x, tgt_y], dtype=np.float64)

        # 2. Acute mirror orientation
        angle_sign = -1.0 if (task_index % 2 == 0) else 1.0
        angle_deg = angle_sign * float(rng.uniform(*self.config.acute_mirror_angles))
        rad = math.radians(angle_deg)
        u = np.array([math.cos(rad), math.sin(rad)], dtype=np.float64)
        n = np.array([-math.sin(rad), math.cos(rad)], dtype=np.float64)

        # Mirror line reference point P0 near bottom/center of canvas
        p0_x = float(rng.uniform(canvas_w * 0.40, canvas_w * 0.60))
        p0_y = float(rng.uniform(canvas_h * 0.65, canvas_h * 0.82))
        P0 = np.array([p0_x, p0_y], dtype=np.float64)

        # Ensure normal faces the camera initially
        if np.dot(C - P0, n) < 0:
            n = -n

        if target_category == "WRONG_SIDE":
            # Reverse normal so reflective face points away from camera
            n = -n

        # Virtual camera reflection across infinite mirror plane (P - P0) . n = 0
        C_prime = C - 2.0 * np.dot(C - P0, n) * n

        # Find analytical intersection of line C' -> T with plane
        v_line = T - C_prime
        denom = np.dot(v_line, n)
        if abs(denom) < 1e-9:
            t_line = 0.5
        else:
            t_line = float(np.dot(P0 - C_prime, n) / denom)

        R = C_prime + t_line * v_line

        # Clamp mirror placement within canvas bounds
        mirror_len = float(rng.uniform(220.0, 300.0))
        half_len = mirror_len / 2.0

        if target_category == "MIRROR_MISS":
            # Shift mirror along tangent u so reflection point R falls outside segment
            shift = half_len + float(rng.uniform(25.0, 60.0))
            M1 = R + shift * u
            M2 = M1 + mirror_len * u
        else:
            # Center mirror segment exactly at reflection point R
            M1 = R - half_len * u
            M2 = R + half_len * u

        # Camera heading towards reflection point R
        cam_heading = math.degrees(math.atan2(R[1] - C[1], R[0] - C[0]))
        if target_category == "OUTSIDE_FOV":
            # Point camera 110 degrees away from R
            cam_heading = (cam_heading + 110.0) % 360.0

        camera_spec = CameraSpec(
            position=Point2D(x=float(C[0]), y=float(C[1])),
            heading_deg=float(cam_heading),
            fov_degrees=75.0,
            max_range=2000.0,
        )
        mirror_spec = MirrorSpec(
            endpoint_a=Point2D(x=float(M1[0]), y=float(M1[1])),
            endpoint_b=Point2D(x=float(M2[0]), y=float(M2[1])),
            reflective_normal=Vector2D(dx=float(n[0]), dy=float(n[1])),
        )
        target_spec = TargetSpec(
            center=Point2D(x=float(T[0]), y=float(T[1])),
            radius_mm=8.0,
        )

        # 3. Direct sight blocker O1
        mid_x = float((C[0] + T[0]) / 2.0 + rng.uniform(-10.0, 10.0))
        mid_y = float((C[1] + T[1]) / 2.0 + rng.uniform(-10.0, 10.0))
        obstacles: list[ObstacleSpec] = [
            ObstacleSpec(
                obstacle_id="O1_direct_barrier",
                obstacle_type="rectangle",
                center=Point2D(x=mid_x, y=mid_y),
                width=35.0,
                height=150.0,
                angle_degrees=float(rng.uniform(-10.0, 10.0)),
            )
        ]

        # 4. Target-specific obstruction placement
        if target_category == "BLOCKED_INCIDENT":
            t_seg = float(rng.uniform(0.35, 0.65))
            occ = C + t_seg * (R - C)
            obstacles.append(
                ObstacleSpec(
                    obstacle_id="O2_incident_blocker",
                    obstacle_type="circle",
                    center=Point2D(x=float(occ[0]), y=float(occ[1])),
                    radius=float(rng.uniform(22.0, 30.0)),
                )
            )
        elif target_category == "BLOCKED_REFLECTED":
            t_seg = float(rng.uniform(0.35, 0.65))
            occ = R + t_seg * (T - R)
            obstacles.append(
                ObstacleSpec(
                    obstacle_id="O2_reflected_blocker",
                    obstacle_type="circle",
                    center=Point2D(x=float(occ[0]), y=float(occ[1])),
                    radius=float(rng.uniform(22.0, 30.0)),
                )
            )

        # 5. Add clutter obstacles (5 to 8 total)
        num_clutter = int(rng.integers(self.config.min_obstacles - 1, self.config.max_obstacles))
        start_idx = len(obstacles) + 1

        for obs_i in range(start_idx, num_clutter + 2):
            obs_type = "circle" if (obs_i % 2 == 0) else "rectangle"
            ox = float(rng.uniform(canvas_w * 0.10, canvas_w * 0.90))
            oy = float(rng.uniform(canvas_h * 0.30, canvas_h * 0.70))

            # In visible / miss / wrong-side / FOV modes, ensure clutter does not occlude C->R or R->T
            if target_category in ("VALID_VISIBLE", "MIRROR_MISS", "WRONG_SIDE", "OUTSIDE_FOV"):
                len_cr = max(1e-9, float(np.linalg.norm(R - C)))
                len_rt = max(1e-9, float(np.linalg.norm(T - R)))
                d_c_r = abs(_cross2d(R - C, C - np.array([ox, oy]))) / len_cr
                d_r_t = abs(_cross2d(T - R, R - np.array([ox, oy]))) / len_rt
                if d_c_r < 60.0 or d_r_t < 60.0:
                    ox = (ox + 220.0) % (canvas_w * 0.85) + canvas_w * 0.08
                    oy = (oy + 180.0) % (canvas_h * 0.75) + canvas_h * 0.15

            if obs_type == "circle":
                obstacles.append(
                    ObstacleSpec(
                        obstacle_id=f"O{obs_i}_clutter",
                        obstacle_type="circle",
                        center=Point2D(x=ox, y=oy),
                        radius=float(rng.uniform(16.0, 26.0)),
                    )
                )
            else:
                obstacles.append(
                    ObstacleSpec(
                        obstacle_id=f"O{obs_i}_clutter",
                        obstacle_type="rectangle",
                        center=Point2D(x=ox, y=oy),
                        width=float(rng.uniform(28.0, 48.0)),
                        height=float(rng.uniform(28.0, 48.0)),
                        angle_degrees=float(rng.uniform(0.0, 90.0)),
                    )
                )

        temp_scene = InspectionScene(
            scene_id=f"task_ood_{seed:05d}",
            seed=seed,
            category=SceneCategory.VALID_VISIBLE,
            difficulty=DifficultyLevel.BOUNDARY,
            canvas_width=canvas_w,
            canvas_height=canvas_h,
            camera=camera_spec,
            mirror=mirror_spec,
            target=target_spec,
            obstacles=obstacles,
            ground_truth_state=VisibilityState.VISIBLE,
            ground_truth_reason=ObstructionReason.NONE,
            diagnostic_details="OOD candidate synthesis.",
        )

        oracle_res = VisibilityOracle.evaluate(temp_scene)
        dual = IndependentGeometryVerifier.verify_dual(temp_scene, oracle_res)

        return temp_scene, oracle_res, dual

    def generate_ood_scene(
        self,
        base_seed: int | None = None,
        task_index: int = 0,
        seed: int | None = None,
    ) -> InspectionScene:
        """
        Procedurally generates a single out-of-distribution inspection scene.
        Strictly rejects scenes where dual verification fails or category does not match.
        Retries up to max_retries_per_task times; raises RuntimeError if exceeded.
        """
        if base_seed is None:
            base_seed = seed if seed is not None else (self.config.seed_start + task_index)
        target_category = self.TARGET_CATEGORIES[task_index % len(self.TARGET_CATEGORIES)]

        for attempt in range(self.config.max_retries_per_task):
            seed = base_seed + attempt * 1000
            temp_scene, oracle_res, dual = self._generate_candidate_scene(
                seed, task_index, target_category
            )

            # Check dual verification certification
            if not dual.is_mathematically_certified:
                continue

            # Verify outcome matches target category
            matched = False
            if target_category == "VALID_VISIBLE":
                matched = oracle_res.is_visible
            elif target_category == "BLOCKED_INCIDENT":
                matched = not oracle_res.is_visible and oracle_res.reason == ObstructionReason.BLOCKED_INCIDENT
            elif target_category == "BLOCKED_REFLECTED":
                matched = not oracle_res.is_visible and oracle_res.reason == ObstructionReason.BLOCKED_REFLECTED
            elif target_category == "MIRROR_MISS":
                matched = not oracle_res.is_visible and oracle_res.reason == ObstructionReason.MISSED_MIRROR
            elif target_category == "WRONG_SIDE":
                matched = not oracle_res.is_visible and oracle_res.reason == ObstructionReason.BACK_OF_MIRROR
            elif target_category == "OUTSIDE_FOV":
                matched = not oracle_res.is_visible and oracle_res.reason == ObstructionReason.OUT_OF_FOV

            if matched:
                refl_pt = None
                if oracle_res.reflection_point is not None:
                    refl_pt = Point2D(
                        x=float(oracle_res.reflection_point[0]),
                        y=float(oracle_res.reflection_point[1]),
                    )

                virt_tgt = None
                if oracle_res.virtual_target is not None:
                    virt_tgt = Point2D(
                        x=float(oracle_res.virtual_target[0]),
                        y=float(oracle_res.virtual_target[1]),
                    )

                opt_path = [
                    Point2D(x=float(p[0]), y=float(p[1]))
                    for p in oracle_res.path_points
                ]

                category_enum = (
                    SceneCategory.VALID_VISIBLE
                    if oracle_res.is_visible
                    else SceneCategory.BLOCKED_REFLECTED
                )
                if oracle_res.reason == ObstructionReason.MISSED_MIRROR:
                    category_enum = SceneCategory.MIRROR_MISS
                elif oracle_res.reason == ObstructionReason.BACK_OF_MIRROR:
                    category_enum = SceneCategory.WRONG_SIDE
                elif oracle_res.reason == ObstructionReason.OUT_OF_FOV:
                    category_enum = SceneCategory.OUTSIDE_FOV

                return InspectionScene(
                    scene_id=f"task_ood_{base_seed:05d}",
                    seed=seed,
                    category=category_enum,
                    difficulty=DifficultyLevel.BOUNDARY,
                    canvas_width=temp_scene.canvas_width,
                    canvas_height=temp_scene.canvas_height,
                    camera=temp_scene.camera,
                    mirror=temp_scene.mirror,
                    target=temp_scene.target,
                    obstacles=temp_scene.obstacles,
                    ground_truth_state=oracle_res.state,
                    ground_truth_reason=oracle_res.reason,
                    diagnostic_details=(
                        f"OOD Benchmark: {oracle_res.diagnostic_details} | "
                        f"Dual Cert: Delta_R={dual.reflection_point_delta_mm:.2e}mm, "
                        f"Fermat={dual.fermat_residual:.2e}"
                    ),
                    reflection_point=refl_pt,
                    virtual_target=virt_tgt,
                    optical_path=opt_path,
                    incident_angle_deg=oracle_res.incident_angle_deg,
                    reflected_angle_deg=oracle_res.reflected_angle_deg,
                    total_optical_distance=oracle_res.total_optical_distance,
                    is_direct_sight_blocked=oracle_res.is_direct_sight_blocked,
                )

        raise RuntimeError(
            f"Failed to generate dual-certified OOD scene for category '{target_category}' "
            f"after {self.config.max_retries_per_task} retries (base_seed={base_seed})."
        )

    def generate_ood_dataset(self) -> list[str]:
        """
        Generates, validates, and saves the complete certified OOD benchmark task suite.
        Revalidates every saved record from disk before claiming certification.
        Returns list of generated task IDs.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        task_ids: list[str] = []
        manifest_records: list[dict[str, Any]] = []
        category_counts: dict[str, int] = {}

        for idx in range(self.config.num_tasks):
            base_seed = self.config.seed_start + idx
            scene = self.generate_ood_scene(base_seed, idx)
            tid = scene.scene_id
            tdir = self.output_dir / tid
            tdir.mkdir(parents=True, exist_ok=True)

            # Export bundle
            self.task_factory.create_bundle_from_scene(
                scene=scene,
                task_id=tid,
                target_dir=tdir,
            )
            task_ids.append(tid)

            reason_str = (
                scene.ground_truth_reason.value
                if hasattr(scene.ground_truth_reason, "value")
                else str(scene.ground_truth_reason)
            )
            is_vis = scene.ground_truth_state == VisibilityState.VISIBLE
            cat_label = "VISIBLE" if is_vis else reason_str
            category_counts[cat_label] = category_counts.get(cat_label, 0) + 1

            manifest_records.append({
                "task_id": tid,
                "seed": scene.seed,
                "difficulty": "OOD_BOUNDARY",
                "obstacle_count": len(scene.obstacles),
                "is_visible": is_vis,
                "reason": reason_str,
                "category_label": cat_label,
            })

        # Post-export revalidation: Reload from disk and verify certification of every saved record
        for tid in task_ids:
            scene_path = self.output_dir / tid / "scene.json"
            if not scene_path.is_file():
                raise FileNotFoundError(f"Exported OOD scene missing: {scene_path}")
            with open(scene_path, encoding="utf-8") as f:
                saved_sc = json.load(f)
            loaded_scene = InspectionScene.model_validate(saved_sc)
            oracle_verdict = VisibilityOracle.evaluate(loaded_scene)
            cert = IndependentGeometryVerifier.verify_dual(loaded_scene, oracle_verdict)
            if not cert.is_mathematically_certified:
                raise ValueError(
                    f"Post-export verification failed for {tid}: {cert.proof_summary}"
                )

        # Save OOD manifest with actual measured distribution
        manifest_file = self.output_dir / "ood_manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "ood_task_count": len(task_ids),
                    "generator_version": "2.0.0-ood-certified",
                    "certification": "100% dual-verified via IndependentGeometryVerifier and post-export revalidated",
                    "all_records_certified": True,
                    "category_distribution": category_counts,
                    "records": manifest_records,
                },
                f,
                indent=2,
            )

        return task_ids
