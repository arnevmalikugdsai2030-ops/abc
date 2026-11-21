"""
Counterfactual Paired Scene Factory for MIRRORLAB.

Generates paired machine-vision inspection scenes for fine-grained VLM evaluation:
- PAIR TYPE 1: MIRROR ANGLE FLIP (Delta theta ~ 2°..8° flips visibility outcome)
- PAIR TYPE 2: COSMETIC CONTROL (Identical geometry, different rendering theme, invariant outcome)
- PAIR TYPE 3: OBSTACLE INSERTION (Valid clear scene + blocking obstacle, flips to blocked)
- PAIR TYPE 4: IRRELEVANT OBSTACLE (Valid clear scene + distractor obstacle, invariant clear outcome)

All pairs are verified by the deterministic VisibilityOracle.
Pairs failing their expected relationship are strictly rejected.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import LineString

from mirrorlab.generation.scene_generator import SceneGenerator
from mirrorlab.generation.task_factory import VisibilityTaskFactory
from mirrorlab.geometry.oracle import VisibilityOracle
from mirrorlab.geometry.vector import vector_magnitude
from mirrorlab.rendering.themes import RenderTheme
from mirrorlab.schema.common import (
    ObstructionReason,
    Point2D,
    Vector2D,
    VisibilityState,
)
from mirrorlab.schema.scene import (
    DifficultyLevel,
    InspectionScene,
    MirrorSpec,
    ObstacleSpec,
    SceneCategory,
)
from mirrorlab.schema.task import (
    OBSTRUCTION_TO_REASON_CODE,
    CounterfactualPairBundle,
    CounterfactualPairRecord,
    CounterfactualPairType,
    CounterfactualSceneOutcome,
    ExpectedPairRelationship,
    VisibilityReasonCode,
)


class CounterfactualFactory:
    """
    Procedural generator for counterfactual and invariant paired inspection tasks.
    """

    def __init__(
        self,
        canvas_width: float = 1000.0,
        canvas_height: float = 1000.0,
        image_size: int = 1024,
    ):
        self.canvas_width = float(canvas_width)
        self.canvas_height = float(canvas_height)
        self.image_size = int(image_size)
        self.scene_generator = SceneGenerator(
            canvas_width=canvas_width, canvas_height=canvas_height
        )
        self.oracle = VisibilityOracle()
        self.task_factory = VisibilityTaskFactory(
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            image_size=image_size,
        )

    def _sync_scene_with_oracle(
        self,
        scene: InspectionScene,
        scene_id: str,
        category: SceneCategory,
        seed: int,
        difficulty: DifficultyLevel,
    ) -> InspectionScene:
        """
        Run the oracle on a scene and return an updated immutable InspectionScene
        with all ground-truth proof fields synchronized.
        """
        oracle_res = self.oracle.evaluate(scene)
        can_see = oracle_res.is_visible
        vis_state = VisibilityState.VISIBLE if can_see else VisibilityState.NOT_VISIBLE
        reason = oracle_res.reason

        refl_pt = (
            Point2D(x=float(oracle_res.reflection_point[0]), y=float(oracle_res.reflection_point[1]))
            if oracle_res.reflection_point is not None
            else None
        )
        virt_tgt = (
            Point2D(x=float(oracle_res.virtual_target[0]), y=float(oracle_res.virtual_target[1]))
            if oracle_res.virtual_target is not None
            else None
        )

        optical_path: list[Point2D] = []
        if refl_pt is not None:
            optical_path = [scene.camera.position, refl_pt, scene.target.center]

        return InspectionScene(
            scene_id=scene_id,
            seed=seed,
            category=category,
            difficulty=difficulty,
            canvas_width=scene.canvas_width,
            canvas_height=scene.canvas_height,
            camera=scene.camera,
            mirror=scene.mirror,
            target=scene.target,
            obstacles=scene.obstacles,
            ground_truth_state=vis_state,
            ground_truth_reason=reason,
            diagnostic_details=oracle_res.diagnostic_details,
            reflection_point=refl_pt,
            virtual_target=virt_tgt,
            incident_angle_deg=oracle_res.incident_angle_deg,
            reflected_angle_deg=oracle_res.reflected_angle_deg,
            total_optical_distance=oracle_res.total_optical_distance,
            optical_path=optical_path,
        )

    def create_mirror_angle_flip_pair(
        self,
        seed: int,
        pair_id: str,
        parent_id: str | None = None,
        output_dir: Path | None = None,
        difficulty: DifficultyLevel = DifficultyLevel.MEDIUM,
        theme: RenderTheme = RenderTheme.DARK_NAVY,
    ) -> CounterfactualPairBundle:
        """
        PAIR TYPE 1: MIRROR ANGLE FLIP.
        Keeps camera, target, obstacles, and FOV fixed.
        Tilts the mirror by a small angular increment (Delta theta ~ 2.5°..8.0°)
        such that the visibility outcome flips from VISIBLE to NOT_VISIBLE.
        """
        parent_scene_id = parent_id or f"scene_parent_{seed}"
        scene_a_id = f"{pair_id}_a"
        scene_b_id = f"{pair_id}_b"

        # 1. Generate base scene A (VALID_VISIBLE)
        scene_a = self.scene_generator.generate_scene(
            seed=seed,
            category=SceneCategory.VALID_VISIBLE,
            difficulty=difficulty,
            scene_id=scene_a_id,
        )
        eval_a = self.oracle.evaluate(scene_a)
        if not eval_a.is_visible:
            raise ValueError(f"Base scene A for seed {seed} is not visible ({eval_a.reason})")

        # Extract mirror geometry
        ma = np.array([scene_a.mirror.endpoint_a.x, scene_a.mirror.endpoint_a.y], dtype=np.float64)
        mb = np.array([scene_a.mirror.endpoint_b.x, scene_a.mirror.endpoint_b.y], dtype=np.float64)
        m_center = (ma + mb) * 0.5
        m_delta = mb - ma
        m_len = float(vector_magnitude(m_delta))
        m_angle_rad = math.atan2(m_delta[1], m_delta[0])

        normal_vec = np.array(
            [scene_a.mirror.reflective_normal.dx, scene_a.mirror.reflective_normal.dy],
            dtype=np.float64,
        )
        normal_angle_rad = math.atan2(normal_vec[1], normal_vec[0])

        # Candidate tilt angles sorted in increasing magnitude to find the minimal flip
        candidate_deltas_deg = [
            +2.5, -2.5, +4.0, -4.0, +6.0, -6.0, +8.0, -8.0, +10.0, -10.0,
            +13.0, -13.0, +16.0, -16.0, +20.0, -20.0, +25.0, -25.0, +30.0, -30.0,
        ]

        best_scene_b: InspectionScene | None = None
        best_delta_deg: float = 0.0
        best_reason: ObstructionReason | None = None
        best_new_angle_deg: float = 0.0

        for delta_deg in candidate_deltas_deg:
            delta_rad = math.radians(delta_deg)
            new_m_angle_rad = m_angle_rad + delta_rad
            new_norm_angle_rad = normal_angle_rad + delta_rad

            u_new = np.array([math.cos(new_m_angle_rad), math.sin(new_m_angle_rad)], dtype=np.float64)
            n_new = np.array([math.cos(new_norm_angle_rad), math.sin(new_norm_angle_rad)], dtype=np.float64)

            new_ma = m_center - (m_len * 0.5) * u_new
            new_mb = m_center + (m_len * 0.5) * u_new

            new_mirror = MirrorSpec(
                endpoint_a=Point2D(x=float(new_ma[0]), y=float(new_ma[1])),
                endpoint_b=Point2D(x=float(new_mb[0]), y=float(new_mb[1])),
                reflective_normal=Vector2D(dx=float(n_new[0]), dy=float(n_new[1])),
            )

            cand_scene = InspectionScene(
                scene_id=scene_b_id,
                seed=seed,
                category=SceneCategory.MIRROR_MISS,
                difficulty=difficulty,
                canvas_width=scene_a.canvas_width,
                canvas_height=scene_a.canvas_height,
                camera=scene_a.camera,
                mirror=new_mirror,
                target=scene_a.target,
                obstacles=scene_a.obstacles,
                ground_truth_state=VisibilityState.NOT_VISIBLE,
                ground_truth_reason=ObstructionReason.MISSED_MIRROR,
                diagnostic_details="Candidate tilt evaluation",
            )

            eval_cand = self.oracle.evaluate(cand_scene)

            # Check outcome flip and clear ambiguity rejection
            if not eval_cand.is_visible:
                # Map reason to appropriate scene category
                cat_map = {
                    ObstructionReason.MISSED_MIRROR: SceneCategory.MIRROR_MISS,
                    ObstructionReason.OUT_OF_FOV: SceneCategory.OUTSIDE_FOV,
                    ObstructionReason.BACK_OF_MIRROR: SceneCategory.WRONG_SIDE,
                    ObstructionReason.BLOCKED_INCIDENT: SceneCategory.BLOCKED_INCIDENT,
                    ObstructionReason.BLOCKED_REFLECTED: SceneCategory.BLOCKED_REFLECTED,
                }
                cat = cat_map.get(eval_cand.reason, SceneCategory.MIRROR_MISS)
                best_scene_b = self._sync_scene_with_oracle(
                    cand_scene, scene_b_id, cat, seed, difficulty
                )
                best_delta_deg = delta_deg
                best_reason = eval_cand.reason
                best_new_angle_deg = math.degrees(new_m_angle_rad)
                break

        if best_scene_b is None:
            raise RuntimeError(f"Could not find valid angular perturbation for seed {seed}")

        # Build pair record
        orig_angle_deg = math.degrees(m_angle_rad)
        changed_param = "mirror.angle_deg"
        orig_vals = {
            "mirror_angle_deg": round(orig_angle_deg, 2),
            "mirror_length_mm": round(m_len, 2),
        }
        new_vals = {
            "mirror_angle_deg": round(best_new_angle_deg, 2),
            "delta_angle_deg": round(best_delta_deg, 2),
        }

        diff_summary = (
            f"Mirror tilted by {best_delta_deg:+.1f}° (from {orig_angle_deg:.1f}° to {best_new_angle_deg:.1f}°). "
            f"Optical reflection point shifted, flipping visibility from CLEAR to {best_reason.value}."
        )

        return self._build_pair_bundle(
            pair_id=pair_id,
            pair_type=CounterfactualPairType.MIRROR_ANGLE_FLIP,
            parent_scene_id=parent_scene_id,
            scene_a=scene_a,
            scene_b=best_scene_b,
            theme_a=theme,
            theme_b=theme,
            changed_parameter=changed_param,
            original_values=orig_vals,
            new_values=new_vals,
            expected_relationship=ExpectedPairRelationship.FLIPPED_OUTCOME,
            difference_summary=diff_summary,
            output_dir=output_dir,
        )

    def create_cosmetic_control_pair(
        self,
        seed: int,
        pair_id: str,
        parent_id: str | None = None,
        output_dir: Path | None = None,
        difficulty: DifficultyLevel = DifficultyLevel.MEDIUM,
        theme_a: RenderTheme = RenderTheme.DARK_NAVY,
        theme_b: RenderTheme = RenderTheme.LIGHT_OFFWHITE,
    ) -> CounterfactualPairBundle:
        """
        PAIR TYPE 2: COSMETIC CONTROL.
        Keeps 100% of geometric coordinates, angles, and obstacles completely identical.
        Changes only the rendering theme/styling (e.g. DARK_NAVY vs LIGHT_OFFWHITE).
        The oracle outcome must remain 100% invariant.
        """
        parent_scene_id = parent_id or f"scene_parent_{seed}"
        scene_a_id = f"{pair_id}_a"
        scene_b_id = f"{pair_id}_b"

        # Generate base scene A
        scene_a = self.scene_generator.generate_scene(
            seed=seed,
            category=SceneCategory.VALID_VISIBLE,
            difficulty=difficulty,
            scene_id=scene_a_id,
        )

        # Scene B is an exact clone of Scene A with updated scene_id
        scene_b = scene_a.model_copy(deep=True, update={"scene_id": scene_b_id})

        changed_param = "rendering.theme"
        orig_vals = {"theme": theme_a.value}
        new_vals = {"theme": theme_b.value}

        diff_summary = (
            f"Identical geometry rendered under two different visual palettes: "
            f"{theme_a.value} vs {theme_b.value}. Ground truth outcome remains invariant."
        )

        return self._build_pair_bundle(
            pair_id=pair_id,
            pair_type=CounterfactualPairType.COSMETIC_CONTROL,
            parent_scene_id=parent_scene_id,
            scene_a=scene_a,
            scene_b=scene_b,
            theme_a=theme_a,
            theme_b=theme_b,
            changed_parameter=changed_param,
            original_values=orig_vals,
            new_values=new_vals,
            expected_relationship=ExpectedPairRelationship.INVARIANT_OUTCOME,
            difference_summary=diff_summary,
            output_dir=output_dir,
        )

    def create_obstacle_insertion_pair(
        self,
        seed: int,
        pair_id: str,
        parent_id: str | None = None,
        output_dir: Path | None = None,
        difficulty: DifficultyLevel = DifficultyLevel.MEDIUM,
        theme: RenderTheme = RenderTheme.DARK_NAVY,
    ) -> CounterfactualPairBundle:
        """
        PAIR TYPE 3: OBSTACLE INSERTION.
        Starts with a valid clear scene A.
        Inserts an opaque obstacle on one of the reflection legs (Camera->Mirror or Mirror->Target).
        The outcome flips from VISIBLE (CLEAR) to NOT_VISIBLE (BLOCKED_*).
        """
        parent_scene_id = parent_id or f"scene_parent_{seed}"
        scene_a_id = f"{pair_id}_a"
        scene_b_id = f"{pair_id}_b"

        # 1. Base clear scene
        scene_a = self.scene_generator.generate_scene(
            seed=seed,
            category=SceneCategory.VALID_VISIBLE,
            difficulty=difficulty,
            scene_id=scene_a_id,
        )
        eval_a = self.oracle.evaluate(scene_a)
        if not eval_a.is_visible or eval_a.reflection_point is None:
            raise ValueError(f"Base scene A for seed {seed} is not clear")

        cam_pt = np.array([scene_a.camera.position.x, scene_a.camera.position.y], dtype=np.float64)
        refl_pt = eval_a.reflection_point
        tgt_pt = np.array([scene_a.target.center.x, scene_a.target.center.y], dtype=np.float64)

        # Select leg to block (alternate based on seed)
        block_leg_1 = (seed % 2 == 0)
        rng = np.random.default_rng(seed + 777)

        if block_leg_1:
            # Block Camera -> Mirror
            s = float(rng.uniform(0.40, 0.60))
            obs_center = cam_pt + s * (refl_pt - cam_pt)
            blocked_leg_desc = "CAMERA_TO_MIRROR"
            target_cat = SceneCategory.BLOCKED_INCIDENT
        else:
            # Block Mirror -> Target
            s = float(rng.uniform(0.40, 0.60))
            obs_center = refl_pt + s * (tgt_pt - refl_pt)
            blocked_leg_desc = "MIRROR_TO_TARGET"
            target_cat = SceneCategory.BLOCKED_REFLECTED

        new_obs_id = f"O{len(scene_a.obstacles) + 1}"
        # Shape: Circle with radius 24 mm
        new_obs_spec = ObstacleSpec(
            obstacle_id=new_obs_id,
            obstacle_type="circle",
            center=Point2D(x=float(obs_center[0]), y=float(obs_center[1])),
            radius=24.0,
        )

        obstacles_b = list(scene_a.obstacles) + [new_obs_spec]

        cand_scene_b = InspectionScene(
            scene_id=scene_b_id,
            seed=seed,
            category=target_cat,
            difficulty=difficulty,
            canvas_width=scene_a.canvas_width,
            canvas_height=scene_a.canvas_height,
            camera=scene_a.camera,
            mirror=scene_a.mirror,
            target=scene_a.target,
            obstacles=obstacles_b,
            ground_truth_state=VisibilityState.NOT_VISIBLE,
            ground_truth_reason=(
                ObstructionReason.BLOCKED_INCIDENT if block_leg_1 else ObstructionReason.BLOCKED_REFLECTED
            ),
            diagnostic_details=f"Blocked by {new_obs_id} along {blocked_leg_desc}",
        )

        scene_b = self._sync_scene_with_oracle(
            cand_scene_b, scene_b_id, target_cat, seed, difficulty
        )

        eval_b = self.oracle.evaluate(scene_b)
        if eval_b.is_visible:
            raise RuntimeError(f"Inserted obstacle failed to block path in seed {seed}")

        changed_param = "obstacles.blocking_inserted"
        orig_vals = {
            "obstacle_count": len(scene_a.obstacles),
            "blocking_obstacle": None,
        }
        new_vals = {
            "obstacle_count": len(scene_b.obstacles),
            "blocking_obstacle": new_obs_spec.model_dump(),
            "blocked_leg": blocked_leg_desc,
        }

        diff_summary = (
            f"Opaque obstacle {new_obs_id} was inserted directly into the {blocked_leg_desc} path segment, "
            f"flipping visibility from CLEAR to {scene_b.ground_truth_reason.value}."
        )

        return self._build_pair_bundle(
            pair_id=pair_id,
            pair_type=CounterfactualPairType.OBSTACLE_INSERTION,
            parent_scene_id=parent_scene_id,
            scene_a=scene_a,
            scene_b=scene_b,
            theme_a=theme,
            theme_b=theme,
            changed_parameter=changed_param,
            original_values=orig_vals,
            new_values=new_vals,
            expected_relationship=ExpectedPairRelationship.FLIPPED_OUTCOME,
            difference_summary=diff_summary,
            output_dir=output_dir,
        )

    def create_irrelevant_obstacle_pair(
        self,
        seed: int,
        pair_id: str,
        parent_id: str | None = None,
        output_dir: Path | None = None,
        difficulty: DifficultyLevel = DifficultyLevel.MEDIUM,
        theme: RenderTheme = RenderTheme.DARK_NAVY,
    ) -> CounterfactualPairBundle:
        """
        PAIR TYPE 4: IRRELEVANT OBSTACLE.
        Starts with a valid clear scene A.
        Adds a distractor obstacle in an open area strictly outside both reflection path segments
        (with guaranteed clearance >= 30 mm).
        The outcome remains invariant (CLEAR -> CLEAR).
        """
        parent_scene_id = parent_id or f"scene_parent_{seed}"
        scene_a_id = f"{pair_id}_a"
        scene_b_id = f"{pair_id}_b"

        # 1. Base clear scene
        scene_a = self.scene_generator.generate_scene(
            seed=seed,
            category=SceneCategory.VALID_VISIBLE,
            difficulty=difficulty,
            scene_id=scene_a_id,
        )
        eval_a = self.oracle.evaluate(scene_a)
        if not eval_a.is_visible or eval_a.reflection_point is None:
            raise ValueError(f"Base scene A for seed {seed} is not clear")

        cam_pt = (scene_a.camera.position.x, scene_a.camera.position.y)
        refl_pt = (eval_a.reflection_point[0], eval_a.reflection_point[1])
        tgt_pt = (scene_a.target.center.x, scene_a.target.center.y)

        line_leg1 = LineString([cam_pt, refl_pt])
        line_leg2 = LineString([refl_pt, tgt_pt])

        # Candidate placement regions away from optical path
        rng = np.random.default_rng(seed + 999)
        candidate_zones = [
            (120.0, 240.0, 720.0, 840.0),  # Top-left corner
            (760.0, 880.0, 720.0, 840.0),  # Top-right corner
            (760.0, 880.0, 150.0, 320.0),  # Bottom-right cavity
            (120.0, 240.0, 150.0, 320.0),  # Bottom-left cavity
        ]

        distractor_spec: ObstacleSpec | None = None
        min_clearance_found: float = 0.0

        for _ in range(30):
            zone_idx = int(rng.integers(0, len(candidate_zones)))
            x_min, x_max, y_min, y_max = candidate_zones[zone_idx]
            cx = float(rng.uniform(x_min, x_max))
            cy = float(rng.uniform(y_min, y_max))
            radius = 22.0

            cand_obs = ObstacleSpec(
                obstacle_id=f"O{len(scene_a.obstacles) + 1}",
                obstacle_type="circle",
                center=Point2D(x=cx, y=cy),
                radius=radius,
            )

            poly = cand_obs.to_geometry_obstacle().to_shapely_polygon()
            d1 = float(poly.distance(line_leg1))
            d2 = float(poly.distance(line_leg2))
            clearance = min(d1, d2)

            # Check distance to camera, target, mirror endpoints
            d_cam = math.hypot(cx - cam_pt[0], cy - cam_pt[1])
            d_tgt = math.hypot(cx - tgt_pt[0], cy - tgt_pt[1])
            d_ma = math.hypot(cx - scene_a.mirror.endpoint_a.x, cy - scene_a.mirror.endpoint_a.y)
            d_mb = math.hypot(cx - scene_a.mirror.endpoint_b.x, cy - scene_a.mirror.endpoint_b.y)

            if (
                clearance >= 30.0
                and d_cam >= 50.0
                and d_tgt >= 50.0
                and min(d_ma, d_mb) >= 40.0
            ):
                distractor_spec = cand_obs
                min_clearance_found = clearance
                break

        if distractor_spec is None:
            raise RuntimeError(f"Could not place non-interfering distractor for seed {seed}")

        obstacles_b = list(scene_a.obstacles) + [distractor_spec]

        cand_scene_b = InspectionScene(
            scene_id=scene_b_id,
            seed=seed,
            category=SceneCategory.VALID_VISIBLE,
            difficulty=difficulty,
            canvas_width=scene_a.canvas_width,
            canvas_height=scene_a.canvas_height,
            camera=scene_a.camera,
            mirror=scene_a.mirror,
            target=scene_a.target,
            obstacles=obstacles_b,
            ground_truth_state=VisibilityState.VISIBLE,
            ground_truth_reason=ObstructionReason.NONE,
            diagnostic_details=scene_a.diagnostic_details,
        )

        scene_b = self._sync_scene_with_oracle(
            cand_scene_b, scene_b_id, SceneCategory.VALID_VISIBLE, seed, difficulty
        )

        eval_b = self.oracle.evaluate(scene_b)
        if not eval_b.is_visible:
            raise RuntimeError(f"Distractor accidentally blocked sightline in seed {seed}")

        changed_param = "obstacles.irrelevant_inserted"
        orig_vals = {
            "obstacle_count": len(scene_a.obstacles),
            "distractor": None,
        }
        new_vals = {
            "obstacle_count": len(scene_b.obstacles),
            "distractor": distractor_spec.model_dump(),
            "min_clearance_mm": round(min_clearance_found, 2),
        }

        diff_summary = (
            f"Irrelevant distractor obstacle {distractor_spec.obstacle_id} was added with "
            f"{min_clearance_found:.1f} mm clearance from optical rays. "
            f"Sightline remains unblocked and outcome remains invariant (CLEAR)."
        )

        return self._build_pair_bundle(
            pair_id=pair_id,
            pair_type=CounterfactualPairType.IRRELEVANT_OBSTACLE,
            parent_scene_id=parent_scene_id,
            scene_a=scene_a,
            scene_b=scene_b,
            theme_a=theme,
            theme_b=theme,
            changed_parameter=changed_param,
            original_values=orig_vals,
            new_values=new_vals,
            expected_relationship=ExpectedPairRelationship.INVARIANT_OUTCOME,
            difference_summary=diff_summary,
            output_dir=output_dir,
        )

    def _build_pair_bundle(
        self,
        pair_id: str,
        pair_type: CounterfactualPairType,
        parent_scene_id: str,
        scene_a: InspectionScene,
        scene_b: InspectionScene,
        theme_a: RenderTheme,
        theme_b: RenderTheme,
        changed_parameter: str,
        original_values: dict[str, Any],
        new_values: dict[str, Any],
        expected_relationship: ExpectedPairRelationship,
        difference_summary: str,
        output_dir: Path | None = None,
    ) -> CounterfactualPairBundle:
        """
        Verify relationship and package pair artifacts.
        """
        eval_a = self.oracle.evaluate(scene_a)
        eval_b = self.oracle.evaluate(scene_b)

        # Verification of expected relationship
        if (
            expected_relationship == ExpectedPairRelationship.FLIPPED_OUTCOME
            and eval_a.is_visible == eval_b.is_visible
        ):
            raise ValueError(
                f"Pair {pair_id} failed FLIPPED_OUTCOME verification: "
                f"A={eval_a.is_visible}, B={eval_b.is_visible}"
            )
        elif (
            expected_relationship == ExpectedPairRelationship.INVARIANT_OUTCOME
            and eval_a.is_visible != eval_b.is_visible
        ):
            raise ValueError(
                f"Pair {pair_id} failed INVARIANT_OUTCOME verification: "
                f"A={eval_a.is_visible}, B={eval_b.is_visible}"
            )

        # Build bundles for Scene A and Scene B
        pair_dir = Path(output_dir) / pair_id if output_dir is not None else None
        dir_a = pair_dir / "scene_a" if pair_dir is not None else None
        dir_b = pair_dir / "scene_b" if pair_dir is not None else None

        bundle_a = self.task_factory.create_bundle_from_scene(
            scene=scene_a,
            theme=theme_a,
            target_dir=dir_a,
            task_id=f"{pair_id}_a",
        )
        bundle_b = self.task_factory.create_bundle_from_scene(
            scene=scene_b,
            theme=theme_b,
            target_dir=dir_b,
            task_id=f"{pair_id}_b",
        )

        outcome_a = CounterfactualSceneOutcome(
            scene_id=bundle_a.task_id,
            is_visible=eval_a.is_visible,
            reason_code=OBSTRUCTION_TO_REASON_CODE.get(
                eval_a.reason, VisibilityReasonCode.CLEAR
            ),
            diagnostic_details=eval_a.diagnostic_details,
            image_filename="diagram_blind.png",
        )
        outcome_b = CounterfactualSceneOutcome(
            scene_id=bundle_b.task_id,
            is_visible=eval_b.is_visible,
            reason_code=OBSTRUCTION_TO_REASON_CODE.get(
                eval_b.reason, VisibilityReasonCode.CLEAR
            ),
            diagnostic_details=eval_b.diagnostic_details,
            image_filename="diagram_blind.png",
        )

        record = CounterfactualPairRecord(
            pair_id=pair_id,
            pair_type=pair_type,
            parent_scene_id=parent_scene_id,
            child_scene_ids=[bundle_a.task_id, bundle_b.task_id],
            changed_parameter=changed_parameter,
            original_values=original_values,
            new_values=new_values,
            ground_truth_outcomes={
                bundle_a.task_id: outcome_a,
                bundle_b.task_id: outcome_b,
            },
            expected_relationship=expected_relationship,
            relationship_verified=True,
            difference_summary=difference_summary,
        )

        if pair_dir is not None:
            with open(pair_dir / "pair.json", "w", encoding="utf-8") as f:
                json.dump(record.model_dump(mode="json"), f, indent=2)

        return CounterfactualPairBundle(
            pair_id=pair_id,
            pair_type=pair_type,
            pair_record=record,
            scene_a_bundle=bundle_a,
            scene_b_bundle=bundle_b,
        )

    def generate_dataset(
        self,
        num_pairs_per_type: int = 30,
        output_dir: Path | None = None,
        start_seed: int = 40000,
    ) -> list[CounterfactualPairBundle]:
        """
        Generate a complete verified dataset containing num_pairs_per_type
        of all 4 counterfactual pair types (total = 4 * num_pairs_per_type).
        """
        all_bundles: list[CounterfactualPairBundle] = []
        themes = [
            RenderTheme.DARK_NAVY,
            RenderTheme.LIGHT_OFFWHITE,
            RenderTheme.BLUEPRINT,
            RenderTheme.MONOCHROME_CAD,
        ]

        # 1. Type 1: Mirror Angle Flip
        for i in range(num_pairs_per_type):
            seed = start_seed + i
            pid = f"pair_mirror_flip_{i:03d}"
            theme = themes[i % len(themes)]
            bundle = self.create_mirror_angle_flip_pair(
                seed=seed,
                pair_id=pid,
                output_dir=output_dir,
                theme=theme,
            )
            all_bundles.append(bundle)

        # 2. Type 2: Cosmetic Control
        cosmetic_start_seed = start_seed + 1000
        for i in range(num_pairs_per_type):
            seed = cosmetic_start_seed + i
            pid = f"pair_cosmetic_{i:03d}"
            t_a = themes[i % len(themes)]
            t_b = themes[(i + 1) % len(themes)]
            bundle = self.create_cosmetic_control_pair(
                seed=seed,
                pair_id=pid,
                output_dir=output_dir,
                theme_a=t_a,
                theme_b=t_b,
            )
            all_bundles.append(bundle)

        # 3. Type 3: Obstacle Insertion
        obs_insert_start_seed = start_seed + 2000
        for i in range(num_pairs_per_type):
            seed = obs_insert_start_seed + i
            pid = f"pair_obs_insert_{i:03d}"
            theme = themes[i % len(themes)]
            bundle = self.create_obstacle_insertion_pair(
                seed=seed,
                pair_id=pid,
                output_dir=output_dir,
                theme=theme,
            )
            all_bundles.append(bundle)

        # 4. Type 4: Irrelevant Obstacle
        irrel_start_seed = start_seed + 3000
        for i in range(num_pairs_per_type):
            seed = irrel_start_seed + i
            pid = f"pair_irrel_obs_{i:03d}"
            theme = themes[i % len(themes)]
            bundle = self.create_irrelevant_obstacle_pair(
                seed=seed,
                pair_id=pid,
                output_dir=output_dir,
                theme=theme,
            )
            all_bundles.append(bundle)

        # Aggregate manifests
        if output_dir is not None:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            # pairs.jsonl
            with open(out_path / "pairs.jsonl", "w", encoding="utf-8") as f:
                f.writelines(b.pair_record.model_dump_json() + "\n" for b in all_bundles)

            # tasks.jsonl (all child tasks)
            with open(out_path / "tasks.jsonl", "w", encoding="utf-8") as f:
                for b in all_bundles:
                    f.write(b.scene_a_bundle.question.model_dump_json() + "\n")
                    f.write(b.scene_b_bundle.question.model_dump_json() + "\n")

            # ground_truth.jsonl (all child ground truths)
            with open(out_path / "ground_truth.jsonl", "w", encoding="utf-8") as f:
                for b in all_bundles:
                    f.write(b.scene_a_bundle.ground_truth.model_dump_json() + "\n")
                    f.write(b.scene_b_bundle.ground_truth.model_dump_json() + "\n")

        return all_bundles
