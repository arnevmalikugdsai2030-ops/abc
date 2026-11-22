"""
Task B: Obstruction Identification Factory for MIRRORLAB.

Generates constraint-verified tasks requiring models to determine:
1. Whether the reflected inspection view is blocked.
2. The specific obstacle label (O1, O2, O3...) causing the obstruction.
3. The exact optical path segment blocked:
   - CAMERA_TO_MIRROR (Leg 1)
   - MIRROR_TO_TARGET (Leg 2)
   - NONE (Unblocked control)

Enforces:
- Geometrically valid base scenes with clear specular reflection.
- Exactly one blocker on exactly one path segment (or zero for unblocked controls).
- Optional irrelevant distractor obstacles maintaining strict clearance (>15mm).
- Rejection of grazing, multi-blocker, or ambiguous layouts.
- Complete mathematical oracle derivations.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from shapely.geometry import LineString

from mirrorlab.generation.scene_generator import SceneGenerator
from mirrorlab.geometry.obstacles import (
    CircleObstacle,
    ObstacleBase,
    PathLeg,
    RectangleObstacle,
    check_segment_occlusion,
)
from mirrorlab.geometry.oracle import evaluate_visibility_oracle
from mirrorlab.rendering.pillow_renderer import PillowDiagramRenderer
from mirrorlab.rendering.themes import RenderMode, RenderTheme
from mirrorlab.schema.common import ObstructionReason, Point2D
from mirrorlab.schema.scene import (
    DifficultyLevel,
    InspectionScene,
    ObstacleSpec,
    SceneCategory,
)
from mirrorlab.schema.task import (
    BlockedSegment,
    ObstructionGroundTruth,
    ObstructionProof,
    ObstructionQuestion,
    ObstructionTaskBundle,
)

DEFAULT_OBSTRUCTION_PROMPT = """You are an industrial machine-vision inspector analyzing a 2D optical inspection diagram.

Apparatus Legend:
- CAMERA: Represented by the blue camera symbol with a directional lens pointing along its optical axis. The translucent cone represents the camera's field of view (FOV).
- MIRROR: Finite flat mirror segment between endpoints A and B. The reflective front face is marked with a vibrant cyan line. The non-reflective back face is marked with diagonal hatch lines.
- TARGET DEFECT: A red 'X' scratch defect in the cavity.
- OBSTACLES: Opaque polygonal barriers labeled O1, O2, O3, etc. Barrier O1 is the cavity partition wall blocking direct sightline between camera and scratch, requiring reflected inspection via the mirror. Other obstacles may be blockers or irrelevant distractors.

Optical Path:
The physical specular reflection path consists of two straight segments:
1. Incident Segment: Camera -> Mirror Reflection Point (CAMERA_TO_MIRROR)
2. Reflected Segment: Mirror Reflection Point -> Target Scratch (MIRROR_TO_TARGET)

Question:
Is the camera's view of the red X blocked by an obstacle? If so, which obstacle blocks which segment of the path?

Analyze:
1. Trace the straight incident ray from the camera to the specular reflection point on the mirror.
2. Trace the straight reflected ray from the mirror to the red X target scratch.
3. Determine whether any obstacle (O1, O2, O3...) intersects either segment:
   - If an obstacle intersects the path from camera to mirror -> blocked on CAMERA_TO_MIRROR.
   - If an obstacle intersects the path from mirror to target -> blocked on MIRROR_TO_TARGET.
   - If both segments are completely unobstructed -> view is UNBLOCKED (NONE).

Respond in JSON format with:
{
  "is_blocked": true or false,
  "blocking_obstacle": "O1" | "O2" | "O3" | "NONE",
  "blocked_segment": "CAMERA_TO_MIRROR" | "MIRROR_TO_TARGET" | "NONE",
  "explanation": "<your step-by-step reasoning identifying the blocking obstacle and path leg>"
}
"""


class ObstructionTaskFactory:
    """
    Procedural generator for Task B: Obstruction Identification.
    """

    def __init__(
        self,
        canvas_width: float = 1000.0,
        canvas_height: float = 1000.0,
        image_size: int = 1024,
        max_attempts: int = 100,
    ):
        self.canvas_width = float(canvas_width)
        self.canvas_height = float(canvas_height)
        self.image_size = int(image_size)
        self.max_attempts = int(max_attempts)
        self.scene_generator = SceneGenerator(
            canvas_width=canvas_width, canvas_height=canvas_height
        )

    def create_task(
        self,
        seed: int,
        target_segment: BlockedSegment = BlockedSegment.CAMERA_TO_MIRROR,
        difficulty: DifficultyLevel = DifficultyLevel.MEDIUM,
        num_distractors: int = 1,
        theme: RenderTheme = RenderTheme.DARK_NAVY,
        output_dir: Path | None = None,
        task_id: str | None = None,
    ) -> ObstructionTaskBundle:
        """
        Generate a single verified Task B bundle.
        """
        rng = np.random.default_rng(seed)
        tid = task_id or f"task_obs_{seed:05d}"

        for attempt in range(1, self.max_attempts + 1):
            candidate = self._generate_candidate(
                rng=rng,
                seed=seed + attempt * 1000,
                target_segment=target_segment,
                difficulty=difficulty,
                num_distractors=num_distractors,
                tid=tid,
            )
            if candidate is not None:
                return self._finalize_task_bundle(
                    candidate=candidate,
                    theme=theme,
                    output_dir=output_dir,
                )

        raise RuntimeError(
            f"Failed to generate valid Task B obstruction scene for seed {seed} after {self.max_attempts} attempts."
        )

    def _generate_candidate(
        self,
        rng: np.random.Generator,
        seed: int,
        target_segment: BlockedSegment,
        difficulty: DifficultyLevel,
        num_distractors: int,
        tid: str,
    ) -> dict | None:
        """Procedurally build and rigorously test an obstruction candidate scene."""
        # 1. Generate base valid scene (unobstructed reflection, direct sight blocked)
        base_scene = self.scene_generator.generate_scene(
            seed=seed,
            category=SceneCategory.VALID_VISIBLE,
            difficulty=difficulty,
            scene_id=f"base_{tid}",
        )

        cam_pt = np.array([base_scene.camera.position.x, base_scene.camera.position.y])
        tgt_pt = np.array([base_scene.target.center.x, base_scene.target.center.y])
        refl_pt = np.array(
            [base_scene.reflection_point.x, base_scene.reflection_point.y]
        )

        # LineStrings for the two optical legs
        leg1_line = LineString([tuple(cam_pt), tuple(refl_pt)])
        leg2_line = LineString([tuple(refl_pt), tuple(tgt_pt)])

        # The partition wall (cavity barrier O1)
        partition_spec = base_scene.obstacles[0]
        partition_obs = RectangleObstacle(
            obstacle_id="partition_wall",
            center=[partition_spec.center.x, partition_spec.center.y],
            width=partition_spec.width,
            height=partition_spec.height,
            angle_degrees=partition_spec.angle_degrees or 0.0,
        )

        # Verify initial clearance of partition wall
        part_poly = partition_obs.to_shapely_polygon()
        if part_poly.distance(leg1_line) < 20.0 or part_poly.distance(leg2_line) < 20.0:
            return None

        obstacles: list[ObstacleBase] = [partition_obs]
        blocker_obs: ObstacleBase | None = None

        # 2. Introduce exactly one blocker if target_segment is not NONE
        if target_segment == BlockedSegment.CAMERA_TO_MIRROR:
            # Place blocker on Leg 1 (Camera -> Mirror)
            t_block = rng.uniform(0.35, 0.65)
            block_center = cam_pt + t_block * (refl_pt - cam_pt)
            radius = rng.uniform(18.0, 26.0)

            blocker_obs = CircleObstacle(
                "blocker_leg1", center=block_center, radius=radius
            )
            b_poly = blocker_obs.to_shapely_polygon()

            # Must intersect Leg 1 cleanly with penetration >= 8.0 mm
            inter1 = b_poly.intersection(leg1_line)
            if inter1.is_empty or inter1.length < 8.0:
                return None

            # Must NOT intersect Leg 2, must have clearance >= 15 mm from Leg 2
            if b_poly.distance(leg2_line) < 15.0:
                return None

            obstacles.append(blocker_obs)

        elif target_segment == BlockedSegment.MIRROR_TO_TARGET:
            # Place blocker on Leg 2 (Mirror -> Target)
            t_block = rng.uniform(0.35, 0.65)
            block_center = refl_pt + t_block * (tgt_pt - refl_pt)
            radius = rng.uniform(18.0, 26.0)

            blocker_obs = CircleObstacle(
                "blocker_leg2", center=block_center, radius=radius
            )
            b_poly = blocker_obs.to_shapely_polygon()

            # Must intersect Leg 2 cleanly with penetration >= 8.0 mm
            inter2 = b_poly.intersection(leg2_line)
            if inter2.is_empty or inter2.length < 8.0:
                return None

            # Must NOT intersect Leg 1, must have clearance >= 15 mm from Leg 1
            if b_poly.distance(leg1_line) < 15.0:
                return None

            obstacles.append(blocker_obs)

        # 3. Add Optional Distractor Objects (must strictly avoid optical rays)
        distractor_count = 0
        distractor_attempts = 0
        while distractor_count < num_distractors and distractor_attempts < 30:
            distractor_attempts += 1
            d_x = rng.uniform(150.0, self.canvas_width - 150.0)
            d_y = rng.uniform(150.0, self.canvas_height - 150.0)
            d_pos = np.array([d_x, d_y])

            # Distractor shape: Circle or Rectangle
            if rng.random() > 0.5:
                d_radius = rng.uniform(15.0, 24.0)
                d_obs = CircleObstacle(
                    f"distractor_{distractor_count + 1}",
                    center=d_pos,
                    radius=d_radius,
                )
            else:
                d_w = rng.uniform(30.0, 48.0)
                d_h = rng.uniform(30.0, 48.0)
                d_angle = rng.uniform(0.0, 90.0)
                d_obs = RectangleObstacle(
                    f"distractor_{distractor_count + 1}",
                    center=d_pos,
                    width=d_w,
                    height=d_h,
                    angle_degrees=d_angle,
                )

            d_poly = d_obs.to_shapely_polygon()

            # Clearance check: must be >= 18 mm away from both optical legs
            if d_poly.distance(leg1_line) < 18.0 or d_poly.distance(leg2_line) < 18.0:
                continue

            # Must be away from camera, target, mirror endpoints, and existing obstacles
            if (
                np.linalg.norm(d_pos - cam_pt) < 60.0
                or np.linalg.norm(d_pos - tgt_pt) < 60.0
                or np.linalg.norm(d_pos - refl_pt) < 60.0
            ):
                continue

            # Check overlap with existing obstacles
            overlap = False
            for exist_obs in obstacles:
                if d_poly.distance(exist_obs.to_shapely_polygon()) < 15.0:
                    overlap = True
                    break
            if overlap:
                continue

            obstacles.append(d_obs)
            distractor_count += 1

        # 4. Strict Rejection Verification via Oracle
        oracle_res = evaluate_visibility_oracle(
            camera_pos=cam_pt,
            camera_forward=base_scene.camera.heading_deg,
            camera_fov_degrees=base_scene.camera.fov_degrees,
            target_pos=tgt_pt,
            mirror_a=[base_scene.mirror.endpoint_a.x, base_scene.mirror.endpoint_a.y],
            mirror_b=[base_scene.mirror.endpoint_b.x, base_scene.mirror.endpoint_b.y],
            obstacles=obstacles,
            reflective_normal=[
                base_scene.mirror.reflective_normal.dx,
                base_scene.mirror.reflective_normal.dy,
            ],
        )

        # Check outcome consistency
        if target_segment == BlockedSegment.NONE:
            if not oracle_res.is_visible or oracle_res.reason != ObstructionReason.NONE:
                return None
        elif target_segment == BlockedSegment.CAMERA_TO_MIRROR:
            if (
                oracle_res.is_visible
                or oracle_res.reason != ObstructionReason.BLOCKED_INCIDENT
            ):
                return None
            if (
                oracle_res.blocking_obstacle_hit is None
                or oracle_res.blocking_obstacle_hit.obstacle_id
                != blocker_obs.obstacle_id
            ):
                return None
        elif target_segment == BlockedSegment.MIRROR_TO_TARGET:
            if (
                oracle_res.is_visible
                or oracle_res.reason != ObstructionReason.BLOCKED_REFLECTED
            ):
                return None
            if (
                oracle_res.blocking_obstacle_hit is None
                or oracle_res.blocking_obstacle_hit.obstacle_id
                != blocker_obs.obstacle_id
            ):
                return None

        # Verify that ONLY the intended blocker touches the path
        # Check Leg 1
        leg1_hit = check_segment_occlusion(
            cam_pt, refl_pt, obstacles, leg=PathLeg.LEG_1_CAMERA_TO_MIRROR
        )
        leg2_hit = check_segment_occlusion(
            refl_pt, tgt_pt, obstacles, leg=PathLeg.LEG_2_MIRROR_TO_TARGET
        )

        if target_segment == BlockedSegment.NONE:
            if leg1_hit is not None or leg2_hit is not None:
                return None
        elif target_segment == BlockedSegment.CAMERA_TO_MIRROR:
            if leg1_hit is None or leg1_hit.obstacle_id != blocker_obs.obstacle_id:
                return None
            if leg2_hit is not None:
                return None
        elif target_segment == BlockedSegment.MIRROR_TO_TARGET:
            if leg1_hit is not None:
                return None
            if leg2_hit is None or leg2_hit.obstacle_id != blocker_obs.obstacle_id:
                return None

        # Calculate minimum distractor clearance
        min_clearance = 999.0
        for obs in obstacles:
            if blocker_obs is not None and obs.obstacle_id == blocker_obs.obstacle_id:
                continue
            poly = obs.to_shapely_polygon()
            d1 = poly.distance(leg1_line)
            d2 = poly.distance(leg2_line)
            min_clearance = min(min_clearance, d1, d2)

        return {
            "tid": tid,
            "seed": seed,
            "difficulty": difficulty,
            "target_segment": target_segment,
            "base_scene": base_scene,
            "obstacles": obstacles,
            "blocker_obs": blocker_obs,
            "oracle_res": oracle_res,
            "min_clearance": min_clearance,
        }

    def _finalize_task_bundle(
        self,
        candidate: dict,
        theme: RenderTheme,
        output_dir: Path | None,
    ) -> ObstructionTaskBundle:
        """Assemble typed ObstructionTaskBundle with all 5 artifacts."""
        tid = candidate["tid"]
        seed = candidate["seed"]
        diff = candidate["difficulty"]
        target_segment = candidate["target_segment"]
        base_scene: InspectionScene = candidate["base_scene"]
        obstacles: list[ObstacleBase] = candidate["obstacles"]
        blocker_obs: ObstacleBase | None = candidate["blocker_obs"]
        oracle_res = candidate["oracle_res"]

        # 1. Convert obstacles to ObstacleSpec with labeled IDs
        # Label convention: O1 is partition wall, O2 is blocker/distractor, O3...
        obstacle_specs: list[ObstacleSpec] = []
        obstacle_labels: list[str] = []
        blocking_label: str | None = None

        for idx, obs in enumerate(obstacles):
            label = f"O{idx + 1}"
            obstacle_labels.append(label)

            if blocker_obs is not None and obs.obstacle_id == blocker_obs.obstacle_id:
                blocking_label = label

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

        # 2. Build Updated InspectionScene
        # Re-map category based on obstruction state
        if target_segment == BlockedSegment.CAMERA_TO_MIRROR:
            cat = SceneCategory.BLOCKED_INCIDENT
        elif target_segment == BlockedSegment.MIRROR_TO_TARGET:
            cat = SceneCategory.BLOCKED_REFLECTED
        else:
            cat = SceneCategory.VALID_VISIBLE

        path = [Point2D(x=float(p[0]), y=float(p[1])) for p in oracle_res.path_points]
        refl_pt = Point2D(
            x=float(base_scene.reflection_point.x),
            y=float(base_scene.reflection_point.y),
        )
        vt_pt = Point2D(
            x=float(base_scene.virtual_target.x), y=float(base_scene.virtual_target.y)
        )

        scene = InspectionScene(
            scene_id=tid,
            seed=seed,
            category=cat,
            difficulty=diff,
            canvas_width=self.canvas_width,
            canvas_height=self.canvas_height,
            camera=base_scene.camera,
            mirror=base_scene.mirror,
            target=base_scene.target,
            obstacles=obstacle_specs,
            ground_truth_state=oracle_res.state,
            ground_truth_reason=oracle_res.reason,
            diagnostic_details=oracle_res.diagnostic_details,
            reflection_point=refl_pt,
            virtual_target=vt_pt,
            optical_path=path,
            incident_angle_deg=base_scene.incident_angle_deg,
            reflected_angle_deg=base_scene.reflected_angle_deg,
            total_optical_distance=oracle_res.total_optical_distance,
            is_direct_sight_blocked=True,
        )

        # 3. Build Proof
        inter_pt: Point2D | None = None
        penetration_depth = None
        if blocker_obs is not None and oracle_res.blocking_obstacle_hit is not None:
            inter_pt = Point2D(
                x=float(oracle_res.blocking_obstacle_hit.hit_point[0]),
                y=float(oracle_res.blocking_obstacle_hit.hit_point[1]),
            )
            if target_segment == BlockedSegment.CAMERA_TO_MIRROR:
                leg1_line = LineString(
                    [
                        (base_scene.camera.position.x, base_scene.camera.position.y),
                        (refl_pt.x, refl_pt.y),
                    ]
                )
                penetration_depth = float(
                    blocker_obs.to_shapely_polygon().intersection(leg1_line).length
                )
            elif target_segment == BlockedSegment.MIRROR_TO_TARGET:
                leg2_line = LineString(
                    [
                        (refl_pt.x, refl_pt.y),
                        (base_scene.target.center.x, base_scene.target.center.y),
                    ]
                )
                penetration_depth = float(
                    blocker_obs.to_shapely_polygon().intersection(leg2_line).length
                )

        proof = ObstructionProof(
            reflection_point=refl_pt,
            virtual_target=vt_pt,
            incident_angle_deg=float(base_scene.incident_angle_deg),
            reflected_angle_deg=float(base_scene.reflected_angle_deg),
            blocked_segment=target_segment,
            blocking_obstacle_label=blocking_label,
            blocking_obstacle_id=blocker_obs.obstacle_id if blocker_obs else None,
            intersection_point=inter_pt,
            penetration_depth_mm=penetration_depth,
            min_distractor_clearance_mm=float(candidate["min_clearance"]),
            optical_path=path,
        )

        # 4. Build Ground Truth
        is_blocked = target_segment != BlockedSegment.NONE
        if is_blocked:
            explanation = (
                f"Mathematical Proof: Path is BLOCKED on segment {target_segment.value} by obstacle '{blocking_label}' "
                f"at ({inter_pt.x:.1f}, {inter_pt.y:.1f}) with penetration depth {penetration_depth:.1f}mm. "
                f"Reflection point R is at ({refl_pt.x:.1f}, {refl_pt.y:.1f}) with theta_i=theta_r={base_scene.incident_angle_deg:.1f} deg. "
                f"Distractor obstacles maintain minimum clearance of {candidate['min_clearance']:.1f}mm."
            )
        else:
            explanation = (
                f"Mathematical Proof: Path is UNBLOCKED. Specular reflection C -> R({refl_pt.x:.1f}, {refl_pt.y:.1f}) -> T "
                f"is 100% clear with theta_i=theta_r={base_scene.incident_angle_deg:.1f} deg. "
                f"All distractor obstacles maintain clear margin (>={candidate['min_clearance']:.1f}mm)."
            )

        ground_truth = ObstructionGroundTruth(
            task_id=tid,
            is_blocked=is_blocked,
            blocking_obstacle_label=blocking_label,
            blocked_segment=target_segment,
            detailed_explanation=explanation,
            proof=proof,
        )

        # 5. Build Question
        question = ObstructionQuestion(
            task_id=tid,
            question="Is the camera's view of the red X blocked by an obstacle? If so, which obstacle blocks which segment of the path?",
            prompt=DEFAULT_OBSTRUCTION_PROMPT,
            image_filename="diagram_blind.png",
            obstacle_labels=obstacle_labels,
        )

        # 6. Render Images
        renderer = PillowDiagramRenderer(
            theme=theme,
            image_width=self.image_size,
            image_height=self.image_size,
        )
        img_blind = renderer.render(scene, mode=RenderMode.BLIND)
        img_overlay = renderer.render(scene, mode=RenderMode.OVERLAY)

        # 7. Persist to Output Directory if provided
        blind_path_str = ""
        overlay_path_str = ""
        if output_dir is not None:
            task_dir = Path(output_dir) / tid
            task_dir.mkdir(parents=True, exist_ok=True)

            with open(task_dir / "scene.json", "w", encoding="utf-8") as f:
                json.dump(scene.model_dump(mode="json"), f, indent=2)

            blind_file = task_dir / "diagram_blind.png"
            img_blind.save(blind_file, format="PNG")
            blind_path_str = str(blind_file.resolve())

            with open(task_dir / "question.json", "w", encoding="utf-8") as f:
                json.dump(question.model_dump(mode="json"), f, indent=2)

            with open(task_dir / "ground_truth.json", "w", encoding="utf-8") as f:
                json.dump(ground_truth.model_dump(mode="json"), f, indent=2)

            overlay_file = task_dir / "diagram_overlay.png"
            img_overlay.save(overlay_file, format="PNG")
            overlay_path_str = str(overlay_file.resolve())

        return ObstructionTaskBundle(
            task_id=tid,
            seed=seed,
            difficulty=diff,
            theme=theme.value,
            scene=scene,
            question=question,
            ground_truth=ground_truth,
            blind_image_path=blind_path_str,
            overlay_image_path=overlay_path_str,
        )

    def create_balanced_dataset(
        self,
        count: int = 30,
        start_seed: int = 20000,
        output_dir: Path | None = None,
        difficulties: list[DifficultyLevel] | None = None,
        themes: list[RenderTheme] | None = None,
    ) -> list[ObstructionTaskBundle]:
        """
        Generate a balanced dataset across the 3 outcomes:
        - CAMERA_TO_MIRROR (Leg 1 blocked)
        - MIRROR_TO_TARGET (Leg 2 blocked)
        - NONE (Unblocked control with distractors)
        """
        diff_list = difficulties or [
            DifficultyLevel.EASY,
            DifficultyLevel.MEDIUM,
            DifficultyLevel.HARD,
        ]
        theme_list = themes or [
            RenderTheme.DARK_NAVY,
            RenderTheme.LIGHT_OFFWHITE,
            RenderTheme.BLUEPRINT,
            RenderTheme.MONOCHROME_CAD,
        ]

        target_segments = [
            BlockedSegment.CAMERA_TO_MIRROR,
            BlockedSegment.MIRROR_TO_TARGET,
            BlockedSegment.NONE,
        ]

        tasks: list[ObstructionTaskBundle] = []
        for i in range(count):
            seed = start_seed + i
            diff = diff_list[i % len(diff_list)]
            theme = theme_list[i % len(theme_list)]
            segment = target_segments[i % len(target_segments)]
            num_distractors = 1 if i % 2 == 0 else 2

            task = self.create_task(
                seed=seed,
                target_segment=segment,
                difficulty=diff,
                num_distractors=num_distractors,
                theme=theme,
                output_dir=output_dir,
            )
            tasks.append(task)

        # Write aggregate manifests if output_dir provided
        if output_dir is not None:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            with open(out_path / "tasks.jsonl", "w", encoding="utf-8") as f:
                f.writelines(t.question.model_dump_json() + "\n" for t in tasks)

            with open(out_path / "ground_truth.jsonl", "w", encoding="utf-8") as f:
                f.writelines(t.ground_truth.model_dump_json() + "\n" for t in tasks)

        return tasks
