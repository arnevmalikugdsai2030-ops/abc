"""
Task C: Mirror-Setting Repair Factory for MIRRORLAB.

Generates constraint-verified tasks requiring models to choose from 3 candidate
mirror configurations (A, B, and C) to restore reflected inspection visibility.

Enforces:
- Fixed camera, fixed red-X scratch defect, and fixed obstacles.
- Three geometrically distinct candidate mirror configurations.
- Independent oracle validation for every candidate.
- Strict uniqueness: exactly ONE candidate restores visibility.
- Random, balanced distribution of correct option slots (A, B, C).
- Complete evaluations of all three candidates stored in ground truth.
- Zero answer leakage in blind task images.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from mirrorlab.generation.scene_generator import SceneGenerator
from mirrorlab.geometry.obstacles import ObstacleBase, RectangleObstacle
from mirrorlab.geometry.oracle import evaluate_visibility_oracle
from mirrorlab.geometry.vector import vector_magnitude, vector_normalize, vector_rotate
from mirrorlab.rendering.pillow_renderer import PillowDiagramRenderer
from mirrorlab.rendering.themes import RenderMode, RenderTheme
from mirrorlab.schema.common import Point2D, Vector2D
from mirrorlab.schema.scene import (
    DifficultyLevel,
    InspectionScene,
    MirrorSpec,
    SceneCategory,
)
from mirrorlab.schema.task import (
    OBSTRUCTION_TO_REASON_CODE,
    RepairCandidateEvaluation,
    RepairGroundTruth,
    RepairQuestion,
    RepairTaskBundle,
    VisibilityReasonCode,
)

DEFAULT_REPAIR_PROMPT = """You are an industrial machine-vision inspector analyzing a 2D optical inspection diagram.

Problem Statement:
The inspection apparatus is misaligned or broken: the camera cannot see the red 'X' scratch defect in the cavity.
The camera, target scratch, and obstacles (O1, etc.) remain fixed in place.
Engineers have proposed three candidate mirror configurations: MIRROR A, MIRROR B, and MIRROR C.

Apparatus Legend:
- CAMERA: Directed blue symbol with translucent FOV cone.
- CANDIDATE MIRRORS (A, B, C): Three distinct finite flat mirror segments. Each mirror's reflective front face is marked in vibrant cyan; its non-reflective back side is marked with diagonal hatch lines.
- TARGET DEFECT: Red 'X' scratch in the cavity.
- OBSTACLES: Opaque barriers. Barrier O1 blocks direct line of sight between camera and scratch, so inspection is only possible via specular reflection.

Law of Reflection:
A candidate mirror successfully restores inspection if and only if:
1. The specular reflection ray from the camera to the target strikes within that candidate's finite mirror segment.
2. The reflection point lies within the camera's field of view cone.
3. The mirror's reflective face (cyan) faces towards the camera and target (not the hatched back side).
4. Neither the incident ray (camera to mirror) nor the reflected ray (mirror to target) is blocked by an obstacle.

Question:
Which candidate mirror configuration (A, B, or C) successfully restores the camera's view of the red X target scratch?

Analyze each candidate (A, B, and C) independently, and respond in JSON format with:
{
  "selected_candidate": "A" | "B" | "C",
  "explanation": "<your step-by-step reasoning explaining why the chosen candidate succeeds and why the other two fail>"
}
"""


class RepairTaskFactory:
    """
    Procedural generator for Task C: Mirror-Setting Repair.
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
        difficulty: DifficultyLevel = DifficultyLevel.MEDIUM,
        force_winner_slot: str | None = None,
        theme: RenderTheme = RenderTheme.DARK_NAVY,
        output_dir: Path | None = None,
        task_id: str | None = None,
    ) -> RepairTaskBundle:
        """
        Generate a single verified Task C bundle with exactly one winning candidate.
        """
        rng = np.random.default_rng(seed)
        tid = task_id or f"task_repair_{seed}"

        # Choose winner slot: 'A', 'B', or 'C'
        winner_slot = force_winner_slot or rng.choice(["A", "B", "C"])

        for attempt in range(1, self.max_attempts + 1):
            candidate = self._generate_candidate(
                rng=rng,
                seed=seed + attempt * 1000,
                difficulty=difficulty,
                winner_slot=winner_slot,
                tid=tid,
            )
            if candidate is not None:
                return self._finalize_task_bundle(
                    candidate=candidate,
                    theme=theme,
                    output_dir=output_dir,
                )

        raise RuntimeError(
            f"Failed to generate valid Task C repair scene for seed {seed} after {self.max_attempts} attempts."
        )

    def _generate_candidate(
        self,
        rng: np.random.Generator,
        seed: int,
        difficulty: DifficultyLevel,
        winner_slot: str,
        tid: str,
    ) -> dict | None:
        """Procedurally generate 3 distinct mirror candidates and verify strict uniqueness."""
        # 1. Base scene: Winning mirror provides clear valid specular reflection
        base_scene = self.scene_generator.generate_scene(
            seed=seed,
            category=SceneCategory.VALID_VISIBLE,
            difficulty=difficulty,
            scene_id=f"base_{tid}",
        )

        cam_pt = np.array([base_scene.camera.position.x, base_scene.camera.position.y])
        tgt_pt = np.array([base_scene.target.center.x, base_scene.target.center.y])

        m_a = np.array([base_scene.mirror.endpoint_a.x, base_scene.mirror.endpoint_a.y])
        m_b = np.array([base_scene.mirror.endpoint_b.x, base_scene.mirror.endpoint_b.y])
        m_norm = np.array(
            [
                base_scene.mirror.reflective_normal.dx,
                base_scene.mirror.reflective_normal.dy,
            ]
        )

        m_dir = m_b - m_a
        m_len = vector_magnitude(m_dir)
        t_unit = vector_normalize(m_dir)
        m_center = (m_a + m_b) / 2.0
        half_len = m_len / 2.0

        # Partition wall obstacle
        partition_spec = base_scene.obstacles[0]
        partition_obs = RectangleObstacle(
            obstacle_id="partition_wall",
            center=[partition_spec.center.x, partition_spec.center.y],
            width=partition_spec.width,
            height=partition_spec.height,
            angle_degrees=partition_spec.angle_degrees or 0.0,
        )
        obstacles: list[ObstacleBase] = [partition_obs]

        # 2. Candidate 1: Winning Mirror
        win_spec = base_scene.mirror

        # 3. Candidate 2: Shifted Mirror (Misses segment)
        # Shift along tangent by half_len + 35..65 mm
        shift_dir = 1.0 if rng.random() > 0.5 else -1.0
        shift_dist = (half_len + rng.uniform(35.0, 65.0)) * shift_dir
        shift_center = m_center + shift_dist * t_unit

        # Check bounds
        if not (
            100.0 < shift_center[0] < self.canvas_width - 100.0
            and 100.0 < shift_center[1] < self.canvas_height - 100.0
        ):
            shift_dist = -shift_dist
            shift_center = m_center + shift_dist * t_unit
            if not (
                100.0 < shift_center[0] < self.canvas_width - 100.0
                and 100.0 < shift_center[1] < self.canvas_height - 100.0
            ):
                return None

        m_shift_a = shift_center - half_len * t_unit
        m_shift_b = shift_center + half_len * t_unit
        lose_spec_shift = MirrorSpec(
            endpoint_a=Point2D(x=float(m_shift_a[0]), y=float(m_shift_a[1])),
            endpoint_b=Point2D(x=float(m_shift_b[0]), y=float(m_shift_b[1])),
            reflective_normal=Vector2D(dx=float(m_norm[0]), dy=float(m_norm[1])),
        )

        # 4. Candidate 3: Rotated Mirror (Tilted so reflection ray is wrong)
        tilt_dir = 1.0 if rng.random() > 0.5 else -1.0
        tilt_deg = rng.uniform(18.0, 32.0) * tilt_dir
        tilt_rad = math.radians(tilt_deg)

        tilt_t = vector_rotate(t_unit, tilt_rad)
        tilt_n = vector_rotate(m_norm, tilt_rad)

        # Optionally offset center slightly in Y to prevent intersection
        tilt_center = m_center + np.array([0.0, rng.uniform(15.0, 35.0)])
        m_tilt_a = tilt_center - half_len * tilt_t
        m_tilt_b = tilt_center + half_len * tilt_t

        lose_spec_tilt = MirrorSpec(
            endpoint_a=Point2D(x=float(m_tilt_a[0]), y=float(m_tilt_a[1])),
            endpoint_b=Point2D(x=float(m_tilt_b[0]), y=float(m_tilt_b[1])),
            reflective_normal=Vector2D(dx=float(tilt_n[0]), dy=float(tilt_n[1])),
        )

        # 5. Verify geometric distinctness between all candidate pairs
        candidates_raw = [win_spec, lose_spec_shift, lose_spec_tilt]
        for i in range(len(candidates_raw)):
            for j in range(i + 1, len(candidates_raw)):
                c1 = candidates_raw[i]
                c2 = candidates_raw[j]
                mid1 = np.array(
                    [
                        (c1.endpoint_a.x + c1.endpoint_b.x) / 2.0,
                        (c1.endpoint_a.y + c1.endpoint_b.y) / 2.0,
                    ]
                )
                mid2 = np.array(
                    [
                        (c2.endpoint_a.x + c2.endpoint_b.x) / 2.0,
                        (c2.endpoint_a.y + c2.endpoint_b.y) / 2.0,
                    ]
                )
                dist = np.linalg.norm(mid1 - mid2)

                t1 = np.array(
                    [
                        c1.endpoint_b.x - c1.endpoint_a.x,
                        c1.endpoint_b.y - c1.endpoint_a.y,
                    ]
                )
                t2 = np.array(
                    [
                        c2.endpoint_b.x - c2.endpoint_a.x,
                        c2.endpoint_b.y - c2.endpoint_a.y,
                    ]
                )
                cos_ang = abs(
                    float(np.dot(t1, t2) / (np.linalg.norm(t1) * np.linalg.norm(t2)))
                )
                angle_diff_deg = math.degrees(math.acos(min(1.0, cos_ang)))

                if dist < 22.0 and angle_diff_deg < 10.0:
                    return None  # Too close / visually ambiguous

        # 6. Assign to Slots ('A', 'B', 'C')
        slots = ["A", "B", "C"]
        losing_slots = [s for s in slots if s != winner_slot]
        rng.shuffle(losing_slots)

        candidate_map: dict[str, MirrorSpec] = {
            winner_slot: win_spec,
            losing_slots[0]: lose_spec_shift,
            losing_slots[1]: lose_spec_tilt,
        }

        # 7. Independent Oracle Evaluation for Every Slot
        eval_map: dict[str, RepairCandidateEvaluation] = {}
        visible_count = 0

        for slot_id in slots:
            m = candidate_map[slot_id]
            res = evaluate_visibility_oracle(
                camera_pos=cam_pt,
                camera_forward=base_scene.camera.heading_deg,
                camera_fov_degrees=base_scene.camera.fov_degrees,
                target_pos=tgt_pt,
                mirror_a=[m.endpoint_a.x, m.endpoint_a.y],
                mirror_b=[m.endpoint_b.x, m.endpoint_b.y],
                obstacles=obstacles,
                reflective_normal=[m.reflective_normal.dx, m.reflective_normal.dy],
            )

            reason_code = OBSTRUCTION_TO_REASON_CODE.get(
                res.reason, VisibilityReasonCode.CLEAR
            )
            refl_pt = (
                Point2D(
                    x=float(res.reflection_point[0]), y=float(res.reflection_point[1])
                )
                if res.reflection_point is not None
                else None
            )

            eval_map[slot_id] = RepairCandidateEvaluation(
                candidate_id=slot_id,
                mirror=m,
                is_visible=res.is_visible,
                reason_code=reason_code,
                diagnostic_details=res.diagnostic_details,
                reflection_point=refl_pt,
                incident_angle_deg=res.incident_angle_deg,
                reflected_angle_deg=res.reflected_angle_deg,
            )

            if res.is_visible:
                visible_count += 1

        # 8. Strict Uniqueness Check: Exactly ONE candidate must restore visibility
        if visible_count != 1:
            return None
        if eval_map[winner_slot].is_visible is not True:
            return None

        return {
            "tid": tid,
            "seed": seed,
            "difficulty": difficulty,
            "winner_slot": winner_slot,
            "base_scene": base_scene,
            "candidate_map": candidate_map,
            "eval_map": eval_map,
        }

    def _finalize_task_bundle(
        self,
        candidate: dict,
        theme: RenderTheme,
        output_dir: Path | None,
    ) -> RepairTaskBundle:
        """Assemble typed RepairTaskBundle and save all 5 artifacts."""
        tid = candidate["tid"]
        seed = candidate["seed"]
        diff = candidate["difficulty"]
        winner_slot = candidate["winner_slot"]
        base_scene: InspectionScene = candidate["base_scene"]
        candidate_map: dict[str, MirrorSpec] = candidate["candidate_map"]
        eval_map: dict[str, RepairCandidateEvaluation] = candidate["eval_map"]

        win_eval = eval_map[winner_slot]
        losing_slots = [s for s in ["A", "B", "C"] if s != winner_slot]

        # 1. Detailed Explanation Derivation
        explanation = (
            f"Mathematical Proof: Candidate [{winner_slot}] uniquely restores reflected visibility. "
            f"Candidate [{winner_slot}]: VISIBLE at reflection point ({win_eval.reflection_point.x:.1f}, {win_eval.reflection_point.y:.1f}) "
            f"with theta_i=theta_r={win_eval.incident_angle_deg:.1f} deg. "
            f"Candidate [{losing_slots[0]}]: FAILS ({eval_map[losing_slots[0]].reason_code.value} - {eval_map[losing_slots[0]].diagnostic_details}). "
            f"Candidate [{losing_slots[1]}]: FAILS ({eval_map[losing_slots[1]].reason_code.value} - {eval_map[losing_slots[1]].diagnostic_details})."
        )

        ground_truth = RepairGroundTruth(
            task_id=tid,
            correct_candidate=winner_slot,
            detailed_explanation=explanation,
            candidates=eval_map,
        )

        # 2. Public Question
        question = RepairQuestion(
            task_id=tid,
            question="Which candidate mirror configuration (A, B, or C) successfully restores the camera's view of the red X target scratch?",
            prompt=DEFAULT_REPAIR_PROMPT,
            image_filename="diagram_blind.png",
            options=["A", "B", "C"],
        )

        # 3. Render Images
        renderer = PillowDiagramRenderer(
            theme=theme,
            image_width=self.image_size,
            image_height=self.image_size,
        )
        img_blind = renderer.render_repair(
            scene=base_scene,
            candidate_mirrors=candidate_map,
            correct_candidate=winner_slot,
            mode=RenderMode.BLIND,
        )
        img_overlay = renderer.render_repair(
            scene=base_scene,
            candidate_mirrors=candidate_map,
            correct_candidate=winner_slot,
            mode=RenderMode.OVERLAY,
        )

        # 4. Save to output directory if requested
        blind_path_str = ""
        overlay_path_str = ""
        if output_dir is not None:
            task_dir = Path(output_dir) / tid
            task_dir.mkdir(parents=True, exist_ok=True)

            with open(task_dir / "scene.json", "w", encoding="utf-8") as f:
                json.dump(base_scene.model_dump(mode="json"), f, indent=2)

            blind_file = task_dir / "diagram_blind.png"
            img_blind.save(blind_file, format="PNG")
            blind_path_str = str(blind_file.resolve())

            with open(task_dir / "question.json", "w", encoding="utf-8") as f:
                json.dump(question.model_dump(mode="json"), f, indent=2)

            with open(task_dir / "ground_truth.json", "w", encoding="utf-8") as f:
                json.dump(ground_truth.model_dump(mode="json"), f, indent=2)

            with open(task_dir / "answer.json", "w", encoding="utf-8") as f:
                json.dump(ground_truth.model_dump(mode="json"), f, indent=2)

            overlay_file = task_dir / "diagram_overlay.png"
            img_overlay.save(overlay_file, format="PNG")
            overlay_path_str = str(overlay_file.resolve())

        return RepairTaskBundle(
            task_id=tid,
            seed=seed,
            difficulty=diff,
            theme=theme.value,
            scene=base_scene,
            candidate_mirrors=candidate_map,
            question=question,
            ground_truth=ground_truth,
            blind_image_path=blind_path_str,
            overlay_image_path=overlay_path_str,
        )

    def create_balanced_dataset(
        self,
        count: int = 30,
        start_seed: int = 30000,
        output_dir: Path | None = None,
        difficulties: list[DifficultyLevel] | None = None,
        themes: list[RenderTheme] | None = None,
    ) -> list[RepairTaskBundle]:
        """
        Generate a balanced dataset across correct option slots (A, B, and C).
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

        slots = ["A", "B", "C"]
        tasks: list[RepairTaskBundle] = []

        for i in range(count):
            seed = start_seed + i
            diff = diff_list[i % len(diff_list)]
            theme = theme_list[i % len(theme_list)]
            winner = slots[i % len(slots)]

            task = self.create_task(
                seed=seed,
                difficulty=diff,
                force_winner_slot=winner,
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
