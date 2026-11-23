"""
Visibility Task Factory for MIRRORLAB.

Generates complete, self-contained multimodal benchmark task bundles for:
Task A: Reflected Visibility Prediction

Each task generates exactly:
1. scene.json: The single source of truth geometry specification.
2. diagram_blind.png: High-DPI schematic strictly omitting answer proof.
3. question.json: Multimodal question given to vision models.
4. ground_truth.json: Mathematically verified answer key and proof geometry.
5. diagram_overlay.png: Complete visual explanation with optical ray paths.
"""

from __future__ import annotations

import json
from pathlib import Path

from mirrorlab.generation.scene_generator import SceneGenerator
from mirrorlab.rendering.pillow_renderer import PillowDiagramRenderer
from mirrorlab.rendering.themes import RenderMode, RenderTheme
from mirrorlab.schema.common import VisibilityState
from mirrorlab.schema.scene import DifficultyLevel, InspectionScene, SceneCategory
from mirrorlab.schema.task import (
    OBSTRUCTION_TO_REASON_CODE,
    ProofGeometry,
    VisibilityGroundTruth,
    VisibilityQuestion,
    VisibilityReasonCode,
    VisibilityTaskBundle,
)

DEFAULT_VISIBILITY_PROMPT = """You are an industrial machine-vision inspector analyzing a 2D optical inspection diagram.

Apparatus Legend:
- CAMERA: Represented by the blue camera symbol with a directional lens pointing along its optical axis. The translucent cone represents the camera's field of view (FOV).
- MIRROR: Finite flat mirror segment between endpoints A and B. The reflective front face is marked with a vibrant cyan line. The non-reflective back face is marked with diagonal hatch lines.
- TARGET DEFECT: A red 'X' scratch defect in the cavity.
- OBSTACLES: Opaque polygonal barriers labeled O1, O2, etc. Direct line-of-sight between camera and scratch is blocked by barrier O1, so inspection is only possible via specular reflection in the mirror.

Law of Reflection:
Light travels in straight lines. Specular reflection occurs at the mirror surface where angle of incidence equals angle of reflection (theta_i = theta_r) relative to the mirror surface normal.

Question:
Can the camera see the center of the red X through the mirror?

Analyze:
1. Does the specular reflection ray from the camera to the target strike within the finite mirror segment between endpoints A and B?
2. Is the reflection point on the mirror within the camera's field of view cone?
3. Does the mirror's reflective face (cyan) face towards the camera and target (not the hatched back side)?
4. Is the optical path blocked by any opaque obstacle (O1, O2, O3) along either the incident leg (camera to mirror) or the reflected leg (mirror to target)?

Respond in JSON format with:
{
  "can_see": true or false,
  "answer": "YES" or "NO",
  "reason": "CLEAR" | "MIRROR_MISS" | "OUTSIDE_FOV" | "WRONG_SIDE" | "BLOCKED_CAMERA_TO_MIRROR" | "BLOCKED_MIRROR_TO_TARGET",
  "explanation": "<your step-by-step reasoning>"
}
"""


class VisibilityTaskFactory:
    """
    Factory for producing complete, self-contained Visibility inspection tasks.
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

    def create_bundle_from_scene(
        self,
        scene: InspectionScene,
        theme: RenderTheme = RenderTheme.DARK_NAVY,
        output_dir: Path | None = None,
        task_id: str | None = None,
        target_dir: Path | None = None,
    ) -> VisibilityTaskBundle:
        """
        Create a full 5-artifact bundle from an already instantiated and oracle-evaluated scene.
        """
        tid = task_id or scene.scene_id

        # 2. Build Ground-Truth Answer Key
        can_see = scene.ground_truth_state == VisibilityState.VISIBLE
        answer_str = "YES" if can_see else "NO"
        reason_code = OBSTRUCTION_TO_REASON_CODE.get(
            scene.ground_truth_reason, VisibilityReasonCode.CLEAR
        )

        explanation = (
            f"Mathematical Proof: {scene.diagnostic_details}. "
            f"Ground truth state is {scene.ground_truth_state.value} with reason {reason_code.value}."
        )

        proof_geo = ProofGeometry(
            reflection_point=scene.reflection_point,
            virtual_target=scene.virtual_target,
            incident_angle_deg=scene.incident_angle_deg,
            reflected_angle_deg=scene.reflected_angle_deg,
            total_optical_distance=scene.total_optical_distance,
            optical_path=scene.optical_path,
        )

        ground_truth = VisibilityGroundTruth(
            task_id=tid,
            can_see=can_see,
            answer=answer_str,
            reason_code=reason_code,
            detailed_explanation=explanation,
            proof_geometry=proof_geo,
        )

        # 3. Build Public Question Schema
        question = VisibilityQuestion(
            task_id=tid,
            question="Can the camera see the center of the red X through the mirror?",
            prompt=DEFAULT_VISIBILITY_PROMPT,
            image_filename="diagram_blind.png",
        )

        # 4. Render Images (Blind diagram and Explanation overlay)
        renderer = PillowDiagramRenderer(
            theme=theme,
            image_width=self.image_size,
            image_height=self.image_size,
        )
        img_blind = renderer.render(scene, mode=RenderMode.BLIND)
        img_overlay = renderer.render(scene, mode=RenderMode.OVERLAY)

        # 5. Persist to Output Directory if provided
        blind_path_str = ""
        overlay_path_str = ""
        effective_task_dir = None
        if target_dir is not None:
            effective_task_dir = Path(target_dir)
        elif output_dir is not None:
            effective_task_dir = Path(output_dir) / tid

        if effective_task_dir is not None:
            effective_task_dir.mkdir(parents=True, exist_ok=True)

            # 1. scene.json
            with open(effective_task_dir / "scene.json", "w", encoding="utf-8") as f:
                json.dump(scene.model_dump(mode="json"), f, indent=2)

            # 2. diagram_blind.png
            blind_file = effective_task_dir / "diagram_blind.png"
            img_blind.save(blind_file, format="PNG")
            blind_path_str = str(blind_file.resolve())

            # 3. question.json
            with open(effective_task_dir / "question.json", "w", encoding="utf-8") as f:
                json.dump(question.model_dump(mode="json"), f, indent=2)

            # 4. ground_truth.json
            with open(
                effective_task_dir / "ground_truth.json", "w", encoding="utf-8"
            ) as f:
                json.dump(ground_truth.model_dump(mode="json"), f, indent=2)

            # 5. diagram_overlay.png
            overlay_file = effective_task_dir / "diagram_overlay.png"
            img_overlay.save(overlay_file, format="PNG")
            overlay_path_str = str(overlay_file.resolve())

        return VisibilityTaskBundle(
            task_id=tid,
            seed=scene.seed,
            category=scene.category,
            difficulty=scene.difficulty,
            theme=theme.value,
            scene=scene,
            question=question,
            ground_truth=ground_truth,
            blind_image_path=blind_path_str,
            overlay_image_path=overlay_path_str,
        )

    def create_task(
        self,
        seed: int,
        category: SceneCategory = SceneCategory.VALID_VISIBLE,
        difficulty: DifficultyLevel = DifficultyLevel.MEDIUM,
        theme: RenderTheme = RenderTheme.DARK_NAVY,
        output_dir: Path | None = None,
        task_id: str | None = None,
    ) -> VisibilityTaskBundle:
        """
        Generate a single verified task with all 5 required artifacts.
        """
        tid = task_id or f"task_vis_{seed:05d}"

        # 1. Generate Scene Geometry verified by Oracle
        scene = self.scene_generator.generate_scene(
            seed=seed,
            category=category,
            difficulty=difficulty,
            scene_id=tid,
        )

        return self.create_bundle_from_scene(
            scene=scene,
            theme=theme,
            output_dir=output_dir,
            task_id=tid,
        )

    def create_balanced_dataset(
        self,
        count: int,
        start_seed: int = 10000,
        output_dir: Path | None = None,
        difficulties: list[DifficultyLevel] | None = None,
        themes: list[RenderTheme] | None = None,
    ) -> list[VisibilityTaskBundle]:
        """
        Create a balanced benchmark dataset with 50% VISIBLE and 50% NOT_VISIBLE (divided across the 5 failure causes).
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

        # 50% VISIBLE, 50% NOT_VISIBLE across the 5 obstacle/miss categories
        not_visible_cats = [
            SceneCategory.MIRROR_MISS,
            SceneCategory.OUTSIDE_FOV,
            SceneCategory.WRONG_SIDE,
            SceneCategory.BLOCKED_INCIDENT,
            SceneCategory.BLOCKED_REFLECTED,
        ]

        tasks: list[VisibilityTaskBundle] = []
        for i in range(count):
            seed = start_seed + i
            diff = diff_list[i % len(diff_list)]
            theme = theme_list[i % len(theme_list)]

            # Even index: VISIBLE; Odd index: Cycle through NOT_VISIBLE causes
            if i % 2 == 0:
                cat = SceneCategory.VALID_VISIBLE
            else:
                cat_idx = (i // 2) % len(not_visible_cats)
                cat = not_visible_cats[cat_idx]

            task = self.create_task(
                seed=seed,
                category=cat,
                difficulty=diff,
                theme=theme,
                output_dir=output_dir,
            )
            tasks.append(task)

        # Write aggregate manifests if output_dir provided
        if output_dir is not None:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            # Public tasks manifest (JSONL)
            with open(out_path / "tasks.jsonl", "w", encoding="utf-8") as f:
                f.writelines(t.question.model_dump_json() + "\n" for t in tasks)

            # Private ground truth manifest (JSONL)
            with open(out_path / "ground_truth.jsonl", "w", encoding="utf-8") as f:
                f.writelines(t.ground_truth.model_dump_json() + "\n" for t in tasks)

        return tasks
