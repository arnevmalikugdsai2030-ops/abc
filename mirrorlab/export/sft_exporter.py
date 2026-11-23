"""
Supervised Fine-Tuning (SFT) Exporter for MIRRORLAB.

Converts machine-vision inspection scenes and ground-truth oracle derivations
into high-density, multi-turn reasoning traces for downstream VLM training
(LLaVA, Qwen2-VL, Gemini, OpenAI JSONL formats).

Every generated training example includes:
1. Standardized neutral prompt requesting optical inspection.
2. Exact step-by-step physical and geometric derivation:
   - Step 1: Optical apparatus identification & coordinate extraction.
   - Step 2: Virtual target point construction via specular plane reflection.
   - Step 3: Ray line intersection with finite mirror aperture segment.
   - Step 4: Mirror normal orientation & non-reflective backside check.
   - Step 5: Camera field-of-view cone angular bounds check.
   - Step 6: Two-leg obstacle occlusion audit (Incident & Reflected rays).
   - Step 7: Final structured JSON verdict.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from mirrorlab.geometry.oracle import VisibilityOracle
from mirrorlab.schema.common import ObstructionReason, VisibilityState
from mirrorlab.schema.scene import InspectionScene


class SFTMessage(BaseModel):
    """A single role-based message in an SFT conversation."""

    role: str = Field(description="'system', 'user', or 'assistant'")
    content: str = Field(description="Textual dialogue content")


class SFTExample(BaseModel):
    """Standardized multi-modal SFT training example."""

    id: str
    image: str = Field(description="Relative path or filename of inspection diagram")
    messages: list[SFTMessage]
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_huggingface_dict(self) -> dict[str, Any]:
        """Formats example for HuggingFace / Axolotl / LLaVA datasets."""
        return {
            "id": self.id,
            "image": self.image,
            "conversations": [
                {"from": m.role if m.role != "assistant" else "gpt", "value": m.content}
                for m in self.messages
            ],
            "metadata": self.metadata,
        }

    def to_openai_dict(self) -> dict[str, Any]:
        """Formats example for OpenAI / Gemini multi-modal fine-tuning."""
        return {
            "messages": [{"role": m.role, "content": m.content} for m in self.messages],
            "metadata": {"task_id": self.id, "image": self.image, **self.metadata},
        }


class SFTDatasetExporter:
    """
    Exports MIRRORLAB benchmark suites into verified SFT training datasets.
    """

    SYSTEM_PROMPT = (
        "You are an expert automated optical inspection system specializing in machine vision, "
        "specular geometry, and rigorous ray-tracing analysis."
    )

    USER_PROMPT_TEMPLATE = (
        "<image>\n"
        "Inspect the 2D machine-vision schematic diagram.\n"
        "A red 'X' indicates a target scratch defect on the specimen.\n"
        "Determine whether the target scratch is visible to the camera via reflection off the planar mirror.\n"
        "Provide your formal step-by-step geometric derivation, followed by your final structured JSON verdict."
    )

    @classmethod
    def generate_reasoning_trace(cls, scene: InspectionScene) -> str:
        """
        Synthesizes a verifiable, mathematically rigorous step-by-step reasoning trace.
        """
        cam = scene.camera
        mirror = scene.mirror
        target = scene.target
        oracle_verdict = VisibilityOracle.evaluate(scene)

        c_x, c_y = cam.position.x, cam.position.y
        t_x, t_y = target.center.x, target.center.y
        ma_x, ma_y = mirror.endpoint_a.x, mirror.endpoint_a.y
        mb_x, mb_y = mirror.endpoint_b.x, mirror.endpoint_b.y
        n_x, n_y = mirror.reflective_normal.dx, mirror.reflective_normal.dy

        trace_lines = [
            "### Formal Geometric & Optical Derivation",
            "",
            "**Step 1: Coordinate Extraction & Optical Apparatus**",
            f"- Camera Position $C = ({c_x:.1f}, {c_y:.1f})$, Heading $\\theta_c = {cam.heading_deg:.1f}^\\circ$, $\\text{{FOV}} = {cam.fov_degrees:.1f}^\\circ$",
            f"- Mirror Segment $AB$: $A = ({ma_x:.1f}, {ma_y:.1f})$, $B = ({mb_x:.1f}, {mb_y:.1f})$, Reflective Normal $\\mathbf{{n}} = ({n_x:.2f}, {n_y:.2f})$",
            f"- Target Defect $T = ({t_x:.1f}, {t_y:.1f})$, Radius $r = {target.radius_mm:.1f}\\text{{ mm}}$",
            f"- Scene Obstacles: {len(scene.obstacles)} elements present in workspace.",
            "",
            "**Step 2: Virtual Target Position Construction**",
        ]

        # Virtual target point
        if scene.virtual_target is not None:
            vt_x, vt_y = scene.virtual_target.x, scene.virtual_target.y
            trace_lines.append(
                f"- Reflect target $T$ across the infinite plane of mirror $AB$ using Householder reflection:\n"
                f"  $$T' = T - 2 ((T - A) \\cdot \\mathbf{{n}}) \\mathbf{{n}} = ({vt_x:.2f}, {vt_y:.2f})$$"
            )
        else:
            trace_lines.append(
                "- Virtual image calculation indicates target is collinear or non-reflectable."
            )

        trace_lines.append("")
        trace_lines.append(
            "**Step 3: Ray-Mirror Intersection & Finite Aperture Verification**"
        )

        if scene.reflection_point is not None:
            rx, ry = scene.reflection_point.x, scene.reflection_point.y
            trace_lines.append(
                f"- The direct sightline from camera $C$ to virtual target $T'$ intersects the mirror segment at:\n"
                f"  $$R = ({rx:.2f}, {ry:.2f})$$\n"
                f"- The reflection point $R$ lies strictly within the finite segment bounds $A \\to B$."
            )
        else:
            trace_lines.append(
                "- The line connecting camera $C$ and virtual target $T'$ does NOT intersect the finite segment $AB$.\n"
                "- Outcome: The ray misses the mirror aperture (Infinite Plane Fallacy trigger)."
            )

        trace_lines.append("")
        trace_lines.append("**Step 4: Surface Normal & Reflective Side Orientation**")
        if oracle_verdict.reason == ObstructionReason.BACK_OF_MIRROR:
            trace_lines.append(
                "- Ray vector $(R - C)$ strikes the backside of the mirror:\n"
                "  $(R - C) \\cdot \\mathbf{n} > 0$. The reverse surface is non-reflective."
            )
        else:
            trace_lines.append(
                "- Ray vector satisfies incidence condition: $(R - C) \\cdot \\mathbf{n} < 0$.\n"
                "- Ray strikes the polished specular reflective surface."
            )

        trace_lines.append("")
        trace_lines.append("**Step 5: Camera Field-of-View Cone Check**")
        if "OUTSIDE_FOV" in str(scene.ground_truth_reason):
            trace_lines.append(
                f"- Angle between camera optical axis and ray $(R - C)$ exceeds half-angle {cam.fov_degrees / 2.0:.1f}^\\circ.\n"
                "- Target reflection lies outside the camera's sensor field of view."
            )
        else:
            trace_lines.append(
                f"- Angle between optical axis and ray $(R - C)$ lies within the {cam.fov_degrees:.1f}^\\circ FOV cone."
            )

        trace_lines.append("")
        trace_lines.append("**Step 6: Two-Leg Occlusion Analysis**")
        if scene.ground_truth_reason == ObstructionReason.BLOCKED_INCIDENT:
            trace_lines.append(
                "- Leg 1 ($C \\to R$, Camera to Mirror): Intersected by obstacle.\n"
                "- The incident ray is physically occluded before reaching the mirror surface."
            )
        elif scene.ground_truth_reason == ObstructionReason.BLOCKED_REFLECTED:
            trace_lines.append(
                "- Leg 1 ($C \\to R$): Clear.\n"
                "- Leg 2 ($R \\to T$, Mirror to Target): Intersected by obstacle.\n"
                "- The reflected ray is physically occluded between the mirror and defect."
            )
        elif scene.ground_truth_reason == ObstructionReason.NONE:
            trace_lines.append(
                "- Leg 1 ($C \\to R$): Clear of all obstacles.\n"
                "- Leg 2 ($R \\to T$): Clear of all obstacles.\n"
                "- Unobstructed specular optical corridor confirmed."
            )
        else:
            trace_lines.append(f"- Occlusion state: {scene.ground_truth_reason.value}.")

        trace_lines.append("")
        trace_lines.append("**Step 7: Conclusion & Verdict**")

        vis_str = (
            "VISIBLE"
            if scene.ground_truth_state == VisibilityState.VISIBLE
            else "NOT_VISIBLE"
        )
        verdict_json = {
            "visibility_state": vis_str,
            "obstruction_reason": scene.ground_truth_reason.value,
            "diagnostic_summary": scene.diagnostic_details,
        }

        trace_lines.append("```json")
        trace_lines.append(json.dumps(verdict_json, indent=2))
        trace_lines.append("```")

        return "\n".join(trace_lines)

    @classmethod
    def export_scene(
        cls,
        scene: InspectionScene,
        image_relative_path: str = "",
    ) -> SFTExample:
        """
        Exports a single scene to an SFTExample.
        """
        image_file = image_relative_path or f"{scene.scene_id}_blind.png"
        assistant_content = cls.generate_reasoning_trace(scene)

        messages = [
            SFTMessage(role="system", content=cls.SYSTEM_PROMPT),
            SFTMessage(role="user", content=cls.USER_PROMPT_TEMPLATE),
            SFTMessage(role="assistant", content=assistant_content),
        ]

        metadata = {
            "scene_id": scene.scene_id,
            "category": scene.category.value
            if hasattr(scene.category, "value")
            else str(scene.category),
            "ground_truth_state": scene.ground_truth_state.value
            if hasattr(scene.ground_truth_state, "value")
            else str(scene.ground_truth_state),
            "ground_truth_reason": scene.ground_truth_reason.value
            if hasattr(scene.ground_truth_reason, "value")
            else str(scene.ground_truth_reason),
            "difficulty": scene.difficulty.value
            if hasattr(scene.difficulty, "value")
            else str(scene.difficulty),
        }

        return SFTExample(
            id=scene.scene_id,
            image=image_file,
            messages=messages,
            metadata=metadata,
        )

    @classmethod
    def export_to_jsonl(
        cls,
        scenes: Sequence[InspectionScene],
        output_file: Path | str,
        format_type: str = "huggingface",
    ) -> int:
        """
        Exports a batch of scenes to a JSONL file.
        """
        out_p = Path(output_file)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(out_p, "w", encoding="utf-8") as f:
            for scene in scenes:
                example = cls.export_scene(scene)
                record = (
                    example.to_huggingface_dict()
                    if format_type == "huggingface"
                    else example.to_openai_dict()
                )
                f.write(json.dumps(record) + "\n")
                count += 1

        return count
