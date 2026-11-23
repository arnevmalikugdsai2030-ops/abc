"""
Scene Laboratory Module for MIRRORLAB.

Interactive, real-time machine-vision cavity inspection workbench:
- Procedural scene generation across all 4 task types (Visibility, Obstruction, Repair, Counterfactual).
- Real-time geometric manipulation of camera, mirror endpoints, defect target, and obstacles.
- Instant recalculation through the deterministic VisibilityOracle.
- Switchable blind diagram vs reference path overlay.
- Complete mathematical proof and telemetry derivation.
- In-memory ZIP task package exporter.
"""

from __future__ import annotations

import io
import json
import math
import random
import zipfile
from datetime import UTC, datetime
from typing import Any

import streamlit as st

from mirrorlab.generation.counterfactual_factory import CounterfactualFactory
from mirrorlab.generation.obstruction_factory import ObstructionTaskFactory
from mirrorlab.generation.repair_factory import RepairTaskFactory
from mirrorlab.generation.scene_generator import SceneGenerator
from mirrorlab.generation.task_factory import (
    DEFAULT_VISIBILITY_PROMPT,
)
from mirrorlab.geometry.oracle import VisibilityOracle, VisibilityOracleResult
from mirrorlab.rendering.pillow_renderer import PillowDiagramRenderer
from mirrorlab.rendering.themes import RenderMode, RenderTheme
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
from mirrorlab.schema.task import (
    OBSTRUCTION_TO_REASON_CODE,
    BlockedSegment,
    ProofGeometry,
    VisibilityGroundTruth,
    VisibilityQuestion,
    VisibilityReasonCode,
)
from mirrorlab.ui.components import (
    render_telemetry_strip,
    render_verdict_badge,
)
from mirrorlab.ui.data_loader import load_tasks_by_family
from mirrorlab.ui.design_tokens import (
    COLOR_CYAN_PRIMARY,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
)
from mirrorlab.ui.optical_plot import create_optical_figure


# ---------------------------------------------------------------------------
# Session State Helpers
# ---------------------------------------------------------------------------
def _init_scene_lab_state() -> None:
    """Ensure session state defaults are initialized."""
    if "scene_lab_scene" not in st.session_state:
        # Default starting scene: standard unobstructed reflection
        sg = SceneGenerator()
        try:
            default_scene = sg.generate_scene(
                seed=42,
                category=SceneCategory.VALID_VISIBLE,
                difficulty=DifficultyLevel.STANDARD,
                scene_id="scene_lab_init_042",
            )
        except (RuntimeError, ValueError):
            # Fallback direct construction
            cam = CameraSpec(
                position=Point2D(x=150.0, y=200.0), heading_deg=35.0, fov_degrees=60.0
            )
            mir = MirrorSpec(
                endpoint_a=Point2D(x=300.0, y=800.0),
                endpoint_b=Point2D(x=700.0, y=800.0),
                reflective_normal=Vector2D(dx=0.0, dy=-1.0),
            )
            tgt = TargetSpec(center=Point2D(x=800.0, y=250.0), radius_mm=8.0)
            obs = [
                ObstacleSpec(
                    obstacle_id="O1",
                    obstacle_type="rectangle",
                    center=Point2D(x=500.0, y=220.0),
                    width=70.0,
                    height=45.0,
                    angle_degrees=0.0,
                )
            ]
            default_scene = InspectionScene(
                scene_id="scene_lab_init_042",
                seed=42,
                category=SceneCategory.VALID_VISIBLE,
                difficulty=DifficultyLevel.STANDARD,
                canvas_width=1000.0,
                canvas_height=1000.0,
                camera=cam,
                mirror=mir,
                target=tgt,
                obstacles=obs,
                ground_truth_state=VisibilityState.VISIBLE,
                ground_truth_reason=ObstructionReason.NONE,
                diagnostic_details="Initial laboratory setup",
            )
        st.session_state["scene_lab_scene"] = default_scene
        st.session_state["scene_lab_task_type"] = "Task A: Reflected Visibility"
        st.session_state["scene_lab_seed"] = 42
        st.session_state["scene_lab_difficulty"] = DifficultyLevel.STANDARD
        st.session_state["scene_lab_view_mode"] = "🔬 Reference Overlay"
        st.session_state["scene_lab_bundle"] = None
        st.session_state["scene_lab_repair_candidates"] = None
        st.session_state["scene_lab_repair_active"] = "A"
        st.session_state["scene_lab_cf_pair"] = None
        st.session_state["scene_lab_cf_active"] = "Scene A"


def _recalculate_scene_oracle(
    scene: InspectionScene,
) -> tuple[InspectionScene, VisibilityOracleResult]:
    """
    Run the deterministic visibility oracle on the current scene and return
    an updated scene with synchronized ground truth attributes along with the result.
    """
    res = VisibilityOracle.evaluate(scene)
    vis_state = (
        VisibilityState.VISIBLE if res.is_visible else VisibilityState.NOT_VISIBLE
    )

    refl_pt = None
    if res.reflection_point is not None:
        refl_pt = Point2D(
            x=float(res.reflection_point[0]), y=float(res.reflection_point[1])
        )

    virt_tgt = None
    if res.virtual_target is not None:
        virt_tgt = Point2D(
            x=float(res.virtual_target[0]), y=float(res.virtual_target[1])
        )

    opt_path: list[Point2D] = []
    if refl_pt is not None:
        opt_path = [scene.camera.position, refl_pt, scene.target.center]

    updated_scene = InspectionScene(
        scene_id=scene.scene_id,
        seed=scene.seed,
        category=scene.category,
        difficulty=scene.difficulty,
        canvas_width=scene.canvas_width,
        canvas_height=scene.canvas_height,
        camera=scene.camera,
        mirror=scene.mirror,
        target=scene.target,
        obstacles=scene.obstacles,
        ground_truth_state=vis_state,
        ground_truth_reason=res.reason,
        diagnostic_details=res.diagnostic_details,
        reflection_point=refl_pt,
        virtual_target=virt_tgt,
        incident_angle_deg=res.incident_angle_deg,
        reflected_angle_deg=res.reflected_angle_deg,
        total_optical_distance=res.total_optical_distance,
        optical_path=opt_path,
    )
    return updated_scene, res


# ---------------------------------------------------------------------------
# Task Package Exporter (.ZIP)
# ---------------------------------------------------------------------------
def build_task_package_zip(
    scene: InspectionScene,
    oracle_res: VisibilityOracleResult,
    task_type_str: str,
    bundle_data: Any | None = None,
) -> bytes:
    """
    Compile a complete, self-contained task bundle into an in-memory ZIP archive:
    1. scene.json
    2. question.json
    3. ground_truth.json
    4. diagram_blind.png
    5. diagram_overlay.png
    6. manifest.json
    """
    zip_buf = io.BytesIO()
    tid = scene.scene_id or f"task_{scene.seed}"

    # 1. Question & Ground-Truth assembly
    can_see = oracle_res.is_visible
    reason_code = OBSTRUCTION_TO_REASON_CODE.get(
        oracle_res.reason, VisibilityReasonCode.CLEAR
    )

    proof_geo = ProofGeometry(
        reflection_point=scene.reflection_point,
        virtual_target=scene.virtual_target,
        incident_angle_deg=oracle_res.incident_angle_deg,
        reflected_angle_deg=oracle_res.reflected_angle_deg,
        total_optical_distance=oracle_res.total_optical_distance,
        optical_path=scene.optical_path,
    )

    ground_truth = VisibilityGroundTruth(
        task_id=tid,
        can_see=can_see,
        answer="YES" if can_see else "NO",
        reason_code=reason_code,
        detailed_explanation=f"Mathematical Proof: {oracle_res.diagnostic_details}",
        proof_geometry=proof_geo,
    )

    question = VisibilityQuestion(
        task_id=tid,
        question="Can the camera see the center of the red X through the mirror?",
        prompt=DEFAULT_VISIBILITY_PROMPT,
        image_filename="diagram_blind.png",
    )

    # 2. High-DPI Pillow Images
    renderer = PillowDiagramRenderer(
        theme=RenderTheme.DARK_NAVY, image_width=1024, image_height=1024
    )
    img_blind = renderer.render(scene, mode=RenderMode.BLIND)
    img_overlay = renderer.render(scene, mode=RenderMode.OVERLAY)

    b_buf = io.BytesIO()
    img_blind.save(b_buf, format="PNG")

    o_buf = io.BytesIO()
    img_overlay.save(o_buf, format="PNG")

    # 3. Manifest
    manifest = {
        "task_id": tid,
        "seed": scene.seed,
        "task_type": task_type_str,
        "difficulty": scene.difficulty.value
        if hasattr(scene.difficulty, "value")
        else str(scene.difficulty),
        "is_visible": can_see,
        "reason_code": reason_code.value,
        "generated_at": datetime.now(UTC).isoformat(),
        "generator_engine": "MIRRORLAB 2.0 Specular Physics Engine",
        "artifacts": [
            "scene.json",
            "question.json",
            "ground_truth.json",
            "diagram_blind.png",
            "diagram_overlay.png",
            "manifest.json",
        ],
    }

    # Write ZIP
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            f"{tid}/scene.json", json.dumps(scene.model_dump(mode="json"), indent=2)
        )
        zf.writestr(
            f"{tid}/question.json",
            json.dumps(question.model_dump(mode="json"), indent=2),
        )
        zf.writestr(
            f"{tid}/ground_truth.json",
            json.dumps(ground_truth.model_dump(mode="json"), indent=2),
        )
        zf.writestr(f"{tid}/diagram_blind.png", b_buf.getvalue())
        zf.writestr(f"{tid}/diagram_overlay.png", o_buf.getvalue())
        zf.writestr(f"{tid}/manifest.json", json.dumps(manifest, indent=2))

    return zip_buf.getvalue()


# ---------------------------------------------------------------------------
# Main Scene Laboratory Component
# ---------------------------------------------------------------------------
def render_scene_laboratory() -> None:
    """
    Renders the complete MIRRORLAB Scene Laboratory interactive workspace.
    """
    _init_scene_lab_state()

    st.subheader("🔬 Specular Scene Laboratory & Synthetic Task Factory")
    st.caption(
        "Interactive computer-vision testbed: generate, inspect, manipulate, and export constraint-verified reflection tasks. "
        "Every geometric modification triggers immediate recalculation through the deterministic oracle."
    )

    # =======================================================================
    # SECTION 1: TASK GENERATION TOOLBAR
    # =======================================================================
    with st.expander("⚡ Procedural Scene Generator & Preset Loader", expanded=True):
        col_type, col_diff, col_seed, col_sub, col_btn = st.columns([3, 2, 2, 3, 2])

        with col_type:
            task_type = st.selectbox(
                "Task Type:",
                [
                    "Task A: Reflected Visibility",
                    "Task B: Obstruction Identification",
                    "Task C: Mirror-Setting Repair",
                    "Task D: Counterfactual Pair",
                ],
                index=0,
                key="sel_task_type",
            )

        with col_diff:
            diff_label = st.selectbox(
                "Difficulty:",
                ["STANDARD", "EASY", "HARD", "BOUNDARY"],
                index=0,
                key="sel_diff",
            )
            difficulty = DifficultyLevel[diff_label]

        with col_seed:
            seed_val = st.number_input(
                "Random Seed:",
                min_value=0,
                max_value=999999,
                value=42,
                step=1,
                key="num_seed",
            )
            if st.button("🎲 Random Seed", use_container_width=True):
                seed_val = random.randint(100, 99999)
                st.session_state["num_seed"] = seed_val

        # Sub-options depending on Task Type
        with col_sub:
            if task_type == "Task A: Reflected Visibility":
                sub_opt = st.selectbox(
                    "Target Outcome:",
                    [
                        "Auto (Random)",
                        "VALID_VISIBLE",
                        "MIRROR_MISS",
                        "OUTSIDE_FOV",
                        "WRONG_SIDE",
                        "BLOCKED_INCIDENT",
                        "BLOCKED_REFLECTED",
                    ],
                    index=0,
                )
            elif task_type == "Task B: Obstruction Identification":
                sub_opt = st.selectbox(
                    "Blocked Path Leg:",
                    ["CAMERA_TO_MIRROR", "MIRROR_TO_TARGET", "NONE (Clear Control)"],
                    index=0,
                )
            elif task_type == "Task C: Mirror-Setting Repair":
                sub_opt = st.selectbox(
                    "Correct Winning Setting:",
                    ["Auto (Random)", "Setting A", "Setting B", "Setting C"],
                    index=0,
                )
            else:  # Task D
                sub_opt = st.selectbox(
                    "Counterfactual Subtype:",
                    [
                        "Mirror Angle Flip",
                        "Cosmetic Invariance",
                        "Obstacle Insertion",
                        "Irrelevant Obstacle",
                    ],
                    index=0,
                )

        with col_btn:
            st.write("")  # Alignment spacer
            st.write("")
            generate_clicked = st.button(
                "⚡ Generate Scene", type="primary", use_container_width=True
            )

        # Handle Generation Actions
        if generate_clicked:
            st.session_state["scene_lab_repair_candidates"] = None
            st.session_state["scene_lab_cf_pair"] = None

            try:
                if task_type == "Task A: Reflected Visibility":
                    sg = SceneGenerator()
                    target_cat = SceneCategory.VALID_VISIBLE
                    if sub_opt != "Auto (Random)":
                        target_cat = SceneCategory(sub_opt)
                    new_scene = sg.generate_scene(
                        seed=seed_val,
                        category=target_cat,
                        difficulty=difficulty,
                        scene_id=f"scene_lab_A_{seed_val}",
                    )
                    st.session_state["scene_lab_scene"] = new_scene
                    st.session_state["scene_lab_task_type"] = task_type
                    st.success(
                        f"Generated Task A scene (Seed {seed_val}, {difficulty.value})"
                    )

                elif task_type == "Task B: Obstruction Identification":
                    ob_factory = ObstructionTaskFactory()
                    target_seg = BlockedSegment.CAMERA_TO_MIRROR
                    if "MIRROR_TO_TARGET" in sub_opt:
                        target_seg = BlockedSegment.MIRROR_TO_TARGET
                    elif "NONE" in sub_opt:
                        target_seg = BlockedSegment.NONE

                    bundle = ob_factory.create_task(
                        seed=seed_val,
                        target_segment=target_seg,
                        difficulty=difficulty,
                        num_distractors=1,
                        task_id=f"scene_lab_B_{seed_val}",
                    )
                    st.session_state["scene_lab_scene"] = bundle.scene
                    st.session_state["scene_lab_bundle"] = bundle
                    st.session_state["scene_lab_task_type"] = task_type
                    st.success(
                        f"Generated Task B obstruction scene (Blocker: {bundle.ground_truth.blocking_obstacle_label})"
                    )

                elif task_type == "Task C: Mirror-Setting Repair":
                    rep_factory = RepairTaskFactory()
                    force_slot = None
                    if "Setting A" in sub_opt:
                        force_slot = "A"
                    elif "Setting B" in sub_opt:
                        force_slot = "B"
                    elif "Setting C" in sub_opt:
                        force_slot = "C"

                    bundle = rep_factory.create_task(
                        seed=seed_val,
                        difficulty=difficulty,
                        force_winner_slot=force_slot,
                        task_id=f"scene_lab_C_{seed_val}",
                    )
                    st.session_state["scene_lab_scene"] = bundle.scene
                    st.session_state["scene_lab_bundle"] = bundle
                    st.session_state["scene_lab_repair_candidates"] = {
                        cid: c.mirror
                        for cid, c in bundle.ground_truth.candidates.items()
                    }
                    st.session_state["scene_lab_repair_active"] = (
                        bundle.ground_truth.correct_candidate
                    )
                    st.session_state["scene_lab_task_type"] = task_type
                    st.success(
                        f"Generated Task C repair task (Winning Setting: {bundle.ground_truth.correct_candidate})"
                    )

                elif task_type == "Task D: Counterfactual Pair":
                    cf_factory = CounterfactualFactory()
                    pair_id = f"pair_lab_{seed_val}"
                    if "Mirror Angle Flip" in sub_opt:
                        cf_bundle = cf_factory.create_mirror_angle_flip_pair(
                            seed=seed_val, pair_id=pair_id, difficulty=difficulty
                        )
                    elif "Cosmetic" in sub_opt:
                        cf_bundle = cf_factory.create_cosmetic_invariance_pair(
                            seed=seed_val, pair_id=pair_id, difficulty=difficulty
                        )
                    elif "Irrelevant" in sub_opt:
                        cf_bundle = cf_factory.create_irrelevant_obstacle_pair(
                            seed=seed_val, pair_id=pair_id, difficulty=difficulty
                        )
                    else:
                        cf_bundle = cf_factory.create_obstacle_insertion_pair(
                            seed=seed_val, pair_id=pair_id, difficulty=difficulty
                        )

                    st.session_state["scene_lab_cf_pair"] = cf_bundle
                    st.session_state["scene_lab_scene"] = cf_bundle.scene_a_bundle.scene
                    st.session_state["scene_lab_cf_active"] = "Scene A"
                    st.session_state["scene_lab_task_type"] = task_type
                    pair_label = (
                        cf_bundle.pair_type.value
                        if hasattr(cf_bundle.pair_type, "value")
                        else str(cf_bundle.pair_type)
                    )
                    st.success(f"Generated Counterfactual Pair: {pair_label}")

            except (RuntimeError, ValueError, OSError) as exc:
                st.error(
                    f"Scene generation error: {exc}. Try another seed or difficulty."
                )

        # Preset Loader
        st.divider()
        preset_col1, preset_col2 = st.columns([8, 2])
        with preset_col1:
            family_key = "visibility"
            if "Obstruction" in task_type:
                family_key = "obstruction"
            elif "Repair" in task_type:
                family_key = "repair"
            elif "Counterfactual" in task_type:
                family_key = "counterfactual"

            available_presets = load_tasks_by_family(family_key)
            preset_labels = (
                [p["id"] for p in available_presets] if available_presets else []
            )
            selected_preset = st.selectbox(
                f"Or Load Existing Benchmark Preset ({family_key.upper()}):",
                preset_labels if preset_labels else ["(No disk presets found)"],
                index=0,
                key="sel_preset_id",
            )
        with preset_col2:
            st.write("")
            st.write("")
            if (
                st.button("📂 Load Preset", use_container_width=True)
                and preset_labels
                and selected_preset in preset_labels
            ):
                target_p = next(
                    (p for p in available_presets if p["id"] == selected_preset), None
                )
                if target_p and "scene" in target_p:
                    st.session_state["scene_lab_scene"] = target_p["scene"]
                    st.session_state["scene_lab_repair_candidates"] = None
                    st.session_state["scene_lab_cf_pair"] = None
                    st.success(f"Loaded benchmark scene: {selected_preset}")

    # =======================================================================
    # SECTION 2: CURRENT GEOMETRY SYNCHRONIZATION & ORACLE RECALCULATION
    # =======================================================================
    current_scene: InspectionScene = st.session_state["scene_lab_scene"]
    current_scene, oracle_res = _recalculate_scene_oracle(current_scene)
    st.session_state["scene_lab_scene"] = current_scene

    # Top Status Banner: Verdict & Telemetry
    verdict_str = (
        "CLEAR // VISIBLE"
        if oracle_res.is_visible
        else (
            oracle_res.reason.value
            if hasattr(oracle_res.reason, "value")
            else str(oracle_res.reason)
        )
    )
    st.markdown(
        f"""
        <div style="background-color:#0A101D; border:1px solid #1E293B; border-radius:6px; padding:12px 18px; margin-bottom:16px; display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span style="font-size:12px; color:{COLOR_TEXT_SECONDARY}; text-transform:uppercase; letter-spacing:0.05em; font-weight:600;">Deterministic Oracle Status</span>
                <div style="margin-top:4px;">{render_verdict_badge(verdict_str, is_visible=oracle_res.is_visible)}</div>
            </div>
            <div style="font-family:monospace; font-size:12px; color:{COLOR_CYAN_PRIMARY}; text-align:right;">
                <div>Scene ID: <strong>{current_scene.scene_id}</strong> | Seed: <strong>{current_scene.seed}</strong></div>
                <div style="color:{COLOR_TEXT_SECONDARY};">Canvas: {current_scene.canvas_width:.0f}×{current_scene.canvas_height:.0f}mm | Path: {(oracle_res.total_optical_distance or 0.0):.1f}mm</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =======================================================================
    # SECTION 3: MAIN WORKSPACE (VIEWPORT + GEOMETRY CONTROLS)
    # =======================================================================
    col_viewport, col_inspector = st.columns([7, 5])

    # -----------------------------------------------------------------------
    # VIEWPORT (Left Column)
    # -----------------------------------------------------------------------
    with col_viewport:
        # View Mode Selector
        view_ctrl_1, view_ctrl_2, view_ctrl_3 = st.columns([4, 3, 3])
        with view_ctrl_1:
            view_mode = st.radio(
                "Display Mode:",
                [
                    "🔬 Reference Overlay",
                    "👁️ Blind Diagram (Model Input)",
                    "🖼️ High-DPI Pillow Schematics",
                ],
                horizontal=True,
                key="radio_view_mode",
            )
        with view_ctrl_2:
            show_fov = st.checkbox("Show FOV Cone", value=True, key="chk_fov")
        with view_ctrl_3:
            show_sight = st.checkbox(
                "Show Direct Sightline", value=False, key="chk_sight"
            )

        # For Task C (Repair): Candidate mirror toggle
        candidate_mirrors = st.session_state.get("scene_lab_repair_candidates")
        active_candidate = st.session_state.get("scene_lab_repair_active", "A")
        if candidate_mirrors:
            st.markdown(
                f"<div style='font-size:12px; font-weight:600; color:{COLOR_CYAN_PRIMARY}; margin-top:4px;'>TASK C CANDIDATE MIRROR INSPECTOR:</div>",
                unsafe_allow_html=True,
            )
            cand_cols = st.columns(len(candidate_mirrors))
            for i, cid in enumerate(sorted(candidate_mirrors.keys())):
                with cand_cols[i]:
                    if st.button(
                        f"Mirror [{cid}]",
                        type="primary" if active_candidate == cid else "secondary",
                        use_container_width=True,
                    ):
                        st.session_state["scene_lab_repair_active"] = cid
                        # Swap mirror in scene to active candidate for live recalculation
                        new_cand_mirror = candidate_mirrors[cid]
                        updated_scene = InspectionScene(
                            scene_id=current_scene.scene_id,
                            seed=current_scene.seed,
                            category=current_scene.category,
                            difficulty=current_scene.difficulty,
                            canvas_width=current_scene.canvas_width,
                            canvas_height=current_scene.canvas_height,
                            camera=current_scene.camera,
                            mirror=new_cand_mirror,
                            target=current_scene.target,
                            obstacles=current_scene.obstacles,
                            ground_truth_state=current_scene.ground_truth_state,
                            ground_truth_reason=current_scene.ground_truth_reason,
                            diagnostic_details="",
                        )
                        st.session_state["scene_lab_scene"] = updated_scene
                        st.rerun()

        # For Task D (Counterfactual): Scene A / B toggle
        cf_pair = st.session_state.get("scene_lab_cf_pair")
        if cf_pair:
            cf_col_a, cf_col_b = st.columns(2)
            with cf_col_a:
                if st.button(
                    "Inspect Scene A",
                    type="primary"
                    if st.session_state.get("scene_lab_cf_active") == "Scene A"
                    else "secondary",
                    use_container_width=True,
                ):
                    st.session_state["scene_lab_cf_active"] = "Scene A"
                    st.session_state["scene_lab_scene"] = cf_pair.scene_a_bundle.scene
                    st.rerun()
            with cf_col_b:
                if st.button(
                    "Inspect Scene B",
                    type="primary"
                    if st.session_state.get("scene_lab_cf_active") == "Scene B"
                    else "secondary",
                    use_container_width=True,
                ):
                    st.session_state["scene_lab_cf_active"] = "Scene B"
                    st.session_state["scene_lab_scene"] = cf_pair.scene_b_bundle.scene
                    st.rerun()

        # Viewport Rendering
        is_blind_mode = "Blind" in view_mode
        is_pillow_mode = "Pillow" in view_mode

        if is_pillow_mode:
            # High-DPI Pillow rendering comparison
            renderer = PillowDiagramRenderer(
                theme=RenderTheme.DARK_NAVY, image_width=800, image_height=800
            )
            img_b = renderer.render(current_scene, mode=RenderMode.BLIND)
            img_o = renderer.render(current_scene, mode=RenderMode.OVERLAY)

            pil_col1, pil_col2 = st.columns(2)
            with pil_col1:
                st.caption("**Blind Diagram (Given to Model)**")
                st.image(img_b, use_container_width=True)
            with pil_col2:
                st.caption("**Answer Overlay (Private Oracle Key)**")
                st.image(img_o, use_container_width=True)
        else:
            # Interactive Plotly CAD Schematic
            fig = create_optical_figure(
                scene=current_scene,
                oracle_result=oracle_res if not is_blind_mode else None,
                show_debug_rays=not is_blind_mode,
                show_fov_cone=show_fov,
                show_direct_sightline=show_sight,
                theme="dark",
                height=560,
                candidate_mirrors=candidate_mirrors,
                active_candidate=active_candidate,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Telemetry Metrics Strip below Viewport
        telemetry_items = [
            (
                "Incidence Angle (θ_i)",
                f"{oracle_res.incident_angle_deg:.1f}°"
                if oracle_res.incident_angle_deg
                else "N/A",
            ),
            (
                "Reflection Angle (θ_r)",
                f"{oracle_res.reflected_angle_deg:.1f}°"
                if oracle_res.reflected_angle_deg
                else "N/A",
            ),
            (
                "Optical Path Length",
                f"{oracle_res.total_optical_distance:.1f} mm"
                if oracle_res.total_optical_distance
                else "N/A",
            ),
            (
                "Direct Sight Blocked",
                "YES (Cavity Wall/Barrier)"
                if oracle_res.is_direct_sight_blocked
                else "NO",
            ),
        ]
        render_telemetry_strip(telemetry_items)

        # Oracle Step-by-Step Proof Derivation Card
        with st.expander(
            "📐 Mathematical Proof & Oracle Verification Derivation", expanded=True
        ):
            st.markdown(f"**Diagnostic Details:** `{oracle_res.diagnostic_details}`")
            if oracle_res.reflection_point is not None:
                st.markdown(
                    f"- **Reflection Point $P$ on Mirror:** `({oracle_res.reflection_point[0]:.2f}, {oracle_res.reflection_point[1]:.2f}) mm`"
                )
            if oracle_res.virtual_target is not None:
                st.markdown(
                    f"- **Virtual Target $S'$ Reflection:** `({oracle_res.virtual_target[0]:.2f}, {oracle_res.virtual_target[1]:.2f}) mm`"
                )
            if (
                oracle_res.occlusion_result is not None
                and oracle_res.occlusion_result.is_path_blocked
            ):
                hit = oracle_res.occlusion_result.first_blocker
                if hit:
                    st.markdown(
                        f"- ⚠️ **Ray Collision Occurred:** Obstacle `{hit.obstacle_id}` blocks **{hit.leg.value}** at `({hit.hit_point[0]:.1f}, {hit.hit_point[1]:.1f})`"
                    )

        # Download Task Package (.ZIP) Button
        zip_bytes = build_task_package_zip(
            scene=current_scene,
            oracle_res=oracle_res,
            task_type_str=task_type,
            bundle_data=st.session_state.get("scene_lab_bundle"),
        )
        task_filename = f"{current_scene.scene_id}.zip"
        st.download_button(
            label=f"📦 Download Complete Task Package ({task_filename})",
            data=zip_bytes,
            file_name=task_filename,
            mime="application/zip",
            type="primary",
            use_container_width=True,
        )

    # -----------------------------------------------------------------------
    # GEOMETRY INSPECTOR & CONTROLS (Right Column)
    # -----------------------------------------------------------------------
    with col_inspector:
        st.markdown(
            f"<div style='font-size:13px; font-weight:700; color:{COLOR_TEXT_PRIMARY}; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:8px;'>Geometry Inspector & Manipulator</div>",
            unsafe_allow_html=True,
        )

        tabs_inspector = st.tabs(
            ["📷 Camera", "🪞 Mirror", "🎯 Target", "🚧 Obstacles"]
        )

        # -------------------------------------------------------------------
        # 1. CAMERA CONTROLS
        # -------------------------------------------------------------------
        with tabs_inspector[0]:
            st.caption(
                "Machine-vision inspection sensor orientation and field of view:"
            )
            new_cx = st.slider(
                "Camera Position X (mm):",
                50.0,
                950.0,
                float(current_scene.camera.position.x),
                5.0,
                key="cam_x",
            )
            new_cy = st.slider(
                "Camera Position Y (mm):",
                50.0,
                950.0,
                float(current_scene.camera.position.y),
                5.0,
                key="cam_y",
            )
            new_head = st.slider(
                "Camera Optical Axis Heading (°):",
                -180.0,
                180.0,
                float(current_scene.camera.heading_deg),
                1.0,
                key="cam_head",
            )
            new_fov = st.slider(
                "Angular Field-of-View (FOV °):",
                10.0,
                120.0,
                float(current_scene.camera.fov_degrees),
                1.0,
                key="cam_fov",
            )
            new_range = st.slider(
                "Max Camera Range (mm):",
                500.0,
                2000.0,
                float(current_scene.camera.max_range),
                50.0,
                key="cam_range",
            )

            # Check if camera changed
            cam_changed = (
                new_cx != current_scene.camera.position.x
                or new_cy != current_scene.camera.position.y
                or new_head != current_scene.camera.heading_deg
                or new_fov != current_scene.camera.fov_degrees
                or new_range != current_scene.camera.max_range
            )
            if cam_changed:
                updated_cam = CameraSpec(
                    position=Point2D(x=new_cx, y=new_cy),
                    heading_deg=new_head,
                    fov_degrees=new_fov,
                    max_range=new_range,
                )
                st.session_state["scene_lab_scene"] = InspectionScene(
                    scene_id=current_scene.scene_id,
                    seed=current_scene.seed,
                    category=current_scene.category,
                    difficulty=current_scene.difficulty,
                    canvas_width=current_scene.canvas_width,
                    canvas_height=current_scene.canvas_height,
                    camera=updated_cam,
                    mirror=current_scene.mirror,
                    target=current_scene.target,
                    obstacles=current_scene.obstacles,
                    ground_truth_state=current_scene.ground_truth_state,
                    ground_truth_reason=current_scene.ground_truth_reason,
                    diagnostic_details="",
                )
                st.rerun()

        # -------------------------------------------------------------------
        # 2. MIRROR CONTROLS
        # -------------------------------------------------------------------
        with tabs_inspector[1]:
            st.caption("Planar first-surface specular reflector geometry:")
            m_a = current_scene.mirror.endpoint_a
            m_b = current_scene.mirror.endpoint_b
            cur_center_x = (m_a.x + m_b.x) / 2.0
            cur_center_y = (m_a.y + m_b.y) / 2.0
            cur_length = math.hypot(m_b.x - m_a.x, m_b.y - m_a.y)
            cur_angle = math.degrees(math.atan2(m_b.y - m_a.y, m_b.x - m_a.x))

            new_mcx = st.slider(
                "Mirror Center X (mm):",
                100.0,
                900.0,
                float(cur_center_x),
                5.0,
                key="mir_cx",
            )
            new_mcy = st.slider(
                "Mirror Center Y (mm):",
                100.0,
                950.0,
                float(cur_center_y),
                5.0,
                key="mir_cy",
            )
            new_mlen = st.slider(
                "Mirror Segment Length (mm):",
                50.0,
                600.0,
                float(cur_length),
                5.0,
                key="mir_len",
            )
            new_mang = st.slider(
                "Mirror Rotation Angle (°):",
                -180.0,
                180.0,
                float(cur_angle),
                1.0,
                key="mir_ang",
            )
            flip_norm = st.checkbox(
                "Invert Normal Vector (Swaps Front/Back Side)",
                value=False,
                key="mir_flip",
            )

            # Calculate derived endpoints
            rad = math.radians(new_mang)
            hl = new_mlen / 2.0
            dx = hl * math.cos(rad)
            dy = hl * math.sin(rad)
            calc_a = Point2D(x=new_mcx - dx, y=new_mcy - dy)
            calc_b = Point2D(x=new_mcx + dx, y=new_mcy + dy)

            # Normal pointing into cavity
            nx = dy / hl
            ny = -dx / hl
            if flip_norm:
                nx = -nx
                ny = -ny

            st.markdown(
                f"""
                <div style="font-family:monospace; font-size:11px; background:#0B1220; padding:8px; border-radius:4px; margin-top:8px;">
                    <div>Endpoint A: ({calc_a.x:.1f}, {calc_a.y:.1f}) mm</div>
                    <div>Endpoint B: ({calc_b.x:.1f}, {calc_b.y:.1f}) mm</div>
                    <div>Normal: ({nx:.2f}, {ny:.2f})</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Check if mirror changed
            mirror_changed = (
                abs(calc_a.x - m_a.x) > 0.1
                or abs(calc_a.y - m_a.y) > 0.1
                or abs(calc_b.x - m_b.x) > 0.1
                or abs(calc_b.y - m_b.y) > 0.1
                or abs(nx - current_scene.mirror.reflective_normal.dx) > 0.05
                or abs(ny - current_scene.mirror.reflective_normal.dy) > 0.05
            )
            if mirror_changed:
                updated_mir = MirrorSpec(
                    endpoint_a=calc_a,
                    endpoint_b=calc_b,
                    reflective_normal=Vector2D(dx=nx, dy=ny),
                )
                st.session_state["scene_lab_scene"] = InspectionScene(
                    scene_id=current_scene.scene_id,
                    seed=current_scene.seed,
                    category=current_scene.category,
                    difficulty=current_scene.difficulty,
                    canvas_width=current_scene.canvas_width,
                    canvas_height=current_scene.canvas_height,
                    camera=current_scene.camera,
                    mirror=updated_mir,
                    target=current_scene.target,
                    obstacles=current_scene.obstacles,
                    ground_truth_state=current_scene.ground_truth_state,
                    ground_truth_reason=current_scene.ground_truth_reason,
                    diagnostic_details="",
                )
                st.rerun()

        # -------------------------------------------------------------------
        # 3. TARGET DEFECT CONTROLS
        # -------------------------------------------------------------------
        with tabs_inspector[2]:
            st.caption("Red-X scratch defect inspection target location:")
            new_tx = st.slider(
                "Target Defect X (mm):",
                50.0,
                950.0,
                float(current_scene.target.center.x),
                5.0,
                key="tgt_x",
            )
            new_ty = st.slider(
                "Target Defect Y (mm):",
                50.0,
                950.0,
                float(current_scene.target.center.y),
                5.0,
                key="tgt_y",
            )
            new_tr = st.slider(
                "Defect Crosshair Radius (mm):",
                3.0,
                25.0,
                float(current_scene.target.radius_mm),
                1.0,
                key="tgt_r",
            )

            target_changed = (
                new_tx != current_scene.target.center.x
                or new_ty != current_scene.target.center.y
                or new_tr != current_scene.target.radius_mm
            )
            if target_changed:
                updated_tgt = TargetSpec(
                    center=Point2D(x=new_tx, y=new_ty),
                    radius_mm=new_tr,
                )
                st.session_state["scene_lab_scene"] = InspectionScene(
                    scene_id=current_scene.scene_id,
                    seed=current_scene.seed,
                    category=current_scene.category,
                    difficulty=current_scene.difficulty,
                    canvas_width=current_scene.canvas_width,
                    canvas_height=current_scene.canvas_height,
                    camera=current_scene.camera,
                    mirror=current_scene.mirror,
                    target=updated_tgt,
                    obstacles=current_scene.obstacles,
                    ground_truth_state=current_scene.ground_truth_state,
                    ground_truth_reason=current_scene.ground_truth_reason,
                    diagnostic_details="",
                )
                st.rerun()

        # -------------------------------------------------------------------
        # 4. OBSTACLE ARRANGEMENT CONTROLS
        # -------------------------------------------------------------------
        with tabs_inspector[3]:
            st.caption("Opaque barriers, distractor blocks, and sightline occluders:")

            # Table of current obstacles
            if current_scene.obstacles:
                obs_data = [
                    {
                        "ID": o.obstacle_id,
                        "Type": o.obstacle_type,
                        "Center X": f"{o.center.x:.1f}",
                        "Center Y": f"{o.center.y:.1f}",
                        "Size": f"{o.radius:.1f}r"
                        if o.obstacle_type == "circle"
                        else f"{o.width:.0f}×{o.height:.0f}",
                    }
                    for o in current_scene.obstacles
                ]
                st.dataframe(obs_data, use_container_width=True)
            else:
                st.info("No obstacles in current scene.")

            # Add New Obstacle Form
            st.markdown("**Add Obstacle:**")
            add_col1, add_col2, add_col3 = st.columns([2, 3, 3])
            with add_col1:
                new_oid = st.text_input(
                    "Label:", f"O{len(current_scene.obstacles) + 1}", key="new_obs_id"
                )
            with add_col2:
                new_otype = st.selectbox(
                    "Type:", ["rectangle", "circle"], key="new_obs_type"
                )
            with add_col3:
                new_ox = st.number_input(
                    "Center X:", 50.0, 950.0, 500.0, 10.0, key="new_obs_x"
                )

            add_col4, add_col5, add_col6 = st.columns(3)
            with add_col4:
                new_oy = st.number_input(
                    "Center Y:", 50.0, 950.0, 400.0, 10.0, key="new_obs_y"
                )
            with add_col5:
                new_ow = st.number_input(
                    "Width / Radius:", 10.0, 200.0, 50.0, 5.0, key="new_obs_w"
                )
            with add_col6:
                new_oh = st.number_input(
                    "Height (Rect only):", 10.0, 200.0, 50.0, 5.0, key="new_obs_h"
                )

            if st.button("➕ Add Obstacle to Scene", use_container_width=True):
                new_spec = ObstacleSpec(
                    obstacle_id=new_oid,
                    obstacle_type=new_otype,
                    center=Point2D(x=new_ox, y=new_oy),
                    width=new_ow if new_otype == "rectangle" else None,
                    height=new_oh if new_otype == "rectangle" else None,
                    radius=new_ow if new_otype == "circle" else None,
                    angle_degrees=0.0,
                )
                updated_obs = list(current_scene.obstacles) + [new_spec]
                st.session_state["scene_lab_scene"] = InspectionScene(
                    scene_id=current_scene.scene_id,
                    seed=current_scene.seed,
                    category=current_scene.category,
                    difficulty=current_scene.difficulty,
                    canvas_width=current_scene.canvas_width,
                    canvas_height=current_scene.canvas_height,
                    camera=current_scene.camera,
                    mirror=current_scene.mirror,
                    target=current_scene.target,
                    obstacles=updated_obs,
                    ground_truth_state=current_scene.ground_truth_state,
                    ground_truth_reason=current_scene.ground_truth_reason,
                    diagnostic_details="",
                )
                st.success(f"Added obstacle {new_oid}")
                st.rerun()

            # Remove Obstacle
            if current_scene.obstacles:
                del_ids = [o.obstacle_id for o in current_scene.obstacles]
                sel_del = st.selectbox(
                    "Select Obstacle to Remove:", del_ids, key="sel_del_obs"
                )
                if st.button("🗑️ Remove Selected Obstacle", use_container_width=True):
                    rem_obs = [
                        o for o in current_scene.obstacles if o.obstacle_id != sel_del
                    ]
                    st.session_state["scene_lab_scene"] = InspectionScene(
                        scene_id=current_scene.scene_id,
                        seed=current_scene.seed,
                        category=current_scene.category,
                        difficulty=current_scene.difficulty,
                        canvas_width=current_scene.canvas_width,
                        canvas_height=current_scene.canvas_height,
                        camera=current_scene.camera,
                        mirror=current_scene.mirror,
                        target=current_scene.target,
                        obstacles=rem_obs,
                        ground_truth_state=current_scene.ground_truth_state,
                        ground_truth_reason=current_scene.ground_truth_reason,
                        diagnostic_details="",
                    )
                    st.success(f"Removed obstacle {sel_del}")
                    st.rerun()
