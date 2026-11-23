"""
Model Arena Module for MIRRORLAB.

Interactive benchmark arena where users challenge Vision-Language Models (VLMs)
against mathematically verified specular reflection tasks:
- Single benchmark task evaluation across Visibility, Obstruction, and Repair.
- Paired counterfactual evaluation across Mirror Angle Flip, Cosmetic Shift, and Obstacle Insertion.
- Strict blind evaluation: zero answer key exposure prior to model prediction lock.
- Visual impact reveal: dramatic side-by-side prediction vs oracle truth comparisons.
- Verbatim raw response capture, structured JSON parsing, and geometric proof overlays.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import streamlit as st

from mirrorlab.evaluation.model_adapter import (
    ModelResponse,
    ModelSettings,
    get_model_adapter,
)
from mirrorlab.evaluation.response_parser import (
    ParsedEvaluationRecord,
    ResponseParser,
)
from mirrorlab.schema.common import TaskType
from mirrorlab.ui.components import (
    render_telemetry_strip,
)
from mirrorlab.ui.data_loader import load_tasks_by_family
from mirrorlab.ui.design_tokens import (
    COLOR_CYAN_PRIMARY,
    COLOR_STATUS_ERROR,
    COLOR_STATUS_SUCCESS,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State Management Helpers
# ---------------------------------------------------------------------------
def _init_arena_state() -> None:
    """Ensure arena session state keys are initialized."""
    if "arena_single_result" not in st.session_state:
        st.session_state["arena_single_result"] = None
    if "arena_single_revealed" not in st.session_state:
        st.session_state["arena_single_revealed"] = False
    if "arena_paired_result" not in st.session_state:
        st.session_state["arena_paired_result"] = None
    if "arena_paired_revealed" not in st.session_state:
        st.session_state["arena_paired_revealed"] = False


def _compute_prediction_lock_hash(
    task_id: str, model_name: str, raw_response: str
) -> str:
    """Generate a cryptographic tamper-evident lock hash for the prediction."""
    payload = f"{task_id}::{model_name}::{raw_response}".encode()
    return hashlib.sha256(payload).hexdigest()[:16].upper()


# ---------------------------------------------------------------------------
# Task & Model Loading Helpers
# ---------------------------------------------------------------------------
AVAILABLE_ARENA_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gpt-4o",
    "gpt-4o-mini",
    "claude-3-7-sonnet",
    "mock-realistic-gemini",
    "mock-realistic-gpt4o",
    "mock-oracle",
    "mock-all-yes",
    "mock-all-no",
    "mock-random",
]


def _execute_model_challenge(
    model_name: str,
    image_path: str,
    question_prompt: str,
    task_type: TaskType,
    task_id: str,
    ground_truth_dict: dict[str, Any],
    api_key_override: str | None = None,
    temperature: float = 0.0,
) -> tuple[ModelResponse, ParsedEvaluationRecord]:
    """
    Executes a model query using the provider-independent vision adapter,
    measures latency, and evaluates the response using ResponseParser.
    """
    # Ground truth mapping for mock models
    gt_map = {task_id: ground_truth_dict}

    settings = ModelSettings(
        temperature=temperature,
        timeout_seconds=45.0,
        max_retries=2,
    )

    adapter = get_model_adapter(
        model_name=model_name,
        api_key=api_key_override,
        ground_truth_map=gt_map,
        settings=settings,
    )

    # Perform blind inference with explicit task identification
    task_type_str = task_type.value if hasattr(task_type, "value") else str(task_type)
    response = adapter.ask(
        image_path=image_path,
        question=question_prompt,
        task_id=task_id,
        task_type=task_type_str,
    )

    # Parse and grade against ground truth
    record = ResponseParser.parse_and_grade(
        raw_response=response.raw_text,
        task_type=task_type,
        task_id=task_id,
        ground_truth=ground_truth_dict,
        api_error=response.error,
    )

    return response, record


# ---------------------------------------------------------------------------
# Main Arena UI Component
# ---------------------------------------------------------------------------
def render_model_arena() -> None:
    """
    Renders the complete MIRRORLAB Model Arena interactive evaluation studio.
    """
    _init_arena_state()

    st.subheader("⚔️ MIRRORLAB Model Arena")
    st.caption(
        "Directly challenge Vision-Language Models against mathematically verified specular inspection cavities. "
        "Strict blind isolation enforces zero ground-truth leakage before model predictions are locked."
    )

    arena_mode = st.radio(
        "Evaluation Arena Mode:",
        ["🎯 Single Benchmark Task Challenge", "⚖️ Paired Counterfactual Evaluation"],
        horizontal=True,
        key="arena_mode_radio",
    )

    st.divider()

    if arena_mode == "🎯 Single Benchmark Task Challenge":
        _render_single_task_arena()
    else:
        _render_paired_task_arena()


# ===========================================================================
# 1. SINGLE TASK ARENA WORKFLOW
# ===========================================================================
def _render_single_task_arena() -> None:
    """
    Workflow for single task challenge:
    1. Select task -> 2. Select model -> 3. View blind diagram -> 4. Challenge ->
    5. Show raw text -> 6. Parse structured -> 7. Lock prediction -> 8. Reveal ->
    9. Verdict -> 10. Geometric overlay.
    """
    col_task_sel, col_model_sel = st.columns([1, 1])

    with col_task_sel:
        st.markdown(
            f"<div style='font-size:12px; font-weight:700; color:{COLOR_CYAN_PRIMARY}; text-transform:uppercase;'>Step 1: Select Benchmark Task</div>",
            unsafe_allow_html=True,
        )
        task_family = st.selectbox(
            "Task Family:",
            [
                "Task A: Reflected Visibility",
                "Task B: Obstruction Identification",
                "Task C: Mirror-Setting Repair",
                "Task E: Out-Of-Distribution (OOD) Stress Test",
            ],
            key="single_task_family",
        )

        fam_key = "visibility"
        t_type = TaskType.TASK_A_VISIBILITY
        if "Obstruction" in task_family:
            fam_key = "obstruction"
            t_type = TaskType.TASK_B_OBSTRUCTION
        elif "Repair" in task_family:
            fam_key = "repair"
            t_type = TaskType.TASK_C_REPAIR
        elif "OOD" in task_family:
            fam_key = "ood"
            t_type = TaskType.TASK_A_VISIBILITY

        tasks_list = load_tasks_by_family(fam_key)
        if not tasks_list:
            st.error(f"No benchmark tasks found for {task_family} in data/ directory.")
            return

        task_labels = [t["id"] for t in tasks_list]
        selected_task_id = st.selectbox(
            "Select Benchmark Task ID:", task_labels, key="single_task_id"
        )
        current_task = next(t for t in tasks_list if t["id"] == selected_task_id)

    with col_model_sel:
        st.markdown(
            f"<div style='font-size:12px; font-weight:700; color:{COLOR_CYAN_PRIMARY}; text-transform:uppercase;'>Step 2: Select Vision-Language Model</div>",
            unsafe_allow_html=True,
        )
        selected_model = st.selectbox(
            "Vision Model:", AVAILABLE_ARENA_MODELS, index=0, key="single_model_name"
        )
        if selected_model.startswith("mock"):
            st.markdown(
                "<div style='font-size:11px; color:#F59E0B; font-weight:700; background:rgba(245,158,11,0.1); padding:4px 8px; border-radius:4px; margin-top:4px; border:1px solid rgba(245,158,11,0.3);'>"
                "⚠️ SIMULATED BASELINE — NOT AN ACTUAL MODEL API RESULT</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='font-size:11px; color:{COLOR_CYAN_PRIMARY}; font-weight:700; background:rgba(0,229,255,0.1); padding:4px 8px; border-radius:4px; margin-top:4px; border:1px solid rgba(0,229,255,0.3);'>"
                "🌐 REAL MULTIMODAL API EVALUATION</div>",
                unsafe_allow_html=True,
            )
        api_key_input = st.text_input(
            "API Key (Optional override, otherwise reads from .env):",
            type="password",
            key="single_api_key",
            help="If not provided, the adapter automatically checks your local .env or system environment.",
        )
        temperature = st.slider(
            "Sampling Temperature:",
            0.0,
            1.0,
            0.0,
            0.1,
            key="single_temp",
            help="0.0 recommended for deterministic benchmark.",
        )

    # Task Metadata & Security Strip
    blind_img_path = current_task.get("diagram_blind")
    overlay_img_path = current_task.get("diagram_overlay")
    question_dict = current_task.get("question", {})
    ground_truth_dict = current_task.get("answer", {})

    prompt_text = question_dict.get("prompt") or question_dict.get(
        "question", "Can the camera see the red X defect?"
    )

    st.markdown(
        f"""
        <div style="background-color:#0A101D; border:1px solid #1E293B; border-radius:6px; padding:10px 16px; margin: 12px 0; display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span style="font-size:11px; color:{COLOR_TEXT_SECONDARY}; text-transform:uppercase; font-weight:600;">Active Task</span>
                <div style="font-family:monospace; color:{COLOR_TEXT_PRIMARY}; font-weight:700;">{selected_task_id} ({t_type.value})</div>
            </div>
            <div>
                <span style="background:rgba(0,229,255,0.12); color:{COLOR_CYAN_PRIMARY}; border:1px solid rgba(0,229,255,0.3); padding:4px 10px; border-radius:4px; font-size:11px; font-weight:700;">
                    🔒 BLIND EVALUATION MODE // ZERO ANSWER LEAKAGE
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------------
    # Step 3: Blind Diagram & Challenge Button
    # -----------------------------------------------------------------------
    col_view, col_prompt = st.columns([1, 1])

    with col_view:
        st.markdown("**Step 3: Blind Diagram (Model Input)**")
        if blind_img_path and Path(blind_img_path).exists():
            st.image(
                blind_img_path,
                caption=f"Blind Inspection Schematic: {selected_task_id}",
                use_container_width=True,
            )
        else:
            st.warning(f"Blind diagram image not found at {blind_img_path}")

    with col_prompt:
        st.markdown("**Benchmark Question & Instruction:**")
        st.text_area("Full Model Prompt:", prompt_text, height=280, disabled=True)

        st.write("")
        challenge_btn = st.button(
            "⚔️ Submit Question & Challenge Model",
            type="primary",
            use_container_width=True,
        )

        if challenge_btn:
            with st.spinner(f"Querying {selected_model}..."):
                try:
                    resp, record = _execute_model_challenge(
                        model_name=selected_model,
                        image_path=blind_img_path,
                        question_prompt=prompt_text,
                        task_type=t_type,
                        task_id=selected_task_id,
                        ground_truth_dict=ground_truth_dict,
                        api_key_override=api_key_input or None,
                        temperature=temperature,
                    )
                    lock_hash = _compute_prediction_lock_hash(
                        selected_task_id, selected_model, resp.raw_text
                    )
                    st.session_state["arena_single_result"] = {
                        "response": resp,
                        "record": record,
                        "lock_hash": lock_hash,
                        "timestamp": datetime.now(UTC).strftime("%H:%M:%S UTC"),
                        "task_id": selected_task_id,
                        "model_name": selected_model,
                    }
                    st.session_state["arena_single_revealed"] = False
                    st.success("Prediction received and strictly locked!")
                except (ValueError, RuntimeError, TimeoutError, OSError) as exc:
                    st.error(f"Inference execution failed: {exc}")

    # -----------------------------------------------------------------------
    # Step 5-7: Raw Response, Structured Answer, Locked Prediction
    # -----------------------------------------------------------------------
    res_data = st.session_state.get("arena_single_result")
    if res_data and res_data.get("task_id") == selected_task_id:
        resp: ModelResponse = res_data["response"]
        record: ParsedEvaluationRecord = res_data["record"]
        lock_hash = res_data["lock_hash"]
        timestamp = res_data["timestamp"]

        st.divider()

        # Prediction Locked Banner
        st.markdown(
            f"""
            <div style="background:rgba(16,185,129,0.08); border:1px solid #10B981; border-radius:6px; padding:12px 18px; margin-bottom:14px; display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span style="font-size:11px; color:#10B981; text-transform:uppercase; font-weight:700; letter-spacing:0.05em;">Step 7: Prediction Locked</span>
                    <div style="font-family:monospace; font-size:13px; font-weight:700; color:{COLOR_TEXT_PRIMARY};">
                        LOCK HASH: <code>{lock_hash}</code> | MODEL: <strong>{res_data["model_name"]}</strong> | TIME: <strong>{timestamp}</strong>
                    </div>
                </div>
                <div>
                    <span style="font-size:12px; color:{COLOR_CYAN_PRIMARY}; font-family:monospace;">⏱️ Latency: {resp.latency_sec * 1000.0:.0f} ms</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_raw, col_parsed = st.columns([1, 1])

        with col_raw:
            st.markdown("**Step 5: Verbatim Raw Model Response:**")
            st.code(
                resp.raw_text if resp.raw_text else "(Empty or Error response)",
                language="json",
            )

        with col_parsed:
            st.markdown("**Step 6: Structured Parsed Answer:**")
            if record.is_valid_format:
                st.markdown(
                    f"""
                    <div style="background:#0B1220; border:1px solid #1E293B; border-radius:6px; padding:12px; font-family:monospace; font-size:12px;">
                        <div><strong>Format Validity:</strong> <span style="color:{COLOR_STATUS_SUCCESS};">VALID JSON ✓</span></div>
                        <div style="margin-top:6px;"><strong>Extracted Prediction:</strong> <code>{record.extracted_prediction}</code></div>
                        <div style="margin-top:6px;"><strong>Detailed Explanation:</strong> <span style="color:{COLOR_TEXT_SECONDARY};">{record.detailed_explanation or "N/A"}</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div style="background:rgba(239,68,68,0.1); border:1px solid {COLOR_STATUS_ERROR}; border-radius:6px; padding:12px; font-family:monospace; font-size:12px;">
                        <div><strong>Format Validity:</strong> <span style="color:{COLOR_STATUS_ERROR};">INVALID FORMAT ✗</span></div>
                        <div style="margin-top:6px;"><strong>Error Classification:</strong> {record.error_classification.value}</div>
                        <div style="margin-top:6px; color:{COLOR_STATUS_ERROR};"><strong>Errors:</strong> {", ".join(record.validation_errors)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # -------------------------------------------------------------------
        # Step 8-10: Visually Impactful Reveal & Geometric Overlay
        # -------------------------------------------------------------------
        st.write("")
        reveal_col1, reveal_col2 = st.columns([3, 1])
        with reveal_col1:
            reveal_btn = st.button(
                "👁️ REVEAL ORACLE ANSWER & GEOMETRIC EXPLANATION",
                type="primary",
                use_container_width=True,
            )
        with reveal_col2:
            if st.button("🔄 Reset Challenge", use_container_width=True):
                st.session_state["arena_single_result"] = None
                st.session_state["arena_single_revealed"] = False
                st.rerun()

        if reveal_btn:
            st.session_state["arena_single_revealed"] = True

        if st.session_state.get("arena_single_revealed"):
            is_correct = record.final_score == 1.0
            gt_ans = (
                ground_truth_dict.get("answer")
                or ground_truth_dict.get("can_see")
                or ground_truth_dict.get("correct_candidate")
            )
            gt_reason = (
                ground_truth_dict.get("reason_code")
                or ground_truth_dict.get("blocked_segment")
                or "N/A"
            )
            gt_proof = ground_truth_dict.get("detailed_explanation", "")

            # Impactful Verdict Banner
            if is_correct:
                verdict_html = f"""
                <div style="background:linear-gradient(90deg, rgba(16,185,129,0.2) 0%, rgba(6,10,18,0.95) 100%); border:2px solid {COLOR_STATUS_SUCCESS}; border-radius:8px; padding:18px 24px; margin: 16px 0; box-shadow: 0 0 25px rgba(16,185,129,0.25);">
                    <div style="font-size:22px; font-weight:800; color:{COLOR_STATUS_SUCCESS}; letter-spacing:0.05em;">🎯 VERDICT: CORRECT (+1.0)</div>
                    <div style="font-size:14px; color:{COLOR_TEXT_PRIMARY}; margin-top:6px;">
                        The model's prediction <code>{record.extracted_prediction}</code> matches the deterministic ground truth (<strong>{gt_ans}</strong> // {gt_reason}).
                    </div>
                </div>
                """
            else:
                verdict_html = f"""
                <div style="background:linear-gradient(90deg, rgba(239,68,68,0.2) 0%, rgba(6,10,18,0.95) 100%); border:2px solid {COLOR_STATUS_ERROR}; border-radius:8px; padding:18px 24px; margin: 16px 0; box-shadow: 0 0 25px rgba(239,68,68,0.25);">
                    <div style="font-size:22px; font-weight:800; color:{COLOR_STATUS_ERROR}; letter-spacing:0.05em;">❌ VERDICT: INCORRECT (0.0)</div>
                    <div style="font-size:14px; color:{COLOR_TEXT_PRIMARY}; margin-top:6px;">
                        Model predicted <code>{record.extracted_prediction}</code>, but true physical oracle is <strong>{gt_ans}</strong> (Reason: <strong>{gt_reason}</strong>).
                    </div>
                </div>
                """
            st.markdown(verdict_html, unsafe_allow_html=True)

            # Step 10: Geometric Explanation Overlay
            st.markdown("**Step 10: Optical Truth Overlay & Mathematical Proof:**")
            comp_col1, comp_col2 = st.columns([1, 1])

            with comp_col1:
                st.caption("**Input Blind Diagram (VLM View)**")
                if blind_img_path and Path(blind_img_path).exists():
                    st.image(blind_img_path, use_container_width=True)

            with comp_col2:
                st.caption(
                    "**Answer Overlay with Specular Ray Tracing (Oracle Truth)**"
                )
                if overlay_img_path and Path(overlay_img_path).exists():
                    st.image(overlay_img_path, use_container_width=True)
                else:
                    st.info(f"Overlay image path: {overlay_img_path}")

            with st.expander("📐 Ground Truth Mathematical Derivation", expanded=True):
                st.markdown(f"**Proof Explanation:** {gt_proof}")
                proof_geo = ground_truth_dict.get("proof_geometry", {})
                if proof_geo:
                    p_items = [
                        (
                            "θ_i (Incidence)",
                            f"{proof_geo.get('incident_angle_deg', 0.0):.1f}°",
                        ),
                        (
                            "θ_r (Reflection)",
                            f"{proof_geo.get('reflected_angle_deg', 0.0):.1f}°",
                        ),
                        (
                            "Total Path",
                            f"{proof_geo.get('total_optical_distance', 0.0):.1f} mm",
                        ),
                        ("Reason Code", str(gt_reason)),
                    ]
                    render_telemetry_strip(p_items)


# ===========================================================================
# 2. PAIRED COUNTERFACTUAL ARENA WORKFLOW
# ===========================================================================
def _render_paired_task_arena() -> None:
    """
    Workflow for paired evaluations:
    - Display both related scenes in blind mode.
    - Run model on both before revealing answers.
    - Reveal oracle results together.
    - Evaluate whether model correctly tracked the change.
    """
    st.markdown(
        f"<div style='font-size:12px; font-weight:700; color:{COLOR_CYAN_PRIMARY}; text-transform:uppercase;'>Paired Counterfactual Evaluation Studio</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Evaluates paired consistency, mirror-angle flip invariance, cosmetic robustness, and obstacle distractors."
    )

    col_pair_sel, col_model_sel = st.columns([1, 1])

    with col_pair_sel:
        pairs = load_tasks_by_family("counterfactual")
        if not pairs:
            st.error("No counterfactual pairs found in data/counterfactual_pairs.")
            return

        pair_labels = [p["id"] for p in pairs]
        selected_pair_id = st.selectbox(
            "Select Counterfactual Pair:", pair_labels, key="cf_pair_id"
        )
        current_pair = next(p for p in pairs if p["id"] == selected_pair_id)

    with col_model_sel:
        selected_model = st.selectbox(
            "Vision Model:", AVAILABLE_ARENA_MODELS, index=0, key="cf_model_name"
        )
        if selected_model.startswith("mock"):
            st.markdown(
                "<div style='font-size:11px; color:#F59E0B; font-weight:700; background:rgba(245,158,11,0.1); padding:4px 8px; border-radius:4px; margin-top:4px; border:1px solid rgba(245,158,11,0.3);'>"
                "⚠️ SIMULATED BASELINE — NOT AN ACTUAL MODEL API RESULT</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='font-size:11px; color:{COLOR_CYAN_PRIMARY}; font-weight:700; background:rgba(0,229,255,0.1); padding:4px 8px; border-radius:4px; margin-top:4px; border:1px solid rgba(0,229,255,0.3);'>"
                "🌐 REAL MULTIMODAL API EVALUATION</div>",
                unsafe_allow_html=True,
            )
        api_key_input = st.text_input(
            "API Key (Optional override):", type="password", key="cf_api_key"
        )

    pair_data = current_pair.get("pair_data", {})
    pair_dir = Path(current_pair["dir"])
    pair_subtype = pair_data.get("pair_type", "COUNTERFACTUAL")
    expected_rel = pair_data.get("expected_relationship", "INVARIANT_OUTCOME")
    diff_summary = pair_data.get("difference_summary", "")

    # Load question & ground truth for Scene A and Scene B
    qa_path = pair_dir / "scene_a" / "question.json"
    gta_path = pair_dir / "scene_a" / "ground_truth.json"
    qb_path = pair_dir / "scene_b" / "question.json"
    gtb_path = pair_dir / "scene_b" / "ground_truth.json"

    q_a = json.loads(qa_path.read_text(encoding="utf-8")) if qa_path.exists() else {}
    gt_a = json.loads(gta_path.read_text(encoding="utf-8")) if gta_path.exists() else {}
    q_b = json.loads(qb_path.read_text(encoding="utf-8")) if qb_path.exists() else {}
    gt_b = json.loads(gtb_path.read_text(encoding="utf-8")) if gtb_path.exists() else {}

    prompt_a = q_a.get("prompt") or q_a.get("question", "Can camera see red X?")
    prompt_b = q_b.get("prompt") or q_b.get("question", "Can camera see red X?")

    blind_a_path = str(pair_dir / "scene_a" / "diagram_blind.png")
    overlay_a_path = str(pair_dir / "scene_a" / "diagram_overlay.png")
    blind_b_path = str(pair_dir / "scene_b" / "diagram_blind.png")
    overlay_b_path = str(pair_dir / "scene_b" / "diagram_overlay.png")

    # Pair Details Banner
    st.markdown(
        f"""
        <div style="background-color:#0A101D; border:1px solid #1E293B; border-radius:6px; padding:12px 18px; margin: 12px 0;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span style="font-size:11px; color:{COLOR_CYAN_PRIMARY}; text-transform:uppercase; font-weight:700;">Subtype: {pair_subtype}</span>
                    <div style="font-size:13px; font-weight:600; color:{COLOR_TEXT_PRIMARY}; margin-top:2px;">{diff_summary}</div>
                </div>
                <div style="font-family:monospace; font-size:11px; color:{COLOR_TEXT_SECONDARY}; text-align:right;">
                    Expected Behavior: <strong style="color:{COLOR_CYAN_PRIMARY};">{expected_rel}</strong>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------------
    # Step 3: Display Both Related Scenes (Blind Mode)
    # -----------------------------------------------------------------------
    st.markdown(
        "**Step 3: Related Scenes in Blind Evaluation Mode (Zero Answers Exposed):**"
    )
    col_scene_a, col_scene_b = st.columns(2)

    with col_scene_a:
        st.markdown(f"**Scene A (Baseline)** — `{selected_pair_id}_a`")
        if Path(blind_a_path).exists():
            st.image(blind_a_path, use_container_width=True)

    with col_scene_b:
        st.markdown(
            f"**Scene B (Perturbed / Counterfactual)** — `{selected_pair_id}_b`"
        )
        if Path(blind_b_path).exists():
            st.image(blind_b_path, use_container_width=True)

    st.write("")
    run_pair_btn = st.button(
        "⚔️ Challenge Model on Both Related Scenes",
        type="primary",
        use_container_width=True,
    )

    if run_pair_btn:
        with st.spinner(f"Evaluating {selected_model} on Scene A and Scene B..."):
            try:
                # Inference Scene A
                resp_a, rec_a = _execute_model_challenge(
                    model_name=selected_model,
                    image_path=blind_a_path,
                    question_prompt=prompt_a,
                    task_type=TaskType.TASK_A_VISIBILITY,
                    task_id=f"{selected_pair_id}_a",
                    ground_truth_dict=gt_a,
                    api_key_override=api_key_input or None,
                )
                # Inference Scene B
                resp_b, rec_b = _execute_model_challenge(
                    model_name=selected_model,
                    image_path=blind_b_path,
                    question_prompt=prompt_b,
                    task_type=TaskType.TASK_A_VISIBILITY,
                    task_id=f"{selected_pair_id}_b",
                    ground_truth_dict=gt_b,
                    api_key_override=api_key_input or None,
                )

                st.session_state["arena_paired_result"] = {
                    "pair_id": selected_pair_id,
                    "model_name": selected_model,
                    "resp_a": resp_a,
                    "rec_a": rec_a,
                    "resp_b": resp_b,
                    "rec_b": rec_b,
                    "lock_a": _compute_prediction_lock_hash(
                        f"{selected_pair_id}_a", selected_model, resp_a.raw_text
                    ),
                    "lock_b": _compute_prediction_lock_hash(
                        f"{selected_pair_id}_b", selected_model, resp_b.raw_text
                    ),
                    "timestamp": datetime.now(UTC).strftime("%H:%M:%S UTC"),
                }
                st.session_state["arena_paired_revealed"] = False
                st.success(
                    "Both predictions locked! Answer keys remain securely hidden."
                )
            except (ValueError, RuntimeError, TimeoutError, OSError) as exc:
                st.error(f"Paired evaluation error: {exc}")

    # -----------------------------------------------------------------------
    # Step 5-7: Both Predictions Locked Before Revealing Answers
    # -----------------------------------------------------------------------
    paired_res = st.session_state.get("arena_paired_result")
    if paired_res and paired_res.get("pair_id") == selected_pair_id:
        rec_a: ParsedEvaluationRecord = paired_res["rec_a"]
        rec_b: ParsedEvaluationRecord = paired_res["rec_b"]

        st.divider()
        st.markdown(
            f"""
            <div style="background:rgba(0,229,255,0.08); border:1px solid {COLOR_CYAN_PRIMARY}; border-radius:6px; padding:12px 18px; margin-bottom:14px; display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span style="font-size:11px; color:{COLOR_CYAN_PRIMARY}; text-transform:uppercase; font-weight:700;">Predictions Locked Simultaneously</span>
                    <div style="font-family:monospace; font-size:13px; font-weight:700; color:{COLOR_TEXT_PRIMARY};">
                        LOCK A: <code>{paired_res["lock_a"]}</code> | LOCK B: <code>{paired_res["lock_b"]}</code> | TIME: <strong>{paired_res["timestamp"]}</strong>
                    </div>
                </div>
                <div>
                    <span style="font-size:12px; color:{COLOR_TEXT_SECONDARY};">Both predictions frozen before reveal.</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_pred_a, col_pred_b = st.columns(2)
        with col_pred_a:
            st.markdown(
                f"""
                <div style="background:#0B1220; border:1px solid #1E293B; border-radius:6px; padding:12px; font-family:monospace; font-size:12px;">
                    <div><strong>Scene A Locked Prediction:</strong> <code>{rec_a.extracted_prediction}</code></div>
                    <div style="margin-top:4px;"><strong>Format:</strong> {"VALID" if rec_a.is_valid_format else "INVALID"}</div>
                    <div style="margin-top:4px; color:{COLOR_TEXT_SECONDARY};">Explanation: {rec_a.detailed_explanation or "N/A"}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_pred_b:
            st.markdown(
                f"""
                <div style="background:#0B1220; border:1px solid #1E293B; border-radius:6px; padding:12px; font-family:monospace; font-size:12px;">
                    <div><strong>Scene B Locked Prediction:</strong> <code>{rec_b.extracted_prediction}</code></div>
                    <div style="margin-top:4px;"><strong>Format:</strong> {"VALID" if rec_b.is_valid_format else "INVALID"}</div>
                    <div style="margin-top:4px; color:{COLOR_TEXT_SECONDARY};">Explanation: {rec_b.detailed_explanation or "N/A"}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # -------------------------------------------------------------------
        # Step 8-10: Reveal Oracle Answers Together
        # -------------------------------------------------------------------
        st.write("")
        reveal_p_btn = st.button(
            "👁️ REVEAL PAIRED ORACLE RESULTS & CONSISTENCY VERDICT",
            type="primary",
            use_container_width=True,
        )

        if reveal_p_btn:
            st.session_state["arena_paired_revealed"] = True

        if st.session_state.get("arena_paired_revealed"):
            both_correct = (rec_a.final_score == 1.0) and (rec_b.final_score == 1.0)

            # Analyze relationship tracking
            pred_a_val = (
                rec_a.extracted_prediction.get("can_see")
                if isinstance(rec_a.extracted_prediction, dict)
                else rec_a.extracted_prediction
            )
            pred_b_val = (
                rec_b.extracted_prediction.get("can_see")
                if isinstance(rec_b.extracted_prediction, dict)
                else rec_b.extracted_prediction
            )

            gt_a_val = gt_a.get("can_see", True)
            gt_b_val = gt_b.get("can_see", True)

            tracking_success = False
            if expected_rel == "INVARIANT_OUTCOME":
                tracking_success = pred_a_val == pred_b_val
            elif expected_rel == "DIRECTIONAL_FLIP":
                tracking_success = pred_a_val != pred_b_val

            track_status = (
                f"<span style='color:{COLOR_STATUS_SUCCESS}; font-weight:700;'>✓ Successfully tracked {expected_rel}</span>"
                if tracking_success
                else f"<span style='color:{COLOR_STATUS_ERROR}; font-weight:700;'>✗ Failed {expected_rel}</span>"
            )

            # Impactful Paired Verdict Banner
            if both_correct:
                paired_banner = f"""
                <div style="background:linear-gradient(90deg, rgba(16,185,129,0.2) 0%, rgba(6,10,18,0.95) 100%); border:2px solid {COLOR_STATUS_SUCCESS}; border-radius:8px; padding:18px 24px; margin: 16px 0; box-shadow: 0 0 25px rgba(16,185,129,0.25);">
                    <div style="font-size:22px; font-weight:800; color:{COLOR_STATUS_SUCCESS}; letter-spacing:0.05em;">🎯 JOINT PAIR VERDICT: PERFECT (+1.0)</div>
                    <div style="font-size:14px; color:{COLOR_TEXT_PRIMARY}; margin-top:6px;">
                        The model correctly solved <strong>both related scenes</strong> (Truth A: <code>{gt_a_val}</code>, Truth B: <code>{gt_b_val}</code>).
                    </div>
                    <div style="font-size:13px; margin-top:6px;">Transformation Tracking: {track_status}</div>
                </div>
                """
            else:
                paired_banner = f"""
                <div style="background:linear-gradient(90deg, rgba(239,68,68,0.2) 0%, rgba(6,10,18,0.95) 100%); border:2px solid {COLOR_STATUS_ERROR}; border-radius:8px; padding:18px 24px; margin: 16px 0; box-shadow: 0 0 25px rgba(239,68,68,0.25);">
                    <div style="font-size:22px; font-weight:800; color:{COLOR_STATUS_ERROR}; letter-spacing:0.05em;">❌ JOINT PAIR VERDICT: INCONSISTENT (0.0)</div>
                    <div style="font-size:14px; color:{COLOR_TEXT_PRIMARY}; margin-top:6px;">
                        Model failed joint pair verification. Scene A Correct: <strong>{rec_a.final_score == 1.0}</strong> (Truth: <code>{gt_a_val}</code>) | Scene B Correct: <strong>{rec_b.final_score == 1.0}</strong> (Truth: <code>{gt_b_val}</code>).
                    </div>
                    <div style="font-size:13px; margin-top:6px;">Transformation Tracking: {track_status}</div>
                </div>
                """
            st.markdown(paired_banner, unsafe_allow_html=True)

            # Paired Overlays Side-by-Side
            st.markdown(
                "**Paired Geometric Explanations (Oracle Ground Truth Overlays):**"
            )
            col_ov_a, col_ov_b = st.columns(2)
            with col_ov_a:
                st.caption(
                    f"**Scene A Truth:** {gt_a.get('answer')} ({gt_a.get('reason_code')})"
                )
                if Path(overlay_a_path).exists():
                    st.image(overlay_a_path, use_container_width=True)
            with col_ov_b:
                st.caption(
                    f"**Scene B Truth:** {gt_b.get('answer')} ({gt_b.get('reason_code')})"
                )
                if Path(overlay_b_path).exists():
                    st.image(overlay_b_path, use_container_width=True)
