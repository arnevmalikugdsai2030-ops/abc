"""
Unit Tests for MIRRORLAB Scene Laboratory.

Validates:
1. Dynamic oracle recalculation upon geometric modification.
2. In-memory task bundle ZIP package compilation and structure.
3. Blind diagram vs reference overlay separation.
4. Candidate mirror rendering for Task C.
5. Obstacle additions, deletions, and ray occlusion hit calculation.
"""

from __future__ import annotations

import io
import json
import zipfile

import pytest

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
from mirrorlab.ui.optical_plot import create_optical_figure
from mirrorlab.ui.scene_lab import _recalculate_scene_oracle, build_task_package_zip


@pytest.fixture
def base_lab_scene() -> InspectionScene:
    """Fixture providing a standard, clear inspection scene in Scene Lab."""
    cam = CameraSpec(position=Point2D(x=150.0, y=200.0), heading_deg=35.0, fov_degrees=60.0)
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
    return InspectionScene(
        scene_id="test_scene_lab_001",
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
        diagnostic_details="",
    )


def test_recalculate_scene_oracle(base_lab_scene: InspectionScene) -> None:
    """Verify that _recalculate_scene_oracle computes deterministic reflection paths."""
    updated_scene, oracle_res = _recalculate_scene_oracle(base_lab_scene)

    assert oracle_res.is_visible is True
    assert oracle_res.reason == ObstructionReason.NONE
    assert oracle_res.reflection_point is not None
    assert updated_scene.ground_truth_state == VisibilityState.VISIBLE
    assert updated_scene.ground_truth_reason == ObstructionReason.NONE
    assert len(updated_scene.optical_path) == 3


def test_oracle_recalculation_upon_obstacle_blocking(base_lab_scene: InspectionScene) -> None:
    """Verify oracle detects ray collision when an obstacle is moved into the incident leg."""
    # Place obstacle directly between Camera (150, 200) and Reflection Point (~500, 800)
    blocking_obs = ObstacleSpec(
        obstacle_id="O_blocker",
        obstacle_type="circle",
        center=Point2D(x=325.0, y=500.0),
        radius=40.0,
        angle_degrees=0.0,
    )
    blocked_scene = InspectionScene(
        scene_id="test_blocked",
        seed=base_lab_scene.seed,
        category=base_lab_scene.category,
        difficulty=base_lab_scene.difficulty,
        canvas_width=base_lab_scene.canvas_width,
        canvas_height=base_lab_scene.canvas_height,
        camera=base_lab_scene.camera,
        mirror=base_lab_scene.mirror,
        target=base_lab_scene.target,
        obstacles=[blocking_obs],
        ground_truth_state=VisibilityState.VISIBLE,
        ground_truth_reason=ObstructionReason.NONE,
        diagnostic_details="",
    )

    updated_scene, oracle_res = _recalculate_scene_oracle(blocked_scene)
    assert oracle_res.is_visible is False
    assert oracle_res.reason == ObstructionReason.BLOCKED_INCIDENT
    assert updated_scene.ground_truth_state == VisibilityState.NOT_VISIBLE
    assert oracle_res.occlusion_result is not None
    assert oracle_res.occlusion_result.is_path_blocked is True
    assert oracle_res.occlusion_result.first_blocker.obstacle_id == "O_blocker"


def test_task_package_zip_generation(base_lab_scene: InspectionScene) -> None:
    """Verify that build_task_package_zip packages all 6 required benchmark artifacts."""
    updated_scene, oracle_res = _recalculate_scene_oracle(base_lab_scene)
    zip_bytes = build_task_package_zip(
        scene=updated_scene,
        oracle_res=oracle_res,
        task_type_str="Task A: Reflected Visibility",
    )

    assert isinstance(zip_bytes, bytes)
    assert len(zip_bytes) > 5000  # Non-trivial ZIP payload

    # Inspect zip contents
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        namelist = zf.namelist()
        prefix = f"{updated_scene.scene_id}/"
        expected_files = [
            f"{prefix}scene.json",
            f"{prefix}question.json",
            f"{prefix}ground_truth.json",
            f"{prefix}diagram_blind.png",
            f"{prefix}diagram_overlay.png",
            f"{prefix}manifest.json",
        ]
        for ef in expected_files:
            assert ef in namelist, f"Missing {ef} in zip"

        # Validate scene.json content
        scene_data = json.loads(zf.read(f"{prefix}scene.json").decode("utf-8"))
        assert scene_data["scene_id"] == updated_scene.scene_id

        # Validate ground_truth.json content
        gt_data = json.loads(zf.read(f"{prefix}ground_truth.json").decode("utf-8"))
        assert gt_data["can_see"] is True
        assert gt_data["reason_code"] == "CLEAR"

        # Validate image payloads are non-empty PNGs
        blind_bytes = zf.read(f"{prefix}diagram_blind.png")
        assert blind_bytes.startswith(b"\x89PNG")
        overlay_bytes = zf.read(f"{prefix}diagram_overlay.png")
        assert overlay_bytes.startswith(b"\x89PNG")


def test_optical_figure_with_candidate_mirrors(base_lab_scene: InspectionScene) -> None:
    """Verify that create_optical_figure renders candidate mirrors for Task C (Repair)."""
    cand_mirrors = {
        "A": MirrorSpec(
            endpoint_a=Point2D(x=200.0, y=800.0),
            endpoint_b=Point2D(x=500.0, y=800.0),
            reflective_normal=Vector2D(dx=0.0, dy=-1.0),
        ),
        "B": MirrorSpec(
            endpoint_a=Point2D(x=350.0, y=820.0),
            endpoint_b=Point2D(x=650.0, y=820.0),
            reflective_normal=Vector2D(dx=0.0, dy=-1.0),
        ),
        "C": MirrorSpec(
            endpoint_a=Point2D(x=500.0, y=800.0),
            endpoint_b=Point2D(x=800.0, y=800.0),
            reflective_normal=Vector2D(dx=0.0, dy=-1.0),
        ),
    }

    fig = create_optical_figure(
        scene=base_lab_scene,
        candidate_mirrors=cand_mirrors,
        active_candidate="B",
        show_debug_rays=True,
    )

    trace_names = [t.name for t in fig.data if t.name]
    assert any("Candidate Mirror A" in name for name in trace_names)
    assert any("Candidate Mirror B (Active)" in name for name in trace_names)
    assert any("Candidate Mirror C" in name for name in trace_names)
