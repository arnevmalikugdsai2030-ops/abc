"""
Unit and Integration Tests for MIRRORLAB UI Components.

Tests Plotly schematic generation, data loading, scene building,
1:1 aspect ratio invariants, and theme tokens.
"""

from __future__ import annotations

import plotly.graph_objects as go
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
from mirrorlab.ui.data_loader import (
    build_interactive_scene,
    load_benchmark_manifest,
    load_sample_scenes,
    load_split_records,
    load_tasks_by_family,
)
from mirrorlab.ui.design_tokens import COLOR_BG_BASE
from mirrorlab.ui.optical_plot import create_optical_figure


@pytest.fixture
def sample_clear_scene() -> InspectionScene:
    """Fixture providing a standard unobstructed inspection scene."""
    return InspectionScene(
        scene_id="test_ui_scene_001",
        seed=101,
        category=SceneCategory.VALID_VISIBLE,
        difficulty=DifficultyLevel.STANDARD,
        canvas_width=1000.0,
        canvas_height=1000.0,
        camera=CameraSpec(
            position=Point2D(x=150.0, y=200.0),
            heading_deg=35.0,
            fov_degrees=60.0,
            max_range=1500.0,
        ),
        mirror=MirrorSpec(
            endpoint_a=Point2D(x=300.0, y=800.0),
            endpoint_b=Point2D(x=700.0, y=800.0),
            reflective_normal=Vector2D(dx=0.0, dy=-1.0),
        ),
        target=TargetSpec(
            center=Point2D(x=800.0, y=250.0),
            radius_mm=8.0,
        ),
        obstacles=[
            ObstacleSpec(
                obstacle_id="O_distract",
                obstacle_type="rectangle",
                center=Point2D(x=500.0, y=300.0),
                width=80.0,
                height=50.0,
                angle_degrees=0.0,
            )
        ],
        ground_truth_state=VisibilityState.VISIBLE,
        ground_truth_reason=ObstructionReason.NONE,
        diagnostic_details="Test fixture clear",
    )


def test_optical_figure_creation(sample_clear_scene: InspectionScene) -> None:
    """Verify that create_optical_figure generates a valid Plotly Figure."""
    fig = create_optical_figure(
        scene=sample_clear_scene,
        show_debug_rays=True,
        show_fov_cone=True,
        show_direct_sightline=True,
        theme="dark",
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0

    # Invariant: Strict 1:1 Euclidean aspect ratio must be enforced
    yaxis_dict = fig.layout.yaxis
    assert yaxis_dict.scaleanchor == "x"
    assert yaxis_dict.scaleratio == 1.0

    # Verify key visual traces exist
    trace_names = [t.name for t in fig.data if t.name]
    assert any("Mirror" in name for name in trace_names)
    assert any("Camera" in name for name in trace_names)
    assert any("Target" in name for name in trace_names)
    assert any("Incident Ray" in name for name in trace_names)


def test_optical_figure_themes(sample_clear_scene: InspectionScene) -> None:
    """Verify theme styling in Plotly schematic."""
    fig_dark = create_optical_figure(scene=sample_clear_scene, theme="dark")
    fig_light = create_optical_figure(scene=sample_clear_scene, theme="light")

    assert fig_dark.layout.plot_bgcolor == COLOR_BG_BASE
    assert fig_light.layout.plot_bgcolor == "#f8f9fa"


def test_interactive_scene_builder() -> None:
    """Verify dynamic scene creation from slider parameters."""
    scene = build_interactive_scene(
        cam_x=150.0,
        cam_y=200.0,
        cam_heading_deg=60.0,
        cam_fov_deg=80.0,
        mirror_cx=500.0,
        mirror_cy=800.0,
        mirror_length=400.0,
        mirror_angle_deg=0.0,
        flip_mirror_normal=False,
        target_x=800.0,
        target_y=250.0,
        obstacles=[],
    )

    assert isinstance(scene, InspectionScene)
    assert scene.camera.position.x == 150.0
    assert scene.target.center.x == 800.0
    assert scene.mirror.endpoint_a.x == 300.0
    assert scene.mirror.endpoint_b.x == 700.0
    # Normal points towards -Y (into cavity towards camera & target)
    assert scene.mirror.reflective_normal.dy < 0
    # Oracle evaluation populated and visible
    assert scene.ground_truth_state == VisibilityState.VISIBLE


def test_interactive_scene_builder_blocked() -> None:
    """Verify that adding an obstacle in the optical path causes obstruction."""
    # Obstacle placed directly in middle of incident leg
    blocker = ObstacleSpec(
        obstacle_id="O_block",
        obstacle_type="rectangle",
        center=Point2D(x=300.0, y=500.0),
        width=150.0,
        height=100.0,
        angle_degrees=0.0,
    )

    scene = build_interactive_scene(
        cam_x=150.0,
        cam_y=200.0,
        cam_heading_deg=60.0,
        cam_fov_deg=80.0,
        mirror_cx=500.0,
        mirror_cy=800.0,
        mirror_length=400.0,
        mirror_angle_deg=0.0,
        flip_mirror_normal=False,
        target_x=800.0,
        target_y=250.0,
        obstacles=[blocker],
    )

    assert scene.ground_truth_state == VisibilityState.NOT_VISIBLE
    assert scene.ground_truth_reason == ObstructionReason.BLOCKED_INCIDENT


def test_load_sample_scenes() -> None:
    """Verify loading pre-generated sample inspection scenes."""
    scenes = load_sample_scenes("data/sample_scenes")
    assert len(scenes) >= 10
    for s in scenes:
        assert isinstance(s, InspectionScene)
        assert s.canvas_width > 0


def test_load_tasks_by_family() -> None:
    """Verify loading benchmark tasks across supported families."""
    vis_tasks = load_tasks_by_family("visibility")
    obs_tasks = load_tasks_by_family("obstruction")
    rep_tasks = load_tasks_by_family("repair")
    pair_tasks = load_tasks_by_family("counterfactual")

    assert len(vis_tasks) > 0
    assert len(obs_tasks) > 0
    assert len(rep_tasks) > 0
    assert len(pair_tasks) > 0

    # Check structure of standard task
    t0 = vis_tasks[0]
    assert "id" in t0
    assert "dir" in t0
    assert "question" in t0

    # Check structure of counterfactual pair
    p0 = pair_tasks[0]
    assert "id" in p0
    assert "pair_data" in p0


def test_load_split_records() -> None:
    """Verify loading dataset splits."""
    dev_data = load_split_records(split_name="dev")
    assert "tasks" in dev_data
    assert "ground_truth" in dev_data
    assert len(dev_data["tasks"]) > 0
    assert len(dev_data["ground_truth"]) > 0


def test_load_benchmark_manifest() -> None:
    """Verify loading benchmark manifest."""
    manifest = load_benchmark_manifest()
    assert manifest is not None
    assert "split_counts" in manifest
    assert manifest["split_counts"]["dev"] == 40
    assert manifest["split_counts"]["val"] == 40
    assert manifest["split_counts"]["test"] == 120
