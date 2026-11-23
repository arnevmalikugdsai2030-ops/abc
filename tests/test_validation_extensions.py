"""
Unit Tests for Phase 2 Validation Extensions:
1. RenderingConsistencyValidator: pixel probes for target, camera, mirror, obstacles, and blind non-leakage.
2. AmbiguityValidator: endpoint grazing, obstacle clearance, FOV margins, and object separation.
"""

from pathlib import Path

import pytest

from mirrorlab.evaluation.ambiguity_validator import AmbiguityValidator
from mirrorlab.evaluation.rendering_validator import RenderingConsistencyValidator
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


@pytest.fixture
def sample_valid_scene() -> InspectionScene:
    """Creates a well-separated, unambiguous inspection scene."""
    cam = CameraSpec(
        position=Point2D(x=200.0, y=200.0), heading_deg=65.0, fov_degrees=65.0
    )
    mirror = MirrorSpec(
        endpoint_a=Point2D(x=150.0, y=700.0),
        endpoint_b=Point2D(x=650.0, y=700.0),
        reflective_normal=Vector2D(dx=0.0, dy=-1.0),
    )
    target = TargetSpec(center=Point2D(x=500.0, y=250.0), radius_mm=8.0)
    obstacles = [
        ObstacleSpec(
            obstacle_id="O1",
            obstacle_type="rectangle",
            center=Point2D(x=350.0, y=225.0),
            width=50.0,
            height=150.0,
            angle_degrees=0.0,
        )
    ]
    return InspectionScene(
        scene_id="test_scene_ambiguity_01",
        seed=101,
        category=SceneCategory.VALID_VISIBLE,
        difficulty=DifficultyLevel.STANDARD,
        ground_truth_state=VisibilityState.VISIBLE,
        ground_truth_reason=ObstructionReason.NONE,
        diagnostic_details="Test scene",
        camera=cam,
        mirror=mirror,
        target=target,
        obstacles=obstacles,
        reflection_point=Point2D(x=328.57, y=700.0),
        virtual_target=Point2D(x=500.0, y=1150.0),
    )


class TestAmbiguityValidator:
    """Tests geometric ambiguity detection rules."""

    def test_unambiguous_scene_passes(self, sample_valid_scene: InspectionScene):
        report = AmbiguityValidator.validate_scene(sample_valid_scene)
        assert report.is_unambiguous is True
        assert len(report.rejection_reasons) == 0

    def test_endpoint_graze_rejected(self, sample_valid_scene: InspectionScene):
        # Move reflection point to within 1.0 mm of endpoint A (150, 700)
        grazing_scene = sample_valid_scene.model_copy(
            update={"reflection_point": Point2D(x=150.8, y=700.0)}
        )
        report = AmbiguityValidator.validate_scene(grazing_scene)
        assert report.is_unambiguous is False
        assert "AMBIGUOUS_MIRROR_ENDPOINT_GRAZE" in report.rejection_reasons

    def test_object_overlap_rejected(self, sample_valid_scene: InspectionScene):
        # Place obstacle right on top of camera
        overlapping_obs = [
            ObstacleSpec(
                obstacle_id="O_overlap",
                obstacle_type="circle",
                center=Point2D(x=202.0, y=202.0),
                radius=10.0,
            )
        ]
        crowded_scene = sample_valid_scene.model_copy(
            update={"obstacles": overlapping_obs}
        )
        report = AmbiguityValidator.validate_scene(crowded_scene)
        assert report.is_unambiguous is False
        assert any("OBJECT_COLLISION" in r for r in report.rejection_reasons)


class TestRenderingConsistencyValidator:
    """Tests pixel-level rendering fidelity probes."""

    def test_rendered_image_passes_all_probes(
        self, sample_valid_scene: InspectionScene, tmp_path: Path
    ):
        img_path = tmp_path / "test_diagram.png"
        renderer = PillowDiagramRenderer(theme=RenderTheme.DARK_NAVY)
        img = renderer.render(sample_valid_scene, mode=RenderMode.BLIND)
        img.save(img_path)

        report = RenderingConsistencyValidator.validate(
            sample_valid_scene, img_path, is_blind_image=True
        )

        assert report.all_passed is True
        assert report.image_width == 1024
        assert report.image_height == 1024

        element_names = [e.element_name for e in report.element_results]
        assert "target_defect_red_cross" in element_names
        assert "camera_apparatus" in element_names
        assert "mirror_reflective_surface" in element_names
        assert "obstacle_O1" in element_names
        assert "blind_non_leakage_optical_corridor" in element_names

    def test_missing_image_fails_gracefully(
        self, sample_valid_scene: InspectionScene, tmp_path: Path
    ):
        non_existent = tmp_path / "does_not_exist.png"
        report = RenderingConsistencyValidator.validate(
            sample_valid_scene, non_existent
        )
        assert report.all_passed is False
        assert "missing" in report.summary.lower()
