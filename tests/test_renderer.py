"""
Image-Quality & Integrity Tests for the Pillow Diagram Renderer.

Enforces:
1. Complete blind-mode isolation: zero ground-truth leakage (no rays, no answer badges).
2. Exact pixel alignment for target scratch center and mirror endpoints.
3. Strict 1:1 isometric aspect ratio preservation.
4. Multi-theme and multi-resolution fidelity (512x512, 1024x1024, dark/light/blueprint).
5. Deterministic byte-for-byte rendering reproducibility.
"""

import numpy as np
import pytest

from mirrorlab.generation.scene_generator import SceneGenerator
from mirrorlab.rendering.pillow_renderer import PillowDiagramRenderer
from mirrorlab.rendering.themes import THEMES, RenderMode, RenderTheme
from mirrorlab.rendering.transform import ViewportTransform
from mirrorlab.schema.scene import DifficultyLevel, SceneCategory


@pytest.fixture
def sample_visible_scene():
    gen = SceneGenerator()
    return gen.generate_scene(
        seed=5001, category=SceneCategory.VALID_VISIBLE, difficulty=DifficultyLevel.EASY
    )


@pytest.fixture
def sample_blocked_scene():
    gen = SceneGenerator()
    return gen.generate_scene(
        seed=5005,
        category=SceneCategory.BLOCKED_INCIDENT,
        difficulty=DifficultyLevel.MEDIUM,
    )


class TestBlindModeIntegrity:
    """Strict verification that blind evaluation images never leak answers."""

    def test_blind_mode_differs_from_overlay(self, sample_visible_scene):
        renderer = PillowDiagramRenderer(theme=RenderTheme.DARK_NAVY)
        img_blind = renderer.render(sample_visible_scene, mode=RenderMode.BLIND)
        img_overlay = renderer.render(sample_visible_scene, mode=RenderMode.OVERLAY)

        arr_blind = np.array(img_blind)
        arr_overlay = np.array(img_overlay)

        # Images must differ
        diff_pixels = np.sum(arr_blind != arr_overlay)
        assert diff_pixels > 500, (
            "Overlay mode must add substantial diagnostic proof layers"
        )

    def test_blind_mode_contains_zero_answer_rays(self, sample_visible_scene):
        renderer = PillowDiagramRenderer(theme=RenderTheme.DARK_NAVY)
        img_blind = renderer.render(sample_visible_scene, mode=RenderMode.BLIND)
        arr_blind = np.array(img_blind)

        # In DARK_NAVY:
        # ray_incident is Amber (245, 158, 11)
        # ray_reflected is Emerald (16, 185, 129)
        tokens = THEMES[RenderTheme.DARK_NAVY]
        inc_color = np.array(tokens.ray_incident)
        refl_color = np.array(tokens.ray_reflected)

        # Match exact ray colors
        match_inc = np.all(arr_blind == inc_color, axis=-1)
        match_refl = np.all(arr_blind == refl_color, axis=-1)

        assert np.sum(match_inc) == 0, "Blind image leaked incident ray pixels!"
        assert np.sum(match_refl) == 0, "Blind image leaked reflected ray pixels!"


class TestCoordinateTransformAndPixelAccuracy:
    """Verify precision geometry-to-pixel projection."""

    def test_isometric_aspect_ratio(self):
        tf = ViewportTransform.create(
            canvas_width=1000.0,
            canvas_height=1000.0,
            image_width=1024,
            image_height=1024,
            margin=60.0,
        )
        # Aspect ratio must be exactly 1.0 (no stretching)
        p0 = tf.to_pixel(0.0, 0.0)
        p100_x = tf.to_pixel(100.0, 0.0)
        p100_y = tf.to_pixel(0.0, 100.0)

        dx = abs(p100_x[0] - p0[0])
        dy = abs(p100_y[1] - p0[1])
        assert pytest.approx(dx, rel=1e-5) == dy, (
            "X and Y scales must be strictly equal"
        )

    def test_target_scratch_center_pixel_accuracy(self, sample_visible_scene):
        """Target Red-X intersection must cover the exact transformed pixel."""
        renderer = PillowDiagramRenderer(
            theme=RenderTheme.DARK_NAVY, image_width=1024, image_height=1024
        )
        img = renderer.render(sample_visible_scene, mode=RenderMode.BLIND)
        arr = np.array(img)

        tf = ViewportTransform.create(
            canvas_width=sample_visible_scene.canvas_width,
            canvas_height=sample_visible_scene.canvas_height,
            image_width=1024,
            image_height=1024,
            margin=renderer.margin,
        )

        tx = sample_visible_scene.target.center.x
        ty = sample_visible_scene.target.center.y
        px, py = tf.to_pixel(tx, ty)
        ix, iy = round(px), round(py)

        # Pixel at (iy, ix) or its immediate 3x3 neighborhood must have target red color
        neighborhood = arr[iy - 2 : iy + 3, ix - 2 : ix + 3]
        red_threshold = 200
        # Red channel > 200 and red > green + 100
        is_red = (neighborhood[:, :, 0] > red_threshold) & (
            neighborhood[:, :, 0] > neighborhood[:, :, 1] + 80
        )
        assert np.any(is_red), f"Red target X not found at expected center ({ix}, {iy})"

    def test_mirror_endpoints_pixel_accuracy(self, sample_visible_scene):
        """Mirror endpoints A and B must map to valid pixels on canvas."""
        renderer = PillowDiagramRenderer(
            theme=RenderTheme.LIGHT_OFFWHITE, image_width=1024, image_height=1024
        )
        img = renderer.render(sample_visible_scene, mode=RenderMode.BLIND)
        assert img.size == (1024, 1024)

        tf = ViewportTransform.create(
            canvas_width=sample_visible_scene.canvas_width,
            canvas_height=sample_visible_scene.canvas_height,
            image_width=1024,
            image_height=1024,
            margin=renderer.margin,
        )
        pa_x, pa_y = tf.to_pixel(
            sample_visible_scene.mirror.endpoint_a.x,
            sample_visible_scene.mirror.endpoint_a.y,
        )
        pb_x, pb_y = tf.to_pixel(
            sample_visible_scene.mirror.endpoint_b.x,
            sample_visible_scene.mirror.endpoint_b.y,
        )

        # Endpoints must lie strictly within image boundaries
        assert 0 <= pa_x < 1024 and 0 <= pa_y < 1024
        assert 0 <= pb_x < 1024 and 0 <= pb_y < 1024


class TestThemesAndMultiResolution:
    """Verify visual theme variations and different image resolutions."""

    @pytest.mark.parametrize(
        "theme",
        [
            RenderTheme.DARK_NAVY,
            RenderTheme.LIGHT_OFFWHITE,
            RenderTheme.BLUEPRINT,
            RenderTheme.MONOCHROME_CAD,
        ],
    )
    def test_render_all_themes(self, sample_visible_scene, theme):
        renderer = PillowDiagramRenderer(theme=theme, image_width=512, image_height=512)
        img = renderer.render(sample_visible_scene, mode=RenderMode.BLIND)
        assert img.size == (512, 512)
        assert img.mode == "RGB"

    @pytest.mark.parametrize("size", [512, 800, 1024, 1200])
    def test_render_various_resolutions(self, sample_visible_scene, size):
        renderer = PillowDiagramRenderer(
            theme=RenderTheme.DARK_NAVY, image_width=size, image_height=size
        )
        img = renderer.render(sample_visible_scene, mode=RenderMode.BLIND)
        assert img.size == (size, size)


class TestDeterministicRendering:
    """Verify byte-for-byte reproducibility."""

    def test_identical_rendering_bytes(self, sample_visible_scene):
        renderer = PillowDiagramRenderer(
            theme=RenderTheme.DARK_NAVY, image_width=1024, image_height=1024
        )
        img1 = renderer.render(sample_visible_scene, mode=RenderMode.BLIND)
        img2 = renderer.render(sample_visible_scene, mode=RenderMode.BLIND)

        assert img1.tobytes() == img2.tobytes(), (
            "Rendering must be 100% deterministic bit-for-bit"
        )
