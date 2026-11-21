"""
Design Tokens, Color Palettes, and Typography for MIRRORLAB Scientific Diagrams.

Supports professional aerospace and machine-vision schematic aesthetics:
- DARK_NAVY: Deep midnight navy background with crisp luminescent strokes.
- LIGHT_OFFWHITE: Industrial paper/clean lab off-white with high-contrast slate.
- BLUEPRINT: Engineering CAD cyan-on-navy technical drafting style.
- MONOCHROME_CAD: High-contrast black & white publication layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from PIL import ImageFont

# RGBA Color Tuple
ColorRGBA = tuple[int, int, int, int]
ColorRGB = tuple[int, int, int]


class RenderTheme(str, Enum):
    """Supported color themes for diagram rendering."""

    DARK_NAVY = "DARK_NAVY"
    LIGHT_OFFWHITE = "LIGHT_OFFWHITE"
    BLUEPRINT = "BLUEPRINT"
    MONOCHROME_CAD = "MONOCHROME_CAD"


class RenderMode(str, Enum):
    """Render mode: blind evaluation image vs complete answer-key overlay."""

    BLIND = "BLIND"
    OVERLAY = "OVERLAY"


@dataclass(frozen=True)
class ThemeTokens:
    """Complete color specification for a diagram visual theme."""

    name: RenderTheme
    background: ColorRGB
    grid_lines: ColorRGBA
    grid_border: ColorRGB
    text_primary: ColorRGB
    text_secondary: ColorRGB
    corner_brackets: ColorRGB

    # Camera tokens
    camera_body: ColorRGB
    camera_lens: ColorRGB
    camera_outline: ColorRGB
    camera_center_dot: ColorRGB
    fov_wedge: ColorRGBA
    fov_boundary: ColorRGBA

    # Mirror tokens
    mirror_reflective: ColorRGB
    mirror_backing: ColorRGB
    mirror_backing_hatch: ColorRGB
    mirror_endpoint: ColorRGB
    mirror_label: ColorRGB

    # Target defect tokens
    target_x: ColorRGB
    target_outline: ColorRGB
    target_label: ColorRGB

    # Obstacle tokens
    obstacle_fill: ColorRGB
    obstacle_outline: ColorRGB
    obstacle_label: ColorRGB
    obstacle_badge_bg: ColorRGB

    # Overlay / Answer-key tokens
    ray_incident: ColorRGB
    ray_reflected: ColorRGB
    ray_virtual: ColorRGBA
    reflection_point: ColorRGB
    normal_vector: ColorRGB
    badge_visible_bg: ColorRGB
    badge_visible_fg: ColorRGB
    badge_blocked_bg: ColorRGB
    badge_blocked_fg: ColorRGB


THEMES: dict[RenderTheme, ThemeTokens] = {
    RenderTheme.DARK_NAVY: ThemeTokens(
        name=RenderTheme.DARK_NAVY,
        background=(15, 23, 42),  # #0F172A Slate 900
        grid_lines=(30, 41, 59, 120),  # #1E293B
        grid_border=(51, 65, 85),  # #334155
        text_primary=(241, 245, 249),  # #F1F5F9
        text_secondary=(148, 163, 184),  # #94A3B8
        corner_brackets=(100, 116, 139),  # #64748B
        camera_body=(30, 58, 138),  # Deep Blue
        camera_lens=(59, 130, 246),  # Sky Blue
        camera_outline=(96, 165, 250),  # Light Blue
        camera_center_dot=(255, 255, 255),
        fov_wedge=(59, 130, 246, 30),  # Transparent cyan-blue
        fov_boundary=(59, 130, 246, 140),
        mirror_reflective=(6, 182, 212),  # Cyan 500
        mirror_backing=(15, 76, 92),  # Dark Teal
        mirror_backing_hatch=(14, 116, 144),  # Cyan 700
        mirror_endpoint=(255, 255, 255),
        mirror_label=(6, 182, 212),
        target_x=(239, 68, 68),  # Vibrant Red #EF4444
        target_outline=(255, 255, 255),
        target_label=(248, 113, 113),
        obstacle_fill=(30, 41, 59),  # Slate 800
        obstacle_outline=(71, 85, 105),  # Slate 600
        obstacle_label=(226, 232, 240),
        obstacle_badge_bg=(51, 65, 85),
        ray_incident=(245, 158, 11),  # Amber 500
        ray_reflected=(16, 185, 129),  # Emerald 500
        ray_virtual=(168, 85, 247, 180),  # Purple 500
        reflection_point=(250, 204, 21),  # Yellow 400
        normal_vector=(56, 189, 248),  # Light blue
        badge_visible_bg=(6, 95, 70),  # Dark Emerald
        badge_visible_fg=(209, 250, 229),
        badge_blocked_bg=(153, 27, 27),  # Dark Red
        badge_blocked_fg=(254, 226, 226),
    ),
    RenderTheme.LIGHT_OFFWHITE: ThemeTokens(
        name=RenderTheme.LIGHT_OFFWHITE,
        background=(248, 250, 252),  # #F8FAFC Slate 50
        grid_lines=(226, 232, 240, 160),  # #E2E8F0
        grid_border=(203, 213, 225),  # #CBD5E1
        text_primary=(15, 23, 42),  # #0F172A
        text_secondary=(100, 116, 139),  # #64748B
        corner_brackets=(148, 163, 184),
        camera_body=(219, 234, 254),  # Soft Blue Fill
        camera_lens=(37, 99, 235),  # Bold Royal Blue
        camera_outline=(30, 64, 175),
        camera_center_dot=(30, 64, 175),
        fov_wedge=(37, 99, 235, 25),  # Soft Blue Wedge
        fov_boundary=(37, 99, 235, 130),
        mirror_reflective=(2, 132, 199),  # Sky 600
        mirror_backing=(148, 163, 184),  # Slate 400
        mirror_backing_hatch=(100, 116, 139),
        mirror_endpoint=(15, 23, 42),
        mirror_label=(2, 132, 199),
        target_x=(220, 38, 38),  # Red 600
        target_outline=(255, 255, 255),
        target_label=(185, 28, 28),
        obstacle_fill=(241, 245, 249),  # Slate 100
        obstacle_outline=(71, 85, 105),  # Slate 600
        obstacle_label=(15, 23, 42),
        obstacle_badge_bg=(226, 232, 240),
        ray_incident=(217, 119, 6),  # Amber 600
        ray_reflected=(5, 150, 105),  # Emerald 600
        ray_virtual=(147, 51, 234, 180),  # Purple 600
        reflection_point=(202, 138, 4),  # Yellow 600
        normal_vector=(2, 132, 199),
        badge_visible_bg=(209, 250, 229),
        badge_visible_fg=(6, 95, 70),
        badge_blocked_bg=(254, 226, 226),
        badge_blocked_fg=(153, 27, 27),
    ),
    RenderTheme.BLUEPRINT: ThemeTokens(
        name=RenderTheme.BLUEPRINT,
        background=(10, 37, 64),  # CAD Blue #0A2540
        grid_lines=(20, 60, 100, 140),
        grid_border=(40, 90, 140),
        text_primary=(255, 255, 255),
        text_secondary=(160, 200, 240),
        corner_brackets=(0, 240, 255),
        camera_body=(18, 55, 95),
        camera_lens=(0, 220, 255),
        camera_outline=(0, 240, 255),
        camera_center_dot=(255, 255, 255),
        fov_wedge=(0, 220, 255, 30),
        fov_boundary=(0, 240, 255, 160),
        mirror_reflective=(0, 255, 255),  # Pure Cyan
        mirror_backing=(15, 45, 75),
        mirror_backing_hatch=(0, 180, 210),
        mirror_endpoint=(255, 255, 255),
        mirror_label=(0, 255, 255),
        target_x=(255, 80, 80),  # Bright Red
        target_outline=(255, 255, 255),
        target_label=(255, 120, 120),
        obstacle_fill=(16, 48, 80),
        obstacle_outline=(60, 120, 180),
        obstacle_label=(255, 255, 255),
        obstacle_badge_bg=(30, 80, 130),
        ray_incident=(255, 190, 0),
        ray_reflected=(0, 255, 150),
        ray_virtual=(210, 120, 255, 180),
        reflection_point=(255, 255, 0),
        normal_vector=(0, 220, 255),
        badge_visible_bg=(0, 100, 60),
        badge_visible_fg=(0, 255, 150),
        badge_blocked_bg=(140, 20, 20),
        badge_blocked_fg=(255, 180, 180),
    ),
    RenderTheme.MONOCHROME_CAD: ThemeTokens(
        name=RenderTheme.MONOCHROME_CAD,
        background=(255, 255, 255),
        grid_lines=(230, 230, 230, 180),
        grid_border=(180, 180, 180),
        text_primary=(0, 0, 0),
        text_secondary=(90, 90, 90),
        corner_brackets=(60, 60, 60),
        camera_body=(235, 235, 235),
        camera_lens=(50, 50, 50),
        camera_outline=(0, 0, 0),
        camera_center_dot=(0, 0, 0),
        fov_wedge=(150, 150, 150, 25),
        fov_boundary=(100, 100, 100, 150),
        mirror_reflective=(0, 0, 0),
        mirror_backing=(160, 160, 160),
        mirror_backing_hatch=(100, 100, 100),
        mirror_endpoint=(0, 0, 0),
        mirror_label=(0, 0, 0),
        target_x=(200, 0, 0),
        target_outline=(255, 255, 255),
        target_label=(200, 0, 0),
        obstacle_fill=(240, 240, 240),
        obstacle_outline=(0, 0, 0),
        obstacle_label=(0, 0, 0),
        obstacle_badge_bg=(220, 220, 220),
        ray_incident=(50, 50, 50),
        ray_reflected=(0, 0, 0),
        ray_virtual=(120, 120, 120, 180),
        reflection_point=(0, 0, 0),
        normal_vector=(80, 80, 80),
        badge_visible_bg=(230, 230, 230),
        badge_visible_fg=(0, 0, 0),
        badge_blocked_bg=(200, 200, 200),
        badge_blocked_fg=(0, 0, 0),
    ),
}


def get_font(
    size: int, bold: bool = False
) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Load system TrueType font or fallback to Pillow default font."""
    font_names = (
        ["arialbd.ttf", "segoeuib.ttf", "calibrib.ttf", "Arial Bold.ttf"]
        if bold
        else ["arial.ttf", "segoeui.ttf", "calibri.ttf", "Arial.ttf"]
    )

    for name in font_names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue

    # Fallback to default
    return ImageFont.load_default()
