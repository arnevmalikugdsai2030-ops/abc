"""
Unit Tests for MIRRORLAB Professional UI Components and Design Tokens.

Tests:
1. Design tokens consistency (colors, fonts, layout dictionaries).
2. CSS injection stylesheet generation.
3. Verdict badge generation across all optical reason states.
4. Metric card and telemetry HTML rendering.
5. Dual-provider KPI row generation.
"""

from __future__ import annotations

from mirrorlab.ui.components import (
    render_comparison_kpi_row,
    render_verdict_badge,
)
from mirrorlab.ui.design_tokens import (
    COLOR_BG_APP,
    COLOR_BG_BASE,
    COLOR_BG_SURFACE,
    COLOR_CYAN_PRIMARY,
    COLOR_STATUS_ERROR,
    COLOR_STATUS_SUCCESS,
    COLOR_TARGET_RED,
    COLOR_TEXT_PRIMARY,
    FONT_INTERFACE,
    FONT_MONOSPACE,
    get_plotly_layout_dict,
    get_theme_css,
)


class TestDesignTokens:
    """Verifies scientific design tokens adhere to aesthetic specifications."""

    def test_color_tokens_valid_hex(self):
        for col in [
            COLOR_BG_BASE,
            COLOR_BG_APP,
            COLOR_BG_SURFACE,
            COLOR_CYAN_PRIMARY,
            COLOR_TARGET_RED,
            COLOR_TEXT_PRIMARY,
            COLOR_STATUS_SUCCESS,
            COLOR_STATUS_ERROR,
        ]:
            assert col.startswith("#")
            assert len(col) in (7, 9)

    def test_deep_navy_and_cyan_contrast(self):
        assert COLOR_BG_BASE == "#060A12"
        assert COLOR_CYAN_PRIMARY == "#00E5FF"
        assert COLOR_TARGET_RED == "#EF4444"

    def test_font_stacks_defined(self):
        assert "sans-serif" in FONT_INTERFACE
        assert "monospace" in FONT_MONOSPACE

    def test_plotly_layout_dict_structure(self):
        layout = get_plotly_layout_dict()
        assert layout["paper_bgcolor"] == COLOR_BG_BASE
        assert layout["plot_bgcolor"] == COLOR_BG_BASE
        assert "xaxis" in layout
        assert "yaxis" in layout
        assert layout["yaxis"]["scaleanchor"] == "x"
        assert layout["yaxis"]["scaleratio"] == 1

    def test_theme_css_generation(self):
        css = get_theme_css()
        assert "<style>" in css
        assert "</style>" in css
        assert COLOR_BG_APP in css
        assert "platform-header" in css
        assert "metric-card-container" in css
        assert "verdict-chip" in css


class TestUIComponents:
    """Tests reusable HTML/Streamlit components."""

    def test_verdict_badge_classes(self):
        chip_clear = render_verdict_badge("CLEAR", is_visible=True)
        assert "chip-visible" in chip_clear
        assert "VISIBLE" in chip_clear

        chip_blocked = render_verdict_badge("BLOCKED_CAMERA_TO_MIRROR", is_visible=False)
        assert "chip-blocked" in chip_blocked
        assert "BLOCKED" in chip_blocked

        chip_miss = render_verdict_badge("MIRROR_MISS", is_visible=False)
        assert "chip-miss" in chip_miss
        assert "MIRROR_MISS" in chip_miss

        chip_fov = render_verdict_badge("OUTSIDE_FOV", is_visible=False)
        assert "chip-fov" in chip_fov
        assert "OUTSIDE_FOV" in chip_fov

        chip_back = render_verdict_badge("WRONG_SIDE", is_visible=False)
        assert "chip-back" in chip_back
        assert "WRONG_SIDE" in chip_back

    def test_comparison_kpi_row_formatting(self):
        row = render_comparison_kpi_row(
            label="Overall Accuracy",
            val_gemini="80.0",
            val_openai="85.0",
            unit="%",
            winner="openai",
        )
        assert "Overall Accuracy" in row
        assert "80.0 %" in row
        assert "85.0 %" in row
        assert "|" in row
