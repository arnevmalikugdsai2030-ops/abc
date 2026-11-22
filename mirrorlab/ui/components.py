"""
MIRRORLAB Professional Reusable UI Components.

Provides clean, typed HTML/Streamlit components adhering to the design specification:
- Header & Engine Telemetry Status Bar.
- High-Density Scientific Metric Cards.
- Standardized Verdict and Status Chips.
- Parameter Telemetry Strips.
- Dual-Provider Benchmark Observability Tables.
"""

from __future__ import annotations

import streamlit as st

from mirrorlab.ui.design_tokens import (
    COLOR_CYAN_PRIMARY,
    COLOR_TEXT_MUTED,
)


def render_header(
    title: str = "MIRRORLAB",
    version: str = "v1.0.0-CVPR",
    subtitle: str = "Industrial Specular Reflection Computer Vision Benchmark",
    engine_status: str = "ORACLE ACTIVE: DETERMINISTIC",
) -> None:
    """Renders the top scientific platform navigation & telemetry status bar."""
    html = f"""
    <div class="platform-header">
        <div class="platform-title-area">
            <h1 class="platform-title">{title}</h1>
            <span class="platform-version">{version}</span>
            <p class="platform-subtitle">| {subtitle}</p>
        </div>
        <div class="engine-status-pill">
            <span style="display:inline-block; width:7px; height:7px; border-radius:50%; background:#10b981;"></span>
            {engine_status}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_metric_card(
    label: str,
    value: str,
    subtext: str | None = None,
    color: str | None = None,
) -> None:
    """Renders a high-density scientific metric card with monospace value."""
    text_color = color or COLOR_CYAN_PRIMARY
    subtext_html = (
        f'<div class="metric-card-subtext">{subtext}</div>' if subtext else ""
    )
    html = f"""
    <div class="metric-card-container">
        <div class="metric-card-label">{label}</div>
        <div class="metric-card-value" style="color: {text_color};">{value}</div>
        {subtext_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_verdict_badge(
    reason_code: str,
    is_visible: bool | None = None,
) -> str:
    """
    Returns an HTML verdict chip string with appropriate semantic styling.
    Can be used inline or embedded in tables and cards.
    """
    code_upper = str(reason_code).upper().strip()

    if "CLEAR" in code_upper or is_visible is True or code_upper in ("VISIBLE", "YES", "PASS"):
        chip_class = "chip-visible"
        label = "VISIBLE // CLEAR"
    elif "BLOCK" in code_upper:
        chip_class = "chip-blocked"
        label = f"BLOCKED // {code_upper}"
    elif "MISS" in code_upper or "MIRROR" in code_upper:
        chip_class = "chip-miss"
        label = "OCCLUDED // MIRROR_MISS"
    elif "FOV" in code_upper:
        chip_class = "chip-fov"
        label = "OCCLUDED // OUTSIDE_FOV"
    elif "WRONG" in code_upper or "SIDE" in code_upper:
        chip_class = "chip-back"
        label = "OCCLUDED // WRONG_SIDE"
    else:
        chip_class = "chip-blocked"
        label = f"NON-VISIBLE // {code_upper}"

    return f'<span class="verdict-chip {chip_class}">{label}</span>'


def render_telemetry_strip(items: list[tuple[str, str]]) -> None:
    """Renders a compact, multi-column telemetry strip for numerical parameters."""
    items_html = "".join(
        f"""
        <div class="telemetry-item">
            <span class="telemetry-label">{label}</span>
            <span class="telemetry-val">{val}</span>
        </div>
        """
        for label, val in items
    )
    html = f'<div class="telemetry-strip">{items_html}</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_comparison_kpi_row(
    label: str,
    val_gemini: str,
    val_openai: str,
    unit: str = "%",
    winner: str | None = None,
) -> str:
    """Builds a formatted table row comparing Gemini and OpenAI."""
    gemini_color = COLOR_CYAN_PRIMARY if winner == "gemini" else COLOR_TEXT_MUTED
    openai_color = COLOR_CYAN_PRIMARY if winner == "openai" else COLOR_TEXT_MUTED
    return (
        f"| **{label}** | <span style='font-family:monospace; color:{gemini_color}'>{val_gemini} {unit}</span> "
        f"| <span style='font-family:monospace; color:{openai_color}'>{val_openai} {unit}</span> |"
    )
