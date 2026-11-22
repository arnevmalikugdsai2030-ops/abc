"""
Interactive Streamlit Optical Studio and Inspector for MIRRORLAB.
"""

from mirrorlab.ui.components import (
    render_comparison_kpi_row,
    render_header,
    render_metric_card,
    render_telemetry_strip,
    render_verdict_badge,
)
from mirrorlab.ui.data_loader import (
    build_interactive_scene,
    load_benchmark_manifest,
    load_sample_scenes,
    load_split_records,
    load_tasks_by_family,
)
from mirrorlab.ui.design_tokens import (
    COLOR_BG_BASE,
    COLOR_CYAN_PRIMARY,
    COLOR_STATUS_ERROR,
    COLOR_STATUS_SUCCESS,
    COLOR_STATUS_WARNING,
    COLOR_TARGET_RED,
    COLOR_TEXT_PRIMARY,
    get_plotly_layout_dict,
    get_theme_css,
)
from mirrorlab.ui.optical_plot import create_optical_figure

__all__ = [
    "COLOR_BG_BASE",
    "COLOR_CYAN_PRIMARY",
    "COLOR_STATUS_ERROR",
    "COLOR_STATUS_SUCCESS",
    "COLOR_STATUS_WARNING",
    "COLOR_TARGET_RED",
    "COLOR_TEXT_PRIMARY",
    "build_interactive_scene",
    "create_optical_figure",
    "get_plotly_layout_dict",
    "get_theme_css",
    "load_benchmark_manifest",
    "load_sample_scenes",
    "load_split_records",
    "load_tasks_by_family",
    "render_comparison_kpi_row",
    "render_header",
    "render_metric_card",
    "render_telemetry_strip",
    "render_verdict_badge",
]
