"""
MIRRORLAB Professional Design Tokens & Visual Architecture.

Defines standardized engineering-grade design tokens for:
- Deep navy background surfaces.
- Cyan optical geometry accents.
- Red target defect markers.
- Neutral high-contrast greyscale typography.
- Restrained, non-distracting visual hierarchy.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Color Tokens
# ---------------------------------------------------------------------------

# Background & Surface Tokens (Deep Navy / Industrial Dark)
COLOR_BG_BASE = "#060A12"  # Root viewport & plot canvas
COLOR_BG_APP = "#0A101D"  # Application container background
COLOR_BG_SURFACE = "#0F172A"  # Primary cards, sidebars, containers
COLOR_BG_SURFACE_HOVER = "#162032"  # Interactive hover state
COLOR_BG_ELEVATED = "#1A2438"  # Telemetry strips, modals, dropdowns
COLOR_BG_CODE = "#080D17"  # Code blocks and raw response containers

# Border & Divider Tokens
COLOR_BORDER_SUBTLE = "#1E293B"  # Structural dividers, card perimeters
COLOR_BORDER_MUTED = "#334155"  # Active inputs, column outlines
COLOR_BORDER_FOCUS = "#00E5FF"  # Active tab, selection indicator

# Optical Geometry Tokens (Cyan & Laser Ray Path)
COLOR_CYAN_PRIMARY = "#00E5FF"  # Primary reflection ray, active geometry
COLOR_CYAN_SECONDARY = "#00B4D8"  # Secondary lines, normal vectors
COLOR_CYAN_DARK = "#0077B6"  # Subtle bounding accents
COLOR_RAY_INCIDENT = "#F59E0B"  # Incident ray (Camera -> Mirror)
COLOR_RAY_REFLECTED = "#10B981"  # Reflected ray (Mirror -> Target)
COLOR_RAY_VIRTUAL = "#A855F7"  # Virtual camera line / sightline cone

# Target Defect Tokens (Red Scratch / Defect Marker)
COLOR_TARGET_RED = "#EF4444"  # Scratch defect center and crosshair
COLOR_DEFECT_RED = COLOR_TARGET_RED  # Semantic alias for defect marker
COLOR_TARGET_RING = "#DC2626"  # Tolerance ring / outer target boundary
COLOR_TARGET_BG = "rgba(239, 68, 68, 0.12)"  # Target boundary halo (low opacity)

# Mirror & Obstacle Anatomy Tokens
COLOR_MIRROR_FACE = "#38BDF8"  # Specular reflective front surface
COLOR_MIRROR_BACK = "#475569"  # Non-reflective backing & hatch marks
COLOR_CAMERA_BODY = "#60A5FA"  # Camera housing & lens barrel
COLOR_CAMERA_FOV = "rgba(96, 165, 250, 0.08)"  # Camera FOV angular cone fill
COLOR_OBSTACLE_FILL = "#1E293B"  # Matte slate barrier polygon fill
COLOR_OBSTACLE_STROKE = "#475569"  # Barrier perimeter stroke

# Text & Typography Tokens (Neutral Greyscale Tiers)
COLOR_TEXT_PRIMARY = "#F8FAFC"  # High-emphasis headings, values (100% white)
COLOR_TEXT_SECONDARY = "#CBD5E1"  # Standard labels, body copy
COLOR_TEXT_MUTED = "#94A3B8"  # Microcopy, unit annotations, captions
COLOR_TEXT_DIM = "#64748B"  # Table headers, disabled placeholders

# Semantic Status Tokens
COLOR_STATUS_SUCCESS = "#10B981"  # Valid, visible, passed
COLOR_STATUS_ERROR = "#EF4444"  # Blocked, missed, failed
COLOR_STATUS_WARNING = "#F59E0B"  # Mirror miss, aperture clipping
COLOR_STATUS_INFO = "#38BDF8"  # Observability info, metadata
COLOR_STATUS_PURPLE = "#A855F7"  # FOV cone exceedance

# ---------------------------------------------------------------------------
# Typography Stacks
# ---------------------------------------------------------------------------
FONT_INTERFACE = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", "Helvetica Neue", Arial, sans-serif'
)
FONT_MONOSPACE = (
    '"JetBrains Mono", "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace'
)

# ---------------------------------------------------------------------------
# Plotly Engineering CAD Theme Configuration
# ---------------------------------------------------------------------------
def get_plotly_layout_dict(height: int | None = None, width: int | None = None, **kwargs: Any) -> dict[str, Any]:
    """Returns standardized Plotly layout dictionary for deep navy CAD viewport."""
    layout: dict[str, Any] = {
        "paper_bgcolor": COLOR_BG_BASE,
        "plot_bgcolor": COLOR_BG_BASE,
        "font": {
            "family": FONT_MONOSPACE,
            "color": COLOR_TEXT_MUTED,
            "size": 11,
        },
        "margin": {"l": 50, "r": 30, "t": 35, "b": 45},
        "xaxis": {
            "gridcolor": "#131C2D",
            "zerolinecolor": "#1E293B",
            "zerolinewidth": 1.5,
            "tickfont": {"family": FONT_MONOSPACE, "size": 10, "color": COLOR_TEXT_DIM},
            "title": {"font": {"family": FONT_INTERFACE, "size": 11, "color": COLOR_TEXT_MUTED}},
            "showgrid": True,
            "gridwidth": 1,
        },
        "yaxis": {
            "gridcolor": "#131C2D",
            "zerolinecolor": "#1E293B",
            "zerolinewidth": 1.5,
            "tickfont": {"family": FONT_MONOSPACE, "size": 10, "color": COLOR_TEXT_DIM},
            "title": {"font": {"family": FONT_INTERFACE, "size": 11, "color": COLOR_TEXT_MUTED}},
            "showgrid": True,
            "gridwidth": 1,
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        "legend": {
            "bgcolor": "rgba(15, 23, 42, 0.85)",
            "bordercolor": COLOR_BORDER_SUBTLE,
            "borderwidth": 1,
            "font": {"family": FONT_INTERFACE, "size": 11, "color": COLOR_TEXT_SECONDARY},
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0.0,
        },
    }
    if height is not None:
        layout["height"] = height
    if width is not None:
        layout["width"] = width
    layout.update(kwargs)
    return layout


# ---------------------------------------------------------------------------
# Complete Streamlit CSS Stylesheet
# ---------------------------------------------------------------------------
def get_theme_css() -> str:
    """
    Returns complete, engineering-grade CSS stylesheet for Streamlit.
    Enforces deep navy canvas, crisp monospace telemetry, restrained borders,
    and high-contrast visual accessibility.
    """
    return f"""
    <style>
    /* -----------------------------------------------------------------------
       Root App & Canvas Styling
       ----------------------------------------------------------------------- */
    .stApp {{
        background-color: {COLOR_BG_APP} !important;
        color: {COLOR_TEXT_SECONDARY} !important;
        font-family: {FONT_INTERFACE} !important;
    }}

    /* Global Headings */
    h1, h2, h3, h4, h5, h6 {{
        color: {COLOR_TEXT_PRIMARY} !important;
        font-weight: 600 !important;
        letter-spacing: -0.015em !important;
    }}

    /* Top Platform Header */
    .platform-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 18px;
        background: {COLOR_BG_SURFACE};
        border: 1px solid {COLOR_BORDER_SUBTLE};
        border-radius: 6px;
        margin-bottom: 20px;
    }}
    .platform-title-area {{
        display: flex;
        align-items: baseline;
        gap: 12px;
    }}
    .platform-title {{
        font-size: 1.25rem;
        font-weight: 700;
        color: {COLOR_TEXT_PRIMARY};
        letter-spacing: -0.02em;
        margin: 0;
    }}
    .platform-version {{
        font-family: {FONT_MONOSPACE};
        font-size: 0.72rem;
        background: {COLOR_BG_ELEVATED};
        color: {COLOR_CYAN_PRIMARY};
        padding: 2px 7px;
        border-radius: 4px;
        border: 1px solid {COLOR_BORDER_MUTED};
        letter-spacing: 0.04em;
    }}
    .platform-subtitle {{
        font-size: 0.82rem;
        color: {COLOR_TEXT_MUTED};
        margin: 0;
    }}
    .engine-status-pill {{
        font-family: {FONT_MONOSPACE};
        font-size: 0.72rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 4px;
        background: rgba(16, 185, 129, 0.12);
        color: {COLOR_STATUS_SUCCESS};
        border: 1px solid rgba(16, 185, 129, 0.35);
        display: flex;
        align-items: center;
        gap: 6px;
    }}

    /* -----------------------------------------------------------------------
       Scientific Metric Card
       ----------------------------------------------------------------------- */
    .metric-card-container {{
        background: {COLOR_BG_SURFACE};
        border: 1px solid {COLOR_BORDER_SUBTLE};
        border-radius: 6px;
        padding: 14px 16px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 94px;
        transition: border-color 0.15s ease;
    }}
    .metric-card-container:hover {{
        border-color: {COLOR_BORDER_MUTED};
    }}
    .metric-card-label {{
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: {COLOR_TEXT_MUTED};
        font-weight: 600;
        margin-bottom: 4px;
    }}
    .metric-card-value {{
        font-family: {FONT_MONOSPACE};
        font-size: 1.65rem;
        font-weight: 700;
        color: {COLOR_CYAN_PRIMARY};
        line-height: 1.1;
        letter-spacing: -0.02em;
    }}
    .metric-card-subtext {{
        font-size: 0.74rem;
        color: {COLOR_TEXT_DIM};
        margin-top: 5px;
        font-family: {FONT_MONOSPACE};
    }}

    /* -----------------------------------------------------------------------
       Telemetry Strip & Parameter Grid
       ----------------------------------------------------------------------- */
    .telemetry-strip {{
        background: {COLOR_BG_BASE};
        border: 1px solid {COLOR_BORDER_SUBTLE};
        border-radius: 5px;
        padding: 10px 14px;
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 12px;
        margin: 12px 0;
    }}
    .telemetry-item {{
        display: flex;
        flex-direction: column;
    }}
    .telemetry-label {{
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: {COLOR_TEXT_DIM};
    }}
    .telemetry-val {{
        font-family: {FONT_MONOSPACE};
        font-size: 0.88rem;
        color: {COLOR_TEXT_PRIMARY};
        font-weight: 600;
    }}

    /* -----------------------------------------------------------------------
       Status & Verdict Badges
       ----------------------------------------------------------------------- */
    .verdict-chip {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 9px;
        border-radius: 4px;
        font-size: 0.76rem;
        font-weight: 700;
        font-family: {FONT_MONOSPACE};
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }}
    .chip-visible {{
        background: rgba(16, 185, 129, 0.12);
        color: {COLOR_STATUS_SUCCESS};
        border: 1px solid rgba(16, 185, 129, 0.35);
    }}
    .chip-blocked {{
        background: rgba(239, 68, 68, 0.12);
        color: {COLOR_STATUS_ERROR};
        border: 1px solid rgba(239, 68, 68, 0.35);
    }}
    .chip-miss {{
        background: rgba(245, 158, 11, 0.12);
        color: {COLOR_STATUS_WARNING};
        border: 1px solid rgba(245, 158, 11, 0.35);
    }}
    .chip-fov {{
        background: rgba(168, 85, 247, 0.12);
        color: {COLOR_STATUS_PURPLE};
        border: 1px solid rgba(168, 85, 247, 0.35);
    }}
    .chip-back {{
        background: rgba(148, 163, 184, 0.12);
        color: {COLOR_TEXT_MUTED};
        border: 1px solid rgba(148, 163, 184, 0.35);
    }}

    /* -----------------------------------------------------------------------
       Sidebar & Form Controls
       ----------------------------------------------------------------------- */
    [data-testid="stSidebar"] {{
        background-color: {COLOR_BG_SURFACE} !important;
        border-right: 1px solid {COLOR_BORDER_SUBTLE} !important;
    }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
        color: {COLOR_TEXT_PRIMARY} !important;
    }}

    /* Streamlit Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background-color: {COLOR_BG_BASE};
        padding: 4px;
        border-radius: 6px;
        border: 1px solid {COLOR_BORDER_SUBTLE};
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 4px !important;
        padding: 6px 16px !important;
        font-weight: 600 !important;
        font-size: 0.84rem !important;
        color: {COLOR_TEXT_MUTED} !important;
        border: none !important;
        background: transparent !important;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {COLOR_BG_SURFACE} !important;
        color: {COLOR_CYAN_PRIMARY} !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.25) !important;
    }}

    /* Streamlit DataFrames & Tables */
    [data-testid="stDataFrame"] {{
        border: 1px solid {COLOR_BORDER_SUBTLE};
        border-radius: 5px;
    }}

    /* Slider and Input Controls */
    .stSlider [data-baseweb="slider"] {{
        color: {COLOR_CYAN_PRIMARY} !important;
    }}
    </style>
    """
