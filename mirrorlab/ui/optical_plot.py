"""
Plotly Interactive 2D Optical Schematic Engine for MIRRORLAB.

Renders machine-vision inspection scenes with real-time specular ray paths,
virtual camera constructions, camera FOV cones, and obstacle occlusions.
Enforces strict 1:1 Euclidean aspect ratio so geometric angles are never distorted.
"""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go

from mirrorlab.geometry.oracle import VisibilityOracle, VisibilityOracleResult
from mirrorlab.schema.scene import (
    CameraSpec,
    InspectionScene,
    MirrorSpec,
    ObstacleSpec,
    TargetSpec,
)
from mirrorlab.ui.design_tokens import (
    COLOR_BG_BASE,
    COLOR_TEXT_SECONDARY,
)


def create_optical_figure(
    scene: InspectionScene,
    oracle_result: VisibilityOracleResult | None = None,
    show_debug_rays: bool = True,
    show_fov_cone: bool = True,
    show_direct_sightline: bool = False,
    theme: str = "dark",
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """
    Generate an interactive Plotly figure representing the 2D inspection scene.

    Args:
        scene: The InspectionScene schema containing camera, mirror, target, obstacles.
        oracle_result: Pre-computed or optional oracle result. If None, evaluated on the fly.
        show_debug_rays: Whether to draw specular rays, virtual camera, and normal vector.
        show_fov_cone: Whether to draw the camera FOV wedge.
        show_direct_sightline: Whether to draw the direct line from Camera to Target.
        theme: "dark" (default, industrial console) or "light" (clean report paper).
        width: Optional fixed pixel width.
        height: Optional fixed pixel height.

    Returns:
        plotly.graph_objects.Figure with 1:1 aspect ratio.
    """
    if oracle_result is None:
        oracle_result = VisibilityOracle.evaluate(scene)

    is_dark = theme.lower() == "dark"
    bg_color = COLOR_BG_BASE if is_dark else "#f8f9fa"
    paper_color = COLOR_BG_BASE if is_dark else "#ffffff"
    grid_color = "#131C2D" if is_dark else "rgba(0, 0, 0, 0.06)"
    text_color = COLOR_TEXT_SECONDARY if is_dark else "#1e293b"
    cavity_wall_color = "#334155" if is_dark else "#cbd5e1"

    fig = go.Figure()

    # 1. Cavity Boundary Outline (0, 0) to (canvas_width, canvas_height)
    cw, ch = scene.canvas_width, scene.canvas_height
    fig.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=cw,
        y1=ch,
        line={"color": cavity_wall_color, "width": 2, "dash": "dash"},
        fillcolor="rgba(0,0,0,0)",
        layer="below",
    )

    # 2. Draw Obstacles
    for obs in scene.obstacles:
        _draw_obstacle(fig, obs, is_dark)

    # 3. Draw Mirror
    _draw_mirror(fig, scene.mirror, is_dark)

    # 4. Draw Camera and FOV Cone
    _draw_camera(fig, scene.camera, show_fov_cone, is_dark)

    # 5. Draw Target Defect Scratch (Red X)
    _draw_target(fig, scene.target, is_dark)

    # 6. Optional: Direct Sightline (Camera -> Target)
    if show_direct_sightline:
        fig.add_trace(
            go.Scatter(
                x=[scene.camera.position.x, scene.target.center.x],
                y=[scene.camera.position.y, scene.target.center.y],
                mode="lines",
                line={"color": "#94a3b8", "width": 1.5, "dash": "dot"},
                name="Direct Sightline (C → S)",
                hoverinfo="name",
            )
        )

    # 7. Debug Overlays (Specular Rays, Virtual Camera, Normal)
    if show_debug_rays and oracle_result is not None:
        _draw_debug_overlays(fig, scene, oracle_result, is_dark)

    # 8. Layout Configurations (Strict 1:1 Aspect Ratio)
    margin_val = 40
    fig.update_layout(
        plot_bgcolor=bg_color,
        paper_bgcolor=paper_color,
        font={"family": "Inter, Roboto, sans-serif", "color": text_color, "size": 12},
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"size": 11, "color": text_color},
        },
        margin={"l": margin_val, "r": margin_val, "t": margin_val + 20, "b": margin_val},
        width=width,
        height=height,
        hovermode="closest",
        dragmode="pan",
    )

    # Enforce 1:1 Euclidean Aspect Ratio
    fig.update_xaxes(
        range=[-50, cw + 50],
        showgrid=True,
        gridcolor=grid_color,
        zeroline=False,
        title="X Position (mm)",
        tickfont={"color": text_color},
        title_font={"color": text_color},
    )
    fig.update_yaxes(
        range=[-50, ch + 50],
        showgrid=True,
        gridcolor=grid_color,
        zeroline=False,
        scaleanchor="x",
        scaleratio=1.0,
        title="Y Position (mm)",
        tickfont={"color": text_color},
        title_font={"color": text_color},
    )

    return fig


def _draw_obstacle(fig: go.Figure, obs: ObstacleSpec, is_dark: bool) -> None:
    """Render a polygonal or circular opaque obstacle."""
    obs_fill = "rgba(71, 85, 105, 0.45)" if is_dark else "rgba(148, 163, 184, 0.35)"
    obs_line = "#64748b" if is_dark else "#475569"

    if obs.obstacle_type == "circle" and obs.radius is not None:
        # Polygonal approximation for circular obstacle
        theta = np.linspace(0, 2 * np.pi, 48)
        xs = obs.center.x + obs.radius * np.cos(theta)
        ys = obs.center.y + obs.radius * np.sin(theta)

        fig.add_trace(
            go.Scatter(
                x=xs.tolist(),
                y=ys.tolist(),
                mode="lines",
                fill="toself",
                fillcolor=obs_fill,
                line={"color": obs_line, "width": 2},
                name=f"Obstacle {obs.obstacle_id}",
                hoverinfo="text",
                text=f"Obstacle: {obs.obstacle_id}<br>Type: Circle (r={obs.radius:.1f}mm)<br>Center: ({obs.center.x:.1f}, {obs.center.y:.1f})",
            )
        )
    elif obs.obstacle_type == "rectangle" and obs.width is not None and obs.height is not None:
        # 4 rotated rectangle corners
        hw = obs.width / 2.0
        hh = obs.height / 2.0
        corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh], [-hw, -hh]])
        rad = np.radians(obs.angle_degrees)
        rot_mat = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
        rot_corners = corners @ rot_mat.T
        xs = rot_corners[:, 0] + obs.center.x
        ys = rot_corners[:, 1] + obs.center.y

        fig.add_trace(
            go.Scatter(
                x=xs.tolist(),
                y=ys.tolist(),
                mode="lines",
                fill="toself",
                fillcolor=obs_fill,
                line={"color": obs_line, "width": 2},
                name=f"Obstacle {obs.obstacle_id}",
                hoverinfo="text",
                text=f"Obstacle: {obs.obstacle_id}<br>Type: Rect ({obs.width:.1f}x{obs.height:.1f}mm)<br>Center: ({obs.center.x:.1f}, {obs.center.y:.1f})<br>Angle: {obs.angle_degrees:.1f}°",
            )
        )

    # Obstacle Label
    fig.add_annotation(
        x=obs.center.x,
        y=obs.center.y,
        text=obs.obstacle_id,
        showarrow=False,
        font={"size": 11, "color": "#cbd5e1" if is_dark else "#1e293b", "family": "monospace"},
    )


def _draw_mirror(fig: go.Figure, mirror: MirrorSpec, is_dark: bool) -> None:
    """Render the finite mirror segment, cyan front face, and hatched backing."""
    ma = mirror.endpoint_a
    mb = mirror.endpoint_b
    norm = mirror.reflective_normal

    # Reflective front line (Vibrant cyan)
    front_color = "#00e5ff"
    fig.add_trace(
        go.Scatter(
            x=[ma.x, mb.x],
            y=[ma.y, mb.y],
            mode="lines+markers",
            line={"color": front_color, "width": 5},
            marker={"size": 7, "color": front_color, "symbol": "circle"},
            name="Planar Mirror",
            hoverinfo="text",
            text=f"Mirror<br>A: ({ma.x:.1f}, {ma.y:.1f})<br>B: ({mb.x:.1f}, {mb.y:.1f})<br>Length: {math.hypot(mb.x - ma.x, mb.y - ma.y):.1f}mm",
        )
    )

    # Mirror non-reflective backing (offset ticks on back side)
    back_norm_x = -norm.dx
    back_norm_y = -norm.dy
    num_ticks = 8
    tick_len = 10.0
    for i in range(num_ticks + 1):
        frac = i / num_ticks
        px = ma.x + frac * (mb.x - ma.x)
        py = ma.y + frac * (mb.y - ma.y)
        tx = px + back_norm_x * tick_len
        ty = py + back_norm_y * tick_len
        fig.add_shape(
            type="line",
            x0=px,
            y0=py,
            x1=tx,
            y1=ty,
            line={"color": "#475569" if is_dark else "#94a3b8", "width": 1.5},
            layer="below",
        )

    # Normal Vector at Mirror Midpoint
    mid_x = (ma.x + mb.x) / 2.0
    mid_y = (ma.y + mb.y) / 2.0
    arrow_len = 35.0
    nx = mid_x + norm.dx * arrow_len
    ny = mid_y + norm.dy * arrow_len
    fig.add_annotation(
        x=nx,
        y=ny,
        ax=mid_x,
        ay=mid_y,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        text="",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.2,
        arrowwidth=2,
        arrowcolor="#14b8a6",
    )


def _draw_camera(fig: go.Figure, cam: CameraSpec, show_fov: bool, is_dark: bool) -> None:
    """Render the camera icon, heading axis, and semi-transparent FOV cone."""
    cx, cy = cam.position.x, cam.position.y
    heading_rad = np.radians(cam.heading_deg)
    fov_half_rad = np.radians(cam.fov_degrees / 2.0)
    cone_len = min(cam.max_range, 350.0)

    # Camera Marker & Heading Axis
    head_len = 35.0
    hx = cx + np.cos(heading_rad) * head_len
    hy = cy + np.sin(heading_rad) * head_len

    fig.add_trace(
        go.Scatter(
            x=[cx],
            y=[cy],
            mode="markers+text",
            marker={"size": 14, "color": "#2b7fff", "symbol": "circle", "line": {"color": "#ffffff", "width": 1.5}},
            name="Camera (C)",
            text=["Camera C"],
            textposition="bottom center",
            hoverinfo="text",
            textfont={"color": "#2b7fff", "size": 11, "family": "sans-serif"},
            hovertext=f"Camera Position: ({cx:.1f}, {cy:.1f})<br>Heading: {cam.heading_deg:.1f}°<br>FOV: {cam.fov_degrees:.1f}°",
        )
    )

    # Optical Axis Arrow
    fig.add_annotation(
        x=hx,
        y=hy,
        ax=cx,
        ay=cy,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        text="",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.2,
        arrowwidth=2.5,
        arrowcolor="#2b7fff",
    )

    # FOV Cone Wedge
    if show_fov:
        arc_angles = np.linspace(heading_rad - fov_half_rad, heading_rad + fov_half_rad, 25)
        arc_x = [cx] + (cx + cone_len * np.cos(arc_angles)).tolist() + [cx]
        arc_y = [cy] + (cy + cone_len * np.sin(arc_angles)).tolist() + [cy]

        fig.add_trace(
            go.Scatter(
                x=arc_x,
                y=arc_y,
                mode="lines",
                fill="toself",
                fillcolor="rgba(43, 127, 255, 0.12)",
                line={"color": "rgba(96, 165, 250, 0.6)", "width": 1.5, "dash": "dash"},
                name="Camera FOV Cone",
                hoverinfo="skip",
            )
        )


def _draw_target(fig: go.Figure, tgt: TargetSpec, is_dark: bool) -> None:
    """Render the high-visibility red-X defect scratch marker."""
    tx, ty = tgt.center.x, tgt.center.y
    r = tgt.radius_mm

    # Red X crossarms
    fig.add_trace(
        go.Scatter(
            x=[tx - r, tx + r, None, tx - r, tx + r],
            y=[ty - r, ty + r, None, ty + r, ty - r],
            mode="lines",
            line={"color": "#ef4444", "width": 4},
            name="Target Scratch (S)",
            hoverinfo="text",
            hovertext=f"Target Defect Scratch<br>Position: ({tx:.1f}, {ty:.1f})<br>Radius: {r:.1f}mm",
        )
    )

    # Center target label
    fig.add_annotation(
        x=tx,
        y=ty - r - 12,
        text="Defect (S)",
        showarrow=False,
        font={"size": 11, "color": "#ef4444", "family": "sans-serif"},
    )


def _draw_debug_overlays(
    fig: go.Figure,
    scene: InspectionScene,
    oracle: VisibilityOracleResult,
    is_dark: bool,
) -> None:
    """Render the complete optical proof overlay: virtual camera, incident & reflected rays."""
    cam_p = scene.camera.position
    tgt_p = scene.target.center

    # 1. Virtual Target or Virtual Camera
    if oracle.virtual_target is not None:
        vx, vy = float(oracle.virtual_target[0]), float(oracle.virtual_target[1])
        fig.add_trace(
            go.Scatter(
                x=[vx],
                y=[vy],
                mode="markers+text",
                marker={"size": 11, "color": "#d946ef", "symbol": "circle-open", "line": {"width": 2, "color": "#d946ef"}},
                text=["Virtual Target S'"],
                textposition="top center",
                name="Virtual Target (S')",
                hoverinfo="text",
                hovertext=f"Virtual Target S'<br>Coordinates: ({vx:.1f}, {vy:.1f})",
            )
        )

        # Dashed construction line from Camera C to Virtual Target S'
        fig.add_trace(
            go.Scatter(
                x=[cam_p.x, vx],
                y=[cam_p.y, vy],
                mode="lines",
                line={"color": "#d946ef", "width": 1.5, "dash": "dot"},
                name="Virtual Construction Ray",
                hoverinfo="name",
            )
        )

    # 2. Reflection Point P on Mirror
    if oracle.reflection_point is not None:
        rx, ry = float(oracle.reflection_point[0]), float(oracle.reflection_point[1])
        fig.add_trace(
            go.Scatter(
                x=[rx],
                y=[ry],
                mode="markers",
                marker={"size": 10, "color": "#00e5ff", "symbol": "diamond", "line": {"color": "#ffffff", "width": 1.5}},
                name="Reflection Point (P)",
                hoverinfo="text",
                hovertext=f"Reflection Point P: ({rx:.1f}, {ry:.1f})<br>θ_i = θ_r = {oracle.incident_angle_deg or 0.0:.1f}°",
            )
        )

        # Incident Ray: Camera C -> Reflection Point P (Amber)
        fig.add_trace(
            go.Scatter(
                x=[cam_p.x, rx],
                y=[cam_p.y, ry],
                mode="lines",
                line={"color": "#f59e0b", "width": 3},
                name="Incident Ray (Leg 1)",
                hoverinfo="text",
                hovertext=f"Leg 1: Camera C → Mirror P<br>Distance: {math.hypot(rx - cam_p.x, ry - cam_p.y):.1f}mm",
            )
        )

        # Midpoint arrow on Incident Ray
        mid_inc_x = (cam_p.x + rx) / 2.0
        mid_inc_y = (cam_p.y + ry) / 2.0
        dir_inc_x = rx - cam_p.x
        dir_inc_y = ry - cam_p.y
        len_inc = math.hypot(dir_inc_x, dir_inc_y)
        if len_inc > 1e-3:
            fig.add_annotation(
                x=mid_inc_x + (dir_inc_x / len_inc) * 10,
                y=mid_inc_y + (dir_inc_y / len_inc) * 10,
                ax=mid_inc_x - (dir_inc_x / len_inc) * 10,
                ay=mid_inc_y - (dir_inc_y / len_inc) * 10,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                text="",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.2,
                arrowwidth=2,
                arrowcolor="#f59e0b",
            )

        # Reflected Ray: Reflection Point P -> Target S
        refl_color = "#10b981" if oracle.is_visible else "#ef4444"
        refl_dash = "solid" if oracle.is_visible else "dash"
        refl_name = "Reflected Ray (Leg 2)" if oracle.is_visible else "Reflected Ray (Blocked)"

        fig.add_trace(
            go.Scatter(
                x=[rx, tgt_p.x],
                y=[ry, tgt_p.y],
                mode="lines",
                line={"color": refl_color, "width": 3, "dash": refl_dash},
                name=refl_name,
                hoverinfo="text",
                hovertext=f"Leg 2: Mirror P → Target S<br>Status: {'Unobstructed' if oracle.is_visible else 'Blocked'}<br>Distance: {math.hypot(tgt_p.x - rx, tgt_p.y - ry):.1f}mm",
            )
        )

        # Midpoint arrow on Reflected Ray
        mid_ref_x = (rx + tgt_p.x) / 2.0
        mid_ref_y = (ry + tgt_p.y) / 2.0
        dir_ref_x = tgt_p.x - rx
        dir_ref_y = tgt_p.y - ry
        len_ref = math.hypot(dir_ref_x, dir_ref_y)
        if len_ref > 1e-3:
            fig.add_annotation(
                x=mid_ref_x + (dir_ref_x / len_ref) * 10,
                y=mid_ref_y + (dir_ref_y / len_ref) * 10,
                ax=mid_ref_x - (dir_ref_x / len_ref) * 10,
                ay=mid_ref_y - (dir_ref_y / len_ref) * 10,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                text="",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.2,
                arrowwidth=2,
                arrowcolor=refl_color,
            )

    # 3. Obstacle Collision Hit Marker
    if oracle.occlusion_result is not None and oracle.occlusion_result.is_path_blocked:
        hit = oracle.occlusion_result.first_blocker
        if hit is not None:
            hx, hy = float(hit.hit_point[0]), float(hit.hit_point[1])
            fig.add_trace(
                go.Scatter(
                    x=[hx],
                    y=[hy],
                    mode="markers+text",
                    marker={"size": 14, "color": "#ef4444", "symbol": "x-thin", "line": {"width": 3, "color": "#ef4444"}},
                    text=[f"Collision: {hit.obstacle_id}"],
                    textposition="bottom right",
                    name="Ray Collision Hit",
                    hoverinfo="text",
                    hovertext=f"Ray Collision Hit!<br>Obstacle: {hit.obstacle_id}<br>Leg: {hit.leg.value}<br>Hit Point: ({hx:.1f}, {hy:.1f})",
                )
            )
