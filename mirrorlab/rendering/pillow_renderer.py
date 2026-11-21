"""
Deterministic, High-DPI Pillow Diagram Renderer for MIRRORLAB.

Generates engineering-grade machine-vision inspection schematics from InspectionScene schemas:
- Strict 1:1 isometric coordinate transformation.
- Complete separation between blind evaluation images and diagnostic answer overlays.
- High-contrast, colorblind-safe technical aesthetic across 4 visual themes.
- Exact sub-pixel accuracy for red-X target defect and mirror endpoints.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mirrorlab.rendering.themes import (
    THEMES,
    RenderMode,
    RenderTheme,
    ThemeTokens,
    get_font,
)
from mirrorlab.rendering.transform import ViewportTransform
from mirrorlab.schema.common import VisibilityState
from mirrorlab.schema.scene import (
    CameraSpec,
    InspectionScene,
    MirrorSpec,
    ObstacleSpec,
    TargetSpec,
)


class PillowDiagramRenderer:
    """
    Renders 2D machine-vision inspection scenes as high-resolution PNG schematics.
    """

    def __init__(
        self,
        theme: RenderTheme = RenderTheme.DARK_NAVY,
        image_width: int = 1024,
        image_height: int = 1024,
        margin: float = 65.0,
    ):
        self.theme_name = theme
        self.tokens: ThemeTokens = THEMES[theme]
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.margin = float(margin)
        self.scale_factor = float(image_width) / 1024.0

    def render(
        self,
        scene: InspectionScene,
        mode: RenderMode = RenderMode.BLIND,
    ) -> Image.Image:
        """
        Render the scene into a PIL Image.

        Args:
            scene: Complete Pydantic InspectionScene.
            mode: RenderMode.BLIND (no answers/rays) or RenderMode.OVERLAY (complete proof).

        Returns:
            PIL.Image.Image in RGB mode.
        """
        tf = ViewportTransform.create(
            canvas_width=scene.canvas_width,
            canvas_height=scene.canvas_height,
            image_width=self.image_width,
            image_height=self.image_height,
            margin=self.margin,
        )

        # Base RGB image
        img = Image.new(
            "RGB", (self.image_width, self.image_height), self.tokens.background
        )
        draw = ImageDraw.Draw(img)

        # Alpha overlay layer for semi-transparent wedges and badges
        overlay = Image.new("RGBA", (self.image_width, self.image_height), (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)

        # 1. Background Grid & Framing
        self._draw_grid_and_framing(draw, tf)

        # 2. Camera Field-of-View (FOV) Cone Wedge (drawn to overlay layer)
        self._draw_camera_fov_wedge(draw_overlay, scene.camera, tf)

        # 3. Obstacles (O1, O2, O3)
        self._draw_obstacles(draw, scene.obstacles, tf)

        # 4. Mirror Segment (Reflective Face, Backing & Endpoints)
        self._draw_mirror(draw, scene.mirror, tf)

        # 5. Camera Body & Directional Lens
        self._draw_camera(draw, scene.camera, tf)

        # 6. Target Scratch Defect (Exact Red X)
        self._draw_target_defect(draw, scene.target, tf)

        # 7. Answer Overlay Layers (STRICTLY when mode == OVERLAY)
        if mode == RenderMode.OVERLAY:
            self._draw_answer_overlay(draw, draw_overlay, scene, tf)

        # Alpha-composite overlay onto base image
        img.paste(overlay, (0, 0), overlay)

        # 8. Legend / Metadata banner
        self._draw_header_metadata(draw, scene, mode)

        return img

    def render_repair(
        self,
        scene: InspectionScene,
        candidate_mirrors: dict[str, MirrorSpec],
        correct_candidate: str,
        mode: RenderMode = RenderMode.BLIND,
    ) -> Image.Image:
        """
        Render a Task C Mirror-Repair diagram featuring candidate mirror configurations A, B, and C.

        Args:
            scene: Base inspection scene.
            candidate_mirrors: Dictionary mapping 'A', 'B', 'C' to candidate MirrorSpec objects.
            correct_candidate: Unique option identifier ('A', 'B', or 'C') that restores visibility.
            mode: RenderMode.BLIND (no answers) or RenderMode.OVERLAY (full proof).

        Returns:
            PIL Image in RGB mode.
        """
        tf = ViewportTransform.create(
            canvas_width=scene.canvas_width,
            canvas_height=scene.canvas_height,
            image_width=self.image_width,
            image_height=self.image_height,
            margin=self.margin,
        )

        img = Image.new(
            "RGB", (self.image_width, self.image_height), self.tokens.background
        )
        draw = ImageDraw.Draw(img)
        overlay = Image.new("RGBA", (self.image_width, self.image_height), (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)

        # 1. Grid & Framing
        self._draw_grid_and_framing(draw, tf)

        # 2. Camera FOV Cone
        self._draw_camera_fov_wedge(draw_overlay, scene.camera, tf)

        # 3. Fixed Obstacles
        self._draw_obstacles(draw, scene.obstacles, tf)

        # 4. Render All 3 Candidate Mirrors (A, B, C)
        for cand_id in sorted(candidate_mirrors.keys()):
            mirror = candidate_mirrors[cand_id]
            is_winner = cand_id == correct_candidate

            if mode == RenderMode.OVERLAY and not is_winner:
                # Dim the losing mirrors slightly in overlay mode
                self._draw_mirror(
                    draw,
                    mirror,
                    tf,
                    label=f"MIRROR {cand_id} [FAILS]",
                    endpoint_labels=(f"{cand_id}1", f"{cand_id}2"),
                    reflective_color=self.tokens.text_secondary,
                )
            else:
                self._draw_mirror(
                    draw,
                    mirror,
                    tf,
                    label=f"MIRROR {cand_id}"
                    + (
                        " [CORRECT]"
                        if (mode == RenderMode.OVERLAY and is_winner)
                        else ""
                    ),
                    endpoint_labels=(f"{cand_id}1", f"{cand_id}2"),
                )

        # 5. Camera Body & Lens
        self._draw_camera(draw, scene.camera, tf)

        # 6. Target Scratch Red X
        self._draw_target_defect(draw, scene.target, tf)

        # 7. Answer Overlay (STRICTLY when mode == OVERLAY)
        if mode == RenderMode.OVERLAY:
            self._draw_answer_overlay(draw, draw_overlay, scene, tf)
            self._draw_repair_badge(draw, correct_candidate, scene)

        img.paste(overlay, (0, 0), overlay)

        # 8. Header metadata
        mode_badge = (
            "[BLIND REPAIR TASK]"
            if mode == RenderMode.BLIND
            else "[GROUND TRUTH OVERLAY]"
        )
        title_text = f"MIRRORLAB REPAIR SCHEMATIC  {mode_badge}"
        font_title = get_font(int(13 * self.scale_factor), bold=True)
        draw.text(
            (self.margin, self.margin - 35 * self.scale_factor),
            title_text,
            fill=self.tokens.text_primary,
            font=font_title,
        )
        meta_text = f"Scene ID: {scene.scene_id}  |  Candidates: A, B, C  |  Theme: {self.theme_name.value}"
        font_meta = get_font(int(10 * self.scale_factor))
        draw.text(
            (self.margin, self.margin - 18 * self.scale_factor),
            meta_text,
            fill=self.tokens.text_secondary,
            font=font_meta,
        )

        return img

    def _draw_repair_badge(
        self, draw: ImageDraw.ImageDraw, winner: str, scene: InspectionScene
    ) -> None:
        """Render prominent repair outcome pill in top-right."""
        font_badge = get_font(int(13 * self.scale_factor), bold=True)
        font_sub = get_font(int(10 * self.scale_factor))

        status_str = f"CORRECT: OPTION [{winner}] RESTORES VIEW"
        sub_str = f"Angle θ={scene.incident_angle_deg:.1f}° | Options: A, B, C"

        bw = 290 * self.scale_factor
        bh = 38 * self.scale_factor
        bx = self.image_width - self.margin - bw
        by = self.margin + 8 * self.scale_factor

        draw.rounded_rectangle(
            [bx, by, bx + bw, by + bh], radius=6, fill=self.tokens.badge_visible_bg
        )
        draw.text(
            (bx + 12 * self.scale_factor, by + 6 * self.scale_factor),
            status_str,
            fill=self.tokens.badge_visible_fg,
            font=font_badge,
        )
        draw.text(
            (bx + 12 * self.scale_factor, by + 22 * self.scale_factor),
            sub_str,
            fill=self.tokens.badge_visible_fg,
            font=font_sub,
        )

    def _draw_grid_and_framing(
        self, draw: ImageDraw.ImageDraw, tf: ViewportTransform
    ) -> None:
        """Draw technical coordinate grid and CAD corner brackets."""
        # Workspace border
        p_bl = tf.to_pixel(0.0, 0.0)
        p_tr = tf.to_pixel(tf.canvas_width, tf.canvas_height)
        border_rect = [p_bl[0], p_tr[1], p_tr[0], p_bl[1]]
        draw.rectangle(
            border_rect,
            outline=self.tokens.grid_border,
            width=int(1 * self.scale_factor),
        )

        # 100mm grid lines
        font_dim = get_font(int(10 * self.scale_factor))
        for x_val in range(100, int(tf.canvas_width), 100):
            p1 = tf.to_pixel(float(x_val), 0.0)
            p2 = tf.to_pixel(float(x_val), tf.canvas_height)
            draw.line([p1, p2], fill=self.tokens.grid_lines[:3], width=1)
            draw.text(
                (p1[0] - 10, p1[1] + 5),
                f"{x_val}",
                fill=self.tokens.text_secondary,
                font=font_dim,
            )

        for y_val in range(100, int(tf.canvas_height), 100):
            p1 = tf.to_pixel(0.0, float(y_val))
            p2 = tf.to_pixel(tf.canvas_width, float(y_val))
            draw.line([p1, p2], fill=self.tokens.grid_lines[:3], width=1)
            draw.text(
                (p1[0] - 25, p1[1] - 6),
                f"{y_val}",
                fill=self.tokens.text_secondary,
                font=font_dim,
            )

        # Corner technical CAD brackets
        bracket_len = 16 * self.scale_factor
        bw = max(2, int(2 * self.scale_factor))
        corners = [
            (p_bl[0], p_bl[1], 1, -1),
            (p_tr[0], p_bl[1], -1, -1),
            (p_bl[0], p_tr[1], 1, 1),
            (p_tr[0], p_tr[1], -1, 1),
        ]
        for cx, cy, dx, dy in corners:
            draw.line(
                [(cx, cy), (cx + dx * bracket_len, cy)],
                fill=self.tokens.corner_brackets,
                width=bw,
            )
            draw.line(
                [(cx, cy), (cx, cy + dy * bracket_len)],
                fill=self.tokens.corner_brackets,
                width=bw,
            )

    def _draw_camera_fov_wedge(
        self, draw_overlay: ImageDraw.ImageDraw, cam: CameraSpec, tf: ViewportTransform
    ) -> None:
        """Render semi-transparent camera FOV cone on the RGBA layer."""
        c_px = tf.to_pixel(cam.position.x, cam.position.y)
        heading_rad = math.radians(cam.heading_deg)
        half_fov_rad = math.radians(cam.fov_degrees / 2.0)

        # Project boundary rays to visual range (e.g. 700 mm)
        range_dist = tf.scale_dist(750.0)
        num_arc_pts = 32
        poly_pts = [c_px]

        start_angle = heading_rad - half_fov_rad
        end_angle = heading_rad + half_fov_rad

        for i in range(num_arc_pts + 1):
            theta = start_angle + (end_angle - start_angle) * (i / num_arc_pts)
            # In screen pixels: +X is right, +Y is down
            # Cartesian angle: dx = cos(theta), dy = sin(theta) -> pixel dy is -sin(theta)
            px = c_px[0] + range_dist * math.cos(theta)
            py = c_px[1] - range_dist * math.sin(theta)
            poly_pts.append((px, py))

        # Fill FOV cone wedge
        draw_overlay.polygon(poly_pts, fill=self.tokens.fov_wedge)

        # Draw boundary rays
        ray1 = poly_pts[1]
        ray2 = poly_pts[-1]
        bw = max(1, int(1.5 * self.scale_factor))
        draw_overlay.line([c_px, ray1], fill=self.tokens.fov_boundary, width=bw)
        draw_overlay.line([c_px, ray2], fill=self.tokens.fov_boundary, width=bw)

    def _draw_camera(
        self, draw: ImageDraw.ImageDraw, cam: CameraSpec, tf: ViewportTransform
    ) -> None:
        """Draw technical camera body, directional lens hood, and label."""
        c_px = tf.to_pixel(cam.position.x, cam.position.y)
        heading_rad = math.radians(cam.heading_deg)

        # Direction unit vector in screen coordinates (+Y down)
        u_screen = np.array([math.cos(heading_rad), -math.sin(heading_rad)])
        # Perpendicular vector
        v_screen = np.array([-u_screen[1], u_screen[0]])

        # Camera dimensions
        body_len = 26.0 * self.scale_factor
        body_width = 32.0 * self.scale_factor
        lens_len = 16.0 * self.scale_factor
        lens_w1 = 20.0 * self.scale_factor
        lens_w2 = 30.0 * self.scale_factor

        # Camera Body Box (centered slightly behind camera position)
        center_body = np.array(c_px) - (body_len / 2.0) * u_screen
        p1 = center_body - (body_len / 2.0) * u_screen - (body_width / 2.0) * v_screen
        p2 = center_body - (body_len / 2.0) * u_screen + (body_width / 2.0) * v_screen
        p3 = center_body + (body_len / 2.0) * u_screen + (body_width / 2.0) * v_screen
        p4 = center_body + (body_len / 2.0) * u_screen - (body_width / 2.0) * v_screen
        body_poly = [tuple(p1), tuple(p2), tuple(p3), tuple(p4)]

        # Lens Hood Trapezoid (protruding forward along heading)
        base_lens = np.array(c_px)
        tip_lens = np.array(c_px) + lens_len * u_screen
        l1 = base_lens - (lens_w1 / 2.0) * v_screen
        l2 = tip_lens - (lens_w2 / 2.0) * v_screen
        l3 = tip_lens + (lens_w2 / 2.0) * v_screen
        l4 = base_lens + (lens_w1 / 2.0) * v_screen
        lens_poly = [tuple(l1), tuple(l2), tuple(l3), tuple(l4)]

        # Render Camera
        stroke_w = max(2, int(2 * self.scale_factor))
        draw.polygon(
            body_poly,
            fill=self.tokens.camera_body,
            outline=self.tokens.camera_outline,
            width=stroke_w,
        )
        draw.polygon(
            lens_poly,
            fill=self.tokens.camera_lens,
            outline=self.tokens.camera_outline,
            width=stroke_w,
        )

        # Center dot representing exact nodal point C
        r_dot = 3.5 * self.scale_factor
        draw.ellipse(
            [c_px[0] - r_dot, c_px[1] - r_dot, c_px[0] + r_dot, c_px[1] + r_dot],
            fill=self.tokens.camera_center_dot,
            outline=self.tokens.camera_outline,
            width=1,
        )

        # Label: "CAMERA" placed behind the camera body
        label_pos = center_body - (body_len / 2.0 + 14 * self.scale_factor) * u_screen
        font_cam = get_font(int(12 * self.scale_factor), bold=True)
        draw.text(
            (
                label_pos[0] - 20 * self.scale_factor,
                label_pos[1] - 7 * self.scale_factor,
            ),
            "CAMERA",
            fill=self.tokens.text_primary,
            font=font_cam,
        )

    def _draw_mirror(
        self,
        draw: ImageDraw.ImageDraw,
        mirror: MirrorSpec,
        tf: ViewportTransform,
        label: str = "MIRROR",
        endpoint_labels: tuple[str, str] = ("A", "B"),
        reflective_color: tuple[int, int, int] | None = None,
        backing_color: tuple[int, int, int] | None = None,
    ) -> None:
        """Render mirror with reflective face, non-reflective backing, and endpoints."""
        p_a = np.array(tf.to_pixel(mirror.endpoint_a.x, mirror.endpoint_a.y))
        p_b = np.array(tf.to_pixel(mirror.endpoint_b.x, mirror.endpoint_b.y))

        m_dir = p_b - p_a
        m_len = np.linalg.norm(m_dir)
        if m_len < 1e-6:
            return
        m_unit = m_dir / m_len

        # Screen normal pointing away from reflective face (in screen space +Y is down)
        cart_n = np.array([mirror.reflective_normal.dx, mirror.reflective_normal.dy])
        screen_n = np.array([cart_n[0], -cart_n[1]])
        norm_len = np.linalg.norm(screen_n)
        if norm_len > 1e-6:
            screen_n = screen_n / norm_len

        # Back face normal points in opposite direction (-screen_n)
        back_dir = -screen_n

        fill_backing = backing_color or self.tokens.mirror_backing
        fill_reflective = reflective_color or self.tokens.mirror_reflective

        # 1. Non-reflective backing stripe and hatch marks
        backing_thickness = 7.0 * self.scale_factor
        back_a = p_a + backing_thickness * back_dir
        back_b = p_b + backing_thickness * back_dir
        draw.polygon(
            [tuple(p_a), tuple(p_b), tuple(back_b), tuple(back_a)],
            fill=fill_backing,
            outline=self.tokens.mirror_backing_hatch,
        )

        # Cross hatch lines along backing
        num_hatches = max(4, int(m_len / (12.0 * self.scale_factor)))
        hatch_w = max(1, int(1.2 * self.scale_factor))
        for i in range(num_hatches + 1):
            h_pt = p_a + (i / num_hatches) * m_dir
            h_end = h_pt + backing_thickness * back_dir - 5 * self.scale_factor * m_unit
            draw.line(
                [tuple(h_pt), tuple(h_end)],
                fill=self.tokens.mirror_backing_hatch,
                width=hatch_w,
            )

        # 2. Polished Reflective Front Face (Bold Cyan Line)
        mirror_w = max(3, int(4.5 * self.scale_factor))
        draw.line([tuple(p_a), tuple(p_b)], fill=fill_reflective, width=mirror_w)

        # 3. Endpoint Dots and Perpendicular Caps
        cap_r = 4.5 * self.scale_factor
        draw.ellipse(
            [p_a[0] - cap_r, p_a[1] - cap_r, p_a[0] + cap_r, p_a[1] + cap_r],
            fill=self.tokens.mirror_endpoint,
        )
        draw.ellipse(
            [p_b[0] - cap_r, p_b[1] - cap_r, p_b[0] + cap_r, p_b[1] + cap_r],
            fill=self.tokens.mirror_endpoint,
        )

        # 4. Technical Labels: Endpoints
        font_ab = get_font(int(12 * self.scale_factor), bold=True)
        lbl_offset_a = (
            p_a - 20 * self.scale_factor * m_unit + 12 * self.scale_factor * back_dir
        )
        lbl_offset_b = (
            p_b + 10 * self.scale_factor * m_unit + 12 * self.scale_factor * back_dir
        )
        draw.text(
            tuple(lbl_offset_a),
            endpoint_labels[0],
            fill=self.tokens.mirror_label,
            font=font_ab,
        )
        draw.text(
            tuple(lbl_offset_b),
            endpoint_labels[1],
            fill=self.tokens.mirror_label,
            font=font_ab,
        )

        # Label: Mirror name near midpoint of back side
        mid_pt = (p_a + p_b) / 2.0 + (
            backing_thickness + 14 * self.scale_factor
        ) * back_dir
        font_mir = get_font(int(11 * self.scale_factor), bold=True)
        draw.text(
            (mid_pt[0] - 26 * self.scale_factor, mid_pt[1] - 6 * self.scale_factor),
            label,
            fill=self.tokens.mirror_label,
            font=font_mir,
        )

    def _draw_obstacles(
        self,
        draw: ImageDraw.ImageDraw,
        obstacles: Sequence[ObstacleSpec],
        tf: ViewportTransform,
    ) -> None:
        """Render opaque obstacles with badges O1, O2, O3..."""
        font_badge = get_font(int(11 * self.scale_factor), bold=True)

        for idx, obs in enumerate(obstacles):
            label_id = f"O{idx + 1}"
            border_w = max(2, int(2 * self.scale_factor))

            if obs.obstacle_type == "circle":
                assert obs.radius is not None
                center_px = tf.to_pixel(obs.center.x, obs.center.y)
                r_px = tf.scale_dist(obs.radius)
                bbox = [
                    center_px[0] - r_px,
                    center_px[1] - r_px,
                    center_px[0] + r_px,
                    center_px[1] + r_px,
                ]
                draw.ellipse(
                    bbox,
                    fill=self.tokens.obstacle_fill,
                    outline=self.tokens.obstacle_outline,
                    width=border_w,
                )
                # Badge label
                self._draw_obstacle_badge(draw, center_px, label_id, font_badge)

            elif obs.obstacle_type == "rectangle":
                assert obs.width is not None and obs.height is not None
                center_px = np.array(tf.to_pixel(obs.center.x, obs.center.y))
                w_px = tf.scale_dist(obs.width)
                h_px = tf.scale_dist(obs.height)
                angle_deg = obs.angle_degrees or 0.0
                rad = math.radians(angle_deg)

                # Cartesian to screen: angle is clockwise in screen coordinates
                cos_a, sin_a = math.cos(rad), -math.sin(rad)
                hw, hh = w_px / 2.0, h_px / 2.0
                corners = [
                    np.array([-hw, -hh]),
                    np.array([hw, -hh]),
                    np.array([hw, hh]),
                    np.array([-hw, hh]),
                ]
                poly_pts = []
                for c in corners:
                    rx = c[0] * cos_a - c[1] * sin_a
                    ry = c[0] * sin_a + c[1] * cos_a
                    poly_pts.append(tuple(center_px + np.array([rx, ry])))

                draw.polygon(
                    poly_pts,
                    fill=self.tokens.obstacle_fill,
                    outline=self.tokens.obstacle_outline,
                )
                # Badge label
                self._draw_obstacle_badge(draw, tuple(center_px), label_id, font_badge)

    def _draw_obstacle_badge(
        self,
        draw: ImageDraw.ImageDraw,
        center_px: tuple[float, float],
        label: str,
        font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    ) -> None:
        """Draw small rounded pill badge inside obstacle center."""
        bw = 20 * self.scale_factor
        bh = 14 * self.scale_factor
        rect = [
            center_px[0] - bw / 2,
            center_px[1] - bh / 2,
            center_px[0] + bw / 2,
            center_px[1] + bh / 2,
        ]
        draw.rounded_rectangle(
            rect,
            radius=3,
            fill=self.tokens.obstacle_badge_bg,
            outline=self.tokens.obstacle_outline,
        )
        draw.text(
            (
                center_px[0] - 7 * self.scale_factor,
                center_px[1] - 7 * self.scale_factor,
            ),
            label,
            fill=self.tokens.obstacle_label,
            font=font,
        )

    def _draw_target_defect(
        self, draw: ImageDraw.ImageDraw, target: TargetSpec, tf: ViewportTransform
    ) -> None:
        """Render precise red-X inspection target scratch."""
        t_px = tf.to_pixel(target.center.x, target.center.y)
        arm_len = 13.0 * self.scale_factor
        x_w = max(3, int(3.5 * self.scale_factor))

        # Target X diagonals (Exact intersection at t_px)
        p1 = (t_px[0] - arm_len, t_px[1] - arm_len)
        p2 = (t_px[0] + arm_len, t_px[1] + arm_len)
        p3 = (t_px[0] - arm_len, t_px[1] + arm_len)
        p4 = (t_px[0] + arm_len, t_px[1] - arm_len)

        draw.line([p1, p2], fill=self.tokens.target_x, width=x_w)
        draw.line([p3, p4], fill=self.tokens.target_x, width=x_w)

        # Precise center coordinate ring indicator
        r_ring = 3.0 * self.scale_factor
        draw.ellipse(
            [t_px[0] - r_ring, t_px[1] - r_ring, t_px[0] + r_ring, t_px[1] + r_ring],
            outline=self.tokens.target_outline,
            width=1,
        )

        # Label: "SCRATCH (T)" offset safely
        font_tgt = get_font(int(11 * self.scale_factor), bold=True)
        draw.text(
            (t_px[0] + 16 * self.scale_factor, t_px[1] - 8 * self.scale_factor),
            "SCRATCH (T)",
            fill=self.tokens.target_label,
            font=font_tgt,
        )

    def _draw_answer_overlay(
        self,
        draw: ImageDraw.ImageDraw,
        draw_overlay: ImageDraw.ImageDraw,
        scene: InspectionScene,
        tf: ViewportTransform,
    ) -> None:
        """
        Draw scientific proof layers: ray paths, virtual target, normal, and diagnostic badge.
        STRICT REQUIREMENT: NEVER CALLED IN BLIND MODE.
        """
        cam_px = tf.to_pixel(scene.camera.position.x, scene.camera.position.y)
        tgt_px = tf.to_pixel(scene.target.center.x, scene.target.center.y)

        # 1. Virtual Target T' and Virtual Construction Ray (if available)
        if scene.virtual_target is not None:
            vt_px = tf.to_pixel(scene.virtual_target.x, scene.virtual_target.y)
            # Dashed line C -> T'
            self._draw_dashed_line(
                draw,
                cam_px,
                vt_px,
                self.tokens.ray_virtual[:3],
                width=int(2 * self.scale_factor),
                dash_len=8,
                gap_len=6,
            )
            # Virtual target circle marker
            vr = 7.0 * self.scale_factor
            draw.ellipse(
                [vt_px[0] - vr, vt_px[1] - vr, vt_px[0] + vr, vt_px[1] + vr],
                outline=self.tokens.ray_virtual[:3],
                width=int(2 * self.scale_factor),
            )
            font_vt = get_font(int(10 * self.scale_factor), bold=True)
            draw.text(
                (vt_px[0] + 10, vt_px[1] - 6),
                "T' (VIRTUAL)",
                fill=self.tokens.ray_virtual[:3],
                font=font_vt,
            )

        # 2. Specular Reflection Point R and Incident/Reflected Rays
        if scene.reflection_point is not None:
            r_px = tf.to_pixel(scene.reflection_point.x, scene.reflection_point.y)

            # Leg 1: Incident Ray C -> R (Solid Amber)
            ray_w = max(2, int(2.5 * self.scale_factor))
            draw.line([cam_px, r_px], fill=self.tokens.ray_incident, width=ray_w)
            self._draw_arrowhead(
                draw,
                cam_px,
                r_px,
                self.tokens.ray_incident,
                size=12 * self.scale_factor,
            )

            # Leg 2: Reflected Ray R -> T (Solid Emerald)
            draw.line([r_px, tgt_px], fill=self.tokens.ray_reflected, width=ray_w)
            self._draw_arrowhead(
                draw,
                r_px,
                tgt_px,
                self.tokens.ray_reflected,
                size=12 * self.scale_factor,
            )

            # Reflection Point Marker R
            rr = 5.0 * self.scale_factor
            draw.ellipse(
                [r_px[0] - rr, r_px[1] - rr, r_px[0] + rr, r_px[1] + rr],
                fill=self.tokens.reflection_point,
                outline=self.tokens.text_primary,
                width=1,
            )
            font_r = get_font(int(11 * self.scale_factor), bold=True)
            draw.text(
                (r_px[0] - 5, r_px[1] - 22 * self.scale_factor),
                "R",
                fill=self.tokens.reflection_point,
                font=font_r,
            )

        # 3. Diagnostic Result Badge in Top-Right Corner
        self._draw_result_badge(draw, scene)

    def _draw_result_badge(
        self, draw: ImageDraw.ImageDraw, scene: InspectionScene
    ) -> None:
        """Render prominent technical outcome pill in top-right."""
        font_badge = get_font(int(13 * self.scale_factor), bold=True)
        font_sub = get_font(int(10 * self.scale_factor))

        is_vis = scene.ground_truth_state == VisibilityState.VISIBLE
        bg_col = (
            self.tokens.badge_visible_bg if is_vis else self.tokens.badge_blocked_bg
        )
        fg_col = (
            self.tokens.badge_visible_fg if is_vis else self.tokens.badge_blocked_fg
        )

        status_str = (
            f"VISIBLE [THETA={scene.incident_angle_deg:.1f} DEG]"
            if is_vis
            else f"NOT VISIBLE: {scene.ground_truth_reason.value}"
        )

        bw = 270 * self.scale_factor
        bh = 38 * self.scale_factor
        bx = self.image_width - self.margin - bw
        by = self.margin + 8 * self.scale_factor

        draw.rounded_rectangle([bx, by, bx + bw, by + bh], radius=6, fill=bg_col)
        draw.text(
            (bx + 12 * self.scale_factor, by + 6 * self.scale_factor),
            status_str,
            fill=fg_col,
            font=font_badge,
        )
        draw.text(
            (bx + 12 * self.scale_factor, by + 22 * self.scale_factor),
            f"Category: {scene.category.value} | Diff: {scene.difficulty.value}",
            fill=fg_col,
            font=font_sub,
        )

    def _draw_header_metadata(
        self, draw: ImageDraw.ImageDraw, scene: InspectionScene, mode: RenderMode
    ) -> None:
        """Render discrete technical banner at top-left."""
        font_title = get_font(int(13 * self.scale_factor), bold=True)
        font_meta = get_font(int(10 * self.scale_factor))

        mode_badge = (
            "[BLIND TASK]" if mode == RenderMode.BLIND else "[GROUND TRUTH OVERLAY]"
        )
        title_text = f"MIRRORLAB INSPECTION SCHEMATIC  {mode_badge}"
        draw.text(
            (self.margin, self.margin - 35 * self.scale_factor),
            title_text,
            fill=self.tokens.text_primary,
            font=font_title,
        )

        meta_text = f"Scene ID: {scene.scene_id}  |  Seed: {scene.seed}  |  Theme: {self.theme_name.value}"
        draw.text(
            (self.margin, self.margin - 18 * self.scale_factor),
            meta_text,
            fill=self.tokens.text_secondary,
            font=font_meta,
        )

    def _draw_arrowhead(
        self,
        draw: ImageDraw.ImageDraw,
        p_from: tuple[float, float],
        p_to: tuple[float, float],
        color: tuple[int, int, int],
        size: float = 12.0,
    ) -> None:
        """Draw directional arrowhead at segment midpoint."""
        mid = ((p_from[0] + p_to[0]) / 2.0, (p_from[1] + p_to[1]) / 2.0)
        dx = p_to[0] - p_from[0]
        dy = p_to[1] - p_from[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-4:
            return

        ux, uy = dx / dist, dy / dist
        vx, vy = -uy, ux

        tip = (mid[0] + ux * size * 0.5, mid[1] + uy * size * 0.5)
        b1 = (
            mid[0] - ux * size * 0.5 + vx * size * 0.35,
            mid[1] - uy * size * 0.5 + vy * size * 0.35,
        )
        b2 = (
            mid[0] - ux * size * 0.5 - vx * size * 0.35,
            mid[1] - uy * size * 0.5 - vy * size * 0.35,
        )

        draw.polygon([tip, b1, b2], fill=color)

    def _draw_dashed_line(
        self,
        draw: ImageDraw.ImageDraw,
        p1: tuple[float, float],
        p2: tuple[float, float],
        color: tuple[int, int, int],
        width: int = 1,
        dash_len: float = 8.0,
        gap_len: float = 6.0,
    ) -> None:
        """Render dashed line segment."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        total_len = math.hypot(dx, dy)
        if total_len < 1e-4:
            return

        ux, uy = dx / total_len, dy / total_len
        curr = 0.0
        while curr < total_len:
            seg_end = min(curr + dash_len, total_len)
            s_pt = (p1[0] + ux * curr, p1[1] + uy * curr)
            e_pt = (p1[0] + ux * seg_end, p1[1] + uy * seg_end)
            draw.line([s_pt, e_pt], fill=color, width=width)
            curr += dash_len + gap_len
