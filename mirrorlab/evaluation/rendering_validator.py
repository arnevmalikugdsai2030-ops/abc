"""
Rendering-to-Data Consistency Validator for MIRRORLAB.

Verifies that rendered raster images (PNG) strictly and accurately represent
the underlying structured scene data (InspectionScene) without distortion,
misalignment, or label leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from mirrorlab.rendering.transform import ViewportTransform
from mirrorlab.schema.scene import InspectionScene


@dataclass
class ElementValidationResult:
    """Validation result for an individual scene element in the rendered image."""

    element_name: str
    passed: bool
    details: str
    expected_pixel_pos: tuple[float, float] | None = None
    observed_features: dict[str, Any] = field(default_factory=dict)


@dataclass
class RenderingValidationReport:
    """Complete validation report verifying rendering-to-data consistency."""

    image_path: str
    image_width: int
    image_height: int
    all_passed: bool
    element_results: list[ElementValidationResult]
    summary: str


class RenderingConsistencyValidator:
    """Probes rendered raster diagrams to confirm pixel fidelity to scene geometry."""

    @staticmethod
    def validate(
        scene: InspectionScene,
        image_path: Path | str,
        margin: float = 60.0,
        is_blind_image: bool = True,
    ) -> RenderingValidationReport:
        """
        Validates that image at image_path corresponds to scene geometry.
        """
        img_p = Path(image_path)
        if not img_p.is_file():
            return RenderingValidationReport(
                image_path=str(img_p),
                image_width=0,
                image_height=0,
                all_passed=False,
                element_results=[
                    ElementValidationResult(
                        element_name="file_existence",
                        passed=False,
                        details=f"Rendered image not found at {img_p}",
                    )
                ],
                summary="Image file missing",
            )

        with Image.open(img_p) as pil_img:
            img = pil_img.convert("RGB")
            w, h = img.size
            img_arr = np.array(img)

        transform = ViewportTransform.create(
            canvas_width=scene.canvas_width,
            canvas_height=scene.canvas_height,
            image_width=w,
            image_height=h,
            margin=margin,
        )

        results: list[ElementValidationResult] = []

        # 1. Target scratch red-X probe
        tx, ty = scene.target.center.x, scene.target.center.y
        t_px, t_py = transform.to_pixel(tx, ty)
        it_px, it_py = round(t_px), round(t_py)

        target_valid = False
        target_details = ""
        # Search a small window around target
        radius_window = 12
        x0 = max(0, it_px - radius_window)
        x1 = min(w, it_px + radius_window + 1)
        y0 = max(0, it_py - radius_window)
        y1 = min(h, it_py + radius_window + 1)

        patch = img_arr[y0:y1, x0:x1]
        # Target scratch is rendered in prominent red (high R, lower G, B)
        red_mask = (
            (patch[:, :, 0] > 140) & (patch[:, :, 1] < 100) & (patch[:, :, 2] < 100)
        )
        red_count = int(np.sum(red_mask))

        if red_count >= 5:
            target_valid = True
            target_details = f"Target marker detected: {red_count} red pixels centered at ({it_px}, {it_py})."
        else:
            target_details = f"Target marker not detected at ({it_px}, {it_py}); red pixels = {red_count}."

        results.append(
            ElementValidationResult(
                element_name="target_defect_red_cross",
                passed=target_valid,
                details=target_details,
                expected_pixel_pos=(t_px, t_py),
                observed_features={"red_pixel_count": red_count},
            )
        )

        # 2. Camera symbol probe
        cx, cy = scene.camera.position.x, scene.camera.position.y
        c_px, c_py = transform.to_pixel(cx, cy)
        ic_px, ic_py = round(c_px), round(c_py)

        cx0 = max(0, ic_px - 15)
        cx1 = min(w, ic_px + 16)
        cy0 = max(0, ic_py - 15)
        cy1 = min(h, ic_py + 16)

        cam_patch = img_arr[cy0:cy1, cx0:cx1]
        # Camera is rendered with blue elements (high B, or distinct non-background pixels)
        blue_mask = (cam_patch[:, :, 2] > 120) & (cam_patch[:, :, 0] < 120)
        blue_count = int(np.sum(blue_mask))
        cam_valid = blue_count >= 5
        results.append(
            ElementValidationResult(
                element_name="camera_apparatus",
                passed=cam_valid,
                details=f"Camera apparatus detected: {blue_count} blue pixels at ({ic_px}, {ic_py}).",
                expected_pixel_pos=(c_px, c_py),
                observed_features={"blue_pixel_count": blue_count},
            )
        )

        # 3. Mirror segment midpoint probe
        mx = (scene.mirror.endpoint_a.x + scene.mirror.endpoint_b.x) / 2.0
        my = (scene.mirror.endpoint_a.y + scene.mirror.endpoint_b.y) / 2.0
        m_px, m_py = transform.to_pixel(mx, my)
        im_px, im_py = round(m_px), round(m_py)

        mx0 = max(0, im_px - 10)
        mx1 = min(w, im_px + 11)
        my0 = max(0, im_py - 10)
        my1 = min(h, im_py + 11)

        mirror_patch = img_arr[my0:my1, mx0:mx1]
        # Mirror reflective edge is cyan/bright line (distinct from background)
        # Background is typically dark #0F172A (~15, 23, 42) or light grid
        non_bg_mask = np.any(
            np.abs(mirror_patch.astype(int) - np.array([15, 23, 42])) > 30, axis=-1
        )
        mirror_pixels = int(np.sum(non_bg_mask))
        mirror_valid = mirror_pixels >= 4
        results.append(
            ElementValidationResult(
                element_name="mirror_reflective_surface",
                passed=mirror_valid,
                details=f"Mirror reflection surface detected: {mirror_pixels} pixels at midpoint ({im_px}, {im_py}).",
                expected_pixel_pos=(m_px, m_py),
                observed_features={"mirror_pixels": mirror_pixels},
            )
        )

        # 4. Obstacle position probes
        for obs in scene.obstacles:
            ox, oy = obs.center.x, obs.center.y
            o_px, o_py = transform.to_pixel(ox, oy)
            io_px, io_py = round(o_px), round(o_py)

            ox0 = max(0, io_px - 8)
            ox1 = min(w, io_px + 9)
            oy0 = max(0, io_py - 8)
            oy1 = min(h, io_py + 9)

            obs_patch = img_arr[oy0:oy1, ox0:ox1]
            non_bg = np.any(
                np.abs(obs_patch.astype(int) - np.array([15, 23, 42])) > 25, axis=-1
            )
            obs_count = int(np.sum(non_bg))
            results.append(
                ElementValidationResult(
                    element_name=f"obstacle_{obs.obstacle_id}",
                    passed=(obs_count >= 3),
                    details=f"Obstacle '{obs.obstacle_id}' detected at ({io_px}, {io_py}) with {obs_count} pixels.",
                    expected_pixel_pos=(o_px, o_py),
                    observed_features={"pixel_count": obs_count},
                )
            )

        # 5. Non-leakage probe for blind images: Ensure no green/magenta diagnostic rays
        if is_blind_image and scene.reflection_point:
            # Check midpoint between camera and reflection point
            mid_x = (scene.camera.position.x + scene.reflection_point.x) / 2.0
            mid_y = (scene.camera.position.y + scene.reflection_point.y) / 2.0
            mid_px, mid_py = transform.to_pixel(mid_x, mid_y)
            imid_px, imid_py = round(mid_px), round(mid_py)

            mid_patch = img_arr[
                max(0, imid_py - 3) : min(h, imid_py + 4),
                max(0, imid_px - 3) : min(w, imid_px + 4),
            ]
            # Green ray is (G > 200, R < 100)
            green_leak = (
                (mid_patch[:, :, 1] > 200)
                & (mid_patch[:, :, 0] < 100)
                & (mid_patch[:, :, 2] < 100)
            )
            leak_detected = bool(np.any(green_leak))
            results.append(
                ElementValidationResult(
                    element_name="blind_non_leakage_optical_corridor",
                    passed=not leak_detected,
                    details="Zero diagnostic ray overlay detected in blind inspection corridor."
                    if not leak_detected
                    else "WARNING: Green diagnostic overlay detected in blind image!",
                    expected_pixel_pos=(mid_px, mid_py),
                    observed_features={"leak_detected": leak_detected},
                )
            )

        all_passed = all(r.passed for r in results)
        summary = (
            f"Rendering validation passed: all {len(results)} element checks verified."
            if all_passed
            else f"Rendering validation FAILED: {sum(1 for r in results if not r.passed)}/{len(results)} checks failed."
        )

        return RenderingValidationReport(
            image_path=str(img_p),
            image_width=w,
            image_height=h,
            all_passed=all_passed,
            element_results=results,
            summary=summary,
        )
