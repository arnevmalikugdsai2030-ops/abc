"""
Coordinate System Transformation for MIRRORLAB Diagrams.

Handles strict 1:1 isometric projection from continuous Cartesian coordinates
(where +Y points UP) to digital image raster coordinates (where +Y points DOWN).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ViewportTransform:
    """
    Affine transformation between Cartesian workspace coordinates and screen raster pixels.
    """

    canvas_width: float
    canvas_height: float
    image_width: int
    image_height: int
    margin: float
    scale: float
    offset_x: float
    offset_y: float

    @classmethod
    def create(
        cls,
        canvas_width: float = 1000.0,
        canvas_height: float = 1000.0,
        image_width: int = 1024,
        image_height: int = 1024,
        margin: float = 60.0,
    ) -> ViewportTransform:
        """
        Create a 1:1 aspect-ratio-preserving viewport transform.
        """
        usable_w = float(image_width) - 2.0 * margin
        usable_h = float(image_height) - 2.0 * margin

        # Strict isometric scale (same scale in X and Y to avoid angular distortion)
        scale = min(usable_w / canvas_width, usable_h / canvas_height)

        # Center the workspace within the image canvas
        drawn_w = canvas_width * scale
        drawn_h = canvas_height * scale
        offset_x = margin + (usable_w - drawn_w) / 2.0
        offset_y = margin + (usable_h - drawn_h) / 2.0

        return cls(
            canvas_width=float(canvas_width),
            canvas_height=float(canvas_height),
            image_width=int(image_width),
            image_height=int(image_height),
            margin=float(margin),
            scale=float(scale),
            offset_x=float(offset_x),
            offset_y=float(offset_y),
        )

    def to_pixel(self, x: float, y: float) -> tuple[float, float]:
        """
        Convert Cartesian (x, y) to image pixel coordinates (px, py).
        Cartesian +Y points UP; image pixel +Y points DOWN.
        """
        px = self.offset_x + x * self.scale
        # Invert Y so that y=0 is at bottom and y=canvas_height is at top
        py = float(self.image_height) - (self.offset_y + y * self.scale)
        return (px, py)

    def to_world(self, px: float, py: float) -> tuple[float, float]:
        """Convert pixel coordinates (px, py) back to Cartesian (x, y)."""
        x = (px - self.offset_x) / self.scale
        y = ((float(self.image_height) - py) - self.offset_y) / self.scale
        return (x, y)

    def scale_dist(self, distance: float) -> float:
        """Scale a Euclidean distance from millimeters/units to pixels."""
        return float(distance) * self.scale
