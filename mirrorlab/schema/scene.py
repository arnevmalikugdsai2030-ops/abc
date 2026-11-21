"""
Pydantic Schemas for Complete Inspection Scenes, Camera Specs, Mirrors, and Obstacles.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mirrorlab.geometry.obstacles import CircleObstacle, ObstacleBase, RectangleObstacle
from mirrorlab.schema.common import (
    ObstructionReason,
    Point2D,
    Vector2D,
    VisibilityState,
)


class DifficultyLevel(str, Enum):
    """Scene complexity and geometric tightness."""

    EASY = "EASY"
    MEDIUM = "MEDIUM"
    HARD = "HARD"


class SceneCategory(str, Enum):
    """The 6 controlled benchmark scenario categories."""

    VALID_VISIBLE = "VALID_VISIBLE"
    MIRROR_MISS = "MIRROR_MISS"
    OUTSIDE_FOV = "OUTSIDE_FOV"
    WRONG_SIDE = "WRONG_SIDE"
    BLOCKED_INCIDENT = "BLOCKED_INCIDENT"
    BLOCKED_REFLECTED = "BLOCKED_REFLECTED"


class CameraSpec(BaseModel):
    """Camera specification within an inspection scene."""

    model_config = ConfigDict(frozen=True)

    position: Point2D = Field(..., description="Camera 2D coordinate [x, y]")
    heading_deg: float = Field(
        ..., description="Camera optical axis direction in degrees (0 = +X, 90 = +Y)"
    )
    fov_degrees: float = Field(
        ..., ge=5.0, le=170.0, description="Angular Field of View in degrees"
    )
    max_range: float = Field(
        default=1500.0, gt=0.0, description="Maximum detection range in millimeters"
    )


class MirrorSpec(BaseModel):
    """Finite planar mirror specification."""

    model_config = ConfigDict(frozen=True)

    endpoint_a: Point2D = Field(..., description="First endpoint of mirror [x, y]")
    endpoint_b: Point2D = Field(..., description="Second endpoint of mirror [x, y]")
    reflective_normal: Vector2D = Field(
        ..., description="Unit normal pointing outward from the reflective face"
    )


class TargetSpec(BaseModel):
    """Inspection defect target (the red-X scratch)."""

    model_config = ConfigDict(frozen=True)

    center: Point2D = Field(..., description="Target scratch coordinate [x, y]")
    radius_mm: float = Field(
        default=8.0, gt=0.0, description="Visual marker radius in mm"
    )


class ObstacleSpec(BaseModel):
    """Polygonal or circular opaque obstacle specification."""

    model_config = ConfigDict(frozen=True)

    obstacle_id: str = Field(
        ..., description="Unique obstacle identifier (e.g. 'partition_1', 'blocker_a')"
    )
    obstacle_type: Literal["circle", "rectangle"] = Field(
        ..., description="Obstacle geometric shape"
    )
    center: Point2D = Field(..., description="Center coordinate [x, y]")

    # Circle specific
    radius: float | None = Field(default=None, description="Radius in mm if circle")

    # Rectangle specific
    width: float | None = Field(default=None, description="Width in mm if rectangle")
    height: float | None = Field(default=None, description="Height in mm if rectangle")
    angle_degrees: float = Field(
        default=0.0, description="Rotation angle in degrees if rectangle"
    )

    def to_geometry_obstacle(self) -> ObstacleBase:
        """Convert Pydantic specification to pure geometry obstacle object."""
        if self.obstacle_type == "circle":
            if self.radius is None:
                raise ValueError(f"Circle obstacle '{self.obstacle_id}' missing radius")
            return CircleObstacle(
                self.obstacle_id, [self.center.x, self.center.y], self.radius
            )
        elif self.obstacle_type == "rectangle":
            if self.width is None or self.height is None:
                raise ValueError(
                    f"Rectangle obstacle '{self.obstacle_id}' missing width/height"
                )
            return RectangleObstacle(
                self.obstacle_id,
                center=[self.center.x, self.center.y],
                width=self.width,
                height=self.height,
                angle_degrees=self.angle_degrees,
            )
        raise ValueError(f"Unknown obstacle type: {self.obstacle_type}")


class InspectionScene(BaseModel):
    """
    Complete, self-contained, reproducible specification of an inspection scene.
    """

    model_config = ConfigDict(frozen=True)

    scene_id: str = Field(..., description="Unique scene identifier")
    seed: int = Field(
        ..., description="Integer random seed used to generate this scene"
    )
    category: SceneCategory = Field(..., description="Target scenario category")
    difficulty: DifficultyLevel = Field(
        default=DifficultyLevel.MEDIUM, description="Difficulty level"
    )

    # Canvas dimensions
    canvas_width: float = Field(default=1000.0, gt=0.0)
    canvas_height: float = Field(default=1000.0, gt=0.0)

    # Core scene elements
    camera: CameraSpec
    mirror: MirrorSpec
    target: TargetSpec
    obstacles: list[ObstacleSpec] = Field(default_factory=list)

    # Ground-truth verdict computed by the deterministic oracle
    ground_truth_state: VisibilityState
    ground_truth_reason: ObstructionReason
    diagnostic_details: str
    reflection_point: Point2D | None = None
    virtual_target: Point2D | None = None
    optical_path: list[Point2D] = Field(default_factory=list)
    incident_angle_deg: float | None = None
    reflected_angle_deg: float | None = None
    total_optical_distance: float | None = None
    is_direct_sight_blocked: bool = False
