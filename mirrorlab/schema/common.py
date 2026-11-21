"""
Common Pydantic schemas, enumerations, and value objects for MIRRORLAB.
"""
from enum import Enum
import math
from pydantic import BaseModel, Field, ConfigDict


class TaskType(str, Enum):
    """The 5 mandatory benchmark task categories."""
    TASK_A_VISIBILITY = "TASK_A_VISIBILITY"
    TASK_B_OBSTRUCTION = "TASK_B_OBSTRUCTION"
    TASK_C_REPAIR = "TASK_C_REPAIR"
    TASK_D_COUNTERFACTUAL = "TASK_D_COUNTERFACTUAL"
    TASK_E_COSMETIC = "TASK_E_COSMETIC"


class VisibilityState(str, Enum):
    """Reflected sightline state."""
    VISIBLE = "VISIBLE"
    NOT_VISIBLE = "NOT_VISIBLE"


class ObstructionReason(str, Enum):
    """Diagnostic reason codes for reflected inspection failure."""
    NONE = "NONE"                          # Fully visible
    OUT_OF_FOV = "OUT_OF_FOV"              # Ray outside camera field of view cone
    MISSED_MIRROR = "MISSED_MIRROR"        # Reflection misses finite mirror line segment
    BLOCKED_INCIDENT = "BLOCKED_INCIDENT"  # Ray from Camera to Mirror blocked by obstacle
    BLOCKED_REFLECTED = "BLOCKED_REFLECTED"# Ray from Mirror to Scratch blocked by obstacle
    BACK_OF_MIRROR = "BACK_OF_MIRROR"      # Ray strikes non-reflective side of mirror
    DIRECT_SIGHT_CLEAR = "DIRECT_SIGHT"    # Direct line-of-sight unblocked (invalid scene for mirror test)


class Point2D(BaseModel):
    """Immutable 2D Cartesian point in millimeters."""
    model_config = ConfigDict(frozen=True)

    x: float = Field(..., description="X coordinate in millimeters")
    y: float = Field(..., description="Y coordinate in millimeters")

    def distance_to(self, other: "Point2D") -> float:
        """Euclidean distance to another point."""
        return math.hypot(self.x - other.x, self.y - other.y)


class Vector2D(BaseModel):
    """Immutable 2D vector."""
    model_config = ConfigDict(frozen=True)

    dx: float = Field(..., description="Delta X component")
    dy: float = Field(..., description="Delta Y component")

    @property
    def magnitude(self) -> float:
        return math.hypot(self.dx, self.dy)

    def normalized(self) -> "Vector2D":
        mag = self.magnitude
        if mag == 0:
            raise ValueError("Cannot normalize zero-length vector")
        return Vector2D(dx=self.dx / mag, dy=self.dy / mag)
