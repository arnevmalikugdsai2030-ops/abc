"""
MIRRORLAB Deterministic Computational Geometry Engine.
"""

from mirrorlab.geometry.camera import (
    Camera2D,
    CameraFOVResult,
    FOVStatus,
    validate_camera_fov,
)
from mirrorlab.geometry.obstacles import (
    CircleObstacle,
    ObstacleBase,
    ObstacleHit,
    PathLeg,
    PathOcclusionResult,
    RectangleObstacle,
    check_segment_occlusion,
    check_two_leg_path_occlusion,
)
from mirrorlab.geometry.oracle import (
    VisibilityOracleResult,
    evaluate_visibility_oracle,
)
from mirrorlab.geometry.reflection import (
    MirrorReflectionResult,
    ReflectionFailureReason,
    calculate_mirror_reflection,
)
from mirrorlab.geometry.vector import (
    EPSILON,
    DegenerateLineError,
    GeometryError,
    ZeroVectorError,
    cross_product_2d,
    dot_product,
    line_tangent_and_normal,
    point_to_line_distance,
    point_to_line_signed_distance,
    reflect_point_across_line,
    scalar_mul,
    vector_add,
    vector_magnitude,
    vector_normalize,
    vector_rotate,
    vector_sub,
)

__all__ = [
    "EPSILON",
    "Camera2D",
    "CameraFOVResult",
    "CircleObstacle",
    "DegenerateLineError",
    "FOVStatus",
    "GeometryError",
    "MirrorReflectionResult",
    "ObstacleBase",
    "ObstacleHit",
    "PathLeg",
    "PathOcclusionResult",
    "RectangleObstacle",
    "ReflectionFailureReason",
    "VisibilityOracleResult",
    "ZeroVectorError",
    "calculate_mirror_reflection",
    "check_segment_occlusion",
    "check_two_leg_path_occlusion",
    "cross_product_2d",
    "dot_product",
    "evaluate_visibility_oracle",
    "line_tangent_and_normal",
    "point_to_line_distance",
    "point_to_line_signed_distance",
    "reflect_point_across_line",
    "scalar_mul",
    "validate_camera_fov",
    "vector_add",
    "vector_magnitude",
    "vector_normalize",
    "vector_rotate",
    "vector_sub",
]
