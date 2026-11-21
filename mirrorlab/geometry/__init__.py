"""
MIRRORLAB Deterministic Computational Geometry Engine.
"""

from mirrorlab.geometry.camera import (
    Camera2D,
    CameraFOVResult,
    FOVStatus,
    validate_camera_fov,
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
    "DegenerateLineError",
    "FOVStatus",
    "GeometryError",
    "MirrorReflectionResult",
    "ReflectionFailureReason",
    "ZeroVectorError",
    "calculate_mirror_reflection",
    "cross_product_2d",
    "dot_product",
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
