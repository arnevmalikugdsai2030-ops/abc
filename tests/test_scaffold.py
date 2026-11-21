"""
Initial Architecture & Environment Scaffold Smoke Tests.
"""
import pytest
from mirrorlab import __version__
from mirrorlab.schema import (
    Point2D,
    Vector2D,
    TaskType,
    VisibilityState,
    ObstructionReason,
)


def test_package_metadata():
    """Verify package version is defined."""
    assert __version__ == "0.1.0"


def test_point2d_euclidean_distance():
    """Verify Point2D Euclidean distance calculation."""
    p1 = Point2D(x=0.0, y=0.0)
    p2 = Point2D(x=3.0, y=4.0)
    assert p1.distance_to(p2) == pytest.approx(5.0)


def test_vector2d_normalization():
    """Verify Vector2D magnitude and normalization."""
    v = Vector2D(dx=0.0, dy=10.0)
    assert v.magnitude == pytest.approx(10.0)
    norm = v.normalized()
    assert norm.dx == pytest.approx(0.0)
    assert norm.dy == pytest.approx(1.0)
    assert norm.magnitude == pytest.approx(1.0)


def test_vector2d_zero_length_raises():
    """Verify normalizing a zero-length vector raises ValueError."""
    v_zero = Vector2D(dx=0.0, dy=0.0)
    with pytest.raises(ValueError, match="Cannot normalize zero-length vector"):
        v_zero.normalized()


def test_enum_completeness():
    """Verify all 5 mandatory task types exist in TaskType enum."""
    expected_tasks = {
        "TASK_A_VISIBILITY",
        "TASK_B_OBSTRUCTION",
        "TASK_C_REPAIR",
        "TASK_D_COUNTERFACTUAL",
        "TASK_E_COSMETIC",
    }
    actual_tasks = {t.value for t in TaskType}
    assert expected_tasks.issubset(actual_tasks)

    # Verify diagnostic reasons
    assert ObstructionReason.MISSED_MIRROR.value == "MISSED_MIRROR"
    assert ObstructionReason.OUT_OF_FOV.value == "OUT_OF_FOV"
    assert ObstructionReason.BLOCKED_INCIDENT.value == "BLOCKED_INCIDENT"
    assert ObstructionReason.BLOCKED_REFLECTED.value == "BLOCKED_REFLECTED"
    assert ObstructionReason.BACK_OF_MIRROR.value == "BACK_OF_MIRROR"
