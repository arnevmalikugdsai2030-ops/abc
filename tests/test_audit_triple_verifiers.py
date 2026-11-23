"""
Adversarial Tests for Triple-Verifier Fail-Closed Invariants.

Verifies that scripts/audit_triple_verifiers.py strictly rejects:
1. VISIBLE solver results lacking reflection coordinates.
2. Non-finite coordinates (NaN, Inf).
3. Reflection coordinates outside mirror segment bounds.
4. Optical geometry violations (angle imbalance, backside approach).
5. Reflection availability disagreements between solvers.
6. Coordinate residual tolerance breaches.
7. Coverage shortfalls against expected benchmark comparisons.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mirrorlab.schema import Point2D
from mirrorlab.schema.scene import CameraSpec, InspectionScene
from scripts.audit_triple_verifiers import (
    audit_all_benchmark_scenes,
    validate_solver_output,
)


@pytest.fixture
def sample_valid_scene() -> InspectionScene:
    """Loads a real valid benchmark scene."""
    p = Path("data/benchmark_tasks/task_vis_10000/scene.json")
    return InspectionScene.model_validate_json(p.read_text(encoding="utf-8"))


class TestTripleVerifierFailClosedInvariants:
    """Test suite ensuring strict fail-closed enforcement across geometric verifiers."""

    def test_visible_lacking_coordinates_fails(
        self, sample_valid_scene: InspectionScene
    ):
        """Invariant: VISIBLE state must have a finite coordinate."""
        valid, err = validate_solver_output(
            solver_name="oracle",
            state_str="VISIBLE",
            pt=None,
            scene=sample_valid_scene,
            is_reflection_on_mirror=True,
        )
        assert valid is False
        assert err is not None
        assert "state is VISIBLE but reflection_point is None" in err

    def test_non_finite_coordinates_fails(self, sample_valid_scene: InspectionScene):
        """Invariant: Any coordinates produced must be finite."""
        # Test NaN
        valid, err = validate_solver_output(
            solver_name="fermat",
            state_str="VISIBLE",
            pt=(float("nan"), 700.0),
            scene=sample_valid_scene,
            is_reflection_on_mirror=True,
        )
        assert valid is False
        assert err is not None
        assert "non-finite coordinates" in err

        # Test Inf
        valid, err = validate_solver_output(
            solver_name="snell",
            state_str="VISIBLE",
            pt=(350.0, float("inf")),
            scene=sample_valid_scene,
            is_reflection_on_mirror=True,
        )
        assert valid is False
        assert err is not None
        assert "non-finite coordinates" in err

    def test_reflection_point_outside_mirror_segment_fails(
        self, sample_valid_scene: InspectionScene
    ):
        """Invariant: Claimed reflection point must lie strictly within finite mirror segment."""
        # Mirror is between x=351.02 and x=612.36 at y=728.91. Point at x=850 is outside segment (u > 1.0)
        valid, err = validate_solver_output(
            solver_name="oracle",
            state_str="VISIBLE",
            pt=(850.0, 728.9130812677677),
            scene=sample_valid_scene,
            is_reflection_on_mirror=True,
        )
        assert valid is False
        assert err is not None
        assert "outside mirror segment" in err

        # Point at x=100 is outside segment (u < 0.0)
        valid, err = validate_solver_output(
            solver_name="fermat",
            state_str="VISIBLE",
            pt=(100.0, 728.9130812677677),
            scene=sample_valid_scene,
            is_reflection_on_mirror=True,
        )
        assert valid is False
        assert err is not None
        assert "outside mirror segment" in err

    def test_reflection_point_off_mirror_line_fails(
        self, sample_valid_scene: InspectionScene
    ):
        """Invariant: Claimed reflection point must lie on the mirror affine line within tolerance."""
        # Mirror line is y=728.913. Point at y=735 is ~6 mm off line
        valid, err = validate_solver_output(
            solver_name="snell",
            state_str="VISIBLE",
            pt=(437.14, 735.0),
            scene=sample_valid_scene,
            is_reflection_on_mirror=True,
            tolerance_mm=0.1,
        )
        assert valid is False
        assert err is not None
        assert "off mirror line" in err

    def test_backside_normal_approach_fails(self, sample_valid_scene: InspectionScene):
        """Invariant: Rays approaching from non-reflective backside must fail."""
        # Mirror is at y=728.91 with reflective normal (0, -1). Camera at y=850 approaches from backside
        backside_scene = sample_valid_scene.model_copy(
            update={
                "camera": CameraSpec(
                    position=Point2D(x=262.38, y=850.0),
                    heading_deg=270.0,
                    fov_degrees=65.0,
                )
            }
        )
        valid, err = validate_solver_output(
            solver_name="oracle",
            state_str="VISIBLE",
            pt=(437.14, 728.9130812677677),
            scene=backside_scene,
            is_reflection_on_mirror=True,
        )
        assert valid is False
        assert err is not None
        assert "approaches from non-reflective mirror backside" in err

    def test_optical_law_of_reflection_asymmetry_fails(
        self, sample_valid_scene: InspectionScene
    ):
        """Invariant: Reflection point must satisfy theta_i == theta_r."""
        # True reflection point is at x ~ 437.14. Passing x = 360.0 produces asymmetric angles
        valid, err = validate_solver_output(
            solver_name="fermat",
            state_str="VISIBLE",
            pt=(360.0, 728.9130812677677),
            scene=sample_valid_scene,
            is_reflection_on_mirror=True,
        )
        assert valid is False
        assert err is not None
        assert "violates optical reflection geometry" in err

    def test_not_visible_missed_mirror_contradiction_fails(
        self, sample_valid_scene: InspectionScene
    ):
        """Invariant: Stating reason MISSED_MIRROR while coordinate is inside segment is inconsistent."""
        valid, err = validate_solver_output(
            solver_name="oracle",
            state_str="NOT_VISIBLE",
            pt=(437.14, 728.9130812677677),  # inside segment
            scene=sample_valid_scene,
            is_reflection_on_mirror=False,
            reason_str="MISSED_MIRROR",
        )
        assert valid is False
        assert err is not None
        assert "reason is MISSED_MIRROR but coordinate lies inside segment" in err

    def test_full_audit_fails_on_solver_availability_disagreement(self, tmp_path: Path):
        """Invariant: Audit explicitly fails if solvers disagree on reflection availability."""
        # Test that audit detects discrepancies and reports FAIL
        with patch(
            "scripts.audit_triple_verifiers.VisibilityOracle.evaluate"
        ) as mock_o:
            mock_res = MagicMock()
            mock_res.state = MagicMock(value="VISIBLE")
            mock_res.reason = MagicMock(value="CLEAR_SPECULAR_PATH")
            mock_res.reflection_point = None  # Contradiction: VISIBLE but None pt
            mock_o.return_value = mock_res

            res = audit_all_benchmark_scenes(
                data_root_path="data",
                tolerance_mm=0.1,
                expected_total_tasks=350,
                save_artifact=False,
            )
            assert res["all_passed"] is False
            assert len(res["failed_scene_ids"]) > 0

    def test_full_audit_fails_on_coverage_shortfall(self):
        """Invariant: Audit explicitly fails if coordinate comparison coverage is short."""
        # Passing an unrealistically high expected coordinate comparison count must fail audit
        res = audit_all_benchmark_scenes(
            data_root_path="data",
            tolerance_mm=0.1,
            expected_total_tasks=350,
            expected_coord_comparisons=999,  # Shortfall: actual is 306 != 999
            save_artifact=False,
        )
        assert res["all_passed"] is False
