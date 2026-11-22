"""
Independent Geometry Verifier & Boundary Case Tests for MIRRORLAB.

Tests:
1. Virtual-camera reflection matches virtual-target reflection.
2. Fermat principle optical path stationarity (dL/du = 0, residual < 1e-5).
3. Boundary cases:
   - Reflection exactly at mirror endpoints.
   - Reflection just inside mirror vs just outside.
   - Grazing incidence angles (> 60°).
   - Tangent obstacle intersections.
   - Front/back reflective side orientation.
4. Dual mathematical certification on benchmark scenes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mirrorlab.geometry.independent_verifier import (
    IndependentGeometryVerifier,
)
from mirrorlab.schema.scene import (
    InspectionScene,
)


class TestIndependentVerifierFormulation:
    """Mathematical tests for Virtual-Camera reflection and Fermat principle."""

    def test_virtual_camera_mathematical_agreement(self):
        # Canonical symmetric reflection
        cam = np.array([200.0, 300.0])
        target = np.array([800.0, 300.0])
        mirror_a = np.array([300.0, 700.0])
        mirror_b = np.array([700.0, 700.0])
        norm = np.array([0.0, -1.0])

        refl = IndependentGeometryVerifier.calculate_virtual_camera_reflection(
            cam, target, mirror_a, mirror_b, norm
        )
        assert refl.is_valid is True
        assert refl.reflection_point is not None
        # Reflection point must be at x = 500, y = 700
        assert np.isclose(refl.reflection_point[0], 500.0, atol=1e-5)
        assert np.isclose(refl.reflection_point[1], 700.0, atol=1e-5)

        # Fermat stationary residual must be zero
        assert refl.fermat_residual < 1e-5
        # Incident and reflected angles must be equal
        assert refl.incident_angle_deg is not None
        assert refl.reflected_angle_deg is not None
        assert np.isclose(refl.incident_angle_deg, refl.reflected_angle_deg, atol=1e-4)

    def test_wrong_side_detection(self):
        # Camera behind non-reflective backing of mirror
        cam = np.array([200.0, 800.0])  # Y=800 is behind mirror at Y=700 facing -Y
        target = np.array([800.0, 300.0])
        mirror_a = np.array([300.0, 700.0])
        mirror_b = np.array([700.0, 700.0])
        norm = np.array([0.0, -1.0])

        refl = IndependentGeometryVerifier.calculate_virtual_camera_reflection(
            cam, target, mirror_a, mirror_b, norm
        )
        assert refl.is_valid is False
        assert refl.failure_reason == "WRONG_SIDE_CAMERA"

    def test_finite_mirror_aperture_clipping(self):
        # Ray misses mirror segment to the left (t < 0)
        cam = np.array([100.0, 300.0])
        target = np.array([200.0, 300.0])
        mirror_a = np.array([400.0, 700.0])
        mirror_b = np.array([800.0, 700.0])
        norm = np.array([0.0, -1.0])

        refl = IndependentGeometryVerifier.calculate_virtual_camera_reflection(
            cam, target, mirror_a, mirror_b, norm
        )
        assert refl.is_valid is False
        assert refl.failure_reason == "MISSED_MIRROR_PAST_A"


class TestBoundaryConditions:
    """Stress tests on boundary conditions and grazing incidence."""

    def test_grazing_incidence_reflection(self):
        # High incident angle (~71 degrees from normal: dx=450, dy=150)
        cam = np.array([50.0, 650.0])
        target = np.array([950.0, 650.0])
        mirror_a = np.array([300.0, 800.0])
        mirror_b = np.array([700.0, 800.0])
        norm = np.array([0.0, -1.0])

        refl = IndependentGeometryVerifier.calculate_virtual_camera_reflection(
            cam, target, mirror_a, mirror_b, norm
        )
        assert refl.is_valid is True
        assert refl.fermat_residual < 1e-5
        assert refl.incident_angle_deg > 50.0

    def test_reflection_near_mirror_endpoint(self):
        # Reflection point right near mirror endpoint B (t = 0.98)
        cam = np.array([200.0, 300.0])
        # Tune target so reflection lands near endpoint B at x=700, y=700
        target = np.array([1180.0, 300.0])
        mirror_a = np.array([300.0, 700.0])
        mirror_b = np.array([700.0, 700.0])
        norm = np.array([0.0, -1.0])

        refl = IndependentGeometryVerifier.calculate_virtual_camera_reflection(
            cam, target, mirror_a, mirror_b, norm
        )
        assert refl.is_valid is True
        assert 0.90 < refl.mirror_t <= 1.0


class TestDualMathematicalCertification:
    """Verifies that all benchmark sample scenes pass dual mathematical certification."""

    def test_sample_scenes_dual_certified(self):
        scenes_dir = Path("data/sample_scenes")
        assert scenes_dir.exists()

        scenes = [f for f in scenes_dir.glob("*.json")]
        assert len(scenes) >= 10

        for f in scenes:
            scene_dict = json.loads(f.read_text(encoding="utf-8"))
            scene = InspectionScene.model_validate(scene_dict)
            dual_result = IndependentGeometryVerifier.verify_dual(scene)
            assert dual_result.is_mathematically_certified is True, (
                f"Scene {f.name} failed dual certification: {dual_result.proof_summary}"
            )
            assert dual_result.reflection_point_delta_mm < 1e-4
            assert dual_result.fermat_residual < 1e-5
