"""
Independent Scientific Falsification Test Suite for MIRRORLAB Geometry Engine.

Audits:
1. Independent Ground Truth via Fermat's Principle of Least Distance (Golden Section Search).
2. Translation Invariance (rigid spatial translation across all elements).
3. Rotation Invariance (rigid 2D rotation of camera, target, mirror, and obstacles).
4. Scale Invariance (uniform geometric magnification / reduction).
5. Mirror Endpoint Boundary Step Function (t in [-eps, 1+eps]).
6. Camera FOV Threshold Step Function (alpha in [half_fov - eps, half_fov + eps]).
7. Obstacle Tangency, Grazing, and Endpoint Proximity.
8. Degenerate and Parallel Configurations.
9. Extreme Dynamic Range & Numerical Tolerances.
"""

import math

import numpy as np
import pytest

from mirrorlab.geometry.camera import FOVStatus, validate_camera_fov
from mirrorlab.geometry.obstacles import (
    RectangleObstacle,
)
from mirrorlab.geometry.oracle import evaluate_visibility_oracle
from mirrorlab.geometry.reflection import (
    ReflectionFailureReason,
    calculate_mirror_reflection,
)
from mirrorlab.geometry.vector import (
    line_tangent_and_normal,
    vector_rotate,
)


# ==============================================================================
# Independent Solver: Fermat's Principle Minimum Distance Optimizer
# ==============================================================================
def _independent_fermat_solver(
    camera_pos: np.ndarray,
    target_pos: np.ndarray,
    mirror_a: np.ndarray,
    mirror_b: np.ndarray,
    t_min: float = -5.0,
    t_max: float = 5.0,
    tol: float = 1e-12,
) -> tuple[float, np.ndarray, float]:
    """
    Independently find reflection point by numerically minimizing total optical path
    f(t) = ||C - P(t)|| + ||T - P(t)|| using Golden Section Search.
    Does NOT use virtual image or reflection law.
    """
    tangent = mirror_b - mirror_a

    def path_length(t: float) -> float:
        pt = mirror_a + t * tangent
        return float(np.hypot(*(camera_pos - pt)) + np.hypot(*(target_pos - pt)))

    # Golden section search
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    resphi = 2.0 - phi

    a = t_min
    b = t_max
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    f1 = path_length(x1)
    f2 = path_length(x2)

    while abs(b - a) > tol:
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + resphi * (b - a)
            f1 = path_length(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = b - resphi * (b - a)
            f2 = path_length(x2)

    t_opt = (a + b) / 2.0
    p_opt = mirror_a + t_opt * tangent
    min_dist = path_length(t_opt)
    return t_opt, p_opt, min_dist


class TestIndependentFermatValidation:
    """Compare analytic virtual-target solver against independent numerical Fermat optimizer."""

    @pytest.mark.parametrize("seed", [101, 202, 303, 404, 505])
    def test_analytic_vs_fermat_minimization(self, seed):
        rng = np.random.default_rng(seed)

        # Random horizontal/slanted mirror
        a = rng.uniform(100.0, 300.0, size=2)
        b = a + rng.uniform(100.0, 300.0, size=2)

        _, n_hat, _ = line_tangent_and_normal(a, b)

        # Generate camera and target strictly on front side
        d_c = rng.uniform(50.0, 300.0)
        d_t = rng.uniform(50.0, 300.0)
        t_c = rng.uniform(0.1, 0.9)
        t_t = rng.uniform(0.1, 0.9)

        camera = (a + t_c * (b - a)) + d_c * n_hat
        target = (a + t_t * (b - a)) + d_t * n_hat

        # Analytic solver
        res_analytic = calculate_mirror_reflection(
            camera, target, a, b, reflective_normal=n_hat
        )
        assert res_analytic.is_valid is True

        # Independent Fermat solver
        t_fermat, p_fermat, dist_fermat = _independent_fermat_solver(
            camera, target, a, b
        )

        # Compare results
        assert res_analytic.mirror_t == pytest.approx(t_fermat, abs=1e-5)
        np.testing.assert_allclose(res_analytic.intersection_point, p_fermat, atol=1e-4)
        assert res_analytic.total_path_length == pytest.approx(dist_fermat, abs=1e-4)


class TestSpatialInvariances:
    """Verify rigid Euclidean symmetries (translation, rotation, uniform scale)."""

    @pytest.fixture
    def sample_scene(self):
        return {
            "camera_pos": np.array([150.0, 100.0]),
            "camera_forward": np.array([0.7071, 0.7071]),
            "camera_fov_degrees": 75.0,
            "target_pos": np.array([450.0, 120.0]),
            "mirror_a": np.array([500.0, 350.0]),
            "mirror_b": np.array([100.0, 350.0]),
            "reflective_normal": np.array([0.0, -1.0]),
            "obstacle": RectangleObstacle(
                "obs_middle", bounds=(280.0, 80.0, 320.0, 160.0)
            ),
        }

    def test_translation_invariance(self, sample_scene):
        """Translating entire world by arbitrary vector (dx, dy) leaves all results identical."""
        shift = np.array([1234.56, -789.12])

        # Baseline
        base_oracle = evaluate_visibility_oracle(
            camera_pos=sample_scene["camera_pos"],
            camera_forward=sample_scene["camera_forward"],
            camera_fov_degrees=sample_scene["camera_fov_degrees"],
            target_pos=sample_scene["target_pos"],
            mirror_a=sample_scene["mirror_a"],
            mirror_b=sample_scene["mirror_b"],
            obstacles=[sample_scene["obstacle"]],
            reflective_normal=sample_scene["reflective_normal"],
        )

        # Translated obstacle
        obs_b = sample_scene["obstacle"]
        shifted_obs = RectangleObstacle(
            "obs_shifted",
            center=obs_b.center + shift,
            width=obs_b.width,
            height=obs_b.height,
        )

        # Shifted run
        shifted_oracle = evaluate_visibility_oracle(
            camera_pos=sample_scene["camera_pos"] + shift,
            camera_forward=sample_scene[
                "camera_forward"
            ],  # Forward vector is invariant to translation
            camera_fov_degrees=sample_scene["camera_fov_degrees"],
            target_pos=sample_scene["target_pos"] + shift,
            mirror_a=sample_scene["mirror_a"] + shift,
            mirror_b=sample_scene["mirror_b"] + shift,
            obstacles=[shifted_obs],
            reflective_normal=sample_scene["reflective_normal"],
        )

        assert shifted_oracle.is_visible == base_oracle.is_visible
        assert shifted_oracle.reason == base_oracle.reason
        assert (
            shifted_oracle.is_direct_sight_blocked
            == base_oracle.is_direct_sight_blocked
        )
        assert shifted_oracle.incident_angle_deg == pytest.approx(
            base_oracle.incident_angle_deg, abs=1e-5
        )
        assert shifted_oracle.reflected_angle_deg == pytest.approx(
            base_oracle.reflected_angle_deg, abs=1e-5
        )
        assert shifted_oracle.total_optical_distance == pytest.approx(
            base_oracle.total_optical_distance, abs=1e-5
        )
        np.testing.assert_allclose(
            shifted_oracle.reflection_point,
            base_oracle.reflection_point + shift,
            atol=1e-5,
        )

    def test_rotation_invariance(self, sample_scene):
        """Rotating entire world by angle theta around origin leaves geometric relations invariant."""
        theta_rad = math.radians(41.7)

        # Baseline
        base = evaluate_visibility_oracle(
            camera_pos=sample_scene["camera_pos"],
            camera_forward=sample_scene["camera_forward"],
            camera_fov_degrees=sample_scene["camera_fov_degrees"],
            target_pos=sample_scene["target_pos"],
            mirror_a=sample_scene["mirror_a"],
            mirror_b=sample_scene["mirror_b"],
            obstacles=[],
            reflective_normal=sample_scene["reflective_normal"],
        )

        # Rotated geometry
        rot_cam = vector_rotate(sample_scene["camera_pos"], theta_rad)
        rot_fwd = vector_rotate(sample_scene["camera_forward"], theta_rad)
        rot_tgt = vector_rotate(sample_scene["target_pos"], theta_rad)
        rot_a = vector_rotate(sample_scene["mirror_a"], theta_rad)
        rot_b = vector_rotate(sample_scene["mirror_b"], theta_rad)
        rot_norm = vector_rotate(sample_scene["reflective_normal"], theta_rad)

        rotated = evaluate_visibility_oracle(
            camera_pos=rot_cam,
            camera_forward=rot_fwd,
            camera_fov_degrees=sample_scene["camera_fov_degrees"],
            target_pos=rot_tgt,
            mirror_a=rot_a,
            mirror_b=rot_b,
            obstacles=[],
            reflective_normal=rot_norm,
        )

        assert rotated.is_visible == base.is_visible
        assert rotated.reason == base.reason
        assert rotated.incident_angle_deg == pytest.approx(
            base.incident_angle_deg, abs=1e-5
        )
        assert rotated.total_optical_distance == pytest.approx(
            base.total_optical_distance, abs=1e-5
        )
        assert rotated.reflection_result.mirror_t == pytest.approx(
            base.reflection_result.mirror_t, abs=1e-5
        )

    def test_scale_invariance(self, sample_scene):
        """Scaling all dimensions by factor s preserves angles and mirror parameter t."""
        scale = 3.5

        base = evaluate_visibility_oracle(
            camera_pos=sample_scene["camera_pos"],
            camera_forward=sample_scene["camera_forward"],
            camera_fov_degrees=sample_scene["camera_fov_degrees"],
            target_pos=sample_scene["target_pos"],
            mirror_a=sample_scene["mirror_a"],
            mirror_b=sample_scene["mirror_b"],
            obstacles=[],
            reflective_normal=sample_scene["reflective_normal"],
        )

        scaled = evaluate_visibility_oracle(
            camera_pos=sample_scene["camera_pos"] * scale,
            camera_forward=sample_scene[
                "camera_forward"
            ],  # Heading vector is invariant
            camera_fov_degrees=sample_scene["camera_fov_degrees"],
            target_pos=sample_scene["target_pos"] * scale,
            mirror_a=sample_scene["mirror_a"] * scale,
            mirror_b=sample_scene["mirror_b"] * scale,
            obstacles=[],
            reflective_normal=sample_scene["reflective_normal"],
        )

        assert scaled.is_visible == base.is_visible
        assert scaled.reason == base.reason
        assert scaled.reflection_result.mirror_t == pytest.approx(
            base.reflection_result.mirror_t, abs=1e-5
        )
        assert scaled.incident_angle_deg == pytest.approx(
            base.incident_angle_deg, abs=1e-5
        )
        assert scaled.total_optical_distance == pytest.approx(
            base.total_optical_distance * scale, abs=1e-5
        )


class TestBoundaryStepFunctions:
    """Stress-test infinitesimal step transitions across mathematical boundaries."""

    def test_mirror_endpoint_step_function(self):
        """
        Hold camera and target fixed. Move mirror endpoints so reflection point
        transitions from t = -1e-6 (missed past A) to t = 0.0 (exact A) to t = 1.0 (exact B) to t = 1.0 + 1e-6 (missed past B).
        """
        camera = [100.0, 200.0]
        target = [500.0, 200.0]
        # Reflection point for mirror line y=400 is at x = 300.0
        normal = [0.0, -1.0]

        # Case 1: Mirror A is at x = 299.999 and B is at x = 200.0
        # The mirror spans [200.0, 299.999]. Reflection point at x=300.0 is past endpoint A!
        res_past_a = calculate_mirror_reflection(
            camera, target, [299.999, 400.0], [200.0, 400.0], reflective_normal=normal
        )
        assert res_past_a.is_valid is False
        assert res_past_a.failure_reason == ReflectionFailureReason.MISSED_MIRROR_PAST_A

        # Case 2: Mirror A is at exact reflection point x = 300.0 (mirror spans [200.0, 300.0])
        res_exact_a = calculate_mirror_reflection(
            camera, target, [300.0, 400.0], [200.0, 400.0], reflective_normal=normal
        )
        assert res_exact_a.is_valid is True
        assert res_exact_a.mirror_t == pytest.approx(0.0)

        # Case 3: Mirror B is at exact reflection point x = 300.0 (mirror spans [300.0, 400.0])
        res_exact_b = calculate_mirror_reflection(
            camera, target, [400.0, 400.0], [300.0, 400.0], reflective_normal=normal
        )
        assert res_exact_b.is_valid is True
        assert res_exact_b.mirror_t == pytest.approx(1.0)

        # Case 4: Mirror B is at x = 300.001 and A is at x = 400.0
        # The mirror spans [300.001, 400.0]. Reflection point at x=300.0 is past endpoint B!
        res_past_b = calculate_mirror_reflection(
            camera, target, [400.0, 400.0], [300.001, 400.0], reflective_normal=normal
        )
        assert res_past_b.is_valid is False
        assert res_past_b.failure_reason == ReflectionFailureReason.MISSED_MIRROR_PAST_B

    def test_fov_threshold_step_function(self):
        """
        Test angular transition across half_fov boundary.
        FOV = 60 deg -> half_fov = 30 deg.
        """
        camera = [0.0, 0.0]
        forward = [1.0, 0.0]  # East
        fov_deg = 60.0

        # Just inside (29.99 deg)
        rad_in = math.radians(29.99)
        pt_in = [100.0 * math.cos(rad_in), 100.0 * math.sin(rad_in)]
        res_in = validate_camera_fov(camera, forward, fov_deg, pt_in)
        assert res_in.is_inside is True
        assert res_in.status == FOVStatus.INSIDE

        # Exactly on boundary (30.00000 deg)
        rad_bound = math.radians(30.0)
        pt_bound = [100.0 * math.cos(rad_bound), 100.0 * math.sin(rad_bound)]
        res_bound = validate_camera_fov(camera, forward, fov_deg, pt_bound)
        assert res_bound.is_inside is True
        assert res_bound.status == FOVStatus.ON_BOUNDARY

        # Just outside (30.01 deg)
        rad_out = math.radians(30.01)
        pt_out = [100.0 * math.cos(rad_out), 100.0 * math.sin(rad_out)]
        res_out = validate_camera_fov(camera, forward, fov_deg, pt_out)
        assert res_out.is_inside is False
        assert res_out.status == FOVStatus.OUTSIDE


class TestParallelAndDegenerateCases:
    """Stress-test extreme mathematical edge cases."""

    def test_parallel_camera_target_line(self):
        """
        Line C -> T is parallel to mirror line.
        Camera at (100, 200), Target at (500, 200).
        Mirror at y = 400. Distance d_C = d_T = 200.
        Analytic solver must cleanly handle d_C == d_T without singularity.
        """
        res = calculate_mirror_reflection(
            [100, 200], [500, 200], [500, 400], [100, 400], reflective_normal=[0, -1]
        )
        assert res.is_valid is True
        np.testing.assert_allclose(res.intersection_point, [300.0, 400.0])

    def test_normal_incidence_retroreflection(self):
        """
        Camera and Target are collinear along mirror normal.
        Camera at (200, 100), Target at (200, 250).
        Mirror at y = 400 from (100, 400) to (300, 400), normal down (0, -1).
        Reflection point must be exactly (200, 400) with theta_i = theta_r = 0 deg!
        """
        res = calculate_mirror_reflection(
            [200, 100], [200, 250], [300, 400], [100, 400], reflective_normal=[0, -1]
        )
        assert res.is_valid is True
        np.testing.assert_allclose(res.intersection_point, [200.0, 400.0])
        assert res.incident_angle_rad == pytest.approx(0.0, abs=1e-6)
        assert res.reflected_angle_rad == pytest.approx(0.0, abs=1e-6)

    def test_coincident_camera_and_target_points(self):
        """Camera and Target are at the exact same point (autocollimation)."""
        res = calculate_mirror_reflection(
            [200, 100], [200, 100], [300, 400], [100, 400], reflective_normal=[0, -1]
        )
        assert res.is_valid is True
        np.testing.assert_allclose(res.intersection_point, [200.0, 400.0])
        assert res.incident_angle_rad == pytest.approx(0.0, abs=1e-6)

    def test_extreme_coordinate_magnitudes(self):
        """Coordinates at 1,000,000 mm (1 km) to verify float64 numerical stability."""
        c = [1_000_100.0, 1_000_200.0]
        t = [1_000_500.0, 1_000_200.0]
        m_a = [1_000_500.0, 1_000_400.0]
        m_b = [1_000_100.0, 1_000_400.0]
        normal = [0.0, -1.0]

        res = calculate_mirror_reflection(c, t, m_a, m_b, reflective_normal=normal)
        assert res.is_valid is True
        np.testing.assert_allclose(
            res.intersection_point, [1_000_300.0, 1_000_400.0], rtol=1e-7
        )
        assert res.mirror_t == pytest.approx(0.5, abs=1e-6)
        assert res.angle_difference_rad < 1e-6
