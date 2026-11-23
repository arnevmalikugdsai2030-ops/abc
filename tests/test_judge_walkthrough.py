"""
Unit Tests for Judge Demo Walkthrough Component.

Verifies:
1. Walkthrough module imports without errors.
2. Moment 2 Triple-Verifier consensus calculation holds on demo scene.
3. Moment 3 Failure taxonomy classification integrates correctly.
4. Moment 5 SFT and Gymnasium environment initialization works on generated scene.
"""

import numpy as np

from mirrorlab.evaluation.failure_taxonomy import (
    FailureCategory,
    FailureTaxonomyClassifier,
)
from mirrorlab.export.sft_exporter import SFTDatasetExporter
from mirrorlab.generation.scene_generator import SceneGenerator
from mirrorlab.geometry.independent_verifier import IndependentGeometryVerifier
from mirrorlab.geometry.oracle import VisibilityOracle
from mirrorlab.geometry.ray_march_verifier import DirectSnellRayVerifier
from mirrorlab.rl.gym_env import MirrorLabEnv
from mirrorlab.schema.common import (
    ObstructionReason,
    Point2D,
    Vector2D,
    VisibilityState,
)
from mirrorlab.schema.scene import (
    CameraSpec,
    DifficultyLevel,
    InspectionScene,
    MirrorSpec,
    SceneCategory,
    TargetSpec,
)


class TestJudgeWalkthroughBackend:
    """Verifies that all computational modules powering the judge demo work flawlessly."""

    def test_moment_2_triple_verifier_consensus(self):
        cam = CameraSpec(
            position=Point2D(x=200.0, y=200.0), heading_deg=65.0, fov_degrees=65.0
        )
        mirror = MirrorSpec(
            endpoint_a=Point2D(x=150.0, y=700.0),
            endpoint_b=Point2D(x=650.0, y=700.0),
            reflective_normal=Vector2D(dx=0.0, dy=-1.0),
        )
        target = TargetSpec(center=Point2D(x=500.0, y=250.0), radius_mm=8.0)
        scene = InspectionScene(
            scene_id="demo_triple_test",
            seed=101,
            category=SceneCategory.VALID_VISIBLE,
            difficulty=DifficultyLevel.STANDARD,
            ground_truth_state=VisibilityState.VISIBLE,
            ground_truth_reason=ObstructionReason.NONE,
            diagnostic_details="Clean path",
            camera=cam,
            mirror=mirror,
            target=target,
            obstacles=[],
        )

        oracle_res = VisibilityOracle.evaluate(scene)
        fermat_res = IndependentGeometryVerifier.evaluate(scene)
        snell_res = DirectSnellRayVerifier.evaluate_scene(scene)

        # Binary consensus
        assert oracle_res.state == VisibilityState.VISIBLE
        assert fermat_res.state == VisibilityState.VISIBLE
        snell_state = snell_res.get("state")
        if hasattr(snell_state, "value"):
            snell_state = snell_state.value
        assert str(snell_state) == "VISIBLE"

        # Continuous reflection point residual
        fermat_pt = [fermat_res.reflection_point[0], fermat_res.reflection_point[1]]
        snell_pt = snell_res.get("reflection_point")

        diff_of = np.linalg.norm(oracle_res.reflection_point - np.array(fermat_pt))
        diff_os = np.linalg.norm(oracle_res.reflection_point - np.array(snell_pt))
        max_res = max(diff_of, diff_os)

        # Must be below 1e-4 mm tolerance
        assert max_res < 1e-4

    def test_moment_3_failure_classification(self):
        diag = FailureTaxonomyClassifier.classify_single_task(
            task_id="DEMO_PUB_042",
            ground_truth={
                "visibility_state": "NOT_VISIBLE",
                "ground_truth_reason": "MISSED_MIRROR",
            },
            predicted_response={
                "visibility_state": "VISIBLE",
                "reasoning": "The defect reflects off the mirror surface line.",
            },
        )
        assert diag.category == FailureCategory.VIRTUAL_IMAGE_REASONING_FAILURE

    def test_moment_5_task_factory_sft_and_gym(self):
        gen = SceneGenerator()
        scene = gen.generate_scene(
            seed=2026,
            category=SceneCategory.VALID_VISIBLE,
            difficulty=DifficultyLevel.STANDARD,
        )

        sft_ex = SFTDatasetExporter.export_scene(scene)
        assert len(sft_ex.messages) == 3

        env = MirrorLabEnv(scene=scene)
        obs, info = env.reset()
        assert obs.shape == (12,)
        assert "visibility_state" in info
