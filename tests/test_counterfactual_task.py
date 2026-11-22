"""
Unit and Integration Tests for Counterfactual Paired Scene Generation in MIRRORLAB.

Tests:
1. Pair Type 1: Mirror Angle Flip (Outcome flips, camera/target/obstacles fixed).
2. Pair Type 2: Cosmetic Control (Geometry identical, theme changes, outcome invariant).
3. Pair Type 3: Obstacle Insertion (Starts clear, blocking obstacle flips outcome).
4. Pair Type 4: Irrelevant Obstacle (Starts clear, distractor preserves outcome with clearance >= 30 mm).
5. Pair Verification Rejection (Mismatched relationships are rejected).
6. Deterministic Reproducibility (Identical seeds yield identical pairs).
7. Artifact Bundle Integrity (5-artifact bundle per child scene + pair.json + manifests).
"""

import json
from pathlib import Path

import pytest

from mirrorlab.generation.counterfactual_factory import CounterfactualFactory
from mirrorlab.rendering.themes import RenderTheme
from mirrorlab.schema.scene import DifficultyLevel
from mirrorlab.schema.task import (
    CounterfactualPairType,
    ExpectedPairRelationship,
    VisibilityReasonCode,
)


@pytest.fixture
def factory() -> CounterfactualFactory:
    """Fixture providing a CounterfactualFactory instance."""
    return CounterfactualFactory(canvas_width=1000.0, canvas_height=1000.0, image_size=512)


class TestCounterfactualPairType1MirrorAngleFlip:
    """Tests for Pair Type 1: Mirror Angle Flip."""

    def test_mirror_angle_flip_flips_visibility(self, factory: CounterfactualFactory):
        bundle = factory.create_mirror_angle_flip_pair(
            seed=40001,
            pair_id="test_flip_01",
            difficulty=DifficultyLevel.MEDIUM,
        )

        record = bundle.pair_record
        assert record.pair_type == CounterfactualPairType.MIRROR_ANGLE_FLIP
        assert record.expected_relationship == ExpectedPairRelationship.FLIPPED_OUTCOME
        assert record.relationship_verified is True
        assert record.changed_parameter == "mirror.angle_deg"

        # Scene A must be CLEAR / VISIBLE
        out_a = record.ground_truth_outcomes["test_flip_01_a"]
        assert out_a.is_visible is True
        assert out_a.reason_code == VisibilityReasonCode.CLEAR

        # Scene B must be NOT VISIBLE
        out_b = record.ground_truth_outcomes["test_flip_01_b"]
        assert out_b.is_visible is False
        assert out_b.reason_code in (VisibilityReasonCode.MIRROR_MISS, VisibilityReasonCode.OUTSIDE_FOV)

        # Invariant components: Camera, Target, Obstacles must match
        sc_a = bundle.scene_a_bundle.scene
        sc_b = bundle.scene_b_bundle.scene
        assert sc_a.camera.position == sc_b.camera.position
        assert sc_a.camera.heading_deg == sc_b.camera.heading_deg
        assert sc_a.camera.fov_degrees == sc_b.camera.fov_degrees
        assert sc_a.target.center == sc_b.target.center
        assert len(sc_a.obstacles) == len(sc_b.obstacles)

        # Perturbation delta must be non-zero
        delta = record.new_values["delta_angle_deg"]
        assert abs(delta) >= 2.0


class TestCounterfactualPairType2CosmeticControl:
    """Tests for Pair Type 2: Cosmetic Control."""

    def test_cosmetic_control_preserves_outcome(self, factory: CounterfactualFactory):
        bundle = factory.create_cosmetic_control_pair(
            seed=41001,
            pair_id="test_cosm_01",
            theme_a=RenderTheme.DARK_NAVY,
            theme_b=RenderTheme.LIGHT_OFFWHITE,
        )

        record = bundle.pair_record
        assert record.pair_type == CounterfactualPairType.COSMETIC_CONTROL
        assert record.expected_relationship == ExpectedPairRelationship.INVARIANT_OUTCOME
        assert record.relationship_verified is True
        assert record.changed_parameter == "rendering.theme"

        out_a = record.ground_truth_outcomes["test_cosm_01_a"]
        out_b = record.ground_truth_outcomes["test_cosm_01_b"]
        assert out_a.is_visible == out_b.is_visible
        assert out_a.reason_code == out_b.reason_code

        # Geometry must be 100% identical
        sc_a = bundle.scene_a_bundle.scene
        sc_b = bundle.scene_b_bundle.scene
        assert sc_a.camera.position == sc_b.camera.position
        assert sc_a.mirror.endpoint_a == sc_b.mirror.endpoint_a
        assert sc_a.mirror.endpoint_b == sc_b.mirror.endpoint_b
        assert sc_a.target.center == sc_b.target.center
        assert len(sc_a.obstacles) == len(sc_b.obstacles)


class TestCounterfactualPairType3ObstacleInsertion:
    """Tests for Pair Type 3: Obstacle Insertion."""

    def test_obstacle_insertion_flips_visibility(self, factory: CounterfactualFactory):
        bundle = factory.create_obstacle_insertion_pair(
            seed=42001,
            pair_id="test_obs_ins_01",
            difficulty=DifficultyLevel.MEDIUM,
        )

        record = bundle.pair_record
        assert record.pair_type == CounterfactualPairType.OBSTACLE_INSERTION
        assert record.expected_relationship == ExpectedPairRelationship.FLIPPED_OUTCOME
        assert record.relationship_verified is True

        out_a = record.ground_truth_outcomes["test_obs_ins_01_a"]
        out_b = record.ground_truth_outcomes["test_obs_ins_01_b"]
        assert out_a.is_visible is True
        assert out_a.reason_code == VisibilityReasonCode.CLEAR

        assert out_b.is_visible is False
        assert out_b.reason_code in (
            VisibilityReasonCode.BLOCKED_CAMERA_TO_MIRROR,
            VisibilityReasonCode.BLOCKED_MIRROR_TO_TARGET,
        )

        sc_a = bundle.scene_a_bundle.scene
        sc_b = bundle.scene_b_bundle.scene
        assert len(sc_b.obstacles) == len(sc_a.obstacles) + 1


class TestCounterfactualPairType4IrrelevantObstacle:
    """Tests for Pair Type 4: Irrelevant Obstacle."""

    def test_irrelevant_obstacle_preserves_visibility(self, factory: CounterfactualFactory):
        bundle = factory.create_irrelevant_obstacle_pair(
            seed=43001,
            pair_id="test_irrel_01",
            difficulty=DifficultyLevel.MEDIUM,
        )

        record = bundle.pair_record
        assert record.pair_type == CounterfactualPairType.IRRELEVANT_OBSTACLE
        assert record.expected_relationship == ExpectedPairRelationship.INVARIANT_OUTCOME
        assert record.relationship_verified is True

        out_a = record.ground_truth_outcomes["test_irrel_01_a"]
        out_b = record.ground_truth_outcomes["test_irrel_01_b"]
        assert out_a.is_visible is True
        assert out_b.is_visible is True
        assert out_a.reason_code == VisibilityReasonCode.CLEAR
        assert out_b.reason_code == VisibilityReasonCode.CLEAR

        # Clearance must be at least 30 mm
        clearance = record.new_values["min_clearance_mm"]
        assert clearance >= 30.0

        sc_a = bundle.scene_a_bundle.scene
        sc_b = bundle.scene_b_bundle.scene
        assert len(sc_b.obstacles) == len(sc_a.obstacles) + 1


class TestPairVerificationAndRejection:
    """Tests for strict rejection of invalid counterfactual pairs."""

    def test_reject_mismatched_flipped_relationship(self, factory: CounterfactualFactory):
        # Two identical visible scenes claiming to be FLIPPED_OUTCOME must raise ValueError
        sc_a = factory.scene_generator.generate_scene(seed=101, scene_id="s1")
        sc_b = sc_a.model_copy(deep=True, update={"scene_id": "s2"})

        with pytest.raises(ValueError, match="failed FLIPPED_OUTCOME verification"):
            factory._build_pair_bundle(
                pair_id="bad_pair_1",
                pair_type=CounterfactualPairType.MIRROR_ANGLE_FLIP,
                parent_scene_id="parent_01",
                scene_a=sc_a,
                scene_b=sc_b,
                theme_a=RenderTheme.DARK_NAVY,
                theme_b=RenderTheme.DARK_NAVY,
                changed_parameter="test",
                original_values={},
                new_values={},
                expected_relationship=ExpectedPairRelationship.FLIPPED_OUTCOME,
                difference_summary="Invalid test",
            )


class TestPairArtifactBundlePersistence:
    """Tests for artifact generation and manifest persistence on disk."""

    def test_pair_bundle_directory_artifacts(self, factory: CounterfactualFactory, tmp_path: Path):
        pair_dir = tmp_path / "pairs"
        factory.create_mirror_angle_flip_pair(
            seed=40002,
            pair_id="pair_persist_01",
            output_dir=pair_dir,
        )

        p_dir = pair_dir / "pair_persist_01"
        assert p_dir.is_dir()
        assert (p_dir / "pair.json").is_file()

        # Check scene_a artifacts
        dir_a = p_dir / "scene_a"
        assert (dir_a / "scene.json").is_file()
        assert (dir_a / "diagram_blind.png").is_file()
        assert (dir_a / "question.json").is_file()
        assert (dir_a / "ground_truth.json").is_file()
        assert (dir_a / "diagram_overlay.png").is_file()

        # Check scene_b artifacts
        dir_b = p_dir / "scene_b"
        assert (dir_b / "scene.json").is_file()
        assert (dir_b / "diagram_blind.png").is_file()
        assert (dir_b / "question.json").is_file()
        assert (dir_b / "ground_truth.json").is_file()
        assert (dir_b / "diagram_overlay.png").is_file()

        # Verify pair.json content
        with open(p_dir / "pair.json", encoding="utf-8") as f:
            data = json.load(f)
        assert data["pair_id"] == "pair_persist_01"
        assert data["pair_type"] == "MIRROR_ANGLE_FLIP"
        assert data["expected_relationship"] == "FLIPPED_OUTCOME"

    def test_deterministic_reproducibility(self):
        """Verifies that running the same seed produces identical pair records across instances."""
        f1 = CounterfactualFactory(canvas_width=1000.0, canvas_height=1000.0, image_size=512)
        f2 = CounterfactualFactory(canvas_width=1000.0, canvas_height=1000.0, image_size=512)
        b1 = f1.create_mirror_angle_flip_pair(seed=45555, pair_id="rep_01")
        b2 = f2.create_mirror_angle_flip_pair(seed=45555, pair_id="rep_01")

        assert b1.pair_record.new_values == b2.pair_record.new_values
        assert b1.scene_a_bundle.scene.model_dump() == b2.scene_a_bundle.scene.model_dump()
        assert b1.scene_b_bundle.scene.model_dump() == b2.scene_b_bundle.scene.model_dump()

    def test_generate_dataset_manifests(self, factory: CounterfactualFactory, tmp_path: Path):
        """Verifies small batch dataset generation and JSONL manifests."""
        out_dir = tmp_path / "cf_batch"
        bundles = factory.generate_dataset(num_pairs_per_type=2, output_dir=out_dir, start_seed=50000)

        assert len(bundles) == 8  # 4 types * 2 pairs = 8 pairs
        assert (out_dir / "pairs.jsonl").is_file()
        assert (out_dir / "tasks.jsonl").is_file()
        assert (out_dir / "ground_truth.jsonl").is_file()

        with open(out_dir / "pairs.jsonl", encoding="utf-8") as f:
            pair_lines = [line.strip() for line in f if line.strip()]
        assert len(pair_lines) == 8

        with open(out_dir / "tasks.jsonl", encoding="utf-8") as f:
            task_lines = [line.strip() for line in f if line.strip()]
        assert len(task_lines) == 16  # 8 pairs * 2 scenes = 16 tasks

