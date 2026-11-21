"""
Unit & Integration Tests for Seeded, Constraint-Based Scene Generator.

Tests:
1. Deterministic Reproducibility: same seed produces identical scenes bit-for-bit.
2. Verified generation of all 6 controlled categories:
   - VALID_VISIBLE
   - MIRROR_MISS
   - OUTSIDE_FOV
   - WRONG_SIDE
   - BLOCKED_INCIDENT
   - BLOCKED_REFLECTED
3. Difficulty presets (EASY, MEDIUM, HARD) scaling parameters.
4. Batch generation of 20 verified sample scenes with balanced distribution.
"""

import pytest

from mirrorlab.generation.scene_generator import SceneGenerator
from mirrorlab.schema.common import ObstructionReason, VisibilityState
from mirrorlab.schema.scene import DifficultyLevel, InspectionScene, SceneCategory


class TestSceneGeneratorReproducibility:
    """Verify bit-for-bit deterministic reproducibility."""

    def test_identical_seed_produces_identical_scene(self):
        gen1 = SceneGenerator()
        gen2 = SceneGenerator()
        scene1 = gen1.generate_scene(seed=12345, category=SceneCategory.VALID_VISIBLE)
        scene2 = gen2.generate_scene(seed=12345, category=SceneCategory.VALID_VISIBLE)

        assert scene1.model_dump() == scene2.model_dump()
        assert scene1.camera.position.x == scene2.camera.position.x
        assert scene1.mirror.endpoint_a.x == scene2.mirror.endpoint_a.x
        assert scene1.target.center.y == scene2.target.center.y

    def test_different_seeds_produce_different_scenes(self):
        gen = SceneGenerator()
        scene1 = gen.generate_scene(seed=111, category=SceneCategory.VALID_VISIBLE)
        scene2 = gen.generate_scene(seed=222, category=SceneCategory.VALID_VISIBLE)

        assert scene1.camera.position != scene2.camera.position


class TestControlledCategoryGeneration:
    """Verify generator accurately produces each of the 6 controlled categories."""

    def test_generate_valid_visible_scene(self):
        gen = SceneGenerator()
        scene = gen.generate_scene(seed=42, category=SceneCategory.VALID_VISIBLE)

        assert scene.category == SceneCategory.VALID_VISIBLE
        assert scene.ground_truth_state == VisibilityState.VISIBLE
        assert scene.ground_truth_reason == ObstructionReason.NONE
        assert scene.is_direct_sight_blocked is True
        assert len(scene.optical_path) == 3

    def test_generate_mirror_miss_scene(self):
        gen = SceneGenerator()
        scene = gen.generate_scene(seed=43, category=SceneCategory.MIRROR_MISS)

        assert scene.category == SceneCategory.MIRROR_MISS
        assert scene.ground_truth_state == VisibilityState.NOT_VISIBLE
        assert scene.ground_truth_reason == ObstructionReason.MISSED_MIRROR

    def test_generate_outside_fov_scene(self):
        gen = SceneGenerator()
        scene = gen.generate_scene(seed=44, category=SceneCategory.OUTSIDE_FOV)

        assert scene.category == SceneCategory.OUTSIDE_FOV
        assert scene.ground_truth_state == VisibilityState.NOT_VISIBLE
        assert scene.ground_truth_reason == ObstructionReason.OUT_OF_FOV

    def test_generate_wrong_side_scene(self):
        gen = SceneGenerator()
        scene = gen.generate_scene(seed=45, category=SceneCategory.WRONG_SIDE)

        assert scene.category == SceneCategory.WRONG_SIDE
        assert scene.ground_truth_state == VisibilityState.NOT_VISIBLE
        assert scene.ground_truth_reason == ObstructionReason.BACK_OF_MIRROR

    def test_generate_blocked_incident_scene(self):
        gen = SceneGenerator()
        scene = gen.generate_scene(seed=46, category=SceneCategory.BLOCKED_INCIDENT)

        assert scene.category == SceneCategory.BLOCKED_INCIDENT
        assert scene.ground_truth_state == VisibilityState.NOT_VISIBLE
        assert scene.ground_truth_reason == ObstructionReason.BLOCKED_INCIDENT

    def test_generate_blocked_reflected_scene(self):
        gen = SceneGenerator()
        scene = gen.generate_scene(seed=47, category=SceneCategory.BLOCKED_REFLECTED)

        assert scene.category == SceneCategory.BLOCKED_REFLECTED
        assert scene.ground_truth_state == VisibilityState.NOT_VISIBLE
        assert scene.ground_truth_reason == ObstructionReason.BLOCKED_REFLECTED


class TestDifficultyTiers:
    """Verify difficulty presets generate corresponding geometry scales."""

    @pytest.mark.parametrize(
        "diff", [DifficultyLevel.EASY, DifficultyLevel.STANDARD, DifficultyLevel.MEDIUM, DifficultyLevel.HARD, DifficultyLevel.BOUNDARY]
    )
    def test_difficulty_tier_generation(self, diff):
        gen = SceneGenerator()
        scene = gen.generate_scene(
            seed=999, category=SceneCategory.VALID_VISIBLE, difficulty=diff
        )

        assert scene.difficulty == diff
        assert scene.ground_truth_state == VisibilityState.VISIBLE

        # Mirror length check matches tier
        m_len = scene.mirror.endpoint_a.distance_to(scene.mirror.endpoint_b)
        if diff == DifficultyLevel.EASY:
            assert 210.0 <= m_len <= 310.0
            assert scene.is_boundary is False
        elif diff == DifficultyLevel.HARD:
            assert 70.0 <= m_len <= 140.0
            assert scene.is_boundary is False
        elif diff == DifficultyLevel.BOUNDARY:
            assert scene.is_boundary is True

    def test_anti_duplicate_geometry_enforcement(self):
        """Verify generator tracks fingerprints and rejects duplicate geometry."""
        gen = SceneGenerator()
        scenes = gen.generate_batch(count=30, start_seed=12000, difficulty=DifficultyLevel.STANDARD)
        assert len(scenes) == 30
        assert len(gen.seen_fingerprints) == 30  # All 30 are geometrically unique


class TestBatchGeneration:
    """Verify balanced batch generation of 20 sample scenes."""

    def test_generate_20_balanced_scenes(self):
        gen = SceneGenerator()
        batch = gen.generate_batch(count=20, start_seed=5000)

        assert len(batch) == 20
        assert all(isinstance(s, InspectionScene) for s in batch)

        # Count categories
        cat_counts = {}
        for s in batch:
            cat_counts[s.category] = cat_counts.get(s.category, 0) + 1

        # Every category should be represented in 20 samples
        assert len(cat_counts) == 6
        assert cat_counts[SceneCategory.VALID_VISIBLE] >= 3
        assert cat_counts[SceneCategory.MIRROR_MISS] >= 3
        assert cat_counts[SceneCategory.BLOCKED_INCIDENT] >= 3


class TestDatasetStatisticsAndCoverageReporting:
    """Verify statistical analysis and distribution report generation."""

    def test_compute_dataset_statistics(self):
        from mirrorlab.reporting.dataset_stats import compute_dataset_statistics, render_distribution_markdown

        gen = SceneGenerator()
        scenes = []
        for d in [DifficultyLevel.EASY, DifficultyLevel.STANDARD, DifficultyLevel.HARD, DifficultyLevel.BOUNDARY]:
            batch = gen.generate_batch(count=12, start_seed=20000 + hash(d.value) % 1000, difficulty=d)
            scenes.extend(batch)

        report = compute_dataset_statistics(scenes)

        assert report.total_scenes == 48
        assert report.duplicate_count == 0
        assert report.unique_fingerprints == 48

        # Verify difficulty breakdown
        assert "EASY" in report.difficulty_counts
        assert "STANDARD" in report.difficulty_counts
        assert "HARD" in report.difficulty_counts
        assert "BOUNDARY" in report.difficulty_counts

        # Verify boundary separation
        assert report.boundary_count == 12
        assert report.headline_count == 36
        assert report.boundary_pct == 25.0
        assert report.headline_pct == 75.0

        # Verify geometric metric stats exist and are positive
        assert report.mirror_length_stats.mean_val > 0.0
        assert report.fov_stats.mean_val > 0.0
        assert report.obstacle_count_stats.mean_val >= 1.0

        # Verify markdown rendering
        md = render_distribution_markdown(report)
        assert "# MIRRORLAB Dataset Distribution & Coverage Report" in md
        assert "Headline Benchmark" in md
        assert "Boundary Stress Test" in md

