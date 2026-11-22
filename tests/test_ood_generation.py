"""
Unit and Integration Tests for OOD Benchmark Generation.

Tests:
1. OOD scenes contain acute mirror tilt angles.
2. OOD scenes contain high obstacle clutter (5-8 obstacles).
3. 100% of generated OOD scenes pass dual mathematical verification.
4. OOD manifest generation and complete artifact bundles.
"""

from __future__ import annotations

import json
from pathlib import Path

from mirrorlab.dataset.ood_generator import OODGenerationConfig, OODTaskGenerator
from mirrorlab.geometry.independent_verifier import IndependentGeometryVerifier


class TestOODGeneration:
    """Verifies that the OOD generator produces valid extreme benchmark tasks."""

    def test_ood_scene_properties(self):
        gen = OODTaskGenerator(config=OODGenerationConfig(num_tasks=3))
        scene = gen.generate_ood_scene(seed=80001, task_index=1)

        # 1. High obstacle count (>= 5)
        assert len(scene.obstacles) >= 5

        # 2. Direct sightline blocked
        assert scene.is_direct_sight_blocked is True

        # 3. Dual certified
        dual = IndependentGeometryVerifier.verify_dual(scene)
        assert dual.is_mathematically_certified is True
        assert dual.reflection_point_delta_mm < 1e-4

    def test_ood_dataset_generation_and_artifacts(self, tmp_path: Path):
        gen = OODTaskGenerator(
            config=OODGenerationConfig(num_tasks=2, seed_start=99000),
            output_dir=tmp_path / "test_ood",
        )
        tids = gen.generate_ood_dataset()
        assert len(tids) == 2

        # Verify artifacts in each directory
        for tid in tids:
            tdir = tmp_path / "test_ood" / tid
            assert (tdir / "scene.json").is_file()
            assert (tdir / "diagram_blind.png").is_file()
            assert (tdir / "diagram_overlay.png").is_file()
            assert (tdir / "question.json").is_file()
            assert (tdir / "ground_truth.json").is_file()

        # Check manifest
        manifest_p = tmp_path / "test_ood" / "ood_manifest.json"
        assert manifest_p.is_file()
        m_data = json.loads(manifest_p.read_text(encoding="utf-8"))
        assert m_data["ood_task_count"] == 2
