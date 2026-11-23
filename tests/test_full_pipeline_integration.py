"""
Full End-to-End Pipeline Integration & Integrity Tests for MIRRORLAB.

Tests:
1. End-to-end task generation, independent auditing, and split partitioning across all suites.
2. Independent obstruction blocker identification and clearance recomputation.
3. Repair candidate multi-winner / zero-winner rejection.
4. Counterfactual child-scene invariant verification.
5. Strict dataset split isolation: public/private separation, 1:1 ID alignment, and duplicate rejection.
6. 14 Deliberate corruption cases:
   (1) delete private pair file,
   (2) delete one child GT record,
   (3) change one pair's child ID,
   (4) duplicate child ID across pairs,
   (5) change flip to invariant without outcome change,
   (6) change invariant to flip without outcome change,
   (7) answer in public task ID,
   (8) answer in image filename,
   (9) answer-revealing sentence in prompt,
   (10) private GT field in public record,
   (11) delete an image,
   (12) duplicate public task ID,
   (13) public task referencing other split's GT,
   (14) manifest count tampering.
7. Determinism and reproducibility:
   - Same-seed pipeline reproducibility (geometry, ground truth, splits).
   - Distinction between semantic, geometric, and pixel-level determinism.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mirrorlab.dataset.ood_generator import OODGenerationConfig, OODTaskGenerator
from mirrorlab.evaluation.dataset_auditor import (
    DatasetAuditor,
    DatasetSplitter,
)
from mirrorlab.generation.counterfactual_factory import (
    CounterfactualFactory,
)
from mirrorlab.generation.obstruction_factory import (
    BlockedSegment,
    ObstructionTaskFactory,
)
from mirrorlab.generation.repair_factory import RepairTaskFactory
from mirrorlab.generation.task_factory import VisibilityTaskFactory
from mirrorlab.rendering.pillow_renderer import PillowDiagramRenderer, RenderMode
from mirrorlab.rendering.themes import RenderTheme
from mirrorlab.schema.scene import DifficultyLevel, SceneCategory


@pytest.fixture
def fresh_data_pipeline(tmp_path: Path) -> Path:
    """
    Constructs a fully populated, valid, synthetic benchmark pipeline from scratch in tmp_path.
    Includes Task A, Task B, Task C, Task D (6 counterfactual pairs), and Task E (OOD suite).
    """
    data_dir = tmp_path / "data"

    # 1. Task A: Visibility Tasks
    task_a_dir = data_dir / "benchmark_tasks"
    task_a_factory = VisibilityTaskFactory()
    for idx in range(3):
        task_id = f"task_vis_100{idx}"
        cat = (
            SceneCategory.VALID_VISIBLE
            if idx % 2 == 0
            else SceneCategory.BLOCKED_INCIDENT
        )
        task_a_factory.create_task(
            seed=100 + idx,
            task_id=task_id,
            category=cat,
            difficulty=DifficultyLevel.EASY,
            output_dir=task_a_dir,
        )

    # 2. Task B: Obstruction Tasks
    task_b_dir = data_dir / "obstruction_tasks"
    task_b_factory = ObstructionTaskFactory()
    for idx, seg in enumerate(
        [
            BlockedSegment.CAMERA_TO_MIRROR,
            BlockedSegment.MIRROR_TO_TARGET,
            BlockedSegment.CAMERA_TO_MIRROR,
        ]
    ):
        task_id = f"task_obs_200{idx}"
        task_b_factory.create_task(
            seed=200 + idx,
            task_id=task_id,
            target_segment=seg,
            difficulty=DifficultyLevel.MEDIUM,
            output_dir=task_b_dir,
        )

    # 3. Task C: Repair Tasks
    task_c_dir = data_dir / "repair_tasks"
    task_c_factory = RepairTaskFactory()
    for idx in range(3):
        task_id = f"task_repair_300{idx}"
        task_c_factory.create_task(
            seed=300 + idx,
            task_id=task_id,
            output_dir=task_c_dir,
        )

    # 4. Task D: Counterfactual Pairs (6 pairs so dev, val, test all get pairs)
    task_d_dir = data_dir / "counterfactual_pairs"
    pair_factory = CounterfactualFactory()
    pair_factory.create_mirror_angle_flip_pair(
        seed=401, pair_id="pair_test_000", output_dir=task_d_dir
    )
    pair_factory.create_cosmetic_control_pair(
        seed=402, pair_id="pair_test_001", output_dir=task_d_dir
    )
    pair_factory.create_obstacle_insertion_pair(
        seed=403, pair_id="pair_test_002", output_dir=task_d_dir
    )
    pair_factory.create_irrelevant_obstacle_pair(
        seed=404, pair_id="pair_test_003", output_dir=task_d_dir
    )
    pair_factory.create_mirror_angle_flip_pair(
        seed=405, pair_id="pair_test_004", output_dir=task_d_dir
    )
    pair_factory.create_cosmetic_control_pair(
        seed=406, pair_id="pair_test_005", output_dir=task_d_dir
    )

    # 5. Task E: OOD Tasks (6 tasks)
    ood_gen = OODTaskGenerator(
        config=OODGenerationConfig(num_tasks=6, seed_start=500),
        output_dir=data_dir / "ood_tasks",
    )
    ood_gen.generate_ood_dataset()

    return data_dir


class TestFullPipelineIntegration:
    """End-to-end integration tests for synthesis, independent auditing, and split isolation."""

    def test_end_to_end_pipeline_and_audit(self, fresh_data_pipeline: Path):
        """Test complete generation, splitting, and independent auditing of all suites."""
        data_root = fresh_data_pipeline
        splits_root = data_root / "benchmark_splits"

        # 1. Create splits
        splitter = DatasetSplitter(
            data_root=data_root,
            splits_root=splits_root,
            dev_ratio=0.33,
            val_ratio=0.33,
            test_ratio=0.34,
            seed=42,
        )
        manifest = splitter.create_splits()
        assert manifest["isolation_verified"] is True
        assert manifest["pair_integrity_verified"] is True
        assert "accounting" in manifest

        # 2. Audit all tasks
        auditor = DatasetAuditor(data_root=data_root)
        report = auditor.audit_all()

        assert report.total_tasks_checked > 0
        assert report.total_scenes_checked > 0
        assert report.total_pairs_checked == 6
        assert report.total_seeds_checked == report.total_scenes_checked
        assert report.total_images_checked > 0
        assert report.total_prompts_checked == report.total_scenes_checked

        for c in report.checks:
            assert c.passed is True, f"Check '{c.name}' failed: {c.failures}"

        assert report.all_passed is True

    def test_split_isolation_and_alignment(self, fresh_data_pipeline: Path):
        """Verify strict 1:1 ID match, non-empty files, and zero leakage in splits."""
        data_root = fresh_data_pipeline
        splits_root = data_root / "benchmark_splits"

        splitter = DatasetSplitter(
            data_root=data_root,
            splits_root=splits_root,
            dev_ratio=0.33,
            val_ratio=0.33,
            test_ratio=0.34,
            seed=42,
        )
        splitter.create_splits()

        auditor = DatasetAuditor(data_root=data_root)
        iso_res = auditor.check_dataset_split_isolation()
        assert iso_res.passed is True, f"Isolation failed: {iso_res.failures}"
        assert iso_res.items_checked > 0

        pair_res = auditor.check_counterfactual_pair_integrity()
        assert pair_res.passed is True, f"Pair integrity failed: {pair_res.failures}"
        assert pair_res.items_checked == 6


class TestPipelineCorruptionDetection:
    """
    Verifies that the independent automated auditor catches all 14 deliberate corruption cases:
    (1) delete private pair file
    (2) delete one child GT record
    (3) change one pair's child ID
    (4) duplicate child ID across pairs
    (5) change flip to invariant without outcome change
    (6) change invariant to flip without outcome change
    (7) answer in public task ID
    (8) answer in image filename
    (9) answer-revealing sentence in prompt
    (10) private GT field in public record
    (11) delete an image
    (12) duplicate public task ID
    (13) public task referencing other split's GT
    (14) manifest count tampering
    """

    @pytest.fixture(autouse=True)
    def setup_splits(self, fresh_data_pipeline: Path):
        self.data_root = fresh_data_pipeline
        self.splits_root = self.data_root / "benchmark_splits"
        splitter = DatasetSplitter(
            data_root=self.data_root,
            splits_root=self.splits_root,
            dev_ratio=0.33,
            val_ratio=0.33,
            test_ratio=0.34,
            seed=42,
        )
        splitter.create_splits()
        self.auditor = DatasetAuditor(data_root=self.data_root)

    def test_corruption_1_delete_private_pair_file(self):
        """(1) Auditor must fail if any private pair file is deleted."""
        p_file = self.splits_root / "test" / "private" / "pairs.jsonl"
        assert p_file.is_file()
        p_file.unlink()

        res = self.auditor.check_counterfactual_pair_integrity()
        assert res.passed is False
        assert any("missing pairs.jsonl" in f.lower() for f in res.failures)

    def test_corruption_2_delete_one_child_gt_record(self):
        """(2) Auditor must fail if one counterfactual child GT record is deleted."""
        gt_file = self.splits_root / "dev" / "ground_truth.jsonl"
        with open(gt_file, encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]

        # Find a counterfactual line and remove it
        cf_idx = next(i for i, l in enumerate(lines) if "TASK_D_COUNTERFACTUAL" in l)
        deleted_tid = json.loads(lines[cf_idx])["task_id"]
        del lines[cf_idx]
        with open(gt_file, "w", encoding="utf-8") as f:
            f.writelines(lines)

        res = self.auditor.check_counterfactual_pair_integrity()
        assert res.passed is False
        assert any(
            f"child id '{deleted_tid}' missing from private ground_truth.jsonl"
            in f.lower()
            for f in res.failures
        )

    def test_corruption_3_change_one_pair_child_id(self):
        """(3) Auditor must fail if a pair's child ID is changed to a nonexistent ID."""
        p_file = self.splits_root / "dev" / "pairs.jsonl"
        with open(p_file, encoding="utf-8") as f:
            recs = [json.loads(line) for line in f if line.strip()]

        recs[0]["child_scene_ids"][0] = "ml_dev_nonexistent_9999"
        with open(p_file, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in recs)

        res = self.auditor.check_counterfactual_pair_integrity()
        assert res.passed is False
        assert any(
            "missing from public tasks.jsonl" in f.lower()
            or "missing from private ground_truth.jsonl" in f.lower()
            for f in res.failures
        )

    def test_corruption_4_duplicate_child_id_across_pairs(self):
        """(4) Auditor must fail if a counterfactual child ID is duplicated across pairs."""
        p_file = self.splits_root / "dev" / "pairs.jsonl"
        with open(p_file, encoding="utf-8") as f:
            recs = [json.loads(line) for line in f if line.strip()]

        assert len(recs) >= 2
        # Set pair 1's first child equal to pair 0's first child
        shared_child = recs[0]["child_scene_ids"][0]
        recs[1]["child_scene_ids"][0] = shared_child
        with open(p_file, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in recs)

        res = self.auditor.check_counterfactual_pair_integrity()
        assert res.passed is False
        assert any("reused in multiple pairs" in f.lower() for f in res.failures)

    def test_corruption_5_change_flip_to_invariant_without_outcome_change(self):
        """(5) Auditor must fail if a FLIP pair is labeled INVARIANT without outcome change."""
        p_file = self.splits_root / "dev" / "pairs.jsonl"
        with open(p_file, encoding="utf-8") as f:
            recs = [json.loads(line) for line in f if line.strip()]

        flip_rec = next(
            r for r in recs if r["expected_relationship"] in {"FLIP", "FLIPPED_OUTCOME"}
        )
        flip_rec["expected_relationship"] = "INVARIANT_OUTCOME"
        with open(p_file, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in recs)

        res = self.auditor.check_counterfactual_pair_integrity()
        assert res.passed is False
        assert any(
            "expected invariant relationship but child outcomes differ" in f.lower()
            for f in res.failures
        )

    def test_corruption_6_change_invariant_to_flip_without_outcome_change(self):
        """(6) Auditor must fail if an INVARIANT pair is labeled FLIP without outcome change."""
        inv_rec = None
        target_p_file = None
        for sname in ["dev", "val", "test"]:
            pf = (
                self.splits_root / sname / "private" / "pairs.jsonl"
                if sname == "test"
                else self.splits_root / sname / "pairs.jsonl"
            )
            if pf.is_file():
                with open(pf, encoding="utf-8") as f:
                    recs = [json.loads(line) for line in f if line.strip()]
                for r in recs:
                    if r["expected_relationship"] in {"INVARIANT", "INVARIANT_OUTCOME"}:
                        inv_rec = r
                        target_p_file = pf
                        break
            if inv_rec:
                break

        assert inv_rec is not None and target_p_file is not None
        inv_rec["expected_relationship"] = "FLIPPED_OUTCOME"
        with open(target_p_file, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in recs)

        res = self.auditor.check_counterfactual_pair_integrity()
        assert res.passed is False
        assert any(
            "expected flip relationship but child outcomes are identical" in f.lower()
            for f in res.failures
        )

    def test_corruption_7_answer_in_public_task_id(self):
        """(7) Auditor must fail if a public task ID contains an answer-revealing category token."""
        t_file = self.splits_root / "test" / "public" / "tasks.jsonl"
        with open(t_file, encoding="utf-8") as f:
            recs = [json.loads(line) for line in f if line.strip()]

        recs[0]["task_id"] = "ml_test_0001_valid_visible"
        with open(t_file, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in recs)

        res = self.auditor.check_dataset_split_isolation()
        assert res.passed is False
        assert any(
            "semantic answer pattern" in f.lower()
            or "delimited outcome token" in f.lower()
            for f in res.failures
        )

    def test_corruption_8_answer_in_image_filename(self):
        """(8) Auditor must fail if a public image filename contains an outcome token."""
        t_file = self.splits_root / "test" / "public" / "tasks.jsonl"
        with open(t_file, encoding="utf-8") as f:
            recs = [json.loads(line) for line in f if line.strip()]

        recs[0]["image_filename"] = "img_test_outside_fov.png"
        with open(t_file, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in recs)

        res = self.auditor.check_dataset_split_isolation()
        assert res.passed is False
        assert any("semantic answer pattern" in f.lower() for f in res.failures)

    def test_corruption_9_answer_revealing_sentence_in_prompt(self):
        """(9) Auditor must fail if question or prompt contains an answer-revealing assertion."""
        # Test acceptance of legitimate phrasing
        legit_p = "You are an inspector. Is the target visible? Choose YES or NO."
        legit_q = "Can the camera see the center of the red X through the mirror?"
        ok_legit, _ = self.auditor.check_prompt_neutrality(legit_p, legit_q, "YES")
        assert ok_legit is True

        # Test rejection of adversarial declarations
        leak_cases = [
            "Notice that the answer is YES because mirror reflects light.",
            "The correct answer is NO.",
            "The scratch is visible from this position.",
            "The target is not visible through the mirror.",
            "Candidate B is the winner.",
            "Obstacle O2 is the blocker.",
        ]
        for leak_text in leak_cases:
            ok_leak, msg = self.auditor.check_prompt_neutrality(
                leak_text, legit_q, "YES"
            )
            assert ok_leak is False, f"Failed to catch leak: '{leak_text}'"
            assert "leaked" in msg.lower()

    def test_corruption_10_private_gt_field_in_public_record(self):
        """(10) Auditor must fail if any private GT field is leaked into public records."""
        t_file = self.splits_root / "test" / "public" / "tasks.jsonl"
        with open(t_file, encoding="utf-8") as f:
            recs = [json.loads(line) for line in f if line.strip()]

        recs[0]["can_see"] = True
        with open(t_file, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in recs)

        res = self.auditor.check_dataset_split_isolation()
        assert res.passed is False
        assert any("private field 'can_see' exposed" in f.lower() for f in res.failures)

    def test_corruption_11_delete_an_image(self):
        """(11) Auditor must fail if any public image file is deleted."""
        img_file = self.splits_root / "test" / "public" / "images" / "img_test_0001.png"
        assert img_file.is_file()
        img_file.unlink()

        res = self.auditor.check_dataset_split_isolation()
        assert res.passed is False
        assert any(
            "public image file does not exist" in f.lower() for f in res.failures
        )

    def test_corruption_12_duplicate_public_task_id(self):
        """(12) Auditor must fail if a public task ID is duplicated within a split."""
        t_file = self.splits_root / "test" / "public" / "tasks.jsonl"
        with open(t_file, encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]

        # Duplicate the first line
        lines.append(lines[0])
        with open(t_file, "w", encoding="utf-8") as f:
            f.writelines(lines)

        res = self.auditor.check_dataset_split_isolation()
        assert res.passed is False
        assert any(
            "duplicate task id in test_public" in f.lower() for f in res.failures
        )

    def test_corruption_13_public_task_referencing_other_split_gt(self):
        """(13) Auditor must fail if a public task references another split's ground-truth ID."""
        t_file = self.splits_root / "val" / "tasks.jsonl"
        with open(t_file, encoding="utf-8") as f:
            recs = [json.loads(line) for line in f if line.strip()]

        recs[0]["task_id"] = "ml_dev_0001"
        with open(t_file, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in recs)

        res = self.auditor.check_dataset_split_isolation()
        assert res.passed is False
        assert any(
            "unaligned ids" in f.lower() or "overlap" in f.lower() for f in res.failures
        )

    def test_corruption_14_manifest_count_tampering(self):
        """(14) Auditor must fail if manifest.json counts are tampered with."""
        m_file = self.splits_root / "manifest.json"
        with open(m_file, encoding="utf-8") as f:
            manifest = json.load(f)

        manifest["model_facing_task_counts"]["test"] = 9999
        with open(m_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        res = self.auditor.check_dataset_split_isolation()
        assert res.passed is False
        assert any(
            "manifest count mismatch" in f.lower()
            or "manifest accounting mismatch" in f.lower()
            for f in res.failures
        )

    def test_catches_badge_color_leakage_in_blind_image(self):
        """Auditor must detect answer badges injected into blind images."""
        t_a_dir = next((self.data_root / "benchmark_tasks").iterdir())
        blind_img_path = t_a_dir / "diagram_blind.png"

        img = Image.open(blind_img_path).convert("RGB")
        w, _h = img.size
        scale = w / 1024.0
        bx = int(w - (65.0 + 270.0) * scale)
        by = int((65.0 + 8.0) * scale)
        pixels = img.load()
        for x in range(bx + 10, bx + 150):
            for y in range(by + 10, by + 30):
                pixels[x, y] = (46, 204, 113)  # Bright green
        img.save(blind_img_path)

        ok, leak = self.auditor.check_image_leakage(
            blind_img_path, t_a_dir / "diagram_overlay.png"
        )
        assert ok is False
        assert "Answer badge colors detected" in leak

    def test_blind_image_verification_never_fails_open(self):
        """Auditor must never silently pass image verification when corrupted or missing."""
        t_a_dir = next((self.data_root / "benchmark_tasks").iterdir())
        blind_path = t_a_dir / "diagram_blind.png"
        overlay_path = t_a_dir / "diagram_overlay.png"

        with open(blind_path, "wb") as f:
            f.write(b"NOT_A_VALID_PNG_FILE_DATA")

        ok, msg = self.auditor.check_image_leakage(blind_path, overlay_path)
        assert ok is False
        assert "image" in msg.lower() or "error" in msg.lower()


class TestDeterminismAndReproducibility:
    """
    Evaluates seed determinism, geometry reproducibility, and distinguishes:
    1. Semantic determinism: identical task questions, options, ground-truth answers, and reason codes.
    2. Geometric determinism: identical coordinates (camera, mirror, target, obstacles) within 1e-9 tolerance.
    3. Pixel-level determinism: identical image rendering hashes from deterministic Pillow rasterization.
    """

    def test_reproducibility_same_seed_produces_identical_dataset(self, tmp_path: Path):
        """Running task synthesis twice with identical seeds yields identical scenes, geometry, and answers."""
        dir1 = tmp_path / "run1"
        dir2 = tmp_path / "run2"

        factory1 = VisibilityTaskFactory()
        factory2 = VisibilityTaskFactory()

        task1 = factory1.create_task(
            seed=7777,
            task_id="task_vis_repro",
            category=SceneCategory.VALID_VISIBLE,
            difficulty=DifficultyLevel.MEDIUM,
            output_dir=dir1,
        )
        task2 = factory2.create_task(
            seed=7777,
            task_id="task_vis_repro",
            category=SceneCategory.VALID_VISIBLE,
            difficulty=DifficultyLevel.MEDIUM,
            output_dir=dir2,
        )

        # 1. Semantic equality
        assert task1.ground_truth.can_see == task2.ground_truth.can_see
        assert task1.ground_truth.answer == task2.ground_truth.answer
        assert task1.ground_truth.reason_code == task2.ground_truth.reason_code

        # 2. Geometric equality within numerical tolerance 1e-9
        sc1 = task1.scene
        sc2 = task2.scene
        assert abs(sc1.camera.position.x - sc2.camera.position.x) < 1e-9
        assert abs(sc1.camera.position.y - sc2.camera.position.y) < 1e-9
        assert abs(sc1.mirror.endpoint_a.x - sc2.mirror.endpoint_a.x) < 1e-9
        assert abs(sc1.mirror.endpoint_b.y - sc2.mirror.endpoint_b.y) < 1e-9
        assert abs(sc1.target.center.x - sc2.target.center.x) < 1e-9

        # Obstacle geometry
        assert len(sc1.obstacles) == len(sc2.obstacles)
        for o1, o2 in zip(sc1.obstacles, sc2.obstacles):
            assert abs(o1.center.x - o2.center.x) < 1e-9
            assert abs(o1.center.y - o2.center.y) < 1e-9

    def test_distinguish_semantic_geometric_pixel_determinism(self, tmp_path: Path):
        """Demonstrates and validates the three tiers of determinism in MIRRORLAB."""
        # Tier 1: Semantic Determinism
        vis_fact1 = VisibilityTaskFactory()
        vis_fact2 = VisibilityTaskFactory()
        t_a = vis_fact1.create_task(
            seed=888, task_id="same_id", output_dir=tmp_path / "a"
        )
        t_b = vis_fact2.create_task(
            seed=888, task_id="same_id", output_dir=tmp_path / "b"
        )
        assert t_a.ground_truth.can_see == t_b.ground_truth.can_see
        assert t_a.ground_truth.reason_code == t_b.ground_truth.reason_code

        # Tier 2: Geometric Determinism (coordinates match exactly)
        coords_a = np.array(
            [
                t_a.scene.camera.position.x,
                t_a.scene.camera.position.y,
                t_a.scene.mirror.endpoint_a.x,
                t_a.scene.mirror.endpoint_a.y,
                t_a.scene.target.center.x,
                t_a.scene.target.center.y,
            ]
        )
        coords_b = np.array(
            [
                t_b.scene.camera.position.x,
                t_b.scene.camera.position.y,
                t_b.scene.mirror.endpoint_a.x,
                t_b.scene.mirror.endpoint_a.y,
                t_b.scene.target.center.x,
                t_b.scene.target.center.y,
            ]
        )
        np.testing.assert_allclose(coords_a, coords_b, atol=1e-9)

        # Tier 3: Pixel-Level Determinism (rendering same scene with same renderer yields zero diff)
        renderer = PillowDiagramRenderer(
            theme=RenderTheme.DARK_NAVY, image_width=512, image_height=512
        )
        img1 = renderer.render(t_a.scene, mode=RenderMode.BLIND)
        img2 = renderer.render(t_b.scene, mode=RenderMode.BLIND)

        arr1 = np.asarray(img1.convert("RGB"))
        arr2 = np.asarray(img2.convert("RGB"))
        assert np.array_equal(arr1, arr2), (
            "Pillow rasterization must produce identical pixels for identical geometry"
        )
