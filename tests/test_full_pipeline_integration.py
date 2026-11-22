"""
Full End-to-End Pipeline Integration & Integrity Tests for MIRRORLAB.

Tests:
1. End-to-end task generation, independent auditing, and split partitioning across all suites.
2. Independent obstruction blocker identification and clearance recomputation.
3. Repair candidate multi-winner / zero-winner rejection.
4. Counterfactual child-scene invariant verification.
5. Strict dataset split isolation: public/private separation, 1:1 ID alignment, and duplicate rejection.
6. Deliberate leakage / corruption detection:
   - Badge color injection into blind images.
   - Leaked private fields in public task records.
   - Ground truth tampering.
   - Split file deletion.
"""

from __future__ import annotations

import json
from pathlib import Path

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
from mirrorlab.schema.scene import DifficultyLevel, SceneCategory


@pytest.fixture
def fresh_data_pipeline(tmp_path: Path) -> Path:
    """
    Constructs a fully populated, valid, synthetic benchmark pipeline from scratch in tmp_path.
    Includes Task A, Task B, Task C, Task D (counterfactual pairs), and Task E (OOD suite).
    """
    data_dir = tmp_path / "data"

    # 1. Task A: Visibility Tasks
    task_a_dir = data_dir / "benchmark_tasks"
    task_a_factory = VisibilityTaskFactory()
    for idx in range(3):
        task_id = f"task_vis_100{idx}"
        cat = SceneCategory.VALID_VISIBLE if idx % 2 == 0 else SceneCategory.BLOCKED_INCIDENT
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
    for idx, seg in enumerate([BlockedSegment.CAMERA_TO_MIRROR, BlockedSegment.MIRROR_TO_TARGET, BlockedSegment.CAMERA_TO_MIRROR]):
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

    # 4. Task D: Counterfactual Pairs
    task_d_dir = data_dir / "counterfactual_pairs"
    pair_factory = CounterfactualFactory()
    pair_factory.create_mirror_angle_flip_pair(
        seed=401,
        pair_id="pair_test_000",
        output_dir=task_d_dir,
    )
    pair_factory.create_cosmetic_control_pair(
        seed=402,
        pair_id="pair_test_001",
        output_dir=task_d_dir,
    )

    # 5. Task E: OOD Tasks (6 tasks, 1 per category)
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

        # 2. Audit all tasks
        auditor = DatasetAuditor(data_root=data_root)
        report = auditor.audit_all()

        assert report.total_tasks_checked > 0
        assert report.total_scenes_checked > 0
        assert report.total_pairs_checked == 2
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


class TestPipelineCorruptionDetection:
    """Verifies that the independent auditor detects deliberate corruptions and security leaks."""

    def test_catches_corrupted_ground_truth_visibility(self, fresh_data_pipeline: Path):
        """Auditor must catch tampered visibility label in Task A."""
        data_root = fresh_data_pipeline
        t_a_dir = next((data_root / "benchmark_tasks").iterdir())
        gt_file = t_a_dir / "ground_truth.json"

        with open(gt_file, encoding="utf-8") as f:
            gt = json.load(f)
        # Flip ground truth label
        vis_key = "can_see" if "can_see" in gt else "is_visible"
        gt[vis_key] = not gt[vis_key]
        with open(gt_file, "w", encoding="utf-8") as f:
            json.dump(gt, f)

        auditor = DatasetAuditor(data_root=data_root)
        a_checks = auditor.audit_visibility_tasks(data_root / "benchmark_tasks")
        math_check = next(c for c in a_checks if "Oracle" in c.name)
        assert math_check.passed is False
        assert len(math_check.failures) > 0
        assert any("oracle=" in fail.lower() for fail in math_check.failures)

    def test_catches_private_field_leak_in_test_public(self, fresh_data_pipeline: Path):
        """Auditor must fail if test public tasks.jsonl leaks private fields."""
        data_root = fresh_data_pipeline
        splits_root = data_root / "benchmark_splits"
        splitter = DatasetSplitter(data_root=data_root, splits_root=splits_root)
        splitter.create_splits()

        test_pub_file = splits_root / "test" / "public" / "tasks.jsonl"
        with open(test_pub_file, encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        # Inject private ground truth field into public file
        lines[0]["answer"] = "YES"
        with open(test_pub_file, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(item) + "\n" for item in lines)

        auditor = DatasetAuditor(data_root=data_root)
        iso_res = auditor.check_dataset_split_isolation()
        assert iso_res.passed is False
        assert any("answer" in fail.lower() for fail in iso_res.failures)

    def test_catches_missing_split_file(self, fresh_data_pipeline: Path):
        """Auditor must fail if any required split file is missing."""
        data_root = fresh_data_pipeline
        splits_root = data_root / "benchmark_splits"
        splitter = DatasetSplitter(data_root=data_root, splits_root=splits_root)
        splitter.create_splits()

        # Delete private ground truth file
        priv_gt = splits_root / "test" / "private" / "ground_truth.jsonl"
        priv_gt.unlink()

        auditor = DatasetAuditor(data_root=data_root)
        iso_res = auditor.check_dataset_split_isolation()
        assert iso_res.passed is False
        assert any("split file missing" in fail.lower() for fail in iso_res.failures)

    def test_catches_badge_color_leakage_in_blind_image(self, fresh_data_pipeline: Path):
        """Auditor must detect answer badges injected into blind images."""
        data_root = fresh_data_pipeline
        t_a_dir = next((data_root / "benchmark_tasks").iterdir())
        blind_img_path = t_a_dir / "diagram_blind.png"

        img = Image.open(blind_img_path).convert("RGB")
        w, _h = img.size
        # Inject an emerald green badge in the top-right badge box
        scale = w / 1024.0
        bx = int(w - (65.0 + 270.0) * scale)
        by = int((65.0 + 8.0) * scale)
        pixels = img.load()
        for x in range(bx + 10, bx + 150):
            for y in range(by + 10, by + 30):
                pixels[x, y] = (46, 204, 113)  # Bright green
        img.save(blind_img_path)

        auditor = DatasetAuditor(data_root=data_root)
        ok, leak = auditor.check_image_leakage(blind_img_path, t_a_dir / "diagram_overlay.png")
        assert ok is False
        assert "Answer badge colors detected" in leak

    def test_catches_semantic_answer_leak_in_task_id(self, fresh_data_pipeline: Path):
        """Auditor must fail if public task ID leaks category outcome tokens."""
        clean_rec = {"task_id": "ml_test_0001", "image_filename": "img_test_0001.png"}
        ok, _ = DatasetAuditor.check_public_record_non_leakage(clean_rec)
        assert ok is True

        leak_ids = [
            "task_vis_10000_valid_visible",
            "task_vis_10003_outside_fov",
            "task_vis_10005_wrong_side",
            "task_obs_25006_camera_to_mirror",
            "task_obs_25029_none",
            "pair_mirror_flip_022_a",
            "pair_cosmetic_001_a",
            "pair_obs_insert_001_b",
        ]
        for lid in leak_ids:
            bad_rec = {"task_id": lid, "image_filename": "img_001.png"}
            is_clean, reasons = DatasetAuditor.check_public_record_non_leakage(bad_rec)
            assert is_clean is False, f"Failed to catch leak in {lid}"
            assert any("semantic answer pattern" in r.lower() or "delimited outcome token" in r.lower() for r in reasons)

    def test_catches_semantic_answer_leak_in_image_filename(self):
        """Auditor must fail if public image filename leaks outcome tokens."""
        leak_imgs = [
            "task_vis_valid_visible.png",
            "img_test_outside_fov.png",
            "pair_mirror_flip_001.png",
            "pair_cosmetic_b.png",
            "candidate_winner_a.png",
        ]
        for limg in leak_imgs:
            bad_rec = {"task_id": "ml_test_0042", "image_filename": limg}
            is_clean, _reasons = DatasetAuditor.check_public_record_non_leakage(bad_rec)
            assert is_clean is False, f"Failed to catch leak in {limg}"

    def test_blind_image_verification_never_fails_open(self, fresh_data_pipeline: Path):
        """Auditor must never silently pass image verification when corrupted or missing."""
        data_root = fresh_data_pipeline
        t_a_dir = next((data_root / "benchmark_tasks").iterdir())
        blind_path = t_a_dir / "diagram_blind.png"
        overlay_path = t_a_dir / "diagram_overlay.png"

        # Corrupt blind image
        with open(blind_path, "wb") as f:
            f.write(b"NOT_A_VALID_PNG_FILE_DATA")

        auditor = DatasetAuditor(data_root=data_root)
        ok, msg = auditor.check_image_leakage(blind_path, overlay_path)
        assert ok is False
        assert "image" in msg.lower() or "error" in msg.lower()

    def test_isolation_verified_reflects_actual_validation(self, fresh_data_pipeline: Path):
        """isolation_verified in manifest.json must only be True when all artifacts pass independent validation."""
        data_root = fresh_data_pipeline
        splits_root = data_root / "benchmark_splits"
        splitter = DatasetSplitter(data_root=data_root, splits_root=splits_root)
        manifest = splitter.create_splits()
        assert manifest["isolation_verified"] is True

        # Tamper with public file: inject semantic leakage
        test_pub = splits_root / "test" / "public" / "tasks.jsonl"
        with open(test_pub, encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        lines[0]["task_id"] = "ml_test_0001_valid_visible"
        with open(test_pub, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(l) + "\n" for l in lines)

        auditor = DatasetAuditor(data_root=data_root)
        iso_res = auditor.check_dataset_split_isolation()
        assert iso_res.passed is False
