"""
Unit and Integration Tests for Benchmark Dataset Auditor & Splitter.

Tests:
1. ID uniqueness detection.
2. Deterministic seed recording checks.
3. Image leakage detection.
4. Prompt neutrality checks.
5. Live audit of generated MIRRORLAB benchmark suites.
6. Benchmark split creation with strict public/private isolation.
"""

import json
from pathlib import Path
from PIL import Image
import pytest

from mirrorlab.evaluation.dataset_auditor import (
    DatasetAuditor,
    DatasetSplitter,
    render_audit_report_markdown,
)


@pytest.fixture
def auditor() -> DatasetAuditor:
    return DatasetAuditor(data_root="data")


class TestAuditorUnitChecks:
    """Unit tests for individual audit verification rules."""

    def test_id_uniqueness_detection(self, auditor: DatasetAuditor):
        # Unique list
        meta_ok = [{"task_id": "t1"}, {"task_id": "t2"}, {"task_id": "t3"}]
        res_ok = auditor.check_id_uniqueness(meta_ok)
        assert res_ok.passed is True
        assert res_ok.items_checked == 3

        # Duplicate list
        meta_dup = [{"task_id": "t1"}, {"task_id": "t2"}, {"task_id": "t1"}]
        res_dup = auditor.check_id_uniqueness(meta_dup)
        assert res_dup.passed is False
        assert len(res_dup.failures) == 1

    def test_seed_recording_checks(self, auditor: DatasetAuditor):
        meta_ok = [{"task_id": "t1", "seed": 100}, {"task_id": "t2", "seed": 200}]
        assert auditor.check_seed_recording(meta_ok).passed is True

        meta_bad = [{"task_id": "t1", "seed": None}, {"task_id": "t2", "seed": -5}]
        res_bad = auditor.check_seed_recording(meta_bad)
        assert res_bad.passed is False
        assert len(res_bad.failures) == 2

    def test_image_leakage_detection(self, auditor: DatasetAuditor, tmp_path: Path):
        # Create identical images -> must be flagged as leakage
        img1 = Image.new("RGB", (64, 64), color=(30, 40, 50))
        p1 = tmp_path / "blind.png"
        p2 = tmp_path / "overlay.png"
        img1.save(p1)
        img1.save(p2)

        passed, msg = auditor.check_image_leakage(p1, p2)
        assert passed is False
        assert "identical" in msg

        # Create distinct images -> must pass
        img2 = Image.new("RGB", (64, 64), color=(255, 200, 0))
        img2.save(p2)
        passed2, msg2 = auditor.check_image_leakage(p1, p2)
        assert passed2 is True
        assert "Clean separation" in msg2

    def test_prompt_neutrality_checks(self, auditor: DatasetAuditor):
        # Neutral prompt
        p_ok = "Can the camera see the scratch? Respond with YES or NO."
        q_ok = "Is it visible?"
        ok, _ = auditor.check_prompt_neutrality(p_ok, q_ok, "YES")
        assert ok is True

        # Leaked prompt
        p_leak = "Notice that the answer is YES because mirror reflects light."
        ok_leak, msg = auditor.check_prompt_neutrality(p_leak, q_ok, "YES")
        assert ok_leak is False
        assert "Leaked phrase detected" in msg


class TestLiveDatasetAudit:
    """Audits the actual generated benchmark suites in data/."""

    def test_audit_all_generated_suites(self, auditor: DatasetAuditor):
        report = auditor.audit_all()
        assert report.all_passed is True
        assert report.total_tasks_checked >= 150
        assert len(report.checks) >= 6

        # Check all checks passed individually
        for c in report.checks:
            assert c.passed is True, f"Check '{c.name}' failed: {c.failures}"


class TestDatasetSplitterIsolation:
    """Verifies that test set splits maintain absolute isolation between public and private items."""

    def test_create_and_verify_splits(self, tmp_path: Path):
        splitter = DatasetSplitter(
            data_root="data",
            splits_root=tmp_path / "splits",
            dev_ratio=0.20,
            val_ratio=0.20,
            test_ratio=0.60,
            seed=123,
        )
        summary = splitter.create_splits()

        splits_dir = tmp_path / "splits"
        assert (splits_dir / "manifest.json").is_file()

        # Check Test public manifest exists and has ZERO answer keys
        test_pub_tasks = splits_dir / "test" / "public" / "tasks.jsonl"
        assert test_pub_tasks.is_file()

        with open(test_pub_tasks, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                assert "answer" not in item
                assert "can_see" not in item
                assert "proof_geometry" not in item
                assert "correct_candidate" not in item

        # Check Test private manifest exists and contains ground truth
        test_priv_gt = splits_dir / "test" / "private" / "ground_truth.jsonl"
        assert test_priv_gt.is_file()

        with open(test_priv_gt, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) > 0

        # Verify images and overlays directories
        assert (splits_dir / "test" / "public" / "images").is_dir()
        assert (splits_dir / "test" / "private" / "overlays").is_dir()
