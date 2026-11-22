"""
Regression tests for Second Independent Audit Remediation.

Tests:
1. MockVisionAdapter with ORACLE strategy requires explicit task_id and fails without it.
2. MockVisionAdapter fails when task_id is not in ground_truth_map.
3. MockVisionAdapter fails when ground truth is missing required fields.
4. MockVisionAdapter does not silently fall back to stale _current ground truth.
5. Sequential queries with different ground truths yield distinct, correct responses.
6. DatasetAuditor catches deliberately injected badge overlays in blind images.
7. DatasetAuditor catches private fields leaked in public task records.
8. DatasetAuditor catches invalid or missing seeds in counterfactual child scenes.
9. load_split_records successfully loads test split public tasks and private ground truth.
10. OOD dataset exhibits balanced categories and zero all-NO distribution.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mirrorlab.evaluation.dataset_auditor import DatasetAuditor
from mirrorlab.evaluation.model_adapter import (
    MockStrategy,
    MockVisionAdapter,
    ModelSettings,
)
from mirrorlab.schema.common import TaskType
from mirrorlab.ui.data_loader import load_split_records


class TestOracleMockStrictness:
    """Verifies that MockVisionAdapter with ORACLE strategy never silently falls back or guesses."""

    @pytest.fixture
    def dummy_image(self, tmp_path: Path) -> Path:
        img_path = tmp_path / "test_blind.png"
        img = Image.new("RGB", (100, 100), color=(10, 22, 40))
        img.save(img_path)
        return img_path

    def test_missing_task_id_raises_value_error(self, dummy_image: Path):
        """Oracle strategy must raise ValueError when task_id is omitted or empty."""
        adapter = MockVisionAdapter(
            model_name="mock-oracle",
            settings=ModelSettings(),
            strategy=MockStrategy.ORACLE,
            ground_truth_map={"task_001": {"can_see": True, "answer": "YES"}},
        )
        with pytest.raises(ValueError, match="task_id is required"):
            adapter.ask(dummy_image, "Can the camera see the scratch?", task_type="TASK_A_VISIBILITY")

    def test_unknown_task_id_raises_key_error(self, dummy_image: Path):
        """Oracle strategy must raise KeyError when task_id is not in ground_truth_map."""
        adapter = MockVisionAdapter(
            model_name="mock-oracle",
            settings=ModelSettings(),
            strategy=MockStrategy.ORACLE,
            ground_truth_map={"task_001": {"can_see": True, "answer": "YES"}},
        )
        with pytest.raises(KeyError, match="Task-specific ground truth not found for task_id='task_999'"):
            adapter.ask(
                dummy_image,
                "Can the camera see the scratch?",
                task_id="task_999",
                task_type="TASK_A_VISIBILITY",
            )

    def test_no_fallback_to_stale_current(self, dummy_image: Path):
        """Oracle strategy must NOT fall back to _current when querying an unregistered task_id."""
        adapter = MockVisionAdapter(
            model_name="mock-oracle",
            settings=ModelSettings(),
            strategy=MockStrategy.ORACLE,
            ground_truth_map={
                "_current": {"can_see": True, "answer": "YES"},
                "task_001": {"can_see": False, "answer": "NO"},
            },
        )
        with pytest.raises(KeyError, match="task_id='task_unregistered'"):
            adapter.ask(
                dummy_image,
                "Can the camera see the scratch?",
                task_id="task_unregistered",
                task_type="TASK_A_VISIBILITY",
            )

    def test_missing_required_fields_raises_value_error(self, dummy_image: Path):
        """Oracle strategy must raise ValueError when ground truth lacks required visibility fields."""
        adapter = MockVisionAdapter(
            model_name="mock-oracle",
            settings=ModelSettings(),
            strategy=MockStrategy.ORACLE,
            ground_truth_map={"task_bad": {"notes": "no can_see or is_visible field"}},
        )
        with pytest.raises(ValueError, match="missing required visibility field"):
            adapter.ask(
                dummy_image,
                "Can the camera see the scratch?",
                task_id="task_bad",
                task_type="TASK_A_VISIBILITY",
            )

    def test_sequential_queries_with_different_ground_truths(self, dummy_image: Path):
        """Sequential requests return the exact ground truth for each specific task ID."""
        gt_map = {
            "task_visible": {"can_see": True, "answer": "YES", "reason_code": "CLEAR"},
            "task_blocked": {"can_see": False, "answer": "NO", "reason_code": "BLOCKED_INCIDENT"},
            "task_repair": {"correct_candidate": "B", "selection_reason": "Candidate B restores view."},
        }
        adapter = MockVisionAdapter(
            model_name="mock-oracle",
            settings=ModelSettings(),
            strategy=MockStrategy.ORACLE,
            ground_truth_map=gt_map,
        )

        resp1 = adapter.ask(dummy_image, "Q1", task_id="task_visible", task_type=TaskType.TASK_A_VISIBILITY)
        data1 = json.loads(resp1.raw_text)
        assert data1["can_see"] is True
        assert data1["answer"] == "YES"

        resp2 = adapter.ask(dummy_image, "Q2", task_id="task_blocked", task_type=TaskType.TASK_A_VISIBILITY)
        data2 = json.loads(resp2.raw_text)
        assert data2["can_see"] is False
        assert data2["answer"] == "NO"
        assert data2["reason"] == "BLOCKED_CAMERA_TO_MIRROR"

        resp3 = adapter.ask(dummy_image, "Q3", task_id="task_repair", task_type=TaskType.TASK_C_REPAIR)
        data3 = json.loads(resp3.raw_text)
        assert data3["selected_candidate"] == "B"
        assert data3["answer"] == "B"


class TestLeakageAndValidationAuditor:
    """Verifies that DatasetAuditor detects deliberate leakages and invalid data."""

    def test_blind_image_with_deliberately_injected_badge_fails_audit(self, tmp_path: Path):
        """Auditor must catch and reject a blind image that contains answer badge colors."""
        blind_file = tmp_path / "diagram_blind.png"
        overlay_file = tmp_path / "diagram_overlay.png"

        # Create base blind image
        img_b = Image.new("RGB", (1024, 1024), color=(10, 22, 40))
        # Deliberately paint a green visible badge into the top-right corner of the blind image
        arr_b = np.array(img_b)
        # Badge coordinates: bx = 1024 - 335 = 689, by = 73, bw = 270, bh = 38
        arr_b[73 : 73 + 38, 689 : 689 + 270] = [16, 185, 129]  # Emerald green badge
        Image.fromarray(arr_b).save(blind_file)

        # Create overlay
        img_o = Image.new("RGB", (1024, 1024), color=(10, 22, 40))
        arr_o = np.array(img_o)
        arr_o[73 : 73 + 38, 689 : 689 + 270] = [16, 185, 129]
        arr_o[200:300, 200:300] = [255, 180, 0]  # Ray
        Image.fromarray(arr_o).save(overlay_file)

        auditor = DatasetAuditor()
        ok, msg = auditor.check_image_leakage(blind_file, overlay_file)
        assert ok is False
        assert "badge" in msg.lower() or "colors detected" in msg.lower()

    def test_public_record_non_leakage_detects_forbidden_fields(self):
        """check_public_record_non_leakage must flag private fields in public tasks."""
        clean_record = {
            "task_id": "task_001",
            "question": "Is the scratch visible in the mirror?",
            "image": "diagram_blind.png",
        }
        ok, leaks = DatasetAuditor.check_public_record_non_leakage(clean_record)
        assert ok is True
        assert len(leaks) == 0

        contaminated_record = {
            "task_id": "task_002",
            "question": "Is the scratch visible in the mirror?",
            "can_see": True,
            "ground_truth_reason": "CLEAR",
            "image": "diagram_overlay.png",
        }
        ok2, leaks2 = DatasetAuditor.check_public_record_non_leakage(contaminated_record)
        assert ok2 is False
        assert any("can_see" in lk for lk in leaks2)
        assert any("overlay" in lk for lk in leaks2)

    def test_counterfactual_child_seeds_validated_without_defaults(self, tmp_path: Path):
        """Auditor must fail when child scenes lack valid seeds."""
        pair_dir = tmp_path / "pair_test_000"
        pair_dir.mkdir(parents=True)
        (pair_dir / "scene_a").mkdir()
        (pair_dir / "scene_b").mkdir()

        # Missing seed in scene_b
        with open(pair_dir / "scene_a" / "scene.json", "w", encoding="utf-8") as f:
            json.dump({"scene_id": "scene_a", "seed": 50000}, f)
        with open(pair_dir / "scene_b" / "scene.json", "w", encoding="utf-8") as f:
            json.dump({"scene_id": "scene_b"}, f)  # Missing seed!

        with open(pair_dir / "pair.json", "w", encoding="utf-8") as f:
            json.dump({
                "pair_id": "pair_test_000",
                "expected_relationship": "FLIPPED_OUTCOME",
                "child_scene_ids": ["scene_a", "scene_b"],
                "ground_truth_outcomes": {},
            }, f)

        # Artifacts
        for sub in ["scene_a", "scene_b"]:
            for req in ["diagram_blind.png", "question.json", "ground_truth.json", "diagram_overlay.png"]:
                fpath = pair_dir / sub / req
                fpath.write_text("{}", encoding="utf-8")

        auditor = DatasetAuditor(data_root=tmp_path)
        checks = auditor.audit_counterfactual_pairs(tmp_path)
        seed_check = next((c for c in checks if "Child Scene Seed Validity" in c.name), None)
        assert seed_check is not None
        assert seed_check.passed is False
        assert any("scene_b missing valid integer seed" in f for f in seed_check.failures)


class TestSplitRecordsLoading:
    """Verifies that load_split_records loads test split public and private files correctly."""

    def test_load_split_records_test_partition(self):
        """load_split_records('test') loads tasks from public/ and ground truth from private/."""
        data = load_split_records(split_name="test")
        tasks = data.get("tasks", [])
        gt = data.get("ground_truth", [])

        assert len(tasks) > 0, "Test tasks should be loaded from public/tasks.jsonl"
        assert len(gt) > 0, "Test ground truth should be loaded from private/ground_truth.jsonl"
        assert len(tasks) == len(gt)


class TestOODDistributionCertification:
    """Verifies that the generated OOD dataset has a balanced distribution and 100% certification."""

    def test_ood_manifest_category_distribution_and_balance(self):
        """OOD dataset must contain varied categories (not all NO)."""
        manifest_path = Path("data/ood_tasks/ood_manifest.json")
        assert manifest_path.is_file(), "data/ood_tasks/ood_manifest.json must exist"

        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest.get("all_records_certified") is True
        cat_dist = manifest.get("category_distribution", {})

        # Must have at least 5 visible cases
        assert cat_dist.get("VISIBLE", 0) >= 5, "OOD suite must contain at least 5 visible cases"
        # Must have multiple failure reasons represented
        assert len(cat_dist) >= 4, f"OOD suite must have at least 4 categories, got {len(cat_dist)}"
        # Verify total count equals 30
        assert sum(cat_dist.values()) == 30
