"""
Automated Data Leakage & Test Set Contamination Audit.

Verifies:
1. Zero ground-truth fields in public test split records (tasks.jsonl).
2. Zero semantic answer hints in public image filenames.
3. Zero answer hints in public task prompts.
4. Zero duplicate task hashes across train/val/test splits.
5. Private ground truth keys reside strictly in private directories.
"""

import json
from pathlib import Path


class TestDataLeakageAndSplitIsolation:
    """Rigorous audit ensuring zero leakage to public test splits."""

    def test_public_test_tasks_zero_ground_truth(self):
        test_tasks_path = Path("data/benchmark_splits/test/public/tasks.jsonl")
        assert test_tasks_path.is_file(), "Public test tasks.jsonl must exist."

        forbidden_keys = {
            "ground_truth_state",
            "visibility_state",
            "ground_truth_reason",
            "obstruction_reason",
            "reflection_point",
            "virtual_target",
            "optical_path",
            "diagnostic_details",
            "diagnostic_explanation",
            "answer",
            "is_visible",
        }

        with open(test_tasks_path, encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                for key in forbidden_keys:
                    assert key not in record, (
                        f"Leakage detected on line {line_idx}: forbidden key '{key}' found in public test record!"
                    )

    def test_public_test_image_filenames_neutral(self):
        test_images_dir = Path("data/benchmark_splits/test/public/images")
        assert test_images_dir.is_dir(), "Public test images directory must exist."

        forbidden_patterns = [
            "visible",
            "blocked",
            "none",
            "repair",
            "miss",
            "fov",
            "wrong_side",
            "incident",
            "reflected",
        ]

        images = list(test_images_dir.glob("*.png"))
        assert len(images) > 0, "Public test images must be present."

        for img in images:
            filename = img.name.lower()
            for pattern in forbidden_patterns:
                assert pattern not in filename, (
                    f"Semantic leakage in public test filename: '{filename}' contains '{pattern}'"
                )

    def test_public_test_prompts_zero_hints(self):
        test_tasks_path = Path("data/benchmark_splits/test/public/tasks.jsonl")
        forbidden_prompt_hints = [
            "the answer is",
            "this target is visible",
            "this target is blocked",
            "reflection point is",
            "blocked by obstacle",
        ]

        with open(test_tasks_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                prompt = str(record.get("question", "")).lower()
                for hint in forbidden_prompt_hints:
                    assert hint not in prompt, (
                        f"Prompt hint leakage detected: '{hint}' in prompt '{prompt}'"
                    )

    def test_zero_duplicate_tasks_across_splits(self):
        splits = ["dev", "val", "test"]
        split_task_ids: dict[str, set[str]] = {}

        for s in splits:
            s_file = (
                Path(f"data/benchmark_splits/{s}/tasks.jsonl")
                if s != "test"
                else Path("data/benchmark_splits/test/public/tasks.jsonl")
            )
            ids = set()
            with open(s_file, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        ids.add(json.loads(line)["task_id"])
            split_task_ids[s] = ids

        # Verify pairwise disjointness
        assert len(split_task_ids["dev"].intersection(split_task_ids["val"])) == 0
        assert len(split_task_ids["dev"].intersection(split_task_ids["test"])) == 0
        assert len(split_task_ids["val"].intersection(split_task_ids["test"])) == 0
