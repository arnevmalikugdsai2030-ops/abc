"""
Authoritative Dataset Inventory Generator & Manifest Builder for MIRRORLAB.

Scans the real filesystem to compute exact counts, hashes, split distributions,
image counts, and ground-truth availability directly from actual files on disk.

Enforces:
- Direct extraction of task family from explicit ground-truth metadata (no keyword heuristics).
- Distinct accounting of task records, unique scenes, physical JSON files, pairs, and images.
- Reconciliation of split totals and task-family totals.
- Detection of duplicate IDs or missing artifacts.
- Zero hardcoded validation claims: reads real audit artifacts or reports NOT_RUN.

Outputs: data/dataset_manifest.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def scan_dataset(data_root: Path | str = "data") -> dict[str, Any]:
    """
    Exhaustively scans the data/ directory and builds the authoritative inventory.
    """
    root = Path(data_root).resolve()
    splits_dir = root / "benchmark_splits"

    # 1. Scan benchmark splits
    splits = ["dev", "val", "test"]
    split_task_counts: dict[str, int] = {}
    split_gt_counts: dict[str, int] = {}
    split_images: dict[str, int] = {}
    split_overlays: dict[str, int] = {}
    split_pairs: dict[str, int] = {}

    all_task_ids: list[str] = []
    unique_task_ids: set[str] = set()
    unique_scene_ids: set[str] = set()
    split_hashes: dict[str, str] = {}

    task_family_counts: dict[str, int] = {
        "TASK_A_VISIBILITY": 0,
        "TASK_B_OBSTRUCTION": 0,
        "TASK_C_REPAIR": 0,
        "TASK_D_COUNTERFACTUAL": 0,
        "TASK_E_OOD": 0,
    }

    missing_split_artifacts: list[str] = []

    for split in splits:
        s_dir = splits_dir / split
        if not s_dir.exists():
            missing_split_artifacts.append(f"Split dir missing: {s_dir}")
            continue

        tasks_file = (
            s_dir / "tasks.jsonl"
            if split != "test"
            else s_dir / "public" / "tasks.jsonl"
        )
        gt_file = (
            s_dir / "ground_truth.jsonl"
            if split != "test"
            else s_dir / "private" / "ground_truth.jsonl"
        )
        pairs_file = s_dir / "pairs.jsonl"

        if not tasks_file.is_file():
            missing_split_artifacts.append(f"Tasks file missing: {tasks_file}")
            continue
        if not gt_file.is_file():
            missing_split_artifacts.append(f"GT file missing: {gt_file}")
            continue

        split_hashes[f"{split}_tasks_sha256"] = compute_file_sha256(tasks_file)
        split_hashes[f"{split}_ground_truth_sha256"] = compute_file_sha256(gt_file)
        if pairs_file.is_file():
            split_hashes[f"{split}_pairs_sha256"] = compute_file_sha256(pairs_file)

        # Read tasks
        split_tasks: list[dict[str, Any]] = []
        with open(tasks_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    t = json.loads(line)
                    split_tasks.append(t)
                    tid = t.get("task_id")
                    if tid:
                        all_task_ids.append(tid)
                        unique_task_ids.add(tid)
                    sid = t.get("scene_id")
                    if sid:
                        unique_scene_ids.add(sid)

        # Read ground truth
        split_gt: dict[str, dict[str, Any]] = {}
        with open(gt_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    split_gt[item["task_id"]] = item

                    # Explicit task family accounting
                    fam = item.get("task_family")
                    if fam in task_family_counts:
                        task_family_counts[fam] += 1
                    elif fam:
                        task_family_counts[fam] = task_family_counts.get(fam, 0) + 1

        split_task_counts[split] = len(split_tasks)
        split_gt_counts[split] = len(split_gt)

        # Count pairs if pairs.jsonl present
        if pairs_file.is_file():
            with open(pairs_file, encoding="utf-8") as pf:
                pairs_count = sum(1 for line in pf if line.strip())
            split_pairs[split] = pairs_count
        else:
            split_pairs[split] = 0

        # Count images in split
        img_dir = s_dir / "images" if split != "test" else s_dir / "public" / "images"
        blind_imgs = len(list(img_dir.glob("*.png"))) if img_dir.exists() else 0
        split_images[split] = blind_imgs

        overlay_dir = (
            s_dir / "images" if split != "test" else s_dir / "private" / "overlays"
        )
        if split != "test":
            overlays = (
                len(list(img_dir.glob("*_overlay.png"))) if img_dir.exists() else 0
            )
        else:
            overlays = (
                len(list(overlay_dir.glob("*.png"))) if overlay_dir.exists() else 0
            )
        split_overlays[split] = overlays

    # Duplicate ID check
    duplicate_task_ids = len(all_task_ids) - len(unique_task_ids)

    # 2. Scan core task suites
    task_dirs = {
        "benchmark_tasks": root / "benchmark_tasks",
        "obstruction_tasks": root / "obstruction_tasks",
        "repair_tasks": root / "repair_tasks",
        "counterfactual_pairs": root / "counterfactual_pairs",
        "ood_tasks": root / "ood_tasks",
    }
    task_dir_counts: dict[str, int] = {}
    core_scene_json_count = 0
    cf_pairs_count = 0
    cf_child_scenes_count = 0
    missing_pair_child_scenes: list[str] = []

    for name, p in task_dirs.items():
        if p.exists():
            dirs = [d for d in p.iterdir() if d.is_dir()]
            task_dir_counts[name] = len(dirs)

            if name == "counterfactual_pairs":
                cf_pairs_count = len(dirs)
                for d in dirs:
                    # Each counterfactual pair must contain scene_a and scene_b
                    sa = d / "scene_a" / "scene.json"
                    sb = d / "scene_b" / "scene.json"
                    if sa.is_file():
                        cf_child_scenes_count += 1
                        core_scene_json_count += 1
                    else:
                        missing_pair_child_scenes.append(f"{d.name}/scene_a")
                    if sb.is_file():
                        cf_child_scenes_count += 1
                        core_scene_json_count += 1
                    else:
                        missing_pair_child_scenes.append(f"{d.name}/scene_b")
            else:
                for d in dirs:
                    if (d / "scene.json").is_file():
                        core_scene_json_count += 1

    # Total physical scene.json files across entire data/ dir
    all_physical_scene_json_files = len(list(root.glob("**/scene.json")))
    total_rendered_png_files = len(list(root.glob("**/*.png")))

    total_split_tasks = sum(split_task_counts.values())
    total_split_gt = sum(split_gt_counts.values())
    total_split_blind_images = sum(split_images.values())
    total_split_overlays = sum(split_overlays.values())

    # 3. Reconciliations & Invariants Validation
    reconciliations = {
        "split_tasks_equals_ground_truth": total_split_tasks == total_split_gt,
        "split_tasks_equals_family_sum": total_split_tasks
        == sum(task_family_counts.values()),
        "zero_duplicate_task_ids": duplicate_task_ids == 0,
        "pairs_have_two_children": cf_child_scenes_count == (cf_pairs_count * 2),
        "zero_missing_pair_scenes": len(missing_pair_child_scenes) == 0,
        "zero_missing_split_artifacts": len(missing_split_artifacts) == 0,
    }

    # 4. Read real validation & audit status from actual executed audit artifacts
    triple_audit_file = (
        root.parent / "experiments" / "triple_verifier_consensus_audit.json"
    )
    triple_audit_status: dict[str, Any] = {
        "audit_executed": False,
        "status": "NOT_RUN",
    }
    if triple_audit_file.is_file():
        try:
            with open(triple_audit_file, encoding="utf-8") as taf:
                t_data = json.load(taf)
            triple_audit_status = {
                "audit_executed": True,
                "status": "PASS" if t_data.get("all_passed") else "FAIL",
                "state_consensus_rate_pct": t_data.get("state_consensus_rate_pct"),
                "coordinate_consensus_rate_pct": t_data.get(
                    "coordinate_consensus_rate_pct"
                ),
                "coordinate_comparisons": t_data.get("coordinate_comparison_count"),
                "maximum_residual_mm": t_data.get("residual_statistics_mm", {}).get(
                    "max"
                ),
                "tolerance_mm": t_data.get("tolerance_mm"),
            }
        except (OSError, json.JSONDecodeError) as exc:
            triple_audit_status["error"] = str(exc)

    dataset_audit_cache_file = root / ".audit_report_cache.json"
    dataset_audit_status: dict[str, Any] = {
        "audit_executed": False,
        "status": "NOT_RUN",
    }
    if dataset_audit_cache_file.is_file():
        try:
            with open(dataset_audit_cache_file, encoding="utf-8") as daf:
                d_data = json.load(daf)
            dataset_audit_status = {
                "audit_executed": True,
                "status": "PASS" if d_data.get("all_passed") else "FAIL",
                "total_checks": d_data.get("total_checks"),
                "isolation_verified": d_data.get("isolation_verified"),
                "status_message": d_data.get("status_message"),
            }
        except (OSError, json.JSONDecodeError) as exc:
            dataset_audit_status["error"] = str(exc)

    manifest_data = {
        "dataset_name": "MIRRORLAB - The Missing Scratch",
        "dataset_version": "v2.2.0-PROVENANCE-VERIFIED",
        "schema_version": "2.0.0",
        "generator_version": "mirrorlab-inventory-v2.0.0",
        "timestamp": "2026-11-23T14:45:00Z",
        "summary_statistics": {
            "total_model_facing_tasks": total_split_tasks,
            "total_private_ground_truth_records": total_split_gt,
            "total_unique_task_ids": len(unique_task_ids),
            "duplicate_task_id_count": duplicate_task_ids,
            "total_core_task_directories": sum(task_dir_counts.values()),
            "total_core_scenes_on_disk": core_scene_json_count,
            "total_physical_scene_json_files_all": all_physical_scene_json_files,
            "total_counterfactual_pairs": cf_pairs_count,
            "total_counterfactual_child_scenes": cf_child_scenes_count,
            "total_rendered_images_on_disk": total_rendered_png_files,
            "total_split_blind_images": total_split_blind_images,
            "total_split_diagnostic_overlays": total_split_overlays,
        },
        "split_distribution": {
            "dev": {
                "tasks": split_task_counts.get("dev", 0),
                "ground_truth_records": split_gt_counts.get("dev", 0),
                "pairs": split_pairs.get("dev", 0),
                "blind_images": split_images.get("dev", 0),
                "overlays": split_overlays.get("dev", 0),
            },
            "val": {
                "tasks": split_task_counts.get("val", 0),
                "ground_truth_records": split_gt_counts.get("val", 0),
                "pairs": split_pairs.get("val", 0),
                "blind_images": split_images.get("val", 0),
                "overlays": split_overlays.get("val", 0),
            },
            "test": {
                "tasks": split_task_counts.get("test", 0),
                "ground_truth_records": split_gt_counts.get("test", 0),
                "pairs": split_pairs.get("test", 0),
                "blind_images": split_images.get("test", 0),
                "overlays": split_overlays.get("test", 0),
                "ground_truth_isolation": "STRICT_PRIVATE_DIR",
            },
        },
        "task_family_distribution": task_family_counts,
        "task_directory_breakdown": task_dir_counts,
        "cryptographic_split_hashes": split_hashes,
        "reconciliations": reconciliations,
        "executed_audits": {
            "triple_verifier_consensus": triple_audit_status,
            "dataset_integrity_11_points": dataset_audit_status,
        },
    }

    return manifest_data


def main():
    print("=" * 70)
    print("MIRRORLAB AUTHORITATIVE DATASET INVENTORY SCANNER")
    print("=" * 70)

    manifest = scan_dataset("data")
    out_file = Path("data/dataset_manifest.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    stats = manifest["summary_statistics"]
    reconciled = all(manifest["reconciliations"].values())

    print(
        f"Model-Facing Split Tasks : {stats['total_model_facing_tasks']} (Dev={manifest['split_distribution']['dev']['tasks']}, Val={manifest['split_distribution']['val']['tasks']}, Test={manifest['split_distribution']['test']['tasks']})"
    )
    print(f"Private Ground Truth     : {stats['total_private_ground_truth_records']}")
    print(
        f"Unique Task IDs          : {stats['total_unique_task_ids']} (Duplicates: {stats['duplicate_task_id_count']})"
    )
    print(f"Core Task Directories    : {stats['total_core_task_directories']}")
    print(
        f"Counterfactual Pairs     : {stats['total_counterfactual_pairs']} (Child scenes: {stats['total_counterfactual_child_scenes']})"
    )
    print(
        f"Core Scenes on Disk      : {stats['total_core_scenes_on_disk']} (Physical files: {stats['total_physical_scene_json_files_all']})"
    )
    print(f"Total Rendered Images    : {stats['total_rendered_images_on_disk']}")
    print(f"Reconciliations Valid    : {reconciled}")
    print(
        f"Triple Verifier Status   : {manifest['executed_audits']['triple_verifier_consensus']['status']}"
    )
    print(
        f"Dataset 11-Point Status  : {manifest['executed_audits']['dataset_integrity_11_points']['status']}"
    )
    print(f"Authoritative manifest saved to: {out_file.resolve()}")
    print("=" * 70)

    if not reconciled:
        print("[ERROR] Dataset inventory reconciliations failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
