"""
Independent Solver Verification & Consensus Audit Engine.

Executes all 3 independent geometric verifiers across every scene in data/benchmark_splits/
and records exact numerical residuals, agreement rates, coordinate availability, and consensus status.

Solvers evaluated:
1. Householder Specular Oracle (closed-form rank-1 affine reflection matrix)
2. Fermat's Principle of Least Time (variational continuous path minimization)
3. Snell's Law / Law of Reflection (Brent numerical root-finding)

Strict Fail-Closed Invariants:
- VISIBLE must have a finite reflection coordinate.
- NOT_VISIBLE must have a physically consistent reason and output.
- A claimed reflection point must lie on the finite mirror segment.
- The point must satisfy the optical reflection geometry (angle balance & normal orientation).
- A solver must never pass merely because another solver returned a missing coordinate.
- Coordinate comparison must run whenever a physical reflection exists.
- The audit explicitly fails if:
  * Any VISIBLE solver result lacks coordinates.
  * Any coordinate is non-finite.
  * A reflection point lies outside the segment.
  * The three solvers disagree on reflection availability.
  * Expected coordinate comparison coverage is not achieved.
  * Coordinate consensus count differs from comparison count.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from mirrorlab.geometry.independent_verifier import IndependentGeometryVerifier
from mirrorlab.geometry.oracle import VisibilityOracle
from mirrorlab.geometry.ray_march_verifier import DirectSnellRayVerifier
from mirrorlab.schema.scene import InspectionScene


def get_git_commit_sha(repo_root: Path) -> str:
    """Returns the current Git HEAD commit SHA, or UNKNOWN if unavailable."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        head_file = repo_root / ".git" / "HEAD"
        if head_file.is_file():
            try:
                ref = head_file.read_text(encoding="utf-8").strip()
                if ref.startswith("ref: "):
                    ref_path = repo_root / ".git" / ref[5:]
                    if ref_path.is_file():
                        return ref_path.read_text(encoding="utf-8").strip()
                return ref
            except OSError:
                pass
        return "UNKNOWN"


def compute_dataset_content_hash(data_root: Path) -> str:
    """Computes combined SHA-256 hash across all benchmark split tasks and ground-truth files."""
    splits_dir = data_root / "benchmark_splits"
    target_files = [
        splits_dir / "dev" / "tasks.jsonl",
        splits_dir / "dev" / "ground_truth.jsonl",
        splits_dir / "val" / "tasks.jsonl",
        splits_dir / "val" / "ground_truth.jsonl",
        splits_dir / "test" / "public" / "tasks.jsonl",
        splits_dir / "test" / "private" / "ground_truth.jsonl",
    ]
    h = hashlib.sha256()
    for f in target_files:
        if f.is_file():
            with open(f, "rb") as fp:
                while chunk := fp.read(65536):
                    h.update(chunk)
    return h.hexdigest()


def resolve_scene_path(
    data_root: Path, split: str, task_id: str, gt_info: dict[str, Any]
) -> Path | None:
    """Deterministically resolves the filesystem path to scene.json for any task in a split."""
    int_id = gt_info.get("internal_task_id", task_id)
    splits_dir = data_root / "benchmark_splits"

    candidates = [
        splits_dir / split / "tasks" / task_id / "scene.json",
        data_root / "benchmark_tasks" / int_id / "scene.json",
        data_root / "obstruction_tasks" / int_id / "scene.json",
        data_root / "repair_tasks" / int_id / "scene.json",
        data_root / "ood_tasks" / int_id / "scene.json",
        data_root / "benchmark_tasks" / task_id / "scene.json",
        data_root / "obstruction_tasks" / task_id / "scene.json",
        data_root / "repair_tasks" / task_id / "scene.json",
        data_root / "ood_tasks" / task_id / "scene.json",
    ]
    for c in candidates:
        if c.is_file():
            return c

    # Check counterfactual pair directories: pair_XXXX_a or pair_XXXX_b
    parts = int_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in ("a", "b"):
        pair_dir = data_root / "counterfactual_pairs" / parts[0]
        child_file = pair_dir / f"scene_{parts[1]}" / "scene.json"
        if child_file.is_file():
            return child_file

    return None


def extract_point_xy(pt: Any) -> tuple[float, float] | None:
    """Extracts (x, y) floats from a Point2D, numpy array, or tuple/list."""
    if pt is None:
        return None
    if hasattr(pt, "x") and hasattr(pt, "y"):
        return float(pt.x), float(pt.y)
    try:
        return float(pt[0]), float(pt[1])
    except (IndexError, TypeError, ValueError):
        return None


def validate_solver_output(
    solver_name: str,
    state_str: str,
    pt: tuple[float, float] | None,
    scene: InspectionScene,
    is_reflection_on_mirror: bool,
    tolerance_mm: float = 0.1,
    reason_str: str | None = None,
) -> tuple[bool, str | None]:
    """Validates physical plausibility and strict invariants of a solver's output."""
    if state_str not in ("VISIBLE", "NOT_VISIBLE"):
        return False, f"{solver_name} output invalid state string: '{state_str}'"

    # Invariant: VISIBLE must have a finite reflection coordinate
    if state_str == "VISIBLE":
        if pt is None:
            return (
                False,
                f"{solver_name}: state is VISIBLE but reflection_point is None",
            )
        if not (math.isfinite(pt[0]) and math.isfinite(pt[1])):
            return False, f"{solver_name}: produced non-finite coordinates: {pt}"

    # Invariant: Any coordinates returned must be finite
    if pt is not None:
        x, y = pt
        if not (math.isfinite(x) and math.isfinite(y)):
            return False, f"{solver_name} produced non-finite coordinates: ({x}, {y})"

    # Invariant: If a physical reflection on the mirror segment exists, coordinate must lie on segment
    if is_reflection_on_mirror:
        if pt is None:
            return (
                False,
                f"{solver_name}: physical reflection exists on mirror, but {solver_name} returned None",
            )
        x, y = pt
        ma = scene.mirror.endpoint_a
        mb = scene.mirror.endpoint_b
        vx = mb.x - ma.x
        vy = mb.y - ma.y
        l2 = vx * vx + vy * vy
        if l2 <= 1e-9:
            return False, f"{solver_name}: mirror has degenerate zero length"

        u = ((x - ma.x) * vx + (y - ma.y) * vy) / l2
        # Strict segment bounds with numerical margin of 1e-3
        if not (-1e-3 <= u <= 1.0 + 1e-3):
            return (
                False,
                f"{solver_name} reflection point lies outside mirror segment (u={u:.6f})",
            )

        px = ma.x + u * vx
        py = ma.y + u * vy
        dist_to_mirror_line = math.hypot(x - px, y - py)
        if dist_to_mirror_line > tolerance_mm:
            return False, (
                f"{solver_name} reflection point off mirror line by "
                f"{dist_to_mirror_line:.6f} mm (tolerance {tolerance_mm} mm)"
            )

        # Normal orientation: camera and target must be on reflective front side
        c = scene.camera.position
        t = scene.target.center
        n = scene.mirror.reflective_normal
        dot_c = (c.x - x) * n.dx + (c.y - y) * n.dy
        dot_t = (t.x - x) * n.dx + (t.y - y) * n.dy
        if dot_c <= 0:
            return (
                False,
                f"{solver_name}: ray approaches from non-reflective mirror backside (dot_c={dot_c:.6f})",
            )
        if dot_t <= 0:
            return (
                False,
                f"{solver_name}: target is behind non-reflective mirror backside (dot_t={dot_t:.6f})",
            )

        # Optical reflection geometry: Law of reflection angle balance
        dist_c = math.hypot(x - c.x, y - c.y)
        dist_t = math.hypot(t.x - x, t.y - y)
        if dist_c <= 1e-6 or dist_t <= 1e-6:
            return False, f"{solver_name}: degenerate zero ray length"

        di_x, di_y = (x - c.x) / dist_c, (y - c.y) / dist_c
        dr_x, dr_y = (t.x - x) / dist_t, (t.y - y) / dist_t
        cos_i = -(di_x * n.dx + di_y * n.dy)
        cos_r = dr_x * n.dx + dr_y * n.dy
        if abs(cos_i - cos_r) > 1e-3:
            return (
                False,
                f"{solver_name}: violates optical reflection geometry (|cos_i - cos_r| = {abs(cos_i - cos_r):.6e} > 1e-3)",
            )

    elif (
        state_str == "NOT_VISIBLE"
        and reason_str
        in (
            "MISSED_MIRROR",
            "BACK_OF_MIRROR",
            "missed_mirror",
            "back_of_mirror",
        )
        and pt is not None
    ):
        ma = scene.mirror.endpoint_a
        mb = scene.mirror.endpoint_b
        vx = mb.x - ma.x
        vy = mb.y - ma.y
        l2 = vx * vx + vy * vy
        if l2 > 1e-9:
            u = ((pt[0] - ma.x) * vx + (pt[1] - ma.y) * vy) / l2
            if 0.0 <= u <= 1.0 and reason_str in (
                "MISSED_MIRROR",
                "missed_mirror",
            ):
                return (
                    False,
                    f"{solver_name}: reason is MISSED_MIRROR but coordinate lies inside segment (u={u:.4f})",
                )

    return True, None


def audit_all_benchmark_scenes(
    data_root_path: str | Path = "data",
    tolerance_mm: float = 0.1,
    expected_total_tasks: int = 350,
    expected_coord_comparisons: int = 306,
    save_artifact: bool = True,
) -> dict[str, Any]:
    """
    Executes the comprehensive triple-verifier consensus audit.
    Returns complete dictionary of results, provenance metadata, and per-scene records.
    """
    print("=" * 75)
    print("MIRRORLAB TRIPLE-VERIFIER CONSENSUS AUDIT (SCIENTIFIC RE-AUDIT)")
    print(
        "Evaluating: Oracle (Householder Affine) vs. Fermat (Variational) vs. Snell (Brent Root)"
    )
    print("=" * 75)

    data_root = Path(data_root_path).resolve()
    repo_root = data_root.parent
    splits_dir = data_root / "benchmark_splits"

    source_commit_sha = get_git_commit_sha(repo_root)
    dataset_content_hash = compute_dataset_content_hash(data_root)
    execution_timestamp = datetime.now(UTC).isoformat()

    manifest_file = data_root / "dataset_manifest.json"
    manifest_sha256 = "NOT_FOUND"
    if manifest_file.is_file():
        h = hashlib.sha256()
        with open(manifest_file, "rb") as mf:
            while chunk := mf.read(65536):
                h.update(chunk)
        manifest_sha256 = h.hexdigest()

    records: list[dict[str, Any]] = []
    missing_scenes: list[dict[str, str]] = []
    failed_scene_ids: list[str] = []

    total_scenes_resolved = 0
    state_consensus_count = 0
    coord_comparison_count = 0
    coord_consensus_count = 0
    measured_residuals: list[float] = []

    for split in ["dev", "val", "test"]:
        tasks_file = (
            splits_dir / split / "tasks.jsonl"
            if split != "test"
            else splits_dir / split / "public" / "tasks.jsonl"
        )
        gt_file = (
            splits_dir / split / "ground_truth.jsonl"
            if split != "test"
            else splits_dir / split / "private" / "ground_truth.jsonl"
        )

        if not tasks_file.is_file():
            print(f"[ERROR] Tasks file missing for split '{split}': {tasks_file}")
            missing_scenes.append(
                {
                    "task_id": f"split_{split}_all",
                    "split": split,
                    "reason": f"Tasks file missing: {tasks_file}",
                }
            )
            continue

        with open(tasks_file, encoding="utf-8") as tf:
            split_tasks = [json.loads(line) for line in tf if line.strip()]

        gt_map: dict[str, dict[str, Any]] = {}
        if gt_file.is_file():
            with open(gt_file, encoding="utf-8") as gf:
                for line in gf:
                    if line.strip():
                        item = json.loads(line)
                        gt_map[item["task_id"]] = item

        for task in split_tasks:
            tid = task["task_id"]
            gt_info = gt_map.get(tid, {})
            fam = gt_info.get("task_family", "UNKNOWN")

            scene_path = resolve_scene_path(data_root, split, tid, gt_info)
            if not scene_path or not scene_path.is_file():
                missing_scenes.append(
                    {
                        "task_id": tid,
                        "split": split,
                        "internal_task_id": gt_info.get("internal_task_id", tid),
                        "reason": "scene.json not found on disk",
                    }
                )
                failed_scene_ids.append(tid)
                continue

            try:
                with open(scene_path, encoding="utf-8") as sf:
                    scene_dict = json.load(sf)
                scene = InspectionScene.model_validate(scene_dict)
            except (json.JSONDecodeError, ValueError, OSError) as exc:
                missing_scenes.append(
                    {
                        "task_id": tid,
                        "split": split,
                        "internal_task_id": gt_info.get("internal_task_id", tid),
                        "reason": f"Corrupt scene.json: {exc}",
                    }
                )
                failed_scene_ids.append(tid)
                continue

            total_scenes_resolved += 1

            # 1. Oracle (Virtual Target via Householder Operator)
            verdict_oracle = VisibilityOracle.evaluate(scene)

            # 2. Fermat (Variational Least-Time Solver)
            verdict_fermat = IndependentGeometryVerifier.evaluate(scene)

            # 3. Snell (Brent Root-Finder)
            snell_dict = DirectSnellRayVerifier.evaluate_scene(scene)

            # Normalize states
            s_oracle_val = (
                verdict_oracle.state.value
                if hasattr(verdict_oracle.state, "value")
                else str(verdict_oracle.state)
            )
            s_fermat_val = (
                verdict_fermat.state.value
                if hasattr(verdict_fermat.state, "value")
                else str(verdict_fermat.state)
            )
            s_snell_val = (
                snell_dict["state"].value
                if hasattr(snell_dict["state"], "value")
                else str(snell_dict["state"])
            )

            state_agree = s_oracle_val == s_fermat_val == s_snell_val
            if state_agree:
                state_consensus_count += 1

            # Extract reflection coordinates
            pt_oracle = extract_point_xy(verdict_oracle.reflection_point)
            pt_fermat = extract_point_xy(
                getattr(verdict_fermat, "reflection_point", None)
            )
            pt_snell = extract_point_xy(snell_dict.get("reflection_point"))

            oracle_reason = (
                verdict_oracle.reason.value
                if hasattr(verdict_oracle.reason, "value")
                else str(verdict_oracle.reason)
            )
            fermat_reason = getattr(verdict_fermat, "reason", None)
            f_reason_val = (
                fermat_reason.value
                if hasattr(fermat_reason, "value")
                else str(fermat_reason)
            )
            snell_reason = snell_dict.get("reason")
            s_reason_val = (
                snell_reason.value
                if hasattr(snell_reason, "value")
                else str(snell_reason)
            )

            # Check reflection availability per solver
            o_has_refl = pt_oracle is not None and oracle_reason.lower() not in (
                "missed_mirror",
                "back_of_mirror",
            )
            f_has_refl = pt_fermat is not None and f_reason_val.lower() not in (
                "missed_mirror",
                "back_of_mirror",
            )
            s_has_refl = pt_snell is not None and s_reason_val.lower() not in (
                "missed_mirror",
                "back_of_mirror",
            )

            # Invariant: Disagreement on reflection availability fails scene
            refl_avail_agree = o_has_refl == f_has_refl == s_has_refl
            has_reflection_on_mirror = (
                refl_avail_agree and o_has_refl and f_has_refl and s_has_refl
            )

            # Validate solver outputs independently
            valid_o, err_o = validate_solver_output(
                "oracle",
                s_oracle_val,
                pt_oracle,
                scene,
                has_reflection_on_mirror,
                tolerance_mm,
                oracle_reason,
            )
            valid_f, err_f = validate_solver_output(
                "fermat",
                s_fermat_val,
                pt_fermat,
                scene,
                has_reflection_on_mirror,
                tolerance_mm,
                f_reason_val,
            )
            valid_s, err_s = validate_solver_output(
                "snell",
                s_snell_val,
                pt_snell,
                scene,
                has_reflection_on_mirror,
                tolerance_mm,
                s_reason_val,
            )

            # Coordinate comparison
            r_res_of: float | None = None
            r_res_os: float | None = None
            r_res_fs: float | None = None
            max_res: float | None = None
            coord_available = False
            coord_agree = False

            if has_reflection_on_mirror:
                coord_available = True
                coord_comparison_count += 1
                ox, oy = pt_oracle  # type: ignore[misc]
                fx, fy = pt_fermat  # type: ignore[misc]
                sx, sy = pt_snell  # type: ignore[misc]

                d_of = math.hypot(ox - fx, oy - fy)
                d_os = math.hypot(ox - sx, oy - sy)
                d_fs = math.hypot(fx - sx, fy - sy)
                max_r = max(d_of, d_os, d_fs)

                r_res_of = round(d_of, 8)
                r_res_os = round(d_os, 8)
                r_res_fs = round(d_fs, 8)
                max_res = round(max_r, 8)
                measured_residuals.append(max_r)

                coord_agree = max_r <= tolerance_mm
                if coord_agree:
                    coord_consensus_count += 1
            elif not refl_avail_agree:
                coord_available = False
                coord_agree = False
            else:
                # No physical reflection point exists on the mirror segment
                # All solvers unanimously agree reflection does not exist
                coord_available = False
                coord_agree = True  # Consistently absent across all 3, residuals null

            # Scene pass/fail determination
            solver_plausible = valid_o and valid_f and valid_s
            scene_passed = (
                state_agree and refl_avail_agree and solver_plausible and coord_agree
            )

            failure_reasons = []
            if not state_agree:
                failure_reasons.append(
                    f"State mismatch: O={s_oracle_val}, F={s_fermat_val}, S={s_snell_val}"
                )
            if not refl_avail_agree:
                failure_reasons.append(
                    f"Reflection availability disagreement: O={o_has_refl}, F={f_has_refl}, S={s_has_refl}"
                )
            if not solver_plausible:
                failure_reasons.extend(
                    [e for e in (err_o, err_f, err_s) if e is not None]
                )
            if coord_available and not coord_agree:
                failure_reasons.append(
                    f"Coordinate residual {max_res} mm exceeds tolerance {tolerance_mm} mm"
                )

            failure_reason_str = "; ".join(failure_reasons) if failure_reasons else None
            if not scene_passed:
                failed_scene_ids.append(tid)

            record = {
                "task_id": tid,
                "scene_id": scene.scene_id,
                "split": split,
                "task_family": fam,
                "solver_outputs": {
                    "oracle": {
                        "state": s_oracle_val,
                        "reason": oracle_reason,
                        "reflection_point": list(pt_oracle)
                        if pt_oracle and has_reflection_on_mirror
                        else None,
                    },
                    "fermat": {
                        "state": s_fermat_val,
                        "reflection_point": list(pt_fermat)
                        if pt_fermat and has_reflection_on_mirror
                        else None,
                    },
                    "snell": {
                        "state": s_snell_val,
                        "reflection_point": list(pt_snell)
                        if pt_snell and has_reflection_on_mirror
                        else None,
                    },
                },
                "state_agreement": state_agree,
                "reflection_availability_agreement": refl_avail_agree,
                "coordinate_available": coord_available,
                "pairwise_residuals_mm": {
                    "oracle_vs_fermat": r_res_of,
                    "oracle_vs_snell": r_res_os,
                    "fermat_vs_snell": r_res_fs,
                },
                "maximum_residual_mm": max_res,
                "tolerance_mm": tolerance_mm,
                "validation_result": "PASS" if scene_passed else "FAIL",
                "failure_reason": failure_reason_str,
            }
            records.append(record)

    # Compute residual statistics
    stats: dict[str, float | None] = {
        "count": len(measured_residuals),
        "max": round(float(np.max(measured_residuals)), 8)
        if measured_residuals
        else None,
        "mean": round(float(np.mean(measured_residuals)), 8)
        if measured_residuals
        else None,
        "median": round(float(np.median(measured_residuals)), 8)
        if measured_residuals
        else None,
        "p95": round(float(np.percentile(measured_residuals, 95)), 8)
        if measured_residuals
        else None,
        "p99": round(float(np.percentile(measured_residuals, 99)), 8)
        if measured_residuals
        else None,
    }

    # Strict Fail-Closed Global Invariants
    all_passed = (
        len(missing_scenes) == 0
        and len(failed_scene_ids) == 0
        and total_scenes_resolved == expected_total_tasks
        and state_consensus_count == total_scenes_resolved
        and coord_comparison_count == expected_coord_comparisons
        and coord_consensus_count == coord_comparison_count
    )

    audit_summary = {
        "audit_version": "2.1.0-STRICT-FAIL-CLOSED",
        "timestamp": execution_timestamp,
        "source_commit_sha": source_commit_sha,
        "dataset_manifest_sha256": manifest_sha256,
        "dataset_content_hash": dataset_content_hash,
        "command_used": "python scripts/audit_triple_verifiers.py",
        "expected_scenes": expected_total_tasks,
        "resolved_scenes": total_scenes_resolved,
        "missing_scenes_count": len(missing_scenes),
        "missing_scenes": missing_scenes,
        "state_consensus_count": state_consensus_count,
        "state_consensus_rate_pct": round(
            (state_consensus_count / total_scenes_resolved * 100.0)
            if total_scenes_resolved
            else 0.0,
            4,
        ),
        "expected_coordinate_comparisons": expected_coord_comparisons,
        "coordinate_comparison_count": coord_comparison_count,
        "coordinate_consensus_count": coord_consensus_count,
        "coordinate_consensus_rate_pct": round(
            (coord_consensus_count / coord_comparison_count * 100.0)
            if coord_comparison_count
            else 0.0,
            4,
        ),
        "tolerance_mm": tolerance_mm,
        "residual_statistics_mm": stats,
        "failed_scene_ids": failed_scene_ids,
        "all_passed": all_passed,
        "records": records,
    }

    # Print summary
    print("\nAUDIT SUMMARY REPORT:")
    print(f"  Source Commit SHA:           {source_commit_sha}")
    print(f"  Dataset Content Hash:        {dataset_content_hash[:16]}...")
    print(f"  Expected Tasks:              {expected_total_tasks}")
    print(f"  Resolved Scenes:             {total_scenes_resolved}")
    print(f"  Missing Scenes:              {len(missing_scenes)}")
    print(
        f"  State Consensus:             {state_consensus_count}/{total_scenes_resolved} "
        f"({audit_summary['state_consensus_rate_pct']}%)"
    )
    print(
        f"  Coordinate Comparisons:      {coord_comparison_count}/{expected_coord_comparisons}"
    )
    print(
        f"  Coordinate Consensus:        {coord_consensus_count}/{coord_comparison_count} "
        f"({audit_summary['coordinate_consensus_rate_pct']}%)"
    )
    if stats["max"] is not None:
        print(f"  Maximum Residual:            {stats['max']:.6e} mm")
        print(f"  Mean Residual:               {stats['mean']:.6e} mm")
        print(f"  Median Residual (P50):       {stats['median']:.6e} mm")
        print(f"  95th Percentile Residual:    {stats['p95']:.6e} mm")
        print(f"  99th Percentile Residual:    {stats['p99']:.6e} mm")
    print(f"  Audit Result:                {'PASS' if all_passed else 'FAIL'}")
    print("=" * 75)

    if save_artifact:
        out_path = repo_root / "experiments" / "triple_verifier_consensus_audit.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(audit_summary, f, indent=2)
        print(f"Full audit report archived to: {out_path.resolve()}")

    return audit_summary


if __name__ == "__main__":
    result = audit_all_benchmark_scenes()
    if not result["all_passed"]:
        print("\n[FATAL] Triple-verifier consensus audit failed verification!")
        sys.exit(1)
    else:
        print(
            "\n[SUCCESS] All 350 benchmark scenes certified by triple-verifier consensus."
        )
        sys.exit(0)
