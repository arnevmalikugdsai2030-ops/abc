"""
Rebuild the complete MIRRORLAB benchmark dataset from scratch with strictly neutral IDs.
Ensures zero semantic answer leakage across all suites and benchmark splits.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from mirrorlab.dataset.ood_generator import OODGenerationConfig, OODTaskGenerator
from mirrorlab.evaluation.dataset_auditor import (
    DatasetAuditor,
    DatasetSplitter,
    render_audit_report_markdown,
)
from mirrorlab.generation.counterfactual_factory import CounterfactualFactory
from mirrorlab.generation.obstruction_factory import ObstructionTaskFactory
from mirrorlab.generation.repair_factory import RepairTaskFactory
from mirrorlab.generation.task_factory import VisibilityTaskFactory


def main():
    print("=" * 75)
    print("REBUILDING MIRRORLAB BENCHMARK WITH STRICTLY NEUTRAL IDENTIFIERS")
    print("=" * 75)

    data_root = Path("data")

    # 1. Rebuild Task A: Visibility Tasks (20 tasks, seeds 10000-10019)
    vis_dir = data_root / "benchmark_tasks"
    if vis_dir.exists():
        shutil.rmtree(vis_dir)
    print("\n[1/5] Generating Task A (Visibility Tasks - 20 tasks, seed=10000)...")
    t0 = time.time()
    vis_factory = VisibilityTaskFactory(image_size=1024)
    vis_tasks = vis_factory.create_balanced_dataset(
        count=20,
        start_seed=10000,
        output_dir=vis_dir,
    )
    print(f"Generated {len(vis_tasks)} visibility tasks in {time.time() - t0:.2f}s")

    # 2. Rebuild Task B: Obstruction Tasks (30 tasks, seeds 25000-25029)
    obs_dir = data_root / "obstruction_tasks"
    if obs_dir.exists():
        shutil.rmtree(obs_dir)
    print("\n[2/5] Generating Task B (Obstruction Tasks - 30 tasks, seed=25000)...")
    t0 = time.time()
    obs_factory = ObstructionTaskFactory(image_size=1024)
    obs_tasks = obs_factory.create_balanced_dataset(
        count=30,
        start_seed=25000,
        output_dir=obs_dir,
    )
    print(f"Generated {len(obs_tasks)} obstruction tasks in {time.time() - t0:.2f}s")

    # 3. Rebuild Task C: Repair Tasks (30 tasks, seeds 30000-30029)
    rep_dir = data_root / "repair_tasks"
    if rep_dir.exists():
        shutil.rmtree(rep_dir)
    print("\n[3/5] Generating Task C (Repair Tasks - 30 tasks, seed=30000)...")
    t0 = time.time()
    rep_factory = RepairTaskFactory(image_size=1024)
    rep_tasks = rep_factory.create_balanced_dataset(
        count=30,
        start_seed=30000,
        output_dir=rep_dir,
    )
    print(f"Generated {len(rep_tasks)} repair tasks in {time.time() - t0:.2f}s")

    # 4. Rebuild Task D: Counterfactual Pairs (120 pairs, seeds 40000+)
    pair_dir = data_root / "counterfactual_pairs"
    if pair_dir.exists():
        shutil.rmtree(pair_dir)
    print("\n[4/5] Generating Task D (Counterfactual Pairs - 120 pairs, seed=40000)...")
    t0 = time.time()
    cf_factory = CounterfactualFactory(image_size=1024)
    pair_bundles = cf_factory.generate_dataset(
        num_pairs_per_type=30,
        output_dir=pair_dir,
        start_seed=40000,
    )
    print(f"Generated {len(pair_bundles)} counterfactual pairs in {time.time() - t0:.2f}s")

    # 5. Rebuild Task E: OOD Tasks (30 tasks, seeds 80000-80029)
    ood_dir = data_root / "ood_tasks"
    if ood_dir.exists():
        shutil.rmtree(ood_dir)
    print("\n[5/5] Generating Task E (OOD Tasks - 30 tasks, seed=80000)...")
    t0 = time.time()
    ood_gen = OODTaskGenerator(
        output_dir=ood_dir,
        config=OODGenerationConfig(
            num_tasks=30,
            seed_start=80000,
        ),
    )
    ood_ids = ood_gen.generate_ood_dataset()
    print(f"Generated {len(ood_ids)} OOD tasks in {time.time() - t0:.2f}s")

    # 6. Generate Stratified Benchmark Splits with strictly neutral IDs
    print("\n" + "=" * 75)
    print("CREATING STRATIFIED SPLITS WITH STRICTLY NEUTRAL IDENTIFIERS")
    print("=" * 75)
    splits_dir = data_root / "benchmark_splits"
    splitter = DatasetSplitter(
        data_root=data_root,
        splits_root=splits_dir,
        dev_ratio=0.20,
        val_ratio=0.20,
        test_ratio=0.60,
        seed=42,
    )
    split_summary = splitter.create_splits()
    print(f"Split Summary: {split_summary}")

    # 7. Run Comprehensive Benchmark Audit
    print("\n" + "=" * 75)
    print("RUNNING COMPREHENSIVE INDEPENDENT AUDIT")
    print("=" * 75)
    auditor = DatasetAuditor(data_root=data_root)
    report = auditor.audit_all()
    report.split_summary = split_summary

    # Export audit report markdown
    report_file = Path("docs/benchmark_audit_report.md")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(render_audit_report_markdown(report))
    print(f"\nAudit Report written to: {report_file.resolve()}")

    print("\n" + "=" * 65)
    print(f"OVERALL AUDIT STATUS: {'PASSED' if report.all_passed else 'FAILED'}")
    print("=" * 65)
    for idx, c in enumerate(report.checks, 1):
        status_str = "[PASS]" if c.passed else "[FAIL]"
        print(f"  {idx:2d}. {status_str} {c.name:<45} ({c.items_checked:3d} items): {c.details}")
        if not c.passed and c.failures:
            for fail in c.failures[:5]:
                print(f"       ! {fail}")


if __name__ == "__main__":
    main()
