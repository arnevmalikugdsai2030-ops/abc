"""
Audit all MIRRORLAB dataset suites and generate stratified Dev, Val, and Blind-Test splits.
"""

from pathlib import Path

from mirrorlab.evaluation.dataset_auditor import (
    DatasetAuditor,
    DatasetSplitter,
    render_audit_report_markdown,
)


def main():
    print("=================================================================")
    print("MIRRORLAB BENCHMARK DATASET AUDITOR & SPLIT ENGINE")
    print("=================================================================")

    # 1. Partition into Dev, Val, Test Splits
    print("\nGenerating stratified benchmark splits (dev: 20%, val: 20%, test: 60%)...")
    splitter = DatasetSplitter(
        data_root="data",
        splits_root="data/benchmark_splits",
        dev_ratio=0.20,
        val_ratio=0.20,
        test_ratio=0.60,
        seed=42,
    )
    split_summary = splitter.create_splits()

    # 2. Run Comprehensive Audit
    auditor = DatasetAuditor(data_root="data")
    print("\nAuditing benchmark datasets (Task A, Task B, Task C, Task D)...")
    report = auditor.audit_all()
    report.split_summary = split_summary

    # 3. Export Markdown Audit Report
    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)
    report_file = docs_dir / "benchmark_audit_report.md"
    md_content = render_audit_report_markdown(report)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\nAudit Report written to: {report_file.resolve()}")

    # 4. Print Summary to Console
    print("\n" + "=" * 65)
    print("AUDIT RESULTS SUMMARY")
    print("=" * 65)
    print(f"Overall Status: {'PASSED' if report.all_passed else 'FAILED'}")
    print(f"Total Scenes Audited: {report.total_scenes_checked}")
    print(f"Total Tasks Audited:  {report.total_tasks_checked}")

    print("\nChecklist:")
    for idx, c in enumerate(report.checks, 1):
        status_str = "[PASS]" if c.passed else "[FAIL]"
        print(
            f"  {idx:2d}. {status_str} {c.name:<45} ({c.items_checked:3d} items): {c.details}"
        )

    print("\nSplit Partition Counts:")
    counts = split_summary.get("split_counts", {})
    total_s = sum(counts.values()) or 1
    for s_name, cnt in counts.items():
        print(
            f"  - {s_name.upper():<5}: {cnt:3d} items ({(cnt / total_s) * 100:5.1f}%)"
        )

    print(f"\nTest Isolation Verified: {split_summary.get('isolation_verified')}")
    print("=================================================================\n")


if __name__ == "__main__":
    main()
