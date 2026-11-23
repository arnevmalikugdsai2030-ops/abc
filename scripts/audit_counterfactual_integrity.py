"""
Authoritative Counterfactual Pair Integrity Audit Engine.

Executes exhaustive verification across all 120 counterfactual pairs in data/counterfactual_pairs/
and validates partition isolation across benchmark splits:
1. Physical scene completeness for scene_a and scene_b across all 120 pairs.
2. Independent mathematical oracle recomputation for both child scenes.
3. Verification of designated relationship satisfaction (FLIP vs INVARIANT).
4. Strict pair isolation: zero pair metadata leakage into public splits.
5. Partition integrity: no counterfactual pairs split across dev/val/test boundaries.

Exits with code 0 on 100% PASS, or code 1 if any violation is detected.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mirrorlab.evaluation.dataset_auditor import DatasetAuditor


def audit_counterfactual_integrity(data_root_path: str | Path = "data") -> bool:
    """Executes the complete counterfactual integrity audit."""
    print("=" * 70)
    print("MIRRORLAB COUNTERFACTUAL PAIR INTEGRITY AUDIT")
    print("=" * 70)

    data_root = Path(data_root_path).resolve()
    auditor = DatasetAuditor(data_root=data_root)

    cf_dir = data_root / "counterfactual_pairs"
    print(f"\n1. Auditing 120 Core Counterfactual Pairs in '{cf_dir}'...")
    pair_checks = auditor.audit_counterfactual_pairs(cf_dir)

    all_passed = True
    for c in pair_checks:
        status_str = "[PASS]" if c.passed else "[FAIL]"
        print(f"  {status_str} {c.name:<45} ({c.items_checked:3d} items): {c.details}")
        if not c.passed:
            all_passed = False
            for f in c.failures[:5]:
                print(f"    - Failure: {f}")

    print("\n2. Auditing Counterfactual Partition Integrity Across Benchmark Splits...")
    split_pair_check = auditor.check_counterfactual_pair_integrity()
    status_str = "[PASS]" if split_pair_check.passed else "[FAIL]"
    print(
        f"  {status_str} {split_pair_check.name:<45} ({split_pair_check.items_checked:3d} items): {split_pair_check.details}"
    )
    if not split_pair_check.passed:
        all_passed = False
        for f in split_pair_check.failures[:5]:
            print(f"    - Failure: {f}")

    print("=" * 70)
    if all_passed:
        print(
            "[SUCCESS] All counterfactual pair integrity checks PASSED (100% Verified)."
        )
    else:
        print("[FATAL] Counterfactual pair integrity audit FAILED.")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    passed = audit_counterfactual_integrity()
    sys.exit(0 if passed else 1)
