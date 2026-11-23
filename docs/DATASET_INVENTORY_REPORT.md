# MIRRORLAB Authoritative Dataset Inventory & Audit Report
**Generated:** 2026-11-23  
**Source Manifest:** `data/dataset_manifest.json`  
**Schema Version:** 1.3.0  
**Verification Status:** 100% Passed (11/11 Integrity Checks)  

---

## 1. Executive Summary

This report provides the single source of truth for all dataset assets committed to the MIRRORLAB repository. All statistics in this document are automatically computed from the physical filesystem by `scripts/generate_dataset_manifest.py`, eliminating manual discrepancies.

---

## 2. Benchmark Split Inventory (Model-Facing)

The headline competition benchmark consists of **350 model-facing tasks** across three partitions:

| Partition | Tasks | Blind Images | Diagnostic Overlays | Ground-Truth Exposure |
|---|---|---|---|---|
| **`dev`** | 70 | 70 | 70 | Public (`ground_truth.jsonl`) |
| **`val`** | 70 | 70 | 70 | Public (`ground_truth.jsonl`) |
| **`test`** | 210 | 210 | 210 | **Sealed** (`private/ground_truth.jsonl`) |
| **Total** | **350** | **350** | **350** | **Zero leakage to test split** |

### Split Hash Integrity:
- `dev_tasks`: `2dc188734e5a95ad18a0e8d052d92131bb48ef2cb80479713e87fb25be243ae7`
- `val_tasks`: `a832d2077e682d354b0c793ff01ff8002df35c75539fa9fa59a421684c7f12e5`
- `test_tasks`: `b144fc600c025066db2b9095627a85dfa07f0699fa845ca35e4e89fe8bf25b6a`

---

## 3. Core Task Repository Directories (`data/`)

In addition to the partitioned evaluation splits, `data/` houses **230 curated task suites** representing individual procedural generations:

| Directory | Task Family | Description | Task Count |
|---|---|---|---|
| `data/benchmark_tasks/` | Task A | Baseline Specular Visibility | 20 |
| `data/obstruction_tasks/` | Task B | Single-Blocker & Leg Attribution | 30 |
| `data/repair_tasks/` | Task C | Active Specular Alignment & Repair | 30 |
| `data/counterfactual_pairs/`| Task D | Minimal Counterfactual Pairs | 120 |
| `data/ood_tasks/` | Task E | Extreme Grazing & High Clutter OOD | 30 |
| **Total Core Task Dirs** | | | **230** |

**Total Rendered Images on Disk:** **2,000 PNG files** (including 512×512, 1024×1024, dark navy, blueprint, high contrast, and diagnostic answer overlays).

---

## 4. Discrepancy Reconciliation Note

- **350 Tasks (Committed Benchmark Splits)**: The actual, physical, frozen evaluation dataset partitioned into `dev` (70), `val` (70), and `test` (210) with verified ground-truth answer keys and zero test leakage.
- **1,500 Tasks (Factory Synthetic Capacity)**: References the full scaling throughput capacity of the `SceneGenerator` and `TaskExplorer` procedural factory under 50-seed condition sweeps. In all documents, 1,500 is now explicitly labeled as the **synthetic factory generation capacity target**, while 350 is the **authoritative frozen competition benchmark**.
