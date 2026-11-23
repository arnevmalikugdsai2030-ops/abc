# MIRRORLAB Authoritative Dataset Inventory & Audit Report
**Generated:** 2026-11-23  
**Source Manifest:** `data/dataset_manifest.json`  
**Schema Version:** 2.0.0  
**Verification Status:** 100% Passed (11/11 Integrity Checks)  

---

## 1. Executive Summary

This report provides the single source of truth for all dataset assets committed to the MIRRORLAB repository. All statistics in this document are automatically computed from the physical filesystem by `scripts/generate_dataset_manifest.py`, eliminating manual discrepancies.

---

## 2. Benchmark Split Inventory (Model-Facing)

The headline competition benchmark consists of **350 model-facing tasks** across three partitions:

| Partition | Tasks | Ground-Truth Records | Counterfactual Pairs | Split Blind Images | Diagnostic Overlays | Ground-Truth Exposure |
|---|---|---|---|---|---|---|
| **`dev`** | 70 | 70 | 24 | 140 | 70 | Public (`ground_truth.jsonl`) |
| **`val`** | 70 | 70 | 24 | 140 | 70 | Public (`ground_truth.jsonl`) |
| **`test`** | 210 | 210 | 0 | 210 | 210 | **Sealed** (`private/ground_truth.jsonl`) |
| **Total** | **350** | **350** | **48** | **490** | **350** | **Zero leakage to test split** |

### Cryptographic Split Hashes (SHA-256):
- `dev_tasks_sha256`: `a6b5f233e45d833f9c0cd40c1fa682feee100910959eae2f3d4506ed8aeb41f6`
- `dev_ground_truth_sha256`: `ab5fc327ee7a8d46012c79340cc384ced00ede2c93fa185e4e4b60c7c2cca9db`
- `dev_pairs_sha256`: `1149b6f1a386bf58fa75a8190cc2b75b224ba9b35e1587af9c30f5904098fdcf`
- `val_tasks_sha256`: `f25b821f8e271030909c26344788aee318bf01983227697e2a599c36d980994a`
- `val_ground_truth_sha256`: `b88819168f7ecc7761e4d773a96d5fc45311379a420f406aaed4d7537c2e779f`
- `val_pairs_sha256`: `2951638722fe0ac427034ca1729cccdd0fbe1a82fe86dffa73cdc1cd03e6fa4e`
- `test_tasks_sha256`: `8c767cf5929f138fab2181771e468f4c958e6bb24aa51ceaea5a5505dbe8b4ab`
- `test_ground_truth_sha256`: `0d3a84273cb4f11856a24e48408a2778dae42080bbcd9a3890717959fd5cfd0a`

---

## 3. Core Task Repository Directories (`data/`)

In addition to the partitioned evaluation splits, `data/` houses **230 curated task suites** representing individual procedural generations:

| Directory | Task Family | Description | Task Suites | Individual Scenes |
|---|---|---|---|---|
| `data/benchmark_tasks/` | Task A | Baseline Specular Visibility | 20 | 20 |
| `data/obstruction_tasks/` | Task B | Single-Blocker & Leg Attribution | 30 | 30 |
| `data/repair_tasks/` | Task C | Active Specular Alignment & Repair | 30 | 30 |
| `data/counterfactual_pairs/`| Task D | Minimal Counterfactual Pairs | 120 | 240 (2 per pair) |
| `data/ood_tasks/` | Task E | Extreme Grazing & High Clutter OOD | 30 | 30 |
| **Total Core Task Dirs** | | | **230** | **350 Scenes** |

**Total Rendered Images on Disk:** **2,000 PNG files** (including 512×512, 1024×1024, dark navy, blueprint, high contrast, and diagnostic answer overlays).

---

## 4. Discrepancy Reconciliation Note

- **350 Tasks (Committed Benchmark Splits)**: The actual, physical, frozen evaluation dataset partitioned into `dev` (70), `val` (70), and `test` (210) with verified ground-truth answer keys and zero test leakage.
- **1,500 Tasks (Factory Synthetic Capacity)**: References the full scaling throughput capacity of the `SceneGenerator` and `TaskExplorer` procedural factory under 50-seed condition sweeps. In all documents, 1,500 is now explicitly labeled as the **synthetic factory generation capacity target**, while 350 is the **authoritative frozen competition benchmark**.
