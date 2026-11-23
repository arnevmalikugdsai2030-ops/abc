# MIRRORLAB Benchmark Dataset Audit & Integrity Report

**Audit Status:** PASSED (Verified Scopes Only) | **Total Tasks:** 230 | **Counterfactual Pairs:** 120 | **Individual Scenes:** 350 | **Deterministic Seeds:** 350 | **Evaluated Images:** 350 | **Inspected Prompts:** 350

## 1. Executive Summary & Audit Checklist

| # | Audit Check Name | Verification Level | Status | Items Checked | Result Summary |
|---|---|---|---|---|---|
| 1 | **ID Uniqueness** | `[STRUCTURALLY_VERIFIED]` | PASS | 470 | All 470 IDs are unique. |
| 2 | **Deterministic Seed Recording** | `[STRUCTURALLY_VERIFIED]` | PASS | 350 | All 350 scenes (including both counterfactual child scenes) record valid integer seeds. |
| 3 | **Oracle Mathematical Agreement** | `[ANALYTICALLY_VERIFIED]` | PASS | 50 | All 50 visibility and OOD scenes 100% match oracle and independently certified (Fermat residual < 1e-5, Delta_R < 1e-4 mm). |
| 4 | **Blind Image Non-Leakage** | `[HEURISTICALLY_SCREENED]` | PASS | 350 | All 350 blind diagrams verified: zero overlay rays, zero badge color leakage. |
| 5 | **Prompt Neutrality & Zero-Leakage** | `[HEURISTICALLY_SCREENED]` | PASS | 350 | All 350 prompts and questions verified neutral across all suites. |
| 6 | **Task C Geometric Reconstruction & Single-Winner Uniqueness** | `[ANALYTICALLY_VERIFIED]` | PASS | 30 | All 30 repair tasks geometrically evaluated: exactly ONE valid winner per task. |
| 7 | **Task B Single-Blocker Clearance Precision** | `[ANALYTICALLY_VERIFIED]` | PASS | 30 | All 30 obstruction tasks strictly enforce single-blocker precision. |
| 8 | **Task D Geometric Recomputation & Relationship Satisfaction** | `[ANALYTICALLY_VERIFIED]` | PASS | 120 | All 120 pairs recomputed independently; 100% satisfy expected relationships. |
| 9 | **Anti-Duplicate Geometry Verification** | `[STATISTICALLY_MEASURED]` | PASS | 350 | Zero duplicate configurations detected across 350 scenes. |
| 10 | **Test Split Partition Isolation** | `[STRUCTURALLY_VERIFIED]` | PASS | 350 | Zero split leakage. Dev=70, Val=70, Blind Test=210 strictly isolated, manifest-consistent, and aligned. |
| 11 | **Counterfactual Split Pair Integrity** | `[STRUCTURALLY_VERIFIED]` | PASS | 120 | All 120 counterfactual pairs across dev, val, and test partitions strictly verified: 100% child task alignment, mutual exclusivity, expected relationship satisfaction, and zero public leakage. |

## 2. Dataset Split Stratification & Physical Isolation

> **Security Guarantee:** Blind test answer keys are stored exclusively in `test/private/ground_truth.jsonl`. Model evaluation inference harnesses access ONLY `test/public/tasks.jsonl` and `test/public/images/`.

### Disaggregated Benchmark Accounting

| Split Partition | Top-Level Packages | Counterfactual Pairs | Model-Facing Tasks | Private GT Records | Public / Private Images | Share |
|---|---|---|---|---|---|---|
| **Dev (`dev`)** | 46 | 24 | 70 | 70 | 70 blind / 70 overlay | 20.0% |
| **Val (`val`)** | 46 | 24 | 70 | 70 | 70 blind / 70 overlay | 20.0% |
| **Test (`test`)** | 138 | 72 | 210 | 210 | 210 blind / 210 overlay | 60.0% |
| **TOTAL** | **230** | **120** | **350** | **350** | **700 total** | **100.0%** |

## 3. Measured Geometric Integrity & Physical Validity Statistics

All physical and geometric statistics are dynamically measured from actual scene geometry on disk (N=350):

| Metric Name | Measured Value | Ratio / Sample Count | Failures | Units | Validation Method | Verification Category |
|---|---|---|---|---|---|---|
| **Direct Sightline Obstruction Rate** | 100.0% | 350 / 350 scenes | 0 | % | Segment-obstacle intersection testing (Camera -> Target) | `[ANALYTICALLY_VERIFIED]` |
| **Distractor Obstacle Clearance (Min)** | 2.9 mm | 848 obstacle-ray pairs (306 scenes) | 0 | mm | Euclidean boundary-to-ray distance along valid path (C -> R -> T) | `[ANALYTICALLY_VERIFIED]` |
| **Distractor Obstacle Clearance (Mean)** | 192.7 mm | 848 obstacle-ray pairs (306 scenes) | 0 | mm | Euclidean boundary-to-ray distance along valid path (C -> R -> T) | `[ANALYTICALLY_VERIFIED]` |
| **Fermat Optical Path Stationary Residual (Max)** | 6.66e-16 | 306 valid reflection paths | 0 | dimensionless | Tangential projection residual $|\mathbf{u}_{\text{inc}} \cdot \mathbf{t} - \mathbf{u}_{\text{refl}} \cdot \mathbf{t}| = |\Delta \sin \theta|$ | `[ANALYTICALLY_VERIFIED]` |
| **Fermat Optical Path Stationary Residual (Mean)** | 1.65e-16 | 306 valid reflection paths | 0 | dimensionless | Tangential projection residual $|\mathbf{u}_{\text{inc}} \cdot \mathbf{t} - \mathbf{u}_{\text{refl}} \cdot \mathbf{t}| = |\Delta \sin \theta|$ | `[ANALYTICALLY_VERIFIED]` |
| **Physical-World Sensor & Material Calibration** | N/A | 0 / 0 physical tests | 0 | N/A | Empirical physical laboratory experiment | `[NOT_VERIFIED]` |

> **Notes on Geometric Measurements:**
- Direct Camera-to-Target obstruction verified by geometric intersection testing (direct sightline occluded in 100.0% of scenes; indirect specular reflection required).
- Distractor clearance is measured in millimetres (`mm`) across all eligible non-intersecting obstacles along valid reflection paths.
- Fermat residual is a **dimensionless** projection residual ($|\Delta \sin \theta|$), representing numerical stationarity of the optical path.

## 4. Verification Scopes, Evidence & Scientific Limitations

> **AUDIT SCOPE:** All executed checks in the 11-point suite have passed within their defined analytical, structural, or heuristic scopes.

### Verification Status by Tier

1. **Structural & Completeness Validation:** `[STRUCTURALLY_VERIFIED]` — 100% of required dataset artifacts (scene models, questions, prompts, ground truth, images) exist, conform to schema, and contain valid IDs and seeds.
2. **Ground-Truth Consistency:** `[ANALYTICALLY_VERIFIED]` — Every ground-truth label verified by the dual-engine analytical oracle and independent ray verifier.
3. **Geometric Integrity:** `[ANALYTICALLY_VERIFIED]` — Finite mirror clipping, normal orientation, camera FOV cone checks, and obstacle occlusions analytically recomputed.
4. **Answer-Leakage Screening:** `[HEURISTICALLY_SCREENED]` — Public tasks, filenames, metadata, and prompts scanned for forbidden fields, outcome tokens, and declarative assertions.
5. **Counterfactual Relationship Validation:** `[ANALYTICALLY_VERIFIED]` — 100% of counterfactual pairs verified for opposite (FLIP) or identical (INVARIANT) child outcomes under controlled parameter perturbations.
6. **Seed Reproducibility:** `[STRUCTURALLY_VERIFIED]` — Re-synthesis from identical integer seeds reproduces identical scene geometry and task ground truth.
7. **Pixel-Level Image Determinism:** `[STATISTICALLY_MEASURED]` — Pillow rasterization confirmed identical under fixed configuration and identical task identifiers.
8. **Physical-World Validation:** `[NOT_VERIFIED]` — MIRRORLAB is a synthetic 2D geometric optics benchmark. Physical sensor calibration, real-world camera noise, and physical material BRDF reflectance are **NOT VERIFIED**.

> **Adversarial Security & Automated Screening Notice:**
> Automated leakage screening enforces an allowlist public schema and scans all nested structures for outcome tokens and declarative answer statements. However, heuristic pattern matching cannot mathematically prove the absence of every conceivable linguistic hint; external audit and human review remain necessary for complete evaluation integrity.
