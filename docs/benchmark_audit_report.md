# MIRRORLAB Benchmark Dataset Audit & Integrity Report

**Audit Status:** PASSED | **Total Tasks:** 230 | **Counterfactual Pairs:** 120 | **Individual Scenes:** 350 | **Deterministic Seeds:** 350 | **Evaluated Images:** 350 | **Inspected Prompts:** 350

## 1. Executive Summary & Audit Checklist

| # | Audit Check Name | Status | Items Checked | Result Summary |
|---|---|---|---|---|
| 1 | **ID Uniqueness** | PASS | 470 | All 470 IDs are unique. |
| 2 | **Deterministic Seed Recording** | PASS | 350 | All 350 scenes (including both counterfactual child scenes) record valid integer seeds. |
| 3 | **Oracle Mathematical Agreement** | PASS | 50 | All 50 visibility and OOD scenes 100% match oracle and independently certified (Fermat residual < 1e-5, Delta_R < 1e-4 mm). |
| 4 | **Blind Image Non-Leakage** | PASS | 350 | All 350 blind diagrams verified: zero overlay rays, zero badge color leakage. |
| 5 | **Prompt Neutrality & Zero-Leakage** | PASS | 350 | All 350 prompts and questions verified neutral across all suites. |
| 6 | **Task C Geometric Reconstruction & Single-Winner Uniqueness** | PASS | 30 | All 30 repair tasks geometrically evaluated: exactly ONE valid winner per task. |
| 7 | **Task B Single-Blocker Clearance Precision** | PASS | 30 | All 30 obstruction tasks strictly enforce single-blocker precision. |
| 8 | **Task D Geometric Recomputation & Relationship Satisfaction** | PASS | 120 | All 120 pairs recomputed independently; 100% satisfy expected relationships. |
| 9 | **Anti-Duplicate Geometry Verification** | PASS | 350 | Zero duplicate configurations detected across 350 scenes. |
| 10 | **Test Split Partition Isolation** | PASS | 350 | Zero split leakage. Dev=70, Val=70, Blind Test=210 strictly isolated and aligned. |

## 2. Dataset Split Stratification & Physical Isolation

> **Security Guarantee:** Blind test answer keys are stored in `test/private/ground_truth.jsonl`. Model evaluation inference harnesses access ONLY `test/public/tasks.jsonl` and `test/public/images/`.

| Split | Tasks / Pairs | Percentage | Answer Key Access | Purpose |
|---|---|---|---|---|
| **Development (`dev`)** | 46 | 20.0% | Open (Public + Private) | Prompt engineering & diagnostic inspection |
| **Validation (`val`)** | 46 | 20.0% | Open (Public + Private) | Hyperparameter tuning & calibration |
| **Blind Test (`test`)** | 138 | 60.0% | **STRICTLY ISOLATED** | Official frozen benchmark evaluation |

## 3. Rejection & Ambiguity Statistics

During generation, the constraint-based generator enforces zero grazing, non-overlap, and physical validity:
- **Direct Sightline Clearances:** 100% of scenes guarantee direct sight $C \to T$ is blocked by barrier $O_1$.

- **Distractor Clearances:** $\ge 25.0\text{ mm}$ minimum clearance to reflection rays.

- **Grazing Intersections Rejected:** Ray-obstacle chords $< 8.0\text{ mm}$ were rejected during synthesis.

## 4. Auditor Certification Statement

All 10 benchmark criteria have been verified analytically. The dataset is certified free of answer leakage, duplicate geometry, and non-deterministic variations.
