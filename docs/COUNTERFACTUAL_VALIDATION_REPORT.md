# MIRRORLAB Counterfactual Pair Validation Report
**Dataset Suite:** Task D (Counterfactual Minimal Pairs)  
**Total Pairs Audited:** 120 (240 child scenes)  
**Audit Date:** 2026-11-23  
**Verification Status:** 100% Passed (All 6 Integrity Checks Verified)  

---

## 1. Executive Summary

Standard VQA and spatial benchmarks permit Vision-Language Models (VLMs) to achieve high apparent accuracy by exploiting surface heuristics and prior biases (such as guessing "VISIBLE" on 80% of test instances).

Task D eliminates this vulnerability through **Counterfactual Minimal Pairs**. Every task in Task D consists of two tightly coupled scenes ($S_A$ and $S_B$) differing by exactly one controlled physical intervention. Under the **Joint Consistency Metric**, a model receives a score of 1.0 for a pair if and only if it correctly answers **both** scenes according to the physical invariant:

$$\text{PairConsistency} = \mathbb{I}\left(R(\hat{y}_A, \hat{y}_B) \land (\hat{y}_A = y_A) \land (\hat{y}_B = y_B)\right)$$

An "All-Yes" or "All-No" heuristic immediately collapses to 0.0% accuracy on counterfactual flip pairs.

---

## 2. The 4 Minimal Perturbation Types

| Pair Type | Intervention Description | Physical Invariant | Expected Outcome |
|---|---|---|---|
| **Type 1: Mirror Angle Flip** | The planar mirror is rotated by a minimal $\pm 1.5^\circ$ to $2.0^\circ$. | Ray deflection angle shifts by $3.0^\circ - 4.0^\circ$, cleanly moving the beam from inside the finite mirror aperture to outside, or unblocking an occluded ray. | **`FLIPPED_OUTCOME`** ($y_A \ne y_B$) |
| **Type 2: Cosmetic Invariance Control** | Background grid noise, color palette, and line weight are randomized while holding all coordinates frozen. | Physical specular ray geometry is mathematically invariant ($C, M, T, O$ identical). | **`INVARIANT_OUTCOME`** ($y_A = y_B$) |
| **Type 3: Obstacle Insertion** | A convex opaque obstacle is inserted directly into the clear optical corridor. | Ray line is physically intersected by the obstacle boundary ($>1.5\text{ mm}$ penetration). | **`FLIPPED_OUTCOME`** ($y_A \ne y_B$) |
| **Type 4: Irrelevant Obstacle Insertion** | An obstacle is placed in the inspection cell far from the optical corridor ($>50\text{ mm}$ clearance). | Ray corridor remains completely unobstructed. | **`INVARIANT_OUTCOME`** ($y_A = y_B$) |

---

## 3. Empirical Audit Results Across 120 Committed Pairs

Audit performed across `data/counterfactual_pairs/` via `DatasetAuditor.audit_counterfactual_pairs()`:

| Verification Dimension | Items Audited | Passed | Success Rate | Details |
|---|---|---|---|---|
| **Unique Scene & Task IDs** | 360 IDs | 360 | **100.0%** | Zero collision across pair IDs, `scene_a`, and `scene_b`. |
| **Deterministic Seed Validity** | 240 seeds | 240 | **100.0%** | All child scenes record valid, non-default integer seeds. |
| **Geometric Recomputation** | 120 pairs | 120 | **100.0%** | Recomputed with Householder Oracle & Dual Fermat Verifier. |
| **Relationship Satisfaction** | 120 pairs | 120 | **100.0%** | 100% of pairs satisfy expected `FLIPPED` or `INVARIANT` outcome. |
| **Artifact Completeness** | 120 bundles | 120 | **100.0%** | All JSON specs, blind PNGs, and overlay PNGs present. |
| **Blind Non-Leakage Integrity** | 240 scenes | 240 | **100.0%** | Zero diagnostic ray lines or answer metadata leaked in blind PNGs. |
| **Prompt Neutrality** | 240 prompts | 240 | **100.0%** | All child prompts verified neutral with zero answer hints. |

---

## 4. Shortcut Resistance Proof (All-Yes vs Pair Consistency)

To mathematically demonstrate why counterfactual pairs defeat guessing shortcuts:

- **Baseline Task A (Single Question)**:
  - If a split has 60% Visible / 40% Obstructed, an "All-Yes" model scores $60.0\%$.
- **Task D (Counterfactual Pairs)**:
  - For $N$ flip pairs where $y_A = \text{VISIBLE}$ and $y_B = \text{NOT\_VISIBLE}$:
  - "All-Yes" model predicts: $\hat{y}_A = \text{VISIBLE}$, $\hat{y}_B = \text{VISIBLE}$.
  - Joint score: $\mathbb{I}(\hat{y}_A = \text{VISIBLE} \land \hat{y}_B = \text{NOT\_VISIBLE}) = 0$.
  - **The "All-Yes" shortcut score drops to exactly $0.0\%$**.

This guarantees that high scores on MIRRORLAB cannot be faked through binary response biases.
