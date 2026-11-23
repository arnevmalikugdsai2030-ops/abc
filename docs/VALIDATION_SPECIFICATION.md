# MIRRORLAB: Multi-Layer Benchmark Validation Specification
## Formal Verification Architecture, Mathematical Consensus & Rendering Fidelity

**Document Version:** 1.0.0  
**Status:** Authoritative Validation Standard  
**Project:** MIRRORLAB — The Missing Scratch  
**Hackathon:** Innodata Frontier AI Hackathon 2026 | Track 2: Visual Task Factory  
**Last Updated:** November 23, 2026  

---

## 1. Overview of the Multi-Layer Validation Pipeline

To ensure the highest scientific integrity and eliminate all LLM hallucination and labeling ambiguity, MIRRORLAB implements a **7-tier modular validation architecture**:

```
+---------------------------------------------------------------------------------+
|                         MIRRORLAB VALIDATION ARCHITECTURE                       |
+---------------------------------------------------------------------------------+
| Layer 1: Schema Validation (Pydantic V2 immutable models, frozen schemas)       |
|    v                                                                            |
| Layer 2: Data-First Generation (Structured scene.json as source of truth)       |
|    v                                                                            |
| Layer 3: Triple-Redundant Independent Solvers (Virtual Target, Fermat, Snell)   |
|    v                                                                            |
| Layer 4: Geometric Ambiguity & Margin Screening (Endpoint & obstacle buffers)   |
|    v                                                                            |
| Layer 5: Rendering-to-Data Pixel Probes (Centroid, marker chrominance, shapes)  |
|    v                                                                            |
| Layer 6: Air-Gap Blind Non-Leakage Auditing (SHA-256 hashes, zero prompt keys)  |
|    v                                                                            |
| Layer 7: Replay & Re-generation Determinism (Seed invariance across runs)       |
+---------------------------------------------------------------------------------+
```

---

## 2. Layer 1 — Schema Validation

Every task, scene, and ground-truth record is validated against strict Pydantic v2 schemas (`ConfigDict(frozen=True)`):

| Artifact | Schema Model | Required Validated Fields |
|---|---|---|
| **Inspection Scene** | `InspectionScene` | `scene_id`, `seed`, `category`, `camera`, `mirror`, `target`, `obstacles`, `ground_truth_state`, `ground_truth_reason` |
| **Public Task** | `VisibilityQuestion` | `task_id`, `task_type`, `question`, `prompt`, `image_filename`, `binary_options`, `reason_options` |
| **Private Ground Truth** | `VisibilityGroundTruth` | `task_id`, `internal_task_id`, `can_see`, `answer`, `reason_code`, `detailed_explanation`, `proof_geometry` |
| **Counterfactual Pair**| `CounterfactualPair` | `pair_id`, `task_a_id`, `task_b_id`, `pair_type`, `parameter_delta`, `expected_consistency` |

Any record missing required fields, containing extra answer-leaking fields, or violating data types fails validation immediately.

---

## 3. Layer 2 — Data-First Architecture

MIRRORLAB maintains strict separation between abstract geometry, rendered images, and model queries:
1. **Source of Truth:** The structured 2D scene (`scene.json`) is generated and validated *first*.
2. **Deterministic Rendering:** The raster diagram (`diagram_blind.png`) is produced by an isometric viewport projection (`ViewportTransform`) directly from the structured scene.
3. **Independent Solution:** The ground truth is calculated purely from floating-point vectors, completely independent of the rendered image or any model inference.

---

## 4. Layer 3 — Triple-Redundant Independent Solver Verification

Every scene is audited by three distinct mathematical engines operating on different physical principles:

| Engine | Primary Principle | Mathematical Formulation | Independence Guarantee |
|---|---|---|---|
| **Method 1: Oracle** (`oracle.py`) | Virtual Target Reflection | Householder affine transformation reflecting $T \to T'$ across mirror line: $T' = T - 2( (T - A) \cdot \mathbf{n} ) \mathbf{n}$. Intersects ray $C \to T'$ with mirror segment $[A, B]$. | Primary reference solver. |
| **Method 2: Fermat** (`independent_verifier.py`) | Principle of Least Time | Reflects Camera $C \to C'$ across mirror line. Intersects ray $C' \to T$ with segment $[A, B]$. Verifies stationary path length $\|C - R\| + \|R - T\|$. | Inverts reflection reference from Target to Camera; independent line intersection. |
| **Method 3: Snell** (`ray_march_verifier.py`) | Direct Snell-Vector Collinearity | Solves $E(t) = \mathbf{u}_{\text{refl}}(t) \times \mathbf{u}_{\text{tgt}}(t) = 0$ over $t \in [0, 1]$ using Brent's method root-finding ($10^{-7}\text{ mm}$ tolerance). | **Does not use virtual points.** Directly root-finds surface normal angle equality. |

### Consensus & Tolerance Rules
- **Binary Agreement:** All 3 engines must agree on `state` (`VISIBLE` vs `NOT_VISIBLE`).
- **Reason Agreement:** All 3 engines must agree on obstruction reason code.
- **Euclidean Reflection Residual:** Pairwise distance between reflection points:
  $$\max(\|R_{\text{oracle}} - R_{\text{fermat}}\|, \|R_{\text{oracle}} - R_{\text{snell}}\|, \|R_{\text{fermat}} - R_{\text{snell}}\|) \le 10^{-4}\text{ mm}$$
- **Measured Empirical Residual:** Benchmarked maximum residual across dataset is **$2.66 \times 10^{-5}\text{ mm}$**, well within tolerance.

---

## 5. Layer 4 — Geometric Non-Ambiguity Screening

Borderline or degenerate geometric configurations are rejected by `AmbiguityValidator`:

| Ambiguity Rule | Tolerance Threshold | Rationale |
|---|---|---|
| **Mirror Endpoint Buffer** | $\ge 2.0\text{ mm}$ | Prevents sub-pixel aperture boundary grazing ambiguity where raster aliasing could obscure whether a ray clips or clears. |
| **Obstacle Corridor Clearance**| $\ge 2.0\text{ mm}$ | Prevents borderline grazing near-misses where optical wave diffraction could blur ray boundaries. |
| **Obstacle Penetration Depth** | $\ge 1.5\text{ mm}$ | Ensures that whenever an obstacle blocks a ray, the optical intersection is decisive. |
| **Camera FOV Boundary Buffer** | $\ge 0.5^\circ$ | Eliminates edge-of-cone optical truncation ambiguity. |
| **Inter-Object Spatial Clearance**| $\ge 15.0\text{ mm}$ | Prevents overlapping visual symbols or labels in the diagram. |

---

## 6. Layer 5 — Rendering-to-Data Consistency Validation

To guarantee that models are evaluated on diagrams that accurately depict scene geometry, `RenderingConsistencyValidator` performs automated pixel probes on rendered images:
1. **Target Scratch Chrominance:** Probes pixel window around transformed target coordinate $(X_T, Y_T)$, verifying prominent red marker pixels ($R > 140, G < 100, B < 100$).
2. **Camera Apparatus:** Verifies camera body presence ($B > 120, R < 120$) at transformed camera position $(X_C, Y_C)$.
3. **Mirror Surface:** Verifies cyan reflective line pixels along the mirror segment midpoint.
4. **Obstacle Alignment:** Confirms each obstacle's bounding box maps to non-background pixels.
5. **Zero Diagnostic Overlay:** In blind evaluation images, probes the optical corridor to verify zero green/magenta ray traces are present.

---

## 7. Layer 6 — Blind Non-Leakage & Air-Gap Verification

To preserve blind benchmark evaluation integrity:
1. **Non-Semantic Hash IDs:** Public task IDs use neutral identifiers (`ml_dev_0001`, `ml_test_0142`) derived from SHA-256 hashes of seed and task index.
2. **Zero Semantic Metadata:** Image filenames (`img_dev_0001.png`) leak zero information about visibility, reason codes, or repair winners.
3. **Private Split Air-Gap:** Ground-truth files (`ground_truth.jsonl`, `pairs.jsonl`) reside exclusively in evaluator-only directories and are never passed to model adapters.
4. **Ablation Proof:** The `filename_only` multimodal ablation condition yields **$50.0\%$ accuracy** (pure chance), proving identifiers contain zero detectable answer signal.

---

## 8. Layer 7 — Reproducibility & Checksum Hashing

Given the same random seed and generator version:
- Structured scene specifications regenerate identically.
- Rendered images match pixel-for-pixel with matching SHA-256 image hashes.
- Ground truth recomputes with identical 64-bit float precision.

---
*Maintained by the MIRRORLAB Scientific Core & Quality Engineering Team.*
