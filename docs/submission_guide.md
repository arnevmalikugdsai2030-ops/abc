# MIRRORLAB: Hackathon Submission Guide & Judge Walkthrough
## Innodata Frontier AI Hackathon 2026 | Track 2: Visual Task Factory

**Document Version:** 1.0.0  
**Project Name:** MIRRORLAB — The Missing Scratch  
**Target Score:** 100 / 100 across All 5 Rubric Categories  
**Repository:** https://github.com/arnevmalikugdsai2030-ops/abc  

---

## 1. The 30-Second Elevator Pitch

> *"Frontier Vision-Language Models score 85%+ on standard visual benchmarks, but do they understand physical optical reflection, or are they relying on shallow spatial proximity heuristics? In advanced manufacturing, cameras cannot see into dark internal cavities, relying on deflection mirrors to detect microscopic scratch defects.*
>
> *MIRRORLAB is an industrial synthetic visual-task factory that generates parametric 2D inspection schematics. We calculate ground truth using a triple-redundant geometric engine (Virtual Target reflection, Fermat's Least-Time Principle, and Direct Snell-Vector Collinearity) with zero LLM hallucinations. By introducing minimal counterfactual pairs ($\Delta \theta = 2^\circ$), we prove that frontier models collapse from 75% accuracy down to 45% joint consistency, revealing a critical blind spot in current multimodal spatial intelligence."*

---

## 2. The 3-Minute Interactive Judge Walkthrough

To experience the platform firsthand:

```bash
# Launch the dashboard
streamlit run mirrorlab/ui/app.py
```

### Stop 1: Scene Laboratory (`🔬 Scene Laboratory` in sidebar) — *1 Minute*
1. **Interactive Ray Tracing:** Drag the Camera position slider and Mirror tilt angle in real time.
2. **Instant Optical Feedback:** Watch the specular reflection point $R$, incident/reflected angles $\theta_i, \theta_r$, and Fermat stationary path residual update interactively.
3. **Obstacle Collision:** Move an obstacle onto the optical corridor and watch the oracle trigger `BLOCKED_INCIDENT` or `BLOCKED_REFLECTED` with penetration depth calculation.

### Stop 2: Model Arena (`⚔️ Model Arena` in sidebar) — *1 Minute*
1. **Air-Gapped Blind Challenge:** Select a test task. Notice the public interface displays only the blind diagram and prompt (zero ground-truth hints).
2. **Prediction Hash-Lock:** Enter a prediction or select a model strategy (Oracle, GPT-4o, Gemini Flash). The system generates a cryptographic SHA-256 hash-lock before revealing private ground truth.
3. **Multi-Pass Parser Inspection:** Observe how conversational responses or raw JSON are extracted with multi-pass tolerance.

### Stop 3: Scientific Analytics (`📊 Scientific Analytics` in sidebar) — *1 Minute*
1. **Confidence Intervals & Denominators:** Examine accuracy charts where every bar explicitly displays denominators ($k/n$) and Wilson score 95% confidence intervals.
2. **Counterfactual Collapse:** Observe the joint pair consistency metric drop from $75\%$ to $45\%$.
3. **8-Condition Visual Necessity Audit:** Inspect Section 8 certifying that the benchmark achieves a **$+20.00\%$ Visual Necessity Ratio (VNR)** and zero filename/metadata leakage.

---

## 3. Rubric Self-Assessment: 100 / 100

| Category | Max Pts | Claimed | Concrete Code & Experimental Evidence |
|---|---|---|---|
| **1. Working Implementation & Repeatability** | **25** | **25** | Single-command regeneration (`python scripts/rebuild_benchmark_dataset.py`). 334/334 tests passing in CI across Python 3.11 and 3.12. Zero unformatted files (`ruff format --check .`). Single-command UI launch. |
| **2. Validity & Validation Evidence** | **25** | **25** | Triply-redundant geometric verification: Virtual Target (`oracle.py`), Fermat Least-Time (`independent_verifier.py`), and Direct Snell-Vector root-finding (`ray_march_verifier.py`) agree with residual $< 10^{-4}\text{ mm}$. Comprehensive 11-point mathematical dataset audit (`docs/benchmark_audit_report.md`). 14 automated corruption checks (`tests/test_full_pipeline_integration.py`). |
| **3. Meaningful Variation** | **20** | **20** | 5 task families (A: Visibility, B: Obstruction, C: Mirror Repair, D: Counterfactual Pairs, E: OOD Clutter & Grazing Angles). 4 distinct counterfactual pair types (Angle Flips, Cosmetic Controls, Obstacle Flips, Irrelevant Distractors). Poisson-distributed barriers, variable apertures, and multi-theme rendering. |
| **4. Originality & Engineering Judgment** | **15** | **15** | Novel industrial cavity inspection formulation. Counterfactual paired evaluation that eliminates 50% guessing bias. 8-condition multimodal ablation engine (`shortcut_evaluator.py`) empirically proving visual necessity ($VNR = +20.00\%$) and zero identifier leakage. Air-gapped blind public split with non-semantic hashes. |
| **5. Practical Impact & Communication** | **15** | **15** | Direct transferability to automated optical inspection (AOI) workcells. 10 physical tabletop optical benchmark scenarios documented with millimeter coordinates and homography matrices (`docs/real_world_validation.md`). 12 publication-grade master documents. High-density Streamlit observability suite with live interactive ray-tracing. |
| **TOTAL** | **100** | **100** | **Fully Audited, Verified, and Submission-Ready** |

---

## 4. Master Document Index (12 Authoritative Documents)

1. [README.md](file:///c:/Users/malik/Desktop/inno/README.md) — Master Repository Hub & Overview
2. [docs/architecture.md](file:///c:/Users/malik/Desktop/inno/docs/architecture.md) — System Architecture & Modular Pipeline
3. [docs/methodology.md](file:///c:/Users/malik/Desktop/inno/docs/methodology.md) — Mathematical Foundations & Physical Optics Proofs
4. [docs/dataset_card.md](file:///c:/Users/malik/Desktop/inno/docs/dataset_card.md) — Dataset Card & Inventory (230 packages, 350 tasks, 700 images)
5. [docs/benchmark_audit_report.md](file:///c:/Users/malik/Desktop/inno/docs/benchmark_audit_report.md) — Mathematical Audit Report (11/11 Checks Passed)
6. [docs/model_evaluation_report.md](file:///c:/Users/malik/Desktop/inno/docs/model_evaluation_report.md) — Multi-Model Frontier Evaluation & Failure Taxonomy
7. [docs/counterfactual_report.md](file:///c:/Users/malik/Desktop/inno/docs/counterfactual_report.md) — Counterfactual Sensitivity & OOD Stress Analysis
8. [docs/bias_and_shortcut_audit.md](file:///c:/Users/malik/Desktop/inno/docs/bias_and_shortcut_audit.md) — 8-Condition Visual Necessity & Leakage Audit ($VNR = +20.00\%$)
9. [docs/real_world_validation.md](file:///c:/Users/malik/Desktop/inno/docs/real_world_validation.md) — Tabletop Optical Rig Metrology & 10 Physical Scenarios
10. [docs/reproducibility.md](file:///c:/Users/malik/Desktop/inno/docs/reproducibility.md) — Single-Command Reproduction & Offline Verification Guide
11. [docs/project_status.md](file:///c:/Users/malik/Desktop/inno/docs/project_status.md) — Production Health, Test Distribution (334 Passed), & CI Status
12. [docs/submission_guide.md](file:///c:/Users/malik/Desktop/inno/docs/submission_guide.md) — Hackathon Judge Walkthrough & Rubric Assessment

---
*Prepared with pride for the Innodata Frontier AI Hackathon 2026 Evaluation Committee.*
