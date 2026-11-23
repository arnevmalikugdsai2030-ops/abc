# MIRRORLAB: Hackathon Submission Guide & Judge Walkthrough
## Innodata Frontier AI Hackathon 2026 | Track 2: Visual Task Factory

**Document Version:** 1.2.0  
**Project Name:** MIRRORLAB — The Missing Scratch  
**Assessment:** Transparent Evidence-Based Rubric Assessment (No Self-Scoring)  
**Repository:** https://github.com/arnevmalikugdsai2030-ops/abc  

---

## 1. The 30-Second Elevator Pitch

> *"Frontier Vision-Language Models score 85%+ on standard visual benchmarks, but do they understand physical optical reflection, or are they relying on shallow spatial proximity heuristics? In advanced manufacturing, cameras cannot see into dark internal cavities, relying on deflection mirrors to detect microscopic scratch defects.*
>
> *MIRRORLAB is an industrial synthetic visual-task factory that generates parametric 2D inspection schematics. We calculate ground truth using a triple-redundant geometric engine (Virtual Target reflection, Fermat's Least-Time Principle, and Direct Snell-Vector Collinearity) with zero LLM hallucinations. By introducing minimal counterfactual pairs ($\Delta \theta = 2^\circ$), we evaluate whether models exhibit consistent geometric understanding or collapse on spatial reasoning, exposing a critical challenge in multimodal spatial intelligence."*

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
2. **Counterfactual Collapse:** Observe the joint pair consistency metric drop from $75\%$ to $45\%$ under calibrated simulation.
3. **8-Condition Visual Necessity Audit:** Inspect Section 8 certifying that the benchmark ablation achieves a **$+37.14\%$ Visual Necessity Ratio (VNR)** under simulation and zero filename/metadata leakage.

---

## 3. Evidence-Based Rubric Assessment

In accordance with scientific integrity standards, MIRRORLAB does **not** self-award points. Instead, we present an evidence-based breakdown categorizing every project capability by its concrete verification tier:

### 3.1 Subsystem Verification Matrix

| Subsystem / Capability | Verification Status | Primary Evidence & Files |
|---|---|---|
| **Synthetic Geometric Optics Engine** | **Mathematically Verified & Unit-Tested** | Triply-redundant consensus: Virtual Target (`oracle.py`), Fermat Least-Time (`independent_verifier.py`), Snell Root-Finding (`ray_march_verifier.py`). Verified to residual $< 10^{-4}\text{ mm}$. |
| **Parametric Task Factory (Tasks A–E)** | **Implemented & Integration-Tested** | Automated generator produces 5 task families with Poisson obstacles, variable apertures, and multi-theme rendering (`task_generator.py`). |
| **Counterfactual Pair Engine (4 Types)** | **Implemented & Unit-Tested** | Minimal perturbations: angle flips ($\Delta \theta = 2^\circ$), cosmetic controls, obstacle shifts, and irrelevant distractors (`counterfactual.py`, `pair_evaluator.py`). |
| **Blind-Split Integrity & Non-Leakage** | **Empirically Measured & Audited** | Public splits use non-semantic SHA-256 derived hashes; filenames and IDs leak zero ground truth (`filename_only` condition yields 50.0% guessing accuracy; `dataset_auditor.py`). |
| **Automated Dataset Auditor** | **Unit-Tested & Integration-Tested** | 11-point mathematical auditor verifying obstacle clearances, boundary tolerances, and counterfactual pair integrity (`docs/benchmark_audit_report.md`). |
| **Structured Output Parsing Engine** | **Unit-Tested (4-Pass Fallback)** | Robust extraction handling strict JSON, markdown code blocks, outermost braces, and regex fallback (`structured_parser.py`). |
| **8-Condition Shortcut Ablation Engine** | **Simulated & Unit-Tested** | Controlled multimodal ablations evaluating VNR, LPB, CIS, GS, and identifier neutrality (`shortcut_evaluator.py`, `docs/bias_and_shortcut_audit.md`). |
| **Offline Calibrated VLM Simulation** | **Simulated Baseline** | Deterministic simulation of spatial optics error modes for offline pipeline testing; clearly marked with `is_mock: True` and `CALIBRATED_SIMULATION`. |
| **Live VLM API Integration** | **Implemented & Integration-Tested** | Official SDK adapters for Gemini 2.5 (`google-genai`), OpenAI GPT-4o, and Anthropic Claude with exponential backoff and token telemetry. Live API execution requires user-provided keys (`.env`). |
| **High-Density Streamlit Dashboard** | **Implemented & AppTest Verified** | Interactive optical laboratory, blind model arena, and scientific analytics dashboard tested with Streamlit AppTest (`mirrorlab/ui/app.py`). |
| **Tabletop Hardware Metrology Protocol** | **Proposed Experimental Design (Not Yet Validated on Physical Hardware)** | Formal 1:1 millimeter bench design, 10 proposed scenarios, homography calibration, and error budget documented in `docs/real_world_validation.md`. Hardware trials have not yet been conducted. |

### 3.2 Track 2 Rubric Alignment Summary

1. **Working Implementation & Repeatability:**
   - Single-command dataset generation (`python scripts/rebuild_benchmark_dataset.py`).
   - Clean CI test pass across Python 3.11 and Python 3.12 without developer log dependencies.
   - Automated code formatting and linting (`ruff check .`, `ruff format --check .`).
2. **Validity & Validation Evidence:**
   - Triple-redundant mathematical consensus on every ray calculation.
   - Comprehensive 11-point dataset audit (`docs/benchmark_audit_report.md`).
   - 14 automated corruption and boundary tests in integration suite.
3. **Meaningful Variation:**
   - 5 task families (Visibility, Obstruction, Repair, Counterfactual Pairs, OOD Clutter).
   - 4 distinct counterfactual pair types isolating optical geometry from cosmetic appearance.
4. **Originality & Engineering Judgment:**
   - Formulates cavity inspection with reflection geometry as a frontier VLM testbed.
   - Eliminates 50% binary guessing bias via paired counterfactual consistency.
   - Air-gapped public evaluation preventing semantic identifier leakage.
5. **Practical Impact & Honest Communication:**
   - Clear sim-to-real transferability analysis with explicit disclosure of 2D model limitations.
   - High-density observability UI with live parameter exploration.
   - Full disclosure distinguishing mathematical simulation from live API and hardware trials.

---

## 4. Master Document Index (12 Authoritative Documents)

1. [README.md](../README.md) — Master Repository Hub & Overview
2. [docs/architecture.md](architecture.md) — System Architecture & Modular Pipeline
3. [docs/methodology.md](methodology.md) — Mathematical Foundations & Physical Optics Proofs
4. [docs/dataset_card.md](dataset_card.md) — Dataset Card & Inventory (230 packages, 350 tasks, 700 images)
5. [docs/benchmark_audit_report.md](benchmark_audit_report.md) — Mathematical Audit Report (11/11 Checks Passed)
6. [docs/model_evaluation_report.md](model_evaluation_report.md) — Multi-Model Frontier Evaluation & Failure Taxonomy
7. [docs/counterfactual_report.md](counterfactual_report.md) — Counterfactual Sensitivity & OOD Stress Analysis
8. [docs/bias_and_shortcut_audit.md](bias_and_shortcut_audit.md) — 8-Condition Visual Necessity & Leakage Audit
9. [docs/real_world_validation.md](real_world_validation.md) — Tabletop Optical Rig Metrology & 10 Physical Scenarios
10. [docs/reproducibility.md](reproducibility.md) — Single-Command Reproduction & Offline Verification Guide
11. [docs/project_status.md](project_status.md) — Production Health, Test Distribution, & CI Status
12. [docs/submission_guide.md](submission_guide.md) — Hackathon Judge Walkthrough & Rubric Assessment

---
*Prepared for the Innodata Frontier AI Hackathon 2026 Evaluation Committee.*
