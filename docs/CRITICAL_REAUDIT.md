# CRITICAL RE-AUDIT & SCIENTIFIC INTEGRITY ASSESSMENT
**Project:** MIRRORLAB — The Missing Scratch  
**Hackathon:** Innodata Frontier AI Hackathon 2026 — Track 2: Visual Task Factory  
**Audit Date:** 2026-11-23  
**Auditor:** Automated Benchmark & Scientific Quality Engineering  
**Head Commit Evaluated:** `4876ec694afa980482b9b99af54185883f015af5`  

---

## 1. Executive Summary of Audit Findings

A rigorous technical re-audit of the repository was conducted following the recent CI run `45115496094`. The audit revealed a combination of strong, reproducible geometric core systems alongside critical CI testing dependencies and serious scientific provenance overclaims that must be remediated immediately.

Specifically:
1. **CI Pytest Failure (6 Failed Tests)**: In fresh CI environments without local untracked `eval_logs/`, 6 analytics and data loader tests fail because they assume developer evaluation logs exist on disk rather than testing through self-contained, schema-valid fixtures.
2. **Mock-Contaminated Shortcut Evaluator**: `scripts/run_shortcut_evaluation.py` and `mirrorlab/evaluation/shortcut_evaluator.py` utilized simulated baselines (`MockStrategy.REALISTIC_GPT4O_BASELINE`) with internal branching and ground-truth access, while the generated reports presented these figures as though they were empirical findings from real frontier models.
3. **Evaluation Provenance Confusion**: Reports and UI displays do not systematically enforce strict provenance metadata separating `live_api`, `mock`, `human`, and `synthetic_baseline` execution.
4. **Documentation Overclaims**: The repository claimed a 100/100 score against the hackathon rubric and described physical optics experiments as completed when they were established only as protocols.

---

## 2. What Is Currently Working

* **Geometric Verification Engine**:
  - Method 1: Virtual Target Reflection (Householder affine transformation).
  - Method 2: Fermat Stationary Path (Virtual Camera reflection ray-line intersection).
  - Method 3: Direct Snell-Vector Collinearity Root-Finding (Brent's method with $10^{-7}\text{ mm}$ tolerance).
  - Triple-redundant consensus verification (`verify_triple()`) agreeing within $< 10^{-4}\text{ mm}$.
* **Dataset Generation & Splitting**:
  - Deterministic generation from integer seeds.
  - Frozen splits in `data/benchmark_splits/` (46 Dev, 46 Val, 138 Test packages; 70 Dev, 70 Val, 210 Test model-facing tasks).
  - Public/private split separation with neutral task IDs (`ml_dev_0001`, etc.) preventing semantic answer leakage.
  - Private ground-truth and counterfactual pair mapping isolated in private split directories.
* **Dataset Auditor**:
  - `DatasetAuditor.audit_all()` independently recomputes ray paths, blocker clearances, candidate mirror geometry, and counterfactual pair relationships across all 350 scenes with 11/11 checks passing.
* **Streamlit UI Application**:
  - Interactive tools: Scene Laboratory, Model Arena, Scientific Analytics, Benchmark Dashboard, and Physical Rig Guide render cleanly with responsive controls.
* **Code Formatting & Linting**:
  - 100% compliant with Ruff (0 lint errors, 129 files formatted).

---

## 3. What Is Broken

### A. CI Pytest Failures in Clean Environment (6 Tests)
When cloned into a fresh environment (such as GitHub Actions CI), `eval_logs/` is ignored by git (`.gitignore:57`). As a result, tests in `tests/test_analytics.py` and `tests/test_data_loader.py` fail:
1. `tests/test_analytics.py::test_load_all_evaluation_records_real_logs` (fails `assert len(records) > 0`)
2. `tests/test_analytics.py::test_load_all_run_aggregates` (fails `assert len(aggregates) > 0`)
3. `tests/test_analytics.py::test_filter_aggregation_counts` (fails because `records` is empty)
4. `tests/test_data_loader.py::TestRealBackendDataLoaders::test_get_recent_evaluation_runs_discovers_disk_runs` (fails `assert len(runs) > 0`)
5. `tests/test_data_loader.py::TestRealBackendDataLoaders::test_load_evaluation_run_details_dual_comparison` (fails `assert comp_run is not None`)
6. `tests/test_data_loader.py::TestRealBackendDataLoaders::test_get_failure_analysis_extracts_failures` (fails because details is None)

**Root Cause**: These tests were written to inspect developer artifacts left on disk rather than utilizing deterministic pytest fixtures with temporary test directories (`tmp_path`).

---

## 4. What Is Simulated vs. Genuinely Measured

### A. Simulated
* **Frontier Model Evaluations (GPT-4o & Claude 3.7)**:
  - All numbers previously reported for GPT-4o and Claude 3.7 were generated via `MockVLMAdapter` running heuristic simulation functions (`_generate_realistic_gpt4o`, `_generate_realistic_vlm`).
  - These simulations were calibrated based on early qualitative probing of vision models on planar reflection, but **do not represent verifiable live API calls**.
* **Visual Necessity & Shortcut Audit Numbers**:
  - `data/shortcut_eval_results.json` and `docs/bias_and_shortcut_audit.md` were generated using `MockVLMAdapter(strategy=MockStrategy.REALISTIC_GPT4O_BASELINE)`.
  - The evaluator contained mock-specific shortcuts that intercepted conditions like `CRITICAL_GEOMETRY_ABLATED` and returned simulated answers.

### B. Genuinely Measured
* **Mathematical Geometry**:
  - All 350 scenes have been independently verified by three separate geometric algorithms (Virtual Target, Fermat Virtual Camera, and Direct Snell Ray-Marching) agreeing within $10^{-4}\text{ mm}$.
* **Audit Integrity**:
  - All 11 checks in `DatasetAuditor` (bounding boxes, camera FOV frustums, ray-obstacle SAT clipping, finite aperture intersection, counterfactual transformation semantics, and split isolation) are genuinely calculated and verified.
* **UI Functionality**:
  - All 5 user journeys in the Streamlit application were tested and verified via `streamlit.testing.v1.AppTest`.

---

## 5. What Is Misleading or Overclaimed

1. **Presentation of Simulated Baselines as Real Model Results**:
   - Claims that "GPT-4o achieves 50.0% accuracy" or "Claude 3.7 collapses to 25% joint pair accuracy" are misleading because they were produced by calibrated mock strategies rather than authenticated API calls.
2. **Visual Necessity Ratio Claims**:
   - Stating that "GPT-4o demonstrates a Visual Necessity Ratio of +20.00%" when evaluated with a mock runner that intercepts conditions is scientifically invalid.
3. **Physical Optical Validation**:
   - Documents claiming real-world physical validation must clarify that an **optical breadboard protocol and calibration pipeline** have been specified, but automated laboratory bench trials remain pending physical camera capture.
4. **Self-Awarded 100/100 Rubric Score**:
   - Self-awarding a perfect 100/100 score in documentation undermines scientific credibility. An honest self-assessment must reflect the real state of empirical verification.

---

## 6. What Must Be Fixed

1. **Fix CI Tests via Isolated Fixtures**:
   - Refactor `tests/test_analytics.py` and `tests/test_data_loader.py` to use a self-contained pytest fixture (`mock_eval_logs_dir`) populated with schema-valid mock runs, empty directories, and malformed files.
   - Ensure tests pass with zero reliance on local untracked `eval_logs/`.
2. **Enforce Strict Evaluation Provenance**:
   - Add explicit provenance typing (`live_api`, `mock`, `human`, `synthetic_baseline`) to `RunConfig`, `ModelResponse`, and `EvaluationMetrics`.
   - Ensure every output JSON, Markdown report, and dashboard table clearly displays provenance.
3. **Refactor Shortcut Evaluator**:
   - Remove all mock-specific branching (`if isinstance(adapter, MockVLMAdapter)`) from `ShortcutEvaluator`.
   - The evaluator must treat all adapters identically through the public `adapter.ask()` interface without leaking ground-truth dictionaries.
4. **Transparently Relabel All Reports**:
   - Update `docs/model_evaluation_report.md`, `docs/bias_and_shortcut_audit.md`, and `README.md` to clearly mark simulated baselines as `[SIMULATED BASELINE]` and separate them from any live API runs.
5. **Calibrate Submission Guide & Rubric Assessment**:
   - Revise `docs/submission_guide.md` with an honest, defensible rubric assessment and acknowledge limitations transparently.

---

## 7. What Can Be Left Unchanged

1. **Geometric Ray Tracing Core**:
   - `mirrorlab/geometry/oracle.py`
   - `mirrorlab/geometry/independent_verifier.py`
   - `mirrorlab/geometry/ray_march_verifier.py`
2. **Benchmark Splits & Manifests**:
   - `data/benchmark_splits/` (230 packages, 350 model tasks, 120 pairs, neutral IDs).
3. **Dataset Auditor Logic**:
   - `mirrorlab/evaluation/dataset_auditor.py` (all 11 checks are mathematically sound).
4. **Streamlit UI Layout & Styling**:
   - `mirrorlab/ui/` design tokens, optical plots, and navigation components.
