# MIRRORLAB: Complete Reproducibility & Verification Guide
## Deterministic Procedural Generation, Offline Verification, and Single-Command Pipelines

**Document Version:** 1.0.0  
**Status:** Authoritative Reproducibility Specification  
**Hackathon:** Innodata Frontier AI Hackathon 2026 | Track 2: Visual Task Factory  

---

## 1. Reproducibility Guarantee & Principles

MIRRORLAB is built to be **100% reproducible offline** with zero dependency on third-party cloud APIs, pre-trained weights, or nondeterministic LLM labeling:

1. **Deterministic Pseudo-Random Generation:** Every geometric scene, obstacle layout, camera pose, and mirror aperture is parameterized by an explicit 32-bit unsigned integer seed (`np.random.default_rng(seed)`).
2. **Triply-Redundant Geometric Verification:** Ground truth is calculated deterministically via three independent mathematical algorithms (Virtual Target, Fermat Least-Time, and Direct Snell Collinearity).
3. **Hermetic Test Suite:** All 334 unit and integration tests run entirely offline with zero network connectivity.
4. **Air-Gapped Blind Partitioning:** The splitting pipeline deterministically packages public tasks with non-semantic identifiers and keeps ground truth in private key files.

---

## 2. Environment Setup

### 2.1 Supported Python Environments
- **Python 3.11:** `python >= 3.11.0`
- **Python 3.12:** `python >= 3.12.0` (Recommended)

### 2.2 Installation Steps

```bash
# Clone the repository
git clone https://github.com/arnevmalikugdsai2030-ops/abc.git
cd abc

# Create and activate a clean virtual environment
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate

# Install dependencies in editable mode
pip install -e .
```

---

## 3. Single-Command End-to-End Reproduction

### Step 1: Rebuild Benchmark Tasks from Deterministic Seeds
Regenerates all task suites across Task A, Task B, Task C, Task D, and Task E:

```bash
python scripts/rebuild_benchmark_dataset.py
```
*Output: 230 top-level task packages in `data/`.*

### Step 2: Deterministic Partitioning & 11-Point Audit
Splits the generated dataset into Dev (46/70), Val (46/70), and Test (138/210) partitions with neutral hash naming and runs the comprehensive 11-point mathematical audit:

```bash
python scripts/audit_and_split_dataset.py
```
*Output: Regenerates `data/benchmark_splits/` and writes `docs/benchmark_audit_report.md` with 11/11 passing checks.*

### Step 3: Run Full Automated Test Suite
Executes all 334 tests covering geometry, obstacle SAT clipping, FOV cones, corruption detection, and auditor checks:

```bash
pytest tests/ -v
```
*Expected: 334 passed, 0 failures.*

### Step 4: Run 8-Condition Visual Necessity & Shortcut Audit
Evaluates benchmark resistance across unimodal baselines, language priors, metadata leaks, and spatial ablations:

```bash
python scripts/run_shortcut_evaluation.py
```
*Output: Produces `data/shortcut_eval_results.json` and updates `docs/bias_and_shortcut_audit.md`.*

### Step 5: Run Model Evaluation
Run offline evaluation using calibrated model baselines or live APIs:

```bash
# Evaluate analytical oracle baseline (sanity check: 100% accuracy)
python scripts/evaluate_models.py --model mock-oracle --split dev

# Evaluate calibrated GPT-4o simulation baseline
python scripts/evaluate_models.py --model mock-gpt4o --split dev

# (Optional) Evaluate live Google Gemini API (requires GEMINI_API_KEY)
python scripts/evaluate_models.py --model gemini-2.5-flash --split dev
```
*Output: Saves structured execution logs in `eval_logs/`.*

### Step 6: Launch the Interactive Web Dashboard
Launches the full scientific observability dashboard:

```bash
streamlit run mirrorlab/ui/app.py
```
*Access via browser at `http://localhost:8501`.*

---

## 4. Verification Checksums & Artifact Accounting

| Partition | Public Task File | Private Ground Truth | Pair File | Blind Images | Overlay Images | Total Images |
|---|---|---|---|---|---|---|
| **Dev Split** | `dev/tasks.jsonl` (70 lines) | `dev/ground_truth.jsonl` (70 lines) | `dev/pairs.jsonl` (24 lines) | 70 PNGs | 70 PNGs | 140 PNGs |
| **Val Split** | `val/tasks.jsonl` (70 lines) | `val/ground_truth.jsonl` (70 lines) | `val/pairs.jsonl` (24 lines) | 70 PNGs | 70 PNGs | 140 PNGs |
| **Test Split**| `test/public/tasks.jsonl` (210) | `test/private/ground_truth.jsonl` (210)| `test/private/pairs.jsonl` (72)| 210 PNGs | 210 PNGs| 420 PNGs |
| **Total** | **350 model tasks** | **350 ground truths** | **120 pairs** | **350 PNGs**| **350 PNGs**| **700 PNGs**|

---
*Verified by the MIRRORLAB Reproducibility & Quality Assurance Team.*
