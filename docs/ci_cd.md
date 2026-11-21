# Continuous Integration (CI) Guide for MIRRORLAB

This document explains our GitHub Actions CI pipeline, how it guards code quality, and how to read the workflow results.

---

## 1. What the CI Workflow Does

Every time you **push code** to `main` or **open a pull request**, GitHub automatically spins up virtual Ubuntu machines running **Python 3.11** and **Python 3.12** to execute the following 7 checks:

```
+-------------------------------------------------------------------------------+
|                             GITHUB ACTIONS PIPELINE                           |
+-------------------------------------------------------------------------------+
  [1] Checkout code
       │
  [2] Setup Python 3.11 & 3.12 runners
       │
  [3] Install dependencies (`pip install -e ".[dev]"`)
       │
  [4] Security Guard: Fail if `.env` or secret keys are committed
       │
  [5] Linter & Formatter: Verify code with `ruff check` and `ruff format`
       │
  [6] Import Test: Confirm `mirrorlab` and third-party libraries load cleanly
       │
  [7] Pytest Suite: Run all automated unit and regression tests
```

If **any single step fails**, the pipeline turns red (❌), flags the exact failing line of code, and blocks merging.

---

## 2. How to Read Workflow Results on GitHub

1. Open your repository on GitHub:  
   **[https://github.com/arnevmalikugdsai2030-ops/abc](https://github.com/arnevmalikugdsai2030-ops/abc)**
2. Click the **"Actions"** tab at the top of the repository page.
3. You will see a list of workflow runs corresponding to your commits.
4. **Icons:**
   - 🟢 **Green checkmark:** All tests, linting, import checks, and secret scans passed.
   - 🟡 **Yellow circle:** The workflow is currently running.
   - 🔴 **Red X:** One of the checks failed.
5. **Viewing Details:**
   - Click on any workflow run to see the jobs (`Lint, Import Check & Test Suite (3.11)` and `(3.12)`).
   - Click on a job to expand individual steps (e.g. `Run Pytest Test Suite`).
   - If a step failed, expanding it will reveal the exact Python traceback and test failure explanation.

---

## 3. Running CI Checks Locally Before Pushing

You don't have to wait for GitHub Actions to find issues! Run the exact same checks locally in your terminal:

```powershell
# 1. Format and lint checks
ruff check .
ruff format --check .

# 2. Package import verification
python -c "import mirrorlab; print(mirrorlab.__version__)"

# 3. Automated test suite with coverage
pytest -v --cov=mirrorlab tests/
```
