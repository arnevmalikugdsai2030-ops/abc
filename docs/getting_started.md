# Getting Started with MIRRORLAB

This guide will walk you through setting up your environment, running tests, and understanding how MIRRORLAB works.

## 1. Environment Setup

We recommend using `uv` for package management because it creates reproducible virtual environments in seconds.

```powershell
# Create a virtual environment using Python 3.12
uv venv .venv --python 3.12

# Activate the virtual environment
.venv\Scripts\activate

# Install all project dependencies including development tools
uv pip install -e ".[dev]"
```

## 2. Running Automated Tests

MIRRORLAB enforces strict test verification. Before submitting any changes, verify all tests pass:

```powershell
uv run pytest tests/ -v
```

## 3. Generating a Benchmark Dataset

To generate 100 balanced inspection tasks across all 5 categories:

```powershell
uv run python scripts/generate_benchmark.py --count 100 --seed 42 --out-dir data/sample
```

## 4. Evaluating a Vision Model

To evaluate a model against generated tasks (ensure `GEMINI_API_KEY` is set in `.env`):

```powershell
uv run python scripts/evaluate_models.py --model gemini-2.5-flash --dataset data/sample
```

## 5. Launching the Interactive Web Studio

```powershell
uv run streamlit run mirrorlab/ui/app.py
```
