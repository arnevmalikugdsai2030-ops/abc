# MIRRORLAB: The Missing Scratch

[![Innodata Frontier AI Hackathon 2026](https://img.shields.io/badge/Hackathon-Track_2:_Visual_Task_Factory-blue.svg)](https://innodata.com)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-brightgreen.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **A Reproducible Synthetic Visual-Task Factory to evaluate whether Vision-Language Models can reason about reflected visibility in machine-vision inspection diagrams.**

---

## 🔍 The Industrial Problem

In high-precision manufacturing (semiconductor wafer handling, automotive cylinder bores, EV battery pouch packaging), cameras frequently cannot maintain direct line-of-sight into deep recessed cavities. Industrial inspection stations position **finite planar mirrors** to view blind spots and detect critical surface scratches or micro-cracks.

Frontier Vision-Language Models (VLMs) demonstrate impressive semantic captioning, but do they actually understand the physics of optical reflection? **Can an AI look at a machine-vision schematic and determine whether a camera sees a scratch in a mirror?**

## 💡 The MIRRORLAB Solution

**MIRRORLAB** is an automated, synthetic benchmark factory that generates mathematically verified inspection diagrams across 5 distinct reasoning tasks:
1. **Task A: Reflected Visibility Prediction** (Binary: Can the camera see the scratch?)
2. **Task B: Obstruction Reason Identification** (Multiple-choice: Why is the scratch blocked?)
3. **Task C: Mirror-Setting Repair** (Inverse optics: Which mirror orientation restores visibility?)
4. **Task D: Counterfactual Paired Evaluation** (Minimal $2^\circ$ or 5mm perturbation flips visibility)
5. **Task E: Cosmetic Invariance Controls** (Style variations with frozen geometry)

### Absolute Mathematical Ground Truth
Ground truth is computed via **deterministic computational geometry** (the Virtual Camera Reflection Method and Euclidean obstacle intersection). **We never use an LLM to generate or evaluate geometric ground truth.**

---

## 🚀 Quickstart

### 1. Prerequisites
- Python 3.11+
- `uv` (recommended) or standard `pip`

### 2. Installation
```powershell
# Clone the repository
git clone https://github.com/your-username/mirrorlab.git
cd mirrorlab

# Create and activate virtual environment with uv
uv venv .venv --python 3.12
.venv\Scripts\activate

# Install dependencies
uv pip install -e ".[dev]"
```

### 3. Run Verification Tests
```powershell
uv run pytest tests/ -v
```

### 4. Launch the Interactive Studio
```powershell
uv run streamlit run mirrorlab/ui/app.py
```

---

## 📁 Repository Architecture

```text
mirrorlab/
├── configs/                # YAML configuration files for generation and evaluation
├── docs/                   # Detailed specifications and mathematical proofs
├── scripts/                # CLI runners for dataset generation and model benchmarking
├── mirrorlab/              # Core Python package
│   ├── geometry/           # Deterministic 2D ray reflection and obstacle math
│   ├── generation/         # Parametric scene generators & ambiguity rejection
│   ├── rendering/          # High-resolution Pillow schematic diagram engine
│   ├── schema/             # Pydantic v2 data models (public tasks vs private keys)
│   ├── evaluation/         # VLM API adapters, structured parsing, scoring
│   ├── reporting/          # Confusion matrices, accuracy, and pair metrics
│   └── ui/                 # Streamlit interactive optical studio
└── tests/                  # Automated pytest test suites
```

---

## 📜 Mandatory Project Rules
- **Geometry Oracle Independence:** Ground truth is purely mathematical.
- **No Answer Leakage:** Public benchmark images and prompts contain zero ray coordinates or virtual camera hints.
- **Counterfactual Rigor:** Models must solve both sides of a minimally perturbed pair to receive credit.
- **Honest Metrics:** All reported evaluation numbers are logged directly from reproducible API runs.
