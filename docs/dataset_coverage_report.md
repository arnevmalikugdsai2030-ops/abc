# MIRRORLAB Dataset Distribution & Coverage Report

**Total Scenes Analyzed:** 130 | **Duplicate Count:** 0 (130 unique fingerprints)

## 1. Headline vs. Boundary Dataset Separation

> **Rule Enforcement:** Boundary cases are strictly segregated from the headline benchmark dataset to prevent noise in headline accuracy metrics.

| Partition | Scene Count | Percentage | Description |
|---|---|---|---|
| **Headline Benchmark** | 110 | 84.6% | Evaluates standard spatial reasoning across Easy, Standard, Hard tiers |
| **Boundary Stress Test** | 20 | 15.4% | Evaluates sub-millimeter edge grazing, FOV boundaries, and mirror limits |

## 2. Difficulty Tier Distribution

| Difficulty Tier | Scene Count | Percentage | Target Parameters |
|---|---|---|---|
| **EASY** | 30 | 23.1% | Large mirrors (220-300mm), FOV 70-90°, zero distractors |
| **STANDARD** | 50 | 38.5% | Moderate mirror (150-220mm), FOV 55-70°, 1-2 distractors |
| **HARD** | 30 | 23.1% | Small mirror (75-130mm), FOV 35-48°, 2-3 obstacles, 2-leg reasoning |
| **BOUNDARY** | 20 | 15.4% | Near mirror endpoint (2-4%), near FOV boundary (1°), grazing obstacles |

## 3. Physical Category Breakdown

| Category | Count | Percentage | Oracle Reason Code |
|---|---|---|---|
| `VALID_VISIBLE` | 23 | 17.7% | Validated via `VisibilityOracle` |
| `MIRROR_MISS` | 23 | 17.7% | Validated via `VisibilityOracle` |
| `OUTSIDE_FOV` | 21 | 16.2% | Validated via `VisibilityOracle` |
| `WRONG_SIDE` | 21 | 16.2% | Validated via `VisibilityOracle` |
| `BLOCKED_INCIDENT` | 21 | 16.2% | Validated via `VisibilityOracle` |
| `BLOCKED_REFLECTED` | 21 | 16.2% | Validated via `VisibilityOracle` |

## 4. Ground Truth Visibility State

| Outcome | Count | Percentage | Balanced Target |
|---|---|---|---|
| **VISIBLE** | 23 | 17.7% | ~50% / ~50% Balanced Split |
| **NOT_VISIBLE** | 107 | 82.3% | ~50% / ~50% Balanced Split |

## 5. Geometric Distribution Metrics

| Geometric Metric | Min | Mean | Median | Max | Std Dev |
|---|---|---|---|---|---|
| **Mirror Length** | 76.5 mm | 175.3 mm | 174.9 mm | 293.7 mm | 56.0 mm |
| **Camera FOV Angle** | 35.0° | 60.2° | 60.8° | 89.2° | 14.0° |
| **Camera Heading Direction** | 0.3° | 98.3° | 108.5° | 352.5° | 53.7° |
| **Incident Angle ($θ_i = θ_r$)** | 26.5° | 34.9° | 32.6° | 51.9° | 7.3° |
| **Total Optical Path ($C \to R \to T$)** | 511.0 mm | 650.7 mm | 633.4 mm | 824.3 mm | 75.0 mm |
| **Obstacles per Scene** | 1.0 | 2.7 | 3.0 | 5.0 | 1.0 |

## 6. Verification Integrity Statement

- **Oracle Enforcement:** 100% of scenes analytically validated by `VisibilityOracle`.

- **Zero Duplicate Geometry:** 130 unique geometry fingerprints verified out of 130 scenes (0 duplicates).

- **Headline Purity:** Boundary stress cases are strictly tagged `is_boundary=True` and segregated into independent evaluation pipelines.
