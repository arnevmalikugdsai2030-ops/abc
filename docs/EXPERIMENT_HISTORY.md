# MIRRORLAB: Task Discovery & Experiment History
## Systematic Exploration, Empirical Screening & Negative Results Log

**Document Version:** 1.0.0  
**Project:** MIRRORLAB — The Missing Scratch  
**Hackathon:** Innodata Frontier AI Hackathon 2026 | Track 2: Visual Task Factory  
**Last Updated:** November 23, 2026  

---

## 1. Overview & Experimental Methodology

In accordance with scientific benchmarking best practices, MIRRORLAB does **not** assume that the initial task families are the only or optimal visual reasoning operations.

We implemented a systematic **Task Condition Exploration Matrix** (`mirrorlab/experiment/task_explorer.py`) to explore parameter spaces across 5 visual reasoning dimensions:
1. **Geometry Regimes:** Planar mirror, narrow aperture, off-axis tilt, grazing incidence, dual-mirror reflections.
2. **Occlusion Regimes:** Unobstructed, mid-corridor barrier, near-mirror obstruction, near-target occlusion, multiple barriers.
3. **Clutter Regimes:** Clean minimal diagrams, moderate annotations, high distractor clutter.
4. **Reasoning Regimes:** Binary visibility (Task A), obstruction localization (Task B), mirror repair selection (Task C), counterfactual pairs (Task D), physical path falsification.
5. **Sensitivity Regimes:** Minimal angle perturbations ($\Delta \theta = 2^\circ$), distance shifts, invariant controls.

Every exploratory experiment is archived with its full random seed, configuration, yield, rejection accounting, and scientific conclusions in `experiments/`.

---

## 2. Chronological Experiment Log

### Summary Table of Exploratory Conditions

| Exp ID | Condition Name | Geometry / Occlusion | Sample | Valid Yield | Rejections & Reasons | Calibrated Difficulty | Benchmark Status |
|---|---|---|---|---|---|---|---|
| **EXP-01** | `planar_standard` | Single planar mirror, corridor partition | 10 | 10 (100%) | 0 rejections | Easy (wide margins) | **ACCEPTED (Task A)** |
| **EXP-02** | `narrow_aperture_tight` | 100mm mirror aperture ($< 15\%$ width) | 10 | 9 (90%) | 1 `AMBIGUOUS_ENDPOINT` | Medium | **ACCEPTED (Task A Hard)** |
| **EXP-03** | `grazing_incidence_oblique` | Extreme incidence angle $\theta_i > 75^\circ$ | 10 | 8 (80%) | 2 `GRAZING_RAY_INSTABILITY` | Hard | **ACCEPTED (Task E OOD)** |
| **EXP-04** | `near_mirror_occlusion` | Obstacle placed within $30\text{ mm}$ of mirror | 10 | 10 (100%) | 0 rejections | Medium / Hard | **ACCEPTED (Task B)** |
| **EXP-05** | `high_clutter_distractors` | 4 distractor objects scattered in canvas | 10 | 7 (70%) | 3 `DISTRACTOR_CORRIDOR_OVERLAP` | Medium | **FILTERED & TUNED (Task E)** |
| **EXP-06** | `dual_mirror_folded_hop` | Two sequential planar mirrors | 10 | 4 (40%) | 6 `MULTI_HOP_ACCUMULATED_ERROR` | Expert | **QUARANTINED (Negative Result)** |
| **EXP-07** | `sub_degree_angle_flip` | Mirror tilt delta $\Delta \theta = 0.5^\circ$ | 10 | 3 (30%) | 7 `PIXEL_SUB_SAMPLING_AMBIGUITY` | Expert | **REJECTED (Resolution Bound)** |

---

## 3. Detailed Experiment Reports & Negative Results

### Experiment EXP-06: Dual-Mirror Folded Ray Paths (Quarantined)
- **Hypothesis:** Multi-hop reflections ($C \to M_1 \to M_2 \to T$) test higher-order optical path planning.
- **Observed Result:** 6 out of 10 scenes failed due to numerical tolerance accumulation across sequential virtual target reflections. Furthermore, at standard $1024 \times 1024$ image resolution, multi-hop ray corridors created visually ambiguous crossings that human judges could not confidently resolve without zooming.
- **Scientific Decision:** **Quarantined from headline benchmark.** The benchmark maintains high standards: *a difficult but ambiguous task is a bad benchmark task*. Multi-hop scenes are preserved in `experiments/` for future higher-resolution benchmarks.

### Experiment EXP-07: Sub-Degree Angular Perturbations ($\Delta \theta = 0.5^\circ$) (Rejected)
- **Hypothesis:** Testing whether VLMs can detect extreme sub-degree perturbations ($\Delta \theta = 0.5^\circ$).
- **Observed Result:** At $\Delta \theta = 0.5^\circ$, the displacement of the reflected ray across a $400\text{ mm}$ optical leg is:
  $$\Delta x = 400 \cdot \tan(2 \cdot 0.5^\circ) \approx 6.98\text{ mm} \approx 7.1\text{ pixels}$$
  In standard $1024 \times 1024$ rendering, a 7-pixel ray displacement is dominated by raster antialiasing and sub-pixel edge sampling, causing human disagreement on whether the ray clears or strikes an obstacle corner.
- **Scientific Decision:** **Rejected.** Set the minimum counterfactual angular perturbation to $\Delta \theta = 2.0^\circ$ ($\approx 28\text{ pixels}$ displacement), ensuring the change is cleanly resolvable by visual inspection while remaining challenging for models.

### Experiment EXP-05: Distractor Clutter Tuning (Tuned)
- **Hypothesis:** Introducing irrelevant distractor objects tests model attention against background noise.
- **Observed Result:** Unconstrained Poisson placement caused distractors to randomly intersect the valid optical corridor, turning valid scenes into unintended occlusions.
- **Scientific Decision:** Implemented a mandatory **Distractor Clearance Filter** ensuring all distractors maintain a buffer $> 20.0\text{ mm}$ from both incident ($C \to R$) and reflected ($R \to T$) rays.

---

## 4. Quantitative Difficulty Calibration Rules

Difficulty is calibrated deterministically via `mirrorlab/experiment/difficulty_calibrator.py`:

| Tier | Aperture Margin ($\Delta d_{\text{mirror}}$) | Obstacle Clearance ($\Delta d_{\text{obs}}$) | Incidence Angle ($\theta_i$) | Perturbation ($\Delta \theta$) | Target Reasoning |
|---|---|---|---|---|---|
| **EASY** | $> 40.0\text{ mm}$ | $> 35.0\text{ mm}$ | $30^\circ - 55^\circ$ | N/A | Clear visibility or distant blockage |
| **MEDIUM** | $20.0 - 40.0\text{ mm}$ | $20.0 - 35.0\text{ mm}$ | $20^\circ - 65^\circ$ | N/A | Multi-obstacle or candidate mirror selection |
| **HARD** | $10.0 - 20.0\text{ mm}$ | $15.0 - 20.0\text{ mm}$ | $65^\circ - 75^\circ$ | $\Delta \theta \le 5.0^\circ$ | Near-aperture reflection, grazing ray |
| **EXPERT** | $< 10.0\text{ mm}$ | $< 15.0\text{ mm}$ | $> 75^\circ$ | $\Delta \theta = 2.0^\circ$ | Minimal counterfactual flip pair, path falsification |

---
*Maintained by the MIRRORLAB Task Discovery Core.*
