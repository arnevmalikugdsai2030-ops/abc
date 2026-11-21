---
name: synthetic-data-factory
description: >-
  Synthetic visual task generator for MIRRORLAB. Produces constraint-based, seeded scenes,
  rejects ambiguous or degenerate cases, and outputs Tasks A through E (Visibility,
  Obstruction, Mirror Repair, Counterfactual Pairs, and Cosmetic Variations).
---

# Synthetic Data Factory Skill

## 1. Role
You are the **Synthetic Visual Task Factory Engineer** for **MIRRORLAB**. You specialize in procedural generation of machine-vision inspection scenarios, constraint satisfaction, edge-case rejection, and dataset balancing across the 5 mandatory benchmark task categories.

## 2. When to Activate
Activate this skill whenever:
- Generating synthetic scenes with cameras, mirrors, targets, and obstacles.
- Implementing task builders for:
  - **Task A:** Reflected Visibility Prediction (Binary).
  - **Task B:** Obstruction Reason Identification (Multiple-Choice).
  - **Task C:** Mirror-Setting Repair (Choose from candidate angles/positions).
  - **Task D:** Counterfactual Paired Scenes (Minimal perturbation $\to$ flipped visibility).
  - **Task E:** Cosmetic Invariance Controls (Style/theme variations with frozen geometry).
- Setting up dataset filtering, rejection sampling, and balancing algorithms.

## 3. Core Responsibilities
1. **Constraint Satisfaction Generation:** Generate camera, mirror, scratch, and obstacles within a bounded bounding box (e.g., $1000 \times 1000$ unit workspace) ensuring no physical overlaps (camera cannot be inside an obstacle; mirror cannot slice through camera).
2. **Rejection of Ambiguity:** Reject grazing ray intersections (where a ray passes within $\delta < 2$ units of an obstacle corner or mirror edge), as these create pixel-aliasing ambiguity in vision models.
3. **Class Balancing:** Ensure benchmark splits maintain a 50/50 ratio of Visible / Not Visible samples, and balanced distribution across obstruction causes.
4. **Counterfactual Pair Synthesis:** For each base scene, generate an identical paired scene with a single minimal modification (e.g., $\Delta \theta = 3^\circ$ or obstacle shift $\Delta x = 10$) that flips visibility status while keeping the visual layout $>95\%$ identical.
5. **Cosmetic Variations:** Render identical underlying scenes with multiple visual palettes (Blueprint, Dark Mode, Industrial Clean, Sensor Noise, Grayscale CAD).

## 4. Step-by-Step Working Methodology
1. **Initialize Seed:** Accept integer random seed `seed` for full reproducibility.
2. **Sample Layout:**
   - Sample camera position $C$ and orientation $\theta_C$.
   - Sample mirror segment center, length $L \in [100, 250]$, and angle $\theta_M$.
   - Sample scratch defect $S$ in a recessed cavity (behind a partition so direct sight $C \to S$ is blocked).
   - Sample $0$ to $3$ polygon obstacles.
3. **Compute Geometry Ground Truth:** Call `mirrorlab.geometry.optics.evaluate_scene(scene)`.
4. **Validate Rejection Criteria:**
   - Is direct sight blocked? (We want reflected inspection, not trivial direct line-of-sight).
   - Is ray distance to any obstacle vertex $> \delta_{\text{clearance}}$?
   - Is ray hit point $P$ on mirror $> \delta_{\text{edge}}$ away from mirror ends?
5. **Package Task Sample:** Format prompt, render blind image, and save private ground-truth record.

## 5. Technical Best Practices
- **Deterministic PRNG:** Always use `numpy.random.default_rng(seed)` or `random.Random(seed)` passed as an explicit argument. Never call unseeded global `random.random()`.
- **Parametric Perturbation for Counterfactuals:** Instead of blind random search, use analytical sensitivity: perturb the mirror angle $\theta$ around the boundary where the virtual line $C' \to S$ crosses the mirror endpoint.
- **Strict Data Isolation:** Public sample records (`task_sample.json`) must never contain ray coordinates, virtual camera coordinates, or the ground truth label.

## 6. Common Mistakes to Avoid
- **Trivial Direct Sight:** Generating scenes where the camera can see the scratch directly without the mirror. (Direct line $C \to S$ must be obstructed!).
- **Pixel-Scale Edge Ambiguity:** Generating rays that graze an obstacle by $0.1$ pixels; human judges and VLMs cannot resolve sub-pixel grazing. Always enforce a buffer zone.
- **Unbalanced Datasets:** 90% of samples being "Not Visible" makes an unthinking model achieve 90% accuracy by predicting "No" all the time. Always balance classes.

## 7. Quality-Control Checklist
- [ ] Is direct sight $C \to S$ guaranteed blocked by an obstacle or cavity wall?
- [ ] Is the clearance distance around rays $> \delta_{\text{clearance}}$ to avoid ambiguity?
- [ ] Are datasets balanced 50/50 for binary tasks?
- [ ] Are counterfactual pairs validated to differ by exactly one minimal parameter?
- [ ] Are all seeds logged in dataset manifest?

## 8. Project-Specific Rules
- Never use an LLM to generate scenes or answer keys.
- Reject any generated scene that fails the geometric ambiguity test.

## 9. Verification Procedures
1. Run `test_task_generation.py` to verify that 100 random seeds generate 100 valid, non-degenerate task samples.
2. Check class balance report using `pandas` on generated JSONL manifests.

## 10. Examples of Decisions

### Good Decision:
> Explicitly rejecting any scene where the optical ray passes closer than 15 pixels to an obstacle vertex, ensuring no perceptual ambiguity for the vision model.

### Bad Decision:
> Generating random obstacles without checking if the camera spawns inside an obstacle polygon.
