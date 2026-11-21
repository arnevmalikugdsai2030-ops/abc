# MIRRORLAB Task Specifications & Reasoning Protocol

MIRRORLAB evaluates Vision-Language Models across 5 standardized machine-vision reasoning tasks.

---

## Task A: Reflected Visibility Prediction (Binary)
- **Goal:** Determine whether the fixed camera can inspect target scratch `X` via reflection in the finite flat mirror.
- **Visual Input:** Blind inspection schematic (camera icon, mirror line, target `X`, obstacle blocks). No ray paths.
- **Prompt:** *"Can the camera see the target scratch marked with red X via reflection in the mirror? Respond in JSON with answer: 'VISIBLE' or 'NOT_VISIBLE'."*
- **Reasoning Challenge:** Solving the angle of incidence equals angle of reflection ($\theta_i = \theta_r$) and checking whether the ray hits the finite mirror span.

---

## Task B: Obstruction Reason Identification (Multiple-Choice)
- **Goal:** If the target scratch is not visible, identify the exact physical obstruction cause.
- **Options:**
  - `A`: Out of camera field-of-view (FOV cone)
  - `B`: Reflected ray misses the finite mirror boundary
  - `C`: Incident ray ($C \to P$) blocked by Obstacle
  - `D`: Reflected ray ($P \to S$) blocked by Obstacle
  - `E`: Hits the non-reflective back face of the mirror
- **Reasoning Challenge:** Multi-hop causal diagnostic reasoning.

---

## Task C: Mirror-Setting Repair (Inverse Optics)
- **Goal:** Given a defective inspection setting where scratch `X` is occluded, identify which candidate mirror angle ($\theta_1, \theta_2, \theta_3$) restores an unobstructed sightline.
- **Visual Input:** Diagram showing current setup and dashed lines indicating 3 candidate mirror orientations.
- **Reasoning Challenge:** Inverse ray projection and spatial rotation.

---

## Task D: Counterfactual Paired Evaluation
- **Goal:** Evaluate the model on two paired scenes $(S, S')$ where a minimal perturbation (e.g. mirror rotated $2.5^\circ$) flips visibility from True to False.
- **Scoring Protocol:** A model receives 1 point if and only if it correctly answers **both** $S$ and $S'$.
- **Scientific Value:** Destroys surface-layout guessing heuristics. Models that rely on shallow visual patterns score near 0%.

---

## Task E: Cosmetic Invariance Controls
- **Goal:** Evaluate whether the model's prediction remains stable when the underlying geometry is held constant while cosmetic rendering styles vary.
- **Themes:** Blueprint, Dark CAD, Industrial Clean, Grayscale Wireframe, Low-Light Sensor Noise.
