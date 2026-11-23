# MIRRORLAB: 5-Minute Technical Judge Demonstration Script

**Competition:** Innodata Frontier AI Hackathon 2026 — Track 2: Visual Task Factory  
**Target Audience:** Technical Judges, AI Metrology Evaluators & Research Scientists  
**Interactive UI:** Streamlit Dashboard -> Select **`🎯 Judge Demo Walkthrough`**  
**Terminal CLI:** `streamlit run mirrorlab/ui/app.py`  

---

## Overview & Demo Narrative

| Minute | Stage | Interactive Feature | Key Takeaway |
|---|---|---|---|
| **0:00 - 1:00** | **Moment 1: The Intuition Trap** | Scene Lab -> Infinite Plane Fallacy | Humans and VLMs guess visible; finite mirror clips ray |
| **1:00 - 2:00** | **Moment 2: Triple-Verifier Proof** | Mathematical Verifiers Panel | 3 distinct solvers agree to $10^{-5}\text{ mm}$ residual |
| **2:00 - 3:00** | **Moment 3: The Model Challenge** | Model Arena -> Live / Simulation Challenge | SOTA models fail; classified into 14-category taxonomy |
| **3:00 - 4:00** | **Moment 4: Counterfactual Invariance** | Pair Gallery -> Angle Flip vs Cosmetic | Models flip on cosmetic variations; pairs penalize shortcuts |
| **4:00 - 5:00** | **Moment 5: The Task Factory** | 1-Click Generator & Auditor | Fresh seeded generation, zero leakage, SFT/RL export |

---

## Detailed Minute-by-Minute Walkthrough

### [0:00 - 1:00] Moment 1: The Human Intuition Trap
1. **Action in UI**:
   - Open Streamlit sidebar and select **`🎯 Judge Demo Walkthrough`**.
   - Navigate to **"Step 1: The Optical Trap"** (Preset Scene Seed: `#5011 - The Aperture Illusion`).
2. **Narration**:
   > *"Welcome to MIRRORLAB. To understand why frontier vision-language models struggle with spatial reasoning, let's start with a human test.*
   > *Look at this inspection cell schematic: A camera is pointing toward a mirror, and a red defect is on the floor. At first glance, the angle looks plausible for reflection.*
   > *Most people and nearly every frontier VLM answer 'VISIBLE'.*
   > *Now watch what happens when we toggle the Diagnostic Overlay:"*
3. **Action**:
   - Flip the toggle **"Show Diagnostic Ray Path"**.
4. **Reveal**:
   > *"The calculated line of sight strikes the floor 14 mm beyond the edge of the mirror segment. The ray misses the aperture completely. This is the **Infinite Plane Fallacy**—a systematic error where models hallucinate reflection because they extend the mirror line infinitely into space."*

---

### [1:00 - 2:00] Moment 2: The Triple-Verifier Proof
1. **Action in UI**:
   - Scroll to **"Step 2: Triple-Verifier Consensus"**.
2. **Narration**:
   > *"How do we know the ground truth is unassailable? In Track 2, an evaluation benchmark is only as credible as its ground truth.*
   > *Rather than trusting a single raycaster, MIRRORLAB executes **three mathematically independent solvers** for every single scene:*
   > *1. **Householder Specular Oracle**: Closed-form affine reflection matrix.*
   > *2. **Fermat's Principle of Least Time**: Variational path minimization ($|C - R| + |R - T|$).*
   > *3. **Snell's Law Root-Finder**: Numerical Brent-method solver matching incident and reflected angles.*
   > *Look at the live residual table: Across all three solvers, the maximum spatial disagreement is **$2.66 \times 10^{-5}\text{ mm}$**—far below our $0.1\text{ mm}$ engineering tolerance. If any solver disagrees by more than $10^{-4}\text{ mm}$, the scene is automatically rejected."*

---

### [2:00 - 3:00] Moment 3: The Model Challenge
1. **Action in UI**:
   - Scroll to **"Step 3: The VLM Evaluation"**.
   - Select model: **GPT-4o / Gemini 2.5 Simulation** or trigger live API call.
   - Click **"Evaluate Blind Schematic"**.
2. **Narration**:
   > *"Notice the prompt given to the model: It receives strictly the high-resolution blind schematic diagram—with zero rays, zero answer badges, and zero leading hints.*
   > *Look at the model's response:*
   > *It predicts 'VISIBLE', claiming the target reflects off the center of the mirror.*
   > *Our automated **14-Category Failure Classifier** intercepts the response and flags it as:*
   > **`VIRTUAL_IMAGE_REASONING_FAILURE`** (Confidence: 0.90, Remedy: 'Incorporate finite segment clipping verification').*
   > *Notice the provenance badge: Every run records an unambiguous cryptographic tag—preventing any mock data from ever being disguised as live API evidence."*

---

### [3:00 - 4:00] Moment 4: The Counterfactual Flip
1. **Action in UI**:
   - Scroll to **"Step 4: Counterfactual Pair Invariance"** (Preset Pair: `CF-PAIR-402`).
2. **Narration**:
   > *"Standard benchmarks allow models to score well by guessing 'VISIBLE' on 80% of samples.*
   > *MIRRORLAB eliminates this shortcut through **Task D: Counterfactual Minimal Pairs**.*
   > *Here are two paired scenes:*
   > *- Scene A has an irrelevant baffle added 200 mm away from the beam path.*
   > *- Scene B rotates the mirror by just $1.5^\circ$, completely severing the optical link.*
   > *Under our Pair Consistency metric, a model gets zero points if it guesses 'VISIBLE' on both. It must correctly track the physical invariant across both variations to score."*

---

### [4:00 - 5:00] Moment 5: The Task Factory & Downstream Exports
1. **Action in UI**:
   - Scroll to **"Step 5: Visual Task Factory & SFT/RL Export"**.
   - Enter Seed: `2026` -> Click **"Generate Fresh Benchmark Scene"**.
2. **Narration**:
   > *"Finally, MIRRORLAB is a complete Visual Task Factory. With one click, it synthesizes new, seeded, high-DPI inspection cells, runs the 7-layer validation gate, verifies non-leakage, and exports data directly for downstream AI training:*
   > *1. **Supervised Fine-Tuning (SFT)**: Full multi-modal HuggingFace/OpenAI JSONL with 7-step formal geometric derivations.*
   > *2. **Reinforcement Learning**: Gymnasium-compatible `MirrorLabEnv` where RL agents learn to align mirrors in active inspection cells.*
   > *100% reproducible, zero data leakage, and scientifically defensible to CVPR standards."*

---

## Quick Judge Checklist

- [x] **Zero Semantic Leakage**: Public test split filenames anonymized (`task_pub_XXXX`).
- [x] **Proven Ground Truth**: 100.0% agreement across Oracle, Fermat, and Snell solvers.
- [x] **Ambiguity Rejection**: Every scene has $>2.0\text{ mm}$ clearance and $>1.5\text{ mm}$ penetration depth.
- [x] **Cryptographic Provenance**: Strict separation of `LIVE_API` vs `CALIBRATED_SIMULATION_MOCK`.
- [x] **Downstream Utility**: Turn-key exports to SFT reasoning traces and Gymnasium RL environments.
