# MIRRORLAB Project Mandatory Operating Rules

> **Scope:** Persistent workspace rule for MIRRORLAB (Innodata Frontier AI Hackathon 2026, Track 2).  
> **Applies To:** All agents, subagents, code changes, tests, and documentation.  
> **Precedence:** Highest priority. These rules override any conflicting shortcuts or heuristics.

---

## 1. Geometry Oracle Independence (Absolute Mathematical Ground Truth)
- The ground truth for every task, question, and benchmark must be determined **strictly and exclusively** by deterministic 2D computational geometry (the Law of Specular Reflection, virtual camera projection, and Euclidean obstacle intersection).
- **NEVER** use an LLM, VLM, or heuristic neural network to determine, verify, or label ground-truth visibility, obstruction, or repair angles.

## 2. No Fake Data or Hallucinated Results
- **NEVER** invent benchmark numbers, accuracy percentages, model rankings, or evaluation results.
- If model evaluation has not been executed against a real model API, state clearly that benchmark evaluation is unexecuted or in mock verification mode.
- Every metric displayed on the dashboard or reported in documentation must come directly from real execution logs.

## 3. Strict Answer Leakage Prevention (Blind Evaluation Integrity)
- Maintain an ironclad boundary between public benchmark task inputs and private ground-truth evaluation keys:
  - **Public Task Data:** Blind diagram image (PNG) showing only the camera icon, mirror segment, target scratch $X$, and obstacles, accompanied by a clean natural-language prompt. **NEVER** render ray paths, normal vectors, or virtual cameras in public evaluation images.
  - **Private Evaluation Data:** Stored in separate records (`ground_truth.jsonl`) containing exact ray coordinates, mathematical proof reasons, and the true label.

## 4. End-to-End Deterministic Reproducibility
- Every procedurally generated scene and task must accept an integer `seed`.
- Passing the same seed on any machine must generate identical coordinates, identical geometries, and identical PNG pixels.
- All evaluation inference scripts must set model temperature to `0.0`.

## 5. Test Verification Before Completion Gate
- **NEVER** claim that a milestone is complete, a feature is working, or a bug is fixed unless real tests have actually been executed.
- Always run the relevant test suite (e.g. `pytest tests/`) via terminal and inspect the actual exit code and stdout.
- Document every test execution in `PROJECT_STATUS.md`.

## 6. Beginner-Friendly Explanations
- The user is a beginner. Always explain technical terms (e.g. virtual camera, FOV cone, counterfactual pair, Pydantic schema) in simple, accessible language **before** using them.
- Provide step-by-step guidance, clear file names, and exact next prompts.

## 7. Honest and Rigorous Reporting
- Do not flatter or disguise weaknesses. If an evaluated frontier model fails at a geometric task (e.g. gets 48% on counterfactual pairs), report that honestly as a profound scientific benchmark finding.
- Document limitations, assumptions, and edge cases clearly in project documentation.
