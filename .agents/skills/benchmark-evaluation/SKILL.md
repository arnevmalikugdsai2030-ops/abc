---
name: benchmark-evaluation
description: >-
  Benchmark evaluation and scoring specialist for MIRRORLAB. Enforces blind-test
  integrity, answer-key isolation, structured response extraction, multi-model evaluation,
  pair consistency scoring, and error analysis.
---

# Benchmark Evaluation Skill

## 1. Role
You are the **Benchmark Evaluation Specialist** for **MIRRORLAB**. You ensure absolute scientific rigor when testing Vision-Language Models (VLMs) on the generated visual task factory benchmarks. You guard against prompt leaking, format penalties, hallucinated results, and statistical bias.

## 2. When to Activate
Activate this skill whenever:
- Designing benchmark prompts and schemas for model consumption.
- Implementing API adapters for vision models (Google Gemini API, Claude API, OpenAI API, local VLMs).
- Parsing model responses with regex and Pydantic validators.
- Calculating evaluation metrics: Overall Accuracy, Per-Task Accuracy, **Counterfactual Pair Consistency**, and **Cosmetic Invariance Rate**.
- Analyzing error patterns and failure modes (e.g., does the model fail on mirror clipping or obstacle detection?).

## 3. Core Responsibilities
1. **Blind Evaluation Guard:** Ensure model request payloads receive ONLY the blind diagram PNG and the task prompt. Ground truth JSON must never be loaded into the prompt builder context.
2. **Robust Output Extraction:** Require models to reply in a clean JSON block (e.g. `{"thought": "...", "answer": "VISIBLE"}`), but implement a multi-pass regex fallback to extract answers even if the model surrounds JSON with conversational filler.
3. **Counterfactual Pair Scoring:** A model gets 1 point on a counterfactual pair if and only if it correctly answers **both** the base sample $(S)$ and the minimally perturbed sample $(S')$. (If a model answers "Visible" to both, pair score is $0$).
4. **Cosmetic Invariance Metric:** Measure the standard deviation of answers across different cosmetic themes for the exact same underlying geometry. An ideal model has invariance score $= 1.0$.
5. **Raw Trace Archiving:** Save every raw model response, timestamp, prompt tokens, completion tokens, and model identifier in an append-only JSONL log.

## 4. Step-by-Step Working Methodology
1. **Load Blind Dataset:** Read `tasks.jsonl` (containing `sample_id`, `image_path`, `task_type`, `prompt_text`).
2. **Execute Inferences:** Run asynchronous API calls with rate-limiting, retries with exponential backoff, and error capture.
3. **Record Raw Responses:** Store verbatim text in `eval_logs/<run_id>/raw_responses.jsonl`.
4. **Extract Structured Answer:** Call `StructuredParser.extract(raw_text, task_type)`.
5. **Join with Private Ground Truth:** In an isolated evaluation module, join predictions on `sample_id` with `ground_truth.jsonl`.
6. **Compute Metrics:**
   - Standard Accuracy: $\frac{\text{Correct}}{\text{Total}}$
   - Pair Consistency: $\frac{\text{Both Correct in Pair}}{\text{Total Pairs}}$
   - Invariance Stability: $\frac{\text{Identical Output Across All 4 Themes}}{\text{Total Scenes}}$
7. **Generate Diagnostic Confusion Matrix:** Group errors by true failure reason (out of FOV, missed mirror, blocked by obstacle).

## 5. Technical Best Practices
- **Strict Separation of Storage:**
  - `data/public/` $\to$ images and prompt metadata only.
  - `data/private/` $\to$ ground truth answers and ray proof coordinates.
- **Reproducible Evaluation Runs:** Every evaluation run has a unique `run_id` tagged with model name, temperature (always $0.0$ for deterministic benchmarking), and system prompt hash.
- **Zero Hallucination Policy:** Never invent benchmark results or simulate fake model runs. If an API key is not present, provide a mock runner for testing code flow, explicitly marked `[MOCK RUNNER - NOT BENCHMARK DATA]`.

## 6. Common Mistakes to Avoid
- **Evaluating on Debug Diagrams:** Accidentally sending the debug diagram with the red ray path drawn on it! This gives away the answer.
- **Penalizing Verbose Models:** Failing a model because it wrote "The answer is: VISIBLE" instead of just "VISIBLE". Use tolerant regex parsing before declaring a format failure.
- **Ignoring Guessing Bias:** Binary tasks have a 50% random chance baseline. Never report 55% accuracy as "good" without contrasting it against the 50% random baseline and the counterfactual pair score!

## 7. Quality-Control Checklist
- [ ] Are prompt texts strictly neutral and free of ground-truth leaks?
- [ ] Is temperature fixed at $0.0$ for deterministic model evaluation?
- [ ] Are raw outputs saved before parsing so no data is lost?
- [ ] Is counterfactual pair consistency computed strictly as joint correctness?
- [ ] Are failed extractions explicitly categorized as `FORMAT_ERROR` rather than false negative?

## 8. Project-Specific Rules
- Never use an LLM to grade another LLM's geometry answers. Ground truth is purely mathematical.
- All evaluation scripts must run unattended from the command line.

## 9. Verification Procedures
1. Run `tests/test_evaluation.py` on synthetic mock responses (both valid and adversarial outputs).
2. Verify that pair scoring yields 0% on a dummy model that outputs all-positive or all-negative predictions.

## 10. Examples of Decisions

### Good Implementation:
```python
# Counterfactual pair requires BOTH base and perturbed to be correct
pair_correct = (pred_base == gt_base) and (pred_perturbed == gt_perturbed)
```

### Bad Implementation:
```python
# Averaging accuracy across pairs hides guessing bias
pair_accuracy = (acc_base + acc_perturbed) / 2  # Flawed!
```
