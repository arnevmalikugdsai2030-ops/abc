# MIRRORLAB: Evaluation Protocol & Provenance Specification

**Benchmark:** MIRRORLAB — The Missing Scratch (Specular Optical Visual Task Factory)  
**Track:** Innodata Frontier AI Hackathon 2026 — Track 2  
**Status:** FROZEN & VERIFIED  
**Version:** 2.1.0  

---

## 1. Executive Summary

This document specifies the scientific protocol governing model evaluation in MIRRORLAB. To prevent benchmark gaming, evaluation leakage, and unreplicable claims, MIRRORLAB mandates:
1. **Blind Evaluation Isolation**: Target models observe solely unannotated schematic diagrams and neutral task queries.
2. **Cryptographic Provenance**: Every run logs strict model metadata, timestamp, SHA-256 task manifest hashes, and provenance badges (`LIVE_API`, `CALIBRATED_SIMULATION_MOCK`, `UNVERIFIED_DEFAULT`).
3. **Deterministic 14-Category Failure Taxonomy**: Quantitative classification of spatial, geometric, and optical errors.
4. **Pair-Consistent Scoring**: Invariance verification across counterfactual pairs (Task D).

---

## 2. Benchmark Partitions & Task Split Integrity

The benchmark consists of 5 core task families ($A$ through $E$) partitioned into non-overlapping splits:

| Split | Description | Task ID Format | GT Exposure | SHA-256 Isolation |
|---|---|---|---|---|
| **`train`** | SFT training & calibration set | `task_vis_XXXX` | Fully transparent | Public & Private files separated |
| **`val`** | Hyperparameter tuning & prompt search | `task_vis_XXXX` | Fully transparent | Public & Private files separated |
| **`test`** | Blind test split for model competition | `task_pub_XXXX` (Neutral) | **Zero public GT** | Private ground truth in answer keys |

### Neutral Public ID Mapping
In the `test` split, all semantic hints (such as `_blocked_incident`, `_valid_visible`, or `_none`) are eliminated:
- Public task ID: `task_pub_0001`
- Diagram image: `image_pub_0001.png`
- Private key file: `data/benchmark_splits/test/private/ground_truth.jsonl` (contains mapping back to internal geometric scene parameters).

---

## 3. Evaluation Provenance Badges

To guarantee scientific honesty, MIRRORLAB enforces three provenance states at the abstract base adapter level (`mirrorlab/evaluation/model_adapter.py`):

```python
class BaseVisionAdapter(ABC):
    @property
    @abstractmethod
    def is_mock(self) -> bool:
        """True if the adapter operates without an external live API call."""
        ...

    @property
    @abstractmethod
    def evaluation_type(self) -> str:
        """Returns 'LIVE_API' or 'SIMULATION'."""
        ...

    @property
    @abstractmethod
    def provenance_status(self) -> str:
        """Returns 'VERIFIED_LIVE_RUN', 'CALIBRATED_SIMULATION_BASELINE', or 'UNVERIFIED_DEFAULT'."""
        ...
```

1. **`LIVE_API` / `VERIFIED_LIVE_RUN`**: Recorded when an authenticated HTTP/gRPC request is made to an external provider (Gemini 2.5 Flash, GPT-4o, Claude 3.5 Sonnet). The API latency, model version, and exact input tokens are hashed.
2. **`SIMULATION` / `CALIBRATED_SIMULATION_MOCK`**: Clearly flagged simulated responses used for pipeline unit testing and baseline sanity checks. NEVER presented as live empirical results.
3. **`UNVERIFIED_DEFAULT`**: Applied if any required provenance metadata is missing.

---

## 4. Prompt Neutrality & Schema

All tasks are presented using zero-shot, standardized system and user prompts with no leading hints:

### Prompt Template: Task A (Defect Detection)
```text
System: You are an expert automated optical inspection system.
User: [IMAGE]
Inspect the schematic diagram. The red 'X' indicates a target scratch defect.
Determine whether the target scratch is visible to the camera via reflection off the planar mirror.

Respond in strict JSON with the following schema:
{
  "visibility_state": "VISIBLE" | "NOT_VISIBLE",
  "reasoning": "<concise step-by-step optical rationale>"
}
```

### Prompt Template: Task B (Obstruction Attribution)
```text
Inspect the schematic diagram. The red 'X' target defect is occluded along the optical ray path.
Identify which obstacle blocks the ray and specify the optical leg on which the blockage occurs.

Respond in strict JSON with the following schema:
{
  "blocked_leg": "CAMERA_TO_MIRROR" | "MIRROR_TO_TARGET",
  "blocking_obstacle_id": "<obstacle id string>",
  "reasoning": "<concise step-by-step optical rationale>"
}
```

---

## 5. Metrics & Scoring Mathematical Definitions

### 5.1 Primary Accuracy
$$\text{Acc} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{y}_i = y_i)$$

### 5.2 Wilson Score 95% Confidence Interval
$$\hat{p} \pm \frac{z}{1 + z^2/N} \sqrt{\frac{\hat{p}(1-\hat{p})}{N} + \frac{z^2}{4N^2}}$$
Where $z = 1.96$ for the 95% confidence level.

### 5.3 Counterfactual Pair Consistency (Task D)
For a pair $P = (S_A, S_B)$ with expected invariant $R(y_A, y_B)$:
$$\text{PairConsistency} = \frac{1}{|P|} \sum_{(S_A, S_B) \in P} \mathbb{I}\left(R(\hat{y}_A, \hat{y}_B) \land (\hat{y}_A = y_A) \land (\hat{y}_B = y_B)\right)$$
A model receives credit for a pair if and only if **both** scenes are evaluated correctly according to the physical invariant.

---

## 6. Deterministic 14-Category Failure Taxonomy

All errors detected during evaluation runs are classified by `FailureTaxonomyClassifier`:

| Category | Definition | Automated Detection Trigger |
|---|---|---|
| **`OBJECT_DETECTION_FAILURE`** | Missed scene target/mirror/obstacle | Extracted entity list missing target or mirror |
| **`MIRROR_IDENTIFICATION_FAILURE`** | Mirror surface not found or misplaced | Failure to recognize planar segment |
| **`GEOMETRIC_ORIENTATION_FAILURE`** | Backside reflection error | Ray hits non-reflective rear, model predicts visible |
| **`VIRTUAL_IMAGE_REASONING_FAILURE`** | Infinite plane fallacy | Ray misses finite segment endpoints, model predicts visible |
| **`REFLECTION_PATH_ERROR`** | Erroneous reflection point $R$ | Computed reflection point deviates from Fermat point |
| **`OCCLUSION_LEG_ERROR`** | Leg 1 vs Leg 2 confusion | Blocked on incident leg, model attributes to reflected leg |
| **`SPATIAL_TRACKING_ERROR`** | Frustum / coordinate error | Ray outside camera FOV cone, model predicts in-bounds |
| **`COUNTERFACTUAL_INCONSISTENCY`** | Invariant broken on pair | Answers flip on cosmetic variation or fail on angle flip |
| **`NUMERICAL_CALCULATION_ERROR`** | Arithmetic/angle error | Explicit angle calculation contradicted by geometry |
| **`INSTRUCTION_FOLLOWING_FAILURE`** | Invalid JSON schema/type | Output not adhering to requested Pydantic schema |
| **`INVALID_MALFORMED_RESPONSE`** | Empty or corrupted output | HTTP timeout, empty payload, or truncation |
| **`CORRECT_ANSWER_INVALID_REASONING`** | Spurious correct guess | Correct boolean answer but contradictory physical rationale |
| **`AMBIGUOUS_GROUND_TRUTH`** | Borderline geometry | Scene quarantined by AmbiguityValidator ($< 2.0\text{ mm}$ margin) |
| **`UNKNOWN_OTHER`** | Unclassified failure | Anomaly not matching above patterns |

---

## 7. Execution CLI Commands

### Run Full Offline Calibrated Benchmark Evaluation:
```bash
python scripts/run_offline_evaluation.py --split test --output logs/evaluations/
```

### Audit Dataset Integrity & Blind Isolation (11 Checks):
```bash
python -c "from mirrorlab.evaluation.dataset_auditor import DatasetAuditor; print(DatasetAuditor().audit_all().summary)"
```

### Verify Triple Consensus (Oracle, Fermat, Snell):
```bash
python scripts/audit_triple_verifiers.py
```
