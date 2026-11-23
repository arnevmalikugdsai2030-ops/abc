# MIRRORLAB Versioned Prompt Library Specification

> **Prompt Library Status:** Verified & Active  
> **Prompts Registered:** 3 Benchmark Tasks (Task A, Task B, Task C)  
> **Schema Standard:** Strict JSON Schema (Draft 2020-12)  
> **Audit Status:** 18 / 18 Prompt Tests Passing | 189 / 189 Total Repository Tests Passing  

---

## 1. Overview & Information Boundary Guard

The **MIRRORLAB Versioned Prompt Library** ([`mirrorlab/prompts/`](file:///c:/Users/malik/Desktop/inno/mirrorlab/prompts/)) standardizes all visual-language instructional prompts presented to VLMs during blind evaluation.

### Zero-Leakage Guarantee
The model prompt receives **ONLY**:
1. High-level apparatus legend (Camera, Mirror, Red 'X' defect, Obstacles).
2. The fundamental physical law (specular reflection $\theta_i = \theta_r$).
3. A short, neutral, unambiguous inspection question.
4. An analysis checklist reminding the model of physical edge conditions.
5. Strict JSON output instructions with schema property definitions.

> [!CAUTION]
> **Strict Physical Isolation:** Under no circumstance does the prompt builder or model payload receive:
> - Ground-truth answers (`YES`, `NO`, `CLEAR`, candidate winner).
> - Ray-tracing proof coordinates or virtual target locations.
> - Geometry overlay diagrams (only blind diagrams are ever transmitted).
> - Diagnostic failure reason codes or tolerance thresholds ($t_{\text{mirror}}$).
> - Boundary tags (`is_boundary=True`).

---

## 2. Versioned Prompt Catalog & SHA-256 Fingerprints

| Task Type | Prompt ID | SemVer | Cryptographic SHA-256 Version Hash | Description |
|---|---|---|---|---|
| **Task A: Visibility** | `prompt_vis_v1` | `1.0.0` | `2e0dd15ca8840c8fa90e964344585c5b4694931a2386fe8fa8024220b30ffbca` | Finite specular mirror visibility and failure mode attribution. |
| **Task B: Obstruction** | `prompt_obs_v1` | `1.0.0` | `01aa99c82ca1a62ebfa235bb7f80db26baef687483ff1c8feec095c464e81552` | Obstacle collision detection, blocker identification, and path segment attribution. |
| **Task C: Repair** | `prompt_repair_v1` | `1.0.0` | `a9069d2d937073286cae1329c4db22ef252f534346eb4a7967b5ebefd38aa8d6` | Mirror configuration repair and single-winner candidate selection. |

---

## 3. Detailed Prompt Specifications & Schemas

### 3.1 Prompt 1: Visibility (`prompt_vis_v1`)
- **Question:** `"Can the camera see the center of the red X through the mirror?"`
- **Required JSON Fields:**
  - `visible`: `boolean` (`true` if visible, `false` otherwise)
  - `answer`: `"YES"` | `"NO"`
  - `reason`: `"CLEAR"` | `"MIRROR_MISS"` | `"OUTSIDE_FOV"` | `"WRONG_SIDE"` | `"BLOCKED_CAMERA_TO_MIRROR"` | `"BLOCKED_MIRROR_TO_TARGET"`
  - `explanation`: `string`

#### Strict JSON Schema:
```json
{
  "type": "object",
  "required": ["visible", "answer", "reason", "explanation"],
  "properties": {
    "visible": { "type": "boolean" },
    "answer": { "type": "string", "enum": ["YES", "NO"] },
    "reason": {
      "type": "string",
      "enum": [
        "CLEAR",
        "MIRROR_MISS",
        "OUTSIDE_FOV",
        "WRONG_SIDE",
        "BLOCKED_CAMERA_TO_MIRROR",
        "BLOCKED_MIRROR_TO_TARGET"
      ]
    },
    "explanation": { "type": "string" }
  },
  "additionalProperties": false
}
```

#### Example Valid Response:
```json
{
  "visible": true,
  "answer": "YES",
  "reason": "CLEAR",
  "explanation": "Specular reflection ray strikes within mirror segment AB, falls within camera FOV, faces the reflective front, and is not obstructed by barriers."
}
```

---

### 3.2 Prompt 2: Obstruction (`prompt_obs_v1`)
- **Question:** `"Is the optical reflection path between the camera and the scratch blocked by an obstacle?"`
- **Required JSON Fields:**
  - `blocked`: `boolean` (`true` if blocked, `false` if clear)
  - `answer`: `"YES"` | `"NO"`
  - `obstacle_id`: `"O1"` | `"O2"` | `"O3"` | `"NONE"`
  - `blocked_leg`: `"CAMERA_TO_MIRROR"` | `"MIRROR_TO_TARGET"` | `"NONE"`
  - `explanation`: `string`

#### Strict JSON Schema:
```json
{
  "type": "object",
  "required": ["blocked", "answer", "obstacle_id", "blocked_leg", "explanation"],
  "properties": {
    "blocked": { "type": "boolean" },
    "answer": { "type": "string", "enum": ["YES", "NO"] },
    "obstacle_id": { "type": "string", "enum": ["O1", "O2", "O3", "NONE"] },
    "blocked_leg": { "type": "string", "enum": ["CAMERA_TO_MIRROR", "MIRROR_TO_TARGET", "NONE"] },
    "explanation": { "type": "string" }
  },
  "additionalProperties": false
}
```

#### Example Valid Response:
```json
{
  "blocked": true,
  "answer": "YES",
  "obstacle_id": "O2",
  "blocked_leg": "CAMERA_TO_MIRROR",
  "explanation": "Barrier O2 physically intersects the incident sightline between the camera lens and mirror reflection point."
}
```

---

### 3.3 Prompt 3: Repair (`prompt_repair_v1`)
- **Question:** `"Which candidate mirror setting (A, B, or C) restores unobstructed visibility from the camera to the red X scratch?"`
- **Required JSON Fields:**
  - `setting`: `"A"` | `"B"` | `"C"`
  - `answer`: `"A"` | `"B"` | `"C"`
  - `explanation`: `string`

#### Strict JSON Schema:
```json
{
  "type": "object",
  "required": ["setting", "answer", "explanation"],
  "properties": {
    "setting": { "type": "string", "enum": ["A", "B", "C"] },
    "answer": { "type": "string", "enum": ["A", "B", "C"] },
    "explanation": { "type": "string" }
  },
  "additionalProperties": false
}
```

#### Example Valid Response:
```json
{
  "setting": "B",
  "answer": "B",
  "explanation": "Candidate setting B tilts the mirror such that specular reflection strikes within mirror bounds inside the camera FOV, while candidates A and C miss the mirror or are blocked by barrier O1."
}
```

---

## 4. Python API Usage

```python
from mirrorlab.prompts import (
    get_prompt,
    render_prompt,
    get_prompt_schema,
    get_prompt_example,
    list_all_prompts,
)
from mirrorlab.schema.common import TaskType

# 1. Render complete prompt text for model input
visibility_prompt_text = render_prompt(TaskType.TASK_A_VISIBILITY)

# 2. Retrieve JSON schema for validation
schema = get_prompt_schema(TaskType.TASK_B_OBSTRUCTION)

# 3. Retrieve prompt by specific version
prompt_v1 = get_prompt(TaskType.TASK_C_REPAIR, version="1.0.0")
print(f"Prompt ID: {prompt_v1.metadata.prompt_id}, Hash: {prompt_v1.version_hash}")

# 4. Validate model response
is_valid, error_msg = prompt_v1.validate_response(
    {"setting": "A", "answer": "A", "explanation": "..."}
)
```
