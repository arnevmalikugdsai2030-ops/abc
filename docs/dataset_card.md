# MIRRORLAB Dataset Card
## Benchmark Dataset for Specular Reflection & Spatial Optical Reasoning

**Document Version:** 1.0.0  
**Standard:** Hugging Face / NeurIPS Dataset Card Specification  
**License:** MIT License  
**Provenance:** 100% Procedurally Synthesized via Triple-Verified Vector Geometry  

---

## 1. Dataset Summary

MIRRORLAB is a benchmark dataset designed to evaluate the physical optical reasoning capabilities of Vision-Language Models (VLMs). The dataset contains 2D vector technical diagrams representing an industrial cavity inspection workcell, wherein a fixed camera attempts to inspect a target scratch defect via a finite planar deflection mirror amidst opaque barriers.

### Key Quantitative Indicators

| Metric | Dev Split | Val Split | Test Split | Benchmark Total |
|---|---|---|---|---|
| **Top-Level Task Packages** | 46 | 46 | 138 | **230** |
| **Model-Facing Task Queries** | 70 | 70 | 210 | **350** |
| **Counterfactual Paired Bundles** | 24 | 24 | 72 | **120** |
| **Unique Inspection Scenes** | 70 | 70 | 210 | **350** |
| **Public Blind Images (PNG)** | 70 | 70 | 210 | **350** |
| **Private Diagnostic Overlays (PNG)** | 70 | 70 | 210 | **350** |
| **Total Rendered Diagrams** | 140 | 140 | 420 | **700** |

---

## 2. Task Family Distribution

The 350 model-facing tasks span 5 distinct task families:

```
  TASK DISTRIBUTION ACROSS BENCHMARK FAMILIES
  
  [ Task A: Visibility Prediction ]     ==================== (110 tasks)
  [ Task B: Obstruction Identification] ============ (60 tasks)
  [ Task C: Mirror Setting Repair ]     ======== (40 tasks)
  [ Task D: Counterfactual Pairs ]      ====================== (120 tasks, 240 child scenes)
  [ Task E: OOD Stress Testing ]        ==== (20 tasks)
```

1. **Task A (Visibility Prediction, N=110):** Binary judgment whether camera detects scratch in mirror.
2. **Task B (Obstruction Identification, N=60):** Identification of blocking obstacle ID (`O1`, `O2`, `O3`, `NONE`) and blocked path leg (`CAMERA_TO_MIRROR`, `MIRROR_TO_TARGET`, `NONE`).
3. **Task C (Mirror-Setting Repair, N=40):** 3-candidate multiple choice (`A`, `B`, `C`) selecting the unique mirror rotation that restores optical sightline.
4. **Task D (Counterfactual Pairs, N=120):** Paired scenes evaluating sensitivity to $\Delta \theta = 2^\circ$ mirror tilts, obstacle insertions, and cosmetic controls.
5. **Task E (Out-of-Distribution, N=20):** Stress-testing on extreme clutter ($5-8$ obstacles), acute grazing angles ($<35^\circ$), and skewed canvas aspect ratios.

---

## 3. Dataset Structure & Formats

### 3.1 Public Task Format (`tasks.jsonl`)
Zero ground-truth leakage. Contains only prompt and image pointers:

```json
{
  "task_id": "ml_test_0001",
  "task_type": "TASK_A_VISIBILITY",
  "question": "Can the camera see the target scratch via the mirror?",
  "prompt": "You are an industrial machine-vision inspector analyzing a 2D optical inspection diagram...",
  "image_filename": "img_test_0001.png"
}
```

### 3.2 Private Ground Truth Format (`ground_truth.jsonl`)
Air-gapped and stored exclusively in private benchmark directories:

```json
{
  "task_id": "ml_test_0001",
  "internal_task_id": "task_vis_10015",
  "internal_scene_id": "task_vis_10015",
  "split": "test",
  "task_family": "TASK_A_VISIBILITY",
  "can_see": false,
  "answer": "NO",
  "reason_code": "WRONG_SIDE",
  "detailed_explanation": "Mathematical Proof: Reflection failed: WRONG_SIDE_BOTH.",
  "proof_geometry": {
    "reflection_point": null,
    "virtual_target": null,
    "incident_angle_deg": null,
    "reflected_angle_deg": null,
    "total_optical_distance": null,
    "optical_path": []
  }
}
```

### 3.3 Counterfactual Pairs Manifest (`pairs.jsonl`)
Specifies child scene relationships and controlled transformation categories:

```json
{
  "pair_id": "pair_test_0001",
  "split": "test",
  "pair_type": "TYPE_1_MIRROR_ANGLE_FLIP",
  "child_a_task_id": "ml_test_0045",
  "child_b_task_id": "ml_test_0046",
  "child_a_image": "img_test_0045.png",
  "child_b_image": "img_test_0046.png",
  "expected_relationship": "FLIPPED",
  "transformation": {
    "category": "MIRROR_ROTATION",
    "changed_parameter": "mirror_angle_degrees",
    "original_values": {"mirror_angle_deg": 180.0},
    "new_values": {"mirror_angle_deg": 183.5},
    "difference_summary": "Perturbed mirror orientation by 3.50 deg"
  }
}
```

---

## 4. Benchmark Quality & Verification Certificates

The dataset is verified against 11 rigorous mathematical integrity criteria:
- **100% ID Uniqueness:** Zero duplicate task IDs across all partitions.
- **Zero Public Leakage:** Zero ground-truth fields (`answer`, `can_see`, `is_blocked`, `blocker_label`, `proof_geometry`) in public files.
- **Zero Semantic Leakage:** All public task IDs (`ml_test_XXXX`) and image filenames (`img_test_XXXX.png`) are non-semantic sequential hashes.
- **Pair File Non-Leakage:** `pairs.jsonl` is strictly private in the test split.
- **Transformation Schema Integrity:** 100% of counterfactual pairs include explicit parameter deltas.
- **Triple-Verifier Consensus:** 100% agreement between Oracle, Independent Verifier, and Direct Snell Ray Verifier.

---

## 5. Intended Use & Limitations

- **Intended Use:** Benchmarking multi-modal Vision-Language Models on physical geometric optics, spatial reasoning, and visual necessity.
- **Limitations:** Limited to 2D planar schematics with flat first-surface mirrors. Does not model volumetric 3D optics, curved mirrors (parabolic/spherical), or wave interference/polarization effects.

---
*Maintained by the MIRRORLAB Dataset Working Group.*
