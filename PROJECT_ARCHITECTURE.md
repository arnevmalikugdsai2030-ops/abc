# MIRRORLAB: System Architecture & Technical Specification

> **Project:** MIRRORLAB — The Missing Scratch  
> **Competition:** Innodata Frontier AI Hackathon 2026 — Track 2: Visual Task Factory  
> **Domain:** Machine-Vision Geometric Optics Reasoning Benchmark for Vision-Language Models (VLMs)  

---

## 1. Executive Summary & Core Concept

Industrial machine vision inspection systems frequently monitor recessed components, engine blocks, or cavity walls where direct line-of-sight is impossible. Such setups deploy planar or articulated mirrors to inspect critical flaws (such as surface scratches, micro-cracks, or weld seams).

Evaluating whether a vision-language model (VLM) can reliably inspect these setups requires more than pattern recognition: it demands **rigorous 2D spatial reasoning and geometric optics understanding** (the Law of Reflection: $\theta_i = \theta_r$).

**MIRRORLAB** is an automated, synthetic Visual Task Factory that produces mathematically validated inspection diagrams and corresponding multi-step reasoning benchmarks across five core task categories. Ground truth is determined strictly through **deterministic computational geometry**, never via an LLM.

```
+-----------------------------------------------------------------------------------+
|                              MIRRORLAB ARCHITECTURE                               |
+-----------------------------------------------------------------------------------+

 [ Scene Parameter Config ]
           │
           ▼
 ┌─────────────────────────────────────────────────────────┐
 │        1. Deterministic Geometry Engine (Core)         │
 │  - Virtual Camera (Mirror Method)                      │
 │  - Finite Mirror Line-Segment Clipping                 │
 │  - Field of View (FOV) Angular Cones                   │
 │  - Obstacle Polygon Occlusion via Shapely/NumPy        │
 └────────────────────────────┬────────────────────────────┘
                              │
               Deterministic Ray Proof & Answer Key
                              │
           ┌──────────────────┴──────────────────┐
           ▼                                     ▼
 ┌─────────────────────────────────┐   ┌─────────────────────────────────┐
 │   2. Task Factory & Generators  │   │     3. Visual Diagram Engine    │
 │ - Task A: Visibility Predict    │   │ - Blind Diagram (Model Input)   │
 │ - Task B: Obstruction Reason    │   │ - Ground-Truth Debug Overlay    │
 │ - Task C: Mirror Angle Repair   │   │ - Cosmetic Invariance Controls  │
 │ - Task D: Counterfactual Pairs  │   │   (styles, line weights, noise) │
 └─────────────────┬───────────────┘   └─────────────────┬───────────────┘
                   │                                     │
                   └──────────────────┬──────────────────┘
                                      ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                       4. Ground-Truth Data Boundary                         │
 │                                                                             │
 │  [ Public Benchmark Sample ]             [ Private Evaluation Key ]         │
 │  - Diagram Image (PNG)                    - Binary Answer (True/False / Id) │
 │  - Question Prompt (JSON)                 - Exact Ray Path Geometry         │
 │  - Task Type & Metadata (blinded)         - Geometric Proof Reason          │
 └────────────────────────────────────┬────────────────────────────────────────┘
                                      │
                                      ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                      5. Evaluation & Analysis Harness                       │
 │  - Vision Model API Runners (Gemini, Claude, GPT-4o, Local VLM)             │
 │  - Strict Structured Parsing & Verification                                 │
 │  - Metrics: Accuracy, Counterfactual Sensitivity, Invariance Stability       │
 └────────────────────────────────────┬────────────────────────────────────────┘
                                      │
                                      ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │              6. Interactive Streamlit Studio & Inspector                    │
 │  - Live Scene Builder with instant optical ray reflection                   │
 │  - Benchmark Inspector (visual failure analysis & confusion matrices)       │
 │  - Physical World Setup Guide & Reproducibility Specs                       │
 └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Mathematical Foundation: Geometric Optics via Virtual Camera

### 2.1 The Principle of Specular Reflection
For a flat mirror represented as a finite segment $M = [M_1, M_2] \subset \mathbb{R}^2$ with unit normal $\hat{n}$:
1. A ray from camera position $C$ hits the mirror at point $P \in M$.
2. The reflected ray reaches target scratch $S$ if and only if the angle of incidence equals the angle of reflection ($\theta_i = \theta_r$).

### 2.2 The Virtual Camera Method
Instead of tracing infinite forward rays or backward cone-sampling, we use the **Virtual Camera Reflection Method**:
1. Reflect camera point $C$ across the infinite mirror line $L_M$ to obtain virtual camera position $C'$.
   $$C' = C + 2 \, d(C, L_M) \, \hat{n}_M$$
2. Draw a straight line segment $L_{C'S}$ from virtual camera $C'$ to scratch $S$.
3. Check the intersection point $P = L_{C'S} \cap L_M$:
   - If $P$ does not lie on the finite mirror segment $[M_1, M_2]$, the reflection does **not** occur on the mirror (missed mirror).
   - If $P$ lies on the back (non-reflective side) of the mirror, the reflection is invalid.
4. If $P \in [M_1, M_2]$, the actual optical path is the polyline $C \to P \to S$.

### 2.3 Field-of-View (FOV) Cones
The real camera has a finite angular FOV $[\theta_{\text{min}}, \theta_{\text{max}}]$ and maximum range $R_{\text{max}}$.
- The initial ray vector is $\vec{v} = P - C$.
- The angle $\angle(\vec{v}, \vec{u}_{\text{camera}})$ must fall within $[-\text{FOV}/2, +\text{FOV}/2]$.
- Distance $\|P - C\| + \|S - P\| \le R_{\text{max}}$.

### 2.4 Obstacle Occlusion
Obstacles are modelled as 2D convex or non-convex polygons $\mathcal{O}_1, \dots, \mathcal{O}_k$.
- **Incident Ray Occlusion:** Segment $C \to P$ must not intersect the interior or boundary of any obstacle $\mathcal{O}_i$.
- **Reflected Ray Occlusion:** Segment $P \to S$ must not intersect any obstacle $\mathcal{O}_i$.
- **Direct Line of Sight:** Optional check whether $C \to S$ is directly visible without mirror.

---

## 3. Five Mandatory Benchmark Tasks

| Task ID | Name | Description | VLM Challenge |
|---|---|---|---|
| **Task A** | **Visibility Prediction** | Binary classification: Can camera $C$ see scratch $X$ through the mirror? | Multi-bounce ray spatial reasoning; testing if model understands mirror boundaries. |
| **Task B** | **Obstruction Identification** | Multiple-choice: If scratch $X$ is not visible, why? Options: `[Out of FOV, Misses Mirror, Blocked by Obstacle 1, Blocked by Obstacle 2, Back of Mirror]` | Diagnostic causal reasoning; pinpointing the geometric failure point. |
| **Task C** | **Mirror-Setting Repair** | Given 3-4 candidate mirror orientations/angles $[\theta_A, \theta_B, \theta_C]$, select the setting that restores visibility of $X$. | Inverse optical problem solving and angular mental rotation. |
| **Task D** | **Counterfactual Paired Evaluation** | Pairs of scenes $(S, S')$ where a minimal perturbation (e.g. mirror rotated $2^\circ$ or obstacle shifted by 5 units) flips visibility from True to False. | Eliminates visual shortcut exploitation. A model guessing by surface layout fails the pair consistency metric. |
| **Task E** | **Cosmetic Invariance Controls** | Identical underlying geometry rendered with distinct visual themes (Blueprint, Dark Mode, Industrial Monochrome, CAD wireframe, noisy sensor). | Assesses whether perception is invariant to diagram style changes while geometry is held constant. |

---

## 4. Software & Directory Architecture

```
mirrorlab/
├── pyproject.toml              # Project dependencies, build config, pytest config
├── README.md                   # Hackathon overview, quickstart, demo instructions
├── PROJECT_STATUS.md           # Milestone tracking, test reports, changelog
├── PROJECT_ARCHITECTURE.md     # This technical design document
│
├── mirrorlab/
│   ├── __init__.py
│   ├── geometry/               # Core deterministic computational geometry
│   │   ├── __init__.py
│   │   ├── primitives.py       # Point2D, Vector2D, LineSegment2D, Polygon2D
│   │   ├── mirror.py           # MirrorSegment2D (normals, front/back face)
│   │   ├── camera.py           # Camera2D (position, heading, FOV, range)
│   │   ├── optics.py           # Virtual camera reflection, ray solver, obstruction tester
│   │   └── validator.py        # Numerical stability and edge-case guards
│   │
│   ├── scene/                  # Scene definitions and Pydantic schemas
│   │   ├── __init__.py
│   │   ├── models.py           # InspectionScene, Obstacle, TargetDefect
│   │   ├── ground_truth.py     # GroundTruthResult, RayPath, OcclusionType
│   │   └── generator.py        # Parametric & randomized scene generators
│   │
│   ├── renderer/               # Visual rendering engine (Pillow / Matplotlib)
│   │   ├── __init__.py
│   │   ├── styles.py           # Themes: Industrial, Blueprint, Dark, CAD, SensorNoise
│   │   ├── diagram.py          # DiagramRenderer (generates clean blind inspection images)
│   │   └── debug_overlay.py    # OverlayRenderer (draws rays, normals, virtual cameras)
│   │
│   ├── tasks/                  # The 5 visual benchmark task factories
│   │   ├── __init__.py
│   │   ├── base.py             # BaseTask, TaskSample, TaskPrompt
│   │   ├── task_a_visibility.py
│   │   ├── task_b_obstruction.py
│   │   ├── task_c_repair.py
│   │   ├── task_d_counterfactual.py
│   │   └── task_e_cosmetic.py
│   │
│   ├── benchmark/              # Dataset generation and evaluation harness
│   │   ├── __init__.py
│   │   ├── dataset.py          # BenchmarkDataset serializer (JSONL + PNG bundle)
│   │   ├── evaluator.py        # Model evaluation driver (API calls, retries)
│   │   ├── parser.py           # Robust JSON / regex response extraction
│   │   └── metrics.py          # Accuracy, Pair Consistency, Style Invariance scores
│   │
│   └── app/                    # Interactive Streamlit Application
│       ├── __init__.py
│       ├── main.py             # Streamlit app entrypoint
│       ├── components/         # Interactive canvas, diagram viewer, benchmark charts
│       └── physical_demo.py    # Guide & printable target for physical camera demonstration
│
└── tests/                      # Comprehensive pytest test suite
    ├── test_geometry.py        # Unit tests for ray reflection, normals, virtual camera
    ├── test_optics.py          # Obstruction tests, FOV tests, edge clipping
    ├── test_tasks.py           # Unit tests for Task A, B, C, D, E generation
    ├── test_renderer.py        # Test image generation and dimensions
    └── test_evaluator.py       # Test parsing and scoring logic
```

---

## 5. Technology Stack & Justification

| Technology | Purpose | Justification |
|---|---|---|
| **Python 3.11+** | Core Language | Universally accessible, fast iteration, deep ecosystem. |
| **uv** | Fast Virtual Environment & Package Manager | Instant dependency installs, zero lockfile discrepancies. |
| **NumPy** | Vector and Matrix Math | Fast vectorized geometric operations and vector reflections. |
| **Shapely** | 2D Computational Geometry | Robust, standard C-backed geometric predicate evaluation (`intersects`, `distance`, `contains`). |
| **Pillow (PIL)** | Diagram Rendering | Precise pixel-level vector diagram rendering without external GUI server dependencies. |
| **Pydantic v2** | Data Modeling & Validation | Strict serialization, schemas, and public/private ground truth separation. |
| **pytest** | Automated Testing | Verified regression testing for every single milestone. |
| **Streamlit & Plotly** | Interactive Web UI | Clean browser app for judges to interact with live mirrors, run benchmarks, and inspect errors. |
| **Google GenAI / Anthropic / OpenAI APIs** | Model Evaluation | Support evaluating state-of-the-art vision models on the generated tasks. |

---

## 6. Ground Truth Integrity & Blind Evaluation Protocol

A strict boundary is maintained between **Public Benchmark Data** and **Ground Truth Data**:
1. **Public Sample Data (`TaskSample`):**
   - Diagram Image: Only camera symbol, mirror line, target scratch `X`, obstacles, and candidate labels are drawn. Ray paths, reflection vectors, and virtual cameras are **never** rendered into the evaluation image.
   - Question Prompt: Markdown-formatted text explaining the camera location, mirror, and defect, asking the specific task question.
2. **Private Ground Truth Key (`GroundTruthProof`):**
   - Stored in a separate protected JSON structure / file.
   - Contains exact calculated coordinates of intersection $P$, ray segments $C \to P$ and $P \to S$, angles $\theta_i, \theta_r$, obstacle intersection details, and the indisputable ground truth answer.
3. **No LLM Hallucinations in Ground Truth:**
   - 100% of answer keys are computed via analytic geometry.

---

## 7. Physical Machine-Vision Inspection Alignment

In industrial manufacturing (e.g. CNC machining, EV battery inspection, automotive powertrain housings):
- Parts have blind cavities where internal bore holes or sidewalls cannot be viewed directly by fixed production line cameras.
- Mirrored borescope or angled inspection mirrors redirect illumination and camera sightlines.
- MIRRORLAB's diagrams represent top-down orthogonal schematics of real-world industrial inspection stations.
- A physical verification test (using a real mirror, camera/phone, printed target, and obstruction block) will be documented with photos and coordinate mapping to establish real-world fidelity.
