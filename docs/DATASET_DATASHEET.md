# MIRRORLAB Dataset Datasheet
*Authored following the Datasheets for Datasets standard (Gebru et al., 2021)*

---

## 1. Motivation

### For what purpose was the dataset created?
Current Vision-Language Models (VLMs) frequently achieve >90% accuracy on standard visual question-answering benchmarks by exploiting semantic context, surface textures, and linguistic priors. However, frontier AI systems systematically fail at **specular optical reasoning**—predicting non-line-of-sight visual paths reflected across finite planar mirrors in industrial machine-vision settings. MIRRORLAB was created to provide a mathematically rigorous, zero-leakage, counterfactually controlled benchmark that tests whether multi-modal models possess true computational geometry and specular reflection capabilities.

### Who created the dataset?
Created by the MIRRORLAB research team for the **Innodata Frontier AI Hackathon 2026 — Track 2: Visual Task Factory**.

### Who funded the creation of the dataset?
Developed independently under open-source academic research terms for the Innodata Frontier AI Hackathon 2026.

---

## 2. Composition

### What do the instances that comprise the dataset represent?
Each instance represents an industrial surface inspection cell schematic containing:
- A camera with calibrated position $C$, heading angle $\theta_c$, and finite field-of-view cone (typically $60^\circ - 75^\circ$).
- A finite planar front-surface or back-surface mirror segment $AB$ with defined reflective normal vector $\mathbf{n}$.
- A specimen surface with a target scratch defect $T$ (indicated by a red cross).
- Convex obstacles (rectangles and circles) representing industrial fixtures, clamps, and baffles.
- Ground-truth specular ray paths computed via three independent mathematical solvers.

### How many instances are there in total?
The authoritative frozen benchmark dataset committed to disk comprises **350 model-facing tasks** partitioned across three splits, verified in `data/dataset_manifest.json`:
- **`dev` split**: 70 tasks (70 blind images, 70 diagnostic overlays).
- **`val` split**: 70 tasks (70 blind images, 70 diagnostic overlays).
- **`test` split**: 210 blind tasks (210 blind images, 210 diagnostic overlays sealed in `private/overlays`).

In addition, `data/` contains **230 core task directories** across the 5 task families (20 Task A, 30 Task B, 30 Task C, 120 Task D counterfactual pairs, 30 Task E OOD), with **2,000 total rendered PNG images** on disk.

*(Note: Prior design documents referenced an aspirational 1,500-instance scaling throughput capacity for the synthetic procedural generator; the authoritative frozen competition evaluation benchmark consists of the 350 partitioned tasks).*

### What data does each instance consist of?
1. **Public Blind Image (`.png`)**: A 1024×1024 high-DPI engineering schematic displaying only the camera, mirror, obstacles, and target scratch. Zero ray paths, zero answers, and zero diagnostic annotations are present.
2. **Public Task Query (`.jsonl`)**: Standardized natural-language prompt specifying the inspection question.
3. **Private Ground-Truth Record (`.jsonl`)**: Retained strictly in private answer keys. Contains floating-point apparatus coordinates, reflection point $R$, virtual target $T'$, exact obstruction leg, blocker IDs, and diagnostic physical explanations.

---

## 3. Collection Process

### How was the data associated with each instance acquired?
All instances are synthetically generated using MIRRORLAB's procedural `SceneGenerator` and `TaskExplorer` constraint solvers. Geometry is generated from deterministic pseudo-random seeds ensuring exact byte-level reproducibility.

### What mechanisms or procedures were used to ensure quality?
Every candidate scene is evaluated through a **7-Layer Automated Quality Gate**:
1. **Triple-Verifier Consensus**: Every scene is independently solved by:
   - *Oracle Solver*: Householder reflection matrix.
   - *Fermat Solver*: Virtual camera variational path minimization ($|C - R| + |R - T|$).
   - *Snell Solver*: Brent root-finding method on direct angle-of-incidence vs angle-of-reflection.
   *Tolerance threshold*: Maximum allowable residual across all three solvers is $1.0 \times 10^{-4}\text{ mm}$. Any scene with disagreement is immediately quarantined.
2. **Ambiguity & Grazing Rejection**: Rejects any scene where:
   - Reflection point $R$ is within $2.0\text{ mm}$ of mirror endpoints $A$ or $B$.
   - Optical ray passes within $2.0\text{ mm}$ of an obstacle boundary without blocking.
   - Obstacle penetration depth is $< 1.5\text{ mm}$.
   - Ray passes within $0.5^\circ$ of camera FOV cone limit.
3. **Blind Image Non-Leakage Probing**: Pixel-level chrominance auditing of the optical corridor in blind PNGs confirms zero diagnostic ray pixels exist.
4. **Anti-Leakage Filename Sanitization**: Public filenames in `test` splits are strictly anonymized (`image_pub_0001.png`), eliminating all semantic leakage.

---

## 4. Preprocessing, Cleaning & Labeling

### Was any preprocessing/cleaning done?
Yes. 52 experimental candidate configurations with dual-mirror reflections and sub-millimeter clearances were logged and quarantined in `docs/EXPERIMENT_HISTORY.md` due to non-uniqueness or numerical ambiguity. Only configurations meeting 100% triple consensus and $>2.0\text{ mm}$ margins are retained in the frozen splits.

---

## 5. Uses

### What tasks is the dataset intended for?
- Benchmarking visual-spatial and specular geometric reasoning in multimodal frontier models.
- Supervised Fine-Tuning (SFT) of vision models using step-by-step formal mathematical reasoning traces.
- Reinforcement Learning (RL) training via the Gymnasium-compatible `MirrorLabEnv`.
- Evaluating robustness against hallucination and spurious correlations via counterfactual pair consistency.

### What tasks is the dataset NOT intended for?
- General natural-image object detection (e.g., COCO-style bounding boxes on photographic scenes).
- Photorealistic rendering benchmarks (MIRRORLAB deliberately uses high-contrast vector schematics to isolate physical reasoning from texture recognition).

---

## 6. Distribution

### How is the dataset distributed?
Distributed via GitHub repository: `https://github.com/arnevmalikugdsai2030-ops/abc`
- Public splits: `data/benchmark_splits/{train,val,test}/public/`
- Ground truth keys: `data/benchmark_splits/{train,val,test}/private/`
- Formats: JSON Lines (`.jsonl`), PNG images (`.png`), and HuggingFace/OpenAI SFT format.

### License:
Apache 2.0 open-source license.

---

## 7. Maintenance

### Who is supporting and maintaining the dataset?
Maintained by the MIRRORLAB open-source team. Errata, updates, and benchmark submissions are managed via GitHub issues and pull requests.
