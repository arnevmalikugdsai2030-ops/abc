# MIRRORLAB: Mathematical Methodology & Physical Optics Proofs
## Formal Geometry, Triply-Redundant Verification, and Statistical Metrics

**Document Version:** 1.0.0  
**Status:** Authoritative Mathematical Methodology  
**Hackathon:** Innodata Frontier AI Hackathon 2026 | Track 2: Visual Task Factory  

---

## 1. Physical Law: Specular Law of Reflection

Specular reflection occurs when a light ray reflects off a smooth boundary separating two optical media such that the angle of incidence equals the angle of reflection:

$$\theta_i = \theta_r$$

For a planar mirror segment defined by endpoints $\mathbf{A}, \mathbf{B} \in \mathbb{R}^2$ with unit normal vector $\hat{\mathbf{n}}$ pointing into the reflective half-space, an incident ray direction $\hat{\mathbf{u}}_{\text{inc}}$ reflects into direction $\hat{\mathbf{u}}_{\text{refl}}$ governed by the Householder reflection tensor:

$$\hat{\mathbf{u}}_{\text{refl}} = \hat{\mathbf{u}}_{\text{inc}} - 2 (\hat{\mathbf{u}}_{\text{inc}} \cdot \hat{\mathbf{n}}) \hat{\mathbf{n}}$$

---

## 2. Triple-Redundant Geometric Verification

To ensure ground-truth infallibility, MIRRORLAB implements three completely independent geometric verification engines sharing zero codebase or mathematical formulation:

```
+---------------------------------------------------------------------------------------+
| METHOD 1: Virtual Target Reflection (Oracle)                                         |
| 1. Reflect target T across infinite mirror line to form virtual image T'              |
| 2. Intersect segment [C, T'] with mirror segment [A, B] -> reflection point R         |
| 3. Check physical constraints: R in [A, B], C in FOV cone, C and T on front face     |
| 4. Ray-cast segments [C, R] and [R, T] against all obstacles                          |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
| METHOD 2: Virtual Camera & Fermat's Least-Time Principle (Independent Verifier)       |
| 1. Reflect camera C across mirror line to form virtual camera C'                      |
| 2. Compute optical path length function L(t) = ||C - M(t)|| + ||M(t) - T||            |
| 3. Enforce Fermat's stationary path condition: dL/dt = 0                             |
| 4. Verify Fermat residual |sin(theta_i) - sin(theta_r)| < 1e-12                       |
| 5. Analytical circle-line and polygon edge discriminant clipping                      |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
| METHOD 3: Direct Snell-Vector Collinearity Root-Finding (Ray-March Verifier)          |
| 1. Parameterize mirror: M(t) = A + t * (B - A), t in [0, 1]                           |
| 2. Compute incident vector u_inc(t) = normalize(M(t) - C)                             |
| 3. Compute reflected ray u_refl(t) via Householder reflection                         |
| 4. Compute target direction u_tgt(t) = normalize(T - M(t))                            |
| 5. Solve scalar cross-product error E(t) = u_refl(t) x u_tgt(t) = 0 via Brent's method|
| 6. Separating Axis Theorem (SAT) convex polygon occlusion verification                |
+---------------------------------------------------------------------------------------+
```

### Consensus Criteria
All three methods must agree on:
1. Discrete visibility outcome: `Method1.is_visible == Method2.is_visible == Method3.is_visible`.
2. Discrete obstruction root cause: `Method1.reason == Method2.reason == Method3.reason`.
3. Spatial reflection point disparity:
   $$\max \left( \| \mathbf{R}_1 - \mathbf{R}_2 \|, \| \mathbf{R}_2 - \mathbf{R}_3 \| \right) < 10^{-4}\text{ mm}$$
4. Dimensionless Fermat residual:
   $$|\Delta \sin \theta| = |\sin \theta_i - \sin \theta_r| < 10^{-6}$$

---

## 3. Obstacle Collision Algorithms

### 3.1 Circle Obstacle Analytical Intersection
For an optical ray segment $\mathbf{P}(s) = \mathbf{P}_1 + s (\mathbf{P}_2 - \mathbf{P}_1)$ with $s \in [0, 1]$ and circular obstacle with center $\mathbf{O}$ and radius $r$:

$$\|\mathbf{P}(s) - \mathbf{O}\|^2 = r^2$$

Expanding into quadratic form $a s^2 + b s + c = 0$:
$$\begin{aligned}
a &= \|\mathbf{P}_2 - \mathbf{P}_1\|^2 \\
b &= 2 (\mathbf{P}_2 - \mathbf{P}_1) \cdot (\mathbf{P}_1 - \mathbf{O}) \\
c &= \|\mathbf{P}_1 - \mathbf{O}\|^2 - r^2
\end{aligned}$$

The discriminant $\Delta = b^2 - 4ac$ determines intersection:
- $\Delta < 0$: No collision.
- $\Delta \ge 0$: Real roots $s_{1,2} = \frac{-b \pm \sqrt{\Delta}}{2a}$.
- Collision occurs if and only if $\exists s \in [\epsilon_{\text{trim}}, 1 - \epsilon_{\text{trim}}]$, where $\epsilon_{\text{trim}} = 10^{-4}$ prevents false self-occlusion at mirror and camera endpoints.

### 3.2 Polygon Obstacle Intersection & SAT Clipping
For rectangular and polygonal barriers, ray segment intersection uses:
1. Segment-segment intersection against all bounding edges via oriented 2D cross-products.
2. Separating Axis Theorem (SAT) projection to reject non-overlapping bounding boxes in $\mathcal{O}(1)$ time before computing exact vertex ray hits.

---

## 4. Counterfactual Pair Mathematics

To eliminate guessing bias ($50\%$ random guessing on binary tasks), Task D pairs scenes under minimal controlled geometric deltas:

### 4.1 Type 1: Mirror-Angle Flip Pairs
- A base scene $\mathcal{S}_A$ with visibility state $V_A \in \{\text{VISIBLE}, \text{NOT\_VISIBLE}\}$.
- Mirror orientation perturbed by $\Delta \theta \in [2.0^\circ, 5.0^\circ]$:
  $$\hat{\mathbf{n}}_B = \mathbf{R}(\Delta \theta) \hat{\mathbf{n}}_A$$
- Such that $V_B = \neg V_A$.
- Ground truth relationship: `FLIPPED`.

### 4.2 Type 2: Cosmetic Invariant Controls
- Identical geometric coordinates: $\mathbf{C}_A = \mathbf{C}_B, \mathbf{M}_A = \mathbf{M}_B, \mathbf{T}_A = \mathbf{T}_B, \mathcal{O}_A = \mathcal{O}_B$.
- Rendering styling altered (Monochrome CAD vs Blueprint vs High-Contrast Darkroom, line weight $1.5\text{ px} \to 3.0\text{ px}$).
- Ground truth relationship: `INVARIANT` ($V_A = V_B$).

### 4.3 Joint Pair Correctness Scoring
A model is scored correct on a counterfactual pair if and only if it correctly answers both child scenes:

$$\text{Score}(\text{Pair}_i) = \mathbb{I}(\hat{y}_{i, A} = y_{i, A}) \times \mathbb{I}(\hat{y}_{i, B} = y_{i, B})$$

Under random guessing:
$$\mathbb{E}[\text{Score}_{\text{random}}] = 0.50 \times 0.50 = 0.25\ (25\%)$$
This halves the expected guessing baseline and completely penalizes degenerate "All YES" or "All NO" heuristic strategies to $0.00$ on flip pairs.

---

## 5. Statistical Inference & Confidence Intervals

### 5.1 Wilson Score Confidence Interval
For $k$ successes out of $n$ trials, the Wilson score interval provides robust coverage for binomial proportions without normal approximation failures near $0\%$ or $100\%$:

$$w = \frac{p + \frac{z^2}{2n} \pm z \sqrt{\frac{p(1-p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$

where $p = k/n$ and $z = 1.95996$ for a $95\%$ two-tailed confidence level.

### 5.2 Paired McNemar Hypothesis Test
When comparing Model 1 and Model 2 on the identical benchmark split, differences are assessed via the McNemar contingency test on discordant pairs:

$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$$

where $b$ is the number of tasks where Model 1 is correct and Model 2 is incorrect, and $c$ is the converse. A difference is declared statistically significant only when $p < 0.05$ ($\chi^2 > 3.841$).

---
*Authored by the MIRRORLAB Scientific Methodology Core.*
