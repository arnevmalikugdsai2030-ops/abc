# MIRRORLAB: Mathematical Foundations of 2D Geometric Optics

> **Document:** Mathematical Specification & Rigorous Derivations  
> **Skill:** `computational-geometry`  
> **Status:** Verified from First Principles  
> **Audience:** Beginners, Engineers, and Competition Judges  

---

## 1. Introduction: Why Pure Math is Our Ground Truth Oracle

In machine-vision inspection, determining whether a camera sees a flaw in a mirror is not a matter of visual intuition or neural network prediction. It is governed strictly by **Euclidean geometry** and the **Law of Specular Reflection** discovered by Euclid and formalized by Hero of Alexandria and Pierre de Fermat.

This document derives every equation used in MIRRORLAB from first principles. We verify signs, normal directions, parametric line bounds, and edge cases with exact numerical calculations.

---

## 2. Fundamental 2D Geometric Primitives

### 2.1 The Cartesian Coordinate Frame
We operate in a 2D Euclidean plane $\mathbb{R}^2$ with standard Cartesian axes:
- **X-axis:** Horizontal, increasing to the right ($\to +X$).
- **Y-axis:** Vertical, increasing upwards ($\uparrow +Y$).
- **Origin:** $(0, 0)$ located at the reference origin of the machine-vision stage.
- **Units:** Millimeters ($\text{mm}$) for spatial coordinates, Radians internally for angles ($180^\circ = \pi \text{ rad}$).

```text
               +Y ^
                  |
                  |     • Point P = (x, y)
                  |
                  |
    --------------+--------------------> +X
                (0,0)
```

> [!IMPORTANT]
> **Cartesian vs. Image Pixel Coordinates:**  
> In mathematics, $+Y$ points **up**. In computer graphics image arrays (like Pillow), the top-left corner is $(0,0)$ and $Y$ points **down**.  
> In MIRRORLAB, **all geometric calculations are strictly performed in Cartesian coordinates**. The rendering engine maps Cartesian coordinates $(x, y)$ to screen pixels $(u, v)$ via an invertible viewport transform only at the moment of drawing.

---

### 2.2 Points and Vectors

- A **Point** $P = (x_P, y_P)$ defines a static location in space.
- A **Vector** $\vec{v} = \begin{pmatrix} v_x \\ v_y \end{pmatrix}$ represents a direction and magnitude (displacement from one point to another).

For two points $A = (x_A, y_A)$ and $B = (x_B, y_B)$, the displacement vector from $A$ to $B$ is:
$$\vec{v} = B - A = \begin{pmatrix} x_B - x_A \\ y_B - y_A \end{pmatrix}$$

#### Magnitude (Euclidean Length)
$$\|\vec{v}\| = \sqrt{v_x^2 + v_y^2}$$

#### Unit Vector (Normalization)
$$\hat{v} = \frac{\vec{v}}{\|\vec{v}\|} = \begin{pmatrix} \frac{v_x}{\|\vec{v}\|} \\ \frac{v_y}{\|\vec{v}\|} \end{pmatrix}, \quad \text{where } \|\hat{v}\| = 1$$
*(If $\|\vec{v}\| = 0$, normalization is undefined and raises an error).*

#### Dot Product (Scalar Product)
$$\vec{a} \cdot \vec{b} = a_x b_x + a_y b_y = \|\vec{a}\| \|\vec{b}\| \cos(\theta)$$
where $\theta$ is the angle between $\vec{a}$ and $\vec{b}$.
- If $\vec{a} \cdot \vec{b} > 0$: $\theta < 90^\circ$ (vectors point in a similar direction).
- If $\vec{a} \cdot \vec{b} = 0$: $\theta = 90^\circ$ (vectors are **perpendicular / orthogonal**).
- If $\vec{a} \cdot \vec{b} < 0$: $\theta > 90^\circ$ (vectors point in opposite directions).

#### 2D Cross Product (Perpendicular Dot Product / Determinant)
$$\vec{a} \times \vec{b} = a_x b_y - a_y b_x = \|\vec{a}\| \|\vec{b}\| \sin(\theta)$$
- $\vec{a} \times \vec{b} > 0 \implies \vec{b}$ is counter-clockwise from $\vec{a}$.
- $\vec{a} \times \vec{b} < 0 \implies \vec{b}$ is clockwise from $\vec{a}$.
- $\vec{a} \times \vec{b} = 0 \implies \vec{a}$ and $\vec{b}$ are collinear (parallel).

---

### 2.3 Parametric Line Segments

A finite line segment $S$ between start point $A$ and end point $B$ is represented parametrically as:
$$P(t) = A + t (B - A), \quad t \in [0, 1]$$

```text
      A                           P(t)                            B
      •─────────────────────────────•─────────────────────────────•
    t = 0.0                       t = 0.5                       t = 1.0
```
- At $t = 0.0$: $P(0) = A$.
- At $t = 1.0$: $P(1) = B$.
- For $0 < t < 1$: $P(t)$ is strictly inside the segment.
- If $t < 0$ or $t > 1$: the point lies on the infinite line containing $A$ and $B$, but **outside** the finite segment.

---

## 3. Finite Planar Mirrors and Normal Vectors

### 3.1 Mirror Representation
A finite planar mirror is defined by:
1. Start endpoint $M_1 = (x_1, y_1)$
2. End endpoint $M_2 = (x_2, y_2)$

The directed **tangent vector** along the mirror is:
$$\vec{t} = M_2 - M_1 = \begin{pmatrix} t_x \\ t_y \end{pmatrix} = \begin{pmatrix} x_2 - x_1 \\ y_2 - y_1 \end{pmatrix}$$
Mirror length: $L = \|\vec{t}\| = \sqrt{t_x^2 + t_y^2}$.  
Unit tangent: $\hat{t} = \frac{\vec{t}}{L}$.

```
             Front / Reflective Face (d > 0)
                          ^
                          |  Unit Normal n̂ = (-t_y, t_x) / L
                          |
             M_1 •═════════════════════• M_2  ───> Tangent t̂
                 ///////////////////////
                 Back / Non-Reflective Face (d < 0)
```

### 3.2 Defining the Reflective Front Face & Unit Normal $\hat{n}$
In 2D, rotating $\vec{t}$ counter-clockwise by $90^\circ$ gives the standard perpendicular vector:
$$\vec{n} = \begin{pmatrix} -t_y \\ t_x \end{pmatrix}$$
Normalizing gives the **unit normal vector**:
$$\hat{n} = \frac{1}{L} \begin{pmatrix} -t_y \\ t_x \end{pmatrix}$$

#### Verification of Orthogonality:
$$\vec{t} \cdot \vec{n} = t_x (-t_y) + t_y (t_x) = -t_x t_y + t_x t_y = 0 \quad \checkmark$$

### 3.3 Signed Distance to the Mirror Line
For any arbitrary point $Q = (x_Q, y_Q)$ in the plane, its **signed perpendicular distance** to the infinite line containing the mirror is:
$$d_{\text{signed}}(Q) = (Q - M_1) \cdot \hat{n}$$

- **Front (Reflective) Side:** $d_{\text{signed}}(Q) > 0$. The point is in front of the reflective coating.
- **Back (Non-Reflective) Side:** $d_{\text{signed}}(Q) < 0$. The point is behind the mirror's opaque backing.
- **On the Mirror Line:** $d_{\text{signed}}(Q) = 0$.

> [!CAUTION]
> **Front-Face Physical Constraint:**  
> Both the camera $C$ and the inspection target scratch $S$ **must lie on the reflective side** of the mirror:
> $$d_{\text{signed}}(C) > 0 \quad \text{and} \quad d_{\text{signed}}(S) > 0$$
> If $d_{\text{signed}}(C) \le 0$ or $d_{\text{signed}}(S) \le 0$, the ray hits the opaque back of the mirror or is trapped behind it. Physical reflection is impossible (`BACK_OF_MIRROR`).

---

## 4. The Law of Specular Reflection and the Virtual Image

### 4.1 The Law of Reflection
When a light ray strikes a planar mirror at point $P$, it obeys:
1. The incident ray, the reflected ray, and the mirror normal $\hat{n}$ all lie in the same plane.
2. The **angle of incidence** $\theta_i$ equals the **angle of reflection** $\theta_r$:
   $$\theta_i = \theta_r$$

```text
               Incident Ray                  Reflected Ray
                     \          ^ Normal n̂         /
                      \         |                 /
                       \        |                /
                        \ θ_i   |   θ_r         /
                         \      |              /
                          \     |             /
             ══════════════•════•════════════•══════════════ Mirror Line
                                P (Reflection Point)
```

---

### 4.2 The Virtual Camera (Image) Reflection Method

Instead of shooting hundreds of sample rays from the camera and guessing if one hits the tiny scratch (forward ray marching), we use the exact analytic **Virtual Camera Reflection Method**.

#### Physical Insight (Fermat's Principle of Least Time)
Light travels along the path that takes the least time. In a uniform medium, this is the shortest geometric distance.
If an observer at camera $C$ looks into a flat mirror, an object at $S$ appears to be located **behind the mirror** at an apparent position $S'$ (the virtual image).
Equivalently, from the perspective of scratch $S$, the camera appears to be located behind the mirror at the **virtual camera position $C'$**.

```
    Real Camera C (d_C > 0)                       Target Scratch S (d_S > 0)
          •                                                •
           \                                              /
            \                                            /
             \               Mirror Line                /
   ═══════════\════════════════════════════════════════•═════════════
               \                                       P
                \                                    . '
                 \                                . '
                  \                            . '
                   \                        . '
                    \                    . '
                     \                . '
                      \            . '
                       \        . '
                        \    . '
                         • C' (Virtual Camera, behind mirror)
```

The straight line connecting virtual camera $C'$ to scratch $S$ intersects the mirror line at the **exact, unique reflection point $P$** that satisfies $\theta_i = \theta_r$!

---

### 4.3 Rigorous Mathematical Derivation of Virtual Camera $C'$

Let camera $C = (x_C, y_C)$ be on the front side of the mirror ($d_{\text{signed}}(C) > 0$).
1. The projection of $C$ onto the infinite mirror line is:
   $$C_{\text{proj}} = C - d_{\text{signed}}(C) \, \hat{n}$$
2. The virtual camera $C'$ lies on the exact opposite side of the mirror line, at the exact same distance $d_{\text{signed}}(C)$ along the $-\hat{n}$ direction:
   $$C' = C_{\text{proj}} - d_{\text{signed}}(C) \, \hat{n} = C - 2 \, d_{\text{signed}}(C) \, \hat{n}$$

Substituting $d_{\text{signed}}(C) = (C - M_1) \cdot \hat{n}$:
$$\mathbf{C' = C - 2 \, \big[(C - M_1) \cdot \hat{n}\big] \, \hat{n}}$$

#### Verification:
Does $C'$ lie at signed distance $-d_{\text{signed}}(C)$?
$$(C' - M_1) \cdot \hat{n} = (C - 2 d_C \hat{n} - M_1) \cdot \hat{n} = (C - M_1) \cdot \hat{n} - 2 d_C (\hat{n} \cdot \hat{n}) = d_C - 2 d_C (1) = -d_C \quad \checkmark$$

---

### 4.4 Analytical Derivation of the Reflection Point $P$

Consider the straight line connecting virtual camera $C'$ to target scratch $S$.
Points along this line can be written parametrically as:
$$X(\lambda) = C' + \lambda (S - C'), \quad \lambda \in \mathbb{R}$$
The mirror line is defined by the condition that all points $X$ satisfy $(X - M_1) \cdot \hat{n} = 0$.

Substitute $X(\lambda)$ into the mirror equation:
$$\big(C' + \lambda(S - C') - M_1\big) \cdot \hat{n} = 0$$
$$(C' - M_1) \cdot \hat{n} + \lambda \big((S - M_1) \cdot \hat{n} - (C' - M_1) \cdot \hat{n}\big) = 0$$

Recall:
- $(C' - M_1) \cdot \hat{n} = -d_C$
- $(S - M_1) \cdot \hat{n} = +d_S$

Substitute these values:
$$-d_C + \lambda \big(d_S - (-d_C)\big) = 0$$
$$-d_C + \lambda (d_S + d_C) = 0$$
$$\mathbf{\lambda = \frac{d_C}{d_C + d_S}}$$

Because $d_C > 0$ and $d_S > 0$, the denominator $d_C + d_S > 0$ is strictly positive.
Furthermore, $0 < \lambda < 1$. The intersection point $P$ always lies strictly between $C'$ and $S$!

Therefore, the exact reflection point $P$ is:
$$\mathbf{P = C' + \left(\frac{d_C}{d_C + d_S}\right) (S - C')}$$

---

### 4.5 Proof that $\theta_i = \theta_r$

Let us prove that $P$ satisfies the law of reflection.
- **Incident ray vector:** $\vec{r}_{\text{inc}} = P - C$.
- **Reflected ray vector:** $\vec{r}_{\text{refl}} = S - P$.

From $P = C' + \lambda(S - C')$, we have:
$$S - P = (1 - \lambda)(S - C') = \left(1 - \frac{d_C}{d_C + d_S}\right)(S - C') = \frac{d_S}{d_C + d_S}(S - C')$$
Thus $\vec{r}_{\text{refl}}$ has direction $\hat{u} = \frac{S - C'}{\|S - C'\|}$.

Now look at the incident vector:
$$\vec{r}_{\text{inc}} = P - C = (C' + \lambda(S - C')) - C = (C' - C) + \lambda(S - C')$$
Since $C' - C = -2 d_C \hat{n}$, and $\lambda = \frac{d_C}{d_C + d_S}$:
The component of $\vec{r}_{\text{inc}}$ perpendicular to the mirror is:
$$\vec{r}_{\text{inc}} \cdot \hat{n} = -2 d_C + \lambda (S - C') \cdot \hat{n} = -2 d_C + \frac{d_C}{d_C + d_S}(d_S + d_C) = -2 d_C + d_C = -d_C$$
The component of $\vec{r}_{\text{refl}}$ perpendicular to the mirror is:
$$\vec{r}_{\text{refl}} \cdot \hat{n} = (S - P) \cdot \hat{n} = \frac{d_S}{d_C + d_S}(d_S + d_C) = +d_S$$

Meanwhile, the components parallel to the mirror tangent $\hat{t}$ are identical:
$$(P - C) \cdot \hat{t} = (C' - C) \cdot \hat{t} + \lambda (S - C') \cdot \hat{t} = 0 + \lambda (S - C') \cdot \hat{t}$$
$$(S - P) \cdot \hat{t} = (1 - \lambda) (S - C') \cdot \hat{t}$$

The ratio of normal component to tangential component at point $P$ gives:
$$\tan \theta_i = \left|\frac{\text{tangential}}{\text{normal}}\right|_i = \left|\frac{\text{tangential}}{\text{normal}}\right|_r = \tan \theta_r$$
$$\implies \mathbf{\theta_i = \theta_r} \quad \blacksquare$$

---

## 5. Finite Mirror Clipping (Checking Mirror Endpoints)

In the real world, mirrors are not infinite lines. A mirror is a segment from $M_1$ to $M_2$.
Does the calculated reflection point $P$ land on the physical mirror?

We project $P - M_1$ onto the mirror tangent $\vec{t} = M_2 - M_1$:
$$P = M_1 + t_{\text{mirror}} \, (M_2 - M_1)$$
Taking the dot product with $\vec{t}$:
$$(P - M_1) \cdot \vec{t} = t_{\text{mirror}} \, (\vec{t} \cdot \vec{t}) = t_{\text{mirror}} \, L^2$$
$$\mathbf{t_{\text{mirror}} = \frac{(P - M_1) \cdot \vec{t}}{\|\vec{t}\|^2} = \frac{(P - M_1) \cdot (M_2 - M_1)}{(x_2 - x_1)^2 + (y_2 - y_1)^2}}$$

### Decision Rule:
- If $\mathbf{0.0 \le t_{\text{mirror}} \le 1.0}$: The reflection ray strikes the physical mirror! $\checkmark$
- If $\mathbf{t_{\text{mirror}} < 0.0}$: The ray misses the mirror past endpoint $M_1$ (`MISSED_MIRROR`).
- If $\mathbf{t_{\text{mirror}} > 1.0}$: The ray misses the mirror past endpoint $M_2$ (`MISSED_MIRROR`).

```text
  Misses (t < 0)              Valid Reflection (0 <= t <= 1)              Misses (t > 1)
 ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄•════════════════════════════════════════════════•┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
               M_1                                              M_2
              t = 0.0                                          t = 1.0
```

---

## 6. Camera Field of View (FOV) and Range

The camera $C$ is a directional sensor with:
1. **Position:** $C = (x_C, y_C)$
2. **Heading Angle:** $\theta_{\text{cam}}$ (direction camera is pointing)
3. **Unit Heading Vector:** $\vec{h} = \begin{pmatrix} \cos \theta_{\text{cam}} \\ \sin \theta_{\text{cam}} \end{pmatrix}$
4. **Field of View Angle:** $\Phi_{\text{FOV}}$ (e.g. $60^\circ = \pi/3 \text{ rad}$)
5. **Maximum Range:** $R_{\max}$ (e.g. $1200\text{ mm}$)

```text
                      \
                       \  +FOV/2
                        \
                         \
    Camera C •────────────> Heading vector ĥ
                         /
                        /
                       /  -FOV/2
                      /
```

### 6.1 Angular FOV Test
The ray leaves camera $C$ towards the reflection point $P$ along vector:
$$\vec{v}_{\text{ray}} = P - C$$
Length: $d_{\text{inc}} = \|\vec{v}_{\text{ray}}\|$.  
Unit ray direction: $\hat{v}_{\text{ray}} = \frac{\vec{v}_{\text{ray}}}{d_{\text{inc}}}$.

The cosine of the angle $\alpha$ between the camera heading $\vec{h}$ and the ray direction $\hat{v}_{\text{ray}}$ is:
$$\cos \alpha = \hat{v}_{\text{ray}} \cdot \vec{h}$$

Since the camera has an angular half-width of $\Phi_{\text{FOV}} / 2$:
$$\mathbf{\cos \alpha \ge \cos\left(\frac{\Phi_{\text{FOV}}}{2}\right)}$$

- If $\cos \alpha \ge \cos(\Phi_{\text{FOV}} / 2)$: The reflection point $P$ is inside the camera's sight cone! $\checkmark$
- If $\cos \alpha < \cos(\Phi_{\text{FOV}} / 2)$: The mirror point $P$ is outside the camera's field of view (`OUT_OF_FOV`).

### 6.2 Distance Range Test
The total optical path length travelled by the photons is:
$$D_{\text{total}} = \|P - C\| + \|S - P\|$$
- If $D_{\text{total}} \le R_{\max}$: Within detectable optical range. $\checkmark$
- If $D_{\text{total}} > R_{\max}$: Beyond range (`OUT_OF_RANGE`).

---

## 7. Two-Leg Obstacle Occlusion

In real machine-vision inspection, opaque machine components or cavity walls can block sightlines.
The full optical path consists of two legs:
1. **Leg 1 (Incident Ray):** Line segment from Camera $C$ to Mirror Point $P$.
2. **Leg 2 (Reflected Ray):** Line segment from Mirror Point $P$ to Target Scratch $S$.
3. **Direct Sightline:** Line segment from Camera $C$ directly to Scratch $S$.

```text
           Real Camera C
                •
                │ \  Leg 1 (Incident)
                │  \
     Obstacle 1 ■   \
                    ─• P (Mirror)
                    /
                   /  Leg 2 (Reflected)
                  /
     Obstacle 2  ■
                /
               • Scratch S
```

### 7.1 Segment-Segment Intersection Algorithm
To check if a ray leg $AB$ intersects an obstacle edge $CD$:
Let:
$$P_1(t) = A + t(B - A), \quad t \in [0, 1]$$
$$P_2(u) = C + u(D - C), \quad u \in [0, 1]$$

Let $\vec{r} = B - A$ and $\vec{s} = D - C$.
At the intersection point:
$$A + t \vec{r} = C + u \vec{s}$$
Take the 2D cross product with $\vec{s}$:
$$(A - C) \times \vec{s} + t (\vec{r} \times \vec{s}) = 0$$
$$\mathbf{t = \frac{(C - A) \times \vec{s}}{\vec{r} \times \vec{s}} = \frac{(x_C - x_A) s_y - (y_C - y_A) s_x}{r_x s_y - r_y s_x}}$$

Similarly, take the 2D cross product with $\vec{r}$:
$$\mathbf{u = \frac{(C - A) \times \vec{r}}{\vec{r} \times \vec{s}} = \frac{(x_C - x_A) r_y - (y_C - y_A) r_x}{r_x s_y - r_y s_x}}$$

#### Intersection Conditions:
1. If $\vec{r} \times \vec{s} = 0$: Segments are parallel (no unique intersection).
2. Segments intersect if and only if:
   $$0.0 \le t \le 1.0 \quad \text{and} \quad 0.0 \le u \le 1.0$$

### 7.2 Numerical Guard: Endpoint Epsilon Offsets
Point $P$ lies directly on the mirror line. If an obstacle touches or aligns with the mirror, testing the exact segment $C \to P$ might trigger a false intersection at endpoint $P$ due to floating-point rounding.

We define an **epsilon safety trimming**:
$$\text{Leg 1 trimmed: } C + \epsilon(P - C) \implies P - \epsilon(P - C)$$
$$\text{Leg 2 trimmed: } P + \epsilon(S - P) \implies S - \epsilon(S - P)$$
with $\epsilon = 10^{-6}$.

### 7.3 Decision Rules:
- If Leg 1 intersects any obstacle: `BLOCKED_INCIDENT`.
- If Leg 2 intersects any obstacle: `BLOCKED_REFLECTED`.
- If Direct Sight ($C \to S$) is NOT blocked: The scene is invalid for mirror inspection (a mirror benchmark must require the mirror!).

---

## 8. Complete Step-by-Step Worked Numerical Example

Let us trace an exact scene with actual numbers and verify every step.

### 8.1 Scene Coordinates
- **Camera $C$:** $(100, 200)$
- **Camera Heading:** $\theta_{\text{cam}} = 0^\circ \implies \vec{h} = (1, 0)$
- **Camera FOV:** $\Phi_{\text{FOV}} = 90^\circ$ ($\cos(\Phi_{\text{FOV}}/2) = \cos(45^\circ) \approx 0.7071$)
- **Mirror $M$:**
  - $M_1 = (300, 400)$
  - $M_2 = (500, 400)$
  - *(A horizontal mirror 200 mm long at $y = 400$)*
- **Target Scratch $S$:** $(500, 200)$

---

### 8.2 Step 1: Compute Mirror Tangent and Normal
$$\vec{t} = M_2 - M_1 = (500 - 300, 400 - 400) = (200, 0)$$
Length $L = \sqrt{200^2 + 0^2} = 200$.  
Unit tangent $\hat{t} = (1, 0)$.

Unit normal (counter-clockwise $90^\circ$):
$$\hat{n} = \frac{1}{200} \begin{pmatrix} -0 \\ 200 \end{pmatrix} = \begin{pmatrix} 0 \\ 1 \end{pmatrix} \quad (\text{points up along } +Y)$$

Wait! Let's check signed distances:
- Camera $C = (100, 200)$:
  $$d_{\text{signed}}(C) = (C - M_1) \cdot \hat{n} = (100 - 300, 200 - 400) \cdot (0, 1) = (-200) \cdot 0 + (-200) \cdot 1 = -200$$
  $d_{\text{signed}}(C) = -200 < 0$!

**Aha! Look at that:**  
Because $M_1 \to M_2$ goes from left to right, $\hat{n} = (0, 1)$ points UP ($+Y$).  
Our camera is at $y = 200$, which is **below** the mirror ($y = 400$).  
This means the reflective coating faces UP towards $y > 400$, while the camera is at $y = 200$ looking at the **back** of the mirror!

#### Adjusting Mirror Direction:
To face the camera at $y = 200$, the mirror's reflective normal must point **DOWN** ($\hat{n} = (0, -1)$).  
To make $\hat{n}$ point down, we define the mirror from right to left:
- $M_1 = (500, 400)$
- $M_2 = (300, 400)$

Now recalculate:
$$\vec{t} = M_2 - M_1 = (300 - 500, 400 - 400) = (-200, 0)$$
Length $L = 200$. Unit tangent $\hat{t} = (-1, 0)$.  
$$\hat{n} = \frac{1}{200} \begin{pmatrix} -0 \\ -200 \end{pmatrix} = \begin{pmatrix} 0 \\ -1 \end{pmatrix} \quad (\text{points DOWN along } -Y!)$$

Now check signed distances:
- Camera $C = (100, 200)$:
  $$d_C = (C - M_1) \cdot \hat{n} = (100 - 500, 200 - 400) \cdot (0, -1) = (-400)(0) + (-200)(-1) = \mathbf{+200 > 0} \quad \checkmark$$
- Scratch $S = (500, 200)$:
  $$d_S = (S - M_1) \cdot \hat{n} = (500 - 500, 200 - 400) \cdot (0, -1) = (0)(0) + (-200)(-1) = \mathbf{+200 > 0} \quad \checkmark$$

Both are on the front reflective side!

---

### 8.3 Step 2: Calculate Virtual Camera $C'$
$$C' = C - 2 d_C \hat{n} = \begin{pmatrix} 100 \\ 200 \end{pmatrix} - 2(200) \begin{pmatrix} 0 \\ -1 \end{pmatrix} = \begin{pmatrix} 100 \\ 200 \end{pmatrix} - \begin{pmatrix} 0 \\ -400 \end{pmatrix} = \begin{pmatrix} 100 \\ 600 \end{pmatrix}$$

The virtual camera is at $(100, 600)$, which is exactly 200 mm behind the mirror line $y = 400$! $\checkmark$

---

### 8.4 Step 3: Find Reflection Point $P$
$$\lambda = \frac{d_C}{d_C + d_S} = \frac{200}{200 + 200} = \frac{200}{400} = \mathbf{0.5}$$

$$P = C' + \lambda(S - C') = \begin{pmatrix} 100 \\ 600 \end{pmatrix} + 0.5 \left(\begin{pmatrix} 500 \\ 200 \end{pmatrix} - \begin{pmatrix} 100 \\ 600 \end{pmatrix}\right)$$
$$P = \begin{pmatrix} 100 \\ 600 \end{pmatrix} + 0.5 \begin{pmatrix} 400 \\ -400 \end{pmatrix} = \begin{pmatrix} 100 + 200 \\ 600 - 200 \end{pmatrix} = \mathbf{\begin{pmatrix} 300 \\ 400 \end{pmatrix}}$$

Point $P$ is $(300, 400)$.

---

### 8.5 Step 4: Finite Mirror Boundary Test
Does $P = (300, 400)$ lie on the mirror segment from $M_1(500, 400)$ to $M_2(300, 400)$?
$$t_{\text{mirror}} = \frac{(P - M_1) \cdot \vec{t}}{\|\vec{t}\|^2} = \frac{((300, 400) - (500, 400)) \cdot (-200, 0)}{200^2} = \frac{(-200, 0) \cdot (-200, 0)}{40000} = \frac{40000}{40000} = \mathbf{1.0}$$

Since $0.0 \le t_{\text{mirror}} \le 1.0$, $P$ is right on the mirror (at endpoint $M_2$)! $\checkmark$

---

### 8.6 Step 5: Camera Field of View Test
Ray vector from Camera $C(100, 200)$ to Mirror $P(300, 400)$:
$$\vec{v}_{\text{ray}} = P - C = (300 - 100, 400 - 200) = (200, 200)$$
Magnitude: $\|\vec{v}_{\text{ray}}\| = \sqrt{200^2 + 200^2} = \sqrt{80000} \approx 282.84\text{ mm}$.  
Unit ray: $\hat{v}_{\text{ray}} = \left(\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}\right) \approx (0.7071, 0.7071)$.

Camera heading $\vec{h} = (1, 0)$.
$$\cos \alpha = \hat{v}_{\text{ray}} \cdot \vec{h} = 0.7071 \cdot 1 + 0.7071 \cdot 0 = \mathbf{0.7071}$$

FOV threshold:
$$\cos(\Phi_{\text{FOV}} / 2) = \cos(45^\circ) = \frac{1}{\sqrt{2}} \approx \mathbf{0.7071}$$

Since $\cos \alpha \ge \cos(\Phi_{\text{FOV}} / 2)$ ($0.7071 \ge 0.7071$), the mirror point is within the camera's FOV! $\checkmark$

---

### 8.7 Step 6: Verify Angle Equality ($\theta_i = \theta_r$)
- Incident vector $\vec{v}_{\text{inc}} = P - C = (200, 200)$.  
  Angle with normal $\hat{n} = (0, -1)$:
  $$\cos \theta_i = \frac{-\vec{v}_{\text{inc}} \cdot \hat{n}}{\|\vec{v}_{\text{inc}}\|} = \frac{-(200)(0) - (200)(-1)}{282.84} = \frac{200}{282.84} \approx 0.7071 \implies \theta_i = 45.0^\circ$$
- Reflected vector $\vec{v}_{\text{refl}} = S - P = (500 - 300, 200 - 400) = (200, -200)$.  
  Angle with normal $\hat{n} = (0, -1)$:
  $$\cos \theta_r = \frac{\vec{v}_{\text{refl}} \cdot \hat{n}}{\|\vec{v}_{\text{refl}}\|} = \frac{(200)(0) + (-200)(-1)}{282.84} = \frac{200}{282.84} \approx 0.7071 \implies \theta_r = 45.0^\circ$$

$$\mathbf{\theta_i = \theta_r = 45.0^\circ} \quad \text{Exact match! } \checkmark$$

---

## 9. Comprehensive Edge Cases & Boundary Handling

| Edge Case | Mathematical Condition | Physical Meaning | Expected Handler Behavior |
|---|---|---|---|
| **Back-face camera** | $d_C \le 0$ | Camera is behind mirror backing | Return `NOT_VISIBLE`, reason `BACK_OF_MIRROR` |
| **Back-face target** | $d_S \le 0$ | Scratch is behind mirror backing | Return `NOT_VISIBLE`, reason `BACK_OF_MIRROR` |
| **Collinear ray & mirror** | $d_C = 0$ or $d_S = 0$ | Grazing zero-thickness edge | Return `NOT_VISIBLE`, reason `BACK_OF_MIRROR` |
| **Missed finite mirror** | $t_{\text{mirror}} < 0$ or $t_{\text{mirror}} > 1$ | Ray reflects on plane but past physical glass | Return `NOT_VISIBLE`, reason `MISSED_MIRROR` |
| **Endpoint grazing** | $|t_{\text{mirror}}| < 0.02$ or $|1 - t_{\text{mirror}}| < 0.02$ | Ray hits within 2% of mirror corner | Reject scene in Task Factory (ambiguous for VLM) |
| **Outside FOV** | $\cos \alpha < \cos(\Phi/2)$ | Camera looking away from mirror | Return `NOT_VISIBLE`, reason `OUT_OF_FOV` |
| **Direct sight clear** | Segment $C \to S$ has no obstacle | Scratch visible directly without mirror | Reject scene in Task Factory (mirror not needed) |
| **Obstacle on Leg 1** | Segment $C \to P$ intersects obstacle | Camera cannot see mirror | Return `NOT_VISIBLE`, reason `BLOCKED_INCIDENT` |
| **Obstacle on Leg 2** | Segment $P \to S$ intersects obstacle | Mirror cannot see scratch | Return `NOT_VISIBLE`, reason `BLOCKED_REFLECTED` |

---

## 10. Summary of Implementation Rules for Milestone 1

1. **Strict Cartesian Coordinates:** Use $(x, y)$ with $y$ pointing up. Never mix pixel screen space with geometry math.
2. **Standard Normal Orientation:** Unit normal $\hat{n} = \frac{(-t_y, t_x)}{\|\vec{t}\|}$ defines the reflective face. Always check that $(C - M_1) \cdot \hat{n} > 0$ and $(S - M_1) \cdot \hat{n} > 0$.
3. **Parametric Solvers:** Use parametric formulation $P(t) = A + t(B - A)$ for all segment intersections to prevent division-by-zero on vertical lines ($x_1 = x_2$).
4. **Tolerance Epsilon:** Set $\epsilon = 10^{-7}$ for numerical comparisons (`math.isclose` or `abs(a - b) < 1e-7`).
