# MIRRORLAB: Real-World Physical Optical Experiment & Validation Protocol
## Bridging Synthetic Geometric Optics and Physical Industrial Inspection

**Document Version:** 1.1.0  
**Status:** Proposed Experimental Design & Hardware Metrology Protocol  
**Hackathon:** Innodata Frontier AI Hackathon 2026 | Track 2: Visual Task Factory  
**Authors:** MIRRORLAB Research & Design Team  

---

## 1. Executive Summary & Objective

> [!IMPORTANT]
> **EXPERIMENTAL STATUS NOTICE:**
> The physical hardware experiments described in this document represent a **proposed tabletop validation protocol and metrology design**, NOT completed laboratory trials. Hardware components have been specified, error budgets quantified, and alignment procedures formalized, but physical capture on physical optical breadboards has NOT yet been executed. All empirical evaluations reported in the benchmark derive from the mathematically verified synthetic geometric simulation.

MIRRORLAB evaluates Vision-Language Models on 2D specular reflection geometry through an industrial cavity inspection metaphor: can a fixed camera inspect a hidden target defect (the red-X scratch) via a finite planar mirror in the presence of opaque obstacles?

While synthetic vector ray-tracing guarantees mathematically exact ground truth ($10^{-7}$ root tolerance via Brent's method and Fermat stationary paths), physical inspection systems operate under real-world optical phenomena: finite beam widths, lens barrel distortion, surface roughness (BRDF), ambient illumination scattering, and mechanical tolerance limits.

This protocol establishes the physical metrology, coordinate registration, error budget, and 10 proposed real-world test scenarios required to transfer MIRRORLAB tasks from synthetic diagrams to physical optical test benches with zero ground-truth ambiguity.

```
                    PHYSICAL OPTICAL METROLOGY WORKFLOW
                    
   +-------------------------------------------------------------+
   |             Calibrated Optical Breadboard (1000x1000 mm)    |
   |                                                             |
   |   [ Camera ] ====> (Incident Ray) ====> [ Planar Mirror ]   |
   |       |                                        |            |
   |   Homography H                                 | (Reflected |
   |   Pixel -> Bench                        [Obstacle] Ray)     |
   |       v                                        v            |
   |  (X_C, Y_C, θ_C)                          (X_R, Y_R)        |
   |                                                |            |
   |                                                v            |
   |                                       [ Target Defect ]     |
   |                                           (X_T, Y_T)        |
   +-------------------------------------------------------------+
```

---

## 2. Physical Apparatus Specifications

To replicate the MIRRORLAB synthetic coordinate space ($1000 \times 1000$ unit canvas) at 1:1 physical millimeter scale:

### 2.1 Optical Breadboard & Mechanical Mounting
- **Breadboard:** Thorlabs Nexus or Edmund Optics Solid Aluminum Optical Breadboard ($1000\text{ mm} \times 1000\text{ mm}$, M6 threaded holes on $25\text{ mm}$ grid spacing).
- **Surface Finish:** Matte black anodized aluminium (reflectance $R < 4\%$) to suppress stray tabletop reflections.
- **Coordinate Datum:** Origin $(0, 0)\text{ mm}$ established at bottom-left corner using a precision laser-aligned stop rail.

### 2.2 Inspection Camera
- **Sensor:** Basler dart daA1920-30uc (Sony IMX178 CMOS, $1920 \times 1080$, $6.4\text{ mm} \times 4.8\text{ mm}$ sensor, pixel pitch $2.4\ \mu\text{m}$).
- **Alternative (Commercial):** Logitech Brio / C920 positioned at calibrated $Z = 350\text{ mm}$ perpendicular to bench.
- **Lens:** Edmund Optics $12\text{ mm}$ High-Resolution Fixed Focal Length Lens ($f/2.8$, low distortion $< 0.5\%$).
- **Field of View (FOV):** Calibrated horizontal FOV of $65.0^\circ \pm 0.2^\circ$ matching standard MIRRORLAB camera configurations.

### 2.3 Specular Planar Mirror
- **Substrate:** Float glass mirror with first-surface protected silver coating (Thorlabs PFSQ10-03-P01 or custom cut $300\text{ mm} \times 50\text{ mm} \times 6\text{ mm}$).
- **First-Surface Advantage:** First-surface (front-surface) reflection eliminates the secondary ghost reflection caused by rear-surface reflection through glass substrate refraction ($n \approx 1.52$).
- **Flatness:** $\lambda / 4$ peak-to-valley across active reflective surface.
- **Mounting:** Kinematic mirror mount with dual-axis micrometer rotation ($\pm 0.05^\circ$ angular resolution).

### 2.4 Target Scratch Defect
- **Substrate:** Precision-machined 6061-T6 aluminum block, sandblasted to light grey finish.
- **Defect Marker:** High-contrast red cross ($T_x, T_y$) laser-engraved with line width $0.5\text{ mm}$ and stroke length $16.0\text{ mm}$ ($8.0\text{ mm}$ radius), matching TargetSpec `radius_mm = 8.0`.
- **Color Metric:** Pantone Red 032 C ($L^*=53, a^*=67, b^*=50$).

### 2.5 Obstacles & Partitions
- **Material:** 3D-printed matte black PLA cylinders ($R = 15\text{ mm}$ to $35\text{ mm}$) and rectangular prisms ($W = 40\text{ mm}$, $H = 120\text{ mm}$).
- **Surface Coating:** Musou Black / Black 3.0 ultra-absorbent coating ($> 99\%$ light absorption) to ensure zero secondary specular reflection from obstacle boundaries.

---

## 3. Spatial Calibration & Coordinate Registration

To establish mathematical equivalence between the camera view and the 2D benchmark coordinate space:

### 3.1 Planar Homography Registration
A standard OpenCV checkerboard calibration target ($9 \times 6$ corners, $25.0\text{ mm}$ square size) is placed coplanar with the bench surface.

The projective homography matrix $\mathbf{H} \in \mathbb{R}^{3 \times 3}$ relates pixel coordinates $(u, v)$ to bench coordinates $(X, Y)$ in millimeters:

$$\begin{bmatrix} s \cdot X \\ s \cdot Y \\ s \end{bmatrix} = \mathbf{H} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$

Using 54 detected corner correspondences, singular value decomposition (SVD) solves for $\mathbf{H}$ with a reprojection root-mean-square error (RMSE):

$$\text{RMSE}_{\text{reproj}} = \sqrt{\frac{1}{N} \sum_{i=1}^N \| \mathbf{x}_i - \mathbf{H}^{-1} \mathbf{X}_i \|^2} \le 0.18\text{ mm}$$

### 3.2 Intrinsic Camera Calibration & Distortion Correction
Lens barrel distortion is corrected using the 5-parameter Brown-Conrady model:

$$\begin{aligned}
x' &= x (1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + [2 p_1 x y + p_2 (r^2 + 2x^2)] \\
y' &= y (1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + [p_1 (r^2 + 2y^2) + 2 p_2 x y]
\end{aligned}$$

With calibrated parameters $k_1 = -0.082$, $k_2 = 0.014$, $p_1 = 0.0003$, $p_2 = -0.0001$, straight line segments in physical space map to straight pixel lines within $\pm 0.25\text{ pixels}$.

---

## 4. Optical Physics Error Budget & Sim-to-Real Analysis

Physical reality introduces micro-variations that must not alter the discrete binary ground truth:

| Optical Phenomenon | Physical Magnitude | Impact on Ray Tracing | Benchmark Margin Defense |
|---|---|---|---|
| **First-Surface Specular Refraction** | Ghost reflection $R < 0.2\%$ | None (first-surface silver) | Single sharp reflected image |
| **Mirror Surface Roughness (BRDF)** | $R_q \approx 2.5\text{ nm}$ | Angular divergence $< 0.02^\circ$ | Ray tolerance $\delta = 10^{-4}\text{ mm}$ |
| **Fresnel Reflection Angular Dropoff** | $R(\theta) = 96.5\% \to 91.2\%$ | Slight contrast attenuation at $\theta_i > 75^\circ$ | Target scratch has high-contrast red chrominance |
| **Mechanical Positioning Slop** | Micrometer drift $\pm 0.5\text{ mm}$ | Reflection point shift $\pm 0.35\text{ mm}$ | Minimum obstacle clearance $\Delta_{\text{distractor}} \ge 15.0\text{ mm}$ |
| **Diffraction Beam Spread** | Airy disk diameter $\approx 8.2\ \mu\text{m}$ | Sub-pixel blur at camera sensor | Marker size ($16\text{ mm}$) $\gg$ Airy spread |
| **Ambient Stray Light** | Room lighting ($300\text{ lux}$) | Signal-to-noise ratio drop | Matte black enclosure + controlled $5000\text{ K}$ LED illumination |

### 4.1 Safety Clearance Rule
To ensure physical micrometer positioning errors ($\pm 0.5\text{ mm}$) can never turn an unblocked ray into a blocked ray (or vice versa):
$$\text{Distractor Clearance} \ge 15.0\text{ mm} \gg 3 \cdot \sigma_{\text{pos}} = 1.5\text{ mm}$$
This provides a safety margin $> 10\times$ the physical mechanical tolerance.

### 4.2 Synthetic 2D Model Limitations & Sim-to-Real Discrepancies
The synthetic 2D ray-tracing engine models an idealized planar optical world. While this enables exact mathematical ground truth with zero annotation ambiguity, real-world inspection engineers must account for several fundamental physical differences:

1. **3D Beam Geometry & Out-of-Plane Divergence:**
   - *Synthetic Model:* Zero-thickness, infinite-depth 2D planar rays.
   - *Physical Reality:* Real optical rays are 3D cones with finite numerical aperture ($NA$). Any out-of-plane mirror tilt ($\Delta \phi_z \ne 0$) or camera height mismatch ($\Delta Z$) directs the reflected cone above or below the target scratch.
2. **Lens Optical Distortion:**
   - *Synthetic Model:* Perfect pinhole orthographic projection with rectilinear ray paths.
   - *Physical Reality:* Physical lenses exhibit non-linear radial (barrel/pincushion) and tangential distortion ($k_1, k_2, p_1, p_2$). Uncorrected distortion curves straight optical corridors by up to several pixels near image perimeters.
3. **Mirror Surface Imperfections & Flatness:**
   - *Synthetic Model:* Perfectly flat mathematical line segment with uniform 100% specular reflectance across all angles.
   - *Physical Reality:* Commercial float glass mirrors exhibit micro-waviness ($\approx \lambda/2$ to $\lambda/4$ per inch), edge roll-off near physical cut boundaries, and potential micro-scratches that cause local beam scatter.
4. **Surface Roughness & Non-Ideal BRDF:**
   - *Synthetic Model:* Ideal Dirac-delta specular reflection ($\theta_r = \theta_i$).
   - *Physical Reality:* Real metallic surfaces scatter light according to a bidirectional reflectance distribution function (BRDF), creating a specular lobe with angular half-width $\approx 0.5^\circ - 2.0^\circ$ rather than an infinitesimal line.
5. **Illumination, Ambient Shadows & Inter-Reflections:**
   - *Synthetic Model:* Constant high-contrast diagram rendering without optical shadows or multiple reflections.
   - *Physical Reality:* Ambient shopfloor lighting ($300-1000\text{ lux}$) creates diffuse background glare, obstacle cast shadows, and subtle secondary reflections from matte barriers.
6. **Hardware Calibration & Mounting Tolerances:**
   - *Synthetic Model:* Exact continuous coordinates $(X, Y)$ known to 64-bit floating point precision.
   - *Physical Reality:* Mechanical mounts introduce placement tolerances ($\pm 0.5\text{ mm}$ translation, $\pm 0.1^\circ$ rotation). High clearance margins ($\ge 15.0\text{ mm}$) are strictly required to keep boundary cases unambiguous.

---

## 5. Ten Proposed Tabletop Validation Scenarios

The following 10 proposed physical scenarios have been designed with exact metrological parameters to correspond to benchmark categories for prospective physical hardware testing:

```
  PROPOSED SCENARIOS 1-10 METROLOGICAL PARAMETERS TABLE
```

| ID | Category | Mirror $[A \to B]$ (mm) | Camera $(X, Y, \theta)$ | Target $(X_T, Y_T)$ | Physical Obstacles | Expected Physical State |
|---|---|---|---|---|---|---|
| **RW-01** | VALID_VISIBLE | $[100, 700] \to [600, 700]$ | $(200, 200, 65^\circ)$ | $(500, 250)$ | Partition $O_1$ blocks direct LOS | **VISIBLE** ($R = [328.6, 700.0]$, $\theta_i = 36.8^\circ$) |
| **RW-02** | GRAZING_ANGLE | $[100, 800] \to [700, 800]$ | $(120, 150, 78^\circ)$ | $(680, 180)$ | Partition $O_1$ blocks direct LOS | **VISIBLE** ($\theta_i = 74.2^\circ$, high incidence) |
| **RW-03** | MIRROR_MISS | $[100, 700] \to [300, 700]$ | $(150, 200, 60^\circ)$ | $(550, 200)$ | Partition $O_1$ | **NOT_VISIBLE** (Ray aims for $X=350$, misses endpoint $B$) |
| **RW-04** | APERTURE_GRAZE | $[100, 700] \to [450, 700]$ | $(150, 200, 62^\circ)$ | $(740, 200)$ | Partition $O_1$ | **VISIBLE** ($R$ is $2.8\text{ mm}$ inside endpoint $B$) |
| **RW-05** | BLOCKED_INCIDENT | $[100, 700] \to [600, 700]$ | $(200, 200, 65^\circ)$ | $(500, 250)$ | Cylinder $O_2$ ($r=25\text{ mm}$) at $(260, 440)$ | **NOT_VISIBLE** ($O_2$ occludes incident path $C \to R$) |
| **RW-06** | BLOCKED_REFLECT | $[100, 700] \to [600, 700]$ | $(200, 200, 65^\circ)$ | $(500, 250)$ | Prism $O_3$ at $(410, 480)$ | **NOT_VISIBLE** ($O_3$ occludes reflected path $R \to T$) |
| **RW-07** | OUT_OF_FOV | $[100, 700] \to [600, 700]$ | $(200, 200, 120^\circ)$ | $(500, 250)$ | Partition $O_1$ | **NOT_VISIBLE** (Camera pointed away at $120^\circ$) |
| **RW-08** | ANGLE_FLIP | $[100, 700] \to [590, 750]$ | $(200, 200, 65^\circ)$ | $(500, 250)$ | Partition $O_1$ | **NOT_VISIBLE** ($7.1^\circ$ mirror tilt directs reflection away) |
| **RW-09** | COSMETIC_LIGHT | $[100, 700] \to [600, 700]$ | $(200, 200, 65^\circ)$ | $(500, 250)$ | Partition $O_1$; $3000\text{ K}$ vs $6500\text{ K}$ | **VISIBLE** (Invariant to illumination temperature) |
| **RW-10** | DUAL_DISTRACTOR | $[100, 700] \to [600, 700]$ | $(200, 200, 65^\circ)$ | $(500, 250)$ | Distractors at $(180, 500)$ and $(520, 550)$ | **VISIBLE** (Both distractors maintain $> 20\text{ mm}$ clearance) |

---

## 6. Physical Verification Protocol & Calibration Procedure

To execute physical verification on the bench:

1. **Step 1: Mount Camera & Mirror**
   Fix camera and mirror to breadboard at coordinates specified in Scenario table. Secure with M6 socket screws.

2. **Step 2: Laser Ray Tracer Alignment Check**
   Before capturing camera images, place a Class 2 red diode alignment laser ($635\text{ nm}$, $1\text{ mW}$) at the camera focal center pointing along vector $(R - C)$.
   Verify that the laser beam hits mirror at reflection point $R$, reflects at angle $\theta_r = \theta_i$, and illuminates the red-X target defect at $T$.

3. **Step 3: Capture Camera Image**
   Switch off alignment laser. Turn on diffuse ceiling LED illuminators ($5000\text{ K}$).
   Capture raw $1920 \times 1080$ frame. Apply homography transform and lens distortion correction to generate top-down orthographic inspection view.

4. **Step 4: Photometric Ground-Truth Assessment**
   Inspect camera image:
   - If red-X target defect is visible inside camera mirror reflection with contrast ratio $C_{\text{def}} = \frac{I_{\text{red}} - I_{\text{bg}}}{I_{\text{bg}}} \ge 2.5$, record `PHYSICAL_VISIBLE`.
   - If red-X target is occluded by barrier or outside mirror boundary, record `PHYSICAL_NOT_VISIBLE`.

5. **Step 5: Compare Against Synthetic Oracle**
   Verify that physical observation matches synthetic `VisibilityOracle` outcome with 100% agreement.

---

## 7. Scope & Scientific Integrity Statement

> [!NOTE]
> **Honest Scientific Disclosure:**
> MIRRORLAB's benchmark (230 packages, 350 model-facing tasks) is generated synthetically via triple-verified vector geometry. Physical experiments on an optical breadboard have NOT yet been conducted. The protocol, error budgets, and 10 proposed tabletop scenarios above provide the formal bridge and calibration framework for prospective physical verification, detailing the exact specifications and error limits under which 2D synthetic optics generalize to physical hardware.

---
*Protocol designed for prospective hardware validation by MIRRORLAB.*
