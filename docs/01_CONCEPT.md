# 01 — Problem Formulation and Observability

> What a low-front camera can and cannot see, stated precisely, before any model is chosen.

---

## 1. The estimation problem

Given a calibrated short video `V = {I₁ … I_T}` of one tyre, camera intrinsics `K`, and rig-to-world calibration `C`:

```
f(V, K, C) → { M_tread, W_map, P_pattern, D_damage, γ, τ, U }
```

The projection model throughout is

```
s · p = K [R | t] P
```

where `P` is a 3-D point in the rig/world frame, `p` its image coordinate, `s` projective scale. The world frame has **x = vehicle travel, y = lateral, z = vertical**. This frame comes from calibration, never from the image border.

---

## 2. Observability analysis — the core of the design

Not everything the project would like to know is recoverable from this viewpoint. Being explicit about this is what separates a research system from an over-claim.

| Quantity | Observable from a low front view? | Evidence | Confidence |
|---|---|---|---|
| Tread crown, both shoulders | **Yes**, directly | Segmentation | High |
| Main grooves, ribs, blocks | **Yes**, directly | Segmentation + orientation fields | High |
| Sipes, fine cracks | **Yes**, with directional light | Photometric stereo + thin-structure segmentation | Medium-high |
| TWI bars | **Yes**, when a groove cross-section is visible | Detection + metric anchor | Medium |
| Relative wear across tread width | **Yes** | Lateral depth/shading profile | High |
| Wear around the circumference | **Partially** — only what rotates into view | Unrolled map + coverage % | Medium |
| Absolute tread depth (mm) | **Not from RGB alone** | Needs structured light / stereo / RGB-D | Low without metric sensor |
| **Camber γ** | **Yes** — produces visible lateral tilt and shoulder asymmetry | Shoulder contours + landmarks | Medium |
| **Individual toe τ** | **Weakly** — small yaw, confusable with camera yaw and approach angle | Groove orientation + landmarks | Low-medium → *screening only* |
| Hidden sidewall, bulges | **No** | — | Out of scope unless sidewall enters view |
| Total axle toe, thrust angle, caster | **No** — requires both wheels / full vehicle | — | **Explicitly out of scope** |
| Inflation pressure | **Indirectly**, weak proxy | Deformation, contact width | Low — report as a hint only |

**The two rules that follow:**

1. **Camber gets a continuous estimate. Toe gets a screening estimate with an interval.** Do not present them with equal confidence.
2. **Anything requiring the second wheel is out of scope and stays out.** Total toe, thrust angle and caster are four-wheel quantities. Claiming them from one tyre is indefensible and an examiner will find it immediately.

---

## 3. Why wear and alignment need different machinery

| | Wear | Alignment |
|---|---|---|
| Nature | Appearance / texture / local shape | Global geometry |
| Best evidence | Shading, normals, texture statistics | Calibrated landmark positions and orientations |
| Right tool | Deep multi-task CNN | Analytic geometry + learned residual |
| Failure mode if you get it wrong | Misses subtle wear | **Learns camera tilt and background shortcuts** |
| Ground truth | Digital gauge, human annotation | Jig or alignment rack |

A single network mapping pixels to `(worn?, γ, τ)` will do well on validation and collapse on a new camera mount, because it has learned the mount. The architecture keeps the two pathways separate and fuses them only at the decision stage.

---

## 4. The confound problem

The recurring difficulty: **many nuisance variables produce the same image change as the signal.**

| Nuisance | Mimics | Mitigation |
|---|---|---|
| Auto-exposure change | Tread darkening / wear | Lock exposure and white balance; record settings |
| Non-uniform illumination | Shoulder wear (one side darker) | Flat-field correction per session |
| Wet / glossy rubber | Smooth worn surface | Cross-polarisation; wetness flag from the quality gate |
| Dirt and brake dust | Cracks, patch wear | Contamination tag; dirt augmentation; quality gate |
| Camera yaw | **Toe** | Calibration + per-session verification; report toe as screening |
| Vehicle approaching at an angle | **Toe** | Estimate travel direction from motion, not from mounting |
| Load and inflation | Camber (contact width change) | Record pressure and load; treat as covariates |
| Steering angle | Toe | Straight-ahead protocol; reject frames with steering input |
| Tyre brand / tread family | Everything | **Unseen-brand test split.** Domain gap is measured, not assumed |

That last row is not hypothetical. [Vivekanandan & Rajeswari (2026)](https://doi.org/10.1016/j.measurement.2026.121509) measured unseen-brand accuracy at **88.2%**, rising to **92.4%** only after explicit domain adaptation. Brand shift is real, quantified, and must be in your test design from day one.

---

## 5. What "recognise every single detail" means operationally

The phrase needs a testable definition or it cannot be evaluated. Ours:

| Structure | Typical scale | Required capability |
|---|---|---|
| Tread crown | 150–250 mm | Segment with IoU > 0.90 |
| Shoulder boundary | ~1 mm edge precision | Boundary F-score > 0.70; drives alignment |
| Main groove | 8–15 mm wide, 1.6–9 mm deep | Segment + measure relative depth |
| Rib / block edge | ~1–2 mm | Detect rounding asymmetry (feathering cue) |
| Sipe | 0.3–1 mm wide | Detect presence and **connectivity** (clDice) |
| TWI bar | ~5 mm wide, 1.6 mm high | Detect and use as metric anchor |
| Crack | 0.2–2 mm wide | Detect with connectivity preserved |

**Resolution requirement.** To resolve a 0.3 mm sipe at ≥3 px you need **≤0.1 mm/px**. Across a 250 mm tread that is 2,500 px — more than a 1080p sensor delivers across the full width.

Two consequences, and they are design-defining:

1. **Use a higher-resolution sensor** (≥ 8 MP) *or* accept that sipes are resolved only in a **cropped centre region**.
2. **Tile the fine-detail pass** (512×512 tiles, 20–25% overlap) rather than downsampling the whole tread.

State the achieved mm/px in every result table. A wear claim without a stated spatial resolution is unfalsifiable.

---

## 6. Metric scale — the TWI anchor

RGB gives you appearance; it does not give you millimetres. But every road-legal tyre carries a moulded ruler:

**Tread Wear Indicator bars sit exactly 1.6 mm above the groove floor**, marked on the sidewall by a triangle, "TWI", or a brand symbol.

If you can segment the TWI bar and the adjacent rib surface, you have an **absolute in-image scale reference** immune to camera-distance error, lens change and tyre-size variation. [Huber et al. (2022)](https://doi.org/10.36001/phmconf.2022.v14i1.3242) use exactly this in TireEye and report **0.57 mm** accuracy on a full vehicle.

```
remaining_depth ≈ 1.6 mm + height(rib_surface − TWI_top)
L_twi = | D̂(rib) − D̂(twi_top) − 1.6 |
```

**Limitation to state honestly:** the TWI must be in view and resolvable. It appears at only a few points around the circumference. So it anchors *scale*, it does not measure *every* location. Report the fraction of passes where a TWI anchor was available.

---

## 7. The wear ↔ alignment cross-check

Both quantities are estimated independently. Comparing them is informative in a way neither is alone.

| Geometry says | Wear pattern says | Interpretation |
|---|---|---|
| Aligned | Uniform | Healthy |
| Misaligned | Matching one-sided / feathered wear | **Chronic** — long-standing misalignment. High confidence |
| Aligned | Strong one-sided wear | **Recently corrected** — damage is historical. Monitor |
| Misaligned | Uniform wear | **Recent** — kerb strike or pothole. Inspect |
| Either | Disagreement without explanation | Lower confidence; `INSPECT` |

**Important discipline:** wear evidence is a **cross-check on confidence**, never the alignment ground truth. Wear is a decade-long integral; alignment is a present-tense measurement. Using one to label the other would be circular.

The physical basis is real: a wheel with static toe angle τ driving straight is permanently operating at **slip angle α = τ**, and brush-model tyre-wear theory gives wear rate as a steep function of slip angle. So the correlation is mechanistic, not coincidental — which is why disagreement is *diagnostic* rather than noise.

This cross-check is Research Contribution 5 and, after the literature audit, the most distinctive element of the project.

---

## 8. Decision output

Every inspection returns exactly one verdict:

| Verdict | Condition |
|---|---|
| `PASS` | Wear below threshold with confidence; geometry within screening tolerance |
| `MONITOR` | Borderline wear, or geometry near tolerance, or partial circumferential coverage |
| `INSPECT` | Wear above threshold, damage detected, anomaly flagged, or geometry out of tolerance |
| `UNABLE_TO_MEASURE` | Quality gate failed, both shoulders not visible, insufficient groove structure, or interval too wide |

Decisions use **interval bounds**, never point estimates. `UNABLE_TO_MEASURE` is a success state, not a failure — a screening instrument that refuses when it cannot see is more trustworthy than one that always answers.

---

## 9. Design decisions log

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Viewpoint | Low front, single tyre, fixed rig | Ground-embedded, wheel-arch | Given by the project setup; rigid mount enables calibration |
| Illumination | **Multi-directional photometric stereo + cross-polarisation** | Single flood / ring light | Rubber structure is shape, not colour; ring light cannot separate stain from cavity |
| Input | Short video | Single still | Circumferential coverage, repeated estimates, quality *selection* |
| Segmentation | SegFormer-B2, tiled | Whole-image downsampled CNN | Downsampling destroys sipes and grooves |
| Thin-structure loss | focal + dice + boundary + **clDice** | dice only | Connectivity of cracks and sipes matters more than mean IoU |
| Wear objective | Ordinal (CORAL) + multi-label | Binary worn/not-worn | Severity is ordered; patterns co-occur |
| Depth | Relative from RGB; metric only with a metric sensor | RGB → mm regression | Not defensible; monocular depth documented weak at close range |
| Alignment | Analytic geometry + learned residual | End-to-end CNN regression | End-to-end learns mount and background shortcuts |
| Toe | Screening estimate with interval | Continuous measurement | Weakly observable from one wheel |
| Backbone use | Fine-tune fully from SSL init | Frozen features + linear probe | DINOv3-vs-ImageNet study: frozen gives no advantage, fine-tuned is best |
| Uncertainty | Heteroscedastic + split conformal | Softmax confidence | Distribution-free coverage guarantee |
| Annotation | SAM2 propagation + bootstrapping | Fully manual | ~8× throughput; manual labelling would consume the project |
