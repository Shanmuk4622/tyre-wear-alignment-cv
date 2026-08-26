# 04 — Model Architecture

> Design principle: **geometry where geometry works, learning where it doesn't.** Detailed wear recognition and alignment estimation use different evidence and must not share one black box.

---

## 0. Pipeline

```
video (RGB + photometric normals + polarised channels)
    │
 ┌──▼─────────────────────────────────────────────────────┐
 │ MODEL 0 · frame-quality gate         MobileNetV3-Small │
 │   focus · exposure · glare/wet · occlusion · coverage  │
 └──┬─────────────────────────────────────────────────────┘
    │ accepted frames only
 ┌──▼─────────────────────────────────────────────────────┐
 │ ROI  ·  fixed calibrated crop  (YOLO11n-seg fallback)  │
 └──┬─────────────────────────────────────────────────────┘
 ┌──▼─────────────────────────────────────────────────────┐
 │ MODEL 1 · SegFormer-B2   two-pass, tiled               │
 │   tyre · tread · shoulders · grooves · ribs            │
 │   sipes · TWI · damage        (class probabilities)    │
 └──┬─────────────────────────────────────────────────────┘
    ├──────────────────────────┬─────────────────────────┐
 ┌──▼───────────────────┐  ┌───▼──────────┐  ┌───────────▼──────────┐
 │ registration +       │  │ MODEL 3      │  │ MODEL 4 · alignment  │
 │ partial unrolling    │  │ PatchCore    │  │  HRNet-W18 landmarks │
 │ LK + RANSAC          │  │ unknown      │  │  + structure tensor  │
 │ (RAFT-S ablation)    │  │ anomalies    │  │  + LSD/Hough/RANSAC  │
 └──┬───────────────────┘  └───┬──────────┘  │  → analytic γ, τ     │
 ┌──▼───────────────────────┐  │             │  → residual MLP      │
 │ MODEL 2 · ConvNeXt-V2-T  │  │             └───────────┬──────────┘
 │  ordinal severity        │  │                         │
 │  multi-label pattern     │  │                         │
 │  wear heatmap            │  │                         │
 │  damage masks            │  │                         │
 │  confidence              │  │                         │
 │  (optional metric depth) │  │                         │
 └──┬───────────────────────┘  │                         │
    └──────────────┬───────────┴─────────────────────────┘
 ┌─────────────────▼──────────────────────────────────────┐
 │ temporal fusion · conformal intervals · cross-check    │
 │        PASS / MONITOR / INSPECT / UNABLE_TO_MEASURE    │
 └────────────────────────────────────────────────────────┘
```

---

## 1. Model 0 — frame-quality gate

**A blurred or occluded tyre must never become a confident safety answer.**

Deterministic measurements first, small classifier second:

| Signal | Method |
|---|---|
| Focus | Variance of Laplacian / high-frequency energy |
| Exposure | % clipped dark and bright pixels |
| Glare / wetness | Specular area; polarised − unpolarised difference |
| Coverage | Tread-mask area, **both-shoulder visibility** |
| Motion blur | Optical-flow consistency across the burst |
| Condition class | **MobileNetV3-Small** → clean / dirty / wet / glare / occluded / blurred |

**Outputs**

```
usable_probability · focus_score · exposure_score
glare_or_wet_probability · occlusion_probability
left_shoulder_visible · right_shoulder_visible · twi_visible
```

**Hard rule:** alignment is computed only when **both shoulders and sufficient groove structure are visible**. Otherwise → `UNABLE_TO_MEASURE`.

---

## 2. Model 1 — SegFormer-B2 segmentation

Hierarchical transformer encoder + lightweight MLP decoder. Multi-scale features handle an image containing both a whole tread crown and 1-px sipes, and it tolerates varying input resolution without positional-encoding interpolation.

**B0** as the speed baseline. **B2** as the research model. B4/B5 only once the data proves more capacity pays.

### Classes

```
0 background   1 tyre_visible   2 tread_crown   3 left_shoulder   4 right_shoulder
5 main_groove  6 rib_or_block   7 sipe          8 twi_bar         9 visible_damage
```

If the pilot set is small, merge 5+6+7 into `tread_structure` for experiment 1; split them once tyre and shoulder masks are reliable.

### Input channels

```
[ R, G, B, n_x, n_y, n_z, albedo, CLAHE_L ]      8 channels
```

Photometric normals as first-class input, not an afterthought — grooves are shape, and this is where shape enters the network. Inflate the pretrained stem by replicating and rescaling RGB weights for the extra channels.

### Resolution

```
pass 1 · whole tyre      768×768 or 896×896
pass 2 · tread tiles     512×512, 20–25% overlap, blend probabilities
```

**Never resize a wide tread crop to 224×224.** It destroys the grooves and sipes the project exists to measure.

### Loss

```
L_seg = L_focal + 0.7·L_dice + 0.3·L_boundary + 0.2·L_clDice
```

- **Focal** — background and large ribs otherwise dominate
- **Dice** — overlap for small classes
- **Boundary** — shoulder and groove edge accuracy, which feeds alignment geometry
- **clDice** — topology preservation for sipes and cracks. A crack broken into three fragments is a serious failure that plain Dice barely penalises

Extra class weighting on `sipe`, `twi_bar`, `visible_damage` (≈ 5–10×).

**Output class probabilities, not hard masks.** Boundary uncertainty feeds shoulder-curve fitting and the decision on whether alignment geometry is trustworthy.

---

## 3. Model 2 — ConvNeXt-V2-Tiny multi-task wear network

Operates on the segmented tread crop and, when video is available, on the partial unrolled map. Convolutional hierarchy suits local texture; MAE pretraining suits small data; practical on dual T4.

### Two-branch input

```
RGB branch:        normalised RGB tread crop      → ConvNeXt-V2-Tiny
structure branch:  CLAHE-L, Scharr magnitude,     → small 3-layer CNN
                   Gabor/black-hat, normals
fusion:            concat → 1×1 conv → SE gate
```

**Keep the raw RGB branch.** Filters delete subtle defects; the structure branch makes groove direction and edge asymmetry explicit. Neither alone is sufficient — and the comparison is Ablation 1.

### Heads

#### A · Ordinal wear severity

```
0 visually healthy   1 mild / monitor   2 significant / inspect   3 critical / replace
```

**CORAL / cumulative binary cross-entropy**, not softmax. Confusing 0 with 3 must cost more than confusing 1 with 2.

**Plus a Siamese ranking head** on the cheap pairwise labels:

```
L_rank = BCE( σ((f(x_a) − f(x_b)) / T), 1[wear_a > wear_b] )
```

Map ranking scores to a physical scale by **isotonic regression** on the gauge-anchored subset. Pairwise labels are near-noise-free and cost seconds each; absolute mm labels are expensive and noisy. On a small dataset this is where precision actually comes from.

**Plus monotonicity** on the longitudinal set — depth only decreases:

```
L_mono = Σ relu( d̂(t₂) − d̂(t₁) + margin ),   t₁ < t₂
```

#### B · Multi-label wear pattern

Nine independent sigmoids (see `03_DATA.md §3.3`). **Class-balanced binary focal loss. Never one softmax across these labels.**

Also regress the **explicit physical statistics** as auxiliary targets — lateral depth gradient, centre/shoulder ratio, rib-edge asymmetry, cupping FFT amplitude and period. They are computable analytically from the predictions, so supervising them is free, and it forces a physically meaningful representation.

#### C · Local wear heatmap

Pixel- or patch-level map of *where* the evidence is. A worn/not-worn scalar does not satisfy "recognise every detail."

#### D · Known-damage head

Masks/instances for `cut · crack · missing_chunk · embedded_object · exposed_cord · bulge`. Focal + Dice. Bulge reported only when the sidewall is visible.

#### E · Confidence head

Predict image-level log-variance. Inputs include segmentation coverage, blur, illumination, dirt, model entropy, temporal disagreement.

#### F · Optional metric depth

Active **only** on samples with gauge / laser / structured-light / RGB-D labels. Masked Huber. RGB-only samples train the ordinal/relative head and **must never be presented as millimetres**.

Plus the TWI anchor when available:

```
L_twi = λ_twi · | D̂(rib) − D̂(twi_top) − 1.6 mm |
```

### Objective

```
L_wear = λ_o·L_ordinal + λ_r·L_rank + λ_m·L_mono
       + λ_p·L_multilabel + λ_h·L_heatmap + λ_d·L_damage
       + λ_u·L_uncertainty + λ_z·L_metric_depth(valid only) + λ_twi·L_twi
```

Weights chosen on validation and **frozen before final testing**. If one head dominates, use uncertainty-based task weighting or GradNorm rather than hand-shrinking every other loss.

---

## 4. Model 3 — PatchCore anomaly detection

A supervised head recognises only annotated defect categories. Tyre damage is open-ended.

- Input: rectified tread tiles, tread mask applied
- Features: intermediate ConvNeXt or WideResNet layers
- Memory bank: **healthy tread only**, across brands and tread designs
- Output: image-level anomaly score + pixel/patch heatmap

**PatchCore is a flagging model, not a classifier.** High score = "unfamiliar local structure, inspect this region." Fuse it *with* the supervised damage head, never instead of it.

**Report false positives per healthy tyre.** That single number decides whether anyone would tolerate this system in practice.

Ablations: **EfficientAD** (ms-scale inference, relevant for the app), **AnomalyDINO** (frozen DINOv2 features, no fine-tuning — attractive when healthy data is scarce).

---

## 5. Model 4 — single-wheel alignment

### Do not regress angles directly from pixels

Camber and toe are geometric quantities relative to a calibrated vertical and travel axis. A CNN will learn camera tilt, approach position and background. Keep the direct-regression arm **only as an ablation that demonstrates this failure**.

```
learned landmarks + classical orientation evidence
        → calibrated analytic angle
        → small learned residual
        → temporal fusion + confidence interval
```

### 5.1 HRNet-W18 landmarks

High-resolution branches maintained throughout — exactly what stable landmarks need.

```
1  left shoulder, near-contact band     4  right upper visible shoulder
2  right shoulder, near-contact band    5  ≥2 centreline points on a dominant groove
3  left upper visible shoulder          6  lowest visible tread point
```

**Must output per-point covariance (heatmap spread) and a visibility flag. Never force a prediction for a hidden landmark.**

### 5.2 Classical geometry, inside the tread mask

Scharr `x`/`y` · Canny · multi-orientation Gabor · structure tensor · LSD / probabilistic Hough · RANSAC groove and shoulder fits · left/right shoulder height and curvature asymmetry.

```
J = G_σ * [[Ix², IxIy], [IxIy, Iy²]]
φ = ½ · atan2(2·J_xy, J_xx − J_yy)
```

**Reject low-coherence pixels rather than inventing a direction.**

### 5.3 Analytic angles

Fused landmarks and groove directions give a unit wheel-plane normal `n = (n_x, n_y, n_z)` in the calibrated rig frame (x travel, y lateral, z vertical):

```
individual toe   τ = atan2(n_x, n_y)
camber           γ = atan2(n_z, √(n_x² + n_y²))
```

> **Define the sign convention once, for left and right wheels, and unit-test it.** Assert that a horizontal flip negates both, and that flipping twice is the identity. Half of all alignment bugs are sign errors.

### 5.4 Residual MLP — not another backbone

```
inputs = [ analytic τ, γ · RANSAC residuals and inlier ratios
           landmark coords + covariances · groove coherence
           segmentation confidence · tyre width / load proxy
           calibration_id embedding ]

MLP 256 → 256 → 128
outputs = [ Δτ, Δγ, log σ_τ, log σ_γ ]

τ_final = τ_analytic + Δτ        γ_final = γ_analytic + Δγ
```

Heteroscedastic Gaussian NLL:

```
L_geom = Σ [ (τ − τ̂)²/(2σ_τ²) + log σ_τ ]  +  [ same for γ ]
```

The residual **corrects repeatable bias**; it never invents the angle.

---

## 6. Registration, unrolling and temporal fusion

**Rolling/rotation speed** must come from tracked tread features, not from assumed vehicle speed — slip and effective-radius change make assumed speed wrong, and a 2% error produces visible seams and metric drift.

**Unrolling**

```
U(θ,w) = Σ_t q_t·M_t(w)·I_t(π_t(θ,w))  /  Σ_t q_t·M_t(w)
```

**Report observed circumference percentage. Never inpaint unseen tread.** Unobserved ≠ healthy.

**Angle fusion** — estimate from every accepted frame outside the strongest contact-deformation region; reject frames inconsistent with calibrated motion:

```
θ̂ = Σ_(t ∈ inliers) q_t·θ_t  /  Σ_(t ∈ inliers) q_t
```

Use a robust median if residuals are non-Gaussian. Model **periodic variation across rotation separately as possible runout** — the mean becomes the estimate, the periodic residual becomes a quality flag. That separation is a genuinely nice detail.

---

## 7. Fusion and the cross-check

Wear evidence adjusts **confidence**, never the alignment value (see `01_CONCEPT.md §7`):

| Geometry | Wear | Action |
|---|---|---|
| Suspicious | Matching one-sided / feathered | Raise confidence → `INSPECT` |
| Normal | Old uneven wear | Historical or recently corrected → `MONITOR` |
| Suspicious | Uniform | Recent event → `INSPECT` |
| Disagreement, unexplained | — | Lower confidence → `INSPECT` |

### Conformal calibration

Split conformal on the dedicated **calibration split** (`03_DATA.md §4`), per output, stratified by tyre size class. Decisions use interval bounds, never point estimates.

---

## 8. Training recipe

| | |
|---|---|
| Optimiser | AdamW, lr 3e-4 heads / 3e-5 backbone, wd 0.05 |
| Schedule | 5-epoch warmup → cosine to 1e-6 |
| Batch | 16/GPU at 512×512, gradient accumulation for larger effective batch |
| Precision | **fp16 + GradScaler** (T4 has no bf16) |
| Memory format | `channels_last` — free speedup on Turing convnets |
| EMA | decay 0.999 — meaningfully helps small-data regression |
| Early stop | on val ordinal MAE, patience 10 |
| Seeds | **3 seeds for every reported ablation** |

### Backbone initialisation — a research finding that changes the default

The [DINOv3 vs ImageNet industrial-inspection study](https://arxiv.org/html/2605.23472) found frozen self-supervised features give **no clear advantage** on RGB industrial tasks, but are the **strongest initialisation once fully fine-tuned**.

**So: initialise from SSL, fine-tune the whole backbone. Do not freeze and linear-probe.** The common small-data instinct to freeze is wrong here.

Also worth doing: **self-supervised pretraining (MAE/DINO) on your own unlabelled tyre frames.** You will capture far more frames than you label; this uses them for free.

### Training order — do not train every head at once

| Phase | What |
|---|---|
| **1 · Classical baseline** | Undistortion, ROI, CLAHE, Scharr, Canny, Gabor, structure tensor, morphology, RANSAC, LK. Debug visuals + non-learning baselines |
| **2 · Segmentation** | SegFormer-B0 → B2. Freeze the taxonomy. Verify **boundary F-score** on shoulders and grooves, not just IoU |
| **3 · Wear recognition** | ConvNeXt ordinal + multi-label. Add heatmap and damage heads once the classifier is stable |
| **4 · Alignment** | HRNet landmarks on the jig. **Evaluate the analytic estimator alone first** — never skip that baseline. Then add the residual MLP |
| **5 · Video + anomalies** | Unrolling, PatchCore, RAFT-Small — only after the frame pipeline is reliable |
| **6 · Fusion** | Temporal fusion, conformal calibration, decision logic |

---

## 9. Ablations

Plan them now, run them at the end, 3 seeds each.

| # | Ablate | Question |
|---|---|---|
| 1 | Raw RGB vs RGB + structure branch | Do classical filters add anything? |
| 2 | **Flat light vs photometric stereo** | **Is the illumination upgrade load-bearing?** |
| 3 | SegFormer-B0 vs B2 | Is the capacity justified? |
| 4 | Whole image vs segmented tread crop | Does segmentation help downstream? |
| 5 | Single frame vs registered video fusion | Is video worth the complexity? |
| 6 | Classical vs direct CNN vs hybrid alignment | Is the hybrid justified? |
| 7 | Lucas–Kanade vs RAFT-Small | Is dense flow needed? |
| 8 | Supervised damage vs + PatchCore | Does anomaly detection add coverage? |
| 9 | RGB-only vs metric-sensor-supervised depth | How much does the teacher buy? |
| 10 | With / without boundary loss | Does edge accuracy matter? |
| 11 | With / without clDice | Does connectivity matter for sipes and cracks? |
| 12 | With / without temporal quality rejection | Does the gate help? |
| 13 | Ordinal vs plain regression | Better on small data? |
| 14 | With / without TWI anchor | Does self-calibration matter? |
| 15 | ImageNet vs DINOv2 init, frozen vs fine-tuned | Confirm the transfer-learning finding |

**Ablation 2 is the one to prioritise.** It is cheap, likely to show a large effect, and it is the project's most distinctive engineering decision.

A component stays in the final system only if it improves the frozen validation result **or** makes failure detection more reliable.

---

## 10. Deployment

| Target | Format | Note |
|---|---|---|
| Laptop demo | PyTorch / ONNX | Sufficient for the capstone |
| Jetson Orin Nano | TensorRT fp16 | Target < 2 s per inspection |
| Mobile app | TFLite / CoreML INT8 | **No photometric stereo, no polarisation → weaker sensor. Recalibrate conformal separately; let the intervals be honestly wider** |

Export path PyTorch → ONNX (opset 17) → TensorRT/TFLite. **Validate numerics after every conversion** — assert max abs difference < 1e-3 on a fixed batch. Silent conversion breakage is common and demoralising to find late.
