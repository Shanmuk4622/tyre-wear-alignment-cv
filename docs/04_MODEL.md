# 04 — Model Architecture

> Design principle: **geometry where geometry works, learning where it doesn't.** End-to-end regression from raw video to alignment angles is the obvious approach and it is the wrong one — it wastes the strong analytic structure this problem hands you for free, needs 10× the data, and is indefensible in a viva.

---

## 1. System overview

```
video pass (N frames × 3 illumination channels)
        │
        ▼
┌───────────────────────────────────────────────────────┐
│ STAGE A — PERCEPTION (per frame)                      │
│  A1  tyre + footprint segmentation   (YOLO11-seg)     │
│  A2  laser line extraction (sub-pixel Gaussian peak)  │
│  A3  groove / rib / sipe / TWI segmentation (UNet)    │
│  A4  sidewall silhouette extraction                   │
└───────────────┬───────────────────────────────────────┘
                ▼
┌───────────────────────────────────────────────────────┐
│ STAGE B — RECONSTRUCTION (per pass)                   │
│  B1  rolling-speed estimation from tread feature flow │
│  B2  unroll + stitch → metric tread map (θ × w)       │
│  B3  laser depth profile → sparse metric depth        │
│  B4  travel-direction fit from footprint centroid track│
└───────────────┬───────────────────────────────────────┘
                ▼
        ┌───────┴────────┐
        ▼                ▼
┌──────────────┐  ┌─────────────────────────────────────┐
│ STAGE C      │  │ STAGE D — GEOMETRY                  │
│ TREAD        │  │  D1 analytic estimators E1,E2,E3    │
│  C1 depth    │  │     → (τ̂₀, γ̂₀) + covariances       │
│  C2 ranking  │  │  D2 learned residual on E-features  │
│  C3 pattern  │  │     → Δτ, Δγ                        │
│  C4 damage   │  │  D3 τ̂ = τ̂₀+Δτ,  γ̂ = γ̂₀+Δγ         │
└──────┬───────┘  └────────────┬────────────────────────┘
       └──────────┬────────────┘
                  ▼
┌───────────────────────────────────────────────────────┐
│ STAGE E — FUSION                                      │
│  physics-consistency loss (train)                     │
│  agreement/disagreement diagnosis (inference)         │
│  conformal intervals                                  │
│  → REPORT                                             │
└───────────────────────────────────────────────────────┘
```

---

## 2. Stage A — Perception

### A1 · Tyre and footprint segmentation

- **YOLO11-seg (nano or small)**, fine-tuned from COCO. Two classes: `tyre`, `contact_footprint`.
- The FTIR channel makes footprint segmentation almost trivial — thresholding gets you 90% there. Use the network for robustness to dirt and partial contact, not because the task is hard.
- Output: instance masks + track IDs across the pass.

### A2 · Laser line extraction

Not a learning problem. Classical, and better than learned here:

1. Take the laser-channel frame, subtract the temporally adjacent flood frame (removes ambient).
2. For each image column, find the intensity peak.
3. Fit a Gaussian to the 5 pixels around the peak → sub-pixel centre (typical σ_sub ≈ 0.1 px).
4. Reject columns with peak SNR below threshold (rubber absorbs 650 nm; some columns will genuinely fail).
5. Back-project through the calibrated laser plane → metric 3D profile.

```python
# scripts/laser_profile.py — core idea
def subpixel_peak(col):
    i = np.argmax(col)
    if i < 1 or i >= len(col) - 1: return None
    a, b, c = np.log(col[i-1:i+2] + 1e-6)
    return i + 0.5 * (a - c) / (a - 2*b + c)   # Gaussian peak, closed form
```

### A3 · Groove / rib / TWI segmentation

- **U-Net or SegFormer-B0**, 4 classes: `rib`, `groove`, `sipe`, `TWI_bar`.
- Input: 3-channel stack `[FTIR, flood-IR, laser-depth-sparse]`. Stacking the modalities as channels is simple and works well; save cross-attention fusion for an ablation.
- TWI bars are the rare class — weight them heavily (`w ≈ 10`) or they will be ignored.

### A4 · Sidewall silhouette

Contour extraction from the tyre mask, split into left/right sidewall arcs. Fit a low-order polynomial to each. The **asymmetry between the two fits** is the camber cue for estimator E1.

---

## 3. Stage B — Reconstruction

### B1 · Rolling-speed estimation (do not skip this)

Naïve stitching uses vehicle speed. **This is wrong** — tyres slip, and the effective rolling radius changes with load and inflation. A 2% speed error compounds into visible seams and metric drift across a 2 m unrolled map.

Correct approach: estimate **angular** velocity directly from tread features.

1. Track distinctive tread features (blocks, sipe corners) between consecutive frames with LK optical flow, restricted to the near-contact band.
2. In the contact band, surface displacement between frames ≈ `R·Δφ`.
3. Robust-fit (RANSAC) a single `Δφ` per frame pair.
4. Integrate → circumferential coordinate `θ(t)`.

This is self-calibrating and immune to slip. Also gives you **effective rolling radius** as a bonus output, which correlates with inflation pressure. That's a free extra signal — mention it.

### B2 · Unrolling and stitching

For each frame, take the band `|x − x_contact| < 25 mm` (the near-zero-blur zone, see `01_CONCEPT.md §2`), rectify with the plate homography to metric millimetres, and place it at circumferential coordinate `θ(t)`.

Blend overlaps with a **feathered weighted average**, weight = `cos²` falloff from the band centre (sharpest at the exact contact point). Output:

```
unrolled map:  (circumference_mm / res) × (tread_width_mm / res)
at res = 0.1 mm/px, 195/65R15  →  19,900 × 1,950 px per full revolution
```

That is large. Store as tiled PNG or Zarr. For model input, downsample to 0.2 mm/px and crop patches.

### B3 · Sparse → dense depth

The laser gives one profile line per frame. Across the pass, those lines densify into a partial depth map. It will be sparse and striped. That is fine — it is *supervision*, not input. Train with a masked loss over valid laser pixels only.

### B4 · Travel direction

Fit a robust line (RANSAC) to the footprint centroid track across frames. This is the pass-specific travel axis. All toe angles are measured relative to it. Removes rig-mounting bias entirely (`02_RIG_BUILD.md §3.4`).

---

## 4. Stage C — Tread heads

### Backbone

`ConvNeXt-V2-Tiny` or `EfficientNetV2-S`, pretrained self-supervised (MAE) on the public tyre image sets plus your own unlabelled captures. Input: patches from the unrolled map, 4 channels `[FTIR, flood, sparse-depth, groove-mask]`.

> Why not a ViT? On this data, texture and local structure dominate and your dataset is small. ConvNeXt will be more sample-efficient and faster on the edge. Try a ViT in the ablation; don't start there.

### C1 · Depth head

Dense prediction head → per-pixel depth in mm over the unrolled map.

```
L_depth = Σ_valid  huber( D̂ − D_laser )        (masked to laser-valid pixels)
        + λ_smooth · TV(D̂)                     (grooves are smooth along θ)
        + λ_twi   · | D̂(TWI_top) − 1.6 |       (the free metric anchor)
```

That third term is `01_CONCEPT.md`'s novelty claim #5 turned into two lines of loss. It self-calibrates absolute scale on **every tyre**, forever, with no per-camera recalibration. Do not skip it.

### C2 · Ranking head

Siamese, shares the backbone. Feeds on your cheap human-labelled pairs.

```
L_rank = BCE( σ( (f(x_a) − f(x_b)) / T ), 1[d_a > d_b] )
```

At inference, map ranking scores to millimetres via **isotonic regression** fitted on the gauge-anchored subset. This gives a monotone, well-conditioned scale that degrades gracefully — far better than raw regression when the test tyre is unlike anything in training.

Add a **monotonicity loss** on your longitudinal set: for the same tyre at times `t₁ < t₂`, enforce `d̂(t₂) ≤ d̂(t₁)`. Free supervision from a physical law.

```
L_mono = Σ  relu( d̂(t₂) − d̂(t₁) + margin )
```

### C3 · Pattern head

Global-pooled features from the full unrolled map → **8 sigmoid outputs** (multi-label, not softmax — see `01_CONCEPT.md §7`). Focal loss, because `uniform` will dominate.

Also emit the **explicit physical statistics** as auxiliary regression targets — lateral gradient `g_w`, centre/shoulder ratio, cupping FFT peak amplitude and frequency, rib-edge asymmetry `A`. These are computable analytically from `D̂`, so supervising them costs nothing and it forces the representation to be physically meaningful. They then feed Stage E.

### C4 · Damage head

Two-track:
- **Supervised detector** for the classes you have examples of (cut, bulge, embedded object) — YOLO11.
- **Unsupervised anomaly detection** (PatchCore / PaDiM) on the unrolled map for everything you've never seen. Fit the memory bank on healthy tyres only.

The unsupervised track is what makes this deployable. You will never enumerate all tyre damage types.

---

## 5. Stage D — Geometry

### D1 · Three analytic estimators

| | Input | Method | Output |
|---|---|---|---|
| **E1** sidewall | left/right sidewall polynomials | asymmetry → camber via a calibrated lookup | γ̂₁, σ₁ |
| **E2** footprint | FTIR intensity field `F(x,y)` | lateral first moment → camber; principal-axis angle vs travel direction → toe | γ̂₂, τ̂₂, σ₂ |
| **E3** grooves | groove centrelines in image | circumferential grooves are coplanar with the wheel mid-plane; their vanishing point gives the plane normal | γ̂₃, τ̂₃, σ₃ |

E3 deserves emphasis. Circumferential grooves are, to good approximation, **circles coaxial with the wheel**. Their projections are conics whose common supporting-plane normal is the wheel-plane normal. With FTIR giving you clean groove masks, this is a well-posed, purely geometric estimator with no learned component at all. Solve with the conic back-projection in `01_CONCEPT.md §4`, disambiguating via the motion consistency check.

### D2 · Learned residual

```
features = [ γ̂₁,σ₁, γ̂₂,τ̂₂,σ₂, γ̂₃,τ̂₃,σ₃,
             footprint area, aspect, load proxy, inflation proxy,
             speed, tyre size, backbone embedding (256-d) ]
        → MLP (3 × 256) → (Δτ, Δγ, log σ_τ, log σ_γ)
```

Predicting **log-variance alongside the mean** (heteroscedastic / Gaussian NLL loss) gives per-sample uncertainty for free and makes the model down-weight hard passes rather than fitting noise:

```
L_geom = Σ [ (τ − τ̂)²/(2σ_τ²) + log σ_τ ] + [ same for γ ]
```

### D3 · Why the analytic prior matters

Report this ablation, it will be one of your cleanest results:

| Variant | Expected behaviour |
|---|---|
| End-to-end CNN → angles | Overfits the jig; collapses on real cars |
| Analytic only | Unbiased but noisy; fails on dirty/partial data |
| **Analytic + learned residual** | Best of both; degrades gracefully |

The middle row failing on dirty data and the top row failing on domain shift, with the hybrid surviving both — that is a compelling figure.

---

## 6. Stage E — Fusion, consistency, diagnosis

### Training: the consistency loss

Full objective (see `01_CONCEPT.md §6` for the derivation):

```
L = L_depth + L_rank + L_mono
  + α · L_pattern
  + β · L_geom
  + λ · [ ‖ g_w − h_γ(γ̂) ‖²  +  ‖ A − h_τ(τ̂) ‖² ]
```

`h_γ`, `h_τ` are **monotone** links. Start with fitted affine functions (fit their coefficients on the jig+longitudinal subset where you have both labels, then freeze). Upgrade to a monotone MLP only if the affine version underfits — and report both.

Schedule `λ`: **0 for the first 20 epochs**, then ramp to target over 10. Applying a consistency loss between two heads that are both still garbage just injects noise.

### Inference: the disagreement diagnostic

This is the output nobody else has. Compute:

```
residual  r = g_w − h_γ(γ̂)          (wear says X, geometry says Y)
```

| Geometry | Wear pattern | `r` | Interpretation |
|---|---|---|---|
| aligned | uniform | ~0 | Healthy |
| misaligned | matching gradient | ~0 | **Chronic misalignment** — has been wrong for a long time |
| aligned | strong gradient | large | **Recently realigned** — damage is historical, monitor |
| misaligned | uniform | large, opposite | **Recent misalignment** — kerb strike, pothole, recent impact |

That table is a paper section. It converts a two-number output into an actionable *narrative*, and it is only possible because you measure geometry and history simultaneously. Lead with it.

### Conformal calibration

Fit on a held-out calibration split, per output, per stratum (stratify by tyre size class — intervals should widen for underrepresented sizes).

---

## 7. Training recipe

| | |
|---|---|
| Optimiser | AdamW, lr 3e-4 (heads) / 3e-5 (backbone), wd 0.05 |
| Schedule | 5-epoch linear warmup → cosine to 1e-6 |
| Batch | 32 unrolled-map patches (512×512 @ 0.2 mm/px) |
| Precision | bf16 mixed (T4s support fp16; use fp16 + GradScaler on Kaggle) |
| Epochs | 60 synthetic pretrain → 40 jig → 25 real fine-tune |
| Augmentation | Flip lateral (**and negate camber/toe labels** — easy to get wrong), brightness/gamma, synthetic dirt overlays, random channel dropout, cutout on the unrolled map |
| Loss weights | α=1.0, β=2.0, λ=0.3 (ramped), λ_twi=5.0, λ_smooth=0.05 |
| EMA | Yes, decay 0.999 — meaningfully helps small-data regression |
| Early stop | On val depth MAE, patience 10 |

> **The lateral-flip augmentation sign flip is a classic bug.** Flipping the image horizontally must negate both camber and toe labels. Write a unit test for it. Seriously — assert that flipping twice returns the original label.

### Ablation grid (plan these now, run them at the end)

| # | Ablate | Question answered |
|---|---|---|
| 1 | λ = 0 | Does physics consistency help? |
| 2 | No synthetic pretraining | Is sim-to-real worth the pipeline? |
| 3 | No analytic prior (end-to-end) | Is the hybrid justified? |
| 4 | FTIR channel removed | Is the rig's key innovation load-bearing? |
| 5 | No laser distillation (RGB-only training) | How much does the teacher buy? |
| 6 | No TWI anchor | Does self-calibration matter? |
| 7 | Regression instead of ranking | Is ordinal learning better on small data? |
| 8 | Single-frame instead of full-pass | Is the unrolled map worth it? |

Ablation 4 is the one reviewers will ask for. Make sure you run it.

---

## 8. Baselines to compare against

Be fair and be thorough — a weak baseline section is the fastest route to rejection.

1. **Human expert.** Get a tyre technician to judge 50 tyres by eye. Report their MAE against the gauge. If you beat a human, say so; if you don't, that is still a real finding and an honest one.
2. **Depth-gauge repeatability.** Measure the same 20 tyres twice, on different days. This is your **noise floor** — no model can beat it, and reporting it shows you understand your own metrology.
3. **Off-the-shelf monocular depth** (Depth Anything V2 fine-tuned) on the same crops. Shows why the task needs a purpose-built approach.
4. **Simple CNN regression** from a single raw frame. The obvious approach; shows the value of the pipeline.
5. **Classical CV alignment**: ellipse fit only, no learning. Shows the value of the residual head.
6. Public-dataset classifiers (the Kaggle ViT baselines) on the *binary* good/bad task, to place your work in existing literature.

---

## 9. Deployment

| Target | Format | Notes |
|---|---|---|
| Rig (Jetson Orin Nano) | TensorRT fp16 | Target < 2 s per pass end-to-end |
| Mobile app | TFLite / CoreML, INT8 | Handheld close-up mode only — no FTIR, no laser, so wider intervals. **Be explicit in the UI that it's a lower-confidence mode.** |
| Cloud demo | Gradio on HF Spaces | Upload a pass, get the report card |

Export path: PyTorch → ONNX (opset 17) → TensorRT / TFLite. Validate numerics after each conversion — assert max abs difference < 1e-3 on a fixed batch. Conversion silently breaking a model is common and demoralising to discover late.

**Mobile mode caveat:** without FTIR and laser, the mobile app is a fundamentally weaker sensor. Do not let it output the same confidence numbers as the rig. Re-run conformal calibration separately for the mobile pipeline and let the intervals be honestly wider.
