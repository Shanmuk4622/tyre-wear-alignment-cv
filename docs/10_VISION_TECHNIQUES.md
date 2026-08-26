# 10 — Vision Technique Catalogue

> Every computer-vision technique considered for this project, with a verdict and a reason.
> The organising question throughout: **"recognise the wheel in every single detail."** A technique earns a place only if it makes fine tread structure — grooves, sipes, rib edges, TWI bars, cracks — more measurable than it was without it.

**Verdict legend**

| | Meaning |
|---|---|
| ★ **Core** | In the main pipeline. Build it. |
| ◆ **Ablation** | Build it, measure it, keep only if it wins. |
| ○ **Optional** | Worth having if time allows. |
| ✗ **Rejected** | Considered and dropped, with reason. |

---

## 1. The central problem: rubber is a terrible imaging subject

Before any technique, understand the physics you are fighting:

| Property | Value | What it breaks |
|---|---|---|
| Albedo (carbon-black tread) | ~4–8% | Low SNR; long exposures; noise-limited |
| Colour information | Essentially none | Colour-based segmentation is useless |
| Surface | Glossy when wet/new, matte when dry/worn | Specular highlights saturate; wet tyres look like different tyres |
| Structure | Defined by **shape**, not reflectance | Flat lighting destroys the primary signal |
| Geometry | Doubly curved, deformable near contact | Perspective and deformation confound scale |

**The single most important consequence:** grooves, sipes and cracks are *geometric* features on a *low-contrast, low-albedo* surface. Under flat frontal illumination they nearly vanish. Under **directional** illumination they cast shadows proportional to their depth.

So the highest-leverage decision in this entire project is not the network. **It is the lighting.**

---

## 2. ★ Photometric stereo — the biggest upgrade available

**What it is.** Capture N ≥ 3 images of a static scene from a fixed camera, each lit from a different known direction. Under Lambertian assumptions, per-pixel intensity `I_i = ρ (n · l_i)` gives a linear system solvable for surface normal `n` and albedo `ρ`. Integrate the normal field for a relative depth/height map.

**Why it belongs here.** Your camera is fixed and you control the light. That is exactly the configuration photometric stereo needs, and it converts a plain RGB camera into a **surface-geometry sensor for the cost of three extra LEDs**. Groove depth, rib-edge rounding, cracks, cupping and feathering are all *shape*, and this measures shape.

The industrial-inspection literature is unambiguous that this helps on exactly the hard cases:

- Photometric stereo combined with convolutional segmentation networks is deployed for quality assurance on **highly specular metal parts** — the same specularity problem wet rubber has ([Sensors 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC8838491/)).
- With a static ring light it is **impossible to distinguish low reflectance caused by a stain from a genuine shadowed cavity**; multi-angle pseudo-colour imaging makes local concave shape obvious and materially improves recall.
- A public [photometric-stereo defect dataset for complex structured parts](https://www.nature.com/articles/s41597-025-04454-6) (Scientific Data, 2025) exists for method development.

**How to implement it cheaply.**

```
4 × IR or white LEDs at 90° spacing around the lens, ~30–45° elevation
    ↓
capture 4 frames in ~80 ms, one LED per frame (GPIO-strobed)
    ↓
solve for normal map N(x,y) and albedo map ρ(x,y)
    ↓
feed as extra channels:  [RGB | n_x | n_y | n_z | ρ | integrated height]
```

**Three ways to use the output, in increasing ambition:**

1. **Extra input channels** to SegFormer and ConvNeXt. Cheapest, almost certainly helps.
2. **Pseudo-colour fusion** — map three illumination directions to R/G/B. Concavities become vividly coloured. Works with an unmodified pretrained backbone, which is a real practical advantage on a small dataset.
3. **Integrated height map** as a relative-depth supervision signal for the tread-depth head — no laser required. Scale is ambiguous, but the **TWI bar gives you the 1.6 mm anchor** to fix it.

**Caveats to test, not ignore.**

- Rubber is not Lambertian. It has a specular lobe and subsurface scattering. Expect normals to be biased on glossy/wet tyres — pair with cross-polarisation (§3).
- Requires the tyre to be **static during the 4-frame burst**. At walking pace it will move. Either capture while stationary, or strobe fast enough that inter-frame motion is sub-pixel, or register the burst first.
- Shadows and inter-reflections inside deep grooves violate the model. Use ≥4 lights and robust (median/RANSAC) normal estimation to reject outlier observations.

**Verdict: ★ Core, and the strongest single recommendation in this document.** Also makes an excellent ablation (§6 of `06_EVALUATION.md`) because the comparison "flat light vs photometric stereo, same network" is clean, cheap and likely to produce a large effect.

---

## 3. ★ Cross-polarisation — free specular suppression

**What it is.** Linear polariser on the light source, second polariser on the lens rotated 90°. Specular reflection preserves polarisation and is blocked; diffuse reflection is depolarised and passes.

**Why it matters here.** Wet or new tyres are glossy. Specular highlights saturate the sensor and *bake reflections into the texture*, which the network will happily learn as "wear". Cross-polarisation removes them and, per the machine-vision literature, "exposes hidden features and defects that specular reflections might obscure."

Practical rule from the literature: keep the polarised source **close to the lens and near-normal to the surface**, so polarisation changes little on reflection and is rejected efficiently.

**Cost:** two polarising sheets, ~₹500. **Verdict: ★ Core.** Capture both polarised and unpolarised frames if you can — the *difference* image isolates the specular component, which is itself a useful wetness/gloss feature.

**○ Shape from Polarisation (SfP)** — degree and angle of polarisation relate to surface normals via Fresnel. A polarisation camera gives single-shot normals with no moving lights. But: azimuthal ambiguity, weak signal on rough diffuse surfaces, and a polarisation camera costs far more than four LEDs. **Verdict: ○ Optional, mention as future work.** Photometric stereo achieves the same end more cheaply for this application.

---

## 4. Acquisition techniques

| Technique | Verdict | Note |
|---|---|---|
| **Locked exposure / white balance / focus** | ★ Core | Auto-exposure changes look exactly like tread-condition changes. Lock everything and record the settings. |
| **Flat-field correction** | ★ Core | Capture a uniform grey card once per session; divide it out. Removes vignetting and non-uniform illumination that mimics shoulder wear. |
| **Global-shutter sensor** | ★ Core | Rolling shutter skews a rotating tyre and corrupts the geometry. Non-negotiable if the wheel turns. |
| **IR (850 nm) illumination** | ◆ Ablation | Less ambient contamination, invisible to the driver. But rubber's IR reflectance differs from visible — test before committing. |
| **Telecentric lens** | ✗ Rejected | Removes perspective scaling, which sounds ideal, but the working distance and FOV needed here make it impractical and expensive. |
| **Ring light** | ✗ Rejected as primary | Explicitly the failure case photometric stereo fixes — it cannot separate a stain from a cavity. Useful only as the flat-light control arm of the ablation. |
| **HDR bracketing** | ○ Optional | Helps if bright sky and dark tread coexist. Adds capture time and motion sensitivity. |

---

## 5. Tyre localisation

| Technique | Verdict | Note |
|---|---|---|
| **Fixed calibrated ROI** | ★ Core | With a rigid mount and one tyre, a detector is usually unnecessary. Simplest thing that works. |
| **YOLO11n-seg** | ★ Core (fallback) | For variable approach position or partial views. Cheap insurance. |
| **SAM2 (prompted)** | ◆ Ablation | Zero-shot masks from a point/box prompt. Excellent for *annotation* (§13); as a runtime component it adds latency for little gain over a fine-tuned YOLO. |
| **Hough circle / ellipse fit on the rim** | ○ Optional | If the rim is visible, its projected ellipse constrains wheel pose directly. From a low front view the rim is often occluded by the tyre itself — treat as a bonus signal, not a dependency. |
| **Background subtraction** | ✗ Rejected | Vehicle body, road and shadows all move. Unreliable. |

---

## 6. Fine-structure segmentation

This is where "every single detail" is won or lost.

### Architectures

| Model | Verdict | Reason |
|---|---|---|
| **SegFormer-B2** | ★ Core | Hierarchical transformer encoder + light MLP decoder. Multi-scale features suit an image containing both a whole tread crown and 1-px sipes. Handles varying input resolution without positional-encoding interpolation. B0 as a speed baseline; B4/B5 only if the data proves more capacity pays. |
| **HRNet-W18** | ★ Core (landmarks) | Maintains high-resolution branches throughout, which is precisely what stable shoulder/groove landmarks need. |
| **Mask2Former** | ◆ Ablation | Stronger on instance-level damage (separate cuts, separate cracks). Heavier. Worth one comparison against the SegFormer + damage-head design. |
| **U-Net / DeepLabV3+** | ○ Optional | Solid baselines. DeepLabV3+ specifically has a precedent on tyre depth imagery ([Ko et al. 2021](https://doi.org/10.3390/app112110376)). Use as the classical-baseline arm. |
| **SAM2 + LoRA** | ◆ Ablation | [FS-SAM2](https://arxiv.org/html/2509.12105) reports meaningful gains with **as few as 50 images per class** — directly relevant to a small tyre dataset. |

### Resolution strategy — do not skip this

Downsampling a wide tread crop to 224×224 destroys the narrow grooves and sipes the project exists to measure.

```
pass 1 · whole tyre view      768×768 or 896×896   → coarse structure
pass 2 · tread tiles          512×512, 20–25% overlap → fine structure
         blend probabilities in the overlap region
```

**◆ Ablation worth running:** super-resolution (Real-ESRGAN or a task-trained SR head) before fine segmentation. Plausible, but SR *hallucinates* texture, which is dangerous for a safety task. Measure whether it invents cracks before trusting it.

### Losses for thin structures — an upgrade on the baseline

The Review-1 spec uses `L_focal + 0.7·L_dice + 0.3·L_boundary`. That is sound. Two research-backed additions:

**★ clDice (topology-preserving)** — computes Dice on *skeletonised* predictions, with a theoretical guarantee of topology preservation up to homotopy equivalence. Designed for tubular structures; explicitly proposed for **crack detection in industrial quality control**. A groove or crack broken into three fragments is a serious failure that ordinary Dice barely penalises. ([Shit et al.](https://arxiv.org/pdf/2003.07311))

**◆ Skeleton Recall Loss** — a cheaper connectivity-preserving alternative, better on memory. ([arXiv 2404.03010](https://arxiv.org/html/2404.03010v1))

Known trade-off, worth stating in the paper: clDice preserves connectivity but is *insensitive to boundary shifts within the structure's radius*, so it slightly costs boundary precision. **Combine it with boundary loss rather than replacing it:**

```
L_seg = L_focal + 0.7·L_dice + 0.3·L_boundary + 0.2·L_clDice
```

Ablate the clDice term specifically on sipe and crack connectivity, not just mean IoU.

### Class taxonomy

```
background · tyre_visible · tread_crown · left_shoulder · right_shoulder
main_groove · rib_or_block · sipe · twi_bar · visible_damage
```

If the pilot dataset is small, merge `main_groove + rib_or_block + sipe` into `tread_structure` for the first experiment and split them once tyre and shoulder masks are reliable.

**Always output class probabilities, not hard masks.** Boundary uncertainty feeds shoulder-curve fitting and the decision on whether alignment geometry is trustworthy.

---

## 7. Classical filters — what each one is actually for

Filters are not preprocessing decoration. Each exists to make a *specific downstream algorithm* work. Keep a raw normalised RGB path alongside; enhanced images are **parallel channels**, never replacements.

### Pipeline order

```
1  decode without re-compression        7  Scharr gradients
2  camera undistortion                  8  Gabor / structure tensor
3  fixed ROI                            9  black-hat (dark grooves)
4  flat-field + exposure normalisation  10 SegFormer mask
5  edge-preserving denoise              11 small-kernel morphology in-mask
6  CLAHE on L channel                   12 RANSAC geometry + temporal fusion
```

| Technique | Verdict | Config | Warning |
|---|---|---|---|
| Undistortion | ★ | Calibrated radial + tangential | Recalibrate after any lens/focus/mount change |
| CLAHE (LAB `L`) | ★ | clip ≈ 2, grid 8×8 | Parallel channel only. Strong CLAHE amplifies dirt into fake cracks |
| Bilateral / guided filter | ★ | radius 3–5 px | Larger radii erase sipes |
| Scharr gradients | ★ | 3×3 x and y | More rotationally accurate than Sobel at 3×3. Denoise first |
| Structure tensor | ★ | σ matched to groove width | Reject low-coherence pixels instead of inventing a direction |
| Gabor bank | ★ | 0, ±15, ±30, ±45, 90°; multiple λ | Restrict to tread mask; normalise each response |
| Black-hat morphology | ★ | oriented kernel ≈ groove width | Large kernels mistake shadows for grooves |
| Canny | ◆ | thresholds tuned on **validation** only | A geometry input, not a wear detector |
| LSD / probabilistic Hough | ★ | — | Groove line evidence for the alignment module |
| RANSAC | ★ | threshold in calibrated px/mm | **Record inlier ratio — low inliers must lower confidence** |
| Opening / closing | ★ | 3×3, max 5×5 | Closing can wrongly join two separate cracks |
| Median filter | ○ | 3×3, conditional | Rounds narrow structures. Not every frame |
| Temporal median | ★ | **after** registration | Never average unregistered frames |

### ✗ Rejected as defaults

Large Gaussian blur (destroys the target signal) · aggressive sharpening (manufactures fake cracks) · global histogram equalisation (over-amplifies bright regions) · large morphological kernels (change the physical width of defects) · fixed Canny thresholds across cameras · geometric augmentations that don't update alignment labels · **feeding only edge maps to the network** (discards shading and texture evidence).

---

## 8. Texture descriptors for wear

Deep features dominate, but hand-crafted texture descriptors remain useful as **interpretable baselines and auxiliary channels** — and examiners like seeing that you compared against them.

| Descriptor | Verdict | What it captures |
|---|---|---|
| **GLCM** (contrast, homogeneity, energy, correlation) | ◆ Ablation | Rubber surface roughness; smooth worn rubber vs textured new rubber |
| **LBP / rotation-invariant LBP** | ◆ Ablation | Micro-texture; cheap, robust to illumination scaling |
| **Gabor energy** | ★ Core | Groove orientation and feathering asymmetry — feeds both wear and alignment |
| **Wavelet / frequency-band energy** | ○ Optional | Cupping is periodic around the circumference — an FFT peak on the unrolled map is a natural cupping detector |
| **HOG** | ○ Optional | Used by [Petrovic et al. 2025](https://doi.org/10.1016/j.pes.2025.100080) on segmented tread. Include for direct comparability with that paper |

**Concretely useful:** compute cupping amplitude and period as an **explicit FFT feature on the unrolled tread map**, and supervise the network to predict it. That converts a hard visual class into a measured quantity.

---

## 9. Depth and 3D from this camera

Ranked by defensibility for millimetre claims.

| Method | Verdict | Accuracy precedent | Cost / caveat |
|---|---|---|---|
| **Structured light / laser line** | ◆ Ablation (gold) | **<0.2 mm** ([Wang et al. 2019](https://doi.org/10.1177/1687814019837828)) | Needs laser + calibration. Use as **training-time teacher**, then infer RGB-only |
| **Photometric stereo** | ★ Core | Relative height, scale from TWI anchor | 4 LEDs. Non-Lambertian bias |
| **Groove cross-section + TWI scale** | ★ Core | **0.57 mm** full-vehicle ([Huber et al. 2022, TireEye](https://doi.org/10.36001/phmconf.2022.v14i1.3242)) | Needs a visible groove cross-section and TWI bar |
| **Stereo / RGB-D** | ◆ Ablation | <0.1° for alignment ([Shi et al. 2026](https://doi.org/10.3390/metrology6010004)) | Second camera. Dark low-texture rubber is hard for stereo matching |
| **Depth from defocus** | ○ Optional | — | Needs a wide aperture and careful calibration; weak at this scale |
| **Monocular depth (Depth Anything V2)** | ✗ **Rejected for metrology** | — | Documented failure exactly where you need it: *"fine details and close-range depths in feature-sparse areas are not represented well."* Dark, low-texture, close-range rubber is the worst case. **Run it once as a negative-result ablation** |

**The recommended structure:** metric sensors supervise at training time; RGB + photometric stereo infers at deployment. RGB-only samples train an **ordinal / relative** depth head and are never reported as millimetres.

**★ The TWI bar is your free ruler.** Every road-legal tyre has tread-wear indicators moulded at exactly **1.6 mm** above the groove floor. TireEye uses precisely this as an in-frame scale reference. Detect the TWI bar, measure the adjacent rib height above it, and you have absolute scale that is immune to camera-distance error — on every tyre, forever, at no cost. Add it as a loss term:

```
L_twi = | D̂(rib) − D̂(twi_top) − 1.6 mm |
```

---

## 10. Video: registration and unrolling

| Technique | Verdict | Note |
|---|---|---|
| **Pyramidal Lucas–Kanade + RANSAC** | ★ Core | Sparse, fast, easy to validate and debug. Reject tracks leaving the tread mask or disagreeing with dominant motion |
| **RAFT-Small** | ◆ Ablation | Dense all-pairs correlation; better when repetitive tread patterns defeat sparse tracking. Heavier — justify with an ablation, don't assume |
| **Feature matching (SIFT/ORB/SuperPoint+LightGlue)** | ○ Optional | Learned matchers help on low-texture rubber; adds a dependency |
| **Video-Depth-Anything** | ○ Optional | Temporally consistent monocular depth. Same fine-detail limits as §9 — context, not metrology |
| **Frame averaging without registration** | ✗ Rejected | Blurs exactly the structures being measured |

### Unrolling

```
U(θ,w) = Σ_t q_t · M_t(w) · I_t(π_t(θ,w))  /  Σ_t q_t · M_t(w)
```

`q_t` = frame quality, `M_t` = valid tread mask, `π_t` maps unrolled coordinates into frame `t`.

**Report observed circumference percentage. Never inpaint or generatively fill unseen tread.** An unobserved region is `unknown`, not `healthy` — that distinction is the difference between a research instrument and a liability.

---

## 11. Alignment geometry

| Technique | Verdict | Role |
|---|---|---|
| **HRNet-W18 landmark heatmaps** | ★ Core | Shoulders (near-contact + upper), groove centreline points, lowest visible tread point. **Must output per-point covariance and a visibility flag** — never force a hidden landmark |
| **Structure tensor orientation field** | ★ Core | `φ = ½·atan2(2J_xy, J_xx − J_yy)`; reject low-coherence pixels |
| **LSD / Hough + RANSAC groove fitting** | ★ Core | Dominant groove direction, robust to dirt and block edges |
| **Shoulder contour asymmetry** | ★ Core | Primary camber cue from a frontal view |
| **Analytic wheel-plane normal → angles** | ★ Core | `τ = atan2(n_x, n_y)`, `γ = atan2(n_z, √(n_x²+n_y²))` |
| **Residual MLP (256→256→128)** | ★ Core | Corrects repeatable bias; outputs heteroscedastic log-variance. **Never replaces the analytic estimate** |
| **Rim ellipse → conic back-projection** | ○ Optional | Rigorous when the rim is visible; usually occluded from a low front view |
| **Direct CNN image→angle regression** | ✗ **Rejected as primary** | Learns camera tilt, approach position and background shortcuts. Include as an ablation arm specifically to demonstrate this failure |

**Sign conventions must be defined once, for left and right wheels, and covered by a unit test.** Assert that a horizontal flip negates both angles and that flipping twice is the identity. Half of all alignment bugs are sign errors.

---

## 12. Anomaly detection for unknown damage

You cannot enumerate every way a tyre can be damaged. A supervised head recognises only annotated categories, so pair it with an unsupervised flagger trained on healthy tread.

| Method | Verdict | Note |
|---|---|---|
| **PatchCore** | ★ Core | Memory bank of healthy mid-level patch features; high score = unfamiliar local structure. Strong, well-established industrial baseline |
| **EfficientAD** | ◆ Ablation | Millisecond-scale inference; relevant if the app runs on-device |
| **AnomalyDINO** | ◆ Ablation | Uses **frozen DINOv2 features** with no fine-tuning — very attractive when healthy data is limited |
| **DRAEM / synthetic-anomaly training** | ○ Optional | Simulates defects on healthy images; useful when real defect examples are scarce |
| **Autoencoder reconstruction error** | ✗ Rejected | Superseded; reconstructs defects too well |

**PatchCore is a flagging model, not a classifier.** Fuse it *with* the supervised damage head, never in place of it. Report false positives per healthy tyre — that number decides whether anyone would tolerate the system in practice.

---

## 13. Backbones and pretraining — a research finding that changes the recipe

Recent work directly comparing **DINOv3 vs ImageNet pretraining on industrial inspection** ([arXiv 2605.23472](https://arxiv.org/html/2605.23472)) found:

> On RGB industrial datasets, DINOv3 gives **no clear advantage when the backbone is frozen**, but becomes the **strongest initialisation once fully fine-tuned**, with faster convergence and better final performance for semantic segmentation and instance-level localisation.

**Practical consequence: initialise from a self-supervised checkpoint and fine-tune the whole backbone. Do not linear-probe frozen features.** A common small-data instinct is to freeze the backbone; on this task that is the wrong call.

| Backbone | Verdict | Note |
|---|---|---|
| **ConvNeXt-V2-Tiny** | ★ Core | Convolutional hierarchy suits local texture; MAE pretraining suits small data; practical on dual T4 |
| **SegFormer-B2 (MiT-B2)** | ★ Core | Segmentation encoder |
| **DINOv2 / DINOv3 ViT-S/B** | ◆ Ablation | Fine-tune fully. Also the feature source for AnomalyDINO |
| **MobileNetV3-Small** | ★ Core (quality gate) | Tiny classifier for clean/dirty/wet/glare/occluded/blurred |
| **Domain-adaptive BN + Deep CORAL** | ◆ Ablation | [Vivekanandan & Rajeswari 2026](https://doi.org/10.1016/j.measurement.2026.121509) report unseen-brand accuracy improving 88.2% → 92.4%. Brand shift is a real, measured problem — plan for it |
| ViT-Large / SAM-H backbones | ✗ Rejected | Data-hungry and slow; unjustifiable at this dataset size |

**◆ Self-supervised pretraining on your own unlabelled tyre frames** (MAE or DINO on every frame you capture, labelled or not) is cheap and well-matched to a small labelled set. High value per unit effort.

---

## 14. Uncertainty

| Method | Verdict | Note |
|---|---|---|
| **Heteroscedastic regression** (predict log-variance) | ★ Core | Per-sample uncertainty for free; down-weights hard frames instead of fitting noise |
| **Split conformal prediction** | ★ Core | Distribution-free finite-sample coverage guarantee, ~5 lines of code. Calibrate on a held-out **tyre-level** split |
| **Deep ensembles** (3–5 seeds) | ◆ Ablation | Best-quality uncertainty; 3–5× training cost. You need multi-seed runs for the ablations anyway — reuse them |
| **MC dropout** | ○ Optional | Cheap, weaker calibration |
| **Softmax confidence alone** | ✗ Rejected | Systematically overconfident |

Decision rules must use the **interval**, not the point estimate:

```
if upper_bound(wear_severity) < threshold      → PASS
elif lower_bound(wear_severity) > threshold    → INSPECT
else                                            → MONITOR / UNABLE_TO_MEASURE
```

Report **coverage under distribution shift** (unseen brand) honestly. It will degrade — conformal assumes exchangeability. Measuring the degradation is a stronger result than pretending it doesn't happen.

---

## 15. Annotation efficiency — the practical bottleneck

Pixel-labelling sipes and grooves on 250–400 tyres by hand is the task most likely to sink the timeline. Two techniques change the arithmetic:

**★ SAM2 with memory propagation.** Prompt on one frame; the mask propagates temporally across the clip. Reported throughput improvement in a comparable annotation workflow: **37.8 s/frame → 4.5 s/frame** — roughly 8×. For video of a rotating tyre this is close to ideal, because consecutive frames are highly correlated.

```
annotate frame 1 with SAM2 point prompts
    → propagate through the clip
    → human corrects only the frames that drift
    → retrain, re-propagate
```

**★ Iterative bootstrapping.** Label 30 tyres → train SegFormer-B0 → pre-annotate the next 30 → correct rather than draw. Correction is roughly 5× faster than annotation from scratch.

| Technique | Verdict |
|---|---|
| SAM2 propagation | ★ Core |
| Model-in-the-loop bootstrapping | ★ Core |
| Active learning (label highest-uncertainty tyres first) | ◆ Ablation |
| Weak/coarse labels for the pilot, refined later | ★ Core |
| Fully manual pixel annotation | ✗ Rejected — will consume the project |

**Write the annotation guideline document with example images before labelling anything.** Then have a second team member label 100 items independently and report **Cohen's κ**. Two hours of work, and it is a paper-quality detail most student projects skip.

---

## 16. Build order

Ordered by (value × certainty) ÷ effort. Do not build downward until the layer above works.

| # | Build | Why first |
|---|---|---|
| 1 | Rigid mount, locked camera settings, flat-field, calibration | Everything else is meaningless without it |
| 2 | **Photometric-stereo illumination (4 LEDs) + cross-polarisation** | Largest signal gain available, ~₹2,000 |
| 3 | Frame-quality gate | Prevents garbage entering every downstream metric |
| 4 | Classical baseline: CLAHE, Scharr, Gabor, structure tensor, RANSAC, LK | Debug visuals + non-learning baselines |
| 5 | SAM2-assisted annotation of a 30–50 tyre pilot | Unblocks all learning |
| 6 | SegFormer-B0 → B2 segmentation | Everything downstream consumes these masks |
| 7 | ConvNeXt-V2-Tiny ordinal + multi-label heads | The primary deliverable |
| 8 | TWI detection + metric anchoring | Converts appearance into a defensible scale |
| 9 | HRNet-W18 landmarks + analytic alignment on the jig | Alignment needs its own calibrated ground truth |
| 10 | Residual MLP + temporal fusion + conformal | Turns estimates into decisions |
| 11 | Partial unrolling + PatchCore | Requires a stable frame pipeline first |
| 12 | RAFT-Small, structured light, RGB-D, app | Extensions — only if 1–11 hold up |

---

## 17. Sources

**Photometric stereo & multi-illumination**
- [Photometric Stereo-Based Defect Detection for Steel Components (Sensors 2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8838491/)
- [A dataset for surface defect detection on complex structured parts based on photometric stereo (Scientific Data 2025)](https://www.nature.com/articles/s41597-025-04454-6)
- [Defect segmentation for multi-illumination quality control systems (MVA 2021)](https://link.springer.com/article/10.1007/s00138-021-01244-z)
- [Model-driven photometric stereo for non-diffuse curved surfaces (CIRP Annals)](https://www.sciencedirect.com/science/article/abs/pii/S0007850619300393)

**Polarisation**
- [Polarization 3D imaging technology: a review (Frontiers in Physics 2023)](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2023.1198457/full)
- [Cross-polarised photogrammetry for complex reflective materials (Springer 2023)](https://link.springer.com/chapter/10.1007/978-3-031-35593-6_7)

**Thin-structure segmentation**
- [clDice — a topology-preserving loss for tubular structures](https://arxiv.org/pdf/2003.07311)
- [Skeleton Recall Loss (arXiv 2404.03010)](https://arxiv.org/html/2404.03010v1)
- [SEMIR: topology-preserving graph minors for thin-structure segmentation](https://arxiv.org/pdf/2606.24935)

**Foundation models & annotation**
- [FS-SAM2: adapting SAM2 for few-shot segmentation via LoRA](https://arxiv.org/html/2509.12105)
- [MGD-SAM2: detail-enhanced SAM2 for high-resolution segmentation](https://arxiv.org/pdf/2503.23786)
- [Rethinking Transfer Learning for Industrial Inspection: DINOv3 vs ImageNet](https://arxiv.org/html/2605.23472)
- [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/html/2304.07193v2)

**Monocular depth (negative result)**
- [Depth Anything V2](https://arxiv.org/html/2406.09414v2) · [Assessing Depth Anything V2 as a LiDAR alternative](https://www.researchgate.net/publication/397955860_Assessing_Depth_Anything_V2_monocular_depth_estimation_as_a_LiDAR_alternative_in_robotics) · [Video-Depth-Anything (CVPR 2025)](https://github.com/DepthAnything/Video-Depth-Anything)

**Core architecture papers**
- [SegFormer (NeurIPS 2021)](https://research.nvidia.com/labs/lpr/publication/xie2021segformer/) · [ConvNeXt V2 (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/html/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.html) · [PatchCore (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html) · [HRNet (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Sun_Deep_High-Resolution_Representation_Learning_for_Human_Pose_Estimation_CVPR_2019_paper.html) · [RAFT (ECCV 2020)](https://arxiv.org/abs/2003.12039)

**Tyre-domain papers** — see `09_RELATED_WORK.md` for the annotated list.
