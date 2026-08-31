# 14 — Explainability Protocol

> In this study XAI is **the measuring instrument**, not a decoration at the end of the results section. Get it wrong and the headline finding is wrong.
>
> Read alongside `13_EXPERIMENT_PLAN.md §5`.

---

## 1. The one methodological trap to avoid

> **Grad-CAM on a Vision Transformer is not the same operation as Grad-CAM on a CNN. Comparing them naively is invalid.**

This is not pedantry — there is a 2026 audit paper devoted to exactly this ambiguity. Several published "Grad-CAM for ViT" methods actually apply gradients directly to attention entries rather than averaging gradients into channel-wise weights over a feature map. Those are *different algorithms sharing a name*, and a cross-architecture comparison built on them measures the method, not the model.

Since our whole study compares attribution **across architecture families**, this would silently corrupt the headline result.

**Our rule:**

1. Every architecture gets **at least two** attribution methods: one gradient-CAM-family and one architecture-native
2. Any Grad-CAM variant on a transformer must **state its exact target layer and `reshape_transform`**, and be reported as `GradCAM-ViT(layer, transform)` — never bare "Grad-CAM"
3. **Faithfulness metrics decide which method we trust per architecture** (§4). We do not assume.
4. Cross-architecture claims are made on the **faithfulness-selected** method for each architecture, and the choice is reported

---

## 2. Method × architecture matrix

| Architecture | Primary | Secondary | Notes |
|---|---|---|---|
| **CNNs** (ResNet, ConvNeXt, EfficientNet, RegNet, DenseNet, VGG) | **Grad-CAM** on the last conv stage | HiResCAM, LayerCAM, XGrad-CAM, Ablation-CAM, Score-CAM | Native territory. HiResCAM is more faithful than Grad-CAM where the classifier is not a plain GAP+linear head |
| **ViT / DeiT** | **Chefer et al. transformer attribution** | AttnLRP, attention rollout, gradient-attention rollout | Chefer's method is the reference for class-specific ViT attribution |
| **Swin / hierarchical ViT** | LayerCAM on stage outputs | Chefer-style adaptation | Has a spatial feature map — CAM methods apply more cleanly than to plain ViT |
| **MaxViT / CoAtNet** | Grad-CAM on the conv stem stages | Attention rollout on the attention stages | Report both; a hybrid should be explained hybrid |
| **DINOv2 / CLIP** frozen probe | Attention rollout on the frozen backbone | Grad-CAM on the probe head | Distinguish "what the backbone sees" from "what the head uses" |
| **YOLO26 / RT-DETR** | EigenCAM on the detection neck | Grad-CAM on the backbone | Detection CAMs are noisier — report qualitatively |
| **SegFormer / U-Net / DeepLabV3+** | Segmentation Grad-CAM | Direct mask output | The mask *is* the localisation |
| **Any model** | **Occlusion sensitivity** | RISE, Integrated Gradients, SHAP | Model-agnostic control. Slow but assumption-free — a useful cross-check |

### Why the model-agnostic control matters

Occlusion sensitivity and RISE make no assumptions about internals. When a gradient method and an occlusion map disagree strongly, **the gradient method is usually the one at fault**. Run occlusion on a subsample (~200 images) as an audit, not on everything.

---

## 3. Attribution → the evidence-location metrics

The core measurement of the study. All against the **hand-corrected masks** in `annotation_v1` — 418 clean images annotated in labelme with SAM2 assistance, propagated to all 4,598. **These are ground truth, not pseudo-labels**, which removes the "unvalidated mask" caveat entirely.

### Pipeline

```
image ──▶ trained model ──▶ attribution map A(x,y) ≥ 0, normalised to sum 1
                                        │
image ──▶ annotation_v1 ──▶ masks: M_tyre, M_tread, M_marking, M_damage, M_bg
                                        │
                                        ▼
                            TER   = Σ A·M_tread
                            BAR   = Σ A·M_bg
                            SAR   = Σ A·M_marking     (low-class shortcut)
                            DmgAR = Σ A·M_damage      (high-class shortcut)
                            DAR   = Σ A·M_dirt
                            EDI   = −Σ A log A / log N
```

### Definitions

| Metric | Formula | Interpretation | Target |
|---|---|---|---|
| **TER** Tread Evidence Ratio | `Σ A·M_tread / Σ A` | evidence on the tread surface | **high** |
| **BAR** Background Attribution Ratio | `Σ A·M_bg / Σ A` | evidence outside the tyre entirely | **low** |
| **SAR** Stripe Attribution Ratio | `Σ A·M_stripe / Σ A` | evidence on factory paint/lettering | **low** |
| **DAR** Dirt Attribution Ratio | `Σ A·M_dirt / Σ A` | evidence on dirt and deposits | **low** |
| **DmgAR** Damage Attribution Ratio | `Σ A·M_damage / Σ A` | evidence on visible damage — **a `high`-class shortcut** | **low** |
| **EDI** Evidence Dispersion Index | normalised entropy of `A` | focused vs diffuse | context |

### Getting the region masks

| Mask | Source | Coverage |
|---|---|---:|
| `M_tyre` | **annotated** — `m > 0` | 418/418 |
| `M_tread` | **annotated** — `(m==2) \| (m==3)` | 418/418 |
| `M_marking` | **annotated** — `m == 3`. Factory paint stripes and lettering | 67 (all `low`) |
| `M_damage` | **annotated** — `m == 4` | 63 (all `high`) |
| `M_bg` | derived — `m == 0` | — |
| `M_dirt` | rule-based: pale, desaturated, low-gradient regions inside `M_tyre` | — |
| ~~`M_shoulder`~~ | **dropped** — effectively empty at this viewpoint | — |

**Use the region accessors in `scripts/annotation_regions.py`, never raw class indices.** The mask is a single indexed layer, so `m == 1` means "tyre minus whatever was painted on top" — which on a head-on tyre photo is nearly empty. That exact mistake produced 160 empty box files in the package's own build.

> **Mask quality is no longer a caveat** — these are hand-drawn, structurally validated (0 problems), and the derivative propagation is numerically verified (`annotations/README.md §5`). The remaining honest limitation is that there is **one annotator and no consistency pass**, so no label-reliability figure exists. State that.

### ⚠ What TER actually measures on `final_v1`

Measured on the delivered annotations (`annotations/README.md §4`):

```
tread / tyre area ratio   median 0.990   min 0.943
114 of 418 images have NO visible shoulder at all
```

The camera faces the tread crown head-on, so the shoulders curve out of frame.
`M_tread` and `M_tyre` are therefore **nearly the same region**, and:

> **On this dataset, TER measures attention on the TYRE versus the BACKGROUND —
> not tread versus shoulder. Word every claim accordingly.**

This is not a loss. Background is a *documented* shortcut here: it varies by
session (concrete, brick wall, parked car, vegetation), and frame occupancy
alone scores **0.535 mean macro-F1** — higher than either the colour or texture
probe. So "is the model looking at the tyre or at the scene around it?" is the
sharpest question this data can answer, and TER answers it.

`M_shoulder` is effectively empty and is dropped from the region set.
**`M_marking` becomes the discriminating sub-region**: 67 images, all `low`
class, all carrying factory paint stripes. SAR is now the crispest shortcut
metric available.

### Normalisation caveat — do not skip this

TER is trivially inflated if the tread fills most of the frame. A model attending uniformly to everything would score high.

**Always report the area-normalised form alongside the raw one:**

```
TER_norm = TER / area_fraction(M_tread)
```

`TER_norm = 1.0` means "attends in exact proportion to area" — no preference. `> 1` is genuine preference. **`TER_norm` is the number that goes in the paper**; raw TER is supporting detail.

**This is now measured, not assumed.** The tyre fills 55%–98% of the frame depending on the image, and the *mean* differs by class (low 72%, mid 62%, high 61%). A model attending uniformly would score a raw TER that tracks class membership for free. Area normalisation is what removes that, and without it the headline metric would be partly measuring the same shortcut the study exists to detect.

---

## 4. Faithfulness — deciding which explanations to trust

An attribution map that does not reflect the model tells us nothing about the model. Faithfulness is measured **first**, and it selects the method we then use for the location metrics.

| Metric | Procedure | Better |
|---|---|---|
| **Insertion AUC** | Reveal pixels in descending saliency order; track confidence | higher |
| **Deletion AUC** | Remove pixels in descending saliency order; track confidence | lower |
| **ROAD** | Remove-and-debias — imputes removed regions to avoid the distribution-shift artefact that inflates naive deletion | higher |
| **Pointing game** | Does the saliency peak fall inside `M_tread`? | higher |
| **Sanity check (randomisation)** | Randomise model weights layer by layer; the map **must** degrade | must degrade |

### Two disciplines

**Prefer ROAD over raw deletion.** Blanking pixels creates out-of-distribution images, so a confidence drop can reflect the corruption rather than the removed evidence. ROAD is the corrected version.

**Run the randomisation sanity check once per method.** A method whose output barely changes when the model's weights are randomised is not explaining the model — it is an edge detector. This is cheap and it has failed for published methods before.

### Selection rule

For each architecture, the sanity score is the **mean decorrelation across two
real images** after last-block weight randomisation. A candidate must exceed
**0.05** (correlation falls by more than 0.05) and must produce finite
insertion/deletion faithfulness. Raw pixelwise MAE is not used: sparse CAMs can
move materially while most zero pixels keep MAE artificially small. Among survivors, use the method with the best
insertion-minus-deletion score for TER/BAR/SAR. **State the choice in the
results table.** Do not use one method everywhere for tidiness — that is how
the ViT ambiguity contaminates the comparison.

If no candidate survives for an architecture, record
`excluded_no_faithful_cam`, publish both failed method rows, and continue the
screen. Do **not** relax the threshold after seeing the result and do not treat
an unfaithful map as evidence merely to keep the architecture in the ranking.
NB07 requires at least five valid architectures before seed confirmation and
at least three valid, three-seed-confirmed architectures before Stage B.

Checkpoint identity is a gate before attribution. NB07 infers known backbone
signatures from saved tensor names/shapes and compares them with the declared
architecture. A mismatch is published as
`excluded_checkpoint_arch_mismatch`; an otherwise unreconstructable checkpoint
is `excluded_checkpoint_incompatible`. Neither is allowed into the TER ranking,
and neither stops the remaining architecture screen. This caught all nine
historical `convnextv2_s` run ids, whose saved weights are ResNet-18.

Undefined per-image evidence is stored as missing numeric data, never as a
value to rank. `tyrelib` may emit the text sentinel `NA` when a CAM has zero
total saliency; NB07 coerces every evidence metric to numeric at creation,
resume, concatenation and summary boundaries. Architecture tables report the
valid/total map count and coverage beside TER/BAR so available-case means
cannot be mistaken for complete 60-image evidence.

### Completed gate result

NB07 revision `2026-08-30-r3` is complete and public. The three-seed locked
Stage-B set is **RegNetY-16GF** (TER_norm 1.5785, BAR 0.0310),
**DenseNet-121** (1.5513, 0.0455), and **ResNet-50** (1.5146, 0.0512).
Their raw valid-map counts are 180/180, 178/180, and 180/180. The public gate
contains 1,208 evidence rows and 35 method-faithfulness rows; each selected
row is `xai_status=ok`, eligible, and confirmed on seeds 1–3. This is the
selection NB06 must consume; it must not substitute an accuracy-ranked model.

This definition is XAI revision `2026-08-30-r3`. The first r2 Kaggle attempt
accidentally scored only the first member of its two-image batch and stopped
before publishing any per-run evidence; r2 and r3 rows are never mixed.

---

## 5. Stability of explanations

An explanation that changes with the random seed is not a property of the task.

| Metric | Definition |
|---|---|
| **Cross-seed saliency IoU** | IoU of top-20% saliency regions between seeds of the same config |
| **Cross-fold saliency IoU** | Same image, models from different folds |
| **Temporal saliency consistency** | On video frames of one tyre — does attention jump between frames? |

**Interpretation:** low cross-seed IoU with high accuracy is the classic fingerprint of a model that found *a* discriminative shortcut rather than *the* signal. Different seeds find different shortcuts.

---

## 6. Shortcut stress tests

Interventions, not correlations. Each answers a causal question.

| Test | Intervention | Reading |
|---|---|---|
| **Background replacement** | Replace outside `M_tyre` with other sessions' backgrounds | drop ⇒ background dependence |
| **Background blanking** | Set outside `M_tyre` to grey | drop ⇒ background dependence |
| **Stripe masking** | Inpaint `M_marking` on `low` images (67) | drop in `low` recall ⇒ factory-marking dependence |
| **Damage masking** | Inpaint `M_damage` on `high` images (63) | drop in `high` recall ⇒ damage dependence. **Symmetric to the above — run both or you measure half the problem** |
| **Tread-only crop** | Evaluate on `M_tread` crop | rise ⇒ context was harmful |
| **Grayscale** | Remove colour | drop ⇒ colour dependence. Colour is not a wear cue |
| **Dirt inpainting** | Remove `M_dirt` regions | drop ⇒ dirt dependence |
| **Session holdout** | Leave one session out entirely | drop ⇒ tyre-identity memorisation |
| **Shuffled labels** | Train with permuted labels | **above chance ⇒ pipeline leak. Fix before anything else** |

Report as a matrix: **models × interventions**, cells = Δ macro-F1 with CI. That single table is Figure 4 and it is one of the most informative things the study will produce.

---

## 7. Tooling

| Purpose | Library |
|---|---|
| CAM family | `pytorch-grad-cam` (Grad-CAM, HiResCAM, LayerCAM, XGrad, Ablation, Score, Eigen) |
| Attribution, model-agnostic | `captum` (IG, occlusion, SHAP, LRP) |
| ViT attribution | Chefer reference implementation; AttnLRP |
| Faithfulness | `quantus` (insertion, deletion, ROAD, sanity checks) |
| Masks | SAM2 (`sam2.1_hiera_large`) |
| Detection CAMs | `pytorch-grad-cam` YOLO/DETR adapters |

Add to `environment.yml`: `grad-cam`, `captum`, `quantus`, `shap`, `scikit-image`.

---

## 8. Compute cost

XAI is inference-only, but not free.

| Method | Cost per image | 418 images × 30 models |
|---|---|---|
| Grad-CAM family | ~1 forward + 1 backward | minutes |
| Attention rollout | ~1 forward | minutes |
| Chefer / AttnLRP | ~1 forward + 1 backward | minutes |
| Integrated Gradients | 50 forwards | ~1 h total |
| **Occlusion** | ~500 forwards | **subsample to 200 images** |
| **RISE** | ~4,000 forwards | **subsample to 100 images** |
| Insertion/deletion/ROAD | ~100 forwards | ~2 h total |

**Budget ≈ 15 GPU-h** for the full XAI pass — small next to training. Subsample the expensive methods and say so.

---

## 9. Reporting standards

Every XAI claim carries:

1. **Method name, target layer, and any transform** — `GradCAM(layer4)`, `GradCAM-ViT(blocks.11.norm1, reshape=14×14)`
2. **Faithfulness scores** for that method on that architecture
3. **The randomisation sanity check result**
4. **Cross-seed stability**
5. **Area-normalised TER**, not just raw
6. **Mask quality** for the regions used

Qualitative saliency panels are for intuition. **Every quantitative claim is a number with a confidence interval**, bootstrapped at the *tyre* level — never at the image level, because 418 images are 12 tyres.

---

## 10. Hypotheses — registered before evidence

Frozen publicly in `analysis/hypotheses.json` at
**2026-08-30T10:06:21Z**, before the first XAI evidence row:

> **H1.** TER_norm predicts cross-fold macro-F1 stability better than validation accuracy.
>
> **H2.** High SAR/DmgAR predicts larger recall loss when markings/damage are masked.
>
> **H3.** Fine-grained architectures (bilinear / attention-bilinear) have higher TER_norm than plain classifiers at matched accuracy.

Report the outcomes **whatever they are**. A pre-registered hypothesis that fails is a finding. A hypothesis invented after seeing the scatter plot is not — and an examiner who knows the difference will ask.
