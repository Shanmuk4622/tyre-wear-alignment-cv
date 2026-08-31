# 04 — Model Zoo and Training Recipes

> Implementation reference for `13_EXPERIMENT_PLAN.md`. **That document says what we run and why; this one says exactly how.**
>
> Every entry: exact `timm`/library identifier, input size, batch size, LR, CAM target layer, and a runtime estimate to calibrate the work-sharding cost table (`05 §3`).

---

## 1. The task

Three-class **ordinal** mileage proxy — `low` (0) · `mid` (1) · `high` (2).

Ordinal, not nominal: confusing `low` with `high` must cost more than confusing `low` with `mid`. Report macro-F1 **and** quadratic weighted kappa **and** mean absolute class error.

**Splits:** the dataset's supplied `splits/cv{0,1,2}_{train,validation}.csv`. Never construct your own.

---

## 2. Standard recipe (Stage A — held fixed while architecture varies)

```yaml
input:        384 × 384          # revisited in Stage B §9 factor 4
normalise:    ImageNet mean/std  # [0.485,0.456,0.406] / [0.229,0.224,0.225]
head:         CORAL ordinal (3 classes → 2 cumulative logits)
loss:         CORAL cumulative BCE
sampler:      class_session_balanced_weight  (from the manifest)
optimiser:    AdamW, wd 0.05
lr:           3e-4 head / 3e-5 backbone
schedule:     5-epoch linear warmup → cosine → 1e-6
max_epochs:   60                 # EQUAL BUDGET. Every run trains all 60.
early_stop:   NONE               # deliberately removed — see below
monitor:      val_qwk            # used to pick the BEST checkpoint, not to stop
batch:        32 (16 per GPU × 2 T4)
precision:    fp16 + GradScaler  # T4 has no bf16
memory:       channels_last      # RegNetY-16GF: contiguous NCHW safety profile
ema:          0.999
augment:      dataset policy only  # derivatives are pre-generated
val augment:  NONE — ever
seeds:        1, 2, 3
```

### No early stopping. Every run trains all 60 epochs.

Early stopping is **removed entirely** from this study. Three reasons it was the wrong call here:

1. **It makes runs unpredictable in length.** The work-sharding estimate (`05 §3`) depends on knowing how long a run takes *before* it starts. A run that might stop at epoch 22 or might go to 60 makes every printed "this will take N hours" a guess.
2. **Fold 2's validation set is 104 images.** Validation QWK is genuinely noisy at that size, so a patience counter fires on noise as often as on convergence — stopping good runs early and keeping bad ones.
3. **It quietly changes the comparison.** If ConvNeXt stops at 28 epochs and ViT-S runs 60, they did not receive the same budget, and the architecture comparison is confounded by a stopping rule.

**Equal budget for every architecture is what keeps the comparison fair.**

`val_qwk` is still tracked every epoch and still selects `ckpt_best.pt` — we just never *stop* on it. Convergence behaviour remains visible in `epochs.csv`: if a model's best epoch is consistently 25 while another's is 58, that is a finding about convergence under small data, and it is recorded rather than acted on mid-run.

**Held constant across all of Stage A.** If the recipe changes, the architecture comparison stops being a comparison.

### Two things that are easy to get wrong

**Validation is clean-originals only, at native aspect.** The derivatives are 768×768 letterboxed; the clean images are 1152×1536. Resize deterministically, never randomly.

**`class_session_balanced_weight` is in the manifest — use it.** Sessions range from 2 to 67 images. Uniform sampling makes `mileage_100000_plus__session_004` dominate the `high` class.

---

## 3. Tier 0 — non-learning baselines

| Model | Implementation | Established result |
|---|---|---|
| Colour probe | `scripts/dataset_shortcut_probe.py` | **mean 0.491** (0.952 / 0.399 / 0.123) |
| Structure probe | same script | **mean 0.483** (0.354 / 0.119 / 0.976) |
| HOG + linear SVM | `skimage.feature.hog` + `sklearn` | — (comparable to Petrovic et al.) |
| Majority class | — | 0.423 accuracy |
| Random-init ResNet-50 | `timm`, `pretrained=False` | isolates transfer contribution |

**These lines appear on every results figure.** Cost: minutes on CPU.

---

## 4. Tier 1–2 — CNNs

| Model | `timm` name | Res | BS | CAM target layer | ~min/run |
|---|---|---:|---:|---|---:|
| ResNet-50 | `resnet50` | 384 | 32 | `layer4[-1]` | 12 |
| ResNeXt-50 | `resnext50_32x4d` | 384 | 32 | `layer4[-1]` | 13 |
| DenseNet-121 | `densenet121` | 384 | 32 | `features.norm5` | 14 |
| VGG-16-BN | `vgg16_bn` | 384 | 16 | `features[-1]` | 18 |
| **ConvNeXt-V2-T** | `convnextv2_tiny.fcmae_ft_in22k_in1k` | 384 | 32 | `stages[-1]` | 16 |
| ConvNeXt-V2-S | `convnextv2_small` | 384 | 16 | `stages[-1]` | — **excluded: no published timm pretrained Small weights** |
| **EfficientNetV2-S** | `tf_efficientnetv2_s.in21k_ft_in1k` | 384 | 32 | `conv_head` | 15 |
| RegNetY-16GF | `regnety_016` | 384 | 32 | `s4` | 14 |
| MobileNetV4-Conv-M | `mobilenetv4_conv_medium` | 384 | 64 | `blocks[-1]` | 8 |

**Expected front-runners:** ConvNeXt-V2-T and EfficientNetV2-S. Modern CNNs remain the strongest option under limited data and compute, and this dataset is small.

**ConvNeXt-V2-S is not a valid pretrained arm.** The former
`convnextv2_small.fcmae_ft_in22k_in1k` identifier used a weight tag that timm
does not publish for the Small topology. The old emergency fallback silently
produced nine ResNet-18 checkpoints under Small run ids; those records are
quarantined and the fallback has been removed. The untagged topology remains
in the registry only so checkpoint-integrity checks can diagnose the mismatch.

**VGG-16 earns its place** as Grad-CAM's original home — a useful reference point for the XAI comparison even though its accuracy will not lead.

---

## 5. Tier 3–4 — transformers and hybrids

| Model | `timm` name | Res | BS | Attribution target | ~min/run |
|---|---|---:|---:|---|---:|
| ViT-S/16 | `vit_small_patch16_384.augreg_in21k_ft_in1k` | 384 | 32 | Chefer; `blocks[-1].norm1` | 20 |
| DeiT III-S | `deit3_small_patch16_384.fb_in22k_ft_in1k` | 384 | 32 | Chefer | 20 |
| **Swin-T** | `swin_tiny_patch4_window7_224` | 224 | 32 | LayerCAM on `layers[-1]` | 14 |
| Swin-S | `swin_small_patch4_window7_224` | 224 | 16 | LayerCAM on `layers[-1]` | 24 |
| **MaxViT-T** | `maxvit_tiny_tf_384.in1k` | 384 | 16 | Grad-CAM on conv stages + rollout on attn | 30 |
| CoAtNet-0 | `coatnet_0_rw_224` | 224 | 32 | Grad-CAM on conv stem | 16 |

> **Swin and CoAtNet are fixed-window at 224.** Do not silently feed them 384 — that is ⚠ Bug 5 from the playbook (an architecture that cannot do what the sweep assumes). Either use a 384 checkpoint variant where one exists, declare 224-only and note the limitation in the results table, or exclude them from the resolution sweep. **Decide before Stage A, not during it.**

**Expectation:** transformers underperform at this data scale. Demonstrating that cleanly, with matched recipes and 3×3 repeats, is a legitimate result — not a failure.

---

## 6. Tier 5 — self-supervised and foundation backbones

Each in **three modes**, because the mode is the interesting variable:

| Mode | What trains |
|---|---|
| `frozen-linear` | backbone frozen, linear head |
| `frozen-attentive` | backbone frozen, attentive-pooling head |
| **`full-ft`** | everything |

| Backbone | Identifier | Res | BS | ~min/run (full-ft) |
|---|---|---:|---:|---:|
| DINOv2 ViT-S/14 | `vit_small_patch14_dinov2.lvd142m` | 392 | 32 | 22 |
| DINOv2 ViT-B/14 | `vit_base_patch14_dinov2.lvd142m` | 392 | 16 | 40 |
| DINOv3 ViT-S | (HF checkpoint) | 384 | 32 | 22 |
| CLIP ViT-B/16 | `vit_base_patch16_clip_384.laion2b_ft_in12k_in1k` | 384 | 16 | 38 |
| SigLIP ViT-B/16 | `vit_base_patch16_siglip_384` | 384 | 16 | 38 |

**The sub-result to report explicitly:** published work on industrial inspection found frozen SSL features give **no clear advantage** on RGB tasks while **fully fine-tuned** SSL initialisation is strongest. We test that on tyres. Whether it replicates or not, it is a clean, citable finding.

---

## 7. Tier 6 — fine-grained specialists

Tyre wear **is** a fine-grained visual classification problem: subtle within-class texture differences on objects of near-identical global shape. FGVC methods are built for exactly this, and I found no prior application to tyre wear. **This is the most promising novelty on the model axis.**

| Method | Implementation | Backbone | Notes |
|---|---|---|---|
| Bilinear CNN | outer product of two stream features | ResNet-50 ×2 | The classic. Expensive — use compact bilinear |
| Compact Bilinear | Tensor Sketch projection | ResNet-50 | ~1 order smaller than full BCNN |
| **Hierarchical Bilinear Pooling** | cross-layer bilinear on stages 3/4/5 | ResNet-50 / ConvNeXt | Captures inter-layer part interaction; SOTA on CUB/Aircraft/Cars |
| Attention-Bilinear (CSAB) | channel-then-spatial attention + bilinear | ResNet-50 | Also yields attention maps → **direct XAI comparison** |
| Coarse2Fine | two-stage: coarse crop → fine classify | ConvNeXt-V2-T | Natural fit given our framing variance |

**Why CSAB is worth prioritising:** it produces attention maps *natively*, so its TER can be computed both from Grad-CAM and from its own attention — a within-model check on whether the two agree. That is a nice, cheap methodological detail.

**Hypothesis H3** (`14 §10`): FGVC architectures achieve higher `TER_norm` than plain classifiers at matched accuracy.

---

## 8. Tier 7 — detection and segmentation

| Model | Identifier | Task | Trained on |
|---|---|---|---|
| **SAM2** | `sam2.1_hiera_large` | zero-shot masks | nothing — the teacher |
| YOLO26-n / -s | `yolo26n.pt` / `yolo26s.pt` | detection | SAM2 pseudo-boxes |
| YOLO26-n-seg | `yolo26n-seg.pt` | instance seg | SAM2 pseudo-masks |
| RT-DETRv2-S | `rtdetr-l.pt` | detection | SAM2 pseudo-boxes |
| SegFormer-B0 / B2 | `nvidia/mit-b0`, `mit-b2` | semantic seg | SAM2 pseudo-masks |
| U-Net (ResNet-34) | `smp.Unet` | semantic seg | SAM2 pseudo-masks |
| DeepLabV3+ | `smp.DeepLabV3Plus` | semantic seg | SAM2 pseudo-masks |

**Use YOLO26, not YOLO11.** Released January 2026: NMS-free end-to-end head, DFL removal, Progressive Loss Balancing, Small-Target-Aware Label Assignment, MuSGD optimiser. Reports up to 43% faster CPU ONNX inference than YOLO11n, and YOLO26x reaches 57.5 mAP on COCO — above RT-DETRv2-x with fewer parameters.

**We now have manual annotations** (`15_ANNOTATION_GUIDE.md`) — 418 hand-corrected images propagated to all 4,598. So detection and segmentation are **genuinely supervised**, not pseudo-label distillation.

Train each on **both** label sources and compare:

| Arm | Labels | What it answers |
|---|---|---|
| `manual` | hand-corrected masks/boxes | the real result |
| `sam2` | SAM2 zero-shot pseudo-labels | how much does manual annotation actually buy? |

That comparison is a free extra result, and it is only possible because both exist.

---

## 8b. Tier 8 — the integrated pipeline (built last)

```
SegFormer (Stage-D winner) → best classifier (Stage-A/B winner) → HRNet → PatchCore → fusion
```

Every component chosen from evidence rather than guessed. Runs **after** Stages A–G, with component-ablation arms (remove one block at a time). ~90 min/run; 6 configs × 9 = 54 runs ≈ 25 GPU-h.

**The question it answers:** do the study's findings *compose*? If a system assembled from empirically-selected parts beats the best single model, that is the paper's closing result. If it does not, that is more interesting than most benchmarks manage.

---

## 9. Stage B — the technique axis (OFAT on the top 3 architectures)

One factor at a time from the standard recipe. **Report effect size with CI, not just a win/lose.**

**Locked by public NB07 r3:** RegNetY-16GF, DenseNet-121, and ResNet-50. The
selection used three-seed TER_norm with a BAR tie-break after method
faithfulness/randomisation gates; accuracy did not enter the rule. NB06 must
load these names from `tables/stage_b_selection.csv`, never from a hard-coded
fallback list.

**No model substitution after the 2026-08-31 NB06 crash.** Public telemetry
showed only ~4–5 GB used on each 16 GB T4; the failure was a linear host-RAM
increase in the mask-based ROI loader. RegNetY-16GF, DenseNet-121 and ResNet-50
remain locked. The repair changes only image/mask loading and memory lifecycle,
and reproduces the previous ROI crop coordinates exactly.

A later clean retry exposed a separate runtime fault: RegNetY-16GF ROI seeds 1
and 2 independently failed in the first grouped convolution with
`CUDNN_STATUS_EXECUTION_FAILED` / `misaligned address`, while each T4 held only
about 1.1 GB. Tyrelib v4 therefore runs **that same RegNet model** with
contiguous NCHW tensors and cuDNN autotuning disabled. Batch 32, 384px, AMP,
weights, optimiser and epoch budget are unchanged; all other models retain the
Stage-A `channels_last` path. This runtime choice is logged in every epoch and
summary rather than hidden in the scientific configuration.

| # | Factor | Levels | Expectation |
|---:|---|---|---|
| 1 | Head | CE · **CORAL** · regression+threshold | CORAL wins — classes are ordered |
| 2 | Loss | CE · focal · label smoothing · class-balanced | small effects |
| 3 | Sampler | uniform · class-weighted · **session-balanced** | session-balanced matters most |
| 4 | Resolution | 224 · 384 · 512 · 768 | **higher should win** — fine detail |
| 5 | **ROI** | full frame · SAM2 tread crop · CAM crop | **likely the biggest single effect** |
| 6 | Augmentation | none · dataset · +photometric · +RandAugment | |
| 7 | Transfer | random · IN1k · IN21k · DINOv2 · CLIP | |
| 8 | FT depth | frozen · last-block · **full** · LoRA | full wins, per the literature |
| 9 | Regularisation | EMA · stoch. depth · wd sweep | |
| 10 | Preprocessing | raw · CLAHE · +Scharr/Gabor · grayscale | grayscale is a **shortcut test**, not an improvement |
| 11 | Ensemble | single · seed · architecture · TTA | |
| 12 | Calibration | none · temperature · **conformal** | |

**Factor 5 (ROI) is the one I would run first.** Framing varies enormously between sessions — some frames crop the shoulders off, others include a parked car. Removing that variance should matter more than any architecture choice.

**Factor 10's grayscale arm is diagnostic.** Colour carries no wear information. If grayscale *hurts*, the model was using colour — and colour here means dirt and paint stripes.

---

## 10. Ensembling and calibration (Stage G)

Reuses existing checkpoints — nearly free.

| Method | Notes |
|---|---|
| Seed ensemble | average logits over 3 seeds. Usually the single cheapest gain |
| Architecture ensemble | top-3 architectures. Report whether diversity helps |
| TTA | horizontal flip + multi-scale. **Flip is safe for this task only** — banned for any signed toe/camber work |
| Temperature scaling | fit on the calibration split |
| **Conformal prediction** | split conformal → prediction *sets*, and an explicit `uncertain` outcome |

The conformal arm matters: the dataset's own recommended experiment order asks for an `uncertain / request another frame` outcome. Prediction sets deliver exactly that, with a distribution-free coverage guarantee.

---

## 11. Cost table for work sharding

Static estimates for `assign_workers` (`05 §3`). **Never update these from measurements** — measurements refine the *printed plan* only (⚠ Bug 7).

```python
STATIC_COST_HINTS = {           # minutes per single run (1 fold, 1 seed, ~45 epochs)
  "mobilenetv4": 11, "swin_t": 12,      "coatnet0": 13,  "swin_s": 21,
  "regnety016": 24,  "vit_s": 26,       "deit3_s": 26,   "resnet50": 27,
  "effnetv2s": 29,   "dinov2_s": 30,    "resnext50": 32, "convnextv2_t": 34,
  "densenet121": 37, "bcnn": 50,        "convnextv2_s": 55,
  "hbp": 55,         "csab": 55,        "vgg16bn": 61,   "coarse2fine": 61,
  "clip_b16": 69,    "siglip_b16": 69,  "maxvit_t": 72,  "dinov2_b": 72,
  "pipeline_t8": 90,
}
```

Derived from measured T4 throughput scaled by relative FLOPs and resolution (`13 §8`). **These replace an earlier estimate that was roughly 2× optimistic.**

**Calibrate against two real runs before the first big sweep, then freeze.** A first guess in a comparable build predicted 1.73 h for a run that took 2.89 h — a 40% underestimate; two measured runs fixed both scale and ratio. After freezing, measurements refine only the *printed plan*, never the assignment (⚠ Bug 7).

---

## 12. Per-model checklist

Before a model enters Stage A:

- [ ] Builds, forwards, backprops at **every** resolution it will see
- [ ] CAM target layer identified and **verified to produce a non-degenerate map**
- [ ] At least one attribution method passes the randomisation sanity check (`14 §4`)
- [ ] Batch size fits 16 GB at the chosen resolution
- [ ] Runtime measured once and added to the cost table
- [ ] `config_hash` stable across processes
- [ ] Included in the preflight notebook

Skipping the CAM-target check is how you reach Stage E and discover a third of the zoo has no usable attribution.
