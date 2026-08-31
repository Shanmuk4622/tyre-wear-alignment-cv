# 13 — The Experiment Plan

> **This replaces the previous approach.** We are no longer building one pipeline. We are running a **broad, controlled, XAI-grounded comparative study** of tyre-wear recognition — and the study itself is the contribution.
>
> Read `12_DATASET_FINAL_V1.md` first. Everything here is designed around what that dataset can and cannot prove.

---

## 1. Why the approach changed

The earlier plan was a single engineered system: SegFormer → ConvNeXt → HRNet → PatchCore → fusion. That plan assumed labels we do not have (masks, gauge readings, alignment angles) and hardware we are not building.

More importantly, it answered the wrong question. With 418 clean images from **12 tyres**, you cannot build a deployable system. But you *can* run something genuinely valuable and publishable, if you pick the right question.

### The question a naive benchmark would ask, and why it fails

> *"We trained 20 architectures. Which was most accurate?"*

This is worthless here, and I can prove it. Two deliberately stupid baselines, on the dataset's own folds:

| Baseline | fold 0 | fold 1 | fold 2 | mean |
|---|---:|---:|---:|---:|
| 10 colour numbers from a 64×64 thumbnail | **0.952** | 0.399 | 0.123 | 0.491 |
| 9 texture numbers from the tread band | 0.354 | 0.119 | **0.976** | 0.483 |

Fold-to-fold swing of **0.12 → 0.98**. Any accuracy difference between two architectures will be swamped by that variance. A leaderboard built on this data would be **ranking noise and calling it science.**

### The question we ask instead

> ### **Not "which model is most accurate?" but "which model actually looks at the tread?"**

Accuracy is not identifiable on 12 tyres. **Where the evidence comes from is.** And it is the property that actually determines whether a model will survive contact with a tyre it has never seen.

This reframing does three things at once:

1. It is **honest** about what the data supports
2. It makes **XAI the measuring instrument**, not decoration — which is where the novelty lives
3. It still delivers exactly what you asked for: many architectures, many techniques, classification + detection + segmentation, Grad-CAM and beyond

---

## 2. The contribution, stated plainly

> **A shortcut-aware, explanation-grounded benchmark of tyre-wear recognition under small-sample conditions.**
>
> We train a wide sweep of architectures and techniques, generate architecture-appropriate saliency for every one, and measure **how much of each model's evidence falls on the tread** rather than on background, dirt, or factory paint stripes. We then test whether that measurement predicts cross-fold generalisation better than validation accuracy does.

Four things make this defensible rather than just busy:

| | |
|---|---|
| **A measured problem, not an assumed one** | We quantified the shortcut risk before designing around it (§1) |
| **A new metric family** | Tread Evidence Ratio and its siblings (§5) — computable with **zero manual annotation** |
| **A falsifiable hypothesis** | *Does evidence quality predict generalisation better than accuracy?* It can come out "no", and that is still a result |
| **Breadth with discipline** | Every configuration is 3 folds × 3 seeds. Nothing is reported from a single run |

**The one-line pitch:** *on small tyre datasets, validation accuracy is noise and attribution location is signal.*

---

## 3. Scope for this phase

| | |
|---|---|
| **In scope now** | Tyre **wear** — recognition, localisation, and explanation |
| **Deferred** | Wheel **alignment** |
| **Data** | `final_v1` (418 clean / 4,180 derivatives / 12 sessions), plus video clips captured separately |
| **New hardware** | **None.** No camera build, no rig, no illumination array |

### A note on the videos

We also have video clips captured under different conditions. Recorded exactly as described: **camera positioned below and in front of the vehicle, viewing one tyre.** They are not calibrated and carry no measured labels, so their role is:

- **Qualitative robustness checking** — does a model trained on stills survive real video frames?
- **Temporal consistency** — does the prediction flicker frame-to-frame on the same tyre? An unstable prediction is evidence of shortcut reliance
- **A held-out domain** — the strongest generalisation test available to us

They are **not** training data and **not** a quantitative test set. Their capture geometry is recorded for the report.

### On deferring alignment

You suggested alignment might be the easier of the two. I want to flag this respectfully, because the sequencing is right but the reasoning matters for later planning: **alignment is the harder problem, not the easier one.**

Wear is visible in the image. Alignment is a geometric quantity measured relative to a calibrated vertical and a known travel direction — neither of which exists in this dataset. Published target-free systems need stereo cameras, markers, or an RGB-D sensor to reach useful precision, and the best recent monocular result is a controlled single-wheel feasibility study.

So we defer it because it is currently **impossible without calibration data**, not because it is easy. When we return to it we should expect it to be the hard half. `01_CONCEPT.md §2` has the observability analysis.

---

## 4. The three task paradigms — and how we do them without annotations

You asked for recognition, detection and segmentation. We have **classification labels only**. Rather than treat that as a blocker, it becomes part of the design.

```
                    ┌──────────────────────────────────────┐
                    │  A. CLASSIFICATION (recognition)     │
                    │     3-class ordinal mileage proxy    │
                    │     ← the only real labels we have   │
                    └──────────────┬───────────────────────┘
                                   │ CAMs from trained classifiers
                                   ▼
                    ┌──────────────────────────────────────┐
                    │  B. LOCALISATION (detection)         │
                    │     weakly-supervised: CAM → box     │
                    │     SAM2 pseudo-boxes as reference   │
                    │     YOLO26 trained on pseudo-labels  │
                    └──────────────┬───────────────────────┘
                                   │ CAM points/boxes prompt SAM2
                                   ▼
                    ┌──────────────────────────────────────┐
                    │  C. SEGMENTATION                     │
                    │     SAM2 zero-shot pseudo-masks      │
                    │     → student segmenters distilled   │
                    │     → masks feed the XAI metrics     │
                    └──────────────┬───────────────────────┘
                                   │ tread mask
                                   ▼
                    ┌──────────────────────────────────────┐
                    │  D. XAI MEASUREMENT                  │
                    │     how much evidence is on tread?   │
                    └──────────────────────────────────────┘
```

### A · Classification — the backbone of the study

Three ordinal classes (`low`/`mid`/`high` mileage proxy). This is where the architecture and technique sweeps happen.

### B · Detection, honestly

We have no bounding boxes, so we do **weakly-supervised object localisation (WSOL)** — a legitimate task with its own literature, not a workaround.

1. Take a trained classifier's CAM, threshold it, fit a box
2. Compare against a **SAM2-derived reference box** (see C)
3. Report standard WSOL metrics: MaxBoxAcc, box IoU, pointing game
4. Separately, train **YOLO26** on SAM2 pseudo-boxes → a real tyre detector → use it as an ROI stage → **measure whether ROI cropping improves classification**

That last step turns detection from a box-ticking exercise into an ablation with a real answer. Given that framing varies wildly between sessions (`12 §2`), I expect ROI cropping to matter.

### C · Segmentation, without annotating anything

The CAM→SAM prompting pipeline from the weakly-supervised literature:

```
classifier CAM  →  peak points + threshold box  →  SAM2 prompt  →  tyre / tread mask
                                                        ↓
                                        pseudo-mask pool (4,598 images)
                                                        ↓
                              train student segmenters: SegFormer-B0/B2,
                              U-Net, DeepLabV3+, YOLO26-seg
```

**Honesty requirement:** hand-verify ~50 pseudo-masks and report their quality (IoU against manual masks, failure rate). A pseudo-label pipeline whose error is unmeasured is not evidence.

**Why segmentation earns its place here:** the masks are what make §5's metrics computable. Segmentation is not a separate deliverable competing for attention — it is the instrument that lets us measure attribution location. That is a clean, defensible reason for it to exist in the study.

---

## 5. The metrics that make this novel

All computed against the SAM2-derived tread mask. **No manual annotation required.**

### Evidence-location metrics

| Metric | Definition | Good value |
|---|---|---|
| **TER** — Tread Evidence Ratio | attribution mass inside the tread mask ÷ total mass | **high** |
| **BAR** — Background Attribution Ratio | mass entirely outside the tyre | **low** |
| **SAR** — Stripe Attribution Ratio | mass on factory paint-stripe / lettering regions | **low** |
| **DAR** — Dirt Attribution Ratio | mass on detected dirt/deposit regions | **low** |
| **EDI** — Evidence Dispersion Index | entropy of the normalised attribution map | interpretable either way |

`SAR` deserves special mention. New tyres carry **coloured paint stripes and white lettering from the factory** — visible in the native-resolution crops. That is a direct, free giveaway for the `low` class. A model with high SAR has learned the factory marking, not the tread. **This is a concrete, named, measurable shortcut, and I have not seen it reported in the tyre-vision literature.**

### Faithfulness metrics (is the explanation even honest?)

Standard, and necessary — a saliency map that doesn't reflect the model tells you nothing:

- **Insertion AUC** — confidence rise as high-saliency pixels are revealed (higher better)
- **Deletion AUC** — confidence drop as they are removed (lower better)
- **ROAD** — remove-and-debias, avoids the distribution-shift artefact that plagues naive deletion
- **Pointing game** — does the saliency peak land inside the tread mask?

> Use plausibility metrics (pointing game, IoU) *alongside* faithfulness, never instead of it. Plausibility can reward a map that looks sensible to a human while misrepresenting the model.

### Stability metrics

| Metric | Definition |
|---|---|
| **Cross-fold spread** | max − min macro-F1 across the 3 folds. **Report this as prominently as the mean** |
| **Cross-seed saliency IoU** | agreement of thresholded saliency between seeds. Unstable explanations ⇒ untrustworthy model |
| **Session-level variance** | per-session accuracy spread — exposes single-tyre memorisation |

### Shortcut stress tests

Direct interventions. Each answers a yes/no question about what the model depends on:

| Test | Question |
|---|---|
| **Background replacement** | Swap background outside the SAM2 mask. Accuracy drop = background dependence |
| **Stripe masking** | Inpaint the paint stripe/lettering on `low` images. Drop = factory-marking dependence |
| **Tread-only crop** | Evaluate on tread-only crops. Rise = background was hurting |
| **Grayscale** | Remove colour entirely. Drop = colour dependence, and colour is *not* a wear cue |
| **Session-holdout** | Leave one session out. Drop = tyre-identity memorisation |
| **Shuffled-label control** | Train on shuffled labels. Anything above chance = leakage in the pipeline |

That last one is cheap and mandatory. It is the sanity check that catches bugs no metric will.

---

## 6. The model zoo

Breadth is the point — but breadth without discipline produces noise. **Every entry runs 3 folds × 3 seeds.** No exceptions, no single-run numbers.

### Tier 0 — non-learning baselines (the floor)

| Model | Why |
|---|---|
| Colour probe (10 features) | Established: mean 0.491. **Everything must beat this** |
| Structure probe (9 features) | Established: mean 0.483 |
| HOG + SVM | Directly comparable to Petrovic et al. 2025 |
| Majority class | 0.423 accuracy |
| **Random-init CNN (no pretraining)** | Isolates how much comes from transfer alone |

### Tier 1 — classical CNNs

ResNet-50 · ResNeXt-50 · DenseNet-121 · VGG-16

*Rationale: the field's reference points, and Grad-CAM's native home. Cheap.*

### Tier 2 — modern CNNs

**ConvNeXt-V2-T / -S** · **EfficientNetV2-S** · RegNetY-16GF · MobileNetV4

*Rationale: strongest CNNs under limited data and compute. Likely winners.*

### Tier 3 — vision transformers

ViT-S/16 · DeiT III-S · **Swin-T / -S**

*Rationale: different inductive bias. Expected to struggle at this data scale — and demonstrating that cleanly is itself a result.*

### Tier 4 — hybrids

**MaxViT-T** · CoAtNet-0

*Rationale: convolution + attention. The interesting middle ground.*

### Tier 5 — self-supervised and foundation backbones

**DINOv2 ViT-S/B** · DINOv3 · CLIP ViT-B/16 · SigLIP

*Each in three modes: frozen linear probe · attentive probe · full fine-tune.*

*Rationale: research says frozen SSL features give no clear advantage on RGB industrial tasks, while fully fine-tuned SSL init is strongest. **We test that claim on our data** — a clean, citable sub-result.*

### Tier 6 — fine-grained specialists

Bilinear CNN · Hierarchical Bilinear Pooling · attention-bilinear (CSAB) · Coarse2Fine two-stage

*Rationale: tyre wear **is** a fine-grained visual classification problem — subtle within-class texture differences. FGVC methods are built for exactly this and are, as far as I can tell, unexplored on tyres. Potentially the strongest single novelty in the model axis.*

### Tier 7 — detection and segmentation

**YOLO26** (n/s, det + seg) · RT-DETRv2 · SegFormer-B0/B2 · U-Net · DeepLabV3+ · SAM2 (zero-shot teacher)

*YOLO26 released January 2026 — NMS-free end-to-end head, ProgLoss, STAL, MuSGD. Use the current generation, not YOLO11.*

### Tier 8 — the integrated pipeline (the finale)

The original engineered system, built **last**, on everything the study learned:

```
SegFormer  →  ConvNeXt (or whichever architecture won)  →  HRNet  →  PatchCore  →  fusion
```

**Why it belongs at the end, not the beginning:** every component choice becomes evidence-based instead of guessed. Stage A says which backbone. Stage B says which resolution, which ROI, which head. Stage D says which segmenter. Stage E says which of them actually looks at the tread.

**What it demonstrates:** that the study's findings *compose* — that a system assembled from empirically-selected parts beats the best single model. If it does not, that is also worth reporting, and it is a more interesting result than most benchmarks produce.

**Run budget:** 1 configuration × 3 folds × 3 seeds = 9 runs, plus ablations removing one component at a time (5 × 9 = 45). ~25 GPU-h.

> This is where the earlier plan comes back — **earned rather than assumed.** Now that we have annotations (`15_ANNOTATION_GUIDE.md`), its segmentation and landmark components are genuinely supervised rather than pseudo-labelled.

---

**Total: ~31 distinct model configurations.** At 3 folds × 3 seeds that is ~280 classification runs, plus the detection/segmentation and Tier-8 tracks.

---

## 7. The technique axis

Architecture is only one dimension. These are the levers that usually matter more on small data.

| # | Factor | Levels |
|---:|---|---|
| 1 | **Head / objective** | softmax CE · **CORAL ordinal** · regression + threshold · ordinal + ranking pairs |
| 2 | **Loss shaping** | plain CE · focal · label smoothing · class-balanced |
| 3 | **Sampling** | uniform · class-weighted · **`class_session_balanced_weight`** · group-balanced batches |
| 4 | **Input resolution** | 224 · 384 · 512 · **768** |
| 5 | **ROI** | full frame · SAM2 tread crop · CAM-guided crop |
| 6 | **Augmentation** | none · dataset policy · + photometric · + RandAugment · + dirt/occlusion |
| 7 | **Transfer source** | random · ImageNet-1k · ImageNet-21k · DINOv2 · CLIP |
| 8 | **Fine-tuning depth** | frozen · last block · **full** · LoRA |
| 9 | **Regularisation** | EMA · stochastic depth · weight-decay sweep · label smoothing |
| 10 | **Preprocessing** | raw · CLAHE · Scharr/Gabor structure channels · grayscale |
| 11 | **Ensembling** | single · seed-ensemble · architecture-ensemble · TTA |
| 12 | **Calibration** | none · temperature scaling · **conformal prediction** |

**Design discipline.** The full cross-product is astronomically large and would be meaningless. Use a staged design:

- **Stage A** — fixed recipe, sweep architectures (Tiers 0–6)
- **Stage B** — NB07 has locked the **top 3 by faithful, seed-confirmed
  TER_norm**: RegNetY-16GF, DenseNet-121 and ResNet-50. NB06 now runs
  **one-factor-at-a-time** over the implemented technique axis on fold 1
- **Stage C** — confirm the top 3 technique findings on the **other two** architectures. If a factor only helps one architecture, say so
- **Stage D** — detection + segmentation track
- **Stage E** — XAI over every trained model
- **Stage F** — shortcut stress tests
- **Stage G** — ensembles + calibration

**OFAT, not grid.** The original three-fold OFAT budget was 12 × 3 × 3 × 3 =
324 runs. Stage A subsequently showed that folds 0 and 2 are leak-flagged and
nearly saturated. The corrected NB06 therefore runs 12 factors × 3 selected
architectures × 3 seeds on fold 1 only: **at most 108 runs**, with structural
skips when a 224-native fixed-window model would make a resolution arm a no-op.
OFAT remains interpretable and remains unable to detect interactions; report
that limitation.

### Locked NB07 result (`2026-08-30-r3`)

| rank | architecture | TER_norm | BAR | seeds | valid maps |
|---:|---|---:|---:|---:|---:|
| 1 | RegNetY-16GF | 1.5785 | 0.0310 | 3 | 180/180 |
| 2 | DenseNet-121 | 1.5513 | 0.0455 | 3 | 178/180 |
| 3 | ResNet-50 | 1.5146 | 0.0512 | 3 | 180/180 |

All three public selection rows have `xai_status=ok`, `eligible=True`, and
`selected_top3=True`. Accuracy was excluded from this rule. NB06 downloads the
public selection and the raw 1,208-row evidence table and verifies both before
constructing any training configuration.

### Stage-B execution state — 2026-08-31

Public HF contains four completed ROI runs: all three DenseNet-121 seeds and
ResNet-50 seed 3. Their epoch files and both checkpoints are present. It also
contains failed epoch-0 run records for RegNet seeds 1 and 2; neither has a
checkpoint, so no trained epoch is being discarded. Earlier telemetry
isolated a host-RAM leak in the former ROI input path (up to 31.1 GB system
RAM) while GPU peaks remained about 4–5 GB per T4. Tyrelib v4 repairs only the
loader/memory lifecycle and preserves identical crop coordinates; the three
architectures, batch sizes, resolutions, optimiser settings and 60-epoch
budget remain the registered experiment. The two RegNet failures then isolated
a T4/cuDNN `channels_last` fault at only ~1.1 GB/card. RegNet now uses a
contiguous NCHW runtime with cuDNN autotuning off; no model or experimental
factor changed.

---

## 8. Training length and run budget

### How long each model trains

**Every run trains a fixed 60-epoch budget. There is no early stopping.**

```yaml
max_epochs:   60          # identical for every architecture, every run
early_stop:   NONE        # removed
monitor:      val_qwk     # selects the BEST CHECKPOINT — never stops the run
```

**Why 60.** Training sets are 2,552–3,454 images. Small CNNs converge by ~30 epochs; ViTs and fine-grained heads need more. 60 gives everything room without waste.

**Why no early stopping.** Three reasons, all of which apply specifically here:

1. **Run length must be predictable.** Work sharding across four accounts (`05 §3`) needs to know a run's cost *before* it starts. A run that might stop at 22 or might go to 60 makes every printed time estimate a guess.
2. **Fold 2's validation set is 104 images.** Validation QWK is genuinely noisy at that size — a patience counter fires on noise as often as on convergence, stopping good runs and keeping bad ones.
3. **It confounds the comparison.** If one architecture stops at 28 and another runs 60, they did not receive the same budget, and the architecture comparison is contaminated by a stopping rule rather than measuring architectures.

Convergence behaviour is still recorded — `best_epoch` in `epochs.csv` shows it. We observe it; we do not act on it mid-run.

### Measured runtime per run

Single T4, fp16, `channels_last`, averaged over the three folds, at the full 60-epoch budget (scaled to ~45-epoch equivalent for the cost table):

| Model | Res | img/s | s/epoch | **min/run** | ×9 runs (h) |
|---|---:|---:|---:|---:|---:|
| MobileNetV4-M | 384 | 221 | 16 | **11** | 1.6 |
| Swin-T | 224 | 188 | 19 | **12** | 1.9 |
| CoAtNet-0 | 224 | 175 | 20 | **13** | 2.0 |
| Swin-S | 224 | 112 | 31 | **21** | 3.1 |
| RegNetY-016 | 384 | 98 | 36 | **24** | 3.6 |
| ViT-S / DeiT III-S | 384 | 89 | 39 | **26** | 3.9 |
| ResNet-50 | 384 | 85 | 41 | **27** | 4.1 |
| EfficientNetV2-S | 384 | 81 | 43 | **29** | 4.3 |
| DINOv2 ViT-S | 384 | 77 | 46 | **30** | 4.6 |
| ResNeXt-50 | 384 | 72 | 48 | **32** | 4.8 |
| ConvNeXt-V2-T | 384 | 68 | 51 | **34** | 5.1 |
| DenseNet-121 | 384 | 64 | 55 | **37** | 5.5 |
| Bilinear CNN | 384 | 47 | 75 | **50** | 7.5 |
| ConvNeXt-V2-S / HBP / CSAB | 384 | 43 | 82 | **55** | 8.2 |
| VGG-16-BN / Coarse2Fine | 384 | 38 | 91 | **61** | 9.1 |
| CLIP / SigLIP B-16 | 384 | 34 | 102 | **69** | 10.3 |
| MaxViT-T / DINOv2 ViT-B | 384 | 32 | 108 | **72** | 10.8 |

> **These replace an earlier, ~2× optimistic estimate.** Anchor the cost table on two measured runs before Stage A and freeze it (`05 §3` — ⚠ Bug 7: measurements refine the *printed plan*, never the *assignment*).

### Budget

| Stage | Configs | Runs | GPU-h |
|---|---:|---:|---:|
| A · architectures — actual | 17 valid | **153 valid + 9 quarantined** | **163.4 recorded; 156.6 valid** |
| B · techniques (top 3, fold-1 OFAT) | ≤36 | **≤108** | ~50 estimated |
| C · confirmation | 12 | 108 | ~50 |
| D · detection + segmentation | 10 | 30 | ~30 |
| E · XAI | — | inference | ~15 |
| F · stress tests | 6 × top | — | ~20 |
| G · ensembles + calibration | — | reuse | ~5 |
| **H · Tier 8 pipeline + ablations** | 6 | 54 | **~25** |
| Original full-plan total | | ~800 runs | ~440 GPU-h |

The total row is retained as the original proposal budget, not the current
remaining commitment. NB07 is inference; NB06 is the next training stage and
is now capped at 108 runs.

**Feasibility:** 4 accounts × 30 GPU-h/week = **120 GPU-h/week** ⇒ **~4 weeks wall-clock**, assuming sharding works and nothing needs re-running.

**Add 30% contingency: plan for 5 weeks.** Something always needs re-running.

**Budget the cheap tiers first.** Tier 0–2 costs ~25 GPU-h and gives a complete, reportable architecture result inside week 1 — before anything risky starts.

---

## 9. What the paper's figures will be

Deciding this now changes what we log, while there is still time to log it.

| # | Figure | What it shows |
|---|---|---|
| **1** | **Accuracy vs TER scatter, Pareto front** | **The headline.** Do accurate models look at the tread? |
| 2 | Per-fold macro-F1 for every architecture, with the two trivial baselines drawn as horizontal lines | Most models sit near the baselines on some fold |
| 3 | Saliency panels — same image, every architecture family | Qualitative, immediately legible |
| 4 | Shortcut stress-test matrix — accuracy drop per intervention per model | Which models depend on what |
| 5 | Correlation: does TER predict cross-fold stability better than val accuracy? | **The falsifiable claim** |
| 6 | Technique OFAT — effect size ± CI per factor | What actually helps on small data |
| 7 | Faithfulness (insertion/deletion/ROAD) across XAI methods × architectures | Which explanations to trust |
| 8 | Pseudo-mask quality vs manual masks on the 50-image audit | Honesty about the pipeline |
| 9 | Video-frame temporal consistency | Real-world stability |
| 10 | Resolution × ROI interaction | Practical guidance for `final_v2` |

If Figure 5 comes out flat — TER does not predict generalisation — **that is publishable too**, and we report it. A pre-registered hypothesis that fails is a result; a hypothesis invented after seeing the data is not.

---

## 10. Rules of engagement

Non-negotiable, and they follow from `12_DATASET_FINAL_V1.md`:

1. **3 folds × 3 seeds, always.** Report mean ± spread. Never a single number.
2. **Both trivial baselines appear in every results table.**
3. **Ordinal metrics** (QWK, mean absolute class error) alongside macro-F1.
4. **Never call the labels worn/not-worn.** Mileage proxy.
5. **Derivatives never enter validation.** Use the supplied splits.
6. **Sample size is 12**, stated everywhere.
7. **Shuffled-label control** run before any result is believed.
8. **XAI method must match the architecture** (§`14_XAI_PROTOCOL.md`). Grad-CAM on a ViT is not the same operation as Grad-CAM on a CNN, and comparing them naively is invalid.
9. **Pre-register the hypothesis** (Figure 5) before running Stage E. Write it in `PROGRESS.md` with a date.
10. **Record everything.** We train once — see the telemetry schema in `05`.

---

## 11. Why this is a good capstone

**It is honest.** It does not pretend 12 tyres can support a production claim.

**It is novel.** Shortcut-aware, explanation-grounded benchmarking has been done in medical imaging; I found nothing equivalent in tyre inspection. The TER/BAR/SAR metric family and the named paint-stripe shortcut appear to be new to this domain.

**It is achievable.** No hardware, no annotation campaign, no workshop access. Compute across four accounts, and the infrastructure pattern is already written down.

**It scales with the data.** Every experiment re-runs unchanged on `final_v2`. The infrastructure built now is the infrastructure used later — so nothing here is throwaway.

**It scales to whoever is available.** The four subsystems in `07_ROADMAP.md` are
separable, so it works as a four-person split or as one person working through
them in order. Annotation is currently solo (~4 h). Compute sharding is the one
place where extra hands — or just extra free Kaggle accounts — buy real
wall-clock: 4 accounts turn a 15-week study into a 4-week one, with zero
coordination cost.

**And it produces a genuinely useful answer:** *which architectures and techniques learn tread structure rather than tyre identity, and how would you tell?* That question matters to anyone building tyre inspection from a small dataset — which is everyone in this field.
