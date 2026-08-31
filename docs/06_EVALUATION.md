# 06 — Evaluation Protocol

> Decide these metrics **now**, before there are results. Choosing metrics after seeing results is how honest people accidentally produce dishonest reports.
>
> Companion to `13_EXPERIMENT_PLAN.md` and `14_XAI_PROTOCOL.md`.

---

## 1. The evaluation principle for this study

> **Accuracy is not identifiable on 12 tyres. Evidence location is.**

Two trivial baselines swing from 0.12 to 0.98 macro-F1 across the three folds. Any accuracy difference between two architectures will be dominated by that variance. So accuracy is reported — carefully, with spread — but it is **not the primary axis**. The primary axis is where the model's evidence comes from.

Every results table has three blocks:

```
[ ACCURACY ]        mean ± spread over 3 folds × 3 seeds, + trivial baselines
[ EVIDENCE ]        TER_norm, BAR, SAR, faithfulness
[ ROBUSTNESS ]      shortcut stress-test deltas
```

A model that wins block 1 and loses blocks 2 and 3 has not won.

---

## 2. Mandatory baselines — on every figure

| Baseline | fold 0 | fold 1 | fold 2 | **mean** |
|---|---:|---:|---:|---:|
| **Frame occupancy only** — `tyre_frac`, `tread_frac`, ratio | 0.181 | 0.455 | **0.968** | **0.535** ← **highest** |
| **Colour only** — 10 stats from a 64×64 thumbnail | **0.952** | 0.399 | 0.123 | **0.491** |
| **Structure only** — 9 contrast-normalised tread-band stats | 0.354 | 0.119 | **0.976** | **0.483** |
| **Annotation side-channel** — `marking→low, damage→high, else mid` | **0.978** | 0.159 | 0.108 | 0.415 |
| Majority class (accuracy) | 0.360 | 0.484 | 0.423 | 0.423 |
| HOG + SVM | — | — | — | *to run* |
| Random-init CNN | — | — | — | *to run* |

Reproduce: `python scripts/dataset_shortcut_probe.py --root "<FINAL>"` (first three) and `annotations/README.md §6` (last two).

> ### **The floor is 0.535, not 0.491.**
>
> The strongest trivial baseline is now **how much of the frame the tyre fills** — nothing else. The tyre occupies 72% of the frame in `low` images, 62% in `mid`, 61% in `high`, so framing alone is a class cue. It beats both colour and texture.
>
> Every one of the four probes is near-perfect on **one** fold and collapses on the others. Four different shortcuts, four different folds. That is the fingerprint of memorising tyres, and it is why single-fold numbers from this dataset are meaningless.

### What Stage A did against that floor (2026-08-27, 36 runs)

The probes predicted this and the real models confirmed it. Full table:
`docs/18_STAGE_A_RESULTS.md`.

| fold | leak-flagged | best model macro-F1 | **final macro-F1 (fixed budget)** | trivial floor |
|---|---|---:|---:|---:|
| 0 | **yes** | 1.000 | 1.000 | 0.181 |
| 1 | no | 0.736 | **0.499** | **0.455** |
| 2 | **yes** | 1.000 | 1.000 | 0.968 |

**Report `final_val_*`, not `best_val_*`.** `best` is the epoch with the highest
validation QWK — chosen by looking at a validation fold of four tyres, then
reported as if it were held out. That is circular, and on fold 1 it is worth
+0.24 macro-F1 of pure selection. The fixed-budget number was chosen by nobody
and is the only one comparable with a baseline.

Quote them side by side. The gap between them is itself a result: it measures
how much of the score is selection.

**Two consequences that change the experiment design:**

1. **A tyre-region crop is a control, not an optimisation.** It removes the frame-occupancy shortcut outright. Run it early (`04 §9` factor 5).
2. **Stress tests must be symmetric.** `marking` appears on **only** `low` images (67) and `damage` on **only** `high` (63) — so mask the paint stripes for `low` *and* the damage regions for `high`. Testing one without the other measures half the problem.

Draw all four baselines as horizontal lines on every accuracy figure.

---

## 3. Classification metrics

### Primary

| Metric | Why |
|---|---|
| **Macro-F1**, mean ± spread over 3 folds | Class-balanced; spread is as important as mean |
| **Quadratic Weighted Kappa** | The classes are **ordered**. QWK penalises `low↔high` more than `low↔mid` |
| **Mean absolute class error** | Interpretable ordinal distance |
| **Per-class recall** | `mid` has only 3 sessions — it will be worst, and hiding that in a macro average is dishonest |
| **Confusion matrix, per fold** | Where the ordinal structure breaks |

### Reporting rules

1. **Always all three folds.** `0.71 ± 0.24 (0.48 / 0.72 / 0.94)` — never `0.71`
2. **Spread is a headline number**, not a footnote. A model with 0.65 ± 0.05 may be more useful than one with 0.71 ± 0.30
3. **Session-level breakdown** — per-session accuracy exposes single-tyre memorisation
4. **Bootstrap CIs at the tyre level**, never the image level. 418 images are 12 tyres
5. **Never report a single-fold number anywhere**, including in conversation

### Calibration

ECE · MCE · NLL · Brier · reliability diagram. Plus, after Stage G:

| Conformal metric | Target |
|---|---|
| Empirical coverage @ 90% | 88–92% on the calibration split |
| Mean prediction-set size | smaller is better at equal coverage |
| **Abstention rate** | the `uncertain / request another frame` outcome the dataset's guidance asks for |

---

## 4. Evidence metrics — the primary axis

Full definitions in `14_XAI_PROTOCOL.md §3`.

| Metric | Target | Note |
|---|---|---|
| **TER_norm** — area-normalised Tread Evidence Ratio | **> 1.0** | `1.0` = attends in proportion to area, i.e. no preference. **This is the paper number**; raw TER is supporting detail |
| **BAR** — Background Attribution Ratio | low | evidence outside the tyre entirely |
| **SAR** — Stripe Attribution Ratio | low | factory paint/lettering — the named shortcut |
| **DAR** — Dirt Attribution Ratio | low | |
| **Insertion AUC / Deletion AUC / ROAD** | high / low / high | is the explanation faithful at all? |
| **Pointing game** | high | plausibility — report alongside faithfulness, never instead |
| **Cross-seed saliency IoU** | high | unstable explanations ⇒ untrustworthy model |

**Prerequisites before any of these are reported:**

- [x] Selected attribution method passes the **weight-randomisation sanity check**
- [x] Method selected per architecture by finite **insertion-minus-deletion** among sanity survivors, and the choice stated
- [x] **Mask quality reported** — NBT1 clean IoU 0.9780, propagated IoU 0.9747, ratio 0.9966, plus the tested-mask fingerprint

An evidence metric built on an unvalidated mask is unfalsifiable.

### Completed Stage-B evidence gate

NB07 revision `2026-08-30-r3` completed on 2026-08-30 and the three public
gate files were independently re-read. The locked architectures are:

| Architecture | Three-seed TER_norm | BAR | Valid maps |
|---|---:|---:|---:|
| RegNetY-16GF (`regnety016`) | 1.5785 | 0.0310 | 180/180 |
| DenseNet-121 (`densenet121`) | 1.5513 | 0.0455 | 178/180 |
| ResNet-50 (`resnet50`) | 1.5146 | 0.0512 | 180/180 |

All three have `xai_status=ok` and evidence from seeds 1–3. Accuracy was not
part of the selection rule. The public aggregate contains 1,208 image-level
evidence rows, 35 faithfulness rows, and 18 screened architecture rows. NB06
is therefore unblocked and must use this exact set; there is no fallback list.

---

## 5. Shortcut stress tests

Report as a matrix — **models × interventions**, cells = Δ macro-F1 with CI. This is one of the most informative tables the study will produce.

| Test | Reading |
|---|---|
| Background replacement | drop ⇒ background dependence |
| Background blanking | drop ⇒ background dependence |
| **Stripe masking** (`low` class) | drop in `low` recall ⇒ factory-marking dependence |
| Tread-only crop | rise ⇒ context was harmful |
| **Grayscale** | drop ⇒ colour dependence — and colour is not a wear cue |
| Dirt inpainting | drop ⇒ dirt dependence |
| Session holdout | drop ⇒ tyre-identity memorisation |
| **Shuffled labels** | **above chance ⇒ pipeline leak. Stop and fix before anything else** |

The shuffled-label control is cheap, mandatory, and run **before** any result is believed.

---

## 6. Detection and segmentation metrics

Against SAM2 pseudo-labels, with the pseudo-label quality stated first.

| Task | Metrics |
|---|---|
| **Pseudo-mask quality** (50-image audit) | IoU vs manual, Dice, failure rate, per-class breakdown |
| WSOL (CAM → box) | MaxBoxAcc, box IoU, pointing game |
| Detection (YOLO26, RT-DETR) | mAP50, mAP50-95 vs pseudo-boxes, inference latency |
| Segmentation (SegFormer, U-Net, DeepLabV3+) | mIoU, Dice, boundary F-score vs pseudo-masks |
| **Downstream value** | Δ classification macro-F1 when the ROI crop is used |

That last row is the one that matters. Detection and segmentation exist here to serve the classification and XAI tracks; their standalone scores are secondary to whether they *help*.

---

## 7. Video evaluation — qualitative and temporal

No labels, so no accuracy. What can be measured honestly:

| Metric | Definition | Reading |
|---|---|---|
| **Temporal prediction consistency** | fraction of consecutive frames with the same predicted class, per clip | low ⇒ shortcut reliance |
| **Temporal logit variance** | variance of class probabilities across frames of one tyre | high ⇒ instability |
| **Temporal saliency IoU** | saliency agreement between adjacent frames | low ⇒ the model is not tracking a stable feature |
| Qualitative panels | saliency overlaid on video frames | for the report |

A model reading tread structure should give a stable answer across consecutive frames of the same tyre. One reading dirt, glare or background will flicker. **This needs no labels and is a genuinely informative generalisation test.**

---

## 8. Statistical rigour

- **Bootstrap CIs on every headline number**, resampled at the **tyre** level, 10,000 draws
- **Paired tests** for technique comparisons — paired bootstrap or Wilcoxon signed-rank on per-fold, per-seed results
- **Holm–Bonferroni** across the OFAT grid (12 factors)
- **Effect sizes**, not only p-values. A significant 0.005 macro-F1 gain is not interesting
- **Report negative results.** If CORAL doesn't beat softmax, that is a finding about ordinal heads on small data
- **H1–H3 registered** (`14 §10`) at `2026-08-30T10:06:21Z`, before the first XAI evidence row

---

## 9. Failure analysis

Aggregates hide everything. Take the **worst 50 predictions** across the top models and categorise by hand:

```
├─ session confusion (which tyre was mistaken for which)
├─ heavy dirt / deposits
├─ specular glare
├─ framing — shoulders cropped off
├─ unusual tread pattern
└─ genuinely ambiguous wear level
```

For each, state what it implies for `final_v2`. Pair with a qualitative figure: 3 successes and 3 failures, showing image → saliency → prediction vs truth.

---

## 10. Anti-patterns

Things that would invalidate the study. Each has bitten someone before.

| Anti-pattern | Why it invalidates |
|---|---|
| Reporting fold 0 alone | A ten-number colour model gets 0.952 there |
| Derivatives in validation | 10 near-copies of a validation image were in training |
| Image-level bootstrap | 418 images are 12 tyres — CIs would be ~3× too tight |
| Tuning on the fold you report | The fold stops being held out |
| One seed | Cross-seed variance is large at this data scale |
| Grad-CAM everywhere, unqualified | It is a different operation on ViTs (`14 §1`) |
| TER without area normalisation | A model attending to everything scores well |
| Claiming "wear detection" | Labels are a mileage proxy |
| Sample size 4,598 | It is 12 |
| Skipping the shuffled-label control | Silent leakage looks like success |
