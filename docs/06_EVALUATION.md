# 06 — Evaluation Protocol

> Decide these metrics **now**, before you have results. Choosing metrics after seeing results is how honest people accidentally produce dishonest reports.

---

## 1. The acceptance principle

> The primary success criterion is **not** the smallest average error under ideal conditions. It is a system that retains **high sensitivity for visibly unsafe wear**, **recognises when its geometric estimate is unreliable**, and **generalises to tyres not represented in training**.

Every metric below serves one of those three.

---

## 2. Establish the noise floor first

**You cannot beat your own reference instrument.** Before evaluating any model (`03_DATA.md §3.2`):

| Measurement | Expected | Purpose |
|---|---|---|
| Gauge test–retest MAE | 0.10–0.15 mm | Absolute floor |
| Inter-operator MAE | 0.15–0.25 mm | Human-label noise |
| Inter-annotator κ (wear pattern) | > 0.70 | Label reliability |

Report all three. A model at 0.35 mm against a 0.15 mm reference is doing well — but that is only visible if you measured the floor. One afternoon; highest credibility-per-hour in the project.

---

## 3. Per-task metrics

### Segmentation

| Metric | Target | Why |
|---|---|---|
| Dice, mean IoU | > 0.85 (tread/shoulders) | Standard |
| Per-class IoU | report **all**, including `sipe`, `twi_bar` | Rare classes are where it fails |
| **Boundary F-score** | > 0.70 on shoulders and grooves | **A mask can have good IoU and still place a shoulder edge too inaccurately for angle estimation** |
| **clDice / connectivity** | report for `sipe`, `crack` | A crack broken into fragments is a serious failure ordinary Dice barely penalises |

Boundary F-score and clDice are not optional extras — they are the metrics that correspond to "recognise every detail."

### Ordinal wear severity

| Metric | Target |
|---|---|
| Macro-F1 | > 0.75 |
| Balanced accuracy | > 0.75 |
| **Ordinal MAE** (mean |ŷ − y| in class units) | < 0.35 |
| **Critical-wear sensitivity** (class 3 recall) | **> 0.95** |

That last row is the safety metric. **Missing a critically worn tyre is the failure that matters.** Report it separately and prominently; optimise the operating point for it even at some precision cost.

### Wear pattern (multi-label)

| Metric | Target |
|---|---|
| Per-label precision / recall / F1 | report all 9 |
| Macro F1 | > 0.70 |
| mAP | > 0.70 |
| Hamming loss | < 0.12 |

Expect `cupping_or_scalloping` and `flat_spot` to be worst. Say so rather than hiding them in a macro average.

### Damage

| Metric | Target |
|---|---|
| Defect-level sensitivity | > 0.85 |
| Dice / IoU on defect masks | > 0.60 |
| Instance mAP | > 0.50 |
| **False positives per healthy tyre** | **< 0.5** |

The FP rate decides whether anyone would tolerate the system. A detector that cries wolf on every clean tyre is useless regardless of its recall.

### PatchCore / anomaly

Image AUROC > 0.90 · pixel AUROC / AUPRO · **false positives on healthy tyres** (again, the deciding number).

### Landmarks

Normalised mean error · PCK@0.05 · **visibility-prediction accuracy** (did it correctly refuse to place a hidden landmark?).

### Alignment

| Metric | Target | Note |
|---|---|---|
| **Camber MAE** | < 0.40° | The observable one |
| Camber RMSE, bias | report | Bias reveals calibration error |
| **Toe MAE** | < 0.60° | Honest — weakly observable from one wheel |
| **Misalignment screening AUROC** | **> 0.90** | **This is the primary alignment claim** |
| Recall @ 95% precision | > 0.80 | False alarms destroy trust |
| Repeatability (σ across clips, same tyre) | < 0.25° | Precision, independent of accuracy |
| Sign accuracy (toe-in vs toe-out) | > 85% | Direction matters diagnostically |

> **Frame toe as binary out-of-spec detection, with continuous MAE as a secondary result.** A workshop rack reaches ±0.05°; we will not. What we offer is screening every wheel with no markers, no stereo and no setup. That is a different, defensible operating point — see `09_RELATED_WORK.md §4`.

### Metric depth

Evaluated in millimetres **only** on samples with physical depth labels. MAE, RMSE, **Bland–Altman bias and limits of agreement** (the correct method-comparison plot against a reference instrument — use this as the headline figure, not a scatter plot), and % within ±0.5 mm.

**Report the fraction of clips where a TWI anchor was available.** Scale claims without it are weaker and must be labelled as such.

### Uncertainty

| Metric | Target |
|---|---|
| Conformal coverage @ 90% | 88–92% empirical |
| Mean interval width | report per output |
| Expected Calibration Error | < 0.05 |
| **Refusal rate** (`UNABLE_TO_MEASURE`) | report; too high is useless, too low is dishonest |
| **Coverage under distribution shift** (unseen brand) | report honestly — **it will degrade** |

Conformal guarantees hold under exchangeability, which brand shift violates. Measuring the degradation is a much stronger result than pretending it doesn't happen.

### Video

Tracking inlier rate · registration reprojection error · **observed circumference coverage %** · fraction of frames accepted by the quality gate.

---

## 4. Stratified reporting — mandatory

**Overall accuracy alone is not sufficient.** Every headline result must also appear broken down by:

```
tyre brand · tread family · tyre size · wear severity
camera / calibration ID · illumination condition
dirt / wet state · approach angle · inflation · load
```

The brand row matters most: [Vivekanandan & Rajeswari 2026](https://doi.org/10.1016/j.measurement.2026.121509) measured an 88.2% → 92.4% gap on unseen brands. If your unseen-brand column is much worse than your overall column, that is the finding — report it.

---

## 5. Robustness sweep

| Axis | Conditions |
|---|---|
| Illumination | Indoor, overcast, direct sun, dusk, night with rig lights |
| Surface | Dry, damp, wet, dusty, muddy |
| Brand | ≥ 6; **2 held out entirely** |
| Size | 13″–17″; one size class held out |
| Tread pattern | Symmetric, asymmetric, directional |
| Inflation | 180 / 200 / 220 / 240 kPa |
| Load | Empty / 2 occupants / loaded |
| Temperature | Cold vs after 20 min driving |
| Camera pose | Nominal ± tolerance, to test calibration sensitivity |
| Rotation | Stationary vs slowly rotating |

**Wet operation will degrade** (specular reflection, changed apparent texture). Measure exactly how much, report it as a named limitation, and cite cross-polarisation as the mitigation. A clearly characterised failure mode is a strength; an uncharacterised one is a hole.

---

## 6. Ablations

Fifteen listed in `04_MODEL.md §9`. Report as one table with Δ from the full model, **3 seeds each, mean ± std**.

| Priority | Ablation | Why it matters most |
|---|---|---|
| **1** | **Flat light vs photometric stereo** | The project's most distinctive engineering decision. Cheap to run, likely large effect |
| **2** | Classical vs direct CNN vs hybrid alignment | Justifies the whole alignment architecture; the CNN arm should visibly fail on unseen mounts |
| **3** | Single frame vs registered video | Justifies the video complexity |
| **4** | With / without clDice | Directly tests the "every detail" claim on connectivity |
| **5** | RGB-only vs metric-sensor supervision | Bounds what RGB alone can deliver |
| 6–15 | Remainder | Complete the story |

Single-seed ablations on a small dataset are noise, and an examiner will say so.

---

## 7. Baselines

Be generous to the baselines. A weak baseline section is the fastest route to a hard question in the viva.

| Baseline | Expected | Demonstrates |
|---|---|---|
| Gauge test–retest | 0.12 mm | The floor |
| **Human technician, visual only** | 0.8–1.5 mm; pattern κ ~0.5 | You should beat this — and it is the honest deployment comparison |
| Classical-only pipeline (CLAHE + Canny + Gabor + RANSAC) | — | Value of learning |
| Petrovic-style Mask R-CNN + HOG | mAP ~0.6 | Direct comparison to [1] |
| MobileNetV2 binary classifier | — | Direct comparison to [6] |
| Single-frame CNN, whole image | — | Value of segmentation + video |
| Direct CNN → angles | — | **Should fail on unseen camera pose. That failure is the result** |
| Depth Anything V2 fine-tuned | poor | **Documented negative result** — generic monocular depth is not metrology |
| Analytic geometry alone (no residual MLP) | — | Value of the residual head |

---

## 8. Statistical rigour

- **Bootstrap CIs on every headline number**, resampled at the **tyre** level, 10,000 draws
- **Paired significance tests** for ablations — paired bootstrap or Wilcoxon signed-rank on per-tyre errors
- **Holm–Bonferroni** correction across the 15-ablation grid
- **Effect sizes**, not only p-values — a significant 0.01 mm improvement is not interesting
- **Report negative results.** If clDice doesn't help, that is a finding about topology losses on this data

---

## 9. Failure analysis — a required section

Aggregates hide everything. Take the **worst 50 predictions** and categorise them by hand:

```
worst-case analysis, n = 50
├─ 14  heavy dirt / brake dust occluding grooves
├─ 11  specular glare (wet or new glossy tread)
├─  8  tread width exceeded FOV (wide tyre)
├─  7  insufficient mm/px — sipes unresolvable
├─  5  unusual tread pattern (directional, unseen family)
├─  3  very worn (<1.5 mm) — no groove structure left
└─  2  motion blur during rotation
```

For each, state what it implies for the next version. This section is often what distinguishes a good capstone from an excellent one — it shows you understand your system rather than just having trained it.

Pair it with a **qualitative figure**: 3 successes and 3 failures, showing input → segmentation → wear heatmap → prediction vs ground truth.

---

## 10. Evaluation timeline

| Milestone | What |
|---|---|
| Pilot complete | Noise floor established; annotation κ measured |
| Segmentation done | Boundary F-score + clDice verified; taxonomy frozen |
| Wear heads done | First val numbers — **sanity only, do not optimise against test** |
| Alignment done | Analytic-only baseline reported **before** the residual MLP |
| Pre-Review-3 | Robustness sweep, ablations (3 seeds), baselines |
| **Final** | **Gold rack test set opened — once** |
| Final | Failure analysis, qualitative figures, stratified tables |

### The gold-set rule

Evaluate on `gold_rack` exactly once, at the end, and report whatever it says. If you peek and then tune, it stops being a test set and every number in the report becomes optimistic. Separate folder. Commit now, while it costs nothing.
