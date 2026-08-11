# 06 — Evaluation Protocol

> Decide these metrics **now**, before you have results. Choosing metrics after seeing results is how honest people accidentally produce dishonest papers.

---

## 1. Primary metrics

### Tread depth

| Metric | Target | Why |
|---|---|---|
| MAE (mm) | **< 0.35** | Gauge repeatability is ~0.10–0.15 mm. Under 0.35 is genuinely useful. |
| RMSE (mm) | < 0.50 | Penalises the tail |
| % within ±0.5 mm | > 85% | The number a workshop actually cares about |
| Pearson r vs gauge | > 0.95 | |
| **Bland–Altman bias & LoA** | bias < 0.1 mm | The correct plot for method-comparison against a reference instrument. Use this, not a scatter plot, as your headline figure. |
| Legal-threshold accuracy (1.6 mm) | > 95% | The safety-critical decision |
| Replace-threshold accuracy (3.0 mm) | > 92% | The commercial decision |

### Alignment

| Metric | Target | Why |
|---|---|---|
| Camber MAE (°) | **< 0.30** | Screening-grade. Workshop spec is ±0.5°, so this is useful. |
| Toe MAE (°) | **< 0.25** | Workshop spec is ±0.1°. **We will not reach that — say so.** |
| Misalignment detection AUC | > 0.90 | The actual product claim |
| Recall @ 95% precision | > 0.80 | False alarms destroy workshop trust |
| Sign accuracy (toe-in vs toe-out) | > 90% | Direction matters for diagnosis |

> **Reposition the toe claim explicitly.** GRIP does not measure toe to spec. GRIP *screens* for toe outside spec. Frame the primary alignment metric as **binary detection of out-of-spec**, with the continuous MAE as a secondary result. This is honest, it is what the system is actually good for, and it converts a weakness into a well-defined contribution.

### Wear pattern

| Metric | Target |
|---|---|
| Macro F1 (8 classes, multi-label) | > 0.75 |
| Per-class F1 | report all; expect `cupping` and `flat_spot` to be worst |
| Hamming loss | < 0.12 |
| Cohen's κ vs human annotator | > 0.70 |

### Uncertainty

| Metric | Target |
|---|---|
| Conformal coverage @ 90% | 88–92% (empirical) |
| Mean interval width (depth) | < 0.7 mm |
| Expected Calibration Error | < 0.05 |
| Coverage under distribution shift (unseen brand) | report honestly — **it will drop** |

That last row matters. Conformal guarantees hold under exchangeability, which distribution shift violates. Measuring and reporting the degradation is a much stronger result than pretending it doesn't happen.

---

## 2. The noise floor — establish this first

**You cannot beat your own reference instrument.** Before any model evaluation:

1. Measure 20 tyres with the digital gauge. Wait a day. Measure again, same chalk-marked positions.
2. Compute test–retest MAE. Expect **0.10–0.15 mm**.
3. Have a second person measure the same 20. Compute inter-operator MAE. Expect **0.15–0.25 mm**.

Report both. Every model number is then interpretable relative to a real floor. A model at 0.30 mm MAE against a reference with 0.15 mm noise is doing very well — but that's only visible if you measured the floor.

This costs one afternoon and it is the highest-credibility-per-hour work in the entire project.

---

## 3. Robustness protocol

Run every axis. Report a table. Reviewers and examiners will ask.

| Axis | Conditions to test |
|---|---|
| **Speed** | 4, 6, 8, 10, 15, 20 km/h |
| **Lighting** | Indoor, overcast, direct sun, dusk, night |
| **Surface** | Dry, damp, wet, dusty, muddy |
| **Tyre brand** | ≥ 6 brands; hold out 2 entirely |
| **Tyre size** | 13"–18"; hold out one size class |
| **Tread pattern** | Symmetric, asymmetric, directional |
| **Load** | Empty, 2 occupants, fully loaded |
| **Inflation** | 180, 200, 220, 240 kPa |
| **Temperature** | Cold tyre vs after 20 min driving |
| **Plate condition** | Clean, dusty, scratched, partially wet |

**Wet operation will fail** (water frustrates TIR — see `01_CONCEPT.md §3`). Measure exactly how badly, report it as a named limitation, and propose the air-knife mitigation. A clearly characterised failure mode is a strength; an uncharacterised one is a hole.

---

## 4. Ablation table (from `04_MODEL.md §7`)

Report as one table with Δ from full model:

| # | Variant | Depth MAE | Camber MAE | Toe AUC | Δ |
|---|---|---|---|---|---|
| — | **Full GRIP** | — | — | — | — |
| 1 | λ = 0 (no consistency) | | | | |
| 2 | No synthetic pretrain | | | | |
| 3 | End-to-end (no analytic prior) | | | | |
| 4 | **No FTIR channel** | | | | |
| 5 | No laser distillation | | | | |
| 6 | No TWI anchor | | | | |
| 7 | Regression instead of ranking | | | | |
| 8 | Single frame (no unrolled map) | | | | |

Run each with **3 seeds** and report mean ± std. Single-seed ablations on a small dataset are noise, and a sharp examiner will say so.

---

## 5. Baselines (from `04_MODEL.md §8`)

| Baseline | Expected outcome | What it demonstrates |
|---|---|---|
| Gauge test–retest | 0.12 mm | The floor |
| **Published image-based SOTA** (QBurst monocular CNN) | **±1.5 mm on 90% of images** | **The number to beat. Hitting 0.35 mm MAE is a ~4× improvement — this is your headline comparison.** |
| Industrial 2D laser profiler | 0.01–0.05 mm | The ceiling. You are closing the camera↔laser gap, not beating lasers. |
| Human technician (visual) | 0.8–1.5 mm | You should beat this comfortably |
| Depth Anything V2 (fine-tuned) | 0.9–1.4 mm | Generic depth ≠ metrology |
| Single-frame CNN regression | 0.6–0.9 mm | Value of the pipeline |
| Classical ellipse-fit alignment | camber 0.5°, toe 0.8° | Value of the learned residual |
| Kaggle ViT binary classifier | N/A (binary only) | Places you in existing literature |

---

## 6. Failure analysis — a required section

Do not just report aggregates. Take the **worst 50 predictions** and categorise them by hand:

```
worst-case analysis, n = 50
├─ 18  heavy dirt/mud occluding grooves
├─ 11  tyre width exceeded FOV (SUV, 235-section)
├─  9  partial pass — wheel clipped the plate edge
├─  6  unusual tread pattern (directional, unseen)
├─  4  very worn (< 1.5 mm) — no groove structure left to measure
└─  2  motion blur (speed > 18 km/h)
```

Then say what each implies for the next version. This section is often what distinguishes a good capstone from an excellent one — it shows you understand your system rather than just having trained it.

Pair it with a **qualitative figure**: 3 successes and 3 failures, unrolled map + prediction + ground truth side by side.

---

## 7. Statistical rigour

- **Confidence intervals on every headline number.** Bootstrap (10,000 resamples) at the *tyre* level, not the frame level.
- **Paired significance tests** for ablations: paired bootstrap or Wilcoxon signed-rank on per-tyre errors. Report p-values.
- **Correct for multiple comparisons** (Holm–Bonferroni) across the ablation grid.
- **Effect sizes**, not just p-values. A statistically significant 0.01 mm improvement is not interesting.
- **Report negative results.** If the consistency loss doesn't help, that is a finding about physics-informed losses on small datasets, and it is publishable.

---

## 8. Timeline of evaluation

| When | What |
|---|---|
| Week 4 | Noise floor established (gauge test–retest) |
| Week 8 | First end-to-end numbers on jig data — sanity only, don't optimise against them |
| Week 12 | Val-set numbers, model selection |
| Week 16 | Robustness sweep |
| Week 18 | Ablations, 3 seeds |
| Week 20 | Baselines |
| **Week 21** | **Gold rack test set — opened once, never again** |
| Week 22 | Failure analysis, qualitative figures |

**The gold set rule:** you evaluate on it exactly once, at the end, and you report whatever it says. If you peek and then tune, it stops being a test set and every number in your paper becomes optimistic. Put it in a separate folder. Write `DO_NOT_OPEN_UNTIL_WEEK_21` on it. This is easy to say and hard to do — commit now, while it costs nothing.
