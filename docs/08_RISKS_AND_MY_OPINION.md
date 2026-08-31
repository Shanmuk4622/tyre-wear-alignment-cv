# 08 — Risks and My Honest Opinion

> ### ⚑ Update, 2026-08-26 — the approach changed, and I think it changed correctly
>
> Sections 1–9 below were written for the **engineered-system** plan: build a rig, collect 300 tyres, chain SegFormer → ConvNeXt → HRNet → PatchCore. That plan is superseded by `13_EXPERIMENT_PLAN.md` — a broad comparative study with XAI as the measuring instrument.
>
> **What I think of the change: it is the right call, and it is right for a reason worth stating.**
>
> The old plan's fatal problem was not ambition. It was that it required labels that do not exist. Masks, gauge readings, alignment angles — none of them are in `final_v1` and none were arriving soon. A plan that cannot start is worse than a smaller plan that can.
>
> The new plan turns the dataset's biggest weakness into the research question. We *measured* that trivial baselines swing 0.12→0.98 across folds. That kills accuracy benchmarking — but it makes "which model actually looks at the tread?" a question worth asking, and one this data **can** answer.
>
> **Three things I would still push back on:**
>
> **1. Breadth is only a virtue with discipline.** "Train many models" becomes noise the moment someone reports a single fold. The 3 folds × 3 seeds rule and the trivial-baseline lines on every figure are what separate this from a leaderboard of random numbers. If those slip, the whole study is worthless. They are cheap — hold them.
>
> **2. XAI is the instrument, so its own validity matters more than usual.** Grad-CAM on a ViT is not the same operation as Grad-CAM on a CNN; there is a 2026 paper devoted to that ambiguity. Comparing them naively would have corrupted the headline result while looking perfectly reasonable. `14 §1` is the most important page in the new plan.
>
> **3. Alignment is the harder half, not the easier one.** Deferring it is right. But the reason is that it needs a calibrated vertical and a known travel direction, neither of which exists here — not that it is simple. Plan for it as the hard part when you return to it.
>
> **The risk register in §7 has been updated for the new plan. §§1–6 and 8–9 are retained as the record of the earlier reasoning** — some of it (the resolution arithmetic, the shortcut warnings, the data-collection priorities) still applies.

---

Written after reading the Review-1 report and the model-stack specification, and after a research pass on the underlying vision methods.

---

## 1. What I think of the project as specified

**The Review-1 report is good.** Better than most capstone proposals I'd expect to see. Specifically right:

- **Refusing the single black box.** Splitting wear (appearance) from alignment (geometry) is the correct call, and it's the call most teams get wrong.
- **The scope discipline.** Explicitly excluding total axle toe, thrust angle, caster and certified alignment — and stating that RGB cannot give defensible millimetres — is the kind of honesty that makes examiners trust everything else you say.
- **`UNABLE_TO_MEASURE` as a first-class output.** Rare and correct for a safety task.
- **Tyre-level splits.** You already identified the leak that sinks most student vision projects.
- **The analytic-plus-residual alignment design.** Right, and defensible under questioning in a way end-to-end regression never is.

So my job here is not to fix the framing. It's to point at the two or three things that will actually determine whether this works.

---

## 2. The single biggest opportunity you're currently missing

**Illumination.**

The Review-1 spec treats lighting as an acquisition detail — locked exposure, flat-field correction, optional cross-polarisation. All correct, all insufficient.

Here is the problem stated plainly:

> Tread grooves, sipes, rib rounding and cracks are **geometric** features on a surface with **4–8% albedo and essentially no colour information**. Under flat frontal illumination they are nearly invisible. Under directional illumination they cast shadows proportional to their depth.

The multi-illumination inspection literature says this outright: with a static ring light it is **impossible to distinguish low reflectance caused by a stain from a genuine shadowed cavity**. That is your exact failure mode — dirt masquerading as a crack, a shadow masquerading as a groove.

**Four LEDs and a GPIO strobe (~₹2,000) turn your camera into a surface-geometry sensor.** Photometric stereo gives you a per-pixel normal map, which is *precisely* the "recognise every detail" signal you're after — and it's proven in industrial inspection on specular metal parts, which is a harder surface than rubber.

If I could change one thing about the project, it would be this. It's cheap, it's fast to build, it's a clean ablation, and it addresses the core problem statement directly rather than obliquely.

Full argument: `10_VISION_TECHNIQUES.md §2`.

---

## 3. What will work

| | Confidence | Why |
|---|---|---|
| Tread / shoulder segmentation | **High** | Precedent exists ([1] mAP 0.61 with 247 images and a weaker model). SegFormer-B2 with 300 tyres will beat that comfortably |
| Wear-pattern multi-label | **High** | Easy vision task on a good crop. Working by P4 week 3 |
| Ordinal severity | **High** | CORAL + ranking pairs is well-suited to small data |
| Damage detection (annotated types) | **Medium-high** | Standard segmentation; limited by how many examples you gather |
| **Camber** | **Medium-high** | Produces visible lateral tilt and shoulder asymmetry. This is the observable angle |
| Photometric stereo helping | **Medium-high** | Strong industrial precedent; rubber is a good case. Rubber's non-Lambertian lobe is the only real risk |
| TWI-anchored relative depth | **Medium** | Huber got 0.57 mm with this principle — but they had a dedicated groove-cross-section view. Yours is more oblique |

---

## 4. What probably won't work, and what to do about it

### Toe. Say it now rather than discovering it in month five.

Toe is a small yaw of the wheel plane. From a monocular front view it is confounded with camera yaw, vehicle approach angle, steering input and tyre deformation. Meanwhile the workshop specification is **±0.1°**.

Even the best recent work needs help: Furferi reached 0.025° **with stereo cameras and NIR markers glued to the tyre**; Shi reached <0.1° **with an RGB-D sensor**, and explicitly called it a controlled feasibility study with no validation against a commercial aligner.

**My advice: stop trying to measure toe. Screen for it.** Make binary out-of-spec detection (AUROC) the primary alignment metric and continuous MAE a secondary result. This is honest, achievable, and it is the more useful product anyway — a workshop doesn't need you to replace their rack; they need to know which cars to put on it.

If the toe head isn't working by the P4 midpoint, **cut it**. "We deliberately scoped to camber because individual toe is not observable at useful precision from a single monocular front view, and here is the analysis showing why" is a *better* result than a toe number nobody believes.

### Sipes, at full tread width.

The arithmetic is unforgiving: a 0.3 mm sipe at 3 px needs ≤0.1 mm/px, which is 2,500 px across a 250 mm tread. `final_v1` sits at roughly 0.21–0.24 mm/px, so main grooves and rib edges resolve well and only the wider sipes do.

**Either buy an 8–12 MP sensor, or accept that sipes are resolved only in a cropped centre region and say so in the report.** Both are fine. Silently claiming sipe detection at 0.13 mm/px is not.

### Wet and glossy tyres.

Specular reflection saturates the sensor and bakes highlights into the texture, which the network will cheerfully learn as "wear". Cross-polarisation helps a great deal but does not eliminate it.

**Characterise the failure precisely, report it as a named limitation, propose cross-polarisation as the mitigation.** A well-measured limitation is a strength.

### The jig→real gap.

Your jig has no suspension compliance, no caster, no dynamic load transfer. The model will learn jig-specific cues. Expect a meaningful MAE increase on real vehicles.

**Mitigate by collecting real-vehicle data early and continuously, not as a final validation step.** If real clips only arrive at the end, you'll discover the gap with no time to respond. And quantify it — "jig→real adds X° MAE" is itself a useful published number.

### The timeline.

Rig build + illumination + calibration + 300-tyre collection + annotation + six models + fifteen ablations + a report, in one semester, alongside coursework, with four people.

**Realistic expectation: you'll hit ~70% of this plan.** That's fine — 70% is still an excellent capstone. `07_ROADMAP.md` has the cut list in priority order. Read it now so the cuts are deliberate in month four rather than panicked in month six.

---

## 4b. Update after seeing `final_v1` (2026-08-26)

I said in §5 that annotation throughput would be the binding constraint. Having analysed the pilot dataset, **I was wrong about which data constraint binds first.** It is not annotation. It is **the number of independent tyres.**

The package is 4,598 files. It is **12 tyres**, photographed in one 22-minute window on one day.

What that does to your evaluation, measured rather than assumed:

| Trivial baseline | fold 0 | fold 1 | fold 2 | mean |
|---|---:|---:|---:|---:|
| Ten colour numbers from a 64×64 thumbnail | **0.952** | 0.399 | 0.123 | 0.491 |
| Nine texture numbers from the tread band | 0.354 | 0.119 | **0.976** | 0.483 |

Two things to take from this:

**A ten-number colour model scores 95.2% on fold 0.** If you train a ConvNeXt and report fold 0, you will have reproduced mean RGB. Any single-fold number from this package is not a result.

**The two probes win on opposite folds.** That is the signature of memorising tyres rather than learning wear. With one to two sessions per class per fold, the class label is very nearly a tyre identifier — for the `mid` class it *is* one physical tyre in every fold.

The encouraging half: the physical signal is real. Deep-groove shadow fraction (`d20`: 0.046 → 0.037 → 0.019) and groove-banding strength (`colstd`: 0.60 → 0.51 → 0.39) are both monotone in the correct direction across all three classes. There is a genuine wear cue in these images. There is just not enough independent data to demonstrate a model is using it.

**So the advice changes.** Do not spend this month on architecture. Spend it on:

1. A ₹900 digital tread gauge, and a measurement for every tyre from now on
2. A printed ruler or ChArUco marker in frame on every capture — mm/px for free
3. **Forty more tyres**, across different days, sites and lighting

Those three things cost under ₹2,000 and a few weekends, and they would move this project further than any model you could train on `final_v1`.

Run the pilot classifier anyway — but for the **training harness**, not for its accuracy. Build the resumable notebook, the HF sync, the evaluation and conformal code. When the real data lands, the infrastructure will already work.

One more thing the images showed: **new tyres carry coloured paint stripes and white lettering from the factory.** That is a direct, free shortcut to the low-mileage class. Crop to the tread band, and run a stripe-masked evaluation before believing any low-class recall number.

---

## 5. The next constraint after data volume

**Annotation throughput. Not the models.**

I'd put it at 70% likely that annotation is what constrains your final dataset size, and near-zero that architecture choice is. Yet the instinct on a project like this is always to start with the model, because that's the fun part.

The failure mode is concrete: two months on architecture, rig finished in month three, annotation starts in month four, dataset ends up at 80 tyres instead of 300, and every per-class metric on rare wear patterns is noise.

**Counter-measures, in priority order:**

1. **Build the SAM2-assisted annotation workflow in P2, before scaling collection.** ~8× throughput is the difference between 300 tyres and 60.
2. **Measure annotation time per tyre in the pilot.** If it exceeds ~15 minutes, stop and fix the workflow.
3. **Bootstrap.** Label 30 → train B0 → pre-annotate the next 30 → correct rather than draw.
4. **Start the longitudinal gauge study in week 1**, before the rig exists. It's the one thing strictly gated by the wall clock.
5. **Collect scrapyard tyres in week 3.** Rare classes are where your metrics will die, and they cost nothing.

---

## 6. Things I'd push back on

**Don't reach for a bigger model when detail is the problem.** The instinct with "recognise every single detail" is SegFormer-B5, more epochs, a larger backbone. Wrong lever. Detail comes from **illumination** (photometric stereo), **resolution** (mm/px and tiling), and **the right loss** (boundary + clDice). Architecture is maybe fifth on that list. Your spec already correctly says B4/B5 shouldn't be first choice — hold that line.

**Don't freeze the backbone.** The small-data instinct is to freeze and linear-probe. The DINOv3-vs-ImageNet industrial-inspection study found frozen SSL features give no clear advantage, while **fully fine-tuned** SSL initialisation is the strongest option. Initialise from SSL, fine-tune everything.

**Don't skip the boring metrology.** The gauge test–retest study is an afternoon of tedium and it's the highest-value afternoon in the project. Without a noise floor none of your numbers are interpretable, and a good examiner will spot that in the first five minutes.

**Split by subsystem, whoever is working on it.** Write down the interfaces (`07_ROADMAP.md`). Multi-person projects fail at the seams, not inside the modules — and a solo project fails by half-finishing four things at once, which is the same failure wearing a different hat.

**If you are running this alone, the binding constraint moves to compute.** Annotation is four hours; that is fine. But Stage A on a single Kaggle account is ~98 GPU-h ≈ 3.5 weeks, and the whole study ~15 weeks. Three more free accounts fix it entirely and need no coordination, because ownership is decided by arithmetic rather than negotiation. If that is not possible, cut the model zoo to Tiers 0–2 (~47 GPU-h) — still a complete, honest, reportable architecture result. Cut seeds last, and never cut folds.

**Do release the dataset.** Of everything here, it's the thing most likely to outlive the project and get cited. Models age out in eighteen months; a well-documented dataset with a real ground-truth protocol gets used for years.

---

## 7. Risk register

### Updated register for the comparative study

| Risk | P | Impact | Mitigation | Trigger |
|---|---|---|---|---|
| **Reporting a single fold as the result** | **High** | **Fatal** | 3 folds × 3 seeds always; baselines on every figure | Any results table |
| **Broken resume / lost-update registry corrupts hundreds of runs** | Medium | **Fatal** | ⚠ Bugs 1–7 in `05`; per-writer shards; post-seam resume test | Before S2 |
| **XAI method mismatched to architecture** | **High** | **Fatal to the headline** | `14 §1`; faithfulness-selected method per architecture; state the choice | Before S6 |
| **Metrics CSV schema changed mid-sweep** | Medium | High | Freeze before S2 | Any schema edit after S2 starts |
| **Pseudo-mask quality unmeasured** | Medium | High | 50-image manual audit published before any dependent number | Before S6 |
| **TER not area-normalised** | Medium | High | `TER_norm` is the reported number | Any TER claim |
| **Hypotheses invented after seeing the scatter** | Medium | High | Pre-register H1–H3 with a date in `PROGRESS.md` | Before S6 |
| Shortcut learning (paint stripes, tread identity, background, dirt) | **High** | Medium | It is now the *subject*, not just a threat — stress-test matrix in `06 §5` | Ongoing |
| HF rate limit hit across 4 accounts | Medium | Medium | Per-token bucket, `cap × n < 128` | Decision #1 in `PROGRESS.md` |
| ROI host-RAM growth kills the kernel without an exception | **Realised, repaired** | High | Bbox-only masks; ROI workers/pinning off; 88% checkpoint-and-push guard (`05`, Bug 17) | `proc_rss_gb_peak` rises across epochs |
| RegNet AMP + `channels_last` selects a failing T4/cuDNN grouped-conv path | **Realised, repaired** | High | Same model/config on contiguous NCHW; cuDNN autotuning off; fatal-context stop (`05`, Bug 18) | `ERROR.txt`; `runtime_cuda_memory_format` |
| Simultaneous workers steal fresh absent runs before the first claim is visible | **Realised, repaired** | High | Absent work stays with static owner; only a real event older than 45 min is stealable (`05`, Bug 18) | duplicate owners for one `run_id` |
| Work imbalance across accounts | Medium | Medium | LPT bin packing on a **static** cost table (⚠ Bug 7) | Print the plan first |
| Too few independent tyres (12) | **Realised** | — | **This is the study's premise, not a blocker.** Still the top `final_v2` priority | — |
| Scope creep — adding models forever | Medium | Medium | The cut list in `07`; Tier 6 cut last | Behind at S4 |

### Retained from the earlier plan

| Risk | P | Impact | Mitigation |
|---|---|---|---|
| No tread-depth measurements | Realised | High | Buy gauge; measure every tyre from now |
| Annotation too slow | Medium | Medium | SAM2 pseudo-labels remove most of it |
| Toe never reaches useful precision | **High** | Medium | Reframe as binary screening; cut if needed | P4 midpoint review |
| Sipes unresolvable at full width | Medium | Medium | Higher-res sensor or cropped-region claim | v0 resolution measurement |
| Photometric stereo fails on rubber | Low-med | Medium | Hand-torch test in P0; fall back to flat + classical | P0 week 1 |
| Calibration drifts | Medium | High | Weekly zero-toe test; calibration ID per clip | Drift > 0.2° |
| Jig→real gap too large | Medium | High | Real clips from P3 start; report the gap | Real MAE > 2× jig MAE |
| Brand shift hurts generalisation | **High** | Medium | Unseen-brand split from day one; domain adaptation ablation | Unseen-brand gap > 10 points |
| Wet/glossy failure | Medium | Low-med | Cross-polarise; wetness flag; document limitation | Robustness sweep |
| No alignment-rack access | Medium | Medium | Jig covers most GT; rack only for the gold set | Start asking in P1 |
| Kaggle GPU quota exhausted | Medium | Medium | Debug locally in `cv_conda`; never debug on Kaggle | > 20 h used mid-week |
| Session crash loses progress | Medium | High | Resume contract in `05_TRAINING_KAGGLE_HF.md` | Run equivalence test in P4 |
| Team interface mismatch | Medium | High | Write the four interface contracts in week 1 | Any integration surprise |
| Timeline overrun | **High** | Medium | Cut list in `07_ROADMAP.md`; decide at P4 midpoint | Behind at end of P3 |
| Scope creep | Medium | High | This document. Re-read monthly | Any new "wouldn't it be cool if" |

---

## 8. If I had to bet

**Most likely outcome, honestly assessed:** you build a well-calibrated front-view rig with photometric-stereo illumination, collect 150–250 unique tyres, produce solid segmentation (boundary F ~0.70) and wear recognition (macro-F1 ~0.75, critical-wear recall > 0.92), get camber to ~0.4° MAE and toe screening to ~0.85 AUROC, cut the app and possibly PatchCore, and submit a strong Review-3 plus a journal-quality manuscript for *Measurement* or IEEE Access.

That is an excellent capstone. Real contribution, defensible in a viva, and the dataset alone justifies the project.

**The upside case** — annotation goes fast, photometric stereo delivers, 350 tyres collected — is meaningfully better, and the **wear↔geometry cross-check** (Contribution 5) is the piece most likely to interest people outside this niche. Nothing in the literature distinguishes recent from chronic misalignment. **Protect that from the cut list.**

**The downside case** is the one where three months go into architecture, the rig lands in month four, annotation starts in month five, and there are 60 tyres at the end. Everything in `07_ROADMAP.md` is structured to prevent exactly that.

---

## 9. What to do this week

1. **Photograph a tyre from the intended viewpoint with a phone.** Can you see a sipe? A TWI bar? Measure the mm/px against a ruler in frame. *(Half a day.)*
2. **Hand-torch photometric test.** Same tyre, four torch positions, four photos. Compare against one flat-lit photo. If grooves and cracks visibly pop, order the LEDs. *(One hour, and it decides the project's most distinctive design choice.)*
3. **Buy the digital tread depth gauge.** Measure four tyres on one car, record with the date. The longitudinal study has now started.
4. **Assign the four roles** (`07_ROADMAP.md`) and write down the four interface contracts.
5. **Order long-lead parts** — camera, lens, polarising film.

Steps 1–3 cost under ₹2,000 and one day, and they de-risk the two things most likely to determine how this project ends.
