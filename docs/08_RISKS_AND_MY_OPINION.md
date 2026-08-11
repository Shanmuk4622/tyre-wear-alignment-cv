# 08 — Risks, and My Honest Opinion

You asked for my opinion. Here it is, without hedging.

---

## 1. What I think of the original idea

**The instinct is good; the framing was doing you a disservice.**

"Detect if a tyre is worn and if the wheel is aligned, while the car is in motion" is, as stated, two unrelated tasks glued together by the phrase "while in motion." That framing has three problems:

1. **"While in motion" reads as a constraint, not a capability.** As written it sounds harder for no benefit — you'd be fighting motion blur to achieve what a parked photo does better. An examiner will ask "why not just take a photo?" and you need a better answer than "because it's harder."

2. **The two tasks look unrelated.** A project that trains two separate models for two separate things is an engineering exercise, not research. Two models is not a contribution; it's two homework assignments.

3. **The precision claim was unbounded.** "Detect even the smallest change" is not a specification. Without a target number and a stated noise floor, you cannot know when you're done, and you cannot defend any result.

**The fixes are all in this repo:**

- Motion became an *asset*, not an obstacle — the rolling constraint gives you zero-blur imaging that a static photo cannot match, and the rig defines the vehicle's reference frame, which is exactly what alignment measurement needs. Motion is now the *reason* the system works.
- The two tasks became one task — wear is the time-integral of alignment. That coupling is the research contribution, and the disagreement diagnostic (`04_MODEL.md §6`) is a capability neither task has alone.
- Precision got numbers, a noise floor, and honest scope limits.

Your instinct that "we can do much better if we think a lot" was right. This is what that looks like.

---

## 2. What will work

I'm confident about these:

| | Confidence | Why |
|---|---|---|
| Zero-blur contact-patch imaging | **Very high** | It's kinematics. `v_surface = 2v·sin(φ/2) = 0` at contact. Not a hypothesis. |
| Tread depth to ~0.3 mm with laser supervision | **High** | Laser triangulation at 0.08 mm/px is standard industrial metrology. The only question is how much the RGB student loses vs the laser teacher. |
| Wear-pattern classification | **High** | On a clean unrolled map this is an easy vision task. It will work in week 12. |
| Camber estimation | **Moderate-high** | Camber produces large, unambiguous footprint asymmetry. This is the observable one. |
| TWI self-calibration | **High** | The bars are there, they're 1.6 mm, they're detectable. |
| Ranking > regression on small data | **High** | Well-established. Your dataset will be small; this is the right call. |

---

## 3. What probably won't work — and what to do about it

### Toe to workshop precision. It won't happen.

Passenger-car toe spec is ±0.1° — six arcminutes. At a 300 mm tyre radius that's **0.5 mm of lateral displacement at the tread edge.** You're inferring it from a deformable rubber structure, under variable load, through glass, from a single wheel.

Meanwhile the *real* toe angle you're trying to measure changes by more than your target precision when the car is loaded, when the tyre is warm, when the bushings flex.

**My advice: stop trying to measure toe. Screen for it.** Reframe as binary out-of-spec detection with an AUC target. This is honest, achievable, and — importantly — it is the *more useful product*. A workshop doesn't need you to replace their rack; they need to know which of the 40 cars that drove in today should go *on* the rack. That's a real commercial gap and nobody fills it.

If you insist on continuous toe: capture **both wheels of an axle**. Differencing cancels load and suspension confounds. Single-wheel continuous toe is, in my view, not achievable at useful precision.

### Wet and dirty operation. It will fail.

Water frustrates TIR. So does mud. Your beautiful high-contrast footprint becomes noise the first time it rains.

**Don't fight this in the capstone.** Characterise the failure precisely, propose the air-knife mitigation, put it in Limitations. A well-measured, clearly-scoped limitation is a strength. A system that quietly degrades in the rain and doesn't say so is a liability.

### The jig→real domain gap. It will be bigger than you expect.

Your jig has no suspension, no compliance, no dynamic load transfer, no caster. The model will learn jig-specific cues. I'd guess a 1.5–2.5× MAE increase going from jig to real vehicles.

**Mitigate by collecting real data early and continuously**, not as a final validation step. If real data only arrives in week 20, you'll discover the gap with no time to fix it. Also: quantify and report the gap. "Jig→real adds 0.18° MAE" is a genuinely useful number for anyone who builds on this.

### The 24-week timeline. It's tight.

You have a hardware build, a data collection campaign, a multi-stage model, a full evaluation, two deployments, and a paper. Solo. In 24 weeks. Alongside coursework.

**Realistic expectation:** you'll hit maybe 70% of this plan. That's fine — 70% of this plan is still an excellent capstone. `07_ROADMAP.md §"What to cut"` tells you what to drop, in order. Read it now so you make the cuts deliberately in week 16 rather than panicking in week 22.

---

## 4. The single biggest risk

**Not the model. The data.**

I'd put it at 70% likely that data collection is what constrains your final results, and near-zero that model architecture is. Yet the instinct on a project like this is always to start with the model, because it's the fun part and it doesn't require going outside.

Concretely, the failure mode is: weeks 1–8 spent on architecture and Blender, rig finished in week 12, data collected in weeks 14–18, and then there's no time to iterate on anything you learn from it.

**Counter-measures:**
- Build v0 in week 1. Two days. Cardboard and a phone.
- Start the longitudinal gauge study in week 1, before the rig exists. You cannot buy back lost calendar time on a longitudinal study — it's the one thing in this project that is strictly gated by the wall clock.
- Collect scrap tyres in week 3. Rare classes are what your metrics will die on.
- Get 20 real-vehicle passes by week 10, even if the rig is imperfect. Imperfect real data beats perfect synthetic data for telling you what's wrong.

---

## 5. Things I'd change about how you're thinking

Said plainly, because you asked for an opinion:

**Don't reach for a bigger model when precision is the problem.** The instinct with "detect the smallest change" is a bigger backbone, a transformer, more epochs. That's the wrong lever. Precision comes from a physical ruler in the frame (TWI bars), a better-conditioned objective (ranking, not regression), and a higher-quality teacher signal (laser). Architecture is maybe the fifth most important factor here.

**Don't skip the boring metrology.** The gauge test–retest study is an afternoon of tedium and it is the highest-value afternoon in the project. Without a noise floor, none of your numbers mean anything, and a good examiner will spot that in the first five minutes.

**Don't be afraid to narrow scope.** A capstone that does *one* thing rigorously beats one that does three things approximately. If in week 16 the toe head isn't working, cut it. "We deliberately scoped to camber because toe is not observable at useful precision from a single wheel, and here is the analysis showing that" is a *better* result than a toe number you don't believe.

**Do release the dataset.** Of everything here, the dataset is most likely to outlive the project and get cited. Models age out in eighteen months; a well-documented dataset with a real ground-truth protocol gets used for years.

**Write earlier than feels natural.** Start the paper in week 20 with empty results tables. Deciding what the figures need to show will change what analysis you run, and it will change it while you still have time to run it.

---

## 6. Risk register

| Risk | P | Impact | Mitigation | Trigger to act |
|---|---|---|---|---|
| FTIR doesn't give usable contrast | Low | High | v0 test in week 1; fallback to flood-lit ground view | Week 1 fingerprint test |
| Rig calibration drifts | Med | Med | Recalibrate weekly; log calibration in every pass's metadata | Zero-toe test drifts > 0.1° |
| Glass scratches / breaks | High | Med | Buy 2 spare plates upfront; sacrificial film | Visible scratching |
| Data collection slips | High | High | Start week 1; overlap phases; cut synthetic first | Behind at week 9 |
| Jig→real gap too large | Med | High | Real data from week 10; domain adaptation; report the gap | Real MAE > 2× jig MAE |
| Toe head never works | **High** | Med | Reframe as binary screening; consider dropping | Week 16 decision point |
| Kaggle quota exhausted | Med | Med | Debug locally in `cv_conda`; never debug on Kaggle | > 20 h used mid-week |
| Session crash loses progress | Med | High | The resume contract in `05_TRAINING_KAGGLE_HF.md` | Run the equivalence test in week 9 |
| No workshop rack access | Med | Med | Jig provides most GT; rack only needed for the gold set | Start asking week 4 |
| Timeline overrun | **High** | Med | Cut-list in `07_ROADMAP.md`; decide deliberately at week 16 | Week 16 review |
| Scope creep | Med | High | This document. Re-read it monthly. | Any new "wouldn't it be cool if" |

---

## 7. If I had to bet

**Most likely outcome, honestly assessed:** you build a working ground-view rig, produce a genuinely novel dataset of a few thousand passes, hit ~0.3–0.4 mm tread depth MAE and ~0.4° camber MAE, get toe screening to around 0.85 AUC, cut the mobile app, and submit a solid 8-page workshop paper.

That is an excellent capstone. It would be a real contribution, it's defensible in a viva, and the dataset alone justifies the project.

**The upside case** — if FTIR works as well as the physics suggests and data collection goes smoothly — is meaningfully better than that, and the disagreement diagnostic (`04_MODEL.md §6`) is the piece most likely to make it interesting to people outside this niche. That idea is the one I'd protect from the cut-list.

**The downside case** is the one where you spend twelve weeks on model architecture, build the rig in week 14, and run out of runway. Everything in `07_ROADMAP.md` is structured to prevent exactly that.

---

## 8. What to do tomorrow

Not next week. Tomorrow.

1. Buy a sheet of glass (any glass, ~₹300), a 12 V LED strip, **a metre of clear paint-protection film (~₹400)**, and cadge a scrap-tyre offcut from any tyre shop.
2. In a dark room, edge-light the glass and photograph four things: your thumb; bare rubber on bare glass; rubber on the film; and the contrast difference between the last two. **This four-step test — not the thumb alone — is the real go/no-go.** (`02_RIG_BUILD.md §2`)
3. Buy the digital tread depth gauge. Measure all four tyres on one car. Write the numbers in a spreadsheet with the date. That's the longitudinal study started.
4. Order the v1 parts — the camera and polished glass have the longest lead times.

Steps 1–3 cost under ₹2,000 and one evening, and they de-risk the two things most likely to sink the project.

---

## 9. Postscript — what the literature audit changed

I went and read the field properly (`09_RELATED_WORK.md`). Three things changed, and my overall confidence went **up**, not down.

**The FTIR idea is prior art.** Cabrera et al. published a validated FTIR tyre-footprint bench in *Sensors* in 2017, and Chodera proposed the principle decades earlier. My original novelty claim #1 was wrong.

But this is a **better** position than I thought I was in. They published fitted curves for *contact area vs camber angle* — meaning someone has already demonstrated, with peer review, that the exact signal I'm asking you to bet the project on genuinely exists. You are no longer gambling on untested optics. You are deploying validated physics in a new setting and inverting the model. That is a smaller claim and a much safer project.

**My optics were wrong in a way that would have cost you a month.** I assumed rubber index-matches glass and therefore frustrates TIR usefully. Index-matching is necessary but not sufficient — the contacting material also has to *scatter light back*, and carbon-black rubber absorbs it. The prior art's plastic lamina isn't incidental; it's load-bearing. Without an interface film you would have built the rig, seen a black screen, and had no idea why. **This is the most valuable thing the research pass produced**, and it cost ₹400 to fix.

**The toe consistency loss got a real physical foundation.** Toe angle *is* a permanently applied slip angle, and the brush tyre model gives wear rate as an explicit function of slip angle. So the link function `h_τ` is derivable rather than fitted. Better still: because wear rate depends *steeply* on slip angle, inferring toe from accumulated wear may be **more sensitive** than measuring it geometrically. If that holds up, it inverts my §3 pessimism about toe — worth testing early.

**What didn't change:** the disagreement diagnostic is still, after reading the field, the most novel thing here. Nothing in the literature distinguishes recent from chronic misalignment. Protect it from the cut-list.
