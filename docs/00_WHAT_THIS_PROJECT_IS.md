# 00 — What This Project Is

> **Purpose of this document.** This is my understanding of the project, written plainly, so you can check whether I have it right. If anything here is wrong, correct it and everything downstream gets corrected with it.
>
> No jargon where plain words will do. No assumed knowledge from the other documents.

---

## 1. The problem in one paragraph

A tyre wears out gradually. The tread — the grooved rubber that touches the road — gets shallower until the tyre can no longer grip properly, especially in rain. Right now, checking this means a person crouching down with a small depth gauge, poking it into a groove in two or three places, and forming a judgement. It is slow, it depends on who is doing it, and it produces almost no record. Most drivers never do it at all.

Separately, if a car's wheels are not pointing exactly where they should be — misaligned — the tyre gets dragged sideways slightly with every metre travelled, and wears unevenly and much faster. Checking that requires a workshop machine costing lakhs.

**We want a camera to do both jobs.**

---

## 2. What we are actually building

A camera is mounted **low and in front of one wheel**, aimed backwards and slightly upwards, so it looks at the front face of that tyre's tread. It records a short video. From that video the system should produce:

1. **A detailed reading of the tyre** — not just "worn" or "not worn," but *where* it is worn, *how* it is worn, and whether there is visible damage
2. **A screening estimate of that wheel's alignment** — is this wheel's camber and toe roughly where it should be?
3. **A confidence figure on everything**, and the ability to say *"I cannot tell from this image"*
4. *(optional, later)* An app that connects to the camera and shows all of it

The full title is **"Vision-Based Detailed Tyre-Wear Recognition and Single-Wheel Alignment Screening."**

---

## 3. The specific thing that makes this hard

The problem statement I was given emphasises one phrase: **"recognising the wheel in every single detail."**

That is the actual difficulty, and it deserves unpacking.

A tyre is close to the worst possible subject for a camera:

- **It is black.** Carbon-black rubber reflects only about 4–8% of light. Very little signal.
- **It has almost no colour information.** You cannot segment it by hue the way you can segment a road sign.
- **Its features are defined by *shape*, not by brightness.** A groove is a *hole*. A crack is a *split*. A worn rib edge is *rounded off*. Under flat, even lighting, all of these nearly disappear because there is nothing to make them stand out.
- **It is curved, it deforms near the road, and it goes shiny when wet.**
- **The interesting details are tiny.** A sipe — those little slits in the tread — is 0.3 to 1 mm wide.

So "detect if the tyre is worn" sounds like a simple image classification task, and it is not. Deciding *how worn, where, and in what pattern* means resolving sub-millimetre structure on a dark, curved, low-contrast surface.

---

## 4. The two tasks are genuinely different, and that shapes everything

This is, in my view, the most important design decision in the project, and it was already correct in the Review-1 report.

| | **Wear** | **Alignment** |
|---|---|---|
| What it is | How the rubber looks — texture, groove depth, patterns | Geometry — angles of the wheel in space |
| Evidence | Shading, shadow, texture, local shape | Positions and orientations, measured against a known reference |
| Right tool | A deep neural network | Calibrated geometry, with a small learned correction |
| If you use the wrong tool | Misses subtle wear | **The network secretly learns the camera's tilt** and appears to work until you move the camera |

If you train one network to output `(worn?, camber, toe)` from raw pixels, it will score beautifully in validation and fall apart the moment the mounting bracket shifts by two degrees — because it learned the bracket, not the wheel.

**So the architecture keeps them on separate paths and only combines them at the end, when making a decision.**

---

## 5. What we are doing *right now* — and why it changed

The description above is the eventual system. **It is not what we are building this phase**, and pretending otherwise would waste the semester.

An earlier plan chained together a segmentation model, a multi-head wear model, an anomaly detector and a geometric alignment estimator. It assumed labels we do not have — masks, gauge readings, alignment angles — and hardware we are not building.

**The current phase is a different kind of work: a broad, controlled, comparative study.**

We train **many** models — many architectures, many techniques, across classification, detection and segmentation — and use explainability (Grad-CAM and its relatives) not as a pretty picture at the end, but as **the measuring instrument**.

### The question, and why it had to change

A normal benchmark asks *"we trained 20 models — which is most accurate?"* On this dataset that would be **ranking noise**. Two deliberately stupid baselines I ran, on the dataset's own folds:

| Baseline | fold 0 | fold 1 | fold 2 | mean |
|---|---:|---:|---:|---:|
| 10 colour numbers from a 64×64 thumbnail | **0.952** | 0.399 | 0.123 | 0.491 |
| 9 texture numbers from the tread band | 0.354 | 0.119 | **0.976** | 0.483 |

A swing from 0.12 to 0.98 between folds will drown any difference between two architectures.

> ### So the question became:
> ### **Not "which model is most accurate?" but "which model actually looks at the tread?"**

Accuracy is not identifiable with 12 tyres. **Where the evidence comes from is** — and it is the property that decides whether a model survives a tyre it has never seen.

### How the three task types fit, given we only have classification labels

```
A. CLASSIFICATION   3-class ordinal mileage proxy  ← the only real labels
        │ take the Grad-CAM heatmaps
        ▼
B. LOCALISATION     turn heatmaps into boxes; SAM2 gives reference boxes;
                    train YOLO26 on those pseudo-labels
        │ use the heatmap to prompt SAM2
        ▼
C. SEGMENTATION     SAM2 produces tyre/tread masks with no annotation;
                    distil them into SegFormer / U-Net / DeepLabV3+
        │ tread mask
        ▼
D. MEASUREMENT      how much of each model's evidence lands on the tread?
```

Segmentation earns its place because **the masks are what make the measurement possible** — not as a separate thing competing for attention.

### The new metrics

| Metric | Meaning |
|---|---|
| **TER** Tread Evidence Ratio | share of the model's attention that falls on the tread (area-normalised) |
| **BAR** Background Attribution Ratio | share falling outside the tyre entirely |
| **SAR** Stripe Attribution Ratio | share falling on **factory paint stripes and lettering** |

That last one matters. New tyres come with coloured paint stripes and white lettering from the factory — clearly visible in the images. That is a free giveaway for the `low` class. A model with high SAR has learned the factory marking, not the tread.

Full design: `13_EXPERIMENT_PLAN.md`. XAI method details: `14_XAI_PROTOCOL.md`.

### What the eventual system would still output

When the labels exist, the pipeline in §2 returns one of four verdicts:

```
PASS  ·  MONITOR  ·  INSPECT  ·  UNABLE_TO_MEASURE
```

That fourth one is deliberate. A system that always gives an answer is more dangerous than one that admits when it cannot see.

---

## 6. What we are *not* claiming

I think this is where the project earns credibility, so I want to state it clearly:

| We claim | We do **not** claim |
|---|---|
| Detailed, localised wear and damage recognition | Certified tread depth in millimetres from a plain photo |
| An ordered wear severity level | A replacement for a depth gauge |
| **One wheel's** camber and toe as a *screening* estimate, with error bars | Total axle toe, thrust angle, caster, or a four-wheel alignment |
| An explicit "I cannot tell" output | A confident answer on every image |

A proper workshop alignment machine measures toe to about ±0.05°. We will not approach that from one uncalibrated camera looking at one wheel. **What we can offer is different and still useful**: screening every wheel, with no markers, no clamps and no setup, to tell you *which* cars need to go on the proper machine.

Similarly, millimetres of tread depth require a metric sensor — a laser line, stereo cameras, or a depth camera. From ordinary RGB we can produce a *relative* or *ordered* estimate. Calling that "millimetres" would be dishonest.

---

## 7. Where the data actually stands — and this is the crux

We have a prepared, verified pilot dataset: `final_v1`.

**On paper:** 4,598 images.
**In reality: 12 tyres**, photographed in a single 22-minute window on one morning.

Here is what that package contains and how to read its folders:

```
images/clean/      real photographs, deduplicated and standardised (418 images)
images/augmented/  artificially varied copies of those same photos (4,180) — TRAINING ONLY
fold_0/1/2         cross-validation groups — NOT wear categories
low/mid/high       mileage-proxy classes, from the original workshop folder names
```

Three things I want to be precise about, because they are easy to get wrong:

**(a) The folds are not classes.** `fold_1` does not mean "medium wear." It means "this group of tyres is held out in cross-validation run 1." Whole capture sessions were assigned to folds so that near-identical photographs of the same tyre never appear on both sides of a train/validation split.

**(b) The augmented images are not new data.** Every one of the 418 real photos has exactly 10 artificially varied copies — rotated slightly, brightened, flipped, etc. They help the model tolerate nuisance variation. They create **zero new tyres**. The dataset itself records `independent_tyre_increment = 0` on every row, which I think is an admirably honest touch. They must never be used for validation.

**(c) The labels are a mileage proxy, not measured wear.** They came from workshop folders named "New Tires," "40,000," "Tires Gone (1L and Above)" — odometer categories, not gauge readings. A carefully maintained 90,000 km tyre can have more tread left than a neglected 40,000 km one. **So the honest description of what we can train today is "a three-class mileage-proxy classifier," not "a tyre wear detector."**

### What this means for the plan

Of the roughly thirteen capabilities described above, **one is trainable today**: the three-class mileage-proxy classifier. Everything else — segmentation, wear patterns, damage, depth in millimetres, alignment — needs labels that do not exist yet, and for alignment, camera calibration that was never captured.

That is not a failure. That is what a pilot dataset is for. But the documents should not pretend otherwise, and now they do not.

---

## 8. The measurement I ran, and why it matters

Rather than assume the dataset was limited, I measured it. I trained two deliberately stupid models on the dataset's own cross-validation folds:

- **Probe A** — ten colour numbers (average red, green, blue, brightness, contrast…) from a 64×64 thumbnail. No texture, no tread structure at all.
- **Probe B** — nine texture numbers from the centre of the tread, with brightness normalised away.

| | fold 0 | fold 1 | fold 2 | mean |
|---|---:|---:|---:|---:|
| Probe A (colour) | **0.952** | 0.399 | 0.123 | 0.491 |
| Probe B (structure) | 0.354 | 0.119 | **0.976** | 0.483 |

*(macro-F1)*

**Ten colour numbers score 95% on fold 0.** If we train a proper neural network and report fold 0, we will have achieved exactly what average RGB already achieves.

**The two probes win on opposite folds.** That is the fingerprint of memorising individual tyres rather than learning wear. With only one or two capture sessions per class per fold, the class label is very nearly a tyre identifier — for the `mid` class it literally *is* one physical tyre in every fold.

**But there is genuine signal underneath.** The fraction of very dark pixels (deep grooves cast deep shadows) runs 0.046 → 0.037 → 0.019 across low → mid → high mileage. Groove banding strength runs 0.60 → 0.51 → 0.39. Both are ordered correctly, and both are physically exactly what a wear model *should* be looking at. The signal is real. There is simply not enough independent data to prove a model is using it.

**Conclusion I draw:** the limiting factor is not the model, the architecture, or the GPU. It is that we have twelve tyres.

---

## 9. What happens next

**This phase — the study.** Roughly 800 training runs, ~440 GPU-h, across four Kaggle accounts (~4 weeks of wall-clock). Stages:

| | |
|---|---|
| **S0** | Infrastructure — resumable training, multi-account sharding, HF as the only permanent store. **Nothing starts until the kill-and-resume test passes** |
| **S1** | Baselines (2 of 5 already done) |
| **S2** | Architecture sweep — ~30 models × 3 folds × 3 seeds |
| **S3** | SAM2 masks + a 50-image manual audit of their quality |
| **S4** | Technique sweep — 12 factors, one at a time, on the top 3 architectures |
| **S5** | Detection + segmentation |
| **S6** | XAI over every trained model — the primary axis |
| **S7** | Shortcut stress tests |
| **S8** | Ensembles + calibration |
| **S9** | Report, Review-3, paper |

**Beyond this phase — the data.** Still the highest long-term leverage, and still worth starting now in parallel:

1. **A digital tread depth gauge (~₹900).** This single item converts the project from mileage-proxy classification to wear *measurement*
2. **A printed ruler or marker in every photo** — free, and it gives millimetres-per-pixel
3. **40+ more tyres**, across different days, lighting and locations — the only thing that breaks the tyre-identity confound

None of this blocks the study. Every experiment we build now re-runs unchanged on `final_v2`, so the infrastructure is not throwaway.

---

## 10. Rules I am treating as non-negotiable

Derived from the dataset package's own guidance and from what the probes showed:

1. **Always report all three folds** with mean and spread, and always print the trivial baselines beside them. A single-fold number from this dataset is not evidence.
2. **Never call the labels `worn` / `not_worn`.** They are mileage proxies until a gauge says otherwise.
3. **Never put augmented images in validation or test.**
4. **Never say the sample size is 4,598.** It is 12.
5. **Group splits by tyre, never by image.**
6. **Horizontal flip is banned** for anything involving toe, camber, or inner-versus-outer wear — flipping the image swaps left and right and reverses the sign of the angle. It is allowed for the current mileage-proxy task only.
7. **Never generate or in-fill parts of the tyre that were not photographed.** Unseen is `unknown`, not `healthy`.
8. **`final_v1` is immutable.** Corrections create `final_v2`.
9. **Nothing here is safety-certified.** Every output is research insight, and physical inspection by a qualified technician remains the authority.

---

## 11. Practical context I am working within

| | |
|---|---|
| Course | Capstone Project, Fall-Sem 2026–27 |
| Department | AI & ML, SCOPE, VIT-AP |
| Team | Bonala Shanmukesh · Gunnamneni Nehru · GV Manu Rohith · Nettem Harish Kumar |
| Guide | Dr. E. Sreenivasa Reddy |
| Stage | Review-1 submitted; pilot dataset delivered |
| Local environment | `conda activate cv_conda` before every Python command |
| Training | Kaggle, dual T4, fp16, must survive session restarts and resume mid-epoch |
| Storage of record | Hugging Face (`Shanmuk4622`), synced every ~30 min and immediately on interrupt |
| Dataset location | `D:\Dataset Download\Tire Dataset Prepared\FINAL` |
| Repository | `tyre-wear-alignment-cv` |
| Ambition | A publishable contribution, not only a passing grade |

---

## 12. My honest summary

**What this project is, long term:** reading a tyre's condition in fine detail from an ordinary camera below and in front of the wheel, and squeezing a useful — though not workshop-grade — alignment hint out of the same image.

**What this project is, this phase:** a broad, controlled study of *what actually works* on that problem when you have 12 tyres and no measurements — with explainability used to tell learning from memorising.

**What makes it interesting:** most benchmarks rank models by accuracy. On a dataset this small, accuracy is noise — I measured a 0.12→0.98 fold swing from trivial baselines. Ranking by **where the model looks** is both more stable and more honest, and it is the property that actually predicts generalisation.

**What makes it hard:** not the modelling. The number of independent tyres, and the fact that the images contain several free shortcuts — factory paint stripes, dirt colour, tread-pattern identity, background — any of which a network will happily take instead of learning wear.

**What I think will work:** the study itself. It is achievable with the compute available, it needs no hardware and no annotation campaign, and it produces a real answer to a question people building tyre inspection from small datasets actually have.

**What I think will not, yet:** anything claiming millimetres or degrees. Those need a gauge and a calibrated camera. Deferring alignment is right — though it is the *harder* half, not the easier one, and worth planning for accordingly.

**The single most useful thing that could happen alongside:** somebody buys a ₹900 tread depth gauge and starts writing down numbers.

---

*If any of this is wrong, tell me which part. Every other document in `docs/` follows from this one.*
