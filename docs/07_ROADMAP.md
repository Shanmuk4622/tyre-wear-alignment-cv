# 07 — 24-Week Roadmap

**Aug 2026 → Jan 2027.** Six phases, each with a hard deliverable and an explicit kill-criterion.

---

## Phase structure

| Phase | Weeks | Theme | Hard deliverable |
|---|---|---|---|
| **P0** Prove | 1–2 | Does the physics work at all? | v0 rig + fingerprint test + one rolled video |
| **P1** Build | 3–6 | Rig + jig + calibration | Calibrated v1 rig, jig at ±0.05°, 200 clean passes |
| **P2** Collect | 5–10 | Data, data, data | 3,000 jig + 300 real passes, synthetic pipeline live |
| **P3** Model | 9–15 | Train the thing | Full model, val MAE < 0.4 mm |
| **P4** Prove it | 14–19 | Evaluation, ablations, robustness | Complete results tables |
| **P5** Ship | 18–22 | Deploy + release | Jetson demo, mobile app, HF release |
| **P6** Write | 20–24 | Paper + report | Submitted manuscript, thesis |

Phases overlap deliberately. Data collection runs in the background of modelling; writing starts before results are final.

---

## P0 · Weeks 1–2 — Prove the physics (do this before anything else)

| Week | Tasks |
|---|---|
| 1 | Glass sheet + LED strip + **clear PPF/PET film** + a scrap-tyre rubber offcut. Run the **4-step go/no-go test** (`02_RIG_BUILD.md §2`) — thumb, bare rubber, rubber-on-film, quantified comparison. Roll a bicycle wheel by hand, record. Start the longitudinal gauge study on 6 vehicles. Order v1 parts (long lead times). |
| 2 | Crude unroll/stitch in Python. Verify grooves are resolvable. Measure actual mm/px. **Read the five Tier-1 papers** in `09_RELATED_WORK.md §9` — especially Cabrera 2017 and the two brush-model theses. Annotated bibliography, 25 papers. |

**Kill-criterion:** if rubber-on-film shows no usable contrast, the FTIR premise is wrong for this application. Fall back to a flood-lit ground-view rig + laser (still novel, still viable) and rewrite `01_CONCEPT.md §3`. **Better to learn this in week 1 than week 12.**

> The literature audit is already done (`09_RELATED_WORK.md`) — week 2 reading is for depth, not discovery. The brush-model theses matter most: they give you the derived link function for the consistency loss.

**Deliverable:** `notebooks/00_feasibility.ipynb` with a stitched tread strip from a bicycle wheel, and a one-page go/no-go memo.

---

## P1 · Weeks 3–6 — Build and calibrate

| Week | Tasks |
|---|---|
| 3 | Assemble v1 frame. Edge-polish glass, mount LEDs, verify FTIR properly. Visit a tyre shop, collect 40 scrap tyres. |
| 4 | Intrinsics + plate homography calibration. Laser mount, laser-plane calibration, step test. Establish gauge noise floor (`06_EVALUATION.md §2`). |
| 5 | Build the alignment jig. Calibrate its verniers against a digital angle gauge. Illumination channel cycling synced to camera. |
| 6 | Write `qc.py`. Zero-toe bias test (20 passes, must be 0 ± 0.1°). First 200 jig passes. |

**Gate to P2:** calibration acceptance criteria all met (`02_RIG_BUILD.md §3`), and the zero-toe bias test passes. **Do not start bulk collection with an uncalibrated rig.** You will collect 3,000 unusable passes and lose a month.

---

## P2 · Weeks 5–10 — Collect

| Week | Tasks |
|---|---|
| 5–6 | (overlaps P1) Synthetic pipeline: Blender parametric tyre, one working render |
| 7 | Jig collection sprint 1: 1,000 passes, uniform over toe × camber × load |
| 8 | Domain randomisation in Blender. Render 20k. Scrap-tyre collection (rare wear classes). |
| 9 | Jig sprint 2: 2,000 more passes, varied tyres. Build the ranking annotation UI. |
| 10 | Real-vehicle collection: 20 vehicles, 300 passes. Annotate patterns + rankings. Release `GRIP-Roll` v0.1 to HF. |

**Gate to P3:** 3,000 jig + 300 real passes with complete metadata, QC-passed, split by tyre ID.

**Risk to watch:** collection always takes 2× longer than planned. Budget slack. If you are behind at week 9, cut the *synthetic* volume, never the real data.

---

## P3 · Weeks 9–15 — Model

| Week | Tasks |
|---|---|
| 9 | Kaggle notebook infrastructure: resume, HF push, dual-T4 DDP. **Verify the 200-step resume-equivalence test.** |
| 10 | Stage A perception: segmentation, laser extraction, groove masks |
| 11 | Stage B reconstruction: rolling-speed estimation, unrolled map pipeline |
| 12 | Stage C tread heads: depth, ranking, pattern. First end-to-end numbers. |
| 13 | Stage D geometry: analytic estimators E1/E2/E3, learned residual |
| 14 | Stage E fusion, consistency loss, λ schedule |
| 15 | Hyperparameter sweep, model selection on val |

**Gate to P4:** val depth MAE < 0.4 mm, camber MAE < 0.4°, misalignment AUC > 0.85.

**If you miss the gate:** the fallback is to drop the toe head and ship a wear + camber system. Camber is far more observable from below. A tight, well-evaluated two-output system beats a sprawling three-output one that doesn't work. Decide by week 16 at the latest.

---

## P4 · Weeks 14–19 — Prove it

| Week | Tasks |
|---|---|
| 16 | Robustness sweep: speed, light, surface, load, inflation |
| 17 | Baselines: human technician, Depth Anything, single-frame CNN, classical |
| 18 | Ablations 1–8, three seeds each |
| 19 | Statistical analysis: bootstrap CIs, paired tests, Bland–Altman |

**Deliverable:** every table in `06_EVALUATION.md` filled in.

---

## P5 · Weeks 18–22 — Ship

| Week | Tasks |
|---|---|
| 18–19 | ONNX export, numerics validation, TensorRT on Jetson |
| 20 | Live rig demo: drive over, report card in < 2 s |
| 21 | Mobile app (TFLite), with separately calibrated (wider) intervals. **Gold rack test set opened — once.** |
| 22 | HF release: `grip-net` model card, `grip-roll` datasheet, Gradio Space, GitHub repo cleanup |

---

## P6 · Weeks 20–24 — Write

| Week | Tasks |
|---|---|
| 20 | Paper outline + figures list. **Start writing before results are final** — figures drive analysis. |
| 21 | Methods + related work (easiest sections, write them first) |
| 22 | Results, failure analysis, qualitative figures |
| 23 | Discussion, limitations, intro/abstract (write the abstract last) |
| 24 | Thesis document, defence slides, submission |

### Venue targets

| Venue | Fit | Deadline pattern |
|---|---|---|
| **IEEE Access** | Strong — applied systems, fast review | Rolling |
| **IEEE T-IV / T-ITS** | Strong — intelligent vehicles | Rolling |
| **WACV** (applications track) | Very good fit for this kind of work | ~Aug |
| **CVPR/ICCV workshops** (AI for Autonomous Driving, Vision for All Seasons) | Realistic and well-matched | ~Mar / ~Jun |
| **arXiv preprint** | Do this regardless, at week 23 | — |

**Recommendation: aim at a CVPR/ICCV workshop or WACV applications track first**, then extend to IEEE Access. Workshop papers are 8 pages, reviewed by people who appreciate systems work, and a strong workshop paper is a realistic and genuinely good capstone outcome. Aiming at a main-track top-tier venue on a 24-week solo project sets you up for a demoralising rejection.

---

## Weekly discipline

Every Friday, 30 minutes:

1. Update the status checkboxes in `README.md`
2. Append to `docs/LOGBOOK.md`: what worked, what broke, what you decided and why
3. Push everything to HF + git
4. Look at next week's tasks; move anything unrealistic **now**, not later

The logbook is not bureaucracy. When you write the paper in week 22 you will need to remember why you chose `λ = 0.3` in week 14, and you will not.

---

## Critical path

```
FTIR works (W1) ──► rig calibrated (W4) ──► jig built (W5) ──► bulk data (W7-10)
                                                                      │
                                                                      ▼
                                             model trained (W15) ──► results (W19)
                                                                      │
                                                                      ▼
                                                              paper submitted (W24)
```

**Everything depends on week 1 and week 4.** If FTIR fails in week 1, or calibration isn't done by week 4, the whole schedule slips. Front-load effort there. The Blender pipeline, the mobile app, and the Jetson deployment are all cuttable; the rig and the data are not.

---

## What to cut if you fall behind

In order, cut these:

1. Mobile app (nice demo, no research value)
2. Jetson deployment (a laptop demo proves the same point)
3. Synthetic data volume (20k → 5k renders)
4. Toe estimation (keep camber, which is much more observable)
5. Damage detection head
6. Ablations 5–8

**Never cut:** the noise floor study, the gold test set, the calibration, the failure analysis, or the three-seed ablations on 1–4. Those are what make the results believable.
