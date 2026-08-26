# 07 — Roadmap and Team Plan

**Capstone Fall-Sem 2026–27 · Review-1 complete · four members.**

---

## Phase structure

| Phase | Theme | Hard deliverable | Gate to proceed |
|---|---|---|---|
| **P0** Prove | Viewpoint + illumination feasibility | v0 photos: grooves, shoulders, sipe, TWI visible | Can you see a TWI bar? |
| **P1** Build | Rig, illumination, calibration | Calibrated v1, photometric stereo working | RMS < 0.3 px; zero-toe 0 ± 0.2° |
| **P2** Pilot | 30–50 tyres, annotation workflow | Pilot dataset + SegFormer-B0 baseline | Taxonomy frozen; κ > 0.70 |
| **P3** Collect | Main dataset | 250–400 tyres, jig clips, scrapyard set | Tyre-level splits verified |
| **P4** Model | All heads trained | Full pipeline, val targets met | Camber MAE < 0.5°, critical-wear recall > 0.90 |
| **P5** Prove | Evaluation | All tables in `06_EVALUATION.md` filled | 3-seed ablations complete |
| **P6** Ship | Report, demo, release | Review-3, paper draft, HF release | — |

Phases overlap deliberately: collection runs in the background of modelling; writing starts before results are final.

---

## P0 · Prove the viewpoint (2 weeks) — start now

| Task | Owner |
|---|---|
| Phone + tyre + side lighting; photograph grooves, shoulders, sipe, **TWI bar** | Hardware |
| **Hand-torch photometric test** — 4 light positions, do grooves pop? | Hardware |
| Measure achieved mm/px against a ruler in frame | Hardware |
| Start longitudinal gauge study on 6 vehicles | Data |
| Order v1 parts (camera, lens, LEDs, polarisers — long lead times) | Hardware |
| Read Tier-1 papers (`09_RELATED_WORK.md §7`) | All |
| Set up repo, environment, `verify_env.py` on all four machines | ML |

**GO/NO-GO:** if you cannot see a TWI bar and a sipe in a v0 photo, the resolution budget (`02_RIG_BUILD.md §2`) is wrong. Fix the sensor or crop tighter **before** ordering anything else.

**Kill-criterion:** if photometric stereo shows no visible improvement over flat light in the hand-torch test, drop it to an ablation and proceed with flat illumination + classical enhancement. Better to know in week 1.

---

## P1 · Build and calibrate (3 weeks)

- Frame built, camera rigidly mounted, all settings locked
- 4-LED array + drivers, GPIO strobing verified
- Cross-polarisation verified on a wet tyre
- Intrinsics (RMS < 0.3 px) · extrinsics (100 mm within 0.5 mm) · light directions (< 5°)
- Flat-field routine
- `qc.py` written **and actually used**
- Alignment jig built; verniers verified to 0.05°
- **Zero-toe bias test: 0 ± 0.2° over 20 clips**

**Do not start bulk collection with an uncalibrated rig.** You will collect 300 unusable tyres and lose a month.

---

## P2 · Pilot (3 weeks)

- 30–50 unique tyres, full metadata
- **Annotation guideline document written first**, with example images per class
- SAM2-assisted annotation workflow running (`03_DATA.md §5`)
- Second annotator labels 100 items → **Cohen's κ reported**
- Label taxonomy frozen
- SegFormer-B0 baseline trained; boundary F-score measured
- Classical baseline pipeline (CLAHE, Scharr, Gabor, structure tensor, RANSAC, LK) with debug visuals
- Scrapyard tyres collected

**Gate:** taxonomy frozen, κ > 0.70, annotation throughput measured. If annotating one tyre takes more than ~15 minutes, fix the workflow before scaling — that is the difference between 300 tyres and 60.

---

## P3 · Main collection (5 weeks, overlapping P4)

- 250–400 unique tyres, balanced across the diversity axes
- ~800 jig clips, uniform over camber × toe × load
- One alignment-rack session, 20–30 vehicles → **gold test set, sealed**
- Longitudinal visits continuing
- Ranking-pair UI built; 3,000 pairs labelled
- Dataset v0.1 pushed to HF (private)

**Collection always takes ~2× longer than planned.** If behind, cut jig clip count before cutting unique-tyre count.

---

## P4 · Modelling (6 weeks)

| Order | Build |
|---|---|
| 1 | Kaggle notebook infra: resume, HF sync, dual-T4 DDP. **Run the 200-step resume-equivalence test** |
| 2 | Frame-quality gate |
| 3 | SegFormer-B2 with tiled fine pass, boundary + clDice |
| 4 | ConvNeXt-V2-T ordinal + multi-label + ranking + monotonicity |
| 5 | Wear heatmap, damage head, TWI anchor |
| 6 | HRNet-W18 landmarks on jig; **analytic-only alignment baseline reported first** |
| 7 | Residual MLP, heteroscedastic |
| 8 | Registration + partial unrolling |
| 9 | PatchCore |
| 10 | Temporal fusion, conformal, decision logic |

**Gate:** camber MAE < 0.5°, critical-wear recall > 0.90, segmentation boundary F > 0.65.

**If the toe head never works** — likely — reframe it as binary screening only and say so. A tight, well-evaluated camber + wear system beats a sprawling three-output one that doesn't. Decide this at the P4 midpoint, deliberately, not in a panic at the end.

---

## P5 · Evaluation (3 weeks)

Robustness sweep → baselines → 15 ablations × 3 seeds → bootstrap CIs and paired tests → stratified tables → **gold set opened once** → failure analysis → qualitative figures.

---

## P6 · Ship (4 weeks, overlapping P5)

- Review-3 report and slides
- Live demo (laptop is sufficient; Jetson optional)
- Optional app (`11_APP.md`)
- HF release: model card + dataset datasheet
- Paper draft — **start writing with empty results tables.** Deciding what the figures must show changes what analysis you run, while there is still time to run it

### Venue targets

| Venue | Fit |
|---|---|
| **IEEE Access** | Strong — applied systems, rolling submission |
| **IEEE T-IM / T-ITS** | Strong — instrumentation and measurement |
| *Measurement* (Elsevier) | Very good — [6] and [10] published there |
| **WACV applications track** | Realistic and well-matched |
| CVPR/ICCV workshops (Vision for All Seasons, AI for Autonomous Driving) | Realistic |
| **arXiv preprint** | Do this regardless |

**Recommendation:** target *Measurement* or IEEE Access. Both publish exactly this kind of applied-vision measurement work, and two of your key references are already in *Measurement* — that is a signal about fit.

---

## Team work split

Four members. Split by **subsystem**, not by "everyone does a bit of everything" — that produces four half-finished modules.

| Role | Owns | Deliverables |
|---|---|---|
| **Hardware & Calibration** | Rig, illumination, photometric stereo, calibration, `qc.py` | Calibrated rig; light-direction calibration; zero-toe verification; capture software |
| **Data & Annotation** | Collection protocol, SAM2 workflow, labels, splits, gauge GT, jig | Dataset + datasheet; κ measurement; noise floor study; ranking UI |
| **Perception ML** | Quality gate, SegFormer, ConvNeXt heads, PatchCore | Segmentation + wear + damage models; training notebook |
| **Geometry & Evaluation** | Landmarks, classical geometry, analytic angles, residual MLP, fusion, conformal, all metrics | Alignment module; evaluation harness; ablation tables; failure analysis |

**Shared, non-negotiable:** everyone reads the Tier-1 papers; everyone can run `verify_env.py`; everyone commits weekly.

**Interfaces between roles — agree these in week 1 and write them down.** Most four-person projects fail at the seams, not inside a module:

```
Hardware → Data        : clip format + metadata JSON schema
Data → Perception      : label format, split files, dataloader contract
Perception → Geometry  : mask + landmark output format, confidence fields
Geometry → Evaluation  : prediction record schema
```

---

## Weekly discipline

Every Friday, 30 minutes as a group:

1. Update the status checkboxes in `README.md`
2. Append to `docs/LOGBOOK.md` — what worked, what broke, what was decided and **why**
3. Commit and push
4. Move anything unrealistic in next week's plan **now**, not later

The logbook is not bureaucracy. When you write Review-3 you will need to remember why a loss weight was set the way it was, and you will not.

---

## Critical path

```
TWI visible (P0) ──► rig calibrated (P1) ──► annotation workflow proven (P2)
                                                        │
                                                        ▼
                              main dataset (P3) ──► models (P4) ──► results (P5)
                                                                          │
                                                                          ▼
                                                                 Review-3 / paper (P6)
```

**Everything depends on P0 and the P2 annotation throughput.** If annotation is slow, the dataset shrinks, and every downstream number weakens. Front-load effort there.

---

## What to cut if you fall behind

In order:

1. Optional app
2. Jetson deployment (a laptop demo proves the same point)
3. RAFT-Small (keep Lucas–Kanade)
4. PatchCore
5. Partial unrolling (fall back to per-frame analysis with coverage reported as 1 frame)
6. **Toe estimation** — keep camber, which is far more observable
7. Ablations 6–15

**Never cut:** the noise floor study · the gold test set · calibration · tyre-level splits · 3-seed ablations on the top 5 · the failure analysis. Those are what make the results believable.
