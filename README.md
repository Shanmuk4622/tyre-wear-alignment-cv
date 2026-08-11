# GRIP — Ground-level Rolling Inspection Pipeline

**Simultaneous tyre-wear quantification and wheel-alignment estimation from a single rolling pass over a ground-embedded optical sensor.**

Capstone project · Shanmukesh · Aug 2026 – Jan 2027

---

## The one-paragraph pitch

A car rolls at walking pace over a glass plate set flush into the ground. Underneath sits a camera looking straight up, a light-guide illuminator, and a laser line. In the ~1.5 seconds the wheel takes to cross the plate, the system reconstructs the tyre's tread surface in metric units, measures the wheel's toe and camber angles from the rolling geometry, reads the contact-pressure footprint, and returns a calibrated report: *remaining tread depth per rib, wear-pattern class, alignment angles with uncertainty bounds, and a plain-language cause diagnosis.* No jacking, no clamps, no reflective targets, no driver involvement.

---

## Why look from *below*? (This is the whole idea)

Three physical facts make the ground-level view not merely convenient but **strictly superior** for this task:

### 1. The contact patch has zero motion blur — for free

A wheel rolling without slipping has its **instantaneous centre of rotation at the contact point**. The top of the tyre moves at `2v`; the contact patch moves at **`0`**.

| Speed | Exposure | Blur at top of tyre | Blur at contact patch |
|---|---|---|---|
| 10 km/h | 1 ms | 5.6 mm | ~0 mm |
| 20 km/h | 1 ms | 11.1 mm | ~0 mm |
| 30 km/h | 1 ms | 16.7 mm | ~0 mm |

Every side-view or wheel-arch rig fights motion blur with expensive strobes and global shutters. **We get sharp sub-millimetre tread imagery from a cheap camera because we image the one part of the tyre that is standing still.** This single insight is what makes the "detect the smallest change" ambition physically achievable.

### 2. Contact with glass enables FTIR — a pressure map, not just a picture

Inject light into the *edge* of the glass plate. It bounces along inside by total internal reflection. Where the tyre presses against the top surface, TIR is **frustrated** and light scatters down into the camera. Result: the tread ribs that are actually carrying load **glow bright white** against a pure black background.

This is the same principle as an optical fingerprint scanner. Applied here it gives:

- A binary-crisp contact footprint with no illumination artefacts
- Brightness ∝ local contact pressure
- **Camber** → lateral pressure asymmetry across the footprint
- **Toe** → longitudinal shear distortion and leading/trailing edge asymmetry
- Inflation pressure and load → footprint area and aspect ratio

> **Two things the research pass established (`09_RELATED_WORK.md`):**
>
> 1. **This is not novel, and that's good.** FTIR tyre-footprint imaging is validated prior art ([Cabrera et al., *Sensors* 2017](https://doi.org/10.3390/s17040707)) — they even published fitted curves for *contact area vs camber angle*. The physics is peer-reviewed; the signal provably exists. **Our novelty is inverting it**: they set camber and measured the footprint on a static lab bench; we observe the footprint of a moving vehicle and estimate unknown alignment.
> 2. **You need a clear interface film on the glass.** Black carbon-filled rubber absorbs the coupled light rather than scattering it back, so bare rubber on bare glass gives almost no contrast. Automotive paint-protection film solves it for ~₹400/m². This is a required design change — see `09_RELATED_WORK.md §3`.

### 3. Toe and camber are *directly* observable from below

From an upward-looking calibrated camera you can recover the wheel's mid-plane from the sidewall silhouette and the bead/rim ellipse. The vehicle's direction of travel is fixed by the rig's own coordinate frame. Then:

- **toe** = angle between (wheel mid-plane ∩ ground plane) and travel direction
- **camber** = angle between wheel mid-plane and vertical

No markers, no clamp-on heads, no wheel-runout compensation spin. The rig *is* the reference frame.

---

## The intellectual core: wear and alignment are the same problem

This is the reframing that turns a two-model engineering exercise into a research contribution.

```
        alignment (geometry, measurable NOW)
                   │
                   │  causes, integrated over ~10,000 km
                   ▼
        wear pattern (history, measurable NOW)
```

A misaligned wheel writes its own diagnosis into the rubber. Excess toe carves feathered rib edges. Excess camber grinds one shoulder away. These are not correlated by accident — they are linked by tyre-contact mechanics.

So GRIP trains **one network with two heads and a physics-consistency loss** that ties them together:

```
L = L_depth + L_pattern + L_geometry + λ · L_consistency

L_consistency enforces:
    lateral wear gradient across tread width  ↔  predicted camber
    rib-edge sharpness asymmetry              ↔  predicted |toe|
    centre-vs-shoulder depth ratio            ↔  inflation history
```

**Why this matters:** alignment labels are expensive (you need a workshop rack). Wear labels are cheap (a ₹300 depth gauge). The consistency loss lets the abundant wear supervision *regularise* the scarce alignment head — a semi-supervised trick grounded in physics rather than in augmentation heuristics.

---

## Novelty claims (what goes in the paper)

Audited against the literature on 2026-08-09. Honest grades — full analysis in `09_RELATED_WORK.md §7`.

| # | Claim | Verdict |
|---|---|---|
| 1 | FTIR footprint imaging for tyres | ✗ **Prior art** (Chodera; Cabrera 2017) |
| 1b | FTIR **in the road plane, in motion, inverted to estimate unknown alignment** | ✓ Novel |
| 2 | Zero-blur imaging via the rolling constraint | ~ Novel framing, weak claim — use as *justification*, not contribution |
| 3 | Laser as train-time teacher; RGB-only at inference | ✓ Novel application |
| 4 | Consistency loss coupling wear to alignment — **with a brush-model-derived link function** | ✓ Novel as reframed |
| 4b | **Disagreement diagnostic**: distinguishes *recent* from *chronic* misalignment | ✓✓ **Strongest claim — nothing comparable found** |
| 5 | TWI bars as an in-image metric ruler | ✓ Novel, small |
| 6 | **GRIP-Roll** dataset | ✓✓ Solid, durable |
| 7 | Conformal intervals | ~ Good practice, not a contribution |
| 8 | **4× improvement over the ±1.5 mm image-based SOTA** | ✓ Strong if achieved |

**Abstract headline order:** 4b → 8 → 6 → 1b. Keep 2 and 7 in Methods.

---

## What the system outputs

```
┌─ GRIP INSPECTION REPORT ─────────────────────────────┐
│ Wheel: front-left        Pass: 3    Speed: 8.2 km/h  │
│                                                       │
│ TREAD DEPTH (mm, ±0.31 @ 90% conformal)              │
│   outer shoulder  3.1  ▓▓▓░░░░░  ⚠ approaching TWI   │
│   outer rib       4.4  ▓▓▓▓▓░░░                       │
│   centre rib      5.9  ▓▓▓▓▓▓▓░                       │
│   inner rib       6.2  ▓▓▓▓▓▓▓░                       │
│   inner shoulder  6.4  ▓▓▓▓▓▓▓▓                       │
│                                                       │
│ WEAR PATTERN:  outer-shoulder wear + light feathering │
│                                                       │
│ ALIGNMENT (screening estimate)                        │
│   camber   -1.42°  ±0.28°   ⚠ outside spec           │
│   toe      +0.21°  ±0.19°   ok                        │
│                                                       │
│ DIAGNOSIS: wear gradient and measured camber agree.   │
│   Likely excessive negative camber, front-left.       │
│   → Recommend alignment check. Confidence: high.      │
│                                                       │
│ REMAINING LIFE: ~11,400 km to 1.6 mm at current rate  │
└───────────────────────────────────────────────────────┘
```

---

## Repository map

| Path | What it is |
|---|---|
| `README.md` | You are here |
| `ENVIRONMENT.md` | Setting up the `cv_conda` environment |
| `environment.yml` | Conda spec |
| `docs/01_CONCEPT.md` | Deep dive on the idea, the physics, the maths |
| `docs/02_RIG_BUILD.md` | Hardware, bill of materials, optics, calibration |
| `docs/03_DATA.md` | Collection protocol, the alignment jig, synthetic pipeline, public sets |
| `docs/04_MODEL.md` | Network architecture, losses, training recipe |
| `docs/05_TRAINING_KAGGLE_HF.md` | Kaggle dual-T4 + Hugging Face resumable workflow |
| `docs/06_EVALUATION.md` | Metrics, ablations, robustness protocol, baselines |
| `docs/07_ROADMAP.md` | 24-week plan, milestones, kill-criteria |
| `docs/08_RISKS_AND_MY_OPINION.md` | My honest read: what will work, what probably won't |
| `docs/09_RELATED_WORK.md` | Literature audit, novelty grading, **required design corrections** |
| `docs/LOGBOOK.md` | Weekly record. Fill it in every Friday. |

**Read them in that order.** `08` is the one to read if you only read one — it is where I tell you what I actually think. `09` is the one that will change what you build.

---

## Quickstart

```bash
conda activate cv_conda            # always. every python command in this repo.
python -c "import torch, cv2; print(torch.__version__, cv2.__version__)"
```

Full setup: see `ENVIRONMENT.md`.

---

## Status

- [x] Concept and novelty framing
- [x] Physics sanity checks (blur, coverage, resolution)
- [ ] Rig v0 built
- [ ] Alignment jig built
- [ ] First 100 real passes captured
- [ ] Synthetic pipeline
- [ ] Baseline model
- [ ] Paper draft

---

## References consulted

- [UVeye — drive-through vehicle inspection](https://uveye.com/how-it-works/)
- [Automatic and Accurate Vision-Based Measurement of Camber and Toe-In Alignment (IEEE Access)](https://ieeexplore.ieee.org/document/9926077/)
- [Design and Assessment of a Machine Vision System for Automatic Vehicle Wheel Alignment](https://journals.sagepub.com/doi/10.5772/55928)
- [Deep learning-based instance segmentation for detection of tire tread area](https://www.sciencedirect.com/science/article/abs/pii/S2950425225000325)
- [Efficient Tire Wear and Defect Detection Based on Deep Learning (code)](https://github.com/wisetrue95/Tire)
- [Michelin — Irregular Tire Wear 101](https://business.michelinman.com/tips-suggestions/irregular-tire-wear-101)
- [Les Schwab — How to Read Tire Wear Patterns](https://www.lesschwab.com/article/tires/how-to-read-tire-wear-patterns.html)
- [Hybrid Synthetic Data Generation with Domain Randomization for Part Inspection](https://arxiv.org/html/2512.00125v1)
