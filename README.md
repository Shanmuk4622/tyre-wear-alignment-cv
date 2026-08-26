# Vision-Based Detailed Tyre-Wear Recognition and Single-Wheel Alignment Screening

**Capstone Project · Fall-Sem 2026–27 · Department of AI & ML, SCOPE, VIT-AP**

| Reg. No. | Name |
|---|---|
| 23BCE20070 | Bonala Shanmukesh |
| 23BCE7016 | Gunnamneni Nehru |
| 23BCE7148 | GV Manu Rohith |
| 23BCE7749 | Nettem Harish Kumar |

**Guide:** Dr. E. Sreenivasa Reddy, Professor-HAG, SCOPE, VIT-AP

---

## What this is

A low-mounted RGB camera sits ahead of one wheel, aimed backward and slightly upward at the front face of the tread. From a short video it must:

1. **Recognise the tyre in detail** — tread crown, both shoulders, main grooves, ribs/blocks, sipes, TWI bars, visible damage
2. **Decide whether it is worn** — ordinal severity, multi-label wear pattern, and a *localised* wear map, not a binary verdict
3. **Screen single-wheel alignment** — camber and individual toe from calibrated geometry, with uncertainty
4. *(optional)* Feed a connected app that reports the evidence

The system is **not** one black box mapping an image to `worn / not worn`. Wear is an appearance problem; alignment is a geometry problem. They need different evidence and different methods, and the architecture keeps them separate on purpose.

---

## The honest scope

This matters more than any accuracy number, so it comes first.

| Claim we make | Claim we do **not** make |
|---|---|
| Detailed, localised wear and damage recognition | Certified tread depth in millimetres from RGB alone |
| Ordinal wear state + calibrated relative depth | Replacement for a digital tread gauge |
| **Single-wheel** camber and individual toe *screening* with confidence intervals | Total axle toe, thrust angle, caster, or four-wheel alignment |
| A refusal output when the image or geometry is inadequate | A confident answer on every frame |

RGB video is appropriate for wear classification and alignment *screening*. Defensible millimetre depth and workshop-grade angles require metric sensing — structured light, a laser line, stereo, or RGB-D. Those are scoped as ablations and extensions, not assumed away.

The system returns one of four verdicts, always:

```
PASS · MONITOR · INSPECT · UNABLE_TO_MEASURE
```

`UNABLE_TO_MEASURE` is a first-class output. A safety system that never admits doubt is a liability.

---

## Problem formulation

Given a calibrated short video `V = {I₁ … I_T}` of one tyre, with camera intrinsics `K` and rig-to-world calibration `C`:

```
f(V, K, C) → { M_tread, W_map, P_pattern, D_damage, γ, τ, U }
```

- `M_tread` — tyre / tread / shoulder / groove / sipe / TWI / damage masks
- `W_map` — localised wear map over the observed tread
- `P_pattern` — multi-label wear-pattern probabilities
- `D_damage` — visible defect regions
- `γ`, `τ` — camber and individual toe
- `U` — uncertainty, quality flags, and observed-circumference percentage

---

## Why the front view is hard, and what follows from it

The camera sees the tread crown and both shoulders. It does **not** see the full wheel, the hidden sidewall, or the whole circumference in one frame. The tyre is curved, dark, deformable near the road, and specular when wet. Small changes in illumination, pressure, load, steering and camera pose can all masquerade as changes in wear or alignment.

Four consequences drive the whole design:

**1. Illumination is the sensor, not a nuisance.** Carbon-black rubber has ~4–8% albedo and grooves are defined by *shape*, not colour. A single flood light throws that information away. We use **multi-directional illumination (photometric stereo)** to recover surface normals, which makes groove depth, cracks and wear geometry explicit. This is the largest single upgrade over a plain RGB pipeline — see `docs/10_VISION_TECHNIQUES.md §2`.

**2. Video beats a still image.** Rotation exposes more circumference, gives repeated angle estimates, and lets frame quality be a *selection* criterion rather than a hope. We register accepted frames and build a **partial unrolled tread map**, reporting the observed circumference percentage and leaving unseen areas explicitly unknown.

**3. Alignment must come from calibrated geometry.** A CNN regressing angles from pixels will learn camera tilt, approach position and background shortcuts. We use learned landmarks + classical orientation evidence → analytic angle → small learned residual → temporal fusion.

**4. Camber is observable; toe barely is.** Camber produces visible lateral tilt. Toe is a small yaw easily confused with camera yaw or an angled approach. So camber gets a continuous estimate; toe gets a screening estimate with an honest interval.

---

## Architecture

```
camera video
    │
    ├─▶ Model 0 · frame-quality gate ──────────── reject blur/glare/occlusion
    │
    ├─▶ fixed calibrated ROI  (YOLO11n-seg only if position varies)
    │
    ├─▶ Model 1 · SegFormer-B2 segmentation
    │        tyre · tread crown · shoulders · grooves · ribs · sipes · TWI · damage
    │
    ├─▶ frame registration + partial tread unrolling
    │        Lucas–Kanade + RANSAC   (RAFT-Small as ablation)
    │
    ├─▶ Model 2 · ConvNeXt-V2-Tiny multi-task
    │        ordinal severity · multi-label pattern · wear heatmap
    │        damage masks · confidence · (optional metric depth)
    │
    ├─▶ Model 3 · PatchCore  ──────────────────── unknown-anomaly flagging
    │
    ├─▶ Model 4 · HRNet-W18 landmarks + Scharr/Gabor/structure-tensor
    │        → analytic camber & toe → residual MLP → uncertainty
    │
    └─▶ temporal fusion + conformal intervals
             PASS / MONITOR / INSPECT / UNABLE_TO_MEASURE
```

**One sentence:** SegFormer-B2 understands the tyre's structure, ConvNeXt-V2-Tiny describes its wear, HRNet-W18 plus calibrated geometry screens its alignment, and classical filters plus temporal tracking supply stable physical features throughout.

---

## Repository map

| Path | What it is |
|---|---|
| `README.md` | You are here |
| `docs/01_CONCEPT.md` | Problem formulation, what a front view can and cannot observe, scope discipline |
| `docs/02_RIG_BUILD.md` | Camera, **photometric-stereo illumination**, calibration, BOM |
| `docs/03_DATA.md` | Collection protocol, alignment jig, label taxonomy, **SAM2-assisted annotation** |
| `docs/04_MODEL.md` | Full model stack, losses, training order |
| `docs/05_TRAINING_KAGGLE_HF.md` | Kaggle dual-T4 + Hugging Face resumable training spec |
| `docs/06_EVALUATION.md` | Metrics, ablations, robustness, acceptance criteria |
| `docs/07_ROADMAP.md` | Review-2 / Review-3 timeline, team work split |
| `docs/08_RISKS_AND_MY_OPINION.md` | What will work, what probably won't, what to cut |
| `docs/09_RELATED_WORK.md` | Annotated bibliography and novelty audit |
| **`docs/10_VISION_TECHNIQUES.md`** | **Technique catalogue: every CV method considered, with verdicts** |
| `docs/11_APP.md` | Optional camera-connected application |
| `docs/LOGBOOK.md` | Weekly record |
| `docs/GITHUB_SETUP.md` | Repo metadata and push instructions |

**Start with `10`** if you want the technique survey. **Start with `08`** if you want the honest assessment.

---

## Research contributions

1. A low-front, single-tyre **video dataset** with tread masks, fine structures, gauge measurements, wear patterns, visible damage and calibrated alignment labels
2. A **high-resolution multi-task model** producing localised wear and damage evidence, not a binary classification
3. A **video-based partial tread-unrolling** method with explicit circumference-coverage reporting
4. A **hybrid alignment method** — learned landmarks + calibrated analytic geometry + temporal fusion
5. A **confidence-aware comparison of current geometry against accumulated wear**, enabling detection of disagreement and of recent-versus-historical conditions
6. An **experimental comparison** of RGB-only, photometric-stereo, structured-light and RGB-D evidence on the same tyre set
7. An optional **app interface** communicating evidence, uncertainty and inspection status without claiming certified alignment

---

## Quickstart

```bash
conda activate cv_conda            # every python command in this repo
python scripts/verify_env.py
```

Setup: `ENVIRONMENT.md`.

---

## Status

- [x] Review-1 report submitted
- [x] Model stack specified
- [x] Literature audit
- [x] Technique catalogue
- [ ] Camera rig + photometric-stereo illumination built
- [ ] Calibration verified
- [ ] Pilot dataset (30–50 tyres)
- [ ] Alignment jig
- [ ] SegFormer-B2 baseline
- [ ] Main dataset (250–400 tyres)
- [ ] Full pipeline + evaluation
- [ ] Review-2 / Review-3

---

## Key references

Petrovic et al. 2025 (front-view tread segmentation) · Huber et al. 2022 (TireEye, 0.57 mm) · Wang et al. 2019 (structured light, <0.2 mm) · Furferi et al. 2013 (stereo alignment, ~0.025°) · Shi et al. 2026 (RGB-D rim registration, <0.1°) · full annotated list in `docs/09_RELATED_WORK.md`.
