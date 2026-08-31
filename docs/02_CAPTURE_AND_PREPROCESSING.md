# 02 — Capture Guidance and Preprocessing

> **No hardware is being built.** We work from `final_v1` plus separately-captured video.
>
> This document keeps the two things that remain useful: **how to photograph a tyre so the images are worth training on**, and **which image-processing operations belong in the pipeline and why**. Everything about rigs, illumination arrays and calibration has been removed.

---

## 1. How the existing data was captured

Recorded for the report, and it matters for interpreting every result:

| | |
|---|---|
| **Position** | Camera **below and in front of the vehicle**, viewing one tyre |
| Framing | Tread crown face-on, tyre roughly vertical in a portrait frame |
| Stills (`final_v1`) | 888 raw → 418 unique, standardised to 1152 × 1536 |
| **Capture window** | One 22-minute window on 2026-08-25 — **all 12 sessions** |
| Video | Separate sessions, different conditions, **same below-front viewpoint** |
| Calibration | **None.** No intrinsics, no pose, no scale reference |
| Handling | Handheld, not rigidly mounted |

### What follows from "no calibration"

- **No millimetres.** Scale is roughly 0.21–0.24 mm/px but varies between sessions and is not recorded. Any size claim is unfalsifiable
- **No alignment.** Toe and camber are measured relative to a calibrated vertical and travel direction. Neither exists
- **Framing is a confound, not a constant.** Some frames crop the shoulders off; others include road, wall, vegetation, a parked car. This is why the ROI ablation (`04 §9` factor 5) is the one I would run first

### The video clips

Not training data, not a quantitative test set. Their role:

| Use | Why it is valid |
|---|---|
| Qualitative robustness | Does a stills-trained model survive real video frames? |
| **Temporal consistency** | Prediction flicker across frames of *the same tyre* is direct evidence of shortcut reliance |
| Held-out domain | Different conditions ⇒ the strongest generalisation check available |

Temporal consistency is the useful one: a model reading tread structure should give a stable answer across consecutive frames of one tyre. A model reading dirt, glare or background will jump around. That is a real measurement, and it needs no labels.

---

## 2. How to photograph tyres for `final_v2`

If anyone captures more data — and they should (`12 §8`) — these are the rules that would most improve the dataset, in priority order.

### Priority 1 — more tyres, not more frames

**One photograph of forty tyres beats forty photographs of one tyre.** The current dataset is 4,598 files and 12 tyres, and that is the binding constraint on everything.

Target: **40+ additional unique tyres**, ideally across different days, sites and lighting.

### Priority 2 — put a scale reference in frame

A printed ruler, a coin, or a small ChArUco marker laid against the tyre. Costs nothing, and converts pixels into millimetres. Without it, no size claim is checkable.

### Priority 3 — measure the tread depth

A digital gauge (₹900, 0.01 mm) at 3–6 circumferential stations across every main groove. Median of 3 readings per point. Record pressure, size, wheel position, contamination, temperature.

**This single item converts the project from mileage-proxy classification to wear measurement.** Nothing else has comparable leverage.

### Priority 4 — consistent framing

| Rule | Why |
|---|---|
| **Full tread width, both shoulders, always in frame** | Lateral wear profile is impossible otherwise; alignment doubly so |
| Camera roughly perpendicular to the tread crown | Reduces foreshortening variance |
| Consistent distance | Makes scale comparable even without a marker |
| Tyre fills 60–80% of the frame | Enough context to segment, not so much that tread resolution is wasted |

### Priority 5 — lock the camera settings

| Setting | Why |
|---|---|
| **Fixed exposure** | Auto-exposure changes look exactly like tread-condition changes |
| **Fixed white balance** | Same problem in colour |
| Fixed focus | Blur varies per-frame otherwise |
| Highest resolution available | Sipes are 0.3–1 mm; you cannot recover detail later |
| **RAW or highest-quality JPEG** | Recompression destroys fine groove edges |

Record the settings in the metadata. If they change between sessions, that becomes a confound you cannot remove afterwards.

### Priority 6 — vary the nuisance conditions deliberately

The current data has **zero** condition diversity — one 22-minute window. Deliberately vary: time of day, overcast vs direct sun, dry vs damp, clean vs dirty, different locations, different phones.

Nuisance variation you *collect* is robustness. Nuisance variation you *don't* is an unmeasurable domain gap.

### Priority 7 — record identity metadata

`vehicle_id` · `tyre_id` · `capture_session_id` · axle/side · brand · model · size · tread pattern · DOT date · odometer.

`tyre_id` is what makes correct grouping possible. Brand and tread pattern are what let you *measure* the tread-identity confound instead of guessing at it.

### What to avoid

| Don't | Because |
|---|---|
| Burst-shoot one tyre 60 times | 60 near-duplicates, still one independent observation |
| Photograph in a single session | Capture order becomes correlated with class |
| Auto-everything | Exposure changes mimic wear |
| Crop tightly in-camera | You can crop later; you cannot un-crop |
| Photograph only the tyres you can reach easily | Systematic bias toward one vehicle type |

---

## 3. Preprocessing pipeline

Filters are not decoration. Each exists to make a **specific downstream step** work. Keep a raw normalised RGB path alongside; enhanced images are **parallel channels**, never replacements.

### Order

```
1  decode without re-compression        5  CLAHE on the L channel
2  EXIF orientation                     6  Scharr gradients
3  deterministic resize (val: no crop)  7  Gabor / structure-tensor orientation
4  exposure / colour normalisation      8  black-hat (dark grooves)
                                        9  ROI crop (SAM2 or CAM-guided)
```

### The table

| Technique | Verdict | Config | Warning |
|---|---|---|---|
| **CLAHE** (LAB `L`) | ◆ Ablation | clip ≈ 2, grid 8×8 | Parallel channel only. Strong CLAHE **amplifies dirt into fake cracks**. The dataset's own guidance calls this an ablation, not an assumed improvement — agreed |
| **Bilateral / guided filter** | ○ | radius 3–5 px | Larger radii erase sipes |
| **Scharr gradients** | ★ | 3×3 x and y | More rotationally accurate than Sobel at 3×3. Denoise first |
| **Structure tensor** | ★ | σ matched to groove width | Reject low-coherence pixels rather than inventing a direction |
| **Gabor bank** | ★ | 0, ±15, ±30, ±45, 90°; multiple λ | Restrict to the tyre mask; normalise each response |
| **Black-hat morphology** | ★ | oriented kernel ≈ groove width | Large kernels mistake shadows for grooves |
| **Canny** | ◆ | thresholds tuned on **validation only** | An input to geometry, not a wear detector |
| **Grayscale** | ★ **as a test** | — | Not an improvement — a **shortcut diagnostic**. Colour carries no wear information (`04 §9` factor 10) |
| **ROI crop** | ★ | SAM2 or CAM-guided | Expected to be the single largest preprocessing effect |
| **Median filter** | ○ | 3×3, conditional | Rounds narrow structures |

★ core · ◆ ablate · ○ optional

### Rejected as defaults

Large Gaussian blur (destroys the target signal) · aggressive sharpening (manufactures fake cracks) · global histogram equalisation (over-amplifies bright regions) · large morphological kernels (change the apparent width of grooves) · fixed Canny thresholds across sources · **feeding only edge maps to the network** (discards shading and texture evidence entirely).

---

## 4. Augmentation

The dataset ships a validated policy in `config/augmentation_policy.yaml` and pre-generated derivatives. **Use it as the baseline.** Additional augmentation is a Stage-B factor, not a default.

### Already applied, per derivative

Aspect-preserving crop/letterbox (all 4,180) · brightness/contrast (2,939) · **horizontal flip (2,122)** · rotation ±4° (1,877) · gamma (1,207) · saturation (1,054) · JPEG recompression (814) · Gaussian noise (723) · CLAHE (510) · coarse dropout (444) · unsharp (440) · Gaussian blur (402) · box/motion blur (359).

### Deliberately disabled — and correctly

Vertical flip · elastic deformation · aggressive perspective · aggressive colour jitter · MixUp · CutMix.

These corrupt physically meaningful tread geometry or conceal small wear evidence. If you ablate MixUp/CutMix in Stage B, report that you are **deliberately violating the dataset's policy** and why.

### ⚠ The horizontal-flip rule

> **Horizontal flip is acceptable for the mileage-proxy task only.**
>
> It is **forbidden** for signed toe/camber, alignment, or inner-versus-outer wear, because flipping swaps left and right and reverses the sign of the angle.

Write `tests/test_augment_signs.py` now, even though alignment is deferred: assert that a horizontal flip negates any signed lateral label, and that flipping twice is the identity. This bug class is invisible until it has cost a month.

### Validation

**Never apply stochastic augmentation to validation or test.** Deterministic resize and normalise only. The dataset states this; it is worth restating because it is the easiest rule to break by accident when refactoring a dataloader.

---

## 5. Model input

| | |
|---|---|
| Clean images | 1152 × 1536 (validation reads these) |
| Derivatives | 768 × 768 (training only) |
| Normalisation | ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` |
| Default input | 384 × 384 |
| Sweep | 224 · 384 · 512 · 768 (`04 §9` factor 4) |

**Compare resolutions experimentally.** Fine tread detail disappears at low resolution, and at ~0.22 mm/px a downsample to 224 throws away most of what distinguishes `mid` from `high`. I expect resolution to be one of the larger effects in Stage B — but it is a measurement, not an assumption.

---

## 6. What is deliberately not here

Removed with the approach change, kept only in git history:

- Camera and lens selection, mounting geometry, bill of materials
- Photometric-stereo illumination arrays, cross-polarisation
- ChArUco calibration, intrinsics, extrinsics, light-direction calibration
- Laser triangulation and structured light
- The alignment jig

None of it is needed for the current phase. If the project later returns to alignment or metric depth, recover it from git — it was correct for that purpose, just not for this one.
