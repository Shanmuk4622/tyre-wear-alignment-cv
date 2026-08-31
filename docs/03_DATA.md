# 03 — Data: Collection, Labels, Annotation

> **The dataset unit is a unique tyre, not a frame.** Everything in this document follows from that one rule.

---

## 0. Where we actually are — `final_v1`

A pilot package exists and is verified: **`D:\Dataset Download\Tire Dataset Prepared\FINAL`**.

| | |
|---|---:|
| Clean unique images | 418 |
| Synthetic derivatives | 4,180 |
| **Independent capture sessions** | **12** |
| Capture window | one 22-minute window, 2026-08-25 |
| Labels | 3-class **mileage proxy** only |
| Tread-depth measurements | **none** |
| Segmentation masks | ✅ **`annotation_v2`** — 418 hand-drawn + 4,180 propagated (v1 propagation was geometrically wrong; see `annotations/README.md §0`) |
| Alignment ground truth | **none** |

Full analysis, capability matrix and difficulty-floor probes: **`12_DATASET_FINAL_V1.md`**.

The rest of this document is the **collection plan that closes the gap**. Read §1 targets as what still needs collecting, not as what exists.

**The binding constraint is 12 sessions.** Not image count, not architecture. Two trivial baselines reach 0.95+ macro-F1 on one fold and collapse to 0.12 on another, because with 1–2 sessions per class per fold, "class" is very nearly "which tyre". More tyres is the only fix.

---

## 1. Dataset targets

| Phase | Unique tyres | Purpose |
|---|---|---|
| ~~**Pilot**~~ ✅ **done — `final_v1`** | **12** | Delivered. Validated the viewpoint; exposed the session-count constraint |
| **Pilot-2 (next)** | **+40 minimum** | Break the tyre-identity confound; add condition diversity; first gauge measurements |
| **Main study** | 250–400 | Train and evaluate all heads |
| **Scrapyard supplement** | 40–60 | Rare classes: worn to cord, cuts, cupping, flat spots |
| **Alignment jig** | 1 wheel × ~800 clips | Balanced camber/toe coverage |
| **Rack-verified** | 20–30 vehicles | Gold alignment test set |
| **Longitudinal** | 6 vehicles × 8–10 visits | Wear-rate, monotonicity supervision |

**More unique tyres beats more frames per tyre.** Several clips per tyre is fine and useful; thousands of near-duplicate frames from 20 tyres is worthless and will produce beautiful, meaningless validation numbers.

### Diversity axes to actively balance

| Axis | Target coverage |
|---|---|
| Brand | ≥ 8 brands; **hold out 2 entirely** |
| Tread family | Symmetric, asymmetric, directional |
| Size | 13″–17″; hold out one size class |
| Wear state | Uniform across the ordinal scale — *not* the natural distribution |
| Condition | Dry, damp, dusty, dirty |
| Illumination | Indoor, overcast, direct sun, dusk |
| Inflation | 180 / 200 / 220 / 240 kPa |
| Load | Empty, 2 occupants, loaded |

**Natural wear distribution is catastrophically imbalanced toward "mostly fine".** Deliberately over-sample worn and damaged tyres — that is what the scrapyard supplement is for.

---

## 2. Scrapyard tyres — do this in week 3

A local tyre shop's discard pile contains every rare class you will never find on a roadworthy car: worn to the cord, cuts, sidewall damage, cupping, flat spots, embedded objects. They will give them to you free.

Mount them on the jig and photograph them properly. **Rare classes are what your per-class metrics will die on**, and this is the cheapest possible fix. One afternoon, zero rupees.

---

## 3. Ground truth

### 3.1 Tread depth — the reference protocol

Digital tread depth gauge, 0.01 mm resolution.

- **3–6 circumferential stations**, chalk-marked so repeat visits sample the same locations
- **Every main groove** across the tread width
- 3 readings per point, record the **median**
- Record: tyre pressure, size, wheel position, contamination state, ambient temperature, operator ID

Technique rules that actually matter:

- Measure in a **main circumferential groove** — never a sipe, never on a TWI bar
- Press the gauge foot flat on the rib surfaces either side. **Tilting is the number-one error source**
- Note which groove is which (outer → inner), consistently

### 3.2 Establish the noise floor — one afternoon, highest credibility-per-hour in the project

Before evaluating any model:

1. Measure 20 tyres. Wait a day. Measure again at the same chalk marks. → **test–retest MAE**, expect 0.10–0.15 mm
2. Have a second team member measure the same 20. → **inter-operator MAE**, expect 0.15–0.25 mm

Report both. Every model number then becomes interpretable. A model at 0.35 mm against a reference with 0.15 mm noise is doing well — but only if you measured the floor.

### 3.3 Wear pattern — multi-label

```
uniform · centre_wear · both_shoulders_wear · inner_edge_wear · outer_edge_wear
feathering · cupping_or_scalloping · flat_spot · irregular_patch_wear
```

**Independent sigmoids, not softmax.** Real tyres carry two or three simultaneously. Every public tyre dataset gets this wrong by using binary good/defective; modelling it properly is a small but real contribution.

### 3.4 Damage — masks or instances

```
cut · crack · missing_chunk · embedded_object · exposed_cord · bulge (only if sidewall visible)
```

Bulges are hard to infer from a front tread view. **Report only when the relevant sidewall is in frame**, and say so.

### 3.5 Alignment — the jig

Alignment ground truth normally requires a workshop rack. You will get one afternoon of access, yielding perhaps 25 vehicles — nowhere near enough to train a geometry head.

**Don't measure alignment. *Set* it.**

```
        ┌──────────────────────────┐
        │  ballast 0–350 kg        │  simulates wheel load
        └────────────┬─────────────┘
        ┌────────────┴─────────────┐
        │  camber stage  ±5°       │  vernier / digital angle gauge, 0.05°
        └────────────┬─────────────┘
        ┌────────────┴─────────────┐
        │  toe turntable  ±3°      │  vernier arc, 0.05°
        └────────────┬─────────────┘
                 [ WHEEL ]
        pushed along a straight guide rail past the camera
```

**Build notes**

- **Toe stage:** used machinist's rotary table (~₹4,000) or lazy-susan bearing with a 300 mm-radius vernier arc — at that radius 0.05° = 0.26 mm of arc, trivially readable
- **Camber stage:** hinged plate + turnbuckle; verify with a digital angle gauge (₹1,200, 0.05°) on the wheel face
- **Load:** gym plates or sandbags; record actual mass — load is a nuisance parameter you *want* varied
- **Guide rail:** two clamped extrusions keep travel straight, so the travel axis is known

**Why this beats a rack**

| | Rack | Jig |
|---|---|---|
| Label nature | A *measurement* (has its own error) | A *setting* (exact by construction) |
| Clips per day | ~20 vehicles | ~400 |
| Angle coverage | Whatever real cars have — **mostly near zero** | **Uniform over ±3°** |
| Extreme angles | Never | Trivial |
| Cost | Favours owed | ~₹8,000, yours |

That "uniform coverage" row is the decisive one. Real alignment data is hopelessly imbalanced toward "roughly fine"; you cannot train a geometry head on it.

**Be honest about the catch.** A jig wheel has no suspension compliance, no caster, no dynamic load transfer. So: **jig trains the geometry head; real vehicles validate and calibrate it.** Quantify the gap — "jig→real adds X° MAE" is itself a publishable number, and honesty here impresses examiners far more than hiding it.

Get **one** rack session with ~25 vehicles as the gold test set. Buy the technician lunch. Bring your own laptop.

### 3.6 Longitudinal study — start in week 1

Six vehicles you can access regularly. Capture every 2–4 weeks for the whole project. 20 minutes a fortnight.

Value out of proportion to effort:

- **Monotonicity supervision** — depth only decreases. Free physical constraint and a strong regulariser
- **Wear-rate estimation** → "remaining life in km", an output nobody else has
- **Before/after alignment correction** if any of those cars gets aligned — the money shot for the wear↔geometry cross-check

**You cannot buy back lost calendar time on a longitudinal study.** It is the one element strictly gated by the wall clock. Start it before the rig exists, with nothing but the gauge and a phone.

---

## 4. Splitting — get this right or every number is fiction

**Group by tyre identity. Never split by frame or by clip.**

> `final_v1` already does this correctly — whole `session_group` values are assigned to folds, and validation contains clean originals only. **Use the supplied `splits/*.csv` files; do not construct your own.** Assertions to run every time are in `12_DATASET_FINAL_V1.md §10`.

```
train : val : calibration : test  =  60 : 15 : 10 : 15   (grouped by tyre_id)
+ unseen_brand      — 2 brands held out entirely
+ gold_rack         — opened ONCE, at the end
```

The `calibration` split is separate and exists solely for conformal interval fitting. Reusing the validation split for conformal invalidates the coverage guarantee.

Consecutive frames of one clip are near-identical. A frame-level split leaks, and you will report an excellent, meaningless score. **This is the single most common way student projects produce numbers that collapse on real data.**

### The gold-set rule

Evaluate on `gold_rack` exactly once, at the end, and report whatever it says. Put it in a separate folder. Commit to this now, while it costs nothing.

---

## 5. Annotation — the real bottleneck

Hand-labelling sipes and grooves across 300 tyres will consume the project. Two techniques change the arithmetic (details in `10_VISION_TECHNIQUES.md §15`):

### SAM2 with memory propagation

Prompt one frame; the mask propagates temporally through the clip. Reported throughput in a comparable workflow: **37.8 s/frame → 4.5 s/frame** (~8×). Video of a rotating tyre is close to the ideal case because consecutive frames are highly correlated.

```
SAM2 point prompts on frame 1
   → propagate through clip
   → human corrects only drifted frames
   → retrain → re-propagate
```

### Model-in-the-loop bootstrapping

Label 30 tyres → train SegFormer-B0 → pre-annotate the next 30 → **correct rather than draw**. Correction runs ~5× faster than annotation from scratch.

### Annotation budget

*(Solo — one annotator. See `15_ANNOTATION_GUIDE.md`.)*

| What | Tool | Volume | Est. time |
|---|---|---|---|
| Tread/shoulder masks | SAM2 + correction | 300 tyres | ~20 h |
| Fine structure (grooves, sipes, TWI) | SAM2 + correction, tiled | 150 tyres | ~25 h |
| Wear-pattern multi-label | Checkbox UI | 400 clips | ~8 h |
| Damage masks | labelme | 300 images | ~6 h |
| Landmarks | labelme points | 400 frames | ~8 h |
| **Ranking pairs** | Custom 2-up UI, ←/→ keys | 3,000 pairs | ~6 h |

**Build the ranking UI.** Two crops side by side, arrow keys, `↑` for too-close-to-call. You will label 3,000 pairs in an evening. Highest labels-per-minute activity in the project, and it feeds the ordinal head — which is where precision on small data actually comes from.

### Quality control

`15_ANNOTATION_GUIDE.md` is that guideline document — read it before labelling anything.

**Working alone, the quality measure is self-consistency, not inter-annotator agreement.** Re-annotate a 30-image subset after finishing everything, without looking at the first pass, and report the IoU. It catches the thing that actually threatens a one-person job: your boundary judgement drifting as your eye improves over several hours. Almost no student project measures this, and it takes 20 minutes.

---

## 6. Augmentation

Reproduce the camera environment without corrupting physical labels.

**Safe:** brightness / contrast / gamma / colour-temperature shifts · physically plausible motion blur · defocus blur · sensor noise · synthetic dirt, dust, water spots, partial glare · small camera translations and scale within rig tolerance · random channel dropout across RGB/normals/polarised · cutout (record the occlusion mask)

**Dangerous — handle or disable:**

| Augmentation | Problem |
|---|---|
| **Horizontal flip** | Swaps inner/outer shoulder **and negates toe and camber signs.** Transform every affected label or disable |
| Vertical flip | Physically invalid |
| Random rotation / perspective | Changes alignment geometry. Only with mathematically updated angle labels |
| MixUp / CutMix | Poor default for local cracks, TWI bars, geometric landmarks |

> **Write `tests/test_augment_signs.py` now.** Assert that a horizontal flip negates both angles, and that flipping twice is the identity. This bug class is invisible until it has cost you a month.

---

## 7. Dataset release

```
tyrevision-front/
├── README.md              # datasheet: motivation, composition, collection, uses, limits
├── LICENSE                # CC BY-NC 4.0
├── metadata.parquet       # one row per clip, all fields from 02_RIG_BUILD §6
├── splits/{train,val,calib,test,unseen_brand,gold_rack}.txt
├── clips/<clip_id>/frames/*.png
├── clips/<clip_id>/normals.npy        # photometric stereo output
├── clips/<clip_id>/meta.json
├── processed/<clip_id>/unrolled.png   # + coverage_pct
├── labels/depth_gauge.csv
├── labels/masks/          ├── labels/patterns.jsonl
├── labels/damage/         ├── labels/landmarks.jsonl
└── labels/rankings.jsonl
```

Publish to Hugging Face under `Shanmuk4622/`. Write a proper **datasheet** (Gebru et al. format): motivation, composition, collection process, preprocessing, uses, distribution, maintenance. An afternoon's work, and it is what separates a released dataset from a dumped folder.

**Privacy, from clip #1:** blur number plates, strip GPS from phone captures, get written consent from vehicle owners. A one-paragraph form is enough. Retrofitting consent is impossible.

**Publication timing:** keep the dataset private on HF until the paper/report is submitted, then flip it public. `.gitignore` already excludes `data/`, `raw/`, `processed/` and weights, so you cannot leak it through git by accident.
