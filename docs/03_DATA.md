# 03 — Data: The Jig, The Protocol, The Synthetic Pipeline

> 80% of your effort belongs here. Models are commodities; **your dataset is the contribution nobody can replicate.**

---

## 1. The alignment jig — the single best idea in this project

### The problem

Alignment ground truth normally requires a ₹25-lakh Hunter/John Bean rack and a cooperative workshop. You will get maybe *one* afternoon of access. That yields perhaps 30 labelled vehicles — nowhere near enough to train a geometry head.

### The solution

**Don't measure alignment. *Set* it.**

Build a wheeled cart carrying one real spare wheel, mounted on a hub whose toe and camber angles are **adjustable and readable to 0.05°**. Then roll the cart over the plate at any angle you like, as many times as you like. The label is not measured — it is *dialled in*. It is exact by construction.

```
         ┌─────────────────────────┐
         │   ballast (0–350 kg)    │  ← simulates wheel load, adjustable
         └───────────┬─────────────┘
                     │
        ┌────────────┴────────────┐
        │   camber goniometer     │  ← rotates about longitudinal axis, ±5°
        │   ├ vernier scale 0.05° │
        └────────────┬────────────┘
                     │
        ┌────────────┴────────────┐
        │   toe turntable         │  ← rotates about vertical axis, ±3°
        │   ├ vernier scale 0.05° │
        └────────────┬────────────┘
                     │
                 [ WHEEL ]
                     │
   ══════════════════╪══════════════════  glass plate
        pushed by hand along a straight guide rail
```

### Build notes

- **Toe stage:** a machinist's rotary table (₹4,000 used) or a lazy-susan bearing with a 300 mm-radius vernier arc. At 300 mm radius, 0.05° = 0.26 mm of arc travel — readable with a steel rule, trivially readable with a dial indicator.
- **Camber stage:** a hinged plate with a micrometer/turnbuckle adjuster. Verify with a **digital angle gauge** (₹1,200, 0.05° resolution) placed on the wheel face.
- **Load:** stack gym plates or sandbags. Record the actual mass. Load matters enormously for footprint shape — you *want* it as a varied nuisance parameter.
- **Guide rail:** two lengths of extrusion clamped to the floor keep the cart travelling straight, so the travel axis is known. This is what makes the toe label meaningful.
- **Push, don't motorise.** Walking pace is exactly your target speed.

### Why this is better than a real alignment rack

| | Rack | Jig |
|---|---|---|
| Label uncertainty | ±0.05° (and it's a *measurement*, so it has its own error) | ±0.05° (a *setting*, exact by construction) |
| Samples per day | ~20 vehicles | ~400 passes |
| Angle coverage | Whatever real cars happen to have (mostly near-zero!) | **Uniform over the full ±3° range** |
| Cost | Workshop access, favours owed | ₹8,000, yours forever |
| Extreme angles | Never — no one drives a car at 4° toe | Trivial |

That third row is the killer. Real-world alignment data is catastrophically imbalanced toward "roughly fine". The jig gives you a **uniform, balanced, extreme-inclusive** training distribution. You cannot buy that.

### The catch — be honest about it

A jig wheel is not a car wheel. Differences:
- No suspension compliance (real camber changes under load and cornering)
- No steering geometry, no caster effect
- Static vertical load only, no lateral force
- Cart wheel may not be under a real vehicle's dynamic load transfer

So: **jig data trains the geometry head; real-car data validates and calibrates it.** Report both. Quantify the gap explicitly — "jig→real domain gap adds X° MAE" is itself a publishable number, and honesty here will impress examiners far more than pretending the gap doesn't exist.

Get *one* workshop session on a real rack with ~25 vehicles as your gold real-world test set. Buy the technician lunch. Bring your own laptop.

---

## 2. Tread-depth ground truth protocol

The laser gives you dense depth. But you still need an independent, trusted reference to validate the laser itself and to anchor the metric scale.

### Manual gauge protocol (do this on every tyre you capture)

Use a **digital tread depth gauge, 0.01 mm**. Measure at a grid:

- **12 circumferential positions** (clock positions, marked with chalk so you can re-find them)
- **5 lateral positions** (outer shoulder, outer rib, centre, inner rib, inner shoulder)
- = **60 readings per tyre**

Rules that actually matter:
- Measure in the **main circumferential groove**, not a sipe, not on a TWI bar.
- Press the gauge foot flat on the *rib surfaces either side* of the groove. Tilting is the #1 source of error.
- Take 3 readings at each point, record the median.
- Mark the tyre with chalk at "clock 12" so repeat visits sample the same locations.
- Record ambient temperature — rubber compliance varies.

This takes ~12 minutes per tyre. It is boring. Do it anyway; it is the foundation of every metric claim you will make.

### Building a longitudinal set (high value, low cost)

Pick **6 vehicles you have regular access to** (yours, family, friends, a fleet if you can find one). Capture every 2–4 weeks for the whole project.

Why this is disproportionately valuable:
- Gives you **monotonicity supervision** — depth only ever decreases. A powerful physics constraint and a free regulariser.
- Gives you **wear-rate estimation**, which is the "remaining life in km" output nobody else has.
- Gives you **paired before/after** if any of those cars gets an alignment done. That is the money shot for the wear↔alignment coupling claim.
- Costs you 20 minutes a fortnight.

**Start this in week 1**, before the rig is finished, using nothing but the gauge and a phone. You cannot buy back lost time on a longitudinal study. If you do only one thing from this document today, do this.

---

## 3. Data collection plan

| Source | Volume target | What it's for | Labels |
|---|---|---|---|
| **Jig passes** | 3,000 passes | Alignment head training | toe, camber (exact), load, tyre ID |
| **Real vehicles** | 300 passes / 80 wheels | Domain adaptation, validation | depth grid, wear class, some alignment |
| **Rack-verified vehicles** | 25 vehicles / 100 wheels | Gold test set | rack toe/camber/caster/thrust |
| **Longitudinal** | 6 vehicles × 10 visits | Wear-rate, monotonicity | depth grid over time |
| **Synthetic (Blender)** | 50,000 renders | Pretraining, extreme coverage | perfect depth, angles, pattern |
| **Public (Kaggle/Roboflow)** | ~5,000 images | Self-supervised backbone pretraining | weak / binary |
| **Scrapyard tyres** | 40 tyres | Extreme wear + damage classes | depth grid, damage annotation |

**On scrapyard tyres:** a local tyre shop's discard pile is a goldmine. Worn to the cord, cuts, bulges, cupping, flat spots — every rare class you'll never see on a roadworthy car. They will give them to you free. Mount them on the jig. **Do this in week 3.** Rare classes are what your metrics will die on.

### Splitting — get this right or your numbers are fiction

Split by **vehicle and tyre identity**, never by frame or by pass.

```
train : val : test  =  70 : 15 : 15   (grouped by tyre_id)
+ a completely held-out "unseen brand" test set
+ the rack-verified gold set, touched exactly once, at the very end
```

Frame-level splitting will leak — consecutive frames of the same pass are nearly identical, and you will report a beautiful, meaningless MAE. This is the single most common way capstone projects produce numbers that collapse on real data.

---

## 4. Synthetic pipeline (Blender)

### Goal

Perfect labels, unlimited volume, uniform coverage of the parameter space — especially the corners real data never reaches.

### Parametric tyre model

Build **one** procedural tyre in Blender with these driven parameters:

| Parameter | Range | Drives |
|---|---|---|
| `tread_depth_mm` | 0.5 – 9.0 | Displacement-map amplitude on the tread band |
| `wear_lateral_gradient` | −1 – +1 | Linear ramp multiplied into the displacement across width |
| `wear_profile` | centre / shoulder / uniform | Parabolic weighting across width |
| `feathering` | 0 – 1 | Asymmetric shear of the rib-edge profile |
| `cupping_amplitude`, `cupping_period` | 0–2 mm, 8–20 per rev | Sinusoidal circumferential modulation |
| `toe_deg` | −3 – +3 | Wheel object rotation about Z |
| `camber_deg` | −5 – +5 | Wheel object rotation about X |
| `load_deflection_mm` | 5 – 30 | Contact-patch flattening (soft-body or shape key) |
| `tread_pattern_id` | 12 variants | Which groove pattern (model a few real ones) |
| `sidewall_text` | random | Brand markings, DOT codes |

**Key implementation note:** drive tread depth with a **displacement modifier on a high-res tread band**, using a greyscale pattern texture as the height map. Then `tread_depth_mm` is literally the modifier's strength, and your ground-truth depth map is the texture × strength. Exact labels, zero annotation.

### Domain randomisation

Randomise everything that is *not* the label:

- Rubber shader: base colour (0.02–0.12 albedo), roughness, subsurface, sheen
- Contamination: dust layer, mud splatter, brake dust, water film, road tar, small embedded stones
- Illumination: IR LED array position/intensity, laser line position/power, ambient leak
- Camera: focal length ±20%, standoff ±30 mm, principal-point jitter, radial distortion, sensor noise, exposure
- Glass: dirt on surface, scratches, minor refraction, slight tint
- Motion: sub-frame blur outside the contact band (physically correct — use the `2v·sin(φ/2)` law!)

That last point is worth stressing: **render the blur correctly per-region using the rolling constraint**. It makes the synthetic data physically faithful in exactly the way that matters, and it is a nice detail for the paper.

### Simulating FTIR in Blender

Approximate rather than ray-trace. For contact regions, compute a contact mask from the deformed mesh's proximity to the glass plane, then render an emission shader whose strength ∝ local penetration depth. This is not physically exact, but it reproduces the *structure* of the signal (bright ribs, black voids, pressure-graded intensity), which is what the network learns from.

Optionally verify against real FTIR captures and tune the falloff curve to match the observed histogram. That is a nice half-page in the paper.

### Sim-to-real strategy

Do **not** just train on synthetic and hope.

```
1. Pretrain on 50k synthetic renders           → learns geometry + depth structure
2. Fine-tune on jig data                       → learns real rubber, real optics
3. Domain-adapt to real vehicles               → CORAL / adversarial feature alignment
4. Calibrate with conformal on real held-out   → honest intervals
```

Report the ablation: synthetic-only, real-only, and the full ladder. **Even if synthetic pretraining only buys 15% MAE reduction, that is a clean, reportable result.** If it buys nothing, report that too — negative results about sim-to-real are genuinely useful and reviewers respect them.

---

## 5. Public datasets — what they're good for, and what they're not

| Dataset | Size | Labels | Honest verdict |
|---|---|---|---|
| [Tyre Quality Classification](https://www.kaggle.com/datasets/warcoder/tyre-quality-classification) | 1,854 | good / defective | Binary, static close-ups. Fine for backbone pretraining only. |
| [Tire Texture Image Recognition](https://www.kaggle.com/datasets/jehanbhathena/tire-texture-image-recognition) | 1,028 | cracked / normal | Texture pretraining. No depth. |
| [Tyre Condition Classification](https://www.kaggle.com/datasets/sameersambhare1/tyre-condition-classification-dataset) | — | new / serviceable / unusable | Closest to ordinal. Useful for the ranking head warm-start. |
| [Full vs Flat Tire](https://www.kaggle.com/datasets/rhammell/full-vs-flat-tire-images) | 900 | full / flat / none | Inflation only. Marginal. |
| [Roboflow tyre datasets](https://universe.roboflow.com/search?q=class%3Atyre) | varies | bboxes, some segmentation | Good for the detector/segmenter warm-start. |

**Use them for exactly two things:**
1. Self-supervised pretraining (DINOv2 / MAE) of the texture backbone — no labels needed, so their weak labels don't matter.
2. Warm-starting the tyre detector/segmenter.

**Do not** benchmark against them and claim victory. They are static, hand-held, binary-labelled photos. Your task is a different task. Say so plainly in the related-work section; that framing *strengthens* your contribution rather than weakening it.

Download via the Kaggle API for speed (see `05_TRAINING_KAGGLE_HF.md`).

---

## 6. Annotation

What actually needs human annotation (everything else is automatic):

| What | How | Volume | Time |
|---|---|---|---|
| Wear-pattern multi-label | CVAT, 8 checkboxes per unrolled map | 500 maps | ~8 h |
| Damage bounding boxes | CVAT | 300 images | ~5 h |
| TWI bar locations | CVAT points, **or** auto-detect + human verify | 400 | ~4 h |
| Ranking pairs | Custom 2-up web UI: "which is more worn?" | 3,000 pairs | ~6 h |

**Build the ranking UI.** Two images side by side, `←`/`→` keys, `↑` for "too close to call". You will label 3,000 pairs in an evening while watching something. It is the highest labels-per-minute activity in the project by a wide margin, and it feeds the loss that matters most.

Annotation guidelines document: write one, with example images for every class, *before* you annotate anything. Then have one other person label 100 items and compute **Cohen's κ**. Report it. Inter-annotator agreement is a paper-quality detail that costs you two hours.

---

## 7. Dataset release: `GRIP-Roll`

Publish it. It is the most durable thing you will produce and the thing most likely to get you cited.

```
grip-roll/
├── README.md                  # datasheet (follow "Datasheets for Datasets")
├── LICENSE                    # CC BY-NC 4.0 recommended
├── metadata.parquet           # one row per pass, all fields from 02_RIG_BUILD §4
├── splits/{train,val,test,unseen_brand,gold_rack}.txt
├── raw/<pass_id>/frames/*.png
├── raw/<pass_id>/meta.json
├── processed/<pass_id>/unrolled_{ftir,flood,laserdepth}.png
├── processed/<pass_id>/depth_gt.npy
├── labels/depth_grid.csv      # the manual 12×5 gauge readings
├── labels/patterns.jsonl
├── labels/rankings.jsonl
└── synthetic/                 # generator script + config, not the 50k renders
```

Ship to Hugging Face under `Shanmuk4622/grip-roll`. Write a proper **datasheet**: motivation, composition, collection process, preprocessing, uses, distribution, maintenance. It takes an afternoon and it is what separates a released dataset from a dumped folder.

**Privacy:** blur number plates, strip GPS from any phone-captured images, get written consent from vehicle owners. A one-paragraph consent form is enough. Do this from pass #1 — retrofitting consent is impossible.
