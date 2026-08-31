# 12 — Dataset `final_v1`: What It Is, and What It Can Actually Support

**Package:** `D:\Dataset Download\Tire Dataset Prepared\FINAL`
**Version:** `final_v1` · prepared 26 Aug 2026 · verification status **PASS**
**Analysed:** 26 Aug 2026 · re-read against the expanded dataset README (428 lines) same day

> The package itself is **excellently engineered** — lineage-tracked, hash-verified, group-split, honestly documented. The preparation work is better than most published datasets.
>
> This document is not a critique of that work. It records what the data can and cannot support, and reports two experiments that measure the difficulty floor. **§6 is the finding that should change what you do next.**

---

## 1. Contents at a glance

| Item | Value |
|---|---:|
| Raw JPEGs audited | 888 |
| Byte-identical duplicates excluded | 470 |
| **Clean unique source images** | **418** |
| Synthetic derivatives (10 per source) | 4,180 |
| Total image files | 4,598 |
| **Independent capture sessions** | **12** |
| Group folds | 3 |
| Clean resolution | 1152 × 1536 RGB |
| Synthetic resolution | 768 × 768 RGB |
| Package size | 1.2 GB |
| Corrupt files | 0 |

### Class distribution (clean images)

| Proxy label | Index | Source folders | Images | Sessions |
|---|---:|---|---:|---:|
| `low_mileage_proxy` | 0 | New Tires, 100–5000 | 169 | 3 |
| `mid_mileage_proxy` | 1 | 40,000 · 70,000 · 90,000 | 97 | 3 |
| `high_mileage_proxy` | 2 | Tires Gone (1L+) | 152 | 6 |

### Sessions — and the number that matters

| Session | Class | Images | Fold |
|---|---|---:|---:|
| `new_tire__session_001` | low | 67 | 0 |
| `mileage_000100_005000__session_001` | low | 40 | 2 |
| `mileage_000100_005000__session_002` | low | 62 | 1 |
| `mileage_040000__session_001` | mid | 20 | 1 |
| `mileage_070000__session_001` | mid | 20 | 2 |
| `mileage_090000__session_001` | mid | 57 | 0 |
| `mileage_100000_plus__session_001` | high | 3 | 2 |
| `mileage_100000_plus__session_002` | high | 2 | 0 |
| `mileage_100000_plus__session_003` | high | 5 | 1 |
| `mileage_100000_plus__session_004` | high | 60 | 0 |
| `mileage_100000_plus__session_005` | high | 41 | 2 |
| `mileage_100000_plus__session_006` | high | 41 | 1 |

**The scientific sample size is 12, not 4,598.** The package README states this plainly and it is correct.

> ### ⚠ And 12 may be optimistic
>
> `session_group` was derived from a **12-second timestamp gap** — a proxy for
> tyre identity, not a measurement of it. Photograph one tyre, walk away for 20
> seconds, come back: the rule creates two "sessions" for one tyre.
>
> A tread-pattern audit (`scripts/tyre_identity_audit.py`) found
> **`mileage_070000__session_001` and `mileage_090000__session_001` are
> indistinguishable** (ratio 0.90 against a same-tyre reference of 1.0) — and
> they sit in **different folds** (2 and 0). For fold 0, that puts the same
> tyre in training and in validation.
>
> Three further sessions (2, 3 and 5 images) are too small to adjudicate.
>
> **What this means:** fold 0's `mid` class may be leak-inflated, and the true
> number of distinct tyres is somewhere between 9 and 12. Report it as a
> limitation; the audit JSON ships with the annotations.

Three further facts the timestamps reveal:

1. **Every session was captured on 2026-08-25 between 11:03 and 11:25** — a single 22-minute window. There is **no lighting diversity, no weather diversity, no time-of-day diversity, and no site diversity.** All observed variation is between tyres, not between conditions.
2. **Three high-mileage sessions contain 2, 3 and 5 images.** They are effectively single observations.
3. **`mid` has exactly one session per fold.** For any fold, the entire mid class in validation is *one physical tyre*.

---

## 1b. Reading the folder structure correctly

The `images/` tree encodes **three independent things at once**, and confusing them is the easiest way to misuse this package. The dataset README (§"Important: how to understand the image folders", added 2026-08-26) states it explicitly:

```
images/ <kind> / <fold> / <proxy_class> / *.jpg
          │        │           └── mileage-proxy class, from the original workshop folders
          │        └────────────── cross-validation group — NOT a wear category
          └─────────────────────── real photo vs artificial derivative
```

| Level | Values | What it means | Common mistake |
|---|---|---|---|
| **kind** | `clean` / `augmented` | `clean` = 418 real, deduplicated, EXIF-corrected, standardised photographs. `augmented` = 4,180 artificial variants, 10 per clean source | Treating augmented images as extra tyres |
| **fold** | `fold_0/1/2` | Leakage-safe CV group. Whole capture sessions assigned together | **Reading `fold_1` as "medium wear."** It is not a class |
| **class** | `low/mid/high_mileage_proxy` | Derived from original folder names | Reading them as measured wear |

### `clean` vs `augmented` — the rule

- **`clean`** may be used for training *or* validation, according to its fold. **Validation must always use clean only.**
- **`augmented`** is **training-only**. Never validation, never test, never counted as an independent observation.

Every derivative is linked to its ancestor by `source_image_id`, source hashes, fold, session, class, seed, and a full transformation trace. The manifest carries `independent_tyre_increment = 0` on every row — an explicit, machine-readable refusal to let augmentation masquerade as new data.

### The CV protocol, restated

| Run | Training | Validation |
|---|---|---|
| CV 0 | clean + augmented from folds 1, 2 | **clean only** from fold 0 |
| CV 1 | clean + augmented from folds 0, 2 | **clean only** from fold 1 |
| CV 2 | clean + augmented from folds 0, 1 | **clean only** from fold 2 |

Each run: 8 training sessions, 4 held-out validation sessions. Fold sizes are deliberately unequal (186 / 128 / 104 clean) because keeping whole sessions together matters more than equal counts.

### Why "proxy" is load-bearing terminology

```
New Tires + 100-5000        → low_mileage_proxy
40,000 + 70,000 + 90,000    → mid_mileage_proxy
Tires Gone (1L and Above)   → high_mileage_proxy      (1L ≈ one lakh ≈ 100,000 km)
```

Six original folders were collapsed to three because some contained only **one** provisional session, making six-class evaluation indefensible.

Actual wear depends on tyre design, road surface, inflation, alignment, load, driving style, weather and maintenance — not on odometer reading alone. These folders are therefore **not** `safe`, `worn`, `dangerous`, remaining-life, alignment, or roadworthiness labels.

---

## 2. What the images actually look like

Viewed: both contact sheets, plus native-resolution centre crops across all three classes and six full frames.

**Viewpoint.** Handheld close-ups of a tyre on a parked vehicle, camera roughly perpendicular to the tread crown, tyre vertical in a portrait frame. This is *close* to the intended low-front viewpoint, and the tread face is well presented. It is **not** rigidly mounted or calibrated.

**The wear signal is genuinely visible.** At native resolution the three classes are clearly distinguishable to the eye:

| Class | Appearance at native resolution |
|---|---|
| low / new | Deep grooves with hard shadow, crisp sipes, sharp rib edges, **coloured paint stripes and white lettering** |
| mid | Grooves present but shallower, sipes intact, whitish deposits in groove floors |
| high | Grooves nearly flush with the surface, polished smooth tread face, sipe structure largely gone, heavy grey dust |

**Approximate scale:** a ~195 mm tread spanning roughly 800–950 px gives **≈0.21–0.24 mm/px** — but framing varies substantially between sessions, so scale is **not constant and not recorded**. Against the resolution budget in `01_CONCEPT.md §5` (≤0.1 mm/px needed for 0.3 mm sipes), this resolves main grooves and rib edges well, and resolves only the wider sipes.

**Framing is inconsistent.** Some frames are cropped so tightly that **the shoulders are cut off**; others include large amounts of road, wall, vegetation and, in one session, a parked white car. Backgrounds differ per session.

**Real damage is present** — at least one high-mileage session shows visible shoulder chunking.

---

## 3. Preparation quality — what was done well

Worth stating explicitly, because it is genuinely good practice and should be preserved in `final_v2`:

- 470 byte-identical duplicates removed by SHA-256 (a 53% reduction — silently training on those would have inflated every metric)
- Burst frames retained but **controlled by session grouping** rather than treated as independent tyres
- EXIF orientation applied, metadata stripped, aspect preserved, standardised encode
- **Whole sessions assigned to folds**, so no burst frame crosses train/validation
- Deterministic derivatives with per-variant seeds and a full JSON operation trace
- **Validation is clean-originals only**; no derivative of a held-out source appears in training
- `independent_tyre_increment = 0` on every row — an explicit refusal to let augmentation masquerade as new data
- Per-file SHA-256 checksums and a machine-readable verification report
- Vertical flip, elastic, aggressive perspective, MixUp and CutMix all disabled, with reasons
- Horizontal flip flagged as **forbidden for any signed toe/camber or inner-vs-outer task**
- The README documents its own folder semantics (§1b) rather than leaving them to be inferred
- Six original classes collapsed to three **because the session count did not support six** — a restraint most datasets skip

That last point matters and matches `03_DATA.md §6` exactly. The augmentation policy is already correct for the mileage-proxy task and already blocks the laterality bug for future alignment work.

---

## 4. Alignment between the package and this project

| Project component | Dataset support | Status |
|---|---|---|
| 3-class ordinal mileage-proxy classification | ✅ Fully supported | **Buildable now** |
| Tyre / tread segmentation | ✅ **Available** — `annotation_v2` | 418 hand-drawn, propagated to 4,598, verified by `NBT1` |
| Marking / damage regions | ✅ **Available** | 67 / 63 images |
| Groove / sipe / TWI segmentation | ❌ No masks | Not annotated |
| Wear-pattern multi-label (9 classes) | ❌ No labels | Needs annotation |
| Localised wear heatmap | ❌ No labels | Needs annotation |
| Damage detection | ❌ No masks; damage visible but unannotated | Needs annotation |
| PatchCore anomaly detection | ⚠️ Possible — needs a curated healthy pool | Partially buildable |
| Tread depth in millimetres | ❌ **No gauge measurements at all** | Needs physical measurement |
| **Camber / toe** | ❌ **No calibration, no pose, no rack data** | **Impossible from this package** |
| Photometric stereo | ❌ Single-illumination captures | Needs new capture rig |
| Partial tread unrolling | ❌ Not video; no rotation sequence | Needs video capture |
| Video / temporal fusion | ❌ Burst frames, not calibrated video | Needs video capture |

**Roughly one of thirteen planned capabilities is trainable today.** That is not a failure — it is a pilot dataset doing exactly what a pilot dataset should do. But the docs must stop implying the rest is imminent, and `07_ROADMAP.md` has been updated accordingly.

---

## 5. Label semantics — the trap to avoid

The labels are a **mileage proxy**, not wear measurement. The distinction is not pedantic:

- A 90,000 km tyre that was rotated regularly and correctly inflated can have **more** remaining tread than a 40,000 km tyre that was run under-inflated on a misaligned axle.
- Odometer reading is a property of the *vehicle*; tread depth is a property of the *tyre*.
- The folder labels came from a workshop's categorisation, not from a gauge.

Consequences, all of which the package README already states and which this project adopts:

1. Report the target as `mileage_proxy`, never as `worn` / `not_worn`.
2. A binary safety claim needs a **written threshold tied to measured tread depth** and a safety standard. There is none yet.
3. Use ordinal metrics — quadratic weighted kappa, mean absolute class error — alongside macro-F1, because the classes are ordered.
4. Any figure, slide or abstract that says "tyre wear detection" must be qualified as mileage-proxy classification until gauge data exists.

---

## 6. Two probes that establish the difficulty floor

**This is the section that should change your plan.**

Both probes use only the dataset's own supplied group folds, clean images only, multinomial logistic regression — no deep learning, no pretraining.

Reproduce with:

```bash
conda activate cv_conda
python scripts/dataset_shortcut_probe.py --root "D:/Dataset Download/Tire Dataset Prepared/FINAL"
```

### Probe A — colour only

Ten global colour statistics from a **64 × 64 thumbnail**: mean R/G/B, brightness, contrast, saturation, a blue-excess fraction, a bright-pixel fraction, a dark-pixel fraction. No texture. No tread structure.

| Fold | n(val) | macro-F1 | Accuracy | Majority-class acc |
|---:|---:|---:|---:|---:|
| 0 | 186 | **0.952** | 0.952 | 0.360 |
| 1 | 128 | 0.399 | 0.430 | 0.484 |
| 2 | 104 | 0.123 | 0.202 | 0.423 |
| **Mean** | | **0.491** | 0.528 | 0.423 |

### Probe B — structure only

Nine contrast-normalised texture statistics from the **centre tread band** — background cropped away, global brightness and contrast normalised out. Groove-shadow fractions at three thresholds, gradient energy, Laplacian variance, and vertical/horizontal banding strength.

| Fold | n(val) | macro-F1 | Accuracy |
|---:|---:|---:|---:|
| 0 | 186 | 0.354 | 0.441 |
| 1 | 128 | 0.119 | 0.172 |
| 2 | 104 | **0.976** | 0.981 |
| **Mean** | | **0.483** | |

### What these results mean

**(a) A single fold proves nothing.** Fold 0 hands a ten-number colour model **95.2% macro-F1**. Fold 2 hands a nine-number texture model **97.6%**. Both average ~0.49 across all three folds — barely above a majority-class baseline of 0.423.

**Two further probes were added once the annotations existed** (`annotations/README.md §6`):

| Probe | f0 | f1 | f2 | mean |
|---|---:|---:|---:|---:|
| **Frame occupancy** — how much of the frame is tyre | 0.181 | 0.455 | **0.968** | **0.535** |
| **Annotation side-channel** — `marking→low, damage→high` | **0.978** | 0.159 | 0.108 | 0.415 |

**Four probes. Four different folds. Four different shortcuts.** And the
strongest is *framing*: the tyre fills 72% of the frame in `low` images, 62% in
`mid`, 61% in `high`. **The floor is 0.535, not 0.491.**

> **If you train a deep model and report 95% on fold 0, you will have reproduced what mean RGB already achieves. Always report all three folds, and always report these baselines beside them.**

**(b) The two probes succeed on *opposite* folds.** Colour wins fold 0 and collapses on fold 2; structure does exactly the reverse. Neither has learned wear. Both have learned whichever tyres happen to be in their training folds, and cross-fold transfer is close to chance.

**(c) The root cause is session count, not model capacity.** With 1–2 sessions per class per fold, "class" is very nearly "which tyre." A bigger backbone will fit the training tyres better and will not fix this.

**(d) But a genuine physical signal does exist.** The structure features are monotone in wear, in the physically correct direction:

| Feature | low | mid | high | Interpretation |
|---|---:|---:|---:|---|
| `d20` deep-groove shadow fraction | **0.046** | 0.037 | **0.019** | Deep grooves cast deep shadows; worn tread has none |
| `colstd` vertical groove banding | **0.604** | 0.513 | **0.387** | Groove structure flattens with wear |
| `gmean` gradient energy | 0.619 | 0.903 | 0.966 | Inverted — worn/dirty surfaces add fine texture noise |

The first two are exactly the cues a tread-wear model *should* use, and they are ordered correctly across all three classes. **The signal is real. There is simply not enough independent data to prove a model is using it rather than memorising tyres.**

### Confirmed shortcut risks

| Shortcut | Evidence | Mitigation |
|---|---|---|
| **Coloured paint stripes / white lettering on new tyres** | Visible in native crops of `new_tire__session_001`; `blueish` feature non-zero only for the low class | Crop to tread band; test the model on stripe-masked images |
| **Tread-pattern identity** | Each class dominated by 1–3 distinct tread designs, clearly different in the contact sheet | More tyres per class. Nothing else fixes it |
| **Dirt and deposit colour** | Grey dust on high, whitish deposits on mid, clean on low | Contrast normalisation; dirt augmentation |
| **Background** | Concrete / road / brick wall / parked car, varying by session | Tyre-region crop before classification |
| **Framing and distance** | Some frames crop the shoulders off entirely | Detect and standardise the tread ROI |
| **Capture order** | All 12 sessions within 22 minutes; class correlates with timestamp | Cannot be fixed in this package. Fix in `final_v2` |

---

## 7. Recommended use of `final_v1`

**Do:**

- Train a 3-class ordinal mileage-proxy classifier as a **pilot**, using the supplied splits
- Report **all three folds**, mean ± spread, with the Probe A and Probe B baselines in the same table
- Use ordinal metrics (QWK, mean absolute class error) alongside macro-F1 and per-class recall
- Apply a **tyre-region crop** before classification, and ablate it — the framing variance makes this a likely win
- Use `class_session_balanced_weight` for sampling
- Run saliency / occlusion analysis and check the model is looking at tread, not at paint stripes or background
- Treat it as the vehicle for building the **training harness** — resumable Kaggle notebook, HF sync, evaluation code — so that when real data arrives the infrastructure already works

**Do not:**

- Report a single fold
- Put derivatives in validation or test
- Count 4,598 as the sample size
- Read `fold_1` as a wear level — folds are CV groups, not classes
- Claim tread depth, safety, roadworthiness, alignment, toe, camber, or inner-vs-outer wear
- Relabel to binary `worn` / `not_worn`
- Publish the images before confirming ownership, consent and licensing

### The dataset's own recommended experiment order

Worth following as written — it is well sequenced, and each step is an ablation rather than an assumption:

1. Clean-only three-class baseline on the supplied group folds
2. Add photometric augmentation; compare all three folds
3. Add blur / noise / compression, then small occlusion — **one family at a time**
4. CLAHE as an ablation, not an assumed improvement
5. Compare input resolutions, and a tyre-region crop / segmentation stage
6. Ordinary class weighting vs `class_session_balanced_weight`
7. **Inspect errors and saliency for shortcut learning before model selection**
8. Calibrate confidence; allow an `uncertain / request another frame` outcome
9. Only after physical measurements exist, add heads for detailed wear, defects, depth and alignment

Steps 5 and 7 are the ones I would prioritise given the probe results in §6.

---

## 8. What `final_v2` needs

Ranked by how much each item improves the science per unit of effort.

| Priority | Need | Why |
|---:|---|---|
| **1** | **More independent tyres — target 60–100+, ≥20 per class** | The binding constraint. Everything else is secondary. 12 sessions cannot support a generalisation claim |
| **2** | **Measured tread depth (mm)** at inner/centre/outer × 3–6 circumferential stations, digital gauge, repeated readings | Converts a proxy classifier into measurement. Unlocks regression, ordinal anchoring, and any safety framing |
| **3** | **Capture-condition diversity** — multiple days, lighting, weather, sites, cameras | Everything currently comes from one 22-minute window. Domain robustness is presently unmeasurable |
| **4** | **A scale reference in frame** (ruler, checkerboard, known-size marker) | Gives mm/px. Cheap, transforms what can be claimed |
| **5** | **Consistent framing** — full tread width, both shoulders, always | Required for lateral wear profile and any geometry |
| **6** | Immutable `vehicle_id`, `tyre_id`, `capture_session_id`, axle/side | Correct grouping, and enables longitudinal tracking |
| **7** | Tyre brand, model, size, tread pattern, DOT date | Lets you *measure* the tread-pattern confound instead of guessing |
| **8** | Segmentation masks — tread, shoulders, grooves, sipes, TWI, damage | Unlocks the whole perception stack. Use SAM2 (`03_DATA.md §5`) |
| **9** | Video clips with wheel rotation | Unlocks unrolling and temporal fusion |
| **10** | Camera intrinsics, distortion, pose, distance | Prerequisite for *any* alignment work |
| **11** | Workshop alignment ground truth, or a jig | The only route to toe/camber labels |
| **12** | Multi-directional illumination captures | Photometric stereo (`10_VISION_TECHNIQUES.md §2`) |

Items 1–5 are achievable with a phone, a ₹900 gauge, a printed ruler and a few weekends. **They would improve this project more than any modelling work.**

---

## 9. Integrity and reproducibility

The package ships everything needed to verify itself:

```bash
conda activate cv_conda
python "FINAL/scripts/verify_final_dataset.py" --root "FINAL"
```

Current status: **PASS**, zero errors, zero warnings. Do not train if that ever changes.

`FINAL` is immutable `final_v1`. Corrections create `final_v2` with a changelog and new checksums; never regenerate into `FINAL`.

**Authoritative order:** `README.md` → `VERSION.json` → `manifests/dataset_manifest.csv` → `splits/*.csv` → `checksums/` + `reports/FINAL_VERIFICATION_REPORT.json`.

---

## 10. Loading it correctly

```python
from pathlib import Path
import pandas as pd

ROOT = Path("/kaggle/input/<slug>")          # dir containing README.md, images/, splits/
train = pd.read_csv(ROOT / "splits/cv0_train.csv")
valid = pd.read_csv(ROOT / "splits/cv0_validation.csv")

# non-negotiable assertions — run them every time
assert set(train.session_group).isdisjoint(set(valid.session_group))
assert set(valid.image_kind) == {"clean_original"}
assert train.relative_path.map(lambda p: (ROOT / p).exists()).all()
```

Read images directly from `/kaggle/input`; never copy 1.2 GB into the 20 GB `/kaggle/working`. Reserve working storage for checkpoints, logs and caches, per `05_TRAINING_KAGGLE_HF.md`.

---

## 11. Verdict

> `final_v1` is a **well-built pilot package that supports exactly one honest experiment**: a three-class ordinal mileage-proxy classifier, evaluated across all three group folds, reported against trivial colour and structure baselines.
>
> It cannot support tread-depth measurement, alignment, segmentation, or damage detection — not because of any flaw in its preparation, but because those labels do not exist and, for alignment, the necessary calibration was never captured.
>
> **The binding constraint is 12 sessions.** No architecture fixes that. The highest-value next action is not a bigger model — it is a gauge, a ruler in frame, and forty more tyres.
