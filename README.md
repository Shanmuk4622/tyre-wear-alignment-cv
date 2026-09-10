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

> ### 📍 Start here
>
> - **New to the project?** → `docs/00_WHAT_THIS_PROJECT_IS.md`
> - **What are we actually doing?** → `docs/13_EXPERIMENT_PLAN.md`
> - **Current status** → `PROGRESS.md`

---

## The current phase, in one paragraph

**Current notebook delivery:** [NB12 S4b training](notebooks/NB12_S4B_Confirmation.ipynb)
adds 18 confirmation runs; [NB12R reporting](notebooks/NB12R_S4B_Report.ipynb)
verifies and reports them afterward. No new annotation. Local checks passed,
Kaggle execution pending. [Run instructions](docs/23_S4B_CONFIRMATION.md).

**Full-plan correction, 2026-09-09:** S5 detection/segmentation and S9 integrated
pipeline have **not started**. All five legacy baseline rows already exist;
the matched ResNet-50 comparison is now complete (9/9 runs × 60 epochs).
All four recovery notebooks are HF-verified; see [the recovery audit](docs/21_RECOVERY_COMPLETION_AUDIT.md).
Finishing NB10 is not completion of the original experimental plan.

The user has accepted the retained **17-architecture S2 sweep as complete
(153/153)**, with nine mislabeled Small runs excluded. Next, the
manual-supervised S5 route will reuse existing masks with no new annotation.
`NB11_S3_Manual_SAM2_Agreement.ipynb` is now optional/deferred at the user's
request; independent SAM2 agreement and blind consistency remain unmeasured.
See [the S3 scope decision](docs/22_S3_MASK_COMPARISON.md).

A camera mounted **below and in front of a vehicle** photographs one tyre. We have a prepared, verified dataset of those images — 418 unique photographs from **12 tyres**, labelled with a three-level *mileage proxy*. We are not building hardware. We are running a **broad, controlled, explainability-grounded comparative study**: many architectures, many techniques, classification + detection + segmentation, with Grad-CAM and its relatives used as a **measuring instrument** rather than a garnish.

**Current execution point:** NB07 is complete and its public, three-seed XAI
gate selected **RegNetY-16GF, DenseNet-121 and ResNet-50**. NB06 Stage-B OFAT
is complete on those three architectures and fold 1 only. Public HF holds
**108/108 completed runs**, each with rolling/best checkpoints and final metrics.
NB08 and NB09 are now executed and verified on HF: 63 stress rows, all 27
prediction files, and all four ensemble/calibration tables. NB10 also executed
and NB10R now publishes all ten implemented figures, including Figure 3. H2 is explicitly inconclusive/undefined,
and the conformal coverage target was not met across folds. See
[the completion audit](docs/19_NB08_NB10_COMPLETION_AUDIT.md) for remaining
reporting work and the current control mean of 0.375184. Known fold leakage
remains. The tyrelib v12 notebook runs each model in a disposable
child Python process. Public telemetry showed the long-lived Jupyter process
retaining 0.17–0.30 GB after every epoch; v10 therefore completed two models,
trained a third through epoch 45, then correctly stopped at the 88% host-RAM
guard. In v11 the child exits after a run and Linux reclaims all model,
optimiser, CUDA, image-library, and serialization state. A RAM-paused child is
automatically restarted to resume the same HF checkpoint. The selected models
and scientific recipe are unchanged. v12 also repairs the common
`ACTIVE_KAGGLE_ACCOUNTS=('acct1')` missing-comma edit and reports untouched
runs as `NOT STARTED`, not `AT RISK`.

**Focus for this phase: tyre wear.** Alignment is deferred — see `docs/13 §3` for why it is the harder half, not the easier one.

---

## The question the study asks

A naive benchmark would ask *"we trained 20 models, which was most accurate?"* On this data that would be **ranking noise**. Two deliberately stupid baselines, on the dataset's own folds:

| Baseline | fold 0 | fold 1 | fold 2 | mean |
|---|---:|---:|---:|---:|
| 10 colour numbers from a 64×64 thumbnail | **0.952** | 0.399 | 0.123 | 0.491 |
| 9 texture numbers from the tread band | 0.354 | 0.119 | **0.976** | 0.483 |

A fold-to-fold swing of 0.12 → 0.98 will swamp any difference between architectures.

> ### So we ask a different question:
> ### **Not "which model is most accurate?" but "which model actually looks at the tread?"**

Accuracy is not identifiable on 12 tyres. **Where the evidence comes from is** — and it is what determines whether a model survives contact with a tyre it has never seen.

---

## The contribution

> A **shortcut-aware, explanation-grounded benchmark** of tyre-wear recognition under small-sample conditions.
>
> We train a wide sweep of architectures and techniques, generate architecture-appropriate saliency for every one, and measure **how much of each model's evidence falls on the tread** rather than on background, dirt, or factory paint stripes. We then test whether that predicts cross-fold generalisation better than validation accuracy does.

| | |
|---|---|
| **A measured problem** | We quantified the shortcut risk before designing around it |
| **A new metric family** | Tread Evidence Ratio and siblings — **zero manual annotation required** |
| **A falsifiable hypothesis** | Pre-registered. It can come out "no", and that is still a result |
| **Breadth with discipline** | Every configuration: 3 folds × 3 seeds. No single-run numbers |

**One line:** *on small tyre datasets, validation accuracy is noise and attribution location is signal.*

---

## How it fits together

```
   A. CLASSIFICATION  ── the only real labels we have
      3-class ordinal mileage proxy
      ~30 architectures × 12 technique factors × 3 folds × 3 seeds
                │ CAMs
                ▼
   B. LOCALISATION  ── weakly supervised, no boxes needed
      CAM → box · SAM2 reference · YOLO26 on pseudo-labels
                │ CAM prompts SAM2
                ▼
   C. SEGMENTATION  ── zero-shot teacher, distilled students
      SAM2 pseudo-masks → SegFormer / U-Net / DeepLabV3+
                │ tread mask
                ▼
   D. XAI MEASUREMENT  ── the primary axis
      TER · BAR · SAR · faithfulness · stress tests
```

Segmentation earns its place by being **the instrument that makes the evidence metrics computable** — not as a competing deliverable.

---

## The metrics that make this new

| Metric | Meaning | Target |
|---|---|---|
| **TER_norm** | area-normalised share of attribution on the tread | **> 1.0** |
| **BAR** | attribution outside the tyre entirely | low |
| **SAR** | attribution on **factory paint stripes / lettering** | low |
| Insertion / Deletion / **ROAD** | is the explanation faithful at all? | — |
| Cross-fold spread, cross-seed saliency IoU | is any of it stable? | — |

`SAR` is worth singling out: new tyres carry coloured paint stripes and white lettering from the factory. That is a direct, free giveaway for the `low` class — a concrete, named, measurable shortcut we have not seen reported in the tyre-vision literature.

---

## Honest scope

| We claim | We do **not** claim |
|---|---|
| A comparative study of what learns tread structure vs tyre identity | A deployable tyre-wear product |
| Ordinal **mileage-proxy** classification | Measured tread depth in millimetres |
| Evidence-location and faithfulness measurements | Certified safety or roadworthiness |
| Deferred alignment, with a stated reason | Toe, camber, thrust angle, or four-wheel alignment |

**Sample size is 12 tyres.** Stated everywhere. The dataset's labels come from workshop odometer folders, not from a gauge — so the honest description is *mileage-proxy classification*, never "worn / not worn".

---

## Repository map

| Path | What it is |
|---|---|
| **`docs/00_WHAT_THIS_PROJECT_IS.md`** | **Plain-language explanation. Start here** |
| **`PROGRESS.md`** | **Live status log — what's done, blocked, next** |
| **`docs/13_EXPERIMENT_PLAN.md`** | **The study design — model zoo, technique axis, run budget, figures** |
| **`docs/14_XAI_PROTOCOL.md`** | **XAI methods, metrics, faithfulness, pre-registered hypotheses** |
| **`docs/15_ANNOTATION_GUIDE.md`** | **Windows install + click-by-click annotation walkthrough (solo)** |
| **`docs/16_HF_REPO_STRUCTURE.md`** | **Hugging Face layout, run IDs, push tiers, retention** |
| **`docs/18_STAGE_A_RESULTS.md`** | **Stage A, 153 valid + 9 quarantined architecture substitutions — and why two folds are not usable as evidence** |
| **`docs/17_DATA_LOGGING_SCHEMA.md`** | **Every column we record — ~185 per epoch** |
| `docs/12_DATASET_FINAL_V1.md` | The dataset: what it supports, what it can't, the difficulty floor |
| `docs/01_CONCEPT.md` | Problem formulation and observability analysis |
| `docs/02_CAPTURE_AND_PREPROCESSING.md` | How the data was captured, capture guidance, filter table |
| `docs/04_MODEL.md` | Model zoo implementation reference — exact configs, CAM layers, runtimes |
| `docs/05_TRAINING_KAGGLE_HF.md` | Multi-account Kaggle + HF infrastructure, and the six bugs |
| `docs/06_EVALUATION.md` | Metrics, statistics, anti-patterns |
| `docs/07_ROADMAP.md` | Stages S0–S9, team split, interfaces |
| `docs/08_RISKS_AND_MY_OPINION.md` | Honest assessment |
| `docs/09_RELATED_WORK.md` | Annotated bibliography |
| `docs/10_VISION_TECHNIQUES.md` | Technique catalogue |
| `docs/11_APP.md` | Optional app (first on the cut list) |
| `docs/LOGBOOK.md` | Long-form decision record |
| `scripts/dataset_shortcut_probe.py` | Reproduces the difficulty-floor baselines |

---

## Scale

| | |
|---|---|
| Runs | ~800 (3 folds × 3 seeds throughout) |
| Compute | **~440 GPU-h** (measured T4 throughput, not guessed) |
| Available | 30 GPU-h/week per Kaggle account (see the note below) |
| Wall-clock | **~4 weeks at 4 accounts · ~15 weeks at 1.** See `docs/07` |
| Training length | **60-epoch equal budget, no early stopping** |
| Annotation | 418 images, SAM2-assisted, **solo — ~3.5–4 h over 3–4 sittings** |

Infrastructure pattern: lock-free coordination by arithmetic, per-writer registry shards, one HF repo as the only permanent store. `docs/05`.

---

## Quickstart

```bash
conda activate cv_conda            # every python command in this repo
python scripts/verify_env.py
python scripts/dataset_shortcut_probe.py --root "D:/Dataset Download/Tire Dataset Prepared/FINAL"
```

---

## Status

- [x] Review-1 submitted · literature audit · technique catalogue
- [x] Pilot dataset `final_v1` prepared, verified, analysed
- [x] Difficulty-floor probes — the baselines every model must beat
- [x] **Approach redesigned as a comparative XAI-grounded study**
- [x] S0 infrastructure + public-HF resumability exercised across 4 accounts
- [x] S1: NB01A reporting and all nine matched ResNet-50 runs HF-verified
- [ ] S2: 153 valid runs; nine quarantined; original broader zoo incomplete
- [ ] S3: manual masks verified; SAM2-only comparison not verified
- [x] **S4 technique OFAT — NB06 complete, 108/108 verified on HF**
- [ ] S5 detection/segmentation
- [x] S6 Stage-B XAI gate
- [x] S7 stress tests — 63/63 rows verified
- [x] S8 ensembles — outputs verified; calibration limitations recorded
- [ ] S9 integrated pipeline + component ablations
- [ ] S10 Review-3 + paper; NB10R reporting recovery verified (10/10 implemented figures)

Live detail in `PROGRESS.md`.
