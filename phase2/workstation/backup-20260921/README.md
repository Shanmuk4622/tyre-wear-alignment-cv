# Vision-Based Detailed Tyre-Wear Recognition and Single-Wheel Alignment Screening

<!-- current-status:start -->
> **Current status (15 September 2026):** [Completed work and remaining validation](docs/CURRENT_STATUS.md). The report is refreshed; the app, learned-geometry integration and target-assisted alignment software exist. Dated plans below retain their original context.
<!-- current-status:end -->

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
> - **Latest: NB31/NB32 verified complete — no rerun** → [Matched geometry results and next integration plan](docs/36_MATCHED_GEOMETRY_RESULTS_AND_INTEGRATION.md). HRNet 1.312% vs SegFormer 1.721% width error on the two-tyre test set. Earlier run-now links below are historical.
> - **Current: NB30 passed; rerun repaired NB31 only** → [Repair/run instructions](docs/35_MATCHED_SEGFORMER_COMPARISON.md). No repeat annotation or HRNet training.
> - **Run next: NB30–NB32 matched SegFormer comparison** → [Instructions](docs/35_MATCHED_SEGFORMER_COMPARISON.md). No new labels; completed HRNet stays unchanged. Kaggle execution pending.
> - **Full illustrated report** → [Read the report](docs/report/REPORT.html) · [Markdown](docs/report/REPORT.md)
> - **All documentation** → [Documentation index](docs/DOCUMENTATION_INDEX.md) · [Repository guide](docs/REPOSITORY_GUIDE.md)
> - **New to the project?** → `docs/00_WHAT_THIS_PROJECT_IS.md`
> - **What are we actually doing?** → `docs/13_EXPERIMENT_PLAN.md`
> - **Current status** → `PROGRESS.md`

---

## The current phase, in one paragraph

**HRNet NB28/NB29 completed and HF-verified:** three seeds ×60 epochs, seed-average
mean error 15.10 px / 1.312% width on two test tyres. No rerun needed. Next is the
same-split segmentation comparison and integration decision, not full S9 closure.
[Verified results](docs/34_HRNET_COMPLETION_AND_RESULTS.md). Older repair notes follow.

**NB26/NB27 are HF-verified; NB28 numerical repair is ready.** Seed 1 has 18 saved
batches of epoch 1. Run updated NB28 only; checkpoint compatibility is preserved.
[Audit and next run](docs/33_HRNET_SMOKE_VERIFIED_AMP_REPAIR.md).

**HRNet notebooks are ready: NB26 preflight → NB27 smoke → NB28 training → NB29 report.**
User confirms 12 distinct tyres; split locked at 72/24/24 images. CPU preflight/report,
T4 smoke/training, one copy only. No new annotations requested. Local tests pass;
Kaggle GPU execution pending. [Run instructions](docs/32_HRNET_NOTEBOOK_RUN_GUIDE.md).

**15 September: NB24/NB25 completed and HF-verified, 120/120 labels received.**
No rerun needed. Next: label-quality review and physical-tyre identity/split lock
before HRNet training. [Verification](docs/31_S9_120_ANNOTATION_COMPLETION.md).
The preparation instructions below are retained for reproduction.

**Current: NB24/NB25 redesigned for ALL 120 images in ONE package.** One 89.51 MiB
ZIP, one annotation page/JSON, no batch switching. CPU, no training. NB23 is verified
(66/66 eligible points; median 8 px). HRNet training follows label and identity/split
review. [Exact run instructions and S9 plan](docs/30_S9_HRNET_AND_COMPLETION_PLAN.md).
The earlier NB23 run-next note below is historical.

**Run next: [NB23 geometry baseline](notebooks/NB23_S9_Geometry_Baseline.ipynb)** —
CPU, Internet/HF_TOKEN, no attachments, Run All. Reuses published segmentation
masks and excludes unresolved P06; no new training. Ready for Kaggle execution,
not yet a completed experiment. [Guide](docs/29_S9_GEOMETRY_BASELINE.md).

**14 September — revised NB22 upload HF-verified:** local annotations match
HF `05bf37c0f2067b0119ef057a2442d7759cf9ac51`; 11/12 mechanically complete.
P06 still needs a visible-issue description and right-middle/lower visibility
correction; its left pair is corrected. No new image batch.
Next: test existing predicted-mask geometry before HRNet training; verified
healthy references remain unavailable. Do not rerun unchanged annotations.
[Audit and next steps](docs/28_S9_PILOT_REVIEW_AND_NEXT_PLAN.md).
These are proposed components, not an already validated final model. The native
prototype and its new calibrated alignment workflow remain separate and untouched.

**Full report prepared for author review, 2026-09-12:** the illustrated
[manuscript](docs/report/REPORT.html) expands the verified S10 evidence into
methods, results, discussion, limitations and references, with22 visuals and
source-generated tables. Reproducibility, claims review, submission checklist
and repository navigation are included. Only3.02MiB of small reporting evidence
was downloaded; no training or HF writes. Next: team/guide review and the required
submission template. [Documentation index](docs/DOCUMENTATION_INDEX.md).
Deferred original experiments remain explicitly uncompleted.

**S5 complete and HF-verified, 2026-09-12:** NB14 semantic36/36, NB15 YOLO36/36,
NB16 RT-DETRv2-R18 9/9, each60epochs; NB17's complete report is public.
No NB13–NB17 rerun needed. Existing manual masks were used, not SAM2 labels.
S9 integration and wider original-plan/reporting work remain unfinished.
[Run instructions and scope](docs/24_S5_MANUAL_DENSE_TASKS.md).

**S4b complete:** [NB12](notebooks/NB12_S4B_Confirmation.ipynb) finished all
18 confirmation runs (60 epochs each); [NB12R](notebooks/NB12R_S4B_Report.ipynb)
published the verified paired report. Random-init hurt both models; sampling
directions did not repeat across both at the primary endpoint. No rerun needed.
[Verified results](docs/23_S4B_CONFIRMATION.md). S5 is now complete; S9 remains unfinished.

**Full-plan reconciliation, updated 2026-09-12:** S5 detection/segmentation is
complete; NB18 exploratory S9 fusion is **complete and HF-verified**, while the
full integrated pipeline remains blocked on inputs/design. No NB18 rerun needed.
All five legacy baseline rows already exist;
the matched ResNet-50 comparison is now complete (9/9 runs × 60 epochs).
All four recovery notebooks are HF-verified; see [the recovery audit](docs/21_RECOVERY_COMPLETION_AUDIT.md).
Finishing NB10 is not completion of the original experimental plan.

The user has accepted the retained **17-architecture S2 sweep as complete
(153/153)**, with nine mislabeled Small runs excluded. Next, the
manual-supervised S5 route reused existing masks with no new annotation.
`NB11_S3_Manual_SAM2_Agreement.ipynb` is now optional/deferred at the user's
request; independent SAM2 agreement and blind consistency remain unmeasured.
See [the S3 scope decision](docs/22_S3_MASK_COMPARISON.md).

A camera positioned **below and in front of a vehicle** photographs one tyre. The prepared dataset has418 unique photographs grouped into **12 capture sessions**, labelled with a three-level *mileage proxy*. Twelve independent tyres are not verified; suspected overlap affects folds0/2. We are not building hardware. The implemented study compares classification, explanation diagnostics and manual-supervised detection/segmentation. Attribution is a diagnostic, not proof of physical wear reasoning or independent generalisation. See the current report for qualified claims.

**Current execution point:** NB07 is complete and its public, three-seed XAI
gate selected **RegNetY-016, DenseNet-121 and ResNet-50**. NB06 Stage-B OFAT
is complete on those three architectures and fold 1 only. Public HF holds
**108/108 completed runs**, each with rolling/best checkpoints and final metrics.
NB08 and NB09 are now executed and verified on HF: 63 stress rows, all 27
prediction files, and all four ensemble/calibration tables. NB10 also executed
and NB10R now publishes all ten implemented figures, including Figure 3. H2 is explicitly inconclusive/undefined,
and observed90% conformal coverage is86.89%/97.56%/93.94% for folds0/1/2;
only fold0 is below nominal, with dependence and empty-set caveats. See
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
| `docs/11_APP.md` | Original app proposal; implemented native Tread Station documented in `prototype/README.md` |
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
- [x] S5 manual-supervised detection/segmentation —81/81 + NB17 report HF-verified
- [x] S6 Stage-B XAI gate
- [x] S7 stress tests — 63/63 rows verified
- [x] S8 ensembles — outputs verified; calibration limitations recorded
- [ ] S9 integrated pipeline + component ablations
- [ ] S10 Review-3 + paper; NB10R reporting recovery verified (10/10 implemented figures)

Live detail in `PROGRESS.md`.
