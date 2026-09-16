# Documentation index

<!-- current-status:start -->
> **Current status (15 September 2026):** [Completed work and remaining validation](CURRENT_STATUS.md). The report is refreshed; the app, learned-geometry integration and target-assisted alignment software exist. Dated plans below retain their original context.
<!-- current-status:end -->

## Start here

**Latest:** [NB31/NB32 verified results and integration plan](36_MATCHED_GEOMETRY_RESULTS_AND_INTEGRATION.md).
No rerun. Matched geometry comparison complete; S9 integration remains next.
Older repair/run-now entries below are historical.

**Current:** [NB30 verified; rerun repaired NB31 only](35_MATCHED_SEGFORMER_COMPARISON.md).
Resume-check numerical repair passed local tests; GPU check pending. NB30 remains valid.

**Run next:** [Matched SegFormer comparison — NB30–NB32](35_MATCHED_SEGFORMER_COMPARISON.md).
Implemented, local checks passed; GPU results pending. No new annotation or HRNet rerun.

**Current:** [HRNet completion and results](34_HRNET_COMPLETION_AND_RESULTS.md).
NB28/NB29 verified, three seeds ×60 epochs; next matched segmentation comparison.
No rerun. Older pending/repair entries below are history.

**Current:** [NB26/NB27 verified; NB28 numerical repair](33_HRNET_SMOKE_VERIFIED_AMP_REPAIR.md).
Resume updated NB28 only, from 18 completed batches in epoch 1. Older stage-run
instructions below are retained as history.

**Current:** [HRNet NB26–NB29 run guide](32_HRNET_NOTEBOOK_RUN_GUIDE.md).
User-confirmed 12 distinct tyres, frozen 72/24/24 image split; implemented notebooks,
local checks passed, Kaggle GPU execution pending. Older identity questions below
are historical, not a new blocker.

**15 September:** [120-image intake complete and HF-verified](31_S9_120_ANNOTATION_COMPLETION.md).
Next: point-quality review and tyre-identity/split lock, then HRNet training.
No NB24/NB25 rerun needed; older preparation notes below are history.

**Current:** [HRNet annotation expansion and full S9 plan](30_S9_HRNET_AND_COMPLETION_PLAN.md).
NB24 prepares ALL 120 in one ZIP; NB25 saves one JSON. No intermediate batch review.
NB23 is verified, not pending. Training implementation follows approved inputs/splits.

**Run next:** [NB23 saved-mask geometry baseline](29_S9_GEOMETRY_BASELINE.md).
CPU, no attachments or training; P06 remains excluded. Kaggle execution pending.

**Current action,14 September:** [NB22 verified; pilot review and next plan](28_S9_PILOT_REVIEW_AND_NEXT_PLAN.md).
Revised upload verified (11/12); P06 right visibility and issue description remain,
then predicted-mask baseline before training. [Prototype updates](../prototype/README.md)
include geometry and target-assisted calibration, separate from the frozen report.

1. [Full illustrated report](report/REPORT.html) — detailed author-review manuscript, 16 visuals and evidence-linked tables.
2. [Current progress](../PROGRESS.md) — stage completion and next actions.
3. [Claims and remaining gaps](report/CLAIMS_AND_LIMITATIONS.md) — supported and unmeasured claims.
4. [Reproducibility appendix](report/REPRODUCIBILITY.md) — revisions, run order and local rebuild.
5. [Repository guide](REPOSITORY_GUIDE.md) — where things belong.
6. [Submission checklist](report/SUBMISSION_CHECKLIST.md) — human/venue checks still required.

Current edition: 12 September 2026. Historical design prose is not proof of implementation. Twelve capture sessions are not twelve verified tyres; mileage is not measured wear; S5 is manual-supervised; full original S9 remains deferred.

## Document catalogue

| Document | Role |
|---|---|
| [00 — Overview](00_WHAT_THIS_PROJECT_IS.md) | Accessible introduction; current report governs scientific claims |
| [01 — Concept](01_CONCEPT.md) | Original system and observability rationale |
| [02 — Capture](02_CAPTURE_AND_PREPROCESSING.md) | Preparation design/history |
| [02 — Rig](02_RIG_BUILD.md) | Historical hardware design, not a completed rig |
| [03 — Data](03_DATA.md) | Earlier design; use final-v1 for executed collection |
| [04 — Model](04_MODEL.md) | Earlier pipeline design, not execution evidence |
| [05 — Training](05_TRAINING_KAGGLE_HF.md) | Kaggle/HF background; later runtime amendments also apply |
| [06 — Evaluation](06_EVALUATION.md) | Evaluation design; reconcile endpoints with report |
| [07 — Roadmap](07_ROADMAP.md) | Stage sequence and current amendments |
| [08 — Risks](08_RISKS_AND_MY_OPINION.md) | Design risk discussion |
| [09 — Related work](09_RELATED_WORK.md) | Historical notes; current reference audit controls submission claims |
| [10 — Techniques](10_VISION_TECHNIQUES.md) | Catalogue, not all implemented |
| [11 — Application](11_APP.md) | App design, not validated deployment |
| [12 — Dataset](12_DATASET_FINAL_V1.md) | Prepared collection and split caveats |
| [13 — Experiment plan](13_EXPERIMENT_PLAN.md) | Original proposal plus scope amendments |
| [14 — XAI](14_XAI_PROTOCOL.md) | Explanation definitions and constraints |
| [15 — Annotation](15_ANNOTATION_GUIDE.md) | Existing annotation workflow |
| [16 — HF structure](16_HF_REPO_STRUCTURE.md) | Persistent artifact organisation |
| [17 — Logging](17_DATA_LOGGING_SCHEMA.md) | Run/log contract |
| [18 — Stage A](18_STAGE_A_RESULTS.md) | Result audit and quarantine |
| [19 — NB08–NB10](19_NB08_NB10_COMPLETION_AUDIT.md) | Earlier audit; exact conformal values in current report |
| [20 — Full-plan closure](20_FULL_PLAN_CLOSURE.md) | Implemented versus missing scope |
| [21 — Recovery](21_RECOVERY_COMPLETION_AUDIT.md) | Baselines, random-init, identity and figures |
| [22 — S3](22_S3_MASK_COMPARISON.md) | Deferred SAM2/blind consistency |
| [23 — S4b](23_S4B_CONFIRMATION.md) | 18-run same-fold confirmation |
| [24 — S5](24_S5_MANUAL_DENSE_TASKS.md) | 81 dense runs and dated repairs |
| [25 — S9](25_S9_FUSION_AND_REMAINING_GAPS.md) | Exploratory fusion and deferred full pipeline |
| [26 — S10](26_S10_REPORTING_AND_NEXT_STEPS.md) | Frozen reporting snapshot and documentation handoff |
| [27 — S9 pilot](27_S9_SMALL_ANNOTATION_PILOT.md) | 12-image guided labels, JSON export/intake, review before training |
| [28 — Pilot audit and next plan](28_S9_PILOT_REVIEW_AND_NEXT_PLAN.md) | NB22/HF verified, visual review, targeted P06 follow-up and segmentation-first decision |
| [Logbook](LOGBOOK.md) | Dated project history |
| [GitHub setup](GITHUB_SETUP.md) | Earlier setup instructions |

## Report companions

- [Editable-text report](report/REPORT.md) and [authoring source](report/manuscript.source.md).
- [References](report/REFERENCES.md), [claims audit](report/CLAIMS_AND_LIMITATIONS.md), [reproducibility](report/REPRODUCIBILITY.md), [submission checklist](report/SUBMISSION_CHECKLIST.md).
- [Evidence manifest](report/evidence/evidence_manifest.json), [build provenance](report/BUILD_PROVENANCE.json), [validation](report/VALIDATION.json).

For numerical claims use: pinned HF artifacts/manifests → dated verification audits → current report/progress → historical proposals. Operational completion does not override scientific quarantine. A local manuscript update does not alter the frozen HF notebook report.
