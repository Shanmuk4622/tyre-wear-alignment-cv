# Project understanding — Improve the prototype models

Last updated: 19 September 2026.

## Purpose and evidence boundary

This is the working context requested by the user, using the exact filename `undertand3.md`. Maintain it when later work changes the models, prototype, experiments, or interpretation. It summarizes the repository; it does not replace frozen protocols, original results, or dated evidence.

This context pass read all 73 pre-existing Markdown paths, including ignored manuscript build copies and the bundled NumPy random-module license. Five paths were byte-identical copies, covered by reading their matching content and checking hashes. The generated report was also compared against its already-read authoring source, with differing content read separately. File structure and selected implementation entry points were inspected. This was not an exhaustive source-code review, fresh model evaluation, or independent re-audit of remote Hugging Face artifacts. Completion and test results below are attributed to existing repository records.

The task was renamed **Improve the prototype models**. The current request is context gathering and this reference file; it does not itself request retraining or prototype changes.

## How to resolve contradictory documentation

Many files prepend new status above older instructions without removing the old text. An old “run now,” unchecked stage, or original proposal is not today's execution instruction.

- Use `docs/CURRENT_STATUS.md`, docs 34–36, and `prototype/LEARNED_GEOMETRY_LOG.md` for completed geometry work and integration.
- Prototype export behavior is further updated on 16 September in its README and VALIDATION.
- The 18–19 September manuscript revision in `ACCESS_latex_template_20260513/` is newer than the 15 September HTML report and many root summaries.
- Earlier claims that folds 0/2 have established tyre leakage are superseded by the newer interpretation: recorded group IDs are disjoint in each fold, and the user confirmed 12 distinct physical tyres. Visual similarity raised a historical suspicion; it did not prove shared identity. The membership audit is not independent physical identification or a replay of every data loader.
- Some older README statements describe zero-label/SAM2 supervision, unbuilt applications, or deferred alignment software. Actual S5 used manual masks; the desktop app and target-assisted alignment software already exist.
- Preserve original records and namespaces. Do not silently rewrite historical results or treat completed notebooks as rerun requests.

## Repository map

| Location | Actual role |
|---|---|
| `prototype/` | Native PySide6/PyTorch Tread Station, inference adapters, video handling, geometry, calibration, evidence and checks |
| `tyrelib/` | Training infrastructure, scientific runtimes, protocols, reporting and notebook generators |
| `notebooks/` | Kaggle deliverables NB00–NB32, recovery variants and NBT1; saved execution outputs are provenance |
| `scripts/` | Annotation preparation, dataset probes, audits, fault/recovery checks, report builders and verifiers |
| `docs/` | Concept, historical plans, numbered protocols/audits, current status and decision log |
| `docs/report/` | September 15 HTML/Markdown manuscript, editable source, figures and frozen evidence |
| `ACCESS_latex_template_20260513/` | Newer IEEE Access manuscript, sections, bibliography, analyses, source package and build verification |
| `outputs/` | Local diagnostics, annotation packages, previews and QA artifacts; not remote training truth |
| `Videos/` | Three supplied workshop clips used for qualitative deployment checks |
| `tyre_study/` | Legacy directory mentioned in older docs; currently empty in this checkout |
| `ENVIRONMENT.md`, `environment.yml` | Local `cv_conda` guidance; not an exact universal Kaggle replay lock |
| `PROGRESS.md`, `docs/LOGBOOK.md` | Extensive dated history; newer conclusions supersede older interpretations |
| `.env` | Ignored local secrets; not read or copied during this context pass |

Large inputs, weights, caches, generated results, and execution archives are generally ignored. Hidden `.build` packages are copies, not new primary manuscripts. The older proposed modular layout in ENVIRONMENT is not the current implemented layout. Do not delete legacy folders or evidence merely because a newer path exists.

## Scientific scope and data

- The observed classification target is a three-level **ordinal mileage proxy**, derived from workshop mileage folders. It is not measured tread depth, certified wear, or roadworthiness.
- Dataset: 418 unique native 1152 × 1536 photos, 4,180 augmented derivatives at 768 × 768, 12 capture groups/user-confirmed distinct tyres. Derivatives are not independent tyres.
- Acquisition diversity is small. External generalization remains unmeasured even with group-disjoint splits.
- There are 418 manual masks. Canonical regions overlap: tyre is all nonzero labels; tread includes tread and marking. The high tread/tyre ratio means attribution localization mostly distinguishes tyre from background, not tread from shoulder.
- Propagated masks originally had incorrect geometric replay; corrected masks passed NBT1. Version strings alone were insufficient, so fingerprints and actual geometry tests matter.
- Colour, texture, framing, and annotation-side-channel baselines expose shortcut risk. High validation scores or plausible saliency do not prove physical wear reasoning.
- Additional 120-image geometry labels contain six visible boundary points per image, at three fixed guide rows. All-visible labels do not train a visibility classifier. “No visible issue” is not independent proof of a healthy reference.

## Completed study, without expanding its claims

| Work | Recorded state |
|---|---|
| Stage A classification | 153 valid runs: 17 architectures × 3 folds × 3 seeds; nine nominal ConvNeXtV2-S runs were actually ResNet-18 substitutions and remain quarantined |
| Recovery baselines | NB01A/B, NB03A and NB10R completed and audited; nine matched random-init ResNet-50 runs exist |
| XAI gate | NB07 selected RegNetY016, DenseNet121 and ResNet50; 1,208 evidence rows and 35 faithfulness rows |
| Stage B OFAT | 108 completed runs on fold 1; NB07 precedes NB06 despite filenames |
| S4b confirmation | 18 completed runs on ConvNeXt Tiny/MobileNet; random initialization harmed both, sampling effects were model/endpoint dependent |
| S5 dense tasks | 81 completed 60-epoch runs: 36 semantic, 36 YOLO, nine genuine RT-DETRv2-R18; NB17 completed |
| Fusion/reporting | NB18 analysis and NB19/20 reporting completed; fixed equal fusion did not improve mean classification macro-F1 |
| Geometry labels | NB24/25 completed: 120 images and 720 visible points |
| HRNet | NB26–29 completed, three seeds × 60 epochs |
| Matched SegFormer | NB30–32 completed, same split and epoch budget; no rerun needed |

S5 evaluates clean originals with manual supervision and native-coordinate metrics. Its crop experiment feeds crops to a frozen full-image classifier; it is not an experiment training classifiers on crops. CORAL predictions use threshold count, which can differ from score argmax. Preserve selected-checkpoint versus fixed-final endpoints.

The XAI gate requires randomization decorrelation above 0.05 and finite insertion/deletion evidence; it should not be rewritten as a positive faithfulness threshold. H2 is undefined with insufficient marking/damage-positive coverage; H3 is untested. Historical conformal coverage and empty-set handling have limitations.

## Geometry result and its limits

Both models use a 72/24/24 image split across 8/2/2 tyres, three seeds, 60 epochs, full-image 384-wide × 512-high inputs. The HRNet-W18 coordinate adapter has 9,603,962 parameters; matched SegFormer-B0 has 3,714,658. Point supervision versus dense-mask supervision differs, so this is not an equal-label architecture ablation.

Recorded seed-average held-out boundary errors:

| Model | Image-width error | Native pixel error |
|---|---:|---:|
| HRNet-W18 | 1.312% | 15.10 px |
| Matched SegFormer-B0 | 1.721% | 19.80 px |

The relative reduction is 23.75% in this experiment. The test covers only two tyres; 432 seed/point records are dependent observations, not 432 subjects. Both have 100% reported boundary coverage on that test. A prior 12-image pilot/66-point mask baseline is a different cohort and cannot be ranked directly against these errors.

Frozen result revisions: HRNet `a92c0f9c5c1b78c6a06e13d51e18722195230658`; matched comparison `bbe586c6f00cf12ae4cac8b2e9cb4f875b272abb`. These were read from repository audits, not freshly fetched during this task.

## Prototype behavior and code ownership

Tread Station is a native desktop application, despite the workspace folder being named “web Applicarion.” It already supports image/video inspection, comparison, original/overlay wipe, evidence restoration, learned boundaries, shape/edge explanations and target-assisted alignment.

| Files | Responsibility |
|---|---|
| `prototype/app.py`, `theme.py` | Qt workflow, persistent inference/stream workers and display controls |
| `engine.py`, `classification.py` | Base-model loading, full-frame classifier contracts, two-region masks, assistance and inspection records |
| `registry.py`, `prepare_models.py` | Four fixed base artifacts, immutable revisions, hash verification and compact exports |
| `prepare_learned.py`, `learned_geometry.py`, `learned_diagram.py` | Two learned models, frozen adapters, raw proposals, review flags and display smoothing |
| `video.py`, `media_export.py` | Portrait decoding and separate-process overlay video export |
| `shape_geometry.py`, `edge_geometry.py`, diagram/comparison modules | Silhouette PCA, boundary-line and rim-ellipse hypotheses, support/rejection evidence |
| `alignment.py`, `alignment_ui.py` | Camera profiles, distinct ground/wheel ChArUco targets and reference-based single-wheel geometry |
| `evidence.py` | Local frame/mask/JSON/HTML records and restoration |
| `bootstrap.py` | Isolated local dependency/cache paths and Qt setup |
| `check_*.py`, `test_contracts.py` | Software, adapter, UI, video, geometry, evidence and export verification |

Base models are MobileNetV4 (default classifier), ResNet50 (comparison), S5 SegFormer-B0 (default region model), and YOLO26n segmentation (alternative). Base artifacts are fold 1/seed 1/final epoch 60; YOLO uses final EMA. Geometry artifacts are the separately declared seed 1/final epoch 60 for both models, not the best test seed.

There are **two different SegFormer roles/checkpoints**: S5 region segmentation and the later same-split matched geometry experiment. Do not mix them.

Learned inference uses FP32, full RGB resized to 384 × 512 and ImageNet normalization; no crop, rotation search or constant fallback. HRNet predicts six unconditional coordinates. Matched segmentation extracts boundary positions at the same frozen guides. Coordinates map through width−1 and the frozen guide normalization. Crossed/collapsed/missing boundaries and disagreement are reported; raw outputs remain saved. Smoothing begins after three consistent frames and is display-only.

Video controls use portrait pixels, a maximum 1280-pixel working dimension and a replaceable frame slot. Result overlays belong to the exact analysed frame. Seek/source/mode changes invalidate stale results. The engine still contains a legacy optional automatic-rotation branch, but the current UI submits rotation=0 after source orientation; do not mistake the unused branch for deployed model-driven rotation selection.

The optional YOLO assistance is separately labelled SegFormer output. Agreement, smoothing, small residuals and a visible overlay are not accuracy evidence. Local records have no automatic uploads.

September 16 exports provide exact shown-frame PNGs and duration-preserving silent 10 fps MP4s. Video export captures settings at launch and uses a separate CPU process, with cancellation and atomic publication of completed output.

## Main documented weakness for model improvement

Workshop video differs from the small still-image training set in framing, aspect, appearance and motion. Reviewed video1/Video2 frames show crossed HRNet boundaries; all three clips show inter-model disagreement, and some matched segmentation boundaries are missing. The clips have no point/pixel ground truth, so those checks establish failure examples and operational behavior, not a measured video accuracy score.

Future improvement should distinguish preprocessing/coordinate defects, deployment-domain model errors, and display behavior. Preserve raw predictions and frozen test identities; do not hide failures with coordinate sorting, a constant fallback, or test-seed selection. No specific retraining intervention is selected by this context pass.

An incidental source observation: `draw_learned` currently duplicates its HRNet width-label drawing block. This is an unmodified rendering duplication, not an established cause of model boundary errors.

## Alignment and deferred work

Image-space PCA, boundary lines, and rim ellipse ratios do not establish camber/toe. The implemented calibrated path uses separate ground and wheel ChArUco boards, camera intrinsics/distortion, a defined vehicle reference frame and wheel-side sign conventions. Synthetic tests exist; real fixture/rack accuracy, calibration, runout/remount repeatability and physical camera validation remain open.

PatchCore lacks an independently justified healthy reference pool. SAM2/manual agreement and blind repeat annotation remain deferred. Broader original Tier 5/6/8 experiments, shared-backbone/rotation ablations, labelled video validation and physical reference measurements are not completed merely because the narrower study and app exist.

## Future training requirements from the user

- Deliver Kaggle-compatible `.ipynb` notebooks with clear execution instructions and troubleshooting. Python source/generators are the maintained implementation; do not hand-edit embedded blobs or erase executed provenance.
- Support the requested dual-T4 session. Document actual GPU use: recent frozen protocols deliberately use one GPU; do not change a completed recipe merely to occupy the second card.
- Prefer direct Kaggle inputs/downloads. Respect the output storage budget; measure scratch capacity rather than assuming 1 TB. Keep bounded caches and delete only verified disposable local artifacts after successful persistence.
- Use Hugging Face account `Shanmuk4622`, secret name `HF_TOKEN`, and the established dataset repository `Shanmuk4622/tyre-wear-study` where applicable. Never expose token values.
- Preserve model/optimizer/scheduler/scaler, RNG, next-batch cursor, partial metrics, logs, configs, identities and provenance. Future resumability must handle completed optimizer steps within an epoch where required, with an actual stop/resume equivalence check.
- Batch normal HF publication around 30 minutes; flush at major cell/model completion and catchable interruption. Respect server backoff and a shared per-user budget; the historical 128/hour figure is a constraint to handle, not an assurance of today's service quota.
- Save locally before publication, use immutable consistent snapshots/journals and authenticate/retry reads as well as writes. Fetch remote state before deciding resume/skip; failed status with a good checkpoint is resumable.
- Report the last successfully published checkpoint. Forced kernel death cannot run an interrupt handler, and deleted ephemeral storage cannot recover unpublished work. Do not promise otherwise.
- Never silently substitute another architecture, lower the scientific budget, discard nonfinite batches, or ignore a failed preflight to make execution finish.

Older classification/S5 workflows mostly resume at completed epochs. Later HRNet/matched runtimes preserve completed optimizer steps. HRNet's repair retries the same batch with scaler backoff then bounded FP32 recovery. Matched SegFormer's deterministic resize repair passed the recorded T4 resume check with zero parameter difference. Preserve repair hashes and frozen protocol compatibility.

Prior failures worth remembering: checkpoints saved remotely but never fetched; stale registry skipping/retraining mistakes; unsafe ownership races; staging inside the 20 GB output area; masks with wrong transform keys; silent architecture substitution; cgroup versus host RAM confusion; growing telemetry buffers; retained process memory; mismatched checkpoint/status snapshots; nondeterministic interpolation backward.

## Local environment and reporting

Every Python command uses `cv_conda` (local interpreter `C:\Users\shanm\dev\envs\cv_conda\python.exe`). The prototype isolates dependencies in `.vendor` and preserves CUDA PyTorch. Local checks were recorded on a GTX 1650 4 GB, using a 32 MB inference-worker stack to avoid the observed Windows stack failure. Published local latency samples are small operational measurements, not guaranteed video throughput.

For the September 15 report, edit `docs/report/manuscript.source.md` and rebuild with the report scripts; frozen evidence stays unchanged. For the newer paper, edit `ACCESS_latex_template_20260513/main.tex` and `sections/`; `access.tex` is the supplied template example. The newer paper has 31 references and ten related-study comparisons, with generated analysis and source-package verification. Some author declarations and physical/external validation remain outstanding.

## Maintenance log

- 2026-09-19: Completed repository Markdown/context review and selected code-structure inspection; created this file. No training, HF writes, model changes, application changes, or tests were run in this context task. Existing validation results are documentation evidence, not newly reproduced results.
- On future updates: change current facts in place, append a short dated action/result entry, retain precise evidence links/revisions, and distinguish implemented, locally tested, Kaggle-completed and scientifically validated claims.
