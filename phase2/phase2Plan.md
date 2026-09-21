# Phase 2 — improve workstation models with labelled video views

> **Final status — 21 September 2026:** [All 15 runs and NB06 are verified complete](phase2Seed3Completion.md). Installed selections still match the final registry. Earlier partial/deferred/run-next statements below describe historical snapshots. No retraining or extra labeling is required.


## Runnable training revision — 20 September 2026

**NB00 has passed on Kaggle and its HF report is independently verified. NB01–NB06 are implemented; see [phase2RunNotebooks.md](phase2RunNotebooks.md) for the current run order and settings.**

The user delegated the remaining decisions. The runnable revision keeps all three original mid tyres and all video frames in training, avoiding overlap under every possible unknown mapping. The fixed split is **386 train / 81 validation / 103 test images (8/2/2 tyres)**. Held-out evaluation covers old low/high tyres only. Exact video mapping is no longer a blocker for this revision and is not fabricated.

SegFormer ignores conflicts; YOLO uses a documented training-only tyre/tread union and disables the incompatible exclusive-class auxiliary semantic branch. HRNet uses new polygon-derived points with weight 0.25 as weak training labels; only existing human points serve as validation/test truth. No further manual labeling is required to run this revision. The original ZIP and raw annotations remain unchanged.

Default: 15 combined-data runs (five models × three seeds), 60 epochs each. Optional old-only comparison adds 15 runs with matched update budgets. Two local GPU processes and optional static multi-session workers are implemented. Real-model GPU resume tests run automatically before long training. Actual Kaggle T4 results, completed training, reporting and workstation integration remain execution stages, not missing preparation.

The earlier 6/3/3 stratified split, mandatory exact identity matching, mandatory acceptance of training point proposals, and default 30-run study below are historical design proposals superseded by this section. Source-v1 still correctly records unassigned splits; the separate training overlay supplies the runnable revision.


Date: 19 September 2026. Status: preparation implemented; labels, identity matching, training and accuracy results pending.

## Objective and isolation

Build a versioned dataset combining the existing tyre data with newly labelled frames from the three workshop videos, then measure whether retraining improves workstation classification, masks and boundary localization. Improvement is a hypothesis, not a guaranteed result of adding frames or increasing model size.

All new files, generated data, notebooks, checkpoints, reports and any future workstation copy live under `phase2/`. No edits to existing data, labels, notebooks, models, registry, `undertand3.md`, reports, or application files. The baseline manifest protects 376 pre-existing source/document/input files by SHA-256. `phase2_protect_originals.py` checks them. Its scope is those listed files, not every dependency/cache byte.

Progress is recorded in [pahse2Progress.md](pahse2Progress.md), retaining the user's requested spelling. Training implementation will follow [phase2TrainingContract.md](phase2TrainingContract.md). Label only the new frames using [phase2LabelingGuide.md](phase2LabelingGuide.md).

## What the user confirmed

The three videos show **three different mid-mileage tyres, all belonging to the original 12**. They provide new views and capture conditions, not additional independent tyres. `manifests/phase2_video_identity.json` preserves these statements. Exact clip-to-original-session matching remains unknown.

The existing geometry contract lists three mid-mileage sessions: `mileage_040000__session_001`, `mileage_070000__session_001`, and `mileage_090000__session_001`. Do not assign these to clips by filename order or apparent wear. The user must confirm correspondence, supported by original contact sheets if needed. If exact identification is unavailable, keep grouping unresolved; visual similarity alone is not conclusive identity.

Mid-mileage is a source-backed classifier label here, not a measurement of wear. A frame without a recognizable tyre must not automatically enter positive classifier training because its video has that class.

All new videos being mid-mileage also creates a domain/class confound: workshop machinery, motion or video compression could become cues for the mid class. Balanced sampling reduces domination but does not remove that association. Evaluate original low/high performance and development-only background/region sensitivity. Future low/high video from independently identified tyres would address this gap; it is not extra labeling requested in this preparation step. Do not promise that these three mid videos will improve every class.

## Extraction already completed

| Source | Video stream duration | Native portrait PNG size | Candidates |
|---|---:|---:|---:|
| `video1.mp4` | 13.0996 s | 2160 × 3840 | 27 |
| `Video2.mp4` | 46.4551 s | 1080 × 1920 | 93 |
| `Video3.mp4` | 15.8000 s | 2160 × 3840 | 32 |
| Total | about 75.35 s | no downscaling/cropping | **152** |

Sampling targets are 0.0, 0.5, 1.0 seconds, etc., taking the first decoded presentation timestamp at or after each target. This accommodates the variable frame cadence instead of assuming every clip is exactly 30 fps. The largest timestamp offset is 0.0338 s. Display-matrix rotation is applied once (90 degrees clockwise); no model chooses orientation. PNG encoding introduces no additional lossy compression, though the original video is already compressed.

All 2,245 source frames were decoded to reach/verify the samples. Source hashes were checked before and after extraction. The 152 PNGs occupy 434,995,206 bytes (about 415 MiB); exact decoded-pixel duplicate groups: zero. Similar neighboring frames still exist and are not independent samples. Nothing was rejected automatically for blur.

Frame files: `data/new_frames/video1/`, `video2/`, `video3/`.
Machine-readable provenance: `manifests/phase2_frames.json` and `.csv`.
Gallery: [review/phase2_review.html](review/phase2_review.html), eight contact sheets.

The extraction manifest is immutable source provenance; user-confirmed class/identity metadata joins separately by `video_id`. Null class fields in the extraction rows are not a contradiction of the class confirmation. The eventual dataset builder must resolve the join and fail on missing/conflicting metadata.

## Dataset and annotation design

Maximum unique-image pool before quality review: **418 existing originals + 152 new frames = 570**. Reuse all existing labels unchanged. Do not relabel old images. Existing 4,180 derivatives are transformations of the 418 originals, not 4,180 additional observations; every derivative follows its parent's split.

Default Phase 2 training uses clean originals plus accepted video frames, with synchronized online augmentation. This avoids inheriting a ten-to-one old-derivative weight advantage. Any reuse of stored derivatives must preserve parent identity and the same effective source weighting, and must be declared before training.

New annotation work is two visible-region polygon labels per usable frame: `tyre` and `tread`, plus `ignore` only where the boundary/region is genuinely unjudgeable. Tyre includes the visible tyre rubber; tread is the visible tread crown, not each groove. These masks overlap. No manual detection boxes are needed: derive boxes and YOLO polygons from the reviewed masks. No per-frame mileage typing: inherit the user's mid label.

For HRNet, first derive six guide-row boundary proposals from reviewed tread polygons, at the original normalized guide locations. Show those points on the same image for human acceptance/correction. Do not require the user to redraw six points everywhere when the polygon already specifies them. Missing, occluded, outside-frame and ambiguous boundary states must be explicit, not fabricated coordinates. If a row has multiple disconnected tread intervals, do not use unconditional outermost extrema; request review. Independently accepted held-out point labels must not simply be asserted correct because the polygon converter generated them.

Before using derived geometry labels, compare the polygon-derived definition with the old 120 human point labels on available development/training examples. This is a compatibility check, not new labeling of old photos or permission to change the old test. Preserve old six-point labels as authoritative where available; other old images can still train classification/segmentation without inventing human geometry labels.

Quality review distinguishes: usable/clear, usable/blurred, occluded, clipped, no target, and unlabelable. Keep realistic hard frames when their ground truth is still identifiable. Reject only with a recorded reason; keep the source PNG. No automatic deletion of difficult cases to inflate test accuracy. Exact duplicates would share a duplicate group; perceptual-hash similarity is a review cue only.

Video2 is much longer, so training must not give its tyre 93/152 of the new-data influence by accident. Use class → tyre → domain/source sampling, described below. Keep all accepted labels available even when the sampler does not see each neighboring frame in every epoch.

## Split policy: identities before frames

Never random-split extracted frames. Every original, augmented derivative and video frame of one physical tyre must have a single role for a given experiment. Exact clip/session matching is a hard gate before dataset freeze, not a blocker to extraction or polygon labeling.

Proposed primary split: **6 train / 3 validation / 3 test tyres**, with one low, one mid and one high tyre in validation and in test, and all remaining tyres in train. Confirm class/group counts from the original manifest first. This gives one of the three video tyres to each role. Assign identities deterministically within class with a recorded seed and freeze before model results. Never select roles by which model performs well. If counts or identity confirmation invalidate this design, revise the protocol openly before training.

This is a new Phase 2 split, not an overwrite of Phase 1 folds. A fair comparison on it requires new general-pretrained initialization: the old tyre-trained checkpoints may already have seen the new held-out tyres. In particular, all three mid sessions belong to the old HRNet training set. Warm-starting those weights and calling a mid-video test tyre unseen would be false.

The known videos have already influenced development, so call this a **group-held-out internal evaluation on a known cohort**, not a pristine external validation. One test video/one mid tyre cannot establish broad video generalization. Report the result with that limitation and per-tyre values. The video-only classifier test contains just one class: report mid recall/confidence/flip rate, not a meaningful three-class macro-F1 from that subset alone.

If exact mappings cannot be recovered, keep all three possible matching original mid groups and their videos together. That avoids identity leakage but prevents the proposed stratified video test. Such work would be a deployment adaptation experiment with fresh independent video required for generalization claims; do not silently substitute random temporal splits.

## Models and controlled comparisons

| Model | Phase 2 role | Planned initialization and treatment |
|---|---|---|
| MobileNetV4 | Fast full-image mileage-proxy classifier | Same verified architecture/head as workstation, general ImageNet-pretrained initialization, new Phase 2 weights |
| ResNet50 | Comparison classifier | General ImageNet pretrained; preserve CORAL threshold-count semantics |
| **YOLO26m-seg** | Stronger instance-segmentation candidate replacing Nano only if qualified | COCO-pretrained **Medium segmentation**, not detection-only; masks and automatically derived boxes |
| SegFormer-B0 | Lightweight overlapping tyre/tread segmentation and boundary extraction | General MiT-B0 initialization, two independent binary region channels |
| HRNet-W18 | Six tread boundary positions with explicit invalid/visibility handling | General pretrained backbone; update the geometry contract for missing labels and portrait-consistent inputs |

YOLO26m-seg is the primary stronger candidate. YOLO26s-seg is an explicitly documented alternative only if Medium fails measured T4 training or GTX1650 deployment constraints. No silent downgrade, no YOLO27 preview dependency, and no reason to jump to the largest model without evidence. Keep old Nano as a frozen operational comparator. The existence of larger variants is confirmed by [Ultralytics](https://docs.ultralytics.com/models/yolo26); its public performance figures are not measurements on our tyres or workstation.

For each of the five architectures:

1. **A — old-only control:** eligible old training images, new Phase 2 split, Phase 2 recipe.
2. **B — combined:** identical setup plus accepted new frames belonging to training tyres only.

Use the same pretrained source, seed, input contract, optimizer budget and augmentation in A/B. Fix optimizer-step budget as well as nominal epochs, because adding data otherwise changes both data and compute. A Phase 2 epoch is a frozen number of sampled updates, not an assertion that every source frame is visited once. Record exposure counts.

Plan three seeds (1, 2, 3) and 60 fixed-budget epochs: **5 architectures × 2 conditions × 3 seeds = 30 measured runs**, after lightweight non-result smoke tests. Do not run the broad old architecture zoo again. Publish both validation-selected deployment checkpoint and fixed epoch-60 endpoint; no early stopping by default. Timing preflight must produce the actual Kaggle-hour estimate before launching the full schedule.

The A/B comparison estimates the effect of new training views. Comparing old workstation weights against new weights is a practical product comparison but also changes training protocol and, for YOLO, architecture; do not attribute all gains to the extra frames.

## Training direction to freeze before implementation

Classifiers: sample classes uniformly, then tyres uniformly within class. For a tyre with both domains, sample old/video with probability 0.5/0.5; for old-only tyres use old images. Thus mid video does not swamp low/high. Dense/geometry tasks: sample training tyres uniformly, then old/video uniformly when both have eligible labels. Log per-domain/per-class/per-tyre counts and never balance using validation/test data.

Use physically plausible training-only brightness/contrast, mild blur/compression and modest affine perturbations to reflect handheld video. Transform images, masks, ignore maps and boundary states together. Avoid vertical flips and arbitrary 90-degree rotations of the canonical portrait contract. Horizontal flips swap semantic left/right points. Arbitrary affine transforms require re-intersecting the transformed tread region with guide rows; simply rotating the six old points does not keep them on their guide rows. Disable a transform for HRNet when valid targets cannot be reconstructed.

Letterbox/resize and coordinate inversion must be shared by training, evaluation and the Phase 2 workstation. Prefer portrait 384 × 512 for HRNet/SegFormer to match the workstation's full view, with padding recorded and ignored in losses. Classifiers retain their documented full-image model resolution; YOLO uses 640 as the initial resource-test proposal. These are proposal values, not a frozen dataset/training contract yet.

Segmentation uses overlapping tyre/tread supervision with ignored pixels excluded; do not collapse it into mutually exclusive softmax classes. HRNet uses valid-point loss masking and a visibility/validity head only if label-state diversity supports learning it. A positive-width paired parameterization or differentiable order penalty is a training candidate to prevent crossed predictions, not post-hoc sorting of outputs. Freeze the chosen adapter after development-only smoke/label checks, apply the same adapter in A/B, and keep raw outputs/rejection coverage visible.

Full training/optimizer parameters, loss weights, effective batch sizes, augmentation ranges and package hashes must be frozen after label intake and resource smoke tests, before measured runs. The current plan intentionally does not invent tested batch sizes or claim that stock YOLO supports exact mid-epoch restart.

## Evaluation and promotion

- Classification: three-class macro-F1, QWK, per-class recall and confusion matrices on original held-out tyres; separately mid recall/confidence on video. Include no-target rejection behavior and known uncalibrated-score limitations.
- Masks: tyre/tread Dice/IoU, boundary error, missing-mask rate, YOLO box/mask AP where defined. Report old-photo and video domains separately, with per-tyre macro averages so the long clip cannot dominate.
- Points: normalized/native-coordinate mean and median errors, tail error, visible-point coverage, crossed/collapsed rate, missing/false-visible outputs. Do not lower error by silently rejecting hard points; show coverage and error together. Geometry test points need reviewer acceptance.
- Video: run models on full held-out clips for latency, visible failure episodes and temporal behavior; score localization accuracy only at labeled timestamps. Report raw model results and display smoothing separately. Raw point motion alone is not an error measure on a moving/rotating tyre.
- Workstation: measure cold load, warm median/p95 latency, memory including practical headroom, whole-pipeline update rate, portrait/seek behavior and evidence export on GTX1650. Initial product target is at least 3 completed inspections/sec with no growing queue; measure it, do not assume it from accelerator benchmarks.

Validation promotion proposal: improved per-tyre video mask/boundary quality, fewer crossed/missing outputs, and no more than 0.02 absolute degradation in old-photo macro-F1 or mean Dice, while meeting memory/throughput limits. These 0.02 tolerances and 3/sec target are engineering defaults to freeze before runs, not scientific significance claims. Select on validation only, then report held-out test once. If the held-out test disappoints, record it and do not repeatedly tune against it while retaining “test” status.

Only promote components that earn their place; all five need not replace their existing counterpart. Create a separate `phase2/workstation/` copy and Phase 2 registry/launcher. The existing Tread Station remains usable. Keep immutable old/new checkpoint revisions and an easy choice between versions.

After evaluation and recipe selection, an optional separate **deployment refit** may use all accepted old/new labels with one predeclared seed. It gets new run IDs and is explicitly trained on the former internal test tyres. The preceding test scores do not certify that all-data refit or make those same clips unseen. A later fresh video cohort is needed for an independent deployment claim.

## Ordered deliverables and gates

| Step | Deliverable | Gate/status |
|---|---|---|
| P2-00 | Isolated docs, progress and original-file hashes | Created |
| P2-01 | 152 native frames, provenance and review gallery | Completed |
| P2-02 | New-image polygon labeling and original identity mapping | Complete; labels reused and all possible mid identity matches are training-only |
| P2-03 | Label QA, reviewed guide points, dataset package and group split | Source ZIP verified; separate 386/81/103 overlay frozen; derived training-label policies implemented |
| P2-04 | Phase 2 Kaggle preflight/resume smoke notebooks | NB00 Kaggle/HF verified; NB01–NB06 delivered; all five local pretrained GPU resume tests pass |
| P2-05 | A/B runs for five architectures, three seeds | Notebooks ready: 15 combined runs by default; optional 30-run A/B; actual T4 smoke precedes training |
| P2-06 | Paired per-tyre report and held-out evaluation | NB06 implemented; awaits trained results |
| P2-07 | Isolated Phase 2 workstation and operational comparison | Planned; no change to existing app |
| P2-08 | Optional all-data deployment refit and fresh-cohort follow-up | Separate from the internal experiment |

No HF publication, Kaggle training, model download/replacement or original-file modification was performed during preparation. All 152 new frames are labeled, the dataset is uploaded, and NB00 passed on Kaggle/HF. The immediate task is NB01, then NB02–NB05 and NB06 per `phase2RunNotebooks.md`. No further manual labels are required for the revised training protocol.
