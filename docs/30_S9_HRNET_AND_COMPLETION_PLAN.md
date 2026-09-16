# S9 expansion and HRNet comparison plan

<!-- current-status:start -->
> **Current status (15 September 2026):** [Completed work and remaining validation](CURRENT_STATUS.md). The report is refreshed; the app, learned-geometry integration and target-assisted alignment software exist. Dated plans below retain their original context.
<!-- current-status:end -->

**Latest:** HRNet and matched SegFormer training/report are now complete and
HF-verified. No more annotation or training is requested. [Results and integration
plan](36_MATCHED_GEOMETRY_RESULTS_AND_INTEGRATION.md) supersede earlier build/run
instructions below. Full S9 and physical-angle claims remain incomplete.

**Completion update:** NB28/NB29 are now HF-verified, all three seeds ×60 epochs.
No rerun needed. [Current results and next step](34_HRNET_COMPLETION_AND_RESULTS.md)
supersede pending-training and repair-run instructions below.

**Implementation update, 15 September:** user confirms 12 sessions are 12 distinct
physical tyres. NB26–NB29 now implement preflight, GPU resume smoke, three-seed
training and report; 72/24/24 images in 8/2/2 tyre groups. Coordinate-only HRNet-W18
replaces the proposed visibility head because all 720 labels are visible. Actual
implementation/settings and test limits are in [docs/32](32_HRNET_NOTEBOOK_RUN_GUIDE.md),
which supersedes the proposed/deferred-training wording below. No more labels needed.

**15 September execution update:** NB24/NB25 now completed and HF-verified;
120/120 records received. No preparation/intake rerun needed. Full quality review
and physical-tyre identity/split lock remain before training. All 12 local identity
entries are blank. [Current audit](31_S9_120_ANNOTATION_COMPLETION.md) supersedes
pending-Kaggle notes below; training is still planned, not executed.

14 September 2026. User authorises a larger annotation effort and notebook preparation.
**Redesigned: NB24 prepares ALL 120 images in ONE package; NB25 accepts ONE JSON.
No intermediate batch review or switching. No training yet.**

## Why 120 more, and what you actually do

Budget **120 new images**, one package, excluding all 12 old pilot images.
This is an operational ceiling for this round, not a power calculation or a
guarantee of enough independent tyres. All 120 are available immediately; annotate
at your pace across sittings with no mandatory intermediate review.
No new photographs, masks, boxes or physical measurements are requested in this round.

At the fixed 25%, 50%, 75% image-height guides, mark left/right tread-face to
shoulder transitions. Image-relative left/right; ignore internal grooves, paint,
lettering and outer silhouette where the intended transition is indistinct.
Use Uncertain, Occluded or Outside frame instead of guessing. Clipped edges are
not visible boundaries. Every Visible issue selection needs a factual short note.
All-invisible images are legitimate observations, not instructions to invent labels.

NB24: **CPU, one copy, Internet/HF_TOKEN, attach the existing prepared dataset,
Run All**. No BATCH setting. Download **ALL_120_IMAGES.zip (93,854,472 bytes /
89.51 MiB)**, extract everything, and open ANNOTATE.html beside its images folder.
Native JPEGs are unchanged and old masks/predictions hidden.
Save JSON frequently; load it to resume. Return JSON or run NB25.

NB25: **CPU, one copy, Internet/HF_TOKEN, attach only one exported JSON, Run All**.
Read the X/120 count. Partial submissions are safely preserved, not called complete.
One export covers all 120 images. No repeated originals or batch switching.
The larger lossless package has a 160 MiB safety cap instead of 20 MiB. HTML loads
only the selected external JPEG, not a huge embedded image array. Save JSON often;
Load saved JSON to resume. NB25 may preserve partial exports but only one final
intake is required. The new r2-all120 HF namespace preserves old packages.

## Identity and split gate — before training

Actual session counts are 40, 62, 20, 20, 57, 3, 2, 5, 60, 41, 41, 67.
Selection rotates across available sessions and naturally exhausts tiny groups;
120 frames do not create 120 independent tyres. Selection uses hashed image IDs,
not model error, wear class, or annotation outcome. Visual near-duplicates still
need review even though exact originals are unique.

The small SESSION_IDENTITY.json ledger lists all 12 session names with blank tyre
IDs. The user can instead explain in chat which sessions depict the same physical
tyre. Unknown is acceptable; invented separate identities are not. No private
records need to be uploaded. Image labels alone cannot resolve physical identity.

After identity review, freeze disjoint tyre groups for training, development and
held-out evaluation, aiming roughly at 60/20/20 of usable images without splitting
a tyre across partitions. Exact counts depend on group sizes; do not promise this
ratio if grouping makes it impossible. If too few confirmed identities exist,
report session-held-out exploratory evidence only, or separately agree new capture.
Old pilot images are development evidence, never new untouched test evidence.

Lock the image lists and protocol hash before fitting. Do not tune on held-out
errors, move hard test images into training, or replace ambiguous targets with
segmentation-generated labels. A small blinded repeat-label check can be proposed
after the full submission; it is additional work and is not silently counted as done.

## HRNet training notebook design — delivered after label/split review

Proposed model: **HRNet-W18 backbone**, explicitly adapted for six fixed-guide
boundary targets, not a human-pose checkpoint silently renamed as a tyre model.
Verify the exact implementation, pretrained weight provenance and parameter
signature in the training notebook before allocating GPU time. A different HRNet
width or Lite-HRNet would require a documented change, not an automatic fallback.
The [official HRNet project](https://github.com/HRNet/HRNet-Human-Pose-Estimation)
provides the architectural reference; it does not validate these tyre targets.

Proposed adapter: predict six horizontal coordinate distributions plus visibility
states; y is the known guide, not an independently learned physical landmark.
Coordinate loss is masked for invisible/uncertain points, with a separate
visibility loss. This is an adapted HRNet geometry experiment, not anatomical
keypoint detection or camber/toe estimation. Inspect completed-assignment visible-point
coverage before approving that target and loss design.

Initial compute budget: one T4, native aspect ratio resized to 384x512, AMP,
batch 2 and gradient accumulation if needed, zero loader workers initially.
Start with a short resume/overfit smoke test. Only after it passes: up to 60 epochs,
three seeds, fixed training schedule, development-only checkpoint/threshold
selection. These are proposed settings to freeze after model/resource preflight,
not a claim that this recipe has run or fits every Kaggle session.

The future training notebook must persist model, optimizer, scaler, scheduler,
epoch/batch cursor, deterministic sample order and RNG states, configs, data hashes,
history and hardware metadata. Save local checkpoints at completed optimizer steps;
resume from the last durable checkpoint, not an interrupted in-flight operation.
Use 30-minute batched HF snapshots, major-completion and catchable-Stop flushes,
bounded disk caches, upload retry/backoff and one active writer per run. A forced
kernel/OS kill can lose work since the last HF snapshot; no notebook can promise
zero loss from an unsaved computation. Test stop/resume equivalence before long runs.

## Fair comparison and decision

Compare segmentation-derived points and HRNet on exactly the same reviewed labels,
visibility policy and evaluation groups. Existing NB23 masks are a pilot baseline;
they cannot automatically be used as a fair new test baseline if their S5 training
set overlaps the new held-out group. Audit that overlap and, if needed, retrain the
segmentation baseline on the same permitted training groups. Do not grant one model
access to test-image supervision while calling the comparison controlled.

Report coordinate error normalised by width, coverage/rejection, visibility confusion,
per-image and per-tyre results, latency and memory. Evaluate performance-versus-coverage
using development-set choices, not just conditional error after rejecting hard cases.
Any uncertainty intervals must respect independent grouping; hundreds of points
are not hundreds of independent samples. No automatic HRNet improvement claim.
Retain HRNet only if its measured value justifies the extra complexity for the task;
a negative comparison is still a valid completed experiment.

## What remains for complete S9

| Workstream | Required next evidence | What these 120 labels do not supply |
|---|---|---|
| Geometry/HRNet | Reviewed coordinates, identities/splits, fair model comparison | Physical wheel plane or camber/toe truth |
| PatchCore | Defined anomaly target, defensible nominal-reference pool, separate evaluation examples and labels | “No visible issue” is not verified healthy |
| Integration | Frozen component inputs, missing-input behaviour, held-out component-removal comparisons | Adding models alone is not validation |
| Reporting | Results, negative findings, provenance, scope and remaining limits | Original-plan completion without evidence |

[PatchCore's original implementation](https://github.com/amazon-science/patchcore-inspection)
uses nominal-reference features. First decide what anomaly and reference condition
can actually be documented here. Existing geometry annotation does not create that
reference evidence. If only visual nominality can be supported, explicitly agree
that narrower claim; do not silently call it physical health. No PatchCore training
notebook is issued before this data decision.

Physical alignment remains the prototype's separate calibration/measurement task.
Completing this geometry experiment would not validate physical angles. Full S9
closure requires either the original evidence or an explicitly accepted narrower
scope. Do not let additional labels imply otherwise.

## Current evidence and deliverable status

NB23 is complete at HF `84bcbdd61b39b9dfccd2461af86a5de562995d38`: 66/66 eligible
points, median error 8 px, mean width-normalised error 0.90%; P06 excluded. All
72 predicted boundaries and summaries were independently reconstructed and checked.
This motivates a fair comparison, not an automatic need for HRNet.

NB24/NB25: local selection, native package size/hash and validation/syntax checks;
Kaggle execution and real new-batch labels pending. Existing NB21–NB23 executions,
prototype and frozen manuscript are preserved. No assistant HF publication.
The training and integration notebooks are planned, not represented as implemented;
their contracts depend on the input and split decisions above.
