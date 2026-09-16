# NB23 — saved segmentation geometry baseline

<!-- current-status:start -->
> **Current status (15 September 2026):** [Completed work and remaining validation](CURRENT_STATUS.md). The report is refreshed; the app, learned-geometry integration and target-assisted alignment software exist. Dated plans below retain their original context.
<!-- current-status:end -->

**Execution update:** NB23 is now HF-verified at
`84bcbdd61b39b9dfccd2461af86a5de562995d38`. Independent reconstruction confirms
all 72 boundaries; 66/66 eligible points scored, median error 8 px, mean normalised
error 0.90%, P06 excluded. No rerun needed. Next: [NB24/NB25 and HRNet plan](30_S9_HRNET_AND_COMPLETION_PLAN.md).
The implementation/pending-execution notes below are historical.

Implementation: 14 September 2026. Ready for Kaggle execution; not yet an executed result.

## What to run

Open [NB23_S9_Geometry_Baseline.ipynb](../notebooks/NB23_S9_Geometry_Baseline.ipynb).
Choose **CPU / Accelerator None**, Internet ON, one copy, enable **HF_TOKEN**,
and **Run All**. Attach nothing. No original images, model weights, dataset or
GPU training are required. No NB21/NB22 rerun is needed to start NB23.

The notebook uses already published native SegFormer-B0 seed-1 masks, not manual
masks, bounding boxes or new model inference. Dependencies are installed into an
isolated directory and used by a child process, preserving the base environment.

## Fixed evidence and unresolved labels

Source HF revision: `05bf37c0f2067b0119ef057a2442d7759cf9ac51`.
Annotation hash: `4110bdcc413ff327d7ab2e6d107efac071008dc73b1bf04143a6c9a116516e18`.
The notebook deliberately uses the last verified submission, not an automatically
selected newest file. All P06 points remain on review hold, even its visible upper
pair; its diagram is diagnostic only. It can therefore run while P06's issue
description and right-side visibility remain unresolved. No user labels are edited.

Each image uses its recorded fold's seed-1 prediction, after checking actual
validation inclusion, training exclusion, source-image hash, model/job identity,
protocol hash, native dimensions and evaluated epoch-60 status. These checks do
not resolve existing fold 0/2 cross-tyre leakage or make 12 sessions independent tyres.

## Interpretation

At each guide row, take the leftmost/rightmost predicted tread-mask pixel.
Reject empty rows and exact image-edge extrema. Compare only complete, visible,
non-held labels. Report absolute pixel error, width-normalised error, coverage
and rejection count. Rejected predictions are not zero-error successes.
There is no success threshold tuned on this pilot, no physical-angle claim,
and no automatic approval/rejection of HRNet.

Output includes all 72 point records with explicit exclusions, per-image and
overall summaries, 12 predicted-mask diagrams, input hashes and implementation.
PNG colours: grey predicted tread mask, cyan submitted points, yellow accepted
predicted boundary. These are not original-image overlays.

Review coverage and per-image diagrams alongside error. The six proposed points
may not mean exactly the same thing as mask extrema. This experiment tests that
feasibility before extra annotation or HRNet training. PatchCore still lacks
verified reference evidence; full S9 is not completed by this notebook.

## Storage and recovery

Read-only source cache and output uploads are each capped at 20 MiB; no checkpoint
downloads. Results use a content-addressed contract namespace under
`s9/s9-geometry-pilot-r1/`. Sources are pinned and cached atomically. An interrupted
comparison can be rebuilt deterministically from small cached inputs; it has no
training epochs to lose. A fresh Kaggle session re-downloads only small JSONs.
One batched upload at completion, every 30 minutes if necessary, and a catchable
Stop/error flush of existing outputs. Rate limits use backoff. Forced termination
cannot upload. Retry Run All after failure; do not start multiple workers.

## Validation and next action

Boundary/empty-row/clipping, coverage, P06 and incomplete-label exclusion, and
notebook syntax tests passed. Public prediction metadata and split identity
checks cover all 12 images. Local end-to-end mask decoding was not run because
the local pycocotools installation did not finish; the attempted installations
were stopped. Kaggle dependency installation, decoding, diagrams and HF upload
remain to be verified by execution. No assistant HF writes were made.

After Run All, return the executed NB23. Verify SUMMARY/POINTS and public output,
then decide whether to clarify the boundary definition, reuse segmentation
geometry, or design a separate HRNet comparison. Do not begin HRNet training yet.
