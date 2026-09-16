# Learned geometry integration log

<!-- current-status:start -->
> **Current status (15 September 2026):** [Completed work and remaining validation](../docs/CURRENT_STATUS.md). The report is refreshed; the app, learned-geometry integration and target-assisted alignment software exist. Dated plans below retain their original context.
<!-- current-status:end -->

## Scope frozen — 15 September 2026

Integrate HRNet-W18 and matched SegFormer-B0 as optional image-space geometry.
Policy: seed 1, final epoch 60 for BOTH models, chosen before local inference;
no seed selection from the published test ranking. Preserve the four existing
models, raw masks, portrait orientation and target-assisted alignment workflow.

Full RGB image → PIL bilinear resize to 384 wide × 512 high → ImageNet
normalisation. No crop, flip, rotation search, CLAHE or train-mean fallback.
Guide rows and point order come from the immutable training contracts.
HRNet's coordinates are unconditional proposals, not visibility confidence.

Planned verification: source/compact hashes and identities; strict tensor loading;
agreement with original inference adapters; coordinate bounds/order and clipping;
blank-input behavior; image/video overlays and stale-frame handling; same-GPU
latency/memory; evidence round trip. Physical accuracy is not evaluated here.

## Implemented and verified

- Both seed-1 epoch-60 checkpoints downloaded at immutable revisions. Source hashes
  match existing audits; compact inference exports are 39.3 MB HRNet / 15.1 MB
  matched SegFormer. Optimizer states are retained only in the cached source files.
- Original model sources and experiments remain unchanged. Inference uses FP32;
  training reports used AMP. This integration is not a fresh reproduction of the
  published aggregate test score or a new benchmark ranking.
- Full-image preprocessing matches the original loader exactly. Strict weights,
  HRNet coordinate decoding and SegFormer native-logit point extraction passed.
  Guide rows map using the exact frozen normalised row positions; x uses width−1.
- The recipe provides Off / HRNet / HRNet + Matched SegFormer. Paired mode is the
  default when both exports exist so disagreement is visible. It adds roughly
  30–40 ms for matched SegFormer in the checked examples. No ensemble averaging.
- Amber dots and connecting sides: HRNet; cyan squares: matched SegFormer;
  purple: HRNet midpoint line; red proposals: heuristic review flags. Widths are
  image pixels. Crossed/collapsed pairs withhold widths and connecting geometry.
  No invisible-boundary confidence, hidden sorting or train-mean substitution.
- Three consistent frames enable display smoothing. Gaps, seeks, source/mode
  changes and flags reset it. Raw points remain in JSON and the explanation table.
  Sliders/redraws do not advance tracking. Original view stays clean. Learned
  overlays can be hidden independently from the older silhouette geometry.
- Saved cards include `learned-boundaries.png`, raw and display points, flags,
  timestamps, source dimensions, model/contract hashes and timings. Restoration
  reproduces the saved overlay without inference. Target-assisted alignment can
  retain these as supporting evidence, never as its physical coordinate reference.

## Evidence from local tests

`check_learned.py` passed on GTX 1650 in cv_conda. Two five-run warm checks on one
native held-out still observed HRNet 51.41–88.46 ms and matched SegFormer
33.32–35.83 ms; the latest five-run sample is retained in the JSON report.
Both include preprocessing, forward pass, extraction and device synchronisation;
exclude weight loading and file decoding. Peak allocated GPU memory for this
two-model check: 182.0 MiB. These are small operational measurements, not a
statistical latency study or whole-workstation FPS.

`check_learned_ui.py` passed on all three supplied videos: paired GPU inference,
portrait dimensions, overlay toggle, Original view, seek reset, diagram, PNG/JSON
evidence, restoration, and disabling releases learned model references. Recorded
full-pipeline peak allocations were about 271–279 MiB at the checked frames.
Detailed reports: `results/learned-integration-check.json` and
`results/learned-ui-check.json`; screenshots and evidence folders are nearby.

**Video quality finding:** the reviewed 1.5-second captures from video1 and Video2
contain crossed HRNet boundaries; all three show HRNet/matched disagreement, and
some matched boundaries are missing. The shown still has much closer agreement.
This demonstrates a deployment-domain limitation, not a software pass on accuracy.
Stable-looking proposals can still be wrong. Existing videos have no point ground
truth, so no localisation accuracy claim is made for them. Frame aspect, framing,
appearance and motion differ from the small training set; no crop/rotation patch
was introduced to conceal that difference.

## Running / reproducing

Use the existing launcher. For a fresh checkout, activate `cv_conda`, run
`python prepare_learned.py` from `prototype`, then launch the app. This reuses
verified exports offline and resumes partial downloads. No Kaggle rerun, label
change, HF upload or global dependency replacement was performed.

Checks: `python check_learned.py` and `python check_learned_ui.py` in the same env.
Next research work is video-domain validation and component evaluation. The
integration is available; full S9 / physical alignment accuracy is not established.
