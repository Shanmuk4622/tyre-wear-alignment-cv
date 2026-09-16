# Matched SegFormer–HRNet comparison — execution complete

<!-- current-status:start -->
> **Current status (15 September 2026):** [Completed work and remaining validation](CURRENT_STATUS.md). The report is refreshed; the app, learned-geometry integration and target-assisted alignment software exist. Dated plans below retain their original context.
<!-- current-status:end -->

**Completion update:** NB31/NB32 are HF-verified at `bbe586c6f00cf12ae4cac8b2e9cb4f875b272abb`.
Three seeds ×60 epochs; T4 smoke delta 0.0. No rerun. [Results and next plan](36_MATCHED_GEOMETRY_RESULTS_AND_INTEGRATION.md).
All repair/run-now instructions below describe the earlier failure and are historical.

## Historical repair instruction: rerun updated NB31 only

NB30 passed on HF at `e32a80300ac415a60e8f142e099b0af7b0969dc7`.
The first NB31 attempt stopped during its smoke test, before seed training.
HF `465112532cdac878904b9fff41a125f58eac2c58` records T4, Torch 2.10.0+cu128,
peak allocated GPU memory 668,976,128 bytes, and maximum resume parameter
difference 0.000166565 versus the 0.00001 threshold. First resumed loss matches
exactly; the next differs by approximately 0.00000435. The namespace contains
only preflight/smoke artifacts, not completed or resumable training seeds.

This is a numerical continuation mismatch, not an out-of-memory traceback.
CUDA bilinear resize gradients are a plausible source of nondeterminism;
[PyTorch documents this limitation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.interpolate).
The exact GPU operation responsible has not been independently isolated.

`segformer_resume_repair.py` uses the same bilinear forward geometry in FP32 and
an equivalent separable-matrix backward to avoid CUDA atomic accumulation.
Strict deterministic algorithms are enabled and TF32 disabled. Both decoder
resizing and loss resizing use this adapter when gradients are needed. Native
evaluation interpolation remains unchanged. Unsupported nondeterministic
operations will fail visibly; the original strict resume tolerance is NOT relaxed.

Local gradient comparison against native CPU bilinear passed (maximum difference
0.00000763), and a real SegFormer model reproduced its stochastic resumed steps
exactly. GPU validation is still pending: NB31 must pass its actual T4 test before
training is permitted. These CPU checks are not a claim that the T4 run passed.

The frozen experiment source is unchanged. The separate numerical adapter and
its hash are disclosed in `RESUME_REPAIR.json` alongside each new GPU snapshot.
The original protocol `8013bf1e6418e6f884a353efa7ce892c31fbaac450f9d7caede636d9e61e2a18`
and completed NB30 remain valid; architecture, data, annotations, epochs and
optimizer budget are unchanged. The failed executed NB31 is preserved under
`notebooks/execution_archives/NB31_SegFormer_Matched_Training_089ce4becbec.ipynb`.

**Open the updated NB31, start a fresh T4 session with the same data and HF_TOKEN,
then Run All. Do not rerun NB30. Run NB32 after all three seeds finish.**

## Original setup and protocol

15 September 2026. NB30–NB32 are ready. This does not change the completed
HRNet experiment or the user's prototype. No additional annotation is requested.

## Run in this order

| Notebook | Environment | What to do |
|---|---|---|
| NB30_SegFormer_Matched_Preflight | CPU | Internet ON, HF_TOKEN enabled. Attach the same prepared dataset used by NB14, including FINAL images and annotations/clean/masks. Run All. |
| NB31_SegFormer_Matched_Training | T4 GPU | Same dataset and secret. Run All in **one session only**. Automatic real-model smoke/resume test, then three seeds. Rerun this notebook if paused as resumable. |
| NB32_SegFormer_HRNet_Comparison | CPU | No dataset needed. Run All after all three seeds finish. Publishes paired results and comparison PNG. |

Leave DATA_ROOT empty for automatic detection when one prepared dataset is
attached. No package download or human labelling task is added. Do not rerun
NB26–NB29. Do not use four copies of NB31: each copy owns all three seeds.

## Frozen comparison

The HRNet contract is taken from immutable HF commit
`a92c0f9c5c1b78c6a06e13d51e18722195230658`, protocol
`351c6638783996f46cf9efd99ec6689de726e045288054f240193cea61385712`.
Rows, point coordinates, guide rows, image hashes and tyre allocation are copied
exactly: **72/24/24 images, 8/2/2 distinct user-confirmed tyres**.

The 72 dense training masks are hash-checked against the frozen S5 protocol
`1f6694577253e0054f7a22df6ec52d30797063cf498b345bd71fe9b98bab93df`.
No validation/test mask is admitted by the training loader. Those images use
the same clicked evaluation points as HRNet. Existing S5 checkpoints are not
reused: their fold allocation is not this tyre-held-out experiment.

Matched settings: seeds 1/2/3, 60 epochs, batch 2, input height/width 512/384,
AdamW learning rate 0.0001, weight decay 0.01, cosine epoch schedule, no
augmentation, frozen batch-normalisation running statistics, fixed final epoch.
There is no test-based seed or checkpoint selection.

SegFormer-B0 has **3,714,658 parameters**, including its new two-channel decoder.
Its ImageNet encoder is `nvidia/mit-b0` at revision
`80983a413c30d36a39c20203974ae7807835e2b4`. Training uses overlapping tyre and
tread masks with binary cross entropy plus soft Dice. This differs from HRNet's
six-point coordinate supervision and pretrained HRNet backbone. It is a matched
split/budget **system comparison, not equal-supervision architectural isolation**.

## Scoring without hiding failures

SegFormer tread logits are interpolated to native resolution, thresholded at
zero (sigmoid 0.5), then intersected with the exact three guide rows. Left/right
extrema produce the six points. Empty rows or boundaries touching the frame are
rejected, following the pilot's conservative extraction rule.

The report includes:

- Raw boundary coverage and conditional error on accepted SegFormer points.
- HRNet error on those exact same accepted points, for a paired subset comparison.
- Full-set SegFormer error with a **predeclared train-only mean-coordinate
  fallback** wherever a boundary is missing, clearly labelled as fallback-assisted.
- All three seeds, each test tyre, all 432 paired point records and a comparison PNG.

Width error is absolute horizontal error divided by native width minus one,
identical to HRNet. Ground-truth coordinates and membership are checked again
in NB32. A missing boundary is not silently removed from the full-set score.

## Saving and failure handling

The new namespace is `s9/segformer-matched-2026-09-15-r1/<contract-hash>` in
the existing `Shanmuk4622/tyre-wear-study` dataset repo. HRNet artifacts are
read-only. The source hashes are part of the new contract; sources accompany
GPU snapshots. The frozen HRNet engine is reused through process-local bindings,
not edited or assigned a new meaning in the old namespace.

Each completed optimizer step saves an atomic local checkpoint: weights,
optimizer, scheduler, AMP scaler, RNG and next-batch cursor. Same-batch numerical
recovery backs off AMP and tries FP32 before aborting; a skipped optimizer update
is never counted as training. A four-step interrupted/uninterrupted model test
must pass on the actual GPU before training.

One synchronous HF writer publishes every 30 minutes, on completed seeds and
catchable Stop/error. A 9-hour guard pauses cleanly. Status is generated from the
durable checkpoint, not a concurrently changing in-memory state. Failed uploads
retain local files and use bounded retries. Hard kernel death or lost Internet
can still lose work since the last successful HF push; no immediate upload can
be guaranteed after a forced kill. Wait for Stop to finish publishing.

One GPU is intentionally used in a dual-T4 session. No dataset copy or old S5
weight downloads; only the small encoder and this experiment's three latest
checkpoints are required. Completed HF seeds are skipped on rerun.

## Verification and decision

Local checks verify all 120 source image hashes, all 72 training masks, exact
HRNet row/split equality, full-resolution CPU model forward/backward, model
parameter count, boundary failure cases, same-batch overflow recovery, stochastic
checkpoint continuation and notebook syntax/schema. Details:
`outputs/segformer_matched_validation/CHECKS.json`.
**Kaggle GPU execution and experimental results remain unverified until run.**

After NB32, review accuracy, coverage and both tyres across all seeds. Do not
choose a winning seed using test error or declare statistical significance from
432 correlated points. Only two test tyres exist, and their HRNet results have
already been inspected; this is exploratory evidence, not a fresh confirmatory
cohort. Timing logged by SegFormer is diagnostic only; an equal-hardware HRNet
timing benchmark would be needed for a comparative speed claim.

Then decide whether the learned HRNet geometry offers enough practical benefit
over the segmentation-derived geometry to integrate. PatchCore reference-data,
physical-angle validation and full S9 claims remain separate unresolved items.
