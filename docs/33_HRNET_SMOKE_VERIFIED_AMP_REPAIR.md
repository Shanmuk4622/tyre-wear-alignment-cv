# NB26/NB27 verified; NB28 numerical recovery repair

<!-- current-status:start -->
> **Current status (15 September 2026):** [Completed work and remaining validation](CURRENT_STATUS.md). The report is refreshed; the app, learned-geometry integration and target-assisted alignment software exist. Dated plans below retain their original context.
<!-- current-status:end -->

**Completion update:** NB28/NB29 are now HF-verified, all three seeds ×60 epochs.
No rerun needed. [Current results and next step](34_HRNET_COMPLETION_AND_RESULTS.md)
supersede pending-training and repair-run instructions below.

15 September 2026. **Run updated NB28 only. Do not repeat NB26/NB27.**

## Verified public progress

- NB26 preflight commit: `32d13296fafd2253426392e009cd98fb315cb91b`.
- NB27 smoke commit: `240dbb9ebfb51686a7d0899959c340f658755c4c`.
- Latest inspected HF main / NB28 emergency upload:
  `92f580df3ae5a5231a4b446f2752dd0014b19a12`.
- Protocol: `351c6638783996f46cf9efd99ec6689de726e045288054f240193cea61385712`.

Public preflight/smoke contracts match the local frozen contract. Preflight reports
72/24/24 images in 8/2/2 user-confirmed tyre groups. Smoke verifies the expected
9,603,962-parameter HRNet-W18, Tesla T4, Torch 2.10.0+cu128, peak allocated GPU
memory 613,136,384 bytes. Resumed losses exactly equal uninterrupted losses and
maximum parameter difference is **0.0**. This validates that four-step path,
not all possible full-training batches or the subsequently added recovery branch.

Seed 1 has **18 completed batches in epoch 1, zero completed epochs**. Published
history has 18 finite losses. The 116,819,541-byte checkpoint's HF LFS SHA-256
matches STATUS: `8635b6d212e8be6004e7572f0e8177a04ed0e7a39ce8a67207bfc6bfbf1d42e2`.
Only 135,573 bytes of metadata were read; the checkpoint tensor payload was not
downloaded. Its hash is checked again by the notebook when restoring.
[Machine audit](../outputs/hrnet_completion_audit/AUDIT.json).

## Cause and repair

The saved traceback is `Nonfinite gradient; prior durable checkpoint retained`,
not CUDA out-of-memory or failed HF publication. The original step unscaled the
gradients, then raised on a nonfinite norm **before** GradScaler could update its
scale. This prevents recovery from an FP16 overflow and repeats the failure when
resuming the same checkpoint. The exact offending GPU operation was not replayed
locally; persistent underlying nonfinites must still be detected, not ignored.

The new checkpoint-compatible runtime adapter:

1. Restores the same batch's RNG state and retries at a lower loss scale.
2. Makes no optimizer update and does not advance the batch cursor on failure.
3. Allows at most eight AMP attempts, then one full-precision forward/backward.
4. Still aborts safely if FP32 remains nonfinite; it never hides corruption by
   skipping a batch or accepting a NaN update.
5. Records repair source/hash, policy and retry events beside each HF snapshot.

Architecture, labels, split, optimizer/LR schedule and checkpoint namespace remain
unchanged. The original protocol files are intentionally frozen to retain existing
preflight/smoke/checkpoint compatibility. `hrnet_amp_repair.py` is an explicit
runtime amendment, not a silent rewrite of the original source provenance.
NB29 was updated to include that runtime revision/hash in its report.

## What to run

Open the **updated NB28_HRNet_Training.ipynb** in Kaggle: T4, one copy, Internet ON,
HF_TOKEN enabled, same prepared dataset attached, **Run All**. No parameter changes.
It runs a tiny injected-overflow test on GPU, then restores seed 1 and starts the
next unsaved batch (batch 19 of epoch 1 at this audit). It will then complete the
remaining seeds sequentially. Keep the same Torch version as the verified smoke.

Use updated NB29 only after training completes. NB26/NB27 executed files were
not rebuilt. The old failed NB28 output was archived before replacing its executable
copy. No user labels, original protocol code, prototype or HF data were modified.

## Tests and limits

Local real CPU GradScaler tests pass for one-overflow recovery, exactly one
optimizer update, eight-overflow FP32 fallback, and persistent-nonfinite safe
abort without weight/optimizer updates. Original protocol hash and notebook entry
point/syntax checks pass. A GPU recovery regression is embedded before training.
The corrected real failing HRNet batch and long GPU training remain unverified
until the user reruns NB28. Existing 30-minute/seed-completion/Stop persistence and
forced-kill limits remain unchanged. No new HF publication was made by the assistant.
