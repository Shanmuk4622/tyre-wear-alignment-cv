# HRNet notebooks: preflight, smoke, training, report

<!-- current-status:start -->
> **Current status (15 September 2026):** [Completed work and remaining validation](CURRENT_STATUS.md). The report is refreshed; the app, learned-geometry integration and target-assisted alignment software exist. Dated plans below retain their original context.
<!-- current-status:end -->

**Completion update:** NB28/NB29 are now HF-verified, all three seeds ×60 epochs.
No rerun needed. [Current results and next step](34_HRNET_COMPLETION_AND_RESULTS.md)
supersede pending-training and repair-run instructions below.

**Execution/repair update:** NB26 and NB27 now HF-verified. T4 smoke continuation
matched exactly. NB28 stopped after 18 saved batches on a nonfinite gradient;
updated NB28 retries the same batch with scaler backoff and bounded FP32 fallback.
No NB26/NB27 rerun required; use updated NB28 and later updated NB29.
[Current audit](33_HRNET_SMOKE_VERIFIED_AMP_REPAIR.md) supersedes generic GPU-pending
notes below. Real corrected full-training execution remains pending.

15 September 2026. **Implemented and locally checked; Kaggle GPU execution pending.**
User confirms the 12 capture sessions depict 12 different physical tyres. This is
recorded as user-confirmed identity, not independently verified collection metadata.
No additional annotation batch or manual configuration of tyre IDs is required.

## Exactly what to run

| Order | Notebook | Kaggle setup | Result |
|---|---|---|---|
| 1 | [NB26 preflight](../notebooks/NB26_HRNet_Preflight.ipynb) | CPU, Internet, HF_TOKEN, prepared dataset | Frozen split, hashes and all 120 point overlays |
| 2 | [NB27 smoke/resume](../notebooks/NB27_HRNet_Smoke_Resume.ipynb) | T4, Internet, HF_TOKEN, same dataset | Four-step GPU save/reload continuation test |
| 3 | [NB28 training](../notebooks/NB28_HRNet_Training.ipynb) | T4, Internet, HF_TOKEN, same dataset | Three seeds sequentially, 60 epochs each |
| 4 | [NB29 report](../notebooks/NB29_HRNet_Report.ipynb) | CPU, Internet, HF_TOKEN, attach nothing | Audited seed/tyre errors and training-only constant baseline |

**One copy of each notebook; Run All with defaults.** No label JSON attachment:
the verified 120-label revision is read from HF. Leave DATA_ROOT blank with only
one Tire Dataset Prepared dataset attached. A dual-T4 session is acceptable, but
only cuda:0 is used. Do not run four workers or duplicate NB28 sessions.

NB27 requires the matching NB26 publication. NB28 requires a passing matching
NB27 plus the same Torch version. A failed smoke test blocks long training rather
than silently substituting a smaller/different network. Send errors if it fails.

## Frozen protocol

- Source annotations: HF `2b4773914e6eefe421d8ddd74e79c5b84606c9aa`, verified hash
  `8e8fd0734b9f4fe4236e699b804c2f8afb85ae4bb847f8d3c83b0abff5a8769e`.
- Split: **72 training images / 8 tyres; 24 validation / 2 tyres; 24 test / 2 tyres**.
  Whole tyres remain together. Selection balances image counts with a fixed hash
  tie-break, without using model error or label outcome. Exact lists are in CONTRACT.json.
- Actual HRNet-W18 multiresolution features, four branches [18,36,72,144], six
  horizontal coordinate distributions; **9,603,962 parameters** including adapter.
- Pinned `timm==1.0.15` and `timm/hrnet_w18.ms_aug_in1k` pretrained revision
  `7e2c5583769f54514fd87e3ba9de408e33eaba0f`. Download only model.safetensors;
  verify backbone tensor compatibility and record its hash. No pretrained weights
  were downloaded during local development checks.
- Input 512 high × 384 wide, batch 2, AMP, frozen BatchNorm running statistics,
  AdamW 1e-4, weight decay .01, cosine schedule, no augmentation, fixed epoch 60.
  Three seeds are run sequentially; no validation/test-based early selection.
- Cross-entropy against Gaussian horizontal-coordinate targets; the y guide is
  fixed by the native annotation definition. It is not a learned physical landmark.

The implementation follows the native multiresolution feature API in
[timm's pinned HRNet source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.15/timm/models/hrnet.py).
No Lite-HRNet, ResNet substitution or human-pose output head is silently used.

## Explicit change from the earlier proposal

All **720 submitted points are visible**. The training notebook is therefore
**coordinate-only**. It does not train a visibility classifier on a single
visibility class or claim unseen-boundary rejection. Coverage in NB29 means every
submitted visible point was predicted, not that rejection quality was validated.
The earlier visibility-head proposal in docs/30 is superseded for this experiment.

Native point overlays and near-edge flags are available from NB26. Mechanical
completion is not independent anatomical/annotation accuracy. No user labels are
modified or auto-replaced by segmentation masks. The experiment uses the submitted
operational labels; visible-issue observations are not used as healthy evidence.

The old pilot had one image from every tyre and informed protocol development.
Although train/test image and tyre sets for fitting are disjoint, this is NOT an
untouched external cohort. Results have only two test tyres; do not count individual
points as independent tyre samples or claim broad generalisation.

## Resume and HF persistence

After each completed optimizer step, atomically replace state.pt containing model,
AdamW, AMP scaler, scheduler, RNG states, deterministic sample-order cursor and
history. Interrupted writes leave the last completed file intact. HF snapshots
are synchronous: status is derived from the durable checkpoint and the folder is
not mutated during publication, preventing mismatched sidecar/checkpoint snapshots.

Publish every 30 minutes, every completed seed and on catchable Stop/error, with
server Retry-After/backoff. There are no per-epoch, claim or heartbeat commits.
Allow Stop to finish its current step and upload. On restart, verify HF checkpoint
hash and restore the same protocol/seed cursor. A fresh session re-downloads only
the needed checkpoint and model weights, not the dataset. One active writer only.

Forced kernel/OS termination cannot flush. Up to the last 30 minutes of work may
need replay from HF; no zero-loss claim is made for unsaved computation. Local
restarts with preserved working files resume at the last saved step. A graceful
approximately nine-hour limit publishes resumable state; rerun NB28 to continue.
Already trained seeds resume at epoch 60, re-evaluate and publish without retraining.

Storage holds one latest checkpoint per seed and one pinned pretrained weight;
no accumulating epoch-weight files. Refuse to load another run with less than
2 GiB free. Dataset files are read from Kaggle input, with zero loader workers.

## What NB29 does and does not establish

Checks all three epoch-60 statuses, exact held-out image/point coverage and
recalculates normalised error. Provides per-tyre and per-seed results plus a
constant coordinate baseline computed only from training labels. Normalisation
uses native width minus one, consistent with coordinate targets.

**This is not yet a matched SegFormer comparison.** Old S5 training may overlap
these new test tyres. A same-split segmentation baseline needs its own overlap
audit/retraining before superiority can be claimed. No full-S9 completion claim:
PatchCore reference validation, physical-angle validation and final component
ablation/integration remain separate. A negative HRNet result remains reportable.

## Tests actually performed

Passed: frozen source/label contract, 120 image hashes, disjoint group lists,
all four notebook syntax checks, actual HRNet-W18 CPU forward at 512×384, exact
parameter/channel identity, frozen BatchNorm, coordinate-loss test, atomic
checkpoint and CPU AdamW/RNG continuation equivalence on a small test model.

Not tested locally: pretrained tensor load, real HRNet GPU/AMP optimizer continuation,
HF upload, long training or final results. NB27 is the required runtime test, not
a claimed completed experiment. No assistant HF writes or dataset/weight downloads.
