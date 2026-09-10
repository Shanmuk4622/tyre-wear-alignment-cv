# S4b — two additional architectures, 2026-09-09

**Status 2026-09-10: 0 finished, 4 resumable, 14 not started; runtime repair ready.**
HF revision `2def8d1b1f00f7c08fd029e8f339c5ab1004b6c8` has classweighted
ConvNeXt seeds 1/2/3 at epochs 28/26/25 and uniform seed 1 at epoch 26.
All four checkpoints remain reusable; no completion claimed.

### Runtime repair — use the updated NB12

**r2 evidence:** the next saved Kaggle output passed ConvNeXt at 0.8216 s/step,
but stopped on MobileNet dual-GPU at 12.0436 s/step before any run claims.
HF `e3516a22a75d4349b1e9fd5fba9d276ce159ba53` still shows the same four paused
runs. MobileNet Stage-A telemetry shows one active GPU, ~48 s/epoch and ~6.46 GB
peak memory. r2 extends single-GPU execution to MobileNet without changing batch
64. Preflight uses the same runtime flag as training, not its own architecture
list. ConvNeXt speed is now measured; repaired MobileNet speed remains unverified.

The first GPU smoke succeeded numerically but did not measure throughput. Full
training then took ~18–20 minutes/epoch with host RAM growth, versus ~90 seconds
for saved single-GPU Stage-A ConvNeXt. Forward passes dominate; input loading
is only ~5%. Three jobs paused for host RAM, one for session time. These are
not lost jobs or evidence of a Hugging Face commit-budget stall. The exact
underlying DataParallel/CPU/CUDA interaction remains unproven.

**Repair r2:** both models now use GPU 0 only: ConvNeXt batch 32, MobileNet
batch 64 unchanged. `_single_gpu` is a hash-excluded runtime flag: original run IDs,
scientific configs and checkpoint hashes remain compatible. A warmed eight-step
preflight rejects median training-step time >=4 seconds before expensive jobs.
This is an operational speed guard, not an experimental-score threshold.
An actual epoch exceeding 600 seconds also checkpoints and pauses with
`runtime_throughput_guard`, preventing repeated 18-minute epochs after preflight.
This never shortens the required 60-epoch budget or marks an unfinished run complete.
Training logs record `runtime_training_gpu_count` for the changed execution path.
The remaining GPU is intentionally idle for both models, matching the faster
historical path; no smaller model or reduced epoch budget is substituted.

Stop old copies, replace NB12, preserve your four account labels and active-account
list, then Run All in fresh Kaggle sessions. Expect resume from 28/26/25/26 (or
newer HF checkpoints), not epoch zero. Repaired GPU speed has not yet been measured
on Kaggle; if the timed guard fails, send its log rather than bypassing it.

## Exact scope

NB06 already ran all 12 implemented factors on three architectures. Repeating
that work would not add confirmation. This declared S4b extension uses the
**other two eligible three-seed-confirmed architectures** in the locked NB07
selection: ConvNeXt-V2 Tiny and MobileNetV4. Neither is a replacement for the
quarantined Small model. Both retain their original Stage-A model/recipe.

Discovery source is pinned to public HF revision
`bf62f9e9cbedacc580aa42542da14a068b8f9215`:
`tables/stage_b_selection.csv` and `tables/stage_b_effects.csv`.
Rank the 12 factors by descending mean **signed paired selected-epoch F1 delta**
across all nine NB06 architecture/seed observations; break ties alphabetically.
This preserves NB06's original reporting endpoint rather than retrospectively
switching endpoints to pick different winners.

| Factor | Discovery mean delta | Exact change |
|---|---:|---|
| sampler_classweighted | +0.052669 | session-balanced → class-weighted sampling |
| transfer_random | -0.020625 | pretrained → random initialisation |
| sampler_uniform | -0.033445 | session-balanced → uniform sampling |

These are the **three highest-ranked factors**, not three demonstrated gains:
only class-weighted sampling had a positive discovery mean. The report may
find the negative effects also carry over, or that any direction fails to repeat.
This explicit operational selection rule is frozen before S4b training; it is
not represented as an original pre-discovery preregistration.

2 architectures × 3 factors × 3 seeds × fold 1 = **18 new runs**, 60 epochs each
(1,080 total new training epochs). Six completed Stage-A fold-1 baselines are
reused, not retrained. ConvNeXt-V2 Tiny stays at 384px/batch 32; MobileNetV4
stays at 384px/batch 64. CORAL, optimiser and all other scientific recipe
settings stay unchanged. No ROI masks, NB11, SAM2 or human annotation needed.

## Run order

1. Upload **NB12_S4B_Confirmation.ipynb**. Internet ON, HF_TOKEN secret,
   Tire Dataset Prepared attached, **GPU T4 x2**. Run All.
2. Default is one worker. For four accounts, use the same
   `ACTIVE_KAGGLE_ACCOUNTS=('acct1','acct2','acct3','acct4')` on all copies;
   set ACCOUNT to that copy's label. Do not change protocol/config values.
   The same full run list is used on every copy so ownership remains stable.
3. A disposable timed forward/backward/optimizer preflight checks both actual
   model parameter counts against published baselines before claiming training;
   both models use GPU 0, using the runtime flag from their actual training configs.
   If it fails, keep the log; do not silently reduce model/batch or change recipe.
4. Let the workers finish or rerun a fresh Kaggle session with the same notebook
   to resume. Do not run overlapping copies with the same ACCOUNT. Change the
   worker count only after stopping the previous set of copies.
5. Once **all 18** jobs show FINISHED, run **NB12R_S4B_Report.ipynb once**.
   CPU sufficient; Internet and HF_TOKEN, no dataset attachment needed.

New IDs: `c-{arch}-s4b_r1_{factor}-f1-s{seed}`. No existing run ID is overwritten.
The output namespace is
`confirmations/s4b-2026-09-09-r1/9f2e32c8c372c1c985d7827a893c6be0bd7c6801019cd4b5456f3f2913bd0e72/`.
Each worker publishes a copy of the same frozen protocol; NB12R is the single
aggregate writer. A worker finishing its shard is not whole-stage completion.

## Resume, storage and HF

Use the proven isolated-process training scheduler; it releases native/GPU
allocations between models. Scratch/checkpoints use `/kaggle/temp`, not the
20-GB output directory. Actual scratch capacity must still be respected.
Normal artifact pushes batch every 30 minutes; major cells and catchable Stop
flush. Idle-worker takeover can require separate coordination commits under
the shared rate budget. Training restores optimiser/scheduler/scaler/RNG state
from the last published complete epoch; an interrupted partial epoch is replayed.
A forced kernel/OS kill cannot upload unpublished work.

S4b enables `_strict_resume`: missing checkpoints with published epoch progress,
unreadable checkpoints or config mismatches fail instead of silently restarting
training. This is opt-in; old notebook bytes and default behaviour are unchanged.
The flag is non-scientific and excluded from config hashing. Existing recipe
fields are checked against all six pinned baseline configurations before training.

## Reporting and completion

NB12R checks all 18 live STATUS/config files, exact histories 1–60, final metrics,
F1 at the recorded best-QWK epoch and best/last checkpoint paths. Presence checks
do not imply tensor-level checkpoint validation. Missing/inconsistent jobs produce
an incomplete coverage report, never partial confirmation averages.

Primary: same-seed selected-epoch F1 delta matching NB06. Secondary: final-epoch
F1 delta at epoch 60, kept separate. Reports include all 18 paired rows, six
architecture/factor summaries, per-factor directional replication and STATUS.
Same-direction means are **descriptive**, not significance tests. With three
seeds and the same fold/data, this is not independent new-tyre confirmation.
S4b execution can be complete even when discovery effects do not replicate.

Known folds 0/2 are not reused to inflate confirmation. Fold-integrity limitations
still constrain the study. This does not complete S5/S9 or wider Tier-5/6 work.

## Validation

`scripts/verify_s4b_notebooks.py` checks the actual frozen HF ranking, 18 unique
one-factor configurations, original batch sizes, paired endpoint arithmetic,
negative-direction interpretation, incomplete-result rejection, strict resume
failure paths and generated notebook syntax. No training was launched locally.
Full GPU/runtime verification occurs in the notebook preflight and subsequent
Kaggle execution; do not claim it has already passed.
