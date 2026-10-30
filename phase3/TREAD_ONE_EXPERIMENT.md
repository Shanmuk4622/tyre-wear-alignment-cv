# TREAD-ONE — frozen experiment and artifact contract

Version r1, 27 October 2026. No measured 60-epoch training results yet.

## Study size and input

**One architecture × three seeds (1,2,3) ×60 epochs = three measured runs.** The network is trained from scratch. Each epoch has 49 balanced batches of eight sampled with replacement, or 392 sample exposures, not a promise to visit every training image exactly once. Each completed seed has **2,940 optimizer updates**, 60 validation records and both selected/final test endpoints.

Input: existing Kaggle `shanmuk4622/tire-dataset-prepared-phase2`. The loader discovers `VERSION.json`, checks the frozen `SHA256SUMS.txt` hash and verifies every listed source file and all 570 image/mask sets before training. The release stays read-only. Old augmented derivatives and raw video files are not duplicated.

The exact Phase 2 source verifier and overlay implementation are copied under `runtime/` and checked byte-for-byte against their protected originals at notebook build time. The new adapter adds each image's guide fractions and the new multi-task batches without changing source annotations.

| Partition | Images | Tyres | Interpretation |
|---|---:|---:|---|
| Train | 386 | 8 | All original mid tyres and all 152 video frames included |
| Validation | 81 | 2 | Old low/high tyres only |
| Test | 103 | 2 | Old low/high tyres only; 25 images have human geometry |

All possible unknown mappings of the three mid-video tyres remain within training. A source-v1 `UNASSIGNED` label is deliberately preserved; this experiment uses a separately hashed overlay. Image hashes, physical groups, weak/human geometry identity and ignore masks are recorded.

## Training recipe

- Direct PIL bilinear RGB resize: height 512, width 384; ImageNet normalization.
- Full image for all tasks; no inference oracle crop or test-time model selection.
- AdamW, learning rate 1e-4, weight decay 0.01, gradient norm clip 1.
- Cosine schedule over 60 completed epochs to 1e-6; no early stopping.
- Batch eight, no gradient accumulation; dual T4 means one independent seed process per device, not DataParallel. Seed 3 queues as a slot frees.
- GroupNorm, no stochastic dropout or running batch statistics; explicit deterministic kernels.
- FP16 AMP, initial scale 512. Numerical overflow retries the same batch/RNG with bounded scale reduction; final retry uses FP32 computation. Persistent failure pauses with the preceding durable update, without skipping samples.
- Stateless class→domain→tyre/clip balanced sampling; frozen Phase 2 flip/point-swap, brightness/contrast and mild blur transforms.
- Exact loss and background-intervention weights are in [the design](TREAD_ONE_DESIGN.md).

The local batch-eight seam passed on a GTX1650. Its small timing sample is not a T4 estimate or an end-to-end training promise. The actual notebook records its GPU smoke timing and resource evidence before training each seed.

## Checkpoint and seed selection

The one exported multi-task state is selected using this predeclared engineering utility:

```text
0.50 * balanced_accuracy_over_observed_validation_classes
+ 0.35 * mean_two_region_Dice
+ 0.15 * max(0, 1 - human_point_MAE_width_fraction / 0.05)
```

Higher is better; earliest epoch wins equal values. Select the export seed by the highest validation utility, with first-seed ties. Keep every seed's validation-selected endpoint and fixed epoch-60 endpoint separately. Do not substitute the best test seed. The weights and 5%-width geometry scale are design choices, not a calibrated common accuracy.

The two classifiers' known 50% test balanced accuracy from Phase 2 motivates investigation but is not a result for this new model. A smaller model can perform worse. No superiority is pre-claimed.

## Evaluation

| Output | Metric / evidence |
|---|---|
| Mileage proxy | Observed-class balanced accuracy, per-class recall, three-class confusion matrix, probabilities/logits; explicitly labelled three-class macro-F1 with missing classes assigned zero |
| Tyre/tread masks | Mean two-region Dice/IoU at input resolution, excluding contradiction pixels; per-image and per-tyre records |
| Localisation | Mask-derived enclosing-box IoU, including zero for missing boxes; no fictitious detector mAP |
| Geometry | Mean absolute horizontal coordinate error / native image width on human-labelled held-out images; raw predictions/targets and heuristic review flags |
| Explanations | Exact logit reconstruction residual; signed evidence and bias; feature-evidence removal, input occlusion and neutral-background diagnostics |
| Runtime | Actual model identity/parameter count, real-model seam, step timings, GPU allocation and live container memory |

Each test endpoint stores raw quarter-resolution mask probabilities, six points, class logits, bias, latent maps, spatial gates and signed evidence in a compressed NPZ, ordered by saved image IDs. The NPZ representation retains FP32 values. Normal inference uses the image and guide fractions only. Human masks define labels/diagnostics, not an oracle inference gate.

Report each seed and descriptive mean/population SD. Images/points/seeds are dependent observations from a tiny number of tyres, not hundreds of independent subjects. The fixed gallery uses the first deterministic example per class and labels its cohort; mid examples are training-cohort illustrations.

## What a completed run requires

The coordinator checks epoch 60/cursor zero, 2,940 unique sequential update records, validation epochs 1–60, consistency of the selected best epoch/utility, 103 unique test image records, observed classes [0,2], and 25 human point images at both endpoints before accepting a completed skip/export. Missing or incompatible state fails visibly rather than restarting from zero.

The final report contains three completed seed rows and separate selected/final summaries. It exports one validation-selected compact state dictionary with SHA-256 and the full preprocessing/protocol/source contract. No app installation or model promotion occurs automatically.

## Persistence

Local state is atomically replaced after every completed update, including model/optimizer/scheduler/scaler, CPU/CUDA/Python/NumPy RNG, next epoch/cursor, partial detailed history, exposures, selection state/best weights and environment. Validation interruption may require repeating validation; it does not repeat completed training updates.

The coordinator owns all three seed jobs, snapshots them under a per-checkpoint lock, and creates immutable ZIPs with member hashes. Uploads read these snapshots, never live rewritten state. Stable ZIP metadata enables unchanged-file deduplication. Remote index and binary hashes are verified at immutable commits. Restore validates member paths, sizes/hashes and protocol before replacing local progress.

Remote namespace:

```text
Shanmuk4622/tyre-wear-study  [dataset repo]
phase3/tread-one-r1/
  anchors/<RUN_TAG>.json                  # binds training source/environment/protocol
  <protocol_sha256>/
    LEASE.json                           # CAS ownership + fencing
    CONTRACT.json / OVERLAY.json / ENVIRONMENT.json
    DATA_PREFLIGHT.json / sources/*.py
    INDEX.json                           # revision-pinned bundle identities
    runs/seed1/latest.zip                 # state, traces, smoke, raw test NPZ, errors/logs
    runs/seed2/latest.zip
    runs/seed3/latest.zip
    report/REPORT.json / EXPORT.json / weights.pt / gallery/*
```

CPU preflight uses a separate `<RUN_TAG>-preflight` anchor because CPU/GPU images can have different Torch builds. Training's anchor prevents an unchanged RUN_TAG from silently creating a new study after source/environment drift. Explicit compatibility review is required for repairs.

Normal publication: 30-minute cycles, seed completion, final reporting and catchable stop/error. Local signals reach children even during publisher backoff. Save local progress first. Final publication waits for child checkpoint writers before releasing the lease. An 8.25-hour coordinator watchdog requests a pause before the expected session limit; actual Kaggle policy still governs.

The publisher budgets 48 application calls/hour within this single coordinator, caches authenticated reads, batches payloads and honours Retry-After/RateLimit delays. SDK calls can contain multiple HTTP requests. Other notebooks using the same HF account do not share this local counter; run only one TREAD-ONE coordinator and avoid overlapping publication-heavy studies. Server quotas remain authoritative.

Forced kill, network outage, driver failure or ephemeral-storage deletion can lose unpublished progress. Report `LAST_REMOTE.json`, pending local paths and the last `HF snapshot verified` revision. “Immediate stop push” means an immediate catchable flush attempt, not bypassing server delays or executing a handler after SIGKILL.

## Storage

Measure capacity instead of assuming a 1 TB temporary disk. Working output contains the small runtime/reports/cache/upload staging. Live checkpoints use `/tmp/treadone_<protocol>/runs`, not a copied dataset. Initial guards require >3 GiB output free and >5 GiB scratch free; workers stop before scratch falls below 2 GiB. Staging has a 4 GiB cap and retains 1 GiB free. Three small state dictionaries/optimizers plus test arrays should be well below those guards, but the executing environment supplies the actual measurement.

Only rolling state and validation-best weights are retained locally. HF Git/LFS revisions preserve published history; this is not a claim that every local per-step snapshot is separately uploaded. Full logs/state are in each latest consistent bundle. Clean up only verified staging files owned by this task; never delete source inputs or unverified live progress.

## Validity and remaining work

No healthy/anomaly, physical wear/depth, visibility classifier, unseen-mid/video or calibrated-alignment capability is added by this architecture. Extra field data, matched ablations, broader group splits and independent annotation studies remain separate tasks. Long training and live HF smoke are executed by the Kaggle notebooks, not inferred from local code tests.
