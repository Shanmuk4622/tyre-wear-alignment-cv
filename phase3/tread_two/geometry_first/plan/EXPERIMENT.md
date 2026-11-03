# Data, experiment, selection and evaluation protocol

**Chosen design values; executable freeze pending.** One candidate, seed 1,60 epochs,49 updates/epoch, batch 8:2,940 updates and 23,520 source draws. Two texture views per eligible draw; at most 47,040 training views. No automatic seed 2/ablation/sweep. Disposable engineering seam updates are a separate namespace, never reported as measured fit.

## Immutable inputs and identity

| Item | Contract |
|---|---|
| Kaggle source |`shanmuk4622/tire-dataset-prepared-phase2`, expanded read-only input, no duplicate output copy |
| SHA256 SUMS identity |`06c6c9d003aff808babf7ebb5786f441e95064eb000f5e5ed5b155d0ee9e7c6a` |
| Historical overlay |`f81a80d20b60ba505d94e31c26615e7e7171a91a8506948be6c392b68107a039` |
| Historical roles |386 train /81 validation /103 already-inspected test; no reassignment |
| Conservative train groups |Low/mid/high 1/1/4; confirmed original tyres 1/3/4 |
| Point target support |Training 70 human /152 weak /164 none; validation/test 25 human images each |
| Label meaning |Mileage proxy retained as registry context; not a GEO optimization target |
| Base checkpoint |Pinned TREAD-ONE seed 3/epoch 20 SHA and URL in PLAN.json; no test substitution |

Sources and point counts follow the existing independently audited registries. The inherited anchor was selected historically using validation, and old held-out outputs have informed development. Therefore all new validation/test results remain development evidence. A two-tyre low/high endpoint cannot confirm full three-class unseen-tyre transfer.

## Deterministic full-coverage epoch plan

The new objective does not need class-balanced replacement draws. Cover every original training image exactly once per epoch, with six explicitly padded draws to reach 392 samples. This yields 23,160 primary draws and 360 padding draws over 60 epochs. Save the complete plan before each epoch; never invent an independent draw probability for this structured schedule.

1. Build queues from the six conservative components; sort IDs, then seed-shuffle by `(seed,epoch,component)`.
2. Identify the largest component (249 images) and its complement (137 images). Reserve 49 distinct images from each, supplying one from each side to each of 49 batches.
3. Seed-shuffle the remaining 288 images, append six padding IDs from the epoch permutation, and fill six remaining slots/batch.
4. Deterministically repair any within-batch duplicate by swapping available remaining slots while preserving the epoch multiset. Fail preflight if the required invariants cannot be met; never silently drop an image.
5. Record all draws, padding flags, component counts, augmentation/mask seeds and plan SHA. Validate 386 unique primary IDs,392 draws,8 distinct source IDs/batch and≥2 components/batch.

The support constraint makes this packing feasible; actual implementation must test many seeded epochs. It does not produce eight independent tyres per batch. SSL-valid support can shrink after predicted masks; separate eligibility checks still apply.

## Views and warm start

Geometry receives one shared source photometric perturbation: brightness, contrast and gamma independently uniform[.9,1.1], applied in a frozen documented order to RGB[0,1]. Geometry targets/guide rows do not move. Its detached atlas then supplies two independent texture photometric perturbations with factors[.8,1.2]. Local contrast is computed after those factors, inside each tile. No elastic transforms, arbitrary geometric crops or label-changing flips in pilot 1.

Training masks/points supervise geometry and provide error targets only; predicted masks generate the normal atlas. Masked tile choice uses predicted valid coverage≥.25. Keep at least one visible valid tile in every maskable band. Invalid/empty support yields logged texture skips, not fabricated texture embeddings.

Before the measured fit, run zero-update controls: exact anchor-versus-zero-adapter agreement, seeded random texture initialization, fixed training-only mean tile-target predictor, and train-only gradient-isolation checks. The mean predictor is computed once from a deterministic identity pass over 386 train images using the anchor atlas, stored by tile/channel/target pixel and valid counts. It supplies a loss baseline, not a learned semantic label or an additional optimized model.

## Optimizer and precision

One AdamW instance with disjoint parameter groups: geometry lr 1e-4, texture lr 1e-3, quality lr 3e-4; wd 1e-4, betas(.9,.999), eps 1e-8, foreach/fused disabled. Cosine LR by global completed update, no warmup; texture objective ramps over first 5 epochs. Separate norm 1.0 clipping for each group. A texture-skipped batch leaves all texture grads None so AdamW does not apply a weight-decay-only texture step. Save global and per-group counters/moments and the explicit scheduler convention.

All FP32. Frozen anchor eval/GroupNorm; no running BatchNorm, dropout, flash attention, SDPA or CUDA grid-sampling backward. Deterministic mode fails on unsupported operations. Local CPU/GTX checks and actual T4 checks are separate; [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/2.11/notes/randomness.html) does not promise CPU/GPU or cross-release equality.

## Validation and original checkpoint selection

Every epoch: all 81 validation images with fixed identity geometry inputs and the same seed-addressed pair of texture views/masks for comparability. Compute baseline anchor geometry on these identical images once before fitting and retain its exact scores. No test-forward checkpoint selection.

| Eligibility condition | Predeclared rule |
|---|---|
| Mean two-region Dice |≥anchor validation mean−.002 absolute |
| Human-point MAE |≤anchor validation MAE+.001 image widths, on identical available human targets |
| Invalid atlas fraction |≤anchor baseline+.01 absolute |
| Projector collapse check |Median per-dimension std≥.1 and covariance effective rank≥4 across validation images; finite values required |

These are engineering non-regression/collapse tolerances, not statistically proven population margins or semantic quality guarantees. Effective rank is `exp(entropy(normalized nonnegative covariance eigenvalues))`, with zero-total covariance reported as rank 0. Stats use one fixed view per image for rank; paired views are not independent samples. Empty/undefined denominators reject eligibility with a reason.

Among eligible epochs minimize `masked_Huber / frozen_training_mean_predictor_Huber +.1*paired_cosine_distance`, earliest tie. The mean predictor is evaluated against the identical validation targets/masks; denominators below 1e-6 make the ratio unavailable and the epoch ineligible. Quality error estimates and mileage labels do not enter selection. If no epoch qualifies, selected checkpoint is unavailable and the candidate fails; retain final/all checkpoints and evidence. Do not silently export the anchor or a test-winning epoch as the trained winner.

## Endpoint evaluation and success

Retain selected and final identity passes on train 386,validation 81,historical test 103, clearly separated. Initial anchor controls stay separately tagged. All human/weak/missing targets remain distinct. Full dense/geometry per-image metrics, boxes/missing boxes, point errors, atlas validity, paired consistency, masked-target loss against the mean predictor, representation rank, retrieval exclusions and error-estimation quality are saved.

Predeclared diagnostics on selected/final historical endpoint: brightness.8/1.2, gamma.8/1.2, grayscale, known synthetic blur, background-neutralized source, reverse bands/columns and top predicted-residual tile masking. Compare geometry changes and texture features under each actual intervention; keep source-patch mapping. Blur/appearance perturbations are synthetic robustness probes, not newly annotated capture defects or independent photos. Background intervention uses masks only in the explicitly marked diagnostic branch, never normal inference.

| Question | Result needed for a useful research candidate |
|---|---|
| Dense preservation |Selected validation floors pass; historical endpoint dense changes are reported, not used to reselect |
| Nontrivial texture prediction |≥5%lower validation masked loss than the frozen mean predictor under identical mask weights |
| Paired consistency |Mean cosine distance≤.05; report variance/rank and nuisance sensitivity together |
| Training support |≥90%training draws receive valid SSL loss, every training source has recorded coverage; otherwise failure/coverage limit recorded |
| Error head |Compare with constant training median; unvalidated scores stay explicitly experimental |
| Classification |No improvement claim from this pilot; no new class-head fitting |

The practical success claim is preserved/possibly refined visible geometry plus a noncollapsed, useful local texture representation on this development cohort. A classifier, depth estimator, defect detector, safety assessor and production application are separate unestablished claims.
