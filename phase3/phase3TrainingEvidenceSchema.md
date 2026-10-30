# TREAD-ONE NB04 — exact CSV data dictionary

**28 October 2026.** Generated from the accepted source schema. This defines engineering-fixture records, not reconstructed r1 history or a completed measured revision. [Guide, formulas, arrays and run instructions](phase3TrainingEvidenceGate.md).

Blank means unavailable/not applicable; observed zero remains zero. Each snapshot has its own SCHEMA/COVERAGE/CURSOR/MANIFEST. Identity columns repeat across the image-level tables. Do not count those repetitions as additional metrics or subjects.

## Common image identity columns

| Column | Units / type | Meaning |
|---|---|---|
| `update` | count | Completed optimizer-update identity; starts at 1 for observed batches. |
| `epoch` | count | One-based surrogate engineering epoch; not an r1 scientific epoch. |
| `cursor` | count | Zero-based batch position before this update; blank for validation fixture rows. |
| `slot` | count | Zero-based example position in the actual batch or fixed validation fixture. |
| `image_id` | identifier | Prepared-source image identity; repeated sampling is retained as repeated occurrences. |
| `image_sha256` | SHA-256 | Original source-image file identity; native augmented arrays have separate manifest hashes. |
| `tyre_id` | identifier | Source physical tyre identifier, where supplied; video-parent uncertainty is not resolved by this field. |
| `source_video` | identifier | Original video source when applicable; blank for photographs. |
| `domain` | category | Original photograph or added video-frame domain from the prepared manifest. |
| `label` | class index | Low=0, mid=1, high=2 mileage proxy; not a measured wear/depth/safety class. |

## `train_steps.csv`

33 columns.

| Column | Units / type | Meaning |
|---|---|---|
| `update` | count | Completed optimizer-update identity; starts at 1 for observed batches. |
| `epoch` | count | One-based surrogate engineering epoch; not an r1 scientific epoch. |
| `cursor` | count | Zero-based batch position before this update; blank for validation fixture rows. |
| `batch_images` | sample occurrences | Actual images in this completed optimizer batch; eight in this gate. |
| `loss` | loss | Actual optimizer objective before this update; includes background consistency when scheduled. |
| `classification` | loss | Batch mean label-smoothed classification CE; coefficient 1. |
| `ordinal` | loss | Batch ordinal-CDF loss before coefficient 0.2. |
| `segmentation` | loss | Valid-pixel BCE plus 1 minus smoothed soft Dice; coefficient 1. |
| `geometry` | loss | Weighted Smooth L1 point loss, beta 0.02, before coefficient 2. |
| `containment` | loss | Valid-pixel positive tread-minus-tyre probability, before coefficient 0.1. |
| `alignment` | loss | Usable guide-row interval/mask consistency loss before coefficient 0.1. |
| `background_kl` | loss | KL(original detached probabilities || background-swap probabilities); coefficient 0.2, zero when not scheduled. |
| `sample_contribution_sum` | loss | Sum of the eight stored objective_contribution values. |
| `loss_reconstruction_error` | absolute loss | Absolute difference between that contribution sum and the actual optimizer loss. |
| `grad_norm_before_clip` | L2 norm | Combined finite unscaled parameter gradient norm before clip 1. |
| `grad_norm_after_clip` | L2 norm | Combined parameter gradient norm measured after clipping. |
| `clip_coefficient` | factor | min(1, 1/(preclip norm+1e-6)); observational description of the clipping step. |
| `amp_scale_before` | scale factor | Actual gradient-scaler scale before/after the attempt or update. |
| `amp_scale_after` | scale factor | Actual gradient-scaler scale before/after the attempt or update. |
| `retries` | count | Failed numeric attempts replayed for this completed update. |
| `lr_used` | learning rate | Observed sole AdamW group learning rate at the indicated boundary. |
| `lr_after_scheduler` | learning rate | Observed sole AdamW group learning rate at the indicated boundary. |
| `next_epoch` | count | One-based epoch to resume after this completed update. |
| `next_cursor` | count | Zero-based next batch cursor; zero after the surrogate boundary. |
| `checkpoint_before_sha256` | SHA-256 | Manifest identity of full native recovery state immediately before/after the completed update. |
| `checkpoint_after_sha256` | SHA-256 | Manifest identity of full native recovery state immediately before/after the completed update. |
| `data_seconds` | seconds | Measured data stage; blank when not executed. Compute timing excludes data, validation, checkpoint and HF upload unless separately named. |
| `forward_seconds` | seconds | Measured forward stage; blank when not executed. Compute timing excludes data, validation, checkpoint and HF upload unless separately named. |
| `backward_seconds` | seconds | Measured backward stage; blank when not executed. Compute timing excludes data, validation, checkpoint and HF upload unless separately named. |
| `optimizer_seconds` | seconds | Measured optimizer stage; blank when not executed. Compute timing excludes data, validation, checkpoint and HF upload unless separately named. |
| `telemetry_seconds` | seconds | Measured telemetry stage; blank when not executed. Compute timing excludes data, validation, checkpoint and HF upload unless separately named. |
| `update_compute_seconds` | seconds | Measured update compute stage; blank when not executed. Compute timing excludes data, validation, checkpoint and HF upload unless separately named. |
| `validation_seconds` | seconds | Measured validation stage; blank when not executed. Compute timing excludes data, validation, checkpoint and HF upload unless separately named. |

## `train_samples.csv`

56 columns; begins with the 10 common identity columns above.

| Column | Units / type | Meaning |
|---|---|---|
| `prediction` | context dependent | Class argmax in image/class tables; normalized horizontal coordinate in geometry_points. |
| `correct` | 0/1 | Observed predicted-class equality to the mileage-proxy label. |
| `confidence` | fraction | Maximum predicted class probability. |
| `probability_0` | fraction | Observed class probability before this update or at the fixed validation fixture. |
| `probability_1` | fraction | Observed class probability before this update or at the fixed validation fixture. |
| `probability_2` | fraction | Observed class probability before this update or at the fixed validation fixture. |
| `logit_0` | logit | Observed class logit before this update. |
| `logit_1` | logit | Observed class logit before this update. |
| `logit_2` | logit | Observed class logit before this update. |
| `nll` | nats | Negative log true-class probability, clamped at 1e-12 for this diagnostic statistic. |
| `brier` | squared probability sum | Sum over the three classes of squared probability minus one-hot truth. |
| `entropy_nats` | nats | Class-probability entropy; natural logarithms. |
| `label_smoothed_ce` | loss | Per-image CE with label smoothing 0.05, before batch averaging. |
| `ordinal_loss` | loss | Per-image mean squared error of the first two class-CDF entries. |
| `seg_bce_numerator` | loss sum | Per-image sum of valid, downsampled BCE terms over both regions. |
| `seg_bce_denominator` | support | Twice the valid low-resolution pixel count for this image; batch sum is the BCE denominator. |
| `seg_dice_loss` | loss | Per-image 1 minus mean two-region soft Dice with +1 smoothing. |
| `geometry_numerator` | weighted loss | Mean six-point Smooth L1 multiplied by the image point weight. |
| `geometry_weight` | weight | Human=1, polygon-weak=0.25, unlabelled=0; batch sum is clamped below at 1. |
| `containment_numerator` | probability sum | Valid-pixel positive tread-minus-tyre sum for this image. |
| `containment_denominator` | support | Valid low-resolution pixels for this image; aggregated across batch. |
| `alignment_numerator` | absolute difference sum | Usable, valid guide-row interval/mask differences for this image. |
| `alignment_denominator` | support | Usable, valid guide-row positions for this image; aggregated across batch. |
| `background_kl` | loss | KL(original detached probabilities || background-swap probabilities); coefficient 0.2, zero when not scheduled. |
| `objective_contribution` | batch loss share | This image contribution using batch-wide component denominators and frozen task weights. |
| `mean_two_region_dice` | fraction | Mean smoothed binary Dice over tyre and tread with ignored pixels excluded. |
| `mean_box_iou` | fraction | Mean two-region box IoU; empty boxes score zero. |
| `point_kind` | category | human / polygon_weak / none; supervision provenance, not a learned visibility label. |
| `human_point_mae` | width fraction | Mean six-point absolute error only for human-labelled images; blank otherwise. |
| `annotation_point_mae` | width fraction | Mean six-point error against available human or weak targets; blank when unlabelled. |
| `point_output_dtype` | dtype | Actual model point-tensor dtype at the observed forward pass. |
| `augmentation_seed` | integer seed | Stateless seed combining engineering seed, epoch, cursor and slot. |
| `flip_horizontal` | 0/1 | Actual deterministic horizontal flip; left/right point indices are swapped. |
| `brightness` | factor | Sampled image brightness multiplier. |
| `contrast` | factor | Sampled image contrast multiplier. |
| `blur_applied` | 0/1 | Whether the deterministic Gaussian-blur branch was applied. |
| `blur_radius` | working-image pixels | Sampled Gaussian radius when applied; blank otherwise. |
| `background_donor_image_id` | identifier | Previous batch slot under roll(1); background pixels only, when scheduled. |
| `background_applied` | 0/1 | Whether this update computed the scheduled background-swap consistency pass. |
| `background_probability_0` | fraction | Background-swap class probability; blank when the branch is not scheduled. |
| `background_probability_1` | fraction | Background-swap class probability; blank when the branch is not scheduled. |
| `background_probability_2` | fraction | Background-swap class probability; blank when the branch is not scheduled. |
| `background_logit_0` | logit | Background-swap class logit; blank when not scheduled. |
| `background_logit_1` | logit | Background-swap class logit; blank when not scheduled. |
| `background_logit_2` | logit | Background-swap class logit; blank when not scheduled. |
| `checkpoint_before_sha256` | SHA-256 | Manifest identity of full native recovery state immediately before/after the completed update. |

## `region_stats.csv`

28 columns; begins with the 10 common identity columns above.

| Column | Units / type | Meaning |
|---|---|---|
| `region` | category | tyre / tread; masks overlap and are not mutually exclusive classes. |
| `tp` | working-pixel count | Binary region confusion count after threshold 0.5, nearest 4× upsampling and ignore exclusion. |
| `fp` | working-pixel count | Binary region confusion count after threshold 0.5, nearest 4× upsampling and ignore exclusion. |
| `fn` | working-pixel count | Binary region confusion count after threshold 0.5, nearest 4× upsampling and ignore exclusion. |
| `tn` | working-pixel count | Binary region confusion count after threshold 0.5, nearest 4× upsampling and ignore exclusion. |
| `valid_pixels` | working pixels | Number of non-ignored 512×384 pixels in the comparison. |
| `ignored_pixels` | working pixels | Number excluded by the transformed ignore mask. |
| `dice` | fraction | (2TP+1)/(2TP+FP+FN+1), using thresholded/upscaled mask. |
| `iou` | fraction | (TP+1)/(TP+FP+FN+1), using thresholded/upscaled mask. |
| `box_iou` | fraction | IoU between valid-pixel-derived half-open prediction/truth boxes; zero for missing box. |
| `pred_box_x0` | working pixels | Half-open box coordinate on 384-wide ×512-high working grid; blank for an empty valid region. |
| `pred_box_y0` | working pixels | Half-open box coordinate on 384-wide ×512-high working grid; blank for an empty valid region. |
| `pred_box_x1_exclusive` | working pixels | Half-open box coordinate on 384-wide ×512-high working grid; blank for an empty valid region. |
| `pred_box_y1_exclusive` | working pixels | Half-open box coordinate on 384-wide ×512-high working grid; blank for an empty valid region. |
| `target_box_x0` | working pixels | Half-open box coordinate on 384-wide ×512-high working grid; blank for an empty valid region. |
| `target_box_y0` | working pixels | Half-open box coordinate on 384-wide ×512-high working grid; blank for an empty valid region. |
| `target_box_x1_exclusive` | working pixels | Half-open box coordinate on 384-wide ×512-high working grid; blank for an empty valid region. |
| `target_box_y1_exclusive` | working pixels | Half-open box coordinate on 384-wide ×512-high working grid; blank for an empty valid region. |

## `geometry_points.csv`

27 columns; begins with the 10 common identity columns above.

| Column | Units / type | Meaning |
|---|---|---|
| `point_index` | 0..5 | Left/right at each of three source guide rows, in model order. |
| `guide_fraction` | height fraction | Source guide y/(native_height-1); unchanged by horizontal flip. |
| `side` | category | left / right in the transformed example. |
| `prediction` | width fraction | Predicted horizontal point coordinate, copied from the actual pre-update forward pass. |
| `optimization_target` | width fraction | Exact transformed point tensor supplied to the objective, including zero-weight placeholders. |
| `target` | width fraction | Actual transformed annotated target; blank for unlabelled images. |
| `source_target` | width fraction | Original unflipped annotated coordinate at source_target_index; blank if unavailable. |
| `source_target_index` | 0..5 | Original index corresponding to this transformed point; swaps each left/right pair on flip. |
| `point_kind` | category | human / polygon_weak / none; supervision provenance, not a learned visibility label. |
| `point_weight` | weight | Supervision weight human=1, polygon-weak=0.25, none=0. |
| `human_eligible` | 0/1 | One only for independently supplied human geometry targets; weak targets excluded. |
| `signed_error_width_fraction` | width fraction | Predicted coordinate minus transformed annotated target, computed with float64 Python scalars. |
| `absolute_error_width_fraction` | width fraction | Absolute signed error; blank without an annotation. |
| `squared_error_width_fraction` | squared width fraction | Square of signed error; blank without an annotation. |
| `absolute_error_native_pixels` | native pixels | Absolute fractional error times (native_width-1). |
| `native_width` | native pixels | Original source-image width. |
| `native_height` | native pixels | Original source-image height. |

## `latent_metrics.csv`

16 columns; begins with the 10 common identity columns above.

| Column | Units / type | Meaning |
|---|---|---|
| `latent` | 0..11 | Positive latent channel index; no named physical concept is inferred. |
| `minimum` | tensor value | Minimum latent/activation value over the declared tensor. |
| `maximum` | tensor value | Maximum latent/activation value over the declared tensor. |
| `mean` | tensor value | Arithmetic mean over the declared tensor. |
| `std` | tensor value | Population standard deviation over spatial latent map or full hooked activation tensor. |
| `gated_mass` | activation | Spatial sum of latent activation times normalized predicted gate. |

## `evidence_metrics.csv`

20 columns; begins with the 10 common identity columns above.

| Column | Units / type | Meaning |
|---|---|---|
| `class_index` | 0..2 | Class whose signed contribution/evidence is summarized. |
| `logit` | logit | Observed class logit before optimizer update. |
| `bias` | logit | Exact class-head bias copied before the optimizer changes parameters. |
| `sum_contribution` | logit | Spatial sum of signed class-evidence map; sum plus bias reconstructs logit up to numeric rounding. |
| `positive_sum` | logit | Sum of positive signed evidence pixels. |
| `negative_sum` | logit | Sum of negative signed evidence pixels; values are nonpositive. |
| `inside_true_tread_sum` | logit | Signed evidence weighted by valid true-tread 4×4 area fractions. |
| `outside_true_tread_sum` | logit | Signed evidence weighted by one minus that area fraction; complement includes ignored area. |
| `evidence_max` | logit per grid pixel | Maximum/minimum signed contribution-map value for this class. |
| `evidence_min` | logit per grid pixel | Maximum/minimum signed contribution-map value for this class. |

## `class_confusion.csv`

5 columns.

| Column | Units / type | Meaning |
|---|---|---|
| `update` | count | Completed optimizer-update identity; starts at 1 for observed batches. |
| `epoch` | count | One-based surrogate engineering epoch; not an r1 scientific epoch. |
| `label` | class index | Low=0, mid=1, high=2 mileage proxy; not a measured wear/depth/safety class. |
| `prediction` | context dependent | Class argmax in image/class tables; normalized horizontal coordinate in geometry_points. |
| `count` | sample occurrences | Class-pair count in the sampled batch; observed zero is retained. |

## `gradient_modules.csv`

7 columns.

| Column | Units / type | Meaning |
|---|---|---|
| `update` | count | Completed optimizer-update identity; starts at 1 for observed batches. |
| `epoch` | count | One-based surrogate engineering epoch; not an r1 scientific epoch. |
| `module` | identifier | Named model module/group; not a whole-model histogram. |
| `parameters_with_gradient` | tensor count | Number of parameter tensors with a gradient in the module group. |
| `gradient_norm` | L2 norm | Combined unscaled module gradient norm before clipping. |
| `maximum_absolute_gradient` | gradient magnitude | Largest absolute unscaled gradient in this module group. |
| `finite` | 0/1 | Whether all inspected values are finite. |

## `activation_modules.csv`

11 columns.

| Column | Units / type | Meaning |
|---|---|---|
| `update` | count | Completed optimizer-update identity; starts at 1 for observed batches. |
| `epoch` | count | One-based surrogate engineering epoch; not an r1 scientific epoch. |
| `phase` | category | normal / background forward; absent background pass is not fabricated. |
| `module` | identifier | Named model module/group; not a whole-model histogram. |
| `elements` | count | Elements in the observed activation tensor, including batch and spatial axes. |
| `mean` | tensor value | Arithmetic mean over the declared tensor. |
| `std` | tensor value | Population standard deviation over spatial latent map or full hooked activation tensor. |
| `minimum` | tensor value | Minimum latent/activation value over the declared tensor. |
| `maximum` | tensor value | Maximum latent/activation value over the declared tensor. |
| `l2_norm` | L2 norm | Square root of the summed squared hooked activation values. |
| `finite` | 0/1 | Whether all inspected values are finite. |

## `validation_predictions.csv`

19 columns; begins with the 10 common identity columns above.

| Column | Units / type | Meaning |
|---|---|---|
| `prediction` | context dependent | Class argmax in image/class tables; normalized horizontal coordinate in geometry_points. |
| `correct` | 0/1 | Observed predicted-class equality to the mileage-proxy label. |
| `probability_0` | fraction | Observed class probability before this update or at the fixed validation fixture. |
| `probability_1` | fraction | Observed class probability before this update or at the fixed validation fixture. |
| `probability_2` | fraction | Observed class probability before this update or at the fixed validation fixture. |
| `nll` | nats | Negative log true-class probability, clamped at 1e-12 for this diagnostic statistic. |
| `brier` | squared probability sum | Sum over the three classes of squared probability minus one-hot truth. |
| `mean_two_region_dice` | fraction | Mean smoothed binary Dice over tyre and tread with ignored pixels excluded. |
| `human_point_mae` | width fraction | Mean six-point absolute error only for human-labelled images; blank otherwise. |

## `epoch_summaries.csv`

11 columns.

| Column | Units / type | Meaning |
|---|---|---|
| `update` | count | Completed optimizer-update identity; starts at 1 for observed batches. |
| `epoch` | count | One-based surrogate engineering epoch; not an r1 scientific epoch. |
| `batches` | count | Accumulated completed batches in the reported surrogate epoch. |
| `images` | sample occurrences | Accumulated sampled images, with repeats retained. |
| `objective_sum` | loss sum | Sum of completed batch objectives; not a pixel-normalized dataset loss. |
| `mean_batch_objective` | loss | objective_sum / completed batches. |
| `classification_correct` | sample occurrences | Number of correct pre-update classifications over that surrogate epoch. |
| `classification_images` | sample occurrences | Denominator for sampled augmented classification accuracy. |
| `classification_accuracy` | fraction | Correct / observed sampled occurrences; not nominal cohort accuracy. |
| `learning_rate_next_epoch` | learning rate | Observed sole AdamW group learning rate at the indicated boundary. |
| `engineering_only` | 0/1 | This is an engineering fixture rather than a scientific accuracy result. |

## `resources.csv`

19 columns.

| Column | Units / type | Meaning |
|---|---|---|
| `update` | count | Completed optimizer-update identity; starts at 1 for observed batches. |
| `epoch` | count | One-based surrogate engineering epoch; not an r1 scientific epoch. |
| `scope` | category | Resource measurement context, currently completed_update. |
| `process_rss_bytes` | bytes | Observed process rss; scope is process/device/container as named, not universal peak memory. |
| `container_used_bytes` | bytes | Observed container used; scope is process/device/container as named, not universal peak memory. |
| `container_limit_bytes` | bytes | Observed container limit; scope is process/device/container as named, not universal peak memory. |
| `container_source` | identifier | Actual cgroup/memory source or unavailable indicator from the frozen helper. |
| `cuda_allocated_bytes` | bytes | Observed cuda allocated; scope is process/device/container as named, not universal peak memory. |
| `cuda_peak_bytes` | bytes | Observed cuda peak; scope is process/device/container as named, not universal peak memory. |
| `working_free_bytes` | bytes | Observed working free; scope is process/device/container as named, not universal peak memory. |
| `gpu_uuid` | identifier | Reported physical GPU UUID, when available. |
| `gpu_name` | text | Reported GPU model name. |
| `gpu_utilization_percent` | percent | Instantaneous utilization sample; not an average for the entire update. |
| `gpu_memory_used_mib` | MiB | Instantaneous GPU memory sample; 1 MiB = 2^20 bytes. |
| `gpu_memory_total_mib` | MiB | Instantaneous GPU memory sample; 1 MiB = 2^20 bytes. |
| `gpu_power_watts` | watts | Instantaneous sampled board power, where supported; not energy. |
| `gpu_temperature_c` | degrees Celsius | Instantaneous temperature when supported. |
| `gpu_mapping` | category | selected_device only when UUID matches; otherwise other_or_unresolved_device. |
| `missing_reason` | text | Reason for unmeasured/scheduled/unsupported resource fields. |

## `events.csv`

9 columns.

| Column | Units / type | Meaning |
|---|---|---|
| `update` | count | Completed optimizer-update identity; starts at 1 for observed batches. |
| `epoch` | count | One-based surrogate engineering epoch; not an r1 scientific epoch. |
| `time_utc` | UTC timestamp | Event creation timestamp; not an exact hardware event clock. |
| `kind` | category | completed_optimizer_update or numerical_retry in current worker. |
| `attempt` | count | Numerical retry attempt when applicable; blank for completion-only events. |
| `amp_scale_before` | scale factor | Actual gradient-scaler scale before/after the attempt or update. |
| `amp_scale_after` | scale factor | Actual gradient-scaler scale before/after the attempt or update. |
| `seconds` | seconds | Event observation duration when applicable. |
| `message` | text | Optional event explanation; exceptions are separately retained in FAILURES.jsonl. |

## Companion records

`class_metrics.csv` is derived by surrogate epoch/class from actual sampled predictions: support, TP/FP/FN/TN, precision, recall and F1; absent denominators remain blank. `STATE_VERSIONS.csv` retains update/next cursor/native SHA-256 and whether that native file is still local. Earlier remote-base hashes do not imply that the corresponding native file was downloaded.

`COMMIT_TIMINGS.jsonl` records observed local serialization/CSV/manifest/pointer duration after the pointer commits; a kill in that gap can omit a timing row. `REMOTE_HISTORY.jsonl` records coordinator operations/status/duration/backoff and verified revisions, not all SDK HTTP requests. Console/error logs are direct companions. These are not hidden inside a model checkpoint.

Full gate inputs, targets and output maps are direct NPY files; full optimizer/RNG recovery is a direct PT file. Native binary files supplement these CSV records. [Retention policy](phase3DataRetentionPolicy.md) prohibits HF ZIP/NPZ publication.
