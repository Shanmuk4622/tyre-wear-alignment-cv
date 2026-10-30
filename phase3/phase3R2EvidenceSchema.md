# TREAD-ONE NB05 — exact evidence dictionary

**Observed audit note, 29 October 2026:** NB05 is complete. All 23 families and
1,561,734 combined core rows are verified; [completion](phase3NB05Completion.md).
The frozen NumPy-2 per-point scalar subtraction may round in FP32 (maximum
observed deviation 2.95×10⁻⁸ width from FP64 subtraction). Annotation-level
human MAE uses FP64 target-array arithmetic, while the declared selection uses
its separately retained frozen Torch field. These are documented precision
paths; stored metrics/checkpoints are preserved, not rewritten.

**28 October 2026.** Source-derived headers and definitions for the paired pilot. [Protocol/run guide](phase3R2PairedPilot.md). Blank means unavailable/not applicable; observed zero remains zero. Repeated identity columns are not independent metrics.

The source defines **23 core CSV families / 498 column positions**. The compatibility `validation_predictions` family is intentionally empty: full validation now lives in the richer `eval_predictions` and companion tables. Normal-only self-comparison fields do not imply an intervention experiment.

## Planned measured record coverage

| Family / scope | Per condition | Both conditions |
|---|---:|---:|
| Completed training updates |2,940|5,880|
| Actual augmented training occurrences |23,520|47,040|
| Validation image observations |4,860|9,720|
| Selected/final development-test image observations |206|412|
| Training point records |141,120|282,240|
| Training latent records |282,240|564,480|

These are planned observation counts for complete runs, not new subjects or already achieved results. A local engineering fixture is much smaller.

## `train_steps.csv`

36 ordered columns.

| Position | Column | Units / type | Meaning |
|---:|---|---|---|
|1|`condition`|category|baseline / shared_projection; stored in every core CSV row.|
|2|`update`|count|Completed optimizer-update identity; starts at 1 for observed batches.|
|3|`epoch`|one-based epoch|Training/validation epoch; selected/final inference rows name the actual checkpoint epoch.|
|4|`cursor`|count|Zero-based batch position before this update; blank for validation fixture rows.|
|5|`batch_images`|sample occurrences|Actual images in this completed optimizer batch; eight in this gate.|
|6|`loss`|loss|Actual optimizer objective before this update; includes background consistency when scheduled.|
|7|`classification`|loss|Batch mean label-smoothed classification CE; coefficient 1.|
|8|`ordinal`|loss|Batch ordinal-CDF loss before coefficient 0.2.|
|9|`segmentation`|loss|Valid-pixel BCE plus 1 minus smoothed soft Dice; coefficient 1.|
|10|`geometry`|loss|Weighted Smooth L1 point loss, beta 0.02, before coefficient 2.|
|11|`containment`|loss|Valid-pixel positive tread-minus-tyre probability, before coefficient 0.1.|
|12|`alignment`|loss|Usable guide-row interval/mask consistency loss before coefficient 0.1.|
|13|`background_kl`|loss|KL(original detached probabilities || background-swap probabilities); coefficient 0.2, zero when not scheduled.|
|14|`sample_contribution_sum`|loss|Sum of the eight stored objective_contribution values.|
|15|`loss_reconstruction_error`|absolute loss|Absolute difference between that contribution sum and the actual optimizer loss.|
|16|`grad_norm_before_clip`|L2 norm|Combined finite unscaled parameter gradient norm before clip 1.|
|17|`grad_norm_after_clip`|L2 norm|Combined parameter gradient norm measured after clipping.|
|18|`clip_coefficient`|factor|min(1, 1/(preclip norm+1e-6)); observational description of the clipping step.|
|19|`amp_scale_before`|scale factor|Actual gradient-scaler scale before/after the attempt or update.|
|20|`amp_scale_after`|scale factor|Actual gradient-scaler scale before/after the attempt or update.|
|21|`retries`|count|Failed numeric attempts replayed for this completed update.|
|22|`lr_used`|learning rate|Observed sole AdamW group learning rate at the indicated boundary.|
|23|`lr_after_scheduler`|learning rate|Compatibility field: rate at update end; actual post-validation scheduler rate is learning_rate_next_epoch in epoch_summaries.|
|24|`next_epoch`|count|One-based epoch to resume after this completed update.|
|25|`next_cursor`|count|Zero-based next batch cursor; zero after the surrogate boundary.|
|26|`checkpoint_before_sha256`|SHA-256|Manifest identity of full native recovery state immediately before/after the completed update.|
|27|`checkpoint_after_sha256`|SHA-256|Manifest identity of full native recovery state immediately before/after the completed update.|
|28|`data_seconds`|seconds|Measured data stage; blank when not executed. Compute timing excludes data, validation, checkpoint and HF upload unless separately named.|
|29|`forward_seconds`|seconds|Measured forward stage; blank when not executed. Compute timing excludes data, validation, checkpoint and HF upload unless separately named.|
|30|`backward_seconds`|seconds|Measured backward stage; blank when not executed. Compute timing excludes data, validation, checkpoint and HF upload unless separately named.|
|31|`optimizer_seconds`|seconds|Measured optimizer stage; blank when not executed. Compute timing excludes data, validation, checkpoint and HF upload unless separately named.|
|32|`telemetry_seconds`|seconds|Measured telemetry stage; blank when not executed. Compute timing excludes data, validation, checkpoint and HF upload unless separately named.|
|33|`update_compute_seconds`|seconds|Measured update compute stage; blank when not executed. Compute timing excludes data, validation, checkpoint and HF upload unless separately named.|
|34|`validation_seconds`|seconds|Compatibility field blank in NB05; validation costs are in eval_timings and commit/lifecycle CSVs.|
|35|`schedule_epoch0`|integer|Zero-based original r1 sampler/augmentation epoch; CSV epoch is one higher.|
|36|`raw_maps_retained`|0/1|Whether this update retained full native augmented input/target/output maps under the frozen schedule.|

## `train_samples.csv`

57 ordered columns.

| Position | Column | Units / type | Meaning |
|---:|---|---|---|
|1|`condition`|category|baseline / shared_projection; stored in every core CSV row.|
|2|`update`|count|Completed optimizer-update identity; starts at 1 for observed batches.|
|3|`epoch`|one-based epoch|Training/validation epoch; selected/final inference rows name the actual checkpoint epoch.|
|4|`cursor`|count|Zero-based batch position before this update; blank for validation fixture rows.|
|5|`slot`|count|Zero-based example position in the actual batch or fixed validation fixture.|
|6|`image_id`|identifier|Prepared-source image identity; repeated sampling is retained as repeated occurrences.|
|7|`image_sha256`|SHA-256|Original source-image file identity; native augmented arrays have separate manifest hashes.|
|8|`tyre_id`|identifier|Source physical tyre identifier, where supplied; video-parent uncertainty is not resolved by this field.|
|9|`source_video`|identifier|Original video source when applicable; blank for photographs.|
|10|`domain`|category|Original photograph or added video-frame domain from the prepared manifest.|
|11|`label`|class index|Low=0, mid=1, high=2 mileage proxy; not a measured wear/depth/safety class.|
|12|`prediction`|context dependent|Class argmax in image/class tables; normalized horizontal coordinate in geometry_points.|
|13|`correct`|0/1|Observed predicted-class equality to the mileage-proxy label.|
|14|`confidence`|fraction|Maximum predicted class probability.|
|15|`probability_0`|fraction|Observed class probability before this update or at the fixed validation fixture.|
|16|`probability_1`|fraction|Observed class probability before this update or at the fixed validation fixture.|
|17|`probability_2`|fraction|Observed class probability before this update or at the fixed validation fixture.|
|18|`logit_0`|logit|Observed class logit before this update.|
|19|`logit_1`|logit|Observed class logit before this update.|
|20|`logit_2`|logit|Observed class logit before this update.|
|21|`nll`|nats|Negative log true-class probability, clamped at 1e-12 for this diagnostic statistic.|
|22|`brier`|squared probability sum|Sum over the three classes of squared probability minus one-hot truth.|
|23|`entropy_nats`|nats|Class-probability entropy; natural logarithms.|
|24|`label_smoothed_ce`|loss|Per-image CE with label smoothing 0.05, before batch averaging.|
|25|`ordinal_loss`|loss|Per-image mean squared error of the first two class-CDF entries.|
|26|`seg_bce_numerator`|loss sum|Per-image sum of valid, downsampled BCE terms over both regions.|
|27|`seg_bce_denominator`|support|Twice the valid low-resolution pixel count for this image; batch sum is the BCE denominator.|
|28|`seg_dice_loss`|loss|Per-image 1 minus mean two-region soft Dice with +1 smoothing.|
|29|`geometry_numerator`|weighted loss|Mean six-point Smooth L1 multiplied by the image point weight.|
|30|`geometry_weight`|weight|Human=1, polygon-weak=0.25, unlabelled=0; batch sum is clamped below at 1.|
|31|`containment_numerator`|probability sum|Valid-pixel positive tread-minus-tyre sum for this image.|
|32|`containment_denominator`|support|Valid low-resolution pixels for this image; aggregated across batch.|
|33|`alignment_numerator`|absolute difference sum|Usable, valid guide-row interval/mask differences for this image.|
|34|`alignment_denominator`|support|Usable, valid guide-row positions for this image; aggregated across batch.|
|35|`background_kl`|loss|KL(original detached probabilities || background-swap probabilities); coefficient 0.2, zero when not scheduled.|
|36|`objective_contribution`|batch loss share|This image contribution using batch-wide component denominators and frozen task weights.|
|37|`mean_two_region_dice`|fraction|Mean smoothed binary Dice over tyre and tread with ignored pixels excluded.|
|38|`mean_box_iou`|fraction|Mean two-region box IoU; empty boxes score zero.|
|39|`point_kind`|category|human / polygon_weak / none; supervision provenance, not a learned visibility label.|
|40|`human_point_mae`|width fraction|Mean six-point absolute error only for human-labelled images; blank otherwise.|
|41|`annotation_point_mae`|width fraction|Mean six-point error against available human or weak targets; blank when unlabelled.|
|42|`point_output_dtype`|dtype|Actual model point-tensor dtype at the observed forward pass.|
|43|`augmentation_seed`|integer seed|Stateless seed combining engineering seed, epoch, cursor and slot.|
|44|`flip_horizontal`|0/1|Actual deterministic horizontal flip; left/right point indices are swapped.|
|45|`brightness`|factor|Sampled image brightness multiplier.|
|46|`contrast`|factor|Sampled image contrast multiplier.|
|47|`blur_applied`|0/1|Whether the deterministic Gaussian-blur branch was applied.|
|48|`blur_radius`|working-image pixels|Sampled Gaussian radius when applied; blank otherwise.|
|49|`background_donor_image_id`|identifier|Previous batch slot under roll(1); background pixels only, when scheduled.|
|50|`background_applied`|0/1|Whether this update computed the scheduled background-swap consistency pass.|
|51|`background_probability_0`|fraction|Background-swap class probability; blank when the branch is not scheduled.|
|52|`background_probability_1`|fraction|Background-swap class probability; blank when the branch is not scheduled.|
|53|`background_probability_2`|fraction|Background-swap class probability; blank when the branch is not scheduled.|
|54|`background_logit_0`|logit|Background-swap class logit; blank when not scheduled.|
|55|`background_logit_1`|logit|Background-swap class logit; blank when not scheduled.|
|56|`background_logit_2`|logit|Background-swap class logit; blank when not scheduled.|
|57|`checkpoint_before_sha256`|SHA-256|Manifest identity of full native recovery state immediately before/after the completed update.|

## `region_stats.csv`

29 ordered columns.

| Position | Column | Units / type | Meaning |
|---:|---|---|---|
|1|`condition`|category|baseline / shared_projection; stored in every core CSV row.|
|2|`update`|count|Completed optimizer-update identity; starts at 1 for observed batches.|
|3|`epoch`|one-based epoch|Training/validation epoch; selected/final inference rows name the actual checkpoint epoch.|
|4|`cursor`|count|Zero-based batch position before this update; blank for validation fixture rows.|
|5|`slot`|count|Zero-based example position in the actual batch or fixed validation fixture.|
|6|`image_id`|identifier|Prepared-source image identity; repeated sampling is retained as repeated occurrences.|
|7|`image_sha256`|SHA-256|Original source-image file identity; native augmented arrays have separate manifest hashes.|
|8|`tyre_id`|identifier|Source physical tyre identifier, where supplied; video-parent uncertainty is not resolved by this field.|
|9|`source_video`|identifier|Original video source when applicable; blank for photographs.|
|10|`domain`|category|Original photograph or added video-frame domain from the prepared manifest.|
|11|`label`|class index|Low=0, mid=1, high=2 mileage proxy; not a measured wear/depth/safety class.|
|12|`region`|category|tyre / tread; masks overlap and are not mutually exclusive classes.|
|13|`tp`|working-pixel count|Binary region confusion count after threshold 0.5, nearest 4× upsampling and ignore exclusion.|
|14|`fp`|working-pixel count|Binary region confusion count after threshold 0.5, nearest 4× upsampling and ignore exclusion.|
|15|`fn`|working-pixel count|Binary region confusion count after threshold 0.5, nearest 4× upsampling and ignore exclusion.|
|16|`tn`|working-pixel count|Binary region confusion count after threshold 0.5, nearest 4× upsampling and ignore exclusion.|
|17|`valid_pixels`|working pixels|Number of non-ignored 512×384 pixels in the comparison.|
|18|`ignored_pixels`|working pixels|Number excluded by the transformed ignore mask.|
|19|`dice`|fraction|(2TP+1)/(2TP+FP+FN+1), using thresholded/upscaled mask.|
|20|`iou`|fraction|(TP+1)/(TP+FP+FN+1), using thresholded/upscaled mask.|
|21|`box_iou`|fraction|IoU between valid-pixel-derived half-open prediction/truth boxes; zero for missing box.|
|22|`pred_box_x0`|working pixels|Half-open box coordinate on 384-wide ×512-high working grid; blank for an empty valid region.|
|23|`pred_box_y0`|working pixels|Half-open box coordinate on 384-wide ×512-high working grid; blank for an empty valid region.|
|24|`pred_box_x1_exclusive`|working pixels|Half-open box coordinate on 384-wide ×512-high working grid; blank for an empty valid region.|
|25|`pred_box_y1_exclusive`|working pixels|Half-open box coordinate on 384-wide ×512-high working grid; blank for an empty valid region.|
|26|`target_box_x0`|working pixels|Half-open box coordinate on 384-wide ×512-high working grid; blank for an empty valid region.|
|27|`target_box_y0`|working pixels|Half-open box coordinate on 384-wide ×512-high working grid; blank for an empty valid region.|
|28|`target_box_x1_exclusive`|working pixels|Half-open box coordinate on 384-wide ×512-high working grid; blank for an empty valid region.|
|29|`target_box_y1_exclusive`|working pixels|Half-open box coordinate on 384-wide ×512-high working grid; blank for an empty valid region.|

## `geometry_points.csv`

28 ordered columns.

| Position | Column | Units / type | Meaning |
|---:|---|---|---|
|1|`condition`|category|baseline / shared_projection; stored in every core CSV row.|
|2|`update`|count|Completed optimizer-update identity; starts at 1 for observed batches.|
|3|`epoch`|one-based epoch|Training/validation epoch; selected/final inference rows name the actual checkpoint epoch.|
|4|`cursor`|count|Zero-based batch position before this update; blank for validation fixture rows.|
|5|`slot`|count|Zero-based example position in the actual batch or fixed validation fixture.|
|6|`image_id`|identifier|Prepared-source image identity; repeated sampling is retained as repeated occurrences.|
|7|`image_sha256`|SHA-256|Original source-image file identity; native augmented arrays have separate manifest hashes.|
|8|`tyre_id`|identifier|Source physical tyre identifier, where supplied; video-parent uncertainty is not resolved by this field.|
|9|`source_video`|identifier|Original video source when applicable; blank for photographs.|
|10|`domain`|category|Original photograph or added video-frame domain from the prepared manifest.|
|11|`label`|class index|Low=0, mid=1, high=2 mileage proxy; not a measured wear/depth/safety class.|
|12|`point_index`|0..5|Left/right at each of three source guide rows, in model order.|
|13|`guide_fraction`|height fraction|Source guide y/(native_height-1); unchanged by horizontal flip.|
|14|`side`|category|left / right in the transformed example.|
|15|`prediction`|width fraction|Predicted horizontal point coordinate, copied from the actual pre-update forward pass.|
|16|`optimization_target`|width fraction|Exact transformed point tensor supplied to the objective, including zero-weight placeholders.|
|17|`target`|width fraction|Actual transformed annotated target; blank for unlabelled images.|
|18|`source_target`|width fraction|Original unflipped annotated coordinate at source_target_index; blank if unavailable.|
|19|`source_target_index`|0..5|Original index corresponding to this transformed point; swaps each left/right pair on flip.|
|20|`point_kind`|category|human / polygon_weak / none; supervision provenance, not a learned visibility label.|
|21|`point_weight`|weight|Supervision weight human=1, polygon-weak=0.25, none=0.|
|22|`human_eligible`|0/1|One only for independently supplied human geometry targets; weak targets excluded.|
|23|`signed_error_width_fraction`|width fraction|Predicted coordinate minus transformed annotated target, computed with float64 Python scalars.|
|24|`absolute_error_width_fraction`|width fraction|Absolute signed error; blank without an annotation.|
|25|`squared_error_width_fraction`|squared width fraction|Square of signed error; blank without an annotation.|
|26|`absolute_error_native_pixels`|native pixels|Absolute fractional error times (native_width-1).|
|27|`native_width`|native pixels|Original source-image width.|
|28|`native_height`|native pixels|Original source-image height.|

## `latent_metrics.csv`

17 ordered columns.

| Position | Column | Units / type | Meaning |
|---:|---|---|---|
|1|`condition`|category|baseline / shared_projection; stored in every core CSV row.|
|2|`update`|count|Completed optimizer-update identity; starts at 1 for observed batches.|
|3|`epoch`|one-based epoch|Training/validation epoch; selected/final inference rows name the actual checkpoint epoch.|
|4|`cursor`|count|Zero-based batch position before this update; blank for validation fixture rows.|
|5|`slot`|count|Zero-based example position in the actual batch or fixed validation fixture.|
|6|`image_id`|identifier|Prepared-source image identity; repeated sampling is retained as repeated occurrences.|
|7|`image_sha256`|SHA-256|Original source-image file identity; native augmented arrays have separate manifest hashes.|
|8|`tyre_id`|identifier|Source physical tyre identifier, where supplied; video-parent uncertainty is not resolved by this field.|
|9|`source_video`|identifier|Original video source when applicable; blank for photographs.|
|10|`domain`|category|Original photograph or added video-frame domain from the prepared manifest.|
|11|`label`|class index|Low=0, mid=1, high=2 mileage proxy; not a measured wear/depth/safety class.|
|12|`latent`|0..11|Positive latent channel index; no named physical concept is inferred.|
|13|`minimum`|tensor value|Minimum latent/activation value over the declared tensor.|
|14|`maximum`|tensor value|Maximum latent/activation value over the declared tensor.|
|15|`mean`|tensor value|Arithmetic mean over the declared tensor.|
|16|`std`|tensor value|Population standard deviation over spatial latent map or full hooked activation tensor.|
|17|`gated_mass`|activation|Spatial sum of latent activation times normalized predicted gate.|

## `evidence_metrics.csv`

21 ordered columns.

| Position | Column | Units / type | Meaning |
|---:|---|---|---|
|1|`condition`|category|baseline / shared_projection; stored in every core CSV row.|
|2|`update`|count|Completed optimizer-update identity; starts at 1 for observed batches.|
|3|`epoch`|one-based epoch|Training/validation epoch; selected/final inference rows name the actual checkpoint epoch.|
|4|`cursor`|count|Zero-based batch position before this update; blank for validation fixture rows.|
|5|`slot`|count|Zero-based example position in the actual batch or fixed validation fixture.|
|6|`image_id`|identifier|Prepared-source image identity; repeated sampling is retained as repeated occurrences.|
|7|`image_sha256`|SHA-256|Original source-image file identity; native augmented arrays have separate manifest hashes.|
|8|`tyre_id`|identifier|Source physical tyre identifier, where supplied; video-parent uncertainty is not resolved by this field.|
|9|`source_video`|identifier|Original video source when applicable; blank for photographs.|
|10|`domain`|category|Original photograph or added video-frame domain from the prepared manifest.|
|11|`label`|class index|Low=0, mid=1, high=2 mileage proxy; not a measured wear/depth/safety class.|
|12|`class_index`|0..2|Class whose signed contribution/evidence is summarized.|
|13|`logit`|logit|Observed class logit before optimizer update.|
|14|`bias`|logit|Exact class-head bias copied before the optimizer changes parameters.|
|15|`sum_contribution`|logit|Spatial sum of signed class-evidence map; sum plus bias reconstructs logit up to numeric rounding.|
|16|`positive_sum`|logit|Sum of positive signed evidence pixels.|
|17|`negative_sum`|logit|Sum of negative signed evidence pixels; values are nonpositive.|
|18|`inside_true_tread_sum`|logit|Signed evidence weighted by valid true-tread 4×4 area fractions.|
|19|`outside_true_tread_sum`|logit|Signed evidence weighted by one minus that area fraction; complement includes ignored area.|
|20|`evidence_max`|logit per grid pixel|Maximum/minimum signed contribution-map value for this class.|
|21|`evidence_min`|logit per grid pixel|Maximum/minimum signed contribution-map value for this class.|

## `class_confusion.csv`

6 ordered columns.

| Position | Column | Units / type | Meaning |
|---:|---|---|---|
|1|`condition`|category|baseline / shared_projection; stored in every core CSV row.|
|2|`update`|count|Completed optimizer-update identity; starts at 1 for observed batches.|
|3|`epoch`|one-based epoch|Training/validation epoch; selected/final inference rows name the actual checkpoint epoch.|
|4|`label`|class index|Low=0, mid=1, high=2 mileage proxy; not a measured wear/depth/safety class.|
|5|`prediction`|context dependent|Class argmax in image/class tables; normalized horizontal coordinate in geometry_points.|
|6|`count`|sample occurrences|Class-pair count in the sampled batch; observed zero is retained.|

## `gradient_modules.csv`

8 ordered columns.

| Position | Column | Units / type | Meaning |
|---:|---|---|---|
|1|`condition`|category|baseline / shared_projection; stored in every core CSV row.|
|2|`update`|count|Completed optimizer-update identity; starts at 1 for observed batches.|
|3|`epoch`|one-based epoch|Training/validation epoch; selected/final inference rows name the actual checkpoint epoch.|
|4|`module`|identifier|Named model module/group; not a whole-model histogram.|
|5|`parameters_with_gradient`|tensor count|Number of parameter tensors with a gradient in the module group.|
|6|`gradient_norm`|L2 norm|Combined unscaled module gradient norm before clipping.|
|7|`maximum_absolute_gradient`|gradient magnitude|Largest absolute unscaled gradient in this module group.|
|8|`finite`|0/1|Whether all inspected values are finite.|

## `activation_modules.csv`

12 ordered columns.

| Position | Column | Units / type | Meaning |
|---:|---|---|---|
|1|`condition`|category|baseline / shared_projection; stored in every core CSV row.|
|2|`update`|count|Completed optimizer-update identity; starts at 1 for observed batches.|
|3|`epoch`|one-based epoch|Training/validation epoch; selected/final inference rows name the actual checkpoint epoch.|
|4|`phase`|category|normal / background forward; absent background pass is not fabricated.|
|5|`module`|identifier|Named model module/group; not a whole-model histogram.|
|6|`elements`|count|Elements in the observed activation tensor, including batch and spatial axes.|
|7|`mean`|tensor value|Arithmetic mean over the declared tensor.|
|8|`std`|tensor value|Population standard deviation over spatial latent map or full hooked activation tensor.|
|9|`minimum`|tensor value|Minimum latent/activation value over the declared tensor.|
|10|`maximum`|tensor value|Maximum latent/activation value over the declared tensor.|
|11|`l2_norm`|L2 norm|Square root of the summed squared hooked activation values.|
|12|`finite`|0/1|Whether all inspected values are finite.|

## `validation_predictions.csv`

20 ordered columns.

| Position | Column | Units / type | Meaning |
|---:|---|---|---|
|1|`condition`|category|baseline / shared_projection; stored in every core CSV row.|
|2|`update`|count|Completed optimizer-update identity; starts at 1 for observed batches.|
|3|`epoch`|one-based epoch|Training/validation epoch; selected/final inference rows name the actual checkpoint epoch.|
|4|`cursor`|count|Zero-based batch position before this update; blank for validation fixture rows.|
|5|`slot`|count|Zero-based example position in the actual batch or fixed validation fixture.|
|6|`image_id`|identifier|Prepared-source image identity; repeated sampling is retained as repeated occurrences.|
|7|`image_sha256`|SHA-256|Original source-image file identity; native augmented arrays have separate manifest hashes.|
|8|`tyre_id`|identifier|Source physical tyre identifier, where supplied; video-parent uncertainty is not resolved by this field.|
|9|`source_video`|identifier|Original video source when applicable; blank for photographs.|
|10|`domain`|category|Original photograph or added video-frame domain from the prepared manifest.|
|11|`label`|class index|Low=0, mid=1, high=2 mileage proxy; not a measured wear/depth/safety class.|
|12|`prediction`|context dependent|Class argmax in image/class tables; normalized horizontal coordinate in geometry_points.|
|13|`correct`|0/1|Observed predicted-class equality to the mileage-proxy label.|
|14|`probability_0`|fraction|Observed class probability before this update or at the fixed validation fixture.|
|15|`probability_1`|fraction|Observed class probability before this update or at the fixed validation fixture.|
|16|`probability_2`|fraction|Observed class probability before this update or at the fixed validation fixture.|
|17|`nll`|nats|Negative log true-class probability, clamped at 1e-12 for this diagnostic statistic.|
|18|`brier`|squared probability sum|Sum over the three classes of squared probability minus one-hot truth.|
|19|`mean_two_region_dice`|fraction|Mean smoothed binary Dice over tyre and tread with ignored pixels excluded.|
|20|`human_point_mae`|width fraction|Mean six-point absolute error only for human-labelled images; blank otherwise.|

## `epoch_summaries.csv`

12 ordered columns.

| Position | Column | Units / type | Meaning |
|---:|---|---|---|
|1|`condition`|category|baseline / shared_projection; stored in every core CSV row.|
|2|`update`|count|Completed optimizer-update identity; starts at 1 for observed batches.|
|3|`epoch`|one-based epoch|Training/validation epoch; selected/final inference rows name the actual checkpoint epoch.|
|4|`batches`|count|Accumulated completed batches in the reported surrogate epoch.|
|5|`images`|sample occurrences|Accumulated sampled images, with repeats retained.|
|6|`objective_sum`|loss sum|Sum of completed batch objectives; not a pixel-normalized dataset loss.|
|7|`mean_batch_objective`|loss|objective_sum / completed batches.|
|8|`classification_correct`|sample occurrences|Number of correct pre-update classifications over that surrogate epoch.|
|9|`classification_images`|sample occurrences|Denominator for sampled augmented classification accuracy.|
|10|`classification_accuracy`|fraction|Correct / observed sampled occurrences; not nominal cohort accuracy.|
|11|`learning_rate_next_epoch`|learning rate|Observed sole AdamW group learning rate at the indicated boundary.|
|12|`engineering_only`|0/1|One for short engineering fixtures; zero for the declared measured pilot.|

## `resources.csv`

20 ordered columns.

| Position | Column | Units / type | Meaning |
|---:|---|---|---|
|1|`condition`|category|baseline / shared_projection; stored in every core CSV row.|
|2|`update`|count|Completed optimizer-update identity; starts at 1 for observed batches.|
|3|`epoch`|one-based epoch|Training/validation epoch; selected/final inference rows name the actual checkpoint epoch.|
|4|`scope`|category|Resource measurement context, currently completed_update.|
|5|`process_rss_bytes`|bytes|Observed process rss; scope is process/device/container as named, not universal peak memory.|
|6|`container_used_bytes`|bytes|Observed container used; scope is process/device/container as named, not universal peak memory.|
|7|`container_limit_bytes`|bytes|Observed container limit; scope is process/device/container as named, not universal peak memory.|
|8|`container_source`|identifier|Actual cgroup/memory source or unavailable indicator from the frozen helper.|
|9|`cuda_allocated_bytes`|bytes|Observed cuda allocated; scope is process/device/container as named, not universal peak memory.|
|10|`cuda_peak_bytes`|bytes|Observed cuda peak; scope is process/device/container as named, not universal peak memory.|
|11|`working_free_bytes`|bytes|Observed working free; scope is process/device/container as named, not universal peak memory.|
|12|`gpu_uuid`|identifier|Reported physical GPU UUID, when available.|
|13|`gpu_name`|text|Reported GPU model name.|
|14|`gpu_utilization_percent`|percent|Instantaneous utilization sample; not an average for the entire update.|
|15|`gpu_memory_used_mib`|MiB|Instantaneous GPU memory sample; 1 MiB = 2^20 bytes.|
|16|`gpu_memory_total_mib`|MiB|Instantaneous GPU memory sample; 1 MiB = 2^20 bytes.|
|17|`gpu_power_watts`|watts|Instantaneous sampled board power, where supported; not energy.|
|18|`gpu_temperature_c`|degrees Celsius|Instantaneous temperature when supported.|
|19|`gpu_mapping`|category|selected_device only when UUID matches; otherwise other_or_unresolved_device.|
|20|`missing_reason`|text|Explicit unavailable/nonfinite/scheduled resource or task-gradient reason.|

## `events.csv`

10 ordered columns.

| Position | Column | Units / type | Meaning |
|---:|---|---|---|
|1|`condition`|category|baseline / shared_projection; stored in every core CSV row.|
|2|`update`|count|Completed optimizer-update identity; starts at 1 for observed batches.|
|3|`epoch`|one-based epoch|Training/validation epoch; selected/final inference rows name the actual checkpoint epoch.|
|4|`time_utc`|UTC timestamp|Event creation timestamp; not an exact hardware event clock.|
|5|`kind`|category|completed_optimizer_update or numerical_retry in current worker.|
|6|`attempt`|count|Numerical retry attempt when applicable; blank for completion-only events.|
|7|`amp_scale_before`|scale factor|Actual gradient-scaler scale before/after the attempt or update.|
|8|`amp_scale_after`|scale factor|Actual gradient-scaler scale before/after the attempt or update.|
|9|`seconds`|seconds|Event observation duration when applicable.|
|10|`message`|text|Optional event explanation; exceptions are separately retained in FAILURES.jsonl.|

## `eval_predictions.csv`

45 ordered columns.

| Position | Column | Units / type | Meaning |
|---:|---|---|---|
|1|`condition`|category|baseline / shared_projection; stored in every core CSV row.|
|2|`unit`|identifier|Condition/endpoint/epoch/image-index inference-unit identity.|
|3|`job`|identifier|Condition identity in the inherited diagnostic schema.|
|4|`seed`|integer|Predeclared seed 1; fixture seed and measured seed share initialization but never checkpoint state.|
|5|`endpoint`|category|epoch_validation / selected / final.|
|6|`epoch`|one-based epoch|Training/validation epoch; selected/final inference rows name the actual checkpoint epoch.|
|7|`split`|category|validation / test; engineering endpoints use validation_fixture and never read test images.|
|8|`group`|identifier|Cohort all, physical tyre, or inherited clip/tyre grouping as named by the table.|
|9|`tyre`|identifier|Physical tyre source ID; alias of tyre_id in training tables.|
|10|`clip`|identifier|Video source where applicable; alias of source_video.|
|11|`domain`|category|Original photograph or added video-frame domain from the prepared manifest.|
|12|`image_id`|identifier|Prepared-source image identity; repeated sampling is retained as repeated occurrences.|
|13|`image_sha256`|SHA-256|Original source-image file identity; native augmented arrays have separate manifest hashes.|
|14|`label`|class index|Low=0, mid=1, high=2 mileage proxy; not a measured wear/depth/safety class.|
|15|`variant`|category|normal only in NB05. Self-comparison delta fields are zero by definition, not intervention measurements.|
|16|`prediction`|context dependent|Class argmax in image/class tables; normalized horizontal coordinate in geometry_points.|
|17|`correct`|0/1|Observed predicted-class equality to the mileage-proxy label.|
|18|`confidence`|fraction|Maximum predicted class probability.|
|19|`class_margin`|fraction|Largest class probability minus the second largest.|
|20|`true_class_probability`|fraction|Probability assigned to the provided mileage-proxy label.|
|21|`nll`|nats|Negative log true-class probability, clamped at 1e-12 for this diagnostic statistic.|
|22|`brier`|squared probability sum|Sum over the three classes of squared probability minus one-hot truth.|
|23|`entropy_nats`|nats|Class-probability entropy; natural logarithms.|
|24|`probability_0`|fraction|Observed class probability before this update or at the fixed validation fixture.|
|25|`probability_1`|fraction|Observed class probability before this update or at the fixed validation fixture.|
|26|`probability_2`|fraction|Observed class probability before this update or at the fixed validation fixture.|
|27|`logit_0`|logit|Observed class logit before this update.|
|28|`logit_1`|logit|Observed class logit before this update.|
|29|`logit_2`|logit|Observed class logit before this update.|
|30|`normal_prediction`|class index|Normal-forward prediction; equals prediction for the normal-only NB05 inference.|
|31|`probability_l1_delta`|fraction|Reserved self-comparison field, zero for normal-only inference.|
|32|`true_probability_delta`|fraction|Reserved self-comparison field, zero for normal-only inference.|
|33|`normal_predicted_class_probability_delta`|fraction|Reserved self-comparison field, zero for normal-only inference.|
|34|`prediction_changed`|0/1|Reserved normal self-comparison field, zero; no input intervention was performed.|
|35|`mean_two_region_dice`|fraction|Mean smoothed binary Dice over tyre and tread with ignored pixels excluded.|
|36|`mean_box_iou`|fraction|Mean two-region box IoU; empty boxes score zero.|
|37|`point_kind`|category|human / polygon_weak / none; supervision provenance, not a learned visibility label.|
|38|`annotation_point_mae`|width fraction|Mean six-point error against available human or weak targets; blank when unlabelled.|
|39|`human_point_mae`|width fraction|Mean six-point absolute error only for human-labelled images; blank otherwise.|
|40|`human_point_available`|0/1|Whether this image has human point targets; weak/unknown targets are not human truth.|
|41|`gate_entropy_nats`|nats|Spatial entropy of the normalized detached predicted tread gate.|
|42|`gate_mass_inside_true_tread`|fraction|Gate mass weighted by valid true-tread4×4 area fractions.|
|43|`gate_support_fraction`|fraction|Gate grid cells with mass above 1e-6, divided by all grid cells.|
|44|`evidence_residual`|absolute logit|Maximum absolute signed-map-sum plus class bias minus class logit.|
|45|`frozen_r1_selection_point_mae`|width fraction|Torch arithmetic used by original r1 utility; distinct from float64 diagnostic coordinate reconstruction.|

## `eval_regions.csv`

33 ordered columns.

| Position | Column | Units / type | Meaning |
|---:|---|---|---|
|1|`condition`|category|baseline / shared_projection; stored in every core CSV row.|
|2|`unit`|identifier|Condition/endpoint/epoch/image-index inference-unit identity.|
|3|`job`|identifier|Condition identity in the inherited diagnostic schema.|
|4|`seed`|integer|Predeclared seed 1; fixture seed and measured seed share initialization but never checkpoint state.|
|5|`endpoint`|category|epoch_validation / selected / final.|
|6|`epoch`|one-based epoch|Training/validation epoch; selected/final inference rows name the actual checkpoint epoch.|
|7|`split`|category|validation / test; engineering endpoints use validation_fixture and never read test images.|
|8|`group`|identifier|Cohort all, physical tyre, or inherited clip/tyre grouping as named by the table.|
|9|`tyre`|identifier|Physical tyre source ID; alias of tyre_id in training tables.|
|10|`clip`|identifier|Video source where applicable; alias of source_video.|
|11|`domain`|category|Original photograph or added video-frame domain from the prepared manifest.|
|12|`image_id`|identifier|Prepared-source image identity; repeated sampling is retained as repeated occurrences.|
|13|`image_sha256`|SHA-256|Original source-image file identity; native augmented arrays have separate manifest hashes.|
|14|`label`|class index|Low=0, mid=1, high=2 mileage proxy; not a measured wear/depth/safety class.|
|15|`variant`|category|normal only in NB05. Self-comparison delta fields are zero by definition, not intervention measurements.|
|16|`region`|category|tyre / tread; masks overlap and are not mutually exclusive classes.|
|17|`tp`|working-pixel count|Binary region confusion count after threshold 0.5, nearest 4× upsampling and ignore exclusion.|
|18|`fp`|working-pixel count|Binary region confusion count after threshold 0.5, nearest 4× upsampling and ignore exclusion.|
|19|`fn`|working-pixel count|Binary region confusion count after threshold 0.5, nearest 4× upsampling and ignore exclusion.|
|20|`tn`|working-pixel count|Binary region confusion count after threshold 0.5, nearest 4× upsampling and ignore exclusion.|
|21|`valid_pixels`|working pixels|Number of non-ignored 512×384 pixels in the comparison.|
|22|`ignored_pixels`|working pixels|Number excluded by the transformed ignore mask.|
|23|`dice`|fraction|(2TP+1)/(2TP+FP+FN+1), using thresholded/upscaled mask.|
|24|`iou`|fraction|(TP+1)/(TP+FP+FN+1), using thresholded/upscaled mask.|
|25|`box_iou`|fraction|IoU between valid-pixel-derived half-open prediction/truth boxes; zero for missing box.|
|26|`pred_box_x0`|working pixels|Half-open box coordinate on 384-wide ×512-high working grid; blank for an empty valid region.|
|27|`pred_box_y0`|working pixels|Half-open box coordinate on 384-wide ×512-high working grid; blank for an empty valid region.|
|28|`pred_box_x1_exclusive`|working pixels|Half-open box coordinate on 384-wide ×512-high working grid; blank for an empty valid region.|
|29|`pred_box_y1_exclusive`|working pixels|Half-open box coordinate on 384-wide ×512-high working grid; blank for an empty valid region.|
|30|`target_box_x0`|working pixels|Half-open box coordinate on 384-wide ×512-high working grid; blank for an empty valid region.|
|31|`target_box_y0`|working pixels|Half-open box coordinate on 384-wide ×512-high working grid; blank for an empty valid region.|
|32|`target_box_x1_exclusive`|working pixels|Half-open box coordinate on 384-wide ×512-high working grid; blank for an empty valid region.|
|33|`target_box_y1_exclusive`|working pixels|Half-open box coordinate on 384-wide ×512-high working grid; blank for an empty valid region.|

## `eval_points.csv`

27 ordered columns.

| Position | Column | Units / type | Meaning |
|---:|---|---|---|
|1|`condition`|category|baseline / shared_projection; stored in every core CSV row.|
|2|`unit`|identifier|Condition/endpoint/epoch/image-index inference-unit identity.|
|3|`job`|identifier|Condition identity in the inherited diagnostic schema.|
|4|`seed`|integer|Predeclared seed 1; fixture seed and measured seed share initialization but never checkpoint state.|
|5|`endpoint`|category|epoch_validation / selected / final.|
|6|`epoch`|one-based epoch|Training/validation epoch; selected/final inference rows name the actual checkpoint epoch.|
|7|`split`|category|validation / test; engineering endpoints use validation_fixture and never read test images.|
|8|`group`|identifier|Cohort all, physical tyre, or inherited clip/tyre grouping as named by the table.|
|9|`tyre`|identifier|Physical tyre source ID; alias of tyre_id in training tables.|
|10|`clip`|identifier|Video source where applicable; alias of source_video.|
|11|`domain`|category|Original photograph or added video-frame domain from the prepared manifest.|
|12|`image_id`|identifier|Prepared-source image identity; repeated sampling is retained as repeated occurrences.|
|13|`image_sha256`|SHA-256|Original source-image file identity; native augmented arrays have separate manifest hashes.|
|14|`label`|class index|Low=0, mid=1, high=2 mileage proxy; not a measured wear/depth/safety class.|
|15|`variant`|category|normal only in NB05. Self-comparison delta fields are zero by definition, not intervention measurements.|
|16|`point_index`|0..5|Left/right at each of three source guide rows, in model order.|
|17|`guide_fraction`|height fraction|Source guide y/(native_height-1); unchanged by horizontal flip.|
|18|`side`|category|left / right in the transformed example.|
|19|`prediction`|width fraction|Predicted horizontal point coordinate at the fixed evaluation checkpoint.|
|20|`target`|width fraction|Actual transformed annotated target; blank for unlabelled images.|
|21|`target_kind`|category|Inherited point provenance: human / polygon_weak / none.|
|22|`human_eligible`|0/1|One only for independently supplied human geometry targets; weak targets excluded.|
|23|`signed_error_width_fraction`|width fraction|Predicted coordinate minus transformed annotated target, computed with float64 Python scalars.|
|24|`absolute_error_width_fraction`|width fraction|Absolute signed error; blank without an annotation.|
|25|`absolute_error_native_pixels`|native pixels|Absolute fractional error times (native_width-1).|
|26|`native_width`|native pixels|Original source-image width.|
|27|`native_height`|native pixels|Original source-image height.|

## `eval_latents.csv`

21 ordered columns.

| Position | Column | Units / type | Meaning |
|---:|---|---|---|
|1|`condition`|category|baseline / shared_projection; stored in every core CSV row.|
|2|`unit`|identifier|Condition/endpoint/epoch/image-index inference-unit identity.|
|3|`job`|identifier|Condition identity in the inherited diagnostic schema.|
|4|`seed`|integer|Predeclared seed 1; fixture seed and measured seed share initialization but never checkpoint state.|
|5|`endpoint`|category|epoch_validation / selected / final.|
|6|`epoch`|one-based epoch|Training/validation epoch; selected/final inference rows name the actual checkpoint epoch.|
|7|`split`|category|validation / test; engineering endpoints use validation_fixture and never read test images.|
|8|`group`|identifier|Cohort all, physical tyre, or inherited clip/tyre grouping as named by the table.|
|9|`tyre`|identifier|Physical tyre source ID; alias of tyre_id in training tables.|
|10|`clip`|identifier|Video source where applicable; alias of source_video.|
|11|`domain`|category|Original photograph or added video-frame domain from the prepared manifest.|
|12|`image_id`|identifier|Prepared-source image identity; repeated sampling is retained as repeated occurrences.|
|13|`image_sha256`|SHA-256|Original source-image file identity; native augmented arrays have separate manifest hashes.|
|14|`label`|class index|Low=0, mid=1, high=2 mileage proxy; not a measured wear/depth/safety class.|
|15|`variant`|category|normal only in NB05. Self-comparison delta fields are zero by definition, not intervention measurements.|
|16|`latent`|0..11|Positive latent channel index; no named physical concept is inferred.|
|17|`minimum`|tensor value|Minimum latent/activation value over the declared tensor.|
|18|`maximum`|tensor value|Maximum latent/activation value over the declared tensor.|
|19|`mean`|tensor value|Arithmetic mean over the declared tensor.|
|20|`std`|tensor value|Population standard deviation over spatial latent map or full hooked activation tensor.|
|21|`gated_mass`|activation|Spatial sum of latent activation times normalized predicted gate.|

## `eval_evidence.csv`

25 ordered columns.

| Position | Column | Units / type | Meaning |
|---:|---|---|---|
|1|`condition`|category|baseline / shared_projection; stored in every core CSV row.|
|2|`unit`|identifier|Condition/endpoint/epoch/image-index inference-unit identity.|
|3|`job`|identifier|Condition identity in the inherited diagnostic schema.|
|4|`seed`|integer|Predeclared seed 1; fixture seed and measured seed share initialization but never checkpoint state.|
|5|`endpoint`|category|epoch_validation / selected / final.|
|6|`epoch`|one-based epoch|Training/validation epoch; selected/final inference rows name the actual checkpoint epoch.|
|7|`split`|category|validation / test; engineering endpoints use validation_fixture and never read test images.|
|8|`group`|identifier|Cohort all, physical tyre, or inherited clip/tyre grouping as named by the table.|
|9|`tyre`|identifier|Physical tyre source ID; alias of tyre_id in training tables.|
|10|`clip`|identifier|Video source where applicable; alias of source_video.|
|11|`domain`|category|Original photograph or added video-frame domain from the prepared manifest.|
|12|`image_id`|identifier|Prepared-source image identity; repeated sampling is retained as repeated occurrences.|
|13|`image_sha256`|SHA-256|Original source-image file identity; native augmented arrays have separate manifest hashes.|
|14|`label`|class index|Low=0, mid=1, high=2 mileage proxy; not a measured wear/depth/safety class.|
|15|`variant`|category|normal only in NB05. Self-comparison delta fields are zero by definition, not intervention measurements.|
|16|`class_index`|0..2|Class whose signed contribution/evidence is summarized.|
|17|`logit`|logit|Observed class logit before optimizer update.|
|18|`bias`|logit|Exact class-head bias copied before the optimizer changes parameters.|
|19|`sum_contribution`|logit|Spatial sum of signed class-evidence map; sum plus bias reconstructs logit up to numeric rounding.|
|20|`positive_sum`|logit|Sum of positive signed evidence pixels.|
|21|`negative_sum`|logit|Sum of negative signed evidence pixels; values are nonpositive.|
|22|`inside_true_tread_sum`|logit|Signed evidence weighted by valid true-tread 4×4 area fractions.|
|23|`outside_true_tread_sum`|logit|Signed evidence weighted by one minus that area fraction; complement includes ignored area.|
|24|`evidence_max`|logit per grid pixel|Maximum/minimum signed contribution-map value for this class.|
|25|`evidence_min`|logit per grid pixel|Maximum/minimum signed contribution-map value for this class.|

## `task_gradients.csv`

21 ordered columns.

| Position | Column | Units / type | Meaning |
|---:|---|---|---|
|1|`condition`|category|baseline / shared_projection; stored in every core CSV row.|
|2|`update`|count|Completed optimizer-update identity; starts at 1 for observed batches.|
|3|`epoch`|one-based epoch|Training/validation epoch; selected/final inference rows name the actual checkpoint epoch.|
|4|`cursor`|count|Zero-based batch position before this update; blank for validation fixture rows.|
|5|`class_objective`|weighted loss|A=CE+0.2 ordinal+0.2 background KL.|
|6|`dense_objective`|weighted loss|B=segmentation+2 point+0.1 containment+0.1 alignment.|
|7|`shared_parameters`|scalar parameter count|Number of declared shared scalar parameters, not the number of tensors.|
|8|`class_norm`|gradient L2 norm|Unscaled shared-gradient norm for A.|
|9|`dense_norm`|gradient L2 norm|Unscaled shared-gradient norm for B.|
|10|`dot_before`|gradient dot product|Combined shared A/B dot product before routing.|
|11|`cosine_before`|[-1,1]|Combined shared A/B cosine before routing; denominator clamped for zero gradients.|
|12|`dot_after`|gradient dot product|Dot product of the two routed components; unchanged in baseline.|
|13|`cosine_after`|[-1,1]|Cosine of routed components; zero-denominator convention is explicit in code.|
|14|`projection_applied`|0/1|One only in the projection condition when the finite shared dot product is negative.|
|15|`class_projected_norm`|gradient L2 norm|Norm of routed class component; original norm when routing is not applied.|
|16|`dense_projected_norm`|gradient L2 norm|Norm of routed dense component; original norm when routing is not applied.|
|17|`shared_gradient_change_norm`|gradient L2 norm|Actual shared update change relative to the original total backward gradient.|
|18|`head_gradient_max_change`|gradient magnitude|Measured maximum task-head gradient change during routing; must be zero.|
|19|`projection_seconds`|seconds|Measured routing/statistics segment; an observed software duration, not pure GPU kernel time.|
|20|`finite`|0/1|Whether all inspected values are finite.|
|21|`missing_reason`|text|Explicit unavailable/nonfinite/scheduled resource or task-gradient reason.|

## `eval_timings.csv`

9 ordered columns.

| Position | Column | Units / type | Meaning |
|---:|---|---|---|
|1|`condition`|category|baseline / shared_projection; stored in every core CSV row.|
|2|`epoch`|one-based epoch|Training/validation epoch; selected/final inference rows name the actual checkpoint epoch.|
|3|`split`|category|validation / test; engineering endpoints use validation_fixture and never read test images.|
|4|`endpoint`|category|epoch_validation / selected / final.|
|5|`image_id`|identifier|Prepared-source image identity; repeated sampling is retained as repeated occurrences.|
|6|`load_seconds`|seconds|Measured inference input-loading or total observation duration as named.|
|7|`forward_seconds`|seconds|Measured forward stage; blank when not executed. Compute timing excludes data, validation, checkpoint and HF upload unless separately named.|
|8|`metrics_seconds`|seconds|Measured metrics stage; blank when not executed. Compute timing excludes data, validation, checkpoint and HF upload unless separately named.|
|9|`total_seconds`|seconds|Measured inference input-loading or total observation duration as named.|

## `eval_summaries.csv`

16 ordered columns.

| Position | Column | Units / type | Meaning |
|---:|---|---|---|
|1|`condition`|category|baseline / shared_projection; stored in every core CSV row.|
|2|`epoch`|one-based epoch|Training/validation epoch; selected/final inference rows name the actual checkpoint epoch.|
|3|`split`|category|validation / test; engineering endpoints use validation_fixture and never read test images.|
|4|`endpoint`|category|epoch_validation / selected / final.|
|5|`level`|category|cohort / tyre.|
|6|`group`|identifier|Cohort all, physical tyre, or inherited clip/tyre grouping as named by the table.|
|7|`images`|sample occurrences|Accumulated sampled images, with repeats retained.|
|8|`accuracy`|fraction|Correct classifications divided by all image observations in the group.|
|9|`balanced_accuracy_observed`|fraction|Mean recall over classes actually supported by that cohort; no held-out mid recall is invented.|
|10|`nll`|nats|Negative log true-class probability, clamped at 1e-12 for this diagnostic statistic.|
|11|`brier`|squared probability sum|Sum over the three classes of squared probability minus one-hot truth.|
|12|`mean_two_region_dice`|fraction|Mean smoothed binary Dice over tyre and tread with ignored pixels excluded.|
|13|`human_point_images`|image count|Human-labelled images contributing point MAE.|
|14|`human_point_mae`|width fraction|Mean six-point absolute error only for human-labelled images; blank otherwise.|
|15|`frozen_r1_selection_point_mae`|width fraction|Torch arithmetic used by original r1 utility; distinct from float64 diagnostic coordinate reconstruction.|
|16|`joint_validation_utility`|selection utility|0.5 observed BA+0.35 mean Dice+0.15 max(0,1-frozen point MAE/0.05).|

## `eval_class_metrics.csv`

15 ordered columns.

| Position | Column | Units / type | Meaning |
|---:|---|---|---|
|1|`condition`|category|baseline / shared_projection; stored in every core CSV row.|
|2|`epoch`|one-based epoch|Training/validation epoch; selected/final inference rows name the actual checkpoint epoch.|
|3|`split`|category|validation / test; engineering endpoints use validation_fixture and never read test images.|
|4|`endpoint`|category|epoch_validation / selected / final.|
|5|`level`|category|cohort / tyre.|
|6|`group`|identifier|Cohort all, physical tyre, or inherited clip/tyre grouping as named by the table.|
|7|`class_index`|0..2|Class receiving the one-versus-rest confusion/support metrics.|
|8|`support`|image count|Ground-truth support for the stated class.|
|9|`tp`|image count|One-versus-rest class confusion count in this group.|
|10|`fp`|image count|One-versus-rest class confusion count in this group.|
|11|`fn`|image count|One-versus-rest class confusion count in this group.|
|12|`tn`|image count|One-versus-rest class confusion count in this group.|
|13|`precision`|fraction|TP/(TP+FP); blank when the denominator is zero.|
|14|`recall`|fraction|TP/(TP+FN); blank for an absent true class.|
|15|`f1`|fraction|2TP/(2TP+FP+FN); blank for a zero denominator.|

## `calibration_bins.csv`

10 ordered columns.

| Position | Column | Units / type | Meaning |
|---:|---|---|---|
|1|`condition`|category|baseline / shared_projection; stored in every core CSV row.|
|2|`epoch`|one-based epoch|Training/validation epoch; selected/final inference rows name the actual checkpoint epoch.|
|3|`split`|category|validation / test; engineering endpoints use validation_fixture and never read test images.|
|4|`endpoint`|category|epoch_validation / selected / final.|
|5|`bin_index`|0..9|Ten equal confidence bins; exact1.0 belongs to the last bin.|
|6|`lower_inclusive`|fraction|Confidence-bin lower edge, inclusive.|
|7|`upper_inclusive_last`|fraction|Upper edge; exclusive except for the final bin.|
|8|`images`|sample occurrences|Accumulated sampled images, with repeats retained.|
|9|`correct`|image count|Number correct in this bin; not an accuracy fraction.|
|10|`confidence_sum`|probability sum|Sum of predicted-class confidences in this bin; retain with images/correct for later ECE calculation.|

## Native arrays and auxiliary observations

| File | Meaning |
|---|---|
| `inputs.npy` | Float32 normalized input; training batch `[8,3,512,384]`, fixed validation example `[3,512,384]` |
| `targets.npy` | Uint8 tyre/tread/ignore masks on the corresponding512×384 working grid |
| `outputs.npy` / endpoint chunk | Float32 mask2, latent12, signed evidence3, gate1 channels on128×96; full endpoint chunks contain up to8 images |
| `INITIALIZATION.json` | Actual initial model tensor-content SHA-256, parameter count, architecture and seed |
| `COMMIT_TIMINGS.csv` | Unit, updates, phase_after_commit, seconds, UTC and post-commit observation scope |
| `WORKER_LIFECYCLE.csv` | Start/resume and exit UTC, main-process elapsed seconds, updates, phase and termination status; includes input/evaluation/checkpoint work, excludes stopped-worker HF waiting |
| `NUMERICAL_ATTEMPTS.csv` | Unique failed-attempt ID, UTC, candidate update/epoch/cursor, AMP scales, reason and explicit uncommitted-attempt status |
| `STATE_VERSIONS.csv` | Every local unit/current optimizer cursor, state hash, unit-row hash, observation-chain hash and native-presence flag |
| `CAPACITY.json` | Actual saved-output logical bytes and free filesystem bytes at publication |
| Native `.pt` | Model, optimizer, scheduler, scaler, Python/NumPy/CPU/CUDA RNG, training/evaluation cursors, partial statistics and row/hash-chain anchors |

Auxiliary post-commit timings or exit rows can be incomplete after a forced kill. They are not the authoritative scientific cursor. The native/journal unit and published manifest establish that cursor. Lifecycle durations are process observations, not a claim of continuous energy or complete end-to-end notebook wall time.
