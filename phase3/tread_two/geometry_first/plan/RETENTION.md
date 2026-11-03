# Evidence retention, native tensors and storage budget

**Design contract; exact fixture byte measurements pending.** All scalar/tabular records are direct CSV. JSON carries manifests/configuration. Bulk tensors use direct uncompressed native files, never a ZIP/NPZ/database envelope. Native bulk is additional evidence, not a substitute for browsable per-image/loss/gradient CSV.

## IDs, types and loss reconstruction

`run` is the 12-character measured protocol prefix; full SHA stays in CONTRACT.json. `sample` identifies one geometry/atlas source pass; `(sample,view)` identifies its two texture views. Sample IDs are monotonically allocated by committed units, not assumed contiguous within one stage. `image_id/source_sha256` join immutable source bytes; `epoch/draw` join the complete schedule; `parameter_update` is the already-applied state used for a pre-update observation. Geometry/error/texture optimizer-step counters are separate. `stage/endpoint` distinguish initial control, online train, validation, selected/final and interventions. `region=0/1` means tyre/tread, `guide=0..2`, `side=0/1` means left/right, `view=0/1` identifies the paired texture views, and row/column/patch indices are zero-based.

Use a single CSV null representation: empty field with an availability flag/count/reason. Absent human targets or unsupported boxes are not numerical zeros. Counts/IDs stay integers; counter seeds preserve full 64-bit identity, with decimal-string JSON/UI representation where needed. Serialize boolean CSV fields as 0/1. Scientific scalars accumulate in FP64 where appropriate. Raw FP32 vector/probability columns use 9 significant decimal digits with a required FP32 roundtrip test; derived FP64 metrics use 17 digits. Exact raw FP32 logits/tensors also remain native. The dense-pixel CSV omits a repeated run column and repeated logits to control size; its file manifest supplies run identity and the indexed native tensor supplies every original logit. The design JSON proposes every field/type; it is not yet an executed runtime schema.

Retain unreduced numerator/count statistics for BCE, weighted point loss, masked Huber, row agreement, drift and quality losses. Retain every 64D projector output so each batch's invariance, sample variance and covariance objective can be independently recomputed. Save raw 96D global embeddings plus normalized retrieval values/normalization rules. Loss accounting includes epoch ramp, per-group clipping, LR, skip flags and exact sample/component denominators.

## Bulk native records — all training observations

Closed, immutable uncompressed .npy shards contain at most 8 completed updates per tensor type and close early at epoch/evaluation boundaries. Active raw FP32 append streams have fixed block-shape metadata, per-unit byte/hash cursors and explicit CSV indexes. Closing writes a self-describing NPY file, verifies it, commits the new index/pointer, then releases the redundant active stream only after that atomic unit is durable. On a crash, only uncommitted suffixes or unselected close outputs are discarded. Each shard/index maps sample/view/patch to slice, shape, dtype, byte length and SHA. Variable masked tiles pad to 8 with an availability mask, never fake loss observations. Both active-stream and NPY roundtrip/cursor cases must pass before freeze.

| Native content | Planned maximum for 60 epochs; excludes validation/endpoints |
|---|---:|
| Every training geometry logit:23,520×2×128×96 FP32 |2,312,110,080 bytes |
| Both pre/post-context patch tokens:47,040×2×32×96 FP32 |1,156,055,040 bytes |
| Every masked prediction and target:47,040×8×2×4×16×16 FP32 |3,082,813,440 bytes |
| Masked-pixel weights:47,040×8×16×16 FP32 |385,351,680 bytes |
| Canonical row parameters:23,520×512×3 FP32 plus validity |Approximately 145M B plus row flags |

These are shape-based upper bounds, not measured file sizes. Ineligible views reduce actual counts and are explicitly logged. Together with global/projector CSV, this is substantially fuller raw evidence than earlier context runs. Every valid training mask logit, masked target/prediction and patch token is retained; original annotated source files remain pinned on Kaggle/HF rather than duplicated per draw.

Validation native chunks retain every epoch's81 mask-logit maps, all points/quality outputs, atlas row parameters and paired token/prediction data. Selected/final identity outputs retain all 570 images, complete dense logits/probabilities, points, predicted support/row parameters, texture tensors and overlays; source-cohort roles remain distinct. Dense CSV exports cover both regions at 128×96, up to 28,016,640 pixel rows across two 570-image endpoints. If no eligible selected checkpoint exists, selected data stays unavailable instead of fabricating another endpoint.

Source RGB plus exact photometric parameters, original preprocessing code/version, predicted row-grid parameters and validity are sufficient to reconstruct the deterministic forward atlas. The full 512×256×2 coordinate grid and repeated RGB atlas for every training observation are not separately duplicated. Selected/final examples include rendered atlas images for inspection. All source grid conventions, clipping and intermediate precision must have a direct reconstruction check.

## Recovery state versus permanent history

At every complete update persist model/trainable state, all optimizer groups/moments and counters, scheduler, scaler-disabled state, Python/NumPy/CPU/CUDA RNGs, sampler/mask plans, cursor, partial epoch accumulators, native chunk positions, CSV byte/row/hash cursors, source/environment identities and current selection. Two recent local native recovery states rotate atomically. Preserve all 60 epoch model checkpoints, selected/final full exports, and every confirmed 30-minute/major/stop remote snapshot. All 40 CSV families and native observation chunks survive rotation.

Every individual update's complete optimizer state is not kept permanently; its contemporaneous outputs/metrics/gradients, unit hash and the rolling recoverable state are retained. There is no claim of 2940 permanent full model/optimizer checkpoint copies. All epoch weights and published snapshots remain available. Do not use a hidden local database as the sole record of any requested metric.

## Output capacity and publication staging

| Category | Planning allowance, to measure before training |
|---|---|
| Full training native observations |About 7.1 GB including weights/row flags; shape math above |
| Direct embeddings/scalars/gradients/points/support CSV |About 0.7–1.5 GB depending actual row width |
| Selected/final dense CSV |About 1.5–3 GB; exact ASCII size must be measured |
| Validation/endpoint native chunks |About 0.8–1.5 GB; fixture-derived projection required |
|60 epoch weights, two rotating states, sources/anchor |Target≤0.7 GB; exact counts pending |
| Figures/interactive examples/indexes |Target≤0.3 GB; bounded gallery, complete numeric records |
| Logical artifact ceiling |15,000,000,000 bytes |
| Projected working peak ceiling |18,000,000,000 bytes, preserving 2,000,000,000 free bytes within a 20 GB assumption |

The ranges may exceed the logical ceiling when pessimistic ends are combined. This is an explicit preflight risk, not a claim that the design already fits. Measure real CSV/native writes from a representative batch and complete validation fixture; include a 1.2 uncertainty factor and active staging. If projected logical or peak exceeds the ceiling, fail before measured updates. Revise the retention contract/implementation before freeze or improve exact-format/chunk encoding. Never silently discard requested telemetry during the measured fit.

Closed epoch chunks are immutable and can be hard-linked into same-filesystem snapshots after an actual hard-link/immutability test. Active CSV/native prefixes must be copied to their committed boundary; never hard-link a file still being appended. Rotating recovery states use replace-by-new-inode, not in-place overwrite. If hard links are unsupported, the full-copy peak must fit the same limit or preflight fails. Do not depend on the reported 1 TBscratch allowance; measure it and use it only as optional cache. Confirmed remote staging can be removed after hash verification without deleting science or active state.

## CSV dictionary and cadence

[SCHEMAS.json](SCHEMAS.json) contains all 40 draft families, every field, key, type, units and missingness. The inventory below is generated from it, so the documentation and machine proposal agree. A typed executable schema and validators must be frozen with the eventual runtime FILES.json.

| Family | Key | Cadence | Fields |
|---|---|---|---:|
| `source_images` | `image_id` | once | 14 |
| `annotation_targets` | `image_id, guide, side` | once | 10 |
| `epoch_schedule` | `epoch, draw` | every epoch before updates | 15 |
| `sample_observations` | `sample` | every train/evaluation source pass | 15 |
| `view_plans` | `sample, view` | two texture views per eligible sample | 12 |
| `region_metrics` | `sample, region` | every sample and 2 regions | 24 |
| `point_values` | `sample, guide, side` | every sample and 6 points | 11 |
| `atlas_bands` | `sample, band` | every sample and 8 bands | 9 |
| `atlas_rows` | `sample, atlas_row` | selected/final source cohorts; all training row parameters additionally native | 9 |
| `patch_support` | `sample, patch` | every sample and 32 tiles | 10 |
| `masked_patch_plan` | `sample, view, patch` | every eligible texture view and masked tile | 12 |
| `masked_patch_metrics` | `sample, view, patch` | every masked tile | 10 |
| `global_embeddings` | `sample, view, dimension` | every eligible view, all 96 coordinates | 5 |
| `projected_embeddings` | `sample, view, dimension` | every eligible view, all 64 coordinates | 5 |
| `pair_losses` | `sample` | every eligible paired draw | 8 |
| `geometry_losses` | `sample` | every sample; unreduced sufficient statistics | 14 |
| `quality_predictions` | `sample, quantity` | every sample and 3 error outputs | 10 |
| `quality_bins` | `stage, epoch, quantity, bin` | each validation epoch and endpoints | 11 |
| `batch_objectives` | `update` | every completed update | 17 |
| `train_updates` | `update` | every completed update | 13 |
| `parameter_gradients` | `update, parameter` | every trainable tensor/update | 11 |
| `group_gradient_summary` | `update, group` | every group/update | 9 |
| `geometry_drift` | `sample` | every sample against same-view anchor | 8 |
| `latent_diagnostics` | `update, stage` | every batch and validation epoch | 13 |
| `epoch_metrics` | `stage, epoch, task, group` | every epoch and endpoint task/group | 12 |
| `source_exposure` | `epoch, image_id` | each epoch per original training image | 10 |
| `pipeline_timings` | `unit` | every update/evaluation unit | 10 |
| `resource_samples` | `unit` | every update and publication boundary | 10 |
| `numeric_events` | `event_id` | only when event occurs; empty table valid | 10 |
| `endpoint_neighbours` | `sample, reference_id` | selected/final diagnostic retrieval | 11 |
| `spatial_maps_index` | `sample, quantity` | every retained native map chunk | 10 |
| `native_chunks_index` | `file` | each closed epoch or snapshot | 10 |
| `selection` | `epoch` | each validation epoch | 15 |
| `interventions` | `sample, variant` | selected/final predeclared probes | 10 |
| `dense_pixels` | `sample, region, row, column` | selected/final identity cohorts only; run identity bound by file manifest; each row is one 4x4 report block; all logits additionally native | 8 |
| `unit_ledger` | `unit` | every complete atomic unit | 8 |
| `HF_requests` | `event_id` | every request category/event; no secrets | 10 |
| `HF_publications` | `attempt` | every publication attempt | 12 |
| `environment` | `record_id` | once and recovery attempts | 14 |
| `gate_checks` | `check` | preflight and final audit | 8 |

## Every declared column

### source_images

`run`, `image_id`, `image_sha256`, `role`, `domain`, `confirmed_tyre_id`, `source_group`, `leakage_component`, `label_proxy`, `point_kind`, `human_target_available`, `weak_target_available`, `mask_sha256`, `ignore_sha256`.

### annotation_targets

`run`, `image_id`, `guide`, `side`, `x_width_fraction`, `y_height_fraction`, `target_kind`, `optimization_weight`, `human_eligible`, `metric_available`.

### epoch_schedule

`run`, `epoch`, `draw`, `batch`, `batch_slot`, `image_id`, `source_sha256`, `leakage_component`, `is_padding`, `base_augmentation_seed`, `view_a_seed`, `view_b_seed`, `mask_a_seed`, `mask_b_seed`, `schedule_sha256`.

### sample_observations

`run`, `sample`, `stage`, `endpoint`, `epoch`, `completed_update`, `parameter_update`, `image_id`, `source_sha256`, `geometry_valid`, `SSL_valid`, `SSL_skip_reason`, `guide_count`, `mask_valid_pixels`, `point_target_kind`.

### view_plans

`run`, `sample`, `view`, `brightness`, `contrast`, `gamma`, `geometry_augmentation_seed`, `texture_seed`, `mask_seed`, `masked_tiles`, `visible_tiles`, `masked_target_pixels`.

### region_metrics

`run`, `sample`, `region`, `tp`, `fp`, `fn`, `valid_pixels`, `target_pixels`, `predicted_pixels`, `dice`, `iou`, `box_iou`, `target_box_available`, `predicted_box_available`, `predicted_box_left`, `predicted_box_top`, `predicted_box_right`, `predicted_box_bottom`, `target_box_left`, `target_box_top`, `target_box_right`, `target_box_bottom`, `bce_sum`, `bce_count`.

### point_values

`run`, `sample`, `guide`, `side`, `prediction`, `target`, `absolute_error_width_fraction`, `smooth_l1_sum`, `target_kind`, `optimization_weight`, `human_metric_available`.

### atlas_bands

`run`, `sample`, `band`, `valid_rows`, `valid_fraction`, `mean_width_width_fraction`, `clipped_fraction`, `point_mask_gap_width_fraction`, `unsupported`.

### atlas_rows

`run`, `sample`, `atlas_row`, `source_y`, `left_x`, `right_x`, `row_valid`, `source_clipped`, `grid_parameter_native_key`.

### patch_support

`run`, `sample`, `patch`, `band`, `column`, `valid_fraction`, `pooling_mass`, `valid_target_fraction`, `source_row_start`, `source_row_end`.

### masked_patch_plan

`run`, `sample`, `view`, `patch`, `masked`, `valid_loss_pixels`, `mask_weight_sum`, `target_mean`, `target_std`, `native_prediction_key`, `native_target_key`, `native_weight_key`.

### masked_patch_metrics

`run`, `sample`, `view`, `patch`, `huber_sum`, `huber_weight_count`, `absolute_error_sum`, `absolute_error_max`, `mean_predictor_huber_sum`, `mean_predictor_weight_count`.

### global_embeddings

`run`, `sample`, `view`, `dimension`, `value`.

### projected_embeddings

`run`, `sample`, `view`, `dimension`, `value`.

### pair_losses

`run`, `sample`, `invariance`, `masked_huber_a`, `masked_huber_b`, `pair_cosine_distance`, `valid_images_in_batch`, `valid_components_in_batch`.

### geometry_losses

`run`, `sample`, `bce_sum`, `bce_count`, `dice_loss_sum`, `dice_region_count`, `point_loss_sum`, `point_weight_count`, `containment_sum`, `containment_count`, `row_agreement_sum`, `row_agreement_count`, `anchor_drift_sum`, `anchor_drift_count`.

### quality_predictions

`run`, `sample`, `quantity`, `predicted_error`, `observed_error`, `target_available`, `human_only_target`, `loss_sum`, `loss_count`, `validated_for_deployment`.

### quality_bins

`run`, `stage`, `endpoint`, `epoch`, `quantity`, `bin`, `count`, `predicted_error_sum`, `observed_error_sum`, `absolute_error_sum`, `median_baseline_error_sum`.

### batch_objectives

`run`, `update`, `epoch`, `geometry_objective`, `quality_objective`, `SSL_objective`, `variance_a`, `variance_b`, `covariance_a`, `covariance_b`, `texture_ramp`, `total_loss`, `geometry_samples`, `quality_mask_targets`, `quality_human_point_targets`, `SSL_pairs`, `texture_step_skipped`.

### train_updates

`run`, `update`, `epoch`, `next_draw`, `geometry_optimizer_steps`, `texture_optimizer_steps`, `quality_optimizer_steps`, `geometry_lr`, `texture_lr`, `quality_lr`, `gradient_finite`, `update_committed`, `unit_sha256`.

### parameter_gradients

`run`, `update`, `group`, `parameter`, `gradient_available`, `numel`, `l2_norm`, `max_absolute`, `mean_absolute`, `zero_fraction`, `finite`.

### group_gradient_summary

`run`, `update`, `group`, `pre_clip_norm`, `post_clip_norm`, `clip_coefficient`, `gradient_count`, `optimizer_step_applied`, `frozen_anchor_gradient_count`.

### geometry_drift

`run`, `sample`, `mask_logit_delta_max`, `mask_probability_delta_mean`, `point_delta_max_width_fraction`, `tyre_box_delta`, `tread_box_delta`, `anchor_weights_unchanged`.

### latent_diagnostics

`run`, `update`, `stage`, `eligible_images`, `components`, `projector_std_min`, `projector_std_median`, `dimensions_below_floor`, `covariance_offdiag_energy`, `effective_rank`, `embedding_norm_min`, `embedding_norm_max`, `finite`.

### epoch_metrics

`run`, `stage`, `endpoint`, `epoch`, `task`, `group`, `numerator`, `denominator`, `value`, `available`, `unit`, `aggregation_definition`.

### source_exposure

`run`, `epoch`, `image_id`, `geometry_draws`, `texture_views`, `padding_draws`, `SSL_valid_draws`, `SSL_skipped_draws`, `first_update`, `last_update`.

### pipeline_timings

`run`, `unit`, `stage`, `feature_seconds`, `transfer_seconds`, `geometry_seconds`, `texture_seconds`, `record_seconds`, `publish_wait_seconds`, `evaluation_seconds`.

### resource_samples

`run`, `unit`, `gpu0_allocated_bytes`, `gpu0_reserved_bytes`, `gpu1_allocated_bytes`, `gpu1_reserved_bytes`, `host_rss_bytes`, `working_free_bytes`, `logical_artifact_bytes`, `staging_bytes`.

### numeric_events

`run`, `event_id`, `unit`, `event_kind`, `stage`, `parameter_or_tensor`, `finite`, `action`, `replay_from_update`, `reason`.

### endpoint_neighbours

`run`, `sample`, `reference_id`, `reference_sha256`, `reference_component`, `reference_stage`, `cosine_similarity`, `rank`, `same_source_blocked`, `same_component_blocked`, `proxy_label_context_only`.

### spatial_maps_index

`run`, `sample`, `quantity`, `native_file`, `native_key`, `shape`, `dtype`, `coordinate_system`, `sha256`, `precision_note`.

### native_chunks_index

`run`, `file`, `kind`, `epoch`, `first_sample`, `last_sample`, `bytes`, `sha256`, `immutable`, `committed_unit`.

### selection

`run`, `epoch`, `validation_mean_dice`, `baseline_mean_dice`, `validation_human_point_MAE`, `baseline_human_point_MAE`, `invalid_atlas_fraction`, `baseline_invalid_atlas_fraction`, `projector_std_median`, `effective_rank`, `geometry_eligible`, `collapse_eligible`, `selection_objective`, `selected_so_far`, `rejection_reasons`.

### interventions

`run`, `sample`, `variant`, `base_sample`, `mask_dice_change`, `point_MAE_change`, `embedding_cosine_change`, `quality_error_change`, `valid_fraction_change`, `scope`.

### dense_pixels

`sample`, `region`, `row`, `column`, `probability`, `predicted`, `target_foreground_subpixels`, `valid_subpixels`.

### unit_ledger

`run`, `unit`, `kind`, `update`, `csv_prefix_sha256`, `native_state_sha256`, `previous_unit_sha256`, `current_unit_sha256`.

### HF_requests

`run`, `event_id`, `category`, `method`, `http_status`, `retry_delay_seconds`, `rate_remaining`, `reset_seconds`, `bytes_transferred`, `outcome`.

### HF_publications

`run`, `attempt`, `reason`, `local_update`, `remote_update`, `revision`, `manifest_sha256`, `new_objects`, `verified_objects`, `elapsed_seconds`, `owner_released`, `pending`.

### environment

`run`, `record_id`, `python`, `torch`, `cuda`, `cudnn`, `numpy`, `pillow`, `HF_SDK`, `devices`, `free_output_bytes`, `source_inventory_sha256`, `dataset_checksum`, `anchor_sha256`.

### gate_checks

`run`, `check`, `stage`, `passed`, `observed`, `required`, `evidence_path`, `fixture_or_measured`.

## Explicit unavailable measurements

No new independent tyres, physical tread depth, safety truth, crack/damage annotations, causal wear explanations or full three-class independent confirmation are created. Full grids/RGB are reconstructible rather than duplicated per observation. No per-update permanent optimizer archive or new classification accuracy result is claimed. Exact wall time/storage/performance remain pending gates. Any further omission discovered during implementation must be recorded before the measured protocol is frozen.

## Arithmetic planning check

[BUDGET.json](BUDGET.json) records all formulas and provisional row-width allowances. Across train/validation/selected+final identity stages, the maximum is 29,520 source passes and 59,040 texture views before optional controls/probes. Native arrays contribute 8,887,173,120 bytes; provisional other outputs give 12,191,728,000 bytes logical before a 1.2 factor. The adjusted estimate is 14,630,073,600 bytes and same-filesystem staging projection 15,166,944,512 bytes. These pass the proposed 15 GB/18 GB ceilings arithmetically, but the 40/42 byte CSV row-width allowances are unmeasured and must be replaced by actual fixture writes. A full-copy staging path can fail even when this same-filesystem plan fits.
