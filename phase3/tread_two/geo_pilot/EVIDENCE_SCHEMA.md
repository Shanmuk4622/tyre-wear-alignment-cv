# GEO executable evidence schema and retention

**40 families /468 fields before measured freeze.** The original443-field proposal
is preserved. The executable contract corrects14 draft types and adds25 fields
for disaggregated drift, complete SSL accounting, masking policy, state digest,
quota/time metadata, timing, box/gap availability and structural-probe mapping.
HF event/publication identities are replay-safe text IDs across fresh sessions;
training/sample/unit IDs remain integers and seeds retain uint64 precision.

CSV bools are0/1, missing values are empty with availability/count/reason context,
raw FP32 vectors/probabilities use9significant digits, derived FP64 uses17.
The actual full specification is embedded in CONTRACT.json.schema and emitted
for local inspection in geo_validation/EXECUTABLE_SCHEMA.json.

## All families and actual columns

| Family | Cadence | Columns |
|---|---|---|
| `HF_publications` | every publication attempt | `run` (utf8), `attempt` (utf8), `reason` (utf8), `local_update` (int64), `remote_update` (int64), `revision` (utf8), `manifest_sha256` (utf8), `new_objects` (int64), `verified_objects` (int64), `elapsed_seconds` (float64), `owner_released` (bool), `pending` (bool) |
| `HF_requests` | every request category/event; no secrets | `run` (utf8), `event_id` (utf8), `category` (utf8), `method` (utf8), `http_status` (int64), `retry_delay_seconds` (float64), `rate_remaining` (int64), `reset_seconds` (float64), `bytes_transferred` (float64), `outcome` (utf8), `server_quota` (int64), `server_window_seconds` (int64), `time_utc` (utf8) |
| `annotation_targets` | once | `run` (utf8), `image_id` (utf8), `guide` (int64), `side` (int64), `x_width_fraction` (float64), `y_height_fraction` (float64), `target_kind` (utf8), `optimization_weight` (float64), `human_eligible` (bool), `metric_available` (bool) |
| `atlas_bands` | every sample and8bands | `run` (utf8), `sample` (int64), `band` (int64), `valid_rows` (int64), `valid_fraction` (float64), `mean_width_width_fraction` (float64), `clipped_fraction` (float64), `point_mask_gap_width_fraction` (float64), `unsupported` (bool), `point_mask_gap_count` (int64), `point_mask_gap_exceeds_flag` (bool) |
| `atlas_rows` | selected/final source cohorts; all training row parameters additionally native | `run` (utf8), `sample` (int64), `atlas_row` (int64), `source_y` (float64), `left_x` (float64), `right_x` (float64), `row_valid` (bool), `source_clipped` (bool), `grid_parameter_native_key` (utf8) |
| `batch_objectives` | every completed update | `run` (utf8), `update` (int64), `epoch` (int64), `geometry_objective` (float64), `quality_objective` (float64), `SSL_objective` (float64), `variance_a` (float64), `variance_b` (float64), `covariance_a` (float64), `covariance_b` (float64), `texture_ramp` (float64), `total_loss` (float64), `geometry_samples` (int64), `quality_mask_targets` (int64), `quality_human_point_targets` (int64), `SSL_pairs` (int64), `texture_step_skipped` (bool), `invariance` (float64), `masked_huber_sum` (float64), `masked_huber_count` (float64), `mean_predictor_huber_sum` (float64), `mean_predictor_huber_count` (float64) |
| `dense_pixels` | selected/final identity cohorts only; run identity bound by file manifest; each row is one 4x4report block; all logits additionally native | `sample` (int64), `region` (int64), `row` (int64), `column` (int64), `probability` (float32), `predicted` (bool), `target_foreground_subpixels` (int64), `valid_subpixels` (int64) |
| `endpoint_neighbours` | selected/final diagnostic retrieval | `run` (utf8), `sample` (int64), `reference_id` (utf8), `reference_sha256` (utf8), `reference_component` (utf8), `reference_stage` (utf8), `cosine_similarity` (float64), `rank` (int64), `same_source_blocked` (bool), `same_component_blocked` (bool), `proxy_label_context_only` (bool) |
| `environment` | once and recovery attempts | `run` (utf8), `record_id` (int64), `python` (utf8), `torch` (utf8), `cuda` (utf8), `cudnn` (utf8), `numpy` (utf8), `pillow` (utf8), `HF_SDK` (utf8), `devices` (utf8), `free_output_bytes` (float64), `source_inventory_sha256` (utf8), `dataset_checksum` (utf8), `anchor_sha256` (utf8) |
| `epoch_metrics` | every epoch and endpoint task/group | `run` (utf8), `stage` (utf8), `endpoint` (utf8), `epoch` (int64), `task` (utf8), `group` (utf8), `numerator` (float64), `denominator` (float64), `value` (float64), `available` (bool), `unit` (utf8), `aggregation_definition` (utf8) |
| `epoch_schedule` | every epoch before updates | `run` (utf8), `epoch` (int64), `draw` (int64), `batch` (int64), `batch_slot` (int64), `image_id` (utf8), `source_sha256` (utf8), `leakage_component` (utf8), `is_padding` (bool), `base_augmentation_seed` (uint64), `view_a_seed` (uint64), `view_b_seed` (uint64), `mask_a_seed` (uint64), `mask_b_seed` (uint64), `schedule_sha256` (utf8) |
| `gate_checks` | preflight and final audit | `run` (utf8), `check` (utf8), `stage` (utf8), `passed` (bool), `observed` (utf8), `required` (utf8), `evidence_path` (utf8), `fixture_or_measured` (utf8) |
| `geometry_drift` | every sample against same-view anchor | `run` (utf8), `sample` (int64), `mask_logit_delta_max` (float64), `mask_probability_delta_mean` (float64), `point_delta_max_width_fraction` (float64), `tyre_box_delta` (float64), `tread_box_delta` (float64), `anchor_weights_unchanged` (bool), `anchor_tyre_box_available` (bool), `anchor_tread_box_available` (bool) |
| `geometry_losses` | every sample; unreduced sufficient statistics | `run` (utf8), `sample` (int64), `bce_sum` (float64), `bce_count` (float64), `dice_loss_sum` (float64), `dice_region_count` (int64), `point_loss_sum` (float64), `point_weight_count` (float64), `containment_sum` (float64), `containment_count` (float64), `row_agreement_sum` (float64), `row_agreement_count` (float64), `anchor_drift_sum` (float64), `anchor_drift_count` (float64), `anchor_mask_sum` (float64), `anchor_mask_count` (int64), `anchor_point_sum` (float64), `anchor_point_count` (int64) |
| `global_embeddings` | every eligible view, all 96coordinates | `run` (utf8), `sample` (int64), `view` (int64), `dimension` (int64), `value` (float32) |
| `group_gradient_summary` | every group/update | `run` (utf8), `update` (int64), `group` (utf8), `pre_clip_norm` (float64), `post_clip_norm` (float64), `clip_coefficient` (float64), `gradient_count` (int64), `optimizer_step_applied` (bool), `frozen_anchor_gradient_count` (int64) |
| `interventions` | selected/final predeclared probes | `run` (utf8), `sample` (int64), `variant` (utf8), `base_sample` (int64), `mask_dice_change` (float64), `point_MAE_change` (float64), `embedding_cosine_change` (float64), `quality_error_change` (float64), `valid_fraction_change` (float64), `scope` (utf8), `probe_available` (bool), `source_patch` (int64), `source_patch_mapping` (utf8), `unavailable_reason` (utf8) |
| `latent_diagnostics` | every batch and validation epoch | `run` (utf8), `update` (int64), `stage` (utf8), `eligible_images` (int64), `components` (int64), `projector_std_min` (float64), `projector_std_median` (float64), `dimensions_below_floor` (int64), `covariance_offdiag_energy` (float64), `effective_rank` (float64), `embedding_norm_min` (float64), `embedding_norm_max` (float64), `finite` (bool) |
| `masked_patch_metrics` | every masked tile | `run` (utf8), `sample` (int64), `view` (int64), `patch` (int64), `huber_sum` (float64), `huber_weight_count` (float64), `absolute_error_sum` (float64), `absolute_error_max` (float64), `mean_predictor_huber_sum` (float64), `mean_predictor_weight_count` (float64) |
| `masked_patch_plan` | every eligible texture view and masked tile | `run` (utf8), `sample` (int64), `view` (int64), `patch` (int64), `masked` (bool), `valid_loss_pixels` (int64), `mask_weight_sum` (float64), `target_mean` (float64), `target_std` (float64), `native_prediction_key` (utf8), `native_target_key` (utf8), `native_weight_key` (utf8) |
| `native_chunks_index` | each closed epoch or snapshot | `run` (utf8), `file` (utf8), `kind` (utf8), `epoch` (int64), `first_sample` (int64), `last_sample` (int64), `bytes` (int64), `sha256` (utf8), `immutable` (bool), `committed_unit` (int64) |
| `numeric_events` | only when event occurs; empty table valid | `run` (utf8), `event_id` (int64), `unit` (int64), `event_kind` (utf8), `stage` (utf8), `parameter_or_tensor` (utf8), `finite` (bool), `action` (utf8), `replay_from_update` (int64), `reason` (utf8) |
| `pair_losses` | every eligible paired draw | `run` (utf8), `sample` (int64), `invariance` (float64), `masked_huber_a` (float64), `masked_huber_b` (float64), `pair_cosine_distance` (float64), `valid_images_in_batch` (int64), `valid_components_in_batch` (int64) |
| `parameter_gradients` | every trainable tensor/update | `run` (utf8), `update` (int64), `group` (utf8), `parameter` (utf8), `gradient_available` (bool), `numel` (int64), `l2_norm` (float64), `max_absolute` (float64), `mean_absolute` (float64), `zero_fraction` (float64), `finite` (bool) |
| `patch_support` | every sample and32tiles | `run` (utf8), `sample` (int64), `patch` (int64), `band` (int64), `column` (int64), `valid_fraction` (float64), `pooling_mass` (float64), `valid_target_fraction` (float64), `source_row_start` (int64), `source_row_end` (int64) |
| `pipeline_timings` | every update/evaluation unit | `run` (utf8), `unit` (int64), `stage` (utf8), `feature_seconds` (float64), `transfer_seconds` (float64), `geometry_seconds` (float64), `texture_seconds` (float64), `record_seconds` (float64), `publish_wait_seconds` (float64), `evaluation_seconds` (float64), `data_loading_seconds` (float64), `backward_optimizer_seconds` (float64), `previous_commit_seconds` (float64) |
| `point_values` | every sample and6 points | `run` (utf8), `sample` (int64), `guide` (int64), `side` (int64), `prediction` (float64), `target` (float64), `absolute_error_width_fraction` (float64), `smooth_l1_sum` (float64), `target_kind` (utf8), `optimization_weight` (float64), `human_metric_available` (bool) |
| `projected_embeddings` | every eligible view, all 64coordinates | `run` (utf8), `sample` (int64), `view` (int64), `dimension` (int64), `value` (float32) |
| `quality_bins` | each validation epoch and endpoints | `run` (utf8), `stage` (utf8), `endpoint` (utf8), `epoch` (int64), `quantity` (utf8), `bin` (int64), `count` (int64), `predicted_error_sum` (float64), `observed_error_sum` (float64), `absolute_error_sum` (float64), `median_baseline_error_sum` (float64) |
| `quality_predictions` | every sample and3error outputs | `run` (utf8), `sample` (int64), `quantity` (utf8), `predicted_error` (float64), `observed_error` (float64), `target_available` (bool), `human_only_target` (bool), `loss_sum` (float64), `loss_count` (float64), `validated_for_deployment` (bool) |
| `region_metrics` | every sample and2regions | `run` (utf8), `sample` (int64), `region` (int64), `tp` (int64), `fp` (int64), `fn` (int64), `valid_pixels` (int64), `target_pixels` (int64), `predicted_pixels` (int64), `dice` (float64), `iou` (float64), `box_iou` (float64), `target_box_available` (bool), `predicted_box_available` (bool), `predicted_box_left` (int64), `predicted_box_top` (int64), `predicted_box_right` (int64), `predicted_box_bottom` (int64), `target_box_left` (int64), `target_box_top` (int64), `target_box_right` (int64), `target_box_bottom` (int64), `bce_sum` (float64), `bce_count` (float64) |
| `resource_samples` | every update and publication boundary | `run` (utf8), `unit` (int64), `gpu0_allocated_bytes` (float64), `gpu0_reserved_bytes` (float64), `gpu1_allocated_bytes` (float64), `gpu1_reserved_bytes` (float64), `host_rss_bytes` (float64), `working_free_bytes` (float64), `logical_artifact_bytes` (float64), `staging_bytes` (float64) |
| `sample_observations` | every train/evaluation source pass | `run` (utf8), `sample` (int64), `stage` (utf8), `endpoint` (utf8), `epoch` (int64), `completed_update` (int64), `parameter_update` (int64), `image_id` (utf8), `source_sha256` (utf8), `geometry_valid` (bool), `SSL_valid` (bool), `SSL_skip_reason` (utf8), `guide_count` (int64), `mask_valid_pixels` (int64), `point_target_kind` (utf8) |
| `selection` | each validation epoch | `run` (utf8), `epoch` (int64), `validation_mean_dice` (float64), `baseline_mean_dice` (float64), `validation_human_point_MAE` (float64), `baseline_human_point_MAE` (float64), `invalid_atlas_fraction` (float64), `baseline_invalid_atlas_fraction` (float64), `projector_std_median` (float64), `effective_rank` (float64), `geometry_eligible` (bool), `collapse_eligible` (bool), `selection_objective` (float64), `selected_so_far` (bool), `rejection_reasons` (utf8) |
| `source_exposure` | each epoch per original training image | `run` (utf8), `epoch` (int64), `image_id` (utf8), `geometry_draws` (int64), `texture_views` (int64), `padding_draws` (int64), `SSL_valid_draws` (int64), `SSL_skipped_draws` (int64), `first_update` (int64), `last_update` (int64) |
| `source_images` | once | `run` (utf8), `image_id` (utf8), `image_sha256` (utf8), `role` (utf8), `domain` (utf8), `confirmed_tyre_id` (utf8), `source_group` (utf8), `leakage_component` (utf8), `label_proxy` (utf8), `point_kind` (utf8), `human_target_available` (bool), `weak_target_available` (bool), `mask_sha256` (utf8), `ignore_sha256` (utf8) |
| `spatial_maps_index` | every retained native map chunk | `run` (utf8), `sample` (int64), `quantity` (utf8), `native_file` (utf8), `native_key` (utf8), `shape` (utf8), `dtype` (utf8), `coordinate_system` (utf8), `sha256` (utf8), `precision_note` (utf8) |
| `train_updates` | every completed update | `run` (utf8), `update` (int64), `epoch` (int64), `next_draw` (int64), `geometry_optimizer_steps` (int64), `texture_optimizer_steps` (int64), `quality_optimizer_steps` (int64), `geometry_lr` (float64), `texture_lr` (float64), `quality_lr` (float64), `gradient_finite` (bool), `update_committed` (bool), `unit_sha256` (utf8) |
| `unit_ledger` | every complete atomic unit | `run` (utf8), `unit` (int64), `kind` (utf8), `update` (int64), `csv_prefix_sha256` (utf8), `native_state_sha256` (utf8), `previous_unit_sha256` (utf8), `current_unit_sha256` (utf8), `state_semantic_sha256` (utf8) |
| `view_plans` | two texture views per eligible sample | `run` (utf8), `sample` (int64), `view` (int64), `brightness` (float64), `contrast` (float64), `gamma` (float64), `geometry_augmentation_seed` (uint64), `texture_seed` (uint64), `mask_seed` (uint64), `masked_tiles` (int64), `visible_tiles` (int64), `masked_target_pixels` (int64), `masking_policy` (utf8) |

## Native bodies and reconstruction

All train/validation/identity endpoint source passes retain FP32 two-mask logits,
six points, three error estimates, atlas row parameters/validity, both views'
pre/post-context32×96 tokens and up-to8masked4×16×16predictions/targets/weights.
Padded slots are explicitly absent in the mask plan, never fake observations.
Original RGB/annotation data is read-only on Kaggle, not duplicated per draw.

spatial_maps_index identifies a logical stream and global sample key. Resolve
that key through native_chunks_index's first/last sample bounds and NPY header.
For active unsealed streams, native state holds exact shape/dtype/first sample/
byte prefix/hash; the snapshot includes only that committed raw prefix. The
complete native state is CPU-readable and requires no model forward. Structural
probe mappings additionally define band/column permutations; they are not new
monotonic geometry warps.

Selected/final endpoints also retain dense CSV: each128×96cell stores probability,
prediction and original4×4foreground/valid subpixel counts. These exactly rebuild
full512×384TP/FP/FN. Boxes enclose native valid masks in half-open pixel coordinates.
All human/weak/missing point targets and target availability remain distinct.

The probe allowance is deliberately narrower: retain full geometry native,
points/error estimates/row parameters and full scalar/vector/mapping CSV, but
not probe texture token/prediction/target native bodies. The native keys for
those unsaved probe bodies are blank. Training/validation/identity endpoint
texture bodies are fully retained. This boundary is fixed before fitting.

## Loss/ledger semantics

Geometry CSV contains independent mask and point anchor-drift sums/counts because
their denominators differ. Batch VIC variance/covariance is recorded once per
batch; each64D projector coordinate permits independent recomputation. Masked
Huber preserves raw sums and channel-weight counts; the mean predictor is
evaluated under identical masks. Quality targets are detached; human-point
availability never includes weak proposals. Every trainable tensor and each
group's pre/post clip norm/count/step state is retained.

Sample predictions record the parameter update that actually generated them,
separately from the global completed training update. A selected epoch 20 endpoint
therefore records parameter_update980 and completed_update2940, rather than
mislabeling its weights as final. Controls are Geo epoch 0. Native chunk epoch
metadata follows the observation's checkpoint/epoch, not the next cursor.

unit_ledger.native_state_sha256/state_semantic_sha256 identifies the semantic
model/optimizer/RNG/cursor content. The selecting POINTER separately hashes the
actual serialized PT bytes. A self-referential file hash is not fabricated.
The ledger's kind describes the executed operation; the state cursor describes
the next operation. Transport batches are acknowledged inside native state so
replayed queue files do not duplicate records.

## Explicit measurement boundaries

Pipeline record_seconds is whole observed-unit wall time before native commit.
Feature time includes frozen inference/transfer, so separate transfer_seconds is
unavailable rather than invented. Geometry time includes refinements/warp/quality;
texture time covers paired forward paths; backward_optimizer_seconds is separate.
Previous commit time is lagged and unavailable immediately after restore. For
interventions the component timers describe the reference forward; whole-unit
time includes the actual intervention. Current allocated/reserved memory is
sampled per unit; these samples do not claim every transient micro-peak.

Transport CSV captures safe HTTP policy/status/time and publication receipts;
session-scoped request IDs and receipt-scoped publication IDs avoid fresh-session
counter collisions. The final commit's own acknowledgement is in the HF
PUBLICATIONS marker and local result/transport log; it cannot be recursively
included in the very manifest it acknowledges. Failed-network bytes not uploaded
remain local. No credentials/alternate token files are recorded.

Store rolling state after every complete update, keep all 60epoch weights and
selected/final exports, retain every verified30-minute/major/stop HF snapshot.
Do not claim2,940permanent full optimizer copies: the two rolling states and
published snapshots have that role, while scientific observations are permanent.
