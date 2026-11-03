# ATLAS-CONTEXT r3 — direct CSV and native evidence dictionary

<!-- NB14_COMPLETED_20261101 -->
## Completed context review — 1 November 2026

The NB13 fit and NB14 review are complete and HF-verified. All 80 review files / 52 CSVs match, and the original fit is preserved. [NB14 completion](phase3NB14Completion.md). The classifier remains unpromoted. Next is the implemented [NB15 existing-data readiness / evidence passport](phase3TreadTwoReadiness.md), not another full fit. Earlier pending-NB14 statements below describe the prior delivery. The original architecture/protocol/schema are unchanged.
<!-- END_NB14_COMPLETED_20261101 -->

<!-- NB13_COMPLETED_20261101 -->
## Completed retention audit — 1 November 2026

The frozen planned schema below was retained exactly: **27 raw CSV families / 58,176,066 scientific rows**. Every published byte, schema and count was independently verified. [Completion proof](phase3NB13Completion.md), [raw audit](tread_two/nb13_completion/8bdf46ec2040/VERIFICATION.json). Actual counts:

| Family | Rows |
|---|---:|
| `source_images` | 570 |
| `bank_cases` | 18 |
| `bank_pixels` | 221,184 |
| `predictions` | 53,754 |
| `sample_exposures` | 47,040 |
| `affinities` | 30,962,304 |
| `patch_support` | 1,720,128 |
| `patch_evidence` | 5,160,384 |
| `coefficients` | 170,578 |
| `reference_embeddings` | 107,136 |
| `train_updates` | 2,940 |
| `parameter_gradients` | 144,060 |
| `dense_pixels` | 14,008,320 |
| `atlas_rows` | 291,840 |
| `dense_regions` | 1,140 |
| `dense_points` | 3,420 |
| `metrics` | 41,434 |
| `confusion` | 16,092 |
| `calibration_bins` | 17,880 |
| `selection` | 60 |
| `explanation_checks` | 618 |
| `case_rules` | 967,572 |
| `view_pairs` | 23,520 |
| `view_patch_consistency` | 752,640 |
| `context_evidence` | 1,720,128 |
| `context_summary` | 53,754 |
| `endpoint_attention` | 1,687,552 |

All 60 epoch weights and complete native recovery are retained directly on HF. Full selected/final attention is retained; full per-update/probe matrices were not saved and are not reconstructed from summaries. The separate [NB14 derived review](phase3TreadTwoContextReview.md) adds 52 CSVs / 585,825 rows without training or additional independent tyres. The pre-fit wording below documents the original frozen contract.
<!-- END_NB13_COMPLETED_20261101 -->

**Freeze before NB13 measured training.** More telemetry is a standing requirement. This revision keeps every prior scientific family and adds context corrections, attention summaries and complete endpoint attention. [Executable schema](tread_two/context_pilot/schemas.py) is the exact column authority. Tables are published directly on HF, never hidden inside ZIP/NPZ/database artifacts.

## Keys, units and planned observations

`run` is the contract's 12-character prefix; the full immutable SHA is in CONTRACT.json. `sample` joins predictions/support/affinities/corrections/attention; `image_index` joins the source registry and full source SHA. `parameter_update` identifies the completed model state used for an observation. Train predictions are pre-update. Epoch weights/native states supply the numeric model state. Timings/resources vary across recovery and are excluded only from deterministic-equality tests.

| Observation source | Count |
|---|---:|
| Training paired views | 47,040 |
| Per-epoch validation | 4,860 |
| Selected + final identity endpoint | 206 |
| Eight diagnostic variants × two endpoints × 103 images | 1,648 |
| **Total planned raw observations** | **53,754** |
| Paired draws | 23,520 |
| Endpoint attention entries: 206 × 2 layers × 4 heads × 32 queries × 32 keys | 1,687,552 |

Logits, signed evidence and corrections use logit units. Affinity/attention/probability/mass are dimensionless. Dense coordinates use declared resized image/atlas systems; boxes are half-open. Point errors are image-width fractions, not physical distance. Blank metrics denote unavailable denominators; zero remains zero. Mid recall in held-out partitions is unavailable.

## Families and complete columns


**27 families / 328 declared columns** (column names repeat across families).

### `source_images` — 13 columns

570-image identity, role, domain, group, hash, point kind and bank/query registry.

`run`, `image_index`, `image_id`, `image_sha256`, `class_index`, `role`, `domain`, `physical_tyre_id`, `source_group`, `image_path`, `point_kind`, `is_bank`, `is_query`

### `bank_cases` — 11 columns

Literal reference provenance and risk/review policy.

`run`, `prototype_index`, `class_index`, `image_index`, `source_image_id`, `source_tyre_id`, `atlas_patch_index`, `valid_fraction`, `clipped_fraction`, `review_policy`, `risk_note`

### `bank_pixels` — 6 columns

Every reference RGB value, long format.

`run`, `prototype_index`, `channel`, `row`, `column`, `value`

### `predictions` — 43 columns

Every observation: three logits/scores, loss, uncertainty, prior anchor output, identity, support, variant, checkpoint and permissions.

`run`, `sample`, `stage`, `endpoint`, `epoch`, `update`, `parameter_update`, `image_index`, `image_id`, `label`, `prediction`, `logit_0`, `logit_1`, `logit_2`, `probability_0`, `probability_1`, `probability_2`, `nll`, `brier`, `entropy`, `correct`, `evidence_residual`, `supported_patches`, `supported_bands`, `clipped_fraction`, `empty_support`, `variant`, `occluded_patch`, `replacement_r`, `replacement_g`, `replacement_b`, `learner_sha256`, `anchor_logit_0`, `anchor_logit_1`, `anchor_logit_2`, `anchor_probability_0`, `anchor_probability_1`, `anchor_probability_2`, `anchor_prediction`, `source_sha256`, `allowed_low`, `allowed_mid`, `allowed_high`

### `sample_exposures` — 16 columns

Every sampled view: exact probability, factors, seed, group and occurrence; distinguish paired views from draws.

`run`, `sample`, `epoch`, `update`, `slot`, `occurrence`, `image_index`, `class_draw`, `domain_draw`, `source_group`, `draw_probability`, `augmentation_seed`, `brightness`, `contrast`, `gamma`, `view`

### `affinities` — 6 columns

All 32×18 patch/reference cosines and affinities per observation, including blocked references; join case_rules before accounting.

`run`, `sample`, `patch`, `prototype`, `cosine`, `affinity`

### `patch_support` — 5 columns

Coverage and exact pooling mass of each tile.

`run`, `sample`, `patch`, `valid_fraction`, `pooling_mass`

### `patch_evidence` — 5 columns

Combined frozen-reference plus context signed class contributions, 32×3 per observation.

`run`, `sample`, `patch`, `class_index`, `contribution`

### `coefficients` — 6 columns

Frozen signed reference coefficients/temperature plus combined base+context bias at recorded parameter updates.

`run`, `parameter_update`, `kind`, `class_index`, `prototype`, `value`

### `reference_embeddings` — 8 columns

Full frozen reference embeddings at retained evaluation/update endpoints.

`run`, `epoch`, `update`, `endpoint`, `prototype`, `dimension`, `value`, `learner_sha256`

### `train_updates` — 26 columns

Loss components, lr, clipping/finite status, frozen-base gradient count, learned-state hashes and bounded timing/resource counters.

`run`, `epoch`, `update`, `cursor`, `loss`, `lr`, `grad_norm_before_clip`, `grad_norm_after_clip`, `gradient_finite`, `trainable_parameters`, `anchor_gradient_count`, `before_learner_sha256`, `after_learner_sha256`, `compute_seconds`, `record_seconds`, `device`, `cuda_peak_allocated_bytes`, `cuda_peak_reserved_bytes`, `host_peak_rss_bytes`, `free_disk_bytes`, `ce_a`, `ce_b`, `js`, `texture`, `prediction_views`, `residual_l2`

### `parameter_gradients` — 10 columns

Every trainable tensor: count/norm/mean/min/max/finite; not full gradient tensors.

`run`, `epoch`, `update`, `parameter`, `elements`, `norm`, `mean`, `minimum`, `maximum`, `finite`

### `dense_pixels` — 6 columns

Frozen dense pixel probabilities on the declared audit grid; native full maps retained.

`run`, `image_index`, `channel`, `row`, `column`, `probability`

### `atlas_rows` — 7 columns

Native-to-atlas geometry/support provenance.

`run`, `image_index`, `atlas_row`, `source_y`, `source_left`, `source_right`, `valid_fraction`

### `dense_regions` — 19 columns

Mask/box intersections, counts, coordinates and metrics.

`run`, `image_index`, `region`, `predicted_pixels`, `target_pixels`, `intersection`, `union`, `valid_pixels`, `dice`, `iou`, `box_iou`, `pred_left`, `pred_top`, `pred_right_exclusive`, `pred_bottom_exclusive`, `truth_left`, `truth_top`, `truth_right_exclusive`, `truth_bottom_exclusive`

### `dense_points` — 11 columns

Point predictions/truth/kind, errors and unavailable reasons.

`run`, `image_index`, `point`, `predicted_x_fraction`, `target_x_fraction`, `point_kind`, `prediction_guide_y_fraction`, `target_guide_y_fraction`, `absolute_error_width_fraction`, `metric_available`, `unavailable_reason`

### `metrics` — 12 columns

Cohort/domain/group/class summaries with denominators and unavailable reasons.

`run`, `stage`, `endpoint`, `epoch`, `update`, `level`, `group`, `class_index`, `metric`, `value`, `n`, `unavailable_reason`

### `confusion` — 10 columns

Complete class confusion counts at reported levels.

`run`, `stage`, `endpoint`, `epoch`, `update`, `level`, `group`, `target_class`, `predicted_class`, `count`

### `calibration_bins` — 15 columns

Ten equal-width confidence bins with sufficient statistics.

`run`, `stage`, `endpoint`, `epoch`, `update`, `level`, `group`, `bin`, `lower`, `upper`, `count`, `correct`, `confidence_sum`, `accuracy`, `mean_confidence`

### `selection` — 10 columns

Every validation checkpoint decision, BA/NLL/tie rule and native artifact.

`run`, `epoch`, `update`, `validation_ba_observed`, `validation_nll`, `selected`, `selected_epoch`, `tie_break`, `checkpoint_artifact`, `learner_sha256`

### `explanation_checks` — 13 columns

Selected/final class-wise signed mass conservation and unsupported mass.

`run`, `sample`, `epoch`, `update`, `endpoint`, `mass_input`, `mass_projected`, `bias`, `logit`, `residual`, `class_index`, `supported_pixels`, `unsupported_mass`

### `case_rules` — 7 columns

All 18 full-source permissions/reference-class normalization scales per observation.

`run`, `sample`, `prototype`, `reference_class`, `allowed`, `same_source_sha256`, `normalization_scale`

### `view_pairs` — 16 columns

Per-draw pair loss components, disagreement and both observation keys; includes residual_l2.

`run`, `epoch`, `update`, `image_index`, `sample_a`, `sample_b`, `ce_a`, `ce_b`, `js`, `texture`, `total`, `label`, `prediction_a`, `prediction_b`, `disagree`, `residual_l2`

### `view_patch_consistency` — 6 columns

Each pair/tile contextual cosine distance and pooling mass.

`run`, `sample_a`, `sample_b`, `patch`, `pooling_mass`, `cosine_distance`

### `context_evidence` — 7 columns

Every observation/tile: source-patch mapping and three weighted spatial logit corrections.

`run`, `sample`, `patch`, `source_patch`, `delta_0`, `delta_1`, `delta_2`

### `context_summary` — 27 columns

Every observation: base prediction/NLL/three logits, three bias corrections, total correction squared and eight attention entropy/max pairs.

`run`, `sample`, `base_prediction`, `base_nll`, `correction_l2`, `base_logit_0`, `base_logit_1`, `base_logit_2`, `bias_delta_0`, `bias_delta_1`, `bias_delta_2`, `entropy_0_0`, `max_attention_0_0`, `entropy_0_1`, `max_attention_0_1`, `entropy_0_2`, `max_attention_0_2`, `entropy_0_3`, `max_attention_0_3`, `entropy_1_0`, `max_attention_1_0`, `entropy_1_1`, `max_attention_1_1`, `entropy_1_2`, `max_attention_1_2`, `entropy_1_3`, `max_attention_1_3`

### `endpoint_attention` — 7 columns

Full selected/final identity attention: layer/head/query/key/value, including unsupported query rows; normalized valid-key support.

`run`, `sample`, `layer`, `head`, `query`, `key`, `value`


## Reconstructing the added measurements

Join each context_evidence row to predictions/sample and patch_support. Deltas already include pooling mass; do not multiply by mass again. Combined patch_evidence minus context_evidence gives the frozen case-branch contribution. Subtract context_summary.bias_delta from coefficients' combined bias to recover base bias. Sum patches plus combined bias to reproduce each logit. The independently checked engineering maximum residual is 8.28×10⁻⁸.

`correction_l2` is mean over three classes of `(sum_patch(delta)+bias_delta)²`. A pair's `residual_l2` is the mean of its two view values. Add `0.02*residual_l2` to paired CE/JS/contextual-distance terms. Context attention matrices have shape [2,4,32,32]. Summaries weight each query row by pooling mass; entropy is divided by `log(max(2, valid_key_count))`. Empty support has zero weighted summaries. Full matrices include all query rows even if their output has zero pooling mass.

## Native state, visuals and explicit limits

| Retained directly | Purpose |
|---|---|
| Native completed-update state | Exact model/optimizer/scheduler/scaler/RNG/order/cursor/partial-stat recovery |
| All 60 native epoch weights | Preserve full training trajectory without refitting |
| Selected/final native image captures | Base/context embeddings, affinities, combined evidence, correction, bias, full attention, atlas grid, original dense outputs and signed projections |
| Immutable JSON/source manifests | Configuration, source/data/environment/recipe IDs, schemas, original identities, object hashes and row cursors |
| Lifecycle/HTTP records | Stop/errors, retry/backoff, publication/ownership/revisions, pending progress |
| PNG + interactive HTML | Curves; endpoint/error/source filtering; literal reference cards; signed 32-tile evidence and explicit frozen-base/context accounting |

**Declared gaps:** no full per-update attention matrices, no full per-update activations/embeddings/gradient tensors, no complete profiler trace or power telemetry, no genuine physical wear/depth/alignment/safety targets, no new independent tyre cohort and no held-out mid targets. Attention summaries are retained every observation; full attention is retained at both identity endpoints. Engineering fixture resource readings are not measured-fit timings. Future derived summaries must not invent unsaved values.

The reporter only reads retained state and marks engineering examples prominently. Reference thumbnails explain the frozen case branch; context evidence must accompany any complete prediction explanation. Conservation does not establish causality or clinical/physical usefulness. All data/protocol/cohort caveats remain attached to exported evidence.

[CSV-only verification](tread_two/context_validation/CSV_ACCOUNTING.json) · [Storage policy](phase3TreadTwoContextProtocol.md) · [Run instructions](tread_two/context_pilot/README.md).
