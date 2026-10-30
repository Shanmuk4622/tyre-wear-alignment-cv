# ATLAS-LC r2 — direct CSV and native evidence dictionary

**Frozen schema:24 families.** Counts below are planned for the60epoch measured run, not completed outcomes. Source roles remain386/81/103. Engineering fixtures use four updates and two validation examples. More rows/views do not create additional independent tyres.

| Family | Planned rows | Join / observation key | Meaning |
|---|---:|---|---|
| `source_images` | 570 | `image_index` | All historical roles,full SHA,label,identity,bank/query flags; eligibility is not observed exposure |
| `bank_cases` | 18 | `prototype_index` | Literal training-photo provenance and atlas patch |
| `bank_pixels` | 221,184 | `prototype_index/channel/row/column` | Raw RGB float pixels; valid masks and LC can be reconstructed from native assets |
| `predictions` | 53,342 | `sample` | Both online views and every evaluation; source SHA and allowed-class counts |
| `sample_exposures` | 47,040 | `sample` | Actual sampled draw probability and view factors; A/B share one occurrence |
| `affinities` | 30,724,992 | `sample/patch/prototype` | All32×18 cosine and temperature-scaled affinities; includes blocked cases for audit |
| `patch_support` | 1,706,944 | `sample/patch` | Coverage and valid-band pooling mass |
| `patch_evidence` | 5,120,832 | `sample/patch/class_index` | Signed contribution after SHA suppression/scaling |
| `coefficients` | 170,578 | `parameter_update/kind/class_index/prototype` | Base signed weights,bias and temperature; update0 plus2,940 updates |
| `reference_embeddings` | 107,136 | `epoch/endpoint/prototype/dimension` | 96D embeddings at60validation and2endpoint starts |
| `train_updates` | 2,940 | `update` | Joint/pair loss components,LR,gradient/resource and before/after learner hashes |
| `parameter_gradients` | 64,680 | `update/parameter` | All22 trainable parameter tensors: preclip norm,mean,min,max,finite |
| `dense_pixels` | ≤14,008,320 | `image_index/channel/row/column` | Frozen quarter-resolution probabilities,once per observed image |
| `atlas_rows` | ≤291,840 | `image_index/atlas_row` | Predicted source coordinates/validity,once per observed image |
| `dense_regions` | ≤1,140 | `image_index/region` | Unignored counts,Dice,IoU and half-open mask-box IoU |
| `dense_points` | ≤3,420 | `image_index/point` | Guide-relative human/weak target comparability and width-fraction error |
| `metrics` | group dependent | `stage/endpoint/epoch/level/group/class_index/metric` | Cohort/domain/tyre/group metrics with denominators and explicit NA |
| `confusion` | group dependent | `stage/endpoint/epoch/level/group/target_class/predicted_class` | All3×3 counts including zeros |
| `calibration_bins` | group dependent | `stage/endpoint/epoch/level/group/bin` | 10equal-width bins with sufficient statistics |
| `selection` | 60 | `epoch` | Validation rule result,selected epoch,checkpoint/hash |
| `explanation_checks` | 618 | `sample/class_index` | Selected/final native image projection conservation |
| `case_rules` | 960,156 | `sample/prototype` | SHA match,allowed flag,reference class and6/n scale |
| `view_pairs` | 23,520 | `sample_a/sample_b` | Per-draw CE_A,CE_B,JS,texture,total,target/prediction disagreement |
| `view_patch_consistency` | 752,640 | `sample_a/sample_b/patch` | 32weighted query embedding distances per draw |

## Joins, denominators and update ownership

`run` joins the full CONTRACT hash. `sample` is a unique monotonic observation key: A/B are separate samples, paired in `view_pairs`; evaluation uses one sample per image/variant. `image_index` joins the fixed570-image registry. Every observation retains source SHA and role is looked up from that registry. Bank prototypes are ordered six per class. Training view rows use `parameter_update=update−1`; base coefficients at that state plus case_rules reconstruct signed evidence.

Training predictions precede optimization. The saved joint update loss averages draw-level paired totals. Do not average47,040 view rows and call them independent tyres. Online epoch metrics mix learner states; validation uses the completed epoch state. Missing mid recall in held-out sets remains NA. All raw probabilities support later calibration analysis; calibration is not fitted here.

## Every retained column

### `source_images`

`run`, `image_index`, `image_id`, `image_sha256`, `class_index`, `role`, `domain`, `physical_tyre_id`, `source_group`, `image_path`, `point_kind`, `is_bank`, `is_query`

### `bank_cases`

`run`, `prototype_index`, `class_index`, `image_index`, `source_image_id`, `source_tyre_id`, `atlas_patch_index`, `valid_fraction`, `clipped_fraction`, `review_policy`, `risk_note`

### `bank_pixels`

`run`, `prototype_index`, `channel`, `row`, `column`, `value`

### `predictions`

`run`, `sample`, `stage`, `endpoint`, `epoch`, `update`, `parameter_update`, `image_index`, `image_id`, `label`, `prediction`, `logit_0`, `logit_1`, `logit_2`, `probability_0`, `probability_1`, `probability_2`, `nll`, `brier`, `entropy`, `correct`, `evidence_residual`, `supported_patches`, `supported_bands`, `clipped_fraction`, `empty_support`, `variant`, `occluded_patch`, `replacement_r`, `replacement_g`, `replacement_b`, `learner_sha256`, `anchor_logit_0`, `anchor_logit_1`, `anchor_logit_2`, `anchor_probability_0`, `anchor_probability_1`, `anchor_probability_2`, `anchor_prediction`, `source_sha256`, `allowed_low`, `allowed_mid`, `allowed_high`

### `sample_exposures`

`run`, `sample`, `epoch`, `update`, `slot`, `occurrence`, `image_index`, `class_draw`, `domain_draw`, `source_group`, `draw_probability`, `augmentation_seed`, `brightness`, `contrast`, `gamma`, `view`

### `affinities`

`run`, `sample`, `patch`, `prototype`, `cosine`, `affinity`

### `patch_support`

`run`, `sample`, `patch`, `valid_fraction`, `pooling_mass`

### `patch_evidence`

`run`, `sample`, `patch`, `class_index`, `contribution`

### `coefficients`

`run`, `parameter_update`, `kind`, `class_index`, `prototype`, `value`

### `reference_embeddings`

`run`, `epoch`, `update`, `endpoint`, `prototype`, `dimension`, `value`, `learner_sha256`

### `train_updates`

`run`, `epoch`, `update`, `cursor`, `loss`, `lr`, `grad_norm_before_clip`, `grad_norm_after_clip`, `gradient_finite`, `trainable_parameters`, `anchor_gradient_count`, `before_learner_sha256`, `after_learner_sha256`, `compute_seconds`, `record_seconds`, `device`, `cuda_peak_allocated_bytes`, `cuda_peak_reserved_bytes`, `host_peak_rss_bytes`, `free_disk_bytes`, `ce_a`, `ce_b`, `js`, `texture`, `prediction_views`

### `parameter_gradients`

`run`, `epoch`, `update`, `parameter`, `elements`, `norm`, `mean`, `minimum`, `maximum`, `finite`

### `dense_pixels`

`run`, `image_index`, `channel`, `row`, `column`, `probability`

### `atlas_rows`

`run`, `image_index`, `atlas_row`, `source_y`, `source_left`, `source_right`, `valid_fraction`

### `dense_regions`

`run`, `image_index`, `region`, `predicted_pixels`, `target_pixels`, `intersection`, `union`, `valid_pixels`, `dice`, `iou`, `box_iou`, `pred_left`, `pred_top`, `pred_right_exclusive`, `pred_bottom_exclusive`, `truth_left`, `truth_top`, `truth_right_exclusive`, `truth_bottom_exclusive`

### `dense_points`

`run`, `image_index`, `point`, `predicted_x_fraction`, `target_x_fraction`, `point_kind`, `prediction_guide_y_fraction`, `target_guide_y_fraction`, `absolute_error_width_fraction`, `metric_available`, `unavailable_reason`

### `metrics`

`run`, `stage`, `endpoint`, `epoch`, `update`, `level`, `group`, `class_index`, `metric`, `value`, `n`, `unavailable_reason`

### `confusion`

`run`, `stage`, `endpoint`, `epoch`, `update`, `level`, `group`, `target_class`, `predicted_class`, `count`

### `calibration_bins`

`run`, `stage`, `endpoint`, `epoch`, `update`, `level`, `group`, `bin`, `lower`, `upper`, `count`, `correct`, `confidence_sum`, `accuracy`, `mean_confidence`

### `selection`

`run`, `epoch`, `update`, `validation_ba_observed`, `validation_nll`, `selected`, `selected_epoch`, `tie_break`, `checkpoint_artifact`, `learner_sha256`

### `explanation_checks`

`run`, `sample`, `epoch`, `update`, `endpoint`, `mass_input`, `mass_projected`, `bias`, `logit`, `residual`, `class_index`, `supported_pixels`, `unsupported_mass`

### `case_rules`

`run`, `sample`, `prototype`, `reference_class`, `allowed`, `same_source_sha256`, `normalization_scale`

### `view_pairs`

`run`, `epoch`, `update`, `image_index`, `sample_a`, `sample_b`, `ce_a`, `ce_b`, `js`, `texture`, `total`, `label`, `prediction_a`, `prediction_b`, `disagree`

### `view_patch_consistency`

`run`, `sample_a`, `sample_b`, `patch`, `pooling_mass`, `cosine_distance`

## Units and missing values

| Field category | Contract |
|---|---|
| run | 12-character prefix; CONTRACT.json maps it to the full immutable SHA-256 |
| sample | Monotonic observation key; joins predictions and dependent evidence rows |
| image_index | Zero-based fixed source registry key; image identity/hash retained in source_images |
| parameter_update | Model state after this many completed optimizer updates; training predictions use the preceding state |
| probability | dimensionless [0,1]; uncalibrated model score |
| contribution | signed class-logit units; bias is separate |
| cosine | dimensionless [-1,1] |
| affinity | exp((cosine-1)/temperature) |
| pooling_mass | dimensionless; valid bands have equal total mass, empty atlas has zero |
| absolute_error_width_fraction | fraction of resized image width; not physical depth/alignment |
| dice | (2*intersection+1)/(predicted_pixels+target_pixels+1) on unignored pixels |
| iou | (intersection+1)/(union+1) on unignored pixels |
| time | seconds; timing/resource observations are not expected to match across resumed executions |
| coordinate | Zero-based resized input (512 high,384 wide) or declared atlas/patch indices; boxes half-open |
| missing | Blank means unavailable/not applicable. Missing class recall is NA; observed zeros remain zero. |
| case_rules | allowed/same-source are0/1; scale6/n_allowed counts references of the reference class, not the query target |
| view losses | CE/JS are nats; texture is band-weighted cosine distance; total=.5(CE_A+CE_B)+.1JS+.05texture |
| appearance | brightness/contrast/gamma dimensionless; independent[.8,1.2]; source SHA refers to original file before augmentation |
| gradient | Per-tensor summaries are before joint gradient clipping; per-objective gradients and dense continuous training traces are not measured |
| resource | CUDA/host/disk in bytes; wall times in seconds; environment dependent,excluded only from deterministic-equivalence comparison |

## Native evidence and reproducibility

| Artifact | Retained content |
|---|---|
| state.pt and two rotating local states | Full model/fixed bank,optimizer,scheduler,disabled scaler,RNG,exact orders,next update/draw,partial metrics,CSV byte/row/native cursors |
| artifacts/epochs/*.pt | All60epoch model states,epoch/update,protocol and learner hash |
| artifacts/dense/*.pt | Deduplicated frozen dense masks,points/guides and source IDs; raw dense probabilities also CSV |
| artifacts/endpoint_maps/*/*.pt | Every selected/final atlas RGB/valid/grid,signed patch evidence,effective coefficients,reference embeddings,aggregate reference evidence,input-space signed maps and bias |
| source/assets/*.npy + prototype_bank.csv | Original reference RGB/valid/source32-byte SHA/labels,case/patch/native-source provenance and planned sampling probabilities |
| CONTRACT/FILES/recipe/environment/history/HTTP/lifecycle JSON | Source hashes,provenance,atomic-unit integrity,recovery/publication attempts,capacity/errors |

Augmented full RGB/feature tensors are not duplicated at every draw. They are recoverable from the original data SHA,frozen geometry/source and exact seed/factors; complete affinities and signed evidence are stored for both views. No invented physical labels,independent tyres,unsaved per-objective gradients or continuous energy/GPU utilization traces are claimed.

Native tensor weights/maps are directPT/NPY; scientific tables are directCSV. No ZIP,NPZ or database upload. See [architecture](../../phase3TreadTwoLCArchitecture.md) and [protocol](../../phase3TreadTwoLCProtocol.md).
