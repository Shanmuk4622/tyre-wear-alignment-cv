"""Explicit per-column CSV dictionary, with units, keys and missing-value rules."""
import csv
import sys
from pathlib import Path
assert Path(sys.prefix).name == 'cv_conda'
PILOT = Path(__file__).resolve().parents[1] / 'tread_two/pilot'
sys.path.insert(0, str(PILOT))
from schemas import SCHEMAS

D = {}
def fields(names, dtype, unit, definition):
    for name in names.split():
        D[name] = (dtype, unit, definition)

fields('run', 'string', 'identifier', '12-character prefix of the full SHA-256 contract; full identity is in CONTRACT.json.')
fields('sample', 'integer', 'observation key', 'Monotonically increasing query-observation ID, shared by predictions and all associated patch/evidence rows.')
fields('image_index', 'integer', 'registry index', 'Zero-based fixed source registry index; joins image ID, original path, role and source SHA-256.')
fields('image_id source_image_id', 'string', 'identifier', 'Original immutable prepared-release image ID; source_image_id identifies a reference case input.')
fields('image_sha256', 'string', 'SHA-256', 'SHA-256 of original source image bytes; not the resized tensor.')
fields('learner_sha256 before_learner_sha256 after_learner_sha256', 'string', 'SHA-256', 'Ordered names and raw FP32 values of trainable parameters. Anchor/bank/source identities are bound separately by the full contract.')
fields('image_path checkpoint_artifact', 'string', 'relative path', 'Prepared-release image path, or immutable epoch checkpoint path relative to artifacts/.')
fields('physical_tyre_id source_tyre_id', 'string', 'known subject identifier', 'Confirmed original physical tyre ID. Missing identity remains blank; clip identities are not invented.')
fields('source_group group', 'string', 'group identifier', 'Declared aggregation/draw group; source_group is known tyre ID or unresolved clip ID, not a new independent tyre.')
fields('role', 'enum', 'data role', 'Frozen historical train, validation or test role; all unresolved clips remain train.')
fields('domain domain_draw', 'enum', 'source domain', 'Original photo or existing video-frame source domain; domain_draw records the sampler route.')
fields('class_index class_draw label prediction predicted_class target_class', 'integer', 'proxy class', '0=low, 1=mid, 2=high mileage proxy. Missing target classes are not physical wear labels.')
fields('epoch selected_epoch', 'integer', 'epoch', 'One-based epoch; selected_epoch is the currently selected validation endpoint.')
fields('update', 'integer', 'completed update', 'Completed-update cursor; training observation rows name the update whose input produced them, before that optimizer step.')
fields('parameter_update', 'integer', 'model-state update', 'Number of completed optimizer updates in the parameters that produced this observation/coefficient record. Training predictions use update-1.')
fields('cursor', 'integer', 'batch position', 'Zero-based batch cursor inside the deterministic epoch schedule.')
fields('occurrence slot', 'integer', 'sample position', 'Zero-based draw position in an epoch, or zero-based slot inside the batch; draws use replacement.')
fields('augmentation_seed', 'integer', 'RNG seed', 'Stateless per-occurrence NumPy generator seed used for the three recorded query-only appearance factors.')
fields('draw_probability', 'float', 'probability', 'Actual hierarchical probability: class x domain x source-group x image draw probabilities.')
fields('brightness contrast saturation', 'float', 'multiplier', 'Recorded query-atlas appearance factor, sampled independently in [0.9,1.1]; support validity stays fixed.')
fields('stage endpoint variant', 'string', 'protocol label', 'Training/evaluation phase, checkpoint endpoint, or frozen diagnostic transformation. Online training summaries mix successive parameter states.')
fields('review_policy risk_note tie_break unavailable_reason', 'string', 'explanation', 'Explicit bank inclusion/risk, validation ordering rule, or reason a numeric result is unavailable.')
fields('level', 'enum', 'aggregation scope', 'cohort, domain, physical_tyre or source_group; unresolved clip IDs never become physical tyre IDs.')
fields('is_bank is_query empty_support metric_available selected correct finite gradient_finite', 'integer', 'boolean 0/1', 'Declared inclusion/availability/selection/correctness/finiteness flag. Empty-support images still receive raw scores and stay in metric denominators.')
fields('logit logit_0 logit_1 logit_2 bias contribution mass_input mass_projected unsupported_mass', 'float', 'class-logit units', 'Signed class evidence or separate bias. Patch contributions plus bias reconstruct the logit; splatting conserves signed patch mass.')
fields('probability probability_0 probability_1 probability_2 confidence_sum mean_confidence', 'float', 'dimensionless probability', 'Raw uncalibrated mask/class probability, or the declared bin sum/mean of maximum class confidence.')
fields('affinity', 'float', 'dimensionless', 'exp((cosine-1)/temperature), comparing the current query patch and actual current-encoder reference input.')
fields('cosine', 'float', 'dimensionless [-1,1]', 'Cosine similarity of L2-normalized 96-dimensional query and reference embeddings.')
fields('pooling_mass', 'float', 'dimensionless weight', 'Patch pooling alpha: equal supported-band total mass, coverage-weighted within band; zero for unsupported patches.')
fields('valid_fraction clipped_fraction', 'float', 'fraction [0,1]', 'Predicted-mask support fraction, or fraction of source rows clipped at an image edge; not calibrated quality/confidence.')
fields('nll validation_nll loss entropy', 'float', 'nats', 'Three-class negative log likelihood, mean cross-entropy objective, or natural-log class entropy as named. No temperature calibration is applied.')
fields('brier', 'float', 'squared probability sum', 'Sum over three classes of (probability-one_hot_target)^2; not divided by three.')
fields('evidence_residual residual', 'float', 'absolute class-logit error', 'Absolute residual between retained signed evidence plus bias and the raw logit. FP32 projection tolerance is 2e-6.')
fields('accuracy validation_ba_observed dice iou box_iou', 'float', 'fraction [0,1]', 'Named metric: BA averages recall only over observed target classes; Dice/IoU use unignored pixels with +1 smoothing; boxes are mask-derived and half-open.')
fields('patch atlas_patch_index', 'integer', 'patch index 0..31', 'Eight atlas bands x four columns; index=band*4+column, each patch 64x64.')
fields('prototype prototype_index', 'integer', 'case index 0..17', 'Literal fixed training reference case; six per class, from only 1/3/4 independent training tyres.')
fields('atlas_row', 'integer', 'atlas coordinate', 'Row/band coordinate as named by table: prototype_bank band is 0..7; atlas_rows pixel row is 0..511.')
fields('row column', 'integer', 'zero-based pixel coordinate', 'Pixel coordinate inside a 64x64 reference tensor or 128x96 dense prediction grid as identified by the CSV family.')
fields('channel region', 'integer', 'channel index', 'RGB channel 0/1/2 in bank_pixels; dense mask/region 0=tyre,1=tread. Masks may overlap.')
fields('point', 'integer', 'ordered point index 0..5', 'Left/right ordered boundary pair at each of three fixed image-relative guide rows.')
fields('point_kind', 'enum', 'target provenance', 'human, polygon_weak or none. Weak targets remain separate from human evaluation truth.')
fields('predicted_x_fraction target_x_fraction absolute_error_width_fraction', 'float', 'fraction of input width', 'Predicted/target boundary x coordinate or absolute error, normalized by width-1. No physical depth/alignment units.')
fields('prediction_guide_y_fraction target_guide_y_fraction', 'float', 'fraction of input height', 'Fixed model guide or target annotation guide y. Point errors are unavailable when guide rows differ by more than 1e-5.')
fields('source_y source_left source_right', 'float', 'resized input pixels', 'Inverse visible-atlas mapping in the 512x384 input. Unsupported rows have blank coordinates; not original-camera or 3D coordinates.')
fields('pred_left pred_top truth_left truth_top', 'integer', 'resized input pixels', 'Inclusive lower box coordinate in the 512x384 image after applying the source ignore policy.')
fields('pred_right_exclusive pred_bottom_exclusive truth_right_exclusive truth_bottom_exclusive', 'integer', 'resized input pixels', 'Exclusive upper box coordinate; absent boxes stay blank.')
fields('predicted_pixels target_pixels intersection union valid_pixels supported_pixels', 'integer', 'pixels', 'Named mask/support sufficient statistic; dense regions use full resized image and unignored pixels, explanation support uses atlas pixels.')
fields('supported_patches supported_bands', 'integer', 'count', 'Number of positive-mass patches out of 32, or supported longitudinal bands out of eight.')
fields('count n elements trainable_parameters anchor_gradient_count', 'integer', 'count', 'Declared row/bin/metric denominator, tensor elements, trainable parameter count or anchor parameters with gradients. Metric n follows the stated denominator.')
fields('bin', 'integer', 'calibration-bin index 0..9', 'Ten equal-width maximum-class-confidence bins; right edge inclusive only for final bin.')
fields('lower upper', 'float', 'probability-bin boundary', 'Lower/upper equal-width confidence boundary; no data-dependent bin tuning.')
fields('parameter', 'string', 'parameter name', 'Named trainable tensor whose actual pre-clipping gradient statistics were collected.')
fields('kind', 'enum', 'coefficient kind', 'signed_weight, bias or temperature. Temperature controls similarity, not probability calibration.')
fields('dimension', 'integer', 'embedding index 0..95', 'Current shared-encoder reference-vector dimension at the first evaluation batch of this immutable checkpoint.')
fields('value', 'float', 'family dependent', 'bank_pixels: exact FP32 RGB [0,1]; coefficients: signed weight/bias/temperature; reference_embeddings: normalized feature; metrics: named metric units/denominator.')
fields('norm mean minimum maximum grad_norm_before_clip grad_norm_after_clip', 'float', 'gradient units', 'Named tensor-gradient L2 norm/element mean/min/max, or total trainable-gradient norm before/after clipping at one.')
fields('lr', 'float', 'optimizer learning rate', 'Learning rate used by the recorded update, before the completed-step cosine scheduler advances.')
fields('compute_seconds record_seconds', 'float', 'seconds', 'Compute includes forward/backward/optimizer/scheduler; record_seconds covers observation construction, before transactional CSV/native I/O. Commit timing is retained separately.')
fields('device', 'string', 'Torch device', 'Actual optimization device; validation uses a separate immutable model copy on the declared evaluation device.')
fields('cuda_peak_allocated_bytes cuda_peak_reserved_bytes host_peak_rss_bytes free_disk_bytes', 'integer', 'bytes', 'Actual update resource counters; optional unavailable host/GPU values are blank. Runtime free space is not an assumed scratch quota.')
fields('occluded_patch', 'integer', 'patch index', 'Highest signed patch contribution for the base predicted class, replaced with recorded mean RGB; may be unsupported in a failed atlas.')
fields('replacement_r replacement_g replacement_b', 'float', 'RGB [0,1]', 'Actual valid-atlas channel means used for the logged patch occlusion; blank outside that diagnostic.')
fields('metric', 'string', 'metric name', 'Explicit named statistic; cohort/group class recall with zero targets is unavailable, not fabricated as zero.')
fields('anchor_logit_0 anchor_logit_1 anchor_logit_2', 'float', 'class-logit units', 'Frozen TREAD-ONE baseline logits on the original FP32 frame; unchanged during query-only atlas diagnostics.')
fields('anchor_probability_0 anchor_probability_1 anchor_probability_2', 'float', 'dimensionless probability', 'Frozen TREAD-ONE baseline raw class probabilities on the original frame; historical AMP scores may differ.')
fields('anchor_prediction', 'integer', 'proxy class 0/1/2', 'Frozen original-frame TREAD-ONE baseline argmax; retained alongside each candidate observation.')

all_columns = {name for columns in SCHEMAS.values() for name in columns}
assert all_columns <= set(D), sorted(all_columns - set(D))
target = PILOT / 'assets/COLUMN_DICTIONARY.csv'
with target.open('w', encoding='utf-8', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=['table','column','dtype','unit','missing','definition'], lineterminator='\n')
    writer.writeheader()
    for table, columns in SCHEMAS.items():
        for column in columns:
            dtype, unit, definition = D[column]
            writer.writerow(dict(table=table, column=column, dtype=dtype, unit=unit,
                                 missing='Blank means unavailable/not applicable; recorded zero remains zero.', definition=definition))
print('Column dictionary:', sum(map(len, SCHEMAS.values())), 'table-column definitions')
