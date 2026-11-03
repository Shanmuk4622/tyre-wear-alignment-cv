# Methods and data dictionary — TRACE aggregation assay

This is a saved-feature investigation, not a new trained candidate. See
[main findings](../phase3TextureAggregationInvestigation.md) and the immutable
[primary protocol](PROTOCOL.json). The mask-layout follow-up is explicitly
exploratory in [its separate protocol](MASK_ADDENDUM_PROTOCOL.json).

## Frozen data and computation

- Original TRACE final epoch 60 /update 2940;570 aligned images,386 train /81 validation /103 historical test. View 0 primary, view 1 supplementary; no new augmentations or photos.
- Local `encoded`, unmasked `full_context` and `masked_context` arrays each have shape `[570,2,32,48]`; recorded masses/hidden plans select support. Targets have `[570,32,6,4,4]`, valid-pixel counts `[570,32,1,4,4]`.
- Descriptor targets/statistics and source SHA/role/component metadata come from the original fixed cache. Geometry saw the full original image before masking; no new geometry/image encoder forward occurs.
- Matrix calculations use CPU FP64. Original targets and position-control normalization use the original FP32 statistics/arithmetic before promotion. Identity decoder replay is checked against every original prediction and per-image loss.
- Decoder is `Linear48→96, SiLU, Linear96→96`, reshaped into 6 fields ×4×4 cells. Its weights are unchanged. The projector replay is `240→48→6` with the original SiLU and weights.

## Aggregation algebra

Let `Z` contain 32 position vectors and `M=diag(m)` the recorded coverage mass.
For a fixed position basis `B`, weighted projection is
`Z_hat = B (Bᵀ M B)^+ Bᵀ M Z`, with pseudoinverse rcond 1e-10.

| Variant | Exact definition |
|---|---|
| identity |Original stored tokens; no projection |
| global_mean |Constant32×1basis; broadcast the weighted mean |
| band_means |Eight disjoint row-band indicators |
| column_means |Four column indicators |
| moments5 |`1,y,x,y*x,y*y−1/3` on the original 8×4 grid |
| random5 |Constant mode plus 4 fixed zero-sum QR random position modes; seed 462225 |
| train_template |Train-view 0 supported-image mean at each position/channel; no per-image variation |
| residual_mean |Train template plus constant-mode projection of `Z−template` |
| residual_moments5 |Train template plus five-moment projection of `Z−template` |

The five-moment projection reconstructs an approximation to tokens. The actual
model only stores the raw moment coefficients for its auxiliary projector; this
reconstruction is a counterfactual analysis, not an existing decoder branch.

## Target and utility reductions

Each descriptor cell uses the original valid-pixel count and SmoothL1 beta 0.1.
Local decoder support is supported tiles; masked support is original hidden
tiles. Every variant shares the intersection with train-position-control
availability. Missing cells have zero weight and missing loss, not a true zero.
No variant silently changes its comparison denominator.

Per-image loss is `sum(weight*Huber)/sum(weighted channel counts)`. Role and
component scores are equal-image means. Retained advantage is
`(position_mean_loss−variant_mean_loss)/(position_mean_loss−original_mean_loss)`.
It can be negative or exceed1; values are not clipped. Per-channel retention is
reported only if its original control advantage exceeds1e-8. Small margins can
produce large negative percentages, so the main report gives absolute losses.

Tile summaries retain raw Huber sum, weighted count, available-image count and
sum of per-image tile losses. Interactive role heatmaps combine component loss
sums/counts, not unweighted averages of component means. Six channel means
reconcile exactly to each matching image-loss summary.

## Diversity, spatial covariance and mask controls

Features are cohort-centered for covariance. Entropy rank uses positive
eigenvalues above 1e-14 of total variance to exclude numerical nullspace noise.
Per-component centering is descriptive and never applied to inference outputs.
All original dimensions/sample counts are retained; no significance test treats
correlated views or images as independent tyres.

For fixed train-mean mass `r`, spatial covariance is
`C[t,u]=sum_i <Zc[i,t],Zc[i,u]>/(n−1)`. Weighted tile variance is
`sum_t r[t]*C[t,t]`; constant mean-mode variance is `rᵀ C r`.
Their difference equals weighted spatial-residual variance. This exact identity
is checked. Primary held-out roles have32 supported positions; train includes two
images with missing support, recorded separately. The fixed-grid training
covariance can include stored placeholder-token effects and is not primary texture evidence.

Train-only per-coordinate standardization uses a 1e-3floor. The inverse is checked
to 1e-12, demonstrating that covariance rank can change without new information.
These rescaled vectors never replace the original checkpoint/gate.

The exploratory mask follow-up computes linear centered-kernel alignment:
`CKA = <K_feature,K_mask>_F/(||K_feature||_F*||K_mask||_F)` after role or component
centering. It is association, not explained-variance percentage or a new mask
randomization. High masked-context rank can contain mask-placement information.

## Direct files and grains

| Family / file | Grain and meaning |
|---|---|
| IMAGE_ALIGNMENT |570 rows:source image SHA,role,component,slot and observation |
| NATIVE_INPUTS /CSV_INPUTS |Full checksums of 249 selected native/metadata and 254 consumed CSV bodies |
| BRIDGE_VERIFICATION |Six exact-source bridge/replay checks with maximum numerical differences |
| TRAIN_TEMPLATES /TRAIN_SCALES |Train-only descriptive parameter, counts and fit role/view |
| FEATURE_SPACES |Layer/representation/role/view/centering; rank,std,variance,dimension and group fraction |
| SPATIAL_COVARIANCE |Layer/role/view/centering; exact mean-mode and spatial variance partition |
| SPATIAL_ENERGY /summary |Per image/view/layer/basis template-centered projection energy; macro and ratio-of-sums summaries remain named separately |
| IMAGE_LOSSES_* |Image/view/branch/variant; raw sums/counts,loss,control margin,coverage and feature error |
| CHANNEL_LOSSES_* |Image/view/branch/variant/channel; raw sum,count and loss |
| detail/TILE_LOSSES_* |Image/view/branch/variant/tile; raw numerator,count,support,hidden state and optional loss |
| UTILITY_SUMMARY /COMPONENT_UTILITY |Equal-image role/component utility for all 9variants and both views |
| CHANNEL_UTILITY_SUMMARY /TILE_UTILITY_SUMMARY |Derived descriptor and per-component canonical-tile reductions |
| detail/SPECTRA /SPATIAL_COVARIANCE_CELLS |All eigenvalues and 32×32covariance/correlation matrices |
| MASK_PLAN_* /SUPPORT_SUMMARY |Exploratory mask association, plan diversity and actual support counts |
| native/prediction_* |All 18 counterfactual prediction arrays, FP64 `[570,2,32,6,4,4]` |
| native/features_* |All compared feature vectors; original coordinates retained |
| *_RECEIPT /ANALYSIS |Protocol/source/output hashes, completed stage reuse and computation scope |

Each CSV is direct and readable. Large per-image/channel/tile/native derivatives
are local/on-disk and ignored by Git; compact summaries/methods/figures remain
versionable. No derived artifact is claimed uploaded to HF in this pass.

## Replay and limits

Use `conda activate cv_conda`, or the existing `phase3/run_cv.cmd` wrapper. These
analysis scripts require the previously verified NB24 local audit mirror and
pinned manifests; they are not new standalone Kaggle training notebooks.
Order:prepare_texture_aggregation →test_texture_aggregation_math →
analyze_texture_aggregation →texture_mask_confounds →summarize_texture_aggregation
→document_texture_aggregation →build_aggregation_explorer →validation.

Completed analysis stages are size/SHA/source-verified before reuse. Source
images, original model code, checkpoints, protocols and both workstations are
not changed. There are zero image/geometry forwards and optimizer updates;
656,640 decoder-token and 1,140 projector-vector readouts are explicitly counted.
Fixed-head interventions are not newly trained heads; lower counterfactual
utility can reflect both information removal and off-manifold readout behavior.
The existing cohorts are already inspected, and proxy labels are not physical truth.
