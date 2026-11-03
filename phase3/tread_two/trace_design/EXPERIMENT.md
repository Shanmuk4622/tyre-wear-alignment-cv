# Draft experiment and rejection rules

<!-- NB24_COMPLETION -->
## Actual execution update — 2 November 2026

NB24 completed the original one-seed fit and review. Original selected=None and
all source/contract identities are preserved. [Completion](../../phase3NB24Completion.md)
and [illustrated findings](../../phase3TreadTwoTRACEFindings.md) supersede pending
execution statements below. No rerun or new fit is required for this cycle.
<!-- END_NB24_COMPLETION -->

<!-- TRACE_BATCH_SUCCESSOR -->
## Executable successor delivered

**2 November 2026:** this file retains the original design-stage record. The
separate [TRACE notebook batch](../../phase3TreadTwoTRACEBatch.md) is now implemented
with NB21–NB24,671 executable fields and local acceptance. Use NB24 for automatic
preparation→gates→pilot→review. Actual Kaggle/HF and measured results remain
pending; the original prototype/code/check receipts remain unchanged.
<!-- END_TRACE_BATCH_SUCCESSOR -->

**Prototype recipe; measured protocol not frozen.** No sweep, new seed fit or
data acquisition is launched by this document. Training can begin only after the
new trainer, complete evidence writers, native storage accounting, interruption
transactions and actual dual-T4/HF continuation gates pass.

## Scope and source roles

| Item | Draft first candidate |
|---|---|
| Name / future namespace | TREAD-TWO / TRACE r1; `phase3/tread-two/trace-r1` |
| Input data | Existing expanded `shanmuk4622/tire-dataset-prepared-phase2` |
| Roles | Train 386 / validation 81 / test 103; original overlay unchanged |
| Geometry | Frozen original selected anchor + GEO unselected final geometry adapters |
| New parameters | 71,022; no geometry/quality/classifier optimizer |
| Measured fits | One seed, proposed seed 1 |
| Budget | Proposed 60 epochs, batch 8, 49 updates/epoch =2,940 complete updates |
| Exposure | All 386 train sources once/epoch, plus six explicitly labelled padding draws |
| Padding policy | Deterministic source choice excluding the two already present in the last batch; eight distinct image SHAs per full update |
| Views | Two deterministic tile-local photometric views, clean fixed targets |
| Optimizer draft | AdamW, LR 0.001, weight decay 0.0001, global texture-only clipping at 1 |
| Precision draft | FP32; no scaler or scheduler, saved explicitly as not applicable |
| Target fitting | Train-only channel/position statistics; no whitening, train/test mixing or labels |
| Evaluation | Original-role validation each epoch; final/selected endpoints after selection; already-inspected test once per endpoint |
| Extra classifier / learned confidence fit | None |

The budget keeps the familiar workflow; it is not a promise that 60 epochs are
necessary or sufficient. Freeze the final duration/weights only after numerical,
gradient-scale and resource gates, before fitting. A later change needs a new
unmeasured revision, not an edit to an active experiment.

Training has six conservative components of sizes 249/67/60/5/3/2, with proxy
low/mid/high component counts 1/1/4. Validation and test each have two components
and no middle class. Distinct image SHAs are not independent physical tyres.
Development roles have been repeatedly inspected, including the historical test
role; no new blind generalisation claim is available.

## Draft eligibility, utility and selection

The following are proposed engineering/scientific decision rules to freeze in
the measured contract. They have **not** been evaluated on a TRACE model.

| Gate | Proposed rule | Why |
|---|---|---|
| Fixed geometry | Pinned source/export and zero parameter drift; endpoint masks/points compared with the declared frozen input | Avoid silently changing the inherited task |
| Exposure | At least 90% of primary draws context-eligible; full source and skip accounting | A low loss on an unrepresentative subset is insufficient |
| Common control support | At least 95% of eligible model hidden-cell weight retained by the common control intersection | Comparisons cannot hide harder missing cells |
| Control denominators | Positive finite common-control count/loss; position and neighbour missingness explicit | No division by zero or invented utility |
| Masked utility | At least 5% lower image-mean standardised Huber than **each** channel, position and visible-band control on the same validation cells | Beat actual easy alternatives |
| Local utility | At least 10% lower local image-mean Huber than the train-position control | Verify direct local target learning |
| Projector diversity | Effective rank ≥4 and median std ≥0.05 in validation view 0 | Six-dimensional projection cannot be the only healthy space |
| Encoder diversity | Local-pooled 48D and contextual spatial-moment 240D validation rank ≥4 | Check upstream degeneracy as well |
| Within-component diversity | Within-component centred validation rank ≥2 in local-pooled and spatial-moment spaces | Do not accept merely separating the two held-out groups |
| Numeric state | Finite loss, gradients, optimizer, feature spectra and retained outputs | Reject corrupt or degenerate measurements |

For eligible epochs, minimise `masked_model_loss / min(control_losses) +
0.1*paired_projector_cosine_distance`, with earliest epoch for exact ties.
Here each loss is an image mean on the declared common support, not an
incompatible sum across different counts. Cosine availability/norm floors must
be explicitly implemented before protocol freeze. MSE remains the training
paired objective; cosine is a separately logged validation diagnostic.

If no epoch passes **all** gates, selected is unavailable. Keep final and all
epoch weights as research evidence. Do not substitute the best test checkpoint,
relax a threshold after seeing results or describe a geometry-only finding as
joint representation success. The inherited GEO `selected=None` stays unchanged
regardless of the new experiment's outcome.

## Development controls and analysis

| Analysis | Fit / evaluation permission | Retained evidence |
|---|---|---|
| Local and masked targets | Deterministic clean image descriptors; stats train only | Complete maps once; per-view target IDs and counts |
| Geometry sentinels | Fixed model, no updates | Input/output hashes, masks, human-only point availability |
| Appearance stability | Fixed deterministic factors; no new fitted transform | Both views, scalar factors, feature deltas, reconstruction losses |
| Layout sensitivity | Fixed row/column reversal probes | Actual token/moment changes; no claimed causal wear interpretation |
| Within/between scatter | Original component registry; no reassignment by features | Exact decomposition plus covariance eigenvalues and counts |
| Retrieval | Training bank only; full-source SHA self exclusion | Neighbour IDs/distances and exclusions; no label-dependent ranking |
| Descriptor residuals | Model vs known image-derived target | Spatial/channel residuals, weight and availability, not calibrated confidence |

The completed local target audit informs architecture and implementation only.
Its six-dimensional image-average rank remains low (1.458), while full common
spatial targets have rank 15.657. Neither is a trained validation result; no
claim that target design solved representation collapse is made.
