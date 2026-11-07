# Claim-to-evidence ledger

**Manuscript revision 3 · author review.** Rounded paper values derive
from the copied evidence below. Selected checkpoints, fixed final checkpoints,
original measured results and post-hoc calculations remain distinct.

| ID | Claim / value | Endpoint and evidence | Interpretation / boundary |
|---|---|---|---|
| C01 |570 images;418 originals +152 old frames;12 user-confirmed tyres | Fixed overlay; [alignment](evidence/aggregation_image_alignment.csv); [readiness](../phase3NB15Completion.md) | Frames add views, not subjects; identities not independently audited |
| C02 |Train/validation/historical test386/81/103; components6/2/2 | Same overlay; [support](evidence/aggregation_support_summary.csv); readiness reports | Mid parents conservatively merged, not asserted identical; no held-out mid truth |
| C03 |7 families;10 executions;600 epochs;29,400 updates | [Fit ledger](evidence/phase3_fits.csv) | Excludes preparation/reviews/fixtures; recovery and wrappers do not multiply fits |
| C04 |Selected r1 mean Dice97.978%; point MAE1.189% width; BA34.146% |3 selected seeds; [endpoints](evidence/classification_endpoints.csv) | BA over2 observed classes; each selected model misses low0/62 |
| C05 |Projection selected BA17.073% versus50%; point1.885% versus1.067% | Matched seed1; [paired endpoints](evidence/paired_endpoints.csv); [review](../phase3NB05Completion.md) | One paired comparison; not a general claim that PCGrad fails |
| C06 |ATLAS/LC/context BA50%; selected NLL1.1783/.9613/2.1748 | [Endpoints](evidence/classification_endpoints.csv) | One seed each; serial revisions, not factorial ablations |
| C07 |GEO point .00808536→.00766220 width; Dice .98986587→.98986230 | G0/final validation; [GEO](evidence/geo_validation.csv); [findings](evidence/geo_findings.json) |5.23% relative ≈.163 pixels at384 width; joint selection unavailable |
| C08 |TRACE71,022 trainable;1,399,227 frozen;1,470,249 total | Original frozen32-file package; [design](../phase3TreadTwoTRACEDesign.md) | Geometry explicitly includes unselected GEO-final research adapter |
| C09 |Local validation36.44%; masked21.63% improvement | Final epoch60/update2940/view0; [science](evidence/nb24_science.json); [roles](evidence/trace_roles.csv) | Local denominator: position control; masked: strongest of3 controls, visible band |
| C10 |Historical-test local37.80%; masked17.85% | Same endpoint/recorded plan; [roles](evidence/trace_roles.csv) | Reused historical role, not a new blind test |
| C11 |60/60 utility;0/60 projector/global diversity; selected=None | [All original epochs](evidence/trace_validation_epochs.csv) | Final is diagnostic; no gate relaxation or test-best selection |
| C12 |Mean retains61.14% local val advantage; template+5 residuals81.76% | [All9 variants](evidence/aggregation_utility_summary.csv); [table](generated/table_aggregation.csv) | Position-control advantage denominator; not accuracy or wear-information fraction |
| C13 |Raw5 moments76.05%; random5 modes65.47%; template7.75% local val retention | Same unchanged decoder/support | One fixed random basis; modes differ in spatial capacity |
| C14 |Mean local test retention low32.03%/high84.02% | [Component utility](evidence/aggregation_component_utility.csv) | One component per class/role; does not prove pooling caused classifier failure |
| C15 |Mean orientation-cross loss .790882 >position .667389 | [Channels](evidence/aggregation_channel_utility_summary.csv), local val | Absolute standardized loss avoids unstable small-margin retention |
| C16 |Within-component constant spatial mode4.15% of local variance | [Covariance](evidence/aggregation_spatial_covariance.csv), val/view0 |95.85% outside mean is variance, not useful wear information; fixed train-mean mass |
| C17 |Context moment rank1.2489→2.9120 by invertible scaling | [Spaces](evidence/aggregation_feature_spaces.csv) | Adds no information; does not replace original gate |
| C18 |Masked rank15.3126; C−H/mask CKA .8879 overall/.9575 within | [Mask alignment](evidence/aggregation_mask_plan_alignment.csv); [addendum](evidence/aggregation_mask_addendum_protocol.json) | Exploratory; deterministic plans; CKA not explained variance; projector reads H |
| C19 |3,553 CSV/47,263,346 rows/6,725 NPY/64 PT/13,506,978,062 bytes | [NB24 verification](evidence/nb24_verification.json); science receipt | Includes297,021 cache rows; richer telemetry is not more subjects |
| C20 |Published full dense audit and later bounded independent audit differ | Science receipt; [completion](../phase3NB24Completion.md) |946-body local audit is not a full independent dense re-download |
| C21 |All63,037,440 counterfactual values reconcile | [Validation](evidence/aggregation_validation.json) | Independent arithmetic reduction, not physical-label validation |
| C22 |2 real examples use lexical-first selection | [Identities](evidence/qualitative/EXAMPLES.json); [cells](evidence/qualitative/orientation_cells.csv) | Not error-ranked; display-only saturation[-1,1]; metrics not clipped |
| C23 | Token PCA retains 89.47% local validation and 89.01% historical-test advantage at five modes | [Matched-budget summary](evidence/spatial_budget_summary.csv), main fit, clean view 0 | Same 240 input coefficients and shared template as fixed moments. Retention is relative to the position control. |
| C24 | Five-mode decoder-aware changes are small and inconsistent across roles | Same summary, Jacobian and decoded-snapshot rows | Surrogate optimality does not imply task-loss optimality. |
| C25 | Six training omissions give 86.45–89.01% local historical-test retention | [Fit cohorts](evidence/spatial_budget_fits.csv) and summary | Templates and eigensystems are refitted. The reference mass remains fixed. |
| C26 | Masked order changes mean absolute fields by 0.10045 with signed loss effect −0.00326 | Five-mode token PCA, clean validation | Output-first uses 480 coefficients. The result accounts for order and does not compare equal-rate codecs. |
| C27 | Fitted subspace overlaps 0.9773–0.9896 with token PCA, excluding the constant | [Angles](evidence/review_subspace_angles.csv), five-mode main fit | Geometric similarity on one training cohort and checkpoint |
| C28 | Median relative Jacobian differences of 31.65% local and 69.22% masked on full-support train tokens | [Jacobian variation](evidence/review_jacobian_variation.csv), [linearization error](evidence/review_linearization_error.csv) | Each comparison uses its branch-specific shared Jacobian |
| C29 | Affine PCA retains 90.21% / 89.91% locally and 18.07% / 10.62% in masked context | [Affine readout](evidence/review_affine_readout.csv), validation / historical test | Train-only fit at the same frozen checkpoint. Each readout defines its own uncompressed margin. |

## Computation accounting clarification

The original aggregation receipt reports **656,640 decoder-token readouts** for
the primary sweep:570×2 views×32 tiles×9 variants×2 branches. Source code also
separately replays both original decoders for bridge checks: **72,960 additional
decoder-token** and **1,140 projector-vector readouts**. The paper separates these
scopes instead of presenting the primary counter as all computation. No original
receipt or source was rewritten.

Paper preparation reads cached evidence and converts two existing atlas images
to display PNGs. The revision adds two synthetic TRACE forwards for shape and
mask checks and two closed-form train-only readout fits. Network optimizer
updates, real-image encoder and geometry forwards, and HF writes are zero.

## Claims excluded

| Excluded statement | Missing support |
|---|---|
| Validated physical wear/depth/safety predictor | No measured depth, damage severity or replacement/safety labels |
| Full three-class external accuracy | No held-out mid truth; few groups; reused historical test |
| Pooling caused low-class classifier failure | Different models/objectives; off-manifold interventions; identity confounding |
| TRACE completed as a selected model | Training complete, selection eligibility0/60 |
| More CSV rows mean more subjects | Same570 images |
| Higher rank establishes semantic improvement | Invertible scaling and mask association counterexamples |
| New derivatives are all public on HF | Aggregation and paper derivatives remain local |
| Authors approved; no conflicts; ethics exemption; unrestricted license | Requires actual author confirmation |

## Definitions

- **BA:** mean recall over ground-truth classes present; predictions may include mid.
- **Dice:** overlapping tyre/tread channel agreement under original ignore policy.
- **Point MAE:** horizontal error/image width, human subset only.
- **NLL:** true-class negative log likelihood over all3 outputs.
- **Reconstruction:** standardized Smooth L1(beta=.1), count-weighted cells,
  channel mean within image, then equal image mean.
- **Retention:** (position loss−variant loss)/(position loss−original loss), same support.
- **Rank:** covariance-energy entropy, distinct from singular-value-weighted RankMe.
- **CKA:** normalized centered-Gram alignment, not explained variance.
