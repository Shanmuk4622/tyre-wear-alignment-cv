# Evidence, design decisions and risks

## Observation → choice → test

| Existing project evidence | Chosen response | What must be measured |
|---|---|---|
|All v1/LC/context selected/final low recalls 0/62 |Remove mileage CE from this first new pilot; target geometry/texture/error objectives |Masked prediction/noncollapse and dense preservation; no classification improvement claim |
|Frozen LC already correct on 47,040 sampled train views; context sharpened confidence |Use self-supervised target prediction and representation diagnostics instead of another confidence-maximizing classifier |Mean-predictor comparison, invariant features with retained variance/rank |
|Shared-gradient projection frequently acted and selected point error worsened |Frozen backbone, disjoint adapter/texture/error learning paths; separate clips |Gradient and optimizer-routing tests; actual geometry floors |
|Strong existing visible geometry and exact project checkpoint available |Warm-start the selected seed 3/epoch 20 and train bounded small refinements |Zero-start equivalence and matched baseline geometry on identical views |
|Source/clip parent uncertainty remains |Preserve NB07 co-assignment and historical roles |Source hashes/schedule/exclusion checks; no new independent support claim |
|Prior quiet stages looked frozen; one recovery required missing-file repair |Self-contained bootstrap, visible phase status, full verifiedHF restore |Actual fresh-directory restart and lost-response/stop tests |
|User requires much richer evidence in direct CSV |All 40 families plus every train geometry logit/token/masked target/prediction native chunk |Exact storage projection and independent sufficient-statistic/native loss reconstruction |

Sources: [NB05 results](../../../phase3NB05Completion.md), [NB15 completion](../../../phase3NB15Completion.md), [research closure](../../../phase3TreadTwoResearchClosure.md), [source hashes/primary references](SOURCES.json). These observations motivate a new hypothesis; they do not prove it will succeed.

## Failure register

| Risk | Prevention / observation | Decision if observed |
|---|---|---|
|Texture collapse or coordinate-only shortcut |Global across-image variance/rank, masked-loss mean baseline, fixed input shuffle controls, no hidden token leakage |Fail eligibility or report insufficient texture representation; no appearance-stability-only success claim |
|Tiny correlated batch makes covariance noisy |N≥4 distinct images/≥2 components; no patch-as-subject counting; coverage/stats retained |Skip invalid texture steps; fail useful-coverage gate if excessive; no mid-fit batch-rule change |
|Geometry adapters overfit or atlas drifts |Small zero-start bounds, anchor drift, no texture gradients into geometry, validation floors |Reject checkpoint; retain exact failed evidence and final state |
|Mask refinement learns annotation noise |Preserve ignore map/native overlapping labels, raw human/weak point provenance |Show affected groups/targets and avoid hidden annotation repairs |
|Error head predicts a constant low error |Training-median baseline, error bins, MAE/RMSE, separate human support |Keep experimental/unvalidated label; no reliability or safety score claim |
|Same-image invariance learns tyre identity |No population/physical-wear claim; per-source/component probes and cross-component exclusions |Report limited representation utility; do not infer new-tyre recognition |
|Masked target is solved by local average |Frozen mean-target baseline plus≥5%gain requirement; hidden pixels absent from masked path |No useful masked-prediction claim if baseline not beaten |
|Raw retention exceeds 20 GB area |Immutable chunking, typed CSV projection, measured staging/hard-link test,15 GB logical/18 GB peak caps |Fail before measured fit; revise explicit source/schema before freezing |
|Cross-device math or unsupported op breaks replay |FP32, explicit boundaries, identical-math local/T4 seams, strictactualenvironment binding |Repair fixtures before fit; never silently relax determinism |
|HF throttling or forced kernel death |30 min/major/stop publication, durable local units, budget/backoff/status, explicit pending receipt |Resume last surviving verified state; unsent destroyed files cannot be fabricated |
|Repeated historical validation shapes model design |Explicit development scope, freeze new selection before fitting, no test reselection |No blind/generalization or formal noninferiority claim |

## Fixed now versus implementation measurements

Fixed design: one candidate/seed/budget, dataset/anchor, branch structures and disjoint gradients, losses/coefficient starting values, structured epoch schedule, no class loss, proposed selection/failure rules,40 requiredCSVfamilies, raw tensor scope, new namespace and notebook stages.

Pending measurements with explicit acceptance: exact instantiated parameter count/shapes, actual forward/backward memory/time, live output/native CSV byte projection and staging capacity, deterministicT4 seam, actualHF restore/quota behavior, reconstruction loss baseline numeric values, baseline validation geometry, and all model-quality results. These are tasks in the plan, not missing user preferences or a reason to invent performance numbers.

No new user permission checkpoint is introduced by these engineering tests. They automatically determine whether the already planned implementation is internally valid. This turn delivers planning; it does not claim implementation or initiate a measured fit. Future runtime refinements before first fit remain possible if all changed choices are documented and revalidated before the measured contract is frozen.
