# Claims, limitations and figure audit

<!-- current-status:start -->
> **Current status (15 September 2026):** [Completed work and remaining validation](../CURRENT_STATUS.md). The report is refreshed; the app, learned-geometry integration and target-assisted alignment software exist. Dated plans below retain their original context.
<!-- current-status:end -->

**15 September geometry addendum:** [Verified matched comparison](../36_MATCHED_GEOMETRY_RESULTS_AND_INTEGRATION.md)
supports “HRNet had 23.75% lower mean horizontal point error than SegFormer-derived
boundaries on this two-tyre test set across three seeds.” It does not support
architecture-only superiority, calibrated alignment, safety, downstream wear-F1
improvement or completion of the original full S9 research design. Prototype
software integration is implemented. Supervision differs; SegFormer coverage
was 100%. The user confirmed 12 sessions represent different physical tyres,
but no independent identity audit was performed. The ledger and main manuscript
are now refreshed; author review remains required before submission.

Updated author-review edition, 15 September 2026. Read together with [REPORT.md](REPORT.md). The matched results and existing prototype are now incorporated into the report body and figures; original research gaps remain separate from completed implementation.

## Claim-to-evidence ledger

| Claim | Evidence | Verdict / permitted wording |
|---|---|---|
| 418 unique photographs | Dataset specification, frozen preparation audit | 418 clean originals; derivatives are not independent samples |
| 12 independent tyres | 15 September user confirmation that 12 sessions represent 12 different physical tyres; no independent audit | Say “12 user-confirmed distinct tyres”; retain grouping/split limitations and do not claim independently verified identity |
| All architecture runs scientifically valid | Master + nine quarantine rows | 153 retained runs / 17 architectures; nine substitutions excluded |
| Best architecture generalises almost perfectly | Selected master means and flagged folds | Unsupported external claim. Specify selected/final endpoint and split |
| Matched random-init comparison | Recovery audit, nine final-epoch runs | ResNet-50 final mean 0.823322, not selected mean; not legacy ResNet-18 |
| XAI selection is better than accuracy | NB07 selection and inherited H1 | Selection completed; H1 advantage not supported |
| S4b is independent replication | Same fold 1, new architectures/seeds | Same-fold confirmation only |
| S5 complete | 81 runs, 60 epochs each, NB17 audit | Manual-supervised dense-task scope complete |
| Model/manual IoU validates annotator reliability | One existing reference set | Unsupported; no blind repeat or independent annotator study |
| Localisation improves classification | 405 ROI rows and paired deltas | No overall mean improvement for the tested predicted crops |
| Fusion improves recognition | 405 fixed fusion rows | Equal full+tyre+tread mean delta −0.013238; descriptive negative result |
| Every conformal fold misses nominal coverage | Fold table | Incorrect: fold 0 below 90%; folds 1/2 above in the observed test subsets |
| Calibration guarantees reliable deployment | Small dependent subsets; empty sets | Unsupported. Report achieved values and set-policy limitation |
| H2/H3 confirmed | Hypothesis outcome and region coverage tables | H2 undefined/inconclusive; H3 untested |
| Full original S9 completed | NB18, completed geometry comparison and local prototype integration | Software integration exists; full Tier8/end-to-end/PatchCore claims remain unsupported |
| HRNet improves point localisation | Frozen matched test, three seeds/two tyres, full coverage | 23.75% lower mean error in this study; not equal supervision or external generalisation |
| Optional app not built | Existing Tread Station and saved local tests | Stale claim: desktop app and learned geometry are implemented |
| Alignment fully deferred | Target-assisted workflow and synthetic tests exist | Software implemented; real camera/fixture/reference accuracy unvalidated |
| Full experiment/manuscript approved | Current artifacts, no institutional sign-off | Implemented report prepared; original gaps and human review remain |

## Open research gaps

| Gap | Why current evidence cannot fill it | Next requirement if authorised |
|---|---|---|
| Independent tyre generalisation | Session grouping does not verify identities | New identified tyres and frozen tyre-held-out evaluation |
| Physical wear/depth | Mileage proxy has no physical calibration | Target-matched measurements and measurement protocol |
| Video-domain learned geometry | Still-study benefit coexists with crossed/missing video boundaries | Define held-out labelled video evaluation; no duplicate point-label pilot by default |
| PatchCore healthy reference | Low-mileage label does not certify healthy condition | Independently justified healthy pool and evaluation definition |
| SAM2/manual and blind consistency | NBT1 is replay, not repeat annotation | Separate independent comparison if user later authorises it |
| H3 fine-grained arms | Planned model arms absent | Predeclared model/budget/evaluation design |
| Video temporal stability | Still-image plots are not video evidence | Suitable held-out clips and defined temporal metrics |
| Crossed resolution × ROI | Separate OFAT arms do not estimate interactions | Frozen crossed design, not a post-hoc interaction claim |
| Broader Tier5/6 and XAI extensions | Implemented tracks cover only declared subset | Explicit scope decision and appropriate new evidence |

Do not schedule annotation or training merely to turn this table green. HRNet training and integration are done; PatchCore remains unresolved. The current task refreshes documentation, not new model work.

## Figure review

All 14 earlier public figures retain their source bytes. The refreshed flow diagram, endpoint chart, two matched-geometry charts and four saved prototype screenshots bring the report to 22 visuals. Prototype screenshots distinguish software evidence, qualitative still agreement, video failure and synthetic alignment; none is mislabelled as real physical accuracy.

| Visual group | Review finding | Presentation decision |
|---|---|---|
| Endpoint comparison | Generated directly from selected/final columns | Explicit endpoint legend; no fake uncertainty bars |
| Study overview | Schematic, not a measured result | Deferred branch visibly distinct |
| NB10R accuracy/per-fold/H1 | Source figures1/2 use final-epoch macro-F1; H1 figure5 retains legacy selected endpoint | Distinguish these individually; “inherited” does not mean every plot uses the same endpoint |
| Saliency | 30 panels; two recorded zero maps; source title overlaps top labels | Retain original in appendix, disclose layout issue; not a raw photograph montage |
| OFAT | Discovery, with source-labeled95% normal intervals over run effects | Separate from confirmation table; not independent-tyre uncertainty or a significance claim |
| Stress | Source uses “causal” title; marking fold1 intervention is a no-op | Caption explicitly limits causal interpretation and rejects a marking-independence claim |
| H1/session source layout | Some H1 labels overlap; session names are abbreviated | Keep source evidence intact; interpret numeric tables and original IDs |
| S10 dense/ROI/fusion | Three-seed SD, existing folds | Caption says SD, not confidence interval; no independent-tyre claim |
| Manual-mask IoU | Model versus existing manual reference | Not SAM2 or annotator repeatability |
| Best epoch/energy/session | Implemented diagnostics | Not missing original video/factorial experiments |

Automated link/image/hash checks complement visual review; they do not substitute for final author inspection of labels at the required printed page size. The tall saliency source should be supplementary or a full-resolution attachment in a constrained page-limit submission.

## Review decisions made in this edition

- Incorporated the later confirmation as “12 user-confirmed distinct tyres”, without claiming an independent identity audit.
- Preserved final versus selected endpoints; explicitly separated the two random-init baselines.
- Reported the actual conformal fold results instead of the inaccurate blanket coverage statement.
- Preserved negative fusion results and avoided fabricated p-values, confidence intervals or causal claims.
- Selected verified primary references and official model documentation; did not copy all historical related-work claims.
- Kept original-plan deferrals visible and did not invent approvals, contributions, funding or permissions.
