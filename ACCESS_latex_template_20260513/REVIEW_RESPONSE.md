# Response to the supplied review — 18 September 2026

The review was treated as criticism to investigate, not as proof of its factual
claims or numerical scores. No new acceptance probability or quality score is
assigned. The revised manuscript addresses the writing and analysis issues below;
additional physical subjects, labels, and training are not manufactured.

## Principal corrections and new analyses

1. **Split interpretation corrected.** A fresh audit of 418 archived image records
   finds zero session intersections between train and validation in each of the
   three folds (8 training / 4 validation groups each). The earlier flag arose
   from resemblance of the 70,000- and 90,000-mileage sessions. The operator later
   confirmed different physical tyres. The manuscript now distinguishes verified
   disjoint group IDs, operator identity confirmation, and the historical visual
   suspicion. It no longer presents suspicion as established leakage. This is an
   allocation audit, not an independent identity inspection or replay of loaders.
2. **Sample size reported proportionately.** The 12 confirmed tyre identities remain
   in the dataset methods because they define the independent unit. Repetitive
   emphasis was removed from the abstract, overview diagram, and discussion.
3. **Endpoint reconstruction.** Eighteen complete 60-epoch histories for MobileNetV4
   and ResNet-50 were analysed. Original selection is the earliest maximum QWK,
   not the maximum F1. Fold-specific selected/final F1, final QWK, and final class
   MAE replace the pooled architecture table in the main result narrative.
4. **Threshold sensitivity.** At thresholds 0.03/0.05, the initial explanation screen
   retains 20 method rows across 10 models. At 0.10 it retains 16 across 8, losing
   RegNetY-016 and MobileNetV4. Initial selected methods are unchanged between
   0.03/0.05. The complete downstream selection is not claimed robust at 0.10.
5. **Fusion stratification.** All four crop/fusion arms have zero mean change on fold
   0, negative change on fold 1, and small positive change on fold 2. For full +
   tyre + tread, the values are 0, -0.04107, +0.00136. This materially refines the
   pooled negative result rather than implying every allocation deteriorates.
6. **Paired geometry detail.** The six targets and three seeds are averaged within
   image: 22/24 image means favour point learning, with paired differences from
   -0.556 to +0.857 percentage points of width. Location gains range from 0.029
   at left-middle to 0.847 at right-upper. No independent-tyre p-value or bootstrap
   interval is made from dependent point/image records.

These are new retrospective analyses of frozen data. No model was retrained and
no result from an existing model's training data is called a new test result.

## Issue-by-issue disposition

| Review topic | Response | Status / remaining work |
|---|---|---|
| 1. Novelty | Contributions narrowed to checkpoint, paired intervention, and boundary-pipeline evaluation; no new-backbone or first-system claim | Revised; novelty remains empirical |
| 2. Literature | 31 references total; 10-study application comparison, recent 2024–2026 work, explicit measurement/task distinctions | Revised with primary-source and DOI metadata checks |
| 3. Dataset/splits | Group-membership audit plus careful reconciliation of historical visual suspicion and operator confirmation | Allocation issue clarified; independent external test still absent |
| 4. Mileage proxy | Recognition explicitly targets mileage groups; physical wear claims excluded | Revised consistently |
| 5. Classification | Fold-specific reconstruction; QWK and MAE added; original earliest-max-QWK policy stated | Added for 18 illustrative runs, not falsely extended to the entire archive |
| 6. Explanation gate | 0.03/0.05/0.10 sensitivity, with changed eligibility reported | Completed on initial screen; downstream retraining not done |
| 7. Semantic Box AP | Threshold, enclosing box, positive-pixel mean score, empty-mask handling and evaluator specified from code | Clarified |
| 8. Crop/fusion | Frozen full-image classifier and paired design retained; fold effects added | Completed reanalysis; crop-trained models not evaluated |
| 9. Calibration | Moved to appendix; CORAL differencing/clamping/normalisation and distinct argmax explained | Revised; does not establish deployment coverage |
| 10. Geometry | Pipeline/formulation language, per-location and per-image effects | Improved; rotated grouped training and shared-backbone ablation remain unexecuted |
| 11. Statistics | Explicit aggregation unit and actual paired variation; no pseudoreplication | Improved descriptive evidence; broader subject-level inference needs more groups |
| 12. Findings | RQ-based narrative with positive, null and negative effects | Revised |
| 13. Reproducibility | Frozen downloads, SHA-256 manifest, deterministic analysis script and audits | Preserved and expanded |
| 14–15. Workstation/alignment | Reduced to failure visibility and implementation context; synthetic GUI figure removed | Revised; no physical-angle validation claim |
| 16. Figures/tables | Outdated full-window screenshot removed; plots alphabetical; new image-effect plot and stratified tables | Revised; saved failure example retained with provenance |
| 17. Grammar | Endpoint terminology, randomly initialised, exercised interruption path, rim phrasing | Revised |
| 18. Internal terms | Stage-A/S5 shorthand removed from the manuscript's main explanation | Revised |
| 19. Abstract | Near-perfect pooled F1 replaced by fold-specific sensitivity and paired effects | Rewritten |
| 20. Conclusion | Answers the three questions without new physical/generalisation claims | Rewritten |
| 21. References | Application literature expanded, metadata checked; no padding to a quota | 31 total including AI software disclosure |
| 22. Coherence | Three research questions; peripheral confidence material in appendix | Revised |
| 23. Submission | Two user-specified authors; confirmed no external funding; factual role biographies; specific AI acknowledgment | Corresponding-author proposal, conflicts, contributions and image permissions need confirmation |
| 24. Validation priorities | Distinguishes resolved analysis issues from genuinely new experiments | Remaining experiments are stated, not relabelled as completed |

## Experiments deliberately not presented as completed

- Rotated geometry group splits require fresh training of both pipelines for each
  split. Reusing the existing models would contaminate evaluation. This is not a
  small reanalysis and was not run locally merely to increase experiment count.
- A shared-backbone ablation requires another controlled training protocol.
- Annotation repeatability needs a genuinely independent annotator; generating
  alternate labels computationally would not measure human agreement.
- Labelled video and real calibrated wheel/depth reference captures require data
  that the existing uncalibrated images cannot provide.

The user requests Kaggle-compatible resumable notebooks for any future training;
no new training code or notebook is delivered in this revision because it contains
only record analysis and document rebuilding.

## Author information supplied by the user

1. Sreenivasa Reddy Edara — Professor, School of Computer Science and Engineering,
   VIT-AP University, Amaravati, India; sreenivasareddy.e@vitap.ac.in.
2. Shanmukesh Bonala — student, same school/university/location;
   shanmukesh.23BCE20070@vitapstudent.ac.in.

No funding confirmed. Contributions are proposed for confirmation, not asserted
as verified facts. Conflicts and image permissions remain pending. Biographies
contain only the supplied positions; no qualifications or publications are invented.

IEEE source checked: https://ieeeaccess.ieee.org/authors/submission-guidelines/
and https://ieeeaccess.ieee.org/authors/preparing-your-article/ (18 September 2026).
They require author biographies and AI disclosure in the acknowledgment. The draft
identifies OpenAI Codex and the scope of assistance. Final human author review is
still required; this document does not assert journal acceptance readiness.
