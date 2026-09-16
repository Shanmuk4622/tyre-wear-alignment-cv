# S9 component analysis and remaining research gaps —2026-09-12

<!-- current-status:start -->
> **Current status (15 September 2026):** [Completed work and remaining validation](CURRENT_STATUS.md). The report is refreshed; the app, learned-geometry integration and target-assisted alignment software exist. Dated plans below retain their original context.
<!-- current-status:end -->

**15 September update:** The 120-image point dataset, HRNet training and matched
SegFormer comparison are now complete. HRNet has lower mean point error in all
three seeds on both test tyres. Geometry integration/end-to-end evaluation remain
next; PatchCore reference evidence remains unresolved. [Verified results](36_MATCHED_GEOMETRY_RESULTS_AND_INTEGRATION.md).
Earlier missing-landmark/pilot-only descriptions below are historical, not current blockers.

## 14 September update — pilot reopened, not full training

**Latest revised intake:** HF `05bf37c0f2067b0119ef057a2442d7759cf9ac51`
matches local labels. Upload succeeded, 11/12 complete: P06's new Visible issue
selection has no description. Left cropped points corrected; right-middle/lower
visibility remains unresolved. See docs/28; the 12/12 audit below is the first submission.

**Subsequent execution verified:** NB22 published all 12 submitted annotations at
HF `1dad525affd32b06dc73f907edad73eb4096b782`; local bytes match. Visual review
requests P06 visibility correction, then a predicted-mask baseline before HRNet.
No verified healthy references or training approval yet.
[Current audit and next plan](28_S9_PILOT_REVIEW_AND_NEXT_PLAN.md).

User now authorises the small guided input pilot. NB21/NB22 are implemented;
12 native images,9.46MiB ZIP, offline six-point/visibility annotation and visual
triage. Human review precedes any next batch or training. Neither a visual
observation nor an operator record claim automatically certifies health.
See [the pilot guide](27_S9_SMALL_ANNOTATION_PILOT.md). The earlier deferral below
is historical; full S9 remains uncompleted. Prototype geometry and target-assisted
calibration work is separate and has been read, not modified by this task.

## Verified completion — NB18

NB18 finished successfully:81/81 source runs processed, one commit, zero upload
failures. Public HF revision `35a178b5a94878bdee95a0aa9ed8cb25cd1aeb6b` contains
`s9/exploratory-fusion-r1/20dca3a2d91884230d555a7a36dabdbab8a075776487f3d147f1d4f9ddd21b87/`.

Independent verification checked all81 payload hashes, pinned source prediction
hashes, classifier configuration hashes/decision rules and implementation bytes.
Recomputed every payload from the original S5 input:56,430 per-image/arm
predictions,405 metric rows and135 summaries. Published CSV hashes and numerical
summaries match. STATUS is `exploratory_analysis_complete` with
`full_s9_complete=false`. Executed notebook remains unchanged; no HF writes by
the assistant. No NB18 rerun needed.

| Arm | Mean macro-F1 delta versus full image |
|---|---:|
| Full | 0 |
| Tyre only | −0.01257 |
| Tread only | −0.01467 |
| Equal tyre+tread | −0.01568 |
| Equal full+tyre+tread | −0.01324 |

These are equally weighted descriptive averages over81 runs, not independent
samples or significance tests. The tested fusion methods did not improve the
average endpoint. This does not mean every model/fold behaves identically or
that all possible fusion designs fail. Do not tune weights on these same results
and present the result as independent confirmation.

Full S9 still needs its missing components and frozen design, or an explicit
scope change. No such scope change is inferred from completing this notebook.

## Implementation and historical repair notes

### NB18 decoder repair

The uploaded notebook stopped on its fifth input (`unet_r34-f1-s2`) with
`Saved prediction/probability mismatch`. S5's classifier can use CORAL ordinal
threshold decisions, not class-probability argmax. NB18 incorrectly assumed
argmax for every classifier. This did not invalidate S5 saved predictions.

The repair reads each of the nine matched classifiers' `config.yaml` from the
frozen S5 classifier-source revision. CORAL uses the count of cumulative class
probabilities exceeding0.5; softmax uses argmax. The same recorded decision rule
is applied to both single views and the fixed averaged probabilities, keeping
the baseline comparison consistent. Classifier-config hashes and decision rules
are recorded in new results. All81 real input files,405 output metric rows and
all single-view decisions passed local verification, not just three sample runs.

HF revision `2951194a7bf997126222689a42ea5f1d1831130f` contains the first four
old-version analyses, preserved under their original code-hash namespace. The
repair recomputes CPU analyses under a new namespace; no trained model repeats.
Executed NB18 is archived before regeneration. Run repaired NB18 on CPU with
Internet/HF_TOKEN, one copy, Run All. **Subsequent audit above verifies repaired
Kaggle completion; these instructions are reproduction guidance, not a rerun request.**

`notebooks/NB18_S9_Fusion_Analysis.ipynb` runs on **Kaggle CPU, one copy,
Internet ON, HF_TOKEN enabled, Run All**. No dataset attachment, training,
GPU time, new annotation, or NB13–NB17 rerun is needed.

This is an explicitly exploratory component analysis, not the planned full S9
pipeline or its54 neural training runs. It reads S5 at fixed HF revision
`a3b29a71f8e6af6c50e68eb64a5bbae9ccf6d1c5`, verifies source hashes/labels/coverage,
and computes five fixed probability-level arms for each of81 S5 runs:

1. Full image only.
2. Predicted tyre crop only.
3. Predicted tread crop only.
4. Equal tyre+tread probability average.
5. Equal full+tyre+tread probability average.

Weights are fixed, not fitted to held-out labels. Oracle masks are not used in
fusion. All arms use the same existing matched-fold/seed Stage-A ResNet50
classifier already evaluated in S5, not an asserted winning classifier. Crops
fall back to full frame as recorded by the original S5 pipeline. No result is
silently promoted to an independently validated deployable winner. Identical
predictions or no improvement are legitimate outcomes.

Outputs: per-image probabilities for each arm,405 per-run metric rows,
135 model/fold/arm three-seed summary rows, source/code hashes and scope status.
HF namespace: `s9/exploratory-fusion-r1/<code_sha256>/`. Results are resumable
per completed source run. Normal pushes every30min, completion/catchable Stop
flush; a forced kill cannot flush unpublished work. The analysis is cheap to
repeat but never retrains a completed S5 model. Scratch holds small prediction
files only, not model weights/full datasets; one CPU worker is enforced.

Local tests passed for known-answer fusion, shuffled row alignment, invalid
probabilities and missing/duplicate prediction rejection, notebook syntax and
embedded-source equality. Public-HF tests passed on SegFormerB2 seed1 in all
three folds initially; the subsequent decoder repair was tested on all81 inputs.
The subsequent full notebook publication and independent verification are
recorded above; no further execution is needed for this analysis.

## Full S9 — deferred by user, not completed

**User decision:** set HRNet/PatchCore aside temporarily and finish the supported
study/reporting work first; revisit if worthwhile. This is not permanent removal,
authorization to claim the original pipeline complete, or evidence that the other
original research gaps are closed. No new annotation or large download is required
now. The user is willing to label later if the scientific purpose is justified
and the guidance/package are manageable.

Before requesting future work, first define which landmarks/normality criteria
the proposed component actually needs and whether the available images support
them. Provide a small pilot package (approximately10–15 images, target under20MB
where adequate visual detail permits), not a bulk300MB+ archive. Include labelled
examples, an exact checklist, ambiguous/unusable-image rules, and export/upload
instructions. Check the user's pilot annotations before scaling to small batches;
preserve image IDs and coordinate transforms and transfer annotation files rather
than duplicating original images. These are future delivery requirements, not a
claim that an annotation package or validated landmark definition already exists.

Marking a tyre as healthy requires a defensible reference criterion; image-only
or mileage-based guesses must not become verified health labels. More annotation
alone cannot establish the full pipeline claim: training, component ablations and
appropriate evaluation would still be required. Existing manual masks remain valid
for the completed S5 study.

The original design combines a learned segmenter, selected classifier, HRNet
landmarks, PatchCore anomaly evidence and fusion, with component-removal
ablations. NB18 explores only fixed fusion of already computed classification
probabilities. It does not train HRNet, PatchCore, a fusion head, or a crop-trained
classifier, run new end-to-end inference, or satisfy the original54-run budget.

| Gap | Current evidence | Required decision/evidence |
|---|---|---|
| HRNet landmarks | Tyre/tread masks exist, landmark labels do not | Supply justified labels or explicitly defer this component; mask extrema are not verified landmarks |
| PatchCore reference | Low-wear proxy labels, not a verified healthy reference pool | Verified healthy pool or a separately declared proxy experiment; no substitution without scope approval |
| Model/fusion selection | S5 descriptive existing-fold results | Freeze selection and evaluation rules before claiming a winning integrated system |
| Independent generalisation | Folds0/2 leak-flagged; fold1 tiny tyre sample | Independent tyre data or retain explicit limitation; no statistical relabelling can repair this |
| SAM2/blind consistency | User declines more annotation | Remains deferred, not passed; not required for NB18 |

No new annotation request is made by NB18. Full S9 cannot honestly be completed
without additional inputs or an explicit user-approved change of scope.

## Reporting/research gap ledger

| Item | Action now possible | Still not established |
|---|---|---|
| S5 localisation and ROI results | Use verified NB17 tables; NB18 extends fixed-probability combinations | Crop superiority/significance or an independent winner |
| Original mask-quality figure | Use S5 manual-ground-truth localisation metrics | Annotator consistency or SAM2-versus-manual study |
| H2 shortcut hypothesis | Report inconclusive/undefined with mask-coverage explanation | A newly informative test without additional variation |
| H3 fine-grained hypothesis | State untested | Preregistered missing FGVC arms |
| Calibration | Report actual achieved coverage and limitations | Promised coverage under independent new-tyre shift |
| Video temporal consistency | Document absence | Temporal study without video evidence |
| Resolution×ROI interaction | Distinguish existing separate interventions from factorial evidence | Missing crossed experiment |
| Full XAI programme | Describe completed NB07 gate accurately | All-run/all-fold/native-method/stability extensions |
| Manuscript | Assemble completed results with this scope ledger | Claim that every original experiment is finished |

Sources: `tyrelib/s9_evidence.py`, `tyrelib/build_s9_notebook.py`,
`scripts/verify_s9_evidence.py`. No old notebooks, experiment records or HF
results were overwritten in this delivery.
