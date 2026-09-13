# Results and limitations — review draft

This is a reproducible reporting package, not a completed manuscript or a claim
that the entire original proposal was executed. Labels represent mileage-proxy
classes, not measured tread depth, verified roadworthiness or alignment.

## Scope and endpoints

The retained architecture sweep contains17 architectures and153 valid runs;
nine mislabeled substitutions remain quarantined. Implemented StageB has108
runs; S4b has18 confirmation runs. S5 has81 evaluated runs at60epochs each.
NB18 analyses81 saved S5 prediction sets with fixed fusion rules; it is not the
full HRNet/PatchCore pipeline, which is deferred by user.

Classification source tables retain both selected-epoch and final-epoch fields.
Do not substitute selected-epoch scores for fixed-budget results. Inherited H1
is explicitly a legacy selected-epoch analysis. S5/S9 use the fixed final
classifier endpoint and preserve CORAL versus softmax decisions.

## Key supported results

S5 localisation and full/predicted/oracle ROI tables are bundled with their
source revision. Localisation quality alone does not demonstrate improved
classification. The fixed fusion comparisons below did not improve the mean
endpoint; averages are descriptive, equally weighted by source run, not
independent observations or significance tests.

| Fusion arm | Mean macro-F1 delta versus full |
|---|---:|
| full | 0.000000 |
| full_tyre_tread_equal | -0.013238 |
| tread_only | -0.014666 |
| tyre_only | -0.012572 |
| tyre_tread_equal | -0.015676 |

## Hypotheses and calibration

See `tables/classification_hypothesis_outcomes.csv` and `tables/conformal.csv`.
H2 is inconclusive/undefined; H3 lacks its planned model arms. Calibration coverage must be reported as achieved, not guaranteed.

## Figures

The10 inherited panels retain their original definitions; four new panels extend dense-task/fusion reporting. Figure14 is model-versus-manual agreement, NOT annotator self-consistency.

### fig01_accuracy_vs_ter

Inherited endpoint and limitations; figures8–10 are best-epoch/energy/session, not missing original experiments

![fig01_accuracy_vs_ter](figures/legacy_fig01_accuracy_vs_ter.png)

### fig02_per_fold

Inherited endpoint and limitations; figures8–10 are best-epoch/energy/session, not missing original experiments

![fig02_per_fold](figures/legacy_fig02_per_fold.png)

### fig03_saliency_panels

Inherited endpoint and limitations; figures8–10 are best-epoch/energy/session, not missing original experiments

![fig03_saliency_panels](figures/legacy_fig03_saliency_panels.png)

### fig04_stress_matrix

Inherited endpoint and limitations; figures8–10 are best-epoch/energy/session, not missing original experiments

![fig04_stress_matrix](figures/legacy_fig04_stress_matrix.png)

### fig05_h1_stability

Inherited endpoint and limitations; figures8–10 are best-epoch/energy/session, not missing original experiments

![fig05_h1_stability](figures/legacy_fig05_h1_stability.png)

### fig06_ofat_effects

Inherited endpoint and limitations; figures8–10 are best-epoch/energy/session, not missing original experiments

![fig06_ofat_effects](figures/legacy_fig06_ofat_effects.png)

### fig07_faithfulness

Inherited endpoint and limitations; figures8–10 are best-epoch/energy/session, not missing original experiments

![fig07_faithfulness](figures/legacy_fig07_faithfulness.png)

### fig08_best_epoch

Inherited endpoint and limitations; figures8–10 are best-epoch/energy/session, not missing original experiments

![fig08_best_epoch](figures/legacy_fig08_best_epoch.png)

### fig09_energy

Inherited endpoint and limitations; figures8–10 are best-epoch/energy/session, not missing original experiments

![fig09_energy](figures/legacy_fig09_energy.png)

### fig10_per_session

Inherited endpoint and limitations; figures8–10 are best-epoch/energy/session, not missing original experiments

![fig10_per_session](figures/legacy_fig10_per_session.png)

### S5 box AP50:95 by model/fold

Existing folds, three seed SD; not new-tyre validation

![S5 box AP50:95 by model/fold](figures/s10_11_box_ap.png)

### Predicted-tread ROI change versus full image

Existing folds, three seed SD; not new-tyre validation

![Predicted-tread ROI change versus full image](figures/s10_12_predicted_roi.png)

### Equal full+tyre+tread fusion change versus full image

Existing folds, three seed SD; not new-tyre validation

![Equal full+tyre+tread fusion change versus full image](figures/s10_13_fixed_fusion.png)

### Tread mask IoU against existing manual masks

Existing folds, three seed SD; not new-tyre validation

![Tread mask IoU against existing manual masks](figures/s10_14_manual_mask_iou.png)

## Limitations and uncompleted work

- Full S9 HRNet/PatchCore: Deferred by user; not executed.
- SAM2 comparison / blind consistency: Deferred; existing masks are not independent repeat labels.
- H2: Inconclusive/undefined in the implemented intervention evidence.
- H3: Untested; preregistered fine-grained arms absent.
- Video temporal consistency: Not measured; no suitable video evidence.
- Resolution x ROI factorial interaction: Not established by separate OFAT arms.
- Independent tyre generalisation: Not established; folds0/2 leak-flagged and fold1 has few tyres.
- Broader Tier5/6 and XAI extensions: Not completed by the implemented tracks.
- Manuscript submission: Requires human review, reference checking and venue formatting.

## Review checklist

- Check terminology, author contributions and source references.
- Discuss fold leakage and sample independence explicitly.
- Choose the submission scope; do not imply full original-plan completion.
- Review all figures and result tables before submission.
