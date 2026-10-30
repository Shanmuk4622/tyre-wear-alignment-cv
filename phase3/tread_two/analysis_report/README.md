# Completed ATLAS pilot — evidence review

**Post-hoc review; no training, new inference or new checkpoint selection.**
The completed seed-1 fit used 60 epochs / 2,940 updates. Selected epoch 18 and
final epoch 60 both have 50% balanced accuracy / 39.806% accuracy on the 103-image
existing endpoint. All 62 low targets are predicted mid; all 41 high targets high.
This corrects an earlier prose statement that all predictions were high.

The saved original-frame frozen control has the same confusion and NLL 1.38694.
The selected candidate improves NLL to 1.17830, while final worsens to 1.42830.
That probability-score difference does not establish calibration or correct
recognition of the low tyre. Frozen-control rows are retained in cohort_metrics.csv.

Training online sampled accuracy reaches 100% from epoch four, while validation
BA stays 50% in every epoch. All 378 query images were used. Grayscale both
query/bank reduces endpoint BA to zero at both checkpoints; query dimming ×0.8
also reduces final BA to zero. This exposes appearance sensitivity, not a
unique causal attribution to paint, light, identity or capture conditions.

Saved affinities, pooling, signed coefficients and biases reconstruct all 1,030
endpoint/probe logits within 2e-6. Low endpoint images match the mid references
much more than the low references. Dense anchor metrics remain strong but do
not validate the new classifier: test mean region Dice 98.418%, human point
MAE 1.196% image width on 25 annotated images. No mid target is held out.

| Open | Purpose |
|---|---|
| [Interactive explorer](ATLAS_Failure_Explorer.html) | All 1,030 image/intervention records, 18 literal references, signed patches, paired scores and filters |
| [CSV inventory](CSV_INDEX.md) | All derived table row counts and field dictionary |
| [Methods](METHODS.md) | Exact definitions, denominators, source boundaries and interpretation |
| [Fit and confusion](figures/fit_and_confusion.png) | Training/validation history and exact endpoint errors |
| [Probes and affinities](figures/probes_and_affinities.png) | Fixed interventions and pooled literal-reference matching |
| [Exposure/embedding/calibration](figures/exposure_embedding_calibration.png) | Repeated image use, literal bank geometry and raw score calibration |

Original input provenance is pinned to HF revision
`82e7e1c5774f54c786571cdef7b3b60c1e96630f`. ANALYSIS.json records the consumed
subset and verification scope. Not every large source CSV family is recounted.
Images are the original endpoint thumbnails; atlas thumbnails remain the base
atlas even when the selected numbers describe a probe. No probe image is
fabricated. Exact additive accounting is not causal explanation or safety proof.

The model is not supported for classifier promotion by this pilot. Review
appearance invariance and training-only reference diversity before freezing
any new recipe; do not train longer or choose another epoch from the old test.
Both workstations and original checkpoints remain untouched. No additional
tyre photography is required for the current scope. NB09 publishes this review
additively as direct CSV/JSON/PNG/HTML/source, never a data archive.
