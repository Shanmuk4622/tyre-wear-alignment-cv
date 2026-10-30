# TREAD-ONE completed run — direct CSV recovery

These directly readable CSV files recover the original r1 training evidence and
add explicitly post-hoc metrics without retraining, changing weights, selecting
on test scores or overwriting the frozen experiment.

Source: `Shanmuk4622/tyre-wear-study`, immutable revision
`a0296f18e02b12d28f7e51377783d68538032209`.

Start with `csv/run_summary.csv`, `csv/validation_epochs.csv` and
`csv/completion_checks.csv`. `CSV_SCHEMA.json` defines columns/units/missing values.
`AUDIT.json` records cryptographic and independent recalculation checks.

Recovered: 8,820 training updates, 70,560 sampled-image exposures, 180 validation
epochs, 14,580 validation-image predictions and 618 test-image endpoint records.
Point predictions, class/calibration/confusion tables, raw-test segmentation
counts, numerical events, GPU recovery checks and provenance are separate CSVs.

Additional NLL/Brier/entropy/ECE/QWK/ordinal/class metrics and segmentation
confusion counts were derived from already saved predictions. They are not
measurements newly collected during training. Missing per-image training
predictions, continuous utilization/energy and complete validation mask maps
cannot be invented retrospectively.

All three runs completed 60 epochs and 2,940 updates. The declared export remains
seed 3 / validation-selected epoch 20. Mean selected test balanced accuracy is
34.1463%, mean Dice is 97.9777% and human boundary error is 1.18937% image width.
All selected checkpoints miss the held-out low tyre. These are two-tyre internal
results, not a deployment, measured-wear or unseen-mid/video claim.

Future HF progress/results must use directly browsable CSV plus native tensor
state/JSON manifests, never external ZIP bundles. Original historical evidence
is retained; this additive export does not rewrite it.
