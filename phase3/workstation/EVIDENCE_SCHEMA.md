# Desktop inference evidence — direct CSV and native arrays

Schema: **tread-one-station-capture-v1**, 29 October 2026.
Each capture includes `CSV_SCHEMA.json` with its exact headers, units, grains,
row counts and raw-array shapes/dtypes, plus a SHA-256/byte manifest.

## Explicit full capture

| CSV | Rows | Grain / values |
|---|---:|---|
| `prediction.csv` | 1 | Native frame/source identity, model/weights/seed/epoch, three probabilities, selected label, timing/device/precision, flags, optional note and missing-reference metrics |
| `class_scores.csv` | 3 | Class, probability, logit, bias, positive/negative/spatial evidence sums and reconstruction error |
| `points.csv` | 6 | Ordered point index, guide, side, x/y fractions and native pixels, source dimensions and scope |
| `regions.csv` | 2 | Overlapping region name, native positive pixels/fraction, enclosing box, threshold and dimensions |
| `latent_mass.csv` | 12 | Latent index, mean activation and normalized-gate-weighted mass; no named damage label |
| `mask_probabilities.csv` | 24,576 | 2 regions × 128 rows × 96 columns; sigmoid probability |
| `mask_logits.csv` | 24,576 | Same grid, pre-sigmoid segmentation logits |
| `signed_evidence.csv` | 36,864 | 3 classes × 128 × 96 signed class-logit contributions |
| `latent_textures.csv` | 147,456 | 12 latent channels × 128 × 96 nonnegative activations |
| `tread_gate.csv` | 12,288 | 1 × 128 × 96 normalized tread weighting |
| **Total** | **245,784** | 245,760 dense values + 24 scalar/geometry/channel records |

The five dense tables share `channel,row,column,value`. Class channels are
0/1/2 = low/mid/high mileage proxy; mask channels are 0 tyre / 1 tread.
Dense coordinates are the **quarter grid**, not native source pixels.

Ten `.npy` files retain the exact output arrays: probabilities, logits, bias,
masks, segmentation logits, points, signed contributions, concepts, tread gate
and guides. They use native NPY with pickle disabled. The PNG files are native
oriented input, actual model-input RGB, illustrative combined overlay, native
tyre mask and native tread mask. `CAPTURE.json` retains complete source/model/
environment/application-source identity, note and interpretation scope.

## Scalar meanings and missing data

| Field / group | Meaning |
|---|---|
| `frame_sha256` | Hash of decoded oriented native RGB bytes; distinct from original source-file SHA |
| `source`, `media_kind`, `frame_index`, `timestamp_seconds` | Original path/reader kind and decoded-frame location; timestamp derivation is in source metadata |
| `native_width`, `native_height` | Oriented source dimensions |
| `model`, `seed`, `epoch`, `weights_sha256` | Pinned model identity, never inferred from filename or test score |
| `probability_low/mid/high`, `probability` | Softmax fractions, sum approximately one; not accuracy, depth or wear measurements |
| `predicted_class`, `predicted_label` | Argmax of those same retained probabilities; no UI relabelling |
| `logit`, `bias`, signed evidence fields | Additive class-logit units; bias is not a spatial pixel |
| `evidence_residual`, `reconstruction_error` | Difference between reconstructed and emitted class logit; full worker maximum ≤1e-5 |
| `model_ms`, `preprocess_ms` | Worker wall time in milliseconds, synchronized around GPU execution; not a full application/codec/export benchmark |
| `device`, `precision` | Actual worker device; FP32/no autocast/TF32 disabled |
| `review_flags` | Uncalibrated empty/clipped tread-row or >5%-width point/mask disagreement hints |
| `reference_proxy_class`, `reference_scope` | Optional known existing-example label and cohort; arbitrary files are unlabelled |
| `accuracy`, `dice`, `point_error` | **Blank**: desktop inference does not consume reference masks/points or compute a benchmark |
| `metrics_missing_reason` | Explicit explanation rather than a fabricated zero metric |
| `note` | Optional user text; raw original note retained in JSON; formula-like CSV text is apostrophe-prefixed |
| `x_width_fraction`, `y_height_fraction` | Native coordinate / (dimension−1), separate from box-exclusive bounds |
| `positive_native_pixels`, `native_area_fraction` | Native display mask pixels and fraction; overlapping masks are counted independently |
| `x0,y0,x1_exclusive,y1_exclusive` | Enclosing native mask box, blank when empty |
| `gated_mass`, `mean_activation` | Latent feature summaries; no human concept/depth label |

Blank means unknown/not applicable. Numeric zero remains an observed zero.
All model arrays are checked finite before acceptance. Capture metadata records
that UI predictions are inference evidence, not newly collected independent
tyres, new human truth or a new evaluation study.

## Automatic session retention

`SESSION.json` identifies this process/session and checkpoint. Every completed
prediction appends one prediction, three class rows, six point rows, two region
rows and twelve latent summaries. Each subordinate row carries the UTC and
native-frame SHA. `events.csv` records open/model-ready/source/save/restore/error/
close events. CSV writes are flushed/fsynced per group.

Session tables append independently; forced process death can leave the last
cross-table group incomplete. Join on UTC/frame SHA and inspect event records.
A completed full capture has the stronger all-file schema/manifest check;
`.partial` folders are not complete captures.

A completed result for a source that was changed while inference ran is still
logged with its original identity; it is not displayed on the newer image.
Raw full-grid arrays for unsaved video snapshots are intentionally not retained.
Use **Save capture + CSV** to preserve them. This bounds storage; no promise
of complete all-frame video telemetry is made.

Training optimizer/RNG/recovery states are not desktop inference outputs.
Earlier training retains its own HF evidence. Desktop captures stay local;
there is no token lookup or automatic HF upload in the application.
