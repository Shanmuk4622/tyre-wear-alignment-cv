# ATLAS v1 — retained evidence schema

**Implemented, 29 October 2026.** [schemas.py](pilot/schemas.py) defines 21
transactional CSV families. The direct [column dictionary](pilot/assets/COLUMN_DICTIONARY.csv)
defines **247 table-column combinations**, including types, units and missing
values. Actual per-shard row counts and columns are written into each snapshot's
`SCHEMA.json`; every file has a byte size and SHA-256 in `MANIFEST.json`.

## Keys and time semantics

| Key | Meaning |
|---|---|
| `run` | Short prefix linked to the full immutable contract SHA; never the only provenance record |
| `image_index` | Fixed source registry key, joining original image/tyre/domain/role and hash |
| `sample` | Monotonic query-observation key across training, validation and diagnostics |
| `update` | Completed update associated with the observation; training inputs precede that step |
| `parameter_update` | Actual parameter-state cursor: a training sample at update u uses u−1 |
| `learner_sha256` | Raw named trainable-parameter fingerprint; anchor/bank/source identities live in the contract |

Baseline anchor scores always use the original frame, including during
query-atlas diagnostics. Raw scores are uncalibrated. `NA` is an unavailable
denominator/target/counter; it must not silently become zero. Class indices
0/1/2 are low/mid/high mileage proxies. Box maxima are exclusive.

## Transactional tables

| Family | Granularity and retained data |
|---|---|
| `source_images` | All 570 metadata records; original roles, hashes, source identity, bank/query flags |
| `bank_cases` | All 18 literal cases, source IDs/tyres, patch/support/clipping and inclusion/risk policy |
| `bank_pixels` | All 221,184 exact float32 reference RGB values |
| `predictions` | Every occurrence: candidate and frozen-control logits/probabilities, target/prediction, loss/Brier/entropy, support, intervention and fingerprint |
| `sample_exposures` | Every training draw: class/domain/group route, probability, occurrence, seed and exact augmentation factors |
| `affinities` | All 32×18 query/reference cosine and kernel values per occurrence |
| `patch_support` | All 32 patch-validity fractions and pooling masses per occurrence |
| `patch_evidence` | All 32×3 signed class contributions per occurrence |
| `coefficients` | 54 signed weights +3 biases +1 temperature at initialization and every completed update |
| `reference_embeddings` | All 18×96 features at the first evaluation batch for each immutable epoch/endpoint checkpoint |
| `train_updates` | Actual objective/LR, gradient norms, before/after fingerprints, timing and memory/storage counters |
| `parameter_gradients` | Every trainable parameter tensor: elements, norm, mean, min/max and finite flag |
| `dense_pixels` | Every unique queried image's 2×128×96 raw dense probabilities, retained once under the fixed anchor contract |
| `atlas_rows` | Every unique queried image's 512 inverse row maps and support fractions |
| `dense_regions` | Full-resolution unignored pixel sufficient statistics, Dice/IoU and mask-derived boxes/box IoU |
| `dense_points` | All predicted/target points, guide rows and provenance; human/weak comparability explicitly flagged |
| `metrics` | Cohort/domain/known-tyre/source-group class/task summaries and missing denominators; original-frame frozen-control comparison |
| `confusion` | Full 3×3 counts at every declared group/endpoint |
| `calibration_bins` | Ten fixed equal-width confidence bins, counts/correct/confidence sums and descriptive ECE inputs |
| `selection` | Every epoch's BA/NLL, ordering rule, selected status/epoch, checkpoint and fingerprint |
| `explanation_checks` | Selected/final per-class input/projected signed mass, separate bias, raw logit, residual and unsupported mass |

The bank photos are support inputs and excluded as queries. Dense caching
therefore covers images actually queried, not a claim that all 570 images were
inferred during fitting. Unknown video parents remain unknown; additional
patch/pixel rows do not increase physical tyre count.

## Additional direct files

| File | Meaning |
|---|---|
| `COLUMN_DICTIONARY.csv` | Direct schema/type/unit/NA definitions, also retained in exact source assets |
| `STATE_VERSIONS.csv` | Unit/update/phase, native SHA and honest local native-presence flag; older state hashes do not imply older binaries remain local |
| `COMMIT_TIMINGS.csv` | CSV, artifact/native, pointer and total commit timing; lifecycle telemetry outside the scientific cursor |
| `report/REFERENCE_CASES.csv` | Derived complete sample/endpoint/image/class/reference signed contributions; six columns, in logit units |
| `LIFECYCLE.jsonl`, `NUMERICAL_EVENTS.jsonl` | Starts/stops/completed units/errors and last durable progress |
| `REMOTE_HISTORY.jsonl`, `HTTP_REQUESTS.jsonl` | Typed request/publication/backoff history with safe headers; no token values |
| `session_history/`, `control_history/` | Prior verified logs and preparation/preflight controls retained across sessions |
| `state.pt` | Complete next-update/evaluation recovery state, bound to CSV cursors and source/data identities |
| `artifacts/epochs/*.pt` | All 60 native candidate checkpoints, with selected/final references kept separately |
| `artifacts/dense/*.pt` | Exact deduplicated raw masks/points/guides for queried source images |
| `artifacts/endpoint_maps/*/*.pt` | Full selected/final atlas/grid/validity, masks, points, signed native projection and actual case-evidence components |
| `report/*.png`, `report/ATLAS_Evidence_Explorer.html` | Durable curves and complete interactive endpoint review, derived without another fit |

Lifecycle timing can miss its last row if a process dies after the scientific
pointer commits and before timing output completes. Such a gap does not alter
the native/CSV scientific cursor. Publication logs cannot include verification
of their own final commit inside that same commit; the immutable publication
marker and pinned revision record that boundary. Pending local/remote gaps
must remain explicit.

## Planned full-run volume and resource limits

| Retained observations | Planned count |
|---|---:|
| Training occurrences | 23,520 |
| Full validation predictions | 4,860 |
| Selected/final endpoint images | 206 |
| Fixed endpoint diagnostic predictions | 824 |
| All query observations | **29,410** |
| Raw patch/reference affinities | **16,940,160** |
| Raw patch/class evidence rows | **2,823,360** |
| Coefficient rows | **170,578** |

Counts assume an uninterrupted completed protocol; recovery replays an
uncommitted unit without duplicate scientific rows. Byte volume depends on
actual output values, native snapshots and metadata; the live gate measures
and extrapolates it before fitting. Full raw activations/per-element gradients,
continuous GPU traces and every intermediate optimizer binary are not promised.
Per-parameter gradients, sufficient statistics, full requested query evidence,
epoch checkpoints and current complete recovery state are retained instead.

There is no ZIP/NPZ/database upload. Direct CSV/native shards are batched and
deduplicated into immutable HF objects. Working snapshots are bounded; scientific
CSV/history is preserved. [Recovery/run guide](PILOT_RUN_GUIDE.md).
