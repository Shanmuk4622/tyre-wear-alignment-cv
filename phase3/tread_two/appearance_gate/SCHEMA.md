# Appearance gate — exact scientific CSV schema

**Version1 · 30 October 2026.** Source role is train throughout. Keys refer to
the new gate registry, not NB08's 570-image integer registry. Full source IDs,
original byte hashes and role overlay/checksum connect the experiments.
The exact field order is executable in [SCHEMAS](gate_worker.py).

| Atomic family | Full-run rows | Key / purpose |
|---|---:|---|
| source_train | 386 | `image_index`; image ID/hash, class/domain, known tyre or unresolved clip group, v1 support/query roles |
| patch_quality | 12,352 | Image ×32 patches; supported fraction, intensity statistics, adjacent-pixel gradient, glare/dark/chroma fractions and low-texture flag |
| predictions | 2,702 | Image ×7 variants; label/prediction, all three logits/probabilities, NLL, support and additive residual |
| patch_evidence | 259,392 | Image ×7 ×32 ×3; signed class-logit contribution and geometric pooling mass |
| affinities | 1,556,352 | Image ×7 ×32 ×18; cosine and temperature-transformed affinity to each original trained reference |
| feature_consistency | 86,464 | Image ×7 ×32; valid/new-clipping fractions, RGB and local-contrast relative L2, local-contrast cosine and floor-scope note |
| execution | 386 | Image; dense/head device, zero optimizer/gradient counts, frozen flag, compute seconds and both device allocator peaks |
| candidate_bank | Up to18; structural target18 | Proposed index, class, image ID/hash, known tyre, atlas patch index, quality fields, train role and explicit UNTRAINED status |

With a full18-case proposal, the eight atomic families total **1,918,052 rows**.
An incomplete proposal is reported as a structural failure, not silently filled
with held-out or duplicate photos. Report-derived cohort/class/invariance/
query-coverage summaries and field dictionary are additional direct CSV files.
Source science remains unchanged in its original HF namespace.

## Field semantics and units

| Fields | Meaning / units / unavailable policy |
|---|---|
| image_index | Zero-based sorted training registry; one per original source image |
| image_id / image_sha256 | Original stable image identifier and file SHA-256; not a new image/tyre |
| class_index | 0 low,1 mid,2 high mileage proxy; not physical depth or safety truth |
| domain / physical_tyre_id / source_group | Original photo vs frame; blank tyre means unresolved parent, not a manufactured subject |
| is_v1_bank / is_v1_query / cohort | Eight original support photos vs378 original query images; support scores are reported separately |
| variant | Frozen transform name; gamma numbers are exponents, not brightness factors |
| patch / prototype | Zero-based 8×4 atlas patch index / original 18-reference index |
| valid_fraction | Supported pixel fraction within a 64×64 patch; invalid pixels are excluded from statistics |
| gray_mean / gray_std | Channel-mean intensity over supported pixels, units[0,1]; population std |
| mean_gradient | Mean absolute adjacent horizontal/vertical luminance difference over jointly supported neighbours |
| glare_fraction / dark_fraction | Supported luminance≥.98 / ≤.02; heuristic fractions, not human labels |
| chroma_fraction | Supported `(maxRGB−minRGB)/maxRGB >.2` and luminance>.3; heuristic, not a paint annotation |
| low_texture | Mean gradient<.008; flag only, not candidate exclusion or a wear label |
| logit_0…2 / contribution | Raw class-logit units; signed additive reference/patch terms plus separate bias |
| probability_0…2 | Softmax scores, dimensionless[0,1], uncalibrated |
| correct / nll | Raw argmax correctness / negative natural log of true-class probability |
| pooling_mass | Geometric band weight; supported bands have equal total mass; not learned saliency |
| cosine / affinity | Embedding cosine[-1,1]; `exp((cosine−1)/saved_temperature)` |
| evidence_residual | Maximum class absolute difference between bias+patch sums and saved logit; required<2e-6 |
| new_clip_fraction | Fraction of supported RGB channel values exceeding1 before probe clamping; direct gamma/gray do not exceed1 |
| rgb_relative_l2 | `norm(variantRGB−baseRGB)/max(norm(baseRGB),1e−6)` on supported pixels |
| local_contrast_relative_l2 | Same relative norm using four masked local-contrast channels |
| local_contrast_cosine | Flattened feature cosine with eps1e−8; zero-feature cases use library epsilon convention, interpreted with support/floor scope |
| local_std_floor_scope | Explicit note that floor.02 and clipping limit exact equivariance; not a learned parameter |
| compute_seconds | Per-image compute/record preparation before atomic store commit; not total wall-clock duration |
| cuda_peak_allocated_dense_bytes / head_bytes | PyTorch allocated-tensor peaks on the named devices; same device in one-GPU engineering runs |
| optimizer_updates / parameters_with_grad | Always0; failure of this condition stops the worker |
| candidate status | Always UNTRAINED proposal; original v1 coefficients are not compatible by assumption |

Empty supported patches have unavailable intensity/quality statistics. Numeric
zero remains zero. No missing human/weak target is replaced with fabricated
truth: this gate does not compute new mask or point-ground-truth error.

## Native and administrative files

Each image retains float32 original atlas RGB, valid mask, sampling grid,
frozen dense masks/points, 32×256 descriptors, quality fields, all probe scores
and source hashes in a direct `.pt` packet. Candidate RGB and four-channel
local-contrast patches are direct `.npy` arrays. Original model weights remain
pinned in the original experiment; their hashes form this gate's contract.

`state.pt`, pointer, unit ledger, HISTORY.json, SCHEMA.json, STATE_VERSIONS.csv
and COMMIT_TIMINGS.csv retain exact image progress and CSV/native boundaries.
Current/prior control logs and audited transport source/history accompany
snapshots. These operational/timing rows are separate from the eight scientific
families and are not counted as additional independent examples.
