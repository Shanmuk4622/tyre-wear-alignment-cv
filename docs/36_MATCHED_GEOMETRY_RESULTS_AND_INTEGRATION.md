# Matched geometry results and next integration plan

<!-- current-status:start -->
> **Current status (15 September 2026):** [Completed work and remaining validation](CURRENT_STATUS.md). The report is refreshed; the app, learned-geometry integration and target-assisted alignment software exist. Dated plans below retain their original context.
<!-- current-status:end -->

**Prototype reconciliation:** The user's subsequent local work has already
implemented and tested learned-geometry integration. The proposal to build NB33
below is historical, not the current next task. The refreshed main report includes
these results and the local app/target-alignment evidence. See [current status](CURRENT_STATUS.md)
and `prototype/LEARNED_GEOMETRY_LOG.md`. Next research work is deployment-domain,
end-to-end and physical validation, not duplicate software integration.

**Subsequent workstation update — 15 September:** direct prototype integration
and local adapter/GPU/UI validation are now implemented. Both seed-1 final-epoch
weights were downloaded and strictly loaded; image/video overlays and evidence
export are available. See the [integration log](../prototype/LEARNED_GEOMETRY_LOG.md)
for actual tests and observed video failures. The NB33 proposal below was not
implemented as a notebook; no new training or aggregate accuracy result is claimed.

## Verified outcome — 15 September 2026

**NB31 and NB32 are complete. No NB30–NB32 rerun is needed.**
HF's current revision at audit was `bbe586c6f00cf12ae4cac8b2e9cb4f875b272abb`,
which contains NB32's comparison. Its training source is
`aeee706c3aa60dad2706fff56939d7ed86ab01ca`.
Namespace: `s9/segformer-matched-2026-09-15-r1/8013bf1e6418e6f884a353efa7ce892c31fbaac450f9d7caede636d9e61e2a18`.

The executed NB31 output is incomplete: portions of the epoch log and final
completion messages are missing. This is not used as proof of failure or success.
HF independently verifies all three seeds, each 60 epochs and 2,160 finite
optimizer-step records: **6,480 steps total**, 180 validation metric files,
three final test files and checkpoint LFS hashes matching their STATUS metadata.
The repaired Tesla T4 smoke test passed with **0.0 maximum parameter difference**.
Each run's repair-source hash matches the local deterministic adapter.

NB32's published results were recomputed from 432 paired point records. Both
models use exactly the same 24 test images and two test tyres, the same targets,
and the same frozen training/validation/test allocation. All SegFormer points
were accepted: **100% boundary coverage; zero fallback substitutions**.

Only about **1.01 MB** of JSON/metadata was downloaded. No model weights or
datasets were downloaded. Checkpoint verification here means metadata/LFS-hash
agreement, not independent tensor loading or repeat inference.
Reproducible audit: `scripts/audit_segformer_completion.py`.
Evidence: `outputs/segformer_completion_audit/AUDIT.json` and its source files.

## What the experiment found

Lower error is better. Error is absolute horizontal point error divided by
native image width minus one; points are on three fixed guide rows.

| Seed | HRNet error (% width) | SegFormer error (% width) |
|---|---:|---:|
| 1 | 1.391 | 1.788 |
| 2 | 1.284 | 1.705 |
| 3 | 1.261 | 1.668 |
| Mean across three seeds | **1.312** | **1.721** |

Mean pixel errors are **15.10 px HRNet versus 19.80 px SegFormer** at the native
image dimensions: an absolute improvement of 4.70 px, or **23.75% lower mean
error relative to SegFormer**. The percentage-point difference is 0.409; do not
describe the result as 23.75 percentage points or an accuracy/F1 increase.

HRNet has lower mean error on **both test tyres in each seed**. This is not a
claim that every individual point is better. SegFormer has no missing points in
this comparison, so its published fallback-assisted and raw scores coincide.

## Supported conclusion and limits

HRNet is a justified **candidate geometry component for prototype integration**:
the dedicated six-point model improved point localisation under this matched
split and budget. SegFormer remains useful for dense tyre/tread segmentation;
this result does not say to replace segmentation with HRNet.

This is not a pure architecture experiment: SegFormer learned dense masks and
HRNet learned point labels, with different pretrained backbones. Only two
user-confirmed physical tyres are in the test set; identity was not independently
verified. Earlier pilot work and inspected test results mean this is exploratory,
not untouched external validation. Do not make a significance claim by treating
432 correlated points as independent samples. Mean-of-seeds results are not an
evaluated ensemble. Do not select seed 3 because its test error is lowest.

These are image-space boundary points, not measured camber/toe, physical angles,
tread depth, healthy/anomalous condition, or road-safety decisions. The experiment
does not demonstrate an improvement in downstream wear recognition or fusion.
No matched hardware latency comparison has yet been completed.

## Next — integration validation, not another training sweep

1. **Freeze a deployable checkpoint policy.** Use a predeclared seed or existing
   validation-only selection rule; do not select on these test results. Record
   the exact checkpoint, preprocessing, point order and numerical adapter.
2. **Build an integration-validation notebook** (proposed NB33; not implemented
   in this verification turn). Load the selected HRNet and the existing
   segmentation component, preserve the trained full-image input convention,
   map points back to native pixels and produce overlays. Do not silently crop
   the HRNet input: that changes the coordinate distribution.
3. **Measure operational behaviour.** Check image dimensions/orientation, invalid
   images, coordinate ordering/bounds, repeatability, latency and peak memory on
   the same hardware. HRNet predicts points unconditionally; its softmax outputs
   must not be called calibrated confidence or validated visibility rejection.
4. **Integrate behind an optional prototype switch**, after validation, preserving
   the user's existing geometry/calibration work. A diagram or image-space slope
   is not a calibrated physical wheel-alignment measurement.
5. **Evaluate component removal/end-to-end effects and refresh the manuscript.**
   Keep the split/selection rule frozen. Report whether the additional component
   improves the intended output, not only point error. Incorporate this addendum
   and the integration evidence into the illustrated report after review.

No new annotations or model retraining are currently requested. This is a plan,
not a claim that NB33 or prototype integration already exists. The user’s
prototype files were not modified during this audit.

## Stage status

- Matched HRNet/SegFormer geometry experiment: **complete and HF-verified**.
- S9 learned geometry integration and end-to-end component evaluation: **next**.
- PatchCore: still requires independently justified reference data and evaluation.
- Physical-angle claims: still require calibration and ground-truth validation.
- Original full S9/Tier8 design: **not complete**.
- S10: previous report exists; these new results and future integration evidence
  require a manuscript/figure refresh, followed by author/venue review.
