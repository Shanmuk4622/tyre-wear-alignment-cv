# HRNet training and report — verified completion

<!-- current-status:start -->
> **Current status (15 September 2026):** [Completed work and remaining validation](CURRENT_STATUS.md). The report is refreshed; the app, learned-geometry integration and target-assisted alignment software exist. Dated plans below retain their original context.
<!-- current-status:end -->

15 September 2026. **NB28 and NB29 completed; no rerun needed.**

Final training commit: `96166fb19f9bf63d1acb2bd2ee8b55b0ffdacadd`.
Report commit: `a92c0f9c5c1b78c6a06e13d51e18722195230658`.
Protocol: `351c6638783996f46cf9efd99ec6689de726e045288054f240193cea61385712`.
[Public results](https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/tree/a92c0f9c5c1b78c6a06e13d51e18722195230658/s9/hrnet-geometry-2026-09-15-r1/351c6638783996f46cf9efd99ec6689de726e045288054f240193cea61385712).

## Verification

Both saved notebooks finish successfully without exception outputs. Three public
seed statuses are completed at epoch 60, cursor zero. Each history has exactly
2,160 finite-loss optimizer-step records in the expected 60×36 order: 6,480 total.
Each run lists all 60 validation files. Published checkpoint LFS hashes match
STATUS; checkpoint tensors were not downloaded or independently re-inferred.

Recomputed every test-point error (144 per seed, 432 records across repeated
evaluations of the same 24 images), checked labels, image/point coverage, per-tyre
and aggregate scores, and independently recomputed the training-only constant
baseline. Report results and repair source hashes agree with run artifacts.
The recorded GPU overflow regression passed; real training logs show AMP recovery
and subsequent completion. Runtime repair provenance is retained, not erased.
Downloaded only 702,000 bytes of metadata/results; no HF writes or model downloads.
[Machine audit](../outputs/hrnet_final_audit/AUDIT.json).

## Results

| Seed | Mean coordinate error / image width | Mean native-pixel error | Median native-pixel error |
|---|---:|---:|---:|
| 1 | 1.391% | 16.01 | 8.75 |
| 2 | 1.284% | 14.77 | 9.09 |
| 3 | 1.261% | 14.52 | 8.31 |
| Mean across seeds | **1.312%** | **15.10** | Not pooled here |

The constant-coordinate baseline (six means estimated only from training labels)
has 3.555% mean width error. HRNet's seed-average error is about **63.1% lower**
than that baseline. This is a descriptive comparison, not a significance claim
or a comparison against segmentation. Do not select seed 3 as the winning model
solely because it has the lowest held-out error; all three seeds are reported.

Across seeds, error averages approximately 0.51% on the held-out high-mileage tyre
and 2.11% on the held-out new-tyre session. There are only **two held-out tyres**,
with identities confirmed by the user. The result demonstrates a completed
coordinate-learning experiment, not robust deployment across tyre populations.
Old pilot images from these identities informed development; this is not an
untouched external cohort. Mileage names are not measured wear/health labels.

All submitted target points were visible. HRNet predicts six coordinates at fixed
guide rows; it has not learned or validated invisible-boundary rejection, physical
camber/toe, tread depth, tyre safety, or anomaly/healthy-reference classification.
NB23's earlier segmentation score used different images/splits and must not be
ranked directly against these HRNet scores.

## Next experiment — no new annotation requested

**Now complete:** [NB31/NB32 matched results](36_MATCHED_GEOMETRY_RESULTS_AND_INTEGRATION.md).
HRNet 1.312% vs SegFormer 1.721% width error, same two test tyres; no rerun.
The next action is integration validation, superseding the comparison-build plan below.

Implementation now available: [NB30–NB32 matched SegFormer comparison](35_MATCHED_SEGFORMER_COMPARISON.md).
Run CPU preflight, GPU smoke/training, CPU paired report. Local checks passed;
no SegFormer comparison result is claimed before Kaggle execution.

1. Audit the existing segmentation training sets against the frozen HRNet split.
   Build a matched same-split SegFormer baseline if existing runs saw any of these
   held-out tyres. Keep evaluation labels and metric definitions identical; record
   any extra dense-mask supervision rather than claiming equal supervision.
2. Compare paired errors, per-tyre behaviour, latency and memory. Do not tune on
   these already inspected test results and call them a fresh confirmatory test.
3. Decide whether HRNet adds useful value. Integrate only the supported geometry
   component, then evaluate component removal under an explicitly frozen scope.
4. PatchCore reference evidence and physical-angle validation remain separate
   unresolved workstreams. Update the manuscript after the fair comparison.

No NB26–NB29 rerun is required for the verified experiment. Full S9 is not complete.
