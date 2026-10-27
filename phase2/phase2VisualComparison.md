# NB07 — visual model comparison

Import `notebooks/phase2_NB07_Visual_Model_Comparison.ipynb` into Kaggle. It is
self-contained and performs inference only, using the five validation-selected
Phase 2 exports already installed in Tread Station. No retraining or paper edits.

## Run

1. Select GPU T4 ×2; enable Internet.
2. Attach `shanmuk4622/tire-dataset-prepared-phase2` directly from Kaggle.
3. Enable the `HF_TOKEN` secret with write access to `Shanmuk4622/tyre-wear-study`.
4. Run all cells. Defaults select three low, three medium and three high photos.
5. View the inline figures. Download the final gallery ZIP from Kaggle Output,
   extract it, and open `gallery.html`.

For custom photos, attach an input directory containing `low/`, `medium/` and
`high/`, then set `CUSTOM_ROOT`. `PER_CLASS` accepts 1–30. The folders supply
category labels; the notebook does not verify their mileage or independence.

## What you see

- Original portrait photograph, retaining its aspect ratio.
- MobileNet V4 and ResNet50 three-class probabilities with expected-label checks.
- YOLO26 Medium and SegFormer B0 transparent tyre/tread overlays.
- HRNet points and boundary lines compared with SegFormer at the same guide rows.
- Missing/clipped intersections, crossed HRNet sides and classifier disagreement.
- An overview of probabilities across all selected examples and a light gallery.

Only the two classifiers predict low/medium/high. Segmentation models predict
regions; HRNet predicts boundary coordinates. They are not given a fictitious
shared accuracy score. Medium prepared examples are training-cohort photos and
are explicitly marked. These are illustrations, not a new generalization study.

## Storage, recovery and HF

The notebook embeds the small inference runtime and registry, then downloads only
five checksum-verified checkpoints at immutable revisions (~278 MiB). It retains
Kaggle's PyTorch installation and keeps one model resident on GPU 0 at a time;
a dual-T4 session is supported without requiring both GPUs for small inference.
Dataset files stay in read-only input storage. It checks for 2 GiB free before
loading a model; it does not assume 1 TB scratch space.

Each completed image/model pair is written atomically. A run ID includes input
hashes, labels, cohorts, model hashes, runtime hashes and the user RUN_TAG.
Unchanged runs restore remote results and skip valid completed pairs; incomplete
or corrupt local NPZ files are recomputed. Keep the same tag when resuming and
never run duplicate writers for the same run ID.

HF destination: `phase2/visual-comparison-v1/<run-id>/results.zip` in the existing
dataset repository. This includes rendered images, numerical predictions, masks,
points and provenance. It does not include tokens or duplicate model weights.
Only use custom photos suitable for your repository's visibility.

Uploads are batched after each model, every 30 minutes during inference, after
rendering, and on catchable interruption/error. Each successful upload is fetched
at its immutable commit and hash-verified. A conservative 60 application-call/hour
budget and Retry-After retries reduce rate pressure; SDK calls may involve multiple
HTTP requests and the server's actual limits still apply. A forced kernel kill,
network failure or second Stop during cleanup can lose progress since the last
verified upload. Local outputs remain available when an upload fails.

## Verification

Locally tested in `cv_conda` with the actual five checkpoints on a GTX 1650:
15 model/image pairs covering one low, medium and high image; probability validity;
strict checkpoint/source loading; resume without changing completed outputs;
missing/clipped boundary handling; notebook schema and Python cell syntax.
The figures were inspected visually. Kaggle execution and live HF publication
are not claimed as tested. See `manifests/phase2_visual_notebook_verification.json`.

The editable runtime is `phase2_visual_compare.py`. After changing it, run
`python phase2/phase2_build_visual_notebook.py` in `cv_conda` to regenerate the
self-contained notebook. Old notebooks and trained checkpoints are preserved.
