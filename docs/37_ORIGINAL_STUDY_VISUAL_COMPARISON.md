# Original paper study — corrected visual comparison

Use `notebooks/NB33_Original_Study_Visual_Comparison.ipynb`.
The previous Phase 2 NB07 was the wrong scope for the requested paper illustrations.
Its executed file is preserved; it is not silently rewritten or relabelled.

## What belongs to this comparison

The existing paper describes the original `final_v1` collection and original
model experiments. The workstation's later Phase 2 replacements are a separate
study and are not substituted into these illustrations.

NB33 includes 17 retained Stage-A classifier architectures, nine S5 localisation
models and the original HRNet/matched-SegFormer geometry pair. It uses one fixed
representative fold/seed per architecture, rather than every training replica:
classification/localisation fold 1, seed 1; geometry seed 1. The incorrectly named
ConvNeXt-S runs remain excluded.

Classifier endpoint is explicit: `best` (default, original maximum-validation-QWK
checkpoint), `last` (epoch 60), or `both`. This is not a new selection using the
displayed photos. Segmentation/detection and geometry use their original final
checkpoints. The original CORAL threshold-count decision is preserved; argmax of
the displayed three class scores must not replace it.

## Run on Kaggle

1. Import NB33, select T4 ×2 (one GPU suffices), enable Internet and `HF_TOKEN`.
2. Attach `shanmuk4622/tire-dataset-prepared`, the **original** Kaggle dataset.
   Do not use `tire-dataset-prepared-phase2`.
3. Leave `DATA_ROOT` empty for detection, or set it to the original `FINAL`
   directory containing `manifests/clean_manifest.csv`.
4. Run All. Default selection is three original fold-1 validation photos per class,
   chosen in deterministic ID order and cycling through available session groups.
5. Inspect classifier score sheets, region boards and geometric overlays.
   Download the final gallery ZIP, extract it, and open `gallery.html`.

The frozen original manifest and selected image hashes are checked before
inference. Wrong data are rejected. The notebook embeds all required source and
configurations; no repository upload or Phase 2 dependency is needed at runtime.

## Outputs and interpretation

- One classifier score sheet per image, including every selected architecture,
  checkpoint epoch, actual ordinal decision and agreement with the supplied label.
- A nine-model region board: transparent masks for segmenters and boxes for
  detectors. Missing detections are labelled instead of being hidden.
- HRNet and matched SegFormer boundary diagrams with extraction coverage and
  disagreement. The geometry cohort is labelled separately because its split
  differs from the classifiers' split.
- PNGs, a light HTML gallery, classification CSV, raw NPZ predictions and complete
  input/checkpoint provenance.

These are internal validation illustrations, not external accuracy estimates.
Small example counts and validation-selected models should not be presented as
a new independent benchmark. A model can still misclassify low-use tyres; the
notebook retains failures and makes endpoint differences inspectable.

## Storage, publication and recovery

Only one model is resident on GPU 0. The notebook downloads each selected original
checkpoint at its pinned HF revision, verifies SHA256 and deletes that downloaded
checkpoint after inference. Full runs require several GB of network transfer,
but do not retain the entire checkpoint collection in the 20 GB output area.
Dataset files stay in read-only Kaggle input. Training-state checkpoints are
loaded only after validating their pinned hash; no training is performed.

Image/model outputs are written atomically. The run ID binds selected input
hashes, model endpoints and runtime hashes. Same-run restarts recover HF results
and skip completed pairs. Uploads are batched every four models, at inference
completion, every 30 minutes during inference, after rendering and on catchable
interruption/error. Retries obey Retry-After; no upload occurs per image. The
publisher uses a conservative application-call budget; actual HTTP limits still
apply. A forced kill or network failure cannot guarantee a final upload.

HF results go to `original-study/visual-comparison-v1/<run-id>/results.zip` in
`Shanmuk4622/tyre-wear-study`, separate from Phase 2. Keep the same RUN_TAG/settings
to resume; do not run concurrent writers for the same run. The source dataset,
trained checkpoints, executed Phase 2 notebook and paper are unchanged.

## Review of the executed Phase 2 NB07

The supplied notebook executed all six code cells without recorded errors. It
ran five models on nine photos and verified the final gallery upload at HF commit
`ed3bc8e3b8d939bcc883b93895b31ae8c5b7868f`.

Both Phase 2 classifiers predicted medium on all three shown low-use photos.
MobileNet assigned about 86.8–87.1% to medium; ResNet50 assigned 70.2–72.7%.
Those three low photos are from the same tyre/session, not three independent tyres.
The display therefore reflects a prediction failure on those examples, not a
missing-overlay or failed-execution problem. That successful Phase 2 execution
does not make it the appropriate experiment for the original paper.

## Maintenance

Local verification completed on 23 September 2026 in `cv_conda`:

- All 45 checkpoint references were checked against immutable HF metadata.
- 63 actual model/image predictions passed: nine photos using MobileNet best and
  final, ResNet50 final, S5 SegFormer B0 and YOLO Nano segmentation, and both
  original geometry checkpoints. Completed-pair resume preserved all outputs.
- All 17 classifier architectures passed random-weight forward checks at their
  declared input sizes. These compatibility checks are not trained predictions.
- Original MobileNet best/final and ResNet50 final predicted low for all three
  low photos shown in the executed Phase 2 notebook. The final classifiers
  predicted low on the three selected medium photos; MobileNet best predicted
  medium. Endpoint choice therefore remains visible.
- Notebook schema/cell syntax, original manifest/image hashes and rejection of
  the Phase 2 dataset passed. Low-use classification and geometry figures were
  visually inspected.

The full 28-pipeline trained inference run, corrected notebook execution on Kaggle
and its HF publication have not been tested. The ResNet best download was slow
during local checking; it was not counted as an executed checkpoint. The notebook
uses resumable HTTP downloads and preserves partial downloads after interruption.
Machine-readable checks are in `notebooks/assets/original_visual_verification.json`.

Runtime: `tyrelib/original_visual_compare.py`.
Builder: `tyrelib/build_original_visual_notebook.py`.
Frozen metadata: `notebooks/assets/original_visual_{assets,registry}.json`.
Run the builder in `cv_conda` after a runtime change. It regenerates NB33 only,
never the user's executed Phase 2 notebook.
