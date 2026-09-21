# Tread Station

> **Current — 21 September 2026:** Phase 2 models are installed in Tread Station: MobileNet V4, ResNet50, YOLO26 Medium, SegFormer B0 and HRNet. All fifteen training runs and final NB06 are HF-verified complete. Installed model selections remain unchanged. GPU/video/UI checks passed. Restart the existing launcher. [Results, limitations and current instructions](../phase2/workstation/README.md). Old checkpoints remain available for rollback. Earlier dated status sections below are historical.

<!-- current-status:start -->
> **Current status (15 September 2026):** [Completed work and remaining validation](../docs/CURRENT_STATUS.md). The report is refreshed; the app, learned-geometry integration and target-assisted alignment software exist. Dated plans below retain their original context.
<!-- current-status:end -->

**Learned geometry is available:** HRNet + Matched SegFormer overlays on the exact
photo/video frame, with six points, tread-width guides, a centreline, temporal
smoothing and disagreement flags. Choose the learned-boundary mode in the recipe;
**Explain learned overlay** shows the diagram and raw coordinates. Both models
are shown by default when their weights are present. Read
[LEARNED_GEOMETRY_LOG.md](LEARNED_GEOMETRY_LOG.md) for verification and video limits.

New: **Compare boundaries + rim** and **Calibrated alignment**. See
[CALIBRATED_ALIGNMENT.md](CALIBRATED_ALIGNMENT.md) for printable targets, camera
calibration, live target measurements, reference comparisons and evidence export.

A native desktop inspection console for the tyre study. It loads four original final-epoch checkpoints plus two optional learned-geometry checkpoints from Hugging Face and runs inference with PyTorch in `cv_conda`. No training is performed.

The workflow is **bring a tyre into view → freeze → compare → preserve the evidence**. The original/overlay wipe lets you inspect region boundaries directly. The comparison bench runs all four models on exactly the same pixels. An evidence tray restores earlier inspections without rerunning a model.

The workstation uses light mode only. For portrait playback, the expanded video controls, labelled SegFormer assistance and tests on your three clips, see [VIDEO_NOTES.md](VIDEO_NOTES.md).

## Start

Double-click **Launch Tread Station.cmd** in this folder. The launcher activates `cv_conda` and opens the desktop application. No browser or server is required.

Alternatively, from a Conda-enabled terminal:

```powershell
conda activate cv_conda
cd 'D:\Documents\norse\web Applicarion\Tyre\prototype'
python app.py
```

If PowerShell only finds `conda.bat` and activation does not change Python, initialize the hook in that terminal first:

```powershell
(& conda 'shell.powershell' 'hook') | Out-String | Invoke-Expression
conda activate cv_conda
python app.py
```

`python app.py --device cpu` explicitly selects CPU. The default uses CUDA when available, otherwise CPU. A model failure is shown; it never silently substitutes another architecture.

## Inspect

1. **Photo / video** opens your own image or video. **Study sample** opens the existing fold-1 image directory without duplicating the dataset. You can put additional photos in `data/incoming/` or select them from anywhere.
2. For a camera, select its number (usually 0) and click **Connect**. The source preview runs independently of inference.
3. Press **Enter** to inspect. For video/camera this holds a frame. **Space** freezes/resumes the source. Videos start live analysis automatically and briefly hold the first frame while models warm up. The cadence control is a ceiling, not guaranteed FPS. Busy inference skips frames rather than accumulating a queue. The result canvas shows the exact analysed frame, never an old mask on a newer frame; its inset shows the current source preview.
4. Drag the vertical divider across the image to compare original pixels with the region overlay. Amber is tyre, mint is tread. These regions overlap by design.
5. **Compare all four on this frame** runs both classifiers and both region models sequentially. The overlay selector switches the region model. Classifier agreement and region overlap are observations, not correctness scores.
6. Add an inspection note and press **Ctrl+S**. Click a tray entry to restore it. Double-click it to open its HTML evidence card. **Open saved records** opens the results folder.

Video is displayed and processed vertically using the file's orientation metadata; there is no model-driven rotation switching. Use **Previous/Next frame** to pause and inspect a particular moment, **−5 s / +5 s** for larger jumps, **0.25×–2×** for playback speed and **Loop video** for repeat playback. The controls below the image switch original/overlay/wipe views, show or hide tyre/tread layers, and adjust overlay strength. **Fit / 150% / 200% / 300%** zooms the display; drag to pan while zoomed in Original or Full overlay mode. They change the display, not the model input or predicted masks.

Records contain `frame.png`, separate binary tyre/tread masks, model overlay images, `inspection.json` and `card.html`. JSON includes scores, model revisions and hashes, device, package versions, timings, capture hints and the operator note. Keep a record folder together when sharing its HTML card. Records are local; there are no automatic uploads.

### Download the overlay (updated 21 September 2026)

- **Download shown frame** writes a PNG directly to your Windows Downloads folder. It saves the exact analysed pixels with the current Original / Full overlay / Compare wipe setting, including the wipe position. Zoom, UI controls and the live preview inset are excluded.
- **Download video** exports the entire opened clip at the **Video export FPS** you choose (1–60, including fractional rates; default 10). It creates a silent MP4 and preserves duration to within one output frame. Rates above the source repeat frames; higher rates take longer. Portrait orientation and maximum 1280-pixel analysis size are preserved; odd dimensions receive a one-pixel encoding border. The optional **Include video information panel** appends a footer below the image with time, export FPS, region model, mileage-proxy prediction/model score, tread tilt, raw widths and point-review status. It adds height without hiding or stretching the image.
- The video uses the recipe, region selection, opacity, visible layers, geometry, FPS, information-panel and learned-overlay controls captured when you click. It exports the full overlay without the display wipe or zoom. With automatic region selection, separately labelled SegFormer assistance can take over on frames where YOLO misses. Changing controls or opening another source does not change the running export.
- Export runs in a separate process using PyTorch on CPU (two threads), leaving the GPU available to the workstation. It can be slower than playback. Progress and **Cancel export** sit below the evidence button; cancellation removes the unfinished video. Finish or cancel before closing the app. Files receive unique names and are published only after the encoded frame count is checked. Camera feeds support frame downloads; full-video export requires a video file.

**Why an edge diagram can have no fitted lines:** the image fitter deliberately withholds geometry when its evidence is insufficient. For example, a predicted mask that touches the side of the frame produces “Side boundary clipped by frame”, even if the visible tyre appears to fit inside the photo. The diagram now shows the amber mask outline, red clipped frame edge, and an explicit **FIT WITHHELD** panel. Capture both sides with a background margin or inspect another frame. Mask outlines and learned point proposals remain separate from an accepted image-edge fit; they do not establish alignment accuracy.

Checks: `python check_media_export.py` covers sampling, PNG view fidelity, cancellation and failure cleanup. `python check_media_export.py --real` exercises a short portrait clip from the supplied videos with real models and the background export process. Local results are under `results/export-check/`.

## Models and interpretation

| Model | Role | Endpoint |
|---|---|---|
| MobileNet V4 | Default full-image ordinal mileage-proxy classifier | Stage A, fold 1 / seed 1 / epoch 60 |
| ResNet 50 | Comparison classifier | Stage A, fold 1 / seed 1 / epoch 60 |
| SegFormer B0 | Default tyre/tread segmentation | S5, fold 1 / seed 1 / epoch 60 |
| YOLO26 Nano segmentation | Alternative region model | S5, fold 1 / seed 1 / epoch 60, EMA |

These are the study's fine-tuned weights, not generic ImageNet/COCO predictions. The immutable registry is `registry.py`; checkpoints are downloaded from the dataset repository `Shanmuk4622/tyre-wear-study`.

Classification retains the study's preprocessing, checkpoint resolution and CORAL threshold-count decision. A CORAL decision can differ from the largest displayed class score. SegFormer retains the study's two-stage bilinear interpolation and two independent sigmoid masks at 0.5. YOLO retains the 512-pixel input, 0.001 prediction threshold and 0.25 mask inclusion threshold, using the final EMA weights. Classifiers operate on the full image: the study did not support enabling crops or fusion by default.

**Low / mid / high describe mileage proxies.** They do not measure tread depth, physical wear, alignment or roadworthiness. Scores are uncalibrated. Capture hints are simple brightness/focus heuristics, not a validated rejection system. Non-tyre images can still receive a classifier label. Region overlap compares two models, not predictions against ground truth.

The study images are internal data. Fold 1 is a small internal comparison; folds 0 and 2 carry the documented suspected tyre-identity leakage concern. New photos are useful for qualitative testing; external accuracy requires independently labelled, tyre-disjoint data.

## Setup or repair

The required additional packages are installed locally under `.vendor/`, leaving the existing CUDA PyTorch installation intact. The desktop toolkit is isolated there too, because the existing Conda Qt installation failed its DLL import check.

From this folder after activating `cv_conda`:

```powershell
python -m pip install --target .vendor --upgrade --no-deps --index-url https://pypi.org/simple -r requirements.txt
python prepare_models.py
python app.py
```

This is a supplement to the existing project environment, not a clean-environment dependency lock. It expects the existing torch, torchvision, huggingface_hub, requests, Pillow and pandas packages. Transformers is pinned locally to the study's version; NumPy and OpenCV are isolated too. OpenCV's headless package still handles cameras and video decoding; Qt owns the windows. The tested versions are recorded in `VALIDATION.md`. Do not reinstall CPU-only PyTorch over the CUDA environment.

Downloads use immutable commits, remote size/hash verification, short Windows paths, resumable `.part` files and bounded retries. Only four selected checkpoint files and the SegFormer configuration are fetched. Run the same command again after interruption. Once prepared, normal inspection works offline. Preparation checks the compact checkpoint hash before reusing it. Optional authentication reads `HF_TOKEN` from the environment; the public study does not require a token.

About 156 MB of compact inference checkpoints are retained, plus about 499 MB of original download files and the local dependencies. Original downloads in `.cache/` allow repair without downloading again; they are not loaded for ordinary inference. Checkpoints, cache, incoming photos and results are ignored by Git.

Checkpoint pickle loading is restricted to these fixed, hash-verified study artifacts. Do not point the loader at untrusted checkpoint files. The application caches at most three models so MobileNet, YOLO and SegFormer assist can stay resident. A four-model comparison evicts and reloads models as needed.

## Checks

```powershell
python check_station.py
python check_ui.py
python test_contracts.py
```

`check_station.py` runs real checkpoints, warm timings, repeatability checks, and lossless evidence round trips on one existing study image. Use `--image PATH` to choose another image. It produces `results/smoke-check.json`.

`check_ui.py` uses an offscreen Qt window, real GPU inference and a generated video fixture to check image loading, comparison, overlays, saving/restoring, video pause/resume and live predictions. It saves `results/station-preview.png` and `results/ui-check.json`. It does not exercise a physical webcam.

Windows CUDA inference uses a worker with a 32 MB stack because the default Python main-thread stack overflowed with this local CUDA installation. The desktop worker and command-line smoke check both apply this workaround. The desktop imports its isolated Qt build before OpenCV and pins the matching Qt plugin directories to avoid mixed Conda/pip DLLs.

## Files

- `app.py`, `theme.py`: native desktop workflow and styling.
- `engine.py`: inference, region overlays and capture hints.
- `classification.py`: lightweight classifier contract, checked against the study source.
- `registry.py`, `prepare_models.py`: fixed artifacts, verified downloads and compact exports.
- `evidence.py`: portable local inspection records.
- `check_station.py`, `check_ui.py`: repeatable verification.
- `data/incoming/`: optional location for future photographs.
- `results/`: saved captures and local validation outputs.

The prototype is separate from the completed training notebooks and report pipeline.

**Visible angles:** learned tread overlays show signed centreline tilt relative to image vertical, using displayed/smoothed points when available; positive means the top leans right. This is not physical camber/toe. The calibrated-target bench prints its measured camber/toe values on its preview and saved image after its existing calibration/target checks pass. No angle or alignment verdict is invented for an invalid fit.
