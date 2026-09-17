# Local verification — 13 September 2026

## Overlay downloads — 16 September 2026

Verified in `cv_conda` with `check_media_export.py`,
`check_media_export.py --real`, and `check_edge_geometry.py`:

- Exact PNG pixels for Original, Full overlay, and a non-central Compare wipe.
- 30 fps and 5 fps inputs exported at 10 fps with correct frame selection,
  repeated low-rate frames, portrait dimensions, duration and visible overlays.
- Failed inference leaves no finished or partial MP4; UI cancellation cleans up
  its subprocess without replacing the inspection frame.
- Real GPU inspection followed by a separate CPU export with the default paired
  learned models on a three-frame portrait excerpt of a supplied video. Output
  decoded as three frames at 10 fps, 360 × 640. Controls remained usable while
  exporting, and changing opacity did not change the export's settings snapshot.
- Existing synthetic boundary/rim acceptance and rejection checks still pass.
  The explanation now distinguishes the predicted search outline from accepted
  geometry and displays a withheld-fit notice instead of empty-looking panels.

Local screenshots and a sample MP4 are in `results/export-check/` (ignored by
Git). These checks verify export behavior, not geometric accuracy. Full-length
export throughput and unusual variable-frame-rate videos have not been tested.

<!-- current-status:start -->
> **Current status (15 September 2026):** [Completed work and remaining validation](../docs/CURRENT_STATUS.md). The report is refreshed; the app, learned-geometry integration and target-assisted alignment software exist. Dated plans below retain their original context.
<!-- current-status:end -->

Tested on NVIDIA GeForce GTX 1650 (4 GB), using the existing `cv_conda` environment and its CUDA-enabled PyTorch. No training or model substitution occurred.

## Real-model checks

All four fixed fold-1, seed-1, final-epoch checkpoints loaded strictly. Repeated inference on the same image produced identical decisions and masks; class scores matched within 1e-6. The original image and binary masks survived a lossless saved-record round trip.

Warm timings below are medians of three repeats on one internal study image (1152 × 1536). They include preprocessing, GPU synchronization and postprocessing, but exclude model loading. These are not camera FPS or an accuracy evaluation.

| Model | Median | Observed range |
|---|---:|---:|
| MobileNet V4 | 32.99 ms | 28.77–34.45 ms |
| ResNet 50 | 36.35 ms | 34.73–37.50 ms |
| YOLO26 Nano segmentation | 50.90 ms | 49.55–51.26 ms |
| SegFormer B0 | 54.17 ms | 48.00–54.84 ms |

The four-model comparison recorded a PyTorch allocated-memory peak of 224.4 MiB. This excludes the CUDA context, memory reserved by the allocator, display usage and other processes. First use includes imports, hash verification, model loading and GPU warmup, and takes appreciably longer.

MobileNet + SegFormer is the default pair: MobileNet is the faster classifier here; SegFormer has stronger region results in the existing study and similar local latency to YOLO. This remains a prototype choice, not an external-data winner.

Machine-readable results: `results/smoke-check.json`. The evidence path recorded there contains the exact input, outputs and checkpoint provenance.

## Desktop and contract checks

- Both the offscreen renderer and actual Windows desktop window created and rendered with real model results.
- Asynchronous four-model comparison completed; UI updates use queued signals.
- Original/region overlay selection worked.
- Evidence saved and restored with its frame hash and operator note intact.
- A generated MJPEG video decoded; freeze, resume, live inference and shutdown worked.
- Preprocessing matched the actual study source exactly for raw, grayscale and CLAHE inputs.
- CORAL decisions and scores matched the study source, including a case where threshold-count differs from argmax.
- Verified downloads reused a valid local file and recovered a completed partial download without a new GET.

Run `python test_contracts.py`, `python check_station.py`, and `python check_ui.py` to reproduce. `python check_ui.py --native` repeats the UI check in an actual visible Windows window, then closes it. Both UI modes passed. The UI check saves `results/ui-check.json` and `results/station-preview.png`. The `.cmd` launcher's activation and argument forwarding were also verified with `--help`.

**Not verified:** physical webcam acquisition, new external photographs, aggregate external accuracy, long-duration camera throughput, and CPU performance. The video fixture is derived from a study image and is only a playback/control test.

## Runtime

| Package | Tested version |
|---|---|
| torch | 2.5.1+cu121 |
| torchvision | 0.20.1+cu121 |
| timm | 1.0.15 |
| ultralytics | 8.4.20 |
| transformers | 4.51.3 |
| tokenizers | 0.21.4 |
| PySide6-Essentials / shiboken6 | 6.8.3 |
| opencv-python-headless | 4.11.0.86 |
| numpy | 1.26.4 |
| huggingface_hub | 0.36.2 |

Additional packages live under the prototype's `.vendor/`; the original Conda environment was not overwritten. The implementation handles the observed Windows CUDA stack limit with a 32 MB inference-worker stack and isolates Qt plugins and fonts from the Conda GUI installation. Inference avoids importing the training library's pandas/Arrow dependencies.
