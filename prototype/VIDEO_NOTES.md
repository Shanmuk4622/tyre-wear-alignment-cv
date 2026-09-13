# Video update and real-clip verification

The original prototype was verified with a video made from a study photograph. That did not establish that YOLO could generalize to actual video. This update was tested with the three supplied files in the repository's `Videos/` folder.

## What changed

- Opening a video starts live analysis. Playback holds its first frame during model warmup instead of running past the interesting frames while models load.
- A persistent GPU worker retains its CUDA context and models across frames. Decoding uses a single replaceable frame slot, so slow analysis cannot create a long frame queue.
- 4K/1080p video is reduced proportionally to a maximum 1280-pixel working dimension. Evidence records store that working frame and the source dimensions; photographs retain the original image path.
- The main canvas shows the exact analysed pixels and masks. An inset shows the moving source independently. The displayed update rate measures completed inspections, not camera FPS.
- Replay and a seek bar make it possible to revisit a moment. Seeking invalidates old results and briefly holds the sought frame for analysis. Pausing preserves its overlay.
- The workstation is light mode only, including the canvas, controls and newly exported HTML cards.
- Video is displayed and processed in portrait orientation. All three supplied files have a −90° display-matrix tag; OpenCV exposes this as 90° clockwise. The decoder previously ignored this tag. It is now applied exactly once before preview and inference. The models receive the same vertical pixels shown in the workstation, with no automatic model rotation search. Untagged landscape inputs are turned clockwise once to maintain the requested portrait mode.
- Previous/Next frame pause on a specific frame; ±5-second jumps complement the seek bar. Playback speeds are 0.25×, 0.5×, 1×, 1.5× and 2×. Loop repeats the video at its end.
- Original, full overlay and comparison-wipe views are selectable. Tyre/tread layers can be toggled independently and opacity adjusted. Display zoom ranges from Fit to 300%, with drag-to-pan in Original or Full overlay mode. Raw masks and inference inputs remain unchanged; exported records also include opacity and layer settings.
- The YOLO score threshold is visible and adjustable; it still defaults to 0.25. Each result reports the strongest candidate score, including when nothing passes the threshold.
- The optional **SegFormer assist** runs a separately labelled model if YOLO fails to return both tyre and tread. Its overlay is explicitly a suggestion for review, never relabelled as a YOLO detection. Uncheck the assist box to examine YOLO alone.
- Saved JSON and HTML record position, rotation, threshold, source/working dimensions and assist provenance.

## Findings on your clips

The earlier test found no accepted YOLO regions in nine samples decoded without their display rotation. That horizontal presentation did not reflect the videos' portrait metadata. The current pipeline fixes the decoder orientation rather than choosing a rotation based on model scores.

Correct presentation does not establish model accuracy. Motion blur can still obscure tread; SegFormer suggestions can miss regions or include background. A visible overlay is not proof of accurate segmentation, and these clips have no pixel-level ground truth in this test.

The portrait desktop playback test exercised all three videos, including seeking during inference, pause/overlay retention, evidence capture, replay and reuse of the same GPU worker. It measured approximately **3.25, 3.54 and 3.14 completed inspections/second**, respectively, at a requested ceiling of 5/sec. Working frames were 720 pixels wide × 1280 high. These are short local observations on the GTX 1650, including assisted inference and GUI work, not guaranteed sustained throughput.

No weights were trained or replaced. Classification still describes mileage proxies, not tread depth or roadworthiness. The video-specific orientation and resolution settings are recorded; those outputs must not be mixed into the original fixed-protocol study metrics.

## Reproduce

Activate `cv_conda` and run these from `prototype/`:

```powershell
python check_videos.py
python check_video_ui.py
python check_portrait_controls.py
```

`check_videos.py` saves sampled model diagnostics and an overlay contact sheet under `results/video-check/`. `check_video_ui.py` tests the actual clips in the desktop application and writes `results/video-ui-check.json` plus per-video screenshots. Neither test proves detection accuracy or tests a physical webcam.

`check_portrait_controls.py` compares decoded portrait pixels against OpenCV's metadata-aware reference decoder and exercises the added playback and overlay controls. Its results are saved in `results/portrait-controls-check.json` with a screenshot at `results/portrait-light-workstation.png`.

Restart Tread Station to load this update. Open a file from `Videos/`; portrait orientation is applied automatically from the file. Use Replay, slow motion or frame stepping to revisit a clear frame, and inspect YOLO's own overlay/score before interpreting an assisted result.
