# Phase 2 workstation — installed 21 September 2026

The existing `prototype/Launch Tread Station.cmd` now loads Phase 2 exports. Close and reopen any running workstation window. No package reinstall or training is required. Internet is not needed for inference.

| Component | Installed export |
|---|---|
| MobileNet V4 Medium | Seed 1, validation-best epoch 4 |
| ResNet50 | Seed 2, validation-best epoch 1 |
| YOLO26 Medium segmentation | Seed 1, validation-best epoch 10; replaces Nano in the current app |
| SegFormer B0 | Seed 2, validation-best epoch 29; regions and learned-boundary comparison |
| HRNet W18 | Seed 3, validation-best epoch 2 |

The weights were selected by validation only using the NB06 first-seed tie rule, then downloaded at immutable HF revision `be0eadb2bbb698091b563d699e2519b1e262a31c` and SHA256-verified. Five files total 290,983,569 bytes (~277.5 MiB). `registry.json` retains the source revision, hash, contract and scores. Checkpoints remain under `phase2/workstation/models/`; original model files are preserved.

## What changed

The existing engine now loads strict Phase 2 state dictionaries and uses the trained input sizes: classifiers 384×384; YOLO, SegFormer and HRNet height 512 × width 384. Images use direct PIL bilinear resizing; YOLO uses RGB [0,1], other models ImageNet normalization. Classifiers use three-class softmax. The masks remain overlapping independent tyre/tread channels, resized back to the original frame. HRNet preserves the fixed guide rows and six horizontal positions. SegFormer boundary comparison uses the new segmentation export, not the earlier matched study weights.

The UI identifies YOLO as Medium. Its historical internal key `yolo26n_seg` and the boundary key `matched` are retained for recipe/evidence compatibility; titles and checkpoint provenance identify the actual Phase 2 models. The geometry explanation now shows the actual HRNet seed and validation-selected status.

## Verification

- All five strict GPU loads passed on NVIDIA GTX 1650.
- Repeated predictions were stable; segmentation adapter output matched the training inference implementation exactly on the checked photos.
- HRNet reproduced the saved exported-checkpoint test coordinates within 0.00002 normalized width.
- Nine sampled frames across all three supplied videos passed same-frame masks/points and evidence round trip.
- Desktop UI passed live video, portrait handling, overlays, seeking/reset, evidence JSON/PNG saving/restoration and releasing learned-model references when disabled.
- Five existing contract tests passed. Four-model station smoke passed.
- Geometry explanation was separately checked after fixing its old hard-coded seed/epoch caption.

Evidence: `validation.json`, `phase2_smoke-check.json`, `phase2_learned-ui-check.json`, and the preview PNGs in this folder. The nine-frame combined pipeline check had about 623 ms warm median across eight subsequent sampled frames; this is not a sustained 3-inspections/sec certification. Cold model loading takes longer. UI test allocations peaked around 311 MiB; the separate nine-frame check recorded about 327 MiB of PyTorch allocated memory, not total GPU or driver memory.

## Results and limits

Selected classifiers each achieved only 50% balanced accuracy on held-out low/high tyres. MobileNet seed 3 scored better on test, but it was not substituted after observing test scores. Selected SegFormer test Dice was 98.957%; YOLO 98.790%; HRNet boundary-position error 0.859% of image width. These are small internal test results, not a before/after improvement guarantee. The three videos are training-cohort data. Do not treat video checks as unseen generalization, tread-depth measurement or safety assessment.

All 15 runs and final NB06 are now HF-verified complete at revision `714a8daa65305f450e4e72ca82dd36b2922e7151`. All five installed hashes match the final registry; YOLO seed 1 remains validation-best across all three seeds. No notebook rerun is required.

## Rollback

Close the app, run `phase2_Restore_Legacy.cmd`, and reopen the normal launcher to use the preserved original models. To return to Phase 2, close the app, run `phase2_Enable_Models.cmd`, then reopen. The Phase 2 switch verifies all five model hashes before enabling. No checkpoint is overwritten or deleted by switching.

`backup-20260921/` contains the pre-change app files and readmes with a hash manifest. The current integration was explicitly authorized by the user's model-replacement request; training source files, raw data, executed notebooks and old weights were not modified.

## Updated video controls

Choose Video export FPS (1–60; default 10), and optionally include the information panel. Learned overlays show image-relative tread tilt; calibrated-target previews print camber/toe. [Detailed usage and checks](../phase2VideoControls.md). Restart the app for new controls.
