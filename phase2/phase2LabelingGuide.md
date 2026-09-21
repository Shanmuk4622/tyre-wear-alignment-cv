# Phase 2 — label only the new video images

> **Final status — 21 September 2026:** [All 15 runs and NB06 are verified complete](phase2Seed3Completion.md). Installed selections still match the final registry. Earlier partial/deferred/run-next statements below describe historical snapshots. No retraining or extra labeling is required.


> 20 September: no additional manual labeling is required for the runnable training revision. Model-specific conflict handling and weak training-point supervision are documented in `phase2RunNotebooks.md`. Earlier manual-review instructions remain available for an optional future annotation revision.

> Current status: all 152 submitted annotations were received and verified. The instructions below remain for reference and targeted corrections; do not redo the whole dataset. See `phase2DatasetAudit.md` for 45 containment disagreements and `phase2KaggleUpload.md` for the completed upload package. Raw annotations are preserved.

The old 418 photos and their labels remain unchanged. You only work on the **152 new PNG frames** in `phase2/data/new_frames/`. The class is already recorded as mid-mileage from your confirmation; do not type it on every image.

## 1. Look at the frames

Open [the review gallery](review/phase2_review.html). It contains eight contact sheets, ordered by video and time. This gallery is for viewing, not drawing labels. Original PNGs retain native resolution; label on those, not screenshots/contact sheets.

| Video | Number of new frames | Folder |
|---|---:|---|
| video1 | 27 | `data/new_frames/video1/` |
| video2 | 93 | `data/new_frames/video2/` |
| video3 | 32 | `data/new_frames/video3/` |

The footage includes a spinning tyre, blur, changes in viewpoint and a red workshop background. Keep identifiable difficult views; don't remove every blurred frame. Those conditions are part of why the models need improvement. Nearby frames are similar, but their outlines still move.

## 2. Open the labeling program

Double-click **phase2_Label_New_Frames.cmd** in this folder. Choose 1, 2 or 3 for the video. It uses the installed LabelMe in `cv_conda`; it does not install packages or open the old dataset. Close one session before switching video folders.

Alternatively, from the repository root in a `cv_conda` terminal:

```powershell
python phase2/phase2_label_new_frames.py --video video1
```

Use `video2` or `video3` for the others. Labels are saved separately under `phase2/annotation_work/video1/`, etc. The launcher uses an explicit Phase 2 configuration, with previous-frame copying disabled and image embedding disabled. This keeps JSONs small and avoids silently carrying a wrong polygon to the next frame.

## 3. Draw two regions on each usable frame

1. Choose **Create Polygons** (use the Edit menu if the toolbar is unfamiliar).
2. Draw the outside of the **visible tyre rubber**. Click along its actual contour, then double-click to finish. Select exactly **tyre**. Exclude floor, machinery, shafts, hands, feet and empty background. Do not trace individual grooves.
3. Draw the visible **tread crown** as a second polygon. Select **tread**. Include tread grooves inside the region; exclude visible sidewall when its boundary can be judged. This polygon lies within the tyre polygon. The two shapes intentionally overlap.
4. Use Edit Polygons to correct vertices at higher zoom. If the tread fills nearly the whole visible tyre, the two regions may be almost identical. Do not invent a shoulder strip merely to make them different.
5. Check the image flag **reviewed** only after checking both shapes, then press **Ctrl+S**. Use Next Image (normally D) to continue; Previous Image is normally A. Confirm a JSON appears in the output directory after the first image.

Use a modest number of vertices along straight boundaries and more where curvature changes. Follow visible evidence; don't create intricate edges that the blur does not support. One shared pair of reviewed regions supports SegFormer and YOLO; no separate rectangle annotation is needed.

Recommended first examples: the first frame of each video. Establish the same tyre/tread interpretation across them before doing the remainder. There is no mandatory wait for a new approval between batches; you may label all images using this guide and correct saved polygons later.

## 4. Difficult images and flags

- **blurred:** tick when motion or focus blur is present, but label normally if the region is still identifiable.
- **clipped:** tick when a tyre boundary leaves the frame. Stop the visible region at the image edge; do not draw an imagined outside shape.
- **occluded:** tick when another object hides part of the tyre. Label visible components only; use multiple polygons of the same class where necessary rather than bridging through the object.
- **ignore:** use this polygon label for an area whose class/boundary genuinely cannot be judged, such as a small ambiguous occlusion boundary. It is excluded from training/evaluation loss, not taught as background. Do not cover the whole image merely to avoid difficult labeling.
- **no_target:** tick if no recognizable tyre is present; don't invent a tyre/tread polygon. Also tick reviewed and save. These can later support rejection/background checks but are not automatic mid-class positive examples.
- **unusable:** tick if the image cannot be annotated reliably, for example severe blur or almost complete obstruction. Tick reviewed and save an explicit empty annotation if appropriate. Keep the original PNG; exclusions will have a recorded reason.

Multiple components/ignore regions need label-intake review before YOLO export; its polygon format must not silently fill occlusions or holes. Conversion fidelity will be checked against the authoritative binary masks.

## 5. HRNet points without redundant labeling

After polygon intake, a Phase 2 converter/review step will propose left/right tread intersections at three fixed image-height guides. We will show all six on each new image for acceptance/correction. This avoids asking you to draw the same boundary twice from scratch.

If a point is hidden, clipped, ambiguous or the guide does not cross the tyre, record that state rather than guessing. Visible points must remain on their guide row. A row crossing multiple disconnected regions needs human review. These later reviewed points will supervise/evaluate HRNet; generated proposals alone are not independent ground truth. This converter/reviewer is **planned, not supplied as an already working tool in this preparation delivery**.

## 6. Save and resume

Save before closing. Reopen the same video with the launcher; saved labels should load from its output folder. Confirm one completed image restores correctly before starting a long labeling session. Keep the native image folders and JSON folders together; JSONs refer to images rather than embedding them.

Do not rename frames, edit their pixels, move images between videos, or overwrite the old annotation files. The frame IDs and source timestamps are provenance. After finishing, return only the `annotation_work` JSON folders (or their ZIP); the PNGs are already present here. Labels can be revised in place until the dataset is frozen; after freeze, revisions get a new dataset version.

The user has confirmed these are different tyres. The only identity detail still needed is which original session matches each video (40,000 / 70,000 / 90,000 mileage-session candidates in the existing manifest). If unsure, say unknown rather than guessing. Labeling can proceed while that mapping remains pending; training splits cannot.

## Troubleshooting

| Symptom | What to do |
|---|---|
| Launcher closes or reports missing Python | Open an Anaconda Prompt, activate `cv_conda`, run the command above and retain the error text |
| Qt/DLL error | Stop and share the error; do not reinstall PyTorch or replace the workstation's dependencies |
| Cannot find the new PNGs | Keep the complete `phase2` folder structure; don't point LabelMe at the original dataset |
| Polygons appear on the wrong frame | Stop and correct the saved JSON; automatic previous-frame copying should be disabled |
| Saved image doesn't restore | Check the chosen video and output folder; don't start labeling the same frame a second time elsewhere |
| Confusing tread/sidewall transition | Use the visible crown boundary consistently; flag/ignore an unjudgeable area for review rather than inventing it |
| Already finished part of a video | Open the same video/output folder; do not rerun extraction to resume labeling |

The LabelMe command-line interface and local Phase 2 configuration were checked during preparation. Actual interactive polygon entry and save/reopen by the user still need confirmation; no human labels were fabricated by the assistant.
