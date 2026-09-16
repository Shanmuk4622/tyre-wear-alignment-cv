# S9 — small guided input pilot

<!-- current-status:start -->
> **Current status (15 September 2026):** [Completed work and remaining validation](CURRENT_STATUS.md). The report is refreshed; the app, learned-geometry integration and target-assisted alignment software exist. Dated plans below retain their original context.
<!-- current-status:end -->

Updated 14 September 2026. **Revised NB22 upload HF-verified; 11/12 mechanically complete.**
P06's left cropped points are corrected; right-middle/lower still need visibility
correction, and its new Visible issue choice needs a description. No new batch or training yet.
[Current audit and next steps](28_S9_PILOT_REVIEW_AND_NEXT_PLAN.md) supersede the
original preparation instructions below; do not rerun NB21 for the current package.

## Why these labels, and are these the final models?

HRNet and PatchCore are candidates in the original integrated design, not automatically the final winning system. Your prototype already uses trained classification/segmentation models and now has image-relative geometry plus a separate target-assisted calibration workflow. Those working changes are preserved. Adding two more models is justified only if their inputs are valid and their measured contribution improves the intended task.

| Pilot input | Purpose | What it does NOT establish |
|---|---|---|
| Six visible tread-boundary points | Test whether a repeatable 2-D geometry target exists that HRNet might learn | Wheel plane, camber, toe, depth or verified anatomical landmarks |
| Visible issue / no visible issue / cannot assess | Document image usability and possible appearance anomalies | A verified healthy tyre, safety assessment or a PatchCore training pool |
| Independent-record availability | Find whether defensible reference evidence already exists | Automatically validating that record or asking you to perform new physical inspection now |

The six points are an **explicitly proposed pilot definition**, because the earlier proposal did not freeze usable landmark semantics. They are not silently substituted for a validated physical landmark task. Existing masks may already provide adequate image geometry: after review, compare the proposed labels with mask-derived boundaries before committing to HRNet annotation/training. If the point task is redundant or not visible, stop/revise it rather than wasting your time.

HRNet is a high-resolution representation/keypoint method, not a ready-made tyre-angle model: [primary paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Sun_Deep_High-Resolution_Representation_Learning_for_Human_Pose_Estimation_CVPR_2019_paper.html). PatchCore models nominal appearance using reference features; its premise motivates a defensible reference set: [primary paper](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html). Both primary records were consulted on 14 September 2026. Neither validates these proposed tyre-specific inputs by itself.

## Exactly what to run

### First: NB21 PREPARE

Open [NB21_S9_Annotation_Pilot.ipynb](../notebooks/NB21_S9_Annotation_Pilot.ipynb) in Kaggle.

1. Accelerator **None / CPU**. Run only **one copy**, not four workers.
2. Internet ON. Enable the existing `HF_TOKEN` secret.
3. Attach the existing **Tire Dataset Prepared** dataset containing `FINAL/manifests/clean_manifest.csv` and clean images. No new dataset download or masks are needed.
4. Leave `DATA_ROOT` blank and select **Run All**. If more than one prepared dataset is attached, remove the duplicate attachment or provide the exact FINAL path.
5. Download **PILOT_12_IMAGES.zip** from the last cell or Kaggle Output → `s9_pilot_outputs` → package folder.
6. Extract the ZIP on your computer and open **ANNOTATE.html** in Chrome or Edge. No server, login, install or GPU is needed to label.

The local real-data package is **9,917,962 bytes (9.46 MiB)**, preserving all twelve original1152×1536 JPEGs byte-for-byte. It includes three schematic worked examples. It does not recompress, resize or crop the originals. The notebook caps the ZIP and individual HTML page at20MiB and fails instead of silently lowering detail if the cap is exceeded. The ZIP and extracted HTML are alternative views of the same package; you download only the ZIP.

Selection is deterministic: median sorted image ID in each of the12 capture sessions. This is diversity/feasibility sampling, not an unbiased benchmark or12 verified independent tyres. Original IDs, hashes, dimensions, folds and session mappings are in `MANIFEST.json`; the annotation screen hides mileage classes and model outputs.

### Second: mark only these12 images

At each of three fixed horizontal guides (25%,50%,75% of native image height), click the left and right visible transition between grooved tread face and shoulder. Left/right is image-relative. Select the named point before clicking; its y-coordinate snaps to the correct guide. Use original-pixel zoom and scroll for detail.

Do **not** substitute the outer tyre silhouette when the tread transition is not visible. Do not follow paint stripes or internal grooves. Choose **Uncertain**, **Occluded**, or **Outside frame** instead of guessing. If all six points are unobservable, recording that is useful and allowed. Three schematic examples show clear, hidden and ambiguous cases; they are teaching diagrams, not fabricated real-image ground truth.

Choose a surface observation. “No visible issue” means only that in this photograph. A visible issue requires a short location/description, not a diagnosis. Independent records may remain **Unknown / unavailable**. Do not put private inspection reports or personal details in notes; an available record can be described by a non-sensitive identifier for later discussion.

Click **Save annotation JSON** every few images and before closing. The downloaded filename begins `annotations_`; repeated downloads may add a browser suffix. Keep the latest JSON. To resume, open the same HTML and **Load saved JSON**. Browser autosave is a convenience, not a persistence guarantee. Unsent edits made after the last JSON export cannot be recovered by Kaggle or HF.

The counter requires all six point states, an image observation and a note for visible issues. It measures completion of choices, **not annotation accuracy**. Stop at12/12 and send only your JSON to the assistant for review. No next batch yet.

### Third: optional NB22 intake to HF

After NB21 has published its manifest, open [NB22_S9_Pilot_Review.ipynb](../notebooks/NB22_S9_Pilot_Review.ipynb).

1. CPU, one copy, Internet ON, HF_TOKEN enabled.
2. Attach **only the exported annotations JSON**, using Kaggle Add Input/upload files. No images or prepared dataset needed.
3. Leave `ANNOTATION_JSON` blank if only one matching JSON is attached; Run All.
4. Send me the JSON or printed HF revision. I will review the actual marked images before suggesting more work.

NB22 checks package/image identity, names, visibility states, finite native coordinates, exact guide y-values and left/right order. Incomplete exports are preserved as partial. It uploads only annotation and review/status files, normally a few KB. It never starts training or converts an operator's “no visible issue” into a verified healthy reference.

If you prefer, send the JSON directly in chat after NB21; NB22 is the persistent-HF intake convenience, not another labelling task.

## Human review before scaling

I will check point meaning and visibility on each image, skipped/ambiguous cases, orientation/coordinate consistency, and whether the labels add useful information beyond existing masks. I will also separate visual appearance observations from actual reference evidence. This is a feasibility/quality review, not an independent annotator-agreement study. Pilot examples must not be promoted to held-out accuracy evidence.

Possible outcomes: accept a small next batch; revise the point definition and revisit only affected pilot images; collect a few clearer views if justified; or decide that another HRNet model is unnecessary. PatchCore's verified-reference requirement remains open if no independent evidence exists. No minimum healthy-pool size, performance guarantee or automatic training approval is invented here.

Only after review should a training design freeze the target, model identity, train/evaluation identity grouping, endpoint and component-removal comparisons. Any training will be delivered separately as resumable Kaggle notebooks. The current two notebooks intentionally require no GPU or training budget.

## Persistence, provenance and validation

HF namespace: `s9/s9-input-pilot-r1/<package-content-hash>/`; reviews append `reviews/<annotation-sha256>/`. NB21 publishes the small ZIP, manifest, status and implementation sources in one batched major-completion commit. NB22 publishes labels/review/status in one commit. There are no claim/heartbeat commits. These short CPU actions have no long training loop requiring periodic30-minute snapshots; catchable Stop attempts to flush already completed local output. Server backoff applies; forced kills and unsaved browser edits cannot flush.

The initial implementation check found HF revision `22d5a6bc9f953ba3bf2a75919edc7db3193b317b`. After the user's run, public main and NB22's successful commit are `1dad525affd32b06dc73f907edad73eb4096b782`; local manifest and annotations match public bytes. No assistant HF write, large download, model loading or training was performed. See docs/28 for the completed audit.

Local tests:12 original-image hashes and package size; notebook syntax and embedded-source equality; rejection of bad IDs, crossed/out-of-range/nonfinite coordinates and guessed invisible coordinates; partial-record handling; no automatic health/training approval; mocked HF upload; JavaScript syntax. Browser checks exercised skip states, progress, navigation, actual1,366-byte test JSON download and restoring that JSON. Test labels are explicitly marked software fixtures and are not human evidence. NB22's subsequent real execution is now verified separately in docs/28.

Source: `tyrelib/s9_pilot.py`, `tyrelib/pilot_template.html`, `tyrelib/build_s9_pilot_notebooks.py`; tests: `scripts/verify_s9_pilot.py`. The local convenience script `scripts/prepare_s9_pilot_local.py` prepares the same package without publishing. The existing prototype modifications and frozen manuscript snapshot are not rewritten by this addition.
