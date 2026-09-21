# Tire Dataset Prepared phase2 — source release v1

Self-contained dataset: **570 unique images = 418 original photos + 152 new video frames**. All images keep their source bytes and native resolution. This adds views of three existing mid-mileage tyres, not three new tyres. The study still contains 12 physical tyres.

This is an upload-ready, verified **source dataset**, not a frozen train/validation/test release. Exact video-to-original tyre matching is unresolved. All Phase 2 splits are deliberately `UNASSIGNED`; do not randomly split nearby video frames.

## Contents

- `images/original/`: 418 original JPEG files, byte-matched to their original manifest.
- `data/new_frames/video1..3/`: 152 native portrait PNGs at two sampling targets/second; 27/93/32 per clip.
- `annotations/original_indexed/`: 418 unchanged manual indexed masks (0 background, 1 tyre, 2 tread, 3 marking, 4 damage).
- `annotation_work/video1..3/`: 152 unchanged submitted LabelMe JSONs; their relative image paths resolve inside this package.
- `masks/tyre/`, `masks/tread/`, `masks/ignore/`: separate binary PNGs, 0/255. Tyre/tread overlap; never use a mutually exclusive softmax interpretation.
- `annotations/boxes_xyxy.json`: boxes derived from masks, native pixel coordinates, upper x/y exclusive. These are not separate human box annotations.
- `geometry/original_human_points.json`: 120 existing six-point human label records, with source protocol retained.
- `geometry/new_point_proposals.json`: 912 polygon-derived point proposals; **not yet human-accepted geometry ground truth**.
- `manifests/images.json` and `.csv`: canonical 570-image index with paths, labels, identities, hashes and pending split.
- `provenance/`: original source manifest/checksums, geometry contract, video frame metadata and label audit.
- `audit/`: validation results and all-new-image contour contact sheets.
- `splits/phase2_split_status.json`: explicit training gate.
- `SHA256SUMS.txt`: SHA-256 for every other packaged file.
- `phase2_dataset_verify.py`: portable local/Kaggle verification (NumPy and Pillow required).

The old 4,180 augmented copies are intentionally excluded. Phase 2 plans online training augmentation from clean sources, preserving identity and avoiding redundant upload/storage. Old photos and labels do not need to be uploaded separately alongside this release. Phase 1 files/folds are unchanged; the copied old manifest is provenance, not the current loader index.

## Label checks and warnings

Every submitted frame has tyre and tread polygons. All 304 polygons have finite coordinates, nonzero area and no proper edge crossings. Fifteen frames use polygon vertices at the outer canvas boundary W/H; this is accepted as canvas-edge geometry and clipped by rasterization, with raw coordinates preserved.

In 45 frames, a small part of tread falls outside the independently drawn tyre outline (largest case 1.251% of tread area). Raw polygons and both binary channels are preserved. `ignore` marks those contradictory pixels plus any user ignore polygons. Losses and metrics must exclude ignored pixels; they must not be taught as background. Standard YOLO segmentation does not automatically consume this ignore channel, so its export/training remains gated on explicit conflict resolution or a verified compatible trainer. No silent clipping/union correction was applied to the user's labels.

The user reports annotation completion. Original `reviewed`, blur, clipping and other flags were all false and remain unchanged. Derived border-touch observations are separate metadata. All 152 new frames were visually screened using contour thumbnails; this is not an independent pixel-accurate human reannotation. Small contour errors, especially around machinery-adjacent protrusions, can remain. Prioritize Video3 at 0.5 s and 2.0 s for the largest containment disagreements.

The old 418 masks are structurally checked and hash-matched to the original annotation checksum manifest. Their canonical tyre is `indexed > 0`; tread is labels 2 or 3. Damage is not assumed to belong to tread. Original labels are retained unchanged, not reinterpreted as new classes.

Class mapping: low=0, mid=1, high=2, all **mileage proxies**. New frames inherit mid from user confirmation. No depth, roadworthiness, anomaly or alignment ground truth is asserted.

## Verification and use

Run `python phase2_dataset_verify.py /path/to/phase2_dataset_v1` in an environment with NumPy and Pillow. It checks every listed hash, all 570 image/mask sets, semantic ignore handling, uniqueness, geometry inventories and the pending-split gate. It does not train, download or upload anything.

For Kaggle, upload the ZIP as a **new dataset** titled **Tire Dataset Prepared phase2**. Keep the existing Tire Dataset Prepared dataset intact. After processing, confirm that VERSION.json and the folder tree are available, then attach the new dataset to the Phase 2 preflight notebook. Paths may include an extra directory layer; discover by VERSION.json rather than guessing the slug. Kaggle documents ZIP support in its [dataset guide](https://www.kaggle.com/docs/datasets).

Training readiness still needs exact identity mapping, a frozen physical-tyre split, reviewed geometry targets/conflict handling, and actual GPU/resume checks. This archive intentionally includes no model checkpoints, tokens, private environment files, raw videos or fabricated split assignments.
