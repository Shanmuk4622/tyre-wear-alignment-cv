# Phase 2 dataset audit — source v1

> **Final status — 21 September 2026:** [All 15 runs and NB06 are verified complete](phase2Seed3Completion.md). Installed selections still match the final registry. Earlier partial/deferred/run-next statements below describe historical snapshots. No retraining or extra labeling is required.


19 September 2026. User-declared labeling completion: all new images labeled. Raw user annotations and original data were preserved.

| Check | Result |
|---|---|
| Combined clean image inventory | 570: 418 original + 152 new; unique image hashes |
| New frames by video | 27 / 93 / 32 |
| Mileage-proxy classes | 169 low, 249 mid, 152 high |
| Physical tyres | 12 total; new videos add views, not identities |
| New labels | 152 JSON files, 304 tyre/tread polygons |
| Structural polygon audit | Finite coordinates, valid image references/dimensions, nonzero area, no proper edge crossings |
| Canvas boundary coordinates | 15 frames use W/H edges; accepted and raw coordinates preserved |
| Containment disagreements | 45 frames, 172,439 pixels; maximum 1.251% of tread area |
| Disagreement handling | Separate raw masks plus explicit ignore mask; no silent label correction |
| Visual screening | All 152 new images screened at contour-thumbnail scale; not pixel-accurate independent reannotation |
| Original labels | 418 indexed masks matched to source checksums; original geometry retained |
| Geometry | 120 original human point sets; 912 new proposals, zero human-accepted new points |
| Quality flags | User flags remain false; no invented reviewed flags |
| Source immutability | 1,140 external/original/new source image and annotation files rehashed unchanged during packaging |
| Archive | 2,883 members; every member decompressed and SHA-256 matched before publication |
| Training split | UNASSIGNED for every image; exact video identity mapping pending |

The package excludes 4,180 old augmented copies. Augmentation belongs inside the future training loader. Masks are overlapping binary tyre/tread channels, not mutually exclusive semantic classes. Ignore pixels must be excluded from losses and metrics, not treated as background.

The largest containment disagreements are Video3 at 0.5 s and 2.0 s. Border-touch occurs in 103 new masks; this is an observation, not an automatic rejection or corrected user flag. Raw source labels remain the authoritative revision; any manual fixes require a new version.

The audit initially classified W/H canvas-edge vertices as out of bounds. That validator assumption was corrected to accept the LabelMe canvas convention; no submitted polygon was changed to satisfy it. All 152 records then passed structural validation with the documented warnings.

Evidence: `manifests/phase2_label_audit.json`, `manifests/phase2_dataset_sources_unchanged.json`, `manifests/phase2_zip_verification.json`, and the frozen package's `audit/` directory. The detailed mechanical audit retains its initial visual-review-pending status; the later visual review completion is recorded separately in `data/phase2_dataset_v1/audit/label_visual_review.json`.

See `phase2KaggleUpload.md` for upload and CPU preflight instructions. No training, accuracy claim or remote publication was performed locally.
