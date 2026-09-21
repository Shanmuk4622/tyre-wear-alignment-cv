# Phase 2 progress

Last updated: 21 September 2026. This current table supersedes dated session-log statements.

**All 15 runs completed 60 epochs and are independently HF-verified. Final NB06 is complete. All five installed exports remain the validation-selected winners.**

| Item | State | Evidence / disposition |
|---|---|---|
| Frame extraction and new labeling | Complete | 152 frames, all labels received; raw sources preserved |
| Dataset audit and Kaggle release | Complete | 570 total; 418 old + 152 new; 2,882 checksums |
| NB00 | Complete / HF-verified | Existing verified release report |
| Identity and split handling | Complete for this revision | All possible mid matches train-only; 386/81/103 split |
| Conflict masks and weak points | Implemented and exercised | Explicit ignore/union policies; HRNet weak points weight 0.25 |
| Five model GPU resume checks | Passed on Kaggle T4 | SMOKE.json artifacts verified for all 15 jobs |
| Classifier training NB02 | Complete | Six runs ×60 epochs; quality limitations retained |
| SegFormer training NB03 | Complete | Three runs ×60 epochs |
| YOLO training NB04 | Complete | All three seeds ×60 epochs; seed 1 remains validation-selected |
| HRNet training NB05 | Complete | Three runs ×60 epochs |
| NB06 audit and export registry | Complete / HF-verified | Final report 1789972637; 15/15 runs; pending list empty |
| Best available weights | Downloaded and byte-hash verified | MobileNet 1, ResNet 2, SegFormer 2, YOLO 1, HRNet 3; validation-only selection |
| Workstation adapters and GPU checks | Complete | Exact input contracts; repeated outputs; nine video frames; evidence round trip |
| Desktop integration | Complete / GPU and UI verified | All three videos, seeking, overlays, save/restore and model release passed; existing launcher, rollback available |
| Video export FPS and labels | Complete / tested | 1–60 FPS, optional panel, image-relative tilt, calibrated angle labels; real 12.5-FPS export passed |
| Old-only A/B controls | Optional / not run | Outside default 15-run combined schedule; no added-data causality claim |
| Deployment refit / independent cohort | Optional follow-up | Requires separate training/new data; not claimed complete |

See [completion audit](phase2CompletionAudit.md) and [workstation guide](workstation/README.md). No required training, local preparation or annotation remains for the 15-run schedule. [Final seed-3 verification](phase2Seed3Completion.md) supersedes earlier incomplete/deferred entries.

## Decisions

- All new artifacts remain under `phase2/`; existing `undertand3.md` is not updated because the latest instruction protects every existing file.
- Use 152 candidates rather than claiming all are already accepted training examples.
- New frames add views of existing identities; total remains 12 tyres, at most 570 unique images before review.
- Use user-confirmed mid label through metadata; do not predict labels from the old models.
- Candidate Nano replacement: YOLO26m-seg, subject to accuracy/runtime tests. Larger size is not a guarantee.
- Primary honest Phase 2 A/B runs start from general pretrained weights. Existing tyre-trained weights can contaminate a new split, especially because all three mid tyres were in old HRNet training.
- Whole physical tyres determine train/validation/test. Do not split neighboring video frames randomly.
- New video data is mid-only; balance classes/tyres/domains and do not report video-only three-class generalization.
- Label new visible tyre/tread regions once, derive boxes and propose points from them, then review geometry. Old labels remain unchanged.
- Preserve realistic blur/occlusion cases and record rejections. Raw proposals remain available even when displayed outputs are withheld.

## Session log

### 2026-09-19 — planning and preparation

Read existing inference/protocol contracts, inspected three video streams with ffprobe, and checked official YOLO26 segmentation availability and HF rate-limit guidance. User confirmed three different mid-mileage tyres from the original 12; exact matching remains unresolved.

Created a resumable native PNG extraction script, frame/probe/summary manifests, candidate review queue and gallery. Extraction decoded 393 + 1,378 + 474 source frames and saved 152 at half-second targets. Applied recorded display rotation once; no scale/crop/contrast/model-driven rotation. No files rejected or auto-labelled. Contact sheets show changes in view, blur and a shared workshop background; these are development observations, not numerical quality labels or inferred identities.

Wrote the dataset/model/split/evaluation plan, later Kaggle persistence contract and new-image labeling guide. Prepared a launcher using the existing LabelMe installation with a local config. No packages installed, source dataset copied, model inference/training launched or remote artifacts published.

Verification found that the installed LabelMe exposes `load_config`, not the older `get_config` API. Corrected the new Phase 2 launcher and used the current `with_image_data` setting; all three launch-path/configuration checks then passed. No existing application/dependency file was edited. Interactive annotation/save/reopen remains a user check.

A second full extraction reused the PNGs after exact decoded-pixel comparison; the frame manifest remained `5a54d9438def6e090342aa853905b9bc611a8f8d851f9eee37d38861041eae6e`. Independent OpenCV metadata-aware decoding matched the first/middle/last sample of each clip exactly. `manifests/phase2_preparation_validation.json` records the checks, and `phase2_originals_check.json` records all 376 protected files unchanged. These are preparation checks, not model-accuracy or annotation-validity results.

### 2026-09-19 — annotation audit and source release

Received all 152 user JSONs and checked paths, dimensions, finite/nondegenerate polygon geometry and proper edge crossings. Corrected an overly strict validator assumption for 15 valid W/H canvas-edge coordinates; no user vertices changed. Screened all new images as contour thumbnails. Found 45 tread-outside-tyre disagreements (172,439 pixels; largest 1.251% of tread); preserved both regions and marked contradictory pixels in ignore masks. User flags remain unchanged.

Combined 418 original clean photos/indexed masks with all 152 new frames and annotations. Reused 120 original human geometry records and derived 912 explicitly unaccepted new point proposals. Excluded old augmented copies. Checked source checksums, decoded all images and masks, and preserved raw annotation bytes. Built the immutable source-v1 package and ZIP; every archive member was independently decompressed and hash-matched. The portable verifier subsequently passed all 2,882 checksummed files and 570 image/mask sets.

Release: `releases/Tire_Dataset_Prepared_phase2_v1.zip`, 777,988,937 bytes. SHA-256: `07c7aa41e84135f578a8cf1d8a1ec42a947b3d5062cbcfa935732799c15ab836`. Evidence: `manifests/phase2_zip_verification.json` and `phase2DatasetAudit.md`.

Built `notebooks/phase2_NB00_Dataset_Preflight.ipynb`, pinned to this release's checksum manifest, with CPU verification, local report retention and bounded HF retry/commit verification. No Kaggle execution or remote publication has occurred. Training remains gated on identity/split, trainer conflict handling, geometry review and GPU/resume implementation/testing.

### 2026-09-20 — runnable notebook release and NB00 verification

Reviewed the earlier classification/YOLO worker notebooks, executed HRNet/SegFormer notebooks, their generators, runtime and numerical-resume repairs. Preserved the executed Phase 2 NB00 and archived its exact bytes. Independently fetched its report at the printed HF commit and verified the release hash, PASS status and all 570 image/mask sets; evidence is `manifests/phase2_NB00_hf_verified.json`.

User delegated the remaining preparation. Replaced the earlier mapping-dependent split with a documented training-only mid-tyre group, making every possible video mapping safe with respect to validation/test. This limits held-out claims to old low/high tyres. Implemented explicit derived overlap handling and weak training-only points without editing source annotations or requesting more labels. The overlay and run guide supersede the earlier manual gates.

Implemented six self-contained Kaggle notebooks, fixed job sharding, two GPU processes, coordinated HF publication, leases/fencing, atomic completed-update checkpoints and fresh-process real-model smoke tests. Added source/protocol/version checks, validation-only checkpoint selection, full trace retention, report auditing and a separate export registry. Default combined-data study is 15 runs; old-only comparison is optional and retains stable IDs.

All five actual architecture CPU tests passed. All five pinned pretrained models then passed full configured-resolution GPU tests on the local GTX 1650, with zero differences in next losses, weights, optimizer, RNG, AMP scaler and noninitial scheduler state after fresh-process restore. This does not replace dual-T4 execution. Fault tests passed, and a full toy-model lifecycle checked stopping/resuming without repeated updates and validation-best export.

Troubleshooting recorded: an intermediate smoke run mixed a newly changed runtime with an older temporary checkpoint and was rejected; the final stable-code rerun passed. An initial native HRNet GPU process exited without a Python traceback; its clean final rerun passed. The local Windows checkpoint writer initially attempted fsync on a read-only descriptor; changed it to flush/fsync the writable save handle, then the pause/resume/export lifecycle passed. Temporary dependency-install attempts did not provide usable completion output; isolated official wheel downloads enabled local tests. Original environments and notebooks were not replaced.

No long Phase 2 training, new accuracy result or workstation promotion is claimed. The remaining execution rows are expected future results, not unfinished dataset preparation. See `phase2RunNotebooks.md` for the exact order and worker settings.

Notebook delivery: `releases/phase2_Kaggle_Notebooks_v1.zip` (216,275 bytes), containing six self-contained notebooks and the run guide. SHA-256: `38afe7d00c8e2e5abdb7a6311a441e8a1c385563db41a2eed0824c7cc36f78c3`. All six notebook schemas/code cells passed validation; each embeds seven runtime files with exact byte matches to the delivered sources. Executed NB00 remains byte-identical to its archived copy. All 376 protected original project files match their baseline.

## Update rule

At each subsequent Phase 2 step, update the table and append a dated entry with inputs, outputs, checks, decisions, blockers and precise local/remote artifact IDs. Keep planned, implemented, locally checked, GPU-verified, trained and evaluated states distinct. Record repairs/failures rather than replacing history with a success claim.

## Active four-notebook audit

User is running NB02–NB05 concurrently. Saved outputs and HF revision `9278e2a65ab45f1831810505d66523f7a6cd1930` confirm eight advancing seed jobs, passing T4 resume checks and checksum-matched remote checkpoints. Long training has started; previous pre-execution status rows are historical. NB02 schedules MobileNetV4 before ResNet50. Its latest uploaded classifier validation is a concern: both seeds confuse all 40 low validation images with mid, yielding 50% balanced accuracy despite low training loss. Seed 1 retains an earlier validation-best epoch. Full details and snapshot boundaries: `phase2ActiveRunAudit.md`. No active recipe/settings were changed.

## YOLO three-session handoff

Observed NB04 failure: an unexpired lease for yolo26m-combined-seed1 prevented a new worker from claiming it. User confirmed the original YOLO session was stopped and its final upload completed. Read-only HF audit found resumable seed-1/2 checkpoints at 23 completed epochs (4,628/4,627 updates), and no published seed-3 training checkpoint. Created three migration copies, IDs 0/1/2 of three, with one-time takeover enabled and explicit stop-all-older-YOLO-copies instructions. No remote lease was changed locally. Embedded runtime/model/source bytes are unchanged, preserving existing checkpoint compatibility. Verified each worker gets exactly its corresponding seed; notebook schemas, code syntax and ZIP integrity passed. Delivery: `releases/phase2_YOLO_3workers.zip`.

## 2026-09-21 — completion audit and workstation replacement

Audited immutable HF revision `be0eadb2bbb698091b563d699e2519b1e262a31c`: 14 completed jobs, one resumable. Verified JSON/log hashes, remote binary hashes, finite histories, T4 smoke, all completed validation epochs, held-out membership/labels and recalculated scores. NB06 partial status is correct. User explicitly deferred seed 3 and requested using completed models now.

Downloaded five validation-selected state dictionaries, verified SHA256 and retained old checkpoint files. Backed up application files before adapting loading, exact resizing/normalization, overlapping masks and six-point geometry. Latest explicit request authorizes these narrow existing-app changes. Local GPU tests passed strict model loads, training-adapter parity, repeatability, HRNet exported test coordinates, nine frames from all three known videos and evidence round trip. Internal metrics and video checks do not prove independent generalization; both selected classifiers score 50% on low/high test balanced accuracy.

Final verification: five existing contract tests passed; four-model GPU station smoke passed; paired desktop video tests passed for all three videos. Corrected the learned-diagram legacy seed/epoch caption and checked its rendered Phase 2 metadata. The existing launcher now selects Phase 2 via the ACTIVE marker; restart any existing process. Old checkpoint files are unchanged. Deferred and optional work remains accurately classified, not checked off as completed.

## 2026-09-21 — seed 3 and final NB06 complete

User finished seed 3. Verified HF revision `714a8daa65305f450e4e72ca82dd36b2922e7151`, all 15 runs ×60 epochs and complete report 1789972637. Seed 3 best epoch 8: validation Dice 99.1866%, test Dice 98.6091%. All five installed export hashes still match the completed registry; YOLO seed 1 remains best by validation. No weights were replaced unnecessarily. See phase2Seed3Completion.md.

## 2026-09-21 — configurable video export and visible angles

Added 1–60 FPS control (default 10, fractional values), snapshotted export settings and truthful progress/completion rates. Added optional fixed-height information footer and signed image-relative tread tilt on learned overlays. Calibrated camber/toe now appear on target-measurement images. No physical alignment values are inferred from ordinary video. Default/fractional/low/high FPS, invalid rates, duration/frame sampling, output-canvas stability, failure/cancel, real Phase 2 CPU export and synthetic calibrated-target checks passed. See phase2VideoControls.md.
