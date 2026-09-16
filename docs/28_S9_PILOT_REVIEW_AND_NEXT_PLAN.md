# S9 pilot: execution audit and next decision

<!-- current-status:start -->
> **Current status (15 September 2026):** [Completed work and remaining validation](CURRENT_STATUS.md). The report is refreshed; the app, learned-geometry integration and target-assisted alignment software exist. Dated plans below retain their original context.
<!-- current-status:end -->

**Implementation follow-up:** [NB23 and its run guide](29_S9_GEOMETRY_BASELINE.md)
are now ready. CPU, no attachments, saved predictions only. P06 is on an explicit
review hold, so you can run the diagnostic before revising P06. Kaggle execution
is pending; planning-only statements below are historical.

## Latest submission — second NB22 run, 14 September 2026

**Upload verified; 11/12 mechanically complete.** Saved NB22 output and public
HF main both report commit `05bf37c0f2067b0119ef057a2442d7759cf9ac51`.
The updated local annotation file exactly matches public HF bytes, as does the
manifest. Independent validation matches all published review records. There
are no notebook exception outputs. The same dependency warning remains nonfatal.
Only 20,838 remote bytes were read; no assistant HF writes or label edits.

New annotation SHA-256:
`4110bdcc413ff327d7ab2e6d107efac071008dc73b1bf04143a6c9a116516e18`.
[Current HF review](https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/tree/05bf37c0f2067b0119ef057a2442d7759cf9ac51/s9/s9-input-pilot-r1/87933cacc9cc06234dc150fbba749e469d81e57850666d51a2f455a746d3fd96/reviews/4110bdcc413ff327d7ab2e6d107efac071008dc73b1bf04143a6c9a116516e18)
and [revision-specific local audit](../outputs/s9_pilot_completion/4110bdcc413ff327d7ab2e6d107efac071008dc73b1bf04143a6c9a116516e18/AUDIT.json).

Only P06 changed; the other 11 annotation records are identical to the previous
submission. Left-middle and left-lower are now correctly recorded Outside frame.
Right-middle and right-lower still retain the old visible coordinates at the
cropped edge. The upper pair is unchanged. There are 70 visible and two
outside-frame points in total. P06's observation changed to `visible_issue`,
but its note is empty. **That missing note causes 11/12 completion**; outside-frame
points are valid annotations and do not reduce completion. HF correctly preserves
this revision as `partial_annotation`, so this is not lost progress or failed upload.

### Exact remaining action — P06 only

1. Load the latest saved JSON into the existing annotation page and open P06.
2. Select **right_middle**, then **Outside frame**. Repeat for **right_lower**.
   Do not move these points onto the image border. Review the upper pair only
   if the intended tread transition is unobservable; use Uncertain when needed.
3. Keep Visible issue if it reflects your observation and add a short factual
   description. Native-image review shows fine crack-like lines along the central
   grooves; for example, if this is the issue you meant: "Fine crack-like lines
   along the central longitudinal grooves." Do not claim depth or safety.
4. Confirm **12/12**, save the JSON, and return that small file. No new images
   or relabelling of the other 11. NB22 CPU is only needed to publish that new
   revision; do not rerun it unchanged or rerun NB21.

### Next after that

The next technical notebook remains a **predicted-mask geometry baseline**, not
HRNet training: reuse available segmentation predictions or limited inference,
check fold membership, and assess point error plus visibility/rejection coverage.
It has not been built or executed in this verification turn. The known-good 11
records can support preliminary work while P06 is finalised; full-pilot results
must explicitly exclude unresolved points or wait for review. PatchCore remains
pending because all 12 independent records are still unknown. No additional
annotation batch or GPU training is requested now.

The first-submission audit below is retained as history; its 12/12 count refers
to the earlier JSON, not this latest revision. Original local evidence is preserved.

## First submission — historical audit

Verified 14 September 2026. **NB22 completed successfully; annotation intake is complete, not training approval.**

## Execution and HF verification

The saved notebook reports 12/12 complete images and successful publication at
`1dad525affd32b06dc73f907edad73eb4096b782`, matching public HF main when checked.
There are no saved exception outputs. The installation output contains a
Transformers/Hugging Face Hub dependency conflict warning, but the intake itself
finished. No rerun is needed to repair this completed publication. Future
bootstraps should avoid disturbing unrelated installed dependencies.

The local `PILOT_12_IMAGES/MANIFEST.json` and annotation JSON match their public
HF counterparts byte-for-byte. Independent local validation matches the published
review records. All 12 embedded original images passed their manifest hashes.
Only 20,824 bytes of remote metadata/annotations were downloaded; no weights,
images or ZIP were downloaded. The remote ZIP itself was not byte-verified.
No assistant HF writes or annotation changes were made.

- Package: `87933cacc9cc06234dc150fbba749e469d81e57850666d51a2f455a746d3fd96`
- Annotation SHA-256: `b2815a30cb1b0599f3c72f977f4b708c79eb3f175f035372461823074e0c91aa`
- [Public review artifacts](https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/tree/1dad525affd32b06dc73f907edad73eb4096b782/s9/s9-input-pilot-r1/87933cacc9cc06234dc150fbba749e469d81e57850666d51a2f455a746d3fd96/reviews/b2815a30cb1b0599f3c72f977f4b708c79eb3f175f035372461823074e0c91aa)
- [Machine audit](../outputs/s9_pilot_completion/AUDIT.json)

The preparation status retains its historical awaiting-annotation state. The
new content-addressed review is the later intake record; this is not a missing run.

## Visual review of all 12 images

All 72 points were marked visible. All 12 observations say `no_visible_issue`,
all independent records are `unknown`, and notes are empty. Mechanical completion
is therefore correct, but does not establish that every boundary is observable
or that these are verified healthy references.

Reviewed overlays: [P01–P06](../outputs/s9_pilot_completion/pilot_review_1.jpg),
[P07–P12](../outputs/s9_pilot_completion/pilot_review_2.jpg), and
[P06 at native resolution](../outputs/s9_pilot_completion/P06_native.png).

**P06 requires a targeted visibility correction.** Its middle/lower left and
right clicks lie at the cropped image edges, not visible tread boundaries.
Those four points should be Outside frame when the intended boundary is beyond
the crop; use Uncertain if the transition itself cannot be identified. Review
the two upper points as well rather than assuming that proximity to the edge
means they are valid. Do not invent replacement coordinates.

The pilot also exposes ambiguity in our proposed boundary definition: tread
transition and external silhouette are not always distinct in these views.
That is a protocol issue to resolve before scaling, not a reason to ask for
hundreds of repeated labels. P12 is useful for clarifying this distinction;
disagreement with its existing mask alone does not prove its clicks are wrong.

Across all current points, the median horizontal difference from existing
manual tread-mask extrema is 7 native pixels (image width 1152). This is a
redundancy diagnostic, **not independent annotation accuracy or a model score**.
Both references can follow the same cropped edge, as P06 demonstrates. Existing
manual masks must not be presented as predictions from the segmentation model.

## Smallest user action

1. Open the existing `PILOT_12_IMAGES/ANNOTATE.html` and **Load saved JSON** using
   your current export before editing.
2. Go to **P06**. Review the middle/lower left and right point visibility as
   described above; inspect the upper pair. Keep the other images unchanged for now.
3. Save a new annotation JSON and return that small file. No images, new download,
   or reannotation of all 12 are needed. Keep the old JSON for provenance.
4. Run NB22 on CPU only if publishing this revised JSON yourself. Its new review
   hash preserves the previous submission. Do not rerun NB21.

## Next technical plan — proposed, not executed

### 1. Existing segmentation baseline before HRNet training

Use the existing segmentation system to extract the same boundary points and
compare them with the reviewed pilot. Prefer saved predicted masks if available;
otherwise run inference for only these 12 images. Use fold-matched held-out
models, not the prototype's fold-1 model for every image. Check actual split
membership and known overlap before interpreting any result as held-out evidence.

Report visible-point error normalised by image width, valid-point coverage and
rejection cases. Treat frame-clipped and ambiguous points explicitly. Keep the
manual-mask comparison separate. This is a feasibility diagnostic, not a
population benchmark or physical-angle evaluation. Do not tune and claim final
accuracy on these same 12 pilot images.

This should be the next small notebook, with bounded downloads and no training.
If segmentation already supplies useful geometry, HRNet may be redundant.
If not, first freeze an observable target and review a small next batch before
designing resumable training. No larger batch is requested now.

### 2. PatchCore reference decision

There are currently **zero verified healthy references in this pilot**.
“No visible issue” is an image observation, not evidence of measured wear,
safe condition or documented health. First define the anomaly being studied
and what qualifies a reference. A separately declared visual-nominality study
could be discussed if independent records are unavailable; that would require
an explicit scope decision, not silently substituting it for healthy-reference
validation. No PatchCore training is approved by NB22.

### 3. Integration after these decisions

Only after target/reference review and a frozen validation design should new
components be trained and compared against the working prototype with component
removal tests. HRNet/PatchCore are candidate components, not guaranteed final
models. Existing classification, S5 and exploratory S9 results remain complete;
the original full S9 claim remains open. Prototype changes and the frozen
manuscript evidence are preserved.
