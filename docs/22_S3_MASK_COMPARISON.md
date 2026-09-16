# S3 mask comparison — NB11 delivery, 2026-09-09

<!-- current-status:start -->
> **Current status (15 September 2026):** [Completed work and remaining validation](CURRENT_STATUS.md). The report is refreshed; the app, learned-geometry integration and target-assisted alignment software exist. Dated plans below retain their original context.
<!-- current-status:end -->

## Current decision: deferred, no new annotation required

The user declines further annotation. **Do not run NB11 as a prerequisite for
S5.** Keep this notebook/protocol as optional future work; the instructions below
record its design, not a current assignment. Neither the SAM2 comparison nor
blind self-consistency has been measured, and neither is reported as passed.

Proceed with the existing manual-supervised mask route. S5 training detection
boxes can be computed directly from existing training masks; segmentation uses
the masks directly. Held-out manual masks are evaluation targets, never training
inputs for their fold. No second annotation pass is necessary to implement this
route. It does not resolve known fold/tyre leakage, certify annotation accuracy,
or supply S9's missing landmark/healthy-pool inputs. NBT1 remains evidence of
annotation/replay pipeline behaviour, not independent human consistency.

The 418 fresh rectangles were for the newly designed independent prompted-SAM2
comparison, not a fundamental requirement for all SAM2 use or for S5. Automatic
SAM2 alternatives exist, but are not run or substituted here merely to claim
completion. Using reference-mask boxes would instead be an oracle-prompted test
and must not be represented as independent mask-quality validation.

Status: notebook generated and locally checked; **not executed on Kaggle or
verified complete on HF**. The 418 manual masks already exist. No classifier
or existing annotation is changed by this delivery.

## What this measures

NB11 compares unedited **SAM2.1 Hiera Small** masks against canonical manual
tyre/tread regions, for all 418 clean images. It reports image, fold and session
IoU/Dice, empty masks, SAM confidence and tread-outside-tyre pixels. Empty versus
empty is undefined, not automatically perfect agreement. No examples are dropped
for poor scores and no manual mask is used to select SAM's output.

SAM2 is promptable, not a tyre/tread semantic classifier. This notebook therefore
uses **independent human box prompts** drawn from the image. It is a prompted,
uncorrected pseudo-label arm, **not a fully automatic zero-human-input arm**.
Do not derive boxes from the existing reference masks or reuse old polygons:
that would make agreement artificially dependent on the reference. Manual mask
hashes are used for provenance; mask content is used only for comparison.

The Small choice bounds memory for one T4 and does not substitute a different
classification model. Source and weight revisions are pinned to the inspected
[official SAM2 implementation](https://github.com/facebookresearch/sam2/tree/2b90b9f5ceec907a1c18123530e92e794ad901a4)
and [SAM2.1 Small weights](https://huggingface.co/facebook/sam2.1-hiera-small/tree/ee5bba1d82bb8749febdf90f45e84b687142ba03).
No epoch training or SAM fine-tuning occurs.

## Run instructions

1. Upload `notebooks/NB11_S3_Manual_SAM2_Agreement.ipynb` to Kaggle. Attach the
   existing Tire Dataset Prepared package; enable Internet and HF_TOKEN.
   Use exactly one notebook, not four accounts. Keep MODE="PREPARE"; CPU works.
2. Run All and download `S3_inputs_to_annotate.zip` from local Kaggle Output.
   Source photos are not enqueued to public HF. Keep the package private.
3. In labelme, do **blind_pass first**: 30 fixed images (10 per fold), independently
   reannotated as tyre/tread polygons plus marking/damage where visible, without
   viewing old masks or model outputs. Original dimensions and filenames must stay.
4. In boxes/, draw exactly two rectangles per image, labelled tyre and tread,
   from the image alone. Do not convert the old masks into bounding boxes.
   This covers 418 images; the work cannot honestly be skipped by copying ground truth.
5. Upload JSON-only boxes/ and blind_pass/ folders as a private Kaggle input.
   Set INPUT_ROOT to the directory containing those folders. Disable labelme's
   embedded imageData where possible; the notebook never republishes source JSON.
6. Select T4 x2 (only GPU 0 used), set MODE="RUN", and truthfully set both
   independence confirmations. Run All. It validates all inputs before GPU work.
7. Review the printed agreement and HF prefix. A failed human consistency gate
   is a result requiring review—not permission to lower the threshold.

## Interpretation and safeguards

The existing blind consistency gate is mean tread IoU **strictly greater than
0.90**, with all 30 images defined. This is a manual self-consistency check,
not SAM2's acceptance threshold. Agreement can be measured, but honesty of the
blind procedure requires the annotator's confirmation; software cannot prove it.

SAM quality is reported without inventing a new pass threshold. The resulting
STATUS remains `review_required`, even after all 418 masks are uploaded. Do not
silently call pseudo-labels fit for S5 merely because inference finished.
Marking/damage are not predicted SAM classes here; tread reference includes
marking and excludes damage per the existing canonical region convention.

Outputs live under `s3/s3-box-sam21-small-r1/<full-protocol-hash>/`. Each compressed
image record contains both masks, confidence and an input/protocol key. HF restore
only reuses matching records; corrupt/mismatched data stops rather than being
silently accepted. Prompt/model/code/input changes create a new namespace.
Local unpublished records are requeued on rerun. Original manual masks remain intact.

Normal commits batch every 30 minutes; protocol publication, inference completion,
catchable Stop and final tables trigger major flushes. A hard kernel kill cannot
flush: rerunning restores HF-published images and recomputes only missing images.
No per-image commit calls. Staging/model caches use the session staging directory;
the private annotation ZIP is capped at 2 GB of original image data before packing.

## What follows

- **S4b**: confirmation of NB06 technique findings; independent of this mask
  comparison unless a new chosen arm explicitly uses these pseudo-labels. Existing
  NB06 already has three architectures, so first reconcile the exact untested
  confirmation claim; do not duplicate completed configurations under new labels.
- **S5**: train/evaluate detector and segmenter arms after reviewing S3 quality,
  with training-fold labels only and manual held-out evaluation. Compare predicted
  ROI classification against the full-image/oracle references. Fix fold-integrity
  decisions before launching expensive new batches.
- **S9**: integrate after S5; landmark ground truth and curated healthy-pool inputs
  are still separate requirements. S3 completion alone does not supply them.

Local validation: `scripts/verify_s3_notebook.py` verifies prompt rejection,
canonical regions, undefined empty masks, strict consistency gate, atomic image
records, fingerprint mismatch rejection, deterministic blind selection and all
418 real reference mask dimensions. Generated cells parse. Full SAM2 GPU execution
and Kaggle Stop/resume still require runtime verification; no untested success claimed.
