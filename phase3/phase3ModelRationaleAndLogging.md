# Model rationale and training-evidence priority

Recorded: 27 October 2026, following the user's clarification while seeds 1 and 2 are running.

**28 October update:** r1 has completed and is [independently audited](phase3CompletionAudit.md). The user rejects HF ZIP publication and requires more retained telemetry; [the new direct-CSV policy](phase3DataRetentionPolicy.md) supersedes the original delivery design. Retained r1 records are recovered into CSV without retraining, with missing measurements still identified honestly.

**NB02 completion:** 18 core CSV / 25 files are now published and independently verified at HF `91ccd3315bd296a76a8b583a6d0f2515ccb3d370`. [The illustrated completion report](phase3NB02Completion.md) records execution, exact hashes, inventory and limits. The next model-related work is fixed-checkpoint diagnosis; publication does not change r1's classification failure.

**Later same-day NB03 update:** fixed-checkpoint diagnosis is now complete locally. All six endpoints fit the nominal training images, oracle gates barely change validation accuracy, and the fixed three-image batch shows opposing class/segmentation gradients. [Results and figures](phase3ClassifierDiagnosticResults.md). This supplies new post-hoc evidence; it does not reconstruct the missing training history. [New independent tyre collection with the user's help](phase3DataGrowth.md), richer logging and useful interactive image exploration remain standing priorities under TREAD-ONE.

## Standing user preference

Models should be tailored to this tyre project using what earlier experiments taught us. Creative choices should connect to an observed need rather than being complexity for its own sake. Give architecture, code, dependencies/resources, data flow, execution and recovery details special documentation care. Preserve as much meaningful training evidence as feasible, including fine-grained metrics, because repeated training should not be required merely to recover missing records.

The user explicitly describes this as a reminder for ongoing/future work, not dissatisfaction or a request to redesign the active experiment. No training code, notebook, recipe, checkpoint or remote artifact was changed in response.

## How earlier observations motivated TREAD-ONE

| Project observation | Corresponding design response | What remains a hypothesis |
|---|---|---|
| Selected Phase 2 classifiers scored 50% balanced accuracy on the small low/high test set despite higher validation scores | Couple classification to a shared supervised region representation; retain the declared validation selection and explicit test limits | Multi-task features will improve classification; this has not yet been demonstrated |
| Earlier colour/structure probes, paint/marking cues and split sensitivity exposed shortcut risks | Predicted-tread evidence gate, background-swap training and final background/occlusion diagnostics | The model will learn more transferable tyre cues; a spatial gate cannot eliminate wide-receptive-field shortcuts |
| Phase 2 segmentation had strong internal region scores and reusable manual masks already exist | Reuse those labels in the same network through overlapping tyre/tread heads, ignore handling and shared decoder features | Sharing with classification/geometry will preserve or improve segmentation |
| Original learned boundaries crossed/disagreed on workshop video, and HRNet/mask-derived boundaries supplied complementary evidence | Ordered left/span parameterization, shared guide-row profiles, mask/point agreement and visible review flags | Ordering prevents crossings structurally; it does not prove that the points are accurate or visible |
| Plausible saliency alone did not establish valid physical reasoning | Build classification logits from exact signed local evidence contributions; retain bias, latent maps and input-intervention diagnostics separately | Algebraic accounting is verified; physical explanation quality and input causality remain empirical questions |
| The workstation loads separate models and the local GPU is resource-constrained | One compact 1,372,159-parameter model, shared encoder/decoder and coarse transformer context | Whole-application speed/memory benefit needs a measured comparison; local smoke memory is not an application benchmark |
| There are only 12 tyres, limited acquisition diversity and unresolved mid-video mapping | Keep the safe Phase 2 group split, weak/human point distinction and explicit low/high-only holdout interpretation | A new architecture cannot create independent mid/video validation or measured-wear truth |
| Previous sessions exposed resume, memory, ownership and HF publication failures | Completed-update atomic state, real-model fresh-process gate, one publisher, immutable snapshots, fencing and bounded retries | Actual live Kaggle/HF operation still needs artifact verification |

The choices were therefore **informed by project evidence**. The exact channel counts, two transformer blocks, 12 latent channels, loss weights and selection utility are new engineering choices, not winners already proven by a controlled architecture/ablation search. The model is a custom composition of established operations. Its running results will determine whether the combination helps.

Relevant sources: [Phase 2 results](../phase2/phase2CompletionAudit.md), [installed-model contracts](../phase2/workstation/README.md), [original dataset probes](../docs/12_DATASET_FINAL_V1.md), [video boundary findings](../prototype/LEARNED_GEOMETRY_LOG.md), [custom design](TREAD_ONE_DESIGN.md) and [frozen experiment](TREAD_ONE_EXPERIMENT.md).

## What the current r1 runtime retains

Checked against `runtime/worker.py` and `runtime/evaluation.py`; this is a code-level inventory, not a fresh audit of the running Kaggle session.

| Level | Retained information |
|---|---|
| Every completed optimizer update | Total objective; component losses; background KL term; learning rate; pre-clipping gradient norm; AMP scale and numerical retry events; epoch/cursor/update; sampled image IDs; measured update compute duration; live memory report and GPU allocated bytes |
| Accumulated training state | Exposure counts by class/domain/tyre-or-clip; summed measured update compute time; full update history; next cursor; validation history and best epoch/score/weights |
| Every validation epoch | Per-image labels/predictions/probabilities/logits, tyre/domain IDs, two-region Dice/IoU, predicted/target boxes and box IoU, points/targets and eligible human error, evidence reconstruction residual, latent-channel mass and review flags; aggregate/per-tyre records and joint utility |
| Final and validation-selected test endpoints | Separate predictions/metrics, background-neutralization and evidence/image-removal diagnostics; raw mask probabilities, points, evidence maps, latent maps, gates, logits, bias and ordered image IDs in NPZ |
| Recovery/provenance | Model, optimizer, scheduler, scaler, all recorded RNG state, protocol/source/package/data identities, environment, smoke checks, logs/errors, selected/final distinction, member/file hashes and verified HF snapshot identities |

Records are saved locally after completed updates and included in consistent HF bundles on the frozen publication cadence. This does not mean every update has its own separate remote checkpoint. No new claim about the latest uploaded epoch or completeness can be made without inspecting the actual immutable HF artifacts.

## Honest gaps and future logging standard

“Every metric” is a retention priority, not a claim that r1 computes every conceivable statistic. The current code does not explicitly log training accuracy/F1 for each batch, full per-parameter gradient histograms, continuous GPU utilization/energy, or every validation pixel-probability map at every epoch. Its update `seconds` and accumulated `elapsed_seconds` describe measured update computation, not full wall time including image decoding, saving, validation and upload.

For a future revision, review the telemetry schema before launch and retain these when meaningful and feasible:

- Step/epoch training and validation loss breakdown, class confusion/precision/recall/F1 and ordinal errors, with denominators and cohort identity.
- Segmentation intersection/union and TP/FP/FN/support counts, coverage/missing predictions and boundary metrics with exact coordinate/ignore conventions.
- Per-point/per-guide geometry errors and validity/coverage, keeping human truth separate from weak targets.
- Explanation reconstruction, signed support/opposition, localization/intervention measures and failure cases.
- All optimizer-group learning rates, gradient/AMP health, sample exposure and numerical-retry history.
- Separate compute/data/checkpoint/evaluation/upload/wall timing, actual GPU/host/container memory and bounded utilization/energy telemetry where supported.
- Raw predictions or sufficient statistics for later recalculation, stable column definitions, source/metric versions, timestamps, hashes and explicit unavailable fields.

Many additional classification metrics can already be calculated later from saved validation probabilities/labels without retraining. Arbitrary new pixel-level validation metrics cannot be assumed recoverable when the required maps/epoch weights were not retained. Record such limits before claiming completeness. Stream/compress/shard large artifacts and preserve scalar/statistical records; avoid duplicating the source dataset or letting telemetry compromise training/storage/HF limits.

These are future requirements for review, not changes to the active r1 run. The present runtime/notebooks/release remain frozen.

## Historical epoch-13 report

**Later implementation update, 28 October:** [NB03](phase3NB03Completion.md) is
now Kaggle-complete and HF-verified. Its transfer/context/task-interaction
observations motivate [NB04](phase3TrainingEvidenceGate.md), the richer training
evidence and native/CSV recovery gate. Actual augmented predictions, loss
sufficient statistics, transformed targets, pixel/point/evidence records and
bounded gradient/activation/resource/timing samples are retained in 13 direct
CSV families. Local original-math/pause/crash/install checks pass; live Kaggle
recovery and a frozen long-run recipe remain pending. These engineering updates
do not change the completed r1 model or establish a better model.

The user reports **seed 1 and seed 2 at epoch 13**. This has not been independently checked against notebook outputs or a pinned HF revision. The wording does not establish completed updates, epoch-13 validation, last uploaded checkpoint or seed-3 status. Do not infer them or alter the active run based on this note.

That paragraph records the 27 October snapshot. The later complete-run audit now verifies all three seeds and supersedes it; do not treat it as today's active state.
