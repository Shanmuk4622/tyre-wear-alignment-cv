# ATLAS-CONTEXT r3 — engineering verification and pending checks

<!-- NB14_COMPLETED_20261101 -->
## Completed context review — 1 November 2026

The NB13 fit and NB14 review are complete and HF-verified. All 80 review files / 52 CSVs match, and the original fit is preserved. [NB14 completion](phase3NB14Completion.md). The classifier remains unpromoted. Next is the implemented [NB15 existing-data readiness / evidence passport](phase3TreadTwoReadiness.md), not another full fit. Earlier pending-NB14 statements below describe the prior delivery. The original architecture/protocol/schema are unchanged.
<!-- END_NB14_COMPLETED_20261101 -->

<!-- NB13_COMPLETED_20261101 -->
## Superseding measured outcome — 1 November 2026

**NB13A completed and independently HF-verified:** actual recovery restored update **2,486** and reached **60 epochs / 2,940 updates** under the original protocol. All 1,503 direct files / 58,176,066 raw CSV rows, native/CSV history and all 60 unchanged frozen-prior epoch states verified; writer released. [Completion, figures and findings](phase3NB13Completion.md).

Do not rerun NB13/NB13A. Context did not repair low recall and worsened NLL; candidate remains unpromoted. Next is [NB14 CPU saved-data review](notebooks/phase3_NB14_TREAD_TWO_CONTEXT_Completed_Review.ipynb), [run guide](tread_two/context_review/README.md). Actual NB14 execution/publication pending. Earlier partial positions and pending gates in this document are dated history, preserved to explain the experiment and repair.
<!-- END_NB13_COMPLETED_20261101 -->


> Superseding recovery update, 31 October 2026: revision 1 failed because local working files were unavailable. [NB13A revision 2](phase3NB13ARepair.md) is now self-contained, prefers retained/attached saved state and otherwise restores the verified HF checkpoint. Use the updated notebook with Run All. Earlier existing-session-only instructions below are historical.

> Current, 31 October 2026: actual dual-T4/live-HF preflight passed; NB13 is partially trained and stopped. HF update 1,294 is verified; notebook output reaches local update 2,467. Preserve the existing session and use [the NB13A recovery instructions](phase3NB13FreezeAndRecovery.md). Earlier implementation/pending-execution statements below record the original delivery.

**Local engineering checks passed. No measured context accuracy exists yet.** Every Python check ran through `phase3/run_cv.cmd` in `cv_conda`, without replacing CUDA/Torch or dependencies. Actual dual T4/live-HF preflight remains required on Kaggle.

| Check | Actual evidence | Boundary |
|---|---|---|
| [Layout premise](tread_two/context_validation/LAYOUT_PROOF.json) | Eight known training tyres, 16 rearrangements; max logit delta 5.96×10⁻⁸ | No proof layout explains endpoint errors |
| [Model math](tread_two/context_validation/MATH_VALIDATION.json) | 231,555 trainable / 1,684,988 total; initial logits/evidence bit-equal to LC; frozen base unchanged after engineering update; finite gradients/empty support | Disposable checks, not a measured fit |
| Context expressiveness | Nonzero disposable-engineering band-order response: 0.0002171993 logit units | Shows ability to respond, not learned useful recognition |
| [Recovery and storage](tread_two/context_validation/LOCAL_VALIDATION.json) | Four actual batch-8 updates; uninterrupted, paused, abrupt-native rollback and fresh-process snapshot restore agree | Single local CUDA device; zero network requests |
| Native equality | Model, optimizer, scheduler, scaler, RNG, all deterministic orders/cursors and nine native artifacts agree | Timings/resources intentionally not equal |
| CSV equality | All 27 scientific families, including context and endpoint attention, agree | Engineering fixture contains 104 observations |
| [CSV-only accounting](tread_two/context_validation/CSV_ACCOUNTING.json) | 312 logits; 32 pair losses; four update losses; 32,768 attention values reconstructed | Zero model forwards or optimizer updates in this audit |
| Accounting precision | Max logit residual 8.27e-08; max pair loss residual 7.1e-08; max attention row-sum error 7.64e-08 | Retained FP32/CSV precision |
| [Fresh notebook package](tread_two/context_validation/PACKAGE_VALIDATION.json) | 33 payload files extracted; actual one-update worker matches reference native state and all 27 CSVs; 16 view rows | Kaggle execution not claimed |
| Interactive controls | Actual generated report JavaScript tested with a synthetic DOM: endpoint/error/empty/source filters, four cases, 18 references, separate base/correction fields | No native browser rendering claim |
| Capacity estimate | 14,372,641,128 bytes conservative peak < 16 GB guard | Actual Kaggle bytes/free disk remeasured automatically |

The 17 offline recovery/publication checks cover four transaction fault boundaries, CSV rollback/tamper rejection, source-contract drift, native/snapshot recovery without archives/databases, immutable snapshot isolation, remote fast-forward, bounded 80-object batches, ambiguous success, pinned-LFS hash verification, deduplication, owner fencing/release and shared budget/server-delay behavior. These are transport fixtures, not live-HF success claims.

## Scope of engineering examples

Recovery uses two validation examples as explicitly labeled fixtures and a two-epoch/two-update-per-epoch recipe. These engineering outputs are isolated from the future measured namespace and do not enter checkpoint selection or accuracy reporting. Train-only math/layout checks use existing cached train packets. The delivery is a single new measured candidate; no old fit or extra measured seed was run locally.

The preflight is intentionally short and disposable. Its full attention matrix accounting, frozen-base assertions and raw-byte extrapolation are exercised before a long run. No test expectation requires improved class accuracy.

## Deliverable and preservation

- [NB13](notebooks/phase3_NB13_TREAD_TWO_ATLAS_CONTEXT_Resumable_Pilot.ipynb): five code cells, 33 embedded files, notebook SHA `59506d92e267e36e37a98213e5f930b57597289c0a86b91f5d816baec1c49e8a`.
- [Frozen source inventory](tread_two/context_pilot/FILES.json): 32 source/asset entries; FILES.json is the 33rd notebook payload file.
- Executed NB08–NB12, v1/r2/gate/review sources, both workstations, earlier paired/r1 runtimes and original Markdown baselines are preservation targets in the final handoff check.
- Git ignores large fixtures/native outputs/HF mirrors and keeps only four explicitly allowed new reference-input arrays, small receipts, current notebook and curated proof figure.

## Pending actual execution

The user must run the new notebook on Kaggle T4×2 with the existing expanded input, Internet and HF_TOKEN. Only that run can establish actual dual-device/live-HF recovery, real storage footprint, final model metrics, learned layout response and final ownership release. Save the executed notebook. Then audit the pinned direct publication before another experiment or workstation decision.

[Architecture](phase3TreadTwoContextArchitecture.md) · [Protocol](phase3TreadTwoContextProtocol.md) · [Evidence dictionary](phase3TreadTwoContextEvidenceSchema.md) · [Run guide](tread_two/context_pilot/README.md).
