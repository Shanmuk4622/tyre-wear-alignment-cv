# Phase 3 — progress

<!-- NB11_COMPLETE_UPDATE_20261030 -->
## Current — NB11 complete; NB12 review ready

**30 October 2026. NB11 is complete and independently HF-verified:** 60 epochs / 2,940 updates, 1,351 direct files and 54,336,368 raw scientific rows. The actual T4×2/live-HF preflight passed; ownership is released. No NB11 rerun is needed. [Full completion and findings](phase3NB11Completion.md).

Validation selected epoch 7 at BA 56.25%. Selected/final historical endpoint BA remains 50%: 62 low→mid and 41 high→high. NLL and appearance stability improve over v1; low recognition does not, and selected raw-score ECE worsens. Keep the candidate as research evidence; no workstation promotion.

Next: [NB12 saved-evidence review](notebooks/phase3_NB12_TREAD_TWO_ATLAS_LC_Completed_Review.ipynb), **CPU / Internet / HF_TOKEN / no dataset attachment**, then Run All. It adds 40 derived CSVs / 380,595 rows, four figures and a 1,442-case explorer with zero fitting or model forwards. Local recovery/cache/publication/package checks pass; actual NB12/HF review publication is pending. [Run guide](tread_two/lc_review/README.md).

<!-- END_NB11_COMPLETE_UPDATE -->
<!-- NB10_LC_UPDATE_20261030 -->
## Latest status — NB10 closed; NB11 ready, 30 October2026

NB10 is completed and independently HF-verified:495 direct files,1,918,052 scientific rows,386native training records,T4×2,zero fitting/held-out forwards and released ownership. No NB10 rerun is needed. [Completion proof](phase3NB10Completion.md).

| Stage | State |
|---|---|
| NB08 / NB09 / NB10 | Complete; independently HF-verified |
| ATLAS-LC r2 implementation | Done; isolated source/recipe/namespace |
| Local recovery + package | Passed;24CSV families,285columns;15.02GB projected peak |
| NB11 actual T4×2/live-HF/fit | Pending Kaggle |

Next is [NB11 / ATLAS-LC r2](notebooks/phase3_NB11_TREAD_TWO_ATLAS_LC_Resumable_Pilot.ipynb): one separately versioned seed,paired LC views,18-photo source-SHA exclusions,all386queries and sqrt-size group sampling. Local batch8 recovery/package checks pass; actual Kaggle/live-HF preflight and measured accuracy remain pending. Use T4×2,Internet,the existing expanded Kaggle dataset and HF_TOKEN. [Architecture](phase3TreadTwoLCArchitecture.md), [protocol](phase3TreadTwoLCProtocol.md).

<!-- END_NB10_LC_UPDATE -->
<!-- NB09_GATE_UPDATE_20261030 -->
## Latest — NB09 verified; full local appearance gate / NB10 delivered

**30 October 2026.** Preserved the supplied executed NB09; all four cells and
30 embedded files match delivery. Independently checked all 50 published
objects plus manifest, all 30 CSVs / 246,157 rows, exact retained-precision
agreement and released owner. [Completion](phase3NB09Completion.md).
The updated executed NB08 is also preserved: its original 25-file payload and
1,177 published scientific objects remain unchanged; administrative restore
history is distinguished explicitly.

Completed the next gate locally: 386 training images, seven probes / 2,702
predictions, eight CSV families / 1,918,052 rows, all native packets and exact
score reconstruction. An 18-photo proposal passes structural coverage but
leaves two high groups with one query photo. Local-contrast features are less
brightness-sensitive on the declared unclipped subset, not a trained classifier.
[Findings/visuals](phase3TreadTwoAppearanceGate.md). Recovery and packaged-worker
checks pass. Next: actual NB10 dual-T4/HF execution. No fit, new tyre photo,
held-out forward or workstation update was added.

## Current update — ATLAS analysis and NB09 delivered, 30 October 2026

Completed the saved-data analysis: all 29,410 predictions, 23,520 draws,
2,940 optimizer records, 64,680 parameter-gradient rows, 60 selection rows,
824 probe pairs and 1,030 exact reference/patch logit decompositions checked.
The actual error is **low → mid**, correcting the previous all-high prose.
Both original endpoints remain BA50%; grayscale reduces both to zero, and
query dimming reduces final BA to zero. [Findings](phase3TreadTwoFindings.md).

Delivered 30 direct derived CSVs / 246,157 rows, three figures, a complete
interactive report and self-contained CPU NB09. Eight acceptance groups pass,
including byte-identical output reproduction from extracted notebook sources
and fake-HF lost-response/release/idempotence checks. Actual NB09 Kaggle run
and new review publication are pending. The completed fit, source/runtime and
both workstations are preserved. No new model forward pass or training occurred.

## Current update — NB08 completed, 30 October 2026

The user reported that execution stopped, then clarified that no error was
displayed. A pinned public HF check finds the measured fit completed at all
2,940 updates / 60 epochs, with selected/final endpoints, all declared
diagnostics and released ownership. No training restart or runtime edit is
needed. [Publication/report verification and remaining review](phase3NB08CompletionStatus.md).
The pending/rerun statements in the dated earlier entries below are historical.

## Latest update — NB08 transport repair, 30 October 2026

The supplied notebook completed the real dual-T4/pinned-HF gate, then its local
96-HTTP-requests/hour policy produced a false timeout during the training
launcher and cleanup. Public pinned logs still showed server capacity and no
429. [Diagnosis, preserved execution and corrected notebook](phase3NB08Repair.md).
The external launcher now uses proper HTTP windows, a separate hourly commit
guard, server-aware waits and snapshot repair provenance. All original 24
source/assets and the model/data/optimizer/selection/namespace remain unchanged.
Transport regression and packaged-worker checks pass. **Next: Run All with the
repaired NB08; measured fit/completion audit remain pending.**

## Latest milestone — TREAD-TWO implementation, 29 October 2026

**29 October 2026 — TREAD-TWO / ATLAS pilot implemented and locally checked.**
The [self-contained NB08 notebook](notebooks/phase3_NB08_TREAD_TWO_ATLAS_Resumable_Pilot.ipynb)
and [run guide](tread_two/PILOT_RUN_GUIDE.md) are ready. One seed-1 classifier
fit is gated behind actual Kaggle dual-T4/pinned-HF recovery and resource checks.
Local batch-eight fixtures pass pause/abrupt/fresh-process equivalence;
**no measured TREAD-TWO result exists yet**.

The isolated runtime retains 21 transactional CSV families, 247 column
definitions, complete native recovery, all epoch checkpoints, frozen-control
scores and full selected/final evidence maps. [Implementation](tread_two/PILOT_IMPLEMENTATION.md),
[schema](tread_two/PILOT_EVIDENCE_SCHEMA.md), [validation](tread_two/PILOT_VALIDATION.md).
Use existing data only. Both workstations and selected prior models are
preserved; the user deferred UI changes. The earlier prototype/app chronology
below remains history, not a request to rerun completed TREAD-ONE work.

<!-- STATION CURRENT -->
**Current scope, 29 October 2026: existing tyre data only. The user will not
collect more tyre photos.** The earlier assumed acquisition campaign is
superseded. The completed app deliverable is **[TREAD-ONE Studio](phase3TreadOneWorkstation.md)**,
a separate native workstation in [workstation/](workstation/README.md).
It uses the declared r1 seed-3/epoch-20 export, has its own launcher/model/Qt/
cache/evidence, and preserves the original workstation. Real model and UI
checks pass. [User guide](workstation/README.md).

Training and NB02–NB07 audits remain complete. Historical classification and
cohort limits remain; the prospective-readiness=false result is preserved as
research evidence and does not block this authorized desktop implementation.
The app is ready for existing media. The latest task plans TREAD-TWO; further
workstation changes are deferred. No new photos or NB07 rerun is required.
<!-- END STATION CURRENT -->

**Earlier NB05/NB06 completion, 29 October 2026:** NB05 is complete / independently HF-verified on both
T4s. Both live preflights and all 3,515 files / 1,561,734 core rows pass the
audit; the selected candidate regresses in classification and point accuracy.
[Completion](phase3NB05Completion.md). The failure atlas is complete, and
[NB06 review/publication](phase3NB06Completion.md) is now complete and
independently verified: 35 direct files / ten CSV tables. The
[data/protocol gate](phase3ProspectiveDataGate.md) has since been implemented as NB07; no further fit is launched.

Updated: 29 October 2026 (Asia/Calcutta).

**Completed state: r1 training and NB02–NB07 Kaggle execution/publication are complete and independently verified.** NB05's selected candidate has test BA 17.073% vs baseline 50%, with point error 1.885% vs 1.067% width. The user subsequently declined additional collection and authorized the separate TREAD-ONE Studio application, now complete. The latest work is the TREAD-TWO proposal/prototype above. Current workstation models stay installed; no further measured fit has started.

| Item | State | Evidence |
|---|---|---|
| Chat named `phase 3` | Complete | Rename tool confirmed the title |
| Separate Phase 3 working folder | Complete | This folder |
| Existing Markdown discovery | Complete | 103 paths / 94 distinct contents, including ignored/hidden Markdown |
| Context review | Complete for orientation | Full-source scan, direct priority-document reads, section excerpts and archived-copy comparisons |
| Folder/source ownership map | Complete | [Repository map](phase3RepositoryMap.md); selected launcher/adapter/engine/build entries inspected |
| Working understanding | Complete | [Understanding](phase3Understanding.md) |
| User training/HF/environment instructions | Recorded | [AGENTS.md](AGENTS.md), [Environment](phase3Environment.md) |
| Stale instructions reconciled | Complete for reviewed scopes | [Conflict ledger](phase3DocumentationConflicts.md) |
| Ongoing trackers | Established | Progress, SessionLog, Decisions, OpenItems, EvidenceRegister and Markdown inventory |
| Initial orientation checks | Passed | Initial 12-note/173-link snapshot retained in `audit/orientation-validation.json` |
| Conda execution health | Working activation route verified | `run_cv.cmd` activates through Conda's Python entry point; intended cv_conda/CUDA interpreter confirmed; hook warnings recorded |
| Custom architecture | Implemented | TREAD-ONE: 1,372,159 parameters, shared transformer/texture decoder, segmentation, ordered points, mask boxes, additive explanations |
| Experiment | Frozen for r1 | One architecture ×3 seeds ×60 epochs, safe Phase 2 split, no pretrained weights, joint validation selection |
| Model/resume checks | Passed locally | Real samples; full 512×384/batch8 fresh-process seam has zero differences |
| Publication fault checks | Passed offline | CAS, ambiguous successful claim, fencing, deduplication, corrupt/atomic saves, revision restore, rate limits |
| Short actual lifecycle | Passed locally | Durable pause/update resume, distinct final/selected evaluation and raw arrays; fixture scores are not study results |
| Kaggle notebooks | Generated and locally validated | NB00 CPU data preflight; NB01 dual-T4 training/evaluation; `releases/TREAD_ONE_Kaggle.zip` |
| Final custom-model delivery checks | Passed | 2,882 source files, 570 image/mask sets, two notebook payload/schema/ZIP checks, 191 local links; 103 original Markdown files unchanged |
| Kaggle training | Complete / HF-verified | Three seeds, 60 epochs, 2,940 updates each; actual T4 fresh-process checks pass |
| Completed-state audit | Passed | All bundle/member hashes, 8,820 updates, 180 validation epochs, labels/cohorts, both raw test endpoints and declared export checked |
| Result interpretation | Complete for r1 | Selected means: BA34.146%, Dice97.978%, point1.189% width; export seed 3/epoch20; all selected seeds miss held-out low tyre |
| Direct CSV recovery | Complete locally | 20 tables, ~32 MiB; includes retained histories and explicitly post-hoc metrics; no retraining |
| HF direct-CSV publication | Complete / independently verified | NB02 executed all 5 code cells; 25 remote file hashes, 18 schemas and 191,904 scientific records checked; [completion](phase3NB02Completion.md) |
| NB02 missing-directory repair | Fixed; exercised by successful Kaggle execution | Original failure and local regression preserved; supplied executed r2 embeds the exact corrected sources; [history](phase3NB02Repair.md) |
| TREAD-ONE classification diagnosis / NB03 | Complete locally / independently checked | 467 train/validation images ×6 fixed checkpoints ×5 variants; no optimizer update or new test evaluation; [results](phase3ClassifierDiagnosticResults.md) |
| Diagnostic recovery and faults | Passed | Actual GPU pause/resume matches five uninterrupted scientific CSVs; 12 offline direct-file/ownership/rate/corruption cases |
| Interactive image explorer | Generated; source/data validated | 78 fixed gallery examples, probability/mask/gate/evidence/boundary controls; browser rendering not verified because local-file preview was blocked |
| Diagnostic HF publication / NB03 | **Complete / independently verified** | Four code cells, no errors; all 27 file hashes, 336,600 core records, 60 summaries, 78 native maps and released writer checked at `75400f019ba0c6741b5060925b9c956c0a1ca0a8`; [completion](phase3NB03Completion.md) |
| Local vs Kaggle diagnosis | Passed class agreement | Zero disagreements over 14,010 rows; maximum probability difference 0.0007585883; environment contracts remain distinct |
| NB04 training-evidence implementation | Implemented in separate `training/` sources | 13 CSV families, actual augmented predictions/targets, sufficient statistics, gradients/activations/resources, direct native optimizer/RNG and atomic row cursors |
| NB04 local engineering acceptance | Passed real-model math/recovery and independent evidence audit | Original r1 math exact; pause/crash/direct-base resume; 11 scientific tables and 9 native arrays match; [receipts](training_gates/EVIDENCE_VALIDATION.json) |
| NB04 fault suite | Passed 22 checks / zero network | Five native-exit save boundaries, corrupt state/CSV, cursor drift, retention, CAS/ownership/ambiguous success/backoff |
| NB04 flush / signal controls | Passed local completed-update marker and SIGINT-entry recovery | Exact native numerical state after continuation; no mid-CUDA-kernel or live-HF Stop proof claimed; [receipt](training_gates/CONTROL_TESTS.json) |
| NB04 exact data dictionary | Complete | [13-family field/units dictionary](phase3TrainingEvidenceSchema.md); 261 column positions including repeated identity fields |
| NB04 Kaggle live-HF acceptance | **Complete / independently verified on both T4s** |572 Git/LFS/manifest files; both update1 sources and update3 states verified; both owners released; [audit](phase3NB04Completion.md) |
| NB05 controlled revision | Implemented in separate `revision2/` sources | Same architecture, one paired seed, original versus symmetric shared-only conflict projection; [protocol](phase3R2PairedPilot.md) |
| NB05 evidence/recovery | New implementation with its own acceptance | Local transactional journal, direct CSV export, immutable bounded HF batches, full validation and bounded map retention; automatic live preflight before measured training |
| NB05 local acceptance | **Passed** |16 offline/analytic checks; both actual-model restore trajectories exact; identical initial weights, samples, augmentations, native inputs and first-forward losses; original baseline math exact |
| NB05 preflight lease repair | **17 local regression checks passed** | Notebook-only safe wait; original rejection/expiry and real subprocess retry checked; all 27 experiment sources preserved; [repair](phase3NB05Repair.md) |
| NB05 actual Kaggle preflight | **Passed / independently checked** | Both pinned update-1 sources, exact native continuation to update 4 and 20 matching scientific CSV families |
| NB05 measured outcomes | **Complete / independently verified** | Both 60 epochs / 2,940 updates; 3,515 direct files and 1,561,734 core rows; selected projection regresses; [audit](phase3NB05Completion.md) |
| NB06 local review | **Completed** | Ten derived CSV tables, three inspected figures, all 103 test images / four endpoints in an interactive atlas |
| NB06 Kaggle/direct review publication | **Complete / independently verified** | Six successful cells, 35 hashed files, ten CSV tables / 758 rows; [completion](phase3NB06Completion.md) |
| Prospective data intake | Empty templates prepared | [Tyre/session/image registration pack](data_intake/README.md); zero new captures/subjects claimed |
| New independent tyre data | Standing priority; user will help later | [Data growth/provenance plan](phase3DataGrowth.md); no new captures claimed by this diagnosis |
| Future format/telemetry | Superseding rule recorded | No HF ZIP bundles; directly browsable CSV and native tensor/JSON state, richer schema before training |
| Evidence-retention priority | Reaffirmed and recorded | [Observation-to-design rationale and logging inventory](phase3ModelRationaleAndLogging.md); future metrics expansions must preserve the active r1 protocol |

Current authorities: [completion/insights](phase3CompletionAudit.md), [next steps](phase3NextSteps.md) and [retention policy](phase3DataRetentionPolicy.md). Earlier epoch-13/pending statements are historical. Full interrupted cross-Kaggle-session restart was not exercised in the supplied long-run record; do not conflate it with the passed fresh-process smoke/hash verification.

## Inherited state

Phase 2 training is recorded complete, all 15 runs and final NB06. Its selected weights are active in the existing workstation. Original-study work and newer paper results retain separate evidence. The selected classifiers' internal test weakness, small tyre count and physical/external validation limits remain explicit.

## Update discipline

After a task, refresh this table and append to [SessionLog](phase3SessionLog.md). Add decisions and precise evidence links where relevant. Preserve failed checks and partial work. Do not turn optional future studies into required reruns or label software checks as accuracy validation.
