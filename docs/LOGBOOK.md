# Project Logbook

<!-- current-status:start -->
> **Current status (15 September 2026):** [Completed work and remaining validation](CURRENT_STATUS.md). The report is refreshed; the app, learned-geometry integration and target-assisted alignment software exist. Dated plans below retain their original context.
<!-- current-status:end -->

## 2026-09-15 — Full report refresh and prototype/status reconciliation

Reconciled the user's later prototype work: native app, learned-geometry integration
and target-assisted alignment software already exist. Saved local evidence supports
software completion, not physical accuracy; video failures are retained. Refreshed
manuscript source, Markdown and HTML with new methods/results/discussion, 22 visuals
and 14 references. Original immutable evidence is unchanged; the new geometry/local
manifest is separate. All maintained Markdown has current-status navigation.
No model/notebook/prototype code edit, new inference or HF publication performed.

## 2026-09-15 — NB31/NB32 complete and independently HF-verified

Compared saved notebook outputs with current HF bbe586c6. NB31's saved logs are
incomplete, but HF verifies 3×60 epochs, 6,480 finite step records, 180 validation
files, three checkpoint LFS/status hash matches and the repaired zero-delta smoke.
Recomputed 432 paired points and all seed/tyre report means. HRNet 1.3118% width
versus SegFormer 1.7205%, 23.75% lower mean error; SegFormer coverage 100%.
Downloaded 1,013,075 bytes of metadata, no weights/images. Added docs/36 and
updated current status/claims. No notebook rerun, prototype edit or HF write.
Next: validation-only/predeclared checkpoint selection and integration validation.

## 2026-09-15 — NB30 verified; NB31 resume mismatch repaired

Read both executed notebooks and pinned HF preflight/smoke metadata. Preflight
passed (e32a8030); smoke failed (46511253), max weight delta 0.000166565, before
training. No run checkpoints exist in that snapshot. Added a separately hashed
deterministic resize adapter without changing the original contract or weakening
the smoke threshold. CPU native-gradient comparison and real-model stochastic
resume passed; T4 verification awaits rerunning updated NB31. Archived failed
NB31 before regenerating only that notebook. NB30/NB32 and HRNet outputs preserved.

## 2026-09-15 — Matched SegFormer comparison implemented

Added NB30 CPU preflight, NB31 single-session T4 smoke/resumable training and
NB32 CPU paired report. Frozen HRNet 72/24/24 image and 8/2/2 tyre allocation
reused, with the existing 72 training masks verified against S5 hashes. No old
S5 weights, new annotations, HRNet reruns or prototype changes. Matched budget
and endpoint; different dense-mask versus point supervision is documented.
Local real-model CPU forward/backward, overflow retry, stochastic continuation,
boundary failure cases and notebook validation passed. Kaggle results pending.
See [protocol/run guide](35_MATCHED_SEGFORMER_COMPARISON.md).

## 2026-09-15 — HRNet training and report completed and verified

NB28 final training commit 96166fb1 and NB29 report a92c0f9c verified. Three seeds
each 60 epochs, 2,160 finite step records; all 432 test-point records and constant
baseline recalculated. Checkpoint LFS hashes match STATUS, no tensor downloads.
Mean seed width error 1.3118%, mean pixel error 15.099; constant baseline 3.5547%.
Only two held-out tyres; unequal per-tyre error and no matched SegFormer comparison.
Repair provenance and passed GPU regression retained. Read 702,000 result/metadata
bytes; no HF writes, code/notebook rebuilds or label changes. Added docs/34 and
updated live status. Next same-split baseline and geometry integration decision;
PatchCore, physical-angle evidence and full S9 remain open.

## 2026-09-15 — NB26/NB27 verified and NB28 AMP recovery repaired

Public preflight 32d13296 and smoke 240dbb9e match notebook outputs and contract.
T4 smoke max parameter delta 0, peak allocation 613,136,384 bytes. NB28 emergency
commit 92f580df preserves seed1 epoch0 cursor18, 117 MB checkpoint; published LFS
hash equals STATUS. Read 135,573 metadata bytes, no checkpoint payload or HF writes.
Root failure: nonfinite-gradient guard raised before scaler recovery. Added explicit
checkpoint-compatible runtime adapter with same-batch retry, RNG restore, bounded
FP32 fallback and persistent-corruption guard. Model/split/LR/checkpoint protocol
unchanged; repair source/events published on future snapshots and included in NB29.
CPU real-scaler regressions and original-hash checks pass. New GPU regression runs
before NB28; actual repaired full training still unverified. Rebuilt NB28 and NB29
only, archived failed NB28 evidence, preserved executed NB26/NB27 and user labels.

## 2026-09-15 — HRNet preflight, smoke, training and report implemented

Recorded user's explicit confirmation of 12 different physical tyres. Added
NB26–NB29: frozen 8/2/2 tyre split (72/24/24 images), all 120 overlay QA outputs,
pinned HRNet-W18 feature adapter (9,603,962 parameters), GPU checkpoint-continuation
smoke gate, three sequential 60-epoch seeds with per-step local durable state,
30-minute/major/Stop HF commits, and small audited test report. All-visible labels
mean coordinate-only training, not visibility classification or healthy references.
NB29 constant baseline uses only training labels; matched SegFormer baseline remains
separate to avoid train/test overlap. No full S9 claim.
Passed local real-model CPU 512x384 forward, channel/parameter identity, split/hash,
notebook syntax, CPU optimizer/RNG continuation and pinned pretrained tensor-header
shape checks. Downloaded small code/metadata only, no pretrained tensor payload,
dataset or HF writes. GPU/AMP, pretrained loading and long training unverified;
NB27 is the required next runtime test after NB26. Existing annotations/notebooks
and prototype changes preserved. Updated PROGRESS and docs/32.

## 2026-09-15 — NB24/NB25 120-image submission verified

Both notebooks finish without exception outputs. HF package 78d90571 and intake
2b477391 match saved commits. Public/local manifest and annotation bytes match;
all 120 native image hashes and independently validated review records pass.
720 points marked visible, 120 no-visible-issue observations, 120 unknown records;
12 blank physical-tyre identities. Read 250,884 remote JSON bytes, no large download
or HF writes. Inspected two near-edge flagged originals; no automatic label changes.
Full point-overlay review remains pending. Updated docs/31 and live entry points;
next identity/split lock and HRNet smoke/training implementation. No new batch asked.

## 2026-09-14 — Single 120-image redesign requested by user

NB24 now delivers one ZIP and NB25 accepts one 120-image JSON. No BATCH setting
or review-after-12 gate. Native images are external files beside the small HTML;
all 120 hashes verified. ZIP 93,854,472 bytes (89.51 MiB), capped at 160 MiB.
Navigation/counter/import and server validation support 120, partial resume and
duplicate rejection. New r2-all120 namespace preserves old HF data. Local package
and syntax tests pass; Kaggle/browser execution pending. No labels or prototype
modified. Earlier ten-batch notes are history; final label/split review remains.

## 2026-09-14 — HRNet annotation expansion authorised and prepared

User accepts a larger annotation budget. Added NB24/NB25: 120 new originals,
ten 12-image batches, old pilot excluded, deterministic session-rotating allocation,
native JPEG preservation, blank factual identity ledger, hash-addressed HF package
and annotation intake. First-batch ZIP 9,785,017 bytes; local selection, validation
and notebook syntax tests pass. Kaggle execution and human labels pending.
Wrote docs/30: HRNet training design, identity-disjoint split gate, matched segmentation
comparison, checkpoint/resume requirements and separate PatchCore/alignment gaps.
No training notebook falsely claimed ready; it follows approved targets/splits.
NB23 verified separately at 84bcbdd61: 66/66 coverage, 8 px median, 0.90% mean width
error, P06 held out of scores. Existing notebooks/prototype/labels untouched;
no assistant HF publication. Current action is first annotation batch, not all 120.

## 2026-09-14 — NB23 saved-mask geometry baseline implemented

Added CPU-only NB23 and embedded runtime: fixed SegFormer-B0 seed-1 native
predictions, pinned verified annotation/source revision, P06 review hold,
per-run split/hash/epoch checks, explicit clipped/empty rejection and coverage,
mask diagrams and content-addressed HF output. No model/dataset download or
training. Public checks pass for all 12 source predictions; unit/syntax tests
pass. Local pycocotools installation stalled and was stopped, so end-to-end
decoding and Kaggle upload remain unverified. User execution is next. Updated
PROGRESS and run guides; no existing notebooks, labels, prototype or HF modified.

## 2026-09-14 — Revised NB22 intake verified (11/12)

Public HF and saved notebook commit match `05bf37c0f2067b0119ef057a2442d7759cf9ac51`.
New local annotation SHA `4110bdcc413ff327d7ab2e6d107efac071008dc73b1bf04143a6c9a116516e18`
matches HF bytes; independent validation matches review records. Only P06 changed:
left-middle/lower now Outside frame, right-middle/lower still visible; new Visible
issue choice lacks a note, causing 11/12 complete. This is successfully preserved
partial annotation, not a notebook failure. Read 20,838 remote bytes and inspected
native P06. Audit outputs now use annotation-hash subdirectories, preserving the
first audit. Updated live guides and docs/28 with exact small follow-up. No labels,
notebooks, prototype, manuscript or HF files changed. Predicted-mask baseline is
still the next planned notebook; no training or new annotation batch requested.

## 2026-09-14 — NB22 execution and real pilot reviewed

Verified saved NB22 success against public HF commit
`1dad525affd32b06dc73f907edad73eb4096b782`: local manifest/export match remote
bytes and independent validation matches published review records (12/12).
Read only 20,824 remote bytes; verified local embedded image hashes and reviewed
all 12 overlays, plus native P06. P06 middle/lower boundary clicks are cropped;
request targeted visibility review, not another batch. Median distance to manual
tread-mask extrema is 7 px, a redundancy diagnostic, not independent accuracy.
All independent records unknown; healthy-pool and training approval remain false.
Next proposed experiment uses fold-matched predicted masks before HRNet training.
Added docs/28 and refreshed live status. No labels, notebooks, prototype code,
frozen report or HF artifacts changed. Earlier pending notes below are history.

## 2026-09-14 — S9 small input pilot implemented

Read the prototype's inspection, video, geometry and calibrated-alignment updates;
preserved all prototype code/Markdown edits. Reopened only input feasibility,
not full pipeline training. Added NB21 CPU preparation and NB22 annotation-only
intake,12 original JPEGs from12 sessions,9.46MiB lossless package, offline point
visibility/triage page, three schematic worked examples, JSON save/import and
partial-safe validation. Proposed2-D boundary targets are not physical angles;
healthy-reference eligibility and training approval always remain false until
separate human review/design. No automatic new batch.

Local contracts, source identity, original-image hash/size checks and browser
skip/navigation/export/import passed. Test exports are software fixtures only.
Kaggle execution/human annotation pending. Read-only HF refs still22d5a6bc9;
no network downloads of models/images and no assistant HF publication. Current
instructions and scope recorded in docs/27, README, PROGRESS and notebook index.

## 2026-09-12 — Full manuscript and documentation handoff

Prepared the full illustrated report in `docs/report/REPORT.md` and `REPORT.html`,
plus references, claims/figure limitations, reproducibility and submission checks.
Added documentation index and additive repository guide; no notebook/source
directories moved or deleted. The source-generated tables and endpoint chart
use the pinned HF reporting package;14 upstream figures remain byte-identical.
With the new chart and study diagram the report contains16 visuals.

Downloaded only3.02MiB of small artifacts, verified38 source hashes and seven
report artifact hashes. No large dataset/checkpoint download, training, HF write
or secret exposure. Corrected current-facing12-session/tyre wording and exact
conformal coverage; retained selected/final distinctions, TER region limitation,
negative fusion, quarantine and deferred original components. Added reproducible
local build/validation scripts. Human/venue review remains; no false approval or
whole-original-project completion is recorded.

## 2026-09-12 — NB19/NB20 executed; S10 reporting package audit

Both uploaded notebooks finish with one commit and zero failures. Public HF
revision `22d5a6bc9f953ba3bf2a75919edc7db3193b317b`; code-version namespace
`s10/reporting-r1/2a333e2a6469905ad8cb822821ea46a357364e6d17bdd53c153c8b7e51fd217e`.
Audited38 source files against pinned originals/manifest, seven report artifacts,
14 figures and report image references. NB19 source revision35a178b5; NB20 evidence
revisione388dd4e. Recorded report-package completion, not manuscript submission or
whole-project completion. Added `docs/26` next plan: manuscript, claims/figure/
reference review, reproducibility handoff, human/venue review. Original experiments
and user-deferred components remain uncompleted. No notebooks or HF data modified.

## Scope decision — temporarily defer HRNet/PatchCore

User asks to put the two components aside and revisit after current project work;
willing to label if justified, but requires small downloads and clear guidance.
Recorded temporary deferral, not completion or permanent exclusion. Prioritise
supported results/reporting; no annotation requested now. Future labelling must
start with a defined scientific purpose, small pilot, worked examples, explicit
uncertainty rules and pilot review before scaling. Healthy-reference validity is
not established by low-wear proxy labels or unsupported image-only guesses.

## 2026-09-12 — NB18 complete and independently verified

Read executed NB18:81/81 processed, one successful85-file commit, zero failures.
HF revision `35a178b5a94878bdee95a0aa9ed8cb25cd1aeb6b`; implementation hash
`20dca3a2d91884230d555a7a36dabdbab8a075776487f3d147f1d4f9ddd21b87`.
Verified all81 payloads by recomputing from pinned S5 probabilities and recorded
classifier heads, including source/config/code hashes,56,430 predictions,
405 metrics and135 summaries. Both published CSV hashes/numerical values match.
Equal full+tyre+tread mean delta−0.01324; equal tyre+tread−0.01568, descriptive
and not significance claims. Updated completion across Markdown records; left
notebook/HF untouched. NB18 needs no rerun. Full S9 remains blocked rather than
silently closed or scoped down; missing landmark/healthy inputs/design remain.

## 2026-09-12 — NB18 classifier decision-rule repair

Uploaded NB18 processed four runs then failed on unet_r34-f1-s2 at an argmax
prediction check. Reviewed S5 evaluation code and frozen classifier configs:
CORAL uses cumulative threshold counting, not argmax. Fixed NB18 to decode each
single/fused probability vector using its recorded classifier head, with config
hash provenance. All81 public S5 inputs and405 metric rows now pass local tests;
single-view decisions reproduce saved S5 decisions exactly. HF
`2951194a7bf997126222689a42ea5f1d1831130f` preserves four old partial analyses.
Archived executed notebook before regeneration. No S5 retraining or HF writes.
Repaired Kaggle execution remains pending. Explained masks versus landmarks and
low-wear proxy labels versus independently verified healthy references.

## 2026-09-12 — NB18 exploratory S9 component/reporting notebook delivered

Reviewed Tier8 scope and remaining gaps. Full HRNet/PatchCore integration lacks
landmark labels and a verified healthy pool; did not fabricate inputs or silently
replace components. Built NB18 CPU fixed-fusion analysis for saved S5 predictions,
with five fixed arms, per-run resumability and HF30min/completion/Stop policy.
Added scope/gap ledger `docs/25`; corrected the experimental plan's misleading
claim that current masks provide landmark supervision. Local math/coverage and
notebook tests pass; real public-input checks passed on all three folds for
SegFormerB2 seed1. No GPU training, full81-run analysis or HF publication performed
by the assistant. User execution/verification remains pending.

## 2026-09-12 — S5 training and NB17 report complete

Read uploaded NB14/NB17 outputs: both finish successfully, zero upload failures;
NB17 prints81/81 verified. Independently audited public HF
`a3b29a71f8e6af6c50e68eb64a5bbae9ccf6d1c5`: semantic36, YOLO36, RT-DETR9,
all60epochs/evaluated with matching checkpoint and evaluation-artifact hashes.
Validated4,860 epoch rows, native/ROI/mask coverage, recomputed405 ROI macro-F1
rows and deltas, and compared published report inventory/run tables/summaries.
NB17's pinned input was `8ffef81e9aebf0464b19349c1956cc3afffb08fc`.
S5 now green; no rerun required. Updated current status across project documents,
retaining dated repair history and scientific limitations. S9 and wider proposal
work remain open. No notebook edits, training or HF writes in this audit.

## 2026-09-12 — NB14 interruption-safe local publication

User traceback shows RAM-guard KeyboardInterrupt followed by local snapshot hash
mismatch. The prior queue fix missed interruption between replacing local weights
and sidecars. Added a pre-replacement metadata journal and journal-based recovery
under the writer lock, including first-save interruption. Cache-aware cgroup guard
keeps dirty/writeback and live working memory counted; no model/recipe reduction.
HF `c9903960450f8fd9a16a5b148ff3da2100ef8654`:73 completed runs with matching
checkpoint hashes (semantic28, YOLO36, RT-DETR9); eight semantic jobs lack public
statuses/checkpoints. Local lost/unpublished progress is not claimed recoverable.
Offline interrupted-save tests pass; actual Kaggle run pending. No assistant HF writes.

## 2026-09-12 — NB14 upload-generation repair and progress audit

Read uploaded NB14 error: `Published checkpoint/status mismatch`, after skipping22
completed jobs. Public HF `971f0c7ad7f9e90aef8f2701e4a76f3f07367397` confirms
67 completed S5 statuses with matching checkpoint hashes: semantic22, YOLO36,
RT-DETR9. Directly inspected SegFormerB0 fold1 seed2 checkpoint: epoch45 with
full history/resume fields, versus stale epoch44 status. One resumable and13
unstarted semantic jobs remain. No GPU/model reduction needed.

Fixed queue-generation splitting/in-flight staging reuse risks using immutable
snapshot directories, atomic batch enqueue and serialized uploads/flushes.
Added validated semantic sidecar recovery with original-status provenance;
checkpoint is not reset or rewritten. NB14 defaults TRAIN + Run All. Added
offline fault-injection tests; preserved all uploaded S5 execution outputs
before rebuilding shared-source notebooks. Updated progress/run instructions.
No remote writes or training launched by this repair; Kaggle resume pending.

## 2026-09-10 — NB12/NB12R completed and audited

Read both executed notebooks and independently audited live HF revision
`dd43b231cfbdd92dd6d8c01b47166ddec4ab05f8`. All 18 runs have exact histories
1–60, completed status, matching configs/endpoints and both checkpoint paths.
Recomputed report matches publication. S4b board is now green; mixed sampling
results are not falsely called universal improvements. Runtime fixes completed
all jobs. No training, notebook overwrite or HF mutations in this audit.

## 2026-09-10 — NB12 MobileNet preflight r2

ConvNeXt repair passed at 0.82 s/step; MobileNet DP path was blocked at 12.04
s/step before training. Its Stage-A baseline also used one GPU (~48 s/epoch).
Enabled one GPU for both S4b models, preserving original batches/config hashes.
Timed guard remains. Actual config now controls smoke routing. HF still has
four resumable runs; repaired MobileNet throughput awaits Kaggle verification.

## 2026-09-10 — NB12 slow runtime, safe checkpoints

Public HF has four paused ConvNeXt runs at 28/26/25/26, not zero progress.
Forward-pass slowdown and RAM growth dominate; three RAM pauses plus watchdog.
Opt-in single-GPU ConvNeXt execution matches the faster Stage-A path with the
same batch/recipe and compatible checkpoint hashes. Added timed preflight,
recorded GPU-count telemetry, archived supplied output and rebuilt NB12/NB12R.
No remote writes; corrected Kaggle speed still unverified. See `docs/23`.

## 2026-09-09 — NB12 S4b training and NB12R reporting ready

Frozen HF selection/effects determine two extra architectures and three ranked
factors. 18 new runs; six reused baselines. No annotation or model substitution.
Added opt-in strict resume guard and CPU regression tests; generated both Kaggle
notebooks without changing executed notebook files. GPU training remains pending.
Full specification: `23_S4B_CONFIRMATION.md`.

## 2026-09-09 — Additional annotation declined; NB11 deferred

User authorizes skipping the newly requested boxes/reannotation. Removed NB11
as a prerequisite. Plan manual-supervised S5 using existing masks and automatic
training-box extraction. Record SAM2 comparison and blind self-consistency as
unmeasured, not successful. No new run or HF mutation. Other scientific/data
limitations remain in force; this is not full-plan completion.

## 2026-09-09 — S2 retained scope accepted; NB11 S3 delivered

Closed the retained S2 sweep green (17 architectures / 153 valid executions),
without counting the nine quarantined substitutions. Added NB11: PREPARE image
package, independent human boxes and 30 blind polygons, then pinned SAM2.1 Small
inference, agreement and per-image HF resume. CPU checks pass on all 418 native
manual masks; GPU execution remains pending. S4b remains separate from S3;
S5 needs mask review, S9 needs S5 plus additional inputs. `docs/22`.

## 2026-09-09 — Recovery execution verified

Read-only public audit at `bf62f9e9cbedacc580aa42542da14a068b8f9215` verifies
NB01A's 15 rows, NB01B's nine 60-epoch runs and final metrics/checkpoint paths,
NB03A's 153 valid / nine quarantined coverage, and NB10R's ten readable figures.
Updated the live PROGRESS board, roadmap and stage ledger rather than leaving
recovery tasks pending. S5/S9 and original-plan gaps remain unfinished.
Evidence and interpretation: `21_RECOVERY_COMPLETION_AUDIT.md`.

## 2026-09-09 — NB01A HTTP 429 repair

Saved output showed anonymous metadata access failing with a 128-second retry
instruction before computation. NB01A r2 now authenticates HF reads, retries
429/transient server errors respecting response delays, and skips unnecessary
repository listing. Original failing notebook archived; generated NB01A only
replaced. Fault-injection and recovery regression tests pass. No remote writes
or experimental changes made; Kaggle rerun remains pending. See PROGRESS session log.

## 2026-09-09 — Full-plan omissions corrected; recovery notebooks generated

Read all 27 existing project Markdown files completely. Public HF still at
`7c5b6461815a78aae589085d0152eaa6ae9995e1`: all five legacy S1 rows and
three 15-epoch ResNet-18 jobs were already present. The "2 of 5" board was
stale. Matched ResNet-50 training is a separate nine-run completion arm.
Generated NB01A, NB01B, NB03A (audit only), NB10R without touching executed
notebooks. Added S3 remainder, S4b, S5, S9 and original-figure gaps to PROGRESS.
Full mask audit: marking/damage-positive counts 67/58, 0/5, 0/0 by fold.
NB10R records H2 as inconclusive and generates privacy-preserving saliency
panels. No Small substitution, remote overwrite, S5 completion or scope cut
is claimed. `20_FULL_PLAN_CLOSURE.md` is the full current stage ledger.

## 2026-09-09 — NB08–NB10 completion verified against HF

Pinned audit commit: `7c5b6461815a78aae589085d0152eaa6ae9995e1`.
NB08: 63 unique stress rows, nine matching per-run tables, current control
mean 0.375184. A later fold-1 HF status supersedes the earlier control snapshot.
NB09: 27 prediction parquets / 3,762 rows; tables contain 39 ensemble, 6 TTA,
6 calibration, 3 conformal rows. The missing RegNet prediction file is restored.
NB10: 17 master rows, nine quarantine rows, three hypothesis rows, nine readable
figures. All submitted notebooks reached final cells without a traceback.

Figure 3 was skipped for absent saliency examples. H2 is undefined despite a
saved False boolean, H3 is not testable, and conformal coverage is
86.89/97.56/93.94%; saved abstention excludes empty sets. Figure 10's quarantine
filter requires review. Notebook execution is complete; reporting closure is
not. See `docs/19_NB08_NB10_COMPLETION_AUDIT.md`. This audit preserved all
executed notebooks and made no remote writes or training changes.

## 2026-09-08 — NB06 complete; NB08 evaluation gate repair

Verified all 108 Stage-B statuses as completed with last/best checkpoints and
final metrics on HF. NB08 trained all three shuffled-label controls but stopped
on validation-selected mean F1 0.498079. The existing docs/06 rule requires
fixed-final-epoch evaluation; full public 12-epoch histories agree with final
summaries at mean 0.353076. The repaired NB08 retains the threshold 0.45, reports
both scores in a revisioned audit, and reuses all completed controls. This
post-result correction is documented, not presented as a new preregistration.
Known fold leakage remains. NB08 embeds v12; its original error output is
archived. Local gate regression checks pass; Kaggle interventions remain pending.

> One entry per week, 30 minutes every Friday, as a group. This is not bureaucracy — at Review-3 you will need to remember why a loss weight was set the way it was, and you will not.

**Template**

```
## Week N — <dates>

**Planned:**  (from 07_ROADMAP.md)

**Done:**

**Broke / didn't work:**

**Decisions made (and why):**

**Numbers:**  (any measurement taken this week, with units)

**Next week:**

**Roadmap changes:**
```

> **Note:** day-to-day status now lives in **`PROGRESS.md`** at the repo root. This logbook keeps the longer-form decision record.

---

## Entry 15 — 2026-09-01 · Mixed epoch-history schema repaired

**Observed failure.** acct1 selected
`b-densenet121-prep_gray-f1-s1`, fetched its valid epoch-60 checkpoint and
history, then pandas raised `Expected 177 fields in line 5, saw 178`. The first
three epoch rows used the v4 schema; v5 rows included the new HF commit-policy
revision but positional append had not expanded the old header.

**Scope audit.** Public HF contains 70 Stage-B epoch files. Sixty-eight have
one internally consistent width; exactly two 60-row DenseNet histories mix 177-
and 178-value rows. Public execution is now 14 completed, 53 paused and 3
running. All 70 status-bearing runs have both checkpoints; nothing trained is
at risk.

**v6 repair.** The reader recognises the exact known revision insertion, adds
the missing header, pads only the old rows, validates every row, and atomically
rewrites the canonical CSV. Future epoch writes merge by column name and epoch
key rather than append position. Unknown width drift leaves the file untouched
and raises. The exact public failure file was tested losslessly: epochs 1–60 and
all 178 columns survived with CUDA and validation fields correctly aligned.

**Finalisation rule.** A checkpoint already at the configured 60 epochs is
finalised as completed metadata without being called “resume from epoch 61” and
without another training epoch.

**Next.** Stop v4/v5 copies, upload NB06 v6 to all four Kaggle accounts, keep
`NUM_WORKERS=4`, and Run All. No model or scientific setting changed.

---

## Entry 14 — 2026-08-31 · NB06 HF commit storm and pause cascade repaired

**Public evidence.** The live HF dataset now has 65 Stage-B status files:
**12 completed and 53 paused**. All 65 have both `ckpt_last.pt` and
`ckpt_best.pt`; every pause reports `host_ram_guard`, so no published epoch is
at risk and the earlier ROI-only memory explanation was incomplete.

**What made the notebooks look idle.** `run_all()` flushed every normal
static-owner claim as its own HF commit. It then continued to the next model
after a host-RAM pause. One already-pressured process produced many tiny partial
runs and reached 45 commits; the per-worker 25/hour limiter then printed
`sleeping 197s`. That wait was foreground scheduler I/O, not GPU training.

**v5 decision.** Automatic work stealing is disabled for NB06. Normal claims
ride the 30-minute batch; only an explicitly stolen claim flushes immediately.
Any paused run ends the worker after its checkpoint is published. The full
checkpoint is serialised once per epoch and atomically snapshotted to
`ckpt_best`; freed checkpoint/load/HF-upload arenas are collected and returned
to Linux with `malloc_trim(0)`. The 88% RAM guard remains.

**Scientific impact.** None: RegNetY-16GF, DenseNet-121, ResNet-50, fold 1,
input sizes, batches, optimiser, AMP and the fixed 60-epoch budget are
unchanged. The repaired notebook embeds tyrelib v5 and records memory,
scheduler and commit-policy revisions in every new epoch/summary.

**Next.** Stop all older v4 copies, upload NB06 v5 to acct1–acct4, leave
`NUM_WORKERS=4`, and Run All. If a v5 worker still reaches the guard, start a
fresh Kaggle session and Run All again; HF resumes at the next epoch.

---

## Entry 13 — 2026-08-31 · RegNet CUDA path and fresh-work race repaired

**New public evidence.** After the ROI host-RAM repair, two independent workers
reached RegNetY-16GF's first training batch. HF now has failed epoch-0 run
records for ROI seeds 1 and 2. Both fail inside RegNet stage `s2` under
`DataParallel`: `CUDNN_STATUS_EXECUTION_FAILED` / `CUDA misaligned address`.
The environments match (PyTorch 2.10.0+cu128, CUDA 12.8, timm 1.0.26, dual T4)
and telemetry shows only ~1.1 GB/card and ~2.6 GB host RAM. This is not an OOM;
neither attempt produced a checkpoint or completed epoch.

**CUDA decision.** Keep the NB07-locked RegNet model and the full registered
recipe. Tyrelib v4 runs RegNet with contiguous NCHW tensors and cuDNN autotuning
off, while retaining 384px, batch 32, AMP/GradScaler, AdamW and 60 epochs. Other
architectures keep `channels_last`. The runtime layout and safety revision are
logged. A fatal launch error is pushed to HF and stops the session before the
poisoned CUDA context can fail another model; cleanup no longer replaces the
root error with a second `empty_cache` exception.

**Scheduler evidence and repair.** The attached account-1 plan called all five
outstanding ROI jobs “picked up from a dead worker” and launched RegNet seed 1,
although the static owner was account 2 and that account also ran it. The cause
was treating a run with no registry event as stale/unclaimed. Fresh absent work
now remains with its static owner. Only a real event older than 45 minutes is
stealable; recent failed/paused work is protected, and its same account can
retry immediately.

**Current state.** Public HF has six Stage-B status files: four complete ROI
runs and two RegNet epoch-0 failures without checkpoints. Accounts 2 and 3
retry those seeds; account 4 owns RegNet seed 3 and ResNet seeds 1/2; account 1
has no remaining ROI job and must not duplicate them. No model, lite variant,
batch, resolution, optimiser or epoch-budget change was made.

---

## Entry 12 — 2026-08-31 · NB06 kernel deaths were host RAM, not the models

**Public progress.** HF contains exactly four completed Stage-B runs, each
with 60 epochs and both checkpoints: DenseNet-121 ROI seeds 1–3 and ResNet-50
ROI seed 3. RegNet registry claims reached epoch 0 and created no run directory
or checkpoint. The completed results are retained and will be skipped.

**Diagnosis.** DenseNet GPU peaks were ~5.0/4.9 GB and ResNet ~4.1/3.9 GB on
16 GB T4s, ruling out GPU OOM. ROI process RSS instead rose almost linearly
from ~3.3 GB to 20.3–20.7 GB over 60 epochs. The sequential ResNet run ended at
27.4 GB process RSS and 31.1 GB/94.9% host RAM; RegNet construction then caused
an OS-level kernel kill with no Python exception. Matched Stage-A full-frame
runs stayed near 3 GB, isolating the mask-based ROI path.

**Repair.** Tyrelib v3 (`2026-08-31-r1`) obtains the same crop from the mask
bounding box without full per-pixel coordinate arrays, closes images
immediately, uses an unpinned zero-worker loader for ROI, shuts loaders down,
and releases each model before constructing the next. At 88% host RAM it
finishes the epoch, checkpoints, pushes, and pauses. Four-copy configuration
now derives worker 0–3 from `acct1`–`acct4`, preventing an accidental all-zero
worker setup.

**Model decision.** No architecture or lite substitute is introduced. GPU
headroom was ample, and changing the NB07-locked set would invalidate the gate.

---

## Entry 11 — 2026-08-30 · NB07 completed and locked the Stage-B set

**Completed.** NB07 finished without an exception and verified its three
public artifacts: `tables/xai_evidence_all.csv` (1,208 rows),
`tables/xai_faithfulness.csv` (35 rows), and
`tables/stage_b_selection.csv` (18 rows), all under XAI revision
`2026-08-30-r3` where applicable.

**Decision.** The three-seed, XAI-valid Stage-B architectures are
RegNetY-16GF (`regnety016`), DenseNet-121 (`densenet121`), and ResNet-50
(`resnet50`). Their TER_norm/BAR values are 1.5785/0.0310, 1.5513/0.0455,
and 1.5146/0.0512. Raw valid-map coverage is 180/180, 178/180, and 180/180.
The selection rule excluded accuracy and used TER_norm with BAR as tie-break.

**Reporting correction.** The saved selection table's coverage denominator
was formed after missing TER rows had already been removed, so DenseNet was
displayed as 178/178 rather than 178/180. This did not change its mean TER,
BAR, seed count, eligibility, or rank. The notebook source now counts all
`xai_status=ok` rows in the denominator, and NB06 independently reconstructs
the raw public coverage before training.

**Next.** Run NB06 on Kaggle with dual T4s. It must print exactly
`['regnety016', 'densenet121', 'resnet50']`, then runs the ROI arm first and
the remaining fold-1 OFAT factors with three seeds. NB08 → NB09 → NB10 follow.

---

## Entry 10 — 2026-08-30 · Persistence formats need a numeric boundary

**Observed after the complete NB07 seed-1 screen.** All 18 public checkpoints
were processed and their r3 XAI files reached HF, but the final architecture
ranking raised `TypeError: agg function failed [how->mean,dtype->object]`.
No CAM computation was lost.

**Cause.** `evidence_metrics()` uses the literal string `NA` for an undefined
metric when a saliency map sums to zero. Newly generated in-memory frames mixed
that string with floats. A frame read back from CSV behaves differently because
pandas normally parses `NA` as missing. The notebook therefore passed its
resume path but failed its same-session path.

**Repair and evidence.** NB07 now coerces every numeric evidence field at all
four boundaries: resumed CSV, newly created frame, concatenated screen, and
three-seed summary. It tests a mixed `[1.25, "NA"]` column before any downloads.
The exact generated function was replayed against all 18 current public r3
files and ranked 10 valid screens successfully. Ranking tables now include
`n_valid`, `n_total`, and coverage so zero-saliency maps remain visible. The
seed-1 shortlist is RegNetY-16GF, ResNet-50, MobileNetV4, DenseNet-121 and
ConvNeXt-V2-T; seed confirmation is still required before the top three lock.

---

## Entry 9 — 2026-08-30 · Checkpoint identity beats the run-id label

**Observed on the second NB07 attempt.** The new exclusion path worked:
ResNeXt-50 and VGG-16-BN failed the locked randomisation gate, published their
failed rows, and the screen continued. It later stopped on
`a-convnextv2_s-base-f1-s1` because timm rejected
`convnextv2_small.fcmae_ft_in22k_in1k` as an invalid pretrained tag.

**Integrity finding.** The downloaded checkpoint declares `convnextv2_s` but
contains ResNet-18 tensors and about 11.19M model parameters. All nine public
statuses for that arm independently report `n_params_total = 11,177,538`, and
all nine best checkpoints are ~134 MB. The old emergency model-construction
fallback therefore substituted ResNet-18 for the entire arm. Execution
completion is not scientific validity: Stage A now has **153 valid runs and 9
quarantined records**. NB05 is unaffected and remains 27/27 valid.

**Repair.** Silent architecture fallback is removed. Checkpoint reconstruction
uses the untagged topology when checkpoint weights are supplied, checks the
saved tensor signature against the declared architecture before loading, and
publishes `excluded_checkpoint_arch_mismatch` rather than terminating NB07.
NB03 no longer schedules ConvNeXt-V2-S because timm has no pretrained Small
weights. The XAI revision remains r3; no metric or threshold changed.

---

## Entry 8 — 2026-08-30 · An XAI gate failure is evidence, not a notebook crash

**Observed on the first real NB07 run.** ResNeXt-50 produced good
insertion-minus-deletion values for both Grad-CAM and HiResCAM (~0.502), but
the old sanity score was 0.01297 and 0.01314. The old cell raised immediately
and stopped the entire architecture screen.

**Diagnosis and decision.** The sanity function constructed two images but
scored only `[0]`, and raw pixelwise MAE suppresses differences between sparse
CAMs. Revision r3 measures mean decorrelation across both images. The 0.05
threshold remains, now interpreted as a correlation drop. Separately, an
architecture for which no corrected candidate survives becomes an explicit
XAI exclusion: NB07 writes the failed rows and an
`excluded_no_faithful_cam` evidence row, frees the checkpoint, and continues.
It still requires five valid screens and three valid three-seed candidates.

**Recovery.** The corrected XAI revision is `2026-08-30-r3`; the r2 metric is
not mixed with it. No Stage-A model is retrained. The session did publish
`analysis/hypotheses.json` first, at
`2026-08-30T10:06:21Z`; neither the ResNet-50 nor ResNeXt-50 per-run XAI CSVs
reached public HF before the exception, so both are recomputed under r3.

---

## Entry 7 — 2026-08-30 · Real annotation PASS and Stage A closed

**NBT1 result.** The real Kaggle run completed all 22 FP32 epochs on dual T4s
and passed Parts A/B/C. The uploaded propagated masks were misaligned, so the
notebook exercised its intended self-healing path and rebuilt all 4,180 from
the 418 clean masks plus transform traces. The exact masks used have fingerprint
`085acfb8fb83c531`. Clean IoU was 0.9780; propagated IoU was 0.9747; their ratio
was 0.9966, compared with shuffled-mask 0.8029 and trivial-mask 0.6709. All
seven revision-specific outputs were verified in the public HF repo.

**Small error closed.** Ignored `DataLoader.__del__` assertions appeared near
epoch 18 because Jupyter/Python 3.12 was cleaning up repeatedly-created
multiprocessing workers for arrays already held in RAM. The run itself did not
fail. Those loaders now use `num_workers=0`, which removes the cleanup race.

**Stage A result (superseded by Entry 9's checkpoint-identity audit).** All 162
expected execution records are public. 160 are labelled `completed`; two VGG runs reached epoch
60 and contain every scientific artifact but retain an old telemetry-only
`failed` status. Fold 0 has 53/54 perfect runs and fold 2 has 46/54, consistent
with the known leakage flags. Fold 1 remains discriminative and is used for the
next selection and OFAT stages. NB05 contributed all 27/27 foundation runs.

**Pipeline decision.** NB07 now precedes NB06. It selects architectures using
faithful, seed-confirmed evidence location rather than accuracy alone and
publishes the locked top three. NB06 refuses to start without that artifact and
runs fold-1 OFAT only. NB08–NB10 were regenerated with resumable public-HF
inputs, a non-colliding shuffled control, real flip TTA, disjoint calibration
and conformal splits, and conditionally generated final figures.

---

## Entry 6 — 2026-08-30 · NBT1 made fail-loud, dual-GPU and mid-epoch resumable

**Problem.** The annotation verification notebook could stop before saving its
diagnostic overlay, checked well-formedness on only 200 of 4,598 masks, used
only one T4, and resumed only at epoch boundaries. A local execution also
reproduced a more serious failure: FP16 produced a non-finite U-Net loss while
GradScaler silently skipped the optimiser steps, leaving a run that could still
reach a normal-looking final report.

**Decision.** Annotation verification prioritises numerical reliability over
speed. NBT1 Part C now uses FP32 and raises immediately on a non-finite loss.
Dual T4s are used through `DataParallel` with eight images per GPU.

**Persistence changes.** Checkpoints are atomic and written every five batches
and every epoch. The checkpoint includes optimizer, scheduler, scaler, Python,
NumPy, CPU/CUDA RNG states, partial-epoch loss counters, next-batch cursor,
configuration hash, notebook revision, and mask fingerprint. Batch order and
horizontal flips are deterministic by epoch/sample, so a restart continues at
the next batch without reshuffling the remainder. SIGTERM and Ctrl-C save the
current cursor before a blocking push. Hugging Face commits run from an actual
30-minute timer with a 100-write/hour cap and retry/backoff. The repaired run
uses `annotation_test/2026-08-30-r1/`, leaving the legacy attempt untouched and
preventing its old FP16 checkpoint from being resumed accidentally.

**Other repairs.** Part A now validates all 4,598 masks and manifest links. Part
B saves its overlay before stopping. Empty optional transform groups are
reported rather than converted into a false failure. Incompatible or unreadable
checkpoints raise instead of being ignored and overwritten.

**Verification.** The regenerated 19-cell notebook passed an end-to-end local
run against a synthetic package containing exactly 418 clean images and 4,180
derivatives. A forced interrupt after batch 2 created an emergency checkpoint;
a new process restored it at batch 3 and completed with Parts A/B/C passing.

**Closed by Entry 7.** The real Kaggle run passed and the revision-specific
Hugging Face verification is recorded in `PROGRESS.md`.

---

## Entry 5 — 2026-08-26 · Approach redesigned as a comparative XAI-grounded study

**Dropped:** the single engineered pipeline (SegFormer → ConvNeXt → HRNet → PatchCore → fusion). It required labels that do not exist. **Dropped:** all hardware — no rig, no camera, no illumination array, no calibration, no jig.

**Adopted:** a broad comparative study. ~30 architectures × 12 technique factors × 3 folds × 3 seeds, plus detection and segmentation via SAM2 pseudo-labels, with **explainability as the measuring instrument**.

**The reframing.** A naive "which model is most accurate?" benchmark would rank noise — our probes show a 0.12→0.98 fold swing. The question became **"which model actually looks at the tread?"** New metric family: TER / BAR / SAR / DAR / EDI, computed from SAM2 masks with zero manual annotation. Hypotheses H1–H3 drafted for pre-registration.

**Key research finding that changed the design:** Grad-CAM on a ViT is not the same operation as Grad-CAM on a CNN — several published "Grad-CAM for ViT" methods compute gradients on attention entries rather than channel-averaged feature maps. Since the whole study compares attribution across architecture families, a naive comparison would have corrupted the headline result while looking entirely reasonable. Hence the hard rule in `docs/14 §1`: architecture-appropriate methods, selected by faithfulness, choice reported.

**Other findings:** CAM→SAM prompting gives pseudo-masks with no annotation · YOLO26 (Jan 2026) supersedes YOLO11 · modern CNNs still lead under limited data · **tyre wear is a fine-grained visual classification problem** and FGVC methods appear unexplored on tyres — the most promising model-axis novelty.

**Infrastructure** adopted wholesale from the supplied Replication Playbook: per-writer registry shards, per-token rate limiter, LPT bin packing on a static cost table, lifecycle guards, ~170-column telemetry, base64 notebook bootstrap, preflight with a real kill-and-resume test.

**Alignment deferred** — and flagged as the *harder* half, not the easier one.

Written: `13_EXPERIMENT_PLAN.md`, `14_XAI_PROTOCOL.md`, `02_CAPTURE_AND_PREPROCESSING.md`. Rewritten: `README`, `04`, `05`, `06`, `07`. Superseded: `02_RIG_BUILD.md`.

---

## Entry 4 — 2026-08-26 · Project-understanding document

Dataset README re-read after expansion (348 → 428 lines). The new *"how to understand the image folders"* section clarifies that `clean`/`augmented` is real-vs-artificial, `fold_n` is a **cross-validation group and not a wear category**, and `low/mid/high` are odometer-derived proxies. Folded into `docs/12 §1b`. No prior analysis invalidated.

Wrote **`docs/00_WHAT_THIS_PROJECT_IS.md`** — a plain-language account of the project written to be checked: what we're building, why the two tasks must stay separate, what we do and don't claim, where the data actually stands, and what should happen next. It is now the entry point to the repo.

---

## Entry 3 — 2026-08-26 · Pilot dataset `final_v1` analysed

Read the full package, viewed the images, ran two difficulty-floor probes. Wrote `docs/12_DATASET_FINAL_V1.md`, `scripts/dataset_shortcut_probe.py` and `PROGRESS.md`.

**Headline:** the package is 4,598 files but **12 independent tyres**, captured in one 22-minute window. Colour-only baseline scores 0.952 macro-F1 on fold 0; structure-only scores 0.976 on fold 2; both average ~0.49 across folds. Neither learned wear — both memorised tyres.

**Decisions:** report all three folds always, with both baselines in the same table · labels described as mileage proxy, never worn/not-worn · tyre-region crop as a standard ablation · data collection prioritised over modelling · pilot classifier built for the training harness, not its accuracy.

Full detail in `docs/12_DATASET_FINAL_V1.md` and `PROGRESS.md`.

---

## Entry 2 — 2026-08-25 · Documentation rebuilt on the Review-1 specification

**Context.** The repository docs had been written against a wrong assumption about the capture setup (a ground-embedded glass-plate rig). The Review-1 report and `VISION_MODELS_AND_FILTERS_README.md` define the actual setup: **a low-mounted camera ahead of the wheel, facing the front of one tyre.** All documents rewritten accordingly.

**Rewritten:** `README.md`, `docs/01`–`04`, `06`–`09`, `ENVIRONMENT.md`, `environment.yml`, `CITATION.cff`, `LICENSE`, `GITHUB_SETUP.md`.
**New:** `docs/10_VISION_TECHNIQUES.md`, `docs/11_APP.md`.
**Removed entirely:** every reference to FTIR, glass plates, contact-patch imaging, drive-over rigs and the rolling-constraint blur argument. None of it applies.

**Research pass — findings that changed the design:**

| Finding | Source | Consequence |
|---|---|---|
| **Photometric stereo** is proven for defect detection on specular/low-contrast industrial surfaces; a static ring light provably cannot distinguish a stain from a shadowed cavity | [Sensors 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC8838491/), [MVA 2021](https://link.springer.com/article/10.1007/s00138-021-01244-z) | **Largest addition to the stack.** 4 LEDs (~₹2,000) turn the camera into a surface-geometry sensor. Now Core + Ablation #2 |
| **clDice / Skeleton Recall** topology-preserving losses, explicitly proposed for industrial crack detection | [arXiv 2003.07311](https://arxiv.org/pdf/2003.07311), [2404.03010](https://arxiv.org/html/2404.03010v1) | Added to `L_seg` for sipes and cracks. Connectivity is now a reported metric, not just IoU |
| **SAM2 memory propagation**: annotation throughput 37.8 s/frame → 4.5 s/frame; FS-SAM2 gains from ~50 imgs/class | [arXiv 2509.12105](https://arxiv.org/html/2509.12105) | Adopted as the core annotation workflow. This is what makes 300 tyres feasible |
| **Frozen SSL features give no clear advantage** on RGB industrial tasks; fully fine-tuned SSL init is strongest | [arXiv 2605.23472](https://arxiv.org/html/2605.23472) | Changed the training recipe: initialise from SSL, **fine-tune the whole backbone**, do not linear-probe |
| **Depth Anything V2** documented weak at fine detail and close range | [arXiv 2406.09414](https://arxiv.org/html/2406.09414v2) | Rejected for metrology. Kept as a **negative-result ablation** |
| **Ko et al.**: stacking depth + equalised depth + height map improved mIoU by >7 points | [doi:10.3390/app112110376](https://doi.org/10.3390/app112110376) | Supports the `[RGB \| normals \| albedo \| CLAHE]` input stack |
| **Huber TireEye**: 0.57 mm using TWI bars as an in-frame scale reference | [doi:10.36001/phmconf.2022.v14i1.3242](https://doi.org/10.36001/phmconf.2022.v14i1.3242) | TWI anchor formalised as a training loss term. Benchmark to beat |
| **Vivekanandan & Rajeswari**: unseen-brand accuracy 88.2% → 92.4% with domain adaptation | [doi:10.1016/j.measurement.2026.121509](https://doi.org/10.1016/j.measurement.2026.121509) | Brand shift is a **measured** gap. Unseen-brand split mandatory from day one |

**Decisions:**

- Illumination promoted from acquisition detail to the project's most distinctive design choice
- `L_seg` = focal + 0.7·dice + 0.3·boundary + **0.2·clDice**
- SAM2-assisted annotation before scaling collection, not after
- Backbone: SSL init, **fully fine-tuned**
- Toe positioned as **binary screening (AUROC)**, continuous MAE secondary
- Resolution budget stated explicitly: **0.3 mm sipe at 3 px requires ≤0.1 mm/px** → ≥8 MP sensor or a cropped-region claim
- Team split by subsystem with four written interface contracts

**Numbers established:**

- Resolution: 1080p across a 250 mm tread = 0.130 mm/px — **insufficient for sipes**; 12 MP = 0.062 mm/px
- Reference benchmarks: on-board optical depth **0.57 mm**; structured light **<0.2 mm**; marker stereo alignment **~0.025°**; RGB-D alignment **<0.1°**; front-view tread segmentation **mAP 0.608**

**Novelty audit (see `09_RELATED_WORK.md §6`):** strongest claims are the wear↔geometry cross-check (recent vs chronic misalignment — nothing comparable found), the dataset, the four-modality controlled comparison, and coverage-reported unrolling. TWI anchoring is prior art (cite Huber). The app is not a research contribution.

**Next week (P0):**

- [ ] Photograph a tyre from the intended viewpoint. **Can you see a sipe? A TWI bar?** Measure mm/px against a ruler in frame ← GO/NO-GO
- [ ] **Hand-torch photometric test** — 4 torch positions vs 1 flat photo. Decides Ablation #2
- [ ] Buy the digital tread depth gauge; measure 4 tyres; longitudinal study starts
- [ ] Assign the four roles and write the four interface contracts
- [ ] Order long-lead parts: camera, lens, polarising film, LEDs
- [ ] All four members read Tier-1 papers (`09_RELATED_WORK.md §7`)

---

## Entry 1 — 2026-08-09 · Repository initialised

Initial scaffold created. *(Superseded by Entry 2 — the capture setup assumed here was incorrect.)*

---

## 2026-09-01 — tyrelib v7

NB06 workers were stopping themselves. Two causes: the host-RAM guard compared the epoch's transient **peak** against 88% and any pause ended the whole cell (Bug 22), and the telemetry buffers were never cleared, leaking **0.54 GB/epoch** until Kaggle killed the kernel (Bug 23). The guard now measures live after releasing memory, uses 88/80 hysteresis, and continues to the next run when the pressure was the model. Telemetry writes incrementally and drops what it wrote. 12 new selftests; all 12 notebooks regenerated on v7.

## 2026-09-01 (b) — tyrelib v8

"2 of 4 workers stopped." Not a crash: `steal_stale=False` made other accounts'
runs permanently unreachable, so a worker that finished its static shard planned
zero runs and exited while others had twenty left (Bug 24). That flag was itself
the fix for Bug 13's duplicate training. v8 answers the question properly with a
two-phase claim — flush the claim, settle, re-read, lowest account wins — used
only once a worker has exhausted its own shard. Four threads racing six runs
produced exactly one winner each. 95 selftests; all 12 notebooks on v8.

## 2026-09-01 (c) — tyrelib v9

A legitimate-looking RAM pause at 89.6% turned out to be measured against the
host's `/proc/meminfo` rather than our cgroup (Bug 25) — so every pause decision
ever made used a number that did not describe our budget. Now reads
`/sys/fs/cgroup/memory.*` and reports which process holds the memory. Separately,
the paused run had logged `dl 0%` on all 49 of its epochs while still running two
pinned loader workers whose RSS counts against the same cgroup (Bug 26); at
≥320 px the loader is now synchronous. 104 selftests; all 12 notebooks on v9.

## 2026-09-02 — tyrelib v10

The latest NB06 output was described as a one-worker attempt, but its own
session banner said `worker=0/4` and the HF cap was 25/hr. It was still using
the four-account plan. The output also proved DenseNet was training: epoch 37
finished in 3.7 minutes, while the serialized epoch-38 widget remained at 0%;
HF later recorded that exact run completed at epoch 60. v10 makes
`ACTIVE_KAGGLE_ACCOUNTS=('acct1',)` the single default source of parallelism,
derives `NUM_WORKERS`/`WORKER_ID`, and prints `MODE=ONE NOTEBOOK`. Every epoch
gets a plain first-batch heartbeat, and the weight-norm telemetry warning is
fixed with an explicit detached no-gradient norm. Public audit: 36/108 Stage-B
runs complete, 39 checkpointed incomplete, 33 absent; all 75 status-bearing
runs have both checkpoints. 106 selftests; model and scientific recipe
unchanged.

## 2026-09-03 — tyrelib v11

v10 completed two RegNet runs, then advanced a third from epoch 3 to 45 before
the live cgroup reached 88.1%. The checkpoint and pause were correctly pushed,
but `run_all` ended. HF epoch telemetry shows the long-lived Jupyter process
retaining about 0.17 GB/epoch on both inspected RegNet runs and 0.30 GB/epoch
on the DenseNet tail even after synchronous loading, telemetry drains,
`gc.collect`, and `malloc_trim`. v11 runs each NB06 model in a disposable child
Python process. The OS reclaims the complete model/optimiser/CUDA/native state
when it exits; a RAM-paused child automatically resumes the same public
checkpoint in a clean child. The parent keeps scheduling and now guards every
new run against the final 45 minutes of the real Kaggle session. Public state:
42/108 complete, 34 checkpointed incomplete, 32 absent; all 76 started runs
have both checkpoints. 114 selftests; scientific recipe unchanged.

## 2026-09-08 — tyrelib v12

Four workers advanced Stage B to 101/108 complete. The public repository now
has 104 Stage-B status files: 101 completed and 3 session-watchdog pauses, with
both rolling and best checkpoints for every started run. The remaining four
planned runs have no files at all, so they are not lost work and the true
at-risk count is zero.

The attached NB06 stopped in Cell 3 because
`ACTIVE_KAGGLE_ACCOUNTS=('acct1')` is a string, not a one-item tuple. v12
normalises that common missing-comma edit to `('acct1',)` while preserving the
validated four-worker tuple and rejecting duplicate labels. Final HF
verification now prints never-created work as `NOT STARTED`; `AT RISK` is
reserved for partial artifacts without a usable checkpoint. 118 selftests;
model, data, split, optimiser, batch sizes, and 60-epoch recipe unchanged.
# 2026-09-10 — S5 notebook delivery (not GPU execution)

Built NB13–NB17 for nine manual-supervised detector/segmenter configurations,
81 planned jobs, no additional annotation. Corrected RT-DETR identity to actual
v2-R18; declared DeepLabV3+ ResNet-34 encoder. Frozen data/protocol, isolated job
processes, one upload owner,30min snapshots, pilot/resume gate, native held-out
predictions and paired full/predicted/oracle ROI classification. Public HF at
`dd43b231cfbdd92dd6d8c01b47166ddec4ab05f8` had no S5 artifacts. Existing executed
notebooks were preserved. GPU/backend verification awaits Kaggle PILOT runs;
S5/S9 are not marked complete. See `24_S5_MANUAL_DENSE_TASKS.md`.
# NB14 setup repair — 2026-09-10

Read the uploaded NB14 outputs: blank PREFIX assertion before training; four
accounts configured in PILOT. Verified published NB13 at HF
`951435416dde3bc65c5a2fa0bbe9e1c90301c6ed`, with matching model/data source hashes.
Added unique compatible protocol discovery and explicit resolved-prefix assignment
to NB14–NB17. Only worker0 runs PILOT with a multi-account setup. No training
recipe/checkpoint change, no HF mutation; NB13 need not rerun. Original executed
NB13/NB14 archived. Discovery and existing S5 regression checks pass.
# NB15 flip-only correction and run-all workflow

User requested the actual corrected notebook after inspection found hidden
Albumentations. Added explicit empty augmentation list and a loader-level gate.
NB15 AUTO validates corrected-policy pilots and then starts/resumes scientific
training; workers1–3 wait for worker0's checks. Original4/4 pilot evidence retained
in HF; new pilot paths carry flip-only-r2. No scientific YOLO statuses were
present at revision028abb34a8395a91153b74c279bc28fe88864498. Preserved NB13 protocol
with an exact-hash runtime compatibility amendment, logged before Kaggle training;
no local HF writes. Local policy/AUTO/compatibility and existing regression checks
passed. Corrected T4 pilot execution remains pending.
# NB16 runtime mismatch repair —2026-09-11

Read the actual child traceback: `check_checkpoint` rejected a runtime mismatch,
not GPU OOM. HF run identities use NumPy2.4.6; pilot runtime is2.0.2. Four RT runs
complete and f0-s3 remains saved at52/60 at revision
d1ffc44d26a4ceafcd81a4e67813a2083879596d. Added exact checkpoint-runtime probing,
isolated per-child NumPy restoration, loaded-version verification and published
worker environment record. Kept strict checkpoint/model/CUDA guards; no model
or protocol change. Fresh runs use the pilot runtime. NB16 defaults TRAIN for
this recovery; executed originals archived. Local fault-injection and existing
regressions pass; repaired Kaggle execution not yet claimed.
