# Notebooks

<!-- current-status:start -->
> **Current status (15 September 2026):** [Completed work and remaining validation](../docs/CURRENT_STATUS.md). The report is refreshed; the app, learned-geometry integration and target-assisted alignment software exist. Dated plans below retain their original context.
<!-- current-status:end -->

## Latest: NB31/NB32 complete — no rerun

HF verifies all three SegFormer seeds ×60 epochs and the paired report.
The T4 repair passed with zero resume difference. Next is geometry integration
validation (proposed NB33, not yet built), not another training run or more labels.
[Results and next plan](../docs/36_MATCHED_GEOMETRY_RESULTS_AND_INTEGRATION.md).
All older “rerun NB31” and NB30–NB32 run-now instructions below are historical.

## Current: rerun repaired NB31 only

NB30 is HF-verified; NB31 previously failed the resume-equivalence check before
training. Open updated NB31 in a fresh T4 session, attach the same dataset,
enable Internet/HF_TOKEN and Run All in one session. No NB30 or HRNet rerun.
The strict check remains enabled with a deterministic resize repair. Local
checks passed; GPU repair verification pending. NB32 follows completed training.
[Details](../docs/35_MATCHED_SEGFORMER_COMPARISON.md). Earlier run sequences below
describe the original workflow.

## Run now: NB30 → NB31 → NB32

Matched SegFormer comparison against completed HRNet: NB30 CPU preflight,
NB31 T4 automatic smoke + resumable three-seed training, NB32 CPU report.
Use ONE NB31 session, not four workers. Attach the same prepared dataset with
manual masks for NB30/NB31; NB32 needs no dataset. Internet and HF_TOKEN enabled.
No new annotations or NB26–NB29 rerun. [Full instructions](../docs/35_MATCHED_SEGFORMER_COMPARISON.md).
Implemented/local-tested; Kaggle results pending.

## NB28/NB29 complete — no rerun

All three HRNet seeds completed 60 epochs; NB29 report verified on HF at
`a92c0f9c5c1b78c6a06e13d51e18722195230658`. Seed-average mean boundary error:
1.312% width (15.10 px), versus 3.555% for the training-mean constant baseline.
Next is a matched segmentation comparison, not more annotations or another
NB26–NB29 run. [Audit](../docs/34_HRNET_COMPLETION_AND_RESULTS.md).
Earlier repair/run-now instructions below are historical.

## Run now: repaired NB28 only

NB26 and NB27 are HF-verified; no rerun. Updated [NB28](NB28_HRNet_Training.ipynb)
resumes the preserved seed-1 checkpoint after batch 18 of epoch 1. T4, one copy,
same prepared dataset, Internet/HF_TOKEN, Run All. Numerical recovery retries the
same batch rather than aborting before AMP scale adjustment. Protocol unchanged.
Use updated NB29 after completion; it includes repair provenance. GPU replay pending.
[Details](../docs/33_HRNET_SMOKE_VERIFIED_AMP_REPAIR.md).

## Run now: HRNet notebooks NB26 → NB29

1. [NB26 Preflight](NB26_HRNet_Preflight.ipynb): CPU, prepared dataset, Internet/HF_TOKEN, Run All.
2. [NB27 Smoke/Resume](NB27_HRNet_Smoke_Resume.ipynb): T4, same dataset, Run All. Must pass before training.
3. [NB28 Training](NB28_HRNet_Training.ipynb): T4, same dataset, Run All. Three seeds sequentially; one copy only.
4. [NB29 Report](NB29_HRNet_Report.ipynb): CPU, no dataset, Run All after all seeds complete.

Internet and HF_TOKEN enabled throughout. Dual T4 is acceptable; only cuda:0 is used.
No new annotation, NB24/NB25 rerun or manual split setting needed. User-confirmed
12 distinct tyres; 72/24/24 image split. Local tests pass; Kaggle GPU runs pending.
[Persistence, limits and scientific scope](../docs/32_HRNET_NOTEBOOK_RUN_GUIDE.md).

## NB24/NB25 completed — no rerun needed

Verified 15 September at HF `2b4773914e6eefe421d8ddd74e79c5b84606c9aa`:
120/120 records, local/public annotation bytes and all 120 image hashes match.
Next is quality review and identity/split approval before HRNet training, not
another annotation batch. [Audit](../docs/31_S9_120_ANNOTATION_COMPLETION.md).
Older run-now instructions below are reproduction history.

## Run now: NB24, then NB25 — new geometry annotations

- [NB24](NB24_S9_Geometry_Annotation_Batches.ipynb): CPU, one copy, Internet/HF_TOKEN,
  prepared dataset attached, Run All. No BATCH setting. Download the one 89.51 MiB ZIP,
  extract everything, open ANNOTATE.html beside its images folder. Annotate all 120;
  save/load one JSON across sittings. No intermediate review pause.
- [NB25](NB25_S9_Geometry_Annotation_Intake.ipynb): CPU, one copy, Internet/HF_TOKEN,
  attach only the saved annotation JSON, Run All. Partial work is preserved.

One package / 120 new images, no duplicates from the old pilot. No GPU
training yet; reviewed labels and identity/split lock come first.
[HRNet and full S9 plan](../docs/30_S9_HRNET_AND_COMPLETION_PLAN.md).
NB23 is now executed and HF-verified; its instructions below are reproduction only.

## Run now: NB23 S9 geometry baseline

[NB23_S9_Geometry_Baseline.ipynb](NB23_S9_Geometry_Baseline.ipynb): **CPU, one copy,
Internet ON, HF_TOKEN enabled, attach nothing, Run All.** Reads saved SegFormer-B0
predicted masks; no dataset or weights, no training. Last verified labels are
pinned; P06 is excluded from scores pending review. No NB22 rerun needed first.
Uploads small summaries, point records and mask diagrams to HF. Public source
and logic checks passed; Kaggle execution pending. [Guide](../docs/29_S9_GEOMETRY_BASELINE.md).

## NB22 completed: review before more work

NB22's revised annotation bytes are verified against public HF
`05bf37c0f2067b0119ef057a2442d7759cf9ac51`. The upload succeeded with 11/12
complete: P06's Visible issue description is blank. Its left-middle/lower are
now Outside frame; right-middle/lower still need that visibility correction.
Save corrected JSON at 12/12; rerun NB22 only to publish revised labels, not
unchanged input. No NB21 rerun or new GPU training now.
Next proposed notebook: existing predicted-mask geometry baseline before HRNet.
[Audit and plan](../docs/28_S9_PILOT_REVIEW_AND_NEXT_PLAN.md).

## NB21/NB22 original run instructions — reproduction only

1. **NB21_S9_Annotation_Pilot.ipynb** — CPU, one copy, Internet/HF_TOKEN,
   existing prepared dataset attached, Run All. Download9.46MiB ZIP, extract,
   open ANNOTATE.html. Mark only12 images with built-in guidance; save JSON.
2. **NB22_S9_Pilot_Review.ipynb** — after NB21, CPU with only annotation JSON
   attached. Mechanical validation and small HF upload; no original-image upload.
3. Send JSON/HF revision for human review. Stop before a larger batch or training.

Local/browser checks passed; NB22 execution and public intake now verified. Neither notebook trains
HRNet/PatchCore or certifies a healthy reference. [Detailed guide](../docs/27_S9_SMALL_ANNOTATION_PILOT.md).

## NB19/NB20 — completed and HF-verified

**No NB13–NB20 reruns needed.** NB19 published the38-file evidence bundle;
NB20 published the14-figure HTML/Markdown reporting package. Both saved outputs
finish with one successful commit and zero failures. HF revision
`22d5a6bc9f953ba3bf2a75919edc7db3193b317b`. See `docs/26` for artifact locations
and the next plan: full manuscript, scientific/editorial review and reproducibility
handoff. This is reporting-package completion, not the whole original proposal.

## NB18 — complete and HF-verified

**Do not rerun NB18.** Its81 analyses,405 metrics,135 summaries and56,430
per-image/arm predictions match independent recalculation. HF revision
`35a178b5a94878bdee95a0aa9ed8cb25cd1aeb6b`; one commit, zero failures.
The exploratory analysis is complete; the full S9 pipeline remains open.
The following run/repair instructions are retained for reproduction only.

`NB18_S9_Fusion_Analysis.ipynb`: Kaggle **CPU**, one copy, Internet ON,
HF_TOKEN enabled, **Run All**. No dataset attachment, GPU training or extra
annotation. Reuses verified S5 predictions and saves fixed-fusion results to HF.
Completed source-run analyses resume/skip on rerun. No NB13–NB17 reruns.
This is not the full HRNet/PatchCore integrated pipeline; see `docs/25` for
missing inputs and the reporting gap ledger. Kaggle execution is now verified.

**NB18 repair:** the saved error was a CORAL ordinal/softmax decoding mismatch,
not bad S5 training. Download the repaired notebook and Run All on CPU. It reads
the frozen classifier configs and preserves each decision rule; all81 public
inputs passed tests. Four old partial analyses stay on HF; new code-version
results are recomputed cheaply. No NB14–NB17 rerun or annotation needed.

## Current action — 2026-09-12

**S5 COMPLETE. Do not rerun NB13–NB17.** NB14 semantic36/36, NB15 YOLO36/36,
NB16 RT-DETR9/9 and NB17 report are HF-verified at
`a3b29a71f8e6af6c50e68eb64a5bbae9ccf6d1c5`. NB14/NB17 saved outputs finish
successfully. All81 runs reached60epochs and completed evaluation. Report
inventory, run tables and summaries agree with the underlying evidence.
S9 inputs/scope and result interpretation are next; no new notebook is supplied
by this verification. Keep uploaded execution outputs unchanged.

### Earlier repair instructions (historical; no rerun now required)

Upload repaired **NB14_S5_Semantic.ipynb** into a fresh Kaggle T4×2 session.
Internet ON, HF_TOKEN enabled, Tire Dataset Prepared attached, blank PREFIX.
**TRAIN is already selected; Run All.** Stop older copies first. It skips28
completed semantic jobs. Saves now journal matching metadata before checkpoint
replacement, so emergency snapshots can recover an interrupted local save.
The RAM guard excludes inactive clean file cache, not live training memory.
No smaller model or recipe change. Preserve old local scratch if still available;
a fresh session cannot recover unpublished work. Local tests pass; Kaggle pending.

HF confirms **73/81 S5 runs complete**: semantic28, YOLO36, RT-DETR9.
Do not rerun NB13/NB15/NB16 or pilots. Finish NB14's remaining8 jobs, then NB17
on CPU. Prior executed notebooks are preserved in `execution_archives/`.

The following dated repair notes describe earlier states, not current run orders.

## NB16 resume repair —2026-09-11

Use the repaired `NB16_S5_RTDETRv2.ipynb` in a fresh T4×2 Kaggle session and
Run All. TRAIN is already selected; no NB13 or pilot rerun. The interrupted
run is saved at52/60. Its exact NumPy version is restored in an isolated worker
directory, leaving the CUDA stack and notebook environment intact. Completed
runs skip, strict checkpoint validation stays enabled. See `../docs/24`.

## Next: S5 — NB13 through NB17

**NB15 now defaults to AUTO:** use the repaired notebook and Run All. It disables
unintended Albumentations, validates the corrected pilots, then trains without
a mode change. No NB13 rerun. With four copies start acct1 first; the other
unique accounts wait for the corrected pilots and then start their shards.
Old pilots are preserved separately, not counted as corrected-policy checks.

**NB14 repair:** NB13's protocol is now public and verified; do not rerun NB13.
Use the repaired NB14 with blank PREFIX (automatic matching-protocol discovery).
Run PILOT in one copy first; with four configured accounts only worker0 runs
the pilot. Original executed NB13/NB14 files are in `execution_archives/`.
The observed crash was the blank-prefix assertion before any training, not GPU OOM.

**Built, locally checked; Kaggle GPU execution not yet verified. No annotation work.**
Run `NB13_S5_Prepare.ipynb` on CPU first and copy its PREFIX to the other notebooks.
Then `NB14_S5_Semantic.ipynb`, `NB15_S5_YOLO.ipynb`, and
`NB16_S5_RTDETRv2.ipynb`, in that order. Each defaults to **PILOT in one copy**;
only after it passes change to TRAIN (one or four workers). T4×2 selected, GPU0
used, second GPU intentionally idle. Finish with `NB17_S5_Report.ipynb` on CPU.
All require HF_TOKEN, Internet and Tire Dataset Prepared with manual masks.
81 planned runs; no per-claim commits, normal30min pushes, epoch-boundary resume.
Complete setup, scope, model identities and limitations: `../docs/24_S5_MANUAL_DENSE_TASKS.md`.

## S4b complete: NB12 and NB12R

**Both executed and HF-verified** at `dd43b231cfbdd92dd6d8c01b47166ddec4ab05f8`:
18/18 runs ×60 epochs, complete paired report. No rerun needed. Next unimplemented
training stage is S5 using the new NB13–NB17 package and existing manual masks. The instructions below are
retained for reproduction/history only; `../docs/23_S4B_CONFIRMATION.md` has results.

**2026-09-10 runtime repair:** replace NB12 after stopping all old copies.
Preserve ACCOUNT/active-account settings. Four runs are resumable at 28/26/25/26
in the audited HF snapshot. ConvNeXt now uses one GPU at the same batch 32;
MobileNet now also uses one GPU, at unchanged batch 64 (repair r2). The second
T4 is intentionally idle. Timed preflight must pass before training. No reset
or new annotation. See `../docs/23_S4B_CONFIRMATION.md`.

`NB12_S4B_Confirmation.ipynb`: 18 new 60-epoch jobs on two additional architectures,
fold 1 only. T4 x2, Internet, HF_TOKEN, Tire Dataset Prepared. One worker by default;
four copies supported with matching active-account lists and distinct ACCOUNT values.
No annotation required. After all 18 finish, run `NB12R_S4B_Report.ipynb` once on
CPU with HF_TOKEN/Internet. It verifies per-run HF evidence and reports paired
effects, not just notebook completion. See `../docs/23_S4B_CONFIRMATION.md`.

## NB11 S3 mask comparison — deferred, do not run now

The user has declined additional annotation. NB11 is **not a required next step**.
S5 will use existing manual masks; blind repeat annotation and the separate
SAM2 comparison remain unmeasured. The following is optional reproduction
guidance only, not a request for more annotation.

`NB11_S3_Manual_SAM2_Agreement.ipynb` is generated and locally checked, not yet
Kaggle-verified. First run PREPARE on CPU to download the private image package;
make independent tyre/tread rectangle prompts and 30 blind polygon reannotations.
Attach their JSON folders, select RUN and one T4 session, and Run All. It uses
SAM2.1 Small with image-by-image HF resume, not neural training or four workers.
This is human-box-prompted unedited SAM2, not fully automatic labelling.
See `../docs/22_S3_MASK_COMPARISON.md` for complete instructions.

## Current recovery delivery — 2026-09-09

**All four recovery notebooks are HF-verified** at `bf62f9e9cbedacc580aa42542da14a068b8f9215`.
NB01A: 15 baseline rows; NB01B: 9/9 runs × 60 epochs; NB03A: 153 valid / 9
quarantined; NB10R: 10/10 readable figures. No routine rerun is needed.
See `../docs/21_RECOVERY_COMPLETION_AUDIT.md` for evidence and remaining stages.

**NB01A read fix r2 (reproduction guidance):** CPU, Internet ON, HF_TOKEN enabled,
Tire Dataset Prepared attached.
HF metadata/download reads now authenticate and retry temporary rate limits;
an `[HF read] ... waiting ...` message is an intentional server-requested pause.
No full-repository inventory is requested by NB01A. Upload timing is unchanged.
Local fault-injection checks pass; the subsequent Kaggle outputs are now verified.

All five legacy S1 baseline rows already exist on HF; do not rerun old NB01
assuming three are missing. The four new files preserve the executed originals:

| Notebook | Purpose | Run mode |
|---|---|---|
| `NB01A_Baseline_Recovery.ipynb` | Reproduce CPU probes; recover per-fold and final-epoch reporting | One session, CPU sufficient |
| `NB01B_Matched_RandomInit.ipynb` | Missing matched ResNet-50: 3 folds × 3 seeds × 60 epochs, new IDs | T4 ×2; one or four explicitly labelled copies |
| `NB03A_Architecture_Audit.ipynb` | 153 valid + 9 quarantined coverage; no model substitution/training | One session, CPU sufficient |
| `NB10R_Analysis_Recovery.ipynb` | Generate Figure 3; fix undefined H2 and quarantine/final-epoch reporting | One T4 session |

Attach Tire Dataset Prepared for NB01A/B and NB10R; enable Internet and HF_TOKEN
for all. Run the recovery notebooks from the top. NB01B can span sessions and
resumes at completed-epoch boundaries, not mid-batch. A hard OS kill cannot
guarantee an emergency upload. Old notebooks and HF artifacts are unchanged.

Generated with `python tyrelib/build_closure_notebooks.py`. Recovery outputs
use `tables/closure_2026-09-09/` and `analysis/closure_2026-09-09/`.
**S5 and S9 are still unimplemented, not silently omitted or completed.**
See `../docs/20_FULL_PLAN_CLOSURE.md` for the entire plan and remaining gaps.

NB00–NB05 are generated from `tyrelib/tyrelib.py` by
`tyrelib/build_notebooks.py`. The corrected NB06–NB10 definitions live in
`tyrelib/build_later_notebooks.py`; the full generator calls that builder too.
NBT1 is generated by `tyrelib/build_test_notebook.py` and embeds the same
library only for its self-healing annotation replay.

> **The `.py` file is the source of truth. Notebooks are generated.**
> Edit Python → run the generator → re-upload. Never hand-edit the base64 blob
> in cell 1; the next rebuild overwrites it.

```bash
conda activate cv_conda
cd tyrelib
python build_notebooks.py ../notebooks
python build_test_notebook.py ../notebooks
# To rebuild only the post-Stage-A notebooks without touching NB00–NB05:
python build_later_notebooks.py ../notebooks
```

The generator verifies the base64 round-trips byte-identically.

Executed NB00–NB05 retain the embedded library snapshot that produced their
saved outputs; that is scientific provenance, not drift to “fix” after the
fact. NB07 is now executed; the repaired, in-progress NB06 and not-yet-run
NB08–NB10 embed the current library byte-for-byte. NBT1's
13 executed cells and 75 outputs are retained while its future-run loader
warning fix and embedded mask-rebuild library are updated in place.

---

## Run order

| # | Notebook | Stage | Runs | GPU-h | Needs |
|---|---|---|---:|---:|---|
| 0 | `NB00_Preflight.ipynb` | S0 | — | 0.2 | — |
| 1 | `NB01_Baselines.ipynb` | S1 | 3 | 0.5 | — |
| 2 | `NB02_StageA_CNN_classic.ipynb` | S2 | 36 | 24 | — |
| 3 | `NB03_StageA_CNN_modern.ipynb` | S2 | 36 valid | ~18 | — |
| 4 | `NB04_StageA_Transformer.ipynb` | S2 | 54 | 26 | — |
| 5 | `NB05_StageA_Foundation.ipynb` | S2 | 27 | 26 | — |
| 6 | **`NB07_XAI.ipynb`** | S6 | inference | ~15 | Stage A checkpoints + **annotations** |
| 7 | **`NB06_StageB_OFAT.ipynb`** | S4 | ~108 | ~50 | `tables/stage_b_selection.csv` from NB07 |
| 8 | `NB08_StressTests.ipynb` | S7 | inference | ~20 | checkpoints + **annotations** |
| 9 | `NB09_Ensembles_Calibration.ipynb` | S8 | reuse | ~2 | per-sample predictions |
| 10 | `NB10_Analysis_Figures.ipynb` | S10 | — | — | everything |
| T1 | `NBT1_Annotation_Test.ipynb` | — | 1 | ~0.4 | **annotations** |

The filenames preserve the original stage numbering. NB07 has completed the
first gate in the corrected order **NB07 → NB06 → NB08 → NB09 → NB10** and
selected `regnety016`, `densenet121`, and `resnet50`. NB00 and NB01 run on
every account. NB02–NB05 were sharded; NB06 retains safe HF reconciliation and
can be sharded if more than one account is used.

The public repository contains nine additional historical run ids labelled
`convnextv2_s`. They are quarantined, not counted in the 36 valid NB03 runs:
all nine statuses report the parameter count of ResNet-18, and the audited
checkpoint tensor signature confirms the old emergency fallback substituted
ResNet-18. The rebuilt NB03 does not schedule that unsupported pretrained arm.

`NBT1` uses `tyrelib.ensure_annotations()` only to produce a known candidate
mask set. Its coverage gate, independent alignment controls, differential U-Net
test, and verdict are written separately in the notebook, so the judging code
does not simply repeat the propagation implementation. Run it after any change
to the masks or transform traces, before anything else consumes them.

### NBT1 `2026-08-30-r1` — verified result

The real Kaggle run passed all three gates. It self-healed all 4,180 propagated
masks and used fingerprint `085acfb8fb83c531`; clean IoU was 0.9780,
propagated IoU 0.9747, and their ratio 0.9966. The ignored data-loader cleanup
assertions printed around epoch 18 were a Jupyter multiprocessing cleanup race,
not a failed training step; the RAM-backed loaders now use `num_workers=0`.

Run it again only after changing the annotations or transform traces:

1. Upload `NBT1_Annotation_Test.ipynb` to Kaggle.
2. Add the `Tire Dataset Prepared` input. The notebook measures the masks and
   can rebuild propagated masks from clean masks + transform traces, so the
   version label is recorded but never trusted.
3. Add the `HF_TOKEN` secret, enable Internet, and select **GPU T4 ×2**.
4. Run All. At startup confirm the revision is `2026-08-30-r1` and save the
   printed 16-character mask fingerprint in `PROGRESS.md` after the real run.
5. Accept the annotation package only when Parts A, B, and C all say PASS and
   the final Hugging Face list has no `MISSING` entries.

This revision writes under `annotation_test/2026-08-30-r1/` in
`Shanmuk4622/tyre-wear-study`. The old annotation-test artifacts remain intact
and cannot be mistaken for a resume checkpoint from this repaired run.

NBT1 checks all 4,598 masks, not a sample. Part C runs in FP32 deliberately:
earlier FP16 execution could generate a non-finite loss while GradScaler skipped
every update. It uses both T4s, checkpoints atomically every five batches, and
can resume inside an epoch with deterministic remaining batches.

---

### NB07 `2026-08-30-r3` — completed result

NB07 completed and verified all three public gate files. The locked Stage-B
architectures are `regnety016` (TER_norm 1.5785, BAR 0.0310), `densenet121`
(1.5513, 0.0455), and `resnet50` (1.5146, 0.0512). All are `xai_status=ok`,
eligible, and confirmed on seeds 1–3. Raw valid-map coverage is 180/180,
178/180, and 180/180 respectively. The public evidence table has 1,208 rows;
the faithfulness table has 35 rows.

---

## What to run now

**Superseded execution guidance:** use the recovery delivery above. The following
paragraphs record the preceding NB08–NB10 completion audit, not full-plan closure.

As of 2026-09-09, **NB08, NB09 and NB10 have executed**. NB08 has 63/63
stress rows; NB09 has all four tables and 27 prediction files. NB10 published
nine figures and its result tables, but skipped Figure 3 because the saliency
examples are absent. No full rerun is required to verify these saved artifacts.

Next is results review: H2 is undefined despite a saved False flag; conformal
coverage varies outside the target band; Figure 10 needs its quarantine filter
reviewed. See `../docs/19_NB08_NB10_COMPLETION_AUDIT.md`. The live control
mean is now 0.375184, below 0.45, following a later fold-1 HF record.
The executed notebooks are preserved. If inference is rerun later, use one
session: NB08–NB10 inference loops are not sharded across account labels.

### Historical NB06 recovery instructions (superseded by completion above)

1. Stop every older NB06 session. Upload the
   **tyrelib v12 `NB06_StageB_OFAT.ipynb`** to Kaggle, attach the prepared
   dataset, choose **T4 ×2**, enable Internet, expose `HF_TOKEN`, and Run All.
   The requested one-notebook mode is now the default:
   `ACTIVE_KAGGLE_ACCOUNTS=('acct1',)` and `ACCOUNT='acct1'`.
   **Cell 1 must print `tyrelib v12 loaded`; the session must print
   `worker=0/1` and `MODE=ONE NOTEBOOK`.**
2. Cell 2 must load public revision `2026-08-30-r3` and print exactly these
   locked architectures: **RegNetY-16GF, DenseNet-121, ResNet-50**. It also
   reconstructs their raw evidence coverage from
   `tables/xai_evidence_all.csv` before it allows training.
3. Public HF currently has **101/108 complete, 3 checkpointed incomplete, and
   4 not started**. Every one of the 104 status-bearing Stage-B runs has both
   checkpoints, so nothing
   recorded needs to be retrained. Completed runs will be skipped. NB06 runs the remaining ROI jobs
   first, then the other 11 OFAT factors, three seeds each, on fold 1 only. It
   saves a rolling checkpoint every epoch, batches HF pushes every 30 minutes,
   and flushes on run completion, a clean pause, important cells, or
   Stop/exception; rerunning
   skips complete jobs and resumes interrupted checkpoints from HF.
4. v6 repairs the `ParserError: Expected 177 fields ... saw 178` found while
   resuming two 60-epoch DenseNet histories. It recognises the v5 revision
   field, inserts a blank for v4 rows, verifies every width, and atomically
   rewrites by column name. The exact public 60-row failure file passes the
   migration with epochs 1–60 and all metrics preserved.
5. **v7 fixes the self-stopping workers (Bugs 22–23).** All 53 pauses were
   host-RAM guards — and most were false. The guard compared the epoch's
   transient **peak** against 88%, and any pause ended the whole cell, so a
   two-second checkpoint spike on the first epoch of a session stopped a worker
   with eighteen runs left. Measured on `b-densenet121-res512-f1-s3`: epoch 1
   peaked at 95.2% and epoch 10 sat at 17.8%; the mean after epoch 5 is 42.9%.
   Underneath it there was a real leak — the telemetry buffers were never
   cleared and grew **0.54 GB/epoch**, 3.5 GB to 28 GB across a run.
   v7 measures live RAM *after* releasing freed arenas, uses 88/80 hysteresis,
   continues to the next run when a pause clears, and writes telemetry
   incrementally. No model, batch size, resolution, optimiser or epoch budget
   changed. See `docs/05 §7`.
6. **v8 fixes the idle workers (Bug 24).** `steal_stale=False` made other
   accounts' runs permanently unreachable, so a worker that finished its 27-run
   shard planned zero runs and exited while two accounts still had twenty each
   — the "2 running, 2 stopped" report. Own work still runs first, but an
   exhausted worker now takes from the shared pool through a two-phase claim
   (write, flush, settle, re-read, lowest account wins a tie). Four accounts
   racing for six runs produced exactly one winner each. No takeover starts in
   the last 90 minutes of a session.
7. **v9 fixes the RAM pauses at the root (Bugs 25–26).** The guard was reading
   `/proc/meminfo`, which inside a container reports the **host's** memory, not
   the cgroup limit the kernel enforces — so every pause decision used a number
   that did not describe our budget. It now reads `/sys/fs/cgroup/memory.*` and
   prints its source and which process holds the memory. And the run that
   paused had logged **`dl 0%` on all 49 of its epochs** while still starting
   two pinned loader workers, whose RSS counts against the same cgroup. At
   ≥320 px the loader is now synchronous: `[LOADER] workers=0 pin_memory=False`,
   with epoch times unchanged because nothing was ever waiting for data.
8. **v10 fixes the false one-worker launch and invisible progress (Bug 27).**
   The submitted output said `worker=0/4`, so it was not the claimed
   one-worker run; it also completed epoch 37 in 3.7 minutes before the saved
   widget displayed epoch 38 at 0%. The public HF status later reached 60/60.
   Parallelism now comes from the single `ACTIVE_KAGGLE_ACCOUNTS` tuple, and a
   plain `[LIVE] ... batch 1/N completed` line proves every epoch is active.
   No model or experimental setting changed.
9. **v11 fixes the two-model session ceiling (Bug 28).** v10 actually trained
   three runs: two completed, and the third advanced from epoch 3 to 45 before
   the 88.1% cgroup guard stopped the cell. Public telemetry shows retained RSS
   growing about 0.17 GB/epoch on RegNet and 0.30 GB/epoch on the inspected
   DenseNet tail. Each model now runs in a disposable child process. Linux
   reclaims the entire child address space at exit; a RAM-paused child is
   automatically restarted on the same HF checkpoint instead of ending NB06.
10. **v12 fixes the account typo and the misleading risk label (Bug 29).**
   Python treats `('acct1')` as a string, which made the attached notebook stop
   in Cell 3 before loading data. It is now normalised to `('acct1',)`
   automatically. Untouched runs with no HF files now print `NOT STARTED`;
   `AT RISK` is reserved for partial artifacts without a usable checkpoint.
11. Two independent RegNet ROI attempts then exposed a separate T4/cuDNN
   `channels_last` fault at only ~1.1 GB/card. RegNet remains unchanged but now
   runs contiguous NCHW with cuDNN autotuning off; other architectures keep
   `channels_last`. FRESH runs stay with their static owner, so account 1 will
   no longer duplicate account 2's RegNet job; an exhausted worker takes leftover
   work only through the v8 two-phase claim (step 6). Before claiming
   work, NB06 proves the exact RegNet batch/resolution in a disposable dual-T4
   child process and publishes the log; look for `CUDA_SMOKE_PASS`.
12. Then run **NB08 → NB09 → NB10**. Each notebook reconciles public HF
   artifacts before doing work, pushes at major milestones and on the
   30-minute cadence, and can continue in a fresh Kaggle session. NB09 also
   rebuilds and publishes a missing predictions parquet from its public best
   checkpoint, so a historical derived-upload gap cannot block ensembles.

---

## Before the first run

1. **Attach the dataset.** Add Input → your `Tire Dataset Prepared` dataset.
   One dataset holds everything:
   ```
   /kaggle/input/<slug>/FINAL/{images,splits,manifests}
   /kaggle/input/<slug>/annotations/{clean,propagated}
   ```
2. **Add the secret.** Add-ons → Secrets → `HF_TOKEN` (write scope, `Shanmuk4622`)
3. **Accelerator:** GPU T4 ×2 · **Internet:** ON
4. **Run `NB00_Preflight.ipynb` and make sure it passes.**

---

## The two lines that define active notebooks

```python
ACTIVE_KAGGLE_ACCOUNTS = ('acct1',)  # one notebook (the current default)
ACCOUNT = 'acct1'                     # this notebook's label
```

For four parallel copies, use
`('acct1', 'acct2', 'acct3', 'acct4')` in every copy and change `ACCOUNT` in
each copy. `NUM_WORKERS` and `WORKER_ID` are derived; do not edit them.

All accounts push to the one HuggingFace account, so the 128-writes-per-hour
budget is shared; `tyrelib` caps each worker at `100/NUM_WORKERS`.

### Changing `NUM_WORKERS` mid-study is safe

It decides only **what this account starts first**. Whether a run is finished,
and what epoch it reached, is read from that run's own files on HuggingFace, so
it is the same answer for every account at every worker count.

Go from 4 workers to 1 and: finished runs are skipped, half-done runs are
**resumed from their checkpoints**, and nothing is trained twice.

> This did not always hold. Resume checked only the local disk, and Kaggle
> wipes that between sessions, so every run restarted at epoch 1 — and runs
> that ended in an exception were re-queued from scratch even though their
> checkpoint was intact. See `docs/05 §7`, Bugs 8–10.

Before a long run, look at the reconcile table:

```python
sess.reconcile(run_ids)
```

```
                   run_id      state  epoch status_file  registry  action
    a-resnet50-base-f0-s1  resumable     47      failed    failed  resume
    a-resnet50-base-f0-s2  completed     60   completed completed    skip
    a-resnet50-base-f0-s3     absent      0          NA         -   train
```

**`state` is from the repository; `registry` is from the run log. When they
disagree the repository is right.** A run that raised at epoch 47 still has a
checkpoint at epoch 47, so its action is `resume`, not `train`.

---

## Reading the training output

```
  ep  13/60  loss 0.0002  val_acc 1.000  val_F1 1.000  val_QWK 1.0000  | 54s  dl 20%
             <-- PERFECT on 4 tyres. NOT a success signal; see split_health.json
```

**That warning is the point.** Each fold validates on ~4 tyres — roughly one per
class. A model only has to tell three specific tyres apart, so a perfect score
is the *expected* outcome, not evidence it learned wear.

Four trivial baselines already score near-perfectly, each on a **different**
fold:

| Baseline | f0 | f1 | f2 | mean |
|---|---:|---:|---:|---:|
| Frame occupancy | 0.181 | 0.455 | **0.968** | **0.535** |
| Colour only | **0.952** | 0.399 | 0.123 | 0.491 |
| Structure only | 0.354 | 0.119 | **0.976** | 0.483 |
| Annotation side-channel | **0.978** | 0.159 | 0.108 | 0.415 |

**Beat 0.535 across all three folds, or nothing was learned.**

| Field | Meaning |
|---|---|
| `val_QWK` | quadratic weighted kappa — the ordinal metric; selects `ckpt_best` |
| `dl` | share of the epoch spent waiting for data. High ⇒ fix the loader, not the model |
| `* best` | new best QWK; checkpoint and per-sample predictions written |

**No early stopping.** Every run trains all 60 epochs. Equal budget keeps the
comparison fair and makes the time estimates honest.

---

## Do the workers allocate runs to each other automatically?

**Yes — each fresh run has one deterministic owner.**

Each account starts with its own shard (LPT-balanced on the static cost table).
An absent run stays reserved for that owner. Automatic work stealing is
disabled in NB06 v6. A deliberate recovery launch may opt in and can take over
another account's run only when HF contains a real claim/run event older than
45 minutes. Before
starting any run the notebook re-reads its state from HuggingFace, so a run
another account finished five minutes ago is skipped rather than repeated.

What is **not** automatic:

| | |
|---|---|
| Starting the notebook on each account | manual — open Kaggle, set `ACCOUNT` to `acct1`…`acct4`; `WORKER_ID` is derived; Run All |
| Moving on to NB03 when NB02 finishes | manual — each notebook is its own set of runs |
| Restarting after Kaggle's 9-hour session cap | manual — re-run the notebook; it resumes |

So: allocation is automatic, *launching* is not. Four accounts mean four
browser tabs, each set to a different `ACCOUNT` label.

**Two coordination caveats, now fixed.** On the NB02 run, `a-vgg16bn-base-f1-s1` was trained
by two accounts at once — ~1.4 GPU-hours spent twice. A worker checked a
registry it had last downloaded hours earlier, and claims were only pushed on
the 30-minute cycle. Both workers were correct on what they could see. Stolen
runs now re-pull the registry and flush the claim immediately. NB06 revealed a
second race: all accounts interpreted a fresh absent run as a dead worker and
could converge before the first claim became visible. Absent work is now
reserved for its static owner; stealing is opt-in and only genuinely stale
events qualify. Ordinary claims ride the 30-minute HF batch; only a genuinely
stolen claim needs an immediate coordination commit.
`aggregate_remote()` also warns if any `run_id` has two owners.

---

## A worker that finishes its shard now helps the others

Before v8, `steal_stale=False` made other accounts' runs permanently
unreachable, so a worker that got through its 27-run shard printed
`will run 0 run(s) this session` and ended — while two other accounts still had
twenty runs each. That is the "2 running, 2 stopped" report.

The plan now shows the shared pool:

```
  starting from scratch  : 19
  available if I go idle : 68   (claimed one at a time, only after my own 19)
```

Nothing is taken from anyone. Own work always runs first; the pool is only
reached once `mine` is empty. Then:

```
[IDLE] my own 19 run(s) are done or running elsewhere. Taking work from the
       shared pool so this GPU is not parked while other accounts still have
       runs left.
[IDLE] b-resnet50-mixup-f1-s2: claimed after settling
```

or, if another account reached for the same run in the same 30-second window:

```
[SKIP] b-resnet50-mixup-f1-s2: yielded to acct2 (claimed the same run)
```

Both are correct. The claim is two-phase — write it, flush it so others can see
it, wait out the race, re-read, and the lowest account name wins a tie. Four
accounts racing for the same six runs produce exactly one winner each.

No takeover starts inside the last 90 minutes of a session: better to stop
cleanly than half-train a model. `docs/05 §7`, Bug 24.

---

## If a worker stops on its own

Two messages mean two different things:

```
[RAM] epoch 12 peaked at 91.4% but sits at 43.1% now -- transient, continuing
```

Normal. The spike was the checkpoint being serialised and handed to the
uploader; it is gone by the next epoch. Nothing stopped.

```
[RAM] host RAM 89.2% after epoch 12; pausing before the kernel is killed.
[RUN] host RAM back to 61.0% (under 80%) once this model was released
      -- continuing with the next run
```

Also fine. The pressure was that model; the worker carries on.

```
[RUN] host RAM still 88.4% after releasing this model. Stopping so the
      kernel is not killed.
```

Real. Start a fresh Kaggle session — the checkpoint is on HuggingFace and the
run resumes at the next epoch.

> Before v7 the guard compared the epoch's **peak** RAM against 88% and any
> pause stopped the whole worker, so a two-second spike on the first epoch of a
> session ended it with eighteen runs untouched. `docs/05 §7`, Bugs 22–23.

---

## What happens when you press Stop

Catchable Stop/error handling attempts a blocking push of durable progress.
Resume granularity and watchdog duration depend on the notebook; follow its
current run guide. A fresh session restores the latest successfully published
compatible checkpoint. Forced kernel death, exhausted RAM or lost Internet can
lose work since the last successful HF push: no notebook can guarantee an upload
after a hard kill. Do not change worker ownership or launch extra copies unless
the notebook explicitly supports that arrangement. NB31 uses one session only.

An exception is not different from a Stop. `failed` runs resume too.

NBT1 is stricter: it saves the most recently completed **batch** before the
blocking stop push and resumes at the next batch. Its checkpoint also contains
the partial-epoch loss counters, CUDA RNG states, annotation fingerprint, and
configuration hash; an incompatible or unreadable checkpoint raises instead of
silently starting over.

---

## Push cadence

| Trigger | Blocking |
|---|---|
| Background cycle, every 30 min | no |
| Every epoch — metrics + checkpoint enqueued | rides the cycle |
| Every 10 epochs — telemetry, per-sample predictions | rides the cycle |
| Ordinary static-owner claim | rides the cycle |
| Explicitly stolen claim | **yes** |
| **A model finishes** | **yes** |
| **A model pauses cleanly** | **yes, then this worker stops** |
| **End of an important cell** (`sess.push_now()`) | **yes** |
| **Stop / SIGTERM / exception** | **yes** |
| `sess.finish()` | yes, then verifies by re-listing the repo |
