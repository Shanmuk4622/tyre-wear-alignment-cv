# S5 — manual-supervised detection/segmentation

<!-- current-status:start -->
> **Current status (15 September 2026):** [Completed work and remaining validation](CURRENT_STATUS.md). The report is refreshed; the app, learned-geometry integration and target-assisted alignment software exist. Dated plans below retain their original context.
<!-- current-status:end -->

## Current position — 2026-09-12

**COMPLETE:81/81 runs and NB17 report independently verified.** Public HF
revision `a3b29a71f8e6af6c50e68eb64a5bbae9ccf6d1c5`; NB17 pinned its input
revision `8ffef81e9aebf0464b19349c1956cc3afffb08fc`. Different input/output
revisions are expected because the report was published after auditing training.

- Semantic36/36 (U-Net9, DeepLab9, SegFormerB0 9, SegFormerB2 9).
- YOLO36/36; genuine RT-DETRv2-R18 9/9. Every run60epochs and evaluated.
- All81 checkpoint hashes, evaluation artifact hashes and4,860 epoch records
  verified, with native prediction and per-region/per-mode validation coverage.
- Recomputed all405 downstream ROI macro-F1 rows against frozen manifest labels;
  verified deltas and report inventory,81-row localisation and405-row ROI tables.
  ROI summary has135 model/fold/mode rows; localisation summary27 model/fold rows.
- Six report files are public: STATUS, inventory, ROI by-run/summary and
  localisation by-run/summary. STATUS is complete, not partial.
- NB14 saved output:23 commits,0 failures; NB17:1 commit,0 failures and81/81.

Across the81 equally weighted runs, predicted-tyre ROI mean macro-F1 delta is
−0.01257 and predicted-tread is−0.01467 versus full frame. These are descriptive
averages, not significance tests or proof cropping helps. Use model/fold summaries
for interpretation; oracle ROI also does not imply a crop-trained classifier.

**No rerun of NB13–NB17 needed.** S5's manual-supervised scope is complete.
S9 integration, broader proposal gaps and writing remain. SAM2 comparison/blind
repeat annotation remain deferred. Existing-fold leakage and limited tyre count
still prohibit independent new-tyre generalisation claims. Notebook files and HF
were not changed by this audit; only project Markdown progress records updated.

### Historical partial-progress and repair records (superseded)

Latest HF revision `c9903960450f8fd9a16a5b148ff3da2100ef8654` verifies73 completed
statuses/checkpoint hashes: semantic28/36, YOLO36/36, RT-DETR9/9. Eight semantic
jobs have no published status/checkpoint; this does not prove no local training
occurred. Latest traceback:90% RAM guard triggered, SIGINT interrupted local
publication, emergency snapshot rejected mismatched checkpoint/status.

The earlier upload fix did not protect interruption BETWEEN local file replacements.
Runtime now atomically journals matching status/history before replacing weights.
Emergency snapshot verifies the journal hash and restores sidecars under the writer
lock, including first-save interruption. Old semantic scratch can use validated
checkpoint-derived sidecar recovery after the child stops. Evaluated mismatches
still fail closed. No checkpoint reset, smaller model or training recipe change.

RAM guard subtracts only inactive clean file cache (dirty/writeback remain counted)
from cgroup usage and retains90% working-memory stop. Missing cache stats retain
the original conservative behavior. Traceback alone does not establish how much
of this incident's RAM was cache. Memory-stop diagnostics are uploaded when saved.
Journal runtime provenance is published by the notebook, not this assistant.

Run repaired NB14, TRAIN + Run All, T4×2, Internet/HF_TOKEN/dataset enabled. It
skips28 completed runs. Preserve local scratch if available; work lost with an old
session and never published cannot be recovered from HF. After36/36 run NB17 CPU.
Fault-injection checks cover stopping before/after checkpoint replacement and
first-ever save; GPU execution of this repair remains pending.

### Earlier audit and repair (superseded)

HF revision `971f0c7ad7f9e90aef8f2701e4a76f3f07367397`: semantic22/36,
YOLO36/36 and RT-DETR9/9 completed (67/81 total). Completed checkpoint hashes
match. All22 completed semantic histories1–60, evaluation artifact hashes,
native prediction coverage and five-mode ROI coverage passed the detailed audit.
Semantic U-Net9/9, DeepLab9/9 and SegFormerB0 4/9 are complete;
SegFormerB2 has not started. Remaining: one interrupted job plus13 unstarted.

NB14 raised `Published checkpoint/status mismatch` on `segformer_b0-f1-s2`.
The public checkpoint SHA is
`eaf3107dbfd127a8a5284fce57a92c1cff0c833cc64a0c1a9063e126e8f51aaa`.
Direct CPU inspection verified epoch45, history1–45, correct plan/job and
model/optimizer/scheduler/scaler/RNG/runtime state. STATUS incorrectly records44
and a different checkpoint hash. This is an upload-generation mismatch, not OOM.

Repair: unique immutable staging per snapshot, all-or-nothing enqueue of its
files, and serialized foreground/background commits. Empty-buffer flush now
waits for in-flight publication before staging cleanup. Non-completed semantic
sidecars can be rebuilt only after downloaded bytes match pinned HF LFS metadata
and embedded checkpoint identity/history/resume fields pass validation. Original
status/revision are retained in `resume_recovery.json`; checkpoint bytes remain
unchanged. Completed mismatches still fail closed. Repair is published by the
rerun notebook before training. No assistant HF writes occurred.

**Next: upload repaired NB14, fresh T4×2, Internet/HF_TOKEN/dataset enabled,
Run All (TRAIN already selected).** It skips22 completed runs and resumes epoch46.
Do not rerun NB13, pilots, NB15 or NB16. Then NB17 CPU after semantic36/36.
Models, recipe, protocol,30min cadence and catchable-Stop flush are unchanged.
Fault-injection and notebook tests pass locally; repaired GPU resume is pending.

## Historical delivery and repair notes (superseded by current position)

### NB16 exact-runtime resume repair —2026-09-11

The uploaded NB16 stopped before resuming `rtdetrv2_r18-f0-s3`: its saved
runtime differed from the fresh session. At HF revision
`d1ffc44d26a4ceafcd81a4e67813a2083879596d`, the run remains resumable at52/60;
four RT-DETR jobs are complete. Published run identities record NumPy2.4.6,
while the pilot used2.0.2; torch2.10.0+cu128 and the pinned model packages match.

**Replace NB16, start a fresh T4×2 session and Run All with MODE='TRAIN'**
(already selected). No NB13/pilot rerun, no reset, no smaller model. For every
resumed worker the parent reads the actual checkpoint's runtime in a short CPU
process. If only NumPy differs, the exact saved version is installed with
`--no-deps` into generated scratch and used only by that child's PYTHONPATH.
Fresh jobs instead use the recorded pilot runtime. Both metadata and the loaded
NumPy version are verified before training; non-NumPy/CUDA mismatches still stop.
The strict checkpoint validation and epoch boundary remain unchanged.

`worker_environment.json` records the expected/verified runtime and is uploaded
with the job. Model/data source and immutable NB13 protocol hashes are unchanged;
no HF artifacts were edited by this repair. Original NB15/NB16 outputs were
archived before regeneration. Fault-injection tests cover isolated NumPy restore,
checkpoint preservation and rejection of CUDA changes; notebook and existing
policy regressions pass. Actual repaired Kaggle resume remains to be executed.

### NB15 flip-only repair and AUTO execution

HF at `028abb34a8395a91153b74c279bc28fe88864498` confirms all4 original YOLO
pilot/resume checks passed, but **zero scientific YOLO run statuses** existed.
The original pilots used unintended Ultralytics Albumentations defaults. This
was an implementation mistake, not a reason to change the declared recipe.

**Now run the repaired NB15 with MODE='AUTO' and Run All. No NB13 rerun or
manual pilot-to-training switch.** One worker works with defaults. With four
copies, set the same active account list and unique ACCOUNTs; start acct1 first.
Worker0 checks the repaired pilots; others wait without claim commits, then
all start their assigned full training. Rerunning AUTO skips valid corrected
pilots and resumes training. A failing pilot still stops safely.

Repair `flip-only-r2` passes `augmentations=[]` explicitly and inspects the
constructed loader before any update. Blur, MedianBlur, grayscale and CLAHE
are disabled. Model sizes, batch, resolution, folds/seeds and epochs remain
unchanged. Old-policy YOLO checkpoints cannot be resumed or reported as repaired
results. Original pilots remain at their original paths; new checks use
`pilots/<yolo_model>/flip-only-r2/`. Scientific runs keep the declared IDs.

The exact pre-repair runtime hash is narrowly recognized for the existing NB13
protocol; other unknown code changes still fail validation. On Kaggle, worker0
publishes a runtime amendment plus the corrected source under
`runtime_amendments/flip-only-r2.*` before training. The original immutable
protocol is not rewritten. Semantic/RT training calculations are unchanged.
NB14–NB17 share this documented compatibility handling; old notebook outputs
were archived before regeneration. No HF records were altered locally.

Local tests passed: hidden-augmentation rejection, explicit empty override
accepted by pinned Ultralytics8.4.20, AUTO pilot-to-training transition, rerun
pilot skipping, narrow source compatibility and existing S5 regression checks.
Corrected T4 execution is still validated by the automatic pilots, not claimed
as already completed. The older manual PILOT/TRAIN instructions below apply
to NB14/NB16; NB15 now uses AUTO.

**NB14 setup repair:** NB13 is now verified at public HF revision
`951435416dde3bc65c5a2fa0bbe9e1c90301c6ed`, protocol hash
`1f6694577253e0054f7a22df6ec52d30797063cf498b345bd71fe9b98bab93df`.
No need to rerun NB13. NB14's saved exception was `Paste the exact PREFIX from
NB13`, with PREFIX blank; training had not started. The dependency conflict
warnings were not the terminating exception in this execution.

Repaired NB14–NB17 discover a unique compatible protocol when PREFIX is blank,
pin the read revision, verify its hash and code/data compatibility, and assign
the resolved PREFIX before execution. Zero/multiple matches stop with an
actionable message; no arbitrary latest-protocol choice. PILOT with four accounts
is owned by worker0 only; other workers explicitly skip. Default remains one
copy. Original NB13/NB14 outputs are preserved in `notebooks/execution_archives/`.
Regression checks passed for blank/explicit/missing/ambiguous discovery and
non-owner PILOT skipping. Model/runtime source hashes still match published NB13;
no recipe, weights, checkpoint namespace or upload interval changed.

**Notebook implementation delivered; Kaggle GPU execution and S5 completion are
not yet verified.** Public HF was read at
`dd43b231cfbdd92dd6d8c01b47166ddec4ab05f8`; no `s5/` artifacts existed.
No remote training was launched or remote results altered by this delivery.
S4b remains complete. S9 remains unimplemented.

No new annotation is requested. Existing manual masks are the only label source.
SAM2/manual comparison and blind reannotation are deferred, not passed.

## Run order

| Notebook | Runtime | Purpose |
|---|---|---|
| `NB13_S5_Prepare.ipynb` | CPU, once | Check every image/mask and split; freeze protocol and model revisions on HF; print PREFIX |
| `NB14_S5_Semantic.ipynb` | T4×2 | Four semantic segmenters; 36 jobs |
| `NB15_S5_YOLO.ipynb` | T4×2 | YOLO26 n/s detection and n/s instance segmentation; 36 jobs |
| `NB16_S5_RTDETRv2.ipynb` | T4×2 | Actual RT-DETRv2-R18; nine jobs |
| `NB17_S5_Report.ipynb` | CPU, once | Public-HF audit and paired report; partial stays partial |

All require Internet, the writable `HF_TOKEN` secret, and the same attached
Tire Dataset Prepared package, including `annotations/clean/masks`.
Leave PREFIX blank in the other four notebooks to discover the unique matching
protocol, or set the exact PREFIX printed by NB13 explicitly. There is
no mutable “latest experiment” pointer that can silently switch an active worker.

For EACH training notebook first run `MODE='PILOT'` in **one copy**. It tests
each model, saves a scratch checkpoint, then reloads it in another process and
performs additional updates. Only a passing pilot unlocks `MODE='TRAIN'`.
Pilots are stored separately and never counted among the 81 runs. Semantic/RT
pilots use six batches twice; YOLO uses two complete pilot epochs to exercise
its native checkpoint callback. Pilots do not imply scientific training is done.

Then run TRAIN in one or four copies. For four, set the SAME active account list
in every Session cell and a different ACCOUNT per copy. Finish one notebook
family before starting the next; do not run 12 copies or duplicate an account.
Static ownership does not change as jobs finish; there is no claim polling or
idle-worker stealing. Stop all old copies before changing the worker count.
Rerunning TRAIN resumes remaining work and skips HF-completed jobs.

GPU0 is used, GPU1 deliberately idle. This avoids the demonstrated DataParallel
slowdown; it is not a claim that two GPUs are accelerating each job. A slow
pilot/epoch or GPU OOM stops without substituting a smaller model or reducing
the experiment. Measured pilot speed, not the old classification cost estimate,
should determine how many sessions are needed.

## Explicit recipe and model identity

**Nine configurations × three folds × three seeds ×60 epochs =81 jobs.**
No smaller scope is silently presented as the original S5 sweep.

| Configuration | Initialization | Task |
|---|---|---|
| U-Net ResNet-34 | ImageNet encoder, `smp.Unet` | Two-channel semantic mask |
| DeepLabV3+ ResNet-34 | ImageNet encoder, `smp.DeepLabV3Plus` | Two-channel semantic mask |
| SegFormer B0/B2 | `nvidia/mit-b0`, `nvidia/mit-b2` | Two-channel semantic mask |
| YOLO26-n/s | `yolo26n.pt`, `yolo26s.pt` | Tyre/tread boxes |
| YOLO26-n/s-seg | `yolo26n-seg.pt`, `yolo26s-seg.pt` | Tyre/tread instances |
| RT-DETRv2-R18 | `PekingU/rtdetr_v2_r18vd` | Tyre/tread boxes |

**Identity correction:** the old `rtdetr-l.pt` entry was not RT-DETRv2-S.
This implementation explicitly uses genuine RT-DETRv2 with an R18 backbone;
it never labels Ultralytics RT-DETR-L as v2-S. DeepLabV3+'s previously unspecified
encoder is explicitly ResNet-34. Both YOLO sizes are retained for both tasks.

The new dense-task recipe uses **clean training originals only**, not the 4,180
pre-generated classification derivatives. Train/validation counts are232/186,
290/128 and314/104 in folds0/1/2. All418 image hashes and masks were checked
locally. Notebook repeats the checks against the actual Kaggle attachment.

Input512px; AdamW lr1e-4, weight decay.01; cosine scheduling; semantic batch4,
RT-DETR batch2, YOLO batch4. Semantic uses BCE+soft Dice with two **overlapping
sigmoid channels**, not an incorrect exclusive tyre/tread softmax. RT-DETR and
YOLO use their native detection/instance losses. Semantic/YOLO have horizontal
flip probability.5; RT-DETR has no online augmentation. YOLO colour/mosaic/mixup
and geometric augmentations other than the flip are disabled. Backend-native
EMA, schedules and pretraining differ; this is not an equal-pretraining ablation.

Canonical labels: tyre=`mask>0`; tread=`mask==2 or mask==3`. Boxes are derived
automatically from these regions. Native masks remain evaluation ground truth.
YOLO polygons bridge disconnected components; a rasterized polygon must retain
at least.98 IoU with its manual region or that export stops, with the affected
image identified. Holes/detail must not silently become “perfect ground truth.”
This representation gate may require a code repair, not new human annotation.

Pinned libraries: Ultralytics8.4.20, transformers4.51.3,
segmentation-models-pytorch0.5.0, timm1.0.15, pycocotools2.0.11.
Installation constrains Kaggle's existing torch, torchvision and NumPy; it
does not silently replace the CUDA stack. Exact runtime versions are recorded
and must match the pilot and checkpoint on resume. Upstream model revisions
for the transformer models are pinned by NB13. Initial weight fingerprints and
actual implementation/parameter identities are recorded per run.

## Evaluation: learned ROI, not oracle localisation

Each trained model is evaluated in another process at fixed epoch60, not a
validation-selected checkpoint. YOLO uses its native final-epoch EMA weights;
other backends use final raw weights. Predictions are made on the held-out
clean fold only and saved image by image as native-coordinate boxes and COCO
RLE masks where available. Detection AP uses confidence≥.001; ROI detection
uses fixed confidence≥.25, and semantic masks use probability≥.5.

Outputs include COCO box AP50:95/AP50, mask AP for segmentation models, and
per-image tyre/tread IoU, Dice and boundary F1 (2native-pixel tolerance).
Empty-ground-truth/empty-prediction mask scores are undefined rather than
invented as perfect. All actual manual images have both regions.

The downstream classifier is the **same frozen Stage-A ResNet-50** at the
matched fold/seed, `ckpt_last.pt`, from the pinned S4b-completion HF revision.
Five modes: full image, predicted tyre crop, predicted tread crop, oracle tyre
crop, oracle tread crop. Padding5%; missing prediction falls back to full image
and is counted. Both per-sample probabilities and paired macro-F1 deltas are
saved. The classifier is not retrained or selected using S5 results. This is
a crop intervention on an existing full-frame classifier, not a separately
crop-trained integrated pipeline or proof that S9 is complete.

NB17 verifies all81 statuses/checkpoint hashes, 60-epoch histories, native
prediction coverage and recomputed downstream F1. It reports each fold's
three-seed summary without presenting photographs as independent tyres.
The known fold0/2 suspected cross-tyre overlap and small fold1 tyre sample
remain limitations; no independent new-tyre generalisation/significance claim.

## Storage, stopping, and HF

HF namespace: `s5/s5-manual-2026-09-10-r1/<protocol_sha256>/`.
`protocol.json`, `pilots/<model>/`, `runs/<model>-f<fold>-s<seed>/`, `report/`.
Per run: full `state.pt`, `STATUS.json`, `epochs.csv`, identity, logs, native
per-image predictions, localisation metrics, ROI predictions and metrics.
Existing classification run IDs/results are untouched.

The **parent Session is the only upload owner**. It retains one shared rate
budget across child processes. Normal snapshots every30min; major action end,
including failures, and catchable Stop trigger immediate flush attempts.
HF-requested backoff still applies; “immediate” cannot bypass a server limit.
There are no claim/heartbeat commits. Immutable upload copies are taken under
the same lock used for checkpoint publication.

Custom checkpoints contain model, optimizer, scheduler, AMP scaler, RNG,
history, runtime and protocol hash. YOLO additionally retains its native
checkpoint/train arguments, full-precision training weights/optimizer, separate
EMA, scheduler, loader generator and scaler/RNG state. This avoids resuming
training from the native half-precision EMA substitute. Restarts resume
at the latest **HF-published completed epoch**, not mid-batch. The native YOLO
resume path is not promised bitwise-identical across hardware; changing the
package environment is rejected. A forced OS/kernel kill cannot flush.

Jobs run in isolated processes; parent monitors container RAM and ends the
child before the safety margin is exhausted where possible. `/kaggle/temp`
holds caches/checkpoints; native image exports are symlinks to the attached
dataset. Only generated per-job scratch is removed after successful upload
and verification. At least5GiB runtime and3GiB snapshot free-space checks are
enforced; no assumption of1TB scratch is made. Cache storage is not permanent.

## Validation and sources

Sources: `tyrelib/s5_data.py`, `s5_runtime.py`, `s5_notebook.py`, and
`build_s5_notebooks.py`; regression entrypoint `scripts/verify_s5_notebooks.py`.
Local checks cover region/box geometry, all81 job IDs and one/four-worker
ownership, actual dataset integrity, checkpoint/optimizer roundtrip,
incompatible-resume rejection, notebook syntax and embedded-source equality.
The pinned U-Net/DeepLab/SegFormer forward interfaces, RT-DETRv2 processor/loss,
YOLO26 detection/segmentation constructors and a known-answer COCO AP fixture
also passed CPU checks in isolated temporary libraries. All836 actual manual
regions passed the polygon gate; minimum reconstruction IoU0.980092. No existing
environment packages were replaced for those checks. Per-device hardware traces
and child-process RAM samples are saved separately for each execution segment.
Kaggle CUDA/backend/pilot execution remains unverified until the user runs it.

Primary implementation references:
[Ultralytics callbacks](https://docs.ultralytics.com/usage/callbacks),
[YOLO26](https://docs.ultralytics.com/models/yolo26),
[RT-DETRv2](https://huggingface.co/docs/transformers/model_doc/rt_detr_v2),
[actual R18 checkpoint](https://huggingface.co/PekingU/rtdetr_v2_r18vd),
[SMP API](https://smp.readthedocs.io/en/latest/models.html).
Ultralytics licensing is AGPL-3.0 or its offered enterprise terms; check before
redistributing an integrated application. This delivery is research notebooks.
