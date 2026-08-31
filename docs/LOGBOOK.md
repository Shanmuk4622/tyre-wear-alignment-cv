# Project Logbook

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
