# PROGRESS

**Live status log. Updated every working session.**
Last updated: **2026-09-03**

> **New to this project?** Read **`docs/00_WHAT_THIS_PROJECT_IS.md`** — a plain-language explanation of what we're building and why. Everything else follows from it.

Project: Vision-Based Detailed Tyre-Wear Recognition and Single-Wheel Alignment Screening
Capstone · Fall-Sem 2026–27 · Dept. of AI & ML, SCOPE, VIT-AP
Team: Bonala Shanmukesh · Gunnamneni Nehru · GV Manu Rohith · Nettem Harish Kumar
Guide: Dr. E. Sreenivasa Reddy

---

## ⬤ Where we are right now

**Stage:** **S2 complete; annotations verified; S6 XAI gate complete; S4 Stage B in progress (42/108 complete, 34 checkpointed incomplete, 32 not started).**

Dataset is done and good. **No hardware is being built.** The approach has been **redesigned** (2026-08-26) from a single engineered pipeline into a **broad, controlled, XAI-grounded comparative study**: many architectures, many techniques, classification + detection + segmentation, with explainability as the measuring instrument.

**Focus: tyre wear.** Alignment deferred — `docs/13 §3`.

### Immediate next action

> 1. **Stop every older v4-v10 NB06 session first.** Replace it with the
>    **2026-09-03 tyrelib v11** `NB06_StageB_OFAT.ipynb`, then Run All.
>    NB07 is complete and the public gate has
>    been independently audited: the locked architectures are **RegNetY-16GF,
>    DenseNet-121 and ResNet-50**, each XAI-valid and confirmed on seeds 1–3.
> 2. In Kaggle attach the prepared dataset, enable Internet, select **GPU T4 ×2**,
>    and expose `HF_TOKEN`. For the requested single-notebook run, leave
>    `ACTIVE_KAGGLE_ACCOUNTS=('acct1',)` and `ACCOUNT='acct1'`. For four real
>    parallel copies, list all four active account labels in every copy. The cell derives
>    `NUM_WORKERS=1` and `WORKER_ID=0`, then must print `worker=0/1` and
>    `MODE=ONE NOTEBOOK`. Cell 2 must print those exact three architectures and the raw
>    public-evidence coverage before any training plan is built.
> 3. NB06 runs the tyre-ROI control first and then the remaining OFAT arms on
>    fold 1 only: at most **108 runs**. The public audit now has **42 completed,
>    34 checkpointed incomplete, and 32 not started; all 76 status-bearing runs have both
>    `ckpt_last.pt` and `ckpt_best.pt`, so zero recorded epochs are at risk.**
>    v11 launches every model in a disposable child process. Its entire memory
>    is reclaimed at process exit; if it pauses at the RAM guard, the parent
>    immediately resumes that same HF checkpoint in a clean child instead of
>    ending the notebook.
> 4. After NB06 completes, run NB08 → NB09 → NB10. Re-cutting the folds remains the highest-value
>    dataset correction before making any generalisation claim.

---

## ⬤ Status board

| Area | Status | Notes |
|---|---|---|
| Project concept & scope | ✅ Done | `docs/00`, `README.md` |
| Literature review | ✅ Done | `docs/09`, `docs/10` |
| **Pilot dataset `final_v1`** | ✅ **Verified PASS** | 418 clean / 4,180 derivatives / **12 sessions** |
| Dataset analysis + difficulty probes | ✅ Done | `docs/12`, `scripts/dataset_shortcut_probe.py` |
| **Experiment design** | ✅ **Done** | `docs/13` — ~800 runs, ~440 GPU-h |
| **XAI protocol** | ✅ **Done** | `docs/14` — TER/BAR/SAR, faithfulness, H1–H3 |
| Model zoo reference | ✅ Done | `docs/04` — configs, CAM layers, cost table |
| Infrastructure spec | ✅ Done | `docs/05` — multi-account, sharding, **twenty-one bugs** |
| Evaluation protocol | ✅ Done | `docs/06` |
| **S0 infrastructure code** | ✅ **Done** | `tyrelib/tyrelib.py` **v11** — per-model process isolation and automatic same-run RAM resume (Bug 28); launch/progress clarity (27); cgroup RAM/loader fixes (25–26). **114 selftest checks** |
| S1 baselines | 🔄 NB01 ready | 2 of 5 done (colour 0.491, structure 0.483); NB01 adds HOG+SVM, majority, random-init |
| S2 architecture sweep | ⚠ **153 valid + 9 quarantined** | All 162 executions are public, but every `convnextv2_s` status reports 11,177,538 parameters and its sampled checkpoint has a ResNet-18 tensor signature. Those nine mislabeled runs are excluded. The other 153 are scientifically usable; NB05 remains 27/27 valid. `docs/18` |
| Annotations `annotation_v2` | ✅ **NBT1 verified PASS** | 418 hand-drawn + 4,180 self-healed propagated masks; actual-used fingerprint `085acfb8fb83c531` |
| S4 technique OFAT | 🔄 **NB06 in progress: 42/108 complete + 34 checkpointed incomplete; 32 not started** | All 76 public Stage-B statuses have both checkpoints. v11 isolates each model in a disposable process and automatically resumes a RAM-paused child; model recipe unchanged |
| S6 XAI | ✅ **NB07 r3 complete and public** | 18 seed-1 screens + 10 seed-confirmation runs; 1,208 evidence rows, 35 faithfulness rows, and verified `tables/stage_b_selection.csv`. Selected top three are XAI-valid and three-seed confirmed |
| S7 stress tests | ⬜ **NB08 ready** | Shuffled-label control runs first |
| **Annotation test** | ✅ **Real Kaggle PASS** | NBT1 `2026-08-30-r1`: A/B/C all PASS; clean IoU 0.9780, propagated 0.9747, ratio 0.9966; all seven revisioned artifacts public. The epoch-18 data-loader cleanup warning is fixed with in-memory `num_workers=0` |
| S8 ensembles + calibration | ⬜ **NB09 ready** | Seed/arch ensembles + conformal sets |
| S10 analysis + figures | ⬜ **NB10 ready** | Master tables, H1 test, 10 figures |
| Alignment | ⏸ Deferred | Needs calibration data that does not exist |
| Optional app | ⬜ | First on the cut list |

**Legend:** ✅ done · 🔄 in progress · ⬜ not started · ⏸ deferred

---

## ⬤ What the dataset lets us do today

One thing, honestly: **a three-class ordinal mileage-proxy classifier.**

| Capability | Supported? |
|---|---|
| 3-class mileage-proxy classification | ✅ Yes |
| Anomaly detection (needs curated healthy pool) | ⚠️ Partially |
| Tyre/tread/marking/damage masks for XAI measurement | ✅ NBT1 A/B/C PASS; fingerprint `085acfb8fb83c531` |
| Tread depth in mm | ❌ No gauge data |
| Camber / toe | ❌ No calibration, no pose, no rack data |
| Photometric stereo · unrolling · video fusion | ❌ Wrong capture modality |

Full breakdown: `docs/12_DATASET_FINAL_V1.md §4`.

---

## ⬤ The finding that governs the next month

**Four** deliberately stupid probes on the dataset's own group folds — no deep learning anywhere:

| Probe | Fold 0 | Fold 1 | Fold 2 | **Mean** |
|---|---:|---:|---:|---:|
| **Frame occupancy** — how much of the frame is tyre | 0.181 | 0.455 | **0.968** | **0.535** ← highest |
| **Colour only** — 10 stats from a 64×64 thumbnail | **0.952** | 0.399 | 0.123 | 0.491 |
| **Structure only** — 9 tread-band texture stats | 0.354 | 0.119 | **0.976** | 0.483 |
| **Annotation side-channel** — `marking→low, damage→high` | **0.978** | 0.159 | 0.108 | 0.415 |
| Majority-class accuracy | 0.360 | 0.484 | 0.423 | 0.423 |

*(macro-F1)*

**Four probes. Each near-perfect on a different fold. Four different shortcuts.**

1. **Never report a single fold.** Ten colour numbers get 95.2% on fold 0; frame occupancy gets 96.8% on fold 2; the annotation side-channel gets 97.8% on fold 0.
2. **None of them learned wear.** Each learned which tyres were in its training folds, by a different route.
3. **The floor is 0.535**, and the strongest shortcut is *framing* — the tyre fills 72% of the frame in `low` images, 62% in `mid`, 61% in `high`.
4. **A real physical signal does exist** — deep-groove shadow fraction and groove banding are monotone in the correct direction across all three classes. There is just not enough independent data to prove a model is using it rather than one of the four shortcuts above.

**Any model must beat 0.535 mean macro-F1, and all four baselines go on every figure.**

Reproduce: `scripts/dataset_shortcut_probe.py` (first two) · `annotations/README.md §6` (last two)

---

## ⬤ Open decisions

| # | Decision | Status |
|---:|---|---|
| 1 | ~~One shared HF account or four?~~ | ✅ **SETTLED — one account, `Shanmuk4622`, token in Kaggle Secrets as `HF_TOKEN`.** Budget is shared; `tyrelib` caps each worker at `100/NUM_WORKERS` (25/hr at 4 workers, 100/hr team total, 28 ops headroom) |
| 2 | ~~Swin / CoAtNet at 224 only?~~ | ✅ **SETTLED — declared 224-only** in `tyrelib.FIXED_224`, run at 224 in NB04, excluded from the Stage B resolution sweep, limitation recorded in the results table |
| 3 | ~~Early stopping?~~ | ✅ **SETTLED — removed entirely.** Every run trains the full 60 epochs. `val_qwk` still selects the best checkpoint; it never stops a run |
| 4 | Manual-audit set size for pseudo-masks | 50 images minimum; 50 unless quality looks marginal |
| 5 | ~~Which 3 architectures go to Stage B?~~ | ✅ **SETTLED by public NB07 r3:** RegNetY-16GF, DenseNet-121, ResNet-50. TER_norm 1.5785 / 1.5513 / 1.5146; all three have seeds 1–3 and `xai_status=ok`. NB06 has no fallback. `docs/18 §3.3` |
| 6 | ~~How many Kaggle accounts?~~ | ✅ **SETTLED — 4.** Proven in practice: NB02's 36 runs finished in one day, ~24 GPU-h, with correct cross-account resume and one duplicated run (now fixed, Bug 13) |
| 7 | **Re-cut the folds?** | ⚠ **NEW, and now the blocker.** Folds 0 and 2 both saturate at 1.000 and both carry `cross_fold_tyre_flags=1`. Options: (a) merge the suspect pair → ~11 tyres, 3 honest folds, invalidates the 24 finished runs on those folds; (b) report fold 1 only → keeps everything, one fold of 4 tyres; (c) leave-one-tyre-out → 11–12 folds, ~4× the compute, the most defensible. `docs/18 §6` |

---

## ⬤ Pre-registered hypotheses

**Frozen publicly before the first XAI evidence row.** The aborted NB07 session
successfully published `analysis/hypotheses.json` before processing ResNet-50.

> **H1.** TER_norm predicts cross-fold macro-F1 stability better than validation accuracy.
>
> **H2.** High SAR/DmgAR predicts larger recall loss when markings/damage are masked.
>
> **H3.** Fine-grained architectures (bilinear / attention-bilinear) have higher TER_norm than plain classifiers at matched accuracy.

Registered on: **2026-08-30T10:06:21Z** · public HF

---

## ⬤ Blockers

| Blocker | Blocks | Owner | Action |
|---|---|---|---|
| ~~S0 infrastructure not built~~ | — | — | ✅ `tyrelib` written, selftest passing, NB00–NB05 generated |
| ~~Metrics schema not frozen~~ | — | — | ✅ Frozen by the 162 public Stage-A runs; later tables add derived files without changing those run records |
| ~~Annotation not done~~ | — | — | ✅ 418 annotated, attached on Kaggle, self-healed, and verified by NBT1 |
| Only 12 sessions | Every generalisation claim | Data | Not a blocker for the study — it *is* the study's premise. Still the top priority for `final_v2` |

---

## ⬤ This week

**Infrastructure (blocking everything else)**

- [x] ~~Decide HF accounts~~ — one account, `Shanmuk4622`
- [x] ~~Library + selftest~~ — `tyrelib/tyrelib.py`, all 56 current checks pass
- [x] ~~Uploader / registry / sharding / lifecycle~~ — built, bugs 1/2/3/7 covered
- [x] ~~Create and re-list the public HF dataset repo~~ — 2,412 files audited anonymously
- [x] ~~Upload and attach `final_v1` on Kaggle~~ — all Stage-A and NBT1 runs consumed it
- [x] ~~Prove kill-and-resume~~ — multiple Stage-A runs resumed across fresh accounts/sessions
- [x] ~~Decide the account count~~ — four workers for Stage A; one account owns public HF
- [x] ~~Freeze the metrics schema~~ — preserved across all 162 Stage-A runs

**Science (cheap, do in parallel)**

- [ ] Finish S1: HOG+SVM, majority class, random-init CNN
- [x] Verify checkpoint architecture identity — 153 valid; nine `convnextv2_s` records quarantined as ResNet-18 substitutions
- [ ] Verify every XAI method has a valid target layer per architecture

**Annotation — solo, parallel to everything else (~4 h)**

- [ ] `pip install labelme` (guide `docs/15` Part A, ~5 min)
- [x] ~~Build batches, annotate 418, import, validate, propagate~~ ✅ **done**
- [x] Upload/attach `annotations/` on Kaggle — real NBT1 run consumed all 4,598 masks
- [ ] *(optional, closes a stated gap)* consistency pass on the 30 self-check images

**Data (not blocking, still the highest long-term leverage)**

- [ ] Buy a digital tread depth gauge (₹900) and start measuring
- [ ] Put a ruler or marker in frame on any future capture

---

## ⬤ Session log

### 2026-09-03 — two models completed, then retained per-epoch RSS stopped the third

The v10 output did not fail silently. It completed
`b-regnety016-head_ce-f1-s3` and `b-regnety016-prep_clahe-f1-s1`, then trained
`b-regnety016-prep_gray-f1-s2` from epoch 3 through epoch 45. At that boundary
the cgroup reached 88.1%, the checkpoint was published, and the guard stopped
the cell. HF confirms all three outcomes and both checkpoints for the partial
run.

#### Bug 28 — a model boundary was not a process-memory boundary

The public epoch CSVs show a repeatable long-session slope: about **+0.17 GB
RSS per epoch** for both inspected RegNet runs and **+0.30 GB/epoch** over the
tail of the inspected DenseNet run. This persists with `num_workers=0`, drained
telemetry, `gc.collect`, and Linux `malloc_trim`, so another in-process cleanup
call is not a reliable fix. It is retained native/library/allocator state in
the long-lived Jupyter process, not GPU OOM.

Tyrelib **v11** gives each model a disposable Python child process. The parent
keeps the HF-backed schedule; the child owns exactly one Trainer. On child exit
the operating system reclaims the complete address space. If a child still
reaches the RAM guard, the parent launches a clean child and resumes the same
published epoch automatically. The parent also refuses to start any model in
the last 45 minutes of the actual Kaggle session. No scientific configuration
changed.

Public HF audit: **42/108 completed, 34 checkpointed incomplete, 32 absent**;
all 76 status-bearing runs have both checkpoints.

### 2026-09-02 — requested one worker, but the saved notebook ran as 0/4

The submitted NB06 output is internally conclusive: cell 1 printed
`account=acct1 worker=0/4`, `rate cap 25/hr`, and a four-row shard table. It
was therefore never a one-worker launch. The same output also proves the model
was training: `b-densenet121-wd_low-f1-s3` completed epoch 37 in 3.7 minutes;
the saved widget then showed epoch 38 at 0%, which is only the last serialized
state of a live `tqdm` widget. HF is the authority and now records that run as
**completed, epoch 60/60**.

#### Bug 27 — two editable worker settings and a misleading saved widget

`tyrelib` **v10** replaces the separate editable worker count/map with one
tuple: `ACTIVE_KAGGLE_ACCOUNTS=('acct1',)` by default. `NUM_WORKERS` and
`WORKER_ID` are derived. A single run must print `worker=0/1` and
`MODE=ONE NOTEBOOK`; four copies list all four account labels. Every epoch now
prints a plain `[LIVE] ... batch 1/N completed -- training is active` heartbeat,
so notebook progress no longer depends on how Kaggle serializes the widget.
`[LOADER] workers=0` is also labelled as zero **CPU input helpers**, not zero
training workers; the model still uses both T4s.
The PyTorch tensor-to-float warning in weight-norm telemetry is fixed with an
explicit detached, no-gradient norm. **No model, input, batch, optimiser,
split, seed, or 60-epoch budget changed.**

Public HF audit at this repair: **36/108 completed, 39 checkpointed incomplete,
33 absent**. All 75 status-bearing runs have both `ckpt_last.pt` and
`ckpt_best.pt`.

### 2026-09-01 (c) — the RAM guard was right; it was measuring the wrong machine

`b-densenet121-wd_low-f1-s3` paused at epoch 36 on a legitimate-looking 89.6%.
Two things were wrong underneath it, and both had been sitting in the logs.

#### Bug 25 — `psutil.virtual_memory()` reports the HOST, not our cgroup

Inside a container `/proc/meminfo` describes the machine, not the limit the
kernel enforces on us. Every pause decision since the guard was written was
made on a number that does not describe our budget: on a busy host it can sit
near 90% regardless of what the notebook does, and it would equally miss a
container genuinely about to be killed.

`container_memory()` now reads `/sys/fs/cgroup/memory.current` + `memory.max`
(v2) or the v1 equivalents, and falls back to psutil only if neither exists.
The source is printed. A pause now also says **where** the memory is —
"this process 4.1 GB, 2 child proc 21.8 GB, rest 2.2 GB" — because a bare
percentage is not something anyone can act on.

#### Bug 26 — two loader workers that did nothing, and cost a cgroup's worth

That run logged **`dl 0%` on every one of its 49 epochs**. Forty-nine
consecutive zeros, at 4.2 min/epoch: the GPU is the bottleneck by a mile. And
the loader was still starting `workers=2 pin_memory=True` — two forked
processes counted against the same cgroup, plus PyTorch's pinned-host
allocator, which caches and never returns.

The ROI arms had already been moved to the synchronous loader when *their* RAM
climbed 3 → 20 GB. The full-frame arms were left alone because the diagnosis
was written up as "an ROI problem". It was a loader problem; ROI just hit it
first.

`dataloading_is_free(cfg)` now makes that an explicit rule: at ≥320 px, no
workers and no pinned memory. Smaller configs keep them.

> Both bugs were visible in data already being collected. `dl 0%` was on every
> epoch line for days, and nobody had checked where the memory percentage came
> from. Recording a metric is not the same as reading it.

#### Delivered

- `tyrelib` **v9** — `container_memory`, `memory_report`,
  `dataloading_is_free`; `RAM_GUARD_REVISION=2026-09-01-r2`.
  **104 selftest checks**, 9 new; the only two failures are pre-existing
  torch-only tests that cannot run offline.
- All 12 notebooks regenerated on v9. `docs/05 §7` Bugs 25–26.

#### What to expect

`[LOADER] workers=0 pin_memory=False` on the 384/512 px arms, and epoch times
unchanged (they were never waiting for data). The RAM line now names its
source, e.g. `[cgroup:v2]`. If a pause still happens it will state which
process is holding the memory — which is the number we have been missing.

---

### 2026-09-01 (b) — "2 running, 2 stopped": the fix for duplicate training was the cause

Not a crash. NB06 ran with `steal_stale=False`, which put every run owned by
another account into `busy` **permanently**. A worker that finished its 27-run
static shard printed

```
  reserved for other static owners: 68
  -> will run 0 run(s) this session
```

and the notebook ended normally, while the other accounts still had twenty runs
each. The shard is LPT-balanced on *estimated* cost and skewed further by pauses
and resumes, so **some worker always runs dry first**.

`steal_stale=False` was itself the fix for Bug 13, where aggressive stealing
trained `a-vgg16bn-base-f1-s1` twice. The two failures sit either side of one
question — *may I take work that is not mine?* "Always" duplicates. "Never"
idles. Neither is the answer.

**v8: yes, once I have nothing of my own, and only after checking properly.**
`claim_or_yield` is a two-phase claim — write the claim and flush it so it is
visible, wait out the race window, re-read, and if two accounts claimed the same
run the lowest account name wins. Both sides compute that from the same bytes,
so exactly one proceeds. Plus: own work always first, each worker enters the
shared pool at a different offset, and no takeover inside the last 90 minutes of
a session.

Verified with four threads racing for six runs on a shared registry: **exactly
one winner each**, both phases observed — two blocked at step 1, one account
that won step 1 correctly yielding at step 2. And a worker whose shard is fully
done now plans 27 runs where it previously planned 0.

#### Delivered

- `tyrelib` **v8** — `Session.claim_or_yield`, takeover pool in `plan()`,
  `takeover_when_idle` on `run_all`. **95 selftest checks**, 9 new; the only two
  failures are pre-existing torch-only tests that cannot run offline.
- NB06 now calls `run_all(..., steal_stale=False, takeover_when_idle=True)`.
- All 12 notebooks regenerated on v8. `docs/05 §7` Bug 24.

#### What to do

1. Stop the running sessions and upload the v8 notebooks. Cell 1 must print
   `tyrelib v8 loaded`.
2. The plan now prints `available if I go idle : N` — that is the pool, not
   work being taken from anyone.
3. When a worker finishes its shard, expect `[IDLE] my own N run(s) are done...`
   followed by `[IDLE] <run>: claimed after settling`, or a `yielded to acctX`
   line if another account got there first. Both are correct.

---

### 2026-09-01 — "the workers stop the notebooks by themselves": one guard, one leak

Two bugs with one symptom. Diagnosed from the public telemetry of
`b-densenet121-res512-f1-s3` rather than from the notebook output, because the
notebook output showed nothing — no error, no message, just an epoch line and
then silence.

#### Bug 22 — the guard fired on a two-second spike, and a pause killed the cell

The host-RAM guard compared **`ram_percent_peak`** — the maximum 1 Hz sample of
the epoch — against 88%, and since v5 *any* pause stopped the whole worker.

| epoch | peak % | RSS GB | |
|---:|---:|---:|---|
| 1 | **95.2** | 28.4 | pauses → stops the worker |
| 2 | **92.8** | 27.8 | |
| 3 | 76.7 | 22.5 | |
| 10 | **17.8** | 3.5 | next session |
| 26 | 46.3 | 12.1 | |

Mean peak after epoch 5 is **42.9%**. The 95.2% is the first epoch of a session
— annotation rebuild, loader warm-up, resume checkpoint load — all released
within two epochs. **One transient ended an eight-hour session with eighteen
runs untouched.**

Fixed: measure live *after* `release_host_memory()`, not the epoch peak;
hysteresis (pause 88%, resume 80%); and a recovered RAM pause now releases the
model, re-measures, and **continues to the next run** instead of ending the
cell. Watchdog and interrupt pauses still stop — those mean there is no time
left.

#### Bug 23 — and there was a real leak underneath

RSS climbed **+0.54 GB per epoch**, 3.5 → 28 GB, which Kaggle kills with no
catchable exception. `HardwareMonitor.dump()` wrote both sample buffers in full
every ten epochs and **never cleared them** — ~300,000 dicts on a four-hour
run, with a full DataFrame rebuilt over all of them each time. `step_traces`
was rewritten whole with `open(..., "w")`.

Fixed: each dump writes only what is new and drops it; concatenated gzip
members still read back as one table. Telemetry flushes **every** epoch now
that it is incremental, so a hard kill loses one epoch of trace instead of
nine. Verified 6 × 500 rows in → buffer empty after each dump → 3,000 rows
back, columns intact, no interior headers.

> Two bugs, one symptom, and fixing either alone would have been wrong. The
> guard turned a slow leak into an abrupt stop; because it fired on a peak it
> also fired when there was no leak at all. Fixing only the guard hides the
> leak until it reaches 88% honestly.

#### Delivered

- `tyrelib` **v7** — `host_ram_percent`, `host_ram_headroom`,
  `HOST_RAM_RESUME_PERCENT`, incremental telemetry. **12 new selftest checks**;
  the only two failures are pre-existing torch-only tests that cannot run
  offline and pass on Kaggle.
- All 12 notebooks regenerated, embedded library byte-identical, no saved error
  outputs.
- `docs/05 §7` Bugs 22–23.

#### What to do

1. **Stop the running v6 sessions** and upload the v7 notebooks.
2. Cell 1 must print `tyrelib v7 loaded`.
3. Expect `transient, continuing` where a session used to stop, and
   `host RAM back to NN% ... continuing with the next run` after a real pause.
4. Nothing is lost: every paused run resumes from its checkpoint.

---

### 2026-08-30 — NB07 completed; public Stage-B gate verified

The final NB07 run completed without an exception, published the selection,
re-listed all required files, and finished with `commits=1 failures=0`. The
saved notebook and an independent anonymous HF audit agree. Public artifacts:

- `tables/stage_b_selection.csv`: 18 rows, revision `2026-08-30-r3`;
- `tables/xai_evidence_all.csv`: 1,208 rows, revision r3;
- `tables/xai_faithfulness.csv`: 35 rows;
- all three selected rows have `eligible=True`, `selected_top3=True`,
  `xai_status=ok`, and `seeds=3`.

**Locked Stage-B architectures:**

| rank | architecture | TER_norm | BAR | raw valid maps |
|---:|---|---:|---:|---:|
| 1 | RegNetY-16GF (`regnety016`) | 1.5785 | 0.0310 | 180/180 |
| 2 | DenseNet-121 (`densenet121`) | 1.5513 | 0.0455 | 178/180 |
| 3 | ResNet-50 (`resnet50`) | 1.5146 | 0.0512 | 180/180 |

The public selection table's original `n_images` denominator counted only
valid rows, so it displayed 100% coverage for DenseNet. This does **not** alter
the ranking, eligibility, seed count, TER_norm, BAR, or selected top three.
The notebook source now computes coverage from all `xai_status=ok` rows, and
NB06 independently reconstructs and prints raw selected-evidence coverage from
the 1,208-row public evidence table before training. The public gate itself is
valid, so **NB06 may start now**.

---

### 2026-08-30 — NB07 third stop: literal `NA` reached `groupby.mean`

The full seed-1 screen completed and published before the exception. Public HF
now contains all 18 r3 evidence/faithfulness pairs: **10 valid architecture
screens**, seven `excluded_no_faithful_cam` rows, and the quarantined
`convnextv2_s` architecture mismatch. The crash happened only when ranking the
completed screen:

```
TypeError: agg function failed [how->mean,dtype->object]
```

`evidence_metrics()` deliberately returns the literal string `NA` when a CAM
has zero total saliency. Fresh in-memory frames therefore mixed strings and
floats; pandas could not average `ter_norm` or `bar`. Resumed CSVs masked the
bug because `read_csv` normally interprets `NA` as missing.

NB07 now normalises every numeric evidence column with `to_numeric(...,
errors="coerce")` when a CSV is resumed, when a new per-run frame is created,
after the screen is concatenated, and before the three-seed summary. A
mixed-float/`NA` assertion runs before any checkpoint work. Replaying the exact
function against all current public r3 files passes and produces 10 ranked
architectures. The seed-1 shortlist is RegNetY-16GF, ResNet-50, MobileNetV4,
DenseNet-121 and ConvNeXt-V2-T; it is not the locked top three until seeds 2/3
finish. Coverage counts are now printed beside each ranking so undefined CAM
maps are visible rather than silently hidden.

---

### 2026-08-30 — NB07 second stop exposed nine mislabeled ConvNeXt-V2-S runs

The corrected r3 gate behaved as intended for two genuine XAI failures:
ResNeXt-50 (`sanity_delta` about 0.013) and VGG-16-BN (about 0) both passed the
faithfulness condition but failed the locked randomisation condition. NB07
published both exclusions and continued. A public audit now finds four
resumable r3 screens: 60-row valid evidence for ResNet-50 and DenseNet-121,
plus the two one-row exclusions. ConvNeXt-V2-T had not been committed before
the exception and will be recomputed. The run then stopped while reconstructing
`a-convnextv2_s-base-f1-s1` because the registered timm pretrained tag does
not exist.

The checkpoint itself revealed the more important fault. Its config says
`convnextv2_s`, while its tensors are an 11.19M-parameter ResNet-18
(`conv1`, `layer1`…`layer4`). A public audit of all nine matching statuses
found the same `n_params_total = 11,177,538` and the same ~134 MB checkpoint
size in every fold/seed. The historical emergency fallback therefore trained
ResNet-18 nine times under ConvNeXt-V2-S run ids. These are complete execution
records but **not valid ConvNeXt-V2-S results**.

The source and notebook are repaired in four places: timm absence can no
longer trigger an architecture substitution; checkpoint reconstruction strips
retired pretrained-weight tags because public checkpoint weights are supplied;
NB07 checks the saved tensor signature before loading; and an incompatible or
mislabeled checkpoint writes `excluded_checkpoint_arch_mismatch`, pushes it
immediately, frees the local file and continues. NB03 no longer plans the
unsupported pretrained Small arm. The valid Stage-A count is consequently
**153**, not 162. XAI stays at revision r3 because its metric and thresholds did
not change.

---

### 2026-08-30 — NB07 first real run: r2 sanity failure exposed two bugs

The first real NB07 pass processed ResNet-50 and then reached ResNeXt-50.
Grad-CAM and HiResCAM had strong insertion-minus-deletion scores (~0.502), but
the old sanity score was only **0.01297** and **0.01314**, below 0.05, and the
cell raised a `RuntimeError`.

Inspection found two separate bugs. First, a no-method survivor stopped all 18
architectures instead of becoming a reported exclusion. Second, the sanity
function constructed a two-image batch but kept only map `[0]`; its pixelwise
MAE was also biased downward for sparse CAMs because most pixels are zero.
Revision `2026-08-30-r3` now averages scale-independent decorrelation across
both maps. The **0.05 threshold is unchanged**, but it now means correlation
must fall by more than 0.05 after randomisation.

If both corrected methods still fail, NB07 publishes the failed rows, records
`excluded_no_faithful_cam`, removes that architecture from TER ranking, and
continues. It requires five valid architectures before confirmation and three
valid three-seed candidates before Stage B. An anonymous audit found the
public hypotheses file but no r2 ResNet-50 or ResNeXt-50 XAI CSVs, so r3 safely
recomputes both. No model is retrained.

---

### 2026-08-30 — NBT1 real PASS; Stage A complete; NB07 is next

**NBT1 passed on the real Kaggle package.** The attached propagated masks were
not aligned, so the notebook correctly rebuilt all 4,180 derivatives from the
418 clean masks and transform traces before testing them. The exact mask set
actually used has fingerprint `085acfb8fb83c531`. Parts A and B passed, and the
differential U-Net test produced clean IoU **0.9780**, propagated IoU **0.9747**,
ratio **0.9966**, shuffled-control IoU **0.8029**, and trivial-mask IoU
**0.6709**. All seven revision-specific artifacts were confirmed public under
`annotation_test/2026-08-30-r1/`.

The only notebook error-like output was a burst of ignored Python
`DataLoader.__del__` assertions at epoch 18. It was a Jupyter multiprocessing
cleanup race after in-memory loaders were repeatedly constructed; it did not
interrupt training or affect the result. NBT1 now uses `num_workers=0` for
those RAM-backed arrays, removing that warning without changing the test.

**Superseded by the later checkpoint-integrity audit above.** The first audit
found all 162 final metrics and best checkpoints, but did not yet compare the
declared architecture with the checkpoint tensor signature.
160 status files say `completed`. Two old VGG runs say `failed` only because a
now-fixed telemetry listing race occurred after epoch 60; both contain their
scientific outputs and must not be rerun. There are 161/162 predictions files:
RegNetY fold-2 seed-2 is missing only that derived parquet, and NB09 now
reconstructs it from the public best checkpoint if selected. NB05's 27/27
foundation runs are complete. Fold 0 and fold 2 remain leak-flagged; fold 1 is
the honest selection fold. Full results and costs are in `docs/18`.

**NB06–NB10 were corrected around that evidence.** NB07 now runs first and
locks a faithfulness-gated, three-seed XAI shortlist in HF. NB06 refuses to use
a fallback shortlist and runs the implemented OFAT factors on fold 1 only.
NB08 uses a revisioned three-fold shuffled-label control and robust stress
measurements; NB09 performs real flip TTA and disjoint calibration/conformal
splits; NB10 builds the final tables and figures from public artifacts. The
shared library now implements ROI cropping, CLAHE, weighted sampling, frozen
fine-tuning, dual-T4 training and CORAL-aware XAI rather than accepting silent
no-op configurations.

---

### 2026-08-30 — NBT1 repair before the real run

The annotation test had several failure modes that could produce either a
misleading notebook error or a plausible-looking run that learned nothing:

- Part B stopped the notebook **before** the diagnostic overlay cell, so the
  evidence needed to diagnose a geometry failure was never saved under Run All.
- Part A sampled only 200 masks while its verdict claimed all 4,598 were valid.
- FP16 could produce a non-finite U-Net loss while GradScaler silently skipped
  every optimiser step.
- Resume existed only at epoch boundaries, used one GPU, did not restore CUDA
  RNG, and a corrupt checkpoint was ignored in favour of a silent fresh start.
- The nominal 30-minute push depended on an epoch finishing; it was not a timer.

**Repaired in NBT1 `2026-08-30-r1`:** all-mask coverage and source-link checks;
diagnostic overlay saved before the Part-B gate; optional transform groups with
too few samples are reported rather than converted into a false failure; FP32
with a hard finite-loss assertion; `DataParallel` over both T4s; atomic
checkpoints every five batches; deterministic mid-epoch batch/flip order;
optimizer, scheduler, scaler, CPU/CUDA RNG and partial loss counters restored;
config/fingerprint mismatch protection; a real 30-minute HF timer with a
100-write/hour cap, retry/backoff, and checkpoint-before-flush on SIGTERM or
Ctrl-C. Results use the revision-specific prefix
`annotation_test/2026-08-30-r1/`, preserving the earlier attempt and preventing
its legacy FP16 checkpoint from skipping the repaired run.

**Verification completed locally:** all 19 cells executed against a synthetic
Kaggle-style package with exactly 418 clean images and 4,180 derivatives. Parts
A/B/C passed. A forced stop after batch 2 wrote an emergency checkpoint; a new
process resumed at batch 3 and completed with all three parts passing. The
real Kaggle gate subsequently passed; see the session entry above.

---

### 2026-08-29 — NB05's dinov2 failure, and NBT1 testing the wrong dataset

Two reports, two different kinds of "the code is fine, something upstream is
not".

#### NB05 — all 18 dinov2 runs died on their first batch

```
AssertionError: Input height (392) doesn't match model (518).
```

`build_model` never told timm what resolution the images would be. Most models
do not care; `vit_*_patch14_dinov2` is created at `img_size=518` and its patch
embedding asserts an exact match. It failed in `forward`, not in
`create_model`, so the old fallback-to-resnet18 `except` — which wrapped only
construction — never fired.

Each run failed in **0 seconds** with no checkpoint, which is why the one-worker
check honestly reported 9 of 27: `state absent`, `action train`. The bookkeeping
was right. CLIP's 9 runs are genuinely finished.

**The part that matters is that NB00 already found this on 26 August.** It
printed `FAIL dinov2_s`, `FAIL dinov2_b`, `17/19 architectures build`, and
`Exclude the failures from the sweep, or fix them, BEFORE Stage A` — and then
returned success. The same run printed `resume equivalence : FAIL`, which was
Bug 8, and that was ignored too.

> **A preflight that reports but does not block is not a preflight.** Two
> accurate diagnoses, printed two days early, cost 18 runs and a four-account
> session because nothing stopped.

Fixed three ways: `build_model` now takes `img_size`, tries
`img_size`/`dynamic_img_size`/plain in order and **verifies each with a real
forward pass** before returning; **NB00 raises** on any architecture or
resume-equivalence failure; and every Stage notebook calls
`tl.assert_zoo_ok(ARCHS)` before planning — 15 s, no download.

#### NBT1 — the test was right, the dataset was stale

NBT1 failed Parts B and C. It should have. Line 4 of its own output:

```
annotation version: v1 | 418 hand-drawn + 4180 propagated
```

**Kaggle still has `annotation_v1`.** The regenerated masks were never
re-uploaded. NBT1's numbers match the v1 measurements exactly — correct 15.69
vs swap control 9.84, margin −0.22, against v2's +13.59 measured locally. The
local `annotations/` folder is v2 and re-verifies clean.

So NBT1 worked precisely as designed: it detected broken propagation on data
that has broken propagation.

**The version guard I added for this was itself wrong** — it refused to run
unless `ANNOTATION_VERSION.json` said `v2`, and blocked a perfectly good
dataset over a label. Worse, it would have happily proceeded on broken masks
carrying a `v2` label. A label is not evidence. Removed.

What the notebook does now:

* **Nothing is gated on the version string.** It is printed for information.
  **Part B measures the masks**, in three minutes, and that measurement is the
  only gate. It catches a stale copy, a mislabelled copy and a genuinely
  broken copy alike.
* **A mask fingerprint at startup** — sha256 of the first eight supplied
  propagated masks, `301de19631fc026b` at that time — with the one-liner to compute the same
  digest from the local folder. This is the only way to answer "is Kaggle
  giving me the same bytes as my machine?" from inside a notebook.
* **Part B failing stops the notebook.** Part C took another 20 minutes to
  reach the same verdict more expensively — the same "reports but does not
  block" mistake as the preflight, in a notebook written the same day.

#### The actual cause: Kaggle pins a dataset version per notebook

The folder on disk was correct the whole time. Attaching a dataset to a
notebook pins **one specific version**; uploading a new version does not move
existing notebooks onto it, and nothing on screen says so. Input panel → the
dataset → version dropdown → newest. Now documented in `docs/15 §E4`.

#### Where Stage A actually stands

**144 of 162 runs (89%).** NB02 36 ✅ · NB03 45 ✅ · NB04 54 ✅ · NB05 clip 9 ✅ ·
NB05 dinov2 0/18 ❌. Verified against the repository's 167 run directories.

#### Delivered

- `tyrelib` — Bug 15 fix, `verify_zoo()`, `assert_zoo_ok()`. **37 selftest
  checks** (4 new, including that `Trainer` passes `input_resolution` through
  and that every patch-based architecture has a divisible resolution).
- NB00 raises. Stage notebooks assert the zoo. NBT1 guards the version and
  stops on a Part B failure.
- `docs/05 §7` Bug 15 · `docs/18` rewritten for all four notebooks.

---

### 2026-08-27 (h) — Stage A complete, 36/36, and the answer is "the folds are broken"

Four accounts, ~24 GPU-hours, one day. **The resume machinery worked.** Runs
resumed from epochs 31, 41, 43 and 51 across account boundaries, 18 runs were
correctly skipped as already finished by another account mid-session, and disk
was reclaimed after each completion.

**Full results: `docs/18_STAGE_A_RESULTS.md`.**

#### The result

Folds 0 and 2 return **exactly 1.000** on QWK, macro-F1 and accuracy for every
one of 24 runs. Both carry `cross_fold_tyre_flags = 1`. Fold 1, the only clean
fold:

| | |
|---|---:|
| mean best val QWK — *what the training log prints* | 0.931 |
| mean best macro-F1 — *selected on the 4-tyre val fold* | 0.736 |
| **mean final macro-F1 — fixed 60 epochs, chosen by nobody** | **0.499** |
| strongest trivial baseline on fold 1 | **0.455** |

**+0.04 macro-F1 over counting how much of the frame is tyre, for 12 GPU-hours.**

Three things worth more than the ranking:

* **Median best epoch is 8 of 60.** `vgg16bn` peaks at 3 and ends at macro-F1
  0.317; one `densenet121` run peaked at **epoch 1**. 232 training photographs
  of 8 tyres are memorised in under ten epochs. Not an argument for early
  stopping — equal budget is what keeps the comparison a comparison — but it
  means `ckpt_best` is optimistically biased and must never be quoted without
  the fixed-budget number beside it.
* **Seed spread exceeds the architecture gap.** resnet50 varies 0.319 across
  its own three seeds; best-vs-worst architecture is 0.134. Open Decision #5
  cannot be answered by Stage A.
* **QWK 0.93 and macro-F1 0.50 are the same models.** Report both, always.

#### Three more bugs, found by the run rather than by reading

* **Bug 12 — the observer killed the run.** `HardwareMonitor.dump()` built a
  DataFrame from a list the 10 Hz sampler was still appending to:
  `ValueError: Length of values (35249) does not match length of index (35250)`.
  Killed two runs after 43 and 66 minutes. The lock was held in `dump()` — and
  bought nothing, because the *append* side was unsynchronised. Now snapshots
  under the lock, and **telemetry can no longer raise at all**: an observer that
  can kill a three-hour run is a liability however good its data is.
* **Bug 13 — a claim nobody can read is not a claim.**
  `a-vgg16bn-base-f1-s1` was trained to completion by **acct1 and acct2**, same
  `config_hash`, ~1.4 GPU-h twice. `can_claim` read registry shards last
  downloaded hours earlier, and `emit` only enqueued the claim against a 30-min
  push cycle. Both workers were correct on the information they had. Now
  re-pulls before stealing and flushes the claim immediately.
* **Bug 14 — two bugs, one symptom.** `confirm_on_hf` tested for
  `summary.json`; `enqueue_light()` never uploaded it. All 36 finished runs
  reported `RESUMABLE`. Neither half is visible alone.

#### Delivered

- `tyrelib` — Bug 12/13/14 fixes, `aggregate_remote()`, `honest_table()`,
  `RemoteInventory.qwk()`. **33 selftest checks** (6 new), including one that
  reproduces the Bug 12 race and proves the fix holds over 400 attempts.
- NB02 Step 6 now reads **all** accounts' results from HuggingFace and prints
  `best` next to `final` with the leak-flagged folds separated.
- `docs/18_STAGE_A_RESULTS.md` — new. `docs/05 §7` — Bugs 12–14.
- Open Decision #6 settled (4 accounts, proven). **New Open Decision #7: re-cut
  the folds** — now the blocker for everything downstream.

#### HuggingFace

Public, ~25 GB, 41 run directories, 12 files each. `tables/all_runs.csv` holds
only the last worker's local subset — use `aggregate_remote()` instead.

---

### 2026-08-27 (g) — Two silent bugs found: ten hours of retraining, and 4,180 wrong masks

Both had the same shape: **code that failed by doing nothing, and left behind
output that looked completely normal.** Neither raised. Neither logged a
warning. Both were caught only by asking a question the code could not answer
about itself.

---

#### Bug 8 — the checkpoints were on HuggingFace and nothing ever fetched them

**Reported as:** "I ran NB02 for 10 hours on 4 workers, then ran it on 1 worker
and it is training the models again."

**It was not a sharding bug.** `Trainer.try_resume` began with

```python
if not self.ckpt_last.exists():
    return False
```

`ckpt_last` is a path on the Kaggle session disk, which Kaggle wipes when the
session ends. So in *every* fresh session that file was absent and the run
restarted at epoch 1 — of a run whose checkpoint had been pushed to
HuggingFace every single epoch, exactly as designed. Saving worked. Pushing
worked. Verification-on-HF worked. The missing piece was the line that brings
them **back**.

It had been restarting at every session boundary all along. Changing the worker
count is just when it became visible.

**Fixed:** `RemoteInventory.fetch_run()` pulls `ckpt_last.pt`, `ckpt_best.pt`
and `metrics/epochs.csv` before `try_resume` looks at the disk.

#### Bug 9 — planning read intentions, not facts

The registry records *who claimed what, on which account, under which
`WORKER_ID`* — all relative to a session. Worse, it treated **`failed` as
"start over"**. The registry showed **26 of 36 Stage A runs as `failed`, each
with a good `best_qwk` already recorded**, and every one was queued for a fresh
start despite having an intact checkpoint.

**Fixed:** state now comes from `runs/<id>/STATUS.json` in the repository, which
says the same thing to every account at every worker count. Three states, and
`failed` is not one of them — `completed` (skip), `resumable` (continue from
its epoch, whatever ended it), `absent` (train).

`NUM_WORKERS` and `WORKER_ID` now decide **only what an account starts first**.
Locked in by a selftest assertion that run state is identical at 1, 2 and 4
workers.

#### Bug 10 — staging fell inside the 20 GB output quota

`Path("/kaggle/temp").exists()` is False on the current Kaggle image until
something creates it, so every session took the `./_work` fallback — inside
`/kaggle/working`, capped at 20 GB. A `vgg16bn` checkpoint is ~1.6 GB and we
keep two per run. This is the likeliest cause of the 26 `failed` runs, and
there was no way to confirm it because the exception type was never recorded.

**Fixed:** create `/kaggle/temp` rather than test for it; print free space and
warn under 20 GB; write `ERROR.json` with the exception type and free space at
the moment of failure; delete a finished run's local checkpoints once
HuggingFace confirms it has them.

---

#### Bug 11 — all 4,180 propagated masks were geometrically wrong

Asked "are the annotations transferred to all the remaining images?", the
coverage answer was yes: 4,598 of 4,598, no orphans, sizes matching, legal class
values. **The masks were in the wrong place.**

`propagate_annotations.py` matched operations by substring and read their
parameters under guessed key names:

```python
if "crop" in name:
    box = p.get("box") or [p.get("x_min"), ...]    # trace records: crop_box
elif "rotat" in name:
    ang = p.get("angle", p.get("limit", 0))        # trace records: degrees
```

Neither key exists in this dataset. `all(v is not None ...)` was False and
`ang` was 0, so **the crop and the rotation were skipped on every derivative**
— and the files were written anyway. Only the flip was applied, and the final
`resize` stretched the full frame instead of letterboxing it.

| | v1 | v2 |
|---|---:|---:|
| IoU against a correct replay | 0.83 | 1.00 |
| Alignment with its own image | 16.1 | **32.9** |
| Alignment of a *different* image's mask (control) | 9.8 | 13.5 |

v1 barely beat a mask belonging to a different photograph.

The hand-drawn 418 were never affected. Rotation sign was settled by
measurement, not assumption: on the largest-|angle| decile `rotate(+degrees)`
scored 33.96 against 28.36 for `rotate(-degrees)`.

**Fixed and regenerated.** The script now dispatches on exact operation names,
letterboxes correctly, **raises on any operation it cannot classify as
geometric or photometric**, and verifies its own output against three
corrupted controls before claiming success. Annotations bumped to
`annotation_v2`; checksums rewritten (5,465 files).

> Neither bug was findable by reading the output. Both were found by making the
> code answer a question about itself that it could fail.

**Delivered**

- `tyrelib` v2 — `RemoteInventory`, `staging_root()`, `fetch_remote_state()`,
  `Session.reconcile()`, `prune_local()`, `ERROR.json`, `last_epoch` tracking.
  **27 selftest checks passing** (8 new).
- All 11 notebooks regenerated; base64 round-trip verified byte-identical.
- **`NBT1_Annotation_Test.ipynb`** — new, standalone, ~25 min. Coverage, then
  a control-based geometry test, then the differential test: train on
  hand-drawn masks only, score on propagated ones.
- `annotations/` regenerated → `annotation_v2`.
- Docs: `05 §3 §7` (Bugs 8–10), `15 §E3 §E4`, `notebooks/README`, `03`, `12`,
  `annotations/README §0`.

**Open**

- The 26 `failed` runs now resume rather than restart — but *why* they failed is
  still unconfirmed. `ERROR.json` will answer it on the next occurrence.
- Run NBT1 to confirm the regenerated masks end-to-end on a GPU.

---

### 2026-08-27 (f) — NB02 diagnosed, notebooks completed, a leak found

**`val_QWK 1.0000` at epoch 13 is not a bug — but it is not a success either.**

Fold 0 validates on **4 tyres, roughly one per class**. A model only has to tell
three specific tyres apart. Four trivial baselines already score near-perfectly
there, so a ConvNet reaching 1.000 is the *expected* outcome. The library now
prints a warning whenever val_QWK ≥ 0.995 and writes `split_health.json` with
the tyre count per fold.

**And a real leak turned up while checking.** `session_group` came from a
12-second timestamp gap — a **proxy for tyre identity, not a measurement**.
A tread-pattern audit (`scripts/tyre_identity_audit.py`) found:

| A | B | ratio | folds |
|---|---|---:|---|
| `mileage_070000__session_001` | `mileage_090000__session_001` | **0.90** | **2 vs 0** |

(1.0 = indistinguishable from two frames of the same tyre.) They are almost
certainly one tyre, and they sit either side of fold 0's split — same tyre in
training and in validation. Fold 0's `mid` result is leak-inflated.

My first pass over-merged and claimed "6 tyres"; small sessions (2, 3, 5 images)
have unreliable within-session baselines that inflate the normalised score. With
a ≥15-image reliability filter the defensible answer is **one confirmed
suspect pair, and 3 sessions too small to adjudicate** — so the true tyre count
is somewhere between 9 and 12, not 12.

The pair is now baked into `tyrelib.KNOWN_CROSS_FOLD_PAIRS` and flagged at the
start of every affected run.

**Fixes**

| Issue | Fix |
|---|---|
| `torch.cuda.amp.autocast` deprecation | `_autocast()` / `_grad_scaler()` use the `torch.amp` API with a fallback |
| Annotations in a separate dataset | One Kaggle dataset now: `FINAL/` + `annotations/` side by side. `find_annotations_root()` added |
| Perfect scores looked like success | Warning + `split_health.json` on every run |
| Floor was stale | `tl.FLOOR = 0.535` (frame occupancy), asserted in the selftest |

**Notebooks — the full set is now generated (11)**

| # | Notebook | Stage |
|---|---|---|
| 00–05 | preflight, baselines, Stage A ×4 | S0–S2 |
| **06** | **Stage B technique OFAT** — 12 factors, ROI first | S4 |
| **07** | **XAI** — TER/BAR/SAR/DmgAR, randomisation sanity, Figure 1 | S6 |
| **08** | **Stress tests** — shuffled-label control first, then 6 interventions | S7 |
| **09** | **Ensembles + conformal** — prediction sets, the `uncertain` outcome | S8 |
| **10** | **Analysis** — master tables, the H1 test, 10 figures | S10 |

All 11 parse clean; base64 round-trip verified. `tyrelib` is **v2**, 96 KB,
19 selftest checks passing.

**NB07 pre-registers H1–H3 before any XAI runs**, writing them to HF with a
timestamp. That ordering is the difference between a hypothesis and a
post-hoc story.

### 2026-08-27 (e) — Annotation delivered, validated, and it changed the baselines

**418 images annotated. Package is complete and verified.**

| | |
|---|---:|
| clean masks / boxes / polygons | 418 each |
| propagated derivative masks | 4,180 |
| COCO annotations | 966 = 418 + 418 + 67 + 63 |
| `marking` (paint stripes) | 67 |
| `damage` | 63 |
| validation problems | **0** |
| package size | 27 MB |

**The validator was wrong, not the annotations.** It flagged 410 images as
"TREAD ≈ TYRE — shoulders not excluded". Checked by rendering overlays: the
camera faces the tread crown head-on, so the shoulders curve out of frame.
Median tread/tyre ratio is **0.990**, and **114 images have no visible shoulder
at all**. Annotating one would mean inventing a region the image does not
contain. Heuristic relaxed; it now reports the ratio as an observation.

**Propagation verified numerically** rather than by eye. 2,122 of 4,180
derivatives carry a horizontal flip. Comparing mask centroids against sources:
flipped → mirrored (error **0.0011** vs 0.0447 unmirrored), unflipped →
unchanged (**0.0002** vs 0.0464). Dimensions match on 200/200 sampled; class
values stay discrete, so nearest-neighbour held.

**A real bug caught: 160 empty box files.** `import_annotations.py` derived the
tyre box from the raw index `m == 1` — which, once `tread` is painted over it,
is empty on a head-on photo. Same trap `annotation_regions.py` was written to
prevent, missed in one place. Fixed, and `rebuild_derived.py` regenerates boxes
and COCO from the authoritative masks: **0 empty boxes, 966 annotations.**

### ⚠ Two findings from the annotations that change the experiment

**1. `marking` and `damage` are perfectly class-separable.**

| Class | has `marking` | has `damage` |
|---|---:|---:|
| low | **67** | 0 |
| mid | 0 | 0 |
| high | 0 | **63** |

A rule of `marking→low, damage→high, else→mid` scores 0.978 / 0.159 / 0.108
(mean 0.415). So the stress tests must be **symmetric**: mask the paint stripes
for `low` *and* the damage for `high`. Added `DmgAR` alongside `SAR`.

**2. Frame occupancy alone is the strongest shortcut found so far.**

The tyre fills **72%** of the frame in `low` images, **62%** in `mid`, **61%**
in `high`. A classifier using only `tyre_frac`, `tread_frac` and their ratio
scores 0.181 / 0.455 / **0.968** — **mean 0.535**.

> **The floor to beat is now 0.535, not 0.491.** Framing beats both colour and
> texture. Four probes now exist, each near-perfect on a *different* fold —
> four different shortcuts, four different folds. That is the whole thesis in
> one table.

Consequences: the tyre-region crop is a **control, not an optimisation**, and
area-normalising TER is now measured-necessary rather than a precaution.

**Also settled:** masks are hand-drawn ground truth, not SAM2 pseudo-labels, so
the "unvalidated mask" caveat is gone from `14_XAI_PROTOCOL`. `M_shoulder` is
dropped (empty at this viewpoint). **TER on this dataset measures tyre-vs-
background, not tread-vs-shoulder** — still the sharpest question the data
supports, but worded correctly now.

**Honest gap:** the consistency pass was skipped, so there is **no measured
label-reliability figure**. Recorded as a limitation in `annotations/README.md
§8`. The 30-image subset is still defined if you want to close it later.

**Updated:** `docs/06`, `docs/14`, `scripts/validate_annotations.py`,
`scripts/import_annotations.py`, `scripts/annotation_regions.py`, new
`scripts/rebuild_derived.py`, new `annotations/README.md`.

### 2026-08-27 (d) — Annotation switched to solo

You are doing the annotation alone, so the four-batch split and the
inter-annotator calibration round no longer apply.

**Changed**

| Was | Now |
|---|---|
| 4 batches, one per person | **One `batch_ALL`**, 418 images, session-ordered |
| 50-image calibration, 4 people, before the bulk | **30-image self-check, re-annotated blind at the END** |
| Inter-annotator IoU | **Self-consistency IoU** (pass 1 vs pass 2) |
| ~45–95 min each | **~3.5–4 h, split over 3–4 sittings** |

**Why the self-check still matters, and arguably more.** With four people you
measure whether they agree with each other. Alone, the risk is that *your own*
judgement drifts — by image 300 your eye is much better at finding the tread
boundary than it was at image 10, so the first and last images were effectively
labelled by two different people, and nothing warns you.

So: re-annotate 30 images at the end without looking at the first attempt, and
compare. If `tread` IoU is below 0.90, the later work is the better work and the
fix is to re-annotate the first ~40 images. Twenty minutes, and it is a
reportable quality figure that almost no student project has.

`annotation_agreement.py` now has `--mode self` (default) and `--mode team`.
`prepare_annotation_batches.py` defaults to solo; `--team` restores the split.

**Flagged honestly:** the compute plan still assumes four Kaggle accounts. On
one account Stage A is ~98 h ≈ 3.5 weeks and the whole study ~15 weeks, which
does not fit a semester. Three extra free accounts is by far the cheapest fix —
they need no coordination, since ownership is arithmetic. Recorded as **Open
Decision #6**, to settle before S2 starts.

### 2026-08-27 (c) — Annotation guide rewritten as a Windows click-by-click walkthrough

The previous guide said *what* to annotate but not *how* — no installation, no buttons, no keystrokes. Rewritten end to end, twice.

**First attempt used CVAT Online. It hit a wall:**

```
Could not create the task
You have reached the maximum number of tasks. Upgrade the account to extend the limits.
```

The free tier caps task count and we need five. **Switched to labelme** — `pip install labelme`, runs entirely on Windows, SAM2 built in and offline, no account, no limits, nothing uploaded.

**`docs/15_ANNOTATION_GUIDE.md` now covers**

| Part | Contents |
|---|---|
| Video | [How to Install and Use LabelMe — Step-by-Step for Beginners](https://www.youtube.com/watch?v=PtUO_H3DEc8) |
| A | **Windows install** — Anaconda Prompt, `pip install labelme`, plus a 6-row error table (PyQt5/opencv conflict, missing VC++ redistributable, wrong terminal) |
| B | Shanmukesh's setup — build batches, `labels.txt`, distribute |
| **C** | **One image, click by click** — AI-Points, left-click adds, **Shift+click removes**, `Enter` accepts, pick label. `D` next, `A` previous. Includes the tread-boundary rule with a diagram |
| D | The 50-image calibration round and the 0.90 tread-IoU gate |
| E–H | Import pipeline, file layout, 14-row troubleshooting table, checklists, alternatives comparison |

**Six scripts, all tested end to end on fabricated annotations**

| Script | Does |
|---|---|
| `prepare_annotation_batches.py` | 418 images → 4 session-aligned batches + stratified 50-image calibration set + `labels.txt` + `filename_map.csv` |
| `import_annotations.py` | labelme JSON (or CVAT zips) → indexed masks, YOLO boxes, COCO |
| `validate_annotations.py` | Missing masks, size mismatches, regions outside the tyre, tread ≈ tyre |
| `propagate_annotations.py` | Replays each derivative's geometric transform onto its source mask; writes 20 overlays to check |
| `annotation_agreement.py` | Pairwise IoU per class with the pass/fail gate; reads labelme JSON directly |
| **`annotation_regions.py`** | **Canonical region definitions — new, and it fixes a real bug** |

**A bug the end-to-end test caught.** Masks are a single indexed PNG, so a later class painted over an earlier one *erases* it — `m == 1` means "tyre **minus** whatever is on top of it", not "the tyre". Comparing raw class indices penalised annotators for disagreeing about the *tread* boundary, because the leftover tyre ring changes shape. Two annotators with pixel-identical tyre outlines scored **0.929** instead of 1.000 — a false FAIL that would have sent the team back to redo the calibration round for no reason.

`annotation_regions.py` now defines the regions once (`tyre = m > 0`, `tread = tread | marking`, etc.), has its own selftest reproducing the exact failure, and is imported everywhere. After the fix: tyre IoU 1.000, tread IoU 0.938 reflecting the genuine 4-px disagreement.

**The gate to remember:** inter-annotator **`tread` IoU must exceed 0.90** before anyone starts their main batch.

---

### 2026-08-27 (b) — `tyrelib` written, notebooks NB00–NB05 generated

**`test.ipynb` passed** — all checks green, the pipeline is wired correctly. Moving to the real thing.

**Built `tyrelib/tyrelib.py`** (83 KB) — the whole infrastructure, selftest passing:

| Module | What it does |
|---|---|
| `SharedRateLimiter` | One bucket **per token, process-wide** (⚠ Bug 1) |
| `Uploader` | 30-min batched commits, `(path,size,mtime)` dedup, 429-hint parsing, failed batch returned to buffer, auth failure breaks immediately |
| `Registry` | **One shard per writer**, merged on read, sticky terminal states (⚠ Bug 2); `can_claim` checks owner before freshness (⚠ Bug 3) |
| `assign_workers` | LPT bin packing on a **static** cost table (⚠ Bug 7). Selftest confirms 1.01× imbalance and order-independence |
| `LifecycleGuard` | SIGTERM + SIGINT + atexit + 8.5 h watchdog, fires exactly once |
| `HardwareMonitor` | 10 Hz per-GPU power/energy, 1 Hz util/temp/clocks/throttle + CPU/**peak RAM** |
| `Trainer` | Fixed-epoch loop, tqdm per epoch, full checkpoint contract, mid-run resume with log truncation |
| `Session` | The façade the notebooks call |

**Generated 6 notebooks**, all cells syntax-checked, base64 round-trip verified:

| Notebook | Runs | GPU-h | 4 workers |
|---|---:|---:|---:|
| NB00 Preflight | — | 0.2 | 12 min |
| NB01 Baselines | 3 | 0.5 | 18 min |
| NB02 CNN classic | 36 | 23.6 | 6.1 h |
| NB03 CNN modern | 45 | 22.9 | 5.8 h |
| NB04 Transformer | 54 | 25.5 | 6.4 h |
| NB05 Foundation | 27 | 25.6 | 6.7 h |
| **Stage A** | **162** | **~98** | **~25 h** |

**Three corrections you called out, all applied**

1. **HF accounts — I asked a question already answered.** Your instructions specify one account (`Shanmuk4622`, `HF_TOKEN` in Secrets). Settled: shared budget, `tyrelib` caps each worker at `100/NUM_WORKERS`.
2. **Early stopping removed entirely.** Not softened — deleted. `RECIPE` has no `patience` or `min_epochs`, and the selftest asserts they are absent so it cannot creep back. Every run trains all 60 epochs.
3. **Push cadence tightened** to match the instructions: 30-min background cycle **+** every epoch (metrics/checkpoints enqueued) **+** blocking flush when each model finishes **+** `sess.push_now()` at the end of every important cell **+** emergency flush on Stop/SIGTERM/exception.

**Also added:** live `tqdm` progress bar per epoch showing loss, accuracy and LR; a summary line carrying `val_QWK` and `dataload_frac`; `NB00` Step 5 kills a run for real and compares **post-seam** per-epoch losses (not a shorter-run extension — ⚠ Bug 6).

**Next:** run NB00 on all four accounts, then NB01, then shard NB02–NB05. Annotation (`docs/15`) runs in parallel.

---

### 2026-08-27 (a) — Annotation decided, infrastructure specified, smoke test built

**Decisions**

| Question | Answer |
|---|---|
| **Annotate manually?** | **Yes — but only the 418 clean images.** The other 4,180 are deterministic transforms; masks propagate by replaying the recorded `augmentation_trace_json`. SAM2 proposes, humans correct. ~2 h per person. `docs/15` |
| **HF repo structure** | One **dataset** repo (CSV/Parquet render in-browser). Per-writer registry shards, tiered pushes, checkpoint retention policy → ~39 GB instead of ~150 GB. `docs/16` |
| **What data do we log?** | ~185 columns per epoch — peak RAM, per-GPU power/energy/throttle, `dataload_frac`, `update_to_weight_ratio`, `amp_scale_decreases`, `nan_or_inf_batches`, plus raw 10 Hz energy and 1 Hz system streams. `docs/17` |
| **How long per model?** | **Equal 60-epoch budget for every architecture. NO early stopping.** `val_qwk` selects the best checkpoint but never stops a run |
| **The old engineered pipeline** | Returns as **Tier 8**, built last, with every component chosen from evidence. `docs/13 §6`, `docs/04 §8b` |

**Corrected: the runtime estimate was ~2× optimistic.** Recomputed from measured T4 throughput scaled by FLOPs and resolution:

| | old | **new** |
|---|---:|---:|
| Stage A | 90 GPU-h | **141 GPU-h** |
| Whole study | ~300 GPU-h | **~440 GPU-h** |
| Wall-clock | 2.5 weeks | **~4 weeks, plan for 5** |

**Also:** annotations mean detection and segmentation are now **genuinely supervised**, not pseudo-label distillation — and training both arms (manual vs SAM2) gives a free extra result on what manual annotation actually buys.

**Written:** `docs/15_ANNOTATION_GUIDE.md`, `docs/16_HF_REPO_STRUCTURE.md`, `docs/17_DATA_LOGGING_SCHEMA.md`, `test.ipynb`
**Updated:** `docs/04`, `docs/07`, `docs/13`, `README.md`

**Open — decide before the first run:** four separate HF accounts (4× rate-limit budget, since the limit is per *user*) or one shared account (25 commits/hr each). See Open Decisions #1.

---

### 2026-08-26 (c) — Approach redesigned as a comparative XAI-grounded study

**The change.** Dropped the single-engineered-pipeline plan (SegFormer → ConvNeXt → HRNet → PatchCore → fusion). It assumed labels we do not have and hardware we are not building. Replaced with a **broad comparative study** — many architectures, many techniques, three task paradigms, XAI as the measuring instrument.

**Also settled:** no camera or rig is being built. We work from `final_v1` plus separately-captured video (same below-and-in-front viewpoint, recorded in `docs/02`). Focus is tyre wear; alignment deferred.

**The reasoning.** A naive "we trained 20 models, which won?" benchmark would rank noise — our own probes show a 0.12→0.98 fold swing. So the question became **"which model actually looks at the tread?"** Accuracy is not identifiable on 12 tyres; attribution location is.

**Research done this session**

| Finding | Source | Consequence |
|---|---|---|
| Faithfulness metrics: insertion/deletion AUC, **ROAD**, pointing game; plausibility ≠ faithfulness | Saliency-Bench, XAI reviews | ROAD adopted over raw deletion; both reported |
| **Grad-CAM on ViTs is methodologically ambiguous** — several "Grad-CAM for ViT" papers compute different things | 2026 taxonomy/audit paper | Architecture-appropriate XAI is now a hard rule (`docs/14 §1`). Naive cross-architecture Grad-CAM comparison would have corrupted the headline |
| Shortcut/spurious-correlation literature: detection via XAI, DFR, group robustness | Spurious Correlations survey; ShortcutProbe; ICCV'25 | Framed our stress-test matrix |
| **CAM → SAM prompting** produces pseudo-masks with no annotation | WSOL/WSSS 2025 literature | This is how we do segmentation and detection without labels |
| **YOLO26** released Jan 2026 — NMS-free, ProgLoss, STAL, MuSGD; beats RT-DETRv2-x with fewer params | Ultralytics, arXiv 2510.09653 | Use YOLO26, not YOLO11 |
| Modern CNNs still lead under limited data/compute; EfficientNetV2 fine-tunes well on small sets | timm/model comparisons | ConvNeXt-V2-T and EfficientNetV2-S are the expected front-runners |
| FGVC methods (bilinear, hierarchical bilinear, attention-bilinear) built for subtle within-class texture | FGVC literature | **Tyre wear _is_ an FGVC problem.** No prior tyre application found — most promising model-axis novelty |

**Decided**

- Primary axis is **evidence location**, not accuracy. New metric family: **TER / BAR / SAR / DAR / EDI**, all computable from SAM2 pseudo-masks with zero manual annotation
- **TER must be area-normalised** — raw TER is inflated when the tread fills the frame
- Attribution method is **selected per architecture by faithfulness**, and the choice is reported
- Three hypotheses drafted for pre-registration before S6
- Run budget: ~730 runs, ~300 GPU-h, ~2.5 weeks across 4 accounts
- Infrastructure adopted wholesale from the supplied Replication Playbook — sharded registry, per-token rate limiter, LPT bin packing, static cost table, lifecycle guards, full telemetry

**Written:** `docs/13_EXPERIMENT_PLAN.md`, `docs/14_XAI_PROTOCOL.md`, `docs/02_CAPTURE_AND_PREPROCESSING.md`
**Rewritten:** `README.md`, `docs/04_MODEL.md`, `docs/05_TRAINING_KAGGLE_HF.md`, `docs/06_EVALUATION.md`, `docs/07_ROADMAP.md`
**Superseded:** `docs/02_RIG_BUILD.md` (stub kept; safe to delete)

**Note on alignment.** You suggested it might be the easier half. Flagging respectfully: it is the harder one. Wear is visible in the image; alignment is a geometric quantity needing a calibrated vertical and a known travel direction, neither of which this dataset has. Deferring is right — the reasoning just matters for later planning. `docs/13 §3`.

---

### 2026-08-26 (b) — Dataset README re-read; project-understanding doc written

**Done**

- Re-read the dataset README after it was expanded from 348 → 428 lines
- Wrote **`docs/00_WHAT_THIS_PROJECT_IS.md`** — a plain-language statement of the whole project, for the team and the guide to verify
- Folded the new folder-semantics guidance into `docs/12_DATASET_FINAL_V1.md §1b`
- Added the dataset's own recommended experiment order to `docs/12 §7`

**What the README added**

A new section, *"Important: how to understand the image folders."* It clarifies three things that are genuinely easy to get wrong:

| Level | Meaning | Mistake it prevents |
|---|---|---|
| `clean` / `augmented` | real photos vs artificial variants | Counting derivatives as extra tyres |
| `fold_0/1/2` | **cross-validation group, not a wear category** | Reading `fold_1` as "medium wear" |
| `low/mid/high_mileage_proxy` | class from the original workshop folder | Reading odometer bands as measured wear |

Also confirmed: `1L` in "Tires Gone (1L and Above)" means one lakh ≈ 100,000 km; and the six original folders were collapsed to three **because some contained only one session** — a restraint worth noting in the report.

Nothing in the earlier analysis changes. The counts, splits, probe results and capability matrix all still hold.

**Decided**

- `docs/00` becomes the entry point for anyone new to the repo — including the guide
- Prioritise steps 5 and 7 of the dataset's recommended experiment order (resolution / tyre-region crop, and the shortcut audit), since the probes showed shortcut risk is the live problem

---

### 2026-08-26 (a) — Dataset received, analysed, probed

**Done**

- Read the full `final_v1` package: README, `VERSION.json`, all manifests, splits, config, provenance, reports, scripts
- Viewed both contact sheets, native-resolution centre crops across all three classes, and six full frames
- Wrote **`docs/12_DATASET_FINAL_V1.md`** — complete capability analysis
- Wrote **`scripts/dataset_shortcut_probe.py`** — reproducible difficulty-floor probes
- Ran Probe A (colour) and Probe B (structure); results above
- Created this file

**Found**

| Finding | Consequence |
|---|---|
| 12 sessions, all within a 22-minute window on one day | No lighting/weather/site diversity. Domain robustness is currently unmeasurable |
| `mid` class has exactly one session per fold | For any fold, the validation mid class *is one physical tyre* |
| Three high-mileage sessions have 2, 3 and 5 images | Effectively single observations |
| Colour-only model: 0.952 macro-F1 on fold 0 | Single-fold reporting would be actively misleading |
| Structure-only model wins fold 2, loses fold 0 | Neither probe generalises; both memorise tyres |
| `d20` and `colstd` monotone across classes | The physical wear signal is real, just under-powered |
| Coloured paint stripes + white lettering on new tyres | Direct shortcut for the low class. Crop and ablate |
| Framing varies; some frames crop the shoulders off | Blocks lateral wear profile and any geometry |
| Handheld, uncalibrated, no scale reference | Alignment is impossible from this package — confirms the package's own warning |

**Decided**

- Dataset labels are a **mileage proxy** and will be described as such everywhere — never as `worn` / `not_worn`
- All results reported across **all three folds**, with the Probe A/B baselines in the same table
- Tyre-region crop before classification, treated as an ablation
- `final_v1` is immutable; corrections create `final_v2`
- Data collection is the priority, not modelling

**Updated:** `README.md`, `docs/03_DATA.md`, `docs/04_MODEL.md`, `docs/06_EVALUATION.md`, `docs/07_ROADMAP.md`, `docs/08_RISKS_AND_MY_OPINION.md`, `docs/LOGBOOK.md`

---

### 2026-08-25 — Documentation rebuilt on the correct capture setup

Earlier docs assumed a ground-embedded glass-plate rig. Corrected to the actual setup: a low camera mounted ahead of the wheel facing one tyre. All FTIR/glass-plate/drive-over content removed. Added `docs/10_VISION_TECHNIQUES.md` and `docs/11_APP.md`.

Key research findings: photometric stereo promoted to core (4 LEDs recover surface normals on low-albedo rubber); clDice added for sipe/crack connectivity; SAM2 propagation adopted for ~8× annotation throughput; SSL backbones must be *fully fine-tuned*, not frozen; Depth Anything V2 rejected for metrology. Details in `docs/LOGBOOK.md`.

---

### 2026-08-09 — Repository initialised

Initial scaffold, environment, git. *(Concept superseded 2026-08-25.)*

---

## ⬤ How to update this file

At the end of every working session, add a dated entry to the **Session log** with: what was done, what was found, what was decided, what changed. Then refresh the **status board**, **blockers** and **this week** sections at the top.

Keep the top of the file answering one question: *if someone opened this repo today, what is the state of the project and what happens next?*
