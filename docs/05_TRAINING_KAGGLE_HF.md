# 05 — Experiment Infrastructure: Multi-Account Kaggle + Hugging Face

> **This is the spec for the training infrastructure.** When you ask for code, the `.ipynb` will be built exactly to this. Read it first so we agree on the contract before any code exists.
>
> Built on the Replication Playbook you supplied. Every "⚠ Bug" below was found by running the pattern, not by reading it — you will hit all of them otherwise.
>
> Scale required: **~800 runs, ~440 GPU-h** (`13_EXPERIMENT_PLAN.md §8`), across **4 team members' Kaggle accounts**.

---

## 1. Constraints that shape everything

| Constraint | Consequence |
|---|---|
| Kaggle session dies at ~9–12 h, without warning | Every run checkpoints and resumes. Nothing may depend on finishing in one session |
| `/kaggle/working` is 20 GB | Cannot hold datasets or large intermediates |
| `/kaggle/temp` is ~1 TB, wiped at session end | Perfect staging area — put everything here |
| Weekly GPU quota 30 h/account | 4 accounts ≈ 120 GPU-h/week ⇒ the study is ~4 weeks wall-clock |
| **HF write limit ~128 commits/hour, _per user_** | Not per repo. See ⚠ Bug 1 |
| **HF has no append operation** | Any shared append-only file loses writes. See ⚠ Bug 2 |
| No shared filesystem between accounts | HF Hub is the only coordination substrate |
| No locking primitive on HF Hub | Coordination must be lock-free |
| GPU: 2× T4, no NVLink | fp16 not bf16; DDP over PCIe |

**Design goal:** if a session dies at any moment — timeout, crash, or you pressing Stop — you lose at most 30 minutes of compute and **zero** metadata.

---

## 2. Architecture

```
 acct 1 (WORKER 0)   acct 2 (WORKER 1)   acct 3 (WORKER 2)   acct 4 (WORKER 3)
 ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
 │ bootstrap    │    │ bootstrap    │    │ bootstrap    │    │ bootstrap    │
 │ lib from b64 │    │              │    │              │    │              │
 │      ↓       │    │              │    │              │    │              │
 │ plan_work()  │    │ plan_work()  │    │ plan_work()  │    │ plan_work()  │
 │ → my slice   │    │ → my slice   │    │ → my slice   │    │ → my slice   │
 │      ↓       │    │              │    │              │    │              │
 │ /kaggle/temp │    │              │    │              │    │              │
 │  staging     │    │              │    │              │    │              │
 │      ↓       │    │              │    │              │    │              │
 │ uploader     │    │              │    │              │    │              │
 │ 30-min batch │    │              │    │              │    │              │
 └──────┬───────┘    └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
        └───────────────────┴───────────────────┴───────────────────┘
                                    ↓
                    ┌────────────────────────────────┐
                    │  HF  Shanmuk4622/tyrevision    │
                    │        (ONE dataset repo)      │
                    │  runs/{run_id}/…               │
                    │  registry/events/*.jsonl       │ ← one shard per writer
                    │  tables/  analysis/  paper/    │
                    └────────────────────────────────┘
```

> **The single most important idea: workers never talk to each other.** Ownership is decided by arithmetic every worker computes independently; progress is read from HF. No locks, no leader, no messages, nothing to deadlock.

---

## 3. Work sharding across 4 accounts

Every worker runs the **same notebook**. Parallelism has one source of truth:

```python
ACTIVE_KAGGLE_ACCOUNTS = ("acct1",)  # one notebook; list all four for four copies
ACCOUNT = "acct1"                     # this copy's label
NUM_WORKERS = len(ACTIVE_KAGGLE_ACCOUNTS)
WORKER_ID = ACTIVE_KAGGLE_ACCOUNTS.index(ACCOUNT)
```

### Sharding reserves fresh work; HF decides truth

This is the distinction that Bug 9 (§7) came from getting wrong, so state it
plainly:

| | source | changes with `NUM_WORKERS`? |
|---|---|---|
| **who owns a fresh absent run** | `assign_workers` — LPT on the static cost table | yes, and that is fine |
| **is this run finished** | `runs/<id>/STATUS.json` on HuggingFace | **no** |
| **what epoch did it reach** | the same file | **no** |
| **is someone on it right now** | registry heartbeat, ≤45 min old | no |

Ownership reserves work; it is never a claim about truth. A worker starts with
its own static shard and resumes partial runs from HF. Work stealing is
**opt-in and disabled in NB06 v5**. An explicit recovery run may take over
somebody else's job only when HF contains a real claim/run event older than 45
minutes. Absence is not staleness. Halving the worker count therefore changes
the owner map, while HF still determines the facts: finished runs are skipped
and half-done runs are resumed.

The test that pins this down (`selftest`):

```python
states = {nw: {r: inv.state(r) for r in runs} for nw in (1, 2, 4)}
assert states[1] == states[2] == states[4]
```

### Do not use hashing for this

Hash-based ownership is elegant for large open-ended universes. Ours is ~800 runs with **6× cost variation** (a MobileNet epoch vs a ViT-B epoch). A phase ends when the **slowest** worker ends, so imbalance multiplies wall-clock directly. Measured on a comparable 45-job/6-worker split:

| Mode | Counts | Imbalance |
|---|---|---|
| hash | `[11, 7, 4, 10, 3, 10]` | 4.91× |
| balanced (round-robin) | `[8, 8, 8, 7, 7, 7]` | 1.31× |
| **cost (greedy LPT)** | `[7, 7, 7, 8, 8, 8]` | **1.02×** |

### Use longest-processing-time-first bin packing

```python
def assign_workers(run_ids, n, costs):
    run_ids = sorted(run_ids)                      # canonical order everywhere
    jobs = sorted(run_ids, key=lambda r: -costs[r])
    load, out = [0.0]*n, {}
    for r in jobs:
        w = int(np.argmin(load)); out[r] = w; load[w] += costs[r]
    return out
```

Deterministic, zero-communication — every worker computes the same packing from the same inputs.

### ⚠ Bug 7 — measurement must never feed back into allocation

**Symptom:** one job abandoned mid-run, another trained twice by two workers.
**Cause:** the scheduler refined its cost table from measured runtimes, so a worker planning early computed a *different* assignment than one planning later. "Identical input" quietly stopped being identical.

**Fix: ownership uses a static cost table, always.** Measurements refine only what you *print*.

```python
measured = estimate_costs_from_history(data_dir)   # median sec/epoch per arch
display_costs = {**STATIC_HINTS, **measured}       # for the printed plan ONLY
owner = assign_workers(universe, N, STATIC_HINTS)  # never `display_costs`
```

**Calibrate the static table once, early.** A first guess predicted 1.73 h for a run that took 2.89 h — a 40% underestimate. Two measured runs fixed both scale and ratio. Anchor any printed "this will take N hours" on a measurement.

### Optional work stealing as a recovery tool

The default is `steal_stale=False`. Only a deliberate recovery launch should
enable it after confirming the old workers are stopped. It can then pick up
runs whose real HF event is stale (>45 min). A run with no event remains
reserved for its static owner:

```python
if steal_stale:
    for r in universe:
        if r in done or owner[r] == me: continue
        st = latest.get(r)
        if st is not None and age(st) >= STALE_SEC:
            stolen.append(r)
```

**Own work always first.** Two live workers then never fight.

**Print the plan before committing hours to it.** A shard report with estimated hours per worker catches a bad split on minute one instead of day three.

---

## 4. The run registry

You need to know, across all four accounts: what exists, what is running, what finished, who owns what.

### ⚠ Bug 2 — lost updates on a shared ledger

**Symptom:** two runs training, the ledger lists one. No error — the file just forgets.
**Cause:** HF has no append. Every worker appending to a shared `runs.jsonl` and pushing means **the last push silently destroys every other worker's lines.**
**Cost:** work planning reads *completion* from the ledger, so a lost `completed` entry makes a finished 3-hour run look unfinished and someone retrains it.

**Fix: shard by writer.**

```
registry/events/{account}_w{worker_id}_{session_id}.jsonl
```

Each writer owns one file nobody else touches. Reads merge every shard.

```python
def latest(self):
    st = {}
    for e in self.entries():                       # merged, sorted by float ts
        rid = e["run_id"]
        # 'completed' is STICKY: a late heartbeat from a stale shard must not
        # resurrect a finished run, or it gets trained twice.
        if st.get(rid, {}).get("state") == "completed" and e["state"] != "completed":
            continue
        st[rid] = e
    return st
```

Two details that matter:

- **Store a float epoch timestamp**, not just ISO. Second granularity makes same-second events across shards sort ambiguously — exactly where ordering must be trustworthy.
- **Terminal states are sticky.** Otherwise out-of-order pushes resurrect finished work.

### ⚠ Bug 3 — the claim protocol blocking self-resume

**Symptom:** `held by acct1 (5 min ago)` — refusing to resume *your own* run.
**Fix:** check owner **before** freshness. Same account → always allowed.
**Lesson:** the most common case (my session died, this is the new one) must be the easy path.

---

## 5. The batched uploader

**Problem:** per-file pushes blow the rate limit; synchronous pushes block training.
**Solution:** one background thread, one buffer keyed by repo path, one commit per cycle.

```python
class BackgroundUploader:
    BATCH_INTERVAL_SEC = 1800                      # 30 min
    def enqueue(self, local_path, repo_path):
        fp = f"{repo_path}|{size}|{mtime}"
        if fp in self._pushed: return False         # unchanged file, skip
        self._buffer[repo_path] = ...               # newer supersedes pending
    def _loop(self):
        while not stop:
            self._wakeup.wait(timeout=self.BATCH_INTERVAL_SEC)
            batch, self._buffer = list(self._buffer.values()), {}
            self._rate_limiter.wait_for_slot()
            api.create_commit(operations=[CommitOperationAdd(...) for f in batch])
```

Four properties that matter:

- **Buffer keyed by repo path** — a rolling checkpoint enqueued five times in one window produces one file in one commit, not five
- **Fingerprint dedup on `(path, size, mtime)`** — config files never change; re-enqueueing is free
- **Failed batch returns to the buffer** via `setdefault`, so a newer version that arrived meanwhile is not clobbered
- **A failed push never kills training** — log it, retry next cycle

### 429 handling — parse the hint, don't guess

HF's response body carries a human-readable retry hint. Parsing it beats blind exponential backoff, which either wastes a window or hammers early.

```python
m = re.search(r"retry after (\d+)\s*second", err, re.I)
if m: return float(m.group(1)) + 2.0
m = re.search(r"in about (\d+)\s*minute", err, re.I)
if m: return float(m.group(1)) * 60.0 + 5.0
```

**Auth failures must break immediately**, not retry eight times. A read-only token will never become writable.

---

## 6. Rate limiting

### ⚠ Bug 1 — the limiter on the wrong object

**The write limit is per _user_, not per repository.** If the limiter lives on the uploader object, N repos multiply your budget by N. In the reference build: two repos at 20 commits/hour each = 40/hour per account; six accounts = **240/hour against a real ceiling near 128.** The cap was decorative.

**Fix: one bucket per token, process-wide.**

```python
class _SharedRateLimiter:
    _buckets = {}                                   # token hash -> bucket
    @classmethod
    def for_token(cls, token, limit):
        key = hashlib.sha256((token or "anon").encode()).hexdigest()[:16]
        with cls._registry_lock:
            b = cls._buckets.setdefault(key, cls(limit))
            b.limit = min(b.limit, limit)           # most conservative wins
            return b
```

### Our budget arithmetic — settled

**One HuggingFace account, `Shanmuk4622`, token in Kaggle Secrets as `HF_TOKEN`.** All four Kaggle accounts use it, so the 128/hr budget is shared by the whole team.

```python
rate_limit = max(6, int(100 / num_workers))     # tyrelib.Session does this for you
# NUM_WORKERS=4 -> 25/hr each -> 100/hr team total, 28 ops of headroom
```

At a 30-minute cycle, one 9-hour session makes ~18 scheduled commits plus a few event-driven ones. Comfortably inside.

**When the cap is hit, sleep — don't fail.** Training continues; only the uploader waits.

---

## 7. Resumability

### ⚠ Bug 8 — the checkpoint was on HuggingFace and nothing ever fetched it

**This one cost ten hours of T4 time before it was noticed.**

`Trainer.try_resume` began:

```python
if not self.ckpt_last.exists():
    return False
```

`ckpt_last` is a path on the session disk. Kaggle wipes the session disk when
the session ends. So in every fresh session that file was absent, `try_resume`
returned `False` without asking anything else, and the run started again at
epoch 1 — of a run that had reached epoch 47 and whose checkpoint had been
pushed to HuggingFace every single epoch, exactly as designed.

Everything around it worked. Checkpoints were saved atomically, pushed on
schedule, verified present in the repository. The one missing line was the one
that brought them **back**.

The symptom looked like a sharding bug, because it showed up as "I changed
`NUM_WORKERS` from 4 to 1 and it started training everything again". It was
not. It restarted at every session boundary regardless of worker count; the
worker change was just when it got noticed.

**Fix:** `RemoteInventory.fetch_run()` downloads `ckpt_last.pt`, `ckpt_best.pt`
and `metrics/epochs.csv` before `try_resume` looks at the disk.

> A resume path that has only ever been tested inside one session has not been
> tested. The seam it has to survive is the session boundary, and that is the
> one place a single-session test never goes.

### ⚠ Bug 9 — planning read intentions instead of facts

Work planning read the registry: who claimed what, on which account, under
which `WORKER_ID`. Every one of those is relative to a session.

* Change `NUM_WORKERS` and the ownership arithmetic reshuffles.
* Run on a different account and `can_claim` no longer recognises a run as yours.
* Lose a shard and a finished run looks unfinished.

And the fatal one: **`failed` was treated as "not done, start over".** A run
that raised at epoch 47 has a checkpoint at epoch 47. It is exactly as
resumable as one the watchdog paused. Twenty-six runs were marked `failed`,
each with a good `best_qwk` recorded, and every one of them was queued for a
fresh start.

**Fix:** a run's state now comes from `runs/<id>/STATUS.json` in the
repository — a file that says the same thing to every account at every worker
count. There are three states and `failed` is not one of them:

| state | meaning |
|---|---|
| `completed` | skip |
| `resumable` | a checkpoint exists — continue from its epoch, whatever ended it |
| `absent` | never started |

`NUM_WORKERS` and `WORKER_ID` now decide **only what this account starts
first**. That is what makes them safe to change mid-study.

### ⚠ Bug 10 — staging fell into the 20 GB output quota

```python
base = Path("/kaggle/temp") if Path("/kaggle/temp").exists() else Path("./_work")
```

On the current Kaggle image `/kaggle/temp` does not exist until something
creates it. So every session took the fallback and staged into `./_work`,
which is inside `/kaggle/working` — capped at 20 GB, and that cap is the size
of your **output**, not your scratch.

A `vgg16bn` checkpoint is ~1.6 GB and we keep two per run. Nine vgg runs is
29 GB. The session hits the quota partway through, the write raises, and the
run is recorded as `failed` — with no record of *why*, because the exception
type was not stored. Which is the likeliest explanation for the 26.

**Fix:** create `/kaggle/temp` rather than testing for it, print free space at
session start, warn under 20 GB, record `ERROR.json` with the exception type
and the free space at the moment of failure, and delete a finished run's local
checkpoints once the repository confirms it has them.

### ⚠ Bug 12 — the observer killed the thing it was observing

```
ValueError: Length of values (35249) does not match length of index (35250)
```

Two runs died on that, after 43 and 66 minutes of training. It is
`HardwareMonitor.dump()`:

```python
with self._lock:
    pd.DataFrame(self.energy_rows).to_csv(...)     # lock held...
```

The lock was held. It bought nothing, because the sampler thread's
`self.energy_rows.append(...)` **was not inside the lock**. `pd.DataFrame` walks
the list while building columns, the 10 Hz sampler added one more row midway,
and the last column came out one element short.

Reproduced in the selftest — the unlocked pattern raises the identical error
within 400 attempts; the fixed one survives.

Two changes, and the second matters more:

1. Copy the buffers under the lock, build the DataFrames outside it. Correct,
   and a slow gzip write no longer stalls the sampler.
2. **`dump()` and `window()` never raise.** Telemetry is an observer. An
   observer that can kill a three-hour training run is a liability however
   good its data is. A lost power trace is a nuisance; a lost run is not.

> Any background thread that writes what a foreground thread produces needs
> this treatment. Check the *append* side, not just the read side — that is
> where this hid.

### ⚠ Bug 13 — a claim nobody can read is not a claim

`a-vgg16bn-base-f1-s1` was trained to completion by **acct1 and acct2**. Same
`config_hash`, same result, ~1.4 GPU-hours spent twice. It is only visible if
you notice one `run_id` has two owners — which is why `aggregate_remote()` now
warns on duplicated run ids.

Two compounding causes:

* `can_claim` reads the **local** copy of other workers' registry shards, last
  downloaded during `sync_state` — hours earlier. A run another account started
  twenty minutes ago still looked idle.
* `registry.emit(rid, "claimed", ...)` only *enqueues*. The background cycle is
  30 minutes, so for half an hour the claim existed nowhere anyone could read.

Both workers were behaving correctly on the information available to them.

**Fix:** before starting a **stolen** run (never for its own — nobody else can
be on those), re-pull the registry, then flush the claim immediately. Two extra
requests, paid only at the moment they buy something.

### ⚠ Bug 14 — two bugs whose only symptom was a report that could never pass

`confirm_on_hf` used the presence of `runs/<id>/summary.json` as its completion
test. `enqueue_light()` never enqueued `summary.json`. So the file was never on
HuggingFace, and all 36 finished runs were reported `RESUMABLE`.

Neither half is visible alone. The upload gap shows up only as a report that is
always pessimistic; the report bug shows up only as a file nobody misses.

**Fix:** upload `summary.json` and `split_health.json`, and judge completion the
way everything else does — `STATUS.json`'s status field, through
`RemoteInventory` — so there is one definition of "finished" in the codebase
instead of two.

### ⚠ Bug 15 — the preflight found it, said so, and was ignored

All 18 `dinov2` runs in NB05 failed on their first batch:

```
AssertionError: Input height (392) doesn't match model (518).
```

`build_model` never told timm what resolution the images would be. Most models
do not care; `vit_*_patch14_dinov2` is created at `img_size=518` and its patch
embedding asserts an exact match. Note *where* it failed — in `forward`, not in
`create_model`. The old fallback-to-resnet18 `except` wrapped only construction,
so it never fired, and the failure surfaced as a training crash rather than as
"this architecture cannot take this input".

**The part that matters is not the bug. It is that NB00 had already found it.**
On 26 August the preflight printed:

```
  FAIL dinov2_s   AssertionError: Input height (392) doesn't match model (518).
  FAIL dinov2_b   AssertionError: Input height (392) doesn't match model (518).
17/19 architectures build, forward and backprop.
Exclude the failures from the sweep, or fix them, BEFORE Stage A.
```

then reported `architectures OK : 17/19` in a summary block where every other
line read PASS, and returned success. The same run printed
`resume equivalence : FAIL` — Bug 8 — and that was ignored too.

> **A preflight that reports but does not block is not a preflight.** It is one
> more line competing with nineteen others. Two accurate diagnoses, printed
> two days early, cost 18 runs and a four-account session because nothing
> stopped.

Fixes, in order of how much they are worth:

1. **NB00 raises.** On any architecture failure, and on resume-equivalence
   failure. Excluding an architecture is now a decision recorded in `ZOO`, not
   a warning nobody scrolled back to.
2. **Every Stage notebook calls `tl.assert_zoo_ok(ARCHS)`** before planning.
   Fifteen seconds, no pretrained download, one forward per architecture at the
   resolution it will actually receive.
3. `build_model` takes `img_size`, tries `img_size` / `dynamic_img_size` /
   plain in that order, and **verifies each candidate with a real forward pass**
   before returning. The cascade is driven by the requirement — can this model
   forward at this size — rather than by whether construction happened to
   succeed.

### ⚠ Bug 16 — a completed run id can still name the wrong model

NB07 later tried to reconstruct `a-convnextv2_s-base-f1-s1`. The registered
`convnextv2_small.fcmae_ft_in22k_in1k` pretrained tag does not exist in timm.
The public checkpoint then showed what the training log had hidden: its config
declared ConvNeXt-V2-S, but the tensors were ResNet-18. All nine statuses for
that arm report the same ResNet-18 parameter count (`11,177,538`). The old
construction exception handler had silently substituted an emergency model,
so nine 60-epoch runs completed under false architecture labels.

**Fix:** there is no model fallback of any kind. If timm is unavailable or a
requested pretrained source does not exist, construction raises before
training. Checkpoint consumers instantiate the untagged topology because the
checkpoint supplies its own weights, then compare known tensor signatures and
strictly load the state. NB07 publishes a structured architecture-mismatch
exclusion and continues; it never relabels the ResNet-18 evidence as
ConvNeXt-V2-S. NB03 no longer schedules the unsupported pretrained Small arm.

### ⚠ Bug 17 — GPU headroom does not prevent a host-RAM kernel kill

The first NB06 ROI pass completed three DenseNet seeds and one ResNet seed,
then every worker died at the start of RegNet with only Kaggle's “Kernel
Restarting” message. This bypassed Python exception handling and the emergency
flush, so diagnosis had to come from public HF telemetry.

The evidence rules out GPU OOM: DenseNet peaked near 5.0 GB and ResNet near
4.1 GB on each 16 GB T4. Instead, the ROI process RSS grew almost linearly from
about 3.3 GB at epoch 1 to 20.3–20.7 GB at epoch 60. The next sequential
ResNet run inherited that pressure and ended at 27.4 GB process RSS / 31.1 GB
system RAM / 94.9%. Matched Stage-A full-frame runs stayed near 3 GB. The old
ROI path decoded a full mask, formed `mask > 0`, then materialised two int64
coordinate arrays for every tyre pixel; persistent workers and pinned batch
buffers retained the resulting host allocations.

**Fix (`2026-08-31-r1`, introduced in tyrelib v3 and retained in v5):** use the mask image's non-zero bounding
box directly, close both image files immediately, and preserve the old
max-minus-min padding exactly. ROI alone uses `num_workers=0` and
`pin_memory=False`; normal full-frame arms keep the proven Stage-A loader.
Persistent loaders are explicitly shut down, model/optimizer/CUDA caches are
released between runs, and an 88% host-RAM guard finishes the current epoch,
writes the checkpoint, publishes it, and returns `paused` before the OS can
kill the kernel. Scientific settings and the locked model set are unchanged.

That was the first repair. The later full Stage-B audit in Bugs 19–20 showed
that the ROI diagnosis was incomplete: standard full-frame arms could also
accumulate host RSS in the same long-lived process.

### ⚠ Bug 18 — a CUDA fault is not an OOM, and absence is not staleness

After the host-RAM repair, two independent workers reached the actual first
RegNetY-16GF ROI batch. Public `ERROR.txt` files for seeds 1 and 2 show the same
failure inside RegNet stage `s2` under `DataParallel`: one reports
`CUDNN_STATUS_EXECUTION_FAILED`, the other `CUDA error: misaligned address`.
Both used Kaggle's Python 3.12.13, PyTorch 2.10.0+cu128, CUDA 12.8, timm 1.0.26
and two Tesla T4s. Telemetry sampled only ~1.1 GB per card and ~2.6 GB host RAM,
so neither GPU nor host capacity was the cause. The repeated stack and the
successful Stage-A controls make the T4 cuDNN grouped-convolution path selected
by AMP + `channels_last` + autotune the supported diagnosis.

**GPU fix (`2026-08-31-r1`, retained in tyrelib v6):** RegNet alone uses contiguous NCHW
and `cudnn.benchmark=False`. It remains the exact NB07-locked `regnety_016`
model at 384px, batch 32, FP16/GradScaler, AdamW and 60 epochs. All other models
retain the Stage-A `channels_last` path. Epoch rows, `summary.json`,
`STATUS.json` and `ERROR.json` record the actual layout and safety revision.
A launch fault marks `cuda_restart_required`, publishes the error, suppresses
the misleading second exception from `empty_cache`, and stops before a poisoned
CUDA context can fail unrelated models.

NB06 also proves the repaired profile before claiming work: a disposable child
process performs an exact batch-32, 384px, dual-T4 RegNet forward, backward and
AdamW step, then publishes its account-specific log under
`preflight/cuda_profiles/`. A failure poisons only the child and blocks the
training plan; a pass prints `CUDA_SMOKE_PASS`.

The attached account-1 output exposed a second problem at the same time: its
plan called all five outstanding ROI runs “picked up from a dead worker” even
though they were fresh jobs statically owned by accounts 2--4. With no registry
event, `can_claim` returned “unclaimed”; simultaneous notebooks could therefore
race before the first claim commit became visible. Public seed 1 was in fact
attempted by its rightful account while account 1 also launched it.

**Scheduler fix:** a fresh absent run stays with its static owner. Work stealing
is now disabled by default. If a recovery session explicitly enables it, only a
real event older than 45 minutes is eligible; a recent `failed` or `paused`
event is protected too, while the same account may retry immediately.

### ⚠ Bug 19 — claim commits and pause cascades exhausted HF while the GPUs idled

The executed NB06 output shows the exact chain. `run_all()` forced an immediate
HF commit for every ordinary static-owner `claimed` event. Separately, after a
run returned `pause_reason=host_ram_guard`, the outer loop started the next
model unless the 8.5-hour watchdog was also near its limit. One high-RAM process
therefore produced dozens of one-to-three-epoch partial runs, and every partial
run produced both a pause commit and the next claim commit. The session reached
45 commits, spent its 25/hour worker allowance, and stopped at:

```text
[HF] flush (claim b-regnety016-wd_low-f1-s2): 1 file(s)
[RATE] budget spent (25/hr); sleeping 197s
```

This wait was scheduler I/O, not training and not a slow GPU.

**Fix (`SCHEDULER_SAFETY_REVISION=2026-08-31-r2`):** NB06 explicitly runs with
`steal_stale=False`. A normal claim is only enqueued and is folded into the
30-minute/major-step batch. Only a genuinely stolen job forces an immediate
claim commit, because only that claim coordinates two possible owners. Any
`paused` result now ends the worker's training cell after the trainer publishes
its checkpoint. Re-run in a fresh Kaggle session; HF resumes the same run at the
next epoch.

### ⚠ Bug 20 — the ROI-only memory diagnosis was incomplete

The expanded public audit now contains 65 Stage-B statuses: **12 completed and
53 paused**, and all 53 pauses report `host_ram_guard`. Those pauses include
ordinary full-frame arms using the standard loader, so dense ROI coordinates
could not be the only source. Epoch telemetry also shows process RSS rising
across otherwise normal runs while VRAM remains well below T4 capacity.

The long-lived process performed a full `torch.save` for both `ckpt_best` and
`ckpt_last` on every improving epoch, and HF/LFS handled the same large files
repeatedly. Freed serialization/upload arenas can remain mapped in glibc and
inflate process RSS until process exit. That matches the reset seen between
fresh Kaggle sessions and gives a direct repair target without changing a
model, batch, image, optimiser or epoch budget.

**Fix (`MEMORY_SAFETY_REVISION=2026-08-31-r2`, tyrelib v5 and retained in v6):** serialize the
complete state once per epoch; on an improving epoch atomically hard-link/copy
that exact file to `ckpt_best`. After checkpoint save/load, HF commit and final
model cleanup, collect unreachable objects and call Linux `malloc_trim(0)` to
return already-free arenas to the host. The 88% guard remains as the fail-safe.
Epoch rows and summaries record both the memory and commit-policy revisions so
the v5 Kaggle result can be distinguished from all prior partial runs.

Public verification at this repair point: **12/108 complete, 53 resumable; all
65 status-bearing Stage-B runs have `ckpt_last.pt` and `ckpt_best.pt`.** No
published epoch needs to be retrained.

### ⚠ Bug 21 — telemetry schema drift made a valid checkpoint look unresumable

Tyrelib v5 added `runtime_hf_commit_policy_revision` immediately after
`runtime_memory_safety_revision`, but the epoch writer still used positional
CSV append. Two DenseNet runs already had 177-column v4 headers. Their first
three rows matched that header; v5 then appended 178 values per row. The
checkpoint reached epoch 60, but resume stopped before finalisation with:

```text
ParserError: Error tokenizing data. C error:
Expected 177 fields in line 5, saw 178
```

The model, optimiser, RNG and both checkpoints were intact. Only the telemetry
table's header was stale. A public audit of all 70 Stage-B epoch files found
exactly two mixed-width files; 68 were internally consistent.

**Fix (`EPOCH_HISTORY_SCHEMA_REVISION=2026-09-01-r1`, tyrelib v6):** parse with
Python's CSV reader, recognise the known revision token at the exact insertion
position, add the missing heading, pad only the older rows, validate every row
width, and atomically rewrite one canonical table. Future writes read the
existing table, union columns by name, de-duplicate the epoch key and atomically
replace the file. Unknown drift raises and leaves the source untouched instead
of guessing or dropping a row.

The repair was run against the exact public 60-epoch failure file: all epochs
1–60, all 178 columns and the shifted CUDA/validation fields were preserved.
An epoch-60 checkpoint whose status still says `running` is now described and
finalised as completed metadata without training an impossible epoch 61.

Public state on the 2026-09-03 follow-up: **42/108 completed, 34 checkpointed
incomplete, 32 absent; all 76 status-bearing runs have both checkpoints.**

### ⚠ Bug 22 — a guard that ended the session over a two-second spike

Reported as *"sometimes the workers are stopping the notebooks by themselves."*

The host-RAM guard read `ram_percent_peak` — the **maximum** of the 1 Hz samples
taken during the epoch — and compared it against 88%. Above that, the run
paused; and since v5, *any* pause stopped the entire worker.

The measured trajectory of `b-densenet121-res512-f1-s3`, from public telemetry:

| epoch | `ram_percent_peak` | RSS GB | |
|---:|---:|---:|---|
| 1 | **95.2** | 28.4 | pauses, and stops the worker |
| 2 | **92.8** | 27.8 | |
| 3 | 76.7 | 22.5 | |
| … | | | |
| 10 | **17.8** | 3.5 | (next session) |
| 26 | 46.3 | 12.1 | |

Mean peak after epoch 5: **42.9%**. The 95.2% was the first epoch of a session —
the annotation rebuild, the loader warm-up and the resume checkpoint load, all
released within two epochs. One transient ended an eight-hour session with
eighteen runs untouched.

A peak answers *"did we ever come close?"*. Before starting another epoch the
question is *"is there room now?"*.

**Fix (`RAM_GUARD_REVISION=2026-09-01-r1`, tyrelib v7):**

* `host_ram_percent()` reads the live value **after** `release_host_memory()`
  has returned freed arenas to the kernel — the checkpoint that was just
  serialised and handed to the uploader is exactly the spike that used to trip
  this, and it is gone by the time the next epoch starts.
* A transient peak now prints `transient, continuing` instead of pausing.
* Hysteresis: pause at 88%, resume below **80%**.
* **A recovered RAM pause no longer ends the cell.** `run_all` releases the
  finished model, re-measures, and continues to the next run if the pressure
  was the model rather than the session. A watchdog pause or an interrupt still
  stops — those genuinely mean there is no time left.

### ⚠ Bug 23 — the telemetry buffers were the leak

Bug 22 explains why one spike stopped a session. It does not explain the slope:
RSS climbed **+0.54 GB per epoch**, 3.5 GB → 28 GB over one run, and Kaggle
kills that kernel with no Python exception to catch.

`HardwareMonitor.dump()` snapshotted both sample buffers, wrote them out in
full, and **never cleared them**. At 10 Hz per GPU a four-hour run accumulates
~300,000 dicts, and every dump rebuilt a DataFrame over all of them. Same for
`step_traces`, which was rewritten whole with `open(..., "w")`.

**Fix (tyrelib v7):** each dump writes only the rows added since the last one
and drops them. Concatenated gzip members are a valid gzip stream, so the file
still reads back as one table with `pd.read_csv` while the process holds at
most one dump-interval of samples. Telemetry now flushes **every** epoch rather
than every ten — cheap once it is incremental, and a hard kill loses one epoch
of trace instead of nine. `step_traces` is capped per epoch, appended, and the
finaliser appends instead of truncating what the per-epoch flush already wrote.

Verified: 6 × 500 rows in, buffer empty after each dump, 3,000 rows read back
with columns intact and no interior header rows.

> Two bugs, one symptom. The guard turned a slow leak into an abrupt stop, and
> because the guard fired on a peak it also fired when there was no leak at all.
> Fixing only the guard would have hidden the leak until it hit 88% for real.

### ⚠ Bug 24 — two workers running, two parked, and the fix for that was the cause

Reported as *"out of 4, 2 are running and 2 stopped"*.

NB06 called `run_all(..., steal_stale=False)`. That put every run owned by
another account into `busy` **permanently** — not "busy until they finish",
just unreachable. So a worker that got through its 27-run static shard printed

```
  reserved for other static owners: 68
  -> will run 0 run(s) this session
```

and the notebook ended normally, while the other accounts still had twenty runs
each. The shard is LPT-balanced on *estimated* cost and skewed further by pauses
and resumes, so shards always finish at different times. **Some worker always
runs dry first.** It was not a crash; it was a scheduler with no idea what to
do next.

And `steal_stale=False` was itself a fix — for Bug 13, where aggressive stealing
trained `a-vgg16bn-base-f1-s1` twice on two accounts. The two failure modes sit
either side of the same question: *may I take work that is not mine?* "Always"
duplicates; "never" idles.

**The answer is "yes, once I have nothing of my own, and only after checking
properly."** `claim_or_yield` is a two-phase claim, which is the standard
protocol when there is nowhere to put a lock:

1. Pull the registry, confirm nobody holds it, write the claim, and **flush it
   immediately** so every other worker can see it.
2. Wait out the race window (25 s + jitter), pull again, and look at every claim
   written for that run in the window. If more than one account claimed it, the
   **lowest account name wins**.

Both sides compute step 2 from the same bytes and get the same answer, so
exactly one proceeds. Cost: one commit and ~30 s, paid only by a worker that
would otherwise be parked.

Three further guards:

* **Own work always first.** `plan.order = mine + pool`, and the pool is only
  reached past `plan.n_mine`. At a simultaneous cold start every worker has its
  own shard to do, so nobody touches the pool and Bug 13 cannot recur.
* **Each worker enters the pool at a different offset** (`worker_id % len`), so
  two workers going idle together do not even reach for the same run before the
  protocol has to arbitrate.
* **No takeover inside the last 90 minutes** of a session — better to stop
  cleanly than half-train a model.

Verified with four threads racing for six runs on a shared registry: **exactly
one winner each**, with both phases observed — two blocked at step 1, and one
account that won step 1 correctly yielding at step 2.

> A scheduler needs a rule for the idle case, not just the contended one. Both
> bugs here came from answering only the half of the question that had recently
> hurt.

### ⚠ Bug 25 — the guard was watching the wrong machine's memory

```
[RAM] host RAM 89.6% still above 88% after releasing (epoch peak 89.9%)
```

`psutil.virtual_memory()` reads `/proc/meminfo`. Inside a container that
reports the **host's** memory, not the cgroup limit the kernel enforces on us.
So the percentage every pause decision was made on did not describe our budget
at all — on a busy host it can sit near 90% no matter what the notebook does,
and it would equally miss a container that is genuinely about to be killed.

**Fix:** `container_memory()` reads `/sys/fs/cgroup/memory.current` and
`memory.max` (v2), or `memory.usage_in_bytes` / `memory.limit_in_bytes` (v1),
and only falls back to psutil when neither exists. That is the number Kaggle's
OOM killer actually uses. The source is printed, so you can see which one you
got.

A pause now says **where** the memory is, because "89.6%" is not something
anyone can act on:

```
[RAM] 89.6% of 32 GB [cgroup:v2] after releasing (epoch peak 89.9%)
      -- this process 4.1 GB, 2 child proc 21.8 GB, rest 2.2 GB
```

### ⚠ Bug 26 — two loader workers that did nothing, at 21 GB

The `wd_low` run that paused at epoch 36 had logged **`dl 0%` on every one of
its 49 epochs**. `dataload_frac` was zero, forty-nine times in a row, at 4.2
minutes per epoch. The GPU is the bottleneck by a wide margin.

And yet:

```
[LOADER] workers=2 pin_memory=True (standard path)
```

Two forked worker processes, whose RSS counts against the same cgroup, plus
PyTorch's pinned-host allocator, which caches its buffers and does not return
them. Buying nothing — the measurement had been saying so for hours.

The ROI arms were moved to the synchronous loader when *their* RAM climbed
3 → 20 GB. The full-frame arms were left on two pinned workers because the
diagnosis was framed as "an ROI problem". It was a loader problem; ROI just
reached it first.

**Fix:** `dataloading_is_free(cfg)` — a named, measured rule rather than a bare
`nw = 0`. At ≥320 px the epoch is GPU-bound and gets no workers and no pinned
memory. Smaller, faster configurations keep them, because that is where
prefetching can genuinely matter. If a future arm is loader-bound its
`dataload_frac` will say so and it can be given workers back deliberately.

> Both of these were visible in data we had already been collecting for days.
> `dl 0%` was printed on every epoch line, and the memory percentage came from
> a source nobody had checked. Recording a metric is not the same as reading it.

### ⚠ Bug 27 — “one worker” still printed `worker=0/4`

The submitted NB06 output settled this without guesswork:

```
[HF] rate cap 25/hr
[SESSION] account=acct1  worker=0/4
```

That session was configured as four Kaggle notebooks, even though only one was
running. The same output did train: epoch 37 completed in 3.7 minutes, and HF
later recorded the run at 60/60. The epoch-38 `0%` line was the serialized
state of Kaggle's live widget, not evidence of an idle GPU.

**Fix (tyrelib v10):** `ACTIVE_KAGGLE_ACCOUNTS` is the only editable
parallelism setting; worker count and id are derived. One notebook prints
`worker=0/1` and `MODE=ONE NOTEBOOK`. Training also emits a plain-text
first-batch heartbeat every epoch, which survives notebook serialization. The
weight-norm diagnostic uses a detached no-gradient tensor and no longer emits
PyTorch's tensor-to-float warning. These are execution/visibility repairs;
the model, data, split, optimiser, batch size, seeds and epoch budget did not
change.

### ⚠ Bug 28 — cleanup calls cannot guarantee process-memory reclamation

The next v10 run completed two models and advanced a third from epoch 3 to 45,
then stopped at the live cgroup guard:

```
[RAM] host RAM 88.1% after epoch 45; pausing before the kernel is killed.
```

This time the guard was correct. Public epoch telemetry shows the long-lived
Jupyter process retaining about **0.17 GB/epoch** across both inspected RegNet
runs and **0.30 GB/epoch** over the inspected DenseNet tail. It happens with a
synchronous loader, drained telemetry buffers, garbage collection, and
`malloc_trim`. Native image libraries, PyTorch serialization/allocator state,
and CUDA runtime state do not all promise to return memory to the OS merely
because their Python objects were deleted.

**Fix (tyrelib v11): process lifetime is the memory boundary.** `run_all` can
launch each model in a fresh Python child. The child owns exactly one Trainer
and publishes the same rolling HF checkpoint. When it exits, Linux reclaims
the entire address space. If the child pauses at the RAM guard after making
epoch progress, the parent launches another clean child and resumes that same
run automatically. The parent retains static ownership, two-phase idle
takeover, and a 45-minute real-session deadline so a fresh child is not started
just before Kaggle terminates the allocation.

NB06 enables `isolate_runs=True`; other notebooks retain their existing
execution path. This changes process topology only, not experimental state.

### The checkpoint contract

Saving weights is not enough:

```python
{
  "epoch": int,                  # last COMPLETED epoch
  "model": state_dict, "optimizer": state_dict,
  "scheduler": state_dict,
  "scaler": state_dict,          # AMP: omit and loss scale resets on resume
  "rng": {"python":…, "numpy":…, "torch":…, "cuda":…},   # ALL FOUR
  "config_hash": str,            # asserted on resume
  "wall_seconds": float,         # cumulative, survives restarts
  "energy_joules": float,
  "custom_state": ...,
}
```

Each field prevents a specific silent corruption:

| Omit | What breaks |
|---|---|
| `scaler` | AMP loss scale resets; post-resume steps behave differently |
| `rng` | Augmentation and shuffling order diverges → **a resumed run is not equivalent to an uninterrupted one** |
| `config_hash` | You resume under an edited config and never notice |
| `wall_seconds` | Cumulative totals restart at zero mid-run |

The RNG one is the subtle killer. **Our study compares seeds** (`13 §6`) — a resume that loses RNG state makes "same config, different seed" stop meaning what we think it means.

### Atomic writes, always

```python
torch.save(state, path.with_suffix(".tmp"))
os.replace(path.with_suffix(".tmp"), path)          # atomic on POSIX
```

A session killed mid-write leaves a truncated file and the run is gone.

### Checkpoint at epoch boundaries

Mid-epoch resumption needs dataloader worker state, which is not reliably serialisable. **Epoch granularity is fine** — our epochs are minutes, not hours.

### Rebuild progress from logs, not status files

A session that died between writing a log and pushing its status leaves them disagreeing. **The log is the honest one.** A run marked complete but truncated by a crash is a "broken stub" — left alone, every future session skips it forever. Detect and demote to `paused`.

### Truncate the log on resume

A milestone push can land after the checkpoint was written, so the log may contain epochs the checkpoint doesn't know about. Without truncation you append duplicate epoch numbers and every cumulative statistic is wrong.

```python
h = pd.read_csv(history_path); h[h.epoch < start_epoch].to_csv(history_path, index=False)
```

---

## 8. Lifecycle guards

Four ways a session ends. Handle all four.

| Exit | Handler |
|---|---|
| `KeyboardInterrupt` | try/except around the loop |
| **`SIGTERM`** | signal handler — **this is the common one on Kaggle** |
| Uncaught exception | except block, mark `failed` |
| Normal/abnormal shutdown | `atexit` |
| Approaching session limit | watchdog: at 8.5 h, push and mark `paused` |

```python
class LifecycleGuard:
    def install(self):
        signal.signal(signal.SIGTERM, self._handle)
        atexit.register(self._atexit)
    def _fire(self, reason):
        if self._fired.is_set(): return              # exactly once
        self._fired.set(); self.on_flush(reason)
```

Catching only `KeyboardInterrupt` misses the platform kill entirely — which is how you lose the last 30 minutes of a 3-hour run. **The watchdog is the civilised one**: detect the limit yourself and stop cleanly rather than being killed mid-epoch.

---

## 9. Repository layout

**One repo**, because the rate limit is per user — two repos means two commits per cycle for no benefit. Make it a **dataset repo**: HF renders CSV and Parquet previews in the browser, so every metrics table is browsable without downloading.

```
runs/{run_id}/
├── config.yaml · config_hash.txt · STATUS.json · summary.json
├── metrics/      epochs.csv · final.csv · confusion_matrix.csv
├── xai/          saliency/*.npz · ter_bar_sar.csv · faithfulness.csv
├── telemetry/    energy_samples.csv · system_samples.csv · step_traces.jsonl
├── per_sample/   predictions.parquet
├── checkpoints/  ckpt_last.pt · ckpt_best.pt
└── env/          environment.json
registry/  events/ claims/ plans/
tables/    all_epochs.csv · all_final.csv · summary.csv · xai_summary.csv
analysis/  paper/
```

### Run IDs — deterministic and readable

```
{stage}-{arch}-{technique}-f{fold}-s{seed}
a-convnextv2t-base-f0-s1
b-convnextv2t-coral768roi-f2-s3
```

**Never auto-generate a UUID.** In six weeks you need to find a run by reading its name.

### Staging on scratch

Put the whole tree on `/kaggle/temp` (~1 TB). The uploader reads from there. `/kaggle/working` stays nearly empty. Losing scratch costs at most one push interval.

### Confirm-then-delete

```python
ok = sync.flush(timeout=1800)
missing = sync.verify_present([f"runs/{run_id}/checkpoints/ckpt_last.pt", ...])
if ok and not missing: shutil.rmtree(run_dir)
```

**A flush that merely didn't time out is not evidence the files arrived. Re-list the repo.**

### Push tiers

| Tier | Contents | When |
|---|---|---|
| light | config, status, metrics CSVs | every 30 min |
| heavy | checkpoints | every 30 min |
| bulk | raw telemetry, saliency `.npz`, parquet | every N epochs + at end |

Normal static-owner claim events are light files and wait for the same batch.
Run completion, a clean pause, an important notebook cell, Stop/SIGTERM, and an
explicitly stolen claim are major boundaries and flush immediately. An ordinary
claim must never consume a standalone commit.

Saliency arrays reach several MB per run; re-uploading them every half hour churns LFS for data nobody reads until the run finishes.

---

## 10. Telemetry

> **You train once. Record everything you could conceivably want** — re-running to recover a forgotten metric is unrecoverable time.

Per epoch, **~185 columns** — full schema in `17_DATA_LOGGING_SCHEMA.md`:

| Group | Columns |
|---|---|
| Identity | run_id, epoch, step, timestamps, account, worker, session, host, config_hash |
| Learning | losses, accuracies, macro/micro/weighted P-R-F1, balanced acc, **QWK**, MCC |
| Calibration | ECE, MCE, NLL, Brier, mean confidence |
| Loss parts | one column per term, `NA` when not in the objective |
| Optimisation | LR per group, grad-norm mean/max/p50/p95/p99, clip-hit rate, weight norm, **update-to-weight ratio**, AMP scale, **AMP scale-decrease count**, **NaN/Inf batch count** |
| Timing | epoch/train/val, **dataload vs compute split**, step-time p50/p90/p99, throughput |
| GPU, **per device** | util, memory, temperature, clocks, power, energy, throttle reasons |
| Host | CPU %, RAM, RSS, free disk |
| Energy | J / Wh / kWh and CO₂, per epoch and cumulative |
| Config echo | batch size, optimizer, scheduler… so the CSV is self-describing |

### The five columns people forget

1. **`dataload_frac`** — time waiting for data ÷ total. High means the GPU is starving and the fix is the loader, not the model. Unrecoverable after the fact
2. **`update_to_weight_ratio`** — ‖Δw‖/‖w‖. Healthy ≈ 1e-3. 1e-1 means LR far too high; 1e-6 means nothing is moving. Tells you before the loss curve does
3. **`nan_or_inf_batches`** — under AMP, non-finite losses are silent; the run continues and learns nothing from those batches
4. **`amp_scale_decreases`** — each is a step whose gradients overflowed and were **discarded**. Invisible by default
5. **`gpu{i}_throttle_reasons`** — non-zero means the card clocked down. Otherwise a slow epoch is a permanent mystery

**Per-device, not aggregated.** Train on one of two GPUs and an aggregate reports ~50% utilisation, hiding that half the allocation is idle.

### Schema discipline

- **Every column always present.** Missing quantity → `NA`, never omitted, never `0`. "This term doesn't exist" and "this term was zero" are different facts
- **Fill from the schema at the end:** `for c in SCHEMA: row.setdefault(c, NA)`
- **Test the schema against the requirements list programmatically** — map each requirement to its column(s) and assert none missing

---

## 11. The base64 library bootstrap

A 5,000-line library cannot live inside a notebook, and `pip install` of a private package needs auth. Embed the library as base64 in cell 1:

```python
import base64, sys
from pathlib import Path
WORK = Path('/kaggle/working')
_LIB = ('...96-char chunks...',)
(WORK / 'tyrelib.py').write_bytes(base64.b64decode(''.join(_LIB)))
if str(WORK) not in sys.path: sys.path.insert(0, str(WORK))
for _m in [m for m in sys.modules if m == 'tyrelib']: del sys.modules[_m]
import tyrelib
```

- **Chunk at 96 chars** — a single 300 KB string literal makes notebook JSON unreadable and some editors choke
- **Delete from `sys.modules`** — otherwise re-running cell 1 after an edit does nothing and you debug a ghost
- **The `.py` file is the source of truth; the notebook is generated.** Write `build_notebooks.py`. Never hand-edit base64
- **Verify the round-trip:** decode the blob back out of the `.ipynb` and byte-compare against the source. A silent truncation means an hour debugging the wrong code

---

## 12. The preflight notebook — build this first

**Run it on every account before anything else.** The cheapest place to find expensive mistakes.

| Check | Failure it prevents |
|---|---|
| Secrets present **and writable** — push a probe, then **re-list the repo** | Nine hours of training with nowhere to save it |
| Dataset attached at `/kaggle/input/...` | Silent fallback to a slow path |
| **Every model in the zoo builds, forwards, backprops** | An architecture whose internals don't match the measurement code |
| Every model at **every input resolution** in the sweep (224/384/512/768) | Shape errors discovered mid-sweep — ⚠ Bug 5 |
| **Every XAI method runs on every architecture** | Discovering at Stage E that Grad-CAM has no valid target layer |
| Cost table sane — no duplicates, plausible ratios | An entire axis being meaningless |
| **Kill-and-resume equivalence** | See below |
| Work split balance printed | One account working 3× longer |

### ⚠ Bug 6 — the resume test that validated nothing

**Do not** test resume by training a shorter run and asking for more epochs. That is a *clean completion* followed by an *extension* — a completely different code path that never touches the interrupt handler, the emergency flush, or the paused state.

```python
ref  = train(cfg)                                    # 1. uninterrupted reference
part = dict(cfg, _debug_interrupt_after_epoch=2)     # excluded from config_hash!
try: train(part)
except KeyboardInterrupt: pass                       # 2. real kill mid-run
res  = train(cfg)                                    # 3. resume in a fresh call
# 4. compare PER-EPOCH LOSS AFTER THE SEAM, not final accuracy
```

**Post-seam loss is where a lost RNG state shows up.** Final accuracy can match by luck; the loss curve cannot.

Add the debug hook to the training loop and **exclude it from the config hash**, or the resumed run fails its own hash check.

---

## 13. Dataset access

```python
# Preferred: attach FINAL as a Kaggle Dataset in the notebook sidebar
DATASET_ROOT = Path("/kaggle/input/<slug>")   # dir containing README.md, images/, splits/
train = pd.read_csv(DATASET_ROOT / "splits/cv0_train.csv")
valid = pd.read_csv(DATASET_ROOT / "splits/cv0_validation.csv")

assert set(train.session_group).isdisjoint(set(valid.session_group))
assert set(valid.image_kind) == {"clean_original"}
assert train.relative_path.map(lambda p: (DATASET_ROOT / p).exists()).all()
```

Read directly from `/kaggle/input`. **Never copy 1.2 GB into `/kaggle/working`.** Cache preprocessed tensors in `/kaggle/temp`.

---

## 14. Notebook conventions

Every notebook must state, per cell:

```
### Cell 6 — Train one fold
# What it does : trains {arch} on fold {k}, seed {s}, with resume
# Runtime      : ~18 min (T4, 384px, bs 32, 40 epochs)  [measured, not guessed]
# Outputs      : /kaggle/temp/runs/{run_id}/…
# Safe to stop : yes — flushes on SIGTERM and marks paused
```

And at the top:

```
TOTAL NOTEBOOK RUNTIME: ~7 h 20 m for this worker's slice (12 runs)
```

**Anchor every time estimate on a measurement.** Two calibration runs are enough.

Per-notebook checklist:

- [ ] `ACCOUNT` and `WORKER_ID` at the top, clearly marked
- [ ] Plan printed before work starts
- [ ] Safe to stop at any moment
- [ ] Safe to re-run from scratch (every cell idempotent)
- [ ] Ends with a blocking flush
- [ ] Markdown explains *why*, not just *what*

---

## 15. Build order

Steps 1–6 are project-agnostic infrastructure. Lift them wholesale.

1. **Library skeleton** — utils, atomic IO, config hashing, seeding, RNG capture
2. **Offline self-test** — `--selftest`, no GPU, no network. Everything below adds tests here
3. **Uploader** — batching, dedup, backoff, 429 parsing, **shared** rate limiter
4. **Registry** — sharded events, claims, ownership-aware `can_claim`
5. **Sharding** — `assign_workers`, `plan_work`, static cost table, shard report
6. **Lifecycle** — SIGTERM + atexit + interrupt + watchdog
7. **Telemetry** — schema, per-device monitors, epoch accumulator
8. **Training loop** — resumable, instrumented, tiered pushes
9. **XAI pass** — attribution, masks, TER/BAR/SAR, faithfulness
10. **Notebook generator** — base64 bootstrap, round-trip verification
11. **Preflight notebook** — including the real kill-and-resume test
12. **The actual science**

---

## 16. Pre-flight checklist

- [ ] Library regenerates notebooks; base64 round-trips byte-identically
- [ ] `--selftest` passes offline, no GPU, no network
- [ ] Token has **write** scope — verified by pushing and re-listing
- [ ] Rate limit is per token, and `cap × n_accounts < 128`
- [ ] Registry shards per writer; **no shared mutable file anywhere**
- [ ] `can_claim` lets a worker resume its own run immediately
- [ ] Work split printed with estimated hours per worker before starting
- [ ] Checkpoint contains optimizer, scheduler, scaler, **all four RNG streams**, cumulative counters
- [ ] Resume asserts `config_hash`
- [ ] Kill-and-resume test compares **post-seam per-epoch losses**
- [ ] SIGTERM, atexit, KeyboardInterrupt and watchdog all flush
- [ ] Dataset attached; staging on `/kaggle/temp`
- [ ] Confirm-then-delete verifies by re-listing the repo
- [ ] Schema tested against the requirements list programmatically
- [ ] Every schema column always present; `NA` where undefined
- [ ] **Shuffled-label control run and confirmed at chance**

**Recurring**

- [ ] Audit the repo: is every expected run present and complete?
- [ ] Watch `dataload_frac`, `nan_or_inf_batches`, `amp_scale_decreases`
- [ ] Sanity-check the science

---

## 17. The two ideas worth carrying to every project

**1. Coordination by arithmetic, not negotiation.** Every worker computes the same assignment from the same inputs and keeps its slice. No locks, no leader, no messages, nothing to deadlock. Stealing is a recovery mechanism, never the primary path.

**2. Record everything, because you train once.** The cost of an extra column is bytes. The cost of a missing one is a re-run you cannot afford. When a quantity doesn't exist, write `NA` — never omit it, never fake it.
