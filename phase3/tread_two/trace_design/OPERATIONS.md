# Operations contract for the future Kaggle batch

<!-- NB24_COMPLETION -->
## Actual execution update — 2 November 2026

NB24 completed the original one-seed fit and review. Original selected=None and
all source/contract identities are preserved. [Completion](../../phase3NB24Completion.md)
and [illustrated findings](../../phase3TreadTwoTRACEFindings.md) supersede pending
execution statements below. No rerun or new fit is required for this cycle.
<!-- END_NB24_COMPLETION -->

<!-- TRACE_BATCH_SUCCESSOR -->
## Executable successor delivered

**2 November 2026:** this file retains the original design-stage record. The
separate [TRACE notebook batch](../../phase3TreadTwoTRACEBatch.md) is now implemented
with NB21–NB24,671 executable fields and local acceptance. Use NB24 for automatic
preparation→gates→pilot→review. Actual Kaggle/HF and measured results remain
pending; the original prototype/code/check receipts remain unchanged.
<!-- END_TRACE_BATCH_SUCCESSOR -->

**Design stage.** No new training notebook is delivered or needs to be run now.
This contract is to be implemented and tested in the new TRACE package. The
completed NB16–NB20 packages are preserved; no source monkey-patching or reruns.

## Resource ownership

| Resource | Planned use / gate |
|---|---|
| Kaggle accelerator | T4×2; verify both actual devices before enabling fit |
| GPU 0 | TRACE encoder/context/decoder/projector and one texture optimizer |
| GPU 1 | Frozen geometry extraction/cache and evaluation work; no optimizer |
| DDP / cross-device batch splitting | None; batch 8 is on GPU 0, not two independent batch-4 covariance losses |
| Data | Attach existing expanded `shanmuk4622/tire-dataset-prepared-phase2` directly; read-only |
| Output | Measure 20 GB area/free bytes and real staging amplification |
| Optional scratch | Use only after filesystem/free-space checks; never assume a 1 TB allocation survives restart |
| Secret | Only Kaggle Secrets `HF_TOKEN` for `Shanmuk4622/tyre-wear-study` |
| Versions | Record/verify Torch, CUDA, GPU capability, NumPy, Python and preprocessing versions; preserve installed CUDA Torch |

GPU-1 extraction should finish/cache immutable geometry before GPU-0 texture
training. An idle GPU after its assigned phase is reported honestly. Sharing
both GPUs does not imply DDP speedup. Reuse only the small pinned geometry
export/required arrays; do not download the entire 10.31 GB old fit just to start
a new texture experiment.

## Complete-update continuation

1. Acquire one local file lock and one remote run owner; reject concurrent writers.
2. Read remote selecting manifest before deciding start/resume/skip. Restore the
   newest compatible complete state after size/SHA verification. Compare local
   and remote lineage; do not overwrite an ahead remote run with stale local data.
3. Verify source/config/data/geometry/target-statistics/environment identities.
   Save model, optimizer, Python/NumPy/CPU and **both CUDA RNG states**, deterministic
   schedule, next complete-update cursor, partial epoch aggregates, counters,
   native shard indexes and committed CSV/native prefixes. Explicitly record
   scheduler/scaler/queue as unused where applicable.
4. Commit atomically after each complete optimizer update. On recovery, truncate
   incomplete CSV/native tails back to the selected committed prefixes; replay
   only an interrupted incomplete update. No epoch restart or hidden extra fit.
5. Before a long fit, exercise kill/stop/corrupt-prefix/disk-pressure fixtures and
   actual HF upload→fresh-worker download→continue equivalence on the real model.
6. Skip verified completed preparation, fit, endpoint and review stages. A clean
   re-run of the master cannot create another seed/run automatically.

Exact bit identity is a same-environment gate, not a cross-hardware promise.
[PyTorch's reproducibility documentation](https://docs.pytorch.org/docs/2.15/notes/randomness.html)
describes deterministic-operation controls and remaining version/platform limits.
The observed current CPU fixture is narrower than a dual-T4 continuation proof.

## HF cadence and quota control

Publish direct CSV/JSON/NPY/PT/PNG/HTML through immutable object staging and a final
selecting manifest. Persist locally at completed updates; publish every **30
minutes**, after major stages and immediately attempt a flush on catchable
Stop/error/SIGINT/SIGTERM. Preserve and report the last verified remote revision
and unsent local updates. A forced kernel death cannot run cleanup or upload
unsent bytes.

Use the established shared account request-budget/owner approach in a separately
vendored and regression-tested transport layer. Batch/deduplicate unchanged
objects, cache authenticated reads, use bounded transient retries and honour
`Retry-After`/server quota/reset headers. Keep a conservative client guard
(historically 96 commit attempts/hour); this does not mean every commit costs one
HTTP request, or that 128 requests/hour is a guaranteed universal server limit.
Other sessions/hosts using the account can consume quota too.
[Current HF rate-limit documentation](https://huggingface.co/docs/hub/rate-limits)
was checked on 2 November 2026; SDK upgrades must be separately tested rather
than replacing the previously proven private SDK without review.

Numeric workers receive no token or hidden credential fallback. The parent is
the only HF writer. Never print a secret, load a project `.env`, search alternate
HF credential files, or put a token into a saved notebook output.

## Troubleshooting that must be built into the notebook

| Symptom | Required behaviour |
|---|---|
| No new epoch output | Show phase, update, heartbeat, last durable commit, resource/staging and publication progress |
| HF 429/backoff | Honour authoritative delay, preserve local state, show next retry without request storms |
| Catchable Stop | Pause numeric worker at complete boundary, snapshot consistent prefix, immediate flush attempt and status receipt |
| Session reset | Same master restores verified HF state; attached local files may supply newer compatible state |
| Missing dataset or wrong checksum | Stop with exact input name; do not substitute a dataset or edit a checksum |
| Fewer than two T4s | Engineering may report available devices; measured dual-T4 gate remains false |
| Less than required storage | Stop before fit, retain gate report, never silently reduce telemetry |
| Too few context-eligible rows | Record missing diversity loss and skip reason; no rank-infeasible covariance penalty |
| Numeric divergence | Commit error/last good state, flush evidence, reject unsafe continuation or explicit protocol drift |

All future runnable notebooks must include exact accelerator/input/Internet/Secret
settings, Run All order, compatible restart steps, native/direct-CSV output
locations and stop limitations in their own Markdown cells.
