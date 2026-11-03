# TRACE notebook acceptance — local proof and pending live gates

<!-- NB24_ACTUAL_RESULT -->
## Current result — original NB24 complete

The actual Kaggle fit and HF review are complete. **Do not rerun the completed
batch.** [Completion and live gate proof](phase3NB24Completion.md) ·
[Actual findings, figures and next research](phase3TreadTwoTRACEFindings.md).
The delivery/local-test/run instructions below are historical reference; their
pending-live statements describe the pre-execution checkpoint.
<!-- END_NB24_ACTUAL_RESULT -->

**2 November 2026.** [NB21–NB24 and Run All instructions](phase3TreadTwoTRACEBatch.md)
are delivered together. This page records actual local checks; no measured
60-epoch TRACE fit or new HF publication has been performed here.

| Check | Actual result | Evidence |
|---|---|---|
| All notebooks |4 valid clean notebooks,33 exact embedded payloads each; fresh extracted setup/imports execute | [Payload proof](tread_two/trace_validation/PAYLOADS.json) |
| Actual model lifecycle |8 existing train images,2 disposable optimizer updates, all phases/endpoints/probes/retrieval | [Lifecycle](tread_two/trace_validation/LOCAL_LIFECYCLE.json) |
| Complete-update restart | Fresh worker from the within-epoch native seam matches model/optimizer/RNG/cursor and scientific files | Same receipt; CPU, not a live HF proof |
| Atomic faults | Evidence/state/pointer interruptions, appended tails, committed corruption, worker error and injected selected export | [Fault proof](tread_two/trace_validation/TRANSACTIONS.json) |
| Preparation restart |16 actual train images; stop after 8; original chunk hashes preserved; review partial output replay and receipt skip/corruption rejection | [Proof](tread_two/trace_validation/PREPARATION_RECOVERY.json) |
| All existing inputs |570 images /72chunks;386/ 81/ 103roles;all 60 source schedules checked | [Full cache](tread_two/trace_validation/FULL_CACHE.json) |
| Source scope of full cache | Scientific input/target modules unchanged after later parent/storage/allocator fixes | [Equivalence](tread_two/trace_validation/FULL_DATA_SOURCE_EQUIVALENCE.json) |
| Per-view prediction audit |884,736fixture prediction values and34,560moment values match native FP32 exactly | Full lifecycle review |
| Objective and selection audit | Every raw local/masked loss and both fixture training objectives reconstructed; original selected=None reproduced | Same review; no claim about future TRACE accuracy |
| Writer availability | All 48 families exercised across their natural/injected fixture stages; 671 declared fields, explicit nullable gaps | [Coverage CSV](tread_two/trace_validation/WRITER_COVERAGE.csv) |
| Parent orchestration | Preparation→gate→fit→review order, completed skip, owner release, stop flush, failed-budget block and finite tail | [Offline injected-service proof](tread_two/trace_validation/COORDINATOR.json) |
| Interactive passport | Actual report JavaScript, role/search/paging, PNG signatures and injected missingness renderer | [Viewer](tread_two/trace_validation/VIEWER.json); synthetic DOM, not native screenshot |
| Preservation / Git | Completed source/notebooks and both workstations checked; caches/fixtures ignored, current deliverables visible | [Handoff](tread_two/trace_validation/DELIVERY.json) |

The full scientific-data construction ran locally on one GTX1650 with CPU target
statistics and zero optimizer updates. Other model/transaction fixtures use CPU;
the host's one CUDA RNG is also retained where available. These do not substitute
for two-T4 arithmetic or remote HF continuation. The actual parent test replaces
HF/numeric services with explicit local fixtures; it makes zero network writes.

## Capacity evidence

The actual all 570 cache is **1,093,059,090 bytes**. Actual formatted
fixture CSV/native sizes, adjusted for this full cache and the declared full
training/validation/two-potential-endpoint/probe schedule, project approximately
**14.26GB logical /17.25GB peak**. This is a local planning
estimate. NB21 repeats the calculation on the real Kaggle full cache/formatter
and requires actual remaining free space plus the 2 GB reserve before enabling
measured initialisation. It never silently drops promised evidence to fit.

## Issues found before delivery

| Issue | Resolution |
|---|---|
| Some Windows fixtures cannot create hardlinks | Permit bounded copies only in offline engineering; measured Kaggle snapshots still require hardlinks |
| NumPy boolean in a selection record | Convert eligibility gates to native bool before canonical recovery serialization |
| Empty CSV shards in review | Skip empty bodies while preserving their declared headers |
| A secondary GPU allocator may be uninitialised before peak reset | Non-random four-byte context warm-up, synchronise, then reset peak counters |
| Final snapshot acknowledgement could race worker exit | Parent records the completed publication/released owner in its callback and reuses that verified receipt |
| Partial review files pre-exist after interruption | Explicit declared artifact sets select complete stage outputs; directory-difference discovery removed |
| Over-frequent phase publication could consume commit quota |30-minute cadence plus major scientific stages; epochs remain durable locally |
| Source changed during development | Rebuilt all new payloads and source-bound checks; no executed historical notebook was regenerated |

## What remains for Run All

Actual T4×2 availability, Kaggle capacity, HF Secret/authentication, a verified
published checkpoint downloaded into a fresh worker, and identical continuation
must pass before the long fit. Then one seed / 60 epochs is measured and the
complete review publishes its observed result. A scientific rejection is still
a valid completed run; final/all epoch weights are retained, selected may remain
unavailable, and no physical-depth/safety or improved-classification claim follows.
