# TRACE executable notebook package

<!-- NB24_COMPLETION -->
## Actual execution update — 2 November 2026

NB24 completed the original one-seed fit and review. Original selected=None and
all source/contract identities are preserved. [Completion](../../phase3NB24Completion.md)
and [illustrated findings](../../phase3TreadTwoTRACEFindings.md) supersede pending
execution statements below. No rerun or new fit is required for this cycle.
<!-- END_NB24_COMPLETION -->

**Separate unmeasured revision, 2 November 2026.** This package implements the
approved fixed-target design without changing the original TRACE prototype,
completed GEO/ATLAS studies or either workstation.

Start with the **[NB24 Run All guide](../../phase3TreadTwoTRACEBatch.md)**.
[Acceptance evidence](../../phase3TreadTwoTRACEValidation.md) distinguishes local
checks from the live Kaggle/HF gates performed by the notebooks.

| Module | Responsibility |
|---|---|
| `config.py`, `DATA_PIN.json`, `FILES.json` | One exact data/source/environment-bound candidate and disposable gate identity |
| `data.py` | Kaggle input verification, historical roles/components and full-coverage schedule |
| `cache.py` | Atomic eight-image chunks, fixed geometry/target construction and train-only statistics |
| `frozen_geometry.py`, `frozen_anchor.py`, `geometry.py`, `atlas_geo.py` | Pinned inherited geometry and predicted-support canonical coordinates |
| `targets.py`, `model.py`, `objective.py` | The checked TRACE math,71,022-parameter branch and all loss terms |
| `schema.py`, `SCHEMAS.json`, `telemetry.py` |48-family/ 671-field executable CSV writers and exact FP32 native pointers |
| `store.py` | Bounded immutable shards, committed prefixes, two recovery slots and atomic unit chain |
| `metrics.py`, `resources.py`, `budget.py` | Fixed controls, selection/spectra, actual allocator peaks and full formatted capacity projection |
| `engineering.py`, `worker.py` | Secret-free numerical gates, complete lifecycle and snapshot/status handling |
| `coordinator.py` | Single parent writer, real HF seam/restore, dependencies,30-minute/major/stop publication and final receipt |
| `review.py` | CPU byte/native/reduction/selection audit, finite stage receipts and interactive passport |
| `publisher.py`, `evidence_hub.py`, `rate_budget.py`, `common.py` | Exact vendored previously tested transport/persistence primitives |
| `VENDORED.json`, `STORE_LINEAGE.json` | Original source identities; TRACE store is explicitly adapted, not claimed byte-identical |

The notebooks embed33 source/configuration payloads, including the selecting
source inventory. Source Python is editable before a measured run; regenerate
the new notebooks together after changes. Once a user starts a measured run,
preserve its source, protocol, notebooks, cache identities and HF namespace.
Transport repairs must not silently create a new fit or rewrite its model.

The normal scientific recipe is one seed / 60 epochs / 2,940 updates at batch 8 on
GPU 0. GPU 1 builds the immutable geometry/cache first. Actual dual-T4 and actual
HF upload→download→fresh-worker continuation are required before measured
initialisation. Offline CPU mode is restricted to disposable engineering.

HF namespace: `phase3/tread-two/trace-r1`, with distinct preparation, engineering,
fit, review and transport-tail prefixes. All tables are direct CSV; configurations
are JSON, tensors PT/NPY and visuals PNG/HTML. No ZIP/NPZ/database envelopes.
Only Kaggle Secret HF_TOKEN is read by the parent; numeric workers receive no
token and make no HF requests.

Rolling states retain the latest complete optimizer/RNG/cursor and source-bound
evidence prefixes. All epoch weights and all declared per-view predictions,
feature bodies, objective terms, controls and failures remain available.
Unavailable fields/errors are explicit, not fabricated observations. Peak
counters are reset and read per update/cache chunk; staging records distinguish
logical referenced bytes from extra copied payload.

No new photos, workstation integration, improved-classification, physical-depth
or safety claim is part of this batch. A fully executed fit can still be rejected
by every validation gate; preserve selected=None and final research evidence.
