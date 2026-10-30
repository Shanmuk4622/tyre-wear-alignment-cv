# ATLAS-LC r2 — code ownership and Kaggle run guide

<!-- NB11_COMPLETE_UPDATE_20261030 -->
## Current — NB11 complete; NB12 review ready

**Measured seed completed on Kaggle and independently verified.** The earlier local delivery notes below are historical. [Completion and findings](../../phase3NB11Completion.md). Next is [CPU review NB12](../../notebooks/phase3_NB12_TREAD_TWO_ATLAS_LC_Completed_Review.ipynb), not another fit. Keep all frozen source/assets and both workstations unchanged.

<!-- END_NB11_COMPLETE_UPDATE -->
Implemented30 October2026. One measured seed is pending. Local batch8 pause,
abrupt-stop and snapshot-restore equivalence passed on the existing GTX1650;
actual dual-T4 and live-HF checks run automatically in NB11 before long fitting.

## Run NB11

1. Open [the self-contained notebook](../../notebooks/phase3_NB11_TREAD_TWO_ATLAS_LC_Resumable_Pilot.ipynb)
   on Kaggle. Select **T4×2**, Internet ON.
2. Attach exactly one **expanded** `shanmuk4622/tire-dataset-prepared-phase2`
   input directly from Kaggle. It remains read-only.
3. Enable write-capable Kaggle Secret **HF_TOKEN** for `Shanmuk4622/tyre-wear-study`.
   Never paste a token or add an alternate credential source.
4. Start with12GiB free. Run All: setup → prepare → live recovery/storage preflight
   → measured fit/endpoints/diagnostics → report. Do not skip a failed preflight.
5. Save the executed notebook when done. Preserve all CSV/native output and HF
   revision for independent audit. If execution exits normally after completion,
   that is expected; unchanged reruns verify completion and skip fitting.

For an ordinary stop, use Stop and allow safe update commit/HF flush. For forced
session loss, rerun **unchanged** with the same input and secret; remote recovery
precedes resume/skip. Unsynced work after the last verified HF snapshot may be
lost. Backoff may delay publication; do not delete leases to bypass a wait.
Inspect `run/console.log`, `LIFECYCLE.jsonl`, `NUMERICAL_EVENTS.jsonl`,
`HTTP_REQUESTS.jsonl`, `REMOTE_HISTORY.jsonl` and `LAST_REMOTE.json` after a failure.

## Source responsibilities

| Files | Responsibility |
|---|---|
| `model_atlas.py`, `lc_features.py`, `objective.py` | Frozen dense anchor, fixed LC transform, trainable directional case head, source-only self suppression, paired loss |
| `geometry.py`, `frozen_anchor.py` | Existing predicted-mask atlas geometry and preserved TREAD-ONE model |
| `data_atlas.py`, `legacy_data.py`, `phase2_training_data.py` | Prepared release/role checks, full source hashes,386query eligibility, seed-addressed two-view draws and sqrt group sampling |
| `assets/` | Pinned NB10 literal RGB/validity/source-SHA/label arrays, case registry and full planned sampling probabilities |
| `worker.py`, `telemetry.py`, `schemas.py` | One optimizer, immutable evaluation barriers,24CSV families, dense deduplication and conserved native image evidence |
| `store.py`, `common.py` | Atomic CSV/native/pointer transactions, suffix rollback, RNG and immutable export/install |
| `coordinator.py`, `acceptance.py` | Kaggle ownership, exact remote recovery, live acceptance, capacity projection, safe publication and owned-child lifecycle |
| `publisher.py`, `evidence_hub.py`, `rate_budget.py` | CAS lease, bounded immutable direct-file batches, deduplication, server backoff and separate HTTP/commit guards |
| `report.py` | Saved-evidence curves, all endpoint cases and literal supporting/opposing reference explorer; no training or selection |
| `spec.py`, `RECIPE.json`, `LC_RULES.json`, `UPSTREAM.json`, `VENDORED.json`, `FILES.json` | Frozen source/asset, environment, experiment and exact upstream provenance |

`FILES.json` freezes30source/assets; NB11 embeds those plus inventory =31files.
Markdown is explanatory and may be updated; it does not alter an active model
contract. Do not rerun source-preparation/freezing tools against a running fit.
New representation/bank/objective/environment requires an explicit compatible
review or separate experiment; completed v1 and workstations remain unchanged.

Dependencies use existing Kaggle PyTorch/NumPy/Pillow plus pandas,Matplotlib,
requests,filelock and tested Hub0.36.2. Setup pins only the Hub client. Locally,
every Python invocation uses `phase3/run_cv.cmd` (`conda activate cv_conda`);
no Torch/CUDA/environment replacement was needed.

See [architecture](../../phase3TreadTwoLCArchitecture.md),
[protocol and resource/recovery limits](../../phase3TreadTwoLCProtocol.md),
[schema](SCHEMA.md), [local acceptance](../lc_validation/LOCAL_VALIDATION.json)
and [notebook packaging](../lc_validation/NOTEBOOK.json).

The source hash for each vendored/adapted file is in the inventory. Historical
provenance remains in VENDORED; r2's geometry/dense/store/publication foundations
come from v1, while model/data/worker/telemetry/schema/acceptance/coordinator and
transport are separately adapted here. This runtime never installs a bank into
the completed v1 model.
