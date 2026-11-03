# NB13 / ATLAS-CONTEXT r3 — Kaggle run and recovery guide

<!-- NB14_COMPLETED_20261101 -->
## Completed family review — 1 November 2026

NB13/NB13A and the NB14 review are complete and independently HF-verified. Preserve these executed notebooks and the original 32-entry runtime. [NB14 completion](../../phase3NB14Completion.md). Current next action is [NB15 existing-data readiness and evidence passport](../readiness_gate/README.md), with CPU / Internet / HF_TOKEN and no image attachment. Earlier training/review run instructions below are historical; no rerun is needed.
<!-- END_NB14_COMPLETED_20261101 -->

<!-- NB13_COMPLETED_20261101 -->
## Historical — original context fit completed; use NB14 next

**1 November 2026:** NB13A restored HF update 2,486 and completed all 60 epochs / 2,940 updates. All 1,503 direct files / 58,176,066 raw CSV rows verified; owner released. Do not rerun the completed fit. [Completion and evidence](../../phase3NB13Completion.md).

The context candidate remains unpromoted: low recall 0/62, historical BA 50%, confidence loss worse. Next is **NB14 CPU / Internet / HF_TOKEN / no image attachment**, Run All. [Review guide](../context_review/README.md). Existing photos only; workstations and frozen training source unchanged. Earlier run/pending statements below are historical.
<!-- END_NB13_COMPLETED_20261101 -->


> Superseding recovery update, 31 October 2026: revision 1 failed because local working files were unavailable. [NB13A revision 2](../../phase3NB13ARepair.md) is now self-contained, prefers retained/attached saved state and otherwise restores the verified HF checkpoint. Use the updated notebook with Run All. Earlier existing-session-only instructions below are historical.

> Current, 31 October 2026: actual dual-T4/live-HF preflight passed; NB13 is partially trained and stopped. HF update 1,294 is verified; notebook output reaches local update 2,467. Preserve the existing session and use [the NB13A recovery instructions](../../phase3NB13FreezeAndRecovery.md). Earlier implementation/pending-execution statements below record the original delivery.

**One new candidate; measured training has not started locally.** NB12 is already HF-verified and needs no rerun. This separate model freezes selected LC epoch 7 and learns a position-aware correction. Its accuracy is unknown. Existing tyre data only; workstation work remains deferred.

## Settings and execution

1. Upload [phase3_NB13_TREAD_TWO_ATLAS_CONTEXT_Resumable_Pilot.ipynb](../../notebooks/phase3_NB13_TREAD_TWO_ATLAS_CONTEXT_Resumable_Pilot.ipynb) to Kaggle.
2. Select **GPU T4 ×2** and enable **Internet**.
3. Attach exactly one **expanded** `shanmuk4622/tire-dataset-prepared-phase2` dataset directly from Kaggle. Keep the input read-only; do not copy its images into working output or attach a ZIP instead.
4. Enable Kaggle Secret **HF_TOKEN** with write access to `Shanmuk4622/tyre-wear-study`. Do not paste or print the token.
5. Begin with at least **12 GiB free** working output. Preflight must also pass its measured peak estimate below **16 GB** and the actual free-space guards. The assumed output budget is 20 GB; extra scratch is not assumed.
6. Choose **Run All**, in order. Do not bypass failed input/recovery/storage checks. Keep the executed notebook after completion.

| Code cell | Work | What to expect |
|---|---|---|
| 1 | Verify/extract 33 embedded files; establish launcher | Source compatibility checks; no measured fitting |
| 2 | `prepare` | Verify data/roles/references, devices, upstream dense and exact selected LC weights; create immutable contract |
| 3 | `preflight` | Offline transaction tests + actual dual-T4/live-HF restoration fixture + measured capacity projection |
| 4 | `train` | Restore or fit one seed for 60 epochs / 2,940 updates, select by validation, final/selected evaluation, eight diagnostics, publish report |
| 5 | Inspect receipt/curves/explorer | Display last verified remote revision, selected/final report and interactive HTML link |

GPU0 fits; GPU1 evaluates an immutable copy at barriers. This is one FP32 model, batch 8 with dense microbatch 1. The notebook pins `huggingface_hub==0.36.2` only; it does not install Torch/CUDA. Source/assets are embedded, so a separate code dataset upload is unnecessary. Upstream weights are downloaded once from pinned HF objects and SHA/size/protocol checked. No local workstation dependency is required.

## Resume and stopping

Work directory: `/kaggle/working/tread_two_atlas_context_r3`. Normal local commits happen after each completed optimizer update; HF flushes normally every 30 minutes and at major stages. Use Kaggle Stop while keeping the session/files alive: the launcher forwards SIGINT, waits for atomic state and attempts an immediate flush subject to HF server delay. Forced kernel/session death cannot publish unsent updates.

Restart by using this **unchanged** notebook with the same dataset/settings and Run All. It reconstructs the exact frozen sources, restores a verified compatible remote/native state, replays an incomplete update and resumes the next completed-update cursor. It never silently restarts a whole completed epoch. Completed compatible preflight/fit stages are verified and skipped. A different environment/device/source contract is not silently treated as the old run.

Target: `Shanmuk4622/tyre-wear-study`, namespace `phase3/tread-two/atlas-context-r3/fit/<contract>`. Engineering preflight is separate. Never run simultaneous writers to the same contract or delete/override a live owner lease. Keep LAST_REMOTE.json, lifecycle/HTTP logs, `run/console.log`, native states and CSVs while a publication is pending. All science is published as direct CSV/native/JSON/code/PNG/HTML files; no ZIP/NPZ/database uploads.

## Troubleshooting

| Symptom | Action |
|---|---|
| Last training cell returns normally after report/owner release | Expected completion; inspect receipt, save executed notebook; no restart needed |
| No valid prepared input or checksum mismatch | Attach the declared expanded Kaggle version exactly once; preserve error details; do not edit checksums to continue |
| Missing HF_TOKEN / permission failure | Enable the authorized Secret and repository write access; never paste a credential in code/output |
| Rate limit / Retry-After / commit delay | Keep session alive; inherited shared budget/backoff/reconciliation handles retries; avoid manual repeated pushes |
| Active owner / stale writer rejection | Preserve identity and logs; do not start another writer or erase ownership records |
| Source/environment/contract mismatch | Preserve old state and stop dependent fitting; report the mismatch for a compatibility review |
| Storage projection or free-space guard fails | Preserve state/receipt; report actual bytes; do not drop CSVs, disable guards or assume extra scratch |
| Unexpected stop with no traceback | Inspect completion phase, LAST_REMOTE.json and published manifest before assuming failure; completed work skips fitting |
| Exception/forced loss | Keep traceback and executed notebook; recovery starts from last verified publication, plus compatible surviving local completed updates |

## Outputs and review

Expect all 60 epoch weights, completed-update recovery, 27 direct scientific CSV families, 53,754 planned observations, full selected/final endpoint attention and native image evidence. Per-observation attention summaries are always retained; full per-update attention/activation/gradient tensors are explicitly not retained. The explorer filters selected/final cases, errors, empty support and source identity, and separates frozen LC logits from context correction. Reference cards explain only the frozen case branch.

After the notebook finishes, return the executed NB13 for independent pinned-HF verification. No improved accuracy, model promotion or workstation change is claimed beforehand. Low/mid/high are mileage proxies; no physical wear/depth/safety conclusion is supported.

## Source and resource map

| Files | Responsibility |
|---|---|
| `RECIPE.json`, `BASE_INPUT.json`, `FILES.json`, `spec.py` | Frozen intent, exact LC input, source bytes/hashes and environment-specific contract |
| `model_atlas.py`, `frozen_lc.py`, `frozen_anchor.py`, `lc_features.py` | New spatial correction; preserved LC/dense calculations; exact local-contrast transform |
| `data_atlas.py`, `legacy_data.py`, `phase2_training_data.py`, `geometry.py` | Prepared inputs, immutable splits, deterministic replacement draws, source SHA and atlas coordinates |
| `assets/*.npy`, `assets/prototype_bank.csv`, `assets/sampling_plan.csv` | Literal frozen reference inputs and query/group provenance; these are not trained context weights |
| `objective.py`, `worker.py` | Paired loss, one completed update, validation/selection/endpoints, diagnostics and frozen-base assertions |
| `schemas.py`, `telemetry.py` | Every direct CSV column, prediction/evidence/attention accounting and metric denominators |
| `store.py`, `common.py` | Atomic CSV/native recovery, hashes, RNG and capacity guards |
| `coordinator.py` | Kaggle stages, child stop forwarding, ownership, 30-minute/major/stop snapshot scheduling |
| `publisher.py`, `evidence_hub.py`, `rate_budget.py` | Direct HF transport, CAS snapshots, retry/backoff, request accounting and fencing |
| `acceptance.py`, `report.py` | Real-model disposable recovery/storage gate; saved-evidence curves and interactive explorer |
| `UPSTREAM.json`, `VENDORED.json`, `LC_RULES.json` | Inherited provenance and fixed representation/reference rules |

The 32-file inventory covers runtime/assets, not this explanatory README. Do not run scaffold scripts over an active/completed experiment. The new source folder is isolated from both completed models and the workstations.

[Architecture](../../phase3TreadTwoContextArchitecture.md) · [Frozen protocol](../../phase3TreadTwoContextProtocol.md) · [Complete CSV dictionary](../../phase3TreadTwoContextEvidenceSchema.md) · [Local checks](../../phase3TreadTwoContextValidation.md) · [NB12 findings](../../phase3NB12Completion.md).
