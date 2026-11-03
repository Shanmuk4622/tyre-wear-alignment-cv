# NB15 — existing-data readiness and evidence passport

<!-- NB15_CLOSED_20261101 -->
## Completed and verified — 1 November 2026

NB15's four cells completed. Independent HF checks pass for all 38 direct files / 16 CSVs / 196,905 rows, three figures and the passport; ownership is released. Expected support readiness remains false. **No rerun is needed.** [Completion](../../phase3NB15Completion.md) · [Consolidated research handoff](../../phase3TreadTwoResearchClosure.md). The original instructions below are historical run/recovery documentation; preserve the executed notebook and four frozen gate modules.
<!-- END_NB15_CLOSED_20261101 -->


**CPU saved-record audit, not training.** [Completed NB14](../../phase3NB14Completion.md) · [readiness findings and module guide](../../phase3TreadTwoReadiness.md).

## Run on Kaggle

1. Import [NB15](../../notebooks/phase3_NB15_TREAD_TWO_Existing_Data_Readiness_And_Evidence_Passport.ipynb).
2. Accelerator **None / CPU**, **Internet ON**, no image dataset attachment.
3. Enable the **HF_TOKEN** Secret with write permission for `Shanmuk4622/tyre-wear-study`.
4. Keep at least **2 GiB free output** and about **4 GB RAM**; inputs total about 70 MB.
5. **Run All**: extract → training-only support/correction analysis → direct HF publication → figures/passport/receipt.
6. Keep the executed notebook and `GATE_PUBLICATION.json` for independent verification.

| Output | Expected evidence |
|---|---|
| `report/SUMMARY.json` | Analysis passed; scientific readiness false; six conditional folds, two missing fit classes |
| Direct CSV | 16 tables / 196,905 rows, including 188,160 algebraic saved-view observations |
| Visuals | Three scientific PNGs and interactive `Evidence_Passport.html` |
| `CAPABILITIES.json` | Classifier research-only/unpromoted; retained geometry/internal-evidence scope; depth/safety unavailable |
| Publication | Released-owner receipt under `phase3/tread-two/readiness-gate/v1/<contract>` |

Working path: `/kaggle/working/tread_two_nb15_readiness/`. The private `publication_sdk/` holds HF SDK 0.36.2 only for the publishing subprocess. No global HF SDK/Torch/CUDA replacement. No original image/model download, model forward, optimizer update or workstation change.

## Stop, restart and troubleshooting

| Situation | Action |
|---|---|
| Analysis interrupted | Preserve files and rerun unchanged. Verified cache is reused; derived tables are atomic and rebuilt. |
| New session | Reimport the same notebook and Run All; approximately 70 MB of pinned evidence downloads again unless preserved output is reused. |
| Scientific readiness false | Expected support finding. Do not rerun a model to change metadata or merge uncertain identities to force a pass. |
| Private SDK install interrupted | Keep files and rerun publication. READY is written only after the private version check succeeds; incomplete installation is retried. |
| HF rate/backoff wait | Allow server delay to finish and keep a single writer. Request/commit budgets and immutable staging preserve completed output. |
| Stop during upload | Catchable Stop lets the completed report flush finish; forced kernel death cannot upload unsent files. Rerun unchanged to reconcile/complete. |
| Secret or compatibility failure | Enable the authorized HF_TOKEN or preserve the exact error. Never paste a token into source or overwrite incompatible manifests/checkpoints. |

No ZIP/NPZ publication. This finite audit publishes after the major analysis completes; it does not claim optimizer recovery for a partially derived report. More observations are richer telemetry from existing data, not additional independent tyres. No NB13/NB13A/NB14 rerun is needed.

[CSV inventory](../readiness_report/CSV_INDEX.md) · [methods](../readiness_report/METHODS.md) · [local acceptance](../readiness_validation/PACKAGE_VALIDATION.json).
