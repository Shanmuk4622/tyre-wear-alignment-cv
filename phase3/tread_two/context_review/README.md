# NB14 — ATLAS-CONTEXT saved-results review

<!-- NB14_COMPLETED_20261101 -->
## Completed — use NB15 next

NB14 completed and independently HF-verified on1November2026: all80directfiles/52CSVs, no errors, released owner and unchanged fit. No rerun needed. [Completion](../../phase3NB14Completion.md). Next: [NB15 readiness/evidence passport](../readiness_gate/README.md). Earlier run instructions below are preserved as historical guidance.
<!-- END_NB14_COMPLETED_20261101 -->


The original NB13/NB13A fit is complete and HF-verified. **Run NB14 next; do not repeat training.** [Completion and findings](../../phase3NB13Completion.md) · [technical review guide](../../phase3TreadTwoContextReview.md).

## Kaggle instructions

1. Import [phase3_NB14_TREAD_TWO_CONTEXT_Completed_Review.ipynb](../../notebooks/phase3_NB14_TREAD_TWO_CONTEXT_Completed_Review.ipynb).
2. Accelerator **None / CPU**, Internet **ON**. No tyre dataset is required. Keep at least **4 GiB free** output and about **8 GB RAM**.
3. Enable Kaggle Secret **HF_TOKEN** with write access to `Shanmuk4622/tyre-wear-study`. Do not paste its value into a notebook or file.
4. **Run All**, in order: unpack/prepare → saved-data analysis → direct HF publication → inspect figures/receipt.
5. Preserve the executed notebook and final `REVIEW_PUBLICATION.json`. Return the executed NB14 for independent HF verification.

| Cell | Expected result |
|---|---|
| Setup | 44 verified source/metadata assets; pinned CPU review ready |
| Analysis | 52 CSVs / 585,825 rows, five figures, 1,854-case explorer, 206 endpoint attention cases |
| Publish | Released-owner receipt for isolated `atlas-context-r3/review/<contract>` namespace |
| Inspect | Printed receipt/summary, context figure and explorer/table links |

Working folder: `/kaggle/working/tread_two_nb14_context_review/`. `cache/` contains verified evidence, `report/` derived artifacts, `bundle/` immutable publication staging. Original full native checkpoints are not loaded; this is a CPU evidence review with zero model forwards and zero fitting.

## Resume and troubleshooting

| Situation | Action |
|---|---|
| Analysis interrupted | Preserve working files; rerun unchanged. Valid cache files are reused and finite derived tables rebuilt atomically. |
| Session reset | Reimport the same NB14 and Run All; pinned inputs redownload. Retained working output can avoid redundant downloads. |
| HF asks for backoff / appears quiet during upload | Allow printed server waits to finish. Shared request budgeting and 80-object batches remain active; do not start another writer. |
| Stop during publication | Catchable Stop lets the completed report flush finish. Forced kernel shutdown cannot upload unsent files; rerun unchanged to reconcile/complete publication. |
| Wrong/missing Secret | Enable `HF_TOKEN` for this notebook and rerun publication. No alternate credential file is used. |
| Source/hash mismatch | Preserve files/error output. Do not edit embedded code, original fit sources or manifests to bypass the check. |
| Insufficient output space | Keep at least 4 GiB available. Do not duplicate the original image dataset/full fit mirror into the review folder. |
| Missing attention for a probe | Expected: full matrices were saved only for selected/final identity endpoints. The viewer explicitly marks other cases unavailable. |

The review publishes complete direct CSV/JSON/PNG/HTML/source records, never ZIP/NPZ. It does not promise optimizer recovery for an unfinished derived report. The completed fit remains read-only, the classifier unpromoted, both workstations unchanged and the existing-photo constraint retained.

## Module and evidence references

| Module | Owns |
|---|---|
| `review_context.py` | Pinned read cache, common audit and output tables |
| `paired_context.py` | Paired objective, exclusions, exposure and gradient checks |
| `context_checks.py` | Frozen-prior/context/attention/layout conservation |
| `render_context.py` | Figures and interactive evidence viewer |
| `publish_context_review.py` | Isolated direct-file HF publication |

[Local package proof](../context_review_validation/PACKAGE_VALIDATION.json) · [CSV inventory](../context_analysis_report/CSV_INDEX.md) · [methods](../context_analysis_report/METHODS.md) · [original 27-family schema](../../phase3TreadTwoContextEvidenceSchema.md).
