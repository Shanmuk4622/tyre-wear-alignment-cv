# NB08–NB10 completion audit — 2026-09-09

<!-- current-status:start -->
> **Current status (15 September 2026):** [Completed work and remaining validation](CURRENT_STATUS.md). The report is refreshed; the app, learned-geometry integration and target-assisted alignment software exist. Dated plans below retain their original context.
<!-- current-status:end -->

**Follow-up:** `20_FULL_PLAN_CLOSURE.md` corrects the broader-stage omissions
and stale S1 status. NB10R is a new recovery notebook; this document remains
the audit of the original public snapshot. S5/S9 were not completed or cut
merely because the classifier notebook sequence finished.

Verified public Hugging Face dataset `Shanmuk4622/tyre-wear-study` at commit
`7c5b6461815a78aae589085d0152eaa6ae9995e1`. The audit pinned all reads to that
commit. Reproduce the checks with `scripts/audit_completed_notebooks.py`
(read-only; it checks the current commit when rerun).

All three submitted notebooks reached their final cells without a recorded
exception. NB08 loaded tyrelib v12; NB09 and NB10 loaded v9. Their completed
outputs were preserved. No remote records or notebooks were changed by this audit.

## Published artifacts

| Notebook | Verified output | Status |
|---|---|---|
| NB08 | `tables/stress_tests.csv`: 63 unique run/intervention pairs; 9 per-run `stress/results.csv` files match the aggregate | Complete |
| NB08 | 3 completed 12-epoch control histories and revisioned gate audit | Published; gate passes |
| NB09 | 27 readable prediction parquets, 3,762 rows; unique images within each run and finite probabilities summing to one | Complete |
| NB09 | `ensemble_metrics.csv`: 39 rows (27 single, 9 seed ensembles, 3 architecture ensembles) | Complete |
| NB09 | `tta.csv`: 6 rows; `calibration.csv`: 6 rows; `conformal.csv`: 3 rows | Complete |
| NB10 | `master_architectures.csv`: 17 rows; `stage_a_quarantined.csv`: 9 rows; `hypothesis_outcomes.csv`: 3 rows | Published |
| NB10 | Figures 1, 2, 4, 5, 6, 7, 8, 9, 10: all PNGs readable | 9 of 10 figures |
| NB10 | Figure 3 saliency panels | Missing: no `analysis/xai_examples/` inputs |

`tables/stage_b_effects.csv` also contains all 108 Stage-B effect rows.
NB06 was previously verified at 108/108 completed; NB07's public gate remains
the selected architecture source. NB09 restored the previously missing
`a-regnety016-base-f2-s2/per_sample/predictions.parquet`.

## NB08 results and provenance

The stress table covers three architectures × three seeds × seven interventions,
each evaluated on 128 fold-1 validation images. All reported stress metrics
are finite. Marking/damage interventions have zero measured deltas in the
saved results; this alone does not prove that those regions are unimportant.

The current control audit has final-epoch F1 values **0.263448, 0.351665,
0.510438**, mean **0.375184**, below the unchanged 0.45 gate. The current
selected-epoch mean is **0.549365**. The prior 2026-09-08 audit recorded mean
0.353076 / selected mean 0.498079. HF's same fold-1 run ID now has a later
status (`2026-09-08T17:37:58Z`, acct1), replacing the earlier acct2 result.
The latest control CSV agrees with all three current epoch histories. Keep
the earlier figures as historical observations; do not mix the two snapshots.
The reason for the fold-1 replacement is not established by this audit.

The gate follows docs/06's fixed-budget rule. Passing this small aggregate
diagnostic does not establish leakage-free folds. NB08's submitted session
was configured as worker 1/3; its inference loop is unsharded, so concurrent
copies can repeat evaluations. No rerun is needed just to change that setting.

## NB09 interpretation limits

The recorded conformal coverage is **86.89%, 97.56%, 93.94%** on folds 0, 1,
and 2. The roadmap's 88–92% target is therefore not met across folds. These
are empirical results from small image subsets, not a verified new-tyre
coverage guarantee: the three-way split separates images rather than tyres,
and predictions come from validation-selected checkpoints.

The saved `abstain_rate` counts only sets larger than one. Empty sets are not
counted, so it must not be described as the total rate of uncertain outcomes.
Mean set sizes below one on folds 0 and 2 establish that empty sets occurred.
Temperature calibration produced finite tables; the saved tensor-to-scalar
warning did not stop the run. None of these findings requires retraining
NB06 to document them correctly.

## NB10 interpretation and remaining work

* **H1:** reported unsupported under the implemented correlation comparison
  (10 architectures; 0.427095 versus the accuracy reference 0.907435).
* **H2:** both correlations are undefined. The CSV says `supported=False`
  because the implementation converts comparisons with NaN to False. Report
  H2 as **undefined/inconclusive**, not as a tested negative result. The zero
  intervention deltas provide no variance for this correlation.
* **H3:** not testable with the current architecture set; no preregistered
  fine-grained architectures were trained.
* **Figure 3:** explicitly skipped; supply saliency examples before calling
  the ten-figure package complete.
* The master table excludes the nine quarantined ConvNeXt-V2-S labels, but
  Figure 10's input loop reads all Stage-A histories without that filter.
  Review/rebuild that figure before claiming all figures apply the quarantine.
* Several NB10 plots use validation-selected best metrics. Preserve that
  labelling and follow docs/06's final-epoch rule for performance claims.

The next phase is results review and targeted analysis corrections, followed
by the write-up. Notebook execution is complete; scientific/reporting closure
and the optional broader stages are not established by that fact. This turn
records the issues without modifying completed scientific artifacts.
