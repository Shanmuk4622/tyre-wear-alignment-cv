# NB06 — CPU review, failure atlas and direct HF publication

**Completion — 29 October 2026:** all six cells and the 35-file HF publication
are independently verified. [Completion, provenance and findings](phase3NB06Completion.md).
**No NB06 rerun is needed.** The clean-delivery identity and run procedure below
are retained as history. The next milestone is the [data/protocol gate](phase3ProspectiveDataGate.md).

**29 October 2026.** This is the next step within TREAD-ONE after the
[completed NB05 audit](phase3NB05Completion.md). No further model fit or forward
inference is needed: the source training predictions, validation trajectories,
task gradients, native endpoint maps and recovery state were already retained.

## State and purpose

| Item | Current evidence |
|---|---|
| NB05 | Complete on both T4s; 3,515 pinned files / 1,561,734 core rows independently checked |
| Candidate | Selected test BA 17.073% vs baseline 50%; point error 1.885% vs 1.067% width |
| New local review | Completed; ten derived CSV tables, three static figures, complete 103-image/four-endpoint explorer |
| Published HF review | **35 direct files / ten CSV tables, independently verified**; [receipt](revision2_validation/nb06_publication/PUBLICATION.json) |
| Original clean NB06 delivery | 516,785 bytes; 36 embedded files; historical source-package identity |
| Original clean-delivery SHA-256 | `6817b3a5d53ae42ca2430d29cfd2d78083e75475b53c008fa41a100f58cd718c` |
| New training/inference | Zero optimizer updates / zero model forward passes |
| New independent tyres | Zero; user-assisted collection remains later |

Notebook: [phase3_NB06_TREAD_ONE_Paired_Review_And_Failure_Atlas.ipynb](notebooks/phase3_NB06_TREAD_ONE_Paired_Review_And_Failure_Atlas.ipynb).
Open the already generated [interactive explorer](revision2_validation/nb05_publication/analysis/TREAD_ONE_NB05_Comparison_Explorer.html)
or [endpoint table](revision2_validation/nb05_publication/analysis/endpoint_metrics.csv)
locally now. NB06 provides the portable Kaggle reproduction and authorized HF write.

## Kaggle instructions

1. Accelerator **None / CPU**; Internet **ON**. No T4 allocation is required.
2. Attach exactly one expanded `shanmuk4622/tire-dataset-prepared-phase2` dataset.
3. Keep at least **6 GiB free**. Final source evidence is about 2.05 GB; cached
   hard-link appearances count more logically. Smaller preflight fixtures and
   the approximately 19.2 MB self-contained review add overhead. Input images
   remain read-only. Do not assume a 1 TB scratch volume exists.
4. Enable **HF_TOKEN** from Kaggle Secrets with write access to
   `Shanmuk4622/tyre-wear-study`. It is read only by the publication cell.
5. **Run All**. No package install, upgrade/downgrade or model training.

| Stage | What happens | Restart behavior |
|---|---|---|
| Setup | Extract exact source payload and preserved executed NB05 | Local session lock prevents duplicate setup/steps |
| Retrieve | Pinned manifests, all published CSV/native objects, SHA-256 and Git/LFS checks | Reuse complete verified objects; an incomplete object is fetched again |
| Recovery/science | Both native preflights, full CSV recalculation and raw mask/evidence comparison | Read-only recomputation; no optimizer or forward inference |
| Review | Ten derived CSVs, three figures and all-image HTML explorer | Derived files are recomputed from fixed sources |
| Publish | Additive direct-file immutable snapshot, ownership/CAS, verify then select manifest | Retry same cell, deduplicate unchanged objects; catchable Stop/error attempts immediate flush |

The read stages explicitly access public revisions without alternate credential
files. Publication uses the explicit Kaggle secret, the frozen bounded publisher
and a separate review namespace:
`phase3/tread-one/r2/review/paired-seed1/<analysis-contract>`.
The contract pins source publications, analysis source hashes and the actual CPU
software environment; its hash can differ between the local and Kaggle runtime.

The local prepared contract is `1f648f415feeb48349786e6327b5939220c224389527fad00e2f04bbf6a49b8a`. It is an analysis identity,
not a new training run or replacement for the original measured study identity.
Training files remain in their original namespaces. CSV, JSON, PNG, HTML and
readable analysis sources are published individually; no ZIP/NPZ/database or
model-weight upload is performed. The frozen training implementations remain
available at their original pinned publications.

## Evidence dictionary for the ten post-hoc tables

| CSV | Rows | Meaning / units |
|---|---:|---|
| endpoint_metrics | 4 | Declared selected/final metrics; accuracy/Dice and error/width are fractions |
| validation_trajectory | 120 | All 60 cohort validation records per condition; selection utility is unchanged |
| epoch_training_gradient | 120 | Augmented training accuracy, conflict/projection counts, cosine, clipping and measured seconds |
| endpoint_class_metrics | 12 | Support/predicted/correct counts and recall/precision; absent support is blank |
| endpoint_tyre_metrics | 8 | One row per condition/endpoint/test tyre; actual subject grouping |
| calibration_reliability | 40 | Ten confidence bins × four endpoints; counts/correct/confidence sums |
| selective_risk | 84 | Fixed 0–1 confidence thresholds; coverage/error on this inspected cohort only |
| validation_prediction_stability | 162 | Sixty-epoch correct counts and consecutive prediction flips per validation image |
| paired_image_deltas | 206 | Per-image condition deltas for selected/final endpoints; NLL/Dice/point error |
| resource_timing_summary | 2 | Retained worker-active seconds, sampled memory bytes and free disk; incomplete-runtime limits explicit |

Repeated image/epoch/point rows are not independent tyres. Blank means
unavailable/not applicable. No missing mid target examples, continuous energy,
new human labels or historical unlogged measurements are invented.

## Validation and limits

Local actual pinned-file/native/CSV audits passed. The fresh-workspace CPU test
uses the **actual embedded NB06 payload**, disables CUDA/network, supplies the
existing prepared dataset explicitly in its harness and reuses verified pinned
objects. It recomputes the audit, all ten CSVs, figures and explorer; resulting
review artifacts match the local originals. A fake HF service exercises the
35-file publication/ambiguous-response path with no real write. See
[NB06 validation](revision2_validation/nb06_validation/VALIDATION.json).
Those were the local acceptance boundaries. The later [NB06 completion audit](phase3NB06Completion.md) now independently verifies actual Kaggle execution and live publication.
The real preflight read audit is a separate passed receipt.

Static scientific figures were visually inspected. Explorer JavaScript syntax,
all 1,854 embedded image blocks and expected shapes are checked. Browser layout
and interaction are not claimed as visually verified. Download/open the HTML
for local offline review; it needs no external scripts or image server.

On Stop, keep the session open for safe cleanup/backoff. Forced kernel death
cannot execute a flush. An unexpired owner remains protected; stop any older
unused writer cleanly or wait for its lease. Do not delete OWNER.json. Preserve
the executed NB06, printed receipt and logs after completion.

## After the review

Use the [empty intake pack](data_intake/README.md) with the user's later help to
register genuinely independent tyres, sessions and annotation provenance before
a prospective tyre-disjoint protocol. The current projected candidate is not
installed or promoted. A later model design is a separate motivated decision;
this notebook does not launch another seed or search for a favorable test result.
