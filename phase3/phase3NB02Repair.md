# NB02 fresh-session repair — 28 October 2026

**Current completion:** the repaired notebook has now executed successfully on Kaggle and published 18 CSV / 25 files at HF `91ccd3315bd296a76a8b583a6d0f2515ccb3d370`. Independent remote verification passes. [Completion evidence and inventory](phase3NB02Completion.md). The repair history below is preserved; no rerun is needed for this completed publication.

## Failure and cause

The supplied Kaggle execution stopped in the historical-evidence retrieval cell:

```text
FileNotFoundError: [Errno 2] No such file or directory:
/kaggle/working/treadone_csv_recovery/completion_20261028/remote_tree.json
```

The notebook's `CalledProcessError` wrapped this underlying error. The fetcher
wrote its repository metadata before creating the output directory. Local audit
work had already created that directory, which hid the fresh-session defect.
This failure preceded the historical bundle downloads and CSV publication; it
does not indicate damaged checkpoints or require retraining.

The exact failed notebook is preserved at
[executed.ipynb](completion_20261028/nb02_repair/c6c6b287249a/executed.ipynb),
SHA-256 `c6c6b287249ac4187a80322d248f46f968b37f7dd407872ef28e837a08f8ccac`.
Its saved error/output inspection is
[inspection.json](completion_20261028/nb02_repair/c6c6b287249a/inspection.json).

## Repair delivered

The current [NB02 notebook](notebooks/phase3_NB02_Audit_And_Direct_CSV.ipynb)
is revision **r2-fresh-session-directories**.

- Its setup creates `completion_20261028` before launching retrieval.
- [fetch_completed_results.py](analysis/fetch_completed_results.py) independently
  creates the output directory at the beginning of `main()`.
- [audit_and_export_csv.py](analysis/audit_and_export_csv.py) creates the CSV
  directory and its missing parents when invoked independently.
- The notebook is generated from the corrected readable sources by
  [build_csv_recovery_notebook.py](analysis/build_csv_recovery_notebook.py).
  Embedded sources match those files exactly.

The frozen training runtime, protocol, original executed training notebooks,
model selection and checkpoints are preserved. The repair only affects the
separate CPU audit/export workflow. Local legacy ZIPs are read to recover old
records; this notebook uploads individual CSV and JSON/source files, with no
ZIP upload.

## Verification

[test_nb02_fresh_session.py](analysis/test_nb02_fresh_session.py) reproduces the
original directory error using the failed notebook's actual embedded fetcher,
then executes the repaired embedded fetcher in an empty workspace. Local pinned
response fixtures replace the remote service; no credential is accessed.

The test checks a deliberately interrupted download, complete retry, cached
restart without download requests, all 11 metadata files and three historical
bundle hashes. It also runs the real CSV audit and publication preparation in
that fresh workspace against the existing read-only dataset. See
[REGRESSION.json](completion_20261028/nb02_repair/REGRESSION.json) for the executed
check results and their limits. Notebook schema/payload and protected-original
checks are recorded in
[REVIEW_VALIDATION.json](completion_20261028/REVIEW_VALIDATION.json).

The fresh-workspace run passed: **18 core CSV tables / 25 prepared files**,
including 8,820 training updates, 14,580 validation predictions and 618 test
predictions. The previous local completion review has two additional comparison
tables and optional figures/registry, accounting for its 20-table / 30-file
package. Neither count represents additional independent training data.

These checks were local CPU and simulated-service checks at repair delivery.
The later supplied Kaggle execution and independently verified publication close
that earlier pending state; see the completion record above.

## Run the repaired notebook

The following instructions are retained for reproduction or a separate recovery
need; the observed publication is already complete.

1. Import the current [NB02 file](notebooks/phase3_NB02_Audit_And_Direct_CSV.ipynb)
   into Kaggle. Confirm the opening cell says **Repair r2**; merely rerunning an
   older Kaggle copy will retain its old embedded code.
2. Set Accelerator **None / CPU**, Internet **ON**, attach
   `shanmuk4622/tire-dataset-prepared-phase2`, and enable Kaggle Secret `HF_TOKEN`
   with write access to `Shanmuk4622/tyre-wear-study`.
3. **Run All** from setup. Completed downloads in the same working directory are
   reused. No training notebook needs rerunning.
4. Confirm the final publication cell reports `published_and_verified` and
   writes `completion_20261028/CSV_PUBLICATION.json` with the new HF revision.
   If only publication fails transiently, retry that cell after the stated
   server delay; keep the recovered local CSV files.

The notebook needs approximately 390 MiB of downloads and at least 3 GiB free
working space. It retains Kaggle's existing package environment. The authorized
credential source remains Kaggle Secrets; this repair adds no alternative
credential lookup.
