# Repository structure and maintenance

<!-- current-status:start -->
> **Current status (15 September 2026):** [Completed work and remaining validation](CURRENT_STATUS.md). The report is refreshed; the app, learned-geometry integration and target-assisted alignment software exist. Dated plans below retain their original context.
<!-- current-status:end -->

<!-- current-status:start -->
> **Current status (15 September 2026):** [Completed work and remaining validation](CURRENT_STATUS.md). The report is refreshed; the app, learned-geometry integration and target-assisted alignment software exist. Dated plans below retain their original context.
<!-- current-status:end -->

15 September comparison addition: NB30–NB32 are built by
`tyrelib/build_segformer_matched_notebooks.py`, with readable runtime in
`tyrelib/segformer_matched.py` and checks in `scripts/verify_segformer_matched.py`.
Only new notebooks are generated; executed HRNet notebooks and prototype changes
are preserved. [Matched comparison guide](35_MATCHED_SEGFORMER_COMPARISON.md).

14 September addition: `prototype/` contains the user's native inspection and
geometry/calibration work; it is separate from training. NB21/NB22 and
`tyrelib/s9_pilot.py` add the small annotation pilot; generated packages are in
`outputs/s9_pilot/`, test-only fixtures in `outputs/s9_pilot_qa/`. Do not treat
software-test annotations as human labels. [Pilot guide](27_S9_SMALL_ANNOTATION_PILOT.md).

Organisation is additive. Existing notebooks, embedded sources and execution archives were not moved or deleted: their paths and saved outputs are part of reproducibility.

```text
Tyre/
├── README.md                  Entry point and project identity
├── PROGRESS.md                Current status plus dated history
├── ENVIRONMENT.md             Local environment guidance
├── environment.yml            Development environment, not every Kaggle runtime
├── CITATION.cff / LICENSE      Citation and project code licence
├── docs/
│   ├── DOCUMENTATION_INDEX.md Reading map
│   ├── REPOSITORY_GUIDE.md     This guide
│   ├── 00_...26_...md          Designs, protocols and audits
│   ├── LOGBOOK.md             Dated history
│   └── report/
│       ├── REPORT.html        Designed offline report
│       ├── REPORT.md          Generated portable manuscript
│       ├── manuscript.source.md  Editable source
│       ├── REFERENCES.md
│       ├── REPRODUCIBILITY.md
│       ├── CLAIMS_AND_LIMITATIONS.md
│       ├── SUBMISSION_CHECKLIST.md
│       ├── BUILD_PROVENANCE.json / VALIDATION.json
│       ├── assets/            New diagram and data-derived endpoint chart
│       └── evidence/          Unmodified pinned small HF package
├── notebooks/                 Kaggle deliverables and saved outputs
│   └── execution_archives/    Prior execution/error evidence
├── tyrelib/                   Current training/reporting/notebook builders
├── tyre_study/                Earlier project implementation
├── scripts/                   Audit, validation and documentation utilities
└── outputs/                   Local diagnostics/previews, not HF truth
```

## Maintenance rules

**Write current results once, link elsewhere.** Use the report for detailed narrative, `PROGRESS.md` for current status, dated audits for verification depth. Add a logbook entry when changing interpretation. Historical repair messages must not become today's run instructions.

**Separate evidence and presentation.** Do not edit `docs/report/evidence/` to improve results or update upstream status. Edit manuscript source or add an explicitly derived artifact, rebuild, and verify. Cached `evidence/REPORT.html` is the shorter upstream draft; the full report is one directory above it.

**Preserve notebook paths.** User-facing training remains Kaggle-compatible `.ipynb` with embedded source and HF persistence. This documentation task changes no training recipe, checkpoint, notebook or remote data. Moving notebooks requires a separate dependency review.

**Keep large data out of the manuscript bundle.** Small tables/figures belong beside the report. Use pinned manifests/links for datasets and weights. A public repository is not permission to redistribute every external asset.

**Protect secrets.** `.env` is local and Git-ignored. Never copy it into outputs, notebook cells, a release archive or documentation. Examples show variable names, not values.

## Build and verify

```powershell
python scripts/build_project_report.py
python scripts/verify_project_documentation.py
```

Both commands are local and do not write to HF. Optional `scripts/fetch_report_evidence.py` reads only the bounded pinned small package. Other scientific audit scripts may have different download costs; inspect scope before running them.

## Why older directories remain

Deleting `tyre_study/`, execution archives or local outputs without a dependency/provenance review could erase useful evidence or break references. A future cleanup can classify individual files. This handoff improves navigation without speculative deletion, and preserves all unrelated user changes.
