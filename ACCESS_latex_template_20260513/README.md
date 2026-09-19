# Revised tyre inspection paper — 19 September 2026

The 19 September editorial pass replaces unnecessary acquisition timestamps with
a concise description of manual collection and limited acquisition diversity.
It strengthens the contribution across the abstract, introduction, discussion,
and conclusion, and removes manuscript-editing commentary from the scientific
narrative. Experimental results and essential study limitations are unchanged.

Read `main.pdf`; edit `main.tex` and `sections/*.tex`. `access.tex` remains the
original supplied template example and is not the manuscript.

**Title:** Evaluating Vision-Based Tyre Inspection: Checkpoint Sensitivity,
Region Interventions, and Boundary Localisation

This revision focuses on three empirical questions. It adds fold-specific ordinal
metrics, a split-membership audit, explanation-threshold sensitivity, fold-specific
fusion effects, and per-image/per-location boundary analyses. The literature
comparison covers ten related application studies; the bibliography has 31 entries.
`REVIEW_RESPONSE.md` addresses the supplied review point by point, including what
still needs new data or training. No new neural-network training was performed.

## Read and edit

- `main.pdf`: compiled author-review manuscript.
- `main.tex`: title, authors, abstract, template settings and section order.
- `sections/`: body, related-study table, appendices and factual author biographies.
- `references.bib`: verified citation entries.
- `REVIEW_RESPONSE.md`: changes, new analyses, unresolved evidence and author actions.
- `CITATION_AUDIT.md`: original audit followed by the superseding revision audit.
- `NUMERICAL_AUDIT.json`: original aggregates and source hashes.
- `REVISION_ANALYSIS.json`: revised diagnostics and their source hashes.
- `VERIFICATION.json`: build/content checks and exact delivered PDF hash.
- `PACKAGE_VERIFICATION.json`: isolated source-package compile check.

## Build locally

```powershell
conda activate cv_conda
./ACCESS_latex_template_20260513/build.ps1
```

The delivered PDF uses Tectonic 0.17.0 with the supplied IEEE Access class and
fonts. The script uses Tectonic (including the existing ignored local cache), or
pdfLaTeX/BibTeX when available. Python verification requires `pypdf`.

To regenerate all tables and plots from repository evidence:

```powershell
conda activate cv_conda
./ACCESS_latex_template_20260513/build.ps1 -RefreshFigures
```

This runs `build_assets.py` and `revision_analysis.py` using `numpy`, `pandas`, and
`matplotlib`; it performs no training. It needs the repository's frozen report
evidence as well as the downloaded `revision_evidence` files. Hugging Face source
URLs, pinned revision, and SHA-256 hashes are in `revision_evidence/manifest.json`.
No network calls are made by the analysis or build scripts.

## Overleaf / standalone source

Upload `tyre-paper-source.zip`, select `main.tex` as the main document, and use
pdfLaTeX. Standard packages come from the TeX installation. The source ZIP contains
all generated manuscript assets and template dependencies; the locally extracted
package is compiled separately with Tectonic as a portability check. The Overleaf
service itself has not been tested.

The supplied class is unchanged. The compatibility file supports XeTeX colour
handling; the main file explicitly handles template font aliases, abstract width,
biography spacing, and draft footers. No dataset, model weights, dependency cache,
or training notebook is included in the manuscript ZIP.

## Before submission

Author order, positions, affiliations, emails, and no external funding are updated
from the user's 18 September confirmation. Still confirm the proposed corresponding
author, proposed contribution assignments, conflicts of interest, and image
permissions. Expand the factual biographies if desired. The manuscript explicitly
marks these outstanding declarations and discloses AI assistance in Acknowledgment.

Do not interpret the revision as new independent physical or external validation.
Rotated geometry training, shared-backbone ablation, independent annotation checks,
and labelled video/physical reference data remain future work. No acceptance score
is assigned. Rebuilding resets the visual-review flag until the new PDF is checked.
Original review comments and previous manuscript copies remain locally preserved.
