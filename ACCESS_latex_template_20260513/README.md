# Tyre inspection paper — author-review draft

Open `main.pdf` to read the paper. Edit `main.tex` and `sections/*.tex`.
The supplied `access.tex` is the original template example, not the manuscript.

**Title:** From Mileage Proxies to Tread Boundaries: An Auditable Pilot Study of Vision-Based Tyre Inspection

The draft presents the recorded classification, localisation, matched boundary,
and workstation results. It distinguishes mileage labels from physical wear,
selected checkpoints from fixed-final results, and image geometry from physical
alignment. It does not claim an independent field trial or calibrated accuracy.

## Compile

For Overleaf, upload `tyre-paper-source.zip`, set the main document to `main.tex`,
and select pdfLaTeX. The package includes the supplied class, bibliography style,
fonts, plots, screenshots, and generated tables. Standard LaTeX packages are
provided by the TeX installation. Overleaf compilation itself has not been tested.

The delivered PDF was compiled locally with Tectonic 0.17.0. `engine_compat.tex`
provides XeTeX colour compatibility without editing the supplied class; pdfLaTeX
uses the original spot-colour implementation. Two explicit font aliases and a
separate abstract width address requests made by the supplied template.

In this repository, use PowerShell:

```powershell
conda activate cv_conda
./ACCESS_latex_template_20260513/build.ps1
```

The script uses Tectonic if available (including the local ignored compiler
cache), otherwise pdfLaTeX and BibTeX. Python verification needs `pypdf`.
For a standalone extracted package, run `./build.ps1` from that folder with a
compiler installed. First-time Tectonic use may download standard packages.

To regenerate tables and analytic figures from the repository evidence:

```powershell
conda activate cv_conda
./ACCESS_latex_template_20260513/build.ps1 -RefreshFigures
```

This requires the full repository evidence and Python `numpy`/`matplotlib`.
It performs no training. The standalone ZIP already includes the generated
assets and does not contain datasets, model weights, or dependency caches.

## Verification records

- `CITATION_AUDIT.md`: primary-source citation checks and claim boundaries.
- `NUMERICAL_AUDIT.json`: recomputed headline results and source-file hashes.
- `VERIFICATION.json`: checks for this exact delivered PDF, including its hash.
- `PACKAGE_VERIFICATION.json`: isolated source-package compilation check.
- `verify_paper.py`: citation, cross-reference, build, PDF, and evidence checks.

The manuscript has 16 references, 8 figures, and 8 tables. Original screenshots
are reproduced without changing predictions. Plots use saved results; the
workflow figure is a schematic. Local PDF review checks every rendered page.
Remaining underfull-box warnings concern spacing, not missing content or
unresolved citations. Recompiling clears the recorded visual-review flag until
the new PDF has been reviewed. Automated checks are not independent scientific
replication or a guarantee of publication suitability.

## Required author review before submission

1. Confirm the four names, order, affiliation, and email imported from `CITATION.cff`.
2. Supply funding, competing-interest, and contribution statements.
3. Confirm image collection, image publication, data redistribution, and model-use permissions.
4. Review all scientific interpretations, the AI-assistance disclosure, and the limitations.
5. Recheck the journal's current submission requirements and replace draft declarations
   only with author-confirmed information. Add biographies if requested by the journal.

No acceptance date, publication DOI, ethics approval, funding declaration, or
absence of conflicts has been invented. The original supplied template files
remain in this directory; unrelated template example images are excluded from
the clean manuscript ZIP.
