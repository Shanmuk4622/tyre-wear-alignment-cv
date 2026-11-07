# Phase 3 IEEE Access paper

## Revision 3 — reviewer response

The new draft includes the source-checked TRACE diagram and layer table,
principal-angle and Jacobian diagnostics, a train-only affine readout, and
both supplied photo biographies. It distinguishes preserved camera texture
from physically calibrated groove depth.
[Current PDF](ACCESS_latex_template_20260513/main.pdf) ·
[Reviewer response](ACCESS_latex_template_20260513/REVIEW_RESPONSE.md).

## Revision 2 — 6 November 2026

The current manuscript is **A Decoder-Aware Spatial Budget Audit for Local
Texture Evidence in Tyre Inspection**. It presents a closed-form comparison of
spatial bases under the same coefficient budget, with a paired decoder-order
diagnostic. At five modes, token PCA retains 89.47% of the original local
validation advantage and 89.01% on the historical test. Fixed moments retain
81.71% and 82.17%, respectively. Decoder-aware bases give small and inconsistent
five-mode changes relative to ordinary PCA.

[Revised PDF](ACCESS_latex_template_20260513/main.pdf) · [Editable package](ACCESS_latex_template_20260513/README.md)
· [Novelty and evidence](ACCESS_latex_template_20260513/NOVELTY_AND_EVIDENCE.md).

The new comparison uses saved final TRACE features and training-only spatial
eigensystems. Its 282,720 paired image records reconcile to 1,488 summaries.
The historical roles remain previously inspected and each has two components.
The record below describes the earlier 3 November draft in revision `7f82528`.

## Revision 1 — historical record

**3 November 2026 — complete, checked author-review draft: 15 pages, seven figures,
five tables, 18 research references and one AI-system citation.** No training or
notebook rerun is required. Numeric and visual receipts are in the audit folder.

> **Local Texture Evidence and Aggregation in Compact Tyre Inspection:
> A Reproducible Limited-Data Study**

Open the [paper PDF](ACCESS_latex_template_20260513/main.pdf) or the
[editable source and package guide](ACCESS_latex_template_20260513/README.md).
The new folder is separate from the original September manuscript.

## What the paper contributes

| Question | Evidence presented | Conclusion |
|---|---|---|
| Do good dense outputs ensure useful proxy predictions? | TREAD-ONE and serial ATLAS revisions | No; internal geometry/mask agreement coexists with poor cross-tyre transfer |
| Can TRACE learn observable texture? | Fixed local fields, controls and all 60 epochs | Local/masked utility improves, but global diversity fails every epoch |
| Does aggregation preserve the useful local evidence? | Nine fixed interventions with unchanged decoder | Retention depends on spatial structure, descriptor and component |
| Does higher rank prove better evidence? | Invertible scaling and exploratory mask-layout CKA | No; rank needs utility and nuisance controls |

```mermaid
flowchart LR
    A[Completed experiments] --> B[Pinned saved evidence]
    B --> C[Focused manuscript]
    C --> D[Figures / tables / numerical checks]
    D --> E[IEEE Access PDF]
    E --> F[Author review and declarations]
```

The primary local validation result is **61.14% retained advantage for a mean**
versus **81.76% for a position template plus five residual moments**. These are
fixed-decoder reconstruction comparisons, not classification accuracy or physical
wear measurements. TRACE remains **selected=None**.

## Package and review

| Deliverable | Link |
|---|---|
| PDF / LaTeX / figures | [Package guide](ACCESS_latex_template_20260513/README.md) |
| Scientific claims and boundaries | [Claim ledger](ACCESS_latex_template_20260513/CLAIMS_AND_LIMITATIONS.md) |
| Source / citation checking | [Citation audit](ACCESS_latex_template_20260513/CITATION_AUDIT.md) |
| Reproduction and provenance | [Reproduction guide](ACCESS_latex_template_20260513/REPRODUCIBILITY.md) |
| Remaining author decisions | [Submission checklist](ACCESS_latex_template_20260513/SUBMISSION_CHECKLIST.md) |
| Preparation history | [Writing log](ACCESS_latex_template_20260513/WRITING_LOG.md) |
| Numeric/PDF checks | [Validation](ACCESS_latex_template_20260513/audit/VALIDATION.json) |
| Latest visual inspection | [Visual review](ACCESS_latex_template_20260513/audit/VISUAL_REVIEW.md) |

Working authors follow the previously recorded Edara–Bonala line. The authors
still need to confirm authorship/contributions, approval, funding/conflicts,
permissions, related-manuscript status and the final availability statement.
There is no journal submission or guarantee of acceptance.

No new photographs, model fit, checkpoint selection, scientific-source rewrite,
workstation change, HF upload or Git push occurred. The paper assets are local;
existing fit/review evidence remains on the previously verified HF revisions.
