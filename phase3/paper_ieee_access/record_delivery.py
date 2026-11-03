"""Add a scoped, dated manuscript record without rewriting scientific history."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
P3=ROOT.parent
marker='IEEE_ACCESS_PAPER_20261103'
def insert(name,body,title=None):
    p=P3/name;text=p.read_text(encoding='utf-8')
    start=f'<!-- {marker} -->';end=f'<!-- END_{marker} -->'
    block=start+'\n'+body.strip()+'\n'+end+'\n\n'
    if start in text:
        i=text.index(start);j=text.index(end,i)+len(end)
        text=text[:i]+block.rstrip()+text[j:]
    else:
        heading,sep,rest=text.partition('\n')
        text=(title or heading)+'\n\n'+block+rest.lstrip('\n')
    p.write_text(text,encoding='utf-8')
common='''## Latest — separate IEEE Access author-review paper

**3 November 2026:** [Paper and handoff](phase3PaperIEEEAccess.md),
[PDF](paper_ieee_access/main.pdf), and [editable source](paper_ieee_access/README.md)
are prepared from completed Phase 3 evidence. The focused contribution is local
texture utility versus global aggregation, with all failed gates and data limits
retained. No new fit, photo, workstation change, HF publication or Git push.

Next is author review and the [submission checklist](paper_ieee_access/SUBMISSION_CHECKLIST.md),
not another notebook. Original manuscript, measured sources and selections remain unchanged.'''
for name in ['AGENTS.md','README.md','phase3Progress.md','phase3Understanding.md']:
    insert(name,common)
insert('phase3NextSteps.md',common+'\n\nEarlier notebook/design directions below are historical for this completed cycle.',
       '# Phase 3 — next steps: manuscript author review')
insert('phase3OpenItems.md','''## Current — manuscript decisions, not a new fit

The separate [IEEE Access paper](phase3PaperIEEEAccess.md) is prepared. Next review
its science and confirm authorship/contributions, author approval, funding/conflicts,
image/data permissions, institutional requirements, related-manuscript status,
public availability and submission-account details. [Full checklist](paper_ieee_access/SUBMISSION_CHECKLIST.md).

No new photo collection, notebook rerun or training is required for this review.
GEO and TRACE remain unselected research; both workstation revisions stay deferred.''')
insert('phase3SessionLog.md','''## 2–3 November 2026 — separate IEEE Access manuscript

1. Created `paper_ieee_access/`; copied compact evidence and template assets with hashes.
2. Wrote nine substantive sections, equations, references, AI disclosure and working-author biographies.
3. Generated seven scientific figures and five tables, including all 60 TRACE epochs and two deterministic saved-image examples.
4. Separated primary sweep decoder reads from additional bridge reads; preserved original rank/gate arithmetic and selected=None.
5. Checked primary citations and the current IEEE template; core class/style hashes match.
6. Compiled the paper and checked numeric tables, gains, execution counts, citations, original-file preservation and PDF layout.
7. Added claim ledger, reproducibility guide, citation audit, writing log and author-action checklist.
8. Kept new paper/aggregation assets local. No new fit, photo, HF upload, Git push, or workstation change.

Final numeric and visual receipts are in the [paper package](paper_ieee_access/README.md).
Next: author review and genuine declaration/availability decisions.''')
insert('phase3Decisions.md','''## Manuscript decisions — 3 November 2026

| ID | Decision | Boundary |
|---|---|---|
| P3-D195 | Isolate the IEEE Access paper under Phase 3 | Preserve original manuscript and all measured sources |
| P3-D196 | Center local evidence and aggregation | Serial model revisions motivate; not a universal model-superiority claim |
| P3-D197 | Retain selected=None and existing-data limits | No new fit, photos, gate relaxation or blind-test claim |
| P3-D198 | Generate figures from verified saved evidence | Deterministic examples, unchanged predictions, direct cell/epoch CSVs |
| P3-D199 | Distinguish primary sweep and bridge readout counts | No claim of zero-computation review; zero new image-model inference |
| P3-D200 | Use primary citations and explicit AI assistance disclosure | No invented references, approval, grant or permission |
| P3-D201 | Deliver editable source, PDF and audit/checklist together | Actual author/submission decisions remain open |
| P3-D202 | Retain curated paper assets in Git, exclude build scratch | No staging, commit, history rewrite or push |
''')
insert('phase3EvidenceRegister.md','''## IEEE Access manuscript evidence

| Record | Scope |
|---|---|
| [Paper PDF](paper_ieee_access/main.pdf) /[source](paper_ieee_access/main.tex) | Complete author-review manuscript |
| [Copied scientific inputs](paper_ieee_access/audit/EVIDENCE_FILES.json) | Original byte identities and source paths |
| [Additional inputs](paper_ieee_access/audit/ADDITIONAL_INPUTS.json) | Epoch records and deterministic atlas/cell evidence |
| [Claim ledger](paper_ieee_access/CLAIMS_AND_LIMITATIONS.md) | Endpoint, denominator and permitted inference |
| [Current template check](paper_ieee_access/audit/CURRENT_TEMPLATE_CHECK.json) | Official core files match supplied template |
| [Numeric/PDF validation](paper_ieee_access/audit/VALIDATION.json) | Tables, headlines, all epochs, citations and preservation |
| [Visual review](paper_ieee_access/audit/VISUAL_REVIEW.md) | Latest rendered-page inspection |
''')
insert('phase3DocumentationInventory.md','''## IEEE Access paper documentation

| Document | Purpose |
|---|---|
| [Phase 3 paper handoff](phase3PaperIEEEAccess.md) | Scope, result and next author actions |
| [Package guide](paper_ieee_access/README.md) | PDF/source/figures and build navigation |
| [Claim ledger](paper_ieee_access/CLAIMS_AND_LIMITATIONS.md) | Scientific evidence and excluded claims |
| [Reproducibility](paper_ieee_access/REPRODUCIBILITY.md) | Build commands, identities and audit boundaries |
| [Citation audit](paper_ieee_access/CITATION_AUDIT.md) | Primary literature and current venue rules |
| [Submission checklist](paper_ieee_access/SUBMISSION_CHECKLIST.md) | Finished preparation versus real author decisions |
| [Writing log](paper_ieee_access/WRITING_LOG.md) | Editorial and preservation decisions |
| [Visual review](paper_ieee_access/audit/VISUAL_REVIEW.md) | Final PDF page QA |
''')
insert('phase3RepositoryMap.md','''## Added — separate IEEE Access manuscript

`paper_ieee_access/` contains the new LaTeX/PDF paper, vector/PNG figures,
compact direct evidence, generators and audits. Its `.build/`, `preview/` and TeX
intermediates are ignored. [Package guide](paper_ieee_access/README.md).
The original `../ACCESS_latex_template_20260513/` manuscript is unchanged.''')
print('11 maintained Phase 3 records updated; scientific reports and sources preserved')
