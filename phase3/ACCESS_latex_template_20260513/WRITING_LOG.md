# Manuscript preparation record

## Figure 1 monochrome schematic

The final figure uses two aligned, bottom-to-top computation paths and a
separate axial-block detail. Square operator boxes distinguish the learned
modules from deterministic steps. The explicit summation nodes, mask-token
replacement rules, and identity skip follow the executed TRACE source.
[Editable schematic](figures/architecture.svg).

## Figure 1 design revision

Replaced the compact computation graph with three panels. The overview traces
the local decoder bypass, the masked-input boundary, and the auxiliary global
readout. The two detail panels expose the four tile encoder stages and the
actual parallel depthwise convolutions, per-channel scale, and identity skip
inside an axial block. Parameter sharing is marked by module letters, and
original-support metadata has its own dashed route.
[Editable figure](figures/architecture.svg) · [source-checked builder](build_trace_figure.py).

The design uses the layer and residual-path legibility of the original
Transformer diagram as a visual reference.
[Vaswani et al., Figure 1](https://arxiv.org/pdf/1706.03762#page=3).

## Revision 3

The reviewer revision adds source-checked architecture, measured subspace
alignment, a train-only affine readout, eight figure refinements, a concise
archival account, and the supplied photo biographies.
[Reviewer response](REVIEW_RESPONSE.md) · [Migration inventory](audit/MIGRATION_INPUTS.json).

## Revision 2 — 6 November 2026

- Reframed the paper around an equal-budget spatial audit and explicit decoder
  order. Compared with primary work on frequency pooling, token pooling, edge
  compression, and gradient-based subspaces.
- Recorded the post-hoc extension protocol before the comparison. Evaluated
  train-only basis fitting, seven budgets, 16 fixed random seeds, both saved
  views, both decoder branches, and six train-component omissions.
- Promoted the fitted-versus-fixed result to the abstract and conclusion.
  Decoder-aware variants and their inconsistent five-mode improvements are
  reported in the same table as ordinary PCA.
- Retained direct image, descriptor, component, fit, and summary evidence.
  The detailed rows reconcile independently against every reported role summary.
- Updated the PDF, editable source, figure assets, citation audit, and Phase 3
  handoff. The earlier paper is recorded in repository revision `7f82528`.

## Scope and status

| Field | Record |
|---|---|
| User authorization | Write a complete IEEE Access-style paper in a separate Phase 3 directory |
| Start / completion | Started 2 November; author-review draft prepared 3 November 2026 |
| Main question | How much useful local texture evidence survives spatial aggregation? |
| Primary evidence | Completed TRACE final endpoint and nine saved-feature interventions |
| Development context | TREAD-ONE, shared-gradient diagnostic, ATLAS/LC/context, GEO |
| Existing-data limit | 570 images; no acquisition, new fit or new workstation work |
| Working author line | Edara then Bonala, carried from the previously confirmed manuscript record |
| Publication state | Local author-review package; no submission, Git push or HF upload |

## Editorial decisions

1. Use a focused representation-analysis paper instead of calling every serial
   revision a successful new architecture. Prior failures motivate the question.
2. Preserve the original failed TRACE/GEO selection. The fixed final endpoint is
   diagnostic, not a retroactively selected deployment model.
3. Separate reconstruction utility from proxy classification and physical wear.
   The dataset does not supply calibrated depth or safety labels.
4. State the independent-subject and repeated-test-use limitations prominently.
   Do not manufacture confidence intervals from dependent photographs.
5. Report all nine variants with the same frozen decoder, support and denominator.
   No test-best aggregation becomes a new trained-model claim.
6. Label mask-layout CKA as an exploratory addendum. Keep invertible scaling a
   diagnostic control, never a way to pass the old rank gate.
7. Generate new publication figures from compact, byte-verified evidence.
   Two examples use lexical-first validation identifiers per observed class.
8. Cite primary literature and disclose substantive AI writing/code assistance.
   Working authorship is not represented as fresh author approval.

## Revisions during preparation

| Issue found | Resolution |
|---|---|
| First data table exceeded a column | Use a column-width table with wrapped labels |
| Generated-row input conflicted with a final table rule | Generate a preamble macro, preserving numeric rows |
| Long revision hashes exceeded the column | Use breakable identifier formatting |
| Plot legends touched bars/curves | Move legends outside the data panels |
| Single orphaned section heading | Reserve space before the data section |
| A short final page and repeated discussion | Tighten repeated prose and reduce the gate table footprint |
| Primary decoder count could be read as all computation | Explicitly add the separate bridge decoder/projector reads |
| Rank numerical floor differed subtly from original gates | State saved-analysis floor separately and preserve original treatment |
| Compound author family name in an old citation | Retain Ristić Durrant together in the new bibliography only |
| Old template identity could be stale | Compare current official core files, all three match |

## Relationship to the original manuscript

`../../ACCESS_latex_template_20260513/` addresses checkpoint sensitivity, region
interventions and boundary localization. It remains byte-identical to the initial
paper-task baseline. This paper reuses the original photographs and task
components but centers the later custom texture model and aggregation analysis.
It does not recast old original-study experiments as newly run Phase 3 results.

The available previous record describes an internal author-review manuscript.
Its actual current submission/publication status is an author-checklist item;
this task does not assert that status on the authors' behalf.

## Remaining actions

Author scientific review, author/declaration/permission confirmation, and a final
availability/submission decision. All technical preparation and QA are recorded
in the local package. No additional notebook execution is requested.
