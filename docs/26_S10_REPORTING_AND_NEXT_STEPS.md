# S10 evidence/report package and next steps

## Current documentation handoff — 12 September 2026

The [full illustrated report](report/REPORT.html) and [Markdown version](report/REPORT.md)
are now prepared for author review, with16 visuals (14 preserved public figures,
one data-derived endpoint chart and one study diagram), generated result tables,
methods, related work, discussion, limitations and12 selected checked references.
Companions: [reproducibility](report/REPRODUCIBILITY.md),
[claims/figure audit](report/CLAIMS_AND_LIMITATIONS.md),
[submission checklist](report/SUBMISSION_CHECKLIST.md), and
[repository/documentation index](DOCUMENTATION_INDEX.md).

The frozen small package was downloaded and hash-verified locally:3.02MiB,
no datasets/checkpoints, no HF writes or new training. Upstream evidence is
unchanged. Current-facing wording now distinguishes12 capture sessions from
unverified tyre identities, selected versus final endpoints, and the actual
conformal fold coverage (only fold0 below nominal90%). TER mainly measures
tyre-versus-background attention because tread/tyre nearly coincide here.

Next is human author/guide review and the actual institution/venue template,
not another training notebook. No institution-specific template, signatures,
funding/contribution declarations or PDF pagination certification is invented.
Full original-plan gaps remain. The steps below record the preceding S10
execution and planning history; manuscript preparation is no longer pending.

## Execution record

NB19_S10_Evidence and NB20_S10_Report both ran successfully on Kaggle CPU.
Each records one successful commit and zero upload failures. Public output:
`s10/reporting-r1/2a333e2a6469905ad8cb822821ea46a357364e6d17bdd53c153c8b7e51fd217e/`.
Audit revision: `22d5a6bc9f953ba3bf2a75919edc7db3193b317b`.
NB19 pinned study inputs at `35a178b5a94878bdee95a0aa9ed8cb25cd1aeb6b`;
NB20 pinned its evidence package at `e388dd4ec0e9f522ac8a8b107f86fbeabe5c6a96`.
These different revisions reflect collection followed by report publication.

NB19 collects38 source files:25 CSV tables,10 inherited NB10R figures and
three upstream status records. Source paths, revisions, byte lengths and SHA256
hashes are recorded in `evidence_manifest.json`. Tables include classification,
quarantine, S4b, XAI selection, stress, calibration, S5 and exploratory S9 evidence.
No model checkpoints or datasets are downloaded/retrained.

NB20 adds four descriptive figures:

1. Box AP50:95 by model and fold.
2. Predicted-tread ROI macro-F1 change versus full image.
3. Fixed equal full+tyre+tread fusion change versus full image.
4. Predicted tread-mask IoU against existing manual masks.

The figure package totals14 panels:10 preserved plus4 new. The new error bars
are seed standard deviations, not confidence intervals. Mask IoU is not a
blind repeat-annotation study. Old figures8–10 are still best-epoch, energy and
per-session analyses; they do not stand in for missing video/factorial experiments.

Outputs include `REPORT.html`, `RESULTS_DRAFT.md`, `figure_manifest.json`,
`EVIDENCE_STATUS.json` and `REPORT_STATUS.json`, alongside tables/figures.
Report status is `report_package_ready_for_review`, with
`full_project_complete=false`. The draft is not a finished submission manuscript.
The preserved classification tables contain selected and final endpoints;
retain their labels rather than mixing them. H1 is legacy selected-epoch;
H2 is inconclusive/undefined; H3 remains untested.

## Completion boundary

**NB19/NB20 reporting execution is complete. No reruns required.** S10 manuscript
development and scientific/editorial review remain. The full original proposal
is not complete. HRNet/PatchCore remain deferred by user, not permanently removed
or silently counted as implemented. Extra annotation is not currently requested.
See `25_S9_FUSION_AND_REMAINING_GAPS.md` for the complete gap ledger.

## Next plan — finish the supported study for review

1. **Freeze the current reporting snapshot.** Use this package as the source of
   truth for the manuscript; preserve earlier evidence. No new training just to
   chase better results or clear a checklist.
2. **Write a full manuscript/report draft.** Abstract, research questions,
   dataset/proxy-label description, methods, experimental protocol, results,
   discussion, limitations and conclusion. The present results draft is a source
   document, not all these sections completed. Clearly distinguish image-level
   performance from independent tyre-level evidence and health/alignment claims.
3. **Review results and figures.** Choose a readable main-text subset; place
   supporting tables/figures in the appendix. Preserve negative fusion results,
   quarantine, limited calibration evidence and unresolved hypotheses. Verify
   references and figure/table citations; do not invent p-values or claim seed
   SD represents independent tyre uncertainty.
4. **Prepare the reproducibility handoff.** Notebook/run order, environment and
   source hashes, HF artifact map, included/deferred scope, and instructions to
   reproduce without overwriting completed work. Review publication rights and
   privacy before any release beyond the current authorised outputs.
5. **Final human review and venue formatting.** Check author details, references,
   wording and required institutional/submission format. Revisit optional original
   experiments only after this phase, with explicit scope and input decisions.

No new notebook or GPU execution is currently necessary for steps1–3.

## Verification

Both saved notebooks contain no error outputs and finish their final HF flush.
The independent audit checks the38 bundled files against pinned originals and
manifest hashes, seven report artifact hashes,14 decodable figure images and
complete image references in both HTML and Markdown. It does not certify every
scientific claim or final manuscript quality. No notebook changes or HF writes
are part of the verification; only project documentation is updated.
