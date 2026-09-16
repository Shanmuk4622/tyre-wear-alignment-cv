# Current project status — 15 September 2026

This is the current-status authority for the maintained Markdown documentation.
Earlier dated plans and logs preserve history; they are not new rerun instructions.

| Area | Implemented / completed | Still requiring evidence or author action |
|---|---|---|
| Classification, S4b and S5 | Frozen implemented tracks and reports complete | Original wider tiers, H2/H3 and external-data limits remain |
| Matched geometry | HRNet and SegFormer each 3 seeds ×60 epochs; HF-verified | Only two test tyres; different supervision limits claims |
| S9 software integration | **Complete for the implemented prototype:** learned HRNet + matched SegFormer modes, image/video overlays, raw evidence and local tests | Video-domain accuracy, end-to-end component benefit, PatchCore and original full Tier8 remain open |
| S10 analysis + figures | **Report refresh complete:** methods/results/discussion, 22 visuals, Markdown + HTML, evidence manifests and appendices | Author/guide approval, venue template and permissions remain |
| Alignment | **Target-assisted software implemented:** camera-profile workflow, dual-target poses, signed outputs and synthetic checks | Real camera/fixture calibration, independent reference error and remount/runout trials remain |
| Optional app | **Implemented:** native Tread Station, local inference, comparison, video, evidence export/restore | Wider field testing and physical webcam verification remain; mobile/web deployment was not built |

The report now contains the new results, rather than only an addendum:
[read REPORT.html](report/REPORT.html) or [REPORT.md](report/REPORT.md).

## Research result

Matched HF report `bbe586c6f00cf12ae4cac8b2e9cb4f875b272abb` verifies HRNet
**1.312% width / 15.10 px** against SegFormer **1.721% / 19.80 px**, or 23.75%
lower mean horizontal point error, with 100% SegFormer coverage. HRNet is lower
on both test tyres in every seed. This is not a physical-angle, wear-F1 or
independent generalisation claim. No NB26–NB32 rerun is requested.

## Prototype evidence and why “pending” changed

The latest local `prototype/LEARNED_GEOMETRY_LOG.md`,
`prototype/results/learned-integration-check.json` and `learned-ui-check.json`
show that the previously proposed geometry integration already exists. A new
integration notebook is not a prerequisite to acknowledge that completed work.
Seed 1 / epoch 60 is used for both optional components; no test-best seed was
selected for the app. FP32 local inference and small GTX 1650 timing samples
are operational evidence, not a re-execution of the AMP benchmark.

The supplied video examples show crossed HRNet boundaries and disagreement;
software assertions passing does not establish prediction accuracy. Alignment
software tests use synthetic target images. Real physical calibration/reference
data remain missing, but describing the entire alignment feature as “deferred”
or the app as “unbuilt” is stale and incorrect.

## Next, without duplicate implementation

1. Review/document the failure cases on supplied clips and define a separate,
   labelled, tyre-disjoint video/field validation protocol before tuning.
2. If pursuing physical alignment accuracy, collect camera/fixture/reference
   measurements and repeat-remount evidence for the existing target workflow.
3. Evaluate end-to-end/component effects under frozen selection and scope;
   independently justify a healthy pool before any PatchCore claim.
4. Complete author/guide and submission-format review of the refreshed report.

The full original experiment is not declared complete merely because software
and documentation are now implemented. No new training, labels, HF upload or
prototype code change is authorised or performed by this documentation refresh.
