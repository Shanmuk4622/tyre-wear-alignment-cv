"""Document actual completed r2 results and the next saved-evidence publication."""
import json,sys
from pathlib import Path
import pandas as pd
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];DONE=Path(json.loads((ROOT/'tread_two/nb11_completion/LATEST.json').read_text())['path'])
v=json.loads((DONE/'VERIFICATION.json').read_text());assert v['status']=='passed'
REPORT=ROOT/'tread_two/lc_analysis_report';a=json.loads((REPORT/'ANALYSIS.json').read_text())
test=json.loads((ROOT/'tread_two/lc_review_validation/PACKAGE_VALIDATION.json').read_text());assert test['status']=='passed' and test['packaged_fresh_analysis_equal']
nb=json.loads((ROOT/'tread_two/lc_review_validation/NOTEBOOK.json').read_text())
cohort=pd.read_csv(REPORT/'cohort_metrics.csv');draw=pd.read_csv(REPORT/'query_draw_exposure.csv');paired=pd.read_csv(REPORT/'paired_epoch_metrics.csv');permissions=pd.read_csv(REPORT/'reference_permission_summary.csv')
old=pd.read_csv(ROOT/'tread_two/lc_review/assets/v1_cohort_metrics.csv')
sel=cohort.query("stage=='selected_eval'").iloc[0];final=cohort.query("stage=='final_eval'").iloc[0];v1=old.query("stage=='selected_eval'").iloc[0]
rel=DONE.relative_to(ROOT).as_posix();rows='\n'.join(f'| `{name}` | {n:,} |' for name,n in v['scientific_rows'].items())
byte_rows=f"{v['bytes_verified']:,}";derived_rows=sum(t['rows'] for t in a['csv_tables'].values());blocked=int(permissions.blocked_rows.sum())
table='\n'.join(f'| {label} | {getattr(v1,f):.6f} | {getattr(sel,f):.6f} | {getattr(final,f):.6f} |' for label,f in [('Accuracy','accuracy'),('Observed-class BA','balanced_accuracy_observed'),('NLL ↓','nll'),('Brier ↓','brier'),('10-bin ECE ↓','ece_10_equal_width')])
text=f'''# NB11 — completed ATLAS-LC r2, independently HF-verified

**30 October 2026. The fit is complete; no NB11 rerun is needed.** All five
notebook cells completed without a notebook exception. The actual dual-T4/live-HF
recovery preflight passed, followed by 60 epochs / 2,940 optimizer updates,
selected/final evaluation and all six diagnostics. The publication owner is released.

**The scientific outcome is mixed:** validation BA reached **56.25%**, selecting
epoch **7** by the frozen rule. The historical endpoint still has **50% BA**,
with **62 low→mid / 41 high→high** at both selected and final checkpoints.
Stability and probability loss improved over v1, while low-tyre recognition did
not. The candidate is not promoted or installed in either workstation.

## Execution and publication identities

| Item | Independently checked value |
|---|---|
| Supplied executed notebook SHA-256 | `{v['executed_notebook_sha256']}` |
| Pinned HF revision | `{v['revision']}` |
| Measured protocol | `{v['protocol']}` |
| Publication | **{v['objects_verified']:,} objects + manifest = {v['files']:,} direct files** |
| Object bytes | **{byte_rows}**; all downloaded bytes checked by size and SHA-256 |
| Scientific rows | **{v['total_scientific_rows']:,}** across all 24 raw CSV families |
| Classifier observations | 53,342, including 47,040 online training views |
| Training | One seed; 60 epochs / 2,940 completed updates; all 60 epoch weights retained |
| Actual devices | Tesla T4 ×2; optimizer on GPU0, immutable evaluation on GPU1 |
| State | Completed; native state hash/cursors/orders/fixed bank/finite values checked |
| Independent subjects | No new tyres; training known low/mid/high support remains 1/3/4 |
| Audit access | Public reads only; no local credential lookup or remote writes |

[Pinned HF publication](https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/tree/{v['revision']}/{v['prefix']}),
[preserved execution]({rel}/executed.ipynb), [inspection]({rel}/INSPECTION.json),
[full verification]({rel}/VERIFICATION.json), [manifest]({rel}/REMOTE_MANIFEST.json).

All 31 embedded files match the delivered payload; all 30 frozen runtime/assets
remain unchanged. Initial unused-package dependency warnings were nonfatal.
No speculative repair, environment replacement or training restart is needed.

## What was checked

```mermaid
flowchart TD
    A["Executed NB11<br/>5 completed cells / original payload"] --> B["Pinned publication / released owner"]
    B --> C["1,350 objects<br/>all size + SHA checks"]
    C --> D["24 raw schemas and row counts<br/>54,336,368 records"]
    D --> E["Native recovery / all60 epoch weights<br/>actual live-HF preflight"]
    E --> F["Scientific review<br/>roles,selection,losses,exclusions,logits"]
    F --> G["40 derived CSVs<br/>380,595 rows / 1,442 image cases"]
    G --> H["Next: CPU NB12<br/>publish review; no new fit"]
```

Every raw CSV header and row count was independently read and compared with its
native cursor. The native completed state has all 60 deterministic epoch orders,
all 570 deduplicated dense-image identities, exact reference buffers and matching
scientific cursors. All 60 checkpoint objects exist and their bytes were hashed.
This does not mean all 60 models were rerun locally; no new model forward was used.

| Raw family | Recounted rows |
|---|---:|
{rows}

The separate [scientific analysis receipt](tread_two/lc_analysis_report/ANALYSIS.json)
checks its 696 consumed objects / 295,152,277 bytes. It reconstructs every paired
objective and all endpoint/probe logits. That consumed-subset scope is distinct
from the exhaustive publication-byte and raw-row audit above.

## Results: distinguish scores, recognition and calibration

All rows below use the same previously inspected 103-image low/high endpoint.
The v1 and r2 recipes differ in representation, bank/query policy, sampler and
objective. This is a descriptive comparison, not an isolated-factor experiment.

| Metric | v1 selected, epoch18 | r2 selected, epoch7 | r2 final, epoch60 |
|---|---:|---:|---:|
{table}

NLL and Brier improve for selected r2, but raw-score ECE worsens from about
0.162 to 0.426. Lower probability loss is not a calibrated-confidence claim.
Neither selected nor final recovers a low endpoint target. Mid recall is
unavailable because this endpoint has no mid targets; it is not recorded as zero.

The frozen dense anchor retains mean two-region Dice **98.418%** and human-point
MAE **1.196% of image width** on the comparable annotated subset. These are
TREAD-ONE outputs; classifier training did not improve the frozen dense model.

![Actual training, validation and endpoint confusion](tread_two/lc_analysis_report/figures/fit_and_confusion.png)

## What the new telemetry shows

| Question | Measured answer | Interpretation |
|---|---|---|
| Did both views train consistently? | All 23,520 paired objectives and 2,940 update losses reconstruct; late online accuracy is 100% | Fits repeated, correlated training observations; not fixed-checkpoint generalization |
| Were self references actually suppressed? | **{blocked:,}** blocked view/reference rows; all 960,156 permission/scaling rows match full source SHA | Identical-photo support was blocked without query labels; same-tyre similarity remains possible |
| Did all queries participate? | All **386** eligible images were drawn; **{int(draw.actual_draws.max())}** is the largest observed per-image draw count | Softer repetition succeeded operationally; no new independent subjects |
| Are fixed appearance probes more stable? | Zero prediction flips across all **1,236** diagnostics | Preserves predictions, including the same wrong low predictions |
| Does grayscale stability prove learned invariance? | Changes are only FP32-scale; LC already uses channel-mean luminance | Largely a construction property; do not credit the paired loss alone |
| Are explanations arithmetically valid? | All **1,442** endpoint/probe logits reconstruct within 2e−6; occlusion indices and saved reference sums agree | Exact signed evidence is model accounting, not physical causal-wear proof |

![Paired objective components and observed disagreement](tread_two/lc_analysis_report/figures/paired_learning.png)

The weighted consistency terms were already small relative to CE at the start.
The plotted joint/CE curves nearly overlap. Their separate contribution cannot
be isolated from LC, the new bank and changed sampling in this bundled pilot.
The persistent train/held-out gap is observed; a unique causal explanation such
as paint, camera, identity or groove texture has not been established.

## Decision and next step

1. Close the measured NB11 fit and preserve its exact source, checkpoints and
   namespace. Do not train longer or choose another test-scoring epoch.
2. Keep this candidate as a completed research result; do not promote its
   classifier or change either workstation.
3. Run [CPU NB12](notebooks/phase3_NB12_TREAD_TWO_ATLAS_LC_Completed_Review.ipynb)
   to reproduce and publish the saved-evidence review in a separate HF namespace.
   It makes no model forwards and needs no tyre dataset attachment or GPU.
4. Review the complete failure/reference evidence before freezing any third fit.
   No additional tyre photography is required or planned for this scope.

[Interactive 1,442-case review](tread_two/lc_analysis_report/ATLAS_Failure_Explorer.html),
[40-table CSV inventory](tread_two/lc_analysis_report/CSV_INDEX.md),
[calculation methods](tread_two/lc_analysis_report/METHODS.md),
[CPU run guide](tread_two/lc_review/README.md),
[local/package validation](tread_two/lc_review_validation/PACKAGE_VALIDATION.json).

NB12 is implemented and locally checked, including fresh extracted-code equality,
cache corruption recovery, viewer controls and offline publication/release/rerun.
Actual NB12 Kaggle execution and its HF review publication remain pending.
The raw NB11 evidence is already published; this next step adds readable derived
records rather than repeating training.
'''
(ROOT/'phase3NB11Completion.md').write_text(text,encoding='utf-8')
marker='<!-- NB11_COMPLETE_UPDATE_20261030 -->'
def front(name,body,title=None):
    p=ROOT/name;s=p.read_text(encoding='utf-8');h,_,rest=s.partition('\n')
    if marker in rest:rest=rest.split(marker,1)[0]+rest.split('<!-- END_NB11_COMPLETE_UPDATE -->',1)[1]
    p.write_text((title or h)+'\n\n'+marker+'\n## Current — NB11 complete; NB12 review ready\n\n'+body+'\n\n<!-- END_NB11_COMPLETE_UPDATE -->\n'+rest.lstrip('\n'),encoding='utf-8')
summary='**30 October 2026. NB11 is complete and independently HF-verified:** 60 epochs / 2,940 updates, 1,351 direct files and 54,336,368 raw scientific rows. The actual T4×2/live-HF preflight passed; ownership is released. No NB11 rerun is needed. [Full completion and findings](phase3NB11Completion.md).'
finding='Validation selected epoch 7 at BA 56.25%. Selected/final historical endpoint BA remains 50%: 62 low→mid and 41 high→high. NLL and appearance stability improve over v1; low recognition does not, and selected raw-score ECE worsens. Keep the candidate as research evidence; no workstation promotion.'
next='Next: [NB12 saved-evidence review](notebooks/phase3_NB12_TREAD_TWO_ATLAS_LC_Completed_Review.ipynb), **CPU / Internet / HF_TOKEN / no dataset attachment**, then Run All. It adds 40 derived CSVs / 380,595 rows, four figures and a 1,442-case explorer with zero fitting or model forwards. Local recovery/cache/publication/package checks pass; actual NB12/HF review publication is pending. [Run guide](tread_two/lc_review/README.md).'
for name in ['README.md','phase3Progress.md','phase3Understanding.md','phase3TreadTwoFindings.md','phase3TreadTwoLCArchitecture.md','phase3TreadTwoLCProtocol.md','phase3TreadTwoLCValidation.md']:
    front(name,summary+'\n\n'+finding+'\n\n'+next)
front('phase3NextSteps.md',summary+'\n\n'+next+'\n\nAfter publication, audit the pinned review and use the saved failures/reference structure to decide a separately frozen next hypothesis. Do not repeat NB08–NB11, add a measured seed, choose by test score, collect new photos or modify workstations as an automatic next step.','# Next steps — publish the completed ATLAS-LC evidence review')
front('AGENTS.md',summary+'\n\n- Preserve executed NB11 (`16fa0443…`), all 30 frozen r2 source/assets, all epoch weights and the completed fit namespace. Do not regenerate executed notebooks or change an active/completed protocol.\n- The candidate is not promoted: endpoint BA stays 50%, low recall is 0/62. Improved stability/NLL are narrower findings; ECE worsened.\n- Existing tyre data only; no new acquisition. Both workstations remain deferred and unchanged.\n- Next is the separate CPU saved-evidence review/publication. Distinguish exhaustive raw publication verification from the 696-input scientific review and pending actual NB12 publication.\n- Keep actual draws separate from two correlated view observations. Preserve all direct CSV/native evidence and exact declared gaps; never ZIP.\n\n'+next)
front('phase3OpenItems.md','| Item | State |\n|---|---|\n| NB11 actual T4×2/live HF/fit/endpoints | Closed; independently verified complete |\n| Raw object/schema/row and scientific-loss/reference audits | Closed |\n| r2 low endpoint recognition | Unresolved: 0/62; no promotion |\n| NB12 review implementation/local checks | Complete |\n| Actual NB12 Kaggle/HF review publication | Pending CPU notebook |\n| More photos / workstation changes | Not part of current scope |\n\n'+next)
front('phase3EvidenceRegister.md','| Record | Scope |\n|---|---|\n| [NB11 full publication audit]('+rel+'/VERIFICATION.json) | Every raw object byte, all 24 raw schemas/counts, native state and actual preflight |\n| [Scientific review](tread_two/lc_analysis_report/ANALYSIS.json) | 53,342 predictions, 23,520 paired losses, source exclusions, selection and all 1,442 endpoint/probe logits |\n| [Review CSV index](tread_two/lc_analysis_report/CSV_INDEX.md) | 40 tables / 380,595 derived rows |\n| [NB12 package checks](tread_two/lc_review_validation/PACKAGE_VALIDATION.json) | Fresh extracted-code equality, cache seam, viewer and offline publication |\n\n'+summary)
front('phase3DocumentationInventory.md','| Document | Purpose |\n|---|---|\n| [NB11 completion](phase3NB11Completion.md) | Exact publication proof, results, interpretation, figures and decision |\n| [Review run guide](tread_two/lc_review/README.md) | CPU settings, recovery/publication boundaries and module ownership |\n| [Derived methods](tread_two/lc_analysis_report/METHODS.md) | Pairing, losses, reference scaling, denominators and comparison caveats |\n| [CSV inventory](tread_two/lc_analysis_report/CSV_INDEX.md) | 40 tables, counts, column dictionary and interactive explorer |\n| [Review validation](tread_two/lc_review_validation/PACKAGE_VALIDATION.json) | Actual local/package/publication-fixture proof; pending actual NB12 clearly marked |')
front('phase3SessionLog.md',summary+'\n\nArchived the supplied NB11, checked payload/execution, audited the pinned publication and all raw rows, and completed the scientific review. Preserved all earlier studies/workstations. Implemented and checked a separate CPU NB12 that adds the derived CSVs, figures and complete explorer. No new fit, source photos or local HF writes occurred.\n\n'+next)
front('tread_two/lc_pilot/README.md','**Measured seed completed on Kaggle and independently verified.** The earlier local delivery notes below are historical. [Completion and findings](../../phase3NB11Completion.md). Next is [CPU review NB12](../../notebooks/phase3_NB12_TREAD_TWO_ATLAS_LC_Completed_Review.ipynb), not another fit. Keep all frozen source/assets and both workstations unchanged.')
p=ROOT/'phase3Decisions.md';s=p.read_text(encoding='utf-8')
if 'P3-D090' not in s:s+='''

## NB11 completion and saved-evidence review — 30 October 2026

| ID | Decision | Evidence / boundary |
|---|---|---|
| P3-D090 | Close NB11; preserve its executed notebook, source, all 60 epoch weights and completed namespace | Actual T4×2/live HF and 2,940 updates complete; exhaustive pinned-byte/raw-row audit passes |
| P3-D091 | Retain original validation-selected epoch 7; do not choose another epoch from test | BA 56.25% validation; endpoint selected/final BA 50%, low recall 0/62 |
| P3-D092 | Separate stability/probability loss from recognition and calibration | Zero saved-probe flips and lower NLL/Brier; unchanged low errors and worse selected ECE |
| P3-D093 | Audit paired objectives, full-source exclusion and draw-vs-view exposure before another fit | All 23,520 paired objectives and 960,156 permission rows checked; 386 queries actually exposed |
| P3-D094 | Deliver CPU NB12 for additive 40-table review publication; no new fit or workstation promotion | Existing photos only; local extracted-code/viewer/offline publication pass; actual NB12 publication pending |
'''
p.write_text(s,encoding='utf-8')
print('Updated completion report and current trackers; NB12 is the next action')
