"""Update the design findings and current Markdown records from actual receipts."""
import json
import sys
from pathlib import Path
assert Path(sys.prefix).name == 'cv_conda'
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'tread_two/trace_design'
REPORT = ROOT / 'tread_two/trace_report'


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def block(path, tag, content):
    old = path.read_text(encoding='utf-8')
    start, end = '<!-- ' + tag + ' -->', '<!-- END_' + tag + ' -->'
    replacement = start + '\n' + content.strip() + '\n' + end
    if start in old:
        a=old.index(start); b=old.index(end,a)+len(end); new=old[:a]+replacement+old[b:]
    else:
        a=old.index('\n')+1;new=old[:a]+'\n'+replacement+'\n'+old[a:]
    path.write_text(new,encoding='utf-8')


numerical=read(BASE/'validation/NUMERICAL.json');audit=read(BASE/'validation/ACTUAL_TARGETS.json');geometry=read(BASE/'validation/GEOMETRY.json')
assert numerical['status']==audit['status']==geometry['status']=='passed'
controls=pd.read_csv(REPORT/'control_losses.csv')
summary=controls.groupby('control').agg(images=('image_id','nunique'),image_mean_huber=('loss','mean'),minimum_common_coverage=('common_control_coverage','min')).reset_index()
summary.to_csv(REPORT/'control_summary.csv',index=False)
controls_md='\n'.join(f"| {r.control} | {r.images} | {r.image_mean_huber:.6f} | {r.minimum_common_coverage:.6%} |" for r in summary.itertuples())
findings=f'''# TRACE design findings — local engineering, not trained performance

**2 November 2026.** The next design has four implemented prototype modules and
an actual all-training-photo target audit. The new measured fit, Kaggle notebook
batch, full telemetry writers and live dual-T4/HF gates are not yet implemented
or run. [Delivery plan](DELIVERY_PLAN.md) · [status](STATUS.json).

## Actual evidence

| Check | Result | Proof |
|---|---|---|
| Existing train scope | 386 unique images /6 conservative components; no held-out photo reads | [Target audit](validation/ACTUAL_TARGETS.json) |
| Target construction | 197,632 cells /1,185,792 values; 197,201 supported cells | Full CSV/NPY maps retained locally; hashes in audit |
| Source integrity | All386 image SHAs and386 ignore masks checked;273 saved geometry/table objects size/SHA checked | [Input inventory](../trace_report/saved_inputs_verified.csv) |
| Numerical target reduction | NumPy FP64 cell reduction vs FP32 implementation maximum absolute difference **{numerical['checks']['independent_target_cell_reduction_max_absolute_difference']:.3g}** | [Numeric proof](validation/NUMERICAL.json) |
| Loss arithmetic | Per-image sufficient-statistic reconstruction difference **{numerical['checks']['per_image_loss_reconstruction_max_absolute_difference']:.3g}** | Same proof |
| Leakage | Hidden-pixel perturbation leaves masked predictions bit-exact; ignore masks never enter student features | Same proof |
| Rank feasibility | 8×6 identity sample covariance witness; fewer than7 eligible rows disables diversity | Same proof |
| Spatial information | Explicit moment pooling responds to row and column reversal | Same proof |
| Gradients | Every new parameter has connected finite gradient | 71,022 new parameters |
| CPU native continuation | Exact model/optimizer/RNG/cursor/history at synthetic update3→6 |12actual disposable updates in current fixture; [CSV](validation/FIXTURE_UPDATES.csv) |
| Frozen geometry | Original CPU logits/masks/points bit-exact for2train images; parent train cannot unfreeze; all state unchanged | [Geometry proof](validation/GEOMETRY.json),4image forwards in this fixture |
| New study / HF publication | No measured candidate, new photos, remote write, local token lookup or workstation change | Scope remains design/engineering |

CPU continuation is not a test of the future transactional store, dual-CUDA RNG,
remote state restoration or GPU numerical determinism. It validates these
prototype modules and a native optimizer/state round trip only.

## The failed first descriptor draft is preserved

The initial v0 used contrast, absolute gradients, squared gradients and their
cross-product. Its six image-average target coordinates have rank **1.329425**,
with **93.4932%** of variation in the leading direction. The revised v2 adds
stabilised orientation, relative curvature and occupancy, with rank **1.458123**
and **91.6687%** leading variance. This is still low; it is not a collapse fix or
a model improvement. Original v0 code, maps, audit and numerical receipt remain
locally in `validation/actual_train_v0/`; the compact [space comparison](../trace_report/target_spaces.csv)
keeps its findings visible in Git.

| Current v2 target space | Effective rank | Leading share | Meaning |
|---|---:|---:|---|
| Six whole-image means | 1.458123 | 91.6687% | Strong common intensity/energy variation remains |
| Mean spatial cells,96D | 2.046700 | 86.3167% | Spatial averaging still discards much variation |
| Full common spatial cells,2,568D | 15.656767 | 48.0424% | Keep local maps and position-sensitive representation |
| Full common spatial,within-component centred | 19.853732 | 46.8627% | Descriptive redistribution after centering, not new subjects |

Only cells observed in **all386** train images enter the full-spatial comparison;
missingness is not treated as visual variation. Different dimensions/centering
answer different questions. No held-out feature transform is fitted. The v2
choice adds interpretable image structure; it does not improve every rank metric
over v0 (v0 full-spatial rank was16.463679).

The architecture therefore supervises full tile/cell maps, keeps the native
local tokens and adds fixed row/column moments. No whitening makes targets look
artificially diverse. Encoder, context and projector all need separate future
checks; a well-spread final projector alone cannot establish useful learning.

## Controls are already strong on training images

One fixed mask plan/image; same common hidden cells for all three controls.
Numbers below are **train-only descriptive image means**, not leave-one-out
control validation, a trained-model result or checkpoint eligibility.

| Control | Images | Image-mean standardised Huber | Minimum common coverage |
|---|---:|---:|---:|
{controls_md}

The visible-band control shows why beating only a constant mean would be weak
evidence. The future candidate must beat all three controls on identical
validation cells and retain coverage/missingness.

## Engineering issues caught and corrected

| Issue in the new prototype/audit | Correction | Verification |
|---|---|---|
| First gradient-energy fields strongly correlated | Keep history; add orientation/curvature/occupancy; preserve full spatial maps | All386 train images, both target versions, no held-out fitting |
| Global averages discard local arrangement | Fixed five-basis spatial moments and axial full context | Row/column reversal fixture |
| Valid batch can shrink below seven rows | Explicitly unavailable variance/covariance terms, logged eligibility | Six-row disabled/seven-row enabled tests |
| FP32 pooling differed from independent FP64 cell sums by a few1e−7 | Bound this FP32-vs-FP64 comparison at1e−6 and report actual error | Max3.565e−7; no data/value replacement |
| Per-image geometry adapters changed batch reduction arithmetic | Preserve original anchor-microbatch then concatenated-adapter execution | Exact original CPU output comparison passes |
| Old local dataset helper default points outside this root | Pass the existing dataset path explicitly in the new fixture | No completed helper/source changed |

## Visual evidence and next work

![Four existing training images and observed fixed target channels](../trace_report/figures/TRACE_Fixed_Targets.png)

[Open the interactive six-channel explorer](../trace_report/TRACE_Design_Explorer.html).
Four examples use fixed training-source positions0/100/200/300. They are observed
descriptor maps, not neural predictions or selected best examples.

Next implement the data/target cache, deterministic scheduler, transaction store
and48planned CSV families, then the actual storage/dual-T4/HF gates and complete
automatic notebook batch. The current user does not need to rerun an older
notebook. All new design evidence is local; direct HF publication belongs to
the future Kaggle preparation stage using only its HF_TOKEN Secret.
'''
(BASE/'FINDINGS.md').write_text(findings,encoding='utf-8')

main='''# Next design — TREAD-TWO / TRACE

**2 November 2026: design and local prototype implemented.** TRACE learns fixed
local image descriptors in the frozen GEO canonical atlas. It has **71,022 new
texture parameters**, explicitly preserves row/column arrangement, and uses a
**6D** auxiliary projector with a seven-eligible-row covariance guard.

| Completed now | Evidence |
|---|---|
| Exact target math, architecture, draft experiment and48-family/669-field retention specification | [Design package](tread_two/trace_design/README.md) |
| All386train-photo target construction:197,632cells /1,185,792values retained | [Findings and audit](tread_two/trace_design/FINDINGS.md) |
| Hidden-pixel leakage, objective/gradient, CPU native resume and frozen-geometry equivalence proofs | [Numeric](tread_two/trace_design/validation/NUMERICAL.json) / [geometry](tread_two/trace_design/validation/GEOMETRY.json) |
| Existing-image interactive descriptor explorer | [Open explorer](tread_two/trace_report/TRACE_Design_Explorer.html) |
| Gated rollout and complete future notebook batch | [Delivery plan](tread_two/trace_design/DELIVERY_PLAN.md) |

```mermaid
flowchart LR
  A[Verified GEO evidence] --> B[TRACE design and prototype: complete]
  B --> C[Full store, writers and resource gates: next]
  C --> D[Actual dual-T4 and HF continuation]
  D --> E[One bounded measured candidate]
  E --> F[Controls, spectra and evidence passport]
```

The train-only audit found that whole-image target averages remain low-rank;
full spatial maps carry much more variation. TRACE therefore retains local maps
and explicit spatial moments. These findings are design evidence, not a trained
accuracy or generalisation result.

**Next:** implement the new transaction/evidence runtime, prove the full storage
and live recovery paths, then deliver preparation/pilot/review notebooks together
with one Run All master. **No new training notebook is ready to run yet.** No old
completed notebook needs a rerun.

Only existing photos are used. The inherited GEO final geometry remains an
explicitly **unselected research input**; the old selected=None result is
preserved. No new classifier, physical-depth/safety claim or workstation change.
New design records are local; their full direct-CSV/native HF publication is
planned for Kaggle preparation, not claimed complete now.

![Observed descriptor examples](tread_two/trace_report/figures/TRACE_Fixed_Targets.png)
'''
(ROOT/'phase3TreadTwoTRACEDesign.md').write_text(main,encoding='utf-8')

current='''## Current — TRACE design and local prototype checked

**2 November 2026:** the user authorised work on the next design. **TREAD-TWO /
TRACE** now has four prototype modules,71,022new trainable parameters, fixed
spatial descriptors and explicit position-sensitive pooling. All386train photos
were audited:197,632target cells /1,185,792values retained. CPU leakage, loss,
gradient, state-restore and frozen-geometry equivalence checks pass.

[Design, findings and interactive examples](phase3TreadTwoTRACEDesign.md) ·
[Detailed rollout](tread_two/trace_design/DELIVERY_PLAN.md).

This is **local design/engineering**, not a measured model, live HF publication,
48implemented telemetry writers or runnable Kaggle batch. Next build the full
data/transaction/evidence runtime and prove storage/dual-T4/HF continuation, then
deliver all justified notebooks with a sequential master. Existing photos only;
both workstations and every completed experiment remain preserved. NB20/GEO stays
closed and unselected; no old notebook rerun is needed.
'''
for name in ['AGENTS.md','README.md','phase3Progress.md','phase3NextSteps.md','phase3OpenItems.md','phase3Understanding.md',
             'phase3DataRetentionPolicy.md','phase3ModelRationaleAndLogging.md','phase3TreadTwoGEONextResearch.md']:
    block(ROOT/name,'TRACE_DESIGN_20261102',current)
    p=ROOT/name;old=p.read_text(encoding='utf-8')
    old=old.replace('## Current — NB20 HF-verified; GEO research cycle complete','## Completed prior cycle — NB20 HF-verified; GEO research cycle complete')
    if name=='phase3NextSteps.md':old=old.replace('# Next steps — GEO cycle closed; proposed texture revision','# Next steps — TRACE engineering after the completed GEO cycle')
    p.write_text(old,encoding='utf-8')

block(ROOT/'tread_two/README.md','TRACE_DESIGN_20261102',current.replace('](phase3TreadTwoTRACEDesign.md)','](../phase3TreadTwoTRACEDesign.md)').replace('](tread_two/trace_design/DELIVERY_PLAN.md)','](trace_design/DELIVERY_PLAN.md)'))
block(ROOT/'phase3Decisions.md','TRACE_DECISIONS','''## TRACE design decisions — 2 November2026

| ID | Decision | Evidence / boundary |
|---|---|---|
| P3-D161 | Separate TRACE texture revision; preserve all completed sources | User authorised next design; no new measured fit or workstation change |
| P3-D162 | Frozen selected anchor + explicit unselected GEO-final geometry only | Exact CPU geometry comparison; original GEO selected=None stays intact |
| P3-D163 | Fixed image-derived local targets and train-only normalisation | All386training photos, no held-out fitting; full target maps retained |
| P3-D164 | Keep failed v0 audit and full spatial evidence | Averaged targets remain low-rank; no claim of a solved collapse |
| P3-D165 | Six-dimensional projector with ≥7eligible-row diversity guard; five spatial moments | Rank-feasibility, mask-leakage and row/column reversal proofs |
| P3-D166 | Draft48CSVfamilies/669fields and storage/continuation gates before fit delivery | Schema is a plan, not implemented measured telemetry |
| P3-D167 | One future candidate and full automatic notebook batch | No new runnable notebooks yet; actual two-T4/HF proof pending |
''')
block(ROOT/'phase3EvidenceRegister.md','TRACE_EVIDENCE','''## TRACE local design evidence — 2 November2026

| Evidence | Location | Scope |
|---|---|---|
| Numerical/leakage/CPU native restore | [JSON](tread_two/trace_design/validation/NUMERICAL.json) | Synthetic engineering;12actual updates in current fixture; no measured fit |
| Original geometry comparison | [JSON](tread_two/trace_design/validation/GEOMETRY.json) |2train images /4CPUforwards; exact output and state preservation |
| All386target construction | [JSON](tread_two/trace_design/validation/ACTUAL_TARGETS.json) |197,632cells, full CSV/native retained locally; no held-out reads or HF writes |
| Spatial target diagnosis | [CSV](tread_two/trace_report/target_spaces.csv) | Initial/revised descriptors; not learned representations |
| Current implementation/preservation proof | [Delivery](tread_two/trace_design/validation/DELIVERY.json) | Scope and old source/notebook identities |
| Interactive observed maps | [Explorer](tread_two/trace_report/TRACE_Design_Explorer.html) |4fixed train examples,6channels |
''')
block(ROOT/'phase3SessionLog.md','TRACE_SESSION','''## 2 November2026 — authorised next design and checked prototype

1. Read current instructions, GEO/NB20 evidence and the proposed fixed-target brief.
2. Implemented isolated TRACE target, model, objective and frozen-geometry modules; no completed source changed.
3. Audited every386train image from verified original pixels and saved final geometry. Preserved the correlated v0 descriptor attempt; refined image fields and retained spatial evidence.
4. Checked leakage, objective reductions, gradients, rank guard, explicit spatial moments, native CPU continuation and exact inherited geometry.
5. Wrote the mathematical/design/experiment/retention/operations/delivery documents,48-family schema and interactive existing-image explorer. Updated current records and Git ignore rules.
6. Full trainer/writers/storage/two-T4/live-HF proofs and runnable notebooks remain separate next engineering work. No measured fit, new photos, local credentials, HF writes or workstation changes.
''')
block(ROOT/'phase3DocumentationInventory.md','TRACE_DOCS','''## TRACE design package

| Document | Purpose |
|---|---|
| [Design overview](phase3TreadTwoTRACEDesign.md) | Current completion and next engineering step |
| [Package index](tread_two/trace_design/README.md) | All architecture/math/experiment/evidence/operations records |
| [Local findings](tread_two/trace_design/FINDINGS.md) | Actual evidence, failed draft, controls and numerical fixes |
| [Rollout](tread_two/trace_design/DELIVERY_PLAN.md) | Implemented vs pending, future notebook sequence |
''')
block(ROOT/'phase3RepositoryMap.md','TRACE_MAP','''## TRACE design locations

| Location | Role |
|---|---|
| `tread_two/trace_design/runtime/` | Four isolated prototype modules; no full fit runner |
| `tread_two/trace_design/*.md`, `*.json` | Technical design, draft schema/budget/source pins/status |
| `tread_two/trace_design/validation/` | Small numeric/geometry/actual-data/delivery receipts and fixture CSV |
| `tread_two/trace_design/validation/actual_train*/` | Full local constructions and v0 history, excluded from Git |
| `tread_two/trace_report/` | Compact direct CSV summaries, observed-data PNG and interactive HTML |
| `analysis/*trace*` | Reproducible prototype checks, target audit, builder, documentation and handoff checks |
''')
# Keep generated prose readable without touching code identifiers, math or paths.
replacements={
    'All386':'All 386', 'all386':'all 386', 'every386':'every 386', '386train':'386 train',
    '386training':'386 training', '386target':'386 target', 'and386':'and 386', ';273':'; 273',
    '/6 conservative':'/ 6 conservative', '/1,185,792':'/ 1,185,792', 'update3':'update 3',
    '|12actual':'| 12 actual', ';12actual':'; 12 actual', 'for2train':'for 2 train',
    '),4image':'), 4 image', 'was16.463679':'was 16.463679', 'and48planned':'and 48 planned',
    'construction:197,632cells':'construction: 197,632 cells', '1,185,792values':'1,185,792 values',
    'modules,71,022new':'modules, 71,022 new', 'audited:197,632target':'audited: 197,632 target',
    '48implemented':'48 implemented', '≥7eligible':'≥7 eligible', 'Draft48CSVfamilies/669fields':'Draft 48 CSV families / 669 fields',
    '|2train images /4CPUforwards':'| 2 train images / 4 CPU forwards', '|197,632cells':'| 197,632 cells',
    'and48-family':'and 48-family', 'and386':'and 386', 'observations0/100/200/300':'observations 0/100/200/300',
    'positions0/100/200/300':'positions 0/100/200/300', 'checked;273':'checked; 273',
}
for path in [*ROOT.glob('*.md'), ROOT/'tread_two/README.md', BASE/'FINDINGS.md']:
    text=path.read_text(encoding='utf-8')
    # Only modify this revision's blocks (or its wholly new documents).
    if path.name in ('FINDINGS.md','phase3TreadTwoTRACEDesign.md'):
        for a,b in replacements.items():text=text.replace(a,b)
    else:
        import re
        def polish(match):
            part=match.group(0)
            for a,b in replacements.items():part=part.replace(a,b)
            return part
        text=re.sub(r'<!-- TRACE_[A-Z0-9_]+ -->[\s\S]*?<!-- END_TRACE_[A-Z0-9_]+ -->',polish,text)
    path.write_text(text,encoding='utf-8')

brief=ROOT/'phase3TreadTwoGEONextResearch.md'
text=brief.read_text(encoding='utf-8')
text=text.replace('**Proposal only, 2 November 2026.** No new runtime, training notebook, measured\nmodel or accuracy claim is delivered by this brief.',
                  '## Original proposal before TRACE implementation\n\n**Historical brief, 2 November 2026.** This proposal preceded the authorised\nTRACE design/prototype above; it delivered no runtime, training notebook or\nmeasured model at that point.')
brief.write_text(text,encoding='utf-8')
lessons=ROOT/'phase3TreadTwoGEOLessons.md'
text=lessons.read_text(encoding='utf-8')
text=text.replace('The immediate next action is NB20 Run All, which publishes NB18 review and NB19\npassport sequentially. No extra full fit or workstation change follows it\nautomatically. A later model needs its own concrete objective and frozen protocol.',
                  'NB20 has now completed and published NB18 review and NB19 passport. The user\nauthorised the separately documented [TRACE design and prototype](phase3TreadTwoTRACEDesign.md).\nIts measured fit still requires a new frozen protocol and the full engineering\ngates; the completed GEO source and workstations remain preserved.')
lessons.write_text(text,encoding='utf-8')
print(json.dumps({'status':'documented','control_summary':summary.to_dict(orient='records'),'new_decisions':'P3-D161–167'},indent=2),flush=True)
