"""Write illustrated completion/insight records from passed, pinned audit receipts."""
import re,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nb24_public import *
v=read(OUT/'VERIFICATION.json');s=read(OUT/'SCIENTIFIC_VERIFICATION.json');assert v['status']==s['status']=='passed'
df=pd.read_csv(OUT/'VALIDATION_GATE_AUDIT.csv');end=pd.read_csv(OUT/'ENDPOINT_COHORTS.csv');f=s['final_validation']
report=ROOT/'tread_two/trace_completion_report';figures=report/'figures';figures.mkdir(parents=True,exist_ok=True)
shutil.copyfile(OUT/'remote/review/TRACE_Validation.png',figures/'TRACE_Validation.png')
shutil.copyfile(OUT/'remote/review/MODEL_CARD.md',report/'MODEL_CARD.md')
for name in ('VALIDATION_GATE_AUDIT.csv','ENDPOINT_COHORTS.csv','INTERVENTION_SUMMARY.csv','CSV_FAMILY_INVENTORY.csv','RESOURCE_SUMMARY.csv'):
    shutil.copyfile(OUT/name,report/name)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
a=axes[0,0]
a.plot(df.epoch,df.masked_loss,color='#166576',lw=2,label='TRACE masked')
a.axhline(f['control_visible_band_mean'],color='#b07434',ls='--',label='Visible-band control')
a.plot(df.epoch,df.local_loss,color='#45a280',lw=2,label='TRACE local')
a.set(title='Useful fixed-texture reconstruction',xlabel='Epoch',ylabel='Validation image-mean Huber');a.legend(fontsize=9)
a=axes[0,1]
for key,label in [('projected_rank','6D projector'),('local_pooled_rank','48D local'),('pooled_rank','240D spatial')]:a.plot(df.epoch,df[key],label=label)
a.axhline(4,color='#aa4545',ls='--',label='Original minimum rank 4')
a.set(title='Overall diversity remains below the gate',xlabel='Epoch',ylabel='Effective rank',ylim=(0,4.5));a.legend(fontsize=9)
a=axes[1,0];x=np.arange(3);ordered=end.set_index('role').loc[['train','validation','test']]
for i,(key,label) in enumerate([('projected_between_fraction','Projector'),('local_pooled_between_fraction','Local'),('pooled_between_fraction','Spatial')]):a.bar(x+(i-1)*.24,ordered[key]*100,.24,label=label)
a.set(xticks=x,xticklabels=['Train (6 groups)','Validation (2)','Test (2)'],ylim=(0,105),ylabel='Between-component variation (%)',title='Held-out feature variation is group-dominated');a.legend(fontsize=9)
a=axes[1,1];gates=s['gates_passed_epochs'];names=list(gates);values=list(gates.values())
a.barh([n.replace('_',' ') for n in names][::-1],values[::-1],color=['#b65858' if z==0 else '#318a80' for z in values][::-1]);a.set(xlim=(0,65),xlabel='Passing epochs / 60',title='Original selection rule stays unchanged')
fig.suptitle('TREAD-TWO / TRACE — completed research; no selected checkpoint',fontsize=17,color='#173c48')
fig.savefig(figures/'TRACE_Completion_Insights.png',dpi=160);plt.close(fig)
fit_url=f'https://huggingface.co/datasets/{REPO}/tree/{v["stages"]["fit"]["revision"]}/{v["stages"]["fit"]["prefix"]}'
passport=url('review',MANIFESTS['review']['files']['TRACE_Evidence_Passport.html']['object'])
audit_rel=OUT.relative_to(ROOT).as_posix()
stage_rows='\n'.join(f'| {name} | `{r["revision"]}` | {r["logical_files_including_manifest"]:,} | {r["CSV_files"]:,} | {r["bytes"]:,} | Released |' for name,r in v['stages'].items())
gate_rows='\n'.join(f'| {name.replace("_"," ")} | {count}/60 |' for name,count in s['gates_passed_epochs'].items())
cohort_rows='\n'.join(f'| {r.role} | {r.images} | {r.projected_rank:.6f} | {r.local_pooled_rank:.6f} | {r.pooled_rank:.6f} | {r.local_pooled_between_fraction*100:.2f}% | {r.pooled_between_fraction*100:.2f}% |' for r in end.itertuples())
def write(path,body):
    body=body.rstrip()+'\n'
    if not path.exists() or path.read_text(encoding='utf-8')!=body:path.write_text(body,encoding='utf-8',newline='\n')
def polish(body):
    body=re.sub(r'\b(All|all|uses|with|establish|step|epoch|Epoch|below|the|old|only|least|rank|Mean|mean|approximately|RSS|declared|projected)(?=\d)',r'\1 ',body)
    body=re.sub(r'(?<=\d)(GB|MB|NPY|PT|families|fields|epochs|updates|existing|train|validation|test|eligible|texture|million|objectives|photos)\b',r' \1',body)
    body=re.sub(r'(?<=[A-Za-z])\*\*(?=\d)',' **',body)
    for a,b in [('November2026','November 2026'),('bothCUDA','both CUDA'),('twoCUDA','two CUDA'),('batch-compatible6D','batch-compatible 6D'),('2GB','2 GB'),('1.02 is','1.02 is'),('GEO\'s4.03%','GEO\'s 4.03%'),('6Dprojector','6D projector')]:body=body.replace(a,b)
    return body
completion=f'''# NB24 complete — TRACE fit and review verified with Hugging Face

**2 November 2026. Execution is complete; no checkpoint passed selection.**
The supplied master completed preparation, real two-T4/HF continuation gates,
one seed / **60 epochs / 2,940 updates**, and the CPU review/passport. Three code
cells completed without saved errors. All **33 embedded files** match the frozen
delivery. **No NB21, NB22, NB23 or NB24 rerun is needed.**

The original rule accepts **0/60 checkpoints**. `selected=None` remains intact;
the final epoch is an unpromoted research artifact. [Observed scientific findings](phase3TreadTwoTRACEFindings.md).

```mermaid
flowchart LR
  A[570 existing photos] --> B[Fixed GEO geometry and TRACE target cache]
  B --> C[Actual two-T4 and HF restart gate passed]
  C --> D[One TRACE seed: 60 epochs / 2940 updates]
  D --> E[Saved CSV/native review and interactive passport]
  E --> F[Independent public HF and numerical audit]
  F --> G[Completed research / selected unavailable]
```

## Pinned identities and publications

| Identity | Exact value |
|---|---|
| Executed NB24 SHA256 | `{v['executed_notebook_sha256']}` |
| Frozen package digest | `050f71ae540b387e33795144cf97894934cbccfad239fed9078a0c9140315356` |
| Actual Kaggle fit protocol | `{v['protocol']}` |
| Review protocol | `{v['review_protocol']}` |
| Fixed cache scientific hash | `{read(OUT/'FIT_RESULT.json')['cache_hash']}` |

| Stage | Checked HF revision | Logical files incl. manifest | CSV files | Selected content bytes | Owner |
|---|---|---:|---:|---:|---|
{stage_rows}

Preparation, engineering and transport-tail are checked at the final repository
revision; that is not presented as their original selecting-commit timestamp.
Fit/review revisions also match the notebook's saved completed result. All five
selecting manifests remain unchanged at the checked latest revision.

[Pinned fit publication]({fit_url}) · [Published interactive residual passport]({passport}) ·
[Audit receipt]({audit_rel}/VERIFICATION.json) · [Object-level methods]({audit_rel}/OBJECT_VERIFICATION.json).

## What was independently checked

| Check | Result | Scope |
|---|---|---|
| Notebook identity/source | Pass | Exact executed bytes archived; all33 payloads match; no notebook regeneration |
| All selected HF objects | 12,262 objects / five stages | Pinned existence and size for every object; LFS SHA256 identities; full body + Git SHA1/SHA256 for bounded Git objects |
| Large remaining Git CSVs | Metadata tier explicitly recorded | Full body was not independently downloaded for every dense CSV; do not call this an all-byte local audit |
| Native science inputs | 946 files /16,130,192 bytes, full checksums | All projected vectors needed for every update; all original validation and final-cohort diagnostic spaces |
| Training objectives | All2,940 reconstructed | All six recorded terms match, maximum absolute difference {s['max_objective_absolute_difference']:.3g} |
| Original validation gates | All60 independently reconstructed | SVD spectra rather than the trainer's covariance eigensolver; identical rejection and selected=None |
| Final complete state | Pass |3,809 contiguous ledger units and chain; optimizer step2940; final weights and epoch60/export identical |
| Retained RNG | Pass | Python, NumPy, CPU and bothCUDA states; FP32; no scheduler/scaler |
| Actual source coverage | Pass |386 primary images each epoch;8 distinct source hashes per batch;6 padding draws separated |
| Earlier HF publications | Seven selecting manifests unchanged | NB13/14/15/16/17 plus both NB20 stages |
| Interactive viewer | Pass |570 cases; actual JavaScript role/search/paging, PNG signatures and missingness branch; DOM fixture, not a native screenshot |

The original published CPU review records **361,795,584 prediction values** and
**14,132,640 moment values** matching native arrays exactly, all raw local/masked
losses reconstructed, all2,940 objectives checked, and original selection
reproduced. That full dense audit is preserved and its file inventory matches
the selected complete state. This local pass independently reconstructs all
objectives/gates; it does **not** repeat every dense prediction/moment comparison.
These two scopes are deliberately distinguishable.

[Independent science]({audit_rel}/SCIENTIFIC_VERIFICATION.json) ·
[Every validation epoch]({audit_rel}/VALIDATION_GATE_AUDIT.csv) ·
[Every training objective]({audit_rel}/INDEPENDENT_OBJECTIVES.csv).

## Retention and meaning of the counts

| Evidence retained in the fit publication | Count / meaning |
|---|---|
| Existing photos |570:386train /81validation /103test; **zero new photos** |
| Per-image observations |29,443; two views each |
| Training draws |23,520 =23,160 primary +360 padding; no extra independent tyres |
| Main fit CSV records |46,966,325 committed rows |
| Fixed-cache CSV records inherited by fit |{v['inherited_preparation_CSV_rows']:,} |
| Full fit publication |**{v['stages']['fit']['CSV_files']:,} direct CSV files / {v['full_fit_publication_CSV_rows']:,} rows**,13.507GB selected content |
| Dense prediction rows |30,149,632, each carrying six local +six masked values |
| Spatial moment coordinates |14,132,640 CSV rows |
| Native arrays |6,725NPY =6,149 observation shards +576 fixed-cache arrays |
| Weights/recovery |60 epoch weights +initial control +target statistics +final export +selected rolling state =64PT |
| Executable schema |48families /671fields;46families populated in fit; errors and review-verification placeholders remain header-only there |

Row totals use the native complete-state counters and preparation chunk/source
counts. They are not described as a fresh independent reparse of every dense CSV.
The all-stage15.766GB logical sum includes repeated references/copies of the
fixed cache across preparation, engineering and fit; it is not unique new data.

[Family counts and exact basis](tread_two/trace_completion_report/CSV_FAMILY_INVENTORY.csv).
Files remain direct CSV/JSON/NPY/PT/PNG/HTML. No ZIP/NPZ/database envelope.
Historical optimizer snapshots for every update and full preprocessed inputs
were not promised or retained: rolling complete recovery plus all epoch weights,
native predictions/tokens, seeds/cache/plans and explicit artifact-gap records
define the actual boundary.

## Actual engineering and resources

The published readiness certificate reports actual Tesla T4×2, a checkpoint
published/downloaded into a fresh worker, exact model/optimizer/RNG/cursor
continuation,244 exact scientific files, and twoCUDA RNG states. Its nested
worker-only `actual_HF_download_asserted_by_parent=false` is a scope marker;
the parent certificate independently records `actual_HF_restore_continue=true`.
It is not treated as a worker having authenticated to HF.

The live storage gate projected14.468GB logical /17.496GB peak with a2GB reserve,
within the declared20GB working-space assumption. Measured fit samples show
at least7.224GB free, maximum sampled host RSS2.331GB, GPU0 peak allocated
586.503MB /reserved765.460MB. These are recorded samples/allocator peaks, not
an assertion about every instant or the entire Kaggle VM. GPU1 constructs the
fixed geometry cache; GPU0 trains TRACE. This is **not** data-parallel training
across two T4s. No measured numerical event is nonfinite; saved error table is empty.

All five owners are released. The finite transport tail states zero pending
numeric updates and explicitly excludes its own upload/later review requests
from its event boundary. Thirty-minute and major-stage publication remains the
frozen recipe; no claim of recovery from an uncatchable hard session kill is added.

## Visual record and next action

![Verified TRACE training and validation](tread_two/trace_completion_report/figures/TRACE_Validation.png)

**No additional training is needed to finish this TRACE run.** Preserve this
executed notebook, source package, registry and all HF evidence. The useful next
research step is saved-evidence analysis of local texture utility and component
dependence before choosing another objective. That proposal is detailed in the
[findings and next-step record](phase3TreadTwoTRACEFindings.md); no NB25 is
implemented or required here. Existing photos only; both workstations remain deferred.
'''
write(ROOT/'phase3NB24Completion.md',polish(completion))
findings=f'''# TRACE findings — useful texture reconstruction, unresolved feature diversity

**2 November2026. Actual NB24/Kaggle/HF result; one measured seed.**
[Execution, provenance and verification boundaries](phase3NB24Completion.md).

TRACE learns a measurable fixed-texture reconstruction advantage on the original
validation cohort. It still fails the predeclared broad representation gates.
The honest outcome is **completed, unselected, unpromoted research**.

![TRACE utility, diversity, group variation and gates](tread_two/trace_completion_report/figures/TRACE_Completion_Insights.png)

## What improved

Validation uses81 previously inspected images. Values below are image-mean
SmoothL1/Huber losses on standardized image-derived targets; lower is better.
The three masked comparisons use identical supported hidden-cell intersections,
with100% common coverage. Controls and normalization were fitted on training only.

| Final epoch60 quantity | TRACE loss | Control loss | Relative reduction | Original minimum |
|---|---:|---:|---:|---:|
| Masked vs channel mean |0.467383113 |0.691030548 |32.36% |5% |
| Masked vs position mean |0.467383113 |0.662255626 |29.43% |5% |
| Masked vs visible-band mean, strongest control |0.467383113 |0.596403013 |**21.63%** |5% |
| Local vs position mean |0.421631580 |0.663375253 |**36.44%** |10% |

These percentages describe **target reconstruction**, not tyre-wear
classification accuracy, physical tread depth, safety or defect-detection accuracy.
The best masked validation loss is0.463746201 at epoch47 (22.24% vs the strongest
control), recorded only as a retrospective diagnostic. Epoch47 fails diversity too;
it is **not selected** and does not replace the original final endpoint.

## Why no checkpoint was selected

| Original gate | Passing epochs |
|---|---:|
{gate_rows}

The geometry entry is an inherited fixed-input/provenance gate. TRACE does not
train geometry and did not newly establish60 independent geometry measurements.
The other passing gates do not override either failed diversity gate.

| Final validation space | Effective rank | Minimum | Leading variation | Within-component rank |
|---|---:|---:|---:|---:|
|6D projector |{f['projected_rank']:.6f} |4 |{f['projected_leading']*100:.2f}% |{f['projected_within_rank']:.3f} |
|48D local pooled features |{f['local_pooled_rank']:.6f} |4 |{f['local_pooled_leading']*100:.2f}% |{f['local_pooled_within_rank']:.3f} |
|240D spatial moments |{f['pooled_rank']:.6f} |4 |{f['pooled_leading']*100:.2f}% |{f['pooled_within_rank']:.3f} |

Projector median coordinate standard deviation is**0.096143**, above the0.05
floor. Its rank fails even though its amplitude passes. Within-component local
and spatial ranks comfortably pass the2.0 floor. The representation is not
literally constant; most total variation is concentrated along one direction.

## Group structure remains the main warning

| Final endpoint role | Images | Projector rank | Local rank | Spatial rank | Local between-group variation | Spatial between-group variation |
|---|---:|---:|---:|---:|---:|---:|
{cohort_rows}

Validation has only two conservative components. They account for**96.78%** of
local variation and**96.56%** of spatial variation; projector between-component
variation is99.35%. Removing component means exposes smaller, more varied
within-component differences. This is compatible with a dominant group/appearance
axis; it does not identify its physical cause or prove a tyre-identity shortcut.

Training effective ranks are also below4, so the limitation is not explained
solely by having two held-out groups. The batch-compatible6D projector fixes
the old64D/batch8 rank-feasibility conflict; feasibility does not guarantee that
this objective learns broad useful features. Reconstruction and a pair
agreement term can coexist with a narrow representation. That mechanism remains
a hypothesis until a separately justified test establishes it.

## Layout and appearance probes

All four nonidentity probes use the103 original test-role images, the final
epoch60 weights and fixed geometry/cache. Targets/support are permuted with the
layout where appropriate. These are descriptive probes, not checkpoint selection.

| Probe | Mean240D spatial cosine change | Mean6D projector cosine change | Mean masked loss |
|---|---:|---:|---:|
| Brightness×0.85 |0.000650 |0.000458 |0.471648 |
| Grayscale |approximately0 |approximately0 |0.472012 |
| Reverse bands |0.023727 |0.019439 |0.553258 |
| Reverse columns |0.021681 |0.006075 |0.548635 |

Reversal leaves the pooled local encoder invariant to numerical precision while
changing contextual spatial evidence and worsening masked reconstruction. This
confirms order sensitivity in the existing model. Spatial moments/position
metadata already make order matter by design; this does not prove semantic
damage localization. Grayscale invariance is largely expected because input
contrast/gradients are constructed from channel-mean grayscale. Do not describe
that built-in property as a newly learned colour-robustness discovery.

[Full probe table](tread_two/trace_completion_report/INTERVENTION_SUMMARY.csv) ·
[Interactive residual passport]({passport}).
Residual heatmaps are standardized image-descriptor reconstruction errors,
with unavailable cells shown in white. They are not measured wear, causal
explanations of damage, calibrated confidence or physical-depth maps. The
mask-leakage gate applies after fixed geometry; original geometry has already
seen the full image before a texture tile is hidden.

## Lessons carried forward from previous models

| Previous observation | TRACE response | Actual outcome |
|---|---|---|
| Joint geometry/texture training did not solve latent collapse | Freeze the saved geometry; train only71,022texture parameters | Reconstruction utility improves; broad diversity still fails |
|64D small-batch covariance objective was rank-infeasible |6D auxiliary projector; diversity requires at least7eligible rows | Feasible and finite; final rank1.02 is still below4 |
| Mean pooling discarded layout | Explicit five spatial moments and axial context | Actual reversal sensitivity; no new semantic wear claim |
| Learned quality estimates lost to constant controls | Omit another confidence head; retain raw residuals and fixed controls | Rich interpretable research evidence, no calibrated reliability claim |
| Earlier collection was too sparse/compressed | Direct full-view CSV/native evidence and all epoch weights |47.263million fit-publication rows with explicit scopes/gaps |

TRACE and GEO have different targets, controls and model roles. Their loss
numbers/ranks are contextual evidence; the21.63% TRACE advantage is not a
matched paired estimate against GEO's4.03% value. No new classification head or
evaluation was run; old classification results are not updated by this notebook.

## What to do next with the existing data

```mermaid
flowchart TD
  A[Preserve completed TRACE and selected=None] --> B[Use saved native tokens and fixed targets]
  B --> C[Compare local texture evidence with fixed-descriptor controls]
  B --> D[Separate total variation from component-centered variation]
  C --> E[Choose a task whose evidence is supported by existing photos]
  D --> E
  E --> F[Predeclare a new protocol only if justified]
```

1. Close the current fit/review cycle without rerunning it or relaxing its gates.
2. Next, analyse saved tile tokens, fixed descriptor maps and per-component
   residuals to distinguish useful local reconstruction from broad embedding
   quality. Use training-only controls; keep repeatedly inspected roles explicit.
3. A future candidate should answer that diagnosis with a separately frozen
   objective and feasible evaluation. Do not launch a weight sweep, add epochs,
   relabel the same data or claim a fresh blind test from the existing roles.

**This is the next research proposal, not another delivered training notebook.**
No new photos, additional optimizer updates, HF writes or workstation changes
were made during this verification. Both workstations remain deferred.
'''
write(ROOT/'phase3TreadTwoTRACEFindings.md',polish(findings))
current='''<!-- NB24_VERIFIED_20261102 -->
## Current — NB24 TRACE completed and HF-verified

**2 November2026:** NB24 completed preparation, actual T4×2/HF continuation,
one seed /60epochs /2,940updates, and the saved-evidence review/passport.
All33 payloads match. **No NB21/NB22/NB23/NB24 rerun needed.**
[Completion and verification scope](phase3NB24Completion.md).

Final validation masked reconstruction beats the strongest declared control by
**21.63%**; local reconstruction improves**36.44%**. Utility passes60/60, but
projector/encoder diversity passes0/60: ranks**1.0225 /1.2296 /1.2489**.
**Selected=None; research-only, unpromoted.**
[Illustrated findings and next research](phase3TreadTwoTRACEFindings.md).

The fit retains**3,553 direct CSVs /47,263,346 rows /6,725NPY /64PT /13.507GB**;
570existing photos only. Counts include inherited cache rows; no new photos.
Independent audit reconstructs all2,940objectives and60validation gates; the
published full dense audit and the bounded local audit have explicit scopes.
All five owners are released; earlier publications remain unchanged.

Next use saved local texture/component evidence to justify any future revision;
no new fit or NB25 is required to complete this cycle. Preserve executed
notebook/runtime/protocol. Both workstations remain deferred.
<!-- END_NB24_VERIFIED_20261102 -->'''
main=['AGENTS.md','README.md','phase3Progress.md','phase3NextSteps.md','phase3OpenItems.md','phase3Understanding.md','phase3DataRetentionPolicy.md','phase3ModelRationaleAndLogging.md','phase3TreadTwoTRACEDesign.md','phase3TreadTwoGEONextResearch.md']
def prepend(path,block):
    block=polish(block)
    body=path.read_text(encoding='utf-8');body=body.replace('## Current — NB21–NB24 TRACE batch delivered; run NB24 once','## Historical — TRACE batch delivery before NB24 execution')
    marker=block.splitlines()[0]
    if marker in body:
        end_marker=block.splitlines()[-1];body=re.sub(re.escape(marker)+r'.*?'+re.escape(end_marker),lambda _:block,body,flags=re.S)
    else:
        first,rest=body.split('\n',1);body=first+'\n\n'+block+'\n'+rest
    write(path,body)
for name in main:prepend(ROOT/name,current)
for name in ['phase3TreadTwoTRACEBatch.md','phase3TreadTwoTRACEValidation.md']:
    prepend(ROOT/name,'''<!-- NB24_ACTUAL_RESULT -->
## Current result — original NB24 complete

The actual Kaggle fit and HF review are complete. **Do not rerun the completed
batch.** [Completion and live gate proof](phase3NB24Completion.md) ·
[Actual findings, figures and next research](phase3TreadTwoTRACEFindings.md).
The delivery/local-test/run instructions below are historical reference; their
pending-live statements describe the pre-execution checkpoint.
<!-- END_NB24_ACTUAL_RESULT -->''')
indexes={
'phase3EvidenceRegister.md':f'''## NB24 verification evidence

| Record | Purpose |
|---|---|
| [HF verification]({audit_rel}/VERIFICATION.json) | Five stages, exact scope, counts, released owners and seven old manifests |
| [Science]({audit_rel}/SCIENTIFIC_VERIFICATION.json) | All2,940 objectives,60 original validation gates,946 native inputs and final complete state |
| [Object methods]({audit_rel}/OBJECT_VERIFICATION.json) | Full-body versus LFS-identity versus large-Git metadata tiers |
| [Validation](tread_two/trace_completion_report/VALIDATION_GATE_AUDIT.csv) | Independent SVD and every original gate |
| [Figures/findings](phase3TreadTwoTRACEFindings.md) | Actual utility, group variation and limits |
''',
'phase3DocumentationInventory.md':'''## NB24 completion documents

| Document | Purpose |
|---|---|
| [NB24 completion](phase3NB24Completion.md) | Actual HF identities, audit scopes, retention, recovery and resources |
| [TRACE findings](phase3TreadTwoTRACEFindings.md) | Illustrated measured gains, failed gates, component/probe diagnosis and next proposal |
| [Research model card](tread_two/trace_completion_report/MODEL_CARD.md) | Exact published unpromoted card |
''',
'phase3RepositoryMap.md':f'''## NB24 completion locations

| Location | Purpose |
|---|---|
| `{audit_rel}/` | Compact public and independent scientific audit receipts |
| `{audit_rel}/remote/` | Ignored checksum-verified downloaded HF bodies |
| `tread_two/trace_completion_report/` | Curated small CSVs, exact model card and documentation figures |
| `analysis/audit_nb24_*.py`, `analysis/verify_nb24_objects.py` | Credential-free audit; no notebook execution/training/HF write |
''',
'phase3SessionLog.md':'''## 2 November2026 — NB24 verified after user execution

1. Archived supplied executed NB24 and verified33 exact payloads /three complete code cells.
2. Checked all five public HF selecting manifests,12,262 object existence/size identities, released owners, bounded full-body hashes and explicit large-Git metadata limits.
3. Independently reconstructed all2,940 objectives and60 validation gates using946 fully checked native inputs and SVD; verified complete state, source coverage, all epoch weights and selected=None.
4. Confirmed original dense-review receipts, actual viewer, seven earlier selecting manifests, direct CSV retention and scientific limitations.
5. Added illustrated completion/findings records and current status updates. No new training, photos, local credentials, HF write or workstation change.
''',
'phase3Decisions.md':'''## NB24 measured result — 2 November2026

| ID | Decision | Evidence / boundary |
|---|---|---|
| P3-D175 | Close NB21–NB24 through actual NB24 master execution |60epochs /2,940updates and completed review; no standalone execution invented |
| P3-D176 | Keep selected=None and original gates | Utility60/60; projector/encoder diversity0/60 |
| P3-D177 | Preserve improved reconstruction without a wear-accuracy claim |21.63% masked /36.44% local control advantage on previously inspected validation |
| P3-D178 | Distinguish published full dense audit from this bounded independent audit | All objectives/gates replayed; every large CSV body was not re-downloaded |
| P3-D179 | Count observations, padding and inherited cache explicitly |47,263,346fit-publication rows;570existing photos; no new independent data |
| P3-D180 | Diagnose saved local/component evidence before another candidate | No automatic refit, post-hoc gate relaxation or workstation deployment |
'''}
for name,block in indexes.items():prepend(ROOT/name,'<!-- NB24_RECORD -->\n'+block.rstrip()+'\n<!-- END_NB24_RECORD -->')
package_notice='''<!-- NB24_COMPLETION -->
## Actual execution update — 2 November2026

NB24 completed the original one-seed fit and review. Original selected=None and
all source/contract identities are preserved. [Completion](../../phase3NB24Completion.md)
and [illustrated findings](../../phase3TreadTwoTRACEFindings.md) supersede pending
execution statements below. No rerun or new fit is required for this cycle.
<!-- END_NB24_COMPLETION -->'''
prepend(ROOT/'tread_two/trace_batch/README.md',package_notice)
for name in ('README.md','DELIVERY_PLAN.md','EXPERIMENT.md','EVIDENCE.md','OPERATIONS.md','FINDINGS.md'):
    prepend(ROOT/'tread_two/trace_design'/name,package_notice)
# Keep the original executed archive and downloaded scientific bodies out of Git.
ignore=ROOT/'.gitignore';body=ignore.read_text(encoding='utf-8')
addition='\n# NB24 immutable execution archive and downloaded public HF audit bodies.\ntread_two/nb24_completion/*/remote/\ntread_two/nb24_completion/*/executed.ipynb\n'
if 'tread_two/nb24_completion/*/remote/' not in body:write(ignore,body.rstrip()+'\n'+addition)
atomic(report/'REPORT.json',dict(verification_sha256=sha(OUT/'VERIFICATION.json'),science_sha256=sha(OUT/'SCIENTIFIC_VERIFICATION.json'),selected=None,
    figures={p.name:file_meta(p) for p in figures.glob('*.png')},new_photos=0,local_optimizer_updates=0,remote_writes=0))
print('Illustrated NB24 records written; frozen runtime/notebook unchanged.',flush=True)
