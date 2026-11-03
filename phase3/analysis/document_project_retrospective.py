"""Illustrated cross-project analysis; all plots use existing saved evidence."""
import json,re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'retrospective_20261102';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,t):
    t=re.sub(r'\b(all|All|uses|with|from|than|on|through|by|to|of|is|and|same|across|every|each|all|only|final|epoch|update|seed|median|rank|below|low|high|recall|step|against|selected|about|original|declared|their|the|its|at|both|into|has|have|under|mean|reads|are|not|exceed|Custom|Matched|HRNet|SegFormer|BA|baseline|representation|gain|diversity)(?=\d)',r'\1 ',t)
    t=re.sub(r'(?<=\d)(photos|images|frames|runs|seeds|seed|tyres|epochs|epoch|updates|parameters|references|pipelines|families|objects|NPY|PT|GB|MB|CSV|million|originals|retained|mid|positions)\b',r' \1',t)
    for a,b in [('Phase3','Phase 3'),('Phase2','Phase 2'),('all3','all 3'),('six recorded','six recorded'),('three controls','three controls'),('September24','September 24')]:t=t.replace(a,b)
    t=t.rstrip()+'\n'
    if not p.exists() or p.read_text(encoding='utf-8')!=t:p.write_text(t,encoding='utf-8',newline='\n')
analysis=read(OUT/'ANALYSIS.json');assert analysis['status']=='passed'
classification=pd.read_csv(OUT/'CLASSIFICATION_ENDPOINTS.csv');groups=pd.read_csv(OUT/'TRACE_COMPONENT_UTILITY.csv');images=pd.read_csv(OUT/'TRACE_IMAGE_UTILITY.csv');roles=pd.read_csv(OUT/'TRACE_ROLE_UTILITY.csv');tiles=pd.read_csv(OUT/'TRACE_TILE_SUMMARY.csv');fits=pd.read_csv(OUT/'PHASE3_FIT_LEDGER.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
a=axes[0,0];sel=classification[(classification.endpoint=='selected')&classification.validation_BA.notna()]
for i,r in enumerate(sel.itertuples()):
    a.scatter(r.validation_BA*100,r.test_observed_BA*100,s=45,alpha=.7)
    if r.model=='TREAD-ONE r1':a.annotate('r1 s'+str(r.seed),(r.validation_BA*100,r.test_observed_BA*100),xytext=(4,5+i*2),textcoords='offset points',fontsize=8)
    if 'projection' in r.model:a.annotate('Shared projection',(r.validation_BA*100,r.test_observed_BA*100),xytext=(-90,7),textcoords='offset points',fontsize=8)
a.plot([0,100],[0,100],ls='--',lw=1,color='#aaa');a.set(xlim=(0,105),ylim=(0,105),xlabel='Selected validation BA (%)',ylabel='Historical endpoint BA (%)',title='Selection scores do not establish transfer')
a=axes[0,1];atlas=pd.read_csv(ROOT/'tread_two/research_release/endpoint_metrics.csv');x=np.arange(3)
for j,endpoint in enumerate(['selected','final']):
    subset=atlas[atlas.endpoint==endpoint].set_index('model').loc[['v1','lc','context']];a.bar(x+(j-.5)*.32,subset.nll,.32,label=endpoint)
a.set(xticks=x,xticklabels=['ATLAS','LC','Context'],ylabel='Three-class NLL (lower better)',title='Same argmax decisions; different probability quality');a.legend()
a=axes[1,0];k=np.arange(5);values=[1.038659162,1.020989995,1.02250744,1.22962649,1.24885870]
a.bar(k,values,color=['#718793']*2+['#1a877e']*3);a.hlines([4,4],[-.4,1.6],[.4,4.4],color='#b55555',ls='--',label='Original gates: GEO projector / TRACE spaces')
a.set(xticks=k,xticklabels=['GEO\nproject64','GEO\npooled96','TRACE\nproject6','TRACE\nlocal48','TRACE\nspatial240'],ylabel='Validation effective rank',ylim=(0,4.5),title='Both representation pilots fail their original gates');a.legend(fontsize=8)
a=axes[1,1];r=roles.set_index('role').loc[['train','validation','test']]
a.bar(['Train','Validation','Historical test'],r.gain_vs_strongest_aggregate_control*100,color=['#778e9b','#218a81','#c48b45'])
a.set(ylabel='Masked loss reduction vs strongest control (%)',title='TRACE has useful reconstruction across roles')
fig.suptitle('Project retrospective — internal evidence, preserved endpoints',fontsize=17,color='#183c49');fig.savefig(FIG/'Project_Findings.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(13,4.8),constrained_layout=True)
g=groups[groups.role!='train'].copy();g['label']=g.role+' / '+g.proxy_class.map({0:'low proxy',2:'high proxy'})+' / n='+g.images.astype(str)
axes[0].barh(g.label,g.gain_visible_band_mean*100,color=['#c48b45','#c48b45','#218a81','#218a81']);axes[0].axvline(5,color='#aa5555',ls='--');axes[0].set(xlabel='Masked loss reduction vs visible-band control (%)',title='All four held-out groups improve descriptively')
for role,color in [('validation','#218a81'),('test','#c48b45')]:
    z=np.sort(images[images.role==role].gain_visible_band_mean.to_numpy()*100);axes[1].plot(np.arange(1,len(z)+1)/len(z),z,label=role,color=color)
axes[1].axhline(5,color='#aa5555',ls='--',label='5% reference line');axes[1].set(xlabel='Fraction of images, sorted within role',ylabel='Per-image gain vs visible-band control (%)',title='Recorded clean view / one fixed hidden plan');axes[1].legend()
fig.suptitle('New saved-only TRACE audit — no fitting or new image forwards',fontsize=15);fig.savefig(FIG/'TRACE_Component_Utility.png',dpi=160);plt.close(fig)
fit_rows='\n'.join(f'| {r.family} | {r.fits} | {r.fit_notebook} | {r.result} | [Record]({r.source}) |' for r in fits.itertuples())
group_rows='\n'.join(f'| {r.role} | {"low" if r.proxy_class==0 else "high"} | {r.images} | {r.gain_visible_band_mean*100:.2f}% | {r.local_gain*100:.2f}% | {r.images_beating_all_controls}/{r.images} |' for r in g.itertuples())
tile_rows='\n'.join(f'| {r.role} | {r.layer} | {r.tiles} | {r.rank_median:.3f} | {r.rank_min:.3f}–{r.rank_max:.3f} |' for r in tiles.itertuples())
main=f'''# Whole-project retrospective — what the evidence now supports

**2 November 2026 · through completed NB24 / TREAD-TWO TRACE.**

The project has built a substantial tyre-vision research system: prepared data,
multiple completed model studies, custom architectures, dense geometry, evidence
visualization, resumable Kaggle/HF execution, two desktop interfaces and a manuscript.
Its strongest measured capabilities are internal region segmentation, image-space
boundary localization and, most recently, fixed-texture reconstruction. Reliable
transfer of the mileage classifier remains unresolved.

The next useful direction is **local geometry and texture evidence whose utility
can be checked on the existing data**, followed by a separately justified model
objective. A larger architecture or another unchanged classifier fit is not the
default next step supported by these results.

This review inventories **236 maintained Markdown documents** and links the
controlling evidence. Detailed synthesis uses40 source records and a new analysis
of444 checksum-verified scientific files,59,176,731 bytes. It makes no fresh HF
availability claim: the earlier pinned HF audits remain the verification authority.
No model forward, training, new photo, source-label change or remote write occurred.

[Interactive project/TRACE explorer](retrospective_20261102/Project_Evidence_Explorer.html) ·
[New component analysis](phase3TRACEComponentReview.md) ·
[Evidence inventory](retrospective_20261102/SOURCE_EVIDENCE.csv) ·
[All-document inventory](retrospective_20261102/MARKDOWN_INVENTORY.csv).

## 1. The project has several distinct completed layers

```mermaid
flowchart TD
  A[Original study:418 photos /12 user-confirmed tyres] --> B[Architecture sweep, XAI, dense tasks, geometry and report]
  A --> C[Phase2:570 photos /same12 tyres]
  C --> D[Five model families /15 completed runs /original workstation]
  C --> E[TREAD-ONE custom shared model /3 seeds]
  E --> F[Diagnostics and matched gradient-projection pair]
  E --> S[Separate TREAD-ONE Studio]
  E --> G[ATLAS reference classifier]
  G --> H[LC appearance revision]
  H --> I[Context residual /same classifier failure]
  E --> J[GEO task-isolated geometry and representation]
  J --> K[TRACE frozen geometry /fixed texture targets]
  K --> L[Useful reconstruction /diversity still rejected]
```

| Layer | Completed work | Main scientific boundary |
|---|---|---|
| Original study |153 retained architecture-sweep runs;108 OFAT runs;18 confirmation runs;81 manual-supervised dense runs; geometry studies and saved-result reviews | Different protocols/roles; suspected old fold overlap, few capture groups, proxy labels and selection effects |
| Original geometry / paper | Matched3-seed HRNet and SegFormer study;28 pipelines ×9-image visual replay; revised manuscript with34 references | HRNet1.312% vs SegFormer1.721% width error is an internal, differently supervised comparison; no external physical validation or submission claim |
| Phase2 |418 originals +152 frames from existing mid tyres; all15 combined-data runs and final NB06 complete; five original installed selections retained | More views of existing tyres; held-out mid absent; original-data-only controls not performed |
| Phase3 TREAD-ONE | Custom1,372,159-parameter multitask model;3 seeds; masks, derived boxes, six points and additive class evidence; independent Studio | Internal dense quality; poor mileage transfer; evidence accounting does not validate semantic reasoning |
| Phase3 TREAD-TWO | ATLAS, LC, context, GEO and TRACE studies, engineering gates and postrun reviews complete | Different revisions answer different hypotheses; no unselected model is promoted |
| Applications | Original Tread Station and isolated TREAD-ONE Studio; evidence export/restore; target-assisted alignment software | Implemented software is separate from field accuracy, physical calibration and reliable classification |

Original-study counts describe named tracks, not a deduplicated grand total of
all training in the repository. [Original sweep](../docs/18_STAGE_A_RESULTS.md),
[confirmation](../docs/23_S4B_CONFIRMATION.md), [dense study](../docs/24_S5_MANUAL_DENSE_TASKS.md),
[paper revision](../ACCESS_latex_template_20260513/SUBMISSION_REVISION_20260924.md).

The original study already showed checkpoint sensitivity and strong trivial
colour/texture baselines on some folds. Two folds carried suspected overlap
flags; later user identity confirmation and recorded-group disjointness do not
turn those historical flags into independently proven physical identity. Its
same-fold S4b confirmation found that random initialization hurt both added
architectures (selected macro-F1 deltas−0.428566 /−0.371504), while sampling
directions did not repeat across both models. Architecture breadth, pretrained
versus custom initialization, sampling and checkpoint choice must therefore be
treated as different experimental questions, not one leaderboard.

## 2. What Phase3 actually trained

There are **seven measured fit families /ten fit executions /600 epochs /
29,400 optimizer updates** through NB24. This excludes disposable engineering
updates. A continuation does not count as a new fit: NB13A finished NB13. Masters
do not double the fit count: NB24 executed the NB22 training stage, and NB20
executed the GEO postrun stages. Reviews and diagnostics are not extra models.

| Family | Fit executions | Training notebook / orchestration | Measured outcome | Authority |
|---|---:|---|---|---|
{fit_rows}

The two paired conditions are fit executions, not two new architectures or new
independent tyre cohorts. One seed for each later candidate gives a measured
result for that configuration; it does not establish seed stability.
[Fit ledger](retrospective_20261102/PHASE3_FIT_LEDGER.csv).

![Cross-experiment observed findings](retrospective_20261102/figures/Project_Findings.png)

The figure is descriptive. The cohorts, tasks, architectures and selection rules
are not interchangeable. No pooled significance test, architecture ranking,
test-selected checkpoint or matched GEO-versus-TRACE effect is inferred.
GEO pooled96 rank is a diagnostic, not an additional original rank gate; the
red threshold is drawn only over the spaces constrained by their actual rules.

## 3. Dense vision is the strongest established part

| Existing selected artifact | Historical task result | Interpretation |
|---|---|---|
| Phase2 SegFormer seed2 /epoch29 |98.957% mean two-region Dice | Strong internal tyre/tread masks |
| Phase2 YOLO26 Medium seed1 /epoch10 |98.790% Dice | Strong internal region prediction under its recipe |
| Phase2 HRNet seed3 /epoch2 |0.859% image-width point MAE,25 human-labelled test photos | Useful image-space boundaries; not millimetres or alignment degrees |
| TREAD-ONE seed3 /epoch20 |98.419% Dice /1.196% point MAE | Compact shared implementation; no demonstrated dense-score improvement over installed models |
| GEO final, validation only |98.986% Dice /0.7662% point MAE | Geometry floors pass60/60; point error5.23% below G0 retrospectively; original joint selection still unavailable |

GEO's validation row must not be ranked against the other rows' historical test
scores. Dice uses overlapping tyre/tread channels at model-input resolution,
excluding ignore pixels. Point error uses eligible human labels and image width;
derived weak targets are not new independent human truth. Boxes obtained from
masks are localization outputs, not a separately human-validated damage detector.
[Phase2 selections](retrospective_20261102/PHASE2_SELECTIONS.csv),
[TREAD-ONE audit](phase3CompletionAudit.md), [GEO audit](phase3NB17Completion.md).

## 4. Classification failure persists through several different designs

| Preserved selected endpoint | Validation BA | Historical observed-class BA | Main finding |
|---|---:|---:|---|
| Phase2 MobileNetV4 seed1 |100% |50% | A perfect selected validation score did not transfer |
| Phase2 ResNet50 seed2 |85% |50% | Same limitation in a different family |
| TREAD-ONE r1 seeds1/2/3 |55% /82.5% /100% |50% /2.439% /50% | Strong seed/selection instability; selected-seed mean34.146% |
| Paired shared projection seed1 |83.75% |17.073% | Worse than matched baseline50%; point error also worse |
| ATLAS v1 |50% |50% |62 low→mid;41 high→high; appearance sensitivity |
| ATLAS-LC |56.25% |50% | Same low failure; selected NLL improves to0.961273 |
| ATLAS-CONTEXT |50% |50% | Same low failure; selected NLL worsens to2.174813 |

The historical endpoint has103 photos but only **two known tyres**,62 low and41
high, with no mid targets. Observed-class BA averages low/high recall while mid
predictions still count as errors. The repeated50% result here is specifically
low recall0% /high recall100%; it is not a three-class success rate or a calibrated
chance benchmark. Mid recall is unavailable, not a measured zero.

Selection is preserved. MobileNet seed3's94.35% test BA does not authorize
substituting it for the original validation-tied seed1. An attractive late or
test-favoured checkpoint is not a new independent result. These endpoints have
already influenced development; they are historical development evidence.
[Endpoint CSV](retrospective_20261102/CLASSIFICATION_ENDPOINTS.csv).

### The revisions were motivated; their fixes answered narrower questions

| Observation → design response | What improved | What remained unresolved |
|---|---|---|
| Opposing task gradients → shared-gradient projection | Intervention occurred on91.84% of candidate updates; shared/head accounting retained | Selected classification regressed32.93 percentage points; human point error rose about76.6% |
| v1 brightness/grayscale sensitivity → LC representation +paired views | Saved appearance probes stabilized; all386 train queries covered; exact-source self-reference blocked | Low recall remained0/62; selected ECE worsened despite better NLL/Brier |
| Equal-band pooling ignored order → bounded positional context | Model became layout-sensitive; corrections and attention were fully recorded | Training confidence increased; low predictions remained wrong and NLL worsened |
| Shared-task concerns → GEO task isolation | Geometry remained useful without shared trainable task gradients | Texture representation failed gates; learned error heads lost to constant controls |
| Correlated/self targets and64D/batch8 issue → TRACE fixed local targets and6D diversity head | Reconstruction beats fixed controls; local tokens retain more variation than pooled summaries | Original projector/encoder gates still reject all60 epochs |

These are completed hypothesis tests, not evidence that every proposed remedy
worked or that a single failure proves the mechanism causing all later errors.

## 5. TRACE provides real local utility despite failing its global gate

The new saved-only analysis partitions final-epoch reconstruction into individual
images and all ten conservative components. It does not fit or select a model.

| Role | Images | Masked gain vs strongest aggregate control | Local gain vs position control | Images beating all3 controls | Images with ≥5% gain vs all3 |
|---|---:|---:|---:|---:|---:|
| Train |386 |45.93% |53.63% |386/386 |386/386 |
| Validation |81 |21.63% |36.44% |81/81 |81/81 |
| Historical test |103 |17.85% |37.80% |103/103 |98/103 |

Both validation and both test components improve descriptively; the gain is not
confined to one component. This is encouraging for image-derived texture
inspection. It is **not** a claim that all570 images satisfy every selection
criterion or that570 independent tyres were evaluated. The calculation uses
final epoch60, clean view0 and its recorded hidden-tile plan, on common support.

The global representation remains dominated by between-component variation:
96.78% for validation local pooling and96.56% for spatial moments. Yet the saved
per-position local encoder rank has median2.899 on validation, compared with
global local-pooling rank1.230; within-component pooled rank is9.782. Local
texture information can exist while aggregate vectors remain narrow. This
motivates investigation of local evidence-preserving aggregation; it does not
prove that a replacement pooling rule would fix classification.

[Complete new diagnostics and caveats](phase3TRACEComponentReview.md).

## 6. The dataset constrains the claims and evaluation

| Quantity | Current fact | Consequence |
|---|---|---|
| Photos |570 =418 originals +152 frames of existing mid tyres | The152 frames add views, not new tyres |
| Known identities |12 user-confirmed tyres; no independent identity audit | Preserve source/group constraints and confidence of identity claims |
| Fixed roles |386 train /81 validation /103 historical test | Keep these roles and provenance; no hidden relabelling or random frame split |
| Training proxy classes |67 low /249 mid /70 high images | Image counts do not describe independent support |
| Known train tyres by class |1 low /3 mid /4 high | Low-class diversity is particularly limited |
| Conservative train components |1 low /1 unresolved-mid component /4 high | Withholding low or mid component removes a fit class; two of six conditional holdouts lack full class support |
| Human point truth |70 train /25 validation /25 test images |152 weak derived proposals and164 train images without point targets remain distinguishable |
| Targets |Mileage proxies; no measured tread-depth/safety truth | No physical wear, roadworthiness or millimetre-depth accuracy claim |
| User scope |Existing photos only; no further acquisition | Adapt task/evidence to available data; new photos are not a prerequisite for current analysis |

Conservative mid grouping preserves unresolved clip-parent relationships. It
does not assert the three known mid tyres are physically identical. Repeated
views, annotations, seeds, CSV cells, synthetic augmentation and model-generated
labels do not increase independent tyre count. A support gate returning false
is an informative result, not an error to fix by rerunning unchanged inputs.
[Readiness evidence](phase3NB15Completion.md), [source release](../phase2/phase2DatasetReadme.md).

## 7. Explainability, reliability and physical measurement are separate

| Implemented evidence | Established property | Additional claim still unsupported |
|---|---|---|
| Signed TREAD-ONE /ATLAS patch contributions | Logits reconstruct from retained contributions and bias | Evidence necessarily follows physical wear or establishes input causality |
| Attention and spatial interventions | Internal distributions and sensitivity to a defined perturbation | Attention is a causal explanation or calibrated damage map |
| TRACE residual maps | Standardized fixed-descriptor reconstruction error with support/missingness | Residual equals depth, damage, hazard or confidence |
| NLL/Brier/ECE | Different properties of saved proxy probabilities | One aggregate number certifies useful classification or deployment reliability |
| Learned GEO error estimates | Raw predictions/control comparison retained | Quality heads are validated uncertainty; they lost to constant controls |
| Target-assisted alignment software | Implemented calibrated-target workflow and synthetic checks | Independently validated physical wheel-angle accuracy or roadworthiness |

The original report also retains negative ROI/fusion findings: predicted tyre
and tread crops changed mean macro-F1 by−0.01257 /−0.01467 in the tested S5
scope; equal full+tyre+tread fusion had mean delta−0.013238. Better segmentation
does not automatically improve a frozen classifier. The original XAI advantage
hypothesis was unsupported, H2 inconclusive/undefined and H3 untested in its
declared scope. [Original claims ledger](../docs/report/CLAIMS_AND_LIMITATIONS.md).

## 8. Engineering progress is substantial and should be retained

The project now has source/data/environment-bound experiments, direct readable
CSV/native evidence, complete-update state, actual HF restore/continue gates,
transaction fault checks, resource accounting, owner fencing, quota/backoff
handling and automatic sequential notebook masters. NB13A actually resumed the
original run at update2,486 and completed its454 remaining updates. Catchable
stop publication is implemented; an uncatchable container kill cannot guarantee
cleanup or recover work never published.

| Evidence layer | Representative retained scope | Count boundary |
|---|---|---|
| TREAD-ONE r1 /NB02 |191,904 scientific rows, direct recovery from old archives | Original missing training-detail fields remain missing |
| Paired TREAD-ONE |1,561,734 core rows,412 endpoint maps |23 core families; broader auxiliary counts have a different scope |
| ATLAS-LC |54,336,368 raw scientific rows |Many affinity/pixel rows per correlated observation |
| ATLAS-CONTEXT |58,176,066 raw rows |Richer same-data evidence; not new subjects |
| GEO |27,514,418 raw CSV rows |Completed fit; no joint selected checkpoint |
| TRACE |47,263,346 fit-publication rows, including297,021 inherited cache rows;13.507GB |3,553 CSVs /6,725NPY /64PT; scope differs from GEO |

These totals must not be added as independent training examples or directly
ranked as model quality. TRACE's published dense audit and the later bounded
independent objective/gate audit have different verified scopes. Every report
must say which it performed. Unrecorded historical optimizer states, continuous
energy, full video truth and physical labels cannot be recreated by prose.

## 9. Application, manuscript and documentation status

Original Tread Station retains the five Phase2 validation-selected models.
Separate TREAD-ONE Studio retains seed3/epoch20 and its own sources/settings/
outputs. Both are implemented; current workstation revisions are deferred by the
user. No ATLAS/GEO/TRACE candidate was installed by this analysis.

The September24 manuscript is a completed author-review artifact for its
original-study scope, with34 references and verified original-model replay.
Phase3 results are not silently incorporated into that manuscript, and no
submission, approval or new external validation is claimed. Original papers,
root documentation and model sources are preserved.

Documentation now spans several scopes and many historical “current/next”
blocks. Resolve status by **experiment namespace → exact artifact/endpoint →
later relevant completion receipt**. A September root-current page does not
replace November Phase3 evidence, and a new Phase3 result does not rewrite an
original paper experiment. This retrospective supplies navigation rather than
deleting earlier decisions. [Precedence ledger](phase3DocumentationConflicts.md).

## 10. What we should do next

**Immediate outcome of this analysis:** the new570-image /10-component TRACE
diagnostic is complete locally. It confirms broad internal reconstruction
utility and a local-versus-pooled representation difference. Its CSVs/HTML are
local research derivatives; they have not been published to HF in this pass.

| Priority | Existing-data action | Decision it should support |
|---|---|---|
|1 | Examine native local tokens, descriptor channels, residuals and support together; preserve train-only controls | Does local evidence add useful structure beyond fixed descriptors and position? |
|2 | Separate component-centred signal, coverage and aggregation effects without using labels to fit held-out roles | Is useful local information being lost or dominated during aggregation? |
|3 | Define an inspection task with labels/targets that actually exist; write its controls, failure criteria and retention first | Is another single custom fit scientifically justified? |
|4 | If justified, freeze one new protocol and deliver the complete automatic notebook batch | Test the declared hypothesis without changing completed runs or calling old test data blind |
| Deferred | Workstation changes, physical alignment validation and paper expansion | Respect user scope; acknowledge existing implementation and missing physical evidence |

Preserve every original selection and `selected=None` for GEO/TRACE. Do not relax
the failed gates after seeing results. Avoid another same-objective sweep, a
transformer substitution justified only by novelty, or more epochs presented as
new independent evidence. The next model should be designed around a supported
local-structure task and demonstrated controls, with honest evaluation limits.
'''
write(ROOT/'phase3ProjectRetrospective.md',main)
component=f'''# TRACE component review — new saved-evidence analysis

**2 November 2026. Zero fitting, zero image forwards, zero new photos.**
This is an additional retrospective analysis of the original final epoch60,
update2,940. It does not alter the original NB24 gate or selected=None.

## Question and data

Is TRACE's aggregate reconstruction gain confined to one group, and does low
pooled rank imply that all local texture information disappeared?

The analysis reads444 fully checksum-verified saved CSV files,59,176,731 bytes,
from the previously verified actual Kaggle fit. The source/cache identity and
completed state supply image roles, conservative components and native record
positions. No model is loaded for inference. Source data remain unchanged.
[Exact consumed-input inventory](retrospective_20261102/CONSUMED_SCIENCE_INPUTS.csv).

## Results by original held-out component

![Component and image utility](retrospective_20261102/figures/TRACE_Component_Utility.png)

| Role | Mileage proxy | Photos | Masked gain vs visible-band control | Local gain vs position control | Images beating all3 controls |
|---|---|---:|---:|---:|---:|
{group_rows}

The four component-level masked gains are23.17% /19.72% for validation low/high
and21.11% /10.52% for test low/high. All ten training/validation/test components
have a positive aggregate advantage over every declared masked control. Every
one of570 images has lower masked loss than all three controls on this saved
view/plan. All81 validation images, and98 of103 test images, exceed5% versus
each control. The five remaining test images still improve, by less than5%
against at least one control.

These are correlated image records from ten conservative components, not570
independent subjects or a new test set. No image-level binomial confidence
interval, p-value or universal perturbation guarantee is added.

## Local tokens and pooled features differ

The following summarizes the **retained per-tile statistics**, not a fresh
native-token covariance replay. Each role/layer has32 positions; only supported
images enter each position's statistic. The counts/dimensions are not directly
interchangeable with one globally pooled vector per image.

| Role | Saved layer | Positions | Median per-position effective rank | Position range |
|---|---|---:|---:|---:|
{tile_rows}

Validation's encoded median rank2.899, full-context median2.127 and globally
pooled local rank1.230 support a local-versus-aggregate distinction. Native
validation within-component ranks9.782 (local) and11.272 (spatial moments)
are preserved from the independent NB24 audit. Training local-token median5.953
and validation2.899 also show that local representations do not transfer
unchanged. This does not isolate a causal effect of pooling or prove that a new
pooling layer would improve mileage recognition.

## Exact calculation and boundaries

For each image and control, `gain = 1 - model_image_mean / control_image_mean`.
Component/role gain is the ratio of **equal-image mean losses**, not the mean
of per-image ratios or a pooled per-pixel loss. Local comparison uses local
supported cells; masked comparison uses common supported hidden cells. Original
normalization/controls remain train-only.

Only final endpoint, clean view0 and its recorded mask plan are analyzed. A
new plan, all appearance variants, every epoch or a physical-damage target is
not implied. The endpoint has already informed development. The previous
validation gate correctly still rejects all60 epochs for global diversity.

| New derivative | Rows / purpose |
|---|---|
| [Image utility](retrospective_20261102/TRACE_IMAGE_UTILITY.csv) |570 image-level comparisons, exact source SHA, role, component and losses |
| [Component utility](retrospective_20261102/TRACE_COMPONENT_UTILITY.csv) |10 component summaries |
| [Role utility](retrospective_20261102/TRACE_ROLE_UTILITY.csv) |3 original-role summaries, reconciled to the saved NB24 endpoint |
| [Tile summaries](retrospective_20261102/TRACE_TILE_SUMMARY.csv) |6 retained-statistic summaries |
| [Analysis receipt](retrospective_20261102/ANALYSIS.json) | Scope, inputs, zero fitting/network activity and reconciled role values |

These derivatives are local and not uploaded to HF during this analysis. The
original raw science is already published at NB24's pinned fit revision.
[Whole-project interpretation](phase3ProjectRetrospective.md) ·
[Original NB24 completion and audit limits](phase3NB24Completion.md).
'''
write(ROOT/'phase3TRACEComponentReview.md',component)
index='''# Whole-project retrospective evidence package

**2 November 2026.** New local analysis/figures/navigation; original science
remains at its previously verified HF revisions. No new HF publication occurred.

[Whole-project report](../phase3ProjectRetrospective.md) ·
[New TRACE analysis](../phase3TRACEComponentReview.md) ·
[Interactive explorer](Project_Evidence_Explorer.html).

| File | Grain / scope |
|---|---|
| MARKDOWN_INVENTORY.csv |236 maintained Markdown documents at the pre-analysis snapshot; size/hash/scope |
| MARKDOWN_SECTIONS.json |Extracted headings/tables for that inventory; navigation, not a rerun instruction |
| SOURCE_EVIDENCE.csv |40 controlling report/data/state inputs with exact hashes |
| CONSUMED_SCIENCE_INPUTS.csv |444 raw scientific CSV bodies read and checked for the new analysis |
| PHASE3_FIT_LEDGER.csv |7 families /10 measured fit executions;600 epochs /29,400 updates; engineering excluded |
| CLASSIFICATION_ENDPOINTS.csv |Preserved selected/final endpoints; BA fractions; no cross-protocol ranking or new selection |
| PHASE2_SELECTIONS.csv |Original five installed validation selections; metric directions explicitly recorded |
| TRACE_IMAGE_UTILITY.csv |570 final clean-view image comparisons; no new masks, targets or predictions |
| TRACE_COMPONENT_UTILITY.csv |10 conservative components; groups are not interchangeable with known physical tyres |
| TRACE_ROLE_UTILITY.csv |3 original-role summaries; equal-image means; reconcile to saved endpoint |
| TRACE_TILE_SUMMARY.csv |6 summaries of saved per-tile statistics; not a new native-token covariance audit |
| ANALYSIS.json |Actual calculation scope and zero-training/forward/network/write receipt |
| figures/ |Two generated scientific figures from the listed source tables |

Missing values in classification comparisons mean unavailable/not applicable,
not zero. Losses and gains are fractions; diagram/chart axes explicitly convert
to percent. Point errors are image-width fractions, not physical units. Counts
are observations, not independent subjects. CSVs are analysis data, not new
training photos. The published raw evidence retains its original protocol.
'''
write(OUT/'README.md',index)
print('Retrospective reports, source tables and two figures prepared.')
