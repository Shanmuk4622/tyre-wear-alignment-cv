"""Figures and evidence-linked interpretation of the completed aggregation assay."""
import re,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'texture_aggregation_20261102';F=D/'figures';F.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,t):
    if p.suffix=='.md':
        t=re.sub(r'\b(All|all|View|view|epoch|update|seed|plus|into|contain|uses|with|from|to|versus|of|for|rank|only|original|keeps|retain|retains|losing|restores|preserves|gives|is|has|in|not|full|by|below|above|performed|and|verified|total|after|before|its|than|on|both|a|the|within|about|readouts|floor|rcond|beta)(?=\d)',r'\1 ',t)
        t=re.sub(r'(?<=\d)(fields|cells|train|validation|historical|position|grid|images|of|decoder|projector|fixed|supported|counterfactual|native|predictions|MB|GB)\b',r' \1',t)
    t=t.rstrip()+'\n'
    if not p.exists() or p.read_text(encoding='utf-8')!=t:p.write_text(t,encoding='utf-8',newline='\n')
a=read(D/'ANALYSIS.json');assert a['status']=='passed'
u=pd.read_csv(D/'UTILITY_SUMMARY.csv');s=pd.read_csv(D/'FEATURE_SPACES.csv');c=pd.read_csv(D/'SPATIAL_COVARIANCE.csv');ch=pd.read_csv(D/'CHANNEL_UTILITY_SUMMARY.csv');comp=pd.read_csv(D/'COMPONENT_UTILITY.csv');cka=pd.read_csv(D/'MASK_PLAN_ALIGNMENT.csv')
cfg=read(D/'PROTOCOL.json');variants=cfg['variants'];names={'identity':'Original unpooled','global_mean':'One average','band_means':'8 band means','column_means':'4 column means','moments5':'5 spatial moments','random5':'5 fixed random modes','train_template':'Fixed train template','residual_mean':'Train template + mean residual','residual_moments5':'Train template + 5 residual moments'}
def get(layer,role,variant,view=0):return u[(u.layer==layer)&(u.role==role)&(u.variant==variant)&(u.view==view)].iloc[0]
def rank(layer,space,role='validation',view=0):return s[(s.layer==layer)&(s.space==space)&(s.role==role)&(s.view==view)&(~s.component_centered)].iloc[0].effective_rank
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(17,6),constrained_layout=True)
x=np.arange(len(variants));ax=axes[0]
for j,(role,color) in enumerate([('validation','#218a81'),('test','#c48b45')]):ax.barh(x+(j-.5)*.34,[get('encoded',role,v).retained_gain_fraction*100 for v in variants],.34,label=role,color=color)
ax.set(yticks=x,yticklabels=[names[v] for v in variants],xlabel='Retained original advantage (%)',title='Local decoder: useful detail survives unevenly',ylim=(-.7,len(variants)+1.2));ax.axvline(100,color='#999',ls='--',lw=1);ax.legend()
ax=axes[1];g=comp[(comp.layer=='encoded')&(comp.view==0)&(comp.role!='train')];groupnames=list(g.component.unique());xx=np.arange(4)
for j,(variant,color) in enumerate([('global_mean','#8aa0ac'),('residual_moments5','#218a81')]):
    rows=g[g.variant==variant].set_index('component').loc[groupnames];ax.bar(xx+(j-.5)*.34,rows.retained_gain_fraction*100,.34,label=names[variant],color=color)
labels=[]
for component in groupnames:
    r=g[g.component==component].iloc[0];labels.append(('Val' if r.role=='validation' else 'Test')+'\n'+('Low proxy' if '000100' in component else 'High proxy'))
ax.set(xticks=xx,xticklabels=labels,ylabel='Retained local control advantage (%)',title='Low-proxy groups need more spatial detail',ylim=(0,110));ax.legend(fontsize=8)
ax=axes[2];channels=list(ch.channel_name.unique());xx=np.arange(6)
for j,(variant,color) in enumerate([('global_mean','#8aa0ac'),('residual_moments5','#218a81')]):
    rows=ch[(ch.layer=='encoded')&(ch.role=='validation')&(ch.view==0)&(ch.variant==variant)].set_index('channel_name').loc[channels]
    ax.bar(xx+(j-.5)*.34,rows.loss_delta,.34,label=names[variant],color=color)
ax.set(xticks=xx,xticklabels=['Contrast','Gradient\nenergy','Axis\norientation','Cross\norientation','Relative\ncurvature','Contrast\noccupancy'],ylabel='Extra standardized Huber loss',title='Averaging especially hurts orientation',ylim=(0,.255));ax.tick_params(axis='x',labelsize=8);ax.legend(fontsize=8)
fig.suptitle('TRACE saved-feature counterfactuals — same frozen decoder, no retraining',fontsize=17,color='#183c49');fig.savefig(F/'Aggregation_Utility.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(13,8.5),constrained_layout=True)
ax=axes[0,0];labels=['Local\nconcat1536','Full context\nconcat1536','Local\nmean48','Context\nmean48','Actual\nspatial240','Actual\nprojected6'];values=[rank('encoded','concat1536'),rank('full_context','concat1536'),rank('encoded','mean48'),rank('full_context','mean48'),rank('full_context','moments240'),rank('full_context','actual_projected6')]
ax.bar(np.arange(6),values,color=['#527e8e','#527e8e','#218a81','#218a81','#c48b45','#c48b45']);ax.set(xticks=np.arange(6),xticklabels=labels,ylabel='Effective rank',title='Context and pooling both change the spectrum');ax.tick_params(axis='x',labelsize=8)
ax=axes[0,1];xx=np.arange(3)
for j,(center,color) in enumerate([(False,'#527e8e'),(True,'#c48b45')]):
    rows=c[(c.role=='validation')&(c.view==0)&(c.component_centered==center)].set_index('layer').loc[['encoded','full_context','masked_context']]
    ax.bar(xx+(j-.5)*.34,rows.mean_mode_energy_fraction*100,.34,label='Component-centered' if center else 'Total variation',color=color)
ax.set(xticks=xx,xticklabels=['Local','Full context','Masked context'],ylabel='Tile variance retained in mean mode (%)',title='Averaging removes most within-group variation');ax.legend(fontsize=8)
ax=axes[1,0];fields=['full_context_concat','masked_context_concat','masked_minus_full_context'];values=[cka[(cka.role=='validation')&(cka.view==0)&(~cka.component_centered)&(cka.feature==k)].iloc[0].linear_CKA_with_mask_plan for k in fields]
ax.bar(['Full context','Masked context','Masked minus full'],values,color=['#527e8e','#c48b45','#c48b45']);ax.set(ylim=(0,1),ylabel='Linear CKA with mask plan',title='Mask-placement variation can inflate feature rank')
ax=axes[1,1];values=[rank('full_context','moments240'),rank('full_context','moments240_train_scaled')];ax.bar(['Actual coordinates','Train-only rescaling'],values,color=['#527e8e','#c48b45']);ax.set(ylabel='Validation effective rank',title='Invertible scaling raises rank without new information');ax.text(.5,max(values)*.6,'Same information; exact inverse verified',ha='center',fontsize=9)
fig.suptitle('Mechanism diagnostics — original validation, clean view0',fontsize=17,color='#183c49');fig.savefig(F/'Aggregation_Mechanism.png',dpi=160);plt.close(fig)
rows=[]
for v in variants:rows.append('| '+names[v]+' | '+' | '.join(f'{get(layer,role,v).retained_gain_fraction*100:.2f}%' for layer,role in [('encoded','validation'),('encoded','test'),('masked_context','validation'),('masked_context','test')])+' |')
table='\n'.join(rows)
chanrows=[]
for channel in channels:
    r=ch[(ch.layer=='encoded')&(ch.role=='validation')&(ch.view==0)&(ch.variant=='global_mean')&(ch.channel_name==channel)].iloc[0]
    chanrows.append(f'| {channel.replace("_"," ")} | {r.identity_loss:.6f} | {r.loss:.6f} | {r.position_control_loss:.6f} |')
main=f'''# TRACE aggregation investigation — where useful texture information survives

**2 November 2026. Investigation complete on the saved NB24 endpoint.**

Useful, image-dependent texture information exists in the local tokens. Simple
averaging discards part of its reconstruction utility and preferentially removes
within-component variation. However, the original local decoder bypasses global
pooling, context already changes diversity before pooling, and rank alone is
not a reliable measure of task usefulness. The evidence supports preserving
local spatial residuals; it does not identify a ready replacement model.

[Interactive aggregation explorer](texture_aggregation_20261102/Aggregation_Explorer.html) ·
[Exact methods/data dictionary](texture_aggregation_20261102/METHODS.md) ·
[Frozen protocol](texture_aggregation_20261102/PROTOCOL.json) ·
[Analysis receipt](texture_aggregation_20261102/ANALYSIS.json).

## 1. The useful decoder and the low-rank vector use different paths

```mermaid
flowchart LR
  Z[Saved local tokens:32 x48] --> LD[Local decoder:fixed texture targets]
  Z --> H[Position metadata and axial context]
  H --> P[Five pooled spatial moments:240D]
  P --> Q[Auxiliary6D projector]
  M[Saved masked-context tokens] --> MD[Masked decoder: hidden fixed targets]
  Z -. Offline aggregation interventions .-> C[Same frozen decoder]
  M -. Offline aggregation interventions .-> C
```

The local reconstruction loss is computed directly from unpooled encoder tokens.
The masked reconstruction loss uses spatial masked-context tokens. Neither
primary decoder reads the auxiliary6D projector output. Consequently, strong local utility
and a narrow auxiliary global representation can coexist. This is established
by the frozen source graph and replay of every saved bridge.

All7,004,160 original local/masked prediction values were replayed through the
unchanged decoder on retained features. Maximum absolute differences were
1.01e-6 /2.23e-6 for local/masked heads. Saved mean, spatial-moment and projector
bridges also reproduce within6.05e-7. No image encoder or geometry inference ran.
[Numerical bridge checks](texture_aggregation_20261102/BRIDGE_VERIFICATION.csv).

## 2. Controlled aggregation loses useful reconstruction detail

Nine comparisons were declared before the primary calculation. All use the same
original final epoch60/update2,940, both retained views, targets, support and
frozen decoder. View0 is primary. Training templates/scales use only386 train
images; no held-out target, label or feature mean fits an inference transform.

The table reports **retained original advantage over the train-position control**:

`R = (L_position_control - L_aggregation) / (L_position_control - L_original)`

Role losses are equal-image means on matched cells.100% means original utility;
0% means the position control; negative means worse than that control. These are
not accuracy percentages. The position control here differs from the stronger
visible-band control used for NB24's original21.63% masked-gain headline.

| Frozen-feature intervention | Local validation | Local historical test | Masked validation | Masked historical test |
|---|---:|---:|---:|---:|
{table}

![Preserved utility, component differences and descriptor errors](texture_aggregation_20261102/figures/Aggregation_Utility.png)

Replacing each local token with the image's weighted average retains61.14% of
the validation advantage, losing38.86% of that margin. A train-position template
plus one global residual restores75.09%; five spatial residual moments restore
81.76%. The template alone preserves only7.75%. The benefit therefore includes
image-dependent information beyond a fixed position template.

Spatial structure helps: five actual spatial moments retain76.05% local
validation utility, versus65.47% for five fixed random spatial modes with a
constant mode. But there is no universal winning compression. Four column means
retain75.90% masked validation utility, while raw moments retain68.39% and the
template-plus-residual moments retain65.82%. All nine comparisons remain visible;
no alternative was test-selected or installed.

These are **fixed-head counterfactuals**. Projected feature vectors may be off
the decoder's training manifold. Loss changes combine discarded information and
compatibility with this decoder; they do not predict what a newly trained head
or a different architecture could achieve.

## 3. Orientation and contrast are especially vulnerable

The following local-decoder validation losses use the same standardized target
units and support. Lower is better. Original control advantages are positive
for all six descriptors, but their sizes differ.

| Image-derived descriptor | Original local loss | One-average loss | Train-position control loss |
|---|---:|---:|---:|
{chr(10).join(chanrows)}

Averaging preserves much of the advantage for gradient energy, relative curvature
and contrast occupancy, while damaging orientation. Cross-orientation becomes
worse than the position control (0.790882 versus0.667389), despite the original
local decoder beating it (0.616969). With template-plus-five residual moments,
axis-orientation loss falls to0.454548 and cross-orientation to0.650077, but
contrast loss remains0.626081 versus original0.575280. Low-order moments recover
some orientation detail; they do not preserve all local structure.

These fields describe image contrast/gradients. They are not measured groove
depth, physical damage or safety labels. [All channel comparisons](texture_aggregation_20261102/CHANNEL_UTILITY_SUMMARY.csv).

## 4. Averaging favors coherent group differences over local variation

![Feature spectra, spatial covariance, mask alignment and scale dependence](texture_aggregation_20261102/figures/Aggregation_Mechanism.png)

On primary validation, concatenated local tokens have effective rank3.473;
concatenated full-context tokens have rank2.416. Thus context already changes
the feature spectrum before global pooling. Local mean rank is1.230, full-context
mean1.132, actual spatial-moment rank1.249 and actual projector rank1.023.
Different dimensions are explicitly named; the concatenated space also has a
larger possible rank. These numbers are descriptive, not matched architecture
comparisons or replacement selection thresholds.

Using a fixed train-derived coverage mass, the constant spatial mean mode keeps
30.10% of total local token variance, but only4.15% after component centering.
Under this defined projection,95.85% of within-component local variance lies
outside the constant mean mode. This is **not95.85% of useful wear information**.
The utility experiment above measures a different, task-specific quantity.

Mean off-diagonal tile correlation drops from0.2731 to0.00947 after component
centering. Shared between-group changes are coherent across positions; many
within-group changes are not. Averaging preferentially retains the former.
The between-component fraction rises from76.56% for concatenated local tokens
to96.78% for the original local mean. This explains the observed statistical
concentration without identifying a physical cause or validating class transfer.

Centering by the already known component is a diagnostic only. It is not a
deployable correction, and components are strongly confounded with proxy class.
Removing identity/group signal indiscriminately could remove class information.
Both held-out roles have full32-tile support. Two train images have unsupported
tiles; the support table records this rather than calling all570 images complete.
[Covariance and exact decomposition](texture_aggregation_20261102/SPATIAL_COVARIANCE.csv) ·
[Support](texture_aggregation_20261102/SUPPORT_SUMMARY.csv).

## 5. The effect differs across the two proxy groups

For the historical low-proxy test component, one-average local utility retention
is32.03%, versus84.02% for the high-proxy component. Template-plus-five residual
moments gives74.82% /87.52%, respectively. The masked branch similarly retains
only23.16% /81.04% under one average.

This indicates that fine spatial detail matters more to the recorded descriptor
task in the low-proxy component. Each role still has only two conservative
components. It does not establish why earlier mileage classifiers failed,
demonstrate a repaired classifier, or separate mileage from tyre identity.
[Every component/variant](texture_aggregation_20261102/COMPONENT_UTILITY.csv).

## 6. Two ways to mistake numerical diversity for useful information

**Rescaling:** train-only coordinate standardization raises actual240D full-context
moment rank from1.249 to2.912 on validation. The inverse recovers the same feature
values exactly within1e-12. It adds no information. Coordinate balance matters
to the rank statistic, so increasing effective rank alone is not a model remedy.
The original unscaled gate stays unchanged and selected=None remains preserved.

**Mask layout:** masked-context concatenation has rank15.31, much higher than
full-context2.42. An exploratory follow-up, separately declared after that result,
finds linear centered-kernel alignment0.888 between the masked-minus-full feature
change and hidden-tile layout; full-context alignment is0.0356. After component
centering, masked-context alignment with layout is0.909.

CKA is an association measure, not a fraction of variance explained. Mask plans
are deterministic image-addressed records, not a new randomized trial. Strong
mask association means high masked-feature rank must not be read as broad
semantic texture understanding. The actual projector uses full-context features,
so this does not mean its collapse was caused by mask noise.
[Exploratory mask protocol](texture_aggregation_20261102/MASK_ADDENDUM_PROTOCOL.json) ·
[Mask alignment results](texture_aggregation_20261102/MASK_PLAN_ALIGNMENT.csv).

The second retained view supports the broad local ordering, but photometric
factors and hidden plans both differ. It does not isolate a causal appearance
or masking effect, and stability is partly built into local contrast preprocessing.

## 7. Implications before choosing another objective

| Evidence | Design implication to investigate | Boundary |
|---|---|---|
| Local benefit bypasses the global vector | Preserve a spatial evidence path rather than requiring every task to use one average | Not proof that any new aggregator improves classification |
| Mean-plus-position restores part, but not all, of the margin | Evaluate **incremental local spatial residual utility** beyond a strong global/position control | Control/support must be frozen before any new fit |
| Contrast/orientation are lost disproportionately | Give local structure a separately measured target and per-channel diagnostics | Image-derived texture, not physical depth truth |
| Canonical moments beat fixed random modes for the local head | Spatial organization is useful to this task | Retain the random and band/column controls; no universal best summary |
| Rank can rise through invertible scaling or mask placement | Pair diversity metrics with saved-target utility, scale conventions and mask controls | No post-hoc gate relaxation |
| Group dominance and only one low training component persist | Keep role/component support explicit; avoid identity-adversarial claims | No new independent generalization evidence |

The most supported next research question is whether a local residual texture
objective can preserve contrast/orientation detail beyond a global/position
baseline. This investigation does **not** yet select an architecture or deliver
another training notebook. Any masked-task conditioning must use visible inputs,
never hidden target values. Geometry remains a fixed full-image input; the
original mask-leakage guarantee is conditional on that cache.

## 8. Execution, retention and what stayed unchanged

The investigation verified249 pinned native/metadata inputs,66,722,671 bytes,
and254 source CSV bodies. It uses570 final images, two saved views, three feature
layers and nine fixed interventions. It performed656,640 frozen-decoder token
readouts and1,140 projector-vector readouts in CPU algebra; **zero image-encoder
or geometry forwards, zero optimizer updates, zero new photos**.

All counterfactual predictions are direct FP64 NPY files. Image/channel/tile
losses, template/scaling parameters, spectra, covariances, source hashes and
summary tables are direct CSV. The primary derivatives total about769MB; the
final inventory includes supplementary summaries and visualizations too.
No ZIP/NPZ or database envelope was created. Large native/detail tables stay
out of Git; compact summaries, code and curated figures remain visible.

New derivatives are **local and not uploaded to HF**. Original raw evidence
remains at pinned fit revision510cbc827337cbc963562c798811226326dc7c0c. All original
notebooks, model sources, selections, earlier reports and both workstations are
preserved. GEO/TRACE selected=None remains unchanged. This is post-hoc research
on previously inspected development cohorts, not independent physical-wear validation.
'''
# Keep numeric prose readable without editing any historical source document.
main=re.sub(r'\b(All|all|within|from|to|versus|of|for|rank|only|original|keeps|retains|losing|restores|preserves|gives|is|has|in|with|not|full|by|below|above|performed|and|verified|uses|total|after|before|its|epoch|update|view|than|on|both)(?=\d)',r'\1 ',main)
main=re.sub(r'(?<=\d)(images|D|MB|GB|NPY|PT|CSV|tile|control|position|spatial|source|train|forward)\b',r' \1',main)
write(ROOT/'phase3TextureAggregationInvestigation.md',main)
methods='''# Methods and data dictionary — TRACE aggregation assay

This is a saved-feature investigation, not a new trained candidate. See
[main findings](../phase3TextureAggregationInvestigation.md) and the immutable
[primary protocol](PROTOCOL.json). The mask-layout follow-up is explicitly
exploratory in [its separate protocol](MASK_ADDENDUM_PROTOCOL.json).

## Frozen data and computation

- Original TRACE final epoch60 /update2940;570 aligned images,386train /81validation /103historical test. View0 primary, view1 supplementary; no new augmentations or photos.
- Local `encoded`, unmasked `full_context` and `masked_context` arrays each have shape `[570,2,32,48]`; recorded masses/hidden plans select support. Targets have `[570,32,6,4,4]`, valid-pixel counts `[570,32,1,4,4]`.
- Descriptor targets/statistics and source SHA/role/component metadata come from the original fixed cache. Geometry saw the full original image before masking; no new geometry/image encoder forward occurs.
- Matrix calculations use CPU FP64. Original targets and position-control normalization use the original FP32 statistics/arithmetic before promotion. Identity decoder replay is checked against every original prediction and per-image loss.
- Decoder is `Linear48→96, SiLU, Linear96→96`, reshaped into6fields ×4×4cells. Its weights are unchanged. The projector replay is `240→48→6` with the original SiLU and weights.

## Aggregation algebra

Let `Z` contain32 position vectors and `M=diag(m)` the recorded coverage mass.
For a fixed position basis `B`, weighted projection is
`Z_hat = B (Bᵀ M B)^+ Bᵀ M Z`, with pseudoinverse rcond1e-10.

| Variant | Exact definition |
|---|---|
| identity |Original stored tokens; no projection |
| global_mean |Constant32×1basis; broadcast the weighted mean |
| band_means |Eight disjoint row-band indicators |
| column_means |Four column indicators |
| moments5 |`1,y,x,y*x,y*y−1/3` on the original8×4grid |
| random5 |Constant mode plus4fixed zero-sum QR random position modes; seed462225 |
| train_template |Train-view0 supported-image mean at each position/channel; no per-image variation |
| residual_mean |Train template plus constant-mode projection of `Z−template` |
| residual_moments5 |Train template plus five-moment projection of `Z−template` |

The five-moment projection reconstructs an approximation to tokens. The actual
model only stores the raw moment coefficients for its auxiliary projector; this
reconstruction is a counterfactual analysis, not an existing decoder branch.

## Target and utility reductions

Each descriptor cell uses the original valid-pixel count and SmoothL1 beta0.1.
Local decoder support is supported tiles; masked support is original hidden
tiles. Every variant shares the intersection with train-position-control
availability. Missing cells have zero weight and missing loss, not a true zero.
No variant silently changes its comparison denominator.

Per-image loss is `sum(weight*Huber)/sum(weighted channel counts)`. Role and
component scores are equal-image means. Retained advantage is
`(position_mean_loss−variant_mean_loss)/(position_mean_loss−original_mean_loss)`.
It can be negative or exceed1; values are not clipped. Per-channel retention is
reported only if its original control advantage exceeds1e-8. Small margins can
produce large negative percentages, so the main report gives absolute losses.

Tile summaries retain raw Huber sum, weighted count, available-image count and
sum of per-image tile losses. Interactive role heatmaps combine component loss
sums/counts, not unweighted averages of component means. Six channel means
reconcile exactly to each matching image-loss summary.

## Diversity, spatial covariance and mask controls

Features are cohort-centered for covariance. Entropy rank uses positive
eigenvalues above1e-14of total variance to exclude numerical nullspace noise.
Per-component centering is descriptive and never applied to inference outputs.
All original dimensions/sample counts are retained; no significance test treats
correlated views or images as independent tyres.

For fixed train-mean mass `r`, spatial covariance is
`C[t,u]=sum_i <Zc[i,t],Zc[i,u]>/(n−1)`. Weighted tile variance is
`sum_t r[t]*C[t,t]`; constant mean-mode variance is `rᵀ C r`.
Their difference equals weighted spatial-residual variance. This exact identity
is checked. Primary held-out roles have32supported positions; train includes two
images with missing support, recorded separately. The fixed-grid training
covariance can include stored placeholder-token effects and is not primary texture evidence.

Train-only per-coordinate standardization uses a1e-3floor. The inverse is checked
to1e-12, demonstrating that covariance rank can change without new information.
These rescaled vectors never replace the original checkpoint/gate.

The exploratory mask follow-up computes linear centered-kernel alignment:
`CKA = <K_feature,K_mask>_F/(||K_feature||_F*||K_mask||_F)` after role or component
centering. It is association, not explained-variance percentage or a new mask
randomization. High masked-context rank can contain mask-placement information.

## Direct files and grains

| Family / file | Grain and meaning |
|---|---|
| IMAGE_ALIGNMENT |570 rows:source image SHA,role,component,slot and observation |
| NATIVE_INPUTS /CSV_INPUTS |Full checksums of249 selected native/metadata and254 consumed CSV bodies |
| BRIDGE_VERIFICATION |Six exact-source bridge/replay checks with maximum numerical differences |
| TRAIN_TEMPLATES /TRAIN_SCALES |Train-only descriptive parameter, counts and fit role/view |
| FEATURE_SPACES |Layer/representation/role/view/centering; rank,std,variance,dimension and group fraction |
| SPATIAL_COVARIANCE |Layer/role/view/centering; exact mean-mode and spatial variance partition |
| SPATIAL_ENERGY /summary |Per image/view/layer/basis template-centered projection energy; macro and ratio-of-sums summaries remain named separately |
| IMAGE_LOSSES_* |Image/view/branch/variant; raw sums/counts,loss,control margin,coverage and feature error |
| CHANNEL_LOSSES_* |Image/view/branch/variant/channel; raw sum,count and loss |
| detail/TILE_LOSSES_* |Image/view/branch/variant/tile; raw numerator,count,support,hidden state and optional loss |
| UTILITY_SUMMARY /COMPONENT_UTILITY |Equal-image role/component utility for all9variants and both views |
| CHANNEL_UTILITY_SUMMARY /TILE_UTILITY_SUMMARY |Derived descriptor and per-component canonical-tile reductions |
| detail/SPECTRA /SPATIAL_COVARIANCE_CELLS |All eigenvalues and32×32covariance/correlation matrices |
| MASK_PLAN_* /SUPPORT_SUMMARY |Exploratory mask association, plan diversity and actual support counts |
| native/prediction_* |All18counterfactual prediction arrays, FP64 `[570,2,32,6,4,4]` |
| native/features_* |All compared feature vectors; original coordinates retained |
| *_RECEIPT /ANALYSIS |Protocol/source/output hashes, completed stage reuse and computation scope |

Each CSV is direct and readable. Large per-image/channel/tile/native derivatives
are local/on-disk and ignored by Git; compact summaries/methods/figures remain
versionable. No derived artifact is claimed uploaded to HF in this pass.

## Replay and limits

Use `conda activate cv_conda`, or the existing `phase3/run_cv.cmd` wrapper. These
analysis scripts require the previously verified NB24 local audit mirror and
pinned manifests; they are not new standalone Kaggle training notebooks.
Order:prepare_texture_aggregation →test_texture_aggregation_math →
analyze_texture_aggregation →texture_mask_confounds →summarize_texture_aggregation
→document_texture_aggregation →build_aggregation_explorer →validation.

Completed analysis stages are size/SHA/source-verified before reuse. Source
images, original model code, checkpoints, protocols and both workstations are
not changed. There are zero image/geometry forwards and optimizer updates;
656,640decoder-token and1,140projector-vector readouts are explicitly counted.
Fixed-head interventions are not newly trained heads; lower counterfactual
utility can reflect both information removal and off-manifold readout behavior.
The existing cohorts are already inspected, and proxy labels are not physical truth.
'''
write(D/'METHODS.md',methods)
write(D/'README.md','''# TRACE saved aggregation investigation

**Completed locally,2 November2026.** [Findings](../phase3TextureAggregationInvestigation.md) ·
[Interactive explorer](Aggregation_Explorer.html) · [Methods and dictionary](METHODS.md).

This package uses original final TRACE features and unchanged decoder weights.
It contains no newly trained model, photo or workstation revision. Original
selected=None remains preserved. New derivatives have not been uploaded to HF.

Start with UTILITY_SUMMARY.csv, FEATURE_SPACES.csv, CHANNEL_UTILITY_SUMMARY.csv,
SPATIAL_COVARIANCE.csv and COMPONENT_UTILITY.csv. Protocol/input/replay receipts
identify exact sources and numerical scope. Full direct CSV/NPY evidence is
retained locally; larger native/detail tables are excluded from Git.
''')
findings=dict(status='completed_saved_analysis',protocol=a['protocol'],local_validation_mean_retained=get('encoded','validation','global_mean').retained_gain_fraction,
    local_validation_template_retained=get('encoded','validation','train_template').retained_gain_fraction,
    local_validation_residual_moments_retained=get('encoded','validation','residual_moments5').retained_gain_fraction,
    local_test_residual_moments_retained=get('encoded','test','residual_moments5').retained_gain_fraction,
    validation_local_concat_rank=rank('encoded','concat1536'),validation_local_mean_rank=rank('encoded','mean48'),
    validation_context_raw_moment_rank=rank('full_context','moments240'),validation_context_scaled_moment_rank=rank('full_context','moments240_train_scaled'),
    original_selected=None,new_model_objective_selected=False,new_neural_training=False,image_encoder_forwards=0,new_optimizer_updates=0,remote_writes=0)
write(D/'FINDINGS.json',json.dumps(findings,indent=2))
print('Two scientific figures, investigation, methods and exact findings written.')
