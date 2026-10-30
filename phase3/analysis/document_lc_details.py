"""Document exact schemas and visualize fixed features/sampling, no fitting."""
import csv,json,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'tread_two/lc_pilot'
sys.path.insert(0,str(SOURCE))
import numpy as np
import torch
from schemas import SCHEMAS,UNITS
from common import read
from data_atlas import dataset,locate
from lc_features import contrast
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'tread_two/lc_validation/figures';OUT.mkdir(exist_ok=True)
rgb=torch.from_numpy(np.load(SOURCE/'assets/bank_rgb.npy',allow_pickle=False));valid=torch.from_numpy(np.load(SOURCE/'assets/bank_valid.npy',allow_pickle=False))
torch.set_num_threads(4)
features=contrast(rgb,valid,read(SOURCE/'LC_RULES.json')).numpy()
fig,axes=plt.subplots(3,5,figsize=(12,7),layout='constrained')
for row,(index,label) in enumerate(zip((0,6,12),('Low proxy','Mid proxy','High proxy'))):
    for col,title in enumerate(('Actual RGB reference','Normalized luminance','Sobel x','Sobel y','Gradient magnitude')):
        ax=axes[row,col]
        if col==0:ax.imshow(rgb[index].numpy().transpose(1,2,0).clip(0,1))
        elif col==4:ax.imshow(features[index,3],cmap='magma',vmin=0,vmax=.4)
        else:ax.imshow(features[index,col-1],cmap='coolwarm',vmin=-1 if col==1 else -.4,vmax=1 if col==1 else .4)
        ax.set_xticks([]);ax.set_yticks([])
        if row==0:ax.set_title(title,fontsize=10)
        if col==0:ax.set_ylabel(label)
fig.suptitle('ATLAS-LC r2 · fixed NB10 reference channels\nIllustrations from training photos; no classifier result or physical depth measurement',fontsize=12)
fig.savefig(OUT/'reference_channels.png',dpi=150);plt.close(fig)
plan,queries=dataset(locate());oldbank=list(csv.DictReader((ROOT/'tread_two/pilot/assets/prototype_bank.csv').open()))
excluded={b['source_image_id'] for b in oldbank};groups={}
old=[r for r in queries if r['image_id'] not in excluded]
for r in old:groups.setdefault(r['class_index'],{}).setdefault(r['domain'],{}).setdefault(r['physical_tyre_id'] or r['source_video'],[]).append(r)
new=list(csv.DictReader((SOURCE/'assets/sampling_plan.csv').open()))
comparisons=[]
for row in new:
    r=next(r for r in queries if r['image_id']==row['image_id']);g=r['physical_tyre_id'] or r['source_video']
    old_expected=0 if r['image_id'] in excluded else 23520/3/len(groups[r['class_index']])/len(groups[r['class_index']][r['domain']])/len(groups[r['class_index']][r['domain']][g])
    comparisons.append(dict(image_id=r['image_id'],class_index=r['class_index'],source_group=g,v1_expected=old_expected,r2_expected=float(row['expected_draws_60_epochs'])))
with (OUT.parent/'sampling_comparison.csv').open('w',newline='',encoding='utf-8') as stream:
    w=csv.DictWriter(stream,fieldnames=list(comparisons[0]));w.writeheader();w.writerows(comparisons)
fig,axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained',sharey=True)
for cls,ax in enumerate(axes):
    selected=[r for r in comparisons if r['class_index']==cls]
    ax.plot(sorted([r['v1_expected'] for r in selected],reverse=True),label='v1: 378 queries / uniform group',color='#a46a59')
    ax.plot(sorted([r['r2_expected'] for r in selected],reverse=True),label='r2: 386 queries / sqrt group',color='#167f88')
    ax.set(title=('Low','Mid','High')[cls]+' proxy',xlabel='Image rank within class');ax.grid(alpha=.2)
axes[0].set_ylabel('Expected draws across 60 epochs');axes[1].legend(fontsize=7)
fig.suptitle('Declared exposure expectations · replacement remains · no new independent tyres')
fig.savefig(OUT/'sampling_expectations.png',dpi=160);plt.close(fig)
descriptions={
 'source_images':('570','image_index','All historical roles,full SHA,label,identity,bank/query flags; eligibility is not observed exposure'),
 'bank_cases':('18','prototype_index','Literal training-photo provenance and atlas patch'),
 'bank_pixels':('221,184','prototype_index/channel/row/column','Raw RGB float pixels; valid masks and LC can be reconstructed from native assets'),
 'predictions':('53,342','sample','Both online views and every evaluation; source SHA and allowed-class counts'),
 'sample_exposures':('47,040','sample','Actual sampled draw probability and view factors; A/B share one occurrence'),
 'affinities':('30,724,992','sample/patch/prototype','All32×18 cosine and temperature-scaled affinities; includes blocked cases for audit'),
 'patch_support':('1,706,944','sample/patch','Coverage and valid-band pooling mass'),
 'patch_evidence':('5,120,832','sample/patch/class_index','Signed contribution after SHA suppression/scaling'),
 'coefficients':('170,578','parameter_update/kind/class_index/prototype','Base signed weights,bias and temperature; update0 plus2,940 updates'),
 'reference_embeddings':('107,136','epoch/endpoint/prototype/dimension','96D embeddings at60validation and2endpoint starts'),
 'train_updates':('2,940','update','Joint/pair loss components,LR,gradient/resource and before/after learner hashes'),
 'parameter_gradients':('64,680','update/parameter','All22 trainable parameter tensors: preclip norm,mean,min,max,finite'),
 'dense_pixels':('≤14,008,320','image_index/channel/row/column','Frozen quarter-resolution probabilities,once per observed image'),
 'atlas_rows':('≤291,840','image_index/atlas_row','Predicted source coordinates/validity,once per observed image'),
 'dense_regions':('≤1,140','image_index/region','Unignored counts,Dice,IoU and half-open mask-box IoU'),
 'dense_points':('≤3,420','image_index/point','Guide-relative human/weak target comparability and width-fraction error'),
 'metrics':('group dependent','stage/endpoint/epoch/level/group/class_index/metric','Cohort/domain/tyre/group metrics with denominators and explicit NA'),
 'confusion':('group dependent','stage/endpoint/epoch/level/group/target_class/predicted_class','All3×3 counts including zeros'),
 'calibration_bins':('group dependent','stage/endpoint/epoch/level/group/bin','10equal-width bins with sufficient statistics'),
 'selection':('60','epoch','Validation rule result,selected epoch,checkpoint/hash'),
 'explanation_checks':('618','sample/class_index','Selected/final native image projection conservation'),
 'case_rules':('960,156','sample/prototype','SHA match,allowed flag,reference class and6/n scale'),
 'view_pairs':('23,520','sample_a/sample_b','Per-draw CE_A,CE_B,JS,texture,total,target/prediction disagreement'),
 'view_patch_consistency':('752,640','sample_a/sample_b/patch','32weighted query embedding distances per draw'),
}
lines=['# ATLAS-LC r2 — direct CSV and native evidence dictionary','',
 '**Frozen schema:24 families.** Counts below are planned for the60epoch measured run, not completed outcomes. Source roles remain386/81/103. Engineering fixtures use four updates and two validation examples. More rows/views do not create additional independent tyres.','',
 '| Family | Planned rows | Join / observation key | Meaning |','|---|---:|---|---|']
for family in SCHEMAS:
    n,key,meaning=descriptions[family];lines.append(f'| `{family}` | {n} | `{key}` | {meaning} |')
lines+=['','## Joins, denominators and update ownership','',
 '`run` joins the full CONTRACT hash. `sample` is a unique monotonic observation key: A/B are separate samples, paired in `view_pairs`; evaluation uses one sample per image/variant. `image_index` joins the fixed570-image registry. Every observation retains source SHA and role is looked up from that registry. Bank prototypes are ordered six per class. Training view rows use `parameter_update=update−1`; base coefficients at that state plus case_rules reconstruct signed evidence.','',
 'Training predictions precede optimization. The saved joint update loss averages draw-level paired totals. Do not average47,040 view rows and call them independent tyres. Online epoch metrics mix learner states; validation uses the completed epoch state. Missing mid recall in held-out sets remains NA. All raw probabilities support later calibration analysis; calibration is not fitted here.','',
 '## Every retained column','']
for family,fields in SCHEMAS.items():
    lines += [f'### `{family}`','',f"`{'`, `'.join(fields)}`",'']
lines += ['## Units and missing values','','| Field category | Contract |','|---|---|']
for key,meaning in UNITS.items():lines.append(f'| {key} | {meaning} |')
lines += ['| case_rules | allowed/same-source are0/1; scale6/n_allowed counts references of the reference class, not the query target |',
 '| view losses | CE/JS are nats; texture is band-weighted cosine distance; total=.5(CE_A+CE_B)+.1JS+.05texture |',
 '| appearance | brightness/contrast/gamma dimensionless; independent[.8,1.2]; source SHA refers to original file before augmentation |',
 '| gradient | Per-tensor summaries are before joint gradient clipping; per-objective gradients and dense continuous training traces are not measured |',
 '| resource | CUDA/host/disk in bytes; wall times in seconds; environment dependent,excluded only from deterministic-equivalence comparison |','',
 '## Native evidence and reproducibility','',
 '| Artifact | Retained content |','|---|---|',
 '| state.pt and two rotating local states | Full model/fixed bank,optimizer,scheduler,disabled scaler,RNG,exact orders,next update/draw,partial metrics,CSV byte/row/native cursors |',
 '| artifacts/epochs/*.pt | All60epoch model states,epoch/update,protocol and learner hash |',
 '| artifacts/dense/*.pt | Deduplicated frozen dense masks,points/guides and source IDs; raw dense probabilities also CSV |',
 '| artifacts/endpoint_maps/*/*.pt | Every selected/final atlas RGB/valid/grid,signed patch evidence,effective coefficients,reference embeddings,aggregate reference evidence,input-space signed maps and bias |',
 '| source/assets/*.npy + prototype_bank.csv | Original reference RGB/valid/source32-byte SHA/labels,case/patch/native-source provenance and planned sampling probabilities |',
 '| CONTRACT/FILES/recipe/environment/history/HTTP/lifecycle JSON | Source hashes,provenance,atomic-unit integrity,recovery/publication attempts,capacity/errors |','',
 'Augmented full RGB/feature tensors are not duplicated at every draw. They are recoverable from the original data SHA,frozen geometry/source and exact seed/factors; complete affinities and signed evidence are stored for both views. No invented physical labels,independent tyres,unsaved per-objective gradients or continuous energy/GPU utilization traces are claimed.','',
 'Native tensor weights/maps are directPT/NPY; scientific tables are directCSV. No ZIP,NPZ or database upload. See [architecture](../../phase3TreadTwoLCArchitecture.md) and [protocol](../../phase3TreadTwoLCProtocol.md).']
(SOURCE/'SCHEMA.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Documented',len(SCHEMAS),'families;',sum(map(len,SCHEMAS.values())),'columns; wrote two scientific illustrations')
