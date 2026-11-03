"""Build publication figures and tables from the byte-verified paper snapshot.

No fitting, model loading, image inference, network request or source mutation.
Run from the repository root with phase3/run_cv.cmd -B <this file>.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent
E = ROOT / 'evidence'
F = ROOT / 'figures'
G = ROOT / 'generated'
for path in (F, G, ROOT / 'audit'):
    path.mkdir(exist_ok=True)
for rec in json.loads((ROOT / 'audit/EVIDENCE_FILES.json').read_text()):
    p = E / rec['file']
    assert hashlib.sha256(p.read_bytes()).hexdigest() == rec['sha256'], p
for rec in json.loads((ROOT/'audit/ADDITIONAL_INPUTS.json').read_text())['outputs']:
    p=ROOT/rec['file']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==rec['sha256'],p

plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':8,
    'axes.titlesize':9, 'axes.labelsize':8, 'xtick.labelsize':7.5,
    'ytick.labelsize':7.5, 'legend.fontsize':7, 'pdf.fonttype':42,
    'ps.fonttype':42, 'axes.spines.top':False, 'axes.spines.right':False,
    'savefig.facecolor':'white'})
BLUE, ORANGE, TEAL, RED, GRAY = '#0072B2', '#E69F00', '#009E73', '#CC6677', '#68737d'
audit = {'input_verification':'all copied evidence SHA-256 values match',
         'new_training':False, 'figure_data':{}, 'metrics':{}}

def save(fig, name, inputs):
    fig.savefig(F / (name+'.pdf'), bbox_inches='tight', pad_inches=.06)
    fig.savefig(F / (name+'.png'), bbox_inches='tight', pad_inches=.06, dpi=240)
    plt.close(fig)
    audit['figure_data'][name] = inputs

def row(df, **kw):
    keep = np.ones(len(df), dtype=bool)
    for key, val in kw.items():
        keep &= (df[key] == val).to_numpy()
    out = df.loc[keep]
    assert len(out) == 1, (kw, len(out))
    return out.iloc[0]

# Actual data-flow schematic; dimensions are explicit and decoder bypasses pooling.
fig, ax = plt.subplots(figsize=(7.15,3.25))
ax.set(xlim=(0,10), ylim=(0,5)); ax.axis('off')
def box(x,y,w,h,text,color=BLUE):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=0.08',
        facecolor=color+'12',edgecolor=color,lw=1))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=7.5)
def arrow(a,b,color=GRAY):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=9,color=color,lw=.9))
box(.05,3.48,1.65,1.05,'Existing image\nFrozen geometry\n512 × 256 atlas',GRAY)
box(2.08,3.48,1.6,1.05,'32 tiles, 64 × 64\nLocal contrast\n4 input channels')
box(4.05,3.48,1.6,1.05,'Shared encoder\nZ: 32 × 48\nNo tile mixing')
box(7.55,3.48,2.28,1.05,'Local decoder D(Z)\n6 fields × 4 × 4 / tile\nDirect local supervision',TEAL)
arrow((1.7,4),(2.08,4)); arrow((3.68,4),(4.05,4)); arrow((5.65,4),(7.55,4),TEAL)
box(4.05,1.85,1.6,1.05,'Add metadata\n2 axial blocks\nH: 32 × 48')
arrow((4.85,3.48),(4.85,2.9))
box(6.05,1.85,1.4,1.05,'5 moments\n240D summary')
box(7.8,1.85,2.03,1.05,'6D projector\nPair / variance /\ncovariance losses',ORANGE)
arrow((5.65,2.37),(6.05,2.37)); arrow((7.45,2.37),(7.8,2.37))
box(.05,.25,3.62,1.05,'Separate masked path\nHide raw tiles before preprocessing\nVisible tiles + fixed metadata')
box(4.05,.25,1.6,1.05,'Masked context\nC: 32 × 48')
box(7.55,.25,2.28,1.05,'Same decoder D(C)\nScore hidden cells only',TEAL)
arrow((3.67,.78),(4.05,.78)); arrow((5.65,.78),(7.55,.78),TEAL)
ax.text(6.65,4.7,'71,022 trainable texture parameters',ha='center',fontsize=8,fontweight='bold')
ax.text(6.6,1.48,'Global summaries do not feed either reconstruction head.',ha='center',fontsize=7,color=GRAY)
save(fig,'architecture',['TRACE frozen source / architecture and objective specification'])

cls = pd.read_csv(E/'classification_endpoints.csv')
selected = cls[(cls.endpoint=='selected') & ~cls.model.str.startswith('Phase2') & (cls.model!='TREAD-ONE paired baseline')].copy()
labels = ['ONE s1','ONE s2','ONE s3','Projection','ATLAS','ATLAS-LC','Context']
assert len(selected)==len(labels)
fig, aa=plt.subplots(1,2,figsize=(7.15,2.55),layout='constrained')
y=np.arange(len(labels))
aa[0].barh(y-.16,selected.validation_BA*100,.29,color=BLUE,label='Validation')
aa[0].barh(y+.16,selected.test_observed_BA*100,.29,color=ORANGE,label='Historical test')
aa[0].set(yticks=y,yticklabels=labels,xlim=(0,105),xlabel='Observed-class balanced accuracy (%)',title='(a) Validation-selected endpoints')
aa[0].invert_yaxis(); aa[0].legend(loc='lower right',frameon=False)
aa[1].barh(y,selected.nll,color=TEAL,height=.58)
aa[1].set(yticks=y,yticklabels=labels,xlim=(0,3),xlabel='Historical-test negative log likelihood',title='(b) Probability errors remain material')
aa[1].invert_yaxis()
for ax in aa: ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
save(fig,'classification',['classification_endpoints.csv; selected endpoints only; same historical roles'])

u = pd.read_csv(E/'aggregation_utility_summary.csv')
variants=['identity','global_mean','band_means','column_means','moments5','random5','train_template','residual_mean','residual_moments5']
names=['Original tokens','One global mean','Eight band means','Four column means','Five spatial moments','Five random modes','Train-position template','Template + mean residual','Template + five residual moments']
table=[]
for variant in variants:
    vals=[100*row(u,layer=layer,role=role,view=0,variant=variant).retained_gain_fraction
          for layer,role in [('encoded','validation'),('encoded','test'),('masked_context','validation'),('masked_context','test')]]
    table.append([variant,*vals])
    audit['metrics'][variant+'_retained_percent']=vals
pd.DataFrame(table,columns=['variant','local_val','local_test','masked_val','masked_test']).to_csv(G/'table_aggregation.csv',index=False)
lines=['% Generated by build_assets.py; values are retained advantage, not accuracy.',r'\newcommand{\aggregationrows}{%']
for name, rec in zip(names,table):
    lines.append(name+' & '+' & '.join(f'{v:.2f}' for v in rec[1:])+r' \\')
lines.append('}')
(G/'aggregation_rows.tex').write_text('\n'.join(lines)+'\n')
fig, aa = plt.subplots(1,2,figsize=(7.15,3.9),layout='constrained',sharey=True)
for ax, indices, title in zip(aa,[(1,2),(3,4)],['(a) Direct local tokens Z','(b) Masked-context tokens C']):
    y=np.arange(9)
    ax.barh(y-.16,[x[indices[0]] for x in table],.3,color=BLUE,label='Validation (81)')
    ax.barh(y+.16,[x[indices[1]] for x in table],.3,color=ORANGE,label='Historical test (103)')
    ax.axvline(0,color=GRAY,lw=.65);ax.axvline(100,color=GRAY,lw=.65,ls=':')
    ax.set(yticks=y,yticklabels=names,xlim=(-38,105),xlabel='Retained advantage over position control (%)',title=title)
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
aa[0].invert_yaxis()
fig.legend(*aa[1].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
save(fig,'aggregation',['aggregation_utility_summary.csv; view 0; equal-image mean losses'])

ch = pd.read_csv(E/'aggregation_channel_utility_summary.csv')
comp = pd.read_csv(E/'aggregation_component_utility.csv')
fig, aa=plt.subplots(1,2,figsize=(7.15,3.0),layout='constrained')
channelnames=['Contrast','Gradient energy','Orientation axis','Orientation cross','Relative curvature','Contrast occupancy']
for variant,col,marker,label in [('identity',TEAL,'o','Original'),('global_mean',ORANGE,'s','Global mean'),('residual_moments5',BLUE,'^','Template + 5 residuals')]:
    part=ch[(ch.layer=='encoded') & (ch.role=='validation') & (ch.view==0) & (ch.variant==variant)].sort_values('channel')
    assert len(part)==6
    aa[0].plot(part.loss,np.arange(6),marker=marker,color=col,lw=0,label=label,ms=4)
    if variant=='identity':aa[0].plot(part.position_control_loss,np.arange(6),'x',color=GRAY,label='Position control',ms=5)
aa[0].set(yticks=np.arange(6),yticklabels=channelnames,xlabel='Standardized Smooth L1 loss',title='(a) Validation local descriptor errors',xlim=(.22,.86))
aa[0].invert_yaxis();aa[0].legend(frameon=False,fontsize=6.5,loc='upper center',bbox_to_anchor=(.5,-.23),ncol=2)
groups=[('validation','000100_005000','Val. low (40)'),('validation','050000_','Val. high (41)'),('test','000100_005000','Test low (62)'),('test','050000_','Test high (41)')]
groupvals=[]
for role,pattern,label in groups:
    c=comp[(comp.layer=='encoded') & (comp.role==role) & (comp.view==0)]
    if 'high' in label:
        c=c[~c.component.str.contains('000100_005000')]
    else:c=c[c.component.str.contains(pattern)]
    vals=[float(c[c.variant==v].retained_gain_fraction.iloc[0])*100 for v in ['global_mean','moments5','residual_moments5']]
    groupvals.append(vals)
for j,(label,col) in enumerate([('Global mean',ORANGE),('5 moments',TEAL),('Template + 5 residuals',BLUE)]):
    aa[1].barh(np.arange(4)+(j-1)*.22,np.array(groupvals)[:,j],height=.2,label=label,color=col)
aa[1].set(yticks=np.arange(4),yticklabels=[x[2] for x in groups],xlabel='Retained local advantage (%)',title='(b) Component-dependent retention',xlim=(0,105))
aa[1].invert_yaxis();aa[1].legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.23),fontsize=6.5,ncol=1)
for ax in aa:ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
save(fig,'descriptor_groups',['aggregation_channel_utility_summary.csv','aggregation_component_utility.csv; each held-out component is one class'])

sp = pd.read_csv(E/'aggregation_feature_spaces.csv')
cov = pd.read_csv(E/'aggregation_spatial_covariance.csv')
mask = pd.read_csv(E/'aggregation_mask_plan_alignment.csv')
fig, aa=plt.subplots(2,2,figsize=(7.15,5.0),layout='constrained')
spaces=[('encoded','concat1536','Local Z, concatenated'),('full_context','concat1536','Context H, concatenated'),('encoded','mean48','Local mean (48D)'),('full_context','moments240','Context moments (240D)'),('full_context','actual_projected6','Projector (6D)')]
for cc,offset,col,label in [(False,-.17,BLUE,'Role centered'),(True,.17,TEAL,'Within component')]:
    values=[row(sp,layer=l,space=s,role='validation',view=0,component_centered=cc).effective_rank for l,s,_ in spaces]
    aa[0,0].barh(np.arange(5)+offset,values,.3,color=col,label=label)
aa[0,0].set(yticks=np.arange(5),yticklabels=[x[2] for x in spaces],xlabel='Covariance-energy entropy rank',title='(a) Local variation and global dominance')
aa[0,0].invert_yaxis();aa[0,0].legend(frameon=False,fontsize=6.5,loc='lower right')
layers=['encoded','full_context','masked_context']
for cc,offset,col,label in [(False,-.17,BLUE,'Role centered'),(True,.17,TEAL,'Within component')]:
    values=[100*row(cov,layer=l,role='validation',view=0,component_centered=cc).mean_mode_energy_fraction for l in layers]
    aa[0,1].barh(np.arange(3)+offset,values,.3,color=col,label=label)
aa[0,1].set(yticks=np.arange(3),yticklabels=['Local Z','Full context H','Masked context C'],xlabel='Variance in the constant spatial mode (%)',title='(b) Fixed train-mass covariance partition',xlim=(0,47))
aa[0,1].invert_yaxis();aa[0,1].legend(frameon=False,fontsize=6.5,loc='lower right')
v=[row(sp,layer='full_context',space=s,role='validation',view=0,component_centered=False).effective_rank for s in ['moments240','moments240_train_scaled']]
aa[1,0].bar(['Raw 240D','Train-scaled 240D'],v,color=[BLUE,ORANGE],width=.55)
aa[1,0].set(ylabel='Covariance-energy entropy rank',ylim=(0,3.5),title='(c) Invertible scaling adds no information')
for i,n in enumerate(v):aa[1,0].text(i,n+.07,f'{n:.3f}',ha='center',fontsize=8)
for cc,offset,col,label in [(False,-.17,BLUE,'Role centered'),(True,.17,TEAL,'Within component')]:
    vals=[row(mask,role='validation',view=0,component_centered=cc,feature=s).linear_CKA_with_mask_plan for s in ['full_context_concat','masked_context_concat','masked_minus_full_context','actual_projected6']]
    aa[1,1].barh(np.arange(4)+offset,vals,.3,color=col,label=label)
aa[1,1].set(yticks=np.arange(4),yticklabels=['Full H','Masked C','C − H','Projector'],xlabel='Linear CKA with recorded mask layout',xlim=(0,1.04),title='(d) Exploratory mask-plan association')
aa[1,1].invert_yaxis();aa[1,1].legend(frameon=False,fontsize=6.5,loc='lower right')
for ax in aa.flat:ax.grid(axis='x' if ax is not aa[1,0] else 'y',alpha=.15);ax.set_axisbelow(True)
save(fig,'representation',['aggregation_feature_spaces.csv','aggregation_spatial_covariance.csv','aggregation_mask_plan_alignment.csv; view 0, validation'])

# Curves are copied from the original measured epoch records, not reconstructed fits.
epochs=pd.read_csv(E/'trace_validation_epochs.csv')
assert len(epochs)==60 and epochs.eligible.sum()==0
science=json.loads((E/'nb24_science.json').read_text())
val=science['final_validation']
fig,aa=plt.subplots(1,2,figsize=(7.15,3.15),layout='constrained')
aa[0].plot(epochs.epoch,epochs.local_huber,color=TEAL,label='Local model',lw=1.5)
aa[0].plot(epochs.epoch,epochs.masked_model_huber,color=BLUE,label='Masked model',lw=1.5)
aa[0].axhline(val['local_control'],color=TEAL,ls=':',lw=1,label='Local position control')
aa[0].axhline(val['control_visible_band_mean'],color=BLUE,ls=':',lw=1,label='Masked visible-band control')
aa[0].set(xlim=(1,60),ylim=(.38,.7),xlabel='Completed epoch',ylabel='Validation Smooth L1 loss',title='(a) Reconstruction improves')
aa[0].legend(frameon=False,fontsize=6.5,loc='upper center',bbox_to_anchor=(.5,-.23),ncol=2)
for field,label,col in [('rank_local_pooled','Local mean (48D)',TEAL),('rank_pooled','Spatial moments (240D)',BLUE),('rank_projected','Projector (6D)',ORANGE)]:
    aa[1].plot(epochs.epoch,epochs[field],label=label,color=col,lw=1.3)
aa[1].axhline(4,color=GRAY,ls='--',lw=1,label='Original rank floor')
aa[1].set(xlim=(1,60),ylim=(.9,4.25),xlabel='Completed epoch',ylabel='Covariance-energy entropy rank',title='(b) No epoch passes global diversity')
aa[1].legend(frameon=False,fontsize=6.5,loc='upper center',bbox_to_anchor=(.5,-.23),ncol=2)
for ax in aa:ax.grid(alpha=.15);ax.set_axisbelow(True)
save(fig,'training',['trace_validation_epochs.csv; all 60 original validation records','nb24_science.json; fixed controls'])

# Real saved atlas examples, selected by image identifier rather than performance.
from PIL import Image
cells=pd.read_csv(E/'qualitative/orientation_cells.csv')
fig,aa=plt.subplots(2,4,figsize=(7.15,4.7),layout='constrained')
cmap=plt.get_cmap('coolwarm').copy();cmap.set_bad('white')
for i,case in enumerate(['low','high']):
    aa[i,0].imshow(Image.open(E/f'qualitative/{case}_atlas.png'))
    part=cells[cells['case']==case]
    for j,field in enumerate(['target','original','mean'],start=1):
        array=np.full((32,16),np.nan)
        for r in part.itertuples():
            if r.valid_pixels>0:
                by,bx=divmod(r.tile,4)
                array[by*4+r.cell_y,bx*4+r.cell_x]=getattr(r,'orientation_axis_'+field)
        im=aa[i,j].imshow(array,cmap=cmap,vmin=-1,vmax=1,interpolation='nearest')
    aa[i,0].set_ylabel(f'{case.capitalize()} proxy example',fontsize=8)
for ax in aa.flat:
    ax.set_xticks([]);ax.set_yticks([])
    for spine in ax.spines.values():spine.set_visible(False)
for ax,title in zip(aa[0],['Saved canonical RGB','Fixed orientation target','Original-token decoder','Global-mean decoder']):ax.set_title(title,fontsize=8)
fig.colorbar(im,ax=aa[:,1:].ravel().tolist(),shrink=.72,pad=.015,label='Orientation axis (unstandardized; display range −1 to 1)')
save(fig,'qualitative',['qualitative/EXAMPLES.json; lexical-first validation example per observed proxy class','qualitative/orientation_cells.csv; unchanged original/mean frozen-decoder outputs'])

# Per-figure and per-input receipts make every new image regenerable.
audit['outputs']=[]
for p in sorted(F.glob('*')):
    if p.suffix in ('.pdf','.png'):
        audit['outputs'].append({'file':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(ROOT/'audit/ASSET_BUILD.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps({'status':'passed','figures':len(audit['figure_data']),'outputs':len(audit['outputs']),'training':False}))
