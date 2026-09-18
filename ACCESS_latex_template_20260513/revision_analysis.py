"""Post-review analyses of frozen records. No training or test-set tuning."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P = Path(__file__).resolve().parent
R = P.parent
E = P/'revision_evidence'
G = P/'generated'
used = {}
def read(path):
    used[path.relative_to(R).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path)
def table(name,caption,header,rows,columns=None):
    columns = columns or ('l'+'r'*(len(header)-1))
    text = '\\begin{table}[t]\n\\caption{'+caption+'}\n\\label{tab:'+name+'}\n\\centering\\small\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{'+columns+'}\n\\toprule\n'
    text += ' & '.join(header)+r'\\'+'\n\\midrule\n'
    text += '\n'.join(' & '.join(map(str,row))+r'\\' for row in rows)
    text += '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    (G/(name+'.tex')).write_text(text,encoding='utf-8')

for entry in json.loads((E/'manifest.json').read_text()):
    assert hashlib.sha256((E/entry['file']).read_bytes()).hexdigest()==entry['sha256']
features=read(E/'baseline_features.csv')
assert len(features)==418 and features.image_id.is_unique
assert features.groupby('sess').fold_id.nunique().eq(1).all()
split=[]
for f in range(3):
    train=features[features.fold_id!=f]; val=features[features.fold_id==f]
    overlap=set(train.sess)&set(val.sess);assert not overlap
    split.append(dict(fold=f,train_images=len(train),val_images=len(val),train_groups=train.sess.nunique(),val_groups=val.sess.nunique(),overlap_groups=sorted(overlap)))

epochs=[]
for file in sorted(E.glob('a-*-epochs.csv')):
    df=read(file).sort_values('epoch')
    assert df.epoch.tolist()==list(range(1,61)),file
    # Original trainer selects first strict maximum QWK, not maximum F1.
    chosen=df.loc[df.val_qwk.idxmax()]; final=df.iloc[-1]
    for endpoint,row in [('selected',chosen),('final',final)]:
        epochs.append(dict(arch=row.arch,fold=int(row.fold),seed=int(row.seed),endpoint=endpoint,epoch=int(row.epoch),f1=row.val_f1_macro,qwk=row.val_qwk,mae=row.val_mae_class))
endpoints=pd.DataFrame(epochs);endpoints.to_csv(E/'endpoint_reanalysis.csv',index=False)
archive=read(R/'docs/report/evidence/tables/classification_master_architectures.csv').set_index('arch')
for arch in endpoints.arch.unique():
    for endpoint,column in [('selected','f1_mean'),('final','final_f1')]:
        value=endpoints[(endpoints.arch==arch)&(endpoints.endpoint==endpoint)].f1.mean()
        assert abs(value-archive.loc[arch,column])<1e-10, (arch,endpoint)
means=endpoints.groupby(['arch','fold','endpoint'])[['f1','qwk','mae']].mean()
rows=[]
for a,label in [('mobilenetv4','MobileNetV4'),('resnet50','ResNet-50')]:
    for f in range(3):
        s=means.loc[(a,f,'selected')];z=means.loc[(a,f,'final')]
        rows.append([label if f==0 else '',f,f'{s.f1:.4f}',f'{z.f1:.4f}',f'{z.qwk:.4f}',f'{z.mae:.4f}'])
table('fold_endpoints','Fold-specific checkpoint analysis. Three seeds per cell; selected means the earliest maximum validation QWK. QWK and class-index MAE are reported at epoch 60. Configurations illustrate the former headline and the downstream frozen classifier, not a new model selection.', ['Model','Fold','Sel. F1','Final F1','Final QWK','MAE'],rows)

xai=read(E/'xai_faithfulness.csv');xai=xai[xai.arch!='convnextv2_s'].copy()
xai['faith']=xai.insertion_auc-xai.deletion_auc
screens=[];selected={}
for threshold in [.03,.05,.10]:
    valid=xai[(xai.sanity_delta>threshold)&xai.faith.notna()]
    best=valid.sort_values('faith',ascending=False,kind='stable').groupby('arch',sort=True).head(1)
    selected[str(threshold)]=dict(zip(best.arch,best.method))
    screens.append([f'{threshold:.2f}',len(valid),best.arch.nunique()])
table('gate_sensitivity','Post-review sensitivity of the archived initial explanation screen: 34 method rows from 17 retained configurations, one seed on fold 1. Thresholds were chosen for this diagnostic; subsequent training was not repeated.',['Threshold','Passing methods','Eligible models'],screens)

rawpath=R/'docs/report/evidence/geometry/comparison_PAIRED_POINTS.json'
used[rawpath.relative_to(R).as_posix()]=hashlib.sha256(rawpath.read_bytes()).hexdigest()
points=pd.read_json(rawpath);assert len(points)==432
im=points.groupby(['tyre','pilot_id'])[['hrnet_error','segformer_error']].mean()*100
im['gain']=im.segformer_error-im.hrnet_error
im.to_csv(E/'geometry_image_effects.csv')
rowstats=points.groupby('point')[['hrnet_error','segformer_error']].mean()*100
rowstats['gain']=rowstats.segformer_error-rowstats.hrnet_error
rows=[[k.replace('_',' '),f'{r.hrnet_error:.3f}',f'{r.segformer_error:.3f}',f'{r.gain:+.3f}'] for k,r in rowstats.iterrows()]
table('boundary_detail','Matched test error by point location (percent of native width minus one). Each row averages the same 24 images and three seeds. Positive difference favours point learning.',['Point','Point model','Mask-derived',r'$\Delta$'],rows)
fusion=read(R/'docs/report/evidence/tables/s9_fusion_by_run.csv')
fm=fusion.groupby(['arm','fold']).delta_vs_full.mean()
rows=[]
for a,label in [('tyre_only','Tyre'),('tread_only','Tread'),('tyre_tread_equal','Tyre + tread'),('full_tyre_tread_equal','Full + tyre + tread')]:
    if a not in fusion.arm.unique():continue
    rows.append([label]+[f'{fm.loc[(a,f)]:+.5f}' for f in range(3)])
print('Fusion arms:',fusion.arm.unique().tolist())
if rows:table('fusion_folds','Paired macro-F1 change by fold; each cell averages nine localisation configurations and three seeds with reused full-image classifiers.',['Input','Fold 0','Fold 1','Fold 2'],rows)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(6.8,2.65))
for idx,(tyre,group) in enumerate(im.groupby(level=0)):
    y=np.sort(group.gain.to_numpy());ax.scatter(np.arange(len(y))+idx*13,y,s=25,label='Test group '+str(idx+1))
ax.axhline(0,color='black',lw=.7);ax.set(xlabel='Images ordered within each test group (three-seed mean)',ylabel='Mask minus point error\n(percentage points of width)')
ax.legend(frameon=False);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(P/'figures/image_effects.pdf',bbox_inches='tight');plt.close(fig)

report=dict(split_audit=split,confirmed_groups=int(features.sess.nunique()),endpoint_means=means.reset_index().to_dict('records'),
            fusion_by_fold=fm.reset_index().to_dict('records'),
            gate_selection=selected,gate_rows=screens,boundary_by_location=rowstats.reset_index().to_dict('records'),
            image_wins=int((im.gain>0).sum()),images=len(im),min_image_gain=float(im.gain.min()),max_image_gain=float(im.gain.max()),
            mean_image_gain=float(im.gain.mean()),sources_sha256=used)
(P/'REVISION_ANALYSIS.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='sources_sha256'},indent=2))
