"""Copy bounded retained evidence; optional public reads of two old RGB cache chunks.

No model is constructed/executed. No credential is read and nothing is uploaded.
The example rule is lexical-first validation image in each observed proxy class.
"""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parent
P3=ROOT.parent
assert Path(sys.prefix).name=='cv_conda'
sys.path.insert(0,str(P3/'analysis'))
import nb24_public as pub
fit=pub.OUT/'remote/fit'
receipt={'fit_revision':pub.META['stages']['fit']['revision'],'new_model_forwards':0,'remote_writes':0,'local_credentials_read':False,'inputs':[]}
def checked(path):
    rel=path.relative_to(fit).as_posix()
    expected=pub.MANIFESTS['fit']['files'][rel]
    body=path.read_bytes()
    assert len(body)==expected['bytes'] and hashlib.sha256(body).hexdigest()==expected['sha256']
    receipt['inputs'].append({'source':rel,'bytes':len(body),'sha256':expected['sha256']})
    return path
def frames(family):
    paths=sorted((fit/'csv'/family).glob('*.csv'))
    assert paths
    return pd.concat([pd.read_csv(checked(p)) for p in paths],ignore_index=True)
metrics=frames('epoch_metrics')
gates=frames('eligibility')
spectra=frames('representation_spectra')
epoch=metrics[metrics.stage=='validate'].pivot(index='epoch',columns='metric',values='value').reset_index()
g=gates[gates.stage=='validate'].drop(columns=['run','stage','update','image_id','view'])
epoch=epoch.merge(g,on='epoch',validate='one_to_one')
s=spectra[(spectra.stage=='validate') & (spectra.component_centered==0) & (spectra.dimension==0)]
s=s.pivot(index='epoch',columns='space',values='effective_rank').add_prefix('rank_').reset_index()
epoch=epoch.merge(s,on='epoch',validate='one_to_one')
assert len(epoch)==60 and epoch.eligible.sum()==0
epoch.to_csv(ROOT/'evidence/trace_validation_epochs.csv',index=False)

alignment=pd.read_csv(ROOT/'evidence/aggregation_image_alignment.csv')
chosen=alignment[alignment.role=='validation'].sort_values('image_id').groupby('proxy_class',sort=True).head(1)
assert len(chosen)==2 and set(chosen.proxy_class)=={0,2}
dest=ROOT/'evidence/qualitative';dest.mkdir(exist_ok=True)
stats_path=checked(fit/'inputs/cache/controls/target_statistics.pt')
stats=torch.load(stats_path,map_location='cpu',weights_only=True)
mu=stats['mean'].numpy();sd=stats['std'].numpy()
native=P3/'texture_aggregation_20261102/native'
sealed=json.loads((P3/'texture_aggregation_20261102/ARTIFACTS.json').read_text())['files']
for variant in ['identity','global_mean']:
    p=native/f'prediction_encoded_{variant}.npy'
    name=p.relative_to(native.parent).as_posix()
    actual={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    assert actual==sealed[name]
    receipt.setdefault('saved_aggregation_inputs',[]).append({'source':p.relative_to(P3).as_posix(),**actual})
pred={v:np.load(native/f'prediction_encoded_{v}.npy',mmap_mode='r') for v in ['identity','global_mean']}
cells=[];examples=[]
for r in chosen.itertuples():
    block,offset=divmod(int(r.slot),8)
    rgbp=checked(pub.fetch('fit',f'inputs/cache/native/cache_rgb/{block:05d}.npy'))
    targetp=checked(fit/f'inputs/cache/native/cache_target/{block:05d}.npy')
    weightp=checked(fit/f'inputs/cache/native/cache_weight/{block:05d}.npy')
    rgb=np.load(rgbp,mmap_mode='r')[offset].transpose(1,2,0)
    assert rgb.shape==(512,256,3) and rgb.min()>=0 and rgb.max()<=1
    name='low' if r.proxy_class==0 else 'high'
    Image.fromarray(np.rint(rgb*255).astype('uint8')).save(dest/(name+'_atlas.png'))
    target=np.load(targetp,mmap_mode='r')[offset]
    weight=np.load(weightp,mmap_mode='r')[offset]
    i=int(alignment.index[alignment.image_id==r.image_id][0])
    originals={v:pred[v][i,0]*sd[None,:,None,None]+mu[None,:,None,None] for v in pred}
    for tile in range(32):
        for cy in range(4):
            for cx in range(4):
                cells.append({'case':name,'tile':tile,'cell_y':cy,'cell_x':cx,
                    'valid_pixels':weight[tile,0,cy,cx], 'orientation_axis_target':target[tile,2,cy,cx],
                    'orientation_axis_original':originals['identity'][tile,2,cy,cx],
                    'orientation_axis_mean':originals['global_mean'][tile,2,cy,cx]})
    examples.append({'case':name,**r._asdict(),'selection_rule':'lexicographically first validation image per observed proxy class; not error-ranked',
        'atlas':'8-bit display conversion of existing FP32 frozen canonical atlas, no new inference'})
pd.DataFrame(cells).to_csv(dest/'orientation_cells.csv',index=False)
(dest/'EXAMPLES.json').write_text(json.dumps(examples,indent=2)+'\n')
for p in [ROOT/'evidence/trace_validation_epochs.csv',*sorted(dest.glob('*'))]:
    receipt.setdefault('outputs',[]).append({'file':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
receipt['public_GETs_this_execution']=pub.public.CALLS
(ROOT/'audit/ADDITIONAL_INPUTS.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'status':'passed','epochs':len(epoch),'examples':len(examples),'cells':len(cells),'public_GETs':pub.public.CALLS}))
