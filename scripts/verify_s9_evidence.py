"""CPU checks for fusion math, row alignment and pinned public S5 compatibility."""
import ast
import base64
import io
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tyrelib'))
import s9_evidence as s

job=dict(run_id='fixture',model='fixture',backend='semantic',fold=0,seed=1)
rows=[]
for mode in ('full','pred_tyre','pred_tread'):
    for i in range(3):
        p=[.1,.1,.1];p[i]=.8
        rows.append(dict(mode=mode,image_id=str(i),truth=i,prediction=i,
                         **dict(zip(('prob_low','prob_mid','prob_high'),p))))
frame=pd.DataFrame(rows).sample(frac=1,random_state=8)
x=s.analyse(frame,job,['0','1','2'],{'0':0,'1':1,'2':2})
assert len(x['predictions'])==15 and all(m['macro_f1']==1 and m['delta_vs_full']==0 for m in x['metrics'])
assert s.decode(np.array([[.475,.11,.415]]),'coral').tolist()==[1]
assert s.decode(np.array([[.475,.11,.415]]),'softmax').tolist()==[0]
assert s.decode(np.array([[.5,0,.5]]),'coral').tolist()==[0]
bad=frame.copy();bad.loc[bad.index[0],'prob_low']=float('nan')
for data in (bad,pd.concat([frame,frame.iloc[:1]]),frame.iloc[:-1]):
    try:s.analyse(data,job,['0','1','2'],{'0':0,'1':1,'2':2})
    except AssertionError:pass
    else:raise AssertionError('Malformed predictions accepted')
nb=json.loads((ROOT/'notebooks/NB18_S9_Fusion_Analysis.ipynb').read_text())
joined='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
for c in nb['cells']:
    if c['cell_type']=='code':ast.parse(''.join(c['source']));assert not c['outputs']
assert base64.b64encode(Path(s.__file__).read_bytes()).decode() in joined
assert nb['metadata']['kaggle']['isGpuEnabled']==False
if '--public' in sys.argv:
    import requests
    import tyrelib as tl
    def get(p):
        r=requests.get(f'https://huggingface.co/datasets/{s.REPO}/resolve/{s.SOURCE}/{p}',timeout=60);r.raise_for_status();return r.content
    plan=json.loads(get(s.S5+'/protocol.json'));labels={r['image_id']:tl.C2I[r['proxy_label']] for r in plan['data']['records']}
    status=json.loads(get(s.S5+'/report/STATUS.json'));assert status['verified_runs']==81
    import concurrent.futures,yaml
    inventory=pd.read_csv(io.BytesIO(get(s.S5+'/report/inventory.csv')))
    configs={}
    for fold in range(3):
        for seed in (1,2,3):
            name=plan['downstream']['classifier'].format(fold=fold,seed=seed)
            r=requests.get(f'https://huggingface.co/datasets/{s.REPO}/resolve/{plan["source_revision"]}/runs/{name}/config.yaml',timeout=60);r.raise_for_status()
            cfg=yaml.safe_load(r.text);configs[(fold,seed)]='coral' if cfg['head_type']=='coral' else 'softmax'
    def check_run(row):
        fold,seed,run=row['fold'],row['seed'],row['run_id'];st=json.loads(get(s.S5+'/runs/'+run+'/STATUS.json'))
        raw=get(s.S5+'/runs/'+run+'/roi_predictions.csv');assert s.digest(raw)==st['artifact_sha256']['roi_predictions.csv']
        frame=pd.read_csv(io.BytesIO(raw));result=s.analyse(frame,st['job'],plan['data']['splits'][str(fold)]['validation'],labels,configs[(fold,seed)])
        for arm,mode in [('full','full'),('tyre_only','pred_tyre'),('tread_only','pred_tread')]:
            actual=pd.DataFrame(result['predictions']);a=actual[actual.arm.eq(arm)].sort_values('image_id');b=frame[frame['mode'].eq(mode)].sort_values('image_id')
            assert a.prediction.tolist()==b.prediction.tolist(), 'Single-view predictions changed'
        return len(result['metrics'])
    counts=list(concurrent.futures.ThreadPoolExecutor(8).map(check_run,inventory.to_dict('records')))
    assert len(counts)==81 and sum(counts)==405
    print('PASS all81 public S5 runs, all9 recorded classifier heads,405 metrics, unchanged single-view decisions')
print('PASS fusion math, shuffled alignment, invalid/duplicate/missing rejection, notebook syntax and embedded source')
