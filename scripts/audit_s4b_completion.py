"""Read-only, revision-pinned audit of S4b run artifacts and published report."""
import concurrent.futures
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import requests
import yaml
import pandas as pd
import numpy as np
from huggingface_hub import HfApi,hf_hub_url
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tyrelib'))
import tyrelib as tl
from s4b_confirmation import *
repo='Shanmuk4622/tyre-wear-study'
api=HfApi(); live=api.repo_info(repo,repo_type='dataset').sha
files=set(api.list_repo_files(repo,repo_type='dataset',revision=live))
print('AUDIT_REVISION',live,flush=True)
def raw(path,revision=live):
    r=requests.get(hf_hub_url(repo,path,repo_type='dataset',revision=revision),timeout=60)
    r.raise_for_status();return r.content
def csv(path,revision=live):
    return pd.read_csv(io.BytesIO(raw(path,revision)))
plan=make_plan(csv('tables/stage_b_selection.csv',SOURCE_REVISION),csv('tables/stage_b_effects.csv',SOURCE_REVISION))
prefix=f'confirmations/{REVISION}/{plan_hash(plan)}'
status=json.loads(raw(prefix+'/STATUS.json'))
assert status['status']=='complete' and status['verified_runs']==18
assert json.loads(raw(prefix+'/protocol.json'))==plan
fake=SimpleNamespace(stage='c');fake.config=lambda *a,**k:tl.Session.config(fake,*a,**k)
def check(cfg):
    p='runs/'+cfg['run_id']
    for f in ('STATUS.json','config.yaml','metrics/final.csv','metrics/epochs.csv','checkpoints/ckpt_last.pt','checkpoints/ckpt_best.pt'):
        assert p+'/'+f in files
    s=json.loads(raw(p+'/STATUS.json')); h=csv(p+'/metrics/epochs.csv'); f=csv(p+'/metrics/final.csv')
    c=yaml.safe_load(raw(p+'/config.yaml'))
    assert s['status']=='completed' and s['epoch']==60
    assert len(f)==1 and f.iloc[0].epochs_trained==60 and f.iloc[0].epochs_planned==60
    assert list(h.epoch)==list(range(1,61))
    assert all(c.get(k)==cfg[k] for k in tl.RECIPE if k!='num_workers')
    assert all(c.get(k)==cfg[k] for k in ('arch','fold','seed','stage','technique','run_id'))
    assert np.isclose(f.iloc[0].final_val_f1_macro,h.iloc[-1].val_f1_macro)
    best=h.loc[h.epoch.eq(int(f.iloc[0].best_epoch))].iloc[0]
    assert np.isclose(best.val_f1_macro,f.iloc[0].best_val_f1_macro)
    print('RUN',cfg['run_id'],'60/60','last_epoch_s',round(h.iloc[-1].epoch_seconds,1),
          'gpus',h.iloc[-1].get('runtime_training_gpu_count'),'nan_batches',f.iloc[0].nan_or_inf_batches_total,flush=True)
    return f.iloc[0].to_dict(),h
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    results=list(pool.map(check,configs(fake,tl,plan)))
baseline=pd.concat([csv(f'runs/a-{a}-base-f1-s{s}/metrics/final.csv',SOURCE_REVISION)
                    for a in plan['architectures'] for s in plan['seeds']],ignore_index=True)
paired,summary=analyse(plan,pd.DataFrame([r[0] for r in results]),baseline)
for name,d,sort in [('paired_effects',paired,['run_id']),('architecture_effects',summary,['arch','factor'])]:
    saved=csv(prefix+'/'+name+'.csv')
    pd.testing.assert_frame_equal(d.sort_values(sort).reset_index(drop=True),saved.sort_values(sort).reset_index(drop=True),check_dtype=False)
coverage=csv(prefix+'/coverage.csv');assert len(coverage)==18 and coverage.verified.all()
decisions=csv(prefix+'/factor_decisions.csv').set_index('factor')
expected=summary.groupby('factor').same_direction.sum()
assert (decisions.same_direction_architectures.sort_index()==expected.sort_index()).all()
print('REPORT',json.dumps(status))
print(summary.to_string(index=False))
print('PASS: 18 completed runs, 1080 unique run/epoch rows, configs and endpoints, 36 checkpoint paths, and recomputed published report')
