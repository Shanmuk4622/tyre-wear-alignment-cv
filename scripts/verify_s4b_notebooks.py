"""CPU/no-training regression tests for S4b. Reads only the frozen public discovery tables."""
import ast
import io
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import requests
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tyrelib'))
import tyrelib as tl
from s4b_confirmation import *

def csv(rel):
    url=f'https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/resolve/{SOURCE_REVISION}/{rel}'
    response=requests.get(url,timeout=60);response.raise_for_status()
    return pd.read_csv(io.BytesIO(response.content))
plan=make_plan(csv('tables/stage_b_selection.csv'),csv('tables/stage_b_effects.csv'))
fake=SimpleNamespace(stage='c')
fake.config=lambda *a,**k:tl.Session.config(fake,*a,**k)
cfgs=configs(fake,tl,plan)
assert len(cfgs)==18 and all(c['fold']==1 and c['_strict_resume'] for c in cfgs)
assert all(c['_single_gpu'] for c in cfgs)
assert tl.config_hash({'a':1})==tl.config_hash({'a':1,'_single_gpu':True})
for cfg in cfgs:
    base=tl.Session.config(fake,cfg['arch'],1,cfg['seed'])
    diffs={k for k in tl.RECIPE if base[k]!=cfg[k]}
    factor=cfg['technique'].removeprefix('s4b_r1_')
    assert diffs==set(FACTORS[factor])
    assert cfg['batch_size']=={'convnextv2_t':32,'mobilenetv4':64}[cfg['arch']]
assert tl.config_hash({'a':1})==tl.config_hash({'a':1,'_strict_resume':True})
baseline=pd.DataFrame([dict(arch=a,fold=1,seed=s,best_val_f1_macro=.5,final_val_f1_macro=.4)
                       for a in plan['architectures'] for s in (1,2,3)])
trained=pd.DataFrame([dict(c,best_val_f1_macro=.55,final_val_f1_macro=.3) for c in cfgs])
paired,summary=analyse(plan,trained,baseline)
assert len(paired)==18 and len(summary)==6
assert summary.loc[summary.factor.eq('sampler_classweighted'),'same_direction'].all()
assert not summary.loc[summary.factor.eq('transfer_random'),'same_direction'].any()
assert np.allclose(summary.mean_final_delta,-.1)
try:
    analyse(plan,trained.iloc[:-1],baseline)
except AssertionError:
    pass
else:
    raise AssertionError('Partial result must not become a complete comparison')
with tempfile.TemporaryDirectory() as d:
    checkpoint=Path(d)/'checkpoint.pt'
    t=SimpleNamespace(cfg={'_strict_resume':True,'config_hash':'expected'},run_id='test',
        ckpt_last=checkpoint,fetch_remote_state=lambda:None,
        sess=SimpleNamespace(inventory=SimpleNamespace(epoch=lambda rid:0)))
    assert tl.Trainer.try_resume(t,None,None,None,None) is False
    def must_stop():
        try:
            tl.Trainer.try_resume(t,None,None,None,None)
        except RuntimeError:
            return
        raise AssertionError('Strict resume did not stop')
    t.sess.inventory.epoch=lambda rid:5
    must_stop()
    checkpoint.write_bytes(b'broken checkpoint')
    must_stop()
    torch.save({'config_hash':'wrong'},checkpoint)
    must_stop()
for name in ('NB12_S4B_Confirmation.ipynb','NB12R_S4B_Report.ipynb'):
    nb=json.loads((ROOT/'notebooks'/name).read_text())
    for i,c in enumerate(nb['cells']):
        if c['cell_type']=='code':
            ast.parse(''.join(c['source']),filename=f'{name}:{i}')
print('PASS: live frozen HF ranking; 18 distinct one-factor configs; original batches; paired endpoints; negative discovery directions; partial-result guard; fresh/missing/corrupt/mismatched strict resume; both notebooks parse')
print('PLAN_HASH',plan_hash(plan))
import build_s4b_notebooks as builder
assert 'single=bool(int(sys.argv[4]))' in builder.SMOKE
assert "str(int(runtime['_single_gpu']))" in builder.TRAIN
assert "str(runtime['batch_size'])" in builder.TRAIN
assert 'assert median < 4.0' in builder.SMOKE
print('PASS: timed preflight GPU/batch routing comes from actual training configs; speed guard retained')
print('DISCOVERY', {k:plan['discovery_mean_delta'][k] for k in plan['factors']})
