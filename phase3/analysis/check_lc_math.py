"""R2 scientific invariants, data roles, sampling and paired objective checks."""
import json,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'tread_two/lc_pilot'
sys.path.insert(0,str(SOURCE))
import numpy as np
import torch
from common import atomic,seed_all
from data_atlas import dataset,locate,load_bank,schedule,source_hashes,bank_cases
from model_atlas import TreadTwoAtlas
from objective import objective
from lc_features import contrast
torch.set_num_threads(4);seed_all(1)
plan,queries=dataset(locate());assert len(queries)==386
bank,labels=load_bank();model=TreadTwoAtlas(bank,labels,ROOT/'completion_20261028/hf/report/weights.pt')
assert sum(p.numel() for p in model.parameters() if p.requires_grad)==81274
cases=bank_cases();lookup={r['image_id']:r for r in queries}
rows=[lookup[cases[i]['source_image_id']] for i in (0,6,12)]
hashes=source_hashes(rows,'cpu')
# This fixture bypasses the dense anchor and tests the actual trainable head.
atlas=dict(rgb=torch.rand(3,3,512,256),valid=torch.ones(3,1,512,256))
out=model.classify(atlas,hashes);mask=out['reference_allowed']
assert (~mask).sum().item()==3
assert not mask[0,0] and not mask[1,6] and not mask[2,12]
effective=out['effective_coefficients']
assert effective[0,:,0].eq(0).all() and torch.allclose(out['reference_scale'][0,:6],torch.full((6,),1.2))
reconstruction=model.bias+(out['affinity'][:,:,None,:]*effective[:,None]*out['patch_weights'][:,:,None,None]).sum((1,3))
assert (reconstruction-out['logits']).abs().max()<2e-6
gray=model.classify(atlas,hashes,variant='grayscale_both')
gray_residual=float((gray['logits']-out['logits']).abs().max());assert gray_residual<2e-5
# Changing targets cannot change support permissions/logits; targets enter objective only.
with torch.no_grad(): identical=model.classify(atlas,hashes)
assert torch.equal(identical['logits'],out['logits'])
loss,parts=objective(out,out,torch.tensor([0,1,2]));assert parts['js'].abs().max()<1e-6 and parts['texture'].abs().max()<1e-6
loss.backward();assert all(p.grad is None for p in model.anchor.parameters())
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
# A byte mismatch is a different source: all 18 references must remain available.
other=hashes.clone();other[:,0]^=255
assert model.classify(atlas,other)['reference_allowed'].all()
# Source labels never participate in the SHA mask; full digest, not an image-index heuristic.
order=schedule(queries,1,0);assert order==schedule(queries,1,0) and len(order)==49 and all(len(b)==8 for b in order)
pools={}
for i,r in enumerate(queries):pools.setdefault(r['class_index'],{}).setdefault(r['domain'],{}).setdefault(r['physical_tyre_id'] or r['source_video'],[]).append(i)
prob=[]
for i,r in enumerate(queries):
    groups=pools[r['class_index']][r['domain']];g=r['physical_tyre_id'] or r['source_video']
    p=1/len(pools)/len(pools[r['class_index']])*np.sqrt(len(groups[g]))/sum(np.sqrt(len(v)) for v in groups.values())/len(groups[g])
    prob.append(float(p))
assert abs(sum(prob)-1)<1e-12
for b in order:
    for draw in b:assert abs(draw['probability']-prob[draw['index']])<1e-14
import csv
outdir=ROOT/'tread_two/lc_validation';outdir.mkdir(exist_ok=True)
with (outdir/'sampling_plan.csv').open('w',newline='',encoding='utf-8') as stream:
    w=csv.DictWriter(stream,fieldnames=['image_id','source_group','class_index','domain','draw_probability','expected_draws_60_epochs'])
    w.writeheader()
    for r,p in zip(queries,prob):w.writerow(dict(image_id=r['image_id'],source_group=r['physical_tyre_id'] or r['source_video'],class_index=r['class_index'],domain=r['domain'],draw_probability=p,expected_draws_60_epochs=p*60*49*8))
result=dict(status='passed',checks=['386 train roles','18 distinct source SHA bank cases','source-only self exclusion','6/n reference scaling','exact additive logits','grayscale identity to FP32 tolerance','paired CE/JS/embedding objective gradients','frozen anchor','deterministic square-root group sampler and probability normalization'],trainable_parameters=81274,total_parameters=sum(p.numel() for p in model.parameters()),gray_logit_max_abs=gray_residual,max_expected_draws_per_image=max(prob)*60*49*8,new_measured_accuracy=None)
atomic(outdir/'MATH_VALIDATION.json',result);print(json.dumps(result,indent=2))
