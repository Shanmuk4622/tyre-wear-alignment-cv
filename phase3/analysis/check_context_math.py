"""Context initialization, spatial sensitivity, frozen-base and evidence invariants."""
import copy,json,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'tread_two/context_pilot'
sys.path.insert(0,str(SOURCE))
import numpy as np
import torch
from common import read,seed_all,atomic
from data_atlas import load_bank
from model_atlas import TreadTwoAtlas
from objective import objective
torch.set_num_threads(4);seed_all(1)
fit=Path(read(ROOT/'tread_two/nb11_completion/LATEST.json')['path']);base_weights=fit/'remote/artifacts/epochs/epoch_0007.pt'
bank,labels=load_bank();device='cuda:0' if torch.cuda.is_available() else 'cpu'
model=TreadTwoAtlas(bank,labels,ROOT/'completion_20261028/hf/report/weights.pt',base_weights).to(device)
count=sum(p.numel() for p in model.parameters() if p.requires_grad);total=sum(p.numel() for p in model.parameters())
recipe=read(SOURCE/'RECIPE.json');recipe['trainable_parameters']=count;atomic(SOURCE/'RECIPE.json',recipe)
gate=Path(read(ROOT/'tread_two/nb10_completion/LATEST.json')['path'])/'remote'
cases=[]
for i in (167,0):
    p=torch.load(gate/f'artifacts/train/{i:04d}.pt',map_location='cpu',weights_only=False);assert p['role']=='train';cases.append(p)
atlas={k:torch.tensor(np.stack([p['atlas'][k] for p in cases]),device=device) for k in ('rgb','valid')}
hashes=torch.tensor([list(bytes.fromhex(p['image_sha256'])) for p in cases],dtype=torch.uint8,device=device)
labels=torch.tensor([p['class_index'] for p in cases],device=device)
with torch.no_grad():base=model.base.classify(atlas,hashes)
out=model.correct(base)
assert torch.equal(out['logits'],base['logits']) and torch.equal(out['patch_evidence'],base['patch_evidence'])
assert out['context_total'].eq(0).all() and out['attention'].shape==(2,2,4,32,32)
assert torch.isfinite(out['attention']).all() and torch.allclose(out['attention'].sum(-1),torch.ones(2,2,4,32,device=device),atol=1e-6)
frozen={k:v.detach().cpu().clone() for k,v in model.base.state_dict().items()}
with torch.no_grad():altered=model.base.classify(atlas,hashes,augmentation=torch.tensor([[.8,1.1,1.2],[1.2,.9,.8]],device=device))
other=model.correct(altered);loss,parts=objective(out,other,labels);loss.backward()
assert all(not p.requires_grad and p.grad is None for p in model.base.parameters())
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001).step()
assert all(torch.equal(v.detach().cpu(),frozen[k]) for k,v in model.base.state_dict().items())
after=model.correct(base)
residual=float((after['patch_evidence'].sum(1)+after['bias']-after['logits']).abs().max())
assert residual<2e-6 and after['context_evidence'].abs().sum(1).le(2.00001).all() and after['context_bias'].abs().le(.5).all()
# Nonzero engineering head is an architectural counterexample, not trained accuracy.
with torch.no_grad():model.head.weight.normal_(0,.03)
normal=model.correct(base);reversed=model.correct(base,'reverse_bands')
order_delta=float((normal['logits']-reversed['logits']).abs().max());assert order_delta>1e-6
# Empty support must remain finite and place no spatial mass in unsupported pixels.
empty={k:(v.clone() if torch.is_tensor(v) else v) for k,v in base.items() if k not in ('_paired',)}
empty['patch_weights']=torch.zeros_like(base['patch_weights']);empty['patch_coverage']=torch.zeros_like(base['patch_coverage'])
empty['patch_evidence']=torch.zeros_like(base['patch_evidence']);empty['logits']=base['bias'][None].expand(2,-1)
e=model.correct(empty);assert torch.isfinite(e['logits']).all() and e['context_evidence'].eq(0).all()
OUT=ROOT/'tread_two/context_validation';OUT.mkdir(exist_ok=True)
result=dict(status='passed',trainable_parameters=count,total_parameters=total,initial_predictions_and_patch_evidence_bit_equal_frozen_LC=True,
  frozen_base_unchanged_after_engineering_update=True,all_trainable_gradients_finite=True,empty_support_finite=True,attention_rows_normalized=True,
  exact_logit_accounting_max_residual=residual,engineering_counterexample_band_order_logit_delta=order_delta,
  measured_context_accuracy=None,scope='Train-only architecture checks and one disposable engineering update; no measured fit')
atomic(OUT/'MATH_VALIDATION.json',result);print(json.dumps(result,indent=2))
