"""Train-only structural proof; frozen selected LC, existing saved atlases only."""
import csv,json,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'tread_two/lc_pilot'
sys.path.insert(0,str(SOURCE))
import numpy as np
import torch
from common import read,sha,seed_all
from data_atlas import load_bank
from model_atlas import TreadTwoAtlas
from geometry import patches
OUT=ROOT/'tread_two/context_validation';OUT.mkdir(exist_ok=True)
gate=Path(read(ROOT/'tread_two/nb10_completion/LATEST.json')['path'])/'remote'
fit=Path(read(ROOT/'tread_two/nb11_completion/LATEST.json')['path'])
manifest=read(fit/'REMOTE_MANIFEST.json');checkpoint=fit/'remote/artifacts/epochs/epoch_0007.pt'
assert sha(checkpoint)==manifest['files']['artifacts/epochs/epoch_0007.pt']['sha256']
bank,labels=load_bank();device='cuda:0' if torch.cuda.is_available() else 'cpu';seed_all(1);torch.set_num_threads(4)
model=TreadTwoAtlas(bank,labels,ROOT/'completion_20261028/hf/report/weights.pt').to(device)
model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=False)['model'],strict=True)
model.eval().requires_grad_(False)
cases=list(csv.DictReader((SOURCE/'assets/prototype_bank.csv').open(encoding='utf-8')));chosen={}
for r in cases:chosen.setdefault(r['source_tyre_id'],r)
assert len(chosen)==8
def rearrange(x,order):
    t=patches(x)[:,order];b,_,c,_,_=t.shape
    return t.reshape(b,8,4,c,64,64).permute(0,3,1,4,2,5).reshape(b,c,512,256)
orders=dict(reverse_bands=[r*4+c for r in reversed(range(8)) for c in range(4)],reverse_columns=[r*4+c for r in range(8) for c in reversed(range(4))])
rows=[]
with torch.inference_mode():
    for r in chosen.values():
        p=torch.load(gate/f'artifacts/train/{int(r["gate_image_index"]):04d}.pt',map_location='cpu',weights_only=False)
        assert p['role']=='train' and p['image_sha256']==r['source_image_sha256']
        atlas={k:torch.tensor(p['atlas'][k][None],device=device) for k in ('rgb','valid')}
        hashes=torch.tensor([list(bytes.fromhex(p['image_sha256']))],dtype=torch.uint8,device=device)
        base=model.classify(atlas,hashes)
        for variant,order in orders.items():
            changed=model.classify({k:rearrange(v,order) for k,v in atlas.items()},hashes)
            residual=float((changed['logits']-base['logits']).abs().max())
            assert residual<2e-6 and torch.allclose(changed['patch_evidence'],base['patch_evidence'][:,order],atol=2e-6)
            rows.append(dict(image_id=p['image_id'],source_tyre=r['source_tyre_id'],class_index=int(r['class_index']),role='train',variant=variant,logit_max_abs_delta=residual,base_prediction=int(base['logits'].argmax()),changed_prediction=int(changed['logits'].argmax())))
with (OUT/'layout_invariance.csv').open('w',encoding='utf-8',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
result=dict(status='passed',training_photos=8,known_training_tyres=8,interventions=16,frozen_head_calls=24,dense_forwards=0,optimizer_updates=0,heldout_forwards=0,
  maximum_logit_delta=max(r['logit_max_abs_delta'] for r in rows),checkpoint_sha256=sha(checkpoint),
  scope='Band-order/within-band-column permutation invariance of the implemented pooling; not proof that layout causes the observed classification errors')
(OUT/'LAYOUT_PROOF.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
