"""CPU numerical-control tests using real GradScaler, no GPU/HF writes."""
import json,sys,ast,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tyrelib'))
import torch
torch.set_num_threads(2)
import hrnet_amp_repair as r
assert r.regression('cpu')['status']=='passed'
# Permanent nonfinite gradients must abort without an optimizer update.
class Bad(torch.nn.Module):
    def __init__(self):
        super().__init__();self.w=torch.nn.Parameter(torch.zeros(1,6,16))
        self.w.register_hook(lambda g:torch.full_like(g,float('inf')))
    def training_mode(self):self.train()
    def forward(self,x):return self.w.expand(x.shape[0],-1,-1)
m=Bad();o=torch.optim.AdamW(m.parameters());s=torch.amp.GradScaler('cpu',init_scale=128.)
before=m.w.detach().clone()
try:
    r.step(m,o,s,(torch.zeros(2,1),torch.full((2,6),.5)))
    raise AssertionError('Persistent corruption accepted')
except RuntimeError as e:assert 'persists' in str(e)
assert torch.equal(m.w,before) and not o.state
# Eight failed AMP attempts then a finite FP32 attempt: still exactly one update.
m=Bad();m.w._backward_hooks.clear();remaining=[8]
def eight(g):
    if remaining[0]:remaining[0]-=1;return torch.full_like(g,float('inf'))
    return g
m.w.register_hook(eight);o=torch.optim.AdamW(m.parameters());s=torch.amp.GradScaler('cpu',init_scale=128.)
loss=r.step(m,o,s,(torch.zeros(2,1),torch.full((2,6),.5)))
assert int(o.state[m.w]['step'])==1 and torch.isfinite(torch.tensor(loss))
cfg=r.p.contract(ROOT/'outputs/hrnet_validation')
assert r.p.sha(r.p.canonical(cfg))=='351c6638783996f46cf9efd99ec6689de726e045288054f240193cea61385712'
nb=json.loads((ROOT/'notebooks/NB28_HRNet_Training.ipynb').read_text())
for c in nb['cells']:
    if c['cell_type']=='code':ast.parse(''.join(c['source']))
assert "str(WORK/'hrnet_amp_repair.py')" in ''.join(nb['cells'][-1]['source'])
assert not any(c.get('outputs') for c in nb['cells'])
print('PASS: real scaler backoff, exactly one successful update, permanent corruption abort, original protocol hash, NB28 syntax/entry point')
