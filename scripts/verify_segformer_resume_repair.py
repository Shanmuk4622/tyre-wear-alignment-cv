"""CPU derivative equivalence and actual-model stochastic continuation regression."""
import os,sys,copy,json,ast,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'outputs/segformer_matched_validation/deps'),str(ROOT/'tyrelib')]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
import requests
import segformer_resume_repair as r
torch.set_num_threads(2)
max_error=0.
for shape,size in [((2,3,4,5),(16,20)),((1,2,5,7),(3,4)),((1,1,1,1),(7,9)),((1,2,5,6),(5,6)),((1,2,16,12),(128,96))]:
    x=torch.randn(shape,requires_grad=True);y=x.detach().clone().requires_grad_()
    a=r.ORIGINAL(x,size=size,mode='bilinear',align_corners=False);b=r.Bilinear.apply(y,size)
    g=torch.randn_like(a);a.backward(g);b.backward(g)
    assert torch.equal(a,b)
    err=float((x.grad-y.grad).abs().max());max_error=max(max_error,err)
    torch.testing.assert_close(x.grad,y.grad,atol=3e-5,rtol=3e-5)
r.install()
cfg=r.s.contract();assert r.s.p.sha(r.s.p.canonical(cfg))=='8013bf1e6418e6f884a353efa7ce892c31fbaac450f9d7caede636d9e61e2a18'
with tempfile.TemporaryDirectory() as tmp:
    response=requests.get(f"https://huggingface.co/{cfg['model']}/resolve/{cfg['pretrained_revision']}/config.json",timeout=30)
    response.raise_for_status();(Path(tmp)/'config.json').write_bytes(response.content)
    local=copy.deepcopy(cfg);local['model']=tmp
    r.s.h.seed_all(5);m=r.s.Segmentation(local);o=torch.optim.AdamW(m.parameters(),lr=1e-4)
    batch=torch.randn(2,3,128,96);target=(torch.rand(2,2,128,96)>.5).float()
    def step(m,o):
        m.training_mode();o.zero_grad(set_to_none=True);loss=r.s.loss(m(batch),target)
        loss.backward();o.step();return float(loss.detach())
    step(m,o);step(m,o)
    saved=copy.deepcopy(dict(model=m.state_dict(),optimizer=o.state_dict(),rng=r.s.h.rng()))
    expected_losses=[step(m,o),step(m,o)];expected=copy.deepcopy(m.state_dict())
    del m,o
    m=r.s.Segmentation(local);o=torch.optim.AdamW(m.parameters(),lr=1e-4)
    m.load_state_dict(saved['model']);o.load_state_dict(saved['optimizer']);r.s.h.set_rng(saved['rng'])
    actual=[step(m,o),step(m,o)]
    assert actual==expected_losses
    assert all(torch.equal(v,expected[k]) for k,v in m.state_dict().items())
import nbformat
nb=nbformat.read(ROOT/'notebooks/NB31_SegFormer_Matched_Training.ipynb',as_version=4);nbformat.validate(nb)
for c in nb.cells:
    if c.cell_type=='code':ast.parse(c.source)
out=ROOT/'outputs/segformer_matched_validation'
r.s.p.write(out/'RESUME_REPAIR_CHECKS.json',dict(status='passed',max_cpu_gradient_difference=max_error,
    actual_segformer_cpu_resume_exact=True,protocol_unchanged=True,strict_smoke_tolerance_unchanged=True,
    kaggle_gpu_execution='unverified; NB31 must pass smoke before training'))
print((out/'RESUME_REPAIR_CHECKS.json').read_text())
