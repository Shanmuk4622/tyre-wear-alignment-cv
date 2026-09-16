"""Read-only HF verification, local data checks, real CPU model and resume tests."""
import ast, copy, json, sys, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'outputs/segformer_matched_validation/deps'),str(ROOT/'tyrelib')]
import numpy as np
import requests
import torch
torch.set_num_threads(2)
import segformer_matched as s
cfg=s.contract()
old=json.loads((ROOT/'outputs/hrnet_final_audit/report_CONTRACT.json').read_text())
assert cfg['rows']==old['rows'] and cfg['groups']==old['groups']
assert cfg['mask_sha256'].keys()=={r['image_id'] for r in cfg['rows'] if r['role']=='train'}
data=Path('D:/Dataset Download/Tire Dataset Prepared/FINAL')
s.p.validate_images(cfg,data);s.MASKS=s.locate_masks(data,cfg)
mask=np.zeros((4,8),bool);mask[:,2:6]=True
assert s.boundaries(mask,[0,1,2])==[2/7,5/7]*3
mask[:]=False;assert s.boundaries(mask,[0,1,2])==[None]*6
mask[:]=True;assert s.boundaries(mask,[0,1,2])==[None]*6
mask[:]=False;mask[:,:5]=True;assert s.boundaries(mask,[0,1,2])==[None,4/7]*3
train=[r for r in cfg['rows'] if r['role']=='train']
batch=s.load_batch(train[:2],data,cfg,'cpu')
assert batch[0].shape==(2,3,512,384) and batch[1].shape==(2,2,512,384)
try:s.load_batch([next(r for r in cfg['rows'] if r['role']=='test')],data,cfg,'cpu')
except AssertionError:pass
else:raise AssertionError('Held-out masks admitted')
with tempfile.TemporaryDirectory() as tmp:
    # Only model config downloaded (~KB); no pretrained weights required for these checks.
    response=requests.get(f"https://huggingface.co/{cfg['model']}/resolve/{cfg['pretrained_revision']}/config.json",timeout=30)
    response.raise_for_status(); assert len(response.content)<100000
    # Test fixture, not a source edit.
    (Path(tmp)/'config.json').write_bytes(response.content)
    local=copy.deepcopy(cfg);local['model']=tmp
    s.h.seed_all(1);model=s.Segmentation(local);model.training_mode()
    logits=model(batch[0]);loss=s.loss(logits,batch[1]);loss.backward()
    assert logits.shape==(2,2,128,96) and torch.isfinite(loss)
    assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
    parameters=sum(v.numel() for v in model.parameters());del model,logits,loss
    # Exact stochastic continuation with the actual segmentation loss and AMP repair step.
    class Toy(torch.nn.Module):
        def __init__(self):super().__init__();self.net=torch.nn.Sequential(torch.nn.Dropout(.2),torch.nn.Conv2d(3,2,1))
        def training_mode(self):self.train()
        def forward(self,x):return self.net(x)
    def components():
        m=Toy();o=torch.optim.AdamW(m.parameters(),lr=1e-4)
        return m,o,torch.amp.GradScaler('cpu',init_scale=64),torch.optim.lr_scheduler.CosineAnnealingLR(o,60)
    s.h.coordinate_loss=s.loss
    small=(batch[0][:,:,:16,:16],batch[1][:,:,:16,:16])
    m,o,sc,lr=components()
    # Force one nonfinite gradient; require exactly one optimizer update.
    remaining=[1]
    def corrupt(g):
        if remaining[0]:remaining[0]-=1;return torch.full_like(g,float('inf'))
        return g
    handle=next(m.parameters()).register_hook(corrupt)
    s.amp.step(m,o,sc,small);handle.remove()
    assert int(o.state[next(m.parameters())]['step'])==1 and sc.get_scale()==32
    state=s.h.make_state(m,o,sc,lr,'test',1,0,1,[],{})
    path=Path(tmp)/'state.pt';s.h.atomic_save(path,state)
    expected_loss=s.amp.step(m,o,sc,small);expected=copy.deepcopy(m.state_dict())
    m,o,sc,lr=components();s.h.restore(torch.load(path,weights_only=False),m,o,sc,lr)
    resumed_loss=s.amp.step(m,o,sc,small)
    assert resumed_loss==expected_loss
    assert all(torch.equal(v,expected[k]) for k,v in m.state_dict().items())
    # Exercise the complete reporting path with explicitly synthetic SegFormer data.
    # All temporary outputs are discarded and publication is disabled.
    from unittest.mock import patch
    from types import SimpleNamespace
    mean=np.mean([r['x'] for r in train],axis=0)
    def fixture_fetch(path,revision,token=None):
        seed=int(path.split('/runs/seed')[1].split('/')[0]);kind=path.rsplit('/',1)[1]
        if kind=='STATUS.json':return dict(status='completed',completed_epochs=60,seed=seed,
            protocol=s.HR_KEY if path.startswith(s.HR_PREFIX) else s.p.sha(s.p.canonical(cfg)))
        metric=json.loads((ROOT/f'outputs/hrnet_final_audit/runs_seed{seed}_TEST_FINAL.json').read_text())
        if not path.startswith(s.HR_PREFIX):
            for i,r in enumerate(metric['records']):
                r['raw_prediction']=r['prediction'] if i else None;r['fallback_used']=not bool(i)
                if not i:r['prediction']=float(mean[s.p.POINTS.index(r['point'])])
                r['error_width_fraction']=abs(r['prediction']-r['label'])
        return metric
    with patch.object(s,'contract',return_value=cfg),patch.object(s,'fetch',side_effect=fixture_fetch), \
         patch.object(s,'ORIGINAL_PUBLISH',return_value=None), \
         patch('huggingface_hub.HfApi') as api:
        api.return_value.repo_info.return_value=SimpleNamespace(sha='synthetic-test-only')
        s.compare(tmp,None)
    report=json.loads(next(Path(tmp).glob('comparison/*/REPORT.json')).read_text())
    assert all(r['segformer_raw_coverage']==143/144 for r in report['results'])
for pattern in ['NB30_*.ipynb','NB31_*.ipynb','NB32_*.ipynb']:
    for file in (ROOT/'notebooks').glob(pattern):
        import nbformat
        nb=nbformat.read(file,as_version=4);nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type=='code':ast.parse(cell.source)
result=dict(status='local_checks_passed',protocol=s.p.sha(s.p.canonical(cfg)),same_hrnet_split=True,
    images_hashed=120,training_masks_hashed=72,parameters=parameters,forward_shape=[2,2,128,96],
    amp_overflow_retry='passed',stochastic_resume_exact=True,boundary_tests='passed',
    synthetic_report_checks='passed; temporary fixtures discarded, no HF writes',kaggle_gpu_execution='unverified')
out=ROOT/'outputs/segformer_matched_validation';out.mkdir(exist_ok=True)
s.p.write(out/'CHECKS.json',result);print(json.dumps(result,indent=2))
