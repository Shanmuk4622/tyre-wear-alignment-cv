"""CPU contract/model regression checks; no pretrained downloads, no HF writes."""
import ast,hashlib,io,json,sys,tempfile,zipfile
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1]
deps=ROOT/'outputs/hrnet_validation/test_deps';deps.mkdir(parents=True,exist_ok=True)
if not (deps/'timm').exists():
    # Small pinned test dependency, not model weights or a base-environment change.
    r=requests.get('https://pypi.org/pypi/timm/1.0.15/json',timeout=30);r.raise_for_status()
    item=next(x for x in r.json()['urls'] if x['filename'].endswith('py3-none-any.whl'))
    r=requests.get(item['url'],timeout=30);r.raise_for_status();assert len(r.content)<5*1024**2
    assert hashlib.sha256(r.content).hexdigest()==item['digests']['sha256']
    with zipfile.ZipFile(io.BytesIO(r.content)) as archive:
        assert all((deps/x.filename).resolve().is_relative_to(deps.resolve()) for x in archive.infolist())
        archive.extractall(deps)
sys.path[:0]=[str(deps),str(ROOT/'tyrelib')]
import torch
torch.set_num_threads(2)
import hrnet_protocol as p
import hrnet_runtime as h
cfg=p.contract(ROOT/'outputs/hrnet_validation')
assert {k:sum(r['role']==k for r in cfg['rows']) for k in cfg['groups']}==dict(train=72,validation=24,test=24)
assert set(cfg['groups']['train']).isdisjoint(cfg['groups']['test'])
assert set(cfg['groups']['validation']).isdisjoint(cfg['groups']['test'])
assert p.split_groups(cfg['rows'][::-1])==cfg['groups']
for n in range(26,30):
    f=next((ROOT/'notebooks').glob(f'NB{n}_HRNet_*.ipynb'));nb=json.loads(f.read_text(encoding='utf-8'))
    assert not any(c.get('outputs') for c in nb['cells'])
    for c in nb['cells']:
        if c['cell_type']=='code':ast.parse(''.join(c['source']))
h.seed_all(9);model=h.Geometry(cfg);model.training_mode()
assert model.backbone.feature_info.channels()==[18,36,72,144]
with torch.inference_mode():logits=model(torch.zeros(1,3,512,384))
assert logits.shape==(1,6,96) and torch.isfinite(logits).all()
assert all(not m.training for m in model.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm))
# Save/reload optimizer + scheduler + cursor + RNG with a small deterministic CPU model.
class Toy(torch.nn.Module):
    def __init__(self):super().__init__();self.layer=torch.nn.Linear(4,6*16)
    def forward(self,x):return self.layer(x).reshape(-1,6,16)
def parts():
    m=Toy();o=torch.optim.AdamW(m.parameters(),lr=1e-3);s=torch.amp.GradScaler('cuda',enabled=False)
    return m,o,s,torch.optim.lr_scheduler.CosineAnnealingLR(o,60)
def update(m,o,x,y):
    o.zero_grad();loss=h.coordinate_loss(m(x),y);loss.backward();o.step()
m,o,s,lr=parts();x=torch.rand(2,4);y=torch.rand(2,6)
update(m,o,x,y)
with tempfile.TemporaryDirectory(dir=ROOT/'outputs/hrnet_validation') as temp:
    path=Path(temp)/'state.pt';h.atomic_save(path,h.make_state(m,o,s,lr,'test',1,0,1,[],{}))
    update(m,o,x,y);expected={k:v.clone() for k,v in m.state_dict().items()}
    m2,o2,s2,lr2=parts();state=torch.load(path,map_location='cpu',weights_only=False)
    assert state['cursor']==1 and state['epoch']==0
    h.restore(state,m2,o2,s2,lr2);update(m2,o2,x,y)
    assert all(torch.equal(v,m2.state_dict()[k]) for k,v in expected.items())
print('PASS: 120-image/group split, four notebook syntax checks, actual HRNet-W18 512x384 CPU forward, frozen BN, masked-target loss, atomic checkpoint and CPU AdamW/RNG continuation')
print('HRNet parameters:',sum(v.numel() for v in model.parameters()))
print('Not tested locally: pretrained weight load, GPU/AMP, real HRNet optimizer continuation, HF upload, long training.')
if '--weight-header' in sys.argv:
    url=f'https://huggingface.co/{cfg["pretrained_repo"]}/resolve/{cfg["pretrained_revision"]}/model.safetensors'
    # Only inspect bounded tensor metadata, never download the pretrained tensor payload.
    with requests.get(url,headers={'Range':'bytes=0-7'},stream=True,timeout=30) as response:
        response.raise_for_status();size=int.from_bytes(response.raw.read(8),'little')
    assert 0<size<1024**2
    with requests.get(url,headers={'Range':f'bytes=8-{size+7}'},stream=True,timeout=30) as response:
        response.raise_for_status()
        if response.status_code==206:
            assert response.headers['Content-Range'].startswith('bytes 8-')
            header=response.raw.read(size)
        else:header=response.raw.read(size+8)[8:]
    tensors=json.loads(header)
    needed=model.backbone.state_dict()
    for k,v in needed.items():
        if k.endswith('num_batches_tracked') and k not in tensors:continue
        assert k in tensors and tensors[k]['shape']==list(v.shape),k
    p.write(ROOT/'outputs/hrnet_validation/PRETRAINED_HEADER_CHECK.json',dict(
        revision=cfg['pretrained_revision'],header_bytes=size,backbone_tensors=len(needed),all_shapes_match=True))
    print('PASS: pinned pretrained tensor header matches every backbone shape; no tensor payload downloaded')
