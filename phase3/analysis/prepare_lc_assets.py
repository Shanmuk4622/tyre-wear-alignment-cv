"""Freeze r2 literal assets from hash-verified pinned Kaggle NB10 objects."""
import csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import requests
import torch
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];NEW=ROOT/'tread_two/lc_pilot'
assert not (NEW/'FILES.json').exists(), 'Preserve delivered/active bank assets; a changed bank needs a new reviewed revision.'
OUT=ROOT/'tread_two/lc_validation';OUT.mkdir(exist_ok=True)
GATE=Path(json.loads((ROOT/'tread_two/nb10_completion/LATEST.json').read_text())['path'])
manifest=json.loads((GATE/'REMOTE_MANIFEST.json').read_text())
revision='62929d235027d3f188174d7f080a034c394afab3'
prefix='phase3/tread-two/atlas-v1/appearance-gate/'+manifest['contract']
session=requests.Session();session.trust_env=False
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fetch(name):
    info=manifest['files'][name];cache=GATE/'remote'/name
    if cache.exists() and sha(cache)==info['sha256']:return cache
    path=OUT/'upstream'/name
    if not path.exists() or sha(path)!=info['sha256']:
        for attempt in range(6):
            response=session.get(f'https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/resolve/{revision}/{prefix}/{info["object"]}',timeout=(20,150))
            if response.status_code==429 or response.status_code>=500:
                time.sleep(max(float(response.headers.get('Retry-After',0)),min(30,2**attempt)));continue
            response.raise_for_status();body=response.content
            assert len(body)==info['bytes'] and hashlib.sha256(body).hexdigest()==info['sha256']
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(body);break
        else:raise RuntimeError('Cannot freeze verified upstream bank')
    return path
bank_path=fetch('csv/epoch_0008/candidate_bank.csv')
bank=list(csv.DictReader(bank_path.open(newline='',encoding='utf-8')))
assert len(bank)==len({b['image_sha256'] for b in bank})==18
rgb=[];valid=[];labels=[];hashes=[];cases=[]
for j,b in enumerate(bank):
    assert int(b['prototype'])==j and b['role']=='train' and int(b['class_index'])==j//6
    path=fetch(f'artifacts/train/{int(b["image_index"]):04d}.pt')
    p=torch.load(path,map_location='cpu',weights_only=False)
    assert p['image_id']==b['image_id'] and p['role']=='train' and p['image_sha256']==b['image_sha256']
    patch=int(b['patch']);y,x=patch//4*64,patch%4*64
    rgb.append(p['atlas']['rgb'][:,y:y+64,x:x+64]);valid.append(p['atlas']['valid'][:,y:y+64,x:x+64])
    labels.append(int(b['class_index']));hashes.append(list(bytes.fromhex(b['image_sha256'])))
    binary=np.repeat(np.repeat(p['masks']>=.5,4,-2),4,-1)
    support=binary[0]&binary[1];xs=np.arange(384)[None,:]
    left=np.where(support,xs,384).min(-1);right=np.where(support,xs,-1).max(-1)
    usable=right-left+1>=16
    clipped=float(((left[usable]==0)|(right[usable]==383)).mean()) if usable.any() else ''
    cases.append(dict(prototype_index=j,class_index=b['class_index'],source_image_id=b['image_id'],source_image_sha256=b['image_sha256'],
       source_tyre_id=b['physical_tyre_id'],atlas_patch_index=patch,valid_fraction=b['valid_fraction'],clipped_source_rows_fraction=clipped,
       gate_image_index=b['image_index'],gate_revision=revision,gate_native_sha256=sha(path),glare_fraction=b['glare_fraction'],chroma_fraction=b['chroma_fraction'],low_texture=b['low_texture']))
for name,data in dict(bank_rgb=np.stack(rgb).astype(np.float32),bank_valid=np.stack(valid).astype(np.float32),bank_labels=np.asarray(labels,dtype=np.int64),bank_sha256=np.asarray(hashes,dtype=np.uint8)).items():
    np.save(NEW/f'assets/{name}.npy',data,allow_pickle=False)
assert np.array_equal(np.load(fetch('artifacts/candidate_rgb.npy'),allow_pickle=False),np.stack(rgb))
with (NEW/'assets/prototype_bank.csv').open('w',newline='',encoding='utf-8') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(cases[0]));writer.writeheader();writer.writerows(cases)
upstream=dict(revision=revision,prefix=prefix,contract=manifest['contract'],candidate_bank_sha256=sha(bank_path),
    bank_native_sha256=[r['gate_native_sha256'] for r in cases],bank_selection_scope='train only; six distinct photos per class; no validation/test scores',
    no_new_photos=True,new_independent_tyres=0,reference_cases_trainable=False,reference_encoder_trainable=True)
(NEW/'UPSTREAM.json').write_text(json.dumps(upstream,indent=2)+'\n',encoding='utf-8')
print('Pinned Kaggle NB10 assets verified and frozen: 18 distinct source SHA; no remote writes')
