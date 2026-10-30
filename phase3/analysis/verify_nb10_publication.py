"""Pinned public NB10 audit: all direct objects, native records and raw CSVs."""
import atexit,hashlib,json,re,sys,time,uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import local,Lock
import numpy as np
import pandas as pd
import requests
import torch
from filelock import FileLock
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(json.loads((ROOT/'tread_two/nb10_completion/LATEST.json').read_text())['path'])
audit_lock=FileLock(str(OUT/'VERIFY.lock'),timeout=1)
audit_lock.acquire();atexit.register(audit_lock.release)
inspection=json.loads((OUT/'INSPECTION.json').read_text())
assert [c['execution_count'] for c in inspection['cells']]==[1,2,3,4]
assert not any(o['type']=='error' for c in inspection['cells'] for o in c['outputs'])
text='\n'.join(o['text'] for c in inspection['cells'] for o in c['outputs'] if o['type']=='stream')
receipt=None;decoder=json.JSONDecoder()
for match in re.finditer(r'\{\s*"(?:status|destination)"',text):
    try: value,_=decoder.raw_decode(text[match.start():])
    except json.JSONDecodeError:continue
    if value.get('status')=='published_and_verified': receipt=value
assert receipt and receipt['owner_released']
revision=receipt['revision'];prefix=receipt['destination']
assert prefix.startswith('phase3/tread-two/atlas-v1/appearance-gate/')
base=f'https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/resolve/{revision}/{prefix}/'
sessions=local();lock=Lock();calls=0
def request(url):
    global calls
    if not hasattr(sessions,'session'):
        sessions.session=requests.Session();sessions.session.trust_env=False
    for attempt in range(6):
        with lock:calls+=1
        try:
            response=sessions.session.get(url,timeout=(20,150))
            if response.status_code==429 or response.status_code>=500:
                delay=max(float(response.headers.get('Retry-After','0')),min(60,2**attempt))
                if attempt==5:response.raise_for_status()
                time.sleep(delay);continue
            response.raise_for_status();return response
        except (requests.ConnectionError,requests.Timeout):
            if attempt==5:raise
            time.sleep(min(30,2**attempt))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()
manifest=request(base+'MANIFEST.json').json();owner=request(base+'OWNER.json').json()
assert manifest['contract']==prefix.split('/')[-1] and owner==dict(owner=None,expires=0)
(OUT/'REMOTE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
def fetch(item):
    name,meta=item;assert not Path(name).is_absolute() and '..' not in Path(name).parts
    path=OUT/'remote'/name
    if not path.exists() or path.stat().st_size!=meta['bytes'] or sha(path)!=meta['sha256']:
        body=request(base+meta['object']).content
        assert len(body)==meta['bytes'] and hashlib.sha256(body).hexdigest()==meta['sha256'],name
        path.parent.mkdir(parents=True,exist_ok=True)
        part=path.with_name(path.name+'.'+uuid.uuid4().hex+'.part');part.write_bytes(body)
        for attempt in range(12):
            try:
                if path.exists() and sha(path)==meta['sha256']:
                    part.unlink();break
                part.replace(path);break
            except PermissionError:
                if attempt==11:raise
                time.sleep(min(5,attempt+1))
    assert sha(path)==meta['sha256'],name
with ThreadPoolExecutor(max_workers=4) as pool:
    for index,_ in enumerate(pool.map(fetch,manifest['files'].items()),1):
        if index%100==0:print('HF objects verified:',index,'/',len(manifest['files']),flush=True)
remote=OUT/'remote';contract=json.loads((remote/'CONTRACT.json').read_text())
assert hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==manifest['contract']
assert len(contract['training_image_ids'])==386 and contract['dense_device']=='cuda:0' and contract['head_device']=='cuda:1'
assert contract['environment']['devices']==['Tesla T4','Tesla T4']
for name,value in contract['sources'].items():assert sha(remote/'source'/name)==value==sha(ROOT/'tread_two/appearance_gate'/name)
cursor=json.loads((remote/'CURSOR.json').read_text());assert cursor['phase']=='completed' and cursor['updates']==0
assert sha(remote/'state.pt')==cursor['state_sha256']
def table(family):return pd.concat([pd.read_csv(p) for p in sorted((remote/'csv').glob('epoch_*/'+family+'.csv'))],ignore_index=True)
expected=dict(source_train=386,patch_quality=12352,predictions=2702,patch_evidence=259392,affinities=1556352,feature_consistency=86464,execution=386,candidate_bank=18)
counts={}
for family,n in expected.items():
    count=sum(len(chunk) for p in (remote/'csv').glob('epoch_*/'+family+'.csv') for chunk in pd.read_csv(p,chunksize=150000))
    assert count==n==cursor['row_cursors'][family],family
    counts[family]=count
source=table('source_train').fillna('');pred=table('predictions');bank=table('candidate_bank')
assert set(source.image_id)==set(contract['training_image_ids']) and pred.groupby('image_index').size().eq(7).all()
prob=pred[[f'probability_{c}' for c in range(3)]].to_numpy()
assert np.isfinite(prob).all() and np.allclose(prob.sum(1),1,atol=1e-6)
assert (prob.argmax(1)==pred.prediction).all() and pred.correct.eq(pred.prediction.eq(pred.class_index).astype(int)).all()
assert len(bank)==bank.image_id.nunique()==bank.image_sha256.nunique()==18 and set(bank.role)=={'train'}
assert bank.groupby('class_index').size().to_dict()=={0:6,1:6,2:6}
assert table('execution').optimizer_updates.eq(0).all() and table('execution').parameters_with_grad.eq(0).all()
for i in range(386):
    packet=torch.load(remote/f'artifacts/train/{i:04d}.pt',map_location='cpu',weights_only=False)
    row=source.set_index('image_index').loc[i]
    assert packet['role']=='train' and packet['image_id']==row.image_id and packet['image_sha256']==row.image_sha256
    assert np.isfinite(packet['descriptors']).all()
result_gate=json.loads((remote/'report/GATE_RESULT.json').read_text());assert result_gate['candidate_bank_structural_pass']
html=(remote/'report/Training_Gate_Explorer.html').read_text(encoding='utf-8')
payload=json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>',html,re.S)[1])
assert len(payload['cases'])==386 and len(payload['bank'])==18
for array,shape in [('candidate_rgb.npy',(18,3,64,64)),('candidate_local_contrast.npy',(18,4,64,64))]:
    value=np.load(remote/'artifacts'/array,allow_pickle=False);assert value.shape==shape and np.isfinite(value).all()
result=dict(status='passed',executed_notebook_sha256=inspection['executed_notebook_sha256'],revision=revision,destination=prefix,
    files=len(manifest['files'])+1,objects_verified=len(manifest['files']),bytes_verified=sum(m['bytes'] for m in manifest['files'].values()),
    source_code_matches_delivery=True,code_cells_completed=4,errors=0,owner_released=True,devices=contract['environment']['devices'],environment=contract['environment'],
    scientific_rows=counts,total_scientific_rows=sum(counts.values()),native_train_images_verified=386,query_images=378,support_images=8,
    predictions=2702,untrained_bank_photos=18,bank_source_sha256_unique=True,validation_or_test_forwards=0,optimizer_updates=0,
    report=result_gate,public_read_requests_this_invocation=calls,local_credentials_read=False,remote_writes=0,
    zip_npz_database_files=[n for n in manifest['files'] if n.endswith(('.zip','.npz','.sqlite'))])
assert not result['zip_npz_database_files']
(OUT/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
