"""Small read-only HF audit; no checkpoint download or remote mutations."""
import json,sys,math
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tyrelib'))
import hrnet_protocol as p
KEY='351c6638783996f46cf9efd99ec6689de726e045288054f240193cea61385712'
PREFIX=f's9/{p.VERSION}/{KEY}';OUT=ROOT/'outputs/hrnet_completion_audit';OUT.mkdir(exist_ok=True)
total=0
def request(url):
    global total
    with requests.get(url,timeout=45,stream=True) as r:
        r.raise_for_status();raw=bytearray()
        for chunk in r.iter_content(65536):
            raw.extend(chunk);total+=len(chunk);assert len(raw)<2*1024**2 and total<4*1024**2
    return json.loads(raw)
refs=request(f'https://huggingface.co/api/datasets/{p.REPO}/refs')
REV=next(x['targetCommit'] for x in refs['branches'] if x['name']=='main')
def read(path,revision=REV):
    value=request(f'https://huggingface.co/datasets/{p.REPO}/resolve/{revision}/{PREFIX}/{path}')
    p.write(OUT/(path.replace('/','_')),value);return value
pre=read('preflight/STATUS.json');smoke=read('smoke/STATUS.json')
assert pre==read('preflight/STATUS.json','32d13296fafd2253426392e009cd98fb315cb91b')
assert smoke==read('smoke/STATUS.json','240dbb9ebfb51686a7d0899959c340f658755c4c')
assert pre['status']=='preflight_passed' and smoke['status']=='smoke_passed'
cfg=read('preflight/CONTRACT.json');assert p.sha(p.canonical(cfg))==KEY
assert cfg==p.contract(ROOT/'outputs/hrnet_validation')
assert read('smoke/CONTRACT.json')==cfg
qa=read('preflight/QA.json');assert qa['counts']==dict(train=72,validation=24,test=24)
ident=read('smoke/IDENTITY.json');assert ident['parameters']==9603962 and ident['channels']==[18,36,72,144]
assert smoke['max_parameter_difference']==0 and smoke['resumed_losses']==smoke['uninterrupted_losses'][2:]
st=read('runs/seed1/STATUS.json');hist=read('runs/seed1/HISTORY.json')
assert st['protocol']==KEY and st['seed']==1
assert len(hist)==st['completed_epochs']*36+st['next_batch_cursor'] and all(math.isfinite(x['loss']) for x in hist)
tree=request(f'https://huggingface.co/api/datasets/{p.REPO}/tree/{REV}/{PREFIX}/runs/seed1')
checkpoint=next(x for x in tree if x['path'].endswith('/state.pt'))
assert checkpoint['lfs']['oid']==st['checkpoint_sha256']
report=dict(revision=REV,preflight='verified',smoke='verified',smoke_gpu=smoke['gpu'],
    peak_gpu_bytes=smoke['peak_gpu_bytes'],resume_parameter_difference=smoke['max_parameter_difference'],
    training_status=st,checkpoint_bytes=checkpoint['size'],downloaded_bytes=total,
    checkpoint_verification='Published LFS hash matches STATUS; tensor payload not downloaded',
    note='Original model/resume smoke verified. New AMP recovery branch requires NB28 GPU regression.')
p.write(OUT/'AUDIT.json',report);print(json.dumps(report,indent=2))
