"""Credential-free, pinned public HF metadata audit of executed NB24."""
import json,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
import nb17_public as public
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'tread_two/trace_batch';sys.path.insert(0,str(SRC))
from common import atomic,read,digest
from review import review_contract
OUT=Path(read(ROOT/'tread_two/nb24_completion/LATEST.json')['path'])
inspection=read(OUT/'INSPECTION.json')
streams='\n'.join(o['text'] for c in inspection['cells'] for o in c['outputs'] if o['type']=='stream')
offset=streams.rfind('{\n  "status": "all_complete"');assert offset>=0
execution,_=json.JSONDecoder().raw_decode(streams[offset:]);atomic(OUT/'EXECUTION_RESULT.json',execution)
assert inspection['embedded_files']==33 and inspection['all_payload_hashes_match']
assert [c['execution_count'] for c in inspection['cells']]==[1,2,3]
assert not any(o['type']=='error' for c in inspection['cells'] for o in c['outputs'])
repo='Shanmuk4622/tyre-wear-study';ns='phase3/tread-two/trace-r1'
latest=public.request('https://huggingface.co/api/datasets/'+repo).json()['sha']
def get(rev,prefix,name):
    return public.request(f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{prefix}/{name}').json()
registry=get(latest,ns,'EXPERIMENT.json');atomic(OUT/'EXPERIMENT.json',registry)
protocol=registry['protocol'];assert digest(registry['config'])==protocol
cfg=registry['config'];assert cfg['sources']==read(SRC/'FILES.json') and cfg['epochs']==60 and cfg['updates']==2940 and cfg['new_photos']==0
assert registry['mode']=='one_measured_candidate' and cfg['mode']=='one_measured_candidate'
review_protocol=digest(review_contract(protocol))
stages={}
def stage(name,prefix,rev,contract):
    m=get(rev,prefix,'MANIFEST.json');atomic(OUT/(name.upper()+'_MANIFEST.json'),m)
    assert m['contract']==contract and m['format']=='immutable-direct-files-v1'
    assert get(latest,prefix,'MANIFEST.json')==m
    owner=get(latest,prefix,'OWNER.json');atomic(OUT/(name.upper()+'_OWNER.json'),owner)
    assert owner==dict(owner=None,expires=0),(name,owner)
    def logical(n):
        v=get(rev,prefix,m['files'][n]['object']);atomic(OUT/(name.upper()+'_'+n.replace('/','_')),v);return v
    stages[name]=dict(revision=rev,prefix=prefix,contract=contract,objects=len(m['files']),bytes=sum(v['bytes'] for v in m['files'].values()),owner_released=True,latest_manifest_unchanged=True)
    print(name,json.dumps(stages[name]),flush=True)
    return m,logical
fit,fg=stage('fit',ns+'/fit/'+protocol,execution['fit_revision'],protocol)
result=fg('RESULT.json');pointer=fg('POINTER.json');fg('CONTRACT.json')
print('FIT_RESULT',json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)
prep,pg=stage('preparation',ns+'/preparation/'+protocol,latest,protocol)
ready=pg('ENGINEERING_READY.json');print('ENGINEERING',json.dumps(ready),flush=True)
gate,gg=stage('engineering',ns+'/engineering/'+ready['gate_protocol'],latest,ready['gate_protocol'])
review,rg=stage('review',ns+'/review/'+review_protocol,execution['review_revision'],review_protocol)
for n in ('RESULT.json','SCIENTIFIC_AUDIT.json','INPUT_PROVENANCE.json'):print(n,json.dumps(rg(n)),flush=True)
tail,tg=stage('transport_tail',ns+'/transport-tail/'+protocol,latest,protocol);tg('COVERAGE.json')
atomic(OUT/'METADATA.json',dict(latest_revision=latest,protocol=protocol,review_protocol=review_protocol,stages=stages,public_read_requests=public.CALLS))
print('METADATA_SAVED',flush=True)
