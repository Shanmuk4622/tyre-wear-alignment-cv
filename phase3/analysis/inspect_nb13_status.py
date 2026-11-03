"""Preserve supplied NB13 and inspect public HF status without credentials/writes."""
import ast,base64,hashlib,json,re,sys,datetime
from pathlib import Path
import requests
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'notebooks/phase3_NB13_TREAD_TWO_ATLAS_CONTEXT_Resumable_Pilot.ipynb'
body=path.read_bytes();identity=hashlib.sha256(body).hexdigest()
OUT=ROOT/'tread_two/context_validation/nb13_status'/identity[:12];OUT.mkdir(parents=True,exist_ok=True)
archive=OUT/'executed.ipynb'
if archive.exists():assert archive.read_bytes()==body
else:archive.write_bytes(body)
def safe(value):
    text=''.join(value) if isinstance(value,list) else str(value)
    return re.sub(r'hf_[A-Za-z0-9_]+','[REDACTED]',re.sub(r'\x1b\[[0-9;]*m','',text))
book=json.loads(body);cells=[];payload=None
for i,c in enumerate(book['cells']):
    if c['cell_type']!='code':continue
    outputs=[]
    for o in c.get('outputs',[]):
        if o['output_type']=='stream':outputs.append(dict(type='stream',text=safe(o.get('text',''))[-35000:]))
        elif o['output_type']=='error':outputs.append(dict(type='error',name=o.get('ename'),value=safe(o.get('evalue','')),traceback=safe(o.get('traceback',[]))))
    cells.append(dict(index=i,execution_count=c.get('execution_count'),metadata=c.get('metadata',{}),outputs=outputs))
    for node in ast.walk(ast.parse(''.join(c['source']))):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode' and node.args and isinstance(node.args[0],ast.Constant):
            assert payload is None;payload={n:base64.b64decode(b) for n,b in json.loads(base64.b64decode(node.args[0].value)).items()}
assert payload
matches=all((ROOT/'tread_two/context_pilot'/n).read_bytes()==b for n,b in payload.items())
inspection=dict(notebook_sha256=identity,bytes=len(body),embedded_files=len(payload),all_payloads_match_current_frozen_source=matches,cells=cells)
(OUT/'INSPECTION.json').write_text(json.dumps(inspection,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(notebook_sha256=identity,bytes=len(body),archive=str(OUT),embedded_files=len(payload),payload_match=matches,cells=[dict(index=c['index'],execution_count=c['execution_count'],errors=[o for o in c['outputs'] if o['type']=='error'],last_stream='\n'.join(o['text'] for o in c['outputs'] if o['type']=='stream')[-10000:]) for c in cells]),indent=2),flush=True)
session=requests.Session();session.trust_env=False
def get(url):
    response=session.get(url,timeout=(20,75));response.raise_for_status();return response.json()
base='https://huggingface.co';repo='Shanmuk4622/tyre-wear-study'
commits=get(f'{base}/api/datasets/{repo}/commits/main?limit=1');latest=commits[0];revision=latest['id']
prefix='phase3/tread-two/atlas-context-r3'
tree=get(f'{base}/api/datasets/{repo}/tree/{revision}/{prefix}?recursive=false&expand=false')
result=dict(checked_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),revision=revision,latest_commit=latest,namespace=prefix,tree=tree,public_reads_only=True,local_credentials_read=False,remote_writes=0,runs=[])
for kind in tree:
    if kind['type']!='directory':continue
    runs=get(f'{base}/api/datasets/{repo}/tree/{revision}/{kind["path"]}?recursive=false&expand=false')
    for run in runs:
        if run['type']!='directory':continue
        path=run['path'];record=dict(prefix=path)
        for name in ['MANIFEST.json','OWNER.json']:
            response=session.get(f'{base}/datasets/{repo}/resolve/{revision}/{path}/{name}',timeout=(20,75))
            if response.status_code==404:record[name]=None;continue
            response.raise_for_status();value=response.json();(OUT/(path.split('/')[-2]+'_'+path.split('/')[-1][:12]+'_'+name)).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
            if name=='MANIFEST.json':record[name]={k:v for k,v in value.items() if k!='files'};record['object_count']=len(value['files']);record['object_names']=list(value['files'])
            else:record[name]=value
        result['runs'].append(record)
(OUT/'PUBLIC_STATUS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
(ROOT/'tread_two/context_validation/NB13_STATUS_LATEST.json').write_text(json.dumps(dict(path=str(OUT.resolve()),sha256=identity),indent=2)+'\n',encoding='utf-8')
print(json.dumps({**result,'runs':[{k:v for k,v in r.items() if k!='object_names'} for r in result['runs']]},indent=2),flush=True)
