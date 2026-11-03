"""Actual saved science/package, range-cache, viewer and offline publication checks."""
import ast,base64,hashlib,importlib.util,json,re,subprocess,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import nbformat
import pandas as pd
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'tread_two/context_analysis_report';CHECK=ROOT/'tread_two/context_review_validation';CHECK.mkdir(exist_ok=True)
SOURCE=ROOT/'tread_two/context_pilot';REVIEW=ROOT/'tread_two/context_review'
sys.path.insert(0,str(SOURCE));import publisher
sys.path.insert(0,str(REVIEW));from review_context import Inputs,PROTOCOL,atomic,sha
from publish_context_review import prepare,publish
fixture=Path(tempfile.mkdtemp(prefix='acceptance_',dir=CHECK));checks=[]
audit=json.loads((OUT/'ANALYSIS.json').read_text())
for name,m in audit['csv_tables'].items():
    p=OUT/(name+'.csv');assert sha(p)==m['sha256'];df=pd.read_csv(p);assert len(df)==m['rows'] and list(df.columns)==m['columns']
assert pd.read_csv(OUT/'logit_reconstruction.csv').residual.max()<2e-6
assert pd.read_csv(OUT/'endpoint_patch_accounting.csv').maximum_patch_residual.max()<2e-6
assert pd.read_csv(OUT/'paired_loss_audit.csv').loss_residual.max()<5e-6
assert len(pd.read_csv(OUT/'context_observations.csv'))==53754 and len(pd.read_csv(OUT/'attention_head_audit.csv'))==1648
assert pd.read_csv(OUT/'attention_head_audit.csv').row_sum_max_residual.max()<3e-7
checks.append('Every derived table hash/count/columns and correction/pair/attention residuals checked')
toy=fixture/'cache';toy.mkdir();body=b'value\n7\n';manifest=fixture/'manifest.json'
atomic(manifest,dict(contract=PROTOCOL,format='immutable-direct-files-v1',files={'toy.csv':dict(bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),object='evidence/toy.csv')}))
reader=Inputs(toy,manifest)
with patch.object(reader,'request',return_value=body) as fetch:
    reader.file('toy.csv');reader.file('toy.csv');assert fetch.call_count==1
    (toy/'toy.csv').write_bytes(b'bad');reader.file('toy.csv');assert fetch.call_count==2
checks.append('Small-object cache reuse and corruption replacement')
large=b'x'*(2*1024**2+19)
atomic(manifest,dict(contract=PROTOCOL,format='immutable-direct-files-v1',files={'large.csv':dict(bytes=len(large),sha256=hashlib.sha256(large).hexdigest(),object='evidence/large.csv')}))
reader=Inputs(toy,manifest)
def ranges(url,byte_range=None):
    assert byte_range is not None;start,end,total=byte_range;assert total==len(large);return large[start:end+1]
with patch.object(reader,'request',side_effect=ranges) as request:
    assert reader.file('large.csv').read_bytes()==large and request.call_count==3
checks.append('Bounded range assembly yields exact declared bytes/SHA offline')
response=SimpleNamespace(status_code=200,headers={},raise_for_status=lambda:None,close=lambda:None)
reader.tls.session=SimpleNamespace(get=lambda *a,**k:response)
try:reader.request('https://huggingface.co/offline',(0,9,100))
except AssertionError:pass
else:raise AssertionError('Ignored byte ranges must be rejected before reading content')
checks.append('Invalid/ignored byte-range response rejected')
ui=subprocess.run(['node',str(ROOT/'analysis/test_context_review_viewer.cjs'),str(OUT/'ATLAS_Failure_Explorer.html')],capture_output=True,text=True,encoding='utf-8')
assert ui.returncode==0,ui.stderr;viewer=json.loads(ui.stdout);atomic(CHECK/'VIEWER.json',viewer)
checks.append('Actual explorer JavaScript controls and explicit attention missingness pass synthetic DOM')
saved=list(sys.path);spec=importlib.util.spec_from_file_location('context_review_service',ROOT/'analysis/check_context_training.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);sys.path[:]=saved
done=Path(json.loads((ROOT/'tread_two/nb13_completion/LATEST.json').read_text())['path'])
prepared=prepare(OUT,SOURCE,done/'REMOTE_MANIFEST.json',fixture/'publication');service=module.Service()
hub=publisher.Publisher.__new__(publisher.Publisher);hub.destination=prepared['destination'];hub.owner='context-review-fixture';hub.api=service;hub.session=service;hub.calls=0;hub.last_verified=None;hub.log=fixture/'REMOTE.jsonl'
service.lost=True
with patch.object(publisher,'Publisher',return_value=hub):
    receipt=publish(prepared,SOURCE,fixture/'publication','fixture');before=service.commits
    again=publish(prepared,SOURCE,fixture/'publication','fixture')
assert receipt['owner_released'] and again['status']=='already_published_same_manifest' and service.commits==before
assert all(n.startswith(prepared['destination']+'/') for n in service.files) and '/fit/' not in prepared['destination']
checks.append('Additive review publication reconciles lost success,releases owner and skips same rerun offline')
delivery=json.loads((CHECK/'NOTEBOOK.json').read_text());nb=nbformat.read(delivery['notebook'],as_version=4);nbformat.validate(nb);payload=None
for cell in nb.cells:
    if cell.cell_type!='code':continue
    assert cell.execution_count is None and not cell.outputs
    for node in ast.walk(ast.parse(cell.source)):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode' and node.args and isinstance(node.args[0],ast.Constant):payload=json.loads(base64.b64decode(node.args[0].value))
assert payload and len(payload)==delivery['embedded_files'] and sha(Path(delivery['notebook']))==delivery['sha256']
extracted=fixture/'extracted'
for name,value in payload.items():
    body=base64.b64decode(value);assert hashlib.sha256(body).hexdigest()==delivery['sources'][name]
    p=extracted/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(body)
# Recompute with the extracted scientific source, using the full immutable cache.
script=extracted/'review/review_context.py';fresh=fixture/'fresh_report'
with (fixture/'fresh_console.log').open('wb') as stream:
    proc=subprocess.run([sys.executable,'-B',str(script),'--cache',str(done/'remote'),'--manifest',str(extracted/'FIT_MANIFEST.json'),'--out',str(fresh)],stdout=stream,stderr=subprocess.STDOUT)
assert proc.returncode==0,(fixture/'fresh_console.log').read_text()[-2500:]
fresh_audit=json.loads((fresh/'ANALYSIS.json').read_text());assert fresh_audit['csv_tables']==audit['csv_tables']
checks.append('Fresh extracted notebook source reproduces every scientific table byte-for-byte')
result=dict(status='passed',checks=checks,notebook_sha256=delivery['sha256'],embedded_files=len(payload),derived_tables=len(audit['csv_tables']),derived_rows=sum(v['rows'] for v in audit['csv_tables'].values()),
    packaged_fresh_analysis_equal=True,viewer=viewer,offline_publication=True,source_hashes={p.name:sha(p) for p in REVIEW.glob('*.py')},
    model_forward_calls=0,training_updates=0,remote_writes=0,actual_Kaggle_NB14_execution=False,actual_HF_review_publication=False,fixture=str(fixture))
atomic(CHECK/'PACKAGE_VALIDATION.json',result);print(json.dumps(result,indent=2))
