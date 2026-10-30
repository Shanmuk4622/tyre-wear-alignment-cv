"""Independent saved-result, packaged CPU execution and publication checks."""
import ast
import base64
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import nbformat
import numpy as np
import pandas as pd

assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'tread_two/analysis_report'
CHECK=ROOT/'tread_two/review_validation'
CHECK.mkdir(exist_ok=True)
fixture=Path(tempfile.mkdtemp(prefix='fixture_',dir=CHECK))
sys.path.insert(0,str(ROOT/'tread_two/review'))
from review_atlas import Inputs,PROTOCOL,atomic,sha
from publish_atlas_review import prepare,publish
tests=[]
audit=json.loads((OUT/'ANALYSIS.json').read_text())
for name,meta in audit['csv_tables'].items():
    path=OUT/(name+'.csv')
    assert sha(path)==meta['sha256']
    frame=pd.read_csv(path)
    assert len(frame)==meta['rows'] and list(frame.columns)==meta['columns']
tests.append('Every derived CSV hash, row count and field order matches its analysis receipt')
p=pd.read_csv(OUT/'prediction_audit.csv')
for endpoint in ['selected','final']:
    original=p[(p.endpoint==endpoint)&p.variant.eq('identity')&p.stage.isin(['selected_eval','final_eval'])]
    assert original.label.value_counts().to_dict()=={0:62,2:41}
    assert original.prediction.value_counts().to_dict()=={1:62,2:41}
    assert original.loc[original.label.eq(0),'prediction'].eq(1).all()
    assert original.loc[original.label.eq(2),'prediction'].eq(2).all()
    gray=p[(p.endpoint==endpoint)&p.variant.eq('grayscale_both')]
    assert len(gray)==103 and gray.prediction.eq(1).all()
assert p.query("endpoint == 'final' and variant == 'brightness_080'").prediction.eq(1).all()
tests.append('Corrected low-to-mid confusion and grayscale/dimming failures verified directly, rather than inferred from BA')
val=pd.read_csv(OUT/'cohort_metrics.csv')
assert val.query("stage == 'validate'").balanced_accuracy_observed.eq(.5).all()
assert val.query("stage == 'train' and epoch >= 4").accuracy.eq(1).all()
control=val.query("stage == 'frozen_control'")
assert len(control)==2 and np.allclose(control.nll,1.386939053656755,atol=1e-10)
assert control.predicted_mid.eq(62).all() and control.predicted_high.eq(41).all()
classes=pd.read_csv(OUT/'class_metrics.csv')
assert classes.query("stage in ['selected_eval','final_eval'] and class_index == 1").recall.isna().all()
assert pd.read_csv(OUT/'logit_reconstruction.csv').residual.max()<2e-6
assert pd.read_csv(OUT/'query_exposure.csv').training_draws.gt(0).sum()==378
tests.append('Training/validation gap, absent mid recall, all query exposure and exact saved logit accounting checked')

# Cache interruption/corruption seam with no network call: verified completed
# objects are reused; bad bytes trigger a complete verified atomic replacement.
toy=fixture/'cache';toy.mkdir()
body=b'value\n7\n'
manifest=fixture/'toy_manifest.json'
atomic(manifest,dict(contract=PROTOCOL,format='immutable-direct-files-v1',files={'toy.csv':dict(bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),object='evidence/toy.csv')}))
reader=Inputs(toy,manifest)
with patch.object(reader,'request',return_value=body) as fetch:
    reader.file('toy.csv');reader.file('toy.csv');assert fetch.call_count==1
    (toy/'toy.csv').write_bytes(b'corrupt')
    reader.file('toy.csv');assert fetch.call_count==2
assert not list(toy.glob('*.part'))
with patch.object(reader,'request',return_value=b'bad'):
    (toy/'toy.csv').write_bytes(b'bad cache')
    try: reader.file('toy.csv')
    except AssertionError: pass
    else: raise AssertionError('Corrupt resolver bytes accepted')
tests.append('Verified-cache reuse, corrupt-cache recovery, interrupted-file atomicity and bad-download rejection pass offline')

# Execute the actual JavaScript against a deliberately small synthetic DOM.
# This tests controls/records, not browser rendering or screenshot layout.
html=(OUT/'ATLAS_Failure_Explorer.html').read_text(encoding='utf-8')
data=re.search(r'<script type="application/json" id="data">(.*?)</script>',html,re.S)[1]
script=re.search(r'<script>(.*?)</script>',html,re.S)[1]
syntax=subprocess.run(['node','--check'],input=script,text=True,encoding='utf-8',capture_output=True)
assert syntax.returncode==0,syntax.stderr
payload=json.loads(data)
assert len(payload['cases'])==1030 and len(payload['bank'])==18
node=r'''const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync(process.argv[2],'utf8'),data=html.match(/<script type="application\/json" id="data">([\s\S]*?)<\/script>/)[1],code=html.match(/<script>([\s\S]*?)<\/script>/)[1];
class Element{constructor(){this.children=[];this.handlers={};this.style={};this.value='';this.textContent=''}append(...v){this.children.push(...v)}replaceChildren(...v){this.children=v}addEventListener(k,fn){this.handlers[k]=fn}}
const elements={};for(const id of ['data','summary','endpoint','variant','target','filter','class','search','count','cases','previous','next'])elements['#'+id]=new Element();
elements['#data'].textContent=data;Object.assign(elements['#endpoint'],{value:'selected'});elements['#variant'].value='identity';elements['#target'].value='all';elements['#filter'].value='all';elements['#class'].value='pred';
vm.runInNewContext(code,{document:{querySelector:s=>elements[s],createElement:()=>new Element()},console});
function set(id,value){elements[id].value=value;elements[id].handlers.input()}
assert.match(elements['#count'].textContent,/^103 records/);assert.equal(elements['#cases'].children.length,8);
elements['#next'].onclick();assert.match(elements['#count'].textContent,/page 2/);elements['#previous'].onclick();
set('#variant','grayscale_both');set('#filter','flip');assert.match(elements['#count'].textContent,/^41 records/);
set('#target','0');assert.match(elements['#count'].textContent,/^0 records/);set('#target','all');
set('#endpoint','final');set('#variant','brightness_080');assert.match(elements['#count'].textContent,/^41 records/);
set('#filter','all');set('#class','0');assert.equal(elements['#cases'].children[0].children[0].children[2].children.length,32);
const one=JSON.parse(data).cases.find(x=>x.endpoint==='final'&&x.variant==='brightness_080').image_id;
set('#search',one);assert.match(elements['#count'].textContent,/^1 records/);assert.equal(elements['#cases'].children.length,1);
console.log('Synthetic DOM controls passed: paging, probes, flips, targets, explained class, search and 32-patch rendering.');'''
(fixture/'ui_test.cjs').write_text(node,encoding='utf-8')
result=subprocess.run(['node',str(fixture/'ui_test.cjs'),str(OUT/'ATLAS_Failure_Explorer.html')],text=True,encoding='utf-8',capture_output=True)
assert result.returncode==0,result.stderr
tests.append('Actual explorer JavaScript syntax and synthetic-DOM paging/probe/flip/target/class/search controls pass; 1030 records retained')

# Extract every delivered byte and run the CPU analysis from that fresh source.
delivery=json.loads((CHECK/'NOTEBOOK.json').read_text())
notebook=Path(delivery['notebook'])
assert sha(notebook)==delivery['notebook_sha256']
nb=nbformat.read(notebook,as_version=4);nbformat.validate(nb)
payload=None
for cell in nb.cells:
    if cell.cell_type!='code': continue
    assert cell.execution_count is None and not cell.outputs
    for node in ast.walk(ast.parse(cell.source)):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode' and isinstance(node.args[0],ast.Constant):
            payload=json.loads(base64.b64decode(node.args[0].value))
assert payload and len(payload)==30
for name,value in payload.items():
    path=(fixture/'extracted'/name).resolve()
    assert path.is_relative_to((fixture/'extracted').resolve())
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(base64.b64decode(value))
    assert sha(path)==delivery['source_hashes'][name]
fresh=fixture/'analysis'
cache=ROOT/'tread_two/pilot_validation/nb08_followup/82e7e1c5774f/fit'
with (fixture/'analysis_console.log').open('w',encoding='utf-8') as log:
    run=subprocess.run([sys.executable,'-B',str(fixture/'extracted/review/review_atlas.py'),'--cache',str(cache),'--manifest',str(fixture/'extracted/FIT_MANIFEST.json'),'--out',str(fresh)],stdout=log,stderr=subprocess.STDOUT)
assert run.returncode==0,'Extracted review failed; inspect analysis_console.log'
for path in OUT.rglob('*'):
    if path.is_file(): assert sha(path)==sha(fresh/path.relative_to(OUT)),path.name
tests.append('Fresh NB09 extracted CPU analysis reproduces every CSV, JSON, Markdown, PNG and full explorer byte-for-byte')

# Exercise the new wrapper with the established immutable fake-HF service.
sys.path.insert(0,str(ROOT/'analysis'))
from check_tread_two_training import Service,fake_publisher
source=ROOT/'tread_two/pilot';launcher=ROOT/'analysis/nb08_transport_launcher.py'
prepared=prepare(OUT,source,launcher,ROOT/'tread_two/pilot_validation/nb08_followup/82e7e1c5774f/fit_MANIFEST.json',fixture/'publication')
import publisher
service=Service()
hub=fake_publisher(service,fixture)
hub.destination=prepared['destination']
assert hub.destination.startswith('phase3/tread-two/atlas-v1/review/') and '/fit/' not in hub.destination
import coordinator
def claim_then_lose_object_reply(h):
    h.claim()
    service.lost=True
with patch.object(publisher,'Publisher',return_value=hub), patch.object(coordinator,'claim',side_effect=claim_then_lose_object_reply):
    receipt=publish(prepared,source,launcher,fixture/'publication','fixture_not_a_real_token')
    count=service.commits
    again=publish(prepared,source,launcher,fixture/'publication','fixture_not_a_real_token')
assert receipt['status']=='published_and_verified' and receipt['owner_released']
assert again['status']=='already_published_same_manifest' and service.commits==count
assert json.loads(service.files[hub.destination+'/OWNER.json'])==dict(owner=None,expires=0)
assert all(name.startswith(hub.destination+'/') for name in service.files)
tests.append('New review-only publisher recovers a lost successful response, verifies direct objects, releases ownership and skips identical manifest on rerun offline')
inventory=json.loads((source/'FILES.json').read_text())
assert all(sha(source/name)==meta['sha256'] for name,meta in inventory.items())
assert sha(ROOT/'notebooks/phase3_NB08_TREAD_TWO_ATLAS_Resumable_Pilot.ipynb')=='214abe5b1c7bbc810805e6fa4012f8597c00c8559136daff544836eff9dab2c8'
tests.append('All24 original model/runtime/assets and original repaired NB08 remain byte-identical')
for doc in [OUT/'README.md',OUT/'METHODS.md',OUT/'CSV_INDEX.md']:
    for link in re.findall(r'\[[^\]]+\]\(([^)]+)\)',doc.read_text(encoding='utf-8')): assert (doc.parent/link).exists(),link
record=dict(status='passed',checks=tests,count=len(tests),derived_csv_tables=len(audit['csv_tables']),derived_csv_rows=sum(meta['rows'] for meta in audit['csv_tables'].values()),
    packaged_files=30,packaged_CPU_analysis_outputs_byte_equal=True,publication_fixture_passed=True,network_requests_in_validation=0,remote_writes=0,
    actual_Kaggle_NB09_execution=False,actual_HF_review_publication=False,browser_render_tested=False,
    browser_attempt='Browser Use could not reach local Windows loopback server (ERR_CONNECTION_TIMED_OUT); synthetic DOM verified behavior instead.',
    source_hashes={p.name:sha(p) for p in (ROOT/'tread_two/review').glob('*.py')},notebook_sha256=sha(notebook),fixture=str(fixture))
atomic(CHECK/'VALIDATION.json',record)
print(json.dumps(record,indent=2))
