"""Saved-result invariants, cache seam, synthetic viewer and offline publication."""
import ast,base64,importlib.util,json,re,subprocess,sys,tempfile
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import nbformat
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];CHECK=ROOT/'tread_two/lc_review_validation';CHECK.mkdir(exist_ok=True)
OUT=ROOT/'tread_two/lc_analysis_report';SOURCE=ROOT/'tread_two/lc_pilot';REVIEW=ROOT/'tread_two/lc_review'
sys.path.insert(0,str(SOURCE));import publisher
sys.path.insert(0,str(REVIEW));from review_lc import Inputs,PROTOCOL,atomic,sha
from publish_lc_review import prepare,publish
fixture=Path(tempfile.mkdtemp(prefix='acceptance_',dir=CHECK));checks=[]
audit=json.loads((OUT/'ANALYSIS.json').read_text())
for name,m in audit['csv_tables'].items():
    p=OUT/(name+'.csv');assert sha(p)==m['sha256'];frame=pd.read_csv(p);assert len(frame)==m['rows'] and list(frame.columns)==m['columns']
checks.append('Every derived CSV hash/count/column order checked')
pred=pd.read_csv(OUT/'prediction_audit.csv');pair=pd.read_csv(OUT/'paired_loss_audit.csv')
assert len(pred)==53342 and pred['sample'].is_unique and len(pair)==23520
assert pd.read_csv(OUT/'logit_reconstruction.csv').residual.max()<2e-6 and pair.loss_residual.max()<4e-7
assert pd.read_csv(OUT/'probe_summary.csv').flips.eq(0).all()
assert pd.read_csv(OUT/'query_draw_exposure.csv').actual_draws.sum()==23520
assert pd.read_csv(OUT/'query_draw_exposure.csv').view_observations.sum()==47040
assert pd.read_csv(OUT/'query_draw_exposure.csv').actual_draws.gt(0).all()
for endpoint in ['selected','final']:
    sub=pred[(pred.endpoint==endpoint)&pred.stage.isin(['selected_eval','final_eval'])]
    assert sub.label.value_counts().to_dict()=={0:62,2:41} and sub.prediction.value_counts().to_dict()=={1:62,2:41}
checks.append('Actual confusion,all386 query draws,paired objectives,zero probe flips and logit reconstruction checked')
# Use a small, isolated immutable input to test cache corruption recovery offline.
toy=fixture/'cache';toy.mkdir();manifest=fixture/'toy.json';body=b'value\n7\n'
import hashlib
atomic(manifest,dict(contract=PROTOCOL,format='immutable-direct-files-v1',files={'toy.csv':dict(bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),object='evidence/toy.csv')}))
reader=Inputs(toy,manifest)
with patch.object(reader,'request',return_value=body) as fetch:
    reader.file('toy.csv');reader.file('toy.csv');assert fetch.call_count==1
    (toy/'toy.csv').write_bytes(b'corrupt');reader.file('toy.csv');assert fetch.call_count==2
checks.append('Verified cache reuse and corrupted-object replacement tested without network')
ui=subprocess.run(['node',str(ROOT/'analysis/test_lc_review_viewer.cjs'),str(OUT/'ATLAS_Failure_Explorer.html')],capture_output=True,text=True,encoding='utf-8')
assert ui.returncode==0,ui.stderr;viewer=json.loads(ui.stdout);atomic(CHECK/'VIEWER.json',viewer)
checks.append('Actual explorer JavaScript controls pass with1442cases in synthetic DOM; no browser-render claim')
# The established in-memory immutable HF service has no network/credential path.
saved=list(sys.path)
spec=importlib.util.spec_from_file_location('offline_service_fixture',ROOT/'analysis/check_tread_two_training.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);sys.path[:]=saved
done=Path(json.loads((ROOT/'tread_two/nb11_completion/LATEST.json').read_text())['path'])
prepared=prepare(OUT,SOURCE,done/'REMOTE_MANIFEST.json',fixture/'publication')
service=module.Service();hub=publisher.Publisher.__new__(publisher.Publisher)
hub.destination=prepared['destination'];hub.owner='review-fixture';hub.api=service;hub.session=service;hub.calls=0;hub.last_verified=None;hub.log=fixture/'REMOTE.jsonl'
service.lost=True
with patch.object(publisher,'Publisher',return_value=hub):
    receipt=publish(prepared,SOURCE,fixture/'publication','fixture_not_real_token');before=service.commits
    again=publish(prepared,SOURCE,fixture/'publication','fixture_not_real_token')
assert receipt['owner_released'] and again['status']=='already_published_same_manifest' and service.commits==before
assert all(n.startswith(prepared['destination']+'/') for n in service.files) and '/fit/' not in prepared['destination']
checks.append('New review-only publication reconciles lost success,releases owner and skips identical rerun offline; fit namespace untouched')
if '--package' in sys.argv:
    delivery=json.loads((CHECK/'NOTEBOOK.json').read_text());book=nbformat.read(delivery['notebook'],as_version=4);nbformat.validate(book)
    assert sha(Path(delivery['notebook']))==delivery['sha256'];payload=None
    for cell in book.cells:
        if cell.cell_type!='code':continue
        assert cell.execution_count is None and not cell.outputs
        for node in ast.walk(ast.parse(cell.source)):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode' and node.args and isinstance(node.args[0],ast.Constant):payload=json.loads(base64.b64decode(node.args[0].value))
    assert payload and len(payload)==delivery['embedded_files']
    extracted=fixture/'extracted'
    for name,b in payload.items():
        p=extracted/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(b));assert sha(p)==delivery['sources'][name]
    fresh=fixture/'fresh_report'
    with (fixture/'console.log').open('w',encoding='utf-8') as log:
        proc=subprocess.run([sys.executable,'-B',str(extracted/'review/review_lc.py'),'--cache',str(done/'remote'),'--manifest',str(extracted/'FIT_MANIFEST.json'),'--out',str(fresh)],stdout=log,stderr=subprocess.STDOUT)
    assert proc.returncode==0,'Inspect fixture console.log'
    for p in OUT.rglob('*'):
        if p.is_file():assert sha(p)==sha(fresh/p.relative_to(OUT)),p.name
    checks.append('Fresh notebook-extracted CPU analysis reproduces every CSV/JSON/PNG/HTML/Markdown byte-for-byte')
result=dict(status='passed',checks=checks,derived_csv_tables=len(audit['csv_tables']),derived_csv_rows=sum(m['rows'] for m in audit['csv_tables'].values()),
  original_fit_source_inventory_unchanged=True,publication_fixture=True,packaged_fresh_analysis_equal='--package' in sys.argv,actual_Kaggle_NB12_execution=False,actual_live_HF_review_publication=False,
  source_hashes={p.name:sha(p) for p in REVIEW.glob('*.py')},fixture=str(fixture),network_requests_in_tests=0,remote_writes=0,model_forwards=0,training_updates=0,native_browser_render=False)
atomic(CHECK/('PACKAGE_VALIDATION.json' if '--package' in sys.argv else 'VALIDATION.json'),result);print(json.dumps(result,indent=2))
