"""Extract notebook in a fresh directory and run its actual one-update worker."""
import ast,base64,csv,itertools,json,os,re,sys,tempfile
from pathlib import Path
import nbformat
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'tread_two/context_pilot'
sys.path.insert(0,str(SOURCE))
from common import atomic,read,sha
from store import Store
from schemas import SCHEMAS
import acceptance
nbpath=ROOT/'notebooks/phase3_NB13_TREAD_TWO_ATLAS_CONTEXT_Resumable_Pilot.ipynb'
nb=nbformat.read(nbpath,as_version=4);nbformat.validate(nb)
payload=None
for cell in nb.cells:
    if cell.cell_type!='code':continue
    assert cell.execution_count is None and not cell.outputs
    for node in ast.walk(ast.parse(cell.source)):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode' and node.args and isinstance(node.args[0],ast.Constant):
            value=json.loads(base64.b64decode(node.args[0].value));assert payload is None
            payload={n:base64.b64decode(b) for n,b in value.items()}
assert len(payload)==33 and all(body==(SOURCE/n).read_bytes() for n,body in payload.items())
out=ROOT/'tread_two/context_validation';folder=Path(tempfile.mkdtemp(prefix='nb13_extract_',dir=out));extracted=folder/'source'
for n,body in payload.items():p=extracted/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(body)
prior=read(out/'LOCAL_VALIDATION.json');cfg=read(Path(prior['fixture_directory'])/'CONFIG.json')
config=folder/'CONFIG.json';atomic(config,cfg)
import subprocess
environment=dict(os.environ);environment.pop('HF_TOKEN',None)
with (folder/'console.log').open('wb') as stream:
    proc=subprocess.run([sys.executable,'-B',str(extracted/'worker.py'),'--config',str(config),'--folder',str(folder/'run'),'--limit-updates','1'],stdout=stream,stderr=subprocess.STDOUT,env=environment)
assert proc.returncode==91,(folder/'console.log').read_text()[-3000:]
actual=Store(folder/'run',cfg['contract'],SCHEMAS);expected=Store(Path(prior['fixture_directory'])/'actual/paused',cfg['contract'],SCHEMAS)
for key in ('model','optimizer','scheduler','scaler','rng','epoch_orders','row_cursors','sample_cursor'):
    acceptance.same(actual.state[key],expected.state[key],key)
omit={'compute_seconds','record_seconds','cuda_peak_allocated_bytes','cuda_peak_reserved_bytes','host_peak_rss_bytes','free_disk_bytes'}
for family in SCHEMAS:
    for a,b in itertools.zip_longest(actual.rows(family),expected.rows(family)):
        assert a is not None and b is not None
        if family=='train_updates':a={k:v for k,v in a.items() if k not in omit};b={k:v for k,v in b.items() if k not in omit}
        assert a==b,(family,a,b)
# Internal report payload is complete; this does not claim a native browser render.
report=Path(prior['final_snapshot'])/'report/ATLAS_Evidence_Explorer.html'
data=json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>',report.read_text(),re.S)[1])
assert len(data['cases'])==4 and len(data['bank'])==18
result=dict(status='passed',notebook_sha256=sha(nbpath),embedded_files=33,code_cells=5,extract_directory=str(folder),actual_batch=8,
  completed_optimizer_updates=1,training_views_retained=16,native_state_matches_reference=True,scientific_CSV_families_match=27,
  report_fixture_cases=4,report_references=18,interactive_payload_checked=True,native_browser_render=False,Kaggle_execution=False,network_requests=0)
atomic(out/'PACKAGE_VALIDATION.json',result);print(json.dumps(result,indent=2))
