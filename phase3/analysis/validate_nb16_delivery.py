"""Final offline handoff: real model math, fresh process restore, notebook payload and preservation."""
import ast
import base64
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

assert Path(sys.prefix).name == 'cv_conda'
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'tread_two/geometry_first/runtime'
sys.path.insert(0,str(SOURCE))
from common import atomic, digest, file_meta, read, sha
from engineering_geo import run_fixture
from coordinator_geo import make_snapshot, storage_report
from report_geo import render

files=read(SOURCE/'FILES.json')
assert all(file_meta(SOURCE/n)==m for n,m in files.items())
folder=ROOT/('tread_two/geometry_first/validation/fixture_delivery_'+digest(files)[:12])
proof=run_fixture(folder,ROOT/'workstation/models/weights.pt',device='cpu',anchor_device='cpu',live=False)
storage=storage_report(folder)
gallery=render(folder)
snapshot=make_snapshot(folder,folder/'publication_fixture',proof['protocol'],resume=folder/'seam_checkpoint')
manifest=read(snapshot/'MANIFEST.json')
assert all(file_meta(snapshot/n)==m for n,m in manifest['files'].items())
assert not any(Path(n).suffix in ('.zip','.npz','.sqlite') for n in manifest['files'])
# Simulate a byte-verified download offline, then execute the real worker in a
# fresh cv_conda process. This is NOT an actual Hub publication claim.
restored=folder/'downloaded_offline'
if not restored.exists():
    shutil.copytree(snapshot,restored)
completed=subprocess.run([sys.executable,'-B',str(SOURCE/'worker_geo.py'),'continue','--output',str(folder),
                          '--restored',str(restored)],capture_output=True,text=True,check=False)
assert completed.returncode==0,completed.stdout+'\n'+completed.stderr
continued=read(folder/'HF_CONTINUATION_PROOF.json')
continued['actual_remote_download_restore_continue']=False
continued['scope']='offline_verified_copy_fresh_process_restore'
# Keep the local receipt unambiguous. The notebook uses the same worker after
# an actual Publisher.download; only its live receipt is an HF proof.
atomic(folder/'HF_CONTINUATION_PROOF.json',continued)
nbpath=ROOT/'notebooks/phase3_NB16_TREAD_TWO_GEO_Preparation_And_Contract_Gate.ipynb'
nb=read(nbpath)
code=[c for c in nb['cells'] if c['cell_type']=='code']
assert len(code)==3 and all(not c['outputs'] and c['execution_count'] is None for c in code)
for cell in code:
    ast.parse(''.join(cell['source']))
tree=ast.parse(''.join(code[0]['source']))
payload=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PAYLOAD' for t in n.targets))
assert set(payload)==set(files)|{'FILES.json'}
extracted=folder/'fresh_extracted_source'
extracted.mkdir(exist_ok=True)
for name,item in payload.items():
    body=base64.b64decode(item['base64'],validate=True)
    assert hashlib.sha256(body).hexdigest()==item['sha256']==sha(SOURCE/name)
    (extracted/name).write_bytes(body)
    if name.endswith('.py'): ast.parse(body,filename=name)
try:
    import nbformat
    nbformat.validate(nb)
    notebook_schema=True
except ImportError:
    notebook_schema='nbformat_unavailable_static_schema_checked'
preserved={}
for folder_name,count in [('pilot',24),('lc_pilot',30),('context_pilot',32)]:
    base=ROOT/'tread_two'/folder_name
    inv=read(base/'FILES.json')
    assert len(inv)==count and all(sha(base/n)==m['sha256'] for n,m in inv.items())
    preserved[folder_name]=count
for name,base,items in [
    ('original_workstation',ROOT.parent,read(ROOT/'workstation_validation/ORIGINAL_BASELINE.json')),
    ('phase3_workstation',ROOT,read(ROOT/'workstation_validation/DELIVERY.json')['source_hashes']),
    ('paired_runtime',ROOT,read(ROOT/'revision2_validation/DELIVERY.json')['sources'])]:
    assert all(sha(base/n)==h for n,h in items.items()),name
    preserved[name]=len(items)
baseline=read(ROOT/'audit/markdown-baseline.json')
assert all(sha(ROOT.parent/item['path'])==item['sha256'] for item in baseline)
for n in (10,11,12,13,14,15):
    done=Path(read(ROOT/f'tread_two/nb{n}_completion/LATEST.json')['path'])
    verification=read(done/'VERIFICATION.json')
    names={10:'phase3_NB10_TREAD_TWO_Train_Only_Appearance_And_Reference_Gate.ipynb',11:'phase3_NB11_TREAD_TWO_ATLAS_LC_Resumable_Pilot.ipynb',12:'phase3_NB12_TREAD_TWO_ATLAS_LC_Completed_Review.ipynb',13:'phase3_NB13A_TREAD_TWO_CONTEXT_Existing_Session_Recovery.ipynb',14:'phase3_NB14_TREAD_TWO_CONTEXT_Completed_Review.ipynb',15:'phase3_NB15_TREAD_TWO_Existing_Data_Readiness_And_Evidence_Passport.ipynb'}
    assert sha(ROOT/'notebooks'/names[n])==verification['executed_notebook_sha256']
for item in read(ROOT/'tread_two/gate_validation/PRESERVATION.json')['notebooks'].values():
    assert sha(item['path'])==item['sha256']
atomic(ROOT/'tread_two/geometry_first/validation/LOCAL_ENGINEERING.json',proof)
result=dict(status='passed',scope='local_CPU_and_offline_transport_only',actual_T4_HF_gate_pending=True,
            measured_fit_ready=False, measured_fit_updates=0,notebook_sha256=sha(nbpath),notebook_bytes=nbpath.stat().st_size,
            notebook_schema=notebook_schema,embedded_files=len(payload),fresh_extracted_sources_verified=True,
            fresh_process_continuation=continued,parameters=proof['parameters'],model_tests=proof,
            engineering_gallery=gallery,storage=storage,
            direct_snapshot_files=len(manifest['files']),snapshot_bytes=sum(m['bytes'] for m in manifest['files'].values()),
            preserved=preserved,original_Markdown_preserved=len(baseline),completed_notebooks_preserved=True)
atomic(ROOT/'tread_two/geometry_first/validation/DELIVERY.json',result)
print(json.dumps({k:v for k,v in result.items() if k not in ('model_tests','storage','fresh_process_continuation')},indent=2))
