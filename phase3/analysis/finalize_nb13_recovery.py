"""Validate recovery payload, documentation links, source preservation and Git policy."""
import ast,base64,hashlib,json,re,subprocess,sys
from pathlib import Path
from urllib.parse import unquote
import nbformat
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'tread_two/context_validation/nb13_status'
sys.path.insert(0,str(ROOT/'tread_two/context_pilot'))
from common import atomic,read,sha
done=Path(read(ROOT/'tread_two/context_validation/NB13_STATUS_LATEST.json')['path'])
v=read(done/'CHECKPOINT_VERIFICATION.json');test=read(OUT/'recovery_validation/VALIDATION.json');delivery=read(OUT/'recovery_validation/NOTEBOOK.json');package=read(OUT/'recovery_validation/PACKAGE_VALIDATION.json')
assert v['status']=='verified_partial_checkpoint' and test['status']=='passed' and len(test['checks'])==18
assert package['status']=='passed' and package['actual_setup_executed_in_empty_directory'] and package['notebook_sha256']==delivery['sha256']
assert test['launcher_sha256']==delivery['launcher_sha256']==sha(ROOT/'analysis/nb13_recovery_launcher.py')
notebook=nbformat.read(delivery['notebook'],as_version=4);nbformat.validate(notebook)
assert sha(Path(delivery['notebook']))==delivery['sha256']
code=[c for c in notebook.cells if c.cell_type=='code'];assert len(code)==2
payload=[]
for cell in code:
    assert cell.execution_count is None and not cell.outputs
    for node in ast.walk(ast.parse(cell.source)):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode' and node.args and isinstance(node.args[0],ast.Constant):payload.append(json.loads(base64.b64decode(node.args[0].value)))
assert len(payload)==1 and len(payload[0])==35
embedded={n:base64.b64decode(value) for n,value in payload[0].items()}
assert embedded['NB13_RECOVERY/nb13_recovery_launcher_r2.py']==(ROOT/'analysis/nb13_recovery_launcher.py').read_bytes()
assert all(embedded['source/'+n]==(ROOT/'tread_two/context_pilot'/n).read_bytes() for n in read(ROOT/'tread_two/context_pilot/FILES.json'))
error=read(OUT/'NB13A_ERROR_LATEST.json');error_folder=Path(error['path'])
assert sha(error_folder/'executed.ipynb')==error['sha256']=='b5a4dd68b4c1882ac5723c4ce0e0bdba5d1733b71ff1ac72fd599c14fdb0bdda'
assert sha(ROOT/'notebooks/phase3_NB13_TREAD_TWO_ATLAS_CONTEXT_Resumable_Pilot.ipynb')==sha(done/'executed.ipynb')=='e0c714be2ababb69f965572e9698d1e2145a90340ab2aa5ad3f7614a786c2f3e'
preserved=read(ROOT/'tread_two/gate_validation/PRESERVATION.json')['notebooks'];assert all(sha(Path(m['path']))==m['sha256'] for m in preserved.values())
for n,name in [(10,'phase3_NB10_TREAD_TWO_Train_Only_Appearance_And_Reference_Gate.ipynb'),(11,'phase3_NB11_TREAD_TWO_ATLAS_LC_Resumable_Pilot.ipynb'),(12,'phase3_NB12_TREAD_TWO_ATLAS_LC_Completed_Review.ipynb')]:
    archive=Path(read(ROOT/f'tread_two/nb{n}_completion/LATEST.json')['path'])
    assert sha(ROOT/'notebooks'/name)==read(archive/'VERIFICATION.json')['executed_notebook_sha256']
for folder,count in [('pilot',24),('lc_pilot',30),('context_pilot',32)]:
    source=ROOT/'tread_two'/folder;inventory=read(source/'FILES.json');assert len(inventory)==count
    assert all(sha(source/n)==m['sha256'] for n,m in inventory.items())
def check(base,mapping):assert all(sha(base/n)==s for n,s in mapping.items());return len(mapping)
preservation={
 'original_workstation':check(ROOT.parent,read(ROOT/'workstation_validation/ORIGINAL_BASELINE.json')),
 'phase3_workstation':check(ROOT,read(ROOT/'workstation_validation/DELIVERY.json')['source_hashes']),
 'paired_runtime':check(ROOT,read(ROOT/'revision2_validation/DELIVERY.json')['sources']),
 'r1_runtime':check(ROOT/'runtime',read(ROOT/'completion_20261028/hf/CONTRACT.json')['sources']),
 'appearance_gate':check(ROOT/'tread_two/appearance_gate',read(ROOT/'tread_two/gate_validation/FULL_VALIDATION.json')['source_hashes']),
 'lc_review':check(ROOT/'tread_two/lc_review',read(ROOT/'tread_two/lc_review_validation/PACKAGE_VALIDATION.json')['source_hashes'])}
baseline=read(ROOT/'audit/markdown-baseline.json');assert all(sha(ROOT.parent/r['path'])==r['sha256'] for r in baseline)
docs=list(ROOT.glob('*.md'))+list((ROOT/'tread_two').glob('*.md'))
for folder in ['review','analysis_report','completion_report','appearance_gate','appearance_gate_results','lc_pilot','lc_review','lc_analysis_report','context_pilot']:
    docs+=list((ROOT/'tread_two'/folder).glob('*.md'))
links=0
for doc in docs:
    for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
        target=target.strip('<>')
        if re.match(r'^(https?://|mailto:|#)',target):continue
        assert (doc.parent/unquote(target.split('#')[0])).resolve().exists(),(doc,target)
        links+=1
def ignored(name):return subprocess.run(['git','check-ignore','--no-index','-q','--',name],cwd=ROOT.parent,capture_output=True).returncode==0
assert ignored('phase3/tread_two/context_validation/nb13_status/recovery_validation/fixture_example/source.py')
assert ignored('phase3/tread_two/context_validation/nb13_status/e0c714be2aba/executed.ipynb')
assert not ignored('phase3/notebooks/phase3_NB13A_TREAD_TWO_CONTEXT_Existing_Session_Recovery.ipynb')
assert not ignored('phase3/analysis/nb13_recovery_launcher.py')
assert not ignored('phase3/tread_two/context_validation/nb13_status/e0c714be2aba/CHECKPOINT_VERIFICATION.json')
assert read(ROOT/'audit/GIT_CONTENT_CHECK.json')['status']=='passed'
result=dict(status='passed',date_Asia_Calcutta='2026-10-31',source_notebook_preserved=True,NB08_to_NB12_executed_preserved=True,
    original_frozen_packages_preserved=True,NB13_source_entries32_unchanged=True,source_inventory_sha256=sha(ROOT/'tread_two/context_pilot/FILES.json'),
    measured_protocol=v['protocol'],HF_pinned_revision=v['revision'],verified_HF_update=1294,saved_notebook_local_update=2467,
    retained_Kaggle_state_not_directly_read=True,all798_snapshot_objects_audited=False,diagnostic_objects_verified=15,
    recovery_checks=len(test['checks']),recovery_revision='r2',recovery_notebook_sha256=delivery['sha256'],recovery_launcher_sha256=delivery['launcher_sha256'],
    NB13A_observed_error='required local run/POINTER.json unavailable',NB13A_failed_execution_archived=True,standalone_payload_files=35,
    missing_local_HF_restore_tested_offline=True,actual_packaged_setup_executes_without_local_pointer=True,
    exact_original_NB13_stop_cause='not established',actual_Kaggle_recovery_pending=True,new_training_updates=0,new_model_forwards=0,new_independent_tyres=0,
    remote_writes=0,local_credentials_read=False,preservation_counts=preservation,original_Markdown_preserved=len(baseline),
    Markdown_documents=len(docs),local_links_checked=links,broken_links=[],Git_visibility_checks=5,Git_content_check_passed=True,
    documentation_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in docs})
atomic(OUT/'recovery_validation/HANDOFF.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='documentation_hashes'},indent=2))
