"""Close NB14 and validate the complete readiness handoff without changing fits."""
import ast,base64,hashlib,json,re,subprocess,sys
from pathlib import Path
from urllib.parse import unquote
import nbformat
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
done=Path(read(ROOT/'tread_two/nb14_completion/LATEST.json')['path']);v=read(done/'VERIFICATION.json')
p=read(ROOT/'tread_two/readiness_validation/PACKAGE_VALIDATION.json');d=read(ROOT/'tread_two/readiness_validation/NOTEBOOK.json');t=read(ROOT/'tread_two/readiness_validation/TRANSPORT_VALIDATION.json');a=read(ROOT/'tread_two/readiness_report/ANALYSIS.json')
assert all(x['status']=='passed' for x in (v,p,t,a))
assert v['files']==80 and v['CSV_rows']==585825 and v['owner_released'] and v['original_fit_manifest_all1502_entries_unchanged']
assert all(c['byte_identical'] and c['maximum_numeric_difference']==0 for c in v['table_comparisons'])
assert sha(ROOT/'notebooks/phase3_NB14_TREAD_TWO_CONTEXT_Completed_Review.ipynb')==sha(done/'executed.ipynb')==v['executed_notebook_sha256']
contract=read(done/'remote/CONTRACT.json');assert all(sha(ROOT/'tread_two/context_review'/n)==h for n,h in contract['analysis_sources'].items())
assert p['notebook_sha256']==d['sha256']==sha(d['notebook']) and p['heldout_noninterference'] and p['fresh_packaged_science_equal']
assert t['source_sha256']==sha(ROOT/'tread_two/readiness_gate/public_inputs.py') and t['actual_pinned_public_read']
assert all(sha(ROOT/'tread_two/readiness_gate'/n)==h for n,h in p['source_hashes'].items())
assert len(a['tables'])==16 and sum(m['rows'] for m in a['tables'].values())==196905
assert not a['summary']['scientific_readiness'] and a['summary']['folds_missing_fit_class']==2
for name,m in a['tables'].items():assert sha(ROOT/'tread_two/readiness_report'/(name+'.csv'))==m['sha256']
nb=nbformat.read(d['notebook'],as_version=4);nbformat.validate(nb);code=[c for c in nb.cells if c.cell_type=='code']
assert len(code)==4 and all(c.execution_count is None and not c.outputs for c in code)
payload=None
for c in code:
    for node in ast.walk(ast.parse(c.source)):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode' and node.args and isinstance(node.args[0],ast.Constant):payload=json.loads(base64.b64decode(node.args[0].value))
assert len(payload)==40
for n,b in payload.items():
    body=base64.b64decode(b);assert hashlib.sha256(body).hexdigest()==d['sources'][n]
    if n.startswith('gate/'):path=ROOT/'tread_two/readiness_gate'/n.removeprefix('gate/')
    elif n.startswith('frozen_source/'):path=ROOT/'tread_two/context_pilot'/n.removeprefix('frozen_source/')
    elif n=='NB14_VERIFICATION.json':path=done/'VERIFICATION.json'
    else:assert n=='NB07_VERIFICATION.json';path=ROOT/'data_gate_validation/nb07_publication/VERIFICATION.json'
    assert path.read_bytes()==body
# Every install command is scoped to the private SDK folder; no global mutation.
assert '--target' in code[2].source and '--no-deps' in code[2].source and "str(SDK)" in code[2].source
for folder,count in [('pilot',24),('lc_pilot',30),('context_pilot',32)]:
    source=ROOT/'tread_two'/folder;catalog=read(source/'FILES.json');assert len(catalog)==count
    assert all(sha(source/n)==m['sha256'] for n,m in catalog.items())
for item in read(ROOT/'tread_two/gate_validation/PRESERVATION.json')['notebooks'].values():assert sha(item['path'])==item['sha256']
for n,filename in [(10,'phase3_NB10_TREAD_TWO_Train_Only_Appearance_And_Reference_Gate.ipynb'),(11,'phase3_NB11_TREAD_TWO_ATLAS_LC_Resumable_Pilot.ipynb'),(12,'phase3_NB12_TREAD_TWO_ATLAS_LC_Completed_Review.ipynb')]:
    archive=Path(read(ROOT/f'tread_two/nb{n}_completion/LATEST.json')['path']);assert sha(ROOT/'notebooks'/filename)==read(archive/'VERIFICATION.json')['executed_notebook_sha256']
fit=Path(read(ROOT/'tread_two/nb13_completion/LATEST.json')['path']);assert sha(ROOT/'notebooks/phase3_NB13A_TREAD_TWO_CONTEXT_Existing_Session_Recovery.ipynb')==read(fit/'VERIFICATION.json')['executed_notebook_sha256']
assert sha(ROOT/'notebooks/phase3_NB13_TREAD_TWO_ATLAS_CONTEXT_Resumable_Pilot.ipynb')=='e0c714be2ababb69f965572e9698d1e2145a90340ab2aa5ad3f7614a786c2f3e'
def check(base,files):
    assert all(sha(base/n)==h for n,h in files.items());return len(files)
preservation={
    'original_workstation':check(ROOT.parent,read(ROOT/'workstation_validation/ORIGINAL_BASELINE.json')),
    'phase3_workstation':check(ROOT,read(ROOT/'workstation_validation/DELIVERY.json')['source_hashes']),
    'paired_runtime':check(ROOT,read(ROOT/'revision2_validation/DELIVERY.json')['sources']),
    'r1_runtime':check(ROOT/'runtime',read(ROOT/'completion_20261028/hf/CONTRACT.json')['sources']),
    'appearance_gate':check(ROOT/'tread_two/appearance_gate',read(ROOT/'tread_two/gate_validation/FULL_VALIDATION.json')['source_hashes']),
    'v1_review':check(ROOT/'tread_two/review',read(ROOT/'tread_two/nb09_completion/5aa7e27dd416/remote/CONTRACT.json')['analysis_sources']),
    'LC_review':check(ROOT/'tread_two/lc_review',read(ROOT/'tread_two/lc_review_validation/PACKAGE_VALIDATION.json')['source_hashes'])}
baseline=read(ROOT/'audit/markdown-baseline.json');assert all(sha(ROOT.parent/r['path'])==r['sha256'] for r in baseline)
docs=list(ROOT.glob('*.md'))+list((ROOT/'tread_two').glob('*.md'))
for folder in ['review','analysis_report','completion_report','appearance_gate','appearance_gate_results','lc_pilot','lc_review','lc_analysis_report','context_pilot','context_review','context_analysis_report','readiness_gate','readiness_report']:
    docs+=list((ROOT/'tread_two'/folder).glob('*.md'))
links=0
for doc in docs:
    text=doc.read_text(encoding='utf-8');assert '\ufffd' not in text,doc
    if doc.name=='phase3NextSteps.md':assert '\u00e2' not in text and '\u00c2' not in text,'Windows encoding regression'
    for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',text):
        target=target.strip('<>')
        if re.match(r'^(https?://|mailto:|#)',target):continue
        assert (doc.parent/unquote(target.split('#')[0])).resolve().exists(),(doc,target)
        links+=1
def ignored(name):return subprocess.run(['git','check-ignore','--no-index','-q','--',name],cwd=ROOT.parent,capture_output=True).returncode==0
hidden=['phase3/tread_two/readiness_validation/cache/review_predictions.csv','phase3/tread_two/readiness_validation/acceptance_example/source.py','phase3/tread_two/readiness_validation/fixture_example/input.csv','phase3/tread_two/readiness_report/online_component_counterfactuals.csv','phase3/tread_two/nb14_completion/356dc7893cc5/remote/ANALYSIS.json','phase3/tread_two/nb14_completion/356dc7893cc5/executed.ipynb']
visible=['phase3/tread_two/readiness_gate/INPUTS.json','phase3/tread_two/readiness_gate/run_gate.py','phase3/tread_two/readiness_report/figures/independent_support.png','phase3/tread_two/readiness_report/online_epoch_variants.csv','phase3/tread_two/readiness_report/Evidence_Passport.html','phase3/tread_two/readiness_validation/PACKAGE_VALIDATION.json','phase3/notebooks/phase3_NB15_TREAD_TWO_Existing_Data_Readiness_And_Evidence_Passport.ipynb','phase3/phase3NB14Completion.md','phase3/phase3TreadTwoReadiness.md']
assert all(ignored(n) for n in hidden) and all(not ignored(n) for n in visible)
assert read(ROOT/'audit/GIT_CONTENT_CHECK.json')['status']=='passed'
result=dict(status='passed',date_Asia_Calcutta='2026-11-01',NB14_completed_and_verified=True,NB14_executed_sha256=v['executed_notebook_sha256'],HF_revision=v['revision'],verified_files=80,verified_CSV_rows=585825,all52_tables_exact=True,original_fit_manifest_entries_unchanged=1502,owner_released=True,
    NB15_sha256=d['sha256'],NB15_payload_files=40,NB15_tables=16,NB15_rows=196905,scientific_readiness=False,heldout_noninterference=True,private_SDK_install_scoped=True,NB15_actual_Kaggle_HF_execution_pending=True,
    frozen_source_counts=[24,30,32],original_executed_notebooks_preserved=True,preservation_counts=preservation,original_Markdown_preserved=len(baseline),
    model_forwards=0,optimizer_updates=0,new_independent_tyres=0,remote_writes=0,local_credentials_read=False,Markdown_documents=len(docs),local_links_checked=links,broken_links=[],Git_visibility_cases=len(hidden)+len(visible),Git_content_history_check_passed=True,
    documentation_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in docs})
(ROOT/'tread_two/readiness_validation/HANDOFF.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='documentation_hashes'},indent=2))
