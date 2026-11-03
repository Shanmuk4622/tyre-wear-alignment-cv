"""Final completion/review handoff: identities, local docs and Git visibility."""
import ast,base64,hashlib,json,re,subprocess,sys
from pathlib import Path
from urllib.parse import unquote
import nbformat
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
done=Path(read(ROOT/'tread_two/nb13_completion/LATEST.json')['path'])
v=read(done/'VERIFICATION.json');ledger=read(done/'LEDGER_VERIFICATION.json');science=read(done/'SCIENTIFIC_FINDINGS.json')
a=read(ROOT/'tread_two/context_analysis_report/ANALYSIS.json')
delivery=read(ROOT/'tread_two/context_review_validation/NOTEBOOK.json');package=read(ROOT/'tread_two/context_review_validation/PACKAGE_VALIDATION.json')
setup=read(ROOT/'tread_two/context_review_validation/SETUP_VALIDATION.json')
assert all(x['status']=='passed' for x in [v,ledger,science,a,package])
assert v['files']==1503 and v['total_scientific_rows']==58176066 and v['phase']=='completed' and v['owner_released']
assert ledger['completed_update_sequence_0_to_2940_without_gaps'] and ledger['all60_native_epoch_protocols_updates_finite_base_unchanged']
assert science['actual_HF_restore_updates']==[2486]
assert len(a['csv_tables'])==52 and sum(m['rows'] for m in a['csv_tables'].values())==585825
assert package['notebook_sha256']==delivery['sha256']==sha(delivery['notebook'])
assert setup['status']=='passed' and setup['notebook_sha256']==delivery['sha256'] and setup['changed_source_rejected']
assert package['packaged_fresh_analysis_equal'] and package['viewer']['full_attention_cases']==206
assert all(sha(ROOT/'tread_two/context_review'/n)==s for n,s in package['source_hashes'].items())
for name,m in a['csv_tables'].items():assert sha(ROOT/'tread_two/context_analysis_report'/(name+'.csv'))==m['sha256']
nb=nbformat.read(delivery['notebook'],as_version=4);nbformat.validate(nb)
code=[c for c in nb.cells if c.cell_type=='code'];assert len(code)==4
assert all(c.execution_count is None and not c.outputs for c in code)
embedded=None
for c in code:
    for node in ast.walk(ast.parse(c.source)):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode' and node.args and isinstance(node.args[0],ast.Constant):embedded=json.loads(base64.b64decode(node.args[0].value))
assert len(embedded)==44 and all(hashlib.sha256(base64.b64decode(b)).hexdigest()==delivery['sources'][n] for n,b in embedded.items())
for n,b in embedded.items():
    path=None
    if n.startswith('review/'):path=ROOT/'tread_two/context_review'/n.removeprefix('review/')
    elif n.startswith('frozen_source/'):path=ROOT/'tread_two/context_pilot'/n.removeprefix('frozen_source/')
    elif n=='FIT_MANIFEST.json':path=done/'REMOTE_MANIFEST.json'
    elif n=='LOCAL_FULL_AUDIT.json':path=done/'VERIFICATION.json'
    assert path and path.read_bytes()==base64.b64decode(b),n
assert sha(ROOT/'notebooks/phase3_NB13A_TREAD_TWO_CONTEXT_Existing_Session_Recovery.ipynb')==sha(done/'executed.ipynb')==v['executed_notebook_sha256']
assert sha(ROOT/'notebooks/phase3_NB13_TREAD_TWO_ATLAS_CONTEXT_Resumable_Pilot.ipynb')=='e0c714be2ababb69f965572e9698d1e2145a90340ab2aa5ad3f7614a786c2f3e'
preserved=read(ROOT/'tread_two/gate_validation/PRESERVATION.json')['notebooks']
assert all(sha(Path(m['path']))==m['sha256'] for m in preserved.values())
for n,filename in [(10,'phase3_NB10_TREAD_TWO_Train_Only_Appearance_And_Reference_Gate.ipynb'),(11,'phase3_NB11_TREAD_TWO_ATLAS_LC_Resumable_Pilot.ipynb'),(12,'phase3_NB12_TREAD_TWO_ATLAS_LC_Completed_Review.ipynb')]:
    archive=Path(read(ROOT/f'tread_two/nb{n}_completion/LATEST.json')['path'])
    assert sha(ROOT/'notebooks'/filename)==read(archive/'VERIFICATION.json')['executed_notebook_sha256']
for folder,count in [('pilot',24),('lc_pilot',30),('context_pilot',32)]:
    source=ROOT/'tread_two'/folder;inventory=read(source/'FILES.json');assert len(inventory)==count
    assert all(sha(source/n)==m['sha256'] for n,m in inventory.items())
def check(base,mapping):
    assert all(sha(base/n)==s for n,s in mapping.items());return len(mapping)
preservation={
    'original_workstation':check(ROOT.parent,read(ROOT/'workstation_validation/ORIGINAL_BASELINE.json')),
    'phase3_workstation':check(ROOT,read(ROOT/'workstation_validation/DELIVERY.json')['source_hashes']),
    'paired_runtime':check(ROOT,read(ROOT/'revision2_validation/DELIVERY.json')['sources']),
    'r1_runtime':check(ROOT/'runtime',read(ROOT/'completion_20261028/hf/CONTRACT.json')['sources']),
    'appearance_gate':check(ROOT/'tread_two/appearance_gate',read(ROOT/'tread_two/gate_validation/FULL_VALIDATION.json')['source_hashes']),
    'v1_review':check(ROOT/'tread_two/review',read(ROOT/'tread_two/nb09_completion/5aa7e27dd416/remote/CONTRACT.json')['analysis_sources']),
    'lc_review':check(ROOT/'tread_two/lc_review',read(ROOT/'tread_two/lc_review_validation/PACKAGE_VALIDATION.json')['source_hashes'])}
baseline=read(ROOT/'audit/markdown-baseline.json');assert all(sha(ROOT.parent/r['path'])==r['sha256'] for r in baseline)
docs=list(ROOT.glob('*.md'))+list((ROOT/'tread_two').glob('*.md'))
for folder in ['review','analysis_report','completion_report','appearance_gate','appearance_gate_results','lc_pilot','lc_review','lc_analysis_report','context_pilot','context_review','context_analysis_report']:
    docs+=list((ROOT/'tread_two'/folder).glob('*.md'))
links=0
for doc in docs:
    for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
        target=target.strip('<>')
        if re.match(r'^(https?://|mailto:|#)',target):continue
        assert (doc.parent/unquote(target.split('#')[0])).resolve().exists(),(doc,target)
        links+=1
def ignored(name):return subprocess.run(['git','check-ignore','--no-index','-q','--',name],cwd=ROOT.parent,capture_output=True).returncode==0
hidden=['tread_two/nb13_completion/8bdf46ec2040/remote/CURSOR.json','tread_two/nb13_completion/8bdf46ec2040/executed.ipynb','tread_two/context_review_validation/acceptance_example/source.py','tread_two/context_analysis_report/context_observations.csv','tread_two/context_analysis_report/endpoint_context_evidence.csv','tread_two/context_analysis_report/ATTENTION_VIEW.json','tread_two/context_analysis_report/ATLAS_Failure_Explorer.html']
visible=['notebooks/phase3_NB14_TREAD_TWO_CONTEXT_Completed_Review.ipynb','tread_two/context_review/review_context.py','tread_two/context_review/assets/lc_selected_identity.csv','tread_two/context_analysis_report/layout_prior_context_delta.csv','tread_two/context_analysis_report/figures/context_and_attention.png','tread_two/context_review_validation/PACKAGE_VALIDATION.json','tread_two/nb13_completion/8bdf46ec2040/VERIFICATION.json','phase3NB13Completion.md','phase3TreadTwoContextReview.md']
assert all(ignored('phase3/'+n) for n in hidden)
assert all(not ignored('phase3/'+n) for n in visible)
assert read(ROOT/'audit/GIT_CONTENT_CHECK.json')['status']=='passed'
result=dict(status='passed',date_Asia_Calcutta='2026-11-01',NB13A_completed_and_verified=True,actual_recovery_update=2486,
    completed_epochs=60,completed_optimizer_updates=2940,source_revision=v['revision'],protocol=v['protocol'],verified_files=v['files'],scientific_rows=v['total_scientific_rows'],
    all60_native_epochs_frozen_prior_unchanged=True,ledger_units=ledger['unit_chain'],ledger_CSV_chunks=ledger['committed_CSV_chunks'],owner_released=True,
    original_completed_notebooks_preserved=True,frozen_source_counts=[24,30,32],preservation_counts=preservation,original_Markdown_preserved=len(baseline),
    NB14_sha256=delivery['sha256'],NB14_embedded_files=44,derived_tables=52,derived_rows=585825,review_model_forwards=0,review_training_updates=0,new_independent_tyres=0,
    fresh_packaged_science_equal=True,actual_NB14_Kaggle_HF_completion_pending=True,remote_writes=0,local_credentials_read=False,
    Git_visibility_cases=len(hidden)+len(visible),Git_content_history_check_passed=True,Markdown_documents=len(docs),local_links_checked=links,broken_links=[],
    documentation_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in docs})
path=ROOT/'tread_two/context_review_validation/HANDOFF.json';path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='documentation_hashes'},indent=2))
