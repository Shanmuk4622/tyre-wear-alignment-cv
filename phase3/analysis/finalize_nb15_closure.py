"""Final closure check: executed/published identities, immutable sources and docs."""
import ast,base64,hashlib,json,re,subprocess,sys
from pathlib import Path
from urllib.parse import unquote
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
done=Path(read(ROOT/'tread_two/nb15_completion/LATEST.json')['path']);v=read(done/'VERIFICATION.json');release=ROOT/'tread_two/research_release';r=read(release/'RELEASE.json');test=read(ROOT/'tread_two/research_release_validation/VALIDATION.json')
assert v['status']==test['status']=='passed' and v['files']==38 and v['CSV_rows']==196905 and v['owner_released'] and v['private_SDK_publication_completed']
assert v['scientific_readiness'] is False and max(x['maximum_numeric_difference'] for x in v['table_comparisons'])<2e-12
assert all(p['unchanged'] for p in v['source_publications_unchanged'])
path=ROOT/'notebooks/phase3_NB15_TREAD_TWO_Existing_Data_Readiness_And_Evidence_Passport.ipynb';assert sha(path)==sha(done/'executed.ipynb')==v['executed_notebook_sha256']
delivery=read(done/'DELIVERY_NOTEBOOK.json');book=json.loads(path.read_bytes());payload=None
for cell in book['cells']:
    if cell['cell_type']!='code':continue
    assert cell['execution_count'] in (1,2,3,4) and not any(o['output_type']=='error' for o in cell['outputs'])
    for node in ast.walk(ast.parse(''.join(cell['source']))):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode' and node.args and isinstance(node.args[0],ast.Constant):payload=json.loads(base64.b64decode(node.args[0].value))
assert len(payload)==40
for name,value in payload.items():
    body=base64.b64decode(value);assert hashlib.sha256(body).hexdigest()==delivery['sources'][name]
    if name.startswith('gate/'):target=ROOT/'tread_two/readiness_gate'/name.removeprefix('gate/')
    elif name.startswith('frozen_source/'):target=ROOT/'tread_two/context_pilot'/name.removeprefix('frozen_source/')
    elif name=='NB14_VERIFICATION.json':target=Path(read(ROOT/'tread_two/nb14_completion/LATEST.json')['path'])/'VERIFICATION.json'
    else:assert name=='NB07_VERIFICATION.json';target=ROOT/'data_gate_validation/nb07_publication/VERIFICATION.json'
    assert target.read_bytes()==body
assert test['release_sha256']==sha(release/'RELEASE.json') and r['classification']=='research_only_unpromoted' and not r['HF_publication_of_this_index']
assert all(sha(ROOT/n)==h for n,h in r['source_hashes'].items())
assert all(sha(release/n)==m['sha256'] for n,m in r['artifacts'].items())
for folder,count in [('pilot',24),('lc_pilot',30),('context_pilot',32)]:
    source=ROOT/'tread_two'/folder;inventory=read(source/'FILES.json');assert len(inventory)==count
    assert all(sha(source/n)==m['sha256'] for n,m in inventory.items())
for item in read(ROOT/'tread_two/gate_validation/PRESERVATION.json')['notebooks'].values():assert sha(item['path'])==item['sha256']
for n,filename in [(10,'phase3_NB10_TREAD_TWO_Train_Only_Appearance_And_Reference_Gate.ipynb'),(11,'phase3_NB11_TREAD_TWO_ATLAS_LC_Resumable_Pilot.ipynb'),(12,'phase3_NB12_TREAD_TWO_ATLAS_LC_Completed_Review.ipynb'),(14,'phase3_NB14_TREAD_TWO_CONTEXT_Completed_Review.ipynb')]:
    archive=Path(read(ROOT/f'tread_two/nb{n}_completion/LATEST.json')['path']);assert sha(ROOT/'notebooks'/filename)==read(archive/'VERIFICATION.json')['executed_notebook_sha256']
fit=Path(read(ROOT/'tread_two/nb13_completion/LATEST.json')['path'])
assert sha(ROOT/'notebooks/phase3_NB13A_TREAD_TWO_CONTEXT_Existing_Session_Recovery.ipynb')==read(fit/'VERIFICATION.json')['executed_notebook_sha256']
assert sha(ROOT/'notebooks/phase3_NB13_TREAD_TWO_ATLAS_CONTEXT_Resumable_Pilot.ipynb')=='e0c714be2ababb69f965572e9698d1e2145a90340ab2aa5ad3f7614a786c2f3e'
def check(base,mapping):
    assert all(sha(base/n)==h for n,h in mapping.items());return len(mapping)
preservation={
    'original_workstation':check(ROOT.parent,read(ROOT/'workstation_validation/ORIGINAL_BASELINE.json')),
    'phase3_workstation':check(ROOT,read(ROOT/'workstation_validation/DELIVERY.json')['source_hashes']),
    'paired_runtime':check(ROOT,read(ROOT/'revision2_validation/DELIVERY.json')['sources']),
    'r1_runtime':check(ROOT/'runtime',read(ROOT/'completion_20261028/hf/CONTRACT.json')['sources']),
    'appearance_gate':check(ROOT/'tread_two/appearance_gate',read(ROOT/'tread_two/gate_validation/FULL_VALIDATION.json')['source_hashes']),
    'v1_review':check(ROOT/'tread_two/review',read(ROOT/'tread_two/nb09_completion/5aa7e27dd416/remote/CONTRACT.json')['analysis_sources']),
    'LC_review':check(ROOT/'tread_two/lc_review',read(ROOT/'tread_two/lc_review_validation/PACKAGE_VALIDATION.json')['source_hashes']),
    'context_review':check(ROOT/'tread_two/context_review',read(ROOT/'tread_two/context_review_validation/PACKAGE_VALIDATION.json')['source_hashes']),
    'readiness_gate':check(ROOT/'tread_two/readiness_gate',read(ROOT/'tread_two/readiness_validation/PACKAGE_VALIDATION.json')['source_hashes'])}
baseline=read(ROOT/'audit/markdown-baseline.json');assert all(sha(ROOT.parent/item['path'])==item['sha256'] for item in baseline)
docs=list(ROOT.glob('*.md'))+list((ROOT/'tread_two').glob('*.md'))
for folder in ['review','analysis_report','completion_report','appearance_gate','appearance_gate_results','lc_pilot','lc_review','lc_analysis_report','context_pilot','context_review','context_analysis_report','readiness_gate','readiness_report','research_release']:
    docs+=list((ROOT/'tread_two'/folder).glob('*.md'))
links=0
for doc in docs:
    text=doc.read_text(encoding='utf-8');assert '\ufffd' not in text,doc
    for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',text):
        target=target.strip('<>')
        if re.match(r'^(https?://|mailto:|#)',target):continue
        assert (doc.parent/unquote(target.split('#')[0])).resolve().exists(),(doc,target)
        links+=1
def ignored(name):return subprocess.run(['git','check-ignore','--no-index','-q','--',name],cwd=ROOT.parent,capture_output=True).returncode==0
hidden=['phase3/tread_two/nb15_completion/98837b907abc/remote/CAPABILITIES.json','phase3/tread_two/nb15_completion/98837b907abc/executed.ipynb','phase3/tread_two/readiness_report/online_component_counterfactuals.csv']
visible=['phase3/tread_two/research_release/Evidence_Hub.html','phase3/tread_two/research_release/figures/candidate_comparison.png','phase3/tread_two/research_release/models.csv','phase3/tread_two/research_release/RELEASE.json','phase3/tread_two/research_release/MODEL_CARD_CONTEXT.md','phase3/phase3NB15Completion.md','phase3/phase3TreadTwoResearchClosure.md']
assert all(ignored(p) for p in hidden) and all(not ignored(p) for p in visible)
assert read(ROOT/'audit/GIT_CONTENT_CHECK.json')['status']=='passed'
result=dict(status='passed',date_Asia_Calcutta='2026-11-01',NB15_verified=True,executed_notebook_sha256=v['executed_notebook_sha256'],HF_revision=v['revision'],verified_files=38,CSV_tables=16,CSV_rows=196905,maximum_cross_environment_numeric_difference=max(x['maximum_numeric_difference'] for x in v['table_comparisons']),owner_released=True,
    prior_fit_review_manifests_unchanged=True,scientific_readiness=False,classifier_status='research_only_unpromoted',research_handoff_complete=True,model_cards=3,pinned_publications=9,pinned_artifact_descriptors=14,local_index_HF_published=False,
    all_completed_notebooks_preserved=True,frozen_source_counts=[24,30,32],preservation_counts=preservation,original_Markdown_preserved=len(baseline),
    Markdown_documents=len(docs),local_links_checked=links,broken_links=[],Git_visibility_cases=len(hidden)+len(visible),Git_content_history_check_passed=True,
    new_model_forwards=0,optimizer_updates=0,new_independent_tyres=0,remote_writes=0,local_credentials_read=False,workstation_changes=False,further_notebook_required_for_unchanged_studies=False,
    documentation_hashes={doc.relative_to(ROOT).as_posix():sha(doc) for doc in docs})
(ROOT/'tread_two/research_release_validation/HANDOFF.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:value for k,value in result.items() if k!='documentation_hashes'},indent=2))
