"""Bind executed NB12, new NB13 and preserved studies without rewriting evidence."""
import ast,base64,json,re,sys
from pathlib import Path
from urllib.parse import unquote
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'tread_two/context_validation'
sys.path.insert(0,str(ROOT/'tread_two/context_pilot'))
from common import atomic,read,sha
from schemas import SCHEMAS
v=read(ROOT/'tread_two/nb12_completion/036132254087/VERIFICATION.json')
checks={name:read(OUT/(name+'.json')) for name in ['LAYOUT_PROOF','MATH_VALIDATION','LOCAL_VALIDATION','PACKAGE_VALIDATION','CSV_ACCOUNTING','VIEWER_VALIDATION']}
assert v['status']=='passed' and all(r['status']=='passed' for r in checks.values())
def check(base,mapping):
    assert all(sha(base/n)==s for n,s in mapping.items())
    return len(mapping)
preserved=read(ROOT/'tread_two/gate_validation/PRESERVATION.json')['notebooks']
assert all(sha(Path(m['path']))==m['sha256'] for m in preserved.values())
names={10:'phase3_NB10_TREAD_TWO_Train_Only_Appearance_And_Reference_Gate.ipynb',11:'phase3_NB11_TREAD_TWO_ATLAS_LC_Resumable_Pilot.ipynb',12:'phase3_NB12_TREAD_TWO_ATLAS_LC_Completed_Review.ipynb'}
notebooks={}
for n,name in names.items():
    done=Path(read(ROOT/f'tread_two/nb{n}_completion/LATEST.json')['path']);receipt=read(done/'VERIFICATION.json')
    assert sha(ROOT/'notebooks'/name)==sha(done/'executed.ipynb')==receipt['executed_notebook_sha256']
    notebooks[name]=receipt['executed_notebook_sha256']
counts={}
for folder,number in [('pilot',24),('lc_pilot',30),('context_pilot',32)]:
    source=ROOT/'tread_two'/folder;inventory=read(source/'FILES.json')
    assert len(inventory)==number and all(sha(source/n)==m['sha256'] and (source/n).stat().st_size==m['bytes'] for n,m in inventory.items())
    counts[folder]=len(inventory)
assert sha(ROOT/'tread_two/pilot/FILES.json')=='5f29f62840452059ae0c3acdfc7703a4022d9b011ce0b66b0b558cce8511d493'
counts['appearance_gate']=check(ROOT/'tread_two/appearance_gate',read(ROOT/'tread_two/gate_validation/FULL_VALIDATION.json')['source_hashes'])
counts['v1_review']=check(ROOT/'tread_two/review',read(ROOT/'tread_two/nb09_completion/5aa7e27dd416/remote/CONTRACT.json')['analysis_sources'])
counts['lc_review']=check(ROOT/'tread_two/lc_review',read(ROOT/'tread_two/lc_review_validation/PACKAGE_VALIDATION.json')['source_hashes'])
counts['original_workstation']=check(ROOT.parent,read(ROOT/'workstation_validation/ORIGINAL_BASELINE.json'))
counts['phase3_workstation']=check(ROOT,read(ROOT/'workstation_validation/DELIVERY.json')['source_hashes'])
counts['paired_runtime']=check(ROOT,read(ROOT/'revision2_validation/DELIVERY.json')['sources'])
counts['r1_runtime']=check(ROOT/'runtime',read(ROOT/'completion_20261028/hf/CONTRACT.json')['sources'])
baseline=read(ROOT/'audit/markdown-baseline.json');assert all(sha(ROOT.parent/r['path'])==r['sha256'] for r in baseline)
nb13=ROOT/'notebooks/phase3_NB13_TREAD_TWO_ATLAS_CONTEXT_Resumable_Pilot.ipynb'
assert sha(nb13)==checks['PACKAGE_VALIDATION']['notebook_sha256']
notebook=read(nb13);payloads=[]
for cell in notebook['cells']:
    if cell['cell_type']!='code':continue
    assert cell['execution_count'] is None and not cell['outputs']
    for node in ast.walk(ast.parse(''.join(cell['source']))):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode' and node.args and isinstance(node.args[0],ast.Constant):payloads.append(json.loads(base64.b64decode(node.args[0].value)))
assert len(payloads)==1 and len(payloads[0])==33
assert all(base64.b64decode(value)==(ROOT/'tread_two/context_pilot'/name).read_bytes() for name,value in payloads[0].items())
docs=list(ROOT.glob('*.md'))+list((ROOT/'tread_two').glob('*.md'))
for folder in ['review','analysis_report','completion_report','appearance_gate','appearance_gate_results','lc_pilot','lc_review','lc_analysis_report','context_pilot']:
    docs+=list((ROOT/'tread_two'/folder).glob('*.md'))
links=0;broken=[]
for doc in docs:
    for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
        target=target.strip('<>')
        if re.match(r'^(https?://|mailto:|#)',target):continue
        if not (doc.parent/unquote(target.split('#')[0])).resolve().exists():broken.append((doc.relative_to(ROOT).as_posix(),target))
        links+=1
assert not broken,broken
assert len(SCHEMAS)==27 and sum(len(c) for c in SCHEMAS.values())==328
git=read(ROOT/'audit/GIT_CONTENT_CHECK.json');policy=read(ROOT/'audit/GIT_IGNORE_POLICY_CHECK.json')
assert git['status']==policy['status']=='passed' and policy['frozen_reference_arrays_visible']==10
result=dict(status='passed',date_Asia_Calcutta='2026-10-30',NB12_revision=v['revision'],NB12_contract=v['contract'],NB12_files=63,NB12_tables=40,NB12_rows=380595,
    all_NB12_published_bytes_verified=True,exact_derived_tables_equal=True,original_fit_manifest_all1350_entries_unchanged=True,NB12_owner_released=True,
    NB08_NB09_executed_preserved=True,executed_notebooks=notebooks,preserved_source_counts=counts,original_Markdown_preserved=len(baseline),
    NB13_notebook_sha256=sha(nb13),NB13_embedded_files=33,new_frozen_source_entries=32,scientific_families=27,declared_columns=328,
    all_local_engineering_checks_passed=True,planned_observations=53754,planned_endpoint_attention_rows=1687552,
    projected_peak_bytes=checks['LOCAL_VALIDATION']['actual_model_proof']['storage_projection']['conservative_peak_output_bytes'],
    actual_Kaggle_context_execution=False,actual_context_live_HF_test=False,measured_context_fit_started=False,measured_context_accuracy=None,
    new_independent_tyres=0,local_remote_writes=0,credentials_read=False,workstations_changed=False,
    Git_content_check_passed=True,Git_policy_check_passed=True,index_changed_this_task=False,commit_or_push=False,
    Markdown_documents=len(docs),local_links_checked=links,broken_links=[],figures_visually_inspected=['context_validation/figures/layout_invariance.png','lc_analysis_report/figures/fit_and_confusion.png'],native_browser_render=False,
    source_inventory_sha256=sha(ROOT/'tread_two/context_pilot/FILES.json'),documentation_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in docs})
atomic(OUT/'HANDOFF.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='documentation_hashes'},indent=2))
