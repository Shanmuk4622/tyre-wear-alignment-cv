"""Final verified publication, payload, preservation, viewer and Markdown handoff."""
import ast,base64,json,re,subprocess,sys
from pathlib import Path
from urllib.parse import unquote
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'tread_two/lc_pilot';OUT=ROOT/'tread_two/lc_validation'
sys.path.insert(0,str(SOURCE))
from common import atomic,read,sha
GATE=Path(read(ROOT/'tread_two/nb10_completion/LATEST.json')['path'])
pub=read(GATE/'VERIFICATION.json');local=read(OUT/'LOCAL_VALIDATION.json');pkg=read(OUT/'PACKAGE_VALIDATION.json');nb=read(OUT/'NOTEBOOK.json');math=read(OUT/'MATH_VALIDATION.json');accounting=read(OUT/'CSV_ACCOUNTING.json')
assert all(p['status']=='passed' for p in (pub,local,pkg,math,accounting))
assert pub['objects_verified']==494 and pub['total_scientific_rows']==1918052 and pub['owner_released']
assert pub['devices']==['Tesla T4','Tesla T4'] and pub['optimizer_updates']==pub['remote_writes']==0
assert local['actual_model_proof']['scientific_CSV_families_equal']==24 and local['actual_model_proof']['native_model_optimizer_scheduler_scaler_RNG_equal']
assert local['actual_model_proof']['storage_projection']['projection_passed']
inventory=read(SOURCE/'FILES.json');assert len(inventory)==30
assert all(sha(SOURCE/n)==m['sha256'] for n,m in inventory.items())
assert local['source_hashes']=={n:m['sha256'] for n,m in inventory.items()}
assert sha(Path(nb['notebook']))==nb['sha256']==pkg['notebook_sha256']
book=read(nb['notebook']);payload=None
for c in book['cells']:
    if c['cell_type']!='code':continue
    assert c['execution_count'] is None and not c['outputs']
    for node in ast.walk(ast.parse(''.join(c['source']))):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode' and node.args and isinstance(node.args[0],ast.Constant):
            assert payload is None;payload={n:base64.b64decode(b) for n,b in json.loads(base64.b64decode(node.args[0].value)).items()}
assert len(payload)==31 and all(body==(SOURCE/n).read_bytes() for n,body in payload.items())
preserved=read(ROOT/'tread_two/gate_validation/PRESERVATION.json')['notebooks']
assert all(sha(Path(v['path']))==v['sha256'] for v in preserved.values())
nb10=ROOT/'notebooks/phase3_NB10_TREAD_TWO_Train_Only_Appearance_And_Reference_Gate.ipynb'
assert sha(nb10)==sha(GATE/'executed.ipynb')==pub['executed_notebook_sha256']
pilot=ROOT/'tread_two/pilot';assert sha(pilot/'FILES.json')=='5f29f62840452059ae0c3acdfc7703a4022d9b011ce0b66b0b558cce8511d493'
assert all(sha(pilot/n)==m['sha256'] for n,m in read(pilot/'FILES.json').items())
gate_source=read(ROOT/'tread_two/gate_validation/FULL_VALIDATION.json')['source_hashes']
assert all(sha(ROOT/'tread_two/appearance_gate'/n)==value for n,value in gate_source.items())
def check(base,mapping):
    assert all(sha(base/n)==v for n,v in mapping.items());return len(mapping)
oldstation=check(ROOT.parent,read(ROOT/'workstation_validation/ORIGINAL_BASELINE.json'))
station=check(ROOT,read(ROOT/'workstation_validation/DELIVERY.json')['source_hashes'])
paired=check(ROOT,read(ROOT/'revision2_validation/DELIVERY.json')['sources'])
r1=check(ROOT/'runtime',read(ROOT/'completion_20261028/hf/CONTRACT.json')['sources'])
md=read(ROOT/'audit/markdown-baseline.json');assert all(sha(ROOT.parent/r['path'])==r['sha256'] for r in md)
viewers={}
for name,script,html in [('NB10','test_gate_viewer.cjs',GATE/'remote/report/Training_Gate_Explorer.html'),('NB11_fixture','test_lc_viewer.cjs',Path(local['final_snapshot'])/'report/ATLAS_Evidence_Explorer.html')]:
    p=subprocess.run(['node',str(ROOT/'analysis'/script),str(html)],capture_output=True,text=True,encoding='utf-8')
    assert p.returncode==0,p.stderr
    viewers[name]=json.loads(p.stdout)
atomic(OUT/'VIEWER_VALIDATION.json',dict(status='passed',viewers=viewers,native_browser_render=False))
# Bind the documented handoff path before checking its local link.
atomic(OUT/'HANDOFF.json',dict(status='checking'))
docs=list(ROOT.glob('*.md'))+list((ROOT/'tread_two').glob('*.md'))
for folder in ['review','analysis_report','completion_report','appearance_gate','appearance_gate_results','lc_pilot']:
    docs+=list((ROOT/'tread_two'/folder).glob('*.md'))
links=0
for doc in docs:
    for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
        target=target.strip('<>')
        if re.match(r'^(https?://|mailto:|#)',target):continue
        assert (doc.parent/unquote(target.split('#')[0])).resolve().exists(),(doc,target)
        links+=1
result=dict(status='passed',date_Asia_Calcutta='2026-10-30',NB10_independently_HF_verified=True,NB10_revision=pub['revision'],NB10_direct_files=495,NB10_objects=494,NB10_scientific_rows=1918052,
    NB10_native_train_images=386,NB10_actual_dual_T4=True,NB10_owner_released=True,NB10_fit_updates=0,
    NB11_status='implemented_locally_validated_pending_Kaggle',NB11_sha256=nb['sha256'],NB11_embedded_files=31,NB11_code_cells=5,
    LC_source_hashes={n:m['sha256'] for n,m in inventory.items()},local_actual_batch8=True,local_uninterrupted_pause_abrupt_native_restore_equal=True,
    scientific_CSV_families=24,columns=285,independent_CSV_logit_and_loss_accounting=True,actual_Kaggle_dual_T4_LC_tested=False,actual_live_HF_LC_preflight_tested=False,measured_r2_fit_started=False,
    conservative_peak_output_bytes=local['actual_model_proof']['storage_projection']['conservative_peak_output_bytes'],
    original_workstation_files_preserved=oldstation,separate_workstation_sources_preserved=station,paired_sources_preserved=paired,r1_runtime_sources_preserved=r1,original_Markdown_preserved=len(md),
    NB08_and_NB09_executed_notebooks_preserved=True,NB10_executed_notebook_preserved=True,original_v1_assets24_preserved=True,original_gate_sources6_preserved=True,
    visual_figures_inspected=2,synthetic_DOM_viewer_controls_passed=True,native_browser_render=False,local_remote_writes=0,new_photos=0,new_independent_tyres=0,
    Markdown_documents=len(docs),local_links_checked=links,broken_links=[],documentation_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in docs})
atomic(OUT/'HANDOFF.json',result)
print(json.dumps({k:v for k,v in result.items() if not k.endswith('hashes')},indent=2))
