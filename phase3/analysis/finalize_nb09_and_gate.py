"""Bind verified NB09, local gate, NB10, current notes and preserved artifacts."""
import json,re,subprocess,sys
from pathlib import Path
from urllib.parse import unquote
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];CHECK=ROOT/'tread_two/gate_validation'
sys.path.insert(0,str(ROOT/'tread_two/pilot'))
from common import atomic,read,sha
pub=read(Path(read(ROOT/'tread_two/nb09_completion/LATEST.json')['path'])/'VERIFICATION.json')
full=read(CHECK/'FULL_VALIDATION.json');tests=read(CHECK/'VALIDATION.json');nb=read(CHECK/'NOTEBOOK.json')
assert pub['status']==full['status']==tests['status']=='passed'
assert pub['csv_rows']==246157 and pub['owner_released'] and pub['original_fit_scientific_files_unchanged']==1177
assert full['total_scientific_rows']==1918052 and full['maximum_logit_residual']<2e-6
assert full['source_hashes']==tests['source_hashes']
gate=ROOT/'tread_two/appearance_gate'
assert all(sha(gate/name)==value for name,value in full['source_hashes'].items())
assert sha(Path(nb['notebook']))==nb['sha256']==full['notebook_sha256']
preserve=read(CHECK/'PRESERVATION.json')['notebooks']
assert all(sha(Path(item['path']))==item['sha256'] for item in preserve.values())
pilot=ROOT/'tread_two/pilot'
assert all(sha(pilot/name)==meta['sha256'] for name,meta in read(pilot/'FILES.json').items())
def check(base,mapping):
    assert all(sha(base/name)==expected for name,expected in mapping.items())
    return len(mapping)
original=check(ROOT.parent,read(ROOT/'workstation_validation/ORIGINAL_BASELINE.json'))
station=check(ROOT,read(ROOT/'workstation_validation/DELIVERY.json')['source_hashes'])
paired=check(ROOT,read(ROOT/'revision2_validation/DELIVERY.json')['sources'])
r1=check(ROOT/'runtime',read(ROOT/'completion_20261028/hf/CONTRACT.json')['sources'])
md=read(ROOT/'audit/markdown-baseline.json');assert all(sha(ROOT.parent/r['path'])==r['sha256'] for r in md)
assert sha(ROOT/'workstation/models/weights.pt')==sha(ROOT/'completion_20261028/hf/report/weights.pt')=='a5b872ed2e4cbfb3b9782b0df563e2606631744cd0274fee2da42a832f985f65'
viewer=subprocess.run(['node',str(ROOT/'analysis/test_gate_viewer.cjs'),str(ROOT/'tread_two/appearance_gate_results/Training_Gate_Explorer.html')],text=True,encoding='utf-8',capture_output=True)
assert viewer.returncode==0,viewer.stderr
atomic(CHECK/'VIEWER_VALIDATION.json',json.loads(viewer.stdout))
docs=list(ROOT.glob('*.md'))+list((ROOT/'tread_two').glob('*.md'))
for name in ['review','analysis_report','completion_report','appearance_gate','appearance_gate_results']:
    docs+=list((ROOT/'tread_two'/name).glob('*.md'))
links=0
for doc in docs:
    for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
        target=target.strip('<>')
        if re.match(r'^(https?://|mailto:|#)',target): continue
        assert (doc.parent/unquote(target.split('#')[0])).resolve().exists(),(doc,target)
        links+=1
result=dict(status='passed',date_Asia_Calcutta='2026-10-30',NB09_complete_HF_verified=True,NB09_revision=pub['revision'],NB09_files=51,NB09_csv_rows=246157,
    original_fit_scientific_objects_unchanged=1177,local_gate_protocol=full['protocol'],local_gate_images=386,local_gate_scientific_rows=1918052,
    local_gate_snapshot_bytes=full['snapshot_bytes'],local_gate_native_and_logit_audit=True,maximum_logit_residual=full['maximum_logit_residual'],
    gate_recovery_and_fake_HF_passed=True,packaged_worker_matches=True,NB10_sha256=nb['sha256'],NB10_bytes=nb['bytes'],NB10_embedded_files=36,
    actual_NB10_Kaggle_dual_T4_execution=False,live_gate_HF_publication=False,local_HF_writes=0,training_updates=0,new_independent_tyres=0,
    proposal_distinct_photos=18,proposal_trained_or_installed=False,source_query_tradeoff='Two high groups retain one original query photo',
    current_executed_notebooks_preserved={key:item['sha256'] for key,item in preserve.items()},original_model_assets24_unchanged=True,
    original_workstation_files_preserved=original,separate_workstation_sources_preserved=station,original_markdown_preserved=len(md),paired_sources_preserved=paired,r1_sources_preserved=r1,
    figures_visually_inspected=2,synthetic_DOM_controls_passed=True,browser_render_tested=False,Markdown_documents=len(docs),local_links_checked=links,broken_links=[],
    gate_source_hashes=full['source_hashes'],documentation_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in docs},
    audit_source_hashes={p.name:sha(p) for p in [ROOT/'analysis/verify_nb09_publication.py',ROOT/'analysis/validate_appearance_gate.py',ROOT/'analysis/verify_full_appearance_gate.py',Path(__file__)]})
atomic(CHECK/'HANDOFF.json',result)
print(json.dumps({k:v for k,v in result.items() if not k.endswith('hashes')},indent=2))
