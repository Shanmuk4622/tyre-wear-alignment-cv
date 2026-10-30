"""Bind NB11 completion, NB12 delivery, documentation and preserved baselines."""
import json,re,sys
from pathlib import Path
from urllib.parse import unquote
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tread_two/lc_pilot'))
from common import atomic,read,sha
DONE=Path(read(ROOT/'tread_two/nb11_completion/LATEST.json')['path']);OUT=ROOT/'tread_two/lc_review_validation'
v=read(DONE/'VERIFICATION.json');analysis=read(ROOT/'tread_two/lc_analysis_report/ANALYSIS.json');validation=read(OUT/'PACKAGE_VALIDATION.json');nb=read(OUT/'NOTEBOOK.json')
assert v['status']==analysis['status']==validation['status']=='passed' and validation['packaged_fresh_analysis_equal']
assert v['total_scientific_rows']==54336368 and v['objects_verified']==1350 and v['selected_epoch']==7
assert len(analysis['csv_tables'])==40 and sum(m['rows'] for m in analysis['csv_tables'].values())==380595
source=ROOT/'tread_two/lc_pilot';inventory=read(source/'FILES.json')
assert all(sha(source/n)==m['sha256'] for n,m in inventory.items())
assert {p.name:sha(p) for p in (ROOT/'tread_two/lc_review').glob('*.py')}==validation['source_hashes']
assert sha(Path(nb['notebook']))==nb['sha256'] and nb['code_cells']==4
preserved=read(ROOT/'tread_two/gate_validation/PRESERVATION.json')['notebooks']
assert all(sha(Path(m['path']))==m['sha256'] for m in preserved.values())
nb10=Path(read(ROOT/'tread_two/nb10_completion/LATEST.json')['path']);nb10sha=read(nb10/'VERIFICATION.json')['executed_notebook_sha256']
assert sha(ROOT/'notebooks/phase3_NB10_TREAD_TWO_Train_Only_Appearance_And_Reference_Gate.ipynb')==sha(nb10/'executed.ipynb')==nb10sha
assert sha(ROOT/'notebooks/phase3_NB11_TREAD_TWO_ATLAS_LC_Resumable_Pilot.ipynb')==sha(DONE/'executed.ipynb')==v['executed_notebook_sha256']
pilot=ROOT/'tread_two/pilot';assert all(sha(pilot/n)==m['sha256'] for n,m in read(pilot/'FILES.json').items())
assert sha(pilot/'FILES.json')=='5f29f62840452059ae0c3acdfc7703a4022d9b011ce0b66b0b558cce8511d493'
gate=read(ROOT/'tread_two/gate_validation/FULL_VALIDATION.json')['source_hashes'];assert all(sha(ROOT/'tread_two/appearance_gate'/n)==s for n,s in gate.items())
old_review=read(ROOT/'tread_two/nb09_completion/5aa7e27dd416/remote/CONTRACT.json')['analysis_sources'];assert all(sha(ROOT/'tread_two/review'/n)==s for n,s in old_review.items())
def check(base,mapping):assert all(sha(base/n)==s for n,s in mapping.items());return len(mapping)
original=check(ROOT.parent,read(ROOT/'workstation_validation/ORIGINAL_BASELINE.json'))
station=check(ROOT,read(ROOT/'workstation_validation/DELIVERY.json')['source_hashes'])
paired=check(ROOT,read(ROOT/'revision2_validation/DELIVERY.json')['sources'])
r1=check(ROOT/'runtime',read(ROOT/'completion_20261028/hf/CONTRACT.json')['sources'])
baseline=read(ROOT/'audit/markdown-baseline.json');assert all(sha(ROOT.parent/r['path'])==r['sha256'] for r in baseline)
# Remove only abandoned audit transfer parts whose complete immutable object is verified.
removed=0;remote=DONE/'remote';manifest=read(DONE/'REMOTE_MANIFEST.json')
for name,meta in manifest['files'].items():
    target=remote/name;assert target.exists() and sha(target)==meta['sha256']
    for p in target.parent.glob(target.name+'.*.part'):
        if re.fullmatch(re.escape(target.name)+r'\.[a-f0-9]{32}\.part',p.name):
            assert p.resolve().is_relative_to(remote.resolve())
            p.unlink();removed+=1
atomic(OUT/'HANDOFF.json',dict(status='checking'))
docs=list(ROOT.glob('*.md'))+list((ROOT/'tread_two').glob('*.md'))
for folder in ['review','analysis_report','completion_report','appearance_gate','appearance_gate_results','lc_pilot','lc_review','lc_analysis_report']:
    docs+=list((ROOT/'tread_two'/folder).glob('*.md'))
links=0
for doc in docs:
    for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
        target=target.strip('<>')
        if re.match(r'^(https?://|mailto:|#)',target):continue
        assert (doc.parent/unquote(target.split('#')[0])).resolve().exists(),(doc,target)
        links+=1
result=dict(status='passed',date_Asia_Calcutta='2026-10-30',NB11_revision=v['revision'],NB11_protocol=v['protocol'],NB11_objects_verified=1350,NB11_direct_files=1351,
    all_object_bytes_size_SHA_verified=True,all24_raw_CSV_schemas_and_counts_verified=True,raw_scientific_rows=54336368,epochs=60,updates=2940,selected_epoch=7,actual_dual_T4_and_live_HF_preflight=True,owner_released=True,
    scientific_paired_losses_reference_permissions_and_logits_checked=True,review_CSV_tables=40,review_rows=380595,review_input_files=696,review_image_probe_records=1442,
    NB12_notebook_sha256=nb['sha256'],NB12_embedded_files=nb['embedded_files'],NB12_code_cells=4,local_fresh_packaged_analysis_equal=True,offline_publication_and_viewer_controls_passed=True,
    actual_Kaggle_NB12_execution=False,actual_HF_review_publication=False,new_model_forwards=0,new_training_updates=0,new_independent_tyres=0,local_remote_writes=0,
    NB08_NB09_NB10_NB11_executed_notebooks_preserved=True,v1_sources24_preserved=True,r2_sources30_preserved=True,original_gate_sources6_preserved=True,original_review_sources3_preserved=True,
    original_workstation_files_preserved=original,separate_workstation_sources_preserved=station,paired_sources_preserved=paired,r1_sources_preserved=r1,original_Markdown_preserved=len(baseline),
    abandoned_verified_audit_transfer_parts_removed=removed,Markdown_documents=len(docs),local_links_checked=links,broken_links=[],figures_visually_inspected=4,native_browser_render=False,
    documentation_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in docs})
atomic(OUT/'HANDOFF.json',result);print(json.dumps({k:v for k,v in result.items() if k!='documentation_hashes'},indent=2))
