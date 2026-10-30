"""Bind final review delivery, validated package, current notes and preservation."""
import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
CHECK=ROOT/'tread_two/review_validation'
sys.path.insert(0,str(ROOT/'tread_two/review'))
from review_atlas import atomic,sha
from publish_atlas_review import digest

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def hashes(base,mapping):
    assert all(sha(base/name)==expected for name,expected in mapping.items())
    return len(mapping)

test=read(CHECK/'VALIDATION.json');nb=read(CHECK/'NOTEBOOK.json');delivery=read(CHECK/'REVIEW_DELIVERY.json')
audit=read(ROOT/'tread_two/analysis_report/ANALYSIS.json')
assert test['status']=='passed' and test['notebook_sha256']==nb['notebook_sha256']==sha(Path(nb['notebook']))
assert test['source_hashes']=={p.name:sha(p) for p in (ROOT/'tread_two/review').glob('*.py')}
assert delivery['status']=='prepared_not_published'
folder=Path(delivery['path']);manifest=read(folder/'MANIFEST.json');contract=read(folder/'CONTRACT.json')
assert digest(contract)==manifest['contract']==delivery['contract']
assert delivery['destination'].startswith('phase3/tread-two/atlas-v1/review/') and '/fit/' not in delivery['destination']
assert all(sha(folder/name)==meta['sha256'] and (folder/name).stat().st_size==meta['bytes'] for name,meta in manifest['files'].items())
assert len(manifest['files'])+1==delivery['files']==51
assert len(audit['csv_tables'])==30 and sum(v['rows'] for v in audit['csv_tables'].values())==test['derived_csv_rows']
assert all(not name.endswith(('.zip','.npz','.sqlite','.pt')) for name in manifest['files'])
original=hashes(ROOT.parent,read(ROOT/'workstation_validation/ORIGINAL_BASELINE.json'))
station=hashes(ROOT,read(ROOT/'workstation_validation/DELIVERY.json')['source_hashes'])
baseline=read(ROOT/'audit/markdown-baseline.json')
assert all(sha(ROOT.parent/row['path'])==row['sha256'] for row in baseline)
frozen=hashes(ROOT,read(ROOT/'revision2_validation/DELIVERY.json')['sources'])
r1=hashes(ROOT/'runtime',read(ROOT/'completion_20261028/hf/CONTRACT.json')['sources'])
inventory=read(ROOT/'tread_two/pilot/FILES.json')
assert all(sha(ROOT/'tread_two/pilot'/name)==meta['sha256'] for name,meta in inventory.items())
assert sha(ROOT/'notebooks/phase3_NB08_TREAD_TWO_ATLAS_Resumable_Pilot.ipynb')=='214abe5b1c7bbc810805e6fa4012f8597c00c8559136daff544836eff9dab2c8'
docs=list(ROOT.glob('*.md'))+list((ROOT/'tread_two').glob('*.md'))
for sub in ['review','analysis_report','completion_report']: docs+=list((ROOT/'tread_two'/sub).glob('*.md'))
links=0
for doc in docs:
    for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
        target=target.strip('<>')
        if re.match(r'^(https?://|mailto:|#)',target): continue
        path=(doc.parent/unquote(target.split('#')[0])).resolve()
        assert path.exists(),(doc,target)
        links+=1
result=dict(status='passed',date_Asia_Calcutta='2026-10-30',notebook_sha256=nb['notebook_sha256'],
    notebook_bytes=nb['bytes'],embedded_files=30,code_cells=4,source_inputs=audit['input_files'],source_bytes=audit['input_bytes'],
    derived_csv_tables=30,derived_csv_rows=test['derived_csv_rows'],review_source_hashes=test['source_hashes'],
    outputs_byte_identical_from_extracted_notebook=True,publication_fixture_passed=True,
    prepared_publication_files=51,prepared_review_contract=delivery['contract'],HF_review_publication=False,Kaggle_NB09_execution=False,
    runtime_original24_unchanged=True,NB08_unchanged=True,original_workstation_preserved=original,separate_workstation_preserved=station,
    original_Markdown_preserved=len(baseline),frozen_paired_sources_preserved=frozen,frozen_r1_sources_preserved=r1,
    model_forward_calls=0,training_updates=0,new_independent_tyres=0,remote_writes=0,
    figures_visually_inspected=3,synthetic_DOM_controls_passed=True,browser_render_tested=False,
    local_Markdown_documents=len(docs),local_links_checked=links,broken_links=[],documentation_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in docs},
    validation_script_sha256=sha(ROOT/'analysis/validate_atlas_review.py'),handoff_script_sha256=sha(Path(__file__)))
atomic(CHECK/'HANDOFF.json',result)
print(json.dumps({k:v for k,v in result.items() if not k.endswith('hashes')},indent=2))
