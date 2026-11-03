"""Check current documentation, package identity and actual gallery behavior; no training/HF."""
import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'tread_two/geometry_first'
sys.path.insert(0,str(BASE/'runtime'))
from common import atomic, digest, file_meta, read, sha
delivery=read(BASE/'validation/DELIVERY.json')
notebook=read(BASE/'validation/NOTEBOOK.json')
assert delivery['notebook_sha256']==notebook['sha256']==sha(notebook['notebook'])
inventory=read(BASE/'runtime/FILES.json')
assert all(file_meta(BASE/'runtime'/n)==m for n,m in inventory.items())
fixture=BASE/('validation/fixture_delivery_'+digest(inventory)[:12])
node=subprocess.run(['node',str(ROOT/'analysis/test_geo_passport.cjs'),str(fixture/'Engineering_Passport.html')],
                    capture_output=True,text=True,check=False)
assert node.returncode==0,node.stderr
viewer=json.loads(node.stdout)
docs=[*ROOT.glob('*.md'),*(ROOT/'tread_two').glob('*.md'),*BASE.glob('*.md'),*(BASE/'plan').glob('*.md')]
links=0
for path in docs:
    text=path.read_text(encoding='utf-8')
    assert '\ufffd' not in text,path
    for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',text):
        target=target.strip('<>').split('#',1)[0]
        if not target or re.match(r'^[a-zA-Z]+:',target): continue
        assert (path.parent/unquote(target)).exists(),(str(path),target)
        links+=1
assert sha(BASE/'runtime/SOURCE_COMPONENTS.csv')==sha(ROOT/'data_gate_validation/nb07_publication/remote/csv/source_images.csv')
for path in [*BASE.joinpath('runtime').glob('*.py'),ROOT/'analysis/build_nb16_geo.py',ROOT/'analysis/document_nb16_geo.py']:
    ast.parse(path.read_text(encoding='utf-8'))
result=dict(status='passed',scope='final_local_handoff_not_live_Kaggle_HF',notebook_sha256=notebook['sha256'],
            current_source_files=len(inventory),Markdown_files_checked=len(docs),local_links_checked=links,
            broken_links=[],actual_gallery_script=viewer,source_component_registry_matches_audited_NB07=True,
            documentation_files={p.name:sha(p) for p in BASE.glob('*.md')},
            live_gate_pending=True,measured_fit_started=False)
atomic(BASE/'validation/HANDOFF.json',result)
print(json.dumps(result,indent=2))
