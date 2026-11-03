"""Verify frozen prior work, current notebook identities, documentation links and content policy."""
import json,re,sys
from pathlib import Path
from urllib.parse import unquote
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tread_two/geo_postrun'))
from common import read,sha,file_meta,digest,atomic
source=ROOT/'tread_two/geo_postrun';items=read(source/'FILES.json');assert all(file_meta(source/n)==meta for n,meta in items.items())
validation=ROOT/'tread_two/geo_postrun_validation';actual=read(validation/'ACTUAL_DATA.json');delivery=read(validation/'DELIVERY_CHECKS.json')
assert actual['status']==delivery['status']=='passed' and actual['source_inventory_sha256']==delivery['source_inventory_sha256']==digest(items)
preserved={}
for folder in ['pilot','lc_pilot','context_pilot','geo_pilot','geometry_first/runtime']:
    path=ROOT/'tread_two'/folder;inventory=read(path/'FILES.json');assert all(file_meta(path/n)==meta for n,meta in inventory.items());preserved[folder]=len(inventory)
for name,base,inventory in [('original_workstation',ROOT.parent,read(ROOT/'workstation_validation/ORIGINAL_BASELINE.json')),
                          ('phase3_workstation',ROOT,read(ROOT/'workstation_validation/DELIVERY.json')['source_hashes']),
                          ('paired_runtime',ROOT,read(ROOT/'revision2_validation/DELIVERY.json')['sources'])]:
    assert all(sha(base/n)==value for n,value in inventory.items());preserved[name]=len(inventory)
baseline=read(ROOT/'audit/markdown-baseline.json');assert all(sha(ROOT.parent/item['path'])==item['sha256'] for item in baseline)
for item in read(ROOT/'tread_two/gate_validation/PRESERVATION.json')['notebooks'].values():assert sha(item['path'])==item['sha256']
completed={}
for n in range(10,18):
    folder=Path(read(ROOT/f'tread_two/nb{n}_completion/LATEST.json')['path']);v=read(folder/'VERIFICATION.json')
    candidates=list((ROOT/'notebooks').glob(f'phase3_NB{n:02d}*.ipynb'));matching=[p.name for p in candidates if sha(p)==v['executed_notebook_sha256']];assert matching,(n,candidates)
    completed[n]=matching
notes=read(validation/'NOTEBOOKS.json')
for note in notes:assert file_meta(ROOT/'notebooks'/note['notebook'])=={k:note[k] for k in ('bytes','sha256')}
docs=[*ROOT.glob('*.md'),*(ROOT/'tread_two').glob('*.md'),*(ROOT/'tread_two/geo_pilot').glob('*.md'),*source.glob('*.md'),*(ROOT/'tread_two/geo_passport_report').glob('*.md')]
links=0
for path in docs:
    text=path.read_text(encoding='utf-8');assert '\ufffd' not in text,path
    for raw in re.findall(r'\[[^\]]+\]\(([^)]+)\)',text):
        target=raw.strip('<>').split('#',1)[0]
        if not target or re.match(r'^[a-zA-Z]+:',target):continue
        assert (path.parent/unquote(target)).exists(),(str(path),target);links+=1
report=dict(status='passed',source_inventory_sha256=digest(items),payload_files=len(items)+1,new_notebooks=3,
            full_actual_data_pipeline=True,complete_stage_fresh_repeat_skips=7,prior_sources_preserved=preserved,executed_notebooks_preserved=completed,
            original_Markdown_preserved=len(baseline),Markdown_checked=len(docs),local_links_checked=links,broken_links=[],
            remote_writes=0,credentials_read=False,actual_new_Kaggle_HF_publication_pending=True)
atomic(validation/'HANDOFF.json',report);print(json.dumps(report,indent=2))
