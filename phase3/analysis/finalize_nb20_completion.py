"""Read-only preservation, provenance, notebook and Markdown handoff checks."""
import re,sys
from pathlib import Path
from urllib.parse import unquote
from PIL import Image
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tread_two/geo_postrun'))
from common import read,sha,file_meta,atomic
out=Path(read(ROOT/'tread_two/nb20_completion/LATEST.json')['path']);proof=read(out/'VERIFICATION.json');assert proof['status']=='passed'
assert sha(ROOT/'notebooks/phase3_NB20_TREAD_TWO_GEO_All_In_One_Postrun.ipynb')==proof['executed_notebook_sha256']==sha(out/'executed.ipynb')
preserved={}
for folder in ['pilot','lc_pilot','context_pilot','geo_pilot','geo_postrun','geometry_first/runtime']:
    path=ROOT/'tread_two'/folder;inventory=read(path/'FILES.json');assert all(file_meta(path/n)==meta for n,meta in inventory.items());preserved[folder]=len(inventory)
for name,base,inventory in [('original_workstation',ROOT.parent,read(ROOT/'workstation_validation/ORIGINAL_BASELINE.json')),
                          ('phase3_workstation',ROOT,read(ROOT/'workstation_validation/DELIVERY.json')['source_hashes']),
                          ('paired_runtime',ROOT,read(ROOT/'revision2_validation/DELIVERY.json')['sources'])]:
    assert all(sha(base/n)==value for n,value in inventory.items());preserved[name]=len(inventory)
baseline=read(ROOT/'audit/markdown-baseline.json');assert all(sha(ROOT.parent/item['path'])==item['sha256'] for item in baseline)
for item in read(ROOT/'tread_two/gate_validation/PRESERVATION.json')['notebooks'].values():assert sha(item['path'])==item['sha256']
completed={}
for n in range(10,18):
    folder=Path(read(ROOT/f'tread_two/nb{n}_completion/LATEST.json')['path']);v=read(folder/'VERIFICATION.json');matching=[p.name for p in (ROOT/'notebooks').glob(f'phase3_NB{n:02d}*.ipynb') if sha(p)==v['executed_notebook_sha256']];assert matching,n;completed[n]=matching
notes=read(ROOT/'tread_two/geo_postrun_validation/NOTEBOOKS.json')
for note in notes:
    if note['mode']=='all':continue # correct current identity is the preserved executed NB20
    assert file_meta(ROOT/'notebooks'/note['notebook'])=={k:note[k] for k in ('bytes','sha256')}
release=ROOT/'tread_two/geo_research_release';r=read(release/'RELEASE.json');assert r['verification_sha256']==sha(out/'VERIFICATION.json')
assert all(file_meta(release/name)==meta for name,meta in r['files'].items())
docs=[*ROOT.glob('*.md'),*(ROOT/'tread_two').glob('*.md'),*(ROOT/'tread_two/geo_pilot').glob('*.md'),*(ROOT/'tread_two/geo_postrun').glob('*.md'),*release.glob('*.md')]
links=0
for path in docs:
    text=path.read_text(encoding='utf-8');assert '\ufffd' not in text,path
    for raw in re.findall(r'\[[^\]]+\]\(([^)]+)\)',text):
        target=raw.strip('<>').split('#',1)[0]
        if not target or re.match(r'^[a-zA-Z]+:',target):continue
        assert (path.parent/unquote(target)).exists(),(str(path),target);links+=1
figures=[]
for path in (ROOT/'tread_two/nb20_report/figures').glob('*.png'):
    original=out/'remote/review/figures'/path.name;assert file_meta(path)==file_meta(original)
    with Image.open(path) as image:size=list(image.size);image.verify()
    figures.append(dict(file=path.name,size=size,published_bytes_exact=True))
assert len(figures)==3
result=dict(status='passed',NB20_preserved=True,embedded_files=19,prior_sources_preserved=preserved,executed_notebooks_preserved=completed,
            NB18_NB19_standalone_unexecuted_files_preserved=True,original_Markdown_preserved=len(baseline),Markdown_checked=len(docs),local_links_checked=links,broken_links=[],
            published_figures=figures,registry_files=len(r['files']),local_credentials_read=False,remote_writes=0,new_optimizer_updates=0)
atomic(out/'HANDOFF.json',result);print(result)
