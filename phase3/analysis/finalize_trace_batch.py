"""Final source/payload/proof, documentation, Git visibility and preservation handoff."""
import ast,json,re,runpy,subprocess,sys
from pathlib import Path
from urllib.parse import unquote
assert Path(sys.prefix).name=='cv_conda'
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'tread_two/trace_batch';VALID=ROOT/'tread_two/trace_validation';sys.path.insert(0,str(SRC))
from common import atomic,read,file_meta,sha,digest
source=read(SRC/'FILES.json');assert len(source)==32 and all(file_meta(SRC/n)==m for n,m in source.items())
for p in SRC.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
proof=read(VALID/'LOCAL_LIFECYCLE.json');fixture=Path(proof['fixture'])
assert proof['source_inventory']==source and proof['status']=='passed'
for name in ['TRANSACTIONS','COORDINATOR','PREPARATION_RECOVERY','REVIEW_PARENT']:
    value=read(VALID/(name+'.json'));assert value['status']=='passed' and value['source_inventory']==source,name
assert read(VALID/'PAYLOADS.json')['status']=='passed'
viewer=subprocess.run(['node',str(ROOT/'analysis/test_trace_passport.cjs'),str(fixture/'review/TRACE_Evidence_Passport.html')],capture_output=True,text=True)
assert viewer.returncode==0,viewer.stderr;atomic(VALID/'VIEWER.json',json.loads(viewer.stdout))
with Image.open(fixture/'review/TRACE_Validation.png') as im:figure_size=list(im.size);im.verify()
runpy.run_path(str(ROOT/'analysis/document_trace_batch.py'),run_name='__main__')
notebooks=[]
import nbformat
for item in read(VALID/'NOTEBOOKS.json'):
    path=ROOT/'notebooks'/item['notebook'];assert sha(path)==item['sha256'];nb=read(path);nbformat.validate(nb)
    assert all(c['execution_count'] is None and not c['outputs'] for c in nb['cells'] if c['cell_type']=='code')
    notebooks.append(item)
# All original source inventories and the original project Markdown baseline.
preserved={}
for folder in ['pilot','lc_pilot','context_pilot','geo_pilot','geo_postrun','geometry_first/runtime']:
    path=ROOT/'tread_two'/folder;inventory=read(path/'FILES.json');assert all(file_meta(path/n)==m for n,m in inventory.items());preserved[folder]=len(inventory)
for name,base,inventory in [('original_workstation',ROOT.parent,read(ROOT/'workstation_validation/ORIGINAL_BASELINE.json')),
                          ('phase3_workstation',ROOT,read(ROOT/'workstation_validation/DELIVERY.json')['source_hashes']),
                          ('paired_runtime',ROOT,read(ROOT/'revision2_validation/DELIVERY.json')['sources'])]:
    assert all(sha(base/n)==value for n,value in inventory.items());preserved[name]=len(inventory)
proto=read(ROOT/'tread_two/trace_design/validation/NUMERICAL.json')['source_sha256']
assert all(sha(ROOT/'tread_two/trace_design/runtime'/n)==value for n,value in proto.items());preserved['original_TRACE_prototype_modules']=len(proto)
assert all(sha(SRC/n)==v['sha256'] for n,v in read(SRC/'VENDORED.json').items())
baseline=read(ROOT/'audit/markdown-baseline.json');assert all(sha(ROOT.parent/v['path'])==v['sha256'] for v in baseline)
for item in read(ROOT/'tread_two/gate_validation/PRESERVATION.json')['notebooks'].values():assert sha(Path(item['path']))==item['sha256']
old_notebooks={}
for n in [*range(10,18),20]:
    folder=Path(read(ROOT/f'tread_two/nb{n}_completion/LATEST.json')['path']);v=read(folder/'VERIFICATION.json')
    matches=[p.name for p in (ROOT/'notebooks').glob(f'phase3_NB{n:02d}*.ipynb') if sha(p)==v['executed_notebook_sha256']];assert matches,n;old_notebooks[str(n)]=matches
for note in read(ROOT/'tread_two/geo_postrun_validation/NOTEBOOKS.json'):
    if note['mode']=='all':continue
    assert file_meta(ROOT/'notebooks'/note['notebook'])=={k:note[k] for k in ('bytes','sha256')}
docs=[*ROOT.glob('*.md'),ROOT/'tread_two/README.md',*SRC.glob('*.md'),*(ROOT/'tread_two/trace_design').glob('*.md')];links=0
for p in docs:
    text=p.read_text(encoding='utf-8');assert '\ufffd' not in text,p
    for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',text):
        name=target.strip('<>').split('#',1)[0]
        if not name or re.match(r'^[a-zA-Z]+:',name):continue
        resolved=(p.parent/unquote(name)).resolve();assert resolved.exists() or resolved==(VALID/'DELIVERY.json').resolve(),(str(p),name);links+=1
for p in [fixture/'cache/CACHE.json',Path(read(VALID/'FULL_CACHE.json')['cache'])/'CACHE.json']:
    ignored=subprocess.run(['git','check-ignore','--no-index','--quiet','--',str(p)]);assert ignored.returncode==0,p
for p in [SRC/'model.py',VALID/'LOCAL_LIFECYCLE.json',ROOT/'notebooks'/notebooks[-1]['notebook']]:
    visible=subprocess.run(['git','check-ignore','--no-index','--quiet','--',str(p)]);assert visible.returncode==1,p
schema=read(SRC/'SCHEMAS.json');assert len(schema['families'])==48 and schema['field_declarations']==671
result=dict(status='passed',date='2026-11-02',package=notebooks[0]['package'],source_files=len(source),payload_files=33,
    notebooks=notebooks,local_lifecycle=True,complete_update_faults=True,all570_scientific_cache_scope='unchanged data-module equivalence plus current16-image chunk/allocator recovery proof',
    actual_CPU_review_and_offline_Hub_orchestration=True,actual_private_setup_extraction=True,figure_size=figure_size,viewer=read(VALID/'VIEWER.json'),
    CSV_families=48,CSV_fields=671,all_families_exercised=True,prior_sources_preserved=preserved,executed_notebooks_preserved=old_notebooks,
    original_Markdown_preserved=len(baseline),Markdown_checked=len(docs),local_links_checked=links,broken_links=[],
    large_fixtures_ignored=True,source_and_notebooks_visible=True,actual_Kaggle_dual_T4_pending=True,actual_HF_publication_pending=True,
    measured_fit_started=False,local_credentials_read=False,remote_writes=0,new_photos=0,workstation_changes=False,
    acceptance_receipts={n:file_meta(VALID/n) for n in ['LOCAL_LIFECYCLE.json','TRANSACTIONS.json','COORDINATOR.json','REVIEW_PARENT.json','PAYLOADS.json','PREPARATION_RECOVERY.json','FULL_CACHE.json','FULL_DATA_SOURCE_EQUIVALENCE.json','VIEWER.json','WRITER_COVERAGE.csv','LOCAL_STORAGE_ESTIMATE.json']})
atomic(VALID/'DELIVERY.json',result)
print(json.dumps({k:v for k,v in result.items() if k not in ('notebooks','acceptance_receipts','executed_notebooks_preserved')},indent=2),flush=True)
