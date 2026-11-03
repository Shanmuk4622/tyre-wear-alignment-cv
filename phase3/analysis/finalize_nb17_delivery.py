"""Freeze matching source/notebook/acceptance, validate documents and prior preservation."""
import ast,base64,json,re,subprocess,sys
from pathlib import Path
from urllib.parse import unquote
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'tread_two/geo_pilot';VALID=ROOT/'tread_two/geo_validation'
sys.path.insert(0,str(SRC))
from common import atomic,digest,file_meta,read,sha
source=read(SRC/'FILES.json');assert all(file_meta(SRC/n)==m for n,m in source.items())
assert set(source)=={p.name for p in SRC.iterdir() if p.suffix in ('.py','.json','.csv') and p.name!='FILES.json'}
resume=read(VALID/'RESUME.json');assert resume['status']=='passed' and resume['source_inventory_sha256']==digest(source)
accounting=read(VALID/'LOSS_ACCOUNTING.json');assert accounting['status']=='passed' and accounting['source_inventory_sha256']==digest(source)
for name in ('TRANSACTIONS','TRANSPORT','SELECTED_BRANCH'):assert read(VALID/(name+'.json'))['status']=='passed'
note=read(VALID/'NOTEBOOK.json');assert sha(note['notebook'])==note['sha256']
nb=read(note['notebook']);code=[c for c in nb['cells'] if c['cell_type']=='code'];assert len(code)==3 and all(not c['outputs'] and c['execution_count'] is None for c in code)
import nbformat
nbformat.validate(nb)
payload=None
for cell in code:
    tree=ast.parse(''.join(cell['source']))
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PAYLOAD' for t in node.targets):payload=ast.literal_eval(node.value)
assert set(payload)==set(source)|{'FILES.json'} and len(payload)==29
for name,entry in payload.items():
    body=base64.b64decode(entry['base64'],validate=True)
    import hashlib
    assert hashlib.sha256(body).hexdigest()==entry['sha256']==sha(SRC/name)
    if name.endswith('.py'):ast.parse(body,filename=name)
old=Path(read(ROOT/'tread_two/nb16_completion/LATEST.json')['path']);verified=read(old/'VERIFICATION.json')
assert sha(ROOT/'notebooks/phase3_NB16_TREAD_TWO_GEO_Preparation_And_Contract_Gate.ipynb')==verified['executed_notebook_sha256']
nb16=ROOT/'tread_two/geometry_first/runtime';assert all(file_meta(nb16/n)==m for n,m in read(nb16/'FILES.json').items())
preserved={}
for directory,count in [('pilot',24),('lc_pilot',30),('context_pilot',32)]:
    folder=ROOT/'tread_two'/directory;items=read(folder/'FILES.json');assert len(items)==count and all(file_meta(folder/n)==m for n,m in items.items());preserved[directory]=count
for name,base,items in [('original_workstation',ROOT.parent,read(ROOT/'workstation_validation/ORIGINAL_BASELINE.json')),
                       ('phase3_workstation',ROOT,read(ROOT/'workstation_validation/DELIVERY.json')['source_hashes']),
                       ('paired_runtime',ROOT,read(ROOT/'revision2_validation/DELIVERY.json')['sources'])]:
    assert all(sha(base/n)==h for n,h in items.items());preserved[name]=len(items)
baseline=read(ROOT/'audit/markdown-baseline.json');assert all(sha(ROOT.parent/item['path'])==item['sha256'] for item in baseline)
for item in read(ROOT/'tread_two/gate_validation/PRESERVATION.json')['notebooks'].values():assert sha(item['path'])==item['sha256']
for n in (10,11,12,13,14,15):
    folder=Path(read(ROOT/f'tread_two/nb{n}_completion/LATEST.json')['path']);v=read(folder/'VERIFICATION.json')
    matching=[p for p in (ROOT/'notebooks').glob('*.ipynb') if sha(p)==v['executed_notebook_sha256']]
    assert matching,('Missing executed notebook',n)
docs=[*ROOT.glob('*.md'),*(ROOT/'tread_two').glob('*.md'),*SRC.glob('*.md'),*(ROOT/'tread_two/geometry_first').glob('*.md'),*(ROOT/'tread_two/geometry_first/plan').glob('*.md')]
links=0
for path in docs:
    text=path.read_text(encoding='utf-8');assert '\ufffd' not in text,path
    for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',text):
        target=target.strip('<>').split('#',1)[0]
        if not target or re.match(r'^[a-zA-Z]+:',target):continue
        assert (path.parent/unquote(target)).exists(),(str(path),target)
        links+=1
schema=read(VALID/'EXECUTABLE_SCHEMA.json');assert schema['CSV_families']==40 and schema['columns']==468
receipt=dict(status='passed',scope='local_source_notebook_math_resume_and_preservation_handoff',notebook_sha256=note['sha256'],notebook_bytes=note['bytes'],embedded_files=29,readable_source_files=len(source),
             source_inventory_sha256=digest(source),CSV_families=40,fields=468,fresh_process_scientific_shards=resume['restore']['scientific_shards_exact'],
             independent_loss_accounting=True,native_rotation_crash_checks=True,transport_replay_and_pinned_read_checks=True,selected_provenance_branch_test=True,
             actual_NB17_live_gate_pending=True,measured_updates_performed=0,remote_writes=0,NB16_preserved=True,prior_preserved=preserved,original_Markdown_preserved=len(baseline),Markdown_checked=len(docs),local_links_checked=links,broken_links=[],
             actual_Kaggle_hardlinks_unproven_locally=True,storage={k:v for k,v in resume['storage'].items() if k not in ('measured_CSV','typed_width_bounds')})
atomic(VALID/'DELIVERY.json',receipt)
print(json.dumps(receipt,indent=2))
