"""Independent handoff checks for the local TRACE design/prototype deliverable."""
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote
assert Path(sys.prefix).name == 'cv_conda'
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'tread_two/trace_design'
REPORT = ROOT / 'tread_two/trace_report'
OUT = BASE / 'validation'


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for body in iter(lambda:stream.read(1024*1024),b''):h.update(body)
    return h.hexdigest()
def meta(path): return {'bytes':path.stat().st_size,'sha256':sha(path)}
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')

numeric=read(OUT/'NUMERICAL.json');geom=read(OUT/'GEOMETRY.json');audit=read(OUT/'ACTUAL_TARGETS.json')
assert numeric['status']==geom['status']==audit['status']=='passed'
assert all(sha(BASE/'runtime'/name)==digest for name,digest in numeric['source_sha256'].items())
assert sha(BASE/'runtime/frozen_geometry.py')==geom['wrapper_sha256']
assert sha(BASE/'runtime/targets.py')==audit['target_source_sha256']
assert numeric['checks']['trainable_parameters']==71022 and geom['frozen_parameters']==1399227
assert audit['roles']==['train'] and audit['images']==386 and audit['heldout_photo_reads']==0
assert read(BASE/'STATUS.json')['measured_training_started'] is False
assert all(meta(BASE/'runtime'/name)==value for name,value in read(BASE/'SOURCES.json')['prototype_files'].items())
construction=OUT/'actual_train'
assert all(meta(construction/name)==value for name,value in audit['artifacts'].items())

# Every cell/value in the newly created direct CSV must round-trip to native FP32.
target=np.load(construction/'full_targets.npy',mmap_mode='r',allow_pickle=False)
weight=np.load(construction/'full_target_weights.npy',mmap_mode='r',allow_pickle=False)
registry=pd.read_csv(construction/'target_registry.csv')
columns=['contrast','gradient_energy','orientation_axis','orientation_cross','relative_curvature','contrast_occupancy']
seen=np.zeros(386*32*4*4,dtype=bool);count=0
for frame in pd.read_csv(construction/'target_cells.csv',chunksize=65536,float_precision='round_trip'):
    i=frame.slot.to_numpy(dtype='int64');t=frame.tile.to_numpy(dtype='int64')
    y=frame.cell_y.to_numpy(dtype='int64');x=frame.cell_x.to_numpy(dtype='int64')
    positions=((i*32+t)*4+y)*4+x
    assert ((i>=0)&(i<386)&(t>=0)&(t<32)&(y>=0)&(y<4)&(x>=0)&(x<4)).all()
    assert len(np.unique(positions))==len(positions) and not seen[positions].any();seen[positions]=True
    expected=target[i,t,:,y,x]
    assert np.array_equal(frame[columns].to_numpy(dtype='float32'),expected)
    assert np.array_equal(frame.valid_pixels.to_numpy(),weight[i,t,0,y,x])
    assert np.array_equal(frame.image_id.to_numpy(),registry.image_id.to_numpy()[i])
    count+=len(frame)
assert count==197632 and seen.all() and np.isfinite(target).all() and int((weight>0).sum())==197201
controls=pd.read_csv(REPORT/'control_losses.csv')
assert len(controls)==1158 and controls.image_id.nunique()==386 and controls.groupby('image_id').control.nunique().eq(3).all()
assert controls.common_control_coverage.eq(1).all()
updates=pd.read_csv(OUT/'FIXTURE_UPDATES.csv');assert len(updates)==12
assert set(['local','masked','pair','variance','covariance','loss','pre_clip_norm']).issubset(updates.columns)
assert np.isfinite(updates[['local','masked','pair','variance','covariance','loss','pre_clip_norm']]).all().all()

schema=read(BASE/'SCHEMAS.json');assert len(schema['families'])==schema['families_count']==48
assert sum(len(f['columns']) for f in schema['families'].values())==schema['field_declarations']==669
assert all(f['writer_status']=='planned_not_implemented' for f in schema['families'].values())
for family in schema['families'].values():
    assert len({c['name'] for c in family['columns']})==len(family['columns'])
    assert all(c['dtype'] in ('utf8','int64','float32','float64','bool') for c in family['columns'])
budget=read(BASE/'BUDGET.json')
assert budget['one_prediction_body_train_bytes']==578027520
assert budget['all_three_token_bodies_train_bytes']==867041280
assert budget['forbidden_all_training_view_input_lc_bytes']==98650030080

viewer=subprocess.run(['node',str(ROOT/'analysis/test_trace_design_viewer.cjs'),str(REPORT/'TRACE_Design_Explorer.html')],capture_output=True,text=True)
assert viewer.returncode==0,viewer.stderr
viewer_result=json.loads(viewer.stdout);write(OUT/'VIEWER.json',viewer_result)
with Image.open(REPORT/'figures/TRACE_Fixed_Targets.png') as image:
    figure_size=list(image.size);image.verify()

# Preserve all original experiment inventories, supplied executions and workstations.
preserved={}
for folder in ['pilot','lc_pilot','context_pilot','geo_pilot','geo_postrun','geometry_first/runtime']:
    path=ROOT/'tread_two'/folder;inventory=read(path/'FILES.json')
    assert all(meta(path/name)==value for name,value in inventory.items());preserved[folder]=len(inventory)
for name,base,inventory in [('original_workstation',ROOT.parent,read(ROOT/'workstation_validation/ORIGINAL_BASELINE.json')),
                          ('phase3_workstation',ROOT,read(ROOT/'workstation_validation/DELIVERY.json')['source_hashes']),
                          ('paired_runtime',ROOT,read(ROOT/'revision2_validation/DELIVERY.json')['sources'])]:
    assert all(sha(base/n)==value for n,value in inventory.items());preserved[name]=len(inventory)
baseline=read(ROOT/'audit/markdown-baseline.json')
assert all(sha(ROOT.parent/item['path'])==item['sha256'] for item in baseline)
for item in read(ROOT/'tread_two/gate_validation/PRESERVATION.json')['notebooks'].values():assert sha(Path(item['path']))==item['sha256']
notebooks={}
for n in [*range(10,18),20]:
    folder=Path(read(ROOT/f'tread_two/nb{n}_completion/LATEST.json')['path']);verification=read(folder/'VERIFICATION.json')
    match=[p.name for p in (ROOT/'notebooks').glob(f'phase3_NB{n:02d}*.ipynb') if sha(p)==verification['executed_notebook_sha256']]
    assert match,n;notebooks[str(n)]=match
for note in read(ROOT/'tread_two/geo_postrun_validation/NOTEBOOKS.json'):
    if note['mode']=='all':continue
    assert meta(ROOT/'notebooks'/note['notebook'])=={k:note[k] for k in ('bytes','sha256')}

docs=[*ROOT.glob('*.md'),ROOT/'tread_two/README.md',*(BASE.glob('*.md'))]
links=0
for path in docs:
    text=path.read_text(encoding='utf-8');assert '\ufffd' not in text,path
    for raw in re.findall(r'\[[^\]]+\]\(([^)]+)\)',text):
        target_link=raw.strip('<>').split('#',1)[0]
        if not target_link or re.match(r'^[a-zA-Z]+:',target_link):continue
        resolved=(path.parent/unquote(target_link)).resolve()
        # This check writes its own receipt only after every check passes.
        assert resolved.exists() or resolved==(OUT/'DELIVERY.json').resolve(),(str(path),target_link)
        links+=1
sources=[*(BASE/'runtime').glob('*.py'),*(ROOT/'analysis').glob('*trace*.py')]
for path in sources:ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
visible=subprocess.run(['git','check-ignore','--no-index','--',str(construction/'target_cells.csv'),str(OUT/'actual_train_v0/targets_v0.py'),str(OUT/'fixture_state/resume.pt')],capture_output=True,text=True)
assert visible.returncode==0 and len(visible.stdout.splitlines())==3
for path in [REPORT/'TRACE_Design_Explorer.html',REPORT/'figures/TRACE_Fixed_Targets.png',BASE/'runtime/targets.py',OUT/'NUMERICAL.json']:
    check=subprocess.run(['git','check-ignore','--no-index','--quiet','--',str(path)])
    assert check.returncode==1,('Deliverable accidentally ignored',str(path))

result={'status':'passed','scope':'design_local_prototype_and_actual_train_target_audit_not_measured_fit',
        'prototype_modules':4,'syntax_checked_files':len(sources),'source_receipts_match':True,
        'target_cells_CSV_native_exact':count,'target_values_CSV_native_exact':count*6,'full_train_construction_SHA_verified':True,
        'prototype_checks':numeric['checks'],'geometry_equivalence':geom['status'],'target_audit':audit['status'],
        'planned_CSV_families':48,'planned_field_declarations':669,'measured_CSV_writer_coverage_claimed':False,
        'viewer':viewer_result,'figure_size':figure_size,'prior_frozen_source_inventories_preserved':preserved,
        'executed_notebooks_preserved':notebooks,'original_Markdown_preserved':len(baseline),
        'Markdown_checked':len(docs),'local_links_checked':links,'broken_links':[],
        'Kaggle_notebook_delivery':False,'dual_T4_live_HF_proof':False,'new_measured_training':False,
        'local_credentials_read':False,'remote_writes':0,'new_photos':0,'workstation_changes':False,
        'large_fixtures_ignored':True,'curated_deliverables_visible':True,
        'deliverables':{str(p.relative_to(ROOT)):meta(p) for p in [*BASE.glob('*.md'),*BASE.glob('*.json'),*BASE.glob('runtime/*.py'),REPORT/'TRACE_Design_Explorer.html',REPORT/'figures/TRACE_Fixed_Targets.png']}}
write(OUT/'DELIVERY.json',result)
print(json.dumps({k:v for k,v in result.items() if k not in ('deliverables','prototype_checks','executed_notebooks_preserved')},indent=2),flush=True)
