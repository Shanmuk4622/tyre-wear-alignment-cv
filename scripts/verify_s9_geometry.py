"""Boundary, coverage and exclusion regression tests; no HF writes."""
import ast
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tyrelib'))
import s9_geometry as g
mask=np.zeros((20,30),bool);mask[:,3:26]=True
assert g.boundary(mask,'left',5)==(3,'visible')
assert g.boundary(mask,'right',5)==(25,'visible')
assert g.boundary(np.ones_like(mask),'right',5)==(None,'frame_clipped')
assert g.boundary(np.zeros_like(mask),'left',5)==(None,'empty_row')
ref=dict(pilot_id='P01',width=30,guide_y=[5,10,15])
ann=dict(points={n:dict(state='visible',x=3 if n.startswith('left') else 25) for n in g.p.POINTS})
r=g.rows_for(ref,ann,mask,True);assert g.summary(r)['coverage']==1 and g.summary(r)['median_error_px']==0
r=g.rows_for(ref,ann,np.ones_like(mask),True)
assert g.summary(r)['coverage']==0 and g.summary(r)['median_error_px'] is None
ref['pilot_id']='P06';assert not any(x['eligible'] for x in g.rows_for(ref,ann,mask,True))
ref['pilot_id']='P01';assert not any(x['eligible'] for x in g.rows_for(ref,ann,mask,False))
nb=json.loads((ROOT/'notebooks/NB23_S9_Geometry_Baseline.ipynb').read_text(encoding='utf-8'))
for c in nb['cells']:
    if c['cell_type']=='code':ast.parse(''.join(c['source']))
print('PASS: boundary, empty/clipped coverage, P06/incomplete exclusions and notebook syntax')
if '--public' in sys.argv:
    cache=ROOT/'outputs/s9_geometry_validation/source_cache'
    plan=json.loads(g.fetch(g.S5+'/protocol.json',cache))
    assert g.p.digest(g.p.canonical(plan))==g.PLAN
    m=json.loads(g.fetch(f's9/{g.p.VERSION}/{g.PACKAGE}/MANIFEST.json',cache))
    for ref in m['images']:
        f=ref['fold'];iid=ref['image_id'];split=plan['data']['splits'][str(f)]
        assert iid in split['validation'] and iid not in split['train']
        job=dict(model='segformer_b0',backend='semantic',fold=f,seed=1,run_id=f'segformer_b0-f{f}-s1')
        prefix=g.S5+'/runs/'+job['run_id']
        st=json.loads(g.fetch(prefix+'/STATUS.json',cache))
        assert st['epoch']==60 and st['evaluated'] and st['job']==job and st['plan_hash']==g.PLAN
        rec=json.loads(g.fetch(prefix+'/predictions/'+iid+'.json',cache))
        assert rec['job']==job and rec['plan_hash']==g.PLAN and rec['image_id']==iid
        assert (rec['width'],rec['height'])==(ref['width'],ref['height'])
        assert next(r for r in plan['data']['records'] if r['image_id']==iid)['image_sha256']==ref['image_sha256']
        masks=[d['mask'] for d in rec['detections'] if d['label']==1]
        assert len(masks)==1 and masks[0]['size']==[ref['height'],ref['width']]
    print('PASS: all 12 public prediction identities, native RLE metadata, epoch-60 evaluated jobs and split membership')
