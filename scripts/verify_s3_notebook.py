"""CPU-only checks: native masks, prompts, blind gate, resume, generated notebook."""
import ast
import json
import tempfile
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tyrelib'))
from s3_mask_audit import *

def must_fail(call):
    try:
        call()
    except (ValueError,AssertionError):
        return
    raise AssertionError('Expected rejection')

with tempfile.TemporaryDirectory() as temp:
    p = Path(temp)/'input.json'
    valid = dict(imageWidth=10,imageHeight=10,shapes=[
        dict(label=n,shape_type='rectangle',points=[[1,1],[9,9]]) for n in ('tyre','tread')])
    p.write_text(json.dumps(valid))
    assert read_boxes(p,10,10)['tyre'] == [1,1,9,9]
    must_fail(lambda:read_boxes(p,11,10))
    valid['shapes'][0]['points'] = [[-1,0],[9,9]]
    p.write_text(json.dumps(valid)); must_fail(lambda:read_boxes(p,10,10))
    m = np.array([[0,1,2,3,4]],np.uint8)
    assert regions(m)['tyre'].sum()==4 and regions(m)['tread'].sum()==2
    assert scores(m>0,m>0)['iou']==1
    assert scores(m==9,m==9)['iou'] is None
    assert scores(m>0,m==9)['iou']==0
    must_fail(lambda:scores(m,m.T))
    rec = Path(temp)/'record.npz'
    save_record(rec,regions(m),dict(key='frozen'))
    assert load_record(rec,'frozen',m.shape)[0]['tyre'].sum()==4
    must_fail(lambda:load_record(rec,'changed',m.shape))
    must_fail(lambda:load_record(rec,'frozen',(4,4)))
    assert consistency_gate([])=='pending'
    assert consistency_gate([dict(region='tread',iou=.9)]*30)=='fail'
    assert consistency_gate([dict(region='tread',iou=.95)]*30)=='pass'
    # Real input coverage and canonical mask geometry.
    final=Path('D:/Dataset Download/Tire Dataset Prepared/FINAL')
    frame=pd.read_csv(final/'manifests/clean_manifest.csv')
    ids=selfcheck_ids(frame)
    assert ids==selfcheck_ids(frame.sample(frac=1,random_state=2))
    assert frame[frame.image_id.isin(ids)].groupby('fold_id').size().tolist()==[10,10,10]
    ann=final.parent/'annotations/clean/masks'
    for row in frame.itertuples():
        with Image.open(ann/f'{row.image_id}.png') as im:
            a=np.asarray(im); assert a.shape==(row.height,row.width); regions(a)
    print('PASS: all 418 actual manual masks and 30 deterministic blind IDs')
nb=json.loads((ROOT/'notebooks/NB11_S3_Manual_SAM2_Agreement.ipynb').read_text())
for i,c in enumerate(nb['cells']):
    if c['cell_type']=='code':
        ast.parse(''.join(c['source']),filename=f'NB11:{i}')
print('PASS: prompt rejection, canonical regions, empty masks, shape guard, resume fingerprints, strict blind gate, notebook syntax')
