"""Read-only NB23 publication audit, independent COCO-RLE boundary reconstruction."""
import json
import hashlib
from pathlib import Path
import sys
import requests
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tyrelib'))
import s9_geometry as g
REV='84bcbdd61b39b9dfccd2461af86a5de562995d38'
KEY='929ae395f3521f988680e8c4019559e34241f2b1cd9f998ff48605a3afebc20b'
PREFIX=f's9/{g.VERSION}/{KEY}'
OUT=ROOT/'outputs/s9_geometry_completion';OUT.mkdir(exist_ok=True)
def get(name):
    r=requests.get(f'https://huggingface.co/datasets/{g.p.REPO}/resolve/{REV}/{PREFIX}/{name}',timeout=45)
    r.raise_for_status();assert len(r.content)<2*1024**2
    (OUT/name).write_bytes(r.content)
    return json.loads(r.content) if name.endswith('.json') else r.content
def decode(rle):
    # COCO compressed signed 5-bit counts, with second-order delta after index 2.
    s=rle['counts'];counts=[];i=0
    while i<len(s):
        x=0;k=0
        while True:
            c=ord(s[i])-48;i+=1;x|=(c&31)<<(5*k);k+=1
            if not c&32:break
        if c&16:x|=-1<<(5*k)
        if len(counts)>2:x+=counts[-2]
        assert x>=0;counts.append(x)
    assert sum(counts)==np.prod(rle['size'])
    return np.repeat(np.arange(len(counts))%2,counts).astype(bool).reshape(rle['size'],order='F')
contract=get('CONTRACT.json');assert g.p.digest(g.p.canonical(contract))==KEY
status=get('STATUS.json');assert status['status']=='complete' and status['images_processed']==12
points=get('POINTS.json');summary=get('SUMMARY.json');manifest=get('MANIFEST.json')
ann=get('ANNOTATIONS.json');provenance=get('PROVENANCE.json')
assert g.p.digest(get('s9_geometry.py'))==contract['source_sha256']
assert g.p.digest(get('s9_pilot.py'))==contract['pilot_validator_sha256']
source_ann=g.fetch(f's9/{g.p.VERSION}/{g.PACKAGE}/reviews/{g.ANNOTATION}/ANNOTATIONS.json',ROOT/'outputs/s9_geometry_validation/source_cache')
assert g.p.digest(source_ann)==contract['annotation_sha256'] and json.loads(source_ann)==ann
assert len(points)==72 and len({(r['pilot_id'],r['point']) for r in points})==72
cache=ROOT/'outputs/s9_geometry_validation/source_cache'
for path,sha in provenance.items():assert g.p.digest(g.fetch(path,cache))==sha
labels={a['pilot_id']:a for a in ann['annotations']}
for ref in manifest['images']:
    path=g.S5+f"/runs/segformer_b0-f{ref['fold']}-s1/predictions/{ref['image_id']}.json"
    rec=json.loads(g.fetch(path,cache));masks=[d['mask'] for d in rec['detections'] if d['label']==1]
    mask=decode(masks[0])
    for row in [r for r in points if r['pilot_id']==ref['pilot_id']]:
        xs=np.flatnonzero(mask[row['y']]);x=int(xs[0] if row['point'].startswith('left') else xs[-1]) if len(xs) else None
        state='empty_row' if x is None else 'frame_clipped' if x in (0,ref['width']-1) else 'visible'
        if state!='visible':x=None
        assert row['predicted_x']==x and row['prediction_state']==state
        a=labels[ref['pilot_id']]['points'][row['point']]
        assert row['label_x']==a.get('x') and row['label_state']==a['state']
        if ref['pilot_id']=='P06': assert not row['scored'] and not row['eligible']
        else:
            assert row['scored'] and row['eligible']
            assert row['error_px']==abs(x-a['x'])
            assert np.isclose(row['error_width_fraction'],abs(x-a['x'])/ref['width'])
assert g.summary(points)=={k:summary[k] for k in g.summary(points)}
for pid,s in summary['per_image'].items():assert g.summary([r for r in points if r['pilot_id']==pid])==s
tree=requests.get(f'https://huggingface.co/api/datasets/{g.p.REPO}/tree/{REV}/{PREFIX}',timeout=30);tree.raise_for_status()
assert {f'P{i:02d}.png' for i in range(1,13)}<={Path(x['path']).name for x in tree.json()}
report=dict(status='verified',revision=REV,prefix=PREFIX,summary=g.summary(points),
    evidence='All 72 boundaries independently reconstructed from pinned native RLE; source hashes and summaries match.',
    diagrams='12 PNGs present; pixel content not downloaded or visually audited',
    training_approved=False)
g.p.write_json(OUT/'AUDIT.json',report);print(json.dumps(report,indent=2))
