"""Portable Phase 2 integrity/structure verifier. No training, downloads or uploads."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def verify(root):
    root=Path(root).resolve()
    checks={}
    for line in (root/'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        digest,name=line.split('  ',1)
        assert name not in checks,'Duplicate checksum path'
        checks[name]=digest
    for i,(name,want) in enumerate(checks.items()):
        p=(root/name).resolve()
        assert p.is_relative_to(root) and p.is_file(),name
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
        assert h.hexdigest()==want, f'Checksum mismatch: {name}'
        if i and i%500==0:print(f'Hash checked {i}/{len(checks)} files',flush=True)
    records=json.loads((root/'manifests/images.json').read_text())
    assert len(records)==570 and len({r['image_id'] for r in records})==570
    assert len({r['image_sha256'] for r in records})==570
    counts=collections.Counter(r['domain'] for r in records)
    assert counts=={'original_photo':418,'new_video_frame':152}
    conflict_count=0
    for r in records:
        assert checks[r['image_path']]==r['image_sha256']
        assert checks[r['source_label_path']]==r['source_label_sha256']
        with Image.open(root/r['image_path']) as im:
            im.load();assert im.size==(r['width'],r['height'])
        masks=[]
        for k in ('tyre','tread','ignore'):
            with Image.open(root/r[f'{k}_mask']) as im:
                a=np.array(im);assert a.shape==(r['height'],r['width']) and set(np.unique(a)).issubset({0,255})
                masks.append(a>0)
        tyre,tread,ignore=masks
        assert tyre.any() and tread.any()
        conflicts=tread&~tyre
        assert not (conflicts&~ignore).any(),f'Unmasked semantic conflict: {r["image_id"]}'
        assert int(conflicts.sum())==r['annotation_conflict_pixels']
        conflict_count+=int(conflicts.sum())
        assert r['split']=='UNASSIGNED','This source release does not contain a frozen training split'
    points=json.loads((root/'geometry/original_human_points.json').read_text())
    proposals=json.loads((root/'geometry/new_point_proposals.json').read_text())
    assert len(points)==120 and len(proposals)==152
    assert all(len(x['points'])==6 for x in points+proposals)
    assert all(not p['human_accepted'] for row in proposals for p in row['points'])
    status=json.loads((root/'splits/phase2_split_status.json').read_text())
    assert not status['training_allowed']
    result=dict(integrity='PASS',checksummed_files=len(checks),image_and_mask_sets=570,
        domains=dict(counts),conflict_pixels_explicitly_ignored=conflict_count,
        source_package_ready=True,training_allowed=False,
        next_steps=['Resolve exact original/video tyre identities and freeze splits',
                    'Resolve label conflicts for YOLO or other trainers without ignore support',
                    'Review derived geometry proposals','Run the forthcoming GPU/resume preflight'])
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',nargs='?',default=str(Path(__file__).resolve().parent))
    verify(p.parse_args().root)
