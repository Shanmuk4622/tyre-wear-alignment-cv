"""Verify extraction integrity, independent orientation decoding and local docs."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()


def main():
    data = json.loads((ROOT/'manifests/phase2_frames.json').read_text())
    rows = data['frames']
    assert len(rows) == 152 and len({r['frame_id'] for r in rows}) == 152
    total = 0
    for row in rows:
        p = (ROOT/row['relative_path']).resolve()
        assert p.is_relative_to((ROOT/'data/new_frames').resolve())
        assert sha(p) == row['png_sha256']
        with Image.open(p) as im:
            assert im.format == 'PNG' and im.size == (row['width'],row['height'])
            im.verify()
        total += p.stat().st_size
        assert row['split'] == 'UNASSIGNED' and row['label_status'] == 'unlabelled'
        assert -1e-8 <= row['timestamp_error_seconds'] < .05
    # Independent decoder path: let OpenCV honor orientation metadata itself.
    orientation_checks = []
    for video_id, count in [('video1',27),('video2',93),('video3',32)]:
        subset = [r for r in rows if r['video_id'] == video_id]
        assert len(subset) == count
        assert [r['target_seconds'] for r in subset] == [i/2 for i in range(count)]
        for row in [subset[0], subset[len(subset)//2], subset[-1]]:
            cap = cv2.VideoCapture(str(ROOT.parent/'Videos'/row['source_video']))
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, row['source_frame_index'])
            ok, reference = cap.read(); cap.release()
            assert ok
            actual = cv2.imread(str(ROOT/row['relative_path']))
            assert reference.shape == actual.shape and np.array_equal(reference, actual), row['frame_id']
            orientation_checks.append(row['frame_id'])
        subprocess.run([sys.executable,'-B',str(ROOT/'phase2_label_new_frames.py'),'--video',video_id,'--check'], check=True)
    for path in ROOT.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    broken = []
    for doc in ROOT.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)', doc.read_text(encoding='utf-8')):
            if '://' not in target and not (doc.parent/target.split('#')[0]).exists():
                broken.append([doc.name,target])
    assert not broken, broken
    identity = json.loads((ROOT/'manifests/phase2_video_identity.json').read_text())
    assert len({r['provisional_tyre_id'] for r in identity['videos']}) == 3
    assert all(r['mileage_proxy'] == 'mid_mileage_proxy' and r['original_session_id'] is None for r in identity['videos'])
    subprocess.run([sys.executable,'-B',str(ROOT/'phase2_protect_originals.py')], check=True)
    result = dict(status='PASS',verified_pngs=len(rows),bytes=total,
        independent_orientation_pixel_checks=orientation_checks,
        labelme_config_and_paths_checked=3,phase2_python_syntax='PASS',local_markdown_links='PASS',
        frames_manifest_sha256=sha(ROOT/'manifests/phase2_frames.json'),
        limits=['Not an interactive LabelMe save/reopen test','No annotation accuracy or model training tested',
                'Exact original tyre matching and splits still pending'])
    (ROOT/'manifests/phase2_preparation_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
