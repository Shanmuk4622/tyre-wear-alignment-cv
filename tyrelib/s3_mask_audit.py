"""Pure S3 input/metric/resume helpers, embedded in NB11. No model or HF imports."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

SAM_COMMIT = '2b90b9f5ceec907a1c18123530e92e794ad901a4'
MODEL_REVISION = 'ee5bba1d82bb8749febdf90f45e84b687142ba03'
MODEL_ID = 'facebook/sam2.1-hiera-small'

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def signature(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def selfcheck_ids(frame):
    # Fixed before seeing any mask or score; ten images in each original fold.
    ids = []
    for fold, group in frame.groupby('fold_id', sort=True):
        ids += sorted(group.image_id, key=lambda x: signature(['s3-blind-4622', x]))[:10]
    assert len(ids) == 30 and len(set(ids)) == 30
    return ids

def regions(mask):
    if mask.ndim != 2 or not np.isin(mask, [0,1,2,3,4]).all():
        raise ValueError('Expected a native-sized indexed mask with labels 0..4')
    return {'tyre': mask > 0, 'tread': (mask == 2) | (mask == 3)}

def scores(a, b):
    if a.shape != b.shape:
        raise ValueError('Mask dimensions differ; do not silently resize annotations')
    a, b = a.astype(bool), b.astype(bool)
    intersection, union = (a & b).sum(), (a | b).sum()
    total = a.sum() + b.sum()
    return dict(iou=float(intersection/union) if union else None,
                dice=float(2*intersection/total) if total else None,
                reference_pixels=int(a.sum()), prediction_pixels=int(b.sum()))

def read_boxes(path, width, height):
    d = json.loads(Path(path).read_text())
    if (d['imageWidth'], d['imageHeight']) != (width, height):
        raise ValueError(f'Wrong original image dimensions: {path}')
    out = {}
    for shape in d.get('shapes', []):
        name = shape.get('label', '').strip().lower()
        if name not in ('tyre', 'tread') or name in out or shape.get('shape_type') != 'rectangle':
            raise ValueError('Prompts require exactly two rectangles labelled tyre and tread; no polygons/duplicates')
        xy = np.asarray(shape['points'], dtype=float)
        if xy.shape != (2,2) or not np.isfinite(xy).all():
            raise ValueError('Invalid rectangle')
        lo, hi = xy.min(0), xy.max(0)
        if not (0 <= lo[0] < hi[0] <= width and 0 <= lo[1] < hi[1] <= height):
            raise ValueError('Rectangle outside image or empty')
        out[name] = [*lo.tolist(), *hi.tolist()]
    if set(out) != {'tyre','tread'}:
        raise ValueError('Both tyre and tread rectangles are required')
    return out

def read_blind(path, width, height):
    d = json.loads(Path(path).read_text())
    if (d['imageWidth'], d['imageHeight']) != (width,height):
        raise ValueError('Blind pass dimensions mismatch')
    labels = {'tyre':1, 'tread':2, 'marking':3, 'damage':4}
    shapes = d.get('shapes', [])
    if not {'tyre','tread'} <= {s.get('label','').strip().lower() for s in shapes}:
        raise ValueError('Blind pass must include tyre and tread polygons')
    im = Image.new('L',(width,height)); draw = ImageDraw.Draw(im)
    for s in shapes:
        if s.get('label','').strip().lower() not in labels or s.get('shape_type','polygon') != 'polygon':
            raise ValueError('Blind pass accepts only canonical label polygons')
    for label, value in labels.items():
        for s in shapes:
            if s['label'].strip().lower() == label:
                p = np.asarray(s['points'],dtype=float)
                if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all():
                    raise ValueError('Invalid blind polygon')
                if (p < 0).any() or (p[:,0] > width).any() or (p[:,1] > height).any():
                    raise ValueError('Blind polygon outside image')
                draw.polygon([tuple(x) for x in p], fill=value)
    return np.asarray(im)

def load_record(path, key, shape):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z['metadata'].item()))
        if meta['key'] != key:
            raise ValueError('Saved record protocol/input fingerprint mismatch')
        masks = {k: z[k].copy() for k in ('tyre','tread')}
        if any(m.shape != shape or m.dtype != np.bool_ for m in masks.values()):
            raise ValueError('Saved record mask invalid')
    return masks, meta

def save_record(path, masks, meta):
    path = Path(path)
    tmp = path.with_suffix('.partial')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **masks, metadata=json.dumps(meta, sort_keys=True))
    tmp.replace(path)

def consistency_gate(rows):
    tread = [r['iou'] for r in rows if r['region'] == 'tread']
    if len(tread) != 30 or any(v is None or not np.isfinite(v) for v in tread):
        return 'pending'
    return 'pass' if float(np.mean(tread)) > .90 else 'fail'
