"""Verify all Phase 2 LabelMe inputs, preserve them, and render review evidence."""
import collections
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'review'/'label_audit'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()


def mask_shapes(doc, label):
    im = Image.new('L', (doc['imageWidth'], doc['imageHeight']))
    draw = ImageDraw.Draw(im)
    for s in doc['shapes']:
        if s['label'] == label:
            draw.polygon([tuple(p) for p in s['points']], fill=255)
    return np.asarray(im)>0


def crossings(points):
    """Count proper intersections of nonadjacent edges; touching vertices reported separately."""
    p = np.asarray(points, float)
    count = 0
    def cross(a,b): return a[...,0]*b[...,1]-a[...,1]*b[...,0]
    for i in range(len(p)):
        js = np.arange(i+2, len(p))
        if i == 0: js = js[js != len(p)-1]
        a,b = p[i],p[(i+1)%len(p)]
        c,d = p[js],p[(js+1)%len(p)]
        ab,cd=b-a,d-c
        count += int(np.sum((cross(ab,c-a)*cross(ab,d-a)<-1e-8)&(cross(cd,a-c)*cross(cd,b-c)<-1e-8)))
    return count


def guide_points(tread, ignore):
    h,w=tread.shape
    # Preserve the exact normalized guide definition of the Phase 1 1536px labels.
    guides=[round(y/1535*(h-1)) for y in (384,768,1151)]
    points=[]
    for level,y in zip(('upper','middle','lower'), guides):
        xs=np.flatnonzero(tread[y] & ~ignore[y])
        runs=0 if not len(xs) else 1+int(np.sum(np.diff(xs)>1))
        for side in ('left','right'):
            if not len(xs): state,x='not_intersecting',None
            elif runs>1: state,x='ambiguous_multiple_intervals',None
            else:
                x=int(xs[0 if side=='left' else -1])
                if x in (0,w-1): state,x='clipped',None
                elif ignore[y,max(0,x-1):min(w,x+2)].any(): state,x='ambiguous_ignore',None
                else: state='visible_proposal'
            points.append(dict(name=f'{side}_{level}',x=x,y=y,state=state,human_accepted=False))
    return points


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    frames=json.loads((ROOT/'manifests/phase2_frames.json').read_text())['frames']
    expected={r['frame_id'] for r in frames}
    actual=list((ROOT/'annotation_work').glob('*/*.json'))
    extras=[str(p.relative_to(ROOT)) for p in actual if p.stem not in expected]
    records=[]
    for i,r in enumerate(frames):
        ann=ROOT/'annotation_work'/r['video_id']/(r['frame_id']+'.json')
        errors,warnings=[],[]
        rec=dict(frame_id=r['frame_id'],video_id=r['video_id'],annotation_path=ann.relative_to(ROOT).as_posix())
        if not ann.exists():
            rec.update(errors=['missing_annotation'],warnings=[]);records.append(rec);continue
        doc=json.loads(ann.read_text(encoding='utf-8'))
        image=ROOT/r['relative_path']
        assert sha(image)==r['png_sha256'], f'Frame changed: {image}'
        if (doc.get('imageWidth'),doc.get('imageHeight')) != (r['width'],r['height']): errors.append('dimension_mismatch')
        linked=(ann.parent/doc.get('imagePath','')).resolve()
        if linked != image.resolve(): errors.append('image_path_mismatch')
        if doc.get('imageData') is not None: warnings.append('embedded_image_present')
        labels=collections.Counter(s.get('label') for s in doc.get('shapes',[]))
        for s in doc.get('shapes',[]):
            if s.get('label') not in ('tyre','tread','ignore') or s.get('shape_type')!='polygon':
                errors.append('unexpected_shape_or_label');continue
            p=np.asarray(s.get('points',[]),float)
            if p.ndim!=2 or p.shape[1]!=2 or len(p)<3 or not np.isfinite(p).all():
                errors.append('invalid_vertices');continue
            # LabelMe polygon coordinates can lie on the outer canvas edge W/H.
            # Pixel indices end at W-1/H-1; rasterization naturally clips the boundary.
            if (p[:,0]<0).any() or (p[:,0]>r['width']+1e-8).any() or (p[:,1]<0).any() or (p[:,1]>r['height']+1e-8).any(): errors.append('out_of_bounds')
            elif (p[:,0]>r['width']-1).any() or (p[:,1]>r['height']-1).any(): warnings.append('canvas_edge_coordinates_raster_clipped')
            if cv2.contourArea(p.astype(np.float32))<1: errors.append('zero_area_polygon')
            count=crossings(p)
            if count: warnings.append(f'{s["label"]}_proper_self_intersections:{count}')
        flags=doc.get('flags',{})
        if not flags.get('reviewed'): warnings.append('reviewed_checkbox_unset_user_declared_complete')
        if not flags.get('no_target') and not flags.get('unusable') and any(labels[k]<1 for k in ('tyre','tread')): errors.append('missing_required_region')
        if errors:
            rec.update(errors=sorted(set(errors)),warnings=warnings,annotation_sha256=sha(ann));records.append(rec);continue
        tyre,tread,ignore=[mask_shapes(doc,k) for k in ('tyre','tread','ignore')]
        outside=tread&~tyre&~ignore
        outside_count=int(outside.sum())
        outside_ratio=outside_count/max(1,int((tread&~ignore).sum()))
        if outside_count: warnings.append(f'tread_outside_tyre:{outside_count}')
        if not tyre.any() or not tread.any(): errors.append('empty_region')
        border=bool(tyre[0].any() or tyre[-1].any() or tyre[:,0].any() or tyre[:,-1].any())
        if border and not flags.get('clipped'): warnings.append('mask_touches_border_clipped_checkbox_unset')
        points=guide_points(tread,ignore)
        rec.update(annotation_sha256=sha(ann),flags=flags,labels=dict(labels),
            tyre_pixels=int(tyre.sum()),tread_pixels=int(tread.sum()),ignore_pixels=int(ignore.sum()),
            tread_outside_tyre_pixels=outside_count,tread_outside_tyre_fraction=outside_ratio,
            tread_tyre_ratio=float(tread.sum()/max(1,tyre.sum())),touches_image_border=border,
            points=points,errors=errors,warnings=warnings)
        records.append(rec)
        with Image.open(image) as im:
            thumb=ImageOps.contain(im.convert('RGB'),(276,480))
        arr=np.asarray(thumb).copy()
        for mask,color in [(tyre,(255,185,15)),(tread,(0,200,220))]:
            small=cv2.resize(mask.astype(np.uint8),thumb.size,interpolation=cv2.INTER_NEAREST)
            cs,_=cv2.findContours(small,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(arr,cs,-1,color,1,cv2.LINE_AA)
        tile=Image.new('RGB',(300,540),'white');tile.paste(Image.fromarray(arr),((300-thumb.width)//2,0))
        draw=ImageDraw.Draw(tile)
        draw.text((5,484),f'{r["video_id"]} {r["target_seconds"]:.1f}s | tyre amber / tread cyan',fill='black')
        draw.text((5,502),f'outside: {100*outside_ratio:.3f}% | border: {border}',fill='black')
        draw.text((5,520),f'{i+1}/152',fill='black')
        tile.save(OUT/(r['frame_id']+'.jpg'),quality=90)
    sheets=[]
    for page,start in enumerate(range(0,len(frames),12),1):
        sheet=Image.new('RGB',(1200,1620),'#eeeeee')
        for j,r in enumerate(frames[start:start+12]):
            p=OUT/(r['frame_id']+'.jpg')
            if p.exists():
                with Image.open(p) as im: sheet.paste(im,((j%4)*300,(j//4)*540))
        name=f'phase2_labels_{page:02d}.jpg';sheet.save(OUT/name,quality=94);sheets.append(name)
    summary=dict(annotation_files=len(actual),expected_frames=len(frames),extra_annotations=extras,
        frames_with_errors=sum(bool(r['errors']) for r in records),
        frames_with_outside_tread=sum(r.get('tread_outside_tyre_pixels',0)>0 for r in records),
        max_outside_fraction=max(r.get('tread_outside_tyre_fraction',0) for r in records),
        self_intersection_frames=[r['frame_id'] for r in records if any('self_intersection' in w for w in r['warnings'])],
        point_state_counts=dict(collections.Counter(p['state'] for r in records for p in r.get('points',[]))),
        sheets=sheets,user_completion_statement='User says all images labelled; original reviewed flags preserved unchanged',
        status='MECHANICAL_AUDIT_COMPLETE_VISUAL_REVIEW_PENDING')
    (ROOT/'manifests/phase2_label_audit.json').write_text(json.dumps(dict(summary=summary,records=records),indent=2),encoding='utf-8')
    (OUT/'phase2_labels.html').write_text('<!doctype html><meta charset="utf-8"><title>Phase 2 label audit</title><h1>All new labels: amber tyre, cyan tread</h1>'+''.join(f'<img style="max-width:100%;display:block" src="{s}">' for s in sheets),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
