"""CPU-only, provenance-pinned pilot comparison using already published predictions."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import requests
from PIL import Image, ImageDraw
import s9_pilot as p

VERSION = 's9-geometry-pilot-r1'
SOURCE = '05bf37c0f2067b0119ef057a2442d7759cf9ac51'
PACKAGE = '87933cacc9cc06234dc150fbba749e469d81e57850666d51a2f455a746d3fd96'
ANNOTATION = '4110bdcc413ff327d7ab2e6d107efac071008dc73b1bf04143a6c9a116516e18'
PLAN = '1f6694577253e0054f7a22df6ec52d30797063cf498b345bd71fe9b98bab93df'
S5 = f's5/s5-manual-2026-09-10-r1/{PLAN}'

def fetch(path, cache):
    """Public immutable sources only; bounded downloads; atomic cache."""
    cache=Path(cache); cache.mkdir(parents=True,exist_ok=True)
    dest=cache/(p.digest(path.encode())+'.json')
    if dest.exists(): return dest.read_bytes()
    for attempt in range(5):
        try:
            with requests.get(f'https://huggingface.co/datasets/{p.REPO}/resolve/{SOURCE}/{path}',
                              timeout=45,stream=True) as response:
                response.raise_for_status(); raw=bytearray()
                for chunk in response.iter_content(65536):
                    raw.extend(chunk)
                    if len(raw)>2*1024**2: raise ValueError('Source exceeds 2 MiB per-file cap')
            if sum(x.stat().st_size for x in cache.glob('*.json'))+len(raw)>20*1024**2:
                raise ValueError('Source cache exceeds 20 MiB budget')
            json.loads(raw)
            tmp=dest.with_suffix('.tmp'); tmp.write_bytes(raw); tmp.replace(dest)
            return bytes(raw)
        except requests.RequestException as exc:
            status=getattr(exc.response,'status_code',None)
            if status is not None and status not in (429,500,502,503,504): raise
            if attempt==4: raise
            hint=getattr(exc.response,'headers',{}).get('Retry-After','0')
            time.sleep(max(5*2**attempt,float(hint) if hint.isdigit() else 0))

def boundary(mask, side, y):
    xs=np.flatnonzero(mask[y])
    if not len(xs): return None,'empty_row'
    x=int(xs[0] if side=='left' else xs[-1])
    if x in (0,mask.shape[1]-1): return None,'frame_clipped'
    return x,'visible'

def rows_for(ref, annotation, mask, complete):
    rows=[]
    for i,name in enumerate(p.POINTS):
        point=annotation.get('points',{}).get(name,{})
        x,state=boundary(mask,name.split('_')[0],ref['guide_y'][i//2])
        reason=('P06_review_hold' if ref['pilot_id']=='P06' else
                'incomplete_annotation' if not complete else
                'label_not_visible' if point.get('state')!='visible' else '')
        eligible=not reason
        scored=eligible and state=='visible'
        rows.append(dict(pilot_id=ref['pilot_id'],point=name,y=ref['guide_y'][i//2],
            label_state=point.get('state','missing'),label_x=point.get('x'),predicted_x=x,
            prediction_state=state,eligible=eligible,scored=scored,exclusion=reason,
            error_px=abs(x-point['x']) if scored else None,
            error_width_fraction=abs(x-point['x'])/ref['width'] if scored else None))
    return rows

def summary(rows):
    eligible=[r for r in rows if r['eligible']]; scored=[r for r in rows if r['scored']]
    return dict(total_points=len(rows),eligible_visible_points=len(eligible),scored_points=len(scored),
        coverage=len(scored)/len(eligible) if eligible else None,
        median_error_px=float(np.median([r['error_px'] for r in scored])) if scored else None,
        mean_error_width_fraction=float(np.mean([r['error_width_fraction'] for r in scored])) if scored else None,
        rejected_eligible_points=len(eligible)-len(scored))

def publish(out, token):
    from huggingface_hub import HfApi
    out=Path(out)
    if sum(f.stat().st_size for f in out.rglob('*') if f.is_file())>20*1024**2:
        raise ValueError('Output upload exceeds 20 MiB')
    for attempt in range(5):
        try:
            result=HfApi(token=token).upload_folder(repo_id=p.REPO,repo_type='dataset',
                folder_path=str(out),path_in_repo=f's9/{VERSION}/{out.name}',
                commit_message='S9 saved-mask geometry pilot; no training')
            print('HF publication succeeded:',result.oid,flush=True); return result.oid
        except Exception as exc:
            status=getattr(getattr(exc,'response',None),'status_code',None)
            if status not in (None,429,500,502,503,504) or attempt==4: raise
            hint=getattr(getattr(exc,'response',None),'headers',{}).get('Retry-After','0')
            wait=max(10*2**attempt,float(hint) if hint.isdigit() else 0)
            print(f'Upload retry in {wait}s; local results are safe.',flush=True); time.sleep(wait)

def run(work, token=None, upload=True):
    from pycocotools import mask as coco
    work=Path(work); cache=work/'source_cache'
    provenance={}
    def read(path):
        raw=fetch(path,cache); provenance[path]=p.digest(raw); return json.loads(raw)
    base=f's9/{p.VERSION}/{PACKAGE}'
    manifest=read(base+'/MANIFEST.json')
    assert p.digest(p.canonical({k:v for k,v in manifest.items() if k!='package_id'}))==PACKAGE
    apath=f'{base}/reviews/{ANNOTATION}/ANNOTATIONS.json'
    annotation=read(apath); assert provenance[apath]==ANNOTATION
    validated=p.validate_annotations(annotation,manifest)
    labels={a['pilot_id']:a for a in annotation['annotations']}
    checks={a['pilot_id']:a for a in validated}
    plan=read(S5+'/protocol.json'); assert p.digest(p.canonical(plan))==PLAN
    contract=dict(version=VERSION,source_revision=SOURCE,annotation_sha256=ANNOTATION,
        plan_hash=PLAN,model='segformer_b0',seed=1,endpoint='saved_epoch60_predictions',
        source_sha256=p.digest(Path(__file__).read_bytes()),
        pilot_validator_sha256=p.digest(Path(p.__file__).read_bytes()),
        review_hold=['P06'],edge_rule='reject exact native frame edge',
        limitations=['pilot feasibility only; no independent new-tyre accuracy',
          'known fold 0/2 identity leakage caveat retained',
          'mask boundary and proposed tread transition may differ',
          'no physical angles, healthy certification or HRNet comparison'])
    out=work/'results'/p.digest(p.canonical(contract)); out.mkdir(parents=True,exist_ok=True)
    p.write_json(out/'CONTRACT.json',contract)
    p.write_json(out/'ANNOTATIONS.json',annotation)
    p.write_json(out/'MANIFEST.json',manifest)
    (out/'s9_geometry.py').write_bytes(Path(__file__).read_bytes())
    (out/'s9_pilot.py').write_bytes(Path(p.__file__).read_bytes())
    rows=[]; last_push=time.monotonic()
    try:
        for ref in manifest['images']:
            fold=ref['fold']; split=plan['data']['splits'][str(fold)]; iid=ref['image_id']
            assert iid in split['validation'] and iid not in split['train'], 'Not held out from this run'
            source_row=next(r for r in plan['data']['records'] if r['image_id']==iid)
            assert source_row['image_sha256']==ref['image_sha256']
            job=dict(model='segformer_b0',backend='semantic',fold=fold,seed=1,
                     run_id=f'segformer_b0-f{fold}-s1')
            prefix=S5+'/runs/'+job['run_id']
            status=read(prefix+'/STATUS.json')
            assert status.get('epoch')==60 and status.get('evaluated') is True, 'Expected evaluated final epoch 60'
            assert status['job']==job and status['plan_hash']==PLAN
            rec=read(prefix+'/predictions/'+iid+'.json')
            assert rec['job']==job and rec['plan_hash']==PLAN and rec['image_id']==iid
            assert (rec['height'],rec['width'])==(ref['height'],ref['width'])
            masks=[det['mask'] for det in rec['detections'] if det['label']==1]
            assert len(masks)<=1, 'Semantic output should contain at most one tread mask'
            mask=np.zeros((ref['height'],ref['width']),dtype=bool)
            if masks:
                assert masks[0]['size']==[ref['height'],ref['width']]
                mask=coco.decode(masks[0]).astype(bool)
            part=rows_for(ref,labels.get(ref['pilot_id'],{}),mask,checks.get(ref['pilot_id'],{}).get('complete',False))
            for row in part: row.update(run_id=job['run_id'],fold=fold,image_id=iid)
            rows.extend(part)
            picture=Image.fromarray((mask.astype('uint8')*150)).convert('RGB'); draw=ImageDraw.Draw(picture)
            for row in part:
                y=row['y']; draw.line((0,y,ref['width']-1,y),fill='gray')
                for x,color in [(row['label_x'],'cyan'),(row['predicted_x'],'yellow')]:
                    if x is not None: draw.ellipse((x-7,y-7,x+7,y+7),outline=color,width=3)
            picture.thumbnail((576,768)); picture.save(out/(ref['pilot_id']+'.png'))
            p.write_json(out/'POINTS.json',rows)
            p.write_json(out/'PROVENANCE.json',provenance)
            p.write_json(out/'STATUS.json',dict(status='partial',images_processed=len(rows)//6,training=False))
            print(f"{ref['pilot_id']} | {job['run_id']} | {summary(part)}",flush=True)
            if upload and time.monotonic()-last_push>=1800:
                publish(out,token); last_push=time.monotonic()
        result=summary(rows)
        result.update(per_image={r['pilot_id']:summary([x for x in rows if x['pilot_id']==r['pilot_id']]) for r in manifest['images']},
            decision='Review coverage and overlays; no automatic HRNet approval or rejection.',
            full_s9_complete=False,healthy_reference_eligible=False)
        p.write_json(out/'SUMMARY.json',result)
        p.write_json(out/'STATUS.json',dict(status='complete',images_processed=12,training=False,full_s9_complete=False))
        (out/'README.md').write_text('# Saved-mask geometry pilot\n\nCPU reuse of published SegFormer-B0 seed-1 native predictions. No new inference or training.\n\nPNG: grey is predicted tread mask, cyan human clicks, yellow accepted predicted boundary. P06 is diagnostic only and excluded. Empty/clipped predictions reduce coverage; they are not zero-error successes.\n\nRead SUMMARY.json, POINTS.json and CONTRACT.json together. Point errors are conditional on accepted predictions; do not interpret the mean without coverage. No automatic HRNet decision.\n',encoding='utf-8')
        if upload: publish(out,token)
        print(json.dumps(result,indent=2)); return out
    except BaseException:
        if upload:
            try: publish(out,token)
            except Exception as exc: print('Stop/error flush failed; retain local output and rerun:',type(exc).__name__)
        raise
