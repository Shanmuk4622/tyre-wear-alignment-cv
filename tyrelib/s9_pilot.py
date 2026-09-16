"""Small, lossless S9 annotation feasibility pilot. No model training or health inference."""
import base64
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import zipfile
from PIL import Image

VERSION = 's9-input-pilot-r1'
REPO = 'Shanmuk4622/tyre-wear-study'
MAX_BYTES = 20 * 1024 * 1024
POINTS = ['left_upper', 'right_upper', 'left_middle', 'right_middle', 'left_lower', 'right_lower']
STATES = ['visible', 'occluded', 'outside_frame', 'uncertain']
TRIAGE = ['no_visible_issue', 'visible_issue', 'unassessable']

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_bytes(canonical(value)); tmp.replace(path)

def discover(root='/kaggle/input'):
    candidates = sorted(Path(root).glob('**/manifests/clean_manifest.csv'))
    if len(candidates) != 1:
        raise ValueError('Attach ONE Tire Dataset Prepared package, or set DATA_ROOT to its FINAL directory.')
    return candidates[0].parent.parent

def prepare(data_root, output, template):
    data_root, output = Path(data_root), Path(output)
    with (data_root/'manifests/clean_manifest.csv').open(encoding='utf-8-sig', newline='') as f:
        all_rows = list(csv.DictReader(f))
    if len(all_rows) != 418 or len({r['image_id'] for r in all_rows}) != 418:
        raise ValueError('Expected the frozen 418 unique clean originals.')
    groups = sorted({r['session_group'] for r in all_rows})
    if len(groups) != 12: raise ValueError('Expected 12 capture sessions, not a new dataset.')
    chosen = []
    for group in groups:
        rr = sorted([r for r in all_rows if r['session_group'] == group], key=lambda r: r['image_id'])
        chosen.append(rr[len(rr)//2])  # Fixed median-ID representative, not selected by model performance.
    records, embedded = [], []
    for i, r in enumerate(chosen):
        p = (data_root/r['relative_path']).resolve()
        if not p.is_relative_to(data_root.resolve()): raise ValueError('Image path escapes data root')
        raw = p.read_bytes()
        if digest(raw) != r['file_sha256']: raise ValueError(f'Image hash mismatch: pilot {i+1}')
        with Image.open(p) as im:
            if im.format != 'JPEG' or im.size != (int(r['width']), int(r['height'])):
                raise ValueError('Expected unchanged native JPEG dimensions')
            width, height = im.size
        item = dict(pilot_id=f'P{i+1:02d}', image_id=r['image_id'], session=r['session_group'],
                    fold=int(r['fold_id']), original_relative_path=r['relative_path'],
                    image_sha256=digest(raw), width=width, height=height,
                    guide_y=[round((height-1)*f) for f in (.25,.5,.75)],
                    transform='identity; native pixels; no crop, resize or recompression')
        records.append(item)
        embedded.append({k:item[k] for k in ['pilot_id','image_sha256','width','height','guide_y']}
                        | {'image':'data:image/jpeg;base64,'+base64.b64encode(raw).decode()})
    template = Path(template).read_text(encoding='utf-8')
    protocol = dict(version=VERSION, manifest_sha256=digest((data_root/'manifests/clean_manifest.csv').read_bytes()),
                    source_sha256=digest(Path(__file__).read_bytes()), template_sha256=digest(template.encode()),
                    points=POINTS, images=records, count=12,
                    purpose='Feasibility only; proposed image-plane tread-boundary labels; no training approval',
                    healthy_reference_eligible=False, human_review_required=True)
    package_id = digest(canonical(protocol))
    protocol['package_id'] = package_id
    dest = output/package_id; dest.mkdir(parents=True, exist_ok=True)
    payload = dict(package_id=package_id, version=VERSION, points=POINTS, images=embedded)
    page = template.replace('__PILOT_DATA__', json.dumps(payload).replace('<','\\u003c'))
    if len(page.encode()) > MAX_BYTES:
        raise ValueError('Lossless 12-image page exceeds 20 MiB: stopped without reducing detail. Request a smaller batch.')
    (dest/'ANNOTATE.html').write_text(page, encoding='utf-8')
    write_json(dest/'MANIFEST.json', protocol)
    zip_path = dest/'PILOT_12_IMAGES.zip'
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ['ANNOTATE.html','MANIFEST.json']:
            info=zipfile.ZipInfo(name,date_time=(2026,9,14,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,(dest/name).read_bytes())
    if zip_path.stat().st_size > MAX_BYTES: raise ValueError('Package exceeds 20 MiB; no upload allowed')
    write_json(dest/'STATUS.json', dict(version=VERSION, package_id=package_id,
        status='awaiting_pilot_annotation', image_count=12, zip_bytes=zip_path.stat().st_size,
        html_bytes=(dest/'ANNOTATE.html').stat().st_size, zip_sha256=digest(zip_path.read_bytes()),
        full_s9_complete=False, training_approved=False, healthy_reference_eligible=False))
    print(f'12 native images; ZIP {zip_path.stat().st_size/2**20:.2f} MiB. Open ANNOTATE.html after extraction.')
    return dest

def validate_annotations(value, manifest):
    if not isinstance(value, dict) or value.get('package_id') != manifest['package_id']:
        raise ValueError('Annotation package identity mismatch. Import the JSON into the matching page.')
    if value.get('version') != VERSION: raise ValueError('Annotation schema mismatch')
    labels = value.get('annotations')
    if not isinstance(labels, list) or len(labels) > 12: raise ValueError('Expected at most 12 annotation records')
    by_id = {r['pilot_id']:r for r in manifest['images']}; seen=set(); checked=[]
    for a in labels:
        if not isinstance(a,dict): raise ValueError('Invalid annotation entry')
        pid=a.get('pilot_id')
        if pid not in by_id or pid in seen: raise ValueError('Unknown or duplicated pilot ID')
        seen.add(pid); ref=by_id[pid]
        if a.get('image_sha256') != ref['image_sha256']: raise ValueError('Image identity mismatch')
        points=a.get('points',{})
        if not isinstance(points,dict) or set(points)-set(POINTS): raise ValueError('Unknown landmark names')
        complete=True; visible=0
        for j,name in enumerate(POINTS):
            p=points.get(name)
            if p is None: complete=False; continue
            if not isinstance(p,dict) or p.get('state') not in STATES: raise ValueError('Invalid visibility state')
            if p['state']=='visible':
                x,y=p.get('x'),p.get('y')
                if any(type(v) not in (float,int) or not math.isfinite(v) for v in (x,y)):
                    raise ValueError('Visible points need finite coordinates')
                if not (0<=x<ref['width'] and 0<=y<ref['height']): raise ValueError('Point out of bounds')
                if y != ref['guide_y'][j//2]: raise ValueError('Point must be on its exact horizontal guide')
                visible+=1
            elif p.get('x') is not None or p.get('y') is not None:
                raise ValueError('Invisible point must not have guessed coordinates')
        for level in ['upper','middle','lower']:
            l,r=points.get('left_'+level,{}),points.get('right_'+level,{})
            if l.get('state')==r.get('state')=='visible' and l['x']>=r['x']:
                raise ValueError('Left/right points are crossed or equal')
        triage=a.get('visual_review')
        if triage is not None and triage not in TRIAGE: raise ValueError('Invalid visual review')
        if triage is None: complete=False
        note=a.get('note','')
        if not isinstance(note,str) or len(note)>2000: raise ValueError('Note exceeds 2000 characters')
        if triage=='visible_issue' and not note.strip(): complete=False
        evidence=a.get('independent_record','unknown')
        if evidence not in ['unknown','available']: raise ValueError('Invalid evidence availability')
        # A user assertion is recorded, NEVER automatically promoted to verified health.
        checked.append(dict(pilot_id=pid, image_id=ref['image_id'], complete=complete,
                            visible_points=visible, visual_review=triage, healthy_reference_eligible=False,
                            independent_record=evidence, note=note))
    return checked

def review(annotation_path, package, output):
    package, output=Path(package),Path(output)
    raw=Path(annotation_path).read_bytes()
    if len(raw)>1024*1024: raise ValueError('Upload the annotation-only JSON (under 1 MiB), not images or ZIP')
    value=json.loads(raw); manifest=json.loads((package/'MANIFEST.json').read_text())
    if digest(canonical({k:v for k,v in manifest.items() if k!='package_id'}))!=manifest.get('package_id'):
        raise ValueError('Manifest content hash mismatch')
    result=validate_annotations(value,manifest)
    dest=output/manifest['package_id']/'reviews'/digest(raw); dest.mkdir(parents=True,exist_ok=True)
    (dest/'ANNOTATIONS.json').write_bytes(raw)
    done=sum(r['complete'] for r in result)
    write_json(dest/'REVIEW.json',dict(status='needs_human_review' if done==12 else 'partial_annotation',
       complete_images=done, expected_images=12, records=result, package_id=manifest['package_id'],
       annotations_sha256=digest(raw), training_approved=False, healthy_reference_eligible=False,
       next_action='Send the JSON and review report to the assistant. Do not label more or train yet.'))
    print(f'Pilot completion: {done}/12. Mechanical checks only; human review still required. No training started.')
    return dest

def publish(folder, token, prefix):
    """One commit per major action; small immutable content. No claim/heartbeat writes."""
    from huggingface_hub import HfApi
    folder=Path(folder)
    allowed=['STATUS.json','MANIFEST.json','PILOT_12_IMAGES.zip','REVIEW.json','ANNOTATIONS.json',
             's9_pilot.py','pilot_template.html']
    files=[p for p in folder.iterdir() if p.name in allowed and p.is_file()]
    if sum(p.stat().st_size for p in files)>MAX_BYTES:
        raise ValueError('Upload exceeds 20 MiB limit')
    for attempt in range(5):
        try:
            result=HfApi(token=token).upload_folder(repo_id=REPO,repo_type='dataset',folder_path=str(folder),
                path_in_repo=prefix,allow_patterns=allowed,commit_message='S9 small annotation pilot; no model training')
            print('HF publication succeeded:',result.oid); return result.oid
        except Exception as exc:
            response=getattr(exc,'response',None); status=getattr(response,'status_code',None)
            if status not in (429,500,502,503,504) or attempt==4:
                raise RuntimeError('HF publication did not complete. Keep local outputs and retry this cell; no training was run.') from None
            hint=getattr(response,'headers',{}).get('Retry-After','')
            wait=max(10*2**attempt,float(hint) if str(hint).isdigit() else 0)
            print(f'HF backoff: {wait:.0f}s. Saved local outputs remain available.')
            until=time.monotonic()+wait
            while time.monotonic()<until: time.sleep(min(5,max(0,until-time.monotonic())))
