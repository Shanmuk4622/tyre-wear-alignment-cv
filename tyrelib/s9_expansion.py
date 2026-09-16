"""Bounded S9 geometry annotation expansion; no training or inferred health labels."""
import base64
import csv
import json
from pathlib import Path
import zipfile
from PIL import Image
import s9_pilot as p

VERSION='s9-geometry-labels-r2-all120'
TARGET=120
MAX_PACKAGE_BYTES=160*1024**2
def selection(rows, pilot):
    excluded={r['image_id'] for r in pilot['images']}
    pools={s:sorted([r for r in rows if r['session_group']==s and r['image_id'] not in excluded],
                   key=lambda r:p.digest(r['image_id'].encode())) for s in sorted({r['session_group'] for r in rows})}
    chosen=[]
    while len(chosen)<TARGET:
        before=len(chosen)
        for s,pool in pools.items():
            if pool and len(chosen)<TARGET: chosen.append(pool.pop(0))
        if len(chosen)==before:raise ValueError('Insufficient unique new images')
    assert len({r['image_id'] for r in chosen})==TARGET
    return chosen

def prepare(root, pilot, template, out):
    root=Path(root);manifest_path=root/'manifests/clean_manifest.csv'
    raw=manifest_path.read_bytes()
    assert p.digest(raw)==pilot['manifest_sha256'], 'Prepared dataset manifest differs from the pilot'
    rows=list(csv.DictReader(raw.decode('utf-8-sig').splitlines()))
    assert len(rows)==418 and len({r['image_id'] for r in rows})==418
    selected=selection(rows,pilot)
    allocation=[dict(image_id=r['image_id'],session=r['session_group'],
                     file_sha256=r['file_sha256']) for i,r in enumerate(selected)]
    plan=dict(version=VERSION,target_images=120,batch_size=120,allocation=allocation,
              pilot_package=pilot['package_id'],split_status='UNLOCKED_requires_physical_tyre_identity_review',
              purpose='six image-relative tread-transition points; not physical alignment',
              training_approved=False,healthy_reference_eligible=False)
    plan_hash=p.digest(p.canonical(plan))
    records=[];embedded=[]
    for i,r in enumerate(selected):
        path=(root/r['relative_path']).resolve()
        assert path.is_relative_to(root.resolve())
        image_raw=path.read_bytes();assert p.digest(image_raw)==r['file_sha256']
        with Image.open(path) as im:
            assert im.format=='JPEG';w,h=im.size
        assert (w,h)==(int(r['width']),int(r['height']))
        rec=dict(pilot_id=f'G{i+1:03d}',image_id=r['image_id'],
                 image_sha256=r['file_sha256'],width=w,height=h,guide_y=[round((h-1)*f) for f in (.25,.5,.75)],
                 session=r['session_group'],fold=int(r['fold_id']),original_relative_path=r['relative_path'])
        records.append(rec);embedded.append(rec|{'image':f'images/G{i+1:03d}.jpg'})
    protocol=dict(version=p.VERSION,expansion_version=VERSION,plan_hash=plan_hash,
        points=p.POINTS,images=records,count=120,training_approved=False,
        source_sha256=p.digest(Path(__file__).read_bytes()),template_sha256=p.digest(Path(template).read_bytes()))
    package=p.digest(p.canonical(protocol));protocol['package_id']=package
    dest=Path(out)/package;dest.mkdir(parents=True,exist_ok=True)
    html=Path(template).read_text(encoding='utf-8')
    # External native JPEGs keep the HTML small; only the current image is decoded.
    html=html.replace('12 images. One small pilot. No training yet.','All 120 images — one annotation project.')
    html=html.replace('This is a feasibility check, not a 418-image assignment.',
        'This is one 120-image geometry assignment. You can annotate all images without batch switching or intermediate approval. Save JSON to work across sittings.')
    html=html.replace('or attach that JSON to Kaggle and run NB22','or attach that JSON to Kaggle and run NB25')
    html=html.replace('12/12','120/120').replace("n+'/12 reviewed'","n+'/'+DATA.images.length+' reviewed'")
    html=html.replace('Math.min(11,index+1)','Math.min(DATA.images.length-1,index+1)')
    html=html.replace('v.annotations.length>12','v.annotations.length>DATA.images.length')
    html=html.replace('stop after 12 images.','all 120 images are available; save again whenever you pause.')
    html=html.replace('Stop there; wait for feedback before any larger batch or model training.',
        'Return your saved JSON when ready. Training follows final label and split review; no intermediate 12-image pause.')
    html=html.replace('S9 — 12-image guided pilot','S9 — all 120 geometry images')
    html=html.replace('<h3>Worked examples',
        '<p class="notice"><b>Quality rule:</b> An image edge is not a visible boundary. Select Outside frame when clipped. If you cannot distinguish tread from shoulder, select Uncertain; do not use the silhouette instead. Visible issue requires a short note. Check all six point states before saving. These are geometry labels, not defect masks or health labels.</p><h3>Worked examples')
    payload=dict(version=p.VERSION,package_id=package,points=p.POINTS,images=[{k:v for k,v in r.items() if k not in ('session','fold','image_id','original_relative_path')} for r in embedded])
    html=html.replace('__PILOT_DATA__',json.dumps(payload).replace('<','\\u003c'))
    assert len(html.encode())<=p.MAX_BYTES, 'Package too large: stopped without degrading originals'
    (dest/'ANNOTATE.html').write_text(html,encoding='utf-8')
    images=dest/'images';images.mkdir(exist_ok=True)
    for ref in records:
        raw_image=(root/ref['original_relative_path']).read_bytes()
        assert p.digest(raw_image)==ref['image_sha256']
        (images/(ref['pilot_id']+'.jpg')).write_bytes(raw_image)
    p.write_json(dest/'MANIFEST.json',protocol);p.write_json(dest/'PLAN.json',plan)
    # Same blank identity ledger in each batch, user supplies factual identity only.
    p.write_json(dest/'SESSION_IDENTITY.json',dict(instruction='Same physical tyre must use same anonymous ID across sessions. Leave unknown if unsure. Do not invent identities.',
        sessions=[dict(session=s,physical_tyre_id=None,record_note='') for s in sorted({r['session_group'] for r in rows})]))
    with zipfile.ZipFile(dest/'ALL_120_IMAGES.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['ANNOTATE.html','MANIFEST.json','SESSION_IDENTITY.json']:
            z.write(dest/name,arcname=name)
        for path in sorted(images.glob('*.jpg')):z.write(path,arcname='images/'+path.name)
    assert (dest/'ALL_120_IMAGES.zip').stat().st_size<=MAX_PACKAGE_BYTES
    p.write_json(dest/'STATUS.json',dict(status='awaiting_annotations',image_count=120,plan_hash=plan_hash,
        package_id=package,zip_bytes=(dest/'ALL_120_IMAGES.zip').stat().st_size,
        training_approved=False,full_s9_complete=False))
    return dest

def intake(annotation, manifest, out):
    annotation=Path(annotation);raw=annotation.read_bytes()
    assert len(raw)<=1024**2
    value=json.loads(raw)
    assert manifest.get('expansion_version')==VERSION
    assert p.digest(p.canonical({k:v for k,v in manifest.items() if k!='package_id'}))==manifest['package_id']
    records=validate(value,manifest)
    done=sum(r['complete'] for r in records)
    dest=Path(out)/p.digest(raw);dest.mkdir(parents=True,exist_ok=True)
    (dest/'ANNOTATIONS.json').write_bytes(raw)
    p.write_json(dest/'REVIEW.json',dict(status='needs_human_review' if done==120 else 'partial_annotation',
        complete_images=done,expected_images=120,records=records,package_id=manifest['package_id'],
        plan_hash=manifest['plan_hash'],annotations_sha256=p.digest(raw),training_approved=False))
    p.write_json(dest/'STATUS.json',dict(status='mechanical_intake_only',full_s9_complete=False,training_approved=False))
    print(f'{done}/120 complete. Progress saved; final human review and split lock still required.')
    return dest

def validate(value,manifest):
    labels=value.get('annotations')
    if not isinstance(labels,list) or len(labels)>120:raise ValueError('Expected at most 120 records')
    ids=[a.get('pilot_id') for a in labels if isinstance(a,dict)]
    if len(ids)!=len(labels) or len(set(ids))!=len(ids):raise ValueError('Invalid/duplicate image records')
    # Reuse strict coordinate/schema validation without altering old NB21/NB22 behaviour.
    result=[]
    for i in range(0,max(1,len(labels)),12):
        result.extend(p.validate_annotations(value|{'annotations':labels[i:i+12]},manifest))
    return result

def publish(folder, token, prefix):
    import time
    from huggingface_hub import HfApi
    allowed=['ALL_120_IMAGES.zip','MANIFEST.json','PLAN.json','SESSION_IDENTITY.json','STATUS.json',
             'ANNOTATIONS.json','REVIEW.json','s9_expansion.py']
    assert sum(f.stat().st_size for f in Path(folder).iterdir() if f.name in allowed)<=MAX_PACKAGE_BYTES+2*1024**2
    for attempt in range(5):
        try:
            r=HfApi(token=token).upload_folder(repo_id=p.REPO,repo_type='dataset',folder_path=str(folder),
                path_in_repo=prefix,allow_patterns=allowed,commit_message='S9 bounded geometry annotation batch; no training')
            print('HF publication succeeded:',r.oid);return r.oid
        except Exception as e:
            status=getattr(getattr(e,'response',None),'status_code',None)
            if status not in (None,429,500,502,503,504) or attempt==4:raise
            hint=getattr(getattr(e,'response',None),'headers',{}).get('Retry-After','0')
            wait=max(10*2**attempt,float(hint) if hint.isdigit() else 0)
            print(f'Upload backoff {wait}s; files safe locally.');time.sleep(wait)
