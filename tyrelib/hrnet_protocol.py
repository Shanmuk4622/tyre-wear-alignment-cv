"""Frozen, user-identity-confirmed HRNet geometry experiment. CPU preflight."""
import collections
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import time
import requests
from PIL import Image, ImageDraw

REPO='Shanmuk4622/tyre-wear-study'
REV='2b4773914e6eefe421d8ddd74e79c5b84606c9aa'
PACKAGE='100dac5b12e1192ede59f9d1231c5aeeed80428850661a9add1cba147fe22f92'
ANN='8e8fd0734b9f4fe4236e699b804c2f8afb85ae4bb847f8d3c83b0abff5a8769e'
BASE=f's9/s9-geometry-labels-r2-all120/packages/{PACKAGE}'
VERSION='hrnet-geometry-2026-09-15-r1'
POINTS=['left_upper','right_upper','left_middle','right_middle','left_lower','right_lower']
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def sha(x):return hashlib.sha256(x).hexdigest()
def file_sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def write(path,x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_bytes(canonical(x));tmp.replace(path)
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def retry(fn):
    for i in range(6):
        try:return fn()
        except Exception as e:
            response=getattr(e,'response',None);code=getattr(response,'status_code',None)
            if code not in (None,429,500,502,503,504) or i==5:raise
            hint=getattr(response,'headers',{}).get('Retry-After','0')
            delay=max(10*2**i,float(hint) if str(hint).isdigit() else 0)
            print(f'HF retry after {delay:.0f}s; durable local state retained.',flush=True)
            until=time.monotonic()+delay
            while time.monotonic()<until:time.sleep(max(0,min(5,until-time.monotonic())))
def source(path,work):
    dest=Path(work)/'sources'/sha(path.encode());dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():
        def download():
            with requests.get(f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/{BASE}/{path}',timeout=45,stream=True) as r:
                r.raise_for_status();b=bytearray()
                for chunk in r.iter_content(65536):
                    b.extend(chunk)
                    if len(b)>2*1024**2:raise ValueError('Metadata cap exceeded')
            return b
        raw=retry(download);tmp=dest.with_suffix('.tmp');tmp.write_bytes(raw);tmp.replace(dest)
    return dest.read_bytes()
def split_groups(images):
    """8/2/2 tyres; choose image-count balance, then a fixed hash tie-break only."""
    counts=collections.Counter(r['session'] for r in images);groups=sorted(counts)
    assert len(groups)==12
    options=[]
    for test in itertools.combinations(groups,2):
        for dev in itertools.combinations([g for g in groups if g not in test],2):
            nt=sum(counts[g] for g in test);nv=sum(counts[g] for g in dev)
            if min(nt,nv)<12:continue
            key=(abs(nt-24)+abs(nv-24),sha(canonical([test,dev])))
            options.append((key,list(test),list(dev)))
    _,test,dev=min(options)
    return dict(train=[g for g in groups if g not in test+dev],validation=dev,test=test)
def contract(work):
    raw=source(f'reviews/{ANN}/ANNOTATIONS.json',work);assert sha(raw)==ANN
    labels=json.loads(raw);manifest=json.loads(source('MANIFEST.json',work))
    assert sha(canonical({k:v for k,v in manifest.items() if k!='package_id'}))==PACKAGE
    import s9_expansion as e
    checks=e.validate(labels,manifest)
    assert len(checks)==120 and all(r['complete'] for r in checks)
    by_id={r['pilot_id']:r for r in labels['annotations']}
    assert all(by_id[r['pilot_id']]['points'][n]['state']=='visible' for r in manifest['images'] for n in POINTS), 'This protocol is coordinate-only for the frozen all-visible submission'
    groups=split_groups(manifest['images'])
    rows=[]
    for ref in manifest['images']:
        a=by_id[ref['pilot_id']]
        role=next(k for k,v in groups.items() if ref['session'] in v)
        rows.append(ref|dict(role=role,x=[a['points'][n]['x']/(ref['width']-1) for n in POINTS]))
    cfg=dict(version=VERSION,source_revision=REV,annotations_sha256=ANN,package_id=PACKAGE,
        identity=dict(source='User confirmation in project conversation, 15 September 2026',
            assertion='12 capture sessions represent 12 different physical tyres',independently_verified=False),
        groups=groups,rows=rows,point_names=POINTS,
        model='hrnet_w18.ms_aug_in1k',feature_location='',feature_indices=[1,2,3,4],
        pretrained_repo='timm/hrnet_w18.ms_aug_in1k',pretrained_revision='7e2c5583769f54514fd87e3ba9de408e33eaba0f',
        packages={'timm':'1.0.15'},input_hw=[512,384],batch_size=2,epochs=60,seeds=[1,2,3],
        optimizer='AdamW',lr=0.0001,weight_decay=0.01,lr_schedule='cosine_epoch',
        augmentation='none; deterministic identity transform',batchnorm='frozen running statistics',
        loss='six horizontal Gaussian targets, sigma=1.5 feature pixels; mean cross entropy',
        endpoint='fixed epoch 60; no best-test or best-validation checkpoint selection',
        visibility='all labels visible; no visibility classifier or rejection validation',
        qa='mechanical checks and review diagrams; human accuracy not certified',
        scripts={n:file_sha(Path(__file__).parent/n) for n in ['hrnet_protocol.py','hrnet_runtime.py','s9_expansion.py','s9_pilot.py']},
        limitations=['only 12 user-confirmed tyres; small two-tyre validation/test sets',
            'old pilot images were development evidence; same tyre identities were seen in pilot analysis',
            'not untouched external-cohort validation; no physical angles or healthy-reference inference',
            'no matched segmentation comparison until baseline train-set overlap is resolved'])
    return cfg
def prefix(cfg):return f's9/{VERSION}/{sha(canonical(cfg))}'
def root_data(root=''):
    if root:return Path(root)
    found=list(Path('/kaggle/input').glob('**/manifests/clean_manifest.csv'))
    if len(found)!=1:raise ValueError('Attach ONE Tire Dataset Prepared dataset, or set DATA_ROOT to FINAL')
    return found[0].parent.parent
def validate_images(cfg,root):
    hashes={k:set() for k in cfg['groups']}
    for r in cfg['rows']:
        path=(Path(root)/r['original_relative_path']).resolve()
        assert path.is_relative_to(Path(root).resolve())
        assert file_sha(path)==r['image_sha256'],r['pilot_id']+' original differs'
        with Image.open(path) as im:assert im.size==(r['width'],r['height'])
        hashes[r['role']].add(r['image_sha256'])
    assert not hashes['train']&hashes['validation'] and not hashes['train']&hashes['test'] and not hashes['test']&hashes['validation']
def publish(folder,path,token):
    from huggingface_hub import HfApi
    folder=Path(folder)
    # One synchronous writer; no checkpoint changes during this atomic repo commit.
    result=retry(lambda:HfApi(token=token).upload_folder(repo_id=REPO,repo_type='dataset',
        folder_path=str(folder),path_in_repo=path,ignore_patterns=['*.tmp','*.lock'],
        commit_message='HRNet geometry durable snapshot'))
    print('HF publication succeeded:',result.oid,flush=True);return result.oid
def preflight(work,root,token,upload=True):
    cfg=contract(work);validate_images(cfg,root)
    folder=Path(work)/'preflight'/sha(canonical(cfg));folder.mkdir(parents=True,exist_ok=True)
    write(folder/'CONTRACT.json',cfg)
    counts={k:sum(r['role']==k for r in cfg['rows']) for k in cfg['groups']}
    flags=[]
    for r in cfg['rows']:
        # Original-image point overlays; not synthetic labels. Every image is included.
        with Image.open(Path(root)/r['original_relative_path']) as im:
            im=im.convert('RGB');draw=ImageDraw.Draw(im)
            for j,x in enumerate(r['x']):
                xx=x*(r['width']-1);y=r['guide_y'][j//2]
                draw.ellipse((xx-8,y-8,xx+8,y+8),outline='cyan',width=3)
                if min(xx,r['width']-1-xx)<=10:flags.append(dict(pilot_id=r['pilot_id'],point=POINTS[j],reason='near edge; diagnostic only'))
            im.thumbnail((576,768));im.save(folder/(r['pilot_id']+'.jpg'),quality=88)
    write(folder/'QA.json',dict(counts=counts,flags=flags,all_visible_points=720,
        label_quality='not independently certified',identity=cfg['identity']))
    write(folder/'STATUS.json',dict(status='preflight_passed',protocol=sha(canonical(cfg)),
        next='Run NB27 smoke test; do not interpret format checks as physical validation'))
    for n in cfg['scripts']:shutil.copy2(Path(__file__).parent/n,folder/n)
    if upload:publish(folder,prefix(cfg)+'/preflight',token)
    print('Locked image counts:',counts,'; tyres:',{k:len(v) for k,v in cfg['groups'].items()},flush=True)
    return folder
