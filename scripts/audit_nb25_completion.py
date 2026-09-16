"""Read-only NB24/NB25 audit; preserves labels and remote state."""
import json,sys,collections
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tyrelib'))
import s9_expansion as e
local=ROOT/'ALL_120_IMAGES';out=ROOT/'outputs/s9_120_completion';out.mkdir(exist_ok=True)
files=list(local.glob('annotations_*.json'));assert len(files)==1
raw=files[0].read_bytes();a=json.loads(raw);sha=e.p.digest(raw);pkg=a['package_id']
rev='2b4773914e6eefe421d8ddd74e79c5b84606c9aa';prep='78d90571a46adab0cb41ed40d5dcd1d63628e19c'
prefix=f's9/{e.VERSION}/packages/{pkg}';total=0
def get(path,revision=rev):
    global total
    with requests.get(f'https://huggingface.co/datasets/{e.p.REPO}/resolve/{revision}/{prefix}/{path}',timeout=45,stream=True) as r:
        r.raise_for_status();b=bytearray()
        for chunk in r.iter_content(65536):
            b.extend(chunk);total+=len(chunk)
            assert len(b)<2*1024**2 and total<4*1024**2
    return bytes(b)
mraw=get('MANIFEST.json');m=json.loads(mraw)
assert mraw==(local/'MANIFEST.json').read_bytes()
assert get('MANIFEST.json',prep)==mraw
assert e.p.digest(e.p.canonical({k:v for k,v in m.items() if k!='package_id'}))==pkg
assert get(f'reviews/{sha}/ANNOTATIONS.json')==raw
review=json.loads(get(f'reviews/{sha}/REVIEW.json'))
records=e.validate(a,m);assert review['records']==records and review['annotations_sha256']==sha
assert review['complete_images']==120 and all(x['complete'] for x in records)
assert review['status']=='needs_human_review' and not review['training_approved']
status=json.loads(get(f'reviews/{sha}/STATUS.json'));assert not status['training_approved']
plan=json.loads(get('PLAN.json'));assert e.p.digest(e.p.canonical(plan))==m['plan_hash']
prepstatus=json.loads(get('STATUS.json'));assert prepstatus['image_count']==120
assert len({r['image_id'] for r in m['images']})==120
for r in m['images']:assert e.p.digest((local/'images'/(r['pilot_id']+'.jpg')).read_bytes())==r['image_sha256']
counts=collections.Counter(p['state'] for x in a['annotations'] for p in x['points'].values())
edge=[]
for x,r in zip(sorted(a['annotations'],key=lambda x:x['pilot_id']),sorted(m['images'],key=lambda x:x['pilot_id'])):
    for name,p in x['points'].items():
        if p['state']=='visible' and min(p['x'],r['width']-1-p['x'])<=10:edge.append(dict(image=x['pilot_id'],point=name,x=p['x']))
identity=json.loads((local/'SESSION_IDENTITY.json').read_text())
result=dict(status='publication_and_mechanical_checks_verified',revision=rev,prepare_revision=prep,
    package_id=pkg,annotation_sha256=sha,complete_images=120,points=dict(counts),
    images_with_visible_points=sum(any(p['state']=='visible' for p in x['points'].values()) for x in a['annotations']),
    observations=dict(collections.Counter(x['visual_review'] for x in a['annotations'])),
    independent_records=dict(collections.Counter(x['independent_record'] for x in a['annotations'])),
    unresolved_session_identities=sum(not x['physical_tyre_id'] for x in identity['sessions']),
    near_edge_visible_points=edge,downloaded_bytes=total,
    note='Edge flag is a diagnostic, not proof of wrong labels. Visual quality review and split approval remain. All 120 local image hashes verified; remote ZIP not downloaded.')
for name,value in [('AUDIT.json',result),('REVIEW.json',review),('PLAN.json',plan)]:e.p.write_json(out/name,value)
print(json.dumps(result,indent=2))
