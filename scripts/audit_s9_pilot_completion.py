"""Read-only HF/pilot audit and scientific review overlays; never edits user labels."""
import base64
from collections import Counter
import io
import json
from pathlib import Path
import re
import sys
import requests
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tyrelib'))
import s9_pilot as p

def main():
    local=ROOT/'PILOT_12_IMAGES'
    raw=(local/'annotations_87933cacc9cc.json').read_bytes(); labels=json.loads(raw)
    manifest_raw=(local/'MANIFEST.json').read_bytes(); manifest=json.loads(manifest_raw)
    package=labels['package_id']; annotation_sha=p.digest(raw)
    out=ROOT/'outputs/s9_pilot_completion'/annotation_sha;out.mkdir(parents=True,exist_ok=True)
    assert p.digest(p.canonical({k:v for k,v in manifest.items() if k!='package_id'}))==package
    checked=p.validate_annotations(labels,manifest)
    session=requests.Session(); total=0
    def get(url):
        nonlocal total
        with session.get(url,timeout=45,stream=True) as r:
            r.raise_for_status();chunks=[];count=0
            for chunk in r.iter_content(65536):
                count+=len(chunk);total+=len(chunk)
                if count>1024*1024 or total>2*1024*1024:raise ValueError('Small audit download budget exceeded')
                chunks.append(chunk)
        return b''.join(chunks)
    refs=json.loads(get(f'https://huggingface.co/api/datasets/{p.REPO}/refs'))
    revision=next(b['targetCommit'] for b in refs['branches'] if b['name']=='main')
    prefix=f's9/{p.VERSION}/{package}'
    public={}
    for name,path in [('manifest',f'{prefix}/MANIFEST.json'),('prepare_status',f'{prefix}/STATUS.json'),
                     ('annotations',f'{prefix}/reviews/{annotation_sha}/ANNOTATIONS.json'),
                     ('review',f'{prefix}/reviews/{annotation_sha}/REVIEW.json'),
                     ('review_status',f'{prefix}/reviews/{annotation_sha}/STATUS.json')]:
        content=get(f'https://huggingface.co/datasets/{p.REPO}/resolve/{revision}/{path}')
        (out/f'HF_{name}.json').write_bytes(content);public[name]=json.loads(content)
        if name=='manifest':assert content==manifest_raw
        if name=='annotations':assert content==raw
    assert public['review']['records']==checked
    assert public['review']['annotations_sha256']==annotation_sha
    assert public['review']['complete_images']==sum(r['complete'] for r in checked)
    assert public['review']['status']==('needs_human_review' if all(r['complete'] for r in checked) else 'partial_annotation')
    assert public['review']['training_approved'] is False
    nb=json.loads((ROOT/'notebooks/NB22_S9_Pilot_Review.ipynb').read_text(encoding='utf-8'))
    errors=[o for c in nb['cells'] for o in c.get('outputs',[]) if o['output_type']=='error'];assert not errors
    streams='\n'.join(''.join(o.get('text',[])) for c in nb['cells'] for o in c.get('outputs',[]))
    commits=re.findall(r'HF publication succeeded: ([0-9a-f]{40})',streams);assert commits
    html=(local/'ANNOTATE.html').read_text(encoding='utf-8')
    data=json.loads(re.search(r'const DATA=(.*);\nconst \$',html).group(1))
    assert data['package_id']==package
    by_id={a['pilot_id']:a for a in labels['annotations']};by_manifest={a['pilot_id']:a for a in manifest['images']}
    comparisons=[];images=[]
    maskroot=Path('D:/Dataset Download/Tire Dataset Prepared/annotations/clean/masks')
    for row in data['images']:
        image_raw=base64.b64decode(row['image'].split(',')[1]);assert p.digest(image_raw)==row['image_sha256']
        im=np.asarray(Image.open(io.BytesIO(image_raw)).convert('RGB'));images.append((row,im))
        maskpath=maskroot/(by_manifest[row['pilot_id']]['image_id']+'.png')
        mask=np.asarray(Image.open(maskpath)) if maskpath.exists() else None
        for name,point in by_id[row['pilot_id']]['points'].items():
            result=dict(pilot_id=row['pilot_id'],point=name,state=point['state'])
            if point['state']=='visible':
                x,y=point['x'],point['y']; result.update(x=x,y=y,edge_distance_px=min(x,row['width']-1-x))
                if mask is not None:
                    for region,sel in [('tyre',mask>0),('tread',(mask==2)|(mask==3))]:
                        xs=np.flatnonzero(sel[y]); boundary=int(xs[0] if name.startswith('left') else xs[-1]) if len(xs) else None
                        result[region+'_mask_x']=boundary
                        result[region+'_distance_px']=abs(x-boundary) if boundary is not None else None
            comparisons.append(result)
    for batch in range(2):
        fig,axes=plt.subplots(3,2,figsize=(12,22))
        for ax,(row,im) in zip(axes.flat,images[batch*6:(batch+1)*6]):
            a=by_id[row['pilot_id']];ax.imshow(im)
            for y in row['guide_y']:ax.axhline(y,color='gold',lw=.8,alpha=.7)
            for name,point in a['points'].items():
                if point['state']=='visible':
                    ax.scatter(point['x'],point['y'],s=45,facecolors='none',edgecolors='#00ffcc',linewidths=1.8)
            ax.set_title(f"{row['pilot_id']} | user points (cyan) | {a['visual_review']}",fontsize=11)
            ax.set_xlim(-20,row['width']+20);ax.set_ylim(row['height']+10,-10);ax.axis('off')
        fig.tight_layout();fig.savefig(out/f'pilot_review_{batch+1}.jpg',dpi=135);plt.close(fig)
    row,im=images[5]
    Image.fromarray(im).save(out/'P06_native.png')
    summary=dict(status='HF_and_local_contract_verified',hf_revision=revision,notebook_commit=commits[-1],
        package_id=package,annotations_sha256=annotation_sha,downloaded_bytes=total,
        complete_images=sum(r['complete'] for r in checked),visible_points=sum(r['state']=='visible' for r in comparisons),
        visual_observations=dict(Counter(a['visual_review'] for a in labels['annotations'])),
        independent_records=dict(Counter(a['independent_record'] for a in labels['annotations'])),
        near_frame_edge_points=[r for r in comparisons if r.get('edge_distance_px',999)<=23],
        median_distance_to_existing_tread_mask_px=float(np.median([r['tread_distance_px'] for r in comparisons if r.get('tread_distance_px') is not None])),
        comparisons= comparisons,training_approved=False,healthy_reference_eligible=False,
        note='Mask distances are disagreement diagnostics, not independent ground-truth accuracy. Visual review required.')
    p.write_json(out/'AUDIT.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='comparisons'},indent=2))

if __name__=='__main__':main()
