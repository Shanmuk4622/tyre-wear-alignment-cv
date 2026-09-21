"""Build a self-contained, hash-verified Phase 2 source dataset without editing inputs."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from phase2_audit_labels import ROOT, sha, mask_shapes

OLD = Path(r'D:\Dataset Download\Tire Dataset Prepared')
DEST = ROOT/'data'/'phase2_dataset_v1'
RELEASE = ROOT/'releases'


def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')


def copy_verified(source,relative,expected=None):
    digest=sha(source)
    if expected and digest!=expected: raise ValueError(f'Source hash mismatch: {source}')
    target=DEST/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if sha(target)!=digest: raise ValueError(f'Existing staged file differs: {target}')
    else: shutil.copyfile(source,target)
    assert sha(target)==digest
    return digest


def save_mask(mask,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(mask.astype(np.uint8)*255).save(path)


def bbox(mask):
    y,x=np.where(mask)
    return [int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)] if len(x) else None


def main():
    if (DEST/'VERSION.json').exists():
        raise SystemExit('Release tree exists. Verify it or create a new version; do not overwrite a frozen release.')
    audit=json.loads((ROOT/'manifests/phase2_label_audit.json').read_text())
    assert audit['summary']['frames_with_errors']==0 and not audit['summary']['extra_annotations']
    newaudit={r['frame_id']:r for r in audit['records']}
    frames=json.loads((ROOT/'manifests/phase2_frames.json').read_text())['frames']
    identities=json.loads((ROOT/'manifests/phase2_video_identity.json').read_text())
    videos={r['video_id']:r for r in identities['videos']}
    oldrows=list(csv.DictReader((OLD/'FINAL/manifests/clean_manifest.csv').open(encoding='utf-8-sig')))
    assert len(oldrows)==418 and len({r['image_id'] for r in oldrows})==418
    checks={}
    for line in (OLD/'annotations/checksums_sha256.txt').read_text().splitlines():
        bits=line.split(None,1)
        if len(bits)==2: checks[bits[1].replace('\\','/').lstrip('*')]=bits[0]
    records=[]; sources={}; boxes=[]
    for row in oldrows:
        id=row['image_id']; image=OLD/'FINAL'/row['relative_path']; mask=OLD/'annotations/clean/masks'/f'{id}.png'
        ip=f'images/original/{id}.jpg'; mp=f'annotations/original_indexed/{id}.png'
        ih=copy_verified(image,ip,row['file_sha256'])
        mh=copy_verified(mask,mp,checks[f'clean/masks/{id}.png'])
        sources[str(image)]=ih;sources[str(mask)]=mh
        with Image.open(image) as im:
            im.load(); assert im.size==(int(row['width']),int(row['height'])) and im.mode=='RGB'
        with Image.open(mask) as mm: indexed=np.array(mm)
        assert indexed.shape==(int(row['height']),int(row['width'])) and set(np.unique(indexed)).issubset({0,1,2,3,4})
        tyre=indexed>0;tread=(indexed==2)|(indexed==3);ignore=np.zeros_like(tyre)
        assert tyre.any() and tread.any()
        records.append(dict(image_id=id,domain='original_photo',image_path=ip,image_sha256=ih,
            width=int(row['width']),height=int(row['height']),proxy_label=row['proxy_label'],
            class_index={'low_mileage_proxy':0,'mid_mileage_proxy':1,'high_mileage_proxy':2}[row['proxy_label']],
            physical_tyre_id=row['session_group'],original_session_id=row['session_group'],
            identity_status='original_recorded_group_user_confirmed_distinct',source_video=None,
            timestamp_seconds=None,source_frame_index=None,legacy_fold=int(row['fold_id']),split='UNASSIGNED',
            source_label_path=mp,source_label_sha256=mh,label_source='unchanged_original_indexed_manual_mask',
            geometry_human_available=False,annotation_conflict_pixels=0))
        add_masks(records[-1],tyre,tread,ignore,boxes)
    for r in frames:
        id=r['frame_id'];a=newaudit[id];ident=videos[r['video_id']]
        image=ROOT/r['relative_path'];ann=ROOT/a['annotation_path']
        ih=copy_verified(image,r['relative_path'],r['png_sha256'])
        ah=copy_verified(ann,a['annotation_path'],a['annotation_sha256'])
        sources[str(image)]=ih;sources[str(ann)]=ah
        d=json.loads(ann.read_text(encoding='utf-8'))
        tyre,tread,manual_ignore=[mask_shapes(d,k) for k in ('tyre','tread','ignore')]
        conflict=tread&~tyre
        # Preserve both human channels. Contradictory pixels are unknown, not background.
        ignore=manual_ignore|conflict
        session=ident['original_session_id']
        records.append(dict(image_id=id,domain='new_video_frame',image_path=r['relative_path'],image_sha256=ih,
            width=r['width'],height=r['height'],proxy_label=ident['mileage_proxy'],class_index=1,
            physical_tyre_id=session,original_session_id=session,provisional_video_tyre_id=ident['provisional_tyre_id'],
            identity_status='mapping_pending' if session is None else 'user_confirmed_mapping',
            source_video=r['video_id'],source_video_sha256=r['source_video_sha256'],
            timestamp_seconds=r['actual_seconds'],target_seconds=r['target_seconds'],source_frame_index=r['source_frame_index'],
            legacy_fold=None,split='UNASSIGNED',source_label_path=a['annotation_path'],source_label_sha256=ah,
            label_source='user_submitted_labelme_polygons',geometry_human_available=False,
            annotation_conflict_pixels=int(conflict.sum()),annotation_conflict_fraction=a['tread_outside_tyre_fraction'],
            original_flags=d['flags'],derived_touches_image_border=a['touches_image_border']))
        add_masks(records[-1],tyre,tread,ignore,boxes)
    assert len(records)==570 and len({r['image_id'] for r in records})==570
    assert len({r['image_sha256'] for r in records})==570, 'Duplicate image bytes; reconcile before release'
    byid={r['image_id']:r for r in records}
    oldcontract=ROOT.parent/'outputs/hrnet_final_audit/report_CONTRACT.json'
    contract=json.loads(oldcontract.read_text())
    oldpoints=[]
    for r in contract['rows']:
        base=byid[r['image_id']]
        assert base['image_sha256']==r['image_sha256'] and base['original_session_id']==r['session']
        base['geometry_human_available']=True
        oldpoints.append(dict(image_id=r['image_id'],label_source='existing_human_six_point_labels',
            source_protocol=sha(oldcontract),guide_y=r['guide_y'],width=r['width'],height=r['height'],
            points=[dict(name=name,x=float(x*(r['width']-1)),y=r['guide_y'][i//2],state='visible',human_accepted=True)
                    for i,(name,x) in enumerate(zip(contract['point_names'],r['x']))]))
    newpoints=[dict(image_id=r['frame_id'],label_source='polygon_derived_proposal_not_human_point_ground_truth',
                   points=r['points'],human_accepted=False) for r in audit['records']]
    write_json(DEST/'geometry/original_human_points.json',oldpoints)
    write_json(DEST/'geometry/new_point_proposals.json',newpoints)
    copy_verified(oldcontract,'provenance/original_geometry_contract.json')
    for name in ['phase2_frames.json','phase2_video_identity.json','phase2_label_audit.json']:
        copy_verified(ROOT/'manifests'/name,'provenance/'+name)
    for source,name in [(OLD/'FINAL/manifests/clean_manifest.csv','original_clean_manifest.csv'),
                        (OLD/'FINAL/manifests/label_map.csv','original_label_map.csv'),
                        (OLD/'annotations/ANNOTATION_VERSION.json','original_annotation_version.json'),
                        (OLD/'annotations/checksums_sha256.txt','original_annotation_checksums.txt')]:
        copy_verified(source,'provenance/'+name)
    write_json(DEST/'manifests/images.json',records)
    scalar_fields=['image_id','domain','image_path','image_sha256','width','height','proxy_label','class_index',
                   'physical_tyre_id','original_session_id','identity_status','source_video','timestamp_seconds',
                   'legacy_fold','split','source_label_path','source_label_sha256','tyre_mask','tread_mask','ignore_mask',
                   'geometry_human_available','annotation_conflict_pixels']
    with (DEST/'manifests/images.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=scalar_fields,extrasaction='ignore');w.writeheader();w.writerows(records)
    write_json(DEST/'annotations/boxes_xyxy.json',dict(coordinates='native pixels; xyxy upper bounds exclusive',boxes=boxes))
    pending=dict(status='NOT_FROZEN',training_allowed=False,physical_tyre_count=12,
        reason='Exact new-video to original-tyre mapping unresolved; no frame-level random split permitted',
        all_rows='UNASSIGNED',proposed_group_counts=dict(train=6,validation=3,test=3),
        original_mid_session_candidates=identities['original_mid_session_candidates_from_existing_geometry_contract'])
    write_json(DEST/'splits/phase2_split_status.json',pending)
    summary=dict(status='VERIFIED_SOURCE_RELEASE_WITH_DOCUMENTED_WARNINGS',images=570,original_images=418,new_images=152,
        class_counts=dict(collections.Counter(r['proxy_label'] for r in records)),
        original_physical_tyres=len({r['original_session_id'] for r in records if r['domain']=='original_photo'}),
        new_additional_physical_tyres=0,new_manual_polygons=304,original_human_geometry_images=120,
        new_derived_point_proposals=912,new_human_accepted_point_labels=0,
        new_tread_outside_tyre_frames=45,conflict_pixels=sum(r['annotation_conflict_pixels'] for r in records),
        handling='Raw masks and annotations preserved; ignore mask = human ignore OR tread outside tyre. Never train on ignored pixels as background.',
        original_images_and_masks_hash_verified=418,all_images_decode_verified=True,
        original_geometry_hash_and_identity_matches=120,raw_annotations_unchanged=True,
        training_ready=False,upload_ready=True,
        open_items=['Confirm exact physical tyre mapping and freeze group split',
                    'Review/correct conflict pixels before trainers without ignore support (including stock YOLO)',
                    'Human acceptance/correction of new geometry point proposals',
                    'Build and GPU-verify training notebooks; no training performed'])
    write_json(DEST/'audit/dataset_validation.json',summary)
    write_json(DEST/'audit/label_visual_review.json',dict(scope='All 152 new frames inspected as contour thumbnails across 13 sheets plus individual missing-slot previews',
        conclusion='No missing/whole-frame swapped contours observed. Local contour disagreements and machinery-adjacent protrusions remain label uncertainty.',
        limitation='Not an independent pixel-accurate human reannotation or label-reliability measurement',
        raw_labels_edited=False,high_conflict_priority=['phase2_video3_t0000500_f000015','phase2_video3_t0002000_f000060']))
    for name in audit['summary']['sheets']:
        copy_verified(ROOT/'review/label_audit'/name,'audit/overlays/'+name)
    copy_verified(ROOT/'phase2_dataset_verify.py','phase2_dataset_verify.py')
    copy_verified(ROOT/'phase2DatasetReadme.md','README.md')
    # Record input protection separately outside the portable release (contains local paths).
    changed=[p for p,h in sources.items() if sha(Path(p))!=h]
    assert not changed, changed
    write_json(ROOT/'manifests/phase2_dataset_sources_unchanged.json',dict(checked_files=len(sources),unchanged=True,sha256=sources))
    version=dict(dataset_title='Tire Dataset Prepared phase2',version='phase2-source-v1',
        schema='phase2-overlapping-masks-v1',upload_ready=True,training_ready=False,
        image_manifest_sha256=sha(DEST/'manifests/images.json'),
        label_audit_sha256=sha(DEST/'provenance/phase2_label_audit.json'),
        note='Self-contained 570-image clean-source package; original 4180 augmented derivatives intentionally excluded; Phase 1 unaffected.')
    write_json(DEST/'VERSION.json',version)
    files=sorted(p for p in DEST.rglob('*') if p.is_file())
    (DEST/'SHA256SUMS.txt').write_text(''.join(f'{sha(p)}  {p.relative_to(DEST).as_posix()}\n' for p in files),encoding='utf-8')
    RELEASE.mkdir(parents=True,exist_ok=True)
    zip_path=RELEASE/'Tire_Dataset_Prepared_phase2_v1.zip'
    if zip_path.exists(): raise ValueError('Refusing to overwrite released ZIP')
    temporary=zip_path.with_suffix('.part')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for p in sorted(DEST.rglob('*')):
            if p.is_file(): z.write(p, 'phase2_dataset_v1/'+p.relative_to(DEST).as_posix())
    # Every archived member is decompressed and compared with the source tree hash.
    with zipfile.ZipFile(temporary) as z:
        expected={'phase2_dataset_v1/'+p.relative_to(DEST).as_posix():sha(p) for p in DEST.rglob('*') if p.is_file()}
        assert len(z.namelist())==len(set(z.namelist())) and set(z.namelist())==set(expected)
        for name,h in expected.items():
            with z.open(name) as f:
                d=hashlib.sha256()
                for b in iter(lambda:f.read(4*1024*1024),b''): d.update(b)
            assert d.hexdigest()==h,name
    temporary.replace(zip_path)
    report=dict(status='PASS',zip_path=zip_path.relative_to(ROOT).as_posix(),bytes=zip_path.stat().st_size,
        sha256=sha(zip_path),members=len(expected),all_members_decompressed_and_sha256_verified=True,
        checksum_manifest_sha256=sha(DEST/'SHA256SUMS.txt'),dataset=summary)
    write_json(ROOT/'manifests/phase2_zip_verification.json',report)
    (RELEASE/'Tire_Dataset_Prepared_phase2_v1.zip.sha256').write_text(report['sha256']+'  '+zip_path.name+'\n')
    print(json.dumps(report,indent=2),flush=True)


def add_masks(record,tyre,tread,ignore,boxes):
    id=record['image_id']
    for label,mask in [('tyre',tyre),('tread',tread),('ignore',ignore)]:
        relative=f'masks/{label}/{id}.png';save_mask(mask,DEST/relative);record[f'{label}_mask']=relative
    for label,mask in [('tyre',tyre),('tread',tread)]:
        boxes.append(dict(image_id=id,label=label,xyxy=bbox(mask),source='derived_from_raw_binary_mask',
                          ignore_pixels=int(ignore.sum()),human_box_annotation=False))


if __name__=='__main__': main()
