"""Fail-closed prospective intake; missing inputs never become invented labels."""
from PIL import Image
import math,re
from collections import Counter,defaultdict
from datetime import date
from pathlib import Path
import numpy as np
from common_gate import CLASSES,inside,read,sha,table
from features_gate import dhash
ISSUE_FIELDS=['entity','record_id','severity','code','detail']

def audit_intake(folder,policy,legacy_rows,legacy_quality=None,media_root=None):
 folder=folder.resolve();media=Path(media_root).resolve() if media_root else folder/'incoming';issues=[];file_hashes={}
 def issue(entity,key,code,detail,severity='error'):issues.append(dict(entity=entity,record_id=key,severity=severity,code=code,detail=str(detail)))
 def load(name,required):
  p=folder/(name+'.csv')
  if not p.exists():issue(name,'','missing_input_csv',p.name);return []
  try:
   records=table(p,required)
   return records
  except (AssertionError,ValueError) as exc:issue(name,'','invalid_csv',exc);return []
 def unique(name,records,key):
  counts=Counter(r.get(key,'') for r in records)
  for value,count in counts.items():
   if not value or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,159}',value):issue(name,value,'invalid_identifier',key)
   if count>1:issue(name,value,'duplicate_identifier',key)
  return {r.get(key,''):r for r in records}
 def truth(value):return str(value).strip().lower() in ('true','1','yes')
 def registered(name,suffixes):
  p=inside(media,name);assert p.suffix.lower() in suffixes,'Unsupported input file type'
  file_hashes[p.relative_to(media.resolve()).as_posix()]=sha(p);return p
 old={r['physical_tyre_id']:r for r in legacy_rows if r['domain']=='original_photo'};old_hashes={r['image_sha256'] for r in legacy_rows}
 tyres=load('physical_tyres',['tyre_id','proxy_class','identity_review_status']);sessions=load('capture_sessions',['session_id','tyre_id']);images=load('images',['image_id','tyre_id','session_id','relative_path','split'])
 tyre_index=unique('tyres',tyres,'tyre_id');session_index=unique('sessions',sessions,'session_id');unique('images',images,'image_id')
 normalized_tyres=[]
 for row in tyres:
  key=row['tyre_id'];label=row.get('proxy_class','').strip().lower();existing=row.get('existing_project_tyre_id','').strip();new=truth(row.get('new_independent_tyre_confirmed'))
  if label not in CLASSES:issue('tyres',key,'invalid_proxy_class','Use low, mid or high with a documented label source')
  if new and (existing or key in old):issue('tyres',key,'existing_tyre_claimed_new','An existing tyre cannot add an independent subject')
  if not new and not existing and key not in old:issue('tyres',key,'identity_kind_unspecified','Declare a confirmed new tyre or an explicit existing original ID')
  if existing and existing not in old:issue('tyres',key,'unknown_existing_tyre_id',existing)
  if existing in old and label in CLASSES and CLASSES[old[existing]['class_index']]!=label:issue('tyres',key,'existing_label_conflict',existing)
  if row.get('identity_review_status','').strip().lower()!='confirmed':issue('tyres',key,'identity_not_confirmed','Confirm identity and provenance before prospective use')
  if not row.get('mileage_source','').strip():issue('tyres',key,'label_provenance_missing','Record the mileage/proxy-label source')
  for field in ('mileage_value','measured_tread_depth_mm'):
   if row.get(field,'')!='':
    try:assert math.isfinite(float(row[field])) and float(row[field])>=0
    except (AssertionError,ValueError):issue('tyres',key,'invalid_measurement',field)
  if row.get('mileage_value','')!='' and row.get('mileage_unit','').lower() not in ('km','miles'):issue('tyres',key,'mileage_unit_missing','Use km or miles for a supplied mileage value')
  if row.get('measured_tread_depth_mm','')!='' and not row.get('depth_method','').strip():issue('tyres',key,'depth_method_missing','A real measurement requires its method')
  normalized_tyres.append(dict(tyre_id=key,existing_project_tyre_id=existing,new_independent_tyre_confirmed=new,proxy_class=label,
   identity_review_status=row.get('identity_review_status',''),mileage_value=row.get('mileage_value',''),mileage_unit=row.get('mileage_unit',''),
   mileage_source=row.get('mileage_source',''),measured_tread_depth_mm=row.get('measured_tread_depth_mm',''),depth_method=row.get('depth_method',''),
   provenance='user_supplied_intake; identity confirmation is a recorded claim, not image-hash proof'))
 for row in sessions:
  key=row['session_id']
  if row.get('tyre_id') not in tyre_index:issue('sessions',key,'unknown_tyre_reference',row.get('tyre_id'))
  for field in ('capture_date','domain','device_id','lighting'):
   if not row.get(field,'').strip():issue('sessions',key,'missing_capture_metadata',field)
  if row.get('capture_date'):
   try:date.fromisoformat(row['capture_date'])
   except ValueError:issue('sessions',key,'invalid_capture_date','Use YYYY-MM-DD')
 normalized_images=[];hash_groups=defaultdict(list);role_groups=defaultdict(set);parents=defaultdict(set)
 for row in images:
  key=row['image_id'];tid=row.get('tyre_id','');sid=row.get('session_id','');role=row.get('split','').strip().lower();tyre=tyre_index.get(tid,{})
  if tid not in tyre_index:issue('images',key,'unknown_tyre_reference',tid)
  if sid not in session_index:issue('images',key,'unknown_session_reference',sid)
  elif session_index[sid]['tyre_id']!=tid:issue('images',key,'session_tyre_conflict',sid)
  if role not in ('','unassigned','train','validation','test'):issue('images',key,'invalid_prospective_split',role)
  if role in ('train','validation','test'):role_groups[tid].add(role)
  else:issue('images',key,'prospective_split_unassigned','Assign roles only under the declared prospective protocol')
  if row.get('proxy_class') and row['proxy_class'].strip().lower()!=tyre.get('proxy_class','').strip().lower():issue('images',key,'image_tyre_label_conflict',row['proxy_class'])
  if row.get('label_review_status','').strip().lower()!='reviewed':issue('images',key,'label_review_not_complete','The class/provenance record needs a completed review')
  if row.get('parent_source_id'):parents[row['parent_source_id']].add(tid)
  if row.get('original_or_derivative','').lower() in ('frame','derivative','augmentation') and not row.get('parent_source_id'):issue('images',key,'missing_parent_source','Frames/derivatives need the original source identity')
  out=dict(image_id=key,tyre_id=tid,session_id=sid,proxy_class=tyre.get('proxy_class','').strip().lower(),proposed_split=role,
   relative_path=row.get('relative_path',''),parent_source_id=row.get('parent_source_id',''),file_sha256=None,width=None,height=None,exif_orientation=None,
   brightness_0_255=None,focus_laplacian_variance=None,dhash64=None,closest_existing_image=None,closest_existing_dhash_distance=None,
   mask_status='missing',point_kind='missing',human_points=0,tread_outside_tyre_pixels=None,tread_outside_tyre_valid_pixels=None,ignored_pixels=None,
   new_independent_tyre_claim=truth(tyre.get('new_independent_tyre_confirmed')),identity_review_status=tyre.get('identity_review_status',''))
  try:
   p=registered(row.get('relative_path',''),('.jpg','.jpeg','.png'))
   out['file_sha256']=sha(p)
   if row.get('image_sha256') and row['image_sha256'].lower()!=out['file_sha256']:issue('images',key,'image_checksum_mismatch','Declared and actual SHA-256 differ')
   if out['file_sha256'] in old_hashes:issue('images',key,'existing_photo_reintroduced','This exact image already belongs to the development dataset')
   hash_groups[out['file_sha256']].append((tid,role,key))
   with Image.open(p) as image:
    out['width'],out['height']=image.size;out['exif_orientation']=image.getexif().get(274,1);out['dhash64']=dhash(image)
    gray=np.asarray(image.convert('L').resize((384,512),Image.Resampling.BILINEAR),dtype=np.float32)
    out['brightness_0_255']=float(gray.mean());lap=gray[:-2,1:-1]+gray[2:,1:-1]+gray[1:-1,:-2]+gray[1:-1,2:]-4*gray[1:-1,1:-1];out['focus_laplacian_variance']=float(lap.var())
   if legacy_quality:
    nearest=min(((int(out['dhash64'],16)^int(r['dhash64'],16)).bit_count(),r['image_id']) for r in legacy_quality)
    out['closest_existing_dhash_distance'],out['closest_existing_image']=nearest
    if nearest[0]<=policy.get('near_similarity_hamming_review_radius',6):issue('images',key,'similarity_review_candidate','Review similarity to '+nearest[1]+'; this is not an identity assignment','warning')
   if out['exif_orientation']!=1:issue('images',key,'orientation_requires_normalization','Normalize image and annotations together before validation')
  except (AssertionError,OSError,ValueError) as exc:issue('images',key,'invalid_image_path_or_file',exc)
  reserved=role in ('validation','test')
  if reserved and (not out['new_independent_tyre_claim'] or tyre.get('existing_project_tyre_id') or tid in old):issue('images',key,'reserved_tyre_not_new','Prospective held-out roles require additional independent tyres')
  if row.get('mask_annotation'):
   try:
    annotation=read(registered(row['mask_annotation'],('.json',)));assert annotation.get('image_id')==key
    shapes=[]
    for field in ('tyre_mask','tread_mask'):
     with Image.open(registered(annotation[field],('.png',))) as image:
      a=np.asarray(image);assert image.size==(out['width'],out['height']) and a.ndim==2 and set(np.unique(a))<=set((0,1,255));shapes.append(a>0)
    assert annotation.get('review_status')=='reviewed','Masks require a documented review'
    out['mask_status']='reviewed';out['tread_outside_tyre_pixels']=int((shapes[1]&~shapes[0]).sum())
    ignored=np.zeros(shapes[0].shape,dtype=bool)
    if annotation.get('ignore_mask'):
     with Image.open(registered(annotation['ignore_mask'],('.png',))) as image:
      a=np.asarray(image);assert image.size==(out['width'],out['height']) and a.ndim==2 and set(np.unique(a))<=set((0,1,255));ignored=a>0
    out['ignored_pixels']=int(ignored.sum());out['tread_outside_tyre_valid_pixels']=int((shapes[1]&~shapes[0]&~ignored).sum())
    if out['tread_outside_tyre_valid_pixels']:issue('images',key,'unignored_mask_containment_conflict','Review disagreement or an explicit ignore policy; the gate does not repair source masks')
   except (AssertionError,OSError,ValueError,KeyError) as exc:issue('images',key,'invalid_mask_annotation',exc)
  elif reserved and policy.get('reserved_roles_require_reviewed_masks'):issue('images',key,'reserved_masks_missing','Provide reviewed overlapping tyre/tread masks')
  if row.get('point_annotation'):
   try:
    annotation=read(registered(row['point_annotation'],('.json',)));pts=annotation['points'];assert annotation.get('image_id')==key and len(pts)==6
    assert annotation.get('point_kind') in ('human','weak_proposal') and annotation.get('review_status')=='reviewed'
    for pt in pts:assert 0<=float(pt['x'])<=out['width']-1 and 0<=float(pt['y'])<=out['height']-1
    yy=[]
    for i in (0,2,4):assert float(pts[i]['x'])<float(pts[i+1]['x']) and float(pts[i]['y'])==float(pts[i+1]['y']);yy.append(float(pts[i]['y']))
    assert yy[0]<yy[1]<yy[2];out['point_kind']=annotation['point_kind'];out['human_points']=6 if annotation['point_kind']=='human' else 0
   except (AssertionError,OSError,ValueError,KeyError,TypeError) as exc:issue('images',key,'invalid_point_annotation',exc)
  normalized_images.append(out)
 for tid,roles in role_groups.items():
  if len(roles)>1:issue('tyres',tid,'physical_tyre_crosses_splits',','.join(sorted(roles)))
 for parent,tids in parents.items():
  if len(tids)>1:issue('parents',parent,'parent_source_crosses_tyres',','.join(sorted(tids)))
 for h,group in hash_groups.items():
  if len({tid for tid,_,_ in group})>1 or len({role for _,role,_ in group})>1:issue('images',h,'duplicate_image_crosses_identity_or_split',';'.join(key for _,_,key in group))
  elif len(group)>1:issue('images',h,'repeated_image_same_tyre',';'.join(key for _,_,key in group),'warning')
 for tid,roles in role_groups.items():
  if roles & {'validation','test'} and policy.get('reserved_roles_require_human_geometry_per_tyre'):
   if not any(r['tyre_id']==tid and r['human_points']==6 for r in normalized_images):issue('tyres',tid,'reserved_human_geometry_missing','At least one reviewed human six-point image is needed for this tyre')
 # Blockers on a tyre/session affect all its dependent records; an invalid row
 # is never quietly dropped to manufacture a passing prospective dataset.
 for row in normalized_images:
  row['eligible_after_validation']=not any(i['severity']=='error' and ((i['entity']=='images' and i['record_id'] in (row['image_id'],row['file_sha256'])) or (i['entity']=='tyres' and i['record_id']==row['tyre_id']) or (i['entity']=='sessions' and i['record_id']==row['session_id']) or (i['entity']=='parents' and i['record_id']==row['parent_source_id'])) for i in issues)
 support=[];minimum=policy.get('minimum_distinct_tyres_per_class_role');gaps=[]
 if not isinstance(minimum,dict) or any(type(minimum.get(role)) is not int or minimum.get(role,0)<1 for role in ('train','validation','test')):
  gaps.append(dict(code='subject_minima_not_declared',detail='Declare positive per-class subject minima for each role under the actual available cohort'))
 for role in ('train','validation','test'):
  for label in CLASSES:
   subjects={r['tyre_id'] for r in normalized_images if r['proposed_split']==role and r['proxy_class']==label and r['eligible_after_validation']}
   new_subjects={r['tyre_id'] for r in normalized_images if r['tyre_id'] in subjects and r['new_independent_tyre_claim']}
   if role=='train' and policy.get('legacy_training_scope')=='existing_train_only':subjects.update(r['physical_tyre_id'] for r in legacy_rows if r['domain']=='original_photo' and r['role']=='train' and CLASSES[r['class_index']]==label)
   required=minimum.get(role) if isinstance(minimum,dict) else None
   enough=type(required) is int and required>0 and len(subjects)>=required
   support.append(dict(role=role,proxy_class=label,confirmed_tyre_records=len(subjects),new_independent_tyre_claims=len(new_subjects),minimum_required=required,minimum_met=enough))
   if not enough:gaps.append(dict(code='prospective_support_incomplete',detail=role+'/'+label+': observed '+str(len(subjects))+', required '+str(required) if required is not None else role+'/'+label+': subject minimum not declared'))
 if not policy.get('frozen'):gaps.append(dict(code='protocol_not_frozen',detail='Freeze the prospective data/evaluation policy after its actual subjects and criteria are defined'))
 if not images:gaps.append(dict(code='no_new_intake_images',detail='No new collection was supplied; existing evidence cannot add independent tyres'))
 if any(i['severity']=='error' for i in issues):gaps.append(dict(code='intake_validation_errors',detail='Resolve every error in intake_issues.csv'))
 ready=bool(images) and not gaps
 return dict(tyres=normalized_tyres,sessions=sessions,images=normalized_images,issues=issues,support=support,gaps=gaps,prospective_training_ready=ready,file_hashes=file_hashes)
