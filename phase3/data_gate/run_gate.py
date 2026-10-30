"""Audit canonical data, collect richer CSV descriptors and validate prospective intake."""
from PIL import Image
import argparse,base64,io,itertools,json,os,shutil,sys
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda' or Path('/kaggle/input').exists()
from filelock import FileLock
from common_gate import ROOT,CLASSES,CHECKSUM,OVERLAY,locate,inside,read,sha,digest,atomic,emit,source_hashes,table,file_meta,field_info
from data import plan
from features_gate import describe
from intake_gate import audit_intake,ISSUE_FIELDS
from exposure_gate import aggregate
from common import environment
import torch

def resolutions(folder,identity):
 known={v['video_id'] for v in identity['videos']};candidates=identity['original_mid_session_candidates_from_existing_geometry_contract'];result={};problems=[]
 p=folder/'video_identity_resolutions.csv'
 if p.exists():
  try:entries=table(p,('video_id','confirmed_original_tyre_id','review_status','evidence_source'))
  except (AssertionError,ValueError) as exc:return {},[dict(entity='video_identity',record_id='',severity='error',code='invalid_resolution_csv',detail=str(exc))]
  seen=set()
  for row in entries:
   vid=row['video_id']
   if vid not in known or vid in seen:problems.append(dict(entity='video_identity',record_id=vid,severity='error',code='invalid_resolution_id',detail='Unknown or duplicate clip ID'));continue
   seen.add(vid)
   if row.get('confirmed_original_tyre_id'):
    if row['confirmed_original_tyre_id'] not in candidates or row.get('review_status')!='confirmed' or not row.get('evidence_source'):
     problems.append(dict(entity='video_identity',record_id=vid,severity='error',code='invalid_video_resolution',detail='A candidate original tyre, confirmed review and supporting source are required'))
    else:result[vid]=row['confirmed_original_tyre_id']
  if len(set(result.values()))!=len(result):problems.append(dict(entity='video_identity',record_id='',severity='error',code='distinct_clips_mapped_to_same_tyre',detail='The source user confirmation says the three clips show different tyres'));result={}
 return result,problems

def build(args):
 root=locate(args.data_root);intake=Path(args.intake or ROOT/'data_intake').resolve();output=Path(args.output or ROOT/'data_gate_outputs').resolve();output.mkdir(parents=True,exist_ok=True)
 cache=output/'cache';cache.mkdir(exist_ok=True);(cache/'features').mkdir(exist_ok=True)
 policy_path=intake/'protocol_policy.json' if (intake/'protocol_policy.json').exists() else ROOT/'data_gate/protocol_policy.json'
 policy=read(policy_path);radius=policy['near_similarity_hamming_review_radius'];neighbors_to_keep=policy['near_neighbors_per_image']
 env=environment(torch.device('cpu'))
 input_hashes={p.name:sha(p) for p in intake.glob('*.csv')};snapshot=cache/'intake'/digest(dict(files=input_hashes,policy=policy))[:12];snapshot.mkdir(parents=True,exist_ok=True)
 for name,h in input_hashes.items():
  target=snapshot/name
  if not target.exists():shutil.copy2(intake/name,target)
  assert sha(target)==h,'Input CSV changed while snapshotting; rerun against consistent inputs'
 assert type(radius) is int and 0<=radius<=64 and type(neighbors_to_keep) is int and 1<=neighbors_to_keep<=10
 assert shutil.disk_usage(output).free>1024**3,'Need 1 GiB free for the CPU gate/cache'
 assert sha(root/'SHA256SUMS.txt')==CHECKSUM,'Prepared dataset version differs; do not silently replace the frozen source'
 p=plan(root);assert digest(p)==OVERLAY;rows=p['rows'];raw=read(root/'manifests/images.json');metadata={r['image_id']:r for r in raw}
 hashes=dict(line.split('  ',1)[::-1] for line in (root/'SHA256SUMS.txt').read_text().splitlines())
 for name,h in hashes.items():
  if name.endswith('.json'):assert sha(inside(root,name))==h,name
 identity=read(root/'provenance/phase2_video_identity.json');mapping,mapping_issues=resolutions(snapshot,identity);mid=identity['original_mid_session_candidates_from_existing_geometry_contract'];remaining=set(mid)-set(mapping.values())
 def component(row):
  if row['domain']=='original_photo':
   return 'unresolved_mid_parent_component' if row['physical_tyre_id'] in remaining and len(mapping)<3 else 'tyre:'+row['physical_tyre_id']
  return 'tyre:'+mapping[row['source_video']] if row['source_video'] in mapping else 'unresolved_mid_parent_component'
 source_rows=[]
 for row in rows:
  source_rows.append(dict(image_id=row['image_id'],domain=row['domain'],class_index=row['class_index'],proxy_class=CLASSES[row['class_index']],
   image_path=row['image_path'],image_sha256=row['image_sha256'],native_width=row['width'],native_height=row['height'],
   confirmed_original_tyre_id=row['physical_tyre_id'] if row['domain']=='original_photo' else mapping.get(row['source_video']),
   provisional_video_tyre_id=metadata[row['image_id']].get('provisional_video_tyre_id'),source_video=row.get('source_video'),
   timestamp_seconds=row.get('timestamp_seconds'),source_frame_index=row.get('source_frame_index'),label_source=row['label_source'],
   canonical_source_split=row['split'],historical_study_split=row['role'],prospective_split=None,leakage_component=component(row),
   point_kind=row['point_kind'],point_optimization_weight=row['point_weight'],new_independent_tyre=False))
 geometry={r['image_id']:r for name in ('geometry/original_human_points.json','geometry/new_point_proposals.json') for r in read(root/name)}
 fingerprint=digest(dict(version=1,feature_code=sha(ROOT/'data_gate/features_gate.py'),collector_code=sha(Path(__file__)),environment=env,grid=[512,384]))
 def feature(row):
  names=[row[k] for k in ('image_path','tyre_mask','tread_mask','ignore_mask')];names.append(row['source_label_path'])
  for name in names:assert sha(inside(root,name))==hashes[name],name
  expected=digest(dict(code=fingerprint,files={n:hashes[n] for n in names},point_record=geometry.get(row['image_id']),row=row))
  target=cache/'features'/(row['image_id']+'.json')
  if target.exists():
   try:
    existing=read(target)
    if existing['input_sha256']==expected and existing['record_sha256']==digest(existing['record']):return existing['record']
   except (ValueError,KeyError,OSError):pass
  record=describe(root,row,geometry)
  # Source previews are part of the interactive report, not additional subjects.
  with Image.open(inside(root,row['image_path'])) as image:
   stream=io.BytesIO();image.convert('RGB').resize((144,192),Image.Resampling.BILINEAR).save(stream,format='JPEG',quality=58)
  record['preview']='data:image/jpeg;base64,'+base64.b64encode(stream.getvalue()).decode()
  atomic(target,dict(input_sha256=expected,record_sha256=digest(record),record=record));return record
 features=[]
 with ThreadPoolExecutor(max_workers=2) as pool:
  for n,future in enumerate(as_completed([pool.submit(feature,row) for row in rows]),1):
   features.append(future.result())
   if n%50==0 or n==len(rows):print('Source image/mask/point descriptors verified',n,'/',len(rows),flush=True)
 features.sort(key=lambda r:r['quality']['image_id']);quality=[r['quality'] for r in features];by_q={r['image_id']:r for r in quality}
 source_by_id={r['image_id']:r for r in source_rows};tyre_rows=[]
 for tid in sorted({r['physical_tyre_id'] for r in rows if r['domain']=='original_photo'}):
  group=[r for r in rows if r['domain']=='original_photo' and r['physical_tyre_id']==tid]
  tyre_rows.append(dict(tyre_id=tid,proxy_class=CLASSES[group[0]['class_index']],original_photo_count=len(group),human_geometry_images=sum(r['point_kind']=='human' for r in group),
   historical_study_split=group[0]['role'],recorded_session_id=group[0]['original_session_id'],mileage_value=None,mileage_unit=None,measured_tread_depth_mm=None,
   label_provenance='source folder/group mileage-proxy convention; not a new odometer or physical-wear measurement',new_independent_tyre=False))
 clips=[];candidate_rows=[]
 for clip in identity['videos']:
  group=[r for r in rows if r['source_video']==clip['video_id']];clips.append(dict(video_id=clip['video_id'],frames=len(group),provisional_tyre_id=clip['provisional_tyre_id'],
   confirmed_original_tyre_id=mapping.get(clip['video_id']),proxy_class='mid',known_existing_tyre=True,known_distinct_from_other_clips=True,new_independent_tyre=False,mapping_status='confirmed_by_intake' if clip['video_id'] in mapping else 'unresolved'))
  for tid in mid:
   photos=[r for r in rows if r['domain']=='original_photo' and r['physical_tyre_id']==tid]
   distance=min((int(by_q[a['image_id']]['dhash64'],16)^int(by_q[b['image_id']]['dhash64'],16)).bit_count() for a in group for b in photos)
   candidate_rows.append(dict(video_id=clip['video_id'],possible_original_tyre_id=tid,min_dhash_hamming_to_original_photos=distance,
    selected_by_confirmed_intake=mapping.get(clip['video_id'])==tid,candidate_allowed_by_confirmed_distinctness=(mapping.get(clip['video_id'])==tid if clip['video_id'] in mapping else tid in remaining),
    identity_evidence='candidate set from source provenance; perceptual similarity is review triage only'))
 neighbors=[];dups=[];hash_groups=defaultdict(list)
 for row in source_rows:hash_groups[row['image_sha256']].append(row)
 for h,rs in hash_groups.items():
  if len(rs)>1:
   for a,b in itertools.combinations(rs,2):dups.append(dict(image_id=a['image_id'],other_image_id=b['image_id'],sha256=h,crosses_historical_split=a['historical_study_split']!=b['historical_study_split']))
 for a in source_rows:
  options=sorted(((int(by_q[a['image_id']]['dhash64'],16)^int(by_q[b['image_id']]['dhash64'],16)).bit_count(),b['image_id']) for b in source_rows if b['image_id']!=a['image_id'])
  for rank,(distance,other) in enumerate(options[:neighbors_to_keep],1):
   b=source_by_id[other];neighbors.append(dict(image_id=a['image_id'],other_image_id=other,rank=rank,dhash_hamming=distance,
    within_review_radius=distance<=radius,crosses_historical_split=a['historical_study_split']!=b['historical_study_split'],
    same_declared_original_tyre=bool(a['confirmed_original_tyre_id'] and a['confirmed_original_tyre_id']==b['confirmed_original_tyre_id']),automatic_identity_assignment=False))
 components=[]
 for comp in sorted({r['leakage_component'] for r in source_rows}):
  group=[r for r in source_rows if r['leakage_component']==comp];roles={r['historical_study_split'] for r in group}
  components.append(dict(leakage_component=comp,images=len(group),confirmed_original_tyres=len({r['confirmed_original_tyre_id'] for r in group if r['confirmed_original_tyre_id']}),
   provisional_video_groups=len({r['source_video'] for r in group if r['source_video']}),historical_split=next(iter(roles)) if len(roles)==1 else None,
   crosses_historical_split=len(roles)>1,constraint_reason='unresolved parent candidates co-assigned conservatively; not an identity equivalence' if comp.startswith('unresolved') else 'confirmed original physical tyre'))
 support=[];descriptors=[]
 for role in ('train','validation','test'):
  for cls,label in enumerate(CLASSES):
   group=[r for r in rows if r['role']==role and r['class_index']==cls];ids={r['image_id'] for r in group};q=[r for r in quality if r['image_id'] in ids]
   subject_ids={r['physical_tyre_id'] for r in group if r['domain']=='original_photo'}
   support.append(dict(historical_split=role,proxy_class=label,images=len(group),confirmed_original_tyres=len(subject_ids),unmapped_existing_video_frames=sum(r['domain']=='new_video_frame' and r['source_video'] not in mapping for r in group),
    human_geometry_images=sum(r['point_kind']=='human' for r in group),weak_geometry_images=sum(r['point_kind']=='polygon_weak' for r in group),is_prospective_blind_cohort=False))
   if q:
    def avg(key):v=[r[key] for r in q if r[key] is not None];return sum(v)/len(v) if v else None
    descriptors.append(dict(historical_split=role,proxy_class=label,images=len(q),confirmed_original_tyres=len(subject_ids),mean_brightness_0_255=avg('brightness_0_255'),
     mean_focus_laplacian_variance=avg('focus_laplacian_variance'),mean_background_brightness_0_255=avg('background_brightness_0_255'),
     mean_foreground_minus_background_brightness=avg('foreground_minus_background_brightness'),scope='image-weighted descriptive capture differences; no causal shortcut or subject-level significance test'))
 prospective=audit_intake(snapshot,policy,rows,quality,intake/'incoming');prospective['issues'].extend(mapping_issues)
 if mapping_issues:prospective['prospective_training_ready']=False;prospective['gaps'].append(dict(code='video_identity_errors',detail='Resolve contradictory identity-resolution inputs'))
 exposure,exposure_receipt=aggregate(cache/'exposure',rows)
 for name,h in prospective['file_hashes'].items():assert sha(inside(intake/'incoming',name))==h,'Native intake file changed during audit'
 run_id=digest(dict(dataset=CHECKSUM,overlay=OVERLAY,code=source_hashes(),intake=input_hashes,intake_media=prospective['file_hashes'],policy=policy,environment=env))
 target=output/'runs'/run_id[:12];target.mkdir(exist_ok=True,parents=True);csvdir=target/'csv';csvdir.mkdir(exist_ok=True);catalog={}
 if (target/'CONTRACT.json').exists():assert read(target/'CONTRACT.json')['run_id']==run_id,'Short directory identity collision'
 def save(name,records,empty=None):
  # JSON cache loading sorts keys; a newly computed record preserves insertion
  # order. Use a canonical union so mixed resumed/fresh records have identical
  # CSV bytes and schema regardless of which units were interrupted.
  records=list(records);fields=sorted({k for r in records for k in r}) if records else empty
  catalog[name]=emit(csvdir,name,records,fields)
 save('source_images',source_rows);save('source_tyres',tyre_rows);save('source_clips',clips);save('video_parent_review',candidate_rows)
 save('image_quality',quality);save('region_geometry',[x for r in features for x in r['regions']]);save('point_geometry',[x for r in features for x in r['points']]);save('point_coverage',[r['point_coverage'] for r in features])
 save('similarity_candidates',neighbors);save('exact_duplicates',dups,['image_id','other_image_id','sha256','crosses_historical_split'])
 save('leakage_components',components);save('historical_class_support',support);save('capture_descriptors',descriptors);save('training_image_exposure',exposure)
 save('intake_tyres',prospective['tyres'],['tyre_id','proxy_class','new_independent_tyre_confirmed','identity_review_status'])
 save('intake_sessions',prospective['sessions'],['session_id','tyre_id','capture_date','domain','device_id','lighting','background_id','view_description','source_video_id','notes'])
 save('intake_images',prospective['images'],['image_id','tyre_id','session_id','proxy_class','proposed_split','relative_path','file_sha256','eligible_after_validation'])
 save('intake_issues',prospective['issues'],ISSUE_FIELDS);save('prospective_support',prospective['support']);save('readiness_gaps',prospective['gaps'],['code','detail'])
 checks=[dict(check='canonical_dataset_hash',passed=True,scope='historical_source',detail=CHECKSUM),dict(check='safe_overlay_hash',passed=True,scope='historical_source',detail=OVERLAY),
  dict(check='historical_physical_component_separation',passed=all(not r['crosses_historical_split'] for r in components),scope='historical_source',detail='Conservative unknown-video constraints included'),
  dict(check='no_cross_split_exact_duplicate',passed=not any(r['crosses_historical_split'] for r in dups),scope='historical_source',detail='Byte SHA-256; similarity alone does not prove identity'),
  dict(check='prospective_training_ready',passed=prospective['prospective_training_ready'],scope='new_intake',detail='All required data, identity, annotation and frozen-protocol conditions must pass')]
 save('gate_checks',checks)
 schema_rows=[]
 for name,info in catalog.items():
  for position,column in enumerate(info['columns'],1):
   kind,units,meaning=field_info(column);schema_rows.append(dict(table=name,column_position=position,column=column,data_type=kind,units=units,meaning=meaning,grain='one '+name+' record'))
 for position,column in enumerate(('table','column_position','column','data_type','units','meaning','grain'),1):schema_rows.append(dict(table='csv_schema',column_position=position,column=column,data_type='text' if column!='column_position' else 'integer',units='metadata',meaning='Column definition, not a new empirical image/subject record',grain='one CSV column'))
 save('csv_schema',schema_rows)
 summary=dict(status='audit_completed',source_integrity_passed=True,historical_split_consistent=all(not r['crosses_historical_split'] for r in components),
  source_images=570,confirmed_existing_tyres=len(tyre_rows),existing_video_clips=len(clips),unresolved_video_mappings=sum(r['mapping_status']=='unresolved' for r in clips),
  containment_disagreement_images=sum(r['tread_outside_tyre_native_pixels']>0 for r in quality),new_intake_images=len(prospective['images']),new_intake_tyre_claims=sum(r['new_independent_tyre_confirmed'] for r in prospective['tyres']),
  prospective_training_ready=prospective['prospective_training_ready'],csv_tables=len(catalog),csv_rows=sum(v['rows'] for v in catalog.values()),
  source_training_observations=exposure_receipt['saved_occurrences'],new_training=False,new_inference=False,new_independent_tyres_collected_by_agent=0)
 atomic(target/'SUMMARY.json',summary);atomic(target/'CSV_SCHEMA.json',catalog);atomic(target/'POLICY.json',policy)
 (target/'input_csv').mkdir(exist_ok=True)
 for name in input_hashes:shutil.copy2(snapshot/name,target/'input_csv'/name)
 atomic(target/'CONTRACT.json',dict(run_id=run_id,dataset_checksum=CHECKSUM,overlay_sha256=OVERLAY,source_hashes=source_hashes(),intake_CSV_hashes=input_hashes,intake_media_hashes=prospective['file_hashes'],policy=policy,environment=env,source_training_evidence=exposure_receipt))
 atomic(target/'GALLERY.json',{r['quality']['image_id']:r['preview'] for r in features})
 atomic(output/'LATEST.json',dict(path=str(target),run_id=run_id));print(json.dumps(summary,indent=2),flush=True)
 return target

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--data-root',default='');parser.add_argument('--intake',default='');parser.add_argument('--output',default='');parser.add_argument('--require-ready',action='store_true');args=parser.parse_args()
 base=Path(args.output or ROOT/'data_gate_outputs');base.mkdir(parents=True,exist_ok=True)
 with FileLock(str(base/'GATE_SESSION.lock'),timeout=1):target=build(args)
 if args.require_ready and not read(target/'SUMMARY.json')['prospective_training_ready']:raise SystemExit('Audit saved; prospective training is not ready. Read readiness_gaps.csv.')

if __name__=='__main__':main()
