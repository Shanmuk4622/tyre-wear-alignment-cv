"""Adversarial prospective-intake checks; all generated fixtures are synthetic."""
from PIL import Image
import copy,csv,json,sys,tempfile
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'data_gate'))
from common_gate import read,sha,atomic,digest,emit
from intake_gate import audit_intake
from run_gate import resolutions
OUT=ROOT/'data_gate_validation';OUT.mkdir(exist_ok=True)
POLICY=read(ROOT/'data_gate/protocol_policy.json')

def write_rows(p,rs,fields=None):
 fields=fields or list(rs[0]);p.parent.mkdir(parents=True,exist_ok=True)
 with open(p,'w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rs)

def fixture(folder):
 folder.mkdir();media=folder/'incoming';media.mkdir();tyres=[];sessions=[];images=[]
 for i,(role,cls) in enumerate((r,c) for r in ('train','validation','test') for c in ('low','mid','high')):
  key=f't{i}';sid=f's{i}';iid=f'i{i}'
  tyres.append(dict(tyre_id=key,existing_project_tyre_id='',new_independent_tyre_confirmed='true',proxy_class=cls,mileage_value='0',mileage_unit='km',mileage_source='synthetic test fixture only',measured_tread_depth_mm='0',depth_method='synthetic test fixture',identity_review_status='confirmed'))
  sessions.append(dict(session_id=sid,tyre_id=key,capture_date='2026-10-29',domain='independent_photo',device_id='test',lighting='synthetic'))
  a=np.random.default_rng(i).integers(0,256,size=(40,30,3),dtype=np.uint8);Image.fromarray(a).save(media/(iid+'.png'))
  mask=np.zeros((40,30),dtype=np.uint8);mask[:,2:28]=255;Image.fromarray(mask).save(media/(iid+'_tyre.png'));mask[:]=0;mask[3:37,7:23]=255;Image.fromarray(mask).save(media/(iid+'_tread.png'))
  atomic(media/(iid+'_mask.json'),dict(image_id=iid,tyre_mask=iid+'_tyre.png',tread_mask=iid+'_tread.png',review_status='reviewed'))
  points=[dict(x=x,y=y) for y in (8,20,32) for x in (8,22)];atomic(media/(iid+'_points.json'),dict(image_id=iid,point_kind='human',review_status='reviewed',points=points))
  images.append(dict(image_id=iid,tyre_id=key,session_id=sid,relative_path=iid+'.png',image_sha256='',original_or_derivative='original',parent_source_id='',proxy_class=cls,split=role,mask_annotation=iid+'_mask.json',point_annotation=iid+'_points.json',label_review_status='reviewed'))
 write_rows(folder/'physical_tyres.csv',tyres);write_rows(folder/'capture_sessions.csv',sessions);write_rows(folder/'images.csv',images)
 return tyres,sessions,images

def main():
 base=Path(tempfile.mkdtemp(prefix='synthetic_',dir=OUT));tyres,sessions,images=fixture(base/'valid');folder=base/'valid';policy=copy.deepcopy(POLICY);policy.update(frozen=True,minimum_distinct_tyres_per_class_role=dict(train=1,validation=1,test=1),legacy_training_scope='none')
 def run():return audit_intake(folder,policy,[])
 def changed(name,rs):write_rows(folder/(name+'.csv'),rs)
 def reject(code):
  result=run();assert not result['prospective_training_ready'];assert any(i['code']==code for i in result['issues']),(code,result['issues'])
 checks=[];ok=run();assert ok['prospective_training_ready'] and len(ok['images'])==9 and all(r['minimum_met'] for r in ok['support']);checks.append('complete_synthetic_cohort_passes_declared_policy')
 before={p.name:sha(p) for p in folder.glob('*.csv')};run();assert before=={p.name:sha(p) for p in folder.glob('*.csv')};checks.append('all_input_CSVs_remain_byte_unchanged')
 assert all(r['mileage_value']=='0' and r['measured_tread_depth_mm']=='0' for r in ok['tyres']);checks.append('observed_zero_measurements_preserved')
 bad=copy.deepcopy(images);bad[0]['relative_path']='../../outside.png';changed('images',bad);reject('invalid_image_path_or_file');changed('images',images);checks.append('path_traversal_rejected')
 bad=copy.deepcopy(images);bad[0]['image_sha256']='0'*64;changed('images',bad);reject('image_checksum_mismatch');changed('images',images);checks.append('incorrect_content_hash_rejected')
 bad=copy.deepcopy(images);bad[0]['split']='test';bad[0]['tyre_id']='t3';bad[0]['session_id']='s3';bad[0]['proxy_class']='low';changed('images',bad);reject('physical_tyre_crosses_splits');changed('images',images);checks.append('one_physical_tyre_cannot_cross_roles')
 bad=copy.deepcopy(images);bad[1]['relative_path']=images[0]['relative_path'];changed('images',bad);reject('duplicate_image_crosses_identity_or_split');changed('images',images);checks.append('duplicate_native_photo_cannot_be_two_subjects')
 bad=copy.deepcopy(images);bad[3]['label_review_status']='pending';changed('images',bad);reject('label_review_not_complete');changed('images',images);checks.append('unreviewed_class_record_is_not_ready')
 point=folder/'incoming/i3_points.json';original=point.read_bytes();p=read(point);p['point_kind']='weak_proposal';atomic(point,p);reject('reserved_human_geometry_missing');point.write_bytes(original);checks.append('weak_points_cannot_supply_reserved_human_truth')
 bad=copy.deepcopy(images);bad[3]['mask_annotation']='';changed('images',bad);reject('reserved_masks_missing');changed('images',images);checks.append('reserved_mask_annotation_requirement_enforced')
 bad=copy.deepcopy(sessions);bad[0]['tyre_id']='t2';changed('capture_sessions',bad);reject('session_tyre_conflict');changed('capture_sessions',sessions);checks.append('session_foreign_key_conflict_rejected')
 p=copy.deepcopy(policy);p['frozen']=False;assert not audit_intake(folder,p,[])['prospective_training_ready'];checks.append('draft_protocol_never_pretends_ready')
 p=copy.deepcopy(policy);p['minimum_distinct_tyres_per_class_role']=None;assert not audit_intake(folder,p,[])['prospective_training_ready'];checks.append('missing_subject_minima_remain_missing')
 b=copy.deepcopy(tyres);b[0]['measured_tread_depth_mm']='-1';changed('physical_tyres',b);reject('invalid_measurement');changed('physical_tyres',tyres);checks.append('negative_measurement_rejected_without_zero_imputation')
 b=copy.deepcopy(tyres);b[3]['existing_project_tyre_id']='old';changed('physical_tyres',b);reject('existing_tyre_claimed_new');changed('physical_tyres',tyres);checks.append('existing_identity_cannot_be_declared_independent_new')
 old=(folder/'physical_tyres.csv').read_bytes();(folder/'physical_tyres.csv').write_text('wrong_header\n',encoding='utf-8');reject('invalid_csv');(folder/'physical_tyres.csv').write_bytes(old);checks.append('empty_malformed_header_is_reported')
 native=folder/'incoming/i0.png';old=native.read_bytes();a=np.asarray(Image.open(native)).copy();a[0,0]=0;Image.fromarray(a).save(native);new=run();assert digest(new['file_hashes'])!=digest(ok['file_hashes']);native.write_bytes(old);checks.append('actual_media_bytes_are_in_the_data_closure')
 source=read(ROOT.parent/'phase2/data/phase2_dataset_v1/provenance/phase2_video_identity.json');resolution=folder/'video_identity_resolutions.csv'
 write_rows(resolution,[dict(video_id='video1',confirmed_original_tyre_id='made_up',review_status='confirmed',evidence_source='test')]);mapped,issues=resolutions(folder,source);assert not mapped and issues;checks.append('invalid_video_parent_never_invents_a_mapping')
 assert run()['prospective_training_ready'];checks.append('all_temporary_changes_restored')
 result=dict(status='passed',checks=checks,check_count=len(checks),synthetic_fixture_directory=str(base),synthetic_fixtures_excluded_from_scientific_outputs=True,network_requests=0,new_training=False,new_inference=False)
 atomic(OUT/'UNIT_TESTS.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
