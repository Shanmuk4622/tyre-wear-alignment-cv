"""Separate CPU data/protocol gate; completed training sources stay frozen."""
import csv,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT/'runtime'),str(ROOT/'training'),str(ROOT/'revision2')])
from transactions import atomic,canonical,digest,read,sha,file_meta,csv_write
CHECKSUM='06c6c9d003aff808babf7ebb5786f441e95064eb000f5e5ed5b155d0ee9e7c6a'
OVERLAY='f81a80d20b60ba505d94e31c26615e7e7171a91a8506948be6c392b68107a039'
CLASSES=('low','mid','high')

def locate(value=''):
 if value:return Path(value).resolve()
 if Path('/kaggle/input').exists():
  candidates=[p.parent for p in Path('/kaggle/input').rglob('VERSION.json') if read(p).get('dataset_title')=='Tire Dataset Prepared phase2']
  assert len(candidates)==1,'Attach one expanded prepared Phase2 dataset';return candidates[0]
 return ROOT.parent/'phase2/data/phase2_dataset_v1'

def inside(folder,name):
 assert name and not Path(name).is_absolute() and '..' not in Path(name).parts,'Use a relative path inside the input folder'
 p=(Path(folder)/name).resolve();assert p.is_relative_to(Path(folder).resolve()) and p!=Path(folder).resolve(),'Path leaves the permitted input folder'
 return p

def table(path,required=()):
 with open(path,encoding='utf-8-sig',newline='') as stream:
  r=csv.DictReader(stream);assert r.fieldnames and len(set(r.fieldnames))==len(r.fieldnames),'Missing/duplicate CSV header'
  assert set(required)<=set(r.fieldnames),'Required CSV columns are absent'
  result=list(r);assert all(None not in row and all(v is not None for v in row.values()) for row in result),'Malformed CSV row width'
 return result

def emit(folder,name,records,fields=None):
 records=list(records);assert fields or records,'Empty tables need an explicit schema'
 fields=fields or list(records[0]);assert all(set(r)<=set(fields) for r in records),(name,'unknown columns')
 csv_write(Path(folder)/(name+'.csv'),records,fields)
 return dict(rows=len(records),columns=fields,**file_meta(Path(folder)/(name+'.csv')))

def source_hashes():
 return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted((ROOT/'data_gate').glob('*')) if p.suffix in ('.py','.json')}

def field_info(column):
 flags={'candidate_allowed_by_confirmed_distinctness','new_independent_tyre','known_existing_tyre','known_distinct_from_other_clips','selected_by_confirmed_intake','automatic_identity_assignment','within_review_radius','same_declared_original_tyre','crosses_historical_split','human_eligible','independent_human_truth','exif_transform_applied','eligible_after_validation','new_independent_tyre_claim','new_independent_tyre_confirmed','minimum_met','is_prospective_blind_cohort','passed'}
 if column in flags:return 'boolean','true/false','Explicit recorded or calculated flag; not encoded only by color'
 if column.endswith('sha256'):return 'text','SHA-256 hex','Content identity of the named source or registered native file'
 if column=='dhash64':return 'text','16 hexadecimal characters','64-bit horizontal difference hash; review aid, not physical identity proof'
 if 'hamming' in column or column=='closest_existing_dhash_distance':return 'integer','different hash bits, 0–64','Hamming distance between declared 64-bit difference hashes'
 if column.endswith('_pixels') or column in ('x0','y0','x1_exclusive','y1_exclusive','native_width','native_height','quality_width','quality_height','source_x_pixels','source_y_pixels','width','height'):return 'number','pixels','Coordinates or counts on the explicitly named native/quality grid; boxes have exclusive upper bounds'
 if column.endswith('_0_255') or column=='foreground_minus_background_brightness':return 'number','intensity, 0–255 (difference may be negative)','Stored-orientation fixed-grid RGB/gray descriptor; not a causal or physical measurement'
 if column=='focus_laplacian_variance' or column.startswith('mean_focus_'):return 'number','intensity squared','Variance of a five-point Laplacian on a fixed 512×384 grid; no blur rejection threshold is fitted'
 if column.endswith('_bits'):return 'number','bits','Shannon entropy from 256 gray-intensity bins'
 if 'fraction' in column or column.endswith('_0_1') or column in ('augmented_accuracy','mean_sampled_confidence'):return 'number','fraction','Ratio with the documented population/grid denominator; blank if unavailable'
 if column.endswith('_seconds'):return 'number','seconds','Recorded source-video time, not a new capture timestamp'
 if column in ('mileage_value','mileage_unit','measured_tread_depth_mm','depth_method'):return ('number' if column in ('mileage_value','measured_tread_depth_mm') else 'text'),('mm' if 'depth' in column else 'declared mileage unit'),'Only actual supplied measurements/provenance; unknown values stay blank'
 if column.endswith('_id') or column in ('possible_original_tyre_id','leakage_component','closest_existing_image'):return 'text','identifier','Stable source/intake identifier; unresolved identity stays blank and is not inferred from similarity'
 if column in ('images','frames','rank','point_index','point_records','human_point_records','human_points','confirmed_original_tyres','original_photo_count','human_geometry_images','weak_geometry_images','unmapped_existing_video_frames','sampled_augmented_occurrences','correct_augmented_occurrences','first_update','last_update','provisional_video_groups','confirmed_tyre_records','new_independent_tyre_claims','minimum_required','column_position'):return 'integer','count/index','Count/index for the named table grain; repeated images/frames are not independent tyres'
 if column in ('class_index','exif_orientation'):return 'integer','category code','Class 0/1/2 means low/mid/high mileage proxy; EXIF orientation is recorded without transforming source pixels'
 if column in ('native_aspect_ratio','optimization_weight','point_optimization_weight','mean_sampled_nll'):return 'number','dimensionless','Aspect ratio, frozen supervision weight or mean saved per-occurrence negative log likelihood as named'
 return 'text','text/path/category','Recorded source metadata, explicit status or method; blank means unknown/not applicable'
