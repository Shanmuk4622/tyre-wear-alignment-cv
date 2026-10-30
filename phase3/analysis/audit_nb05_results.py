"""Recompute NB05 observations from pinned CSV/native evidence, without training."""
from PIL import Image
import csv,json,math,sys
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda' or Path('/kaggle/input').exists()
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'revision2'))
import spec
from transactions import atomic,digest,read,sha,csv_write
from metrics_r2 import SCHEMAS
from common import delta
from data import plan,schedule
from telemetry import augmentation
PUB=ROOT/'revision2_validation/nb05_publication'
OUT=PUB/'analysis'

def rows(folder,table):
 for path in sorted((folder/'csv').glob('epoch_*/'+table+'.csv')):
  with open(path,encoding='utf-8',newline='') as f:yield from csv.DictReader(f)

def f(row,key):return float(row[key]) if row.get(key) not in ('',None) else None

def close(a,b,tol=2e-10):
 if a is None or b in ('',None):assert a is None and b in ('',None),(a,b)
 else:assert abs(float(a)-float(b))<=tol,(a,b,abs(float(a)-float(b)))

def classify(rs):
 conf=np.bincount([3*int(r['label'])+int(r['prediction']) for r in rs],minlength=9).reshape(3,3)
 support=conf.sum(1);human=[f(r,'human_point_mae') for r in rs if f(r,'human_point_mae') is not None]
 frozen=[f(r,'frozen_r1_selection_point_mae') for r in rs if f(r,'frozen_r1_selection_point_mae') is not None]
 value=dict(images=len(rs),accuracy=float(np.trace(conf)/len(rs)),balanced_accuracy_observed=float(np.mean(np.diag(conf)[support>0]/support[support>0])),
  nll=float(np.mean([f(r,'nll') for r in rs])),brier=float(np.mean([f(r,'brier') for r in rs])),mean_two_region_dice=float(np.mean([f(r,'mean_two_region_dice') for r in rs])),
  human_point_images=len(human),human_point_mae=float(np.mean(human)) if human else None,frozen_r1_selection_point_mae=float(np.mean(frozen)) if frozen else None)
 point=value['frozen_r1_selection_point_mae'];value['joint_validation_utility']=.5*value['balanced_accuracy_observed']+.35*value['mean_two_region_dice']+.15*max(0,1-point/.05) if point is not None else None
 return value,conf

def main():
 assert read(PUB/'DOWNLOAD.json')['status']=='passed';OUT.mkdir(exist_ok=True)
 prepared=spec.locate();p=plan(prepared);source={r['image_id']:r for r in p['rows']};expected_order={}
 for e in range(60):
  src,batches=schedule(p,1,e)
  for cursor,indices in enumerate(batches):expected_order[(e+1,cursor)]=[src[i]['image_id'] for i in indices]
 summaries_out=[];training_out=[];class_out=[];tyre_out=[];calibration_out=[];risk_out=[];stability_out=[];resource_out=[]
 cases=[];paired_train={};endpoint_rows={};identities=[];native_input_hashes={};raw_exactness=[]
 for condition in spec.CONDITIONS:
  folder=PUB/'remote'/condition;cfg=read(folder/'CONTRACT.json');cursor=read(folder/'CURSOR.json');schema=read(folder/'SCHEMA.json');manifest=read(folder/'MANIFEST.json')
  assert digest(cfg)==cursor['protocol']==manifest['contract'] and cfg['mode']=='measured_paired_pilot'
  assert cfg['source_checksum']==spec.CHECKSUM and cfg['overlay_sha256']==spec.OVERLAY
  assert cfg['device']==('cuda:0' if condition=='baseline' else 'cuda:1') and cfg['environment']['device']=='Tesla T4'
  assert cfg['sources']==read(PUB/'NOTEBOOK_INSPECTION.json')['source_hashes']
  for name,h in cfg['sources'].items():assert sha(folder/'source'/name)==h
  counts=Counter()
  for name,definition in schema['files'].items():
   with open(folder/name,encoding='utf-8',newline='') as stream:
    reader=csv.DictReader(stream);assert reader.fieldnames==SCHEMAS[definition['family']]
    count=0
    for row in reader:assert row['condition']==condition;count+=1
   assert count==definition['rows'];counts[definition['family']]+=count
  assert dict(counts)==cursor['row_cursors'] and counts['train_steps']==2940 and counts['train_samples']==23520 and counts['eval_predictions']==5066
  assert counts['validation_predictions']==0,'Compatibility table intentionally empty'
  state=torch.load(folder/'state.pt',map_location='cpu',weights_only=False)
  assert sha(folder/'state.pt')==cursor['state_sha256'] and state['protocol']==cursor['protocol']
  assert state['phase']=='completed' and state['updates']==2940 and state['epoch0']==60 and state['cursor']==state['eval_cursor']==0
  assert state['row_cursors']==cursor['row_cursors'] and state['best']==cursor['best']
  assert len(state['losses'])==2940 and len(state['history'])==60
  with open(folder/'STATE_VERSIONS.csv',encoding='utf-8',newline='') as stream:versions=list(csv.DictReader(stream))
  assert len(versions)==3628 and int(versions[-1]['unit'])==cursor['unit']==3627 and versions[-1]['state_sha256']==cursor['state_sha256']
  assert versions[-1]['history_sha256']==state['history_sha256'] and versions[-1]['unit_rows_sha256']==state['unit_rows_sha256']
  initial=read(folder/'artifacts/INITIALIZATION.json');assert initial['parameters']==1372159;identities.append(initial['model_tensor_sha256'])
  native_input_hashes[condition]={n:v['sha256'] for n,v in manifest['files'].items() if n.startswith('artifacts/train_maps/') and n.endswith(('/inputs.npy','/targets.npy'))}
  steps=list(rows(folder,'train_steps'));samples=list(rows(folder,'train_samples'));gradients=list(rows(folder,'task_gradients'))
  assert [int(r['update']) for r in steps]==list(range(1,2941)) and len(gradients)==2940
  sample_groups=defaultdict(list)
  signature=('update','epoch','cursor','slot','image_id','image_sha256','label','augmentation_seed','flip_horizontal','brightness','contrast','blur_applied','blur_radius','background_donor_image_id')
  paired_train[condition]=[{k:r[k] for k in signature} for r in samples]
  for row in samples:
   original=source[row['image_id']];assert original['role']=='train' and original['image_sha256']==row['image_sha256'] and original['class_index']==int(row['label'])
   e,c,slot=int(row['epoch']),int(row['cursor']),int(row['slot']);assert row['image_id']==expected_order[(e,c)][slot]
   for k,v in augmentation(1,e-1,c,slot).items():close(v,row[k])
   assert int(row['correct'])==int(row['label']==row['prediction'])
   sample_groups[int(row['update'])].append(row)
  reconstruction=0.
  for step in steps:
   u=int(step['update']);e=int(step['epoch']);assert e==(u-1)//49+1 and int(step['schedule_epoch0'])==e-1 and int(step['cursor'])==(u-1)%49
   ss=sample_groups[u];assert len(ss)==8 and [int(r['slot']) for r in ss]==list(range(8))
   total=lambda k:sum(float(r[k]) for r in ss)
   loss=total('label_smoothed_ce')/8+.2*total('ordinal_loss')/8+total('seg_bce_numerator')/max(total('seg_bce_denominator'),1)+total('seg_dice_loss')/8
   loss+=2*total('geometry_numerator')/max(total('geometry_weight'),1)+.1*total('containment_numerator')/max(total('containment_denominator'),1)+.1*total('alignment_numerator')/max(total('alignment_denominator'),1)+.2*total('background_kl')/8
   reconstruction=max(reconstruction,abs(loss-float(step['loss'])));assert reconstruction<2e-5
   close(total('objective_contribution'),step['sample_contribution_sum'],1e-8)
   close(state['losses'][u-1],step['loss'],1e-10)
  assert all(int(r['finite'])==1 and float(r['head_gradient_max_change'])==0 for r in gradients)
  conflicts=sum(float(r['dot_before'])<0 for r in gradients);applied=sum(int(r['projection_applied']) for r in gradients)
  assert applied==(0 if condition=='baseline' else conflicts)
  if condition=='shared_projection':
   for r in gradients:
    assert int(r['projection_applied'])==int(float(r['dot_before'])<0)
    assert float(r['dot_after'])>=-1e-6*max(float(r['class_projected_norm'])*float(r['dense_projected_norm']),1e-20)
  for e in range(1,61):
   ss=[r for r in samples if int(r['epoch'])==e];gg=[r for r in gradients if int(r['epoch'])==e];ts=steps[(e-1)*49:e*49]
   training_out.append(dict(condition=condition,epoch=e,training_occurrences=len(ss),training_augmented_accuracy=np.mean([int(r['correct']) for r in ss]),
    mean_batch_objective=np.mean([float(r['loss']) for r in ts]),mean_classification_objective=np.mean([float(r['classification']) for r in ts]),
    observed_conflicts=sum(float(r['dot_before'])<0 for r in gg),applied_projections=sum(int(r['projection_applied']) for r in gg),
    mean_cosine_before=np.mean([float(r['cosine_before']) for r in gg]),mean_clip_coefficient=np.mean([float(r['clip_coefficient']) for r in ts]),
    data_seconds=sum(float(r['data_seconds']) for r in ts),update_compute_seconds=sum(float(r['update_compute_seconds']) for r in ts)))
  predictions=list(rows(folder,'eval_predictions'));by_unit={r['unit']:r for r in predictions};assert len(by_unit)==5066
  groups=defaultdict(list);cohort_checks=0;reconstruction_metrics=0.
  for r in predictions:
   original=source[r['image_id']];assert original['image_sha256']==r['image_sha256'] and original['class_index']==int(r['label']) and r['split']==original['role']
   assert r['variant']=='normal' and int(r['prediction_changed'])==0 and float(r['probability_l1_delta'])==0
   probs=np.array([float(r['probability_'+str(c)]) for c in range(3)]);assert np.isfinite(probs).all() and abs(probs.sum()-1)<1e-6
   assert int(probs.argmax())==int(r['prediction']) and int(r['correct'])==int(r['label']==r['prediction'])
   close(-math.log(max(probs[int(r['label'])],1e-12)),r['nll'])
   close(float(np.square(probs-np.eye(3)[int(r['label'])]).sum()),r['brier'])
   groups[(int(r['epoch']),r['split'],r['endpoint'])].append(r)
  for key,rs in groups.items():
   expected=sorted(r['image_id'] for r in p['rows'] if r['role']==key[1]);assert sorted(r['image_id'] for r in rs)==expected
  assert len(groups)==62
  for r in rows(folder,'eval_summaries'):
   key=(int(r['epoch']),r['split'],r['endpoint']);members=groups[key]
   if r['level']=='tyre':members=[v for v in members if str(v['tyre'])==r['group']]
   metric,conf=classify(members)
   for name,value in metric.items():close(value,r[name])
   if r['level']=='cohort':
    cohort_checks+=1;summaries_out.append(dict(condition=condition,**{k:v for k,v in r.items() if k!='condition'}))
   if r['endpoint'] in ('selected','final'):
    if r['level']=='tyre':tyre_out.append(dict(r))
    if r['level']=='cohort':
     endpoint_rows[(condition,r['endpoint'])]=members
     for cls in range(3):
      tp=int(conf[cls,cls]);fp=int(conf[:,cls].sum()-tp);fn=int(conf[cls].sum()-tp)
      class_out.append(dict(condition=condition,endpoint=r['endpoint'],epoch=r['epoch'],class_index=cls,support=int(conf[cls].sum()),predicted=int(conf[:,cls].sum()),tp=tp,fp=fp,fn=fn,
       recall=tp/(tp+fn) if tp+fn else None,precision=tp/(tp+fp) if tp+fp else None))
     for i in range(10):
      binrows=[v for v in members if min(int(float(v['confidence'])*10),9)==i]
      calibration_out.append(dict(condition=condition,endpoint=r['endpoint'],bin_index=i,images=len(binrows),correct=sum(int(v['correct']) for v in binrows),confidence_sum=sum(float(v['confidence']) for v in binrows)))
     for threshold in np.linspace(0,1,21):
      kept=[v for v in members if float(v['confidence'])>=threshold]
      risk_out.append(dict(condition=condition,endpoint=r['endpoint'],confidence_threshold=threshold,retained=len(kept),total=len(members),coverage=len(kept)/len(members),
        error_rate=1-np.mean([int(v['correct']) for v in kept]) if kept else None,scope='posthoc descriptive confidence thresholds; not a calibrated deployment rule'))
  val=[r for r in summaries_out if r['condition']==condition and r['endpoint']=='epoch_validation'];best=max(val,key=lambda r:float(r['joint_validation_utility']))
  assert int(best['epoch'])==state['best']['epoch'];close(best['joint_validation_utility'],state['best']['score'])
  for n,r in enumerate(val):
   for k in ('balanced_accuracy_observed','mean_two_region_dice','human_point_mae'):close(state['history'][n][k],r[k])
  for image_id in sorted(r['image_id'] for r in p['rows'] if r['role']=='validation'):
   rr=sorted([r for r in predictions if r['image_id']==image_id and r['endpoint']=='epoch_validation'],key=lambda r:int(r['epoch']))
   assert len(rr)==60;stability_out.append(dict(condition=condition,image_id=image_id,tyre=rr[0]['tyre'],label=rr[0]['label'],epochs_correct=sum(int(r['correct']) for r in rr),
    successive_prediction_flips=sum(a['prediction']!=b['prediction'] for a,b in zip(rr,rr[1:])),selected_prediction=rr[state['best']['epoch']-1]['prediction'],mean_true_probability=np.mean([float(r['true_class_probability']) for r in rr])))
  region_groups=defaultdict(list);point_groups=defaultdict(list);point_scalar_rounding=0.
  for r in rows(folder,'eval_regions'):
   tp,fp,fn,tn=(int(r[k]) for k in ('tp','fp','fn','tn'));assert tp+fp+fn+tn==int(r['valid_pixels'])
   close((2*tp+1)/(2*tp+fp+fn+1),r['dice']);close((tp+1)/(tp+fp+fn+1),r['iou']);region_groups[r['unit']].append(r)
  for r in rows(folder,'eval_points'):
   if r['target']!='':
    exact=abs(float(r['prediction'])-float(r['target']));point_scalar_rounding=max(point_scalar_rounding,abs(exact-float(r['absolute_error_width_fraction'])))
    # NumPy 2 scalar subtraction in the frozen recorder can round in FP32;
    # annotation-level MAE instead uses an explicit FP64 target array.
    close(exact,r['absolute_error_width_fraction'],1.2e-7)
   point_groups[r['unit']].append(r)
  for unit,r in by_unit.items():
   assert len(region_groups[unit])==2 and len(point_groups[unit])==6
   close(np.mean([float(v['dice']) for v in region_groups[unit]]),r['mean_two_region_dice'])
   if r['point_kind']=='human':close(np.mean([abs(float(v['prediction'])-float(v['target'])) for v in point_groups[unit]]),r['human_point_mae'])
  # Every raw selected/final test map is checked against saved class evidence and
  # the prepared label masks. No new forward pass or optimizer update is made.
  from data import legacy,HW
  raw_residual=0.;maps_checked=0;max_csv_region_error=0.
  for endpoint in ('selected','final'):
   weights_name=state['best']['weights'] if endpoint=='selected' else 'final_weights.pt'
   weights=torch.load(folder/'artifacts'/weights_name,map_location='cpu',weights_only=False)
   assert weights['protocol']==state['protocol'] and weights['epoch']==(state['best']['epoch'] if endpoint=='selected' else 60)
   if endpoint=='final':assert delta(state['model'],weights['model'])==0
   bias=weights['model']['class_bias'].float().numpy();rr=sorted(endpoint_rows[(condition,endpoint)],key=lambda r:r['image_id'])
   for path in sorted((folder/'artifacts/test_maps'/endpoint).glob('*.npy')):
    start,end=map(int,path.stem[1:].split('_'));array=np.load(path,allow_pickle=False);assert array.shape==(end-start,18,128,96) and np.isfinite(array).all()
    for j,a in enumerate(array):
     r=rr[start+j];gt=legacy.sample(prepared,source[r['image_id']],HW,None,'segformer')[1];valid=gt[2]<.5
     for k,region in enumerate(('tyre','tread')):
      binary=np.repeat(np.repeat(a[k]>=.5,4,0),4,1)&valid;truth=(gt[k]>.5)&valid
      tp=int((binary&truth).sum());fp=int((binary&~truth&valid).sum());fn=int((~binary&truth&valid).sum())
      record=next(x for x in region_groups[r['unit']] if x['region']==region)
      assert [tp,fp,fn]==[int(record[x]) for x in ('tp','fp','fn')]
     assert np.min(a[2:14])>=0 and np.min(a[17])>=0
     close(float(a[17].sum()),1,2e-5)
     residual=float(np.max(np.abs(a[14:17].sum((1,2))+bias-np.array([float(r['logit_'+str(k)]) for k in range(3)]))))
     raw_residual=max(raw_residual,residual);close(residual,r['evidence_residual'],2e-6);maps_checked+=1
   assert maps_checked>=103
  assert maps_checked==206
  resources=list(rows(folder,'resources'));life=[]
  with open(folder/'WORKER_LIFECYCLE.csv',encoding='utf-8',newline='') as stream:life=list(csv.DictReader(stream))
  resource_out.append(dict(condition=condition,worker_processes_started=sum(r['event']=='started_or_resumed' for r in life),
   reported_worker_active_seconds=sum(float(r['process_elapsed_seconds']) for r in life if r['event']=='exited'),
   peak_sampled_cuda_bytes=max(float(r['cuda_peak_bytes']) for r in resources if r['cuda_peak_bytes']),
   peak_sampled_process_rss_bytes=max(float(r['process_rss_bytes']) for r in resources if r['process_rss_bytes']),
   minimum_sampled_free_disk_bytes=min(float(r['working_free_bytes']) for r in resources if r['working_free_bytes']),
   scope='sampled resources; retained worker exit intervals; excludes publisher waiting and forced-kill gaps'))
  cases.append(dict(condition=condition,contract=cursor['protocol'],counts=dict(counts),core_rows=sum(counts.values()),native_state_completed=True,
   validation_cohort_summaries_checked=cohort_checks,selected_epoch=state['best']['epoch'],gradient_conflicts=conflicts,projections=applied,
   max_loss_reconstruction_error=reconstruction,max_point_scalar_FP32_rounding=point_scalar_rounding,test_native_maps_checked=maps_checked,max_additive_logit_residual=raw_residual,environment=cfg['environment']))
  print(condition,'scientific checks passed; conflicts',conflicts,'projections',applied,flush=True)
 assert len(set(identities))==1 and paired_train['baseline']==paired_train['shared_projection'] and native_input_hashes['baseline']==native_input_hashes['shared_projection']
 paired=[]
 for endpoint in ('selected','final'):
  a={r['image_id']:r for r in endpoint_rows[('baseline',endpoint)]};b={r['image_id']:r for r in endpoint_rows[('shared_projection',endpoint)]};assert set(a)==set(b)
  for image_id,x in sorted(a.items()):
   y=b[image_id];paired.append(dict(endpoint=endpoint,image_id=image_id,tyre=x['tyre'],label=x['label'],baseline_epoch=x['epoch'],projection_epoch=y['epoch'],baseline_prediction=x['prediction'],projection_prediction=y['prediction'],
    baseline_correct=x['correct'],projection_correct=y['correct'],baseline_confidence=x['confidence'],projection_confidence=y['confidence'],nll_delta=float(y['nll'])-float(x['nll']),
    dice_delta=float(y['mean_two_region_dice'])-float(x['mean_two_region_dice']),human_point_mae_delta=float(y['human_point_mae'])-float(x['human_point_mae']) if x['human_point_mae'] else None))
 datasets=dict(endpoint_metrics=[r for r in summaries_out if r['endpoint'] in ('selected','final')],validation_trajectory=[r for r in summaries_out if r['endpoint']=='epoch_validation'],
  epoch_training_gradient=training_out,endpoint_class_metrics=class_out,endpoint_tyre_metrics=tyre_out,calibration_reliability=calibration_out,selective_risk=risk_out,
  validation_prediction_stability=stability_out,paired_image_deltas=paired,resource_timing_summary=resource_out)
 for name,records in datasets.items():csv_write(OUT/(name+'.csv'),records,list(records[0]))
 result=dict(status='passed',date='2026-10-29',cases=cases,paired_initialization_sha256=identities[0],paired_samples_augmentation_and_native_inputs_identical=True,
  schema_families=23,core_rows=sum(c['core_rows'] for c in cases),scientific_derivative_tables={n:len(r) for n,r in datasets.items()},
  notebooks_executed_by_audit=0,new_optimizer_updates=0,remote_writes=0,independent_new_tyres=0,
  caveats=['One paired seed; no superiority/generalization inference','Observed test cohort remains development evidence','SQL observation hash chain is not independently replayed from CSV strings; native tail anchors and all published hashes/cursors are checked','Resource samples are not continuous energy or full notebook runtime'])
 atomic(PUB/'SCIENTIFIC_VERIFICATION.json',result);print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))

if __name__=='__main__':main()
