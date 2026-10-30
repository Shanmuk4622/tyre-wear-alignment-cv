"""Measured augmented-batch evidence, sufficient statistics and explicit schemas."""
from PIL import Image
import importlib.util,math,subprocess,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('tread_diagnostic_metric_contract',ROOT/'diagnostic/metrics.py')
diag=importlib.util.module_from_spec(spec);spec.loader.exec_module(diag)
ID=['update','epoch','cursor','slot','image_id','image_sha256','tyre_id','source_video','domain','label']
SCHEMAS={
 'train_steps':['update','epoch','cursor','batch_images','loss','classification','ordinal','segmentation','geometry','containment','alignment','background_kl',
   'sample_contribution_sum','loss_reconstruction_error','grad_norm_before_clip','grad_norm_after_clip','clip_coefficient','amp_scale_before','amp_scale_after','retries',
   'lr_used','lr_after_scheduler','next_epoch','next_cursor','checkpoint_before_sha256','checkpoint_after_sha256',
   'data_seconds','forward_seconds','backward_seconds','optimizer_seconds','telemetry_seconds','update_compute_seconds','validation_seconds'],
 'train_samples':ID+['prediction','correct','confidence','probability_0','probability_1','probability_2','logit_0','logit_1','logit_2','nll','brier','entropy_nats',
   'label_smoothed_ce','ordinal_loss','seg_bce_numerator','seg_bce_denominator','seg_dice_loss','geometry_numerator','geometry_weight',
   'containment_numerator','containment_denominator','alignment_numerator','alignment_denominator','background_kl','objective_contribution',
   'mean_two_region_dice','mean_box_iou','point_kind','human_point_mae','annotation_point_mae','point_output_dtype',
   'augmentation_seed','flip_horizontal','brightness','contrast','blur_applied','blur_radius','background_donor_image_id','background_applied',
   'background_probability_0','background_probability_1','background_probability_2','background_logit_0','background_logit_1','background_logit_2','checkpoint_before_sha256'],
 'region_stats':ID+['region','tp','fp','fn','tn','valid_pixels','ignored_pixels','dice','iou','box_iou']+[f'{p}_{k}' for p in ('pred_box','target_box') for k in ('x0','y0','x1_exclusive','y1_exclusive')],
 'geometry_points':ID+['point_index','guide_fraction','side','prediction','optimization_target','target','source_target','source_target_index','point_kind','point_weight','human_eligible',
   'signed_error_width_fraction','absolute_error_width_fraction','squared_error_width_fraction','absolute_error_native_pixels','native_width','native_height'],
 'latent_metrics':ID+['latent','minimum','maximum','mean','std','gated_mass'],
 'evidence_metrics':ID+['class_index','logit','bias','sum_contribution','positive_sum','negative_sum','inside_true_tread_sum','outside_true_tread_sum','evidence_max','evidence_min'],
 'class_confusion':['update','epoch','label','prediction','count'],
 'gradient_modules':['update','epoch','module','parameters_with_gradient','gradient_norm','maximum_absolute_gradient','finite'],
 'activation_modules':['update','epoch','phase','module','elements','mean','std','minimum','maximum','l2_norm','finite'],
 'validation_predictions':ID+['prediction','correct','probability_0','probability_1','probability_2','nll','brier','mean_two_region_dice','human_point_mae'],
 'epoch_summaries':['update','epoch','batches','images','objective_sum','mean_batch_objective','classification_correct','classification_images','classification_accuracy','learning_rate_next_epoch','engineering_only'],
 'resources':['update','epoch','scope','process_rss_bytes','container_used_bytes','container_limit_bytes','container_source','cuda_allocated_bytes','cuda_peak_bytes','working_free_bytes','gpu_uuid','gpu_name','gpu_utilization_percent','gpu_memory_used_mib','gpu_memory_total_mib','gpu_power_watts','gpu_temperature_c','gpu_mapping','missing_reason'],
 'events':['update','epoch','time_utc','kind','attempt','amp_scale_before','amp_scale_after','seconds','message']
}

def augmentation(seed,epoch,cursor,slot):
 value=seed*10000019+epoch*100003+cursor*97+slot
 rng=np.random.default_rng(value);flip=rng.random()<.5;brightness=float(rng.uniform(.8,1.2));contrast=float(rng.uniform(.85,1.15))
 blur=rng.random()<.2;radius=float(rng.uniform(.2,.8)) if blur else None
 return dict(augmentation_seed=value,flip_horizontal=int(flip),brightness=brightness,contrast=contrast,blur_applied=int(blur),blur_radius=radius)

def cpu_output(out):
 return {k:v.detach().float().cpu().numpy().copy() for k,v in out.items() if k in ('masks','points','concepts','gate','contributions','logits','probabilities','bias')}

def decomposition(out,b,altered):
 """Numerators/denominators reconstruct the frozen r1 batch objective."""
 with torch.no_grad():
  p=out['masks'].float();n,_,h,w=b['masks'][:,:2].shape
  y=b['masks'][:,:2].reshape(n,2,h//4,4,w//4,4).mean((3,5))
  valid=(1-b['masks'][:,2:3]).reshape(n,1,h//4,4,w//4,4).amin((3,5))
  bce=(F.binary_cross_entropy_with_logits(out['seg_logits'].float(),y,reduction='none')*valid).sum((1,2,3))
  vd=valid.sum((1,2,3));dice=1-((2*(p*y*valid).sum((-2,-1))+1)/((p*valid).sum((-2,-1))+(y*valid).sum((-2,-1))+1)).mean(1)
  ce=F.cross_entropy(out['logits'].float(),b['labels'],reduction='none',label_smoothing=.05)
  prob=out['probabilities'].float();truth=F.one_hot(b['labels'],3).float()
  cdf=torch.stack([prob[:,0],prob[:,0]+prob[:,1]],-1);target=torch.stack([truth[:,0],truth[:,0]+truth[:,1]],-1)
  ordinal=(cdf-target).square().mean(1)
  geo=F.smooth_l1_loss(out['points'].float(),b['points'].float(),reduction='none',beta=.02).mean(1)*b['point_weight']
  containment=(F.relu(p[:,1:2]-p[:,:1])*valid).sum((1,2,3))
  idx=(out['guides']*(p.shape[-2]-1)).round().long()
  rows=torch.stack([p[i,1,idx[i]] for i in range(n)]);row_valid=torch.stack([valid[i,0,idx[i]] for i in range(n)])
  reference=torch.stack([y[i,1,idx[i]] for i in range(n)])
  usable=((reference[...,0]<.1)&(reference[...,-1]<.1)&(reference.sum(-1)>2)).float()
  x=torch.linspace(0,1,p.shape[-1],device=p.device)[None,None];q=out['points'].reshape(n,3,2)
  interval=((x-q[...,:1])/.015).sigmoid()*((q[...,1:]-x)/.015).sigmoid()
  alignment=((interval-rows.detach()).abs()*row_valid*usable[...,None]).sum((1,2))
  ad=(row_valid*usable[...,None]).sum((1,2))
  kl=F.kl_div(altered['probabilities'].clamp_min(1e-7).log(),out['probabilities'].detach(),reduction='none').sum(1) if altered is not None else torch.zeros(n,device=p.device)
  contribution=ce/n+.2*ordinal/n+bce/(2*vd.sum()).clamp_min(1)+dice/n+2*geo/b['point_weight'].sum().clamp_min(1)+.1*containment/vd.sum().clamp_min(1)+.1*alignment/ad.sum().clamp_min(1)+.2*kl/n
  values=dict(label_smoothed_ce=ce,ordinal_loss=ordinal,seg_bce_numerator=bce,seg_bce_denominator=2*vd,seg_dice_loss=dice,
     geometry_numerator=geo,geometry_weight=b['point_weight'],containment_numerator=containment,containment_denominator=vd,
     alignment_numerator=alignment,alignment_denominator=ad,background_kl=kl,objective_contribution=contribution)
  return [{key:float(value[i]) for key,value in values.items()} for i in range(n)]

def observations(rows,b,out,altered,update,epoch,cursor,seed,before_hash):
 outputs=cpu_output(out);targets=b['masks'].detach().cpu().numpy();points=b['points'].detach().cpu().numpy()
 portions=decomposition(out,b,altered);tables={t:[] for t in SCHEMAS}
 job=dict(key='engineering-fixture',seed=seed,endpoint='engineering',epoch=epoch)
 for i,source in enumerate(rows):
  aug=augmentation(seed,epoch,cursor,i);row=dict(source,points=points[i].tolist() if source['points'] is not None else None)
  p,regions,geometry,latents,evidence=diag.records(row,job,'augmented_pre_update',outputs,i,targets[i],outputs)
  identity=dict(update=update,epoch=epoch,cursor=cursor,slot=i,image_id=source['image_id'],image_sha256=source['image_sha256'],tyre_id=source['physical_tyre_id'],
                source_video=source.get('source_video'),domain=source['domain'],label=source['class_index'])
  selected={key:p[key] for key in ['prediction','correct','confidence','nll','brier','entropy_nats','mean_two_region_dice','mean_box_iou','point_kind','human_point_mae','annotation_point_mae']}
  selected.update({f'probability_{c}':p[f'probability_{c}'] for c in range(3)});selected.update({f'logit_{c}':p[f'logit_{c}'] for c in range(3)})
  background={f'background_{kind}_{c}':float(altered[key][i,c].detach()) if altered is not None else None for kind,key in [('probability','probabilities'),('logit','logits')] for c in range(3)}
  tables['train_samples'].append(dict(identity,**selected,**portions[i],**aug,**background,point_output_dtype=str(out['points'].dtype),
    background_donor_image_id=rows[(i-1)%len(rows)]['image_id'] if altered is not None else None,background_applied=int(altered is not None),checkpoint_before_sha256=before_hash))
  for r in regions:tables['region_stats'].append(dict(identity,**{k:r[k] for k in SCHEMAS['region_stats'] if k not in ID}))
  for k,r in enumerate(geometry):
   source_index=(1,0,3,2,5,4)[k] if aug['flip_horizontal'] else k
   target=float(points[i,k]) if source['points'] is not None else None
   error=float(outputs['points'][i,k])-target if target is not None else None
   tables['geometry_points'].append(dict(identity,point_index=k,guide_fraction=row['guide_fractions'][k//2],side='left' if k%2==0 else 'right',prediction=float(outputs['points'][i,k]),
     optimization_target=float(points[i,k]),target=target,source_target=source['points'][source_index] if source['points'] is not None else None,source_target_index=source_index,
     point_kind=source['point_kind'],point_weight=source['point_weight'],human_eligible=int(source['point_kind']=='human'),
     signed_error_width_fraction=error,absolute_error_width_fraction=abs(error) if error is not None else None,squared_error_width_fraction=error**2 if error is not None else None,
     absolute_error_native_pixels=abs(error)*(source['width']-1) if error is not None else None,native_width=source['width'],native_height=source['height']))
  for table,records in (('latent_metrics',latents),('evidence_metrics',evidence)):
   for r in records:tables[table].append(dict(identity,**{k:r[k] for k in SCHEMAS[table] if k not in ID}))
 for y in range(3):
  for p in range(3):tables['class_confusion'].append(dict(update=update,epoch=epoch,label=y,prediction=p,count=sum(r['label']==y and r['prediction']==p for r in tables['train_samples'])))
 raw=np.concatenate([outputs['masks'],outputs['concepts'],outputs['contributions'],outputs['gate']],axis=1)
 extras={'inputs.npy':b['img'].detach().float().cpu().numpy().copy(),'targets.npy':targets.astype(np.uint8),'outputs.npy':raw}
 return tables,extras,float(sum(r['objective_contribution'] for r in portions))

def gradients(model,update,epoch):
 grouped={}
 for name,p in model.named_parameters():
  if p.grad is not None:grouped.setdefault(name.split('.')[0],[]).append(p.grad.detach().float())
 result=[]
 for module,values in grouped.items():
  result.append(dict(update=update,epoch=epoch,module=module,parameters_with_gradient=len(values),gradient_norm=float(torch.sqrt(sum(v.square().sum() for v in values))),
      maximum_absolute_gradient=max(float(v.abs().max()) for v in values),finite=int(all(bool(torch.isfinite(v).all()) for v in values))))
 return result

def resources(device,update,epoch,working,memory):
 import shutil
 try:
  import psutil
  rss=psutil.Process().memory_info().rss
 except ImportError:rss=None
 m=memory();base=dict(update=update,epoch=epoch,scope='completed_update',process_rss_bytes=rss,container_used_bytes=m.get('used'),container_limit_bytes=m.get('limit'),container_source=m['source'],
   cuda_allocated_bytes=torch.cuda.memory_allocated(device) if device.type=='cuda' else None,cuda_peak_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
   working_free_bytes=shutil.disk_usage(working).free,gpu_uuid=None,gpu_name=None,gpu_utilization_percent=None,gpu_memory_used_mib=None,gpu_memory_total_mib=None,gpu_power_watts=None,gpu_temperature_c=None,gpu_mapping=None,missing_reason=None)
 if update not in (1,3):return [dict(base,missing_reason='nvidia-smi sampling scheduled at updates 1 and 3 only')]
 try:
  command=['nvidia-smi','--query-gpu=uuid,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu','--format=csv,noheader,nounits']
  completed=subprocess.run(command,capture_output=True,text=True,timeout=5,check=True)
  import csv,io
  records=[];target=str(getattr(torch.cuda.get_device_properties(device),'uuid','')) if device.type=='cuda' else ''
  for row in csv.reader(io.StringIO(completed.stdout)):
   row=[v.strip() for v in row]
   if len(row)!=7:continue
   def numeric(s):
    try:return float(s)
    except ValueError:return None
   records.append(dict(base,gpu_uuid=row[0],gpu_name=row[1],gpu_utilization_percent=numeric(row[2]),gpu_memory_used_mib=numeric(row[3]),gpu_memory_total_mib=numeric(row[4]),
     gpu_power_watts=numeric(row[5]),gpu_temperature_c=numeric(row[6]),gpu_mapping='selected_device' if target and row[0] in target else 'other_or_unresolved_device',
     missing_reason='unsupported numeric fields are blank; instantaneous sample, not energy integration'))
  return records or [dict(base,missing_reason='nvidia-smi returned no usable rows')]
 except (OSError,subprocess.SubprocessError) as exc:return [dict(base,missing_reason='nvidia-smi unavailable: '+type(exc).__name__)]
