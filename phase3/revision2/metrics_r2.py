"""Source-defined raw measurements and full-cohort summaries for the paired pilot."""
from PIL import Image
import math,time
from collections import defaultdict
import numpy as np,torch
import spec
from telemetry import SCHEMAS as BASE,diag,cpu_output,observations
from transactions import canonical

SCHEMAS={k:['condition']+v for k,v in BASE.items()}
SCHEMAS['train_steps']+=['schedule_epoch0','raw_maps_retained']
for table in ('predictions','regions','points','latents','evidence'):
 SCHEMAS['eval_'+table]=['condition']+diag.SCHEMAS[table]+(['frozen_r1_selection_point_mae'] if table=='predictions' else [])
SCHEMAS['task_gradients']=['condition','update','epoch','cursor','class_objective','dense_objective','shared_parameters','class_norm','dense_norm','dot_before','cosine_before','dot_after','cosine_after','projection_applied','class_projected_norm','dense_projected_norm','shared_gradient_change_norm','head_gradient_max_change','projection_seconds','finite','missing_reason']
SCHEMAS['eval_timings']=['condition','epoch','split','endpoint','image_id','load_seconds','forward_seconds','metrics_seconds','total_seconds']
SCHEMAS['eval_summaries']=['condition','epoch','split','endpoint','level','group','images','accuracy','balanced_accuracy_observed','nll','brier','mean_two_region_dice','human_point_images','human_point_mae','frozen_r1_selection_point_mae','joint_validation_utility']
SCHEMAS['eval_class_metrics']=['condition','epoch','split','endpoint','level','group','class_index','support','tp','fp','fn','tn','precision','recall','f1']
SCHEMAS['calibration_bins']=['condition','epoch','split','endpoint','bin_index','lower_inclusive','upper_inclusive_last','images','correct','confidence_sum']

def number(row,key):return float(row[key]) if row.get(key) not in (None,'') else None

def summaries(records,epoch,split,endpoint):
 groups=[('cohort','all',records)]+[('tyre',str(g),[r for r in records if r['tyre']==g]) for g in sorted({r['tyre'] for r in records},key=str)]
 tables={t:[] for t in SCHEMAS};utility=None
 for level,group,rs in groups:
  conf=np.bincount([3*int(r['label'])+int(r['prediction']) for r in rs],minlength=9).reshape(3,3);support=conf.sum(1);seen=support>0
  ba=float((np.diag(conf)[seen]/support[seen]).mean());human=[number(r,'human_point_mae') for r in rs if number(r,'human_point_mae') is not None]
  frozen=[number(r,'frozen_r1_selection_point_mae') for r in rs if number(r,'frozen_r1_selection_point_mae') is not None]
  dice=float(np.mean([float(r['mean_two_region_dice']) for r in rs]));point=float(np.mean(frozen)) if frozen else None
  score=.5*ba+.35*dice+.15*max(0,1-point/.05) if point is not None else None
  if level=='cohort':utility=score
  tables['eval_summaries'].append(dict(epoch=epoch,split=split,endpoint=endpoint,level=level,group=group,images=len(rs),accuracy=float(np.trace(conf)/len(rs)),balanced_accuracy_observed=ba,
   nll=float(np.mean([float(r['nll']) for r in rs])),brier=float(np.mean([float(r['brier']) for r in rs])),mean_two_region_dice=dice,human_point_images=len(human),human_point_mae=float(np.mean(human)) if human else None,
   frozen_r1_selection_point_mae=point,joint_validation_utility=score))
  for c in range(3):
   tp=int(conf[c,c]);fp=int(conf[:,c].sum()-tp);fn=int(conf[c].sum()-tp);tn=len(rs)-tp-fp-fn
   tables['eval_class_metrics'].append(dict(epoch=epoch,split=split,endpoint=endpoint,level=level,group=group,class_index=c,support=tp+fn,tp=tp,fp=fp,fn=fn,tn=tn,
    precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None))
 for i in range(10):
  rs=[r for r in records if min(int(float(r['confidence'])*10),9)==i]
  tables['calibration_bins'].append(dict(epoch=epoch,split=split,endpoint=endpoint,bin_index=i,lower_inclusive=i/10,upper_inclusive_last=(i+1)/10,images=len(rs),correct=sum(int(r['correct']) for r in rs),confidence_sum=sum(float(r['confidence']) for r in rs)))
 return tables,utility

def evaluate_one(model,row,root,device,epoch,split,endpoint,condition,index,seed):
 from data import batch
 begin=time.perf_counter();b=batch(root,[row],device);torch.cuda.synchronize(device);loaded=time.perf_counter()-begin;t=time.perf_counter()
 with torch.inference_mode(),torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):out=model(b['img'],b['guides'])
 torch.cuda.synchronize(device);forward=time.perf_counter()-t;t=time.perf_counter();outputs=cpu_output(out)
 job=dict(key=condition,seed=seed,endpoint=endpoint,epoch=epoch)
 records=diag.records(dict(row,role=split),job,'normal',outputs,0,b['masks'][0].cpu().numpy(),outputs);tables={k:[] for k in SCHEMAS}
 pred=records[0];pred['frozen_r1_selection_point_mae']=float((out['points'][0]-b['points'][0]).abs().mean()) if row['point_kind']=='human' else None
 for name,rs in zip(('predictions','regions','points','latents','evidence'),([pred],*records[1:])):
  for r in rs:r['unit']=f'{condition}/{endpoint}/e{epoch:04d}/i{index:05d}'
  tables['eval_'+name]=rs
 raw=np.concatenate([outputs['masks'],outputs['concepts'],outputs['contributions'],outputs['gate']],axis=1)[0]
 tables['eval_timings'].append(dict(epoch=epoch,split=split,endpoint=endpoint,image_id=row['image_id'],load_seconds=loaded,forward_seconds=forward,metrics_seconds=time.perf_counter()-t,total_seconds=time.perf_counter()-begin))
 return tables,pred,raw,b
