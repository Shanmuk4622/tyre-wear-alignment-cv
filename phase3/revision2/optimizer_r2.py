"""Same forward objective; optional symmetric conflict projection on shared modules."""
from PIL import Image
import csv,math,os,time,uuid
from datetime import datetime,timezone
from pathlib import Path
import spec
import torch
from torch.nn import functional as F
from common import rng,set_rng
from data import counterfactual
from model import loss_terms
from telemetry import observations,gradients
from metrics_r2 import SCHEMAS
from transactions import canonical
import importlib.util
_s=importlib.util.spec_from_file_location('frozen_nb04_worker',spec.ROOT/'training/worker.py');base=importlib.util.module_from_spec(_s);_s.loader.exec_module(base)

def product(a,b):return sum((x.double()*y.double()).sum() for x,y in zip(a,b))

def project(a,b,enabled,epsilon=1e-20):
 dot=product(a,b);aa=product(a,a);bb=product(b,b);apply=enabled and float(dot)<0
 if apply:
  ca=(dot/bb.clamp_min(epsilon)).float();cb=(dot/aa.clamp_min(epsilon)).float()
  x=[u-ca*v for u,v in zip(a,b)];y=[v-cb*u for u,v in zip(a,b)]
 else:x,y=a,b
 after=product(x,y);xx=product(x,x);yy=product(y,y)
 info=dict(class_norm=math.sqrt(float(aa)),dense_norm=math.sqrt(float(bb)),dot_before=float(dot),cosine_before=float(dot/(aa*bb).sqrt().clamp_min(epsilon)),
  dot_after=float(after),cosine_after=float(after/(xx*yy).sqrt().clamp_min(epsilon)),projection_applied=int(apply),class_projected_norm=math.sqrt(float(xx)),dense_projected_norm=math.sqrt(float(yy)))
 return x,y,info

def attempt_log(folder,update,epoch,cursor,attempt,scale,next_scale,reason):
 p=Path(folder)/'NUMERICAL_ATTEMPTS.csv';fields=['attempt_id','time_utc','candidate_update','epoch','cursor','attempt','amp_scale_before','amp_scale_after','reason','committed_training_observation']
 exists=p.exists()
 with open(p,'a',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields)
  if not exists:w.writeheader()
  w.writerow(dict(attempt_id=uuid.uuid4().hex,time_utc=datetime.now(timezone.utc).isoformat(),candidate_update=update,epoch=epoch,cursor=cursor,attempt=attempt,
   amp_scale_before=scale,amp_scale_after=next_scale,reason=reason,committed_training_observation=0));f.flush();os.fsync(f.fileno())

def update(model,opt,scaler,b,rows,device,cfg,epoch0,cursor,update_number,before_hash,folder,stop):
 before=rng();events=[];begin=time.perf_counter();hooks=base.Activations(model) if (update_number==1 or update_number%10==0) else None
 shared=[(n,p) for n,p in model.named_parameters() if n.split('.')[0] in spec.SHARED];parameters=[p for _,p in shared]
 torch.cuda.reset_peak_memory_stats(device)
 try:
  for attempt in range(9):
   if stop():raise InterruptedError('Stop before incomplete update; last journal transaction retained')
   set_rng(before);opt.zero_grad(set_to_none=True);model.train()
   if hooks:hooks.values=[];hooks.phase='normal'
   t=time.perf_counter()
   with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda' and attempt<8):
    out=model(b['img'],b['guides']);terms=loss_terms(out,b);altered=None;kl=terms['total'].new_zeros(())
    if cursor%4==0:
     if hooks:hooks.phase='background'
     altered=model(counterfactual(b),b['guides']);kl=F.kl_div(altered['probabilities'].clamp_min(1e-7).log(),out['probabilities'].detach(),reduction='batchmean')
    value=terms['total']+.2*kl
    classification=terms['classification']+.2*terms['ordinal']+.2*kl
    dense=terms['segmentation']+2*terms['geometry']+.1*terms['containment']+.1*terms['alignment']
   torch.cuda.synchronize(device);forward=time.perf_counter()-t;finite=bool(torch.isfinite(value));part_finite=False;backward=0.;projection_seconds=0.
   if finite:
    t=time.perf_counter();scale=float(scaler.get_scale())
    a=[g.detach().float()/scale for g in torch.autograd.grad(scaler.scale(classification),parameters,retain_graph=True)]
    d=[g.detach().float()/scale for g in torch.autograd.grad(scaler.scale(dense),parameters,retain_graph=True)]
    scaler.scale(value).backward();scaler.unscale_(opt)
    finite=all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
    part_finite=all(bool(torch.isfinite(g).all()) for g in a+d)
    torch.cuda.synchronize(device);backward=time.perf_counter()-t;t=time.perf_counter()
    info={k:None for k in ('class_norm','dense_norm','dot_before','cosine_before','dot_after','cosine_after','class_projected_norm','dense_projected_norm','shared_gradient_change_norm','head_gradient_max_change')}
    info.update(projection_applied=0,missing_reason=None if part_finite else 'Nonfinite task observation; baseline can proceed only if original total gradient is finite')
    if part_finite:
     x,y,stats=project(a,d,cfg['condition']=='shared_projection',cfg['projection_epsilon']);info.update(stats)
     change=0.;heads={n:p.grad.clone() for n,p in model.named_parameters() if n.split('.')[0] not in spec.SHARED and p.grad is not None}
     if stats['projection_applied'] and finite:
      change=sum(float((p.grad-(u+v)).double().square().sum()) for p,u,v in zip(parameters,x,y))
      for p,u,v in zip(parameters,x,y):p.grad.copy_(u+v)
     info['shared_gradient_change_norm']=math.sqrt(change)
     info['head_gradient_max_change']=max([float((p.grad-heads[n]).abs().max()) for n,p in model.named_parameters() if n in heads] or [0.])
    elif cfg['condition']=='shared_projection':finite=False
    projection_seconds=time.perf_counter()-t
    module_rows=gradients(model,update_number,epoch0+1)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);finite=finite and bool(torch.isfinite(norm))
   if finite:
    t=time.perf_counter();tables,extras,contribution=observations(rows,b,out,altered,update_number,epoch0,cursor,cfg['seed'],before_hash)
    for rs in tables.values():
     for r in rs:r['epoch']=epoch0+1
    tables['gradient_modules']=module_rows
    tables['activation_modules']=[dict(update=update_number,epoch=epoch0+1,**r) for r in hooks.values] if hooks else []
    tables['events'].extend(events);post=math.sqrt(sum(float(p.grad.detach().float().square().sum()) for p in model.parameters() if p.grad is not None))
    capture=cursor==0 and epoch0+1 in cfg['capture_epochs']
    telemetry_seconds=time.perf_counter()-t;t=time.perf_counter();lr=opt.param_groups[0]['lr'];scaler.step(opt);scaler.update();torch.cuda.synchronize(device);optimizer_seconds=time.perf_counter()-t
    row=dict(update=update_number,epoch=epoch0+1,cursor=cursor,batch_images=len(rows),loss=float(value.detach()),**{k:float(terms[k].detach()) for k in ('classification','ordinal','segmentation','geometry','containment','alignment')},
      background_kl=float(kl.detach()),sample_contribution_sum=contribution,loss_reconstruction_error=abs(contribution-float(value.detach())),grad_norm_before_clip=float(norm),grad_norm_after_clip=post,
      clip_coefficient=min(1.,1./(float(norm)+1e-6)),amp_scale_before=scale,amp_scale_after=float(scaler.get_scale()),retries=len(events),lr_used=lr,lr_after_scheduler=lr,next_epoch=epoch0+1,next_cursor=cursor+1,
      checkpoint_before_sha256=before_hash,checkpoint_after_sha256=None,data_seconds=None,forward_seconds=forward,backward_seconds=backward+projection_seconds,optimizer_seconds=optimizer_seconds,telemetry_seconds=telemetry_seconds,
      update_compute_seconds=time.perf_counter()-begin,validation_seconds=None,schedule_epoch0=epoch0,raw_maps_retained=int(capture))
    assert row['loss_reconstruction_error']<2e-5
    tables['train_steps'].append(row)
    tables.update({t:[] for t in SCHEMAS if t not in tables})
    tables['task_gradients'].append(dict(update=update_number,epoch=epoch0+1,cursor=cursor,class_objective=float(classification.detach()),dense_objective=float(dense.detach()),
      shared_parameters=sum(p.numel() for p in parameters),projection_seconds=projection_seconds,finite=int(part_finite),**info))
    return tables,extras if capture else {},row
   opt.zero_grad(set_to_none=True);old=float(scaler.get_scale());new=max(old*.5,1e-8)
   attempt_log(folder,update_number,epoch0+1,cursor,attempt+1,old,new,'Nonfinite objective/gradient or required projected task gradient')
   if attempt==8 or not scaler.is_enabled():set_rng(before);raise FloatingPointError('All same-batch/RNG retries failed; no sample skipped; NUMERICAL_ATTEMPTS.csv retained')
   scaler.update(new_scale=new);events.append(base.event(update_number,epoch0+1,'numerical_retry',attempt=attempt+1,amp_scale_before=old,amp_scale_after=new,message='Same batch/RNG replay'))
 finally:
  if hooks:hooks.close()
