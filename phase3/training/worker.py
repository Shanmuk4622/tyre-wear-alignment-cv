"""Three-update engineering gate using the real TREAD-ONE/data/AMP recipe.

This cannot start a measured 60-epoch run. Frozen r1 model/checkpoints are never
updated; each gate creates a separately identified disposable engineering model.
"""
from PIL import Image
import argparse,json,math,os,shutil,signal,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1]
assert Path(sys.prefix).name=='cv_conda' or Path('/kaggle/input').exists()
sys.path.insert(0,str(ROOT/'runtime'))
import numpy as np,torch
from torch.nn import functional as F
from filelock import FileLock
from common import deterministic,environment,cpu,rng,set_rng,memory
from data import batch,counterfactual,plan,schedule
from model import TreadOne,loss_terms
import importlib.util
spec=importlib.util.spec_from_file_location('frozen_tread_worker',ROOT/'runtime/worker.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
from transactions import Store,atomic,canonical,digest,read,sha
from telemetry import SCHEMAS,observations,gradients,resources,cpu_output
STOP=False
def stop(*_):
 global STOP
 STOP=True
def now():return datetime.now(timezone.utc).isoformat()
def sync(device):
 if device.type=='cuda':torch.cuda.synchronize(device)
def event(update,epoch,kind,**kw):
 return dict(update=update,epoch=epoch,time_utc=now(),kind=kind,attempt=kw.get('attempt'),amp_scale_before=kw.get('amp_scale_before'),
             amp_scale_after=kw.get('amp_scale_after'),seconds=kw.get('seconds'),message=kw.get('message'))

class Activations:
 def __init__(self,model):
  self.phase='normal';self.values=[];self.handles=[]
  for name in ('stem','s1','s2','s3','s4','context','d1','point_features'):
   def hook(module,args,value,name=name):
    v=value.detach().float()
    self.values.append(dict(phase=self.phase,module=name,elements=v.numel(),mean=float(v.mean()),std=float(v.std(unbiased=False)),
       minimum=float(v.min()),maximum=float(v.max()),l2_norm=float(v.square().sum().sqrt()),finite=int(bool(torch.isfinite(v).all()))))
   self.handles.append(getattr(model,name).register_forward_hook(hook))
 def close(self):
  for h in self.handles:h.remove()

def instrumented_update(model,opt,scaler,b,rows,device,update,epoch,cursor,seed,before_hash):
 before=rng();retry_events=[];start=time.perf_counter();hooks=Activations(model)
 if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
 try:
  for attempt in range(9):
   if STOP:raise InterruptedError('Stop before incomplete update; prior generation remains the recovery cursor.')
   set_rng(before);opt.zero_grad(set_to_none=True);model.train();hooks.values=[];hooks.phase='normal'
   t=time.perf_counter()
   with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda' and attempt<8):
    out=model(b['img'],b['guides']);terms=loss_terms(out,b);kl=terms['total'].new_zeros(());altered=None
    if cursor%4==0:
     hooks.phase='background';altered=model(counterfactual(b),b['guides'])
     kl=F.kl_div(altered['probabilities'].clamp_min(1e-7).log(),out['probabilities'].detach(),reduction='batchmean')
    value=terms['total']+.2*kl
   sync(device);forward=time.perf_counter()-t;finite=bool(torch.isfinite(value));backward=0.;raw_norm=float('nan')
   if finite:
    t=time.perf_counter();scaler.scale(value).backward();scaler.unscale_(opt)
    module_rows=gradients(model,update,epoch)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);finite=bool(torch.isfinite(norm));raw_norm=float(norm)
    sync(device);backward=time.perf_counter()-t
   if finite:
    scale=float(scaler.get_scale());t=time.perf_counter()
    # Capture actual pre-update outputs before parameters/bias are modified.
    tables,extras,contribution=observations(rows,b,out,altered,update,epoch,cursor,seed,before_hash)
    activation_rows=[dict(update=update,epoch=epoch,**r) for r in hooks.values]
    post_norm=math.sqrt(sum(float(p.grad.detach().float().square().sum()) for p in model.parameters() if p.grad is not None))
    telemetry_time=time.perf_counter()-t;t=time.perf_counter();lr=opt.param_groups[0]['lr']
    scaler.step(opt);scaler.update();sync(device);optimizer=time.perf_counter()-t
    tables['gradient_modules']=module_rows;tables['activation_modules']=activation_rows
    tables['events'].extend(retry_events)
    scalar=dict(update=update,epoch=epoch,cursor=cursor,batch_images=len(rows),loss=float(value.detach()),
      **{k:float(terms[k].detach()) for k in ('classification','ordinal','segmentation','geometry','containment','alignment')},background_kl=float(kl.detach()),
      sample_contribution_sum=contribution,loss_reconstruction_error=abs(contribution-float(value.detach())),
      grad_norm_before_clip=raw_norm,grad_norm_after_clip=post_norm,clip_coefficient=min(1.,1./(raw_norm+1e-6)),amp_scale_before=scale,
      amp_scale_after=float(scaler.get_scale()),retries=len(retry_events),lr_used=lr,lr_after_scheduler=None,next_epoch=None,next_cursor=None,
      checkpoint_before_sha256=before_hash,checkpoint_after_sha256=None,data_seconds=None,forward_seconds=forward,backward_seconds=backward,
      optimizer_seconds=optimizer,telemetry_seconds=telemetry_time,update_compute_seconds=time.perf_counter()-start)
    assert scalar['loss_reconstruction_error']<2e-5,('Loss decomposition mismatch',scalar['loss_reconstruction_error'])
    tables['train_steps'].append(scalar)
    return tables,extras,scalar
   opt.zero_grad(set_to_none=True)
   if attempt==8 or not scaler.is_enabled():
    set_rng(before);raise FloatingPointError('Nonfinite update; no sample skipped and prior generation is preserved.')
   scale=float(scaler.get_scale());scaler.update(new_scale=max(scale*.5,1e-8))
   retry_events.append(event(update,epoch,'numerical_retry',attempt=attempt+1,amp_scale_before=scale,amp_scale_after=float(scaler.get_scale()),message='Same batch/RNG replay'))
 finally:hooks.close()

def validate_images(model,rows,root,device,update,epoch):
 model.eval();result=[]
 with torch.inference_mode():
  for i,row in enumerate(rows):
   b=batch(root,[row],device)
   with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):out=model(b['img'],b['guides'])
   # The same sufficient-statistic decoder; no training augmentation here.
   import telemetry
   rec,*_=telemetry.diag.records(row,dict(key='engineering',seed=0,endpoint='engineering',epoch=epoch),'validation',cpu_output(out),0,b['masks'][0].cpu().numpy(),cpu_output(out))
   p=rec['probabilities'] if 'probabilities' in rec else [rec[f'probability_{c}'] for c in range(3)]
   result.append(dict(update=update,epoch=epoch,cursor=None,slot=i,image_id=row['image_id'],image_sha256=row['image_sha256'],tyre_id=row['physical_tyre_id'],source_video=row.get('source_video'),domain=row['domain'],label=row['class_index'],
     prediction=rec['prediction'],correct=int(rec['prediction']==row['class_index']),probability_0=p[0],probability_1=p[1],probability_2=p[2],nll=rec['nll'],brier=rec['brier'],
     mean_two_region_dice=rec['mean_two_region_dice'],human_point_mae=rec['human_point_mae']))
 model.train();return result

def capture(state,model,opt,scaler,scheduler):
 state.update(model=cpu(model.state_dict()),optimizer=cpu(opt.state_dict()),scaler=scaler.state_dict(),scheduler=scheduler.state_dict(),rng=rng())
 return state

def restore(state,model,opt,scaler,scheduler,protocol):
 assert state['protocol']==protocol
 model.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler']);scheduler.load_state_dict(state['scheduler'])
 if state['updates']>0:opt._opt_called=True
 set_rng(state['rng'])

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--folder',required=True)
 parser.add_argument('--stop-after',type=int,default=3);parser.add_argument('--fault',default='');parser.add_argument('--fault-update',type=int,default=2)
 parser.add_argument('--reference',action='store_true');args=parser.parse_args()
 config=read(args.config);assert config['mode']=='engineering_gate' and config['updates']==3 and config['batches_per_epoch']==2
 assert 0<=args.stop_after<=3,'This gate cannot launch a full training study.'
 device=torch.device(config['device']);deterministic(config['seed'])
 for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
 root=Path(config['data_root']);assert sha(root/'SHA256SUMS.txt')=='06c6c9d003aff808babf7ebb5786f441e95064eb000f5e5ed5b155d0ee9e7c6a','Prepared checksum list changed'
 p=plan(root);assert digest(p)==config['overlay_sha256']
 for name,h in config['dataset_metadata_hashes'].items():assert sha(root/name)==h,name
 for name,h in config['source_hashes'].items():assert sha(ROOT/name)==h,'Gate source drift: '+name
 assert environment(device)==config['environment']
 contract={k:v for k,v in config.items() if k!='data_root'}
 store=Store(args.folder,contract,SCHEMAS);protocol=digest(contract)
 hashes=dict(line.split('  ',1)[::-1] for line in (root/'SHA256SUMS.txt').read_text().splitlines());verified=set()
 model,opt,scaler,scheduler=legacy.components(device)
 state=store.state()
 if state is None:
  state=dict(protocol=protocol,updates=0,next_epoch=1,next_cursor=0,partial=dict(batches=0,images=0,loss_sum=0.,correct=0),losses=[],row_cursors={t:0 for t in SCHEMAS},engineering_only=True,environment=config['environment'])
  store.commit(capture(state,model,opt,scaler,scheduler),{})
 else:restore(state,model,opt,scaler,scheduler,protocol)
 val=[]
 for cls in (0,2):
  options=sorted([r for r in p['rows'] if r['role']=='validation' and r['class_index']==cls],key=lambda r:(r['point_kind']!='human',r['image_id']))
  val.append(options[0])
 for row in val:
  for key in ('image_path','tyre_mask','tread_mask','ignore_mask'):
   name=row[key];assert sha(root/name)==hashes[name];verified.add(name)
 try:
  with FileLock(str(Path(args.folder)/'WRITER.lock'),timeout=1):
   while state['updates']<args.stop_after:
    if STOP:raise InterruptedError('Catchable stop at completed-update cursor')
    epoch,cursor=state['next_epoch'],state['next_cursor'];update=state['updates']+1
    source_rows,order=schedule(p,config['seed'],epoch);chosen=[source_rows[i] for i in order[cursor][:8]]
    assert len(chosen)==8 and all(r['role']=='train' for r in chosen)
    t=time.perf_counter()
    for row in chosen:
     for key in ('image_path','tyre_mask','tread_mask','ignore_mask'):
      name=row[key]
      if name not in verified:assert sha(root/name)==hashes[name];verified.add(name)
    b=batch(root,chosen,device,config['seed'],epoch,cursor);sync(device);load=time.perf_counter()-t
    before_hash=store.chain[-1]['files']['state.pt']['sha256']
    if args.reference:
     value=legacy.update(model,opt,scaler,b,cursor)
     tables={t:[] for t in SCHEMAS};extras=None;scalar=dict(loss=value['loss'])
    else:tables,extras,scalar=instrumented_update(model,opt,scaler,b,chosen,device,update,epoch,cursor,config['seed'],before_hash)
    state['updates']=update;state['losses'].append(scalar['loss']);partial=state['partial']
    partial['batches']+=1;partial['images']+=8;partial['loss_sum']+=scalar['loss']
    partial['correct']+=sum(r['correct'] for r in tables['train_samples'])
    state['next_cursor']=cursor+1
    if state['next_cursor']==2:
     if not args.reference:
      t=time.perf_counter();tables['validation_predictions']=validate_images(model,val,root,device,update,epoch)
      sync(device);scalar['validation_seconds']=time.perf_counter()-t
     scheduler.step();state['next_epoch']+=1;state['next_cursor']=0
     if not args.reference:
      tables['epoch_summaries'].append(dict(update=update,epoch=epoch,batches=partial['batches'],images=partial['images'],objective_sum=partial['loss_sum'],
       mean_batch_objective=partial['loss_sum']/partial['batches'],classification_correct=partial['correct'],classification_images=partial['images'],
       classification_accuracy=partial['correct']/partial['images'],learning_rate_next_epoch=opt.param_groups[0]['lr'],engineering_only=1))
     state['partial']=dict(batches=0,images=0,loss_sum=0.,correct=0)
    if not args.reference:
     scalar.update(data_seconds=load,lr_after_scheduler=opt.param_groups[0]['lr'],next_epoch=state['next_epoch'],next_cursor=state['next_cursor'])
     tables['resources']=resources(device,update,epoch,ROOT,memory)
     tables['events'].append(event(update,epoch,'completed_optimizer_update',seconds=scalar['update_compute_seconds']))
    t=time.perf_counter();store.commit(capture(state,model,opt,scaler,scheduler),tables,extras,
      fault=args.fault if update==args.fault_update else None)
    timing=dict(update=update,seconds=time.perf_counter()-t,scope='local native-state/CSV transaction; post-pointer observation')
    atomic(Path(args.folder)/'LAST_COMMIT.json',timing)
    with open(Path(args.folder)/'COMMIT_TIMINGS.jsonl','ab') as f:f.write(canonical(timing)+b'\n');f.flush();os.fsync(f.fileno())
    print('Engineering update committed:',update,'next epoch/cursor:',state['next_epoch'],state['next_cursor'],flush=True)
    if (Path(args.folder).parent/'FLUSH_REQUEST.json').exists():
     atomic(Path(args.folder)/'FLUSH_ACK.json',dict(update=update,status='paused_at_complete_update'))
     raise SystemExit(91)
   atomic(Path(args.folder)/'WORKER_DONE.json',dict(updates=state['updates'],next_epoch=state['next_epoch'],next_cursor=state['next_cursor'],engineering_only=True))
 except BaseException as exc:
  if isinstance(exc,SystemExit) and exc.code==91:raise
  p=Path(args.folder)/'FAILURES.jsonl'
  with open(p,'ab') as f:f.write(canonical(dict(time_utc=now(),type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))+b'\n');f.flush();os.fsync(f.fileno())
  raise

if __name__=='__main__':main()
