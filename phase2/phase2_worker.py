"""One GPU process, durable optimizer-step state, real-model fresh-process seam test."""
from PIL import Image
import argparse,copy,json,math,os,random,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from filelock import FileLock
import phase2_training_data as d
import phase2_models as m

STOP=False
STEP_SECONDS=[]
def environment():
 import importlib.metadata
 return dict(packages={k:importlib.metadata.version(k) for k in ('torch','torchvision','numpy','Pillow','timm','transformers','ultralytics')},gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
def stop(*_):
 global STOP
 STOP=True
def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
def set_rng(s):
 random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'])
 if s['cuda']:torch.cuda.set_rng_state_all(s['cuda'])
def cpu(x):
 if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
 if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
 if isinstance(x,list):return [cpu(v) for v in x]
 if isinstance(x,tuple):return tuple(cpu(v) for v in x)
 return copy.deepcopy(x)
def delta(a,b):
 if isinstance(a,torch.Tensor):
  assert a.shape==b.shape and a.dtype==b.dtype
  return float((a.double()-b.double()).abs().max()) if a.numel() else 0.
 if isinstance(a,np.ndarray):return float(np.abs(a.astype(float)-b.astype(float)).max())
 if isinstance(a,dict):
  assert a.keys()==b.keys();return max([delta(a[k],b[k]) for k in a] or [0.])
 if isinstance(a,(tuple,list)):
  assert len(a)==len(b);return max([delta(x,y) for x,y in zip(a,b)] or [0.])
 if isinstance(a,(float,int)):return abs(a-b)
 assert a==b;return 0.

def components(job,assets,device,initialize):
 model=m.build(job['model'],assets,initialize,device)
 opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
 scaler=torch.amp.GradScaler(device.type,enabled=device.type=='cuda',init_scale=1024.)
 sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=60,eta_min=1e-6)
 return model,opt,scaler,sched

def update(model,opt,scaler,b,job,epoch):
 start=rng();buffers={k:v.detach().clone() for k,v in model.named_buffers()};events=[]
 for attempt in range(9):
  set_rng(start)
  for k,v in model.named_buffers():v.copy_(buffers[k])
  m.training_mode(model);opt.zero_grad(set_to_none=True)
  with torch.autocast(b['img'].device.type,dtype=torch.float16,enabled=b['img'].is_cuda and attempt<8):value=m.loss(model,b,job['model'],epoch)
  scaled=scaler.scale(value);finite=bool(torch.isfinite(value))
  if finite:
   scaled.backward();scaler.unscale_(opt)
   norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);finite=bool(torch.isfinite(norm))
  if finite:
   scaler.step(opt);scaler.update();return float(value.detach()),events
  opt.zero_grad(set_to_none=True)
  if attempt==8 or not scaler.is_enabled():
   set_rng(start);raise RuntimeError('Nonfinite update: previous durable checkpoint retained; no sample skipped.')
  scale=float(scaler.get_scale());scaler.update(new_scale=max(scale*.5,1e-8));events.append(dict(attempt=attempt+1,scale=scale,status='retry_same_batch'))
 raise AssertionError('unreachable')

def state(model,opt,scaler,sched,job,protocol,epoch,cursor,history,validation,best,best_score,updates):
 return dict(model=model.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),scheduler=sched.state_dict(),rng=rng(),job=job,protocol=protocol,torch_version=torch.__version__,environment=environment(),epoch=epoch,cursor=cursor,history=history,validation=validation,best=best,best_score=best_score,updates=updates)
def save(folder,s,status='resumable'):
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
 with FileLock(str(folder/'checkpoint.lock')):
  tmp=folder/'state.tmp'
  with open(tmp,'wb') as f:
   torch.save(s,f);f.flush();os.fsync(f.fileno())
  os.replace(tmp,folder/'state.pt')
  d.write(folder/'LOCAL.json',dict(status=status,epoch=s['epoch'],cursor=s['cursor'],updates=s['updates'],protocol=s['protocol'],job=s['job']))
def restore(s,model,opt,scaler,sched,protocol,job):
 assert s['protocol']==protocol and s['job']==job,'Incompatible checkpoint: never silently restart'
 assert s['torch_version']==torch.__version__,'Torch changed since the checkpoint; needs an explicit compatibility check, never restart silently'
 assert s['environment']==environment(),'Checkpoint runtime/GPU differs; explicit compatibility review required, no silent restart'
 model.load_state_dict(s['model'],strict=True);opt.load_state_dict(s['optimizer']);scaler.load_state_dict(s['scaler']);sched.load_state_dict(s['scheduler']);set_rng(s['rng'])

def evaluate(model,rows,root,job,device):
 name=job['model'];model.eval();records=[]
 with torch.inference_mode():
  for row in rows:
   if STOP:raise InterruptedError('Evaluation paused; training state is durable')
   if name=='hrnet' and row['point_kind']!='human':continue
   b=m.batch(root,[row],name,device=device);t=time.perf_counter()
   with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):pred=m.predict(model,b,name)
   if device.type=='cuda':torch.cuda.synchronize()
   rec=dict(image_id=row['image_id'],tyre=row['physical_tyre_id'],domain=row['domain'],seconds=time.perf_counter()-t)
   if name in ('mobilenetv4','resnet50'):
    rec.update(label=row['class_index'],prediction=int(pred[0].argmax()),probabilities=pred[0].cpu().tolist())
   elif name=='hrnet':
    rec.update(target=row['points'],prediction=pred[0].cpu().tolist(),mae=float((pred[0]-b['points'][0]).abs().mean()))
   else:
    # Score original raw masks, excluding contradiction pixels, at model input resolution.
    _,raw,_=d.sample(root,row,d.SPECS[name]['hw']);target=torch.tensor(raw[:2],device=device);valid=torch.tensor(~raw[2],device=device)
    values=[]
    for c in (0,1):
     p=pred[0,c]&valid;y=target[c]&valid;inter=int((p&y).sum());den=int(p.sum()+y.sum());union=int((p|y).sum())
     values.append(dict(dice=(2*inter+1)/(den+1),iou=(inter+1)/(union+1),predicted_pixels=int(p.sum()),target_pixels=int(y.sum())))
    rec['regions']=values
   records.append(rec)
 assert records
 if name in ('mobilenetv4','resnet50'):
  recalls={str(c):sum(r['prediction']==c for r in records if r['label']==c)/sum(r['label']==c for r in records) for c in sorted({r['label'] for r in records})}
  score=float(np.mean(list(recalls.values())));metric='balanced_accuracy_on_observed_low_high_classes'
 elif name=='hrnet':score=-float(np.mean([r['mae'] for r in records]));metric='negative_mean_width_fraction_error'
 else:score=float(np.mean([x['dice'] for r in records for x in r['regions']]));metric='mean_two_region_dice_at_input_resolution'
 return dict(metric=metric,score=score,records=records,per_tyre={tyre:sum(r['tyre']==tyre for r in records) for tyre in sorted({r['tyre'] for r in records})},scope='known held-out old low/high tyres only; not unseen mid/video evidence')

def run_steps(s,model,opt,scaler,sched,plan,root,job,device,n):
 rows,batches=d.schedule(plan,job,0)
 losses=[]
 for step in range(s['cursor'],s['cursor']+n):
  started=time.monotonic()
  indices=batches[step%len(batches)]
  b=m.batch(root,[rows[i] for i in indices],job['model'],0,step,job['seed'],device)
  value,_=update(model,opt,scaler,b,job,0);losses.append(value)
  if device.type=='cuda':torch.cuda.synchronize()
  STEP_SECONDS.append(time.monotonic()-started)
 return losses

def smoke(config,folder,device,continuation=False):
 plan=config['plan'];job=config['job'];protocol=config['protocol'];root=config['root'];folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
 seed_all(job['seed']);model,opt,scaler,sched=components(job,config['assets'],device,False if continuation else True)
 if continuation:
  s=torch.load(folder/'seam.pt',map_location='cpu',weights_only=False);restore(s,model,opt,scaler,sched,protocol,job)
  losses=run_steps(s,model,opt,scaler,sched,plan,root,job,device,2)
  torch.save(dict(model=cpu(model.state_dict()),optimizer=cpu(opt.state_dict()),scaler=scaler.state_dict(),scheduler=sched.state_dict(),rng=rng(),losses=losses),folder/'actual.pt');return
 start=time.monotonic();initial=dict(cursor=0)
 run_steps(initial,model,opt,scaler,sched,plan,root,job,device,2)
 # Exercise a non-initial scheduler state as well as optimizer/scaler state.
 sched.step()
 torch.save(state(model,opt,scaler,sched,job,protocol,0,2,[],[],None,-math.inf,2),folder/'seam.pt')
 losses=run_steps(dict(cursor=2),model,opt,scaler,sched,plan,root,job,device,2)
 expected=dict(model=cpu(model.state_dict()),optimizer=cpu(opt.state_dict()),scaler=scaler.state_dict(),scheduler=sched.state_dict(),rng=rng(),losses=losses)
 # Release VRAM before independent process reconstructs and restores the real model.
 del model,opt,scaler,sched;torch.cuda.empty_cache()
 args=[sys.executable,'-B',__file__,'continue-smoke','--config',str(folder/'config.json'),'--folder',str(folder),'--device',str(device)]
 d.write(folder/'config.json',config);subprocess.run(args,check=True)
 actual=torch.load(folder/'actual.pt',map_location='cpu',weights_only=False)
 differences={k:delta(expected[k],actual[k]) for k in expected}
 passed=max(differences.values())<=1e-5
 report=dict(status='passed' if passed else 'failed',protocol=protocol,job=job,differences=differences,seconds=time.monotonic()-start,torch=torch.__version__,gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',peak_gpu_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else None,source_sha256=d.sha(__file__),scope='real model, real transformed batches; independent-process mid-epoch state restoration; noninitial scheduler state')
 steps=len(d.schedule(plan,job,0)[1]);report['measured_step_seconds']=STEP_SECONDS[-4:]
 report['estimated_training_compute_hours_excluding_checkpoint_validation_upload']=float(np.mean(STEP_SECONDS[-4:]))*steps*60/3600
 d.write(folder/'SMOKE.json',report)
 for name in ('seam.pt','actual.pt'):(folder/name).unlink()
 if not passed:raise RuntimeError('Real-model fresh-process resume test failed; long run was not started')
 print('GPU resume passed. Estimated training compute hours for this run (excludes checkpoint, validation and uploads):',round(report['estimated_training_compute_hours_excluding_checkpoint_validation_upload'],2),flush=True)
 return report

def train(config,folder,device):
 plan=config['plan'];job=config['job'];protocol=config['protocol'];root=config['root'];folder=Path(folder)
 smoke_dir=folder/'smoke';result=smoke(config,smoke_dir,device)
 if STOP:return
 d.write(folder/'SMOKE.json',result)
 seed_all(job['seed']);existing=(folder/'state.pt').exists()
 model,opt,scaler,sched=components(job,config['assets'],device,not existing)
 identity=dict(model=d.SPECS[job['model']],parameters=sum(p.numel() for p in model.parameters()),tensor_signature=d.digest({k:list(v.shape) for k,v in model.state_dict().items()}))
 d.write(folder/'IDENTITY.json',identity)
 if hasattr(model,'config'):d.write(folder/'MODEL_CONFIG.json',model.config.to_dict())
 elif hasattr(model,'yaml'):d.write(folder/'MODEL_CONFIG.json',model.yaml)
 if existing:
  s=torch.load(folder/'state.pt',map_location='cpu',weights_only=False);restore(s,model,opt,scaler,sched,protocol,job)
 else:s=state(model,opt,scaler,sched,job,protocol,0,0,[],[],None,-math.inf,0);save(folder,s)
 epoch=s['epoch'];cursor=s['cursor'];history=s['history'];validation=s['validation'];best=s['best'];best_score=s['best_score'];updates=s['updates'];del s
 started=time.monotonic()
 try:
  while epoch<60:
   rows,batches=d.schedule(plan,job,epoch)
   while cursor<len(batches):
    if STOP or time.monotonic()-started>8.5*3600:return
    if __import__('shutil').disk_usage(folder).free<2*1024**3:raise RuntimeError('Less than 2 GiB free; pause before an unsafe checkpoint write')
    ids=batches[cursor];b=m.batch(root,[rows[i] for i in ids],job['model'],epoch,cursor,job['seed'],device)
    t=time.monotonic();value,events=update(model,opt,scaler,b,job,epoch);cursor+=1;updates+=1
    history.append(dict(epoch=epoch,batch=cursor,loss=value,lr=opt.param_groups[0]['lr'],seconds=time.monotonic()-t,image_ids=[rows[i]['image_id'] for i in ids],numerical_retries=events))
    save(folder,state(model,opt,scaler,sched,job,protocol,epoch,cursor,history,validation,best,best_score,updates));del b
   val=evaluate(model,[r for r in plan['rows'] if r['role']=='validation'],root,job,device)
   validation.append(dict(epoch=epoch+1,result=val))
   if val['score']>best_score:best=cpu(model.state_dict());best_score=val['score']
   sched.step();epoch+=1;cursor=0
   save(folder,state(model,opt,scaler,sched,job,protocol,epoch,cursor,history,validation,best,best_score,updates))
   print(f"{job['id']} epoch {epoch}/60 | validation {val['metric']} {val['score']:.5f}",flush=True)
  model.load_state_dict(best,strict=True)
  test=evaluate(model,[r for r in plan['rows'] if r['role']=='test'],root,job,device)
  d.write(folder/'FINAL.json',dict(protocol=protocol,job=job,validation_selection='highest validation score; earliest tie',best_score=best_score,test=test,training_steps=updates,limitations=plan['decisions']))
  tmp=folder/'weights.tmp';torch.save(best,tmp);tmp.replace(folder/'weights.pt')
  d.write(folder/'EXPORT.json',dict(protocol=protocol,job=job,weights_sha256=d.sha(folder/'weights.pt'),identity=identity,input_hw=d.SPECS[job['model']]['hw'],preprocessing='RGB direct resize; YOLO [0,1], others ImageNet normalization',classes=['low','mid','high'] if job['model'] in ('mobilenetv4','resnet50') else ['tyre','tread'] if job['model']!='hrnet' else ['left_upper','right_upper','left_middle','right_middle','left_lower','right_lower'],selected_by='validation only',runtime_sources=config['sources']))
  d.write(folder/'LOCAL.json',dict(status='completed',job=job,protocol=protocol,epoch=60,cursor=0,updates=updates))
 except InterruptedError:return
 except BaseException as exc:
  d.write(folder/'ERROR.json',dict(type=type(exc).__name__,traceback=traceback.format_exc(),time=time.time()))
  raise

def main():
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['train','smoke','continue-smoke']);a.add_argument('--config',required=True);a.add_argument('--folder',required=True);a.add_argument('--device',default='cuda:0');args=a.parse_args()
 signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
 torch.set_num_threads(2);m.deterministic();cfg=d.read(args.config);device=torch.device(args.device)
 if args.mode=='train':train(cfg,args.folder,device)
 else:smoke(cfg,args.folder,device,args.mode=='continue-smoke')
if __name__=='__main__':main()
