"""Resumable full-cohort paired training; original r1/NB04 sources remain frozen."""
from PIL import Image
import argparse,csv,json,os,signal,sys,time,traceback,hashlib
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import spec
import numpy as np,torch
from filelock import FileLock
from common import deterministic,environment,cpu,rng,set_rng,memory
from data import plan,schedule,batch
from model import TreadOne
from transactions import atomic,canonical,read,sha
from journal import Journal
from metrics_r2 import SCHEMAS,evaluate_one,summaries
import optimizer_r2
STOP=False
def stop(*_):
 global STOP
 STOP=True
def stopped():return STOP

def append_csv(path,row):
 exists=path.exists() and path.stat().st_size>0
 with open(path,'a',encoding='utf-8',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(row))
  if not exists:writer.writeheader()
  writer.writerow(row);f.flush();os.fsync(f.fileno())

def capture(state,model,opt,scaler,scheduler):
 state.update(model=cpu(model.state_dict()),optimizer=cpu(opt.state_dict()),scaler=scaler.state_dict(),scheduler=scheduler.state_dict(),rng=rng());return state

def restore(state,model,opt,scaler,scheduler,cfg):
 assert state['protocol']==spec.protocol(cfg)
 model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler']);scheduler.load_state_dict(state['scheduler'])
 if state['updates']:opt._opt_called=True
 set_rng(state['rng'])

def resource_rows(device,update,epoch):
 # Reuse the measured sampler, scheduling its optional nvidia-smi branch here.
 from telemetry import resources
 sample=update==1 or update%49==0
 rows=resources(device,1 if sample else 2,epoch,spec.ROOT,memory)
 for row in rows:row['update']=update;row['scope']='completed_update_scheduled_gpu_sample' if sample else 'completed_update_memory_only'
 return rows

def eval_model(model,weights,device):
 before=rng();m=TreadOne().to(device);m.load_state_dict(weights['model']);m.eval();set_rng(before);return m

def main():
 process_started=time.perf_counter();process_started_utc=optimizer_r2.base.now()
 parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--folder',required=True)
 parser.add_argument('--max-train-updates',type=int);parser.add_argument('--fault',default='');parser.add_argument('--fault-unit',type=int,default=2);args=parser.parse_args()
 cfg=read(args.config);root=Path(cfg['data_root']);device=torch.device(cfg['device'])
 assert cfg['mode'] in ('engineering_smoke','measured_paired_pilot')
 if cfg['mode']=='measured_paired_pilot':assert cfg['epochs']==60 and cfg['seed']==1 and cfg['batches_per_epoch']==49 and args.max_train_updates is None and not args.fault
 assert sha(root/'SHA256SUMS.txt')==spec.CHECKSUM
 for n,h in cfg['sources'].items():assert sha(spec.ROOT/n)==h,n
 for n,h in cfg['metadata_hashes'].items():assert sha(root/n)==h,n
 assert environment(device)==cfg['environment'];p=plan(root)
 from transactions import digest
 assert digest(p)==spec.OVERLAY
 deterministic(cfg['seed'])
 for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
 folder=Path(args.folder);journal=Journal(folder,spec.contract(cfg),SCHEMAS)
 model,opt,scaler,scheduler=optimizer_r2.base.legacy.components(device);state=journal.load()
 if state is None:
  state=dict(protocol=spec.protocol(cfg),updates=0,epoch0=0,cursor=0,phase='train',eval_cursor=0,best=None,history=[],losses=[],validation_records=[],
   partial=dict(batches=0,images=0,loss_sum=0.,correct=0),row_cursors={t:0 for t in SCHEMAS},engineering_only=cfg['mode']=='engineering_smoke')
  initial=cpu(model.state_dict());h=hashlib.sha256()
  for name,tensor in sorted(initial.items()):h.update(name.encode());h.update(str(tensor.dtype).encode());h.update(canonical(list(tensor.shape)));h.update(tensor.contiguous().numpy().tobytes())
  identity=dict(seed=cfg['seed'],model_tensor_sha256=h.hexdigest(),parameters=sum(p.numel() for p in model.parameters()),architecture='Frozen TREAD-ONE-r1',condition=cfg['condition'])
  journal.commit(capture(state,model,opt,scaler,scheduler),artifacts={'INITIALIZATION.json':identity})
 else:restore(state,model,opt,scaler,scheduler,cfg)
 hashes=dict(line.split('  ',1)[::-1] for line in (root/'SHA256SUMS.txt').read_text().splitlines());verified=set()
 def verify(rows):
  for row in rows:
   for key in ('image_path','tyre_mask','tread_mask','ignore_mask'):
    n=row[key]
    if n not in verified:assert sha(root/n)==hashes[n],n;verified.add(n)
 def commit(tables,artifacts):
  t=time.perf_counter();journal.commit(capture(state,model,opt,scaler,scheduler),tables,artifacts,fault=args.fault if journal.seq+1==args.fault_unit else None)
  append_csv(folder/'COMMIT_TIMINGS.csv',dict(unit=journal.seq,updates=state['updates'],phase_after_commit=state['phase'],seconds=time.perf_counter()-t,time_utc=optimizer_r2.base.now(),scope='post_commit_observation'))
 val_rows=spec.eval_rows(p,cfg,'validation');gallery=set()
 for cls in (0,2):
  options=sorted([r for r in val_rows if r['class_index']==cls],key=lambda r:(r['point_kind']!='human',r['image_id']))
  gallery.update(r['image_id'] for r in options[:cfg['validation_gallery_per_class']])
 active_eval=None;active_endpoint=None;termination='returned'
 append_csv(folder/'WORKER_LIFECYCLE.csv',dict(event='started_or_resumed',time_utc=optimizer_r2.base.now(),process_started_utc=process_started_utc,process_elapsed_seconds=0.,updates=state['updates'],phase=state['phase'],status='running'))
 try:
  with FileLock(str(folder/'WRITER.lock'),timeout=1):
   while state['phase']!='completed':
    if STOP:raise InterruptedError('Catchable stop; completed journal unit is recoverable')
    if (folder.parent/'FLUSH_REQUEST.json').exists():raise SystemExit(91)
    import shutil
    if shutil.disk_usage(folder).free<3*1024**3:raise RuntimeError('Free disk below 3 GiB; last native/CSV unit retained')
    mem=memory()
    if mem.get('source')=='cgroup' and (mem.get('fraction') or 0)>.9:raise RuntimeError('Container memory above 90%; last committed unit retained')
    if state['phase']=='train' and state['cursor']==0:spec.capacity()
    tables={k:[] for k in SCHEMAS};artifacts={};epoch=state['epoch0']+1
    if state['phase']=='train':
     if args.max_train_updates is not None and state['updates']>=args.max_train_updates:break
     e0,cursor=state['epoch0'],state['cursor'];source,order=schedule(p,cfg['seed'],e0);chosen=[source[i] for i in order[cursor]]
     t=time.perf_counter();verify(chosen);b=batch(root,chosen,device,cfg['seed'],e0,cursor);torch.cuda.synchronize(device);load=time.perf_counter()-t
     before=journal.db.execute('SELECT sha256 FROM commits WHERE seq=?',(journal.seq,)).fetchone()[0]
     tables,raw,step=optimizer_r2.update(model,opt,scaler,b,chosen,device,cfg,e0,cursor,state['updates']+1,before,folder,stopped)
     step['data_seconds']=load;state['updates']+=1;state['losses'].append(step['loss']);state['cursor']+=1
     partial=state['partial'];partial['batches']+=1;partial['images']+=len(chosen);partial['loss_sum']+=step['loss'];partial['correct']+=sum(r['correct'] for r in tables['train_samples'])
     tables['resources']=resource_rows(device,state['updates'],epoch)
     tables['events'].append(optimizer_r2.base.event(state['updates'],epoch,'completed_optimizer_update',seconds=step['update_compute_seconds']))
     if raw:
      for n,value in raw.items():artifacts[f'train_maps/e{epoch:04d}/u{state["updates"]:08d}/{n}']=value
     if state['cursor']==cfg['batches_per_epoch']:
      state['phase']='validation';state['cursor']=0;state['eval_cursor']=0;state['validation_records']=[];step['next_epoch']=epoch+1;step['next_cursor']=0
     commit(tables,artifacts)
     if state['updates']==1 or state['updates']%cfg['batches_per_epoch']==0:print(cfg['condition'],'updates',state['updates'],'epoch',epoch,'phase',state['phase'],flush=True)
    else:
     validation=state['phase']=='validation'
     endpoint='epoch_validation' if validation else ('selected' if state['phase']=='test_selected' else 'final')
     split='validation' if validation else ('validation_fixture' if cfg['mode']=='engineering_smoke' else 'test')
     rows=val_rows if validation else spec.eval_rows(p,cfg,'test')
     checkpoint_epoch=epoch if validation else (state['best']['epoch'] if endpoint=='selected' else cfg['epochs'])
     if validation:active=model;active.eval()
     else:
      if active_endpoint!=endpoint:
       name=state['best']['weights'] if endpoint=='selected' else 'final_weights.pt'
       meta=journal.db.execute('SELECT sha256 FROM artifacts WHERE name=?',('artifacts/'+name,)).fetchone();assert meta and sha(folder/'artifacts'/name)==meta[0]
       weights=torch.load(folder/'artifacts'/name,map_location='cpu',weights_only=False);assert weights['protocol']==state['protocol']
       active_eval=eval_model(model,weights,device);active_endpoint=endpoint
      active=active_eval
     start=state['eval_cursor'];end=min(start+cfg['validation_chunk'],len(rows));raw_chunk=[];verify(rows[start:end])
     for i in range(start,end):
      if STOP:raise InterruptedError('Evaluation chunk interrupted; replay at its last committed cursor')
      part,pred,raw,b=evaluate_one(active,rows[i],root,device,checkpoint_epoch,split,endpoint,cfg['condition'],i,cfg['seed'])
      for k,rs in part.items():tables[k].extend(rs)
      state['validation_records'].append(pred);raw_chunk.append(raw)
      if validation and epoch in cfg['capture_epochs'] and rows[i]['image_id'] in gallery:
       prefix=f'validation_maps/e{epoch:04d}/i{i:05d}/'
       artifacts[prefix+'inputs.npy']=b['img'][0].detach().float().cpu().numpy();artifacts[prefix+'targets.npy']=b['masks'][0].cpu().numpy().astype(np.uint8);artifacts[prefix+'outputs.npy']=raw
       artifacts[prefix+'IDENTITY.json']=dict(image_id=rows[i]['image_id'],image_sha256=rows[i]['image_sha256'],epoch=epoch,split=split)
     if not validation:artifacts[f'test_maps/{endpoint}/i{start:05d}_{end:05d}.npy']=np.stack(raw_chunk)
     state['eval_cursor']=end
     if end==len(rows):
      summary,utility=summaries(state['validation_records'],checkpoint_epoch,split,endpoint)
      for k,rs in summary.items():tables[k].extend(rs)
      cohort=summary['eval_summaries'][0]
      if validation:
       assert utility is not None
       if state['best'] is None or utility>state['best']['score']:
        name=f'selected/epoch_{epoch:04d}.pt';state['best']=dict(epoch=epoch,score=utility,weights=name)
        artifacts[name]=dict(protocol=state['protocol'],epoch=epoch,score=utility,model=cpu(model.state_dict()),selected_by=cfg['selection'])
       scheduler.step();partial=state['partial']
       tables['epoch_summaries'].append(dict(update=state['updates'],epoch=epoch,batches=partial['batches'],images=partial['images'],objective_sum=partial['loss_sum'],mean_batch_objective=partial['loss_sum']/partial['batches'],
        classification_correct=partial['correct'],classification_images=partial['images'],classification_accuracy=partial['correct']/partial['images'],learning_rate_next_epoch=opt.param_groups[0]['lr'],engineering_only=int(cfg['mode']=='engineering_smoke')))
       state['history'].append(cohort);state['epoch0']+=1;state['phase']='train';state['partial']=dict(batches=0,images=0,loss_sum=0.,correct=0)
       if state['epoch0']==cfg['epochs']:
        artifacts['final_weights.pt']=dict(protocol=state['protocol'],epoch=epoch,model=cpu(model.state_dict()));state['phase']='test_selected'
      else:
       artifacts[f'endpoint_results/{endpoint}.json']=dict(summary=cohort,epoch=checkpoint_epoch,scope=cfg['test_scope'])
       state['phase']='test_final' if endpoint=='selected' else 'completed'
      state['eval_cursor']=0;state['validation_records']=[]
     commit(tables,artifacts)
    if (folder.parent/'FLUSH_REQUEST.json').exists():raise SystemExit(91)
   atomic(folder/'WORKER_STATUS.json',dict(phase=state['phase'],updates=state['updates'],epoch0=state['epoch0'],best=state['best'],protocol=state['protocol']))
   if state['phase']=='completed':assert state['updates']==cfg['epochs']*cfg['batches_per_epoch']
 except BaseException as exc:
  termination='flush_boundary' if isinstance(exc,SystemExit) and exc.code==91 else type(exc).__name__
  if isinstance(exc,SystemExit) and exc.code==91:raise
  with open(folder/'FAILURES.jsonl','ab') as f:f.write(canonical(dict(time_utc=optimizer_r2.base.now(),type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),last_durable_unit=journal.seq))+b'\n');f.flush();os.fsync(f.fileno())
  raise
 finally:
  journal.close()
  append_csv(folder/'WORKER_LIFECYCLE.csv',dict(event='exited',time_utc=optimizer_r2.base.now(),process_started_utc=process_started_utc,process_elapsed_seconds=time.perf_counter()-process_started,updates=state['updates'],phase=state['phase'],status=termination))

if __name__=='__main__':main()
