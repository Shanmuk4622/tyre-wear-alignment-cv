"""Kaggle coordinator: two GPU workers, one HF writer, static cross-session sharding."""
from PIL import Image
import argparse,json,os,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
from filelock import FileLock
import torch
import phase2_training_data as d
from phase2_hub import Hub,REPO

STOP=False
CHILDREN=[]
def stop(*_):
 global STOP
 if STOP:return
 STOP=True
 # Forward promptly even if the coordinator is inside an HF retry/backoff.
 for child in list(CHILDREN):
  if child.poll() is None:child.send_signal(signal.SIGINT)
 print('Stop requested: waiting for completed-update checkpoints, then pushing to HF.',flush=True)

def prepare_assets(hub,folder,models):
 from huggingface_hub import hf_hub_download
 import requests
 folder=Path(folder);identities={}
 for name in models:
  spec=d.SPECS[name];out=folder/name;out.mkdir(parents=True,exist_ok=True)
  if name=='yolo26m':
   target=out/'yolo26m-seg.pt'
   if not target.exists():
    def download():
     with requests.get(spec['url'],stream=True,timeout=60) as response:
      response.raise_for_status()
      with open(target.with_suffix('.part'),'wb') as f:
       for chunk in response.iter_content(1024**2):f.write(chunk)
     target.with_suffix('.part').replace(target)
    hub.retry(download)
   if d.sha(target)!=spec['sha256']:raise ValueError('Official YOLO weights hash mismatch')
  else:
   names=[spec['file']]+(['config.json'] if name=='segformer' else [])
   for fn in names:
    target=out/fn
    if not target.exists():
     src=hub.retry(lambda:hf_hub_download(spec['repo'],fn,revision=spec['revision'],token=hub.token,cache_dir=str(folder/'cache')))
     shutil.copy2(src,target)
  identities[name]={p.name:d.sha(p) for p in out.iterdir() if p.is_file()}
 return identities

def snapshot(hub,work,runroot,selected,protocol,release=False):
 staging=work/'snapshot';staging.mkdir(exist_ok=True);files={}
 needed=sum(p.stat().st_size for j in selected for p in (runroot/j['id']).glob('*.pt'))
 reusable=sum(p.stat().st_size for p in staging.rglob('*.pt'))
 output_bytes=sum(p.stat().st_size for p in work.rglob('*') if p.is_file())
 if output_bytes+needed-reusable>18*1024**3:
  raise RuntimeError('Snapshot would exceed the 18 GiB working-output budget; checkpoints retained in scratch. Resume in a fresh session from the last verified HF snapshot.')
 if shutil.disk_usage(work).free+reusable<needed+1024**3:
  raise RuntimeError('Insufficient free space for an immutable HF snapshot; live checkpoints retained in '+str(runroot))
 # Each immutable snapshot is replaced only after the preceding upload returned.
 for job in selected:
  folder=runroot/job['id'];dest=staging/job['id'];dest.mkdir(exist_ok=True)
  if (folder/'REMOTE.json').exists() and d.read(folder/'REMOTE.json')['status']=='completed':continue
  status=folder/'LOCAL.json';checkpoint=folder/'state.pt'
  names=[];summary=dict(status='not_started',updates=0,job=job,protocol=protocol)
  if checkpoint.exists():
   with FileLock(str(folder/'checkpoint.lock')):
    shutil.copy2(checkpoint,dest/'state.pt')
    local=d.read(status) if status.exists() else {}
   saved=torch.load(dest/'state.pt',map_location='cpu',weights_only=False)
   assert saved['protocol']==protocol and saved['job']==job
   summary.update(epoch=saved['epoch'],cursor=saved['cursor'],updates=saved['updates'],status='completed' if local.get('status')=='completed' and saved['epoch']==60 else 'resumable')
   d.write(dest/'HISTORY.json',saved['history']);d.write(dest/'VALIDATION.json',saved['validation']);del saved
   names+=['state.pt','HISTORY.json','VALIDATION.json']
  elif (folder/'REMOTE.json').exists() and d.read(folder/'REMOTE.json')['status']=='completed':continue
  for name in ('SMOKE.json','IDENTITY.json','MODEL_CONFIG.json','ERROR.json','FINAL.json','EXPORT.json','weights.pt'):
   source=folder/name
   if source.exists():shutil.copy2(source,dest/name);names.append(name)
  # Logs are supplementary; checkpoint history is authoritative and atomic.
  if (folder/'console.log').exists():shutil.copy2(folder/'console.log',dest/'console.log');names.append('console.log')
  if not names:continue
  summary['files']={name:d.sha(dest/name) for name in names};d.write(dest/'MANIFEST.json',summary)
  for name in names+['MANIFEST.json']:files[hub.prefix+'/runs/'+job['id']+'/'+name]=dest/name
 for name in ('CONTRACT.json','OVERLAY.json','ENVIRONMENT.json','SESSION.json','ASSETS.json','SESSION_ERROR.json'):
  if (work/name).exists():files[hub.prefix+'/sessions/'+hub.owner+'/'+name]=work/name
 for source in sorted(Path(__file__).parent.glob('phase2_*.py')):
  # Explicit embedded runtime directory, never arbitrary project or secret files.
  files[hub.prefix+'/sources/'+source.name]=source
 # Remove unchanged payloads from repeat commits while retaining manifests.
 prior=d.read(work/'LAST_REMOTE.json').get('files',{}) if (work/'LAST_REMOTE.json').exists() else {}
 files={k:v for k,v in files.items() if prior.get(k)!=d.sha(v) or k.endswith('/MANIFEST.json')}
 revision=hub.commit(files,'Phase2 checkpoint batch'+(' and release' if release else ''),release=release)
 # Commit is transactional; file hashes are recorded in revision-pinned manifests.
 # Keep local pending state if any call above fails. Clean only our staging tree.
 for child in staging.iterdir():
  assert child.resolve().parent==staging.resolve()
  shutil.rmtree(child)
 return revision

def run(args):
 work=Path(args.work).resolve();work.mkdir(parents=True,exist_ok=True)
 root=d.locate(args.root)
 from phase2_dataset_verify import verify
 verify(root)
 plan=d.overlay(root);models=args.models.split(',');conditions=args.conditions.split(',')
 assert set(models)<=set(d.MODELS) and set(conditions)<={'combined','old_only'}
 assert 0<=args.worker<args.workers<=4
 # Stable full study order means different model notebooks use the same ownership.
 alljobs=d.jobs(('combined','old_only'));selected=[j for i,j in enumerate(alljobs) if j['model'] in models and j['condition'] in conditions and i%args.workers==args.worker]
 if not selected:print('No jobs assigned to this worker.');return
 assert torch.cuda.is_available(),'Select Kaggle GPU T4 x2'
 slots=min(2,torch.cuda.device_count());assert slots>=1
 import psutil
 if psutil.virtual_memory().available<4*1024**3:raise RuntimeError('Need at least 4 GiB available host RAM before starting GPU workers')
 sources={p.name:d.sha(p) for p in Path(__file__).parent.glob('phase2_*.py')}
 import importlib.metadata
 versions={x:importlib.metadata.version(x) for x in ['torch','torchvision','timm','transformers','ultralytics','huggingface_hub','numpy','Pillow','safetensors','filelock']}
 contract=dict(version=d.REVISION,source_checksum=d.RELEASE_SHA,overlay_sha256=d.digest(plan),models=d.SPECS,sources=sources,packages={k:v for k,v in versions.items() if k not in ('torch','torchvision','numpy','Pillow')},
  epochs=60,seeds=[1,2,3],lr=.0001,weight_decay=.01,precision='AMP FP16 with same-batch bounded retries and logged FP32 final attempt',
  sampler='class then domain then original-tyre/video-clip balanced; stateless replacement; equal old/combined update budgets',
  augmentation='stateless horizontal flip with point swap; brightness/contrast; occasional mild blur',
  checkpoint='every completed optimizer update; cursor, RNG, optimizer, scaler, scheduler, history, validation and best weights',
  selection='validation best, earliest tie; test only at completion',decisions=plan['decisions'])
 protocol=d.digest(contract);prefix='phase2/training-r1/'+protocol
 print('RUN_PREFIX =',prefix,flush=True)
 print('Training overlay:',plan['counts'],'GPU processes:',slots,'assigned jobs:',len(selected),flush=True)
 d.write(work/'CONTRACT.json',contract);d.write(work/'OVERLAY.json',plan)
 d.write(work/'ENVIRONMENT.json',dict(packages=versions,python=sys.version,gpus=[torch.cuda.get_device_name(i) for i in range(slots)],gpu_bytes=[torch.cuda.get_device_properties(i).total_memory for i in range(slots)],host_ram_available=psutil.virtual_memory().available,output_free=shutil.disk_usage(work).free,scratch_free=shutil.disk_usage('/tmp').free))
 d.write(work/'SESSION.json',dict(worker=args.worker,workers=args.workers,selected=selected,slots=slots,mode=args.mode,conditions=conditions))
 hub=Hub(work,args.workers);hub.claim(prefix,selected,args.take_over)
 runroot=Path('/tmp')/('phase2_'+protocol)/('worker'+str(args.worker))/'runs';runroot.mkdir(parents=True,exist_ok=True)
 assets=work/'assets';active={};logs={};remaining=[];start=time.monotonic();last_push=start
 try:
  asset_ids=prepare_assets(hub,assets,models);d.write(work/'ASSETS.json',asset_ids)
  rev=hub.revision()
  for job in selected:
   folder=runroot/job['id'];folder.mkdir(parents=True,exist_ok=True)
   remote=hub.restore(job,folder,rev)
   if remote and remote['status']=='completed' and args.mode=='train':print('Verified completed; skip',job['id'],flush=True);continue
   remaining.append(job)
  if args.mode=='smoke':
   # One real-model seam per selected architecture, not every seed/condition.
   seen=set();remaining=[j for j in remaining if not (j['model'] in seen or seen.add(j['model']))]
  while remaining or active:
   if time.monotonic()-start>8.5*3600:stop()
   for slot in range(slots):
    if STOP or slot in active or not remaining:continue
    job=remaining.pop(0);folder=runroot/job['id'];cfg=dict(plan=plan,job=job,protocol=protocol,root=str(root),assets=str(assets),sources=sources,asset_ids=asset_ids)
    config=folder/'config.json';d.write(config,cfg)
    env=os.environ.copy();env.pop('HF_TOKEN',None);env['CUDA_VISIBLE_DEVICES']=str(slot)
    env['HF_HUB_OFFLINE']='1';env['TRANSFORMERS_OFFLINE']='1';env['CUBLAS_WORKSPACE_CONFIG']=':4096:8';env['OMP_NUM_THREADS']='2';env['PYTHONDONTWRITEBYTECODE']='1'
    env['YOLO_CONFIG_DIR']=str(work/'yolo_settings')
    (work/'yolo_settings'/'Ultralytics').mkdir(parents=True,exist_ok=True)
    log=open(folder/'console.log','a',encoding='utf-8');logs[slot]=log
    process=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).with_name('phase2_worker.py')),args.mode,'--config',str(config),'--folder',str(folder),'--device','cuda:0'],env=env,stdout=log,stderr=subprocess.STDOUT)
    CHILDREN.append(process)
    active[slot]=(process,job,False);print(f"GPU {slot}: {job['id']} started (resume check first)",flush=True)
   finished=False
   for slot,(process,job,signaled) in list(active.items()):
    if STOP and not signaled:
     process.send_signal(signal.SIGINT);active[slot]=(process,job,True)
    code=process.poll()
    if code is not None:
     logs.pop(slot).close();del active[slot];finished=True
     CHILDREN.remove(process)
     tail=(runroot/job['id']/'console.log').read_text(encoding='utf-8',errors='replace').splitlines()[-12:]
     print('\n'.join(tail),flush=True)
     if code and not STOP:raise RuntimeError(f"{job['id']} failed; see retained console.log")
     print(job['id'],'worker stopped; publishing progress',flush=True)
   if finished or time.monotonic()-last_push>=1800:
    snapshot(hub,work,runroot,selected,protocol);last_push=time.monotonic()
   if STOP and not active:break
   # Small output status every minute; training logs retain detailed epoch lines.
   if int(time.monotonic()-start)%60<2:
    for _,job,_ in active.values():
     status=runroot/job['id']/'LOCAL.json'
     print(job['id'],d.read(status) if status.exists() else 'model/resume preflight running',flush=True)
   time.sleep(2)
 except BaseException:
  d.write(work/'SESSION_ERROR.json',dict(traceback=traceback.format_exc(),time=time.time()))
  raise
 finally:
  # Catchable errors stop all child writers before the emergency snapshot.
  for process,_,_ in active.values():
   if process.poll() is None:process.send_signal(signal.SIGINT)
  for process,_,_ in active.values():process.wait()
  for log in logs.values():log.close()
  snapshot(hub,work,runroot,selected,protocol,release=True)
 print('Session finished. Rerun unchanged notebook for pending jobs; verified completed jobs are skipped.',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','smoke']);p.add_argument('--models',default=','.join(d.MODELS));p.add_argument('--conditions',default='combined');p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);p.add_argument('--root',default='');p.add_argument('--work',required=True);p.add_argument('--take-over',action='store_true');a=p.parse_args()
 signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
 run(a)
