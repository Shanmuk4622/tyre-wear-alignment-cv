"""Coordinate real-model logging/resume tests. No measured study is launched."""
from PIL import Image
import argparse,csv,hashlib,importlib.util,json,os,shutil,signal,subprocess,sys,time,uuid
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1]
assert Path(sys.prefix).name=='cv_conda' or Path('/kaggle/input').exists()
sys.path.insert(0,str(ROOT/'runtime'))
import torch
from filelock import FileLock
from common import environment,delta
from data import plan
from transactions import Store,atomic,canonical,digest,read,sha,file_meta
from telemetry import SCHEMAS
from report import case_report
from evidence_hub import Hub
STOP=False
def stop(*_):
 global STOP
 STOP=True

def locate(value):
 if value:return Path(value).resolve()
 if Path('/kaggle/input').exists():
  roots=[p.parent for p in Path('/kaggle/input').rglob('VERSION.json') if read(p).get('dataset_title')=='Tire Dataset Prepared phase2']
  assert len(roots)==1,'Attach exactly one prepared Phase 2 dataset';return roots[0]
 return ROOT.parent/'phase2/data/phase2_dataset_v1'

def configuration(root,device):
 assert sha(root/'SHA256SUMS.txt')=='06c6c9d003aff808babf7ebb5786f441e95064eb000f5e5ed5b155d0ee9e7c6a','Prepared checksum list changed'
 frozen=read(ROOT/'training/FROZEN.json')
 for name,h in frozen.items():assert sha(ROOT/'runtime'/name)==h
 p=plan(root);assert p['source_checksum']=='06c6c9d003aff808babf7ebb5786f441e95064eb000f5e5ed5b155d0ee9e7c6a'
 assert digest(p)=='f81a80d20b60ba505d94e31c26615e7e7171a91a8506948be6c392b68107a039','Frozen safe overlay changed'
 hashes=dict(line.split('  ',1)[::-1] for line in (root/'SHA256SUMS.txt').read_text().splitlines())
 metadata={name:hashes[name] for name in hashes if name.endswith('.json')}
 for name,h in metadata.items():assert sha(root/name)==h,name
 sources={str(v.relative_to(ROOT)).replace('\\','/'):sha(v) for v in (ROOT/'training').glob('*') if v.is_file() and v.suffix in ('.py','.json')}
 sources.update({'runtime/'+name:h for name,h in frozen.items()});sources['diagnostic/metrics.py']=sha(ROOT/'diagnostic/metrics.py')
 return dict(mode='engineering_gate',study_training_allowed=False,family='TREAD-ONE',seed=462204,device=str(device),environment=environment(torch.device(device)),
  data_root=str(root),overlay_sha256=digest(p),dataset_metadata_hashes=metadata,source_hashes=sources,updates=3,batches_per_epoch=2,batch=8,input_hw=[512,384],
  optimizer='Frozen r1 AdamW/loss/AMP/cosine math; two-batch surrogate epochs for engineering only',
  telemetry='Actual pre-update augmented observations; no fabricated historical data',local_native_recovery_states=2,
  native_map_schedule='Every gate update only; long-run schedule not selected',full_training_epochs=None)
def protocol(cfg):return digest({k:v for k,v in cfg.items() if k!='data_root'})
def store(case,cfg):return Store(case,{k:v for k,v in cfg.items() if k!='data_root'},SCHEMAS)

def run_worker(case,cfg_path,steps=3,reference=False,fault=None):
 case=Path(case);case.mkdir(parents=True,exist_ok=True)
 command=[sys.executable,'-B','-u',str(ROOT/'training/worker.py'),'--config',str(cfg_path),'--folder',str(case),'--stop-after',str(steps)]
 if reference:command+=['--reference']
 if fault:command+=['--fault',fault,'--fault-update','2']
 env=dict(os.environ);env.pop('HF_TOKEN',None)
 if STOP:raise InterruptedError('Coordinator stop before next worker')
 with open(case/'console.log','ab') as output:
  child=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT,env=env)
  try:
   while child.poll() is None:
    try:child.wait(timeout=2)
    except subprocess.TimeoutExpired:
     if STOP:child.send_signal(signal.SIGINT)
   code=child.returncode
  except KeyboardInterrupt:
   child.send_signal(signal.SIGINT)
   try:child.wait(timeout=20)
   except subprocess.TimeoutExpired:pass
   raise
 if code==91:
  while (case.parent/'FLUSH_REQUEST.json').exists():
   if STOP:raise InterruptedError('Coordinator stopped while waiting for snapshot')
   time.sleep(2)
  return run_worker(case,cfg_path,steps,reference,fault)
 if fault:
  assert code==86,case.joinpath('console.log').read_text(errors='replace')[-5000:]
 else:assert code==0,case.joinpath('console.log').read_text(errors='replace')[-5000:]

def comparison(a,b):
 fields=('model','optimizer','scheduler','scaler','rng','losses','next_epoch','next_cursor','updates')
 result={k:delta(a[k],b[k]) for k in fields}
 assert max(result.values())==0,result
 for key in ('batches','images','loss_sum'):assert a['partial'][key]==b['partial'][key]
 return result

def local_suite(cfg,path):
 path=Path(path);path.mkdir(parents=True,exist_ok=True);cp=path/'CONFIG.json';atomic(cp,cfg)
 result_path=path/'LOCAL_RESULTS.json'
 if result_path.exists() and read(result_path)['status']=='passed':return read(result_path)
 for name,ref in (('math_reference',True),('uninterrupted',False)):
  atomic(path/'ACTIVE_CASE.json',dict(case=name));run_worker(path/name,cp,reference=ref)
 reference=store(path/'math_reference',cfg).state();complete=store(path/'uninterrupted',cfg).state()
 original_math=comparison(reference,complete)
 paused=store(path/'paused',cfg)
 proof=path/'PAUSE_PROOF.json'
 if not proof.exists():
  assert not paused.chain or paused.state()['updates']<=1
  atomic(path/'ACTIVE_CASE.json',dict(case='paused'));run_worker(path/'paused',cp,steps=1)
  paused=store(path/'paused',cfg);assert paused.state()['updates']==1
  atomic(proof,dict(first_generation=paused.chain[-1]['generation'],first_manifest_sha256=sha(paused.generation()/'MANIFEST.json'),checkpoint_sha256=paused.chain[-1]['files']['state.pt']['sha256']))
 run_worker(path/'paused',cp);resumed=store(path/'paused',cfg)
 saved=read(proof);original=path/'paused/generations'/saved['first_generation']/'MANIFEST.json'
 if original.exists():assert sha(original)==saved['first_manifest_sha256']
 else:
  with open(path/'paused/BASE_VERSION_HISTORY.csv',newline='',encoding='utf-8') as f:
   assert next(r for r in csv.DictReader(f) if r['update']=='1')['state_sha256']==saved['checkpoint_sha256']
 pause_delta=comparison(complete,resumed.state())
 journal=path/'CRASH_PROOF.json'
 if not journal.exists():
  atomic(path/'ACTIVE_CASE.json',dict(case='abrupt'));run_worker(path/'abrupt',cp,fault='after_csv')
  broken=store(path/'abrupt',cfg);assert broken.state()['updates']==1
  orphan=[p for p in (path/'abrupt/generations').iterdir() if p.is_dir() and p.name not in {m['generation'] for m in broken.chain}]
  assert orphan;atomic(journal,dict(durable_update_after_native_crash=1,uncommitted_generations_ignored=len(orphan)))
 run_worker(path/'abrupt',cp);crash_delta=comparison(complete,store(path/'abrupt',cfg).state())
 # Snapshot/install is a fresh native optimizer+CSV base, not the old local path.
 paused1=path/'published_base_fixture'
 if not (paused1/'POINTER.json').exists():
  atomic(path/'ACTIVE_CASE.json',dict(case='published_base_fixture'));run_worker(paused1,cp,steps=1)
 source=store(paused1,cfg);snapshot=source.export('paused',case_report)
 installed=store(path/'installed',cfg)
 if not installed.chain:installed.install(snapshot)
 run_worker(path/'installed',cp);install_delta=comparison(complete,store(path/'installed',cfg).state())
 actual=store(path/'uninterrupted',cfg);cursor=actual.state()['row_cursors']
 assert cursor['train_samples']==24 and cursor['train_steps']==3 and cursor['geometry_points']==144 and actual.state()['next_epoch']==2 and actual.state()['next_cursor']==1
 result=dict(status='passed',protocol=protocol(cfg),device=cfg['device'],original_r1_update_math=original_math,
  fresh_process_pause_resume=pause_delta,abrupt_native_exit_replay=crash_delta,direct_snapshot_install_resume=install_delta,
  row_cursors=cursor,engineering_only=True,measured_study_started=False)
 atomic(result_path,result);return result

def envelope(path,cfg,status,proof=None):
 path=Path(path);destination=path/'envelopes'/f'{status}-{uuid.uuid4().hex[:10]}';destination.mkdir(parents=True)
 atomic(destination/'CONTRACT.json',{k:v for k,v in cfg.items() if k!='data_root'})
 (destination/'cases').mkdir()
 cases={}
 for name in ('math_reference','uninterrupted','paused','abrupt','published_base_fixture','installed','live_resume'):
  case=path/name
  if not (case/'POINTER.json').exists():continue
  with FileLock(str(case/'WRITER.lock'),timeout=55):
   s=store(case,cfg);export=s.export(status,case_report);target=destination/'cases'/name
   shutil.copytree(export,target);cases[name]=dict(updates=s.state()['updates'],row_cursors=s.state()['row_cursors'])
 for name in ('LOCAL_RESULTS.json','PAUSE_PROOF.json','CRASH_PROOF.json','REMOTE_RESUME_SOURCE.json','LIVE_RESULTS.json','coordinator.log','REMOTE_HISTORY.jsonl'):
  if (path/name).exists():shutil.copy2(path/name,destination/name)
 atomic(destination/'GATE_STATUS.json',dict(status=status,cases=cases,engineering_only=True,source_protocol=protocol(cfg)))
 for name in cfg['source_hashes']:
  target=destination/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
  assert sha(target)==cfg['source_hashes'][name]
 files={str(p.relative_to(destination)).replace('\\','/'):file_meta(p) for p in destination.rglob('*') if p.is_file()}
 atomic(destination/'MANIFEST.json',dict(contract=protocol(cfg),files=files))
 atomic(path/'LATEST_ENVELOPE.json',dict(path=str(destination),status=status))
 return destination

def install_envelope(path,cfg,downloaded):
 downloaded=Path(downloaded)
 for case in (downloaded/'cases').iterdir():
  local=store(Path(path)/case.name,cfg)
  if not local.chain:local.install(case)
 for name in ('LOCAL_RESULTS.json','PAUSE_PROOF.json','CRASH_PROOF.json','REMOTE_RESUME_SOURCE.json','LIVE_RESULTS.json'):
  if (downloaded/name).exists() and not (Path(path)/name).exists():shutil.copy2(downloaded/name,Path(path)/name)

def live_suite(path,cfg,hub):
 path=Path(path);parent=hub.head();manifest=hub.get_json('MANIFEST.json',parent)
 if manifest:
  current=hub.download(parent,path/'downloads/current',protocol(cfg));install_envelope(path,cfg,current)
  prior=read(path/'REMOTE_RESUME_SOURCE.json') if (path/'REMOTE_RESUME_SOURCE.json').exists() else None
  source_revision=prior['revision'] if prior else parent
  suitable=(current/'cases/published_base_fixture/CURSOR.json').exists() and read(current/'cases/published_base_fixture/CURSOR.json')['updates']==1
 else:suitable=False
 if not suitable:
  paused=store(path/'published_base_fixture',cfg);assert paused.state()['updates']==1
  receipt=hub.publish(envelope(path,cfg,'paused'))
  source_revision=receipt['revision']
 atomic(path/'REMOTE_RESUME_SOURCE.json',dict(revision=source_revision,case='published_base_fixture',expected_updates=1))
 downloaded=hub.download(source_revision,path/'downloads/resume_source',protocol(cfg))
 source=downloaded/'cases/published_base_fixture'
 assert read(source/'CURSOR.json')['updates']==1
 target=store(path/'live_resume',cfg)
 if not target.chain:target.install(source)
 atomic(path/'ACTIVE_CASE.json',dict(case='live_resume'));run_worker(path/'live_resume',path/'CONFIG.json')
 differences=comparison(store(path/'uninterrupted',cfg).state(),store(path/'live_resume',cfg).state())
 result=dict(status='passed',published_source_revision=source_revision,restored_completed_updates=1,continued_to_updates=3,differences=differences,
   row_cursors=store(path/'live_resume',cfg).state()['row_cursors'],engineering_only=True)
 atomic(path/'LIVE_RESULTS.json',result)
 receipt=hub.publish(envelope(path,cfg,'passed'),release=True);atomic(path/'PUBLICATION.json',receipt)
 return result,receipt

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--data-root',default='');parser.add_argument('--devices',default='auto');parser.add_argument('--publish',action='store_true')
 parser.add_argument('--local-config');parser.add_argument('--work');args=parser.parse_args()
 if args.local_config:
  print(json.dumps(local_suite(read(args.local_config),args.work),indent=2));return
 for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
 assert shutil.disk_usage(ROOT).free>12*1024**3,'Need 12 GiB free for two-device cases, direct snapshots, downloads and recovery.'
 root=locate(args.data_root);devices=[f'cuda:{i}' for i in range(min(torch.cuda.device_count(),2))] if args.devices=='auto' else args.devices.split(',')
 assert devices,'Enable at least one CUDA GPU for the actual AMP recovery gate.'
 jobs=[];hubs={};children=[];locks=[]
 try:
  for device in devices:
   cfg=configuration(root,device);tag=protocol(cfg);path=ROOT/'training_gates'/tag;path.mkdir(parents=True,exist_ok=True);atomic(path/'CONFIG.json',cfg)
   lock=FileLock(str(path/'COORDINATOR.lock'),timeout=1);lock.acquire();locks.append(lock)
   if (path/'FLUSH_REQUEST.json').exists():(path/'FLUSH_REQUEST.json').unlink()
   if args.publish:
    hub=Hub('phase3/tread-one/engineering/evidence-gate-r1/'+tag,os.environ.get('HF_TOKEN'),path/'REMOTE_HISTORY.jsonl');hub.claim();hubs[tag]=hub
    parent=hub.head()
    if hub.get_json('MANIFEST.json',parent):install_envelope(path,cfg,hub.download(parent,path/'downloads/restart',tag))
   jobs.append((cfg,path))
  # Compute-only coordinators have no secret; both T4 devices are exercised in parallel.
  for cfg,path in jobs:
   env=dict(os.environ);env.pop('HF_TOKEN',None)
   output=open(path/'coordinator.log','ab')
   child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__)),'--local-config',str(path/'CONFIG.json'),'--work',str(path)],stdout=output,stderr=subprocess.STDOUT,env=env)
   children.append((child,output,path));print('Running engineering gate on',cfg['device'],'protocol',protocol(cfg),flush=True)
  last_flush=time.monotonic()
  while any(c.poll() is None for c,_,_ in children):
   if STOP:
    for child,_,_ in children:
     if child.poll() is None:child.send_signal(signal.SIGINT)
    raise InterruptedError('Catchable gate stop requested; completed updates will be snapshotted.')
   for child,_,path in children:
    if child.poll() not in (None,0):raise RuntimeError((path/'coordinator.log').read_text(errors='replace')[-7000:])
   if args.publish and time.monotonic()-last_flush>=1800:
    for _,path in jobs:atomic(path/'FLUSH_REQUEST.json',dict(reason='30_minute_snapshot'))
    # Workers leave their next completed-update boundary and wait for marker removal.
    for cfg,path in jobs:
     receipt=hubs[protocol(cfg)].publish(envelope(path,cfg,'running'));atomic(path/'PUBLICATION.json',receipt)
     (path/'FLUSH_REQUEST.json').unlink(missing_ok=True)
    last_flush=time.monotonic()
   time.sleep(2)
  result=[]
  for cfg,path in jobs:
   local=read(path/'LOCAL_RESULTS.json');assert local['status']=='passed'
   live=receipt=None
   if args.publish:live,receipt=live_suite(path,cfg,hubs[protocol(cfg)])
   else:envelope(path,cfg,'passed_local')
   result.append(dict(protocol=protocol(cfg),device=cfg['device'],local=local,live=live,publication=receipt,path=str(path)))
  atomic(ROOT/'training_gates/GATE_COMPLETION.json',dict(status='passed',devices=result,engineering_only=True,new_study_started=False,live_hf_tested=args.publish))
  print(json.dumps(dict(status='passed',devices=[r['device'] for r in result],live_hf_tested=args.publish,engineering_only=True),indent=2))
 except BaseException as exc:
  for child,_,_ in children:
   if child.poll() is None:
    child.send_signal(signal.SIGINT)
    try:child.wait(timeout=25)
    except subprocess.TimeoutExpired:pass
  for cfg,path in jobs:
   hub=hubs.get(protocol(cfg))
   if hub:
    try:
     receipt=hub.publish(envelope(path,cfg,'paused' if isinstance(exc,(InterruptedError,KeyboardInterrupt)) else 'error'),release=True);atomic(path/'PUBLICATION.json',receipt)
    except BaseException as flush:print('Final gate publication pending:',type(flush).__name__,'Local data retained:',path,flush=True)
  raise
 finally:
  for child,output,_ in children:output.close()
  for lock in locks:lock.release()

if __name__=='__main__':main()
