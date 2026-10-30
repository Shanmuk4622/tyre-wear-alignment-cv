"""One HF parent, automatic live preflight, then two matched T4 training conditions."""
from PIL import Image
import argparse,csv,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import spec
import torch
from filelock import FileLock
from journal import Journal,native
from metrics_r2 import SCHEMAS
from transactions import atomic,read,canonical,digest
from publisher_r2 import Publisher
from report_r2 import report
import acceptance
STOP=False
def stop(*_):
 global STOP
 STOP=True

def publisher(cfg,folder,prefix,token):
 hub=Publisher(prefix+'/'+cfg['study_id']+'/'+cfg['condition'],token,Path(folder)/'REMOTE_HISTORY.jsonl')
 parent=hub.head();m=hub.get_json('MANIFEST.json',parent)
 if m:assert m['contract']==spec.protocol(cfg),'Saved environment/source contract differs; compatibility review required before any new training'
 hub.claim();return hub

def preflight(cfg,folder,token):
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
 hub=publisher(cfg,folder,'phase3/tread-one/r2/engineering',token)
 try:
  parent=hub.head();m=hub.get_json('MANIFEST.json',parent)
  if m:
   downloaded=hub.download(parent,folder/'remote_preflight',spec.protocol(cfg))
   if (downloaded/'PREFLIGHT_PROOF.json').exists() and read(downloaded/'PREFLIGHT_PROOF.json')['status']=='passed':
    proof=read(downloaded/'PREFLIGHT_PROOF.json');receipt=hub.publish(downloaded,release=True);atomic(folder/'PUBLICATION.json',receipt);return proof
   if (downloaded/'PREFLIGHT_CASE.json').exists():
    name=read(downloaded/'PREFLIGHT_CASE.json')['case'];assert name in ('expected','paused','restored')
    j=Journal(folder/name,spec.contract(cfg),SCHEMAS)
    if j.seq<0:j.install(downloaded)
    j.close()
  source=acceptance.prepare(cfg,folder);receipt=hub.publish(source)
  downloaded=hub.download(receipt['revision'],folder/'pinned_source',spec.protocol(cfg));assert read(downloaded/'CURSOR.json')['updates']==1
  final,proof=acceptance.finish(cfg,folder,downloaded,receipt['revision']);receipt=hub.publish(final,release=True)
  atomic(folder/'PUBLICATION.json',receipt);return proof
 except BaseException:
  # Existing committed preflight units are preserved and flushed if possible.
  try:
   name=next((n for n in ('restored','paused','expected') if (folder/n/'journal.sqlite').exists()),'paused')
   j=Journal(folder/name,spec.contract(cfg),SCHEMAS)
   if j.seq>=0:
    snapshot=j.export(spec.ROOT,'preflight_error',report);atomic(snapshot/'PREFLIGHT_CASE.json',dict(case=name))
    from transactions import file_meta
    manifest=read(snapshot/'MANIFEST.json');manifest['files']['PREFLIGHT_CASE.json']=file_meta(snapshot/'PREFLIGHT_CASE.json');atomic(snapshot/'MANIFEST.json',manifest)
    hub.publish(snapshot,release=True)
   else:release_only(hub)
   j.close()
  except BaseException:pass
  raise

def release_only(hub):
 from huggingface_hub import CommitOperationAdd
 parent=hub.owned();hub.api.create_commit(hub.REPO,repo_type='dataset',parent_commit=parent,
  operations=[CommitOperationAdd(path_in_repo=hub.destination+'/OWNER.json',path_or_fileobj=canonical(dict(owner=None,expires=0)))],commit_message='TREAD-ONE r2 clean writer release')

def heartbeat(cfg,hub):
 # Lease-only control traffic does not publish or change the data manifest.
 hub.commit([],f'LEASES/{time.time_ns()}.json',dict(contract=spec.protocol(cfg),writer=hub.owner),
   'TREAD-ONE r2 writer lease heartbeat')

def publish_case(cfg,folder,hub,status,release=False):
 with FileLock(str(Path(folder)/'WRITER.lock'),timeout=60):
  j=Journal(folder,spec.contract(cfg),SCHEMAS)
  if j.seq<0:j.close();return None
  snap=j.export(spec.ROOT,status,report);atomic(snap/'CAPACITY.json',spec.capacity(False))
  from transactions import file_meta
  manifest=read(snap/'MANIFEST.json');manifest['files']['CAPACITY.json']=file_meta(snap/'CAPACITY.json');atomic(snap/'MANIFEST.json',manifest)
  receipt=hub.publish(snap,release=release);atomic(Path(folder)/'PUBLICATION.json',receipt)
  j.prune_snapshots();j.close();return receipt

def launch(cfg,folder):
 cp=Path(folder)/'CONFIG.json'
 if cp.exists():assert spec.contract(read(cp))==spec.contract(cfg)
 else:atomic(cp,cfg)
 env=dict(os.environ);env.pop('HF_TOKEN',None);output=open(Path(folder)/'console.log','ab')
 child=subprocess.Popen([sys.executable,'-B','-u',str(spec.ROOT/'revision2/train.py'),'--config',str(cp),'--folder',str(folder)],stdout=output,stderr=subprocess.STDOUT,env=env)
 return child,output

def pair_artifacts(jobs):
 records={};states={}
 for cfg,folder,_ in jobs:
  j=Journal(folder,spec.contract(cfg),SCHEMAS);states[cfg['condition']]=j.load()
  records[cfg['condition']]=[r for r in j.rows('eval_summaries') if r['level']=='cohort' and r['endpoint'] in ('selected','final')];j.close()
 rows=[]
 for endpoint in ('selected','final'):
  a=next(r for r in records['baseline'] if r['endpoint']==endpoint);b=next(r for r in records['shared_projection'] if r['endpoint']==endpoint)
  for metric in ('accuracy','balanced_accuracy_observed','nll','brier','mean_two_region_dice','human_point_mae','frozen_r1_selection_point_mae'):
   x,y=float(a[metric]),float(b[metric]);rows.append(dict(endpoint=endpoint,metric=metric,baseline=x,shared_projection=y,projected_minus_baseline=y-x,seed=1,scope='Unreplicated paired development pilot; inspected low/high test cohort'))
 payload=dict(status='completed',seed=1,conditions=list(spec.CONDITIONS),study_id=jobs[0][0]['study_id'],selected={k:v['best'] for k,v in states.items()},new_independent_tyres=0,
   superiority_or_promotion_established=False,scope='One paired seed; no new blind generalization result')
 # CSV bytes are created once, then retained in each condition's artifact set.
 import io
 stream=io.StringIO(newline='');w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 for cfg,folder,_ in jobs:
  j=Journal(folder,spec.contract(cfg),SCHEMAS);state=j.load()
  # Native JSON companion is transactional; the CSV is indexed in the same local transaction below.
  target=Path(folder)/'artifacts/PAIR_COMPARISON.csv';target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_name(target.name+'.tmp')
  with open(temp,'w',encoding='utf-8',newline='') as f:f.write(stream.getvalue());f.flush();os.fsync(f.fileno())
  temp.replace(target)
  from transactions import file_meta
  j.commit(state,{},artifacts={'PAIR_COMPLETION.json':payload},known_artifacts={'artifacts/PAIR_COMPARISON.csv':file_meta(target)});j.close()
 return payload

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--data-root',default='');parser.add_argument('--publish',action='store_true');args=parser.parse_args()
 assert args.publish,'Measured delivery always requires the explicit Kaggle HF_TOKEN publisher'
 assert torch.cuda.device_count()>=2,'Use Kaggle T4 x2; this paired protocol assigns one condition to each device'
 assert shutil.disk_usage(spec.ROOT).free>12*1024**3,'Start a clean working area with 12 GiB free'
 token=os.environ.get('HF_TOKEN');assert token,'Enable Kaggle HF_TOKEN secret; no alternative credential files are read'
 for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
 root=spec.locate(args.data_root);preflight_proofs=[];jobs=[];children={};locks=[]
 session_lock=FileLock(str(spec.ROOT/'R2_SESSION.lock'),timeout=1);session_lock.acquire();locks.append(session_lock)
 try:
  for i,condition in enumerate(spec.CONDITIONS):
   cfg=spec.configuration(root,f'cuda:{i}',condition,True);folder=spec.ROOT/'revision2_runs'/cfg['study_id']/('preflight_'+condition)
   folder.mkdir(parents=True,exist_ok=True)
   lock=FileLock(str(folder.parent/(condition+'.PREFLIGHT.lock')),timeout=1);lock.acquire();locks.append(lock)
   if STOP:raise InterruptedError('Stop before automatic preflight')
   print('Automatic live preflight:',condition,flush=True);preflight_proofs.append(preflight(cfg,folder,token))
  assert all(p['status']=='passed' and p['actual_live_HF_restore'] for p in preflight_proofs)
  for i,condition in enumerate(spec.CONDITIONS):
   cfg=spec.configuration(root,f'cuda:{i}',condition);folder=spec.ROOT/'revision2_runs'/cfg['study_id']/condition;folder.mkdir(parents=True,exist_ok=True)
   lock=FileLock(str(folder/'COORDINATOR.lock'),timeout=1);lock.acquire();locks.append(lock)
   hub=publisher(cfg,folder,'phase3/tread-one/r2/paired-seed1',token);jobs.append((cfg,folder,hub))
   j=Journal(folder,spec.contract(cfg),SCHEMAS);parent=hub.head()
   if j.seq<0 and hub.get_json('MANIFEST.json',parent):j.install(hub.download(parent,folder/'remote_resume',spec.protocol(cfg)))
   j.close();(folder.parent/'FLUSH_REQUEST.json').unlink(missing_ok=True)
  # Finish both remote restores before starting either measured condition.
  for cfg,folder,hub in jobs:heartbeat(cfg,hub)
  for cfg,folder,hub in jobs:
   owner=hub.get_json('OWNER.json',hub.head());assert owner and owner['owner']==hub.owner and owner['expires']-time.time()>2100,'Writer lease too short after server delays; rerun unchanged'
  for cfg,folder,hub in jobs:
   children[cfg['condition']]=launch(cfg,folder)
   print('Measured paired condition:',cfg['condition'],'seed 1, 60 epochs, device',cfg['device'],flush=True)
  last=time.monotonic();last_lease=last
  while any(c.poll() is None or c.returncode==91 for c,_ in children.values()):
   if STOP:raise InterruptedError('Catchable notebook stop; flushing last committed native/CSV units')
   for name,(child,_) in children.items():
    if child.poll() not in (None,0,91):raise RuntimeError(name+' worker failed; inspect its console.log')
   if time.monotonic()-last_lease>=900:
    for cfg,folder,hub in jobs:heartbeat(cfg,hub)
    last_lease=time.monotonic()
   if time.monotonic()-last>=1800:
    marker=jobs[0][1].parent/'FLUSH_REQUEST.json';atomic(marker,dict(reason='30_minute_publication'))
    while any(c.poll() is None for c,_ in children.values()):
     if STOP:raise InterruptedError('Stop while awaiting safe publication boundary')
     time.sleep(1)
    for cfg,folder,hub in jobs:publish_case(cfg,folder,hub,'periodic')
    marker.unlink(missing_ok=True)
    for cfg,folder,_ in jobs:
     child,output=children[cfg['condition']]
     if child.returncode==91:output.close();children[cfg['condition']]=launch(cfg,folder)
    last=time.monotonic()
   time.sleep(2)
  for cfg,folder,_ in jobs:
   j=Journal(folder,spec.contract(cfg),SCHEMAS);s=j.load();assert s['phase']=='completed' and s['updates']==2940;j.close()
  pair=pair_artifacts(jobs);receipts=[]
  for cfg,folder,hub in jobs:receipts.append(publish_case(cfg,folder,hub,'completed',True))
  result=dict(status='completed',study=pair,preflight=preflight_proofs,publications=receipts,measured_runs=2,epochs_per_run=60,optimizer_updates_per_run=2940)
  atomic(jobs[0][1].parent/'PAIR_DELIVERY.json',result);atomic(spec.ROOT/'revision2_runs/LATEST_PAIR.json',dict(path=str(jobs[0][1].parent),status='completed'))
  print(json.dumps(result,indent=2),flush=True)
 except BaseException:
  for child,output in children.values():
   if child.poll() is None:
    child.send_signal(signal.SIGINT)
    try:child.wait(timeout=40)
    except subprocess.TimeoutExpired:pass
  for cfg,folder,hub in jobs:
   try:
    receipt=publish_case(cfg,folder,hub,'paused_or_error',True)
    if receipt is None:release_only(hub)
   except BaseException as e:print('HF flush pending:',cfg['condition'],type(e).__name__,'local state retained',folder,flush=True)
  raise
 finally:
  for child,output in children.values():output.close()
  for lock in locks:lock.release()

if __name__=='__main__':main()
