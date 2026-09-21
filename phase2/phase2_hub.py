"""One publisher per notebook; immutable snapshots, fenced leases and bounded retries."""
import email.utils
import os
import random
import shutil
import time
import uuid
from collections import deque
from datetime import datetime,timezone
from pathlib import Path
from phase2_training_data import read,write,sha

REPO='Shanmuk4622/tyre-wear-study'
class Hub:
 def __init__(self,work,workers=1):
  from huggingface_hub import HfApi
  self.work=Path(work);self.api=HfApi(token=os.environ['HF_TOKEN']);self.token=os.environ['HF_TOKEN']
  self.calls=deque();self.limit=max(8,100//workers);self.owner=uuid.uuid4().hex
  self.budget_file=self.work/'request_budget.json'
  if self.budget_file.exists():self.calls.extend(read(self.budget_file))
 def retry(self,fn):
  for attempt in range(6):
   now=time.time()
   while self.calls and self.calls[0]<now-3600:self.calls.popleft()
   while len(self.calls)>=self.limit:
    time.sleep(min(2,max(.01,self.calls[0]+3600-time.time())))
    while self.calls and self.calls[0]<time.time()-3600:self.calls.popleft()
   self.calls.append(time.time());write(self.budget_file,list(self.calls))
   try:return fn()
   except Exception as e:
    response=getattr(e,'response',None);status=getattr(response,'status_code',None)
    if status not in (None,429,500,502,503,504) or attempt==5:raise
    hint=getattr(response,'headers',{}).get('Retry-After','0')
    try:delay=float(hint)
    except (ValueError,TypeError):
     try:delay=max(0,(email.utils.parsedate_to_datetime(hint)-datetime.now(timezone.utc)).total_seconds())
     except Exception:delay=0
    delay=max(delay,min(300,10*2**attempt)+random.Random(attempt).random())
    print(f'HF transient HTTP {status}; retry in {delay:.0f}s. Local files retained.',flush=True)
    end=time.monotonic()+delay
    while time.monotonic()<end:time.sleep(min(2,max(.01,end-time.monotonic())))
 def revision(self):return self.retry(lambda:self.api.repo_info(REPO,repo_type='dataset').sha)
 def fetch(self,path,revision,missing=False):
  from huggingface_hub import hf_hub_download
  from huggingface_hub.errors import EntryNotFoundError
  try:return Path(self.retry(lambda:hf_hub_download(REPO,path,repo_type='dataset',revision=revision,token=self.token,cache_dir=str(self.work/'hub_cache'))))
  except EntryNotFoundError:
   if missing:return None
   raise
 def claim(self,prefix,jobs,takeover=False):
  from huggingface_hub import CommitOperationAdd
  self.prefix=prefix;self.leases={}
  for attempt in range(8):
   rev=self.revision();path=self.fetch(prefix+'/LEASES.json',rev,True);leases=read(path) if path else {}
   for job in jobs:
    entry=leases.get(job['id'],{})
    if entry.get('owner') not in (None,self.owner) and entry.get('expires',0)>time.time() and not takeover:
     raise RuntimeError('Run already leased: '+job['id']+'. Stop its old notebook first. Set TAKE_OVER only after confirming it stopped.')
    leases[job['id']]=dict(owner=self.owner,expires=time.time()+5400)
   try:
    self.retry(lambda:self.api.create_commit(REPO,repo_type='dataset',parent_commit=rev,
     operations=[CommitOperationAdd(path_in_repo=prefix+'/LEASES.json',path_or_fileobj=__import__('json').dumps(leases).encode())],commit_message='Phase2 claim '+self.owner))
    self.leases={j['id']:self.owner for j in jobs};return
   except Exception as e:
    if getattr(getattr(e,'response',None),'status_code',None) not in (409,412) or attempt==7:raise
 def commit(self,files,message,release=False):
  from huggingface_hub import CommitOperationAdd
  # files is remote-path -> immutable local file. Uploading never reads live state.pt.
  for attempt in range(8):
   rev=self.revision();lp=self.fetch(self.prefix+'/LEASES.json',rev,True);leases=read(lp) if lp else {}
   for job,owner in self.leases.items():
    if leases.get(job,{}).get('owner')!=owner:raise RuntimeError('Lease fencing rejected stale writer for '+job)
    leases[job]=dict(owner=None if release else owner,expires=0 if release else time.time()+5400)
   operations=[CommitOperationAdd(path_in_repo=k,path_or_fileobj=str(v)) for k,v in files.items()]
   operations.append(CommitOperationAdd(path_in_repo=self.prefix+'/LEASES.json',path_or_fileobj=__import__('json').dumps(leases).encode()))
   try:
    commit=self.retry(lambda:self.api.create_commit(REPO,repo_type='dataset',operations=operations,parent_commit=rev,commit_message=message))
    # Verify the committed small manifests at the exact immutable revision.
    for remote,local in files.items():
     if remote.endswith('/MANIFEST.json'):
      if sha(self.fetch(remote,commit.oid))!=sha(local):raise RuntimeError('Remote manifest mismatch')
    large={k:v for k,v in files.items() if v.stat().st_size>10*1024**2}
    if large:
     infos=self.retry(lambda:self.api.get_paths_info(REPO,list(large),repo_type='dataset',revision=commit.oid))
     assert {v.path for v in infos}==set(large)
     for info in infos:
      if not info.lfs or info.lfs.sha256!=sha(large[info.path]):raise RuntimeError('Remote large-file SHA-256 mismatch')
    prior=read(self.work/'LAST_REMOTE.json').get('files',{}) if (self.work/'LAST_REMOTE.json').exists() else {}
    prior.update({k:sha(v) for k,v in files.items()})
    write(self.work/'LAST_REMOTE.json',dict(revision=commit.oid,time=time.time(),files=prior))
    print('HF snapshot committed and manifest verified:',commit.oid,flush=True);return commit.oid
   except Exception as e:
    if getattr(getattr(e,'response',None),'status_code',None) not in (409,412) or attempt==7:raise
 def restore(self,job,folder,revision):
  remote=self.prefix+'/runs/'+job['id'];mp=self.fetch(remote+'/MANIFEST.json',revision,True)
  if not mp:return None
  manifest=read(mp)
  if manifest['job']!=job:raise ValueError('Remote job identity mismatch')
  folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
  allowed={'state.pt','weights.pt','HISTORY.json','VALIDATION.json','SMOKE.json','IDENTITY.json','MODEL_CONFIG.json','ERROR.json','FINAL.json','EXPORT.json','console.log'}
  if not set(manifest['files'])<=allowed:raise ValueError('Unexpected remote manifest filename')
  if manifest['status']=='completed':
   infos=self.retry(lambda:self.api.get_paths_info(REPO,[remote+'/weights.pt',remote+'/state.pt'],repo_type='dataset',revision=revision))
   assert len(infos)==2,'Completed run has missing weights/checkpoint'
   for info in infos:
    name=info.path.rsplit('/',1)[-1]
    if not info.lfs or info.lfs.sha256!=manifest['files'][name]:raise ValueError('Completed checkpoint/weights integrity mismatch')
  if (folder/'state.pt').exists():
   from PIL import Image
   import torch
   local=torch.load(folder/'state.pt',map_location='cpu',weights_only=False)
   assert local['job']==job and local['protocol']==manifest['protocol']
   newer=local['updates']>=manifest.get('updates',0)
   del local
   if newer and manifest['status']!='completed':return manifest
  # Completed jobs need no model download. Verify small FINAL metrics instead.
  wanted=['FINAL.json','EXPORT.json'] if manifest['status']=='completed' else list(manifest['files'])
  for name in wanted:
   expected=manifest['files'][name];target=folder/name
   if target.exists() and sha(target)==expected:continue
   downloaded=self.fetch(remote+'/'+name,revision)
   if sha(downloaded)!=expected:raise ValueError('Remote payload checksum mismatch: '+name)
   shutil.copy2(downloaded,target)
  write(folder/'REMOTE.json',manifest)
  return manifest
