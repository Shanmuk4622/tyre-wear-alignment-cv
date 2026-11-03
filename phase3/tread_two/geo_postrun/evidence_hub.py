"""One-owner training-evidence snapshots, explicit token, bounded retries.

No token lookup outside the caller-provided Kaggle secret. Every coordinator
request is recorded; workers never make Hub requests. No ZIP/NPZ publication.
"""
import json,os,re,time,uuid
from pathlib import Path
import requests
from huggingface_hub import HfApi,CommitOperationAdd
from common import atomic,canonical,read,sha

def delay_for(exc,attempt):
 response=getattr(exc,'response',None);code=getattr(response,'status_code',None)
 transport=isinstance(exc,(requests.RequestException,ConnectionError,TimeoutError))
 try:
  import httpx
  transport=transport or isinstance(exc,httpx.TransportError)
 except ImportError:pass
 if code not in (429,500,502,503,504) and not (code is None and transport):return None
 headers=getattr(response,'headers',{}) or {}
 try:delay=float(headers.get('Retry-After','0'))
 except ValueError:
  from email.utils import parsedate_to_datetime
  delay=max(0.,parsedate_to_datetime(headers['Retry-After']).timestamp()-time.time())
 match=re.search(r'\bt=(\d+)',headers.get('RateLimit',''))
 return max(delay,float(match.group(1)) if match else 0,min(60,5*2**attempt))
def wait(delay):
 print('HF server/backoff delay:',delay,'seconds',flush=True)
 while delay>0:
  value=min(30.,delay);time.sleep(value);delay-=value

class Hub:
 REPO='Shanmuk4622/tyre-wear-study'
 def __init__(self,destination,token,log):
  assert token,'Enable Kaggle HF_TOKEN. No alternate credential files are read.'
  self.destination=destination;self.owner=uuid.uuid4().hex;self.log=Path(log);self.last_verified=None
  from rate_budget import configure,BudgetSession
  budget=configure(self.log.parent.parent,self.log.parent/'HTTP_REQUESTS.jsonl')
  self.api=HfApi(token=token);self.session=BudgetSession(budget,self.log.parent/'HTTP_REQUESTS.jsonl');self.session.trust_env=False
  self.session.headers['Authorization']='Bearer '+token;self.calls=0
 def record(self,operation,**values):
  self.log.parent.mkdir(parents=True,exist_ok=True)
  obj=dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),operation=operation,**values)
  with open(self.log,'ab') as f:f.write(canonical(obj)+b'\n');f.flush();os.fsync(f.fileno())
 def call(self,fn,operation):
  for attempt in range(6):
   self.calls+=1;begin=time.monotonic()
   try:
    result=fn();self.record(operation,status='ok',attempt=attempt,seconds=time.monotonic()-begin,coordinator_calls=self.calls);return result
   except Exception as exc:
    delay=delay_for(exc,attempt)
    self.record(operation,status='error',attempt=attempt,error_type=type(exc).__name__,http_status=getattr(getattr(exc,'response',None),'status_code',None),retry_delay=delay)
    if delay is None or attempt==5:raise
    wait(delay)
 def head(self):return self.call(lambda:self.api.repo_info(self.REPO,repo_type='dataset').sha,'repo_info')
 def get_json(self,name,revision):
  def load():
   url=f'https://huggingface.co/datasets/{self.REPO}/resolve/{revision}/{self.destination}/{name}'
   with self.session.get(url,timeout=(20,120)) as r:
    if r.status_code==404:return None
    r.raise_for_status();return r.json()
  return self.call(load,'resolve_json:'+name)
 def claim(self):
  assert self.call(lambda:self.api.whoami(),'whoami')['name'].lower()=='shanmuk4622'
  for attempt in range(8):
   parent=self.head();owner=self.get_json('OWNER.json',parent)
   if owner and owner['owner']==self.owner and owner['expires']>time.time():
    self.record('claim',status='lost_response_acknowledged');return
   if owner and owner['owner'] not in (None,self.owner) and owner['expires']>time.time():raise RuntimeError('Another live gate owns this namespace; stop it cleanly or wait for its 45-minute lease expiry.')
   try:
    self.api.create_commit(self.REPO,repo_type='dataset',parent_commit=parent,
      operations=[CommitOperationAdd(path_in_repo=self.destination+'/OWNER.json',path_or_fileobj=canonical(dict(owner=self.owner,expires=time.time()+2700)))],
      commit_message='TREAD-TWO ATLAS evidence-gate writer claim')
    self.record('claim',status='verified_by_next_parent_read');return
   except Exception as exc:
    code=getattr(getattr(exc,'response',None),'status_code',None);delay=delay_for(exc,attempt)
    self.record('claim',status='error',attempt=attempt,error_type=type(exc).__name__,http_status=code,retry_delay=delay)
    if code not in (409,412) and (delay is None or attempt==7):raise
    if delay is not None:wait(delay)
  raise RuntimeError('Unable to claim engineering writer')
 def publish(self,snapshot,release=False):
  snapshot=Path(snapshot);manifest=read(snapshot/'MANIFEST.json');files=dict(manifest['files'])
  assert not any(n.endswith(('.zip','.npz')) for n in files)
  files['MANIFEST.json']=dict(sha256=sha(snapshot/'MANIFEST.json'),bytes=(snapshot/'MANIFEST.json').stat().st_size)
  for attempt in range(8):
   parent=self.head();owner=self.get_json('OWNER.json',parent)
   assert owner and owner['owner']==self.owner,'Stale writer fenced before publication'
   previous=self.get_json('MANIFEST.json',parent)
   changed=[n for n,m in files.items() if n=='MANIFEST.json' or not previous or previous['files'].get(n)!=m]
   operations=[CommitOperationAdd(path_in_repo=self.destination+'/'+n,path_or_fileobj=str(snapshot/n)) for n in changed]
   operations.append(CommitOperationAdd(path_in_repo=self.destination+'/OWNER.json',path_or_fileobj=canonical(dict(owner=None if release else self.owner,expires=0 if release else time.time()+2700))))
   try:
    commit=self.api.create_commit(self.REPO,repo_type='dataset',parent_commit=parent,operations=operations,
      commit_message='TREAD-TWO ATLAS engineering gate: direct CSV and complete optimizer/RNG state')
    revision=commit.oid
   except Exception as exc:
    current=self.head();remote=self.get_json('MANIFEST.json',current);actual_owner=self.get_json('OWNER.json',current)
    if remote==manifest and actual_owner and actual_owner['owner']==(None if release else self.owner):revision=current
    else:
     code=getattr(getattr(exc,'response',None),'status_code',None);delay=delay_for(exc,attempt)
     if code not in (409,412) and (delay is None or attempt==7):raise
     if delay is not None:wait(delay)
     continue
   infos=self.call(lambda:list(self.api.list_repo_tree(self.REPO,path_in_repo=self.destination,revision=revision,recursive=True,repo_type='dataset')),'verify_tree')
   lookup={v.path[len(self.destination)+1:]:v for v in infos if hasattr(v,'size')}
   import hashlib
   for name,spec in files.items():
    info=lookup[name];assert info.size==spec['bytes']
    if info.lfs:assert info.lfs.sha256==spec['sha256'],name
    else:
     body=(snapshot/name).read_bytes();assert info.blob_id==hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest(),name
   self.last_verified=revision
   result=dict(status='published_and_verified',revision=revision,destination=self.destination,files=len(files),zip_uploads=0,owner_released=release)
   self.record('publish',**result);return result
  raise RuntimeError('Publication exhausted; local committed updates remain retained.')
 def download(self,revision,folder,protocol):
  manifest=self.get_json('MANIFEST.json',revision);assert manifest and manifest['contract']==protocol
  folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
  for name,spec in manifest['files'].items():
   assert not Path(name).is_absolute() and '..' not in Path(name).parts
   target=folder/name;target.parent.mkdir(parents=True,exist_ok=True)
   if target.exists() and sha(target)==spec['sha256']:continue
   def transfer():
    url=f'https://huggingface.co/datasets/{self.REPO}/resolve/{revision}/{self.destination}/{name}'
    part=target.with_name(target.name+'.part')
    with self.session.get(url,timeout=(20,120),stream=True) as response:
     response.raise_for_status()
     with open(part,'wb') as f:
      for block in response.iter_content(4*1024**2):f.write(block)
    part.replace(target)
   self.call(transfer,'resolve_file:'+name)
   assert target.stat().st_size==spec['bytes'] and sha(target)==spec['sha256'],name
  atomic(folder/'MANIFEST.json',manifest)
  return folder
