"""Bounded immutable direct-file staging; one final manifest selects a snapshot."""
import hashlib,json,time,uuid
from pathlib import Path
from huggingface_hub import CommitOperationAdd
from evidence_hub import Hub,delay_for,wait
from common import atomic,canonical,digest,read,sha

class Publisher(Hub):
 def owned(self):
  parent=self.head();owner=self.get_json('OWNER.json',parent)
  assert owner and owner['owner']==self.owner,'Publication fenced: another writer owns this namespace'
  return parent
 def commit(self,operations,marker_name,marker_value,message,release=False):
  for attempt in range(8):
   parent=self.owned()
   if self.get_json(marker_name,parent)==marker_value:
    self.record('ambiguous_or_replayed_commit',status='acknowledged',revision=parent,marker=marker_name);return parent
   owner=canonical(dict(owner=None if release else self.owner,expires=0 if release else time.time()+2700))
   ops=operations+[CommitOperationAdd(path_in_repo=self.destination+'/'+marker_name,path_or_fileobj=canonical(marker_value)),CommitOperationAdd(path_in_repo=self.destination+'/OWNER.json',path_or_fileobj=owner)]
   try:
    result=self.api.create_commit(self.REPO,repo_type='dataset',parent_commit=parent,operations=ops,commit_message=message)
    self.record('commit',status='ok',revision=result.oid,files=len(ops));return result.oid
   except Exception as exc:
    current=self.head()
    if self.get_json(marker_name,current)==marker_value:
     self.record('ambiguous_commit',status='acknowledged',revision=current);return current
    code=getattr(getattr(exc,'response',None),'status_code',None);delay=delay_for(exc,attempt)
    self.record('commit',status='error',error_type=type(exc).__name__,http_status=code,retry_delay=delay)
    if code not in (409,412) and (delay is None or attempt==7):raise
    if delay is not None:wait(delay)
  raise RuntimeError('Bounded publication attempts exhausted; native/local evidence retained')
 def publish(self,snapshot,release=False):
  snapshot=Path(snapshot);local=read(snapshot/'MANIFEST.json');remote_files={}
  for name,value in local['files'].items():
   assert not name.endswith(('.zip','.npz','.sqlite'))
   assert sha(snapshot/name)==value['sha256'] and (snapshot/name).stat().st_size==value['bytes']
   p=Path(name);object_name=(p.parent/(p.stem+'-'+value['sha256'][:20]+p.suffix)).as_posix()
   remote_files[name]=dict(value,object='evidence/'+object_name)
  manifest=dict(contract=local['contract'],format='immutable-direct-files-v1',files=remote_files)
  parent=self.owned();previous=self.get_json('MANIFEST.json',parent)
  needed=[n for n,v in remote_files.items() if not previous or previous['files'].get(n)!=v]
  for offset in range(0,len(needed),80):
   group=needed[offset:offset+80];marker=dict(contract=local['contract'],files={n:remote_files[n] for n in group});key='BATCHES/'+digest(marker)+'.json'
   operations=[CommitOperationAdd(path_in_repo=self.destination+'/'+remote_files[n]['object'],path_or_fileobj=str(snapshot/n)) for n in group]
   self.commit(operations,key,marker,'TREAD-TWO ATLAS immutable direct CSV/native objects')
  staged=self.owned()
  entries=self.call(lambda:list(self.api.list_repo_tree(self.REPO,path_in_repo=self.destination,revision=staged,recursive=True,repo_type='dataset')),'verify_immutable_tree')
  lookup={v.path[len(self.destination)+1:]:v for v in entries if hasattr(v,'size')}
  for name,value in remote_files.items():
   obj=lookup[value['object']];assert obj.size==value['bytes']
   if obj.lfs:assert obj.lfs.sha256==value['sha256'],name
   else:
    body=(snapshot/name).read_bytes();assert obj.blob_id==hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest(),name
  # Only verified immutable objects can be selected by the final manifest.
  marker=dict(contract=local['contract'],manifest_sha256=digest(manifest),owner_released=release,writer=self.owner,publication_nonce=uuid.uuid4().hex)
  key='PUBLICATIONS/'+digest(marker)+'.json'
  operations=[CommitOperationAdd(path_in_repo=self.destination+'/MANIFEST.json',path_or_fileobj=canonical(manifest))]
  revision=self.commit(operations,key,marker,'TREAD-TWO ATLAS select verified recovery snapshot',release=release)
  assert self.get_json('MANIFEST.json',revision)==manifest
  actual_owner=self.get_json('OWNER.json',revision);assert actual_owner['owner']==(None if release else self.owner)
  self.last_verified=revision
  result=dict(status='published_and_verified',revision=revision,destination=self.destination,files=len(remote_files)+1,uploaded_objects=len(needed),zip_uploads=0,owner_released=release)
  self.record('publish',**result);return result
 def download(self,revision,folder,protocol):
  manifest=self.get_json('MANIFEST.json',revision);assert manifest and manifest['contract']==protocol and manifest['format']=='immutable-direct-files-v1'
  folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);local={}
  for name,value in manifest['files'].items():
   if hasattr(self,'transfer_tick'):self.transfer_tick()
   assert not Path(name).is_absolute() and '..' not in Path(name).parts and '..' not in Path(value['object']).parts
   target=folder/name;target.parent.mkdir(parents=True,exist_ok=True)
   if not target.exists() or sha(target)!=value['sha256']:
    def transfer():
     with self.session.get(f'https://huggingface.co/datasets/{self.REPO}/resolve/{revision}/{self.destination}/{value["object"]}',timeout=(20,150),stream=True) as response:
      response.raise_for_status()
      with open(target.with_name(target.name+'.part'),'wb') as f:
       for block in response.iter_content(4*1024**2):f.write(block)
     target.with_name(target.name+'.part').replace(target)
    self.call(transfer,'resolve_immutable:'+name)
   assert target.stat().st_size==value['bytes'] and sha(target)==value['sha256'],name
   local[name]={k:v for k,v in value.items() if k!='object'}
  atomic(folder/'REMOTE_MANIFEST.json',manifest);atomic(folder/'MANIFEST.json',dict(contract=protocol,files=local));return folder
