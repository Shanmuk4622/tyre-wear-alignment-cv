"""TREAD-ONE evidence transactions. CSV and native state share one commit cursor.

This module has no model/training decision and no credential lookup. Historical
CSV is retained; local recovery state rotates only after a new pointer commits.
"""
import csv,hashlib,json,os,shutil,uuid
from pathlib import Path
import torch

def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
 with open(tmp,'wb') as f:f.write(canonical(v));f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def csv_write(p,rows,fields):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
 with open(tmp,'w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='raise');w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def file_meta(p):return dict(sha256=sha(p),bytes=Path(p).stat().st_size)

class Store:
 def __init__(self,root,contract,schemas):
  self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
  self.contract,self.schemas=contract,schemas;self.protocol=digest(contract)
  p=self.root/'CONTRACT.json'
  if p.exists():assert read(p)==contract,'Contract drift: do not mix or silently restart this run.'
  else:atomic(p,contract)
  self.chain=[]
  self._load_chain()

 def _load_chain(self):
  pointer=self.root/'POINTER.json'
  if not pointer.exists():return
  item=read(pointer);name=item['generation'];expected=item['manifest_sha256'];reverse=[]
  pruned=read(self.root/'PRUNED_STATES.json') if (self.root/'PRUNED_STATES.json').exists() else {}
  while name:
   p=self.root/'generations'/name/'MANIFEST.json';assert sha(p)==expected,'Generation manifest altered'
   meta=read(p);assert meta['protocol']==self.protocol
   for relative,spec in meta['files'].items():
    target=p.parent/relative
    if relative=='state.pt' and not target.exists():
     assert pruned.get(name)==spec['sha256'],'Undeclared missing recovery state'
    else:assert target.stat().st_size==spec['bytes'] and sha(target)==spec['sha256'],str(target)
   reverse.append(meta);name=meta['parent'];expected=meta['parent_manifest_sha256']
  self.chain=list(reversed(reverse))
  first=self.chain[0]['update']
  assert first==0 or self.chain[0].get('remote_base')
  assert [m['update'] for m in self.chain]==list(range(first,first+len(self.chain)))
  assert (self.generation()/'state.pt').exists(),'Latest recovery state cannot be pruned'

 def generation(self):return self.root/'generations'/self.chain[-1]['generation'] if self.chain else None
 def state(self):
  if not self.chain:return None
  # Only this run's manifest-verified own state is deserialized.
  state=torch.load(self.generation()/'state.pt',map_location='cpu',weights_only=False)
  assert state['protocol']==self.protocol and state['updates']==self.chain[-1]['update']
  assert state['row_cursors']==self.chain[-1]['row_cursors']
  return state

 def commit(self,state,rows,extras=None,fault=None):
  assert state['protocol']==self.protocol
  expected=self.chain[-1]['update']+1 if self.chain else 0
  assert state['updates']==expected,'Completed updates must be sequential'
  parent=self.chain[-1] if self.chain else None
  counts=dict(parent['row_cursors']) if parent else {t:0 for t in self.schemas}
  for table in self.schemas:counts[table]+=len(rows.get(table,[]))
  state['row_cursors']=counts
  name=f'{state["updates"]:08d}-'+uuid.uuid4().hex[:10]
  folder=self.root/'generations'/name;folder.mkdir(parents=True)
  with open(folder/'state.pt','wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
  if fault=='after_state':os._exit(86)
  state_sha=sha(folder/'state.pt')
  for row in rows.get('train_steps',[]):row['checkpoint_after_sha256']=state_sha
  for table,fields in self.schemas.items():csv_write(folder/'tables'/(table+'.csv'),rows.get(table,[]),fields)
  if extras:
   import numpy as np
   for filename,value in extras.items():
    assert Path(filename).name==filename and filename.endswith('.npy')
    with open(folder/filename,'wb') as f:np.save(f,value,allow_pickle=False);f.flush();os.fsync(f.fileno())
  if fault=='after_csv':os._exit(86)
  files={str(p.relative_to(folder)).replace('\\','/'):file_meta(p) for p in folder.rglob('*') if p.is_file()}
  meta=dict(generation=name,update=state['updates'],protocol=self.protocol,row_cursors=counts,
    next_epoch=state['next_epoch'],next_cursor=state['next_cursor'],files=files,
    parent=parent['generation'] if parent else None,
    parent_manifest_sha256=sha(self.root/'generations'/parent['generation']/'MANIFEST.json') if parent else None)
  atomic(folder/'MANIFEST.json',meta)
  if fault in ('after_manifest','before_pointer'):os._exit(86)
  atomic(self.root/'POINTER.json',dict(generation=name,manifest_sha256=sha(folder/'MANIFEST.json'),updates=state['updates']))
  self.chain.append(meta)
  if fault=='after_pointer':os._exit(86)
  self.prune_states(keep=2)
  return state_sha

 def prune_states(self,keep=2):
  """Keep all CSV/maps and the latest two recovery states; record every rotation."""
  ledger=self.root/'PRUNED_STATES.json';pruned=read(ledger) if ledger.exists() else {}
  for m in self.chain[:-keep]:
   p=(self.root/'generations'/m['generation']/'state.pt').resolve()
   assert p.is_relative_to(self.root) and p.name=='state.pt'
   if p.exists():
    assert sha(p)==m['files']['state.pt']['sha256']
    pruned[m['generation']]=sha(p);atomic(ledger,pruned)
    p.unlink()  # Only a superseded native state inside this newly created run.

 def export(self,status,report_builder=None):
  assert self.chain
  name=f'{self.chain[-1]["update"]:08d}-{status}-'+uuid.uuid4().hex[:8]
  folder=self.root/'snapshots'/name;folder.mkdir(parents=True)
  shutil.copy2(self.generation()/'state.pt',folder/'state.pt')
  atomic(folder/'CONTRACT.json',self.contract)
  grouped={t:{} for t in self.schemas};version_rows=[]
  if (self.root/'BASE_VERSION_HISTORY.csv').exists():
   with open(self.root/'BASE_VERSION_HISTORY.csv',encoding='utf-8',newline='') as f:version_rows=list(csv.DictReader(f))
   # Restoring a snapshot retains historical hashes, not every historical state.
   for row in version_rows:row['local_native_state_present']=0
  if (self.root/'BASE_MAPS').exists():shutil.copytree(self.root/'BASE_MAPS',folder/'maps')
  for meta in self.chain:
   gen=self.root/'generations'/meta['generation']
   for table in self.schemas:
    with open(gen/'tables'/(table+'.csv'),encoding='utf-8',newline='') as f:
     for row in csv.DictReader(f):grouped[table].setdefault(int(row['epoch']),[]).append(row)
   version_rows.append(dict(update=meta['update'],next_epoch=meta['next_epoch'],next_cursor=meta['next_cursor'],
      state_sha256=meta['files']['state.pt']['sha256'],local_native_state_present=int((gen/'state.pt').exists())))
  schema={};total={t:0 for t in self.schemas}
  for table,epochs in grouped.items():
   if not epochs:epochs={0:[]}
   for epoch,rows in epochs.items():
    relative=f'csv/epoch_{epoch:04d}/{table}.csv';path=folder/relative
    csv_write(path,rows,self.schemas[table]);schema[relative]=dict(columns=self.schemas[table],rows=len(rows),**file_meta(path));total[table]+=len(rows)
  assert total==self.chain[-1]['row_cursors'],'State/CSV cursor disagreement'
  versions={int(r['update']):r for r in version_rows}
  csv_write(folder/'STATE_VERSIONS.csv',[versions[k] for k in sorted(versions)],['update','next_epoch','next_cursor','state_sha256','local_native_state_present'])
  for meta in self.chain:
   gen=self.root/'generations'/meta['generation']
   for p in gen.glob('*.npy'):
    target=folder/'maps'/f'update_{meta["update"]:08d}'/p.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
  for event in ('FAILURES.jsonl','COMMIT_TIMINGS.jsonl','console.log','LAST_COMMIT.json'):
   if (self.root/event).exists():shutil.copy2(self.root/event,folder/event)
  atomic(folder/'CURSOR.json',dict(status=status,protocol=self.protocol,updates=self.chain[-1]['update'],
    next_epoch=self.chain[-1]['next_epoch'],next_cursor=self.chain[-1]['next_cursor'],row_cursors=total,state_sha256=sha(folder/'state.pt')))
  atomic(folder/'SCHEMA.json',dict(version='tread-one-evidence-1',files=schema,missing='Blank means unavailable/not applicable; zero is observed zero.',
    scope='Engineering fixture only. Not a measured training study or model comparison.',
    units='Probabilities/Dice/IoU and point errors are fractions; native pixel errors explicit; times seconds; memory bytes; power watts; temperature C.',
    retention='All CSV retained; latest two local native states retained; remote current native state and immutable commit history; no ZIP/NPZ.'))
  if report_builder:report_builder(folder)
  files={str(p.relative_to(folder)).replace('\\','/'):file_meta(p) for p in folder.rglob('*') if p.is_file()}
  assert not any(n.endswith(('.zip','.npz')) for n in files)
  atomic(folder/'MANIFEST.json',dict(contract=self.protocol,files=files))
  atomic(self.root/'LATEST_SNAPSHOT.json',dict(path=str(folder),status=status))
  return folder

 def install(self,snapshot):
  """Restore direct native state and epoch CSV into one new transactional base."""
  snapshot=Path(snapshot);assert not self.chain,'Remote installation requires an empty local cursor'
  manifest=read(snapshot/'MANIFEST.json');assert manifest['contract']==self.protocol
  for n,m in manifest['files'].items():
   assert not Path(n).is_absolute() and '..' not in Path(n).parts
   assert sha(snapshot/n)==m['sha256'] and (snapshot/n).stat().st_size==m['bytes']
  assert read(snapshot/'CONTRACT.json')==self.contract
  cursor=read(snapshot/'CURSOR.json');state=torch.load(snapshot/'state.pt',map_location='cpu',weights_only=False)
  assert cursor['state_sha256']==sha(snapshot/'state.pt') and state['protocol']==self.protocol
  # A remote base already contains earlier updates. Preserve their CSV as a
  # base generation, rather than pretending those updates were just executed.
  name='remote-base-'+uuid.uuid4().hex[:10];folder=self.root/'generations'/name;folder.mkdir(parents=True)
  shutil.copy2(snapshot/'state.pt',folder/'state.pt')
  counts={}
  for table,fields in self.schemas.items():
   rows=[]
   for p in sorted((snapshot/'csv').glob(f'epoch_*/{table}.csv')):
    with open(p,encoding='utf-8',newline='') as f:
     reader=csv.DictReader(f);assert reader.fieldnames==fields;rows.extend(reader)
   counts[table]=len(rows);csv_write(folder/'tables'/(table+'.csv'),rows,fields)
  assert counts==state['row_cursors']==cursor['row_cursors']
  assert all(state[k]==cursor[k] for k in ('updates','next_epoch','next_cursor'))
  # Native map paths retain original update ownership on export via a base map tree.
  if (snapshot/'maps').exists():shutil.copytree(snapshot/'maps',self.root/'BASE_MAPS',dirs_exist_ok=True)
  shutil.copy2(snapshot/'STATE_VERSIONS.csv',self.root/'BASE_VERSION_HISTORY.csv')
  files={str(p.relative_to(folder)).replace('\\','/'):file_meta(p) for p in folder.rglob('*') if p.is_file()}
  meta=dict(generation=name,update=state['updates'],protocol=self.protocol,row_cursors=counts,
       next_epoch=state['next_epoch'],next_cursor=state['next_cursor'],files=files,parent=None,parent_manifest_sha256=None,remote_base=True)
  atomic(folder/'MANIFEST.json',meta);atomic(self.root/'POINTER.json',dict(generation=name,manifest_sha256=sha(folder/'MANIFEST.json'),updates=state['updates']))
  self.chain=[meta]
