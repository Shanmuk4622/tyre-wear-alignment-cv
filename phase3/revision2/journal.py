"""Atomic local SQLite units; published evidence is ordinary direct CSV/native files."""
import csv,json,os,shutil,sqlite3,uuid
from pathlib import Path
import numpy as np,torch
from transactions import atomic,canonical,digest,read,sha,file_meta,csv_write

def inside(root,name):
 p=(Path(root)/name).resolve();assert p.is_relative_to(Path(root).resolve()) and p!=Path(root).resolve();return p

def link(source,target):
 target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists():assert sha(source)==sha(target);return
 try:os.link(source,target)
 except OSError:shutil.copy2(source,target)

def native(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
 with open(tmp,'wb') as f:
  if path.suffix=='.npy':np.save(f,value,allow_pickle=False)
  elif path.suffix=='.pt':torch.save(value,f)
  else:f.write(canonical(value))
  f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
 if os.name!='nt':
  fd=os.open(path.parent,os.O_DIRECTORY)
  try:os.fsync(fd)
  finally:os.close(fd)

class Journal:
 def __init__(self,root,contract,schemas):
  self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True);self.contract=contract;self.protocol=digest(contract);self.schemas=schemas
  if (self.root/'CONTRACT.json').exists():assert read(self.root/'CONTRACT.json')==contract,'Incompatible run; do not restart under the old identity'
  else:atomic(self.root/'CONTRACT.json',contract)
  self.db=sqlite3.connect(self.root/'journal.sqlite',timeout=60,isolation_level=None)
  self.db.execute('PRAGMA synchronous=FULL');self.db.execute('PRAGMA journal_mode=DELETE')
  self.db.executescript('CREATE TABLE IF NOT EXISTS records(family TEXT,idx INTEGER,epoch INTEGER,seq INTEGER,payload TEXT,PRIMARY KEY(family,idx)); CREATE TABLE IF NOT EXISTS commits(seq INTEGER PRIMARY KEY,state_name TEXT,sha256 TEXT,bytes INTEGER,cursor TEXT,pruned INTEGER DEFAULT 0); CREATE TABLE IF NOT EXISTS artifacts(name TEXT PRIMARY KEY,sha256 TEXT,bytes INTEGER);')
  self.db.execute('CREATE INDEX IF NOT EXISTS records_unit ON records(seq,family,idx)')
  latest=self.db.execute('SELECT seq FROM commits ORDER BY seq DESC LIMIT 1').fetchone();self.seq=latest[0] if latest else -1
 def close(self):self.db.close()
 def load(self):
  if self.seq<0:return None
  _,name,h,size,cursor,_=self.db.execute('SELECT * FROM commits WHERE seq=?',(self.seq,)).fetchone();p=inside(self.root,name)
  assert p.stat().st_size==size and sha(p)==h,'Corrupt latest native checkpoint; no silent reset'
  state=torch.load(p,map_location='cpu',weights_only=False);c=json.loads(cursor)
  assert state['protocol']==self.protocol and state['row_cursors']==c['row_cursors']
  actual={t:0 for t in self.schemas}
  actual.update({t:n for t,n in self.db.execute('SELECT family,COUNT(*) FROM records GROUP BY family')})
  assert actual==state['row_cursors'],'Native/CSV observation cursor disagreement'
  assert all(state[k]==c[k] for k in ('updates','epoch0','cursor','phase','eval_cursor'))
  assert self.db.execute('PRAGMA quick_check').fetchone()[0]=='ok'
  parent=None
  for seq,cursor in self.db.execute('SELECT seq,cursor FROM commits ORDER BY seq'):
   rs={t:[] for t in self.schemas}
   for family,payload in self.db.execute('SELECT family,payload FROM records WHERE seq=? ORDER BY family,idx',(seq,)):
    row=json.loads(payload);row.pop('checkpoint_after_sha256',None);rs[family].append(row)
   h=digest(rs);meta=json.loads(cursor);assert h==meta['unit_rows_sha256'],'Corrupt recorded observations'
   parent=digest(dict(parent=parent,unit=h));assert parent==meta['history_sha256'],'Broken observation hash chain'
  assert parent==state['history_sha256'] and state['unit_rows_sha256']==c['unit_rows_sha256']
  return state
 def rows(self,table,epoch=None):
  sql='SELECT payload FROM records WHERE family=?';args=[table]
  if epoch is not None:sql+=' AND epoch=?';args.append(epoch)
  sql+=' ORDER BY idx'
  for (payload,) in self.db.execute(sql,args):yield json.loads(payload)
 def commit(self,state,tables=None,artifacts=None,fault=None,restoring=False,known_artifacts=None):
  tables=tables or {};artifacts=artifacts or {};assert state['protocol']==self.protocol
  assert set(tables)<=set(self.schemas)
  counts={t:0 for t in self.schemas} if self.seq<0 else json.loads(self.db.execute('SELECT cursor FROM commits WHERE seq=?',(self.seq,)).fetchone()[0])['row_cursors']
  before=dict(counts)
  normalized={t:[dict(r,condition=self.contract['condition']) for r in tables.get(t,[])] for t in self.schemas}
  for table in self.schemas:
   for row in tables.get(table,[]):assert set(row)<=set(self.schemas[table]),(table,set(row)-set(self.schemas[table]))
   counts[table]+=len(tables.get(table,[]))
  digest_rows={t:[{k:v for k,v in r.items() if k!='checkpoint_after_sha256'} for r in rs] for t,rs in normalized.items()}
  state['unit_rows_sha256']=digest(digest_rows)
  previous=None if self.seq<0 else json.loads(self.db.execute('SELECT cursor FROM commits WHERE seq=?',(self.seq,)).fetchone()[0])['history_sha256']
  state['history_sha256']=digest(dict(parent=previous,unit=state['unit_rows_sha256']))
  state['row_cursors']=counts;seq=self.seq+1;name=f'checkpoints/unit_{seq:08d}-{uuid.uuid4().hex[:8]}.pt';native(inside(self.root,name),state)
  checkpoint=file_meta(inside(self.root,name));artifact_meta=dict(known_artifacts or {})
  for relative,value in artifacts.items():
   target=inside(self.root,'artifacts/'+relative);native(target,value);artifact_meta['artifacts/'+relative]=file_meta(target)
  if fault=='after_native':os._exit(86)
  cursor={k:state[k] for k in ('updates','epoch0','cursor','phase','eval_cursor','row_cursors','unit_rows_sha256','history_sha256')}
  self.db.execute('BEGIN IMMEDIATE')
  try:
   for table in self.schemas:
    records=[]
    for i,row in enumerate(normalized[table]):
     r=dict(row)
     if table=='train_steps' and not restoring:r['checkpoint_after_sha256']=checkpoint['sha256']
     records.append((table,before[table]+i,int(r.get('epoch') or 0),seq,canonical(r).decode()))
    self.db.executemany('INSERT INTO records VALUES(?,?,?,?,?)',records)
   if fault=='after_rows':os._exit(86)
   self.db.execute('INSERT INTO commits(seq,state_name,sha256,bytes,cursor) VALUES(?,?,?,?,?)',(seq,name,checkpoint['sha256'],checkpoint['bytes'],canonical(cursor).decode()))
   for n,m in artifact_meta.items():self.db.execute('INSERT OR REPLACE INTO artifacts VALUES(?,?,?)',(n,m['sha256'],m['bytes']))
   self.db.execute('COMMIT')
  except BaseException:
   self.db.execute('ROLLBACK');raise
  self.seq=seq
  if fault=='after_commit':os._exit(86)
  for old,n,h in self.db.execute('SELECT seq,state_name,sha256 FROM commits WHERE pruned=0 AND seq<?',(seq-1,)).fetchall():
   p=inside(self.root,n)
   if p.exists():assert sha(p)==h;self.db.execute('UPDATE commits SET pruned=1 WHERE seq=?',(old,));p.unlink()
  return checkpoint['sha256']
 def export(self,source_root,status,report=None):
  state=self.load();assert state is not None
  target=self.root/'snapshots'/f'unit_{self.seq:08d}-{uuid.uuid4().hex[:8]}';target.mkdir(parents=True)
  latest=self.db.execute('SELECT state_name FROM commits WHERE seq=?',(self.seq,)).fetchone()[0];link(inside(self.root,latest),target/'state.pt')
  atomic(target/'CONTRACT.json',self.contract);cache=self.root/'csv_cache';cache.mkdir(exist_ok=True)
  catalog=read(cache/'CATALOG.json') if (cache/'CATALOG.json').exists() else {};schema={}
  for table,fields in self.schemas.items():
   epochs=self.db.execute('SELECT epoch,COUNT(*),MAX(seq) FROM records WHERE family=? GROUP BY epoch ORDER BY epoch',(table,)).fetchall() or [(0,0,-1)]
   for epoch,count,last in epochs:
    name=f'csv/epoch_{epoch:04d}/{table}.csv';p=cache/name;key=dict(rows=count,last_seq=last)
    if name not in catalog or catalog[name]['key']!=key or not p.exists() or sha(p)!=catalog[name]['sha256']:
     csv_write(p,self.rows(table,epoch),fields);catalog[name]=dict(key=key,**file_meta(p))
    link(p,target/name);schema[name]=dict(family=table,columns=fields,rows=count,**file_meta(p))
  atomic(cache/'CATALOG.json',catalog)
  for name,h,size in self.db.execute('SELECT * FROM artifacts'):
   p=inside(self.root,name);assert sha(p)==h and p.stat().st_size==size;link(p,target/name)
  versions=[]
  for seq,name,h,size,cursor,pruned in self.db.execute('SELECT * FROM commits ORDER BY seq'):
   c=json.loads(cursor);versions.append(dict(unit=seq,updates=c['updates'],epoch0=c['epoch0'],cursor=c['cursor'],phase=c['phase'],state_sha256=h,unit_rows_sha256=c['unit_rows_sha256'],history_sha256=c['history_sha256'],local_native_present=int(inside(self.root,name).exists())))
  csv_write(target/'STATE_VERSIONS.csv',versions,['unit','updates','epoch0','cursor','phase','state_sha256','unit_rows_sha256','history_sha256','local_native_present'])
  atomic(target/'CURSOR.json',dict(status=status,protocol=self.protocol,unit=self.seq,state_sha256=sha(target/'state.pt'),**{k:state[k] for k in ('updates','epoch0','cursor','phase','eval_cursor','row_cursors','best')}))
  atomic(target/'SCHEMA.json',dict(version='tread-one-r2-evidence-1',files=schema,missing='Blank means unavailable/not applicable; observed zeros retained.'))
  for n,h in self.contract['sources'].items():
   p=target/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(Path(source_root)/n,p);assert sha(p)==h
  for n in ('console.log','FAILURES.jsonl','COMMIT_TIMINGS.csv','WORKER_LIFECYCLE.csv','NUMERICAL_ATTEMPTS.csv','REMOTE_BASE.json','REMOTE_HISTORY.jsonl'):
   if (self.root/n).exists():shutil.copy2(self.root/n,target/n)
  if report:report(target)
  files={p.relative_to(target).as_posix():file_meta(p) for p in target.rglob('*') if p.is_file()}
  assert not any(n.endswith(('.zip','.npz','.sqlite')) for n in files)
  atomic(target/'MANIFEST.json',dict(contract=self.protocol,files=files));atomic(self.root/'LATEST_SNAPSHOT.json',dict(path=str(target),status=status));return target
 def install(self,snapshot):
  assert self.seq<0,'Install only into an empty local journal'
  snapshot=Path(snapshot);m=read(snapshot/'MANIFEST.json');assert m['contract']==self.protocol and read(snapshot/'CONTRACT.json')==self.contract
  for n,v in m['files'].items():
   p=inside(snapshot,n);assert sha(p)==v['sha256'] and p.stat().st_size==v['bytes']
  state=torch.load(snapshot/'state.pt',map_location='cpu',weights_only=False);assert sha(snapshot/'state.pt')==read(snapshot/'CURSOR.json')['state_sha256']
  tables={t:[] for t in self.schemas}
  for n,v in read(snapshot/'SCHEMA.json')['files'].items():
   with open(snapshot/n,encoding='utf-8',newline='') as f:
    r=csv.DictReader(f);assert r.fieldnames==self.schemas[v['family']];tables[v['family']].extend(r)
  expected=state['row_cursors'];assert expected=={t:len(r) for t,r in tables.items()}
  # CSV strings round-trip scientifically; model/optimizer/RNG state remains native.
  artifacts={}
  for n,v in m['files'].items():
   if n.startswith('artifacts/'):
    link(snapshot/n,self.root/n);artifacts[n]=v
  self.commit(state,tables,restoring=True,known_artifacts=artifacts)
  atomic(self.root/'REMOTE_BASE.json',dict(manifest_sha256=sha(snapshot/'MANIFEST.json'),restored_updates=state['updates']))
 def prune_snapshots(self,keep=2):
  base=(self.root/'snapshots').resolve()
  for p in sorted(base.iterdir(),key=lambda q:q.stat().st_mtime)[:-keep]:
   resolved=p.resolve();assert resolved.is_relative_to(base) and resolved.name.startswith('unit_') and resolved.is_dir()
   shutil.rmtree(resolved) # Only derived, verified-published local snapshots; DB/CSV/artifacts remain.
