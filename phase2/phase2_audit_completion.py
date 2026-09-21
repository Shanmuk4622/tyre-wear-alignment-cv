import requests,json,pathlib,hashlib,time
root=pathlib.Path('phase2/completion');root.mkdir(exist_ok=True)
tree=json.loads(pathlib.Path('phase2/manifests/phase2_completion_tree.json').read_text());rev=tree['revision'];prefix=tree['prefix']; entries={e['path']:e for e in tree['entries']}
s=requests.Session()
def fetch(path):
 p=root/path.removeprefix(prefix+'/');p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():
  for attempt in range(5):
   r=s.get('https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/resolve/'+rev+'/'+path,timeout=120)
   if r.status_code==429 or r.status_code>=500:time.sleep(2**attempt);continue
   r.raise_for_status();p.write_bytes(r.content);break
  else:raise RuntimeError(path)
 return p
runs=[]
for path in entries:
 if '/runs/' not in path or not path.endswith('/MANIFEST.json'):continue
 mf=json.loads(fetch(path).read_text());base=path.rsplit('/',1)[0];row={'run':base.rsplit('/',1)[1],'manifest':mf,'checks':{}}
 for name,sha in mf['files'].items():
  if name.endswith(('.json', '.log')):
   p=fetch(base+'/'+name);assert hashlib.sha256(p.read_bytes()).hexdigest()==sha,(path,name);row['checks'][name]=True
  elif base+'/'+name in entries:
   row['checks'][name]=entries[base+'/'+name].get('lfs',{}).get('oid')==sha
 runs.append(row); print(row['run'],mf['status'],mf.get('epoch'),flush=True)
for path in entries:
 if '/reports/' in path and path.endswith('.json'):fetch(path)
(root/'audit.json').write_text(json.dumps(dict(revision=rev,prefix=prefix,runs=runs),indent=2))
for p in root.glob('runs/*/FINAL.json'):
 f=json.loads(p.read_text());print(p.parent.name,'val',f['best_score'],'test',f['test']['score'])
