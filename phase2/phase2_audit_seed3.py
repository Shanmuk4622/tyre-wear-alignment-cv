import requests,json,pathlib,hashlib,shutil
s=requests.Session();repo='Shanmuk4622/tyre-wear-study';rev=s.get('https://huggingface.co/api/datasets/'+repo,timeout=60).json()['sha'];prefix='phase2/training-r1/661e97642e10bf8cfcbb8f581142069733789eb8f413dc70654aa00dd7854af3';base=f'https://huggingface.co/datasets/{repo}/resolve/{rev}/'
r=s.get(f'https://huggingface.co/api/datasets/{repo}/tree/{rev}/{prefix}?recursive=true&limit=1000',timeout=60);r.raise_for_status();entries={e['path']:e for e in r.json()};root=pathlib.Path('phase2/completion_seed3');root.mkdir(exist_ok=True);runs=[]
for path in entries:
 if '/runs/' not in path or not path.endswith('/MANIFEST.json'):continue
 r=s.get(base+path,timeout=60);r.raise_for_status();m=r.json();dest=root/'runs'/m['job']['id'];dest.mkdir(parents=True,exist_ok=True);(dest/'MANIFEST.json').write_bytes(r.content);checks={}
 for name,sha in m['files'].items():
  if name.endswith(('.json','.log')):
   old=pathlib.Path('phase2/completion/runs')/m['job']['id']/name
   if old.exists() and hashlib.sha256(old.read_bytes()).hexdigest()==sha:shutil.copy2(old,dest/name)
   else:
    rr=s.get(base+path.rsplit('/',1)[0]+'/'+name,timeout=120);rr.raise_for_status();(dest/name).write_bytes(rr.content)
   checks[name]=hashlib.sha256((dest/name).read_bytes()).hexdigest()==sha
  else:checks[name]=entries[path.rsplit('/',1)[0]+'/'+name].get('lfs',{}).get('oid')==sha
 assert all(checks.values()),checks;runs.append(dict(run=m['job']['id'],manifest=m,checks=checks));print(m['job']['id'],m['status'],m['epoch'],flush=True)
(root/'audit.json').write_text(json.dumps(dict(revision=rev,prefix=prefix,runs=runs),indent=2));(root/'tree.json').write_text(json.dumps(list(entries.values()),indent=2))
f=json.loads((root/'runs/yolo26m-combined-seed3/FINAL.json').read_text());print('SEED3',rev,'VAL',f['best_score'],'TEST',f['test']['score']);print('REPORTS',[p for p in entries if p.endswith('/REPORT.json')])
