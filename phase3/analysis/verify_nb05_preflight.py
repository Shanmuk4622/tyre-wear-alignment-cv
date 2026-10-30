"""Independent pinned-source/native/CSV check of NB05's two live preflights."""
from PIL import Image
import csv,hashlib,json,sys
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda' or Path('/kaggle/input').exists()
import torch
from huggingface_hub import HfApi
from verify_nb05_publication import ROOT,OUT,REPO,get,safe,retry
sys.path.insert(0,str(ROOT/'revision2'))
import spec
from transactions import read,sha,atomic,digest
from common import delta
from metrics_r2 import SCHEMAS

def main():
 result=read(OUT/'NOTEBOOK_COMPLETION.json');api=HfApi(token=False);reports=[]
 # Study intent excludes the device/environment; audit on CPU is sufficient.
 smoke_id=spec.configuration(spec.locate(),'cpu','baseline',True)['study_id']
 for condition,proof in zip(spec.CONDITIONS,result['preflight']):
  dest='phase3/tread-one/r2/engineering/'+smoke_id+'/'+condition
  revision=result['publications'][-1]['revision'];folder=OUT/'preflight'/condition;folder.mkdir(parents=True,exist_ok=True)
  manifest=read(get(dest,revision,'MANIFEST.json',folder/'REMOTE_MANIFEST.json'));assert manifest['contract']==proof['protocol']
  tree=retry(lambda:list(api.list_repo_tree(REPO,path_in_repo=dest,revision=revision,recursive=True,repo_type='dataset')))
  lookup={v.path[len(dest)+1:]:v for v in tree if hasattr(v,'size')}
  def fetch(name):
   v=manifest['files'][name];target=safe(folder,name);entry=lookup[v['object']]
   if not target.exists() or sha(target)!=v['sha256']:get(dest,revision,v['object'],target)
   assert sha(target)==v['sha256'] and target.stat().st_size==v['bytes']==entry.size
   if entry.lfs:assert entry.lfs.sha256==v['sha256']
   else:
    body=target.read_bytes();assert entry.blob_id==hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()
   return name
  names=['CONTRACT.json','CURSOR.json','PREFLIGHT_PROOF.json','state.pt','reference_case/state.pt','reference_case/SCHEMA.json','SCHEMA.json']
  families=set(SCHEMAS)-{'resources','events','eval_timings'}
  names += [n for n in manifest['files'] if (n.startswith('csv/') or n.startswith('reference_case/csv/')) and Path(n).stem in families]
  if condition=='baseline':names+=['FROZEN_UPDATE_REFERENCE.pt','ORIGINAL_MATH_PROOF.json']
  with ThreadPoolExecutor(max_workers=4) as pool:
   for future in as_completed([pool.submit(fetch,name) for name in names]):future.result()
  owner=read(get(dest,revision,'OWNER.json',folder/'OWNER.json'));assert owner==dict(owner=None,expires=0)
  assert read(folder/'PREFLIGHT_PROOF.json')==proof and digest(read(folder/'CONTRACT.json'))==proof['protocol']
  a=torch.load(folder/'reference_case/state.pt',map_location='cpu',weights_only=False);b=torch.load(folder/'state.pt',map_location='cpu',weights_only=False)
  keys=('model','optimizer','scaler','scheduler','rng','losses','updates','epoch0','cursor','phase','eval_cursor','best','partial','history','row_cursors')
  differences={k:delta(a[k],b[k]) for k in keys};assert max(differences.values())==0 and b['updates']==4 and b['phase']=='completed'
  compared=[]
  for table in SCHEMAS:
   if table not in families:continue
   omitted={'checkpoint_before_sha256','checkpoint_after_sha256'}|{n for n in SCHEMAS[table] if n.endswith('_seconds')}
   def rows(base):
    result=[]
    for p in sorted((base/'csv').glob('epoch_*/'+table+'.csv')):
     with open(p,encoding='utf-8',newline='') as stream:
      r=csv.DictReader(stream);assert r.fieldnames==SCHEMAS[table];result.extend({k:v for k,v in row.items() if k not in omitted} for row in r)
    return result
   assert rows(folder)==rows(folder/'reference_case'),table;compared.append(table)
  source_rev=proof['published_source_revision'];source=folder/'pinned_update1';source.mkdir(exist_ok=True)
  source_manifest=read(get(dest,source_rev,'MANIFEST.json',source/'REMOTE_MANIFEST.json'))
  for name in ('state.pt','CURSOR.json'):
   value=source_manifest['files'][name];get(dest,source_rev,value['object'],source/name);assert sha(source/name)==value['sha256']
  original=torch.load(source/'state.pt',map_location='cpu',weights_only=False);assert original['updates']==1 and read(source/'CURSOR.json')['updates']==1
  math_diff=None
  if condition=='baseline':
   frozen=torch.load(folder/'FROZEN_UPDATE_REFERENCE.pt',map_location='cpu',weights_only=False)
   math_diff={k:delta(v,original[k]) for k,v in frozen.items()};assert max(math_diff.values())==0
  reports.append(dict(condition=condition,protocol=proof['protocol'],source_revision=source_rev,final_read_revision=revision,native_differences=differences,
    scientific_tables_equal=compared,original_r1_update_math=math_diff,owner_released=True,downloaded_verified_preflight_files=len(names)+2,
    source_updates=1,completed_updates=4,new_measured_updates_by_verifier=0,scope='Independent native states and 20 scientific CSV families; optional engineering maps not all downloaded'))
  print(condition,'independent live source/native/20-table proof passed',flush=True)
 atomic(OUT/'PREFLIGHT_VERIFICATION.json',dict(status='passed',cases=reports,network_read_only=True,remote_writes=0,new_optimizer_updates=0))

if __name__=='__main__':main()
