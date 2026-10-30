"""Read-only, pinned NB05 audit. Parses notebook data; never trains or publishes."""
from PIL import Image
import argparse,ast,base64,csv,gzip,hashlib,json,os,shutil,sys,threading
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda' or Path('/kaggle/input').exists()
import nbformat,numpy as np,requests
from huggingface_hub import HfApi
from nb05_public_reads import retry
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'revision2'))
import spec
from transactions import atomic,canonical,digest,read,sha
from metrics_r2 import SCHEMAS
OUT=ROOT/'revision2_validation/nb05_publication'
REPO='Shanmuk4622/tyre-wear-study'
LOCAL=threading.local()

def safe(base,name):
 p=(Path(base)/name).resolve();assert p.is_relative_to(Path(base).resolve()) and p!=Path(base).resolve()
 assert not Path(name).is_absolute() and '..' not in Path(name).parts
 return p

def session():
 if not hasattr(LOCAL,'session'):
  LOCAL.session=requests.Session();LOCAL.session.trust_env=False
 return LOCAL.session

def get(dest,rev,name,target):
 target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
 def transfer():
  with session().get(f'https://huggingface.co/datasets/{REPO}/resolve/{rev}/{dest}/{name}',timeout=(20,150),stream=True) as response:
   response.raise_for_status()
   with open(target.with_name(target.name+'.part'),'wb') as f:
    for block in response.iter_content(4*1024**2):f.write(block)
  target.with_name(target.name+'.part').replace(target)
 retry(transfer);return target

def verify_control_files(folder,tree):
 for logical,local in (('MANIFEST.json','REMOTE_MANIFEST.json'),('OWNER.json','OWNER.json')):
  body=(folder/local).read_bytes();item=tree[logical];assert len(body)==item['bytes']
  if item['lfs_sha256']:assert hashlib.sha256(body).hexdigest()==item['lfs_sha256']
  else:assert hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()==item['blob_id']

def inspect_notebook():
 OUT.mkdir(parents=True,exist_ok=True)
 path=ROOT/'notebooks/phase3_NB05_TREAD_ONE_Shared_Gradient_Paired_Pilot.ipynb';raw=path.read_bytes();archive=OUT/'executed.ipynb'
 if archive.exists():assert archive.read_bytes()==raw,'Preserve distinct supplied executions separately'
 else:archive.write_bytes(raw)
 nb=nbformat.reads(raw.decode(),as_version=4);nbformat.validate(nb);sources={};streams=[];counts=[];completed=[]
 for cell in nb.cells:
  if cell.cell_type!='code':continue
  counts.append(cell.execution_count);assert cell.execution_count and not any(o.output_type=='error' for o in cell.outputs)
  for output in cell.outputs:
   if output.output_type!='stream':continue
   text=output.get('text','');text=''.join(text) if isinstance(text,list) else text;streams.append(text)
   try:value=json.loads(text)
   except (ValueError,TypeError):continue
   if isinstance(value,dict) and value.get('status')=='completed' and value.get('measured_runs')==2:completed.append(value)
  for node in ast.walk(ast.parse(cell.source)):
   if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode' and node.args and isinstance(node.args[0],ast.Constant):
    payload=json.loads(gzip.decompress(base64.b64decode(node.args[0].value)))
    for n,v in payload.items():
     body=base64.b64decode(v);assert body==safe(ROOT,n).read_bytes(),n;sources[n]=hashlib.sha256(body).hexdigest()
 assert counts==[1,2,3] and len(sources)==27 and completed
 result=completed[-1];assert result['study']['study_id']=='16dfddd7672e70c3b321cc6adc4044d5a6c4453a75d7b84a29f72fdd6ab6c1dc'
 assert result['epochs_per_run']==60 and result['optimizer_updates_per_run']==2940
 assert len(result['preflight'])==2 and len(result['publications'])==2
 assert all(p['status']=='passed' and p['actual_live_HF_restore'] and max(p['native_differences'].values())==0 for p in result['preflight'])
 assert sources==read(ROOT/'revision2_validation/DELIVERY.json')['sources']
 assert (ROOT/'analysis/nb05_launcher.py').read_text(encoding='utf-8') in nb.cells[1].source
 atomic(OUT/'NOTEBOOK_COMPLETION.json',result)
 (OUT/'notebook_streams.txt').write_text('\n'.join(streams),encoding='utf-8')
 atomic(OUT/'NOTEBOOK_INSPECTION.json',dict(sha256=sha(archive),bytes=len(raw),code_cells=counts,errors=0,source_hashes=sources,experiment_sources=27,launcher_source_unchanged=True))
 return result

def metadata(result):
 api=HfApi(token=False);inventory=[]
 for condition,receipt in zip(spec.CONDITIONS,result['publications']):
  assert receipt['destination'].endswith('/'+condition) and receipt['owner_released'] and receipt['zip_uploads']==0
  dest=receipt['destination'];rev=receipt['revision'];folder=OUT/'remote'/condition;folder.mkdir(parents=True,exist_ok=True)
  tree=retry(lambda:list(api.list_repo_tree(REPO,path_in_repo=dest,revision=rev,recursive=True,repo_type='dataset')))
  lookup={item.path[len(dest)+1:]:dict(bytes=item.size,lfs_sha256=item.lfs.sha256 if item.lfs else None,blob_id=item.blob_id) for item in tree if hasattr(item,'size')}
  atomic(folder/'REMOTE_TREE.json',lookup)
  m=read(get(dest,rev,'MANIFEST.json',folder/'REMOTE_MANIFEST.json'));owner=read(get(dest,rev,'OWNER.json',folder/'OWNER.json'))
  verify_control_files(folder,lookup)
  assert m['format']=='immutable-direct-files-v1' and len(m['files'])+1==receipt['files'] and owner==dict(owner=None,expires=0)
  for name,v in m['files'].items():
   safe(folder,name);safe(folder,v['object']);item=lookup[v['object']]
   assert item['bytes']==v['bytes']
   if item['lfs_sha256']:assert item['lfs_sha256']==v['sha256']
   assert not name.endswith(('.zip','.npz','.sqlite'))
  for name in ('CONTRACT.json','CURSOR.json','SCHEMA.json','artifacts/endpoint_results/selected.json','artifacts/endpoint_results/final.json','artifacts/PAIR_COMPLETION.json','artifacts/PAIR_COMPARISON.csv','artifacts/INITIALIZATION.json'):
   value=m['files'][name];p=get(dest,rev,value['object'],folder/name);assert sha(p)==value['sha256']
  contract=read(folder/'CONTRACT.json');assert digest(contract)==m['contract'] and contract['sources']==read(OUT/'NOTEBOOK_INSPECTION.json')['source_hashes']
  counts=Counter();sizes=Counter()
  for name,v in m['files'].items():counts[Path(name).suffix]+=1;sizes[Path(name).suffix]+=v['bytes']
  inventory.append(dict(condition=condition,revision=rev,destination=dest,contract=m['contract'],files=len(m['files'])+1,bytes=sum(v['bytes'] for v in m['files'].values()),counts_by_extension=dict(counts),bytes_by_extension=dict(sizes),cursor=read(folder/'CURSOR.json'),selected=read(folder/'artifacts/endpoint_results/selected.json'),final=read(folder/'artifacts/endpoint_results/final.json')))
  print(condition,'manifest',len(m['files']),'files; endpoints',json.dumps(dict(selected=inventory[-1]['selected'],final=inventory[-1]['final'])),flush=True)
 atomic(OUT/'INVENTORY.json',inventory);return inventory

def download(result):
 tasks={};targets=[]
 for condition,receipt in zip(spec.CONDITIONS,result['publications']):
  folder=OUT/'remote'/condition;m=read(folder/'REMOTE_MANIFEST.json');tree=read(folder/'REMOTE_TREE.json')
  for name,value in m['files'].items():
   targets.append((folder,name,value));tasks.setdefault(value['sha256'],(receipt,value,tree[value['object']]))
 objects=OUT/'objects';objects.mkdir(exist_ok=True)
 def fetch(item):
  h,(receipt,value,tree)=item;p=objects/h
  if not p.exists() or p.stat().st_size!=value['bytes'] or sha(p)!=h:get(receipt['destination'],receipt['revision'],value['object'],p)
  assert p.stat().st_size==value['bytes'] and sha(p)==h
  if tree['lfs_sha256']:assert tree['lfs_sha256']==h
  else:
   b=p.read_bytes();assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==tree['blob_id']
  return h
 with ThreadPoolExecutor(max_workers=4) as pool:
  futures=[pool.submit(fetch,item) for item in tasks.items()]
  for i,future in enumerate(as_completed(futures),1):
   future.result()
   if i%100==0 or i==len(tasks):print('Verified content objects',i,'/',len(tasks),flush=True)
 for folder,name,value in targets:
  p=safe(folder,name);p.parent.mkdir(parents=True,exist_ok=True);source=objects/value['sha256']
  if p.exists():assert sha(p)==value['sha256'];continue
  try:os.link(source,p)
  except OSError:shutil.copy2(source,p)
 for condition in spec.CONDITIONS:
  folder=OUT/'remote'/condition;m=read(folder/'REMOTE_MANIFEST.json')
  verify_control_files(folder,read(folder/'REMOTE_TREE.json'))
  atomic(folder/'MANIFEST.json',dict(contract=m['contract'],files={n:{k:v for k,v in s.items() if k!='object'} for n,s in m['files'].items()}))
 atomic(OUT/'DOWNLOAD.json',dict(status='passed',logical_files=len(targets),published_files_including_manifests=len(targets)+len(spec.CONDITIONS),unique_content_objects=len(tasks),all_payload_SHA256_verified=True,Git_LFS_identities_verified=True,manifest_and_owner_Git_LFS_verified=True,remote_writes=0))

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--metadata',action='store_true');args=parser.parse_args()
 result=inspect_notebook()
 if args.metadata or not (OUT/'INVENTORY.json').exists():metadata(result)
 if not args.metadata:download(result)

if __name__=='__main__':main()
