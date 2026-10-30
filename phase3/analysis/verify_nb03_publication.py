"""Archive executed NB03 and independently verify its pinned public HF snapshot."""
from PIL import Image
import ast,base64,csv,gzip,hashlib,json,math,re,sys
from datetime import datetime,timezone
from pathlib import Path
from collections import defaultdict
assert Path(sys.prefix).name=='cv_conda'
import nbformat,numpy as np,requests
from huggingface_hub import HfApi
from verify_nb02_publication import retry

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'diagnostics/nb03_publication'
REPO='Shanmuk4622/tyre-wear-study'
REV='75400f019ba0c6741b5060925b9c956c0a1ca0a8'
CONTRACT='975c08501981976bb992620aba89478dda90d18505f05d3bbe43376baff3fa0d'
PROTOCOL='371c40c0c6aaedaf5c5970f7721a284b042494db7bb330065f6cbde8fc3a0b8c'
DEST='phase3/tread-one-r1/'+PROTOCOL+'/diagnostics/classifier-r1/'+CONTRACT
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def table(p):
 with open(p,encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 notebook=ROOT/'notebooks/phase3_NB03_TREAD_ONE_Classifier_Diagnostic.ipynb'
 raw=notebook.read_bytes();archive=OUT/'executed.ipynb'
 if archive.exists():assert archive.read_bytes()==raw
 else:archive.write_bytes(raw)
 nb=nbformat.reads(raw.decode(),as_version=4);nbformat.validate(nb)
 embedded={};streams=[];counts=[]
 for c in nb.cells:
  if c.cell_type!='code':continue
  assert c.execution_count and not any(o.output_type=='error' for o in c.outputs)
  counts.append(c.execution_count)
  streams.extend(''.join(o.get('text',[])) if isinstance(o.get('text'),list) else o.get('text','') for o in c.outputs if o.output_type=='stream')
  for call in ast.walk(ast.parse(c.source)):
   if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=='b64decode' and call.args and isinstance(call.args[0],ast.Constant):
    payload=json.loads(gzip.decompress(base64.b64decode(call.args[0].value)))
    for name,value in payload.items():
     body=base64.b64decode(value);assert body==(ROOT/name).read_bytes(),name
     embedded[name]=hashlib.sha256(body).hexdigest()
 assert counts==[1,2,3,4] and len(embedded)==15
 output='\n'.join(streams)
 milestones=re.findall(r'Verified HF diagnostic snapshot: ([0-9a-f]{40}) files: (\d+)',output)
 assert milestones and milestones[-1]==(REV,'27')
 receipts=[];decoder=json.JSONDecoder()
 for offset,ch in enumerate(output):
  if ch!='{':continue
  try:obj,_=decoder.raw_decode(output[offset:])
  except json.JSONDecodeError:continue
  if isinstance(obj,dict) and obj.get('status')=='published_and_verified':receipts.append(obj)
 receipt=receipts[-1]
 assert receipt['revision']==REV and receipt['destination']==DEST and receipt['owner_released'] and receipt['zip_uploads']==0
 (OUT/'PUBLICATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
 (OUT/'notebook_streams.txt').write_text(output,encoding='utf-8')
 api=HfApi(token=False)
 entries=retry(lambda:list(api.list_repo_tree(REPO,path_in_repo=DEST,revision=REV,recursive=True,repo_type='dataset')))
 files={v.path[len(DEST)+1:]:v for v in entries if hasattr(v,'size')}
 session=requests.Session();session.trust_env=False
 def fetch(name):
  assert name in files and '..' not in Path(name).parts and not Path(name).is_absolute()
  p=OUT/'remote'/name;p.parent.mkdir(parents=True,exist_ok=True);item=files[name]
  def valid():
   if not p.exists() or p.stat().st_size!=item.size:return False
   if item.lfs:return sha(p)==item.lfs.sha256
   body=p.read_bytes();return hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()==item.blob_id
  if not valid():
   def transfer():
    url=f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/{DEST}/{name}'
    with session.get(url,timeout=(20,120),stream=True) as response:
     response.raise_for_status()
     with open(p.with_name(p.name+'.part'),'wb') as f:
      for block in response.iter_content(4*1024**2):f.write(block)
    p.with_name(p.name+'.part').replace(p)
   retry(transfer)
  assert valid(),name
  print('Verified published file:',name,p.stat().st_size,flush=True)
  return p
 manifest=read(fetch('MANIFEST.json'));assert manifest['contract']==CONTRACT
 assert set(files)==set(manifest['files'])|{'MANIFEST.json','OWNER.json'}
 assert len(manifest['files'])+1==27 and not any(n.endswith(('.zip','.npz')) for n in files)
 for name,meta in manifest['files'].items():
  p=fetch(name);assert sha(p)==meta['sha256'] and p.stat().st_size==meta['bytes']
 owner=read(fetch('OWNER.json'));assert owner=={'owner':None,'expires':0}
 root=OUT/'remote';contract=read(root/'CONTRACT.json');assert hashlib.sha256(canonical(contract)).hexdigest()==CONTRACT
 assert contract['source_protocol']==PROTOCOL and contract['source_revision']=='a0296f18e02b12d28f7e51377783d68538032209'
 assert len(contract['rows'])==467 and len(contract['jobs'])==6 and not contract['fixture']
 for name,h in contract['sources'].items():assert sha(root/'source'/name)==h==embedded[name]
 cursor=read(root/'CURSOR.json');assert cursor['status']=='complete' and len(cursor['units'])==96
 schema=read(root/'SCHEMA.json');rowcounts={}
 for name,spec in schema['tables'].items():
  p=root/'csv'/(name+'.csv');assert sha(p)==spec['sha256']
  with open(p,encoding='utf-8',newline='') as f:
   reader=csv.DictReader(f);assert reader.fieldnames==spec['columns'];rowcounts[name]=sum(1 for _ in reader)
  assert rowcounts[name]==spec['rows']
 assert rowcounts=={'predictions':14010,'regions':28020,'points':84060,'latents':168120,'evidence':42030,'timings':90,'gradients':270}
 rows=table(root/'csv/predictions.csv');overlay={r['image_id']:r for r in read(ROOT/'completion_20261028/hf/OVERLAY.json')['rows']}
 groups=defaultdict(list)
 for r in rows:
  original=overlay[r['image_id']];assert r['split']==original['role'] and r['split'] in ('train','validation')
  assert int(r['label'])==original['class_index'] and r['image_sha256']==original['image_sha256']
  p=np.array([float(r[f'probability_{c}']) for c in range(3)])
  assert np.isfinite(p).all() and abs(p.sum()-1)<1e-6 and int(p.argmax())==int(r['prediction'])
  assert abs(float(r['nll'])+math.log(max(p[int(r['label'])],1e-12)))<1e-10
  groups[(r['job'],r['split'],r['variant'])].append(r)
 summaries=table(root/'summary/cohort_group_metrics.csv');checked=0
 for r in summaries:
  if r['level']!='cohort':continue
  members=groups[(r['job'],r['split'],r['variant'])]
  conf=np.bincount([3*int(v['label'])+int(v['prediction']) for v in members],minlength=9).reshape(3,3)
  support=conf.sum(1);seen=support>0;ba=float(np.mean(np.diag(conf)[seen]/support[seen]))
  assert abs(ba-float(r['balanced_accuracy_observed']))<1e-12 and len(members)==int(r['images']);checked+=1
 assert checked==60
 local=read(ROOT/'diagnostics/DIAGNOSTIC_COMPLETION.json');local_root=Path(local['snapshot'])
 local_rows={(r['job'],r['image_id'],r['variant']):r for r in table(local_root/'csv/predictions.csv')}
 changed=0;max_probability_difference=0.
 for r in rows:
  old=local_rows[(r['job'],r['image_id'],r['variant'])];changed+=r['prediction']!=old['prediction']
  max_probability_difference=max(max_probability_difference,max(abs(float(r[f'probability_{c}'])-float(old[f'probability_{c}'])) for c in range(3)))
 assert [j['tensor_sha256'] for j in contract['jobs']]==[j['tensor_sha256'] for j in read(local_root/'CONTRACT.json')['jobs']]
 arrays=np.load(root/'gallery/arrays.npy',mmap_mode='r',allow_pickle=False)
 assert arrays.shape==(78,3,19,128,96) and arrays.dtype==np.float32 and np.isfinite(arrays).all()
 result=dict(status='passed',verified_at_utc=datetime.now(timezone.utc).isoformat(),revision=REV,destination=DEST,
  notebook_sha256=sha(archive),executed_code_cells=4,notebook_errors=0,embedded_sources=15,
  source_protocol=PROTOCOL,diagnostic_contract=CONTRACT,published_files=27,remote_owner_released=True,
  core_csv_rows=rowcounts,core_records=sum(rowcounts.values()),cohort_summaries_independently_verified=checked,
  gallery_shape=list(arrays.shape),all_published_hashes_verified=True,zip_uploads=0,
  local_vs_kaggle_prediction_disagreements=changed,max_local_vs_kaggle_probability_difference=max_probability_difference,
  environment=contract['environment'],baseline_cohorts=[r for r in summaries if r['level']=='cohort' and r['variant']=='normal'],
  retained_console_publication_milestones=milestones,console_is_not_a_complete_publication_history=len(milestones)!=6,
  files={n:dict(bytes=files[n].size,sha256=sha(root/n)) for n in manifest['files']},
  remote_writes_by_this_audit=0,training_updates_by_this_audit=0)
 (OUT/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ('files','baseline_cohorts')},indent=2))

if __name__=='__main__':main()
