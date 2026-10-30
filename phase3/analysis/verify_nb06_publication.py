"""Read-only verification of executed NB06 and its pinned direct-file review."""
from PIL import Image
import ast,base64,csv,gzip,hashlib,io,json,math,re,shutil,subprocess,sys,threading
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
import nbformat,numpy as np,requests
from huggingface_hub import HfApi
from nb05_public_reads import retry
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'revision2'))
import spec
from transactions import atomic,read,sha,digest
OUT=ROOT/'revision2_validation/nb06_publication';REPO='Shanmuk4622/tyre-wear-study';LOCAL=threading.local()

def session():
 if not hasattr(LOCAL,'session'):LOCAL.session=requests.Session();LOCAL.session.trust_env=False
 return LOCAL.session

def safe(folder,name):
 path=(folder/name).resolve();assert path.is_relative_to(folder.resolve()) and path!=folder.resolve();return path

def main():
 OUT.mkdir(parents=True,exist_ok=True);notebook=ROOT/'notebooks/phase3_NB06_TREAD_ONE_Paired_Review_And_Failure_Atlas.ipynb';raw=notebook.read_bytes()
 archive=OUT/'executed.ipynb'
 if archive.exists():assert archive.read_bytes()==raw,'Preserve distinct supplied executions separately'
 else:archive.write_bytes(raw)
 nb=nbformat.reads(raw.decode(),as_version=4);nbformat.validate(nb);streams=[];counts=[];embedded={};receipts=[]
 for cell in nb.cells:
  if cell.cell_type!='code':continue
  counts.append(cell.execution_count);assert cell.execution_count and not any(o.output_type=='error' for o in cell.outputs)
  for output in cell.outputs:
   if output.output_type!='stream':continue
   text=output.get('text','');text=''.join(text) if isinstance(text,list) else text;streams.append(text)
   decoder=json.JSONDecoder()
   for offset,char in enumerate(text):
    if char!='{':continue
    try:value,_=decoder.raw_decode(text[offset:])
    except ValueError:continue
    if isinstance(value,dict) and value.get('status')=='published_and_verified' and value.get('files')==35:receipts.append(value)
  for node in ast.walk(ast.parse(cell.source)):
   if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode' and node.args and isinstance(node.args[0],ast.Constant):
    data=json.loads(gzip.decompress(base64.b64decode(node.args[0].value)))
    for name,value in data.items():
     body=base64.b64decode(value);assert body==safe(ROOT,name).read_bytes(),name;embedded[name]=hashlib.sha256(body).hexdigest()
 assert counts==[1,2,3,4,5,6] and len(embedded)==36 and receipts and all(r==receipts[0] for r in receipts)
 delivery=read(ROOT/'revision2_validation/nb06_delivery/DELIVERY.json');assert embedded==delivery['embedded_files']
 receipt=receipts[-1];assert receipt['owner_released'] and receipt['zip_uploads']==0
 atomic(OUT/'PUBLICATION.json',receipt);(OUT/'notebook_streams.txt').write_text('\n'.join(streams),encoding='utf-8')
 dest=receipt['destination'];revision=receipt['revision'];folder=OUT/'remote';folder.mkdir(exist_ok=True)
 api=HfApi(token=False);entries=retry(lambda:list(api.list_repo_tree(REPO,path_in_repo=dest,revision=revision,recursive=True,repo_type='dataset')))
 tree={v.path[len(dest)+1:]:v for v in entries if hasattr(v,'size')}
 def fetch(remote,local,expected=None):
  entry=tree[remote];target=safe(folder,local);target.parent.mkdir(parents=True,exist_ok=True)
  def valid():
   if not target.exists() or target.stat().st_size!=entry.size:return False
   body=target.read_bytes()
   if entry.lfs:return hashlib.sha256(body).hexdigest()==entry.lfs.sha256
   return hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()==entry.blob_id
  if not valid():
   def transfer():
    with session().get(f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/{dest}/{remote}',timeout=(20,150),stream=True) as response:
     response.raise_for_status()
     with open(target.with_name(target.name+'.part'),'wb') as f:
      for block in response.iter_content(4*1024**2):f.write(block)
    target.with_name(target.name+'.part').replace(target)
   retry(transfer)
  assert valid(),local
  if expected:assert sha(target)==expected['sha256'] and target.stat().st_size==expected['bytes']
  return target
 manifest=read(fetch('MANIFEST.json','REMOTE_MANIFEST.json'));owner=read(fetch('OWNER.json','OWNER.json'))
 assert manifest['format']=='immutable-direct-files-v1' and len(manifest['files'])+1==35 and owner==dict(owner=None,expires=0)
 assert dest.endswith('/'+manifest['contract'])
 assert not any(name.endswith(('.zip','.npz','.sqlite','.pt')) for name in manifest['files'])
 with ThreadPoolExecutor(max_workers=4) as pool:
  jobs=[pool.submit(fetch,value['object'],name,value) for name,value in manifest['files'].items()]
  for i,job in enumerate(as_completed(jobs),1):job.result();print('Verified review payload',i,'/ 34',flush=True)
 contract=read(folder/'CONTRACT.json');assert digest(contract)==manifest['contract'] and contract['independent_new_tyres']==0
 assert contract['source_executed_notebook_sha256']==sha(ROOT/'notebooks/phase3_NB05_TREAD_ONE_Shared_Gradient_Paired_Pilot.ipynb')
 assert contract['source_publications']==read(ROOT/'revision2_validation/nb05_publication/NOTEBOOK_COMPLETION.json')['publications']
 for name,h in contract['analysis_sources'].items():assert sha(folder/'source'/name)==h==sha(ROOT/name)
 expected={'endpoint_metrics':4,'validation_trajectory':120,'epoch_training_gradient':120,'endpoint_class_metrics':12,'endpoint_tyre_metrics':8,'calibration_reliability':40,'selective_risk':84,'validation_prediction_stability':162,'paired_image_deltas':206,'resource_timing_summary':2}
 counts_csv={};max_numeric_difference=0.
 for name,count in expected.items():
  with open(folder/'analysis'/(name+'.csv'),encoding='utf-8',newline='') as f:r=csv.DictReader(f);fields=r.fieldnames;actual=list(r)
  with open(ROOT/'revision2_validation/nb05_publication/analysis'/(name+'.csv'),encoding='utf-8',newline='') as f:r=csv.DictReader(f);assert fields==r.fieldnames;local=list(r)
  assert len(actual)==len(local)==count;counts_csv[name]=count
  for a,b in zip(actual,local):
   for key in fields:
    if a[key]==b[key]:continue
    try:x,y=float(a[key]),float(b[key])
    except ValueError:raise AssertionError((name,key,a[key],b[key]))
    assert math.isfinite(x) and math.isfinite(y) and abs(x-y)<=1e-9*max(1.,abs(y)),(name,key,x,y)
    max_numeric_difference=max(max_numeric_difference,abs(x-y))
 assert len(list((folder/'analysis').glob('*.csv')))==10
 scientific=read(folder/'verification/SCIENTIFIC_VERIFICATION.json');assert scientific['status']=='passed' and scientific['core_rows']==1561734
 assert all(case['test_native_maps_checked']==206 for case in scientific['cases'])
 assert read(folder/'verification/DOWNLOAD.json')['published_files_including_manifests']==3515
 preflight=read(folder/'verification/PREFLIGHT_VERIFICATION.json');assert preflight['status']=='passed' and all(max(c['native_differences'].values())==0 for c in preflight['cases'])
 html=(folder/'TREAD_ONE_NB05_Comparison_Explorer.html').read_text(encoding='utf-8');script=re.search(r'<script>(.*?)</script>',html,re.S).group(1)
 syntax=OUT/'explorer_syntax.js';syntax.write_text(script,encoding='utf-8');checked=subprocess.run([shutil.which('node'),'--check',str(syntax)],capture_output=True,text=True);assert checked.returncode==0,checked.stderr
 data=json.JSONDecoder().raw_decode(script.split('const DATA=',1)[1])[0];assert len(data['images'])==103 and len(data['records'])==412 and len(data['validation'])==120
 localhtml=(ROOT/'revision2_validation/nb05_publication/analysis/TREAD_ONE_NB05_Comparison_Explorer.html').read_text(encoding='utf-8')
 localdata=json.JSONDecoder().raw_decode(re.search(r'<script>(.*?)</script>',localhtml,re.S).group(1).split('const DATA=',1)[1])[0]
 image_blocks=0;maximum_heat_pixel_difference=0
 for image_id,item in data['images'].items():
  assert item['image_sha256']==localdata['images'][image_id]['image_sha256']
  for field in ('image','truth_masks'):
   with Image.open(io.BytesIO(base64.b64decode(item[field].split(',',1)[1]))) as image:assert image.size==(384,512);image.verify();image_blocks+=1
 for record,localrecord in zip(data['records'],localdata['records']):
  for key in ('condition','endpoint','image_id','epoch','prediction','correct','probabilities','points'):assert record[key]==localrecord[key]
  for field in ('masks','heats'):
   actual=[record[field]] if field=='masks' else record[field];original=[localrecord[field]] if field=='masks' else localrecord[field]
   for a,b in zip(actual,original):
    with Image.open(io.BytesIO(base64.b64decode(a.split(',',1)[1]))) as image:
     assert image.size==((384,512) if field=='masks' else (96,128));pixels=np.array(image)
    with Image.open(io.BytesIO(base64.b64decode(b.split(',',1)[1]))) as image:other=np.array(image)
    change=int(np.abs(pixels.astype(int)-other.astype(int)).max());assert change<=(0 if field=='masks' else 1)
    maximum_heat_pixel_difference=max(maximum_heat_pixel_difference,change);image_blocks+=1
 assert image_blocks==1854
 for figure in (folder/'figures').glob('*.png'):
  with Image.open(figure) as image:image.verify()
 report=dict(status='passed',date='2026-10-29',verified_utc=datetime.now(timezone.utc).isoformat(),executed_notebook_sha256=sha(archive),executed_notebook_bytes=len(raw),
  code_cells=counts,notebook_errors=0,embedded_files=36,revision=revision,destination=dest,contract=manifest['contract'],published_files=35,
  all_manifest_Git_LFS_and_SHA256_hashes_verified=True,owner_released=True,CSV_tables=counts_csv,derived_rows=sum(counts_csv.values()),max_numeric_difference_from_local=max_numeric_difference,
  original_NB05_publications_pinned=True,original_NB05_core_rows_reverified_on_Kaggle=1561734,interactive_source_images=103,interactive_endpoint_records=412,
  image_blocks_verified=image_blocks,maximum_mask_or_heat_pixel_difference=maximum_heat_pixel_difference,JS_syntax_valid=True,browser_render_verified=False,
  remote_explorer_bytes=(folder/'TREAD_ONE_NB05_Comparison_Explorer.html').stat().st_size,environment=contract['environment'],new_training=False,new_inference=False,
  new_independent_tyres=0,zip_uploads=0,remote_writes_by_verifier=0)
 atomic(OUT/'VERIFICATION.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
