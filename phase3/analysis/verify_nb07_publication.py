"""Independently verify NB07's pinned direct files using public read-only access."""
from PIL import Image
import base64,csv,hashlib,io,json,math,re,shutil,subprocess,sys,threading
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
import requests,numpy as np
from huggingface_hub import HfApi
from inspect_nb07_execution import inspect
from nb05_public_reads import retry
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'data_gate'))
from common_gate import read,sha,digest,atomic,inside,table,CHECKSUM,OVERLAY
OUT=ROOT/'data_gate_validation/nb07_publication';REPO='Shanmuk4622/tyre-wear-study';LOCAL=threading.local()

def session():
 if not hasattr(LOCAL,'session'):LOCAL.session=requests.Session();LOCAL.session.trust_env=False
 return LOCAL.session

def main():
 inspected=inspect();receipt=read(OUT/'PUBLICATION.json');dest=receipt['destination'];revision=receipt['revision'];folder=OUT/'remote';folder.mkdir(exist_ok=True)
 api=HfApi(token=False);entries=retry(lambda:list(api.list_repo_tree(REPO,path_in_repo=dest,revision=revision,recursive=True,repo_type='dataset')))
 tree={v.path[len(dest)+1:]:v for v in entries if hasattr(v,'size')}
 def fetch(remote,local,expected=None):
  entry=tree[remote];target=inside(folder,local);target.parent.mkdir(parents=True,exist_ok=True)
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
      for block in response.iter_content(1024**2):f.write(block)
    target.with_name(target.name+'.part').replace(target)
   retry(transfer)
  assert valid(),local
  if expected:assert sha(target)==expected['sha256'] and target.stat().st_size==expected['bytes'],local
  return target
 manifest=read(fetch('MANIFEST.json','REMOTE_MANIFEST.json'));owner=read(fetch('OWNER.json','OWNER.json'))
 assert manifest['format']=='immutable-direct-files-v1' and len(manifest['files'])+1==46 and owner==dict(owner=None,expires=0)
 assert dest=='phase3/tread-one/data-gate/v1/'+manifest['contract']
 assert not any(n.endswith(('.zip','.npz','.sqlite','.pt')) for n in manifest['files'])
 with ThreadPoolExecutor(max_workers=4) as pool:
  jobs=[pool.submit(fetch,v['object'],n,v) for n,v in manifest['files'].items()]
  for i,j in enumerate(as_completed(jobs),1):j.result();print('Verified NB07 direct payload',i,'/ 45',flush=True)
 publication=read(folder/'CONTRACT.json');contract=read(folder/'DATA_CONTRACT.json');summary=read(folder/'SUMMARY.json');schema=read(folder/'CSV_SCHEMA.json')
 assert digest(publication)==manifest['contract'] and publication['data_contract']==contract
 assert contract['dataset_checksum']==CHECKSUM and contract['overlay_sha256']==OVERLAY
 assert contract['source_hashes']==read(ROOT/'data_gate_outputs/CONTRACT.json')['source_hashes']
 assert contract['policy']==read(ROOT/'data_gate_outputs/POLICY.json') and read(folder/'POLICY.json')==contract['policy']
 for name,h in contract['source_hashes'].items():assert sha(folder/'source'/name)==h==sha(ROOT/name)
 assert publication['sources']=={Path(n).name:h for n,h in contract['source_hashes'].items()}
 assert contract['run_id']==digest(dict(dataset=CHECKSUM,overlay=OVERLAY,code=contract['source_hashes'],intake=contract['intake_CSV_hashes'],intake_media=contract['intake_media_hashes'],policy=contract['policy'],environment=contract['environment']))
 for n,h in contract['intake_CSV_hashes'].items():assert sha(folder/'input_csv'/n)==h==sha(ROOT/'data_intake'/n)
 assert not contract['intake_media_hashes'] and summary==read(ROOT/'data_gate_outputs/SUMMARY.json')
 assert len(schema)==22 and sum(v['rows'] for v in schema.values())==7614
 assert sum(n.endswith('.csv') for n in manifest['files'])==26
 localrun=Path(read(ROOT/'data_gate_outputs/LATEST.json')['path']);differences={};exact=[]
 for name,info in schema.items():
  p=folder/'csv'/(name+'.csv');assert sha(p)==info['sha256'] and p.stat().st_size==info['bytes'];actual=table(p)
  with p.open(encoding='utf-8',newline='') as f:assert csv.DictReader(f).fieldnames==info['columns']
  assert len(actual)==info['rows'];original=table(localrun/'csv'/p.name);assert len(actual)==len(original)
  if sha(p)==sha(localrun/'csv'/p.name):exact.append(name);continue
  changed=[]
  for index,(a,b) in enumerate(zip(actual,original)):
   assert set(a)==set(b)
   for key in a:
    if a[key]==b[key]:continue
    try:x,y=float(a[key]),float(b[key]);assert math.isfinite(x) and math.isfinite(y);delta=abs(x-y)
    except ValueError:changed.append(dict(row=index,column=key,kind='text',published=a[key],local=b[key]));continue
    changed.append(dict(row=index,column=key,kind='number',absolute_difference=delta))
  by_column={}
  for v in changed:
   if v['kind']!='number':continue
   values=by_column.setdefault(v['column'],dict(changed_cells=0,max_absolute_difference=0.));values['changed_cells']+=1;values['max_absolute_difference']=max(values['max_absolute_difference'],v['absolute_difference'])
  differences[name]=dict(changed_cells=len(changed),max_numeric_difference=max((v['absolute_difference'] for v in changed if v['kind']=='number'),default=0),numeric_by_column=by_column,text_changes=[v for v in changed if v['kind']=='text'])
 atomic(OUT/'CROSS_ENVIRONMENT_COMPARISON.json',dict(byte_equal_tables=exact,differing_tables=differences))
 # Source registries, annotation truth, identity, support, exposure and readiness
 # must agree exactly; only image resampling descriptors may vary by library.
 assert all(n in ('image_quality','capture_descriptors') and not d['text_changes'] for n,d in differences.items()),differences
 for n,d in differences.items():
  assert all(k.endswith('_0_255') or k.endswith('_0_1') or k in ('gray_entropy_bits','focus_laplacian_variance','foreground_minus_background_brightness','mean_focus_laplacian_variance','mean_foreground_minus_background_brightness') for k in d['numeric_by_column']),d
 point=table(folder/'csv/point_geometry.csv');assert sum(r['human_eligible']=='True' for r in point)==720 and len(point)==1632
 quality=table(folder/'csv/image_quality.csv');assert len(quality)==570 and sum(int(r['tread_outside_tyre_native_pixels'])>0 for r in quality)==45 and all(int(r['tread_outside_tyre_valid_pixels'])==0 for r in quality)
 assert len(table(folder/'csv/leakage_components.csv'))==10 and len(table(folder/'csv/source_tyres.csv'))==12
 exposure=table(folder/'csv/training_image_exposure.csv');assert sum(int(r['sampled_augmented_occurrences']) for r in exposure)==47040 and all(int(r['sampled_augmented_occurrences'])==0 for r in exposure if r['historical_split']!='train')
 assert len(table(folder/'csv/readiness_gaps.csv'))==12 and not summary['prospective_training_ready'] and not table(folder/'csv/intake_issues.csv')
 html=(folder/'TREAD_ONE_Data_Gate.html').read_text(encoding='utf-8');script=re.search(r'<script>(.*?)</script>',html,re.S).group(1)
 syntax=OUT/'explorer_syntax.js';syntax.write_text(script,encoding='utf-8');result=subprocess.run([shutil.which('node'),'--check',str(syntax)],capture_output=True,text=True);assert result.returncode==0,result.stderr
 data=json.JSONDecoder().raw_decode(script.split('const DATA=',1)[1])[0];assert len(data['images'])==570 and data['summary']==summary
 localhtml=(localrun/'TREAD_ONE_Data_Gate.html').read_text(encoding='utf-8');localdata=json.JSONDecoder().raw_decode(re.search(r'<script>(.*?)</script>',localhtml,re.S).group(1).split('const DATA=',1)[1])[0]
 maximum_preview_pixel_difference=0
 for a,b in zip(data['images'],localdata['images']):
  assert a['image_id']==b['image_id'] and a['image_sha256']==b['image_sha256']
  with Image.open(io.BytesIO(base64.b64decode(a['preview'].split(',',1)[1]))) as image:assert image.size==(144,192);pixels=np.asarray(image).astype(int)
  with Image.open(io.BytesIO(base64.b64decode(b['preview'].split(',',1)[1]))) as image:other=np.asarray(image).astype(int)
  maximum_preview_pixel_difference=max(maximum_preview_pixel_difference,int(np.abs(pixels-other).max()))
 for figure in (folder/'figures').glob('*.png'):
  with Image.open(figure) as image:image.verify()
 protected=read(ROOT/'audit/markdown-baseline.json');assert all(sha(ROOT.parent/r['path'])==r['sha256'] for r in protected)
 assert all(sha(ROOT/n)==h for n,h in read(ROOT/'revision2_validation/DELIVERY.json')['sources'].items())
 report=dict(status='passed',verified_utc=datetime.now(timezone.utc).isoformat(),executed_notebook_sha256=inspected['sha256'],executed_notebook_bytes=inspected['bytes'],code_cells=inspected['execution_counts'],notebook_errors=0,embedded_files=42,
  revision=revision,destination=dest,publication_contract=manifest['contract'],run_id=contract['run_id'],published_files=46,published_CSV_files=26,structured_CSV_tables=22,structured_rows=7614,all_Git_LFS_and_manifest_SHA256_hashes_verified=True,owner_released=True,
  source_images=570,original_tyres=12,conservative_leakage_components=10,human_points=720,weak_points=912,raw_disagreement_images=45,unignored_disagreement_images=0,saved_training_occurrences=47040,
  source_integrity_passed=True,historical_split_consistent=True,prospective_training_ready=False,readiness_gaps=12,intake_errors=0,unresolved_video_mappings=3,
  byte_equal_CSV_tables=len(exact),differing_CSV_tables=differences,embedded_previews_verified=570,maximum_preview_pixel_difference=maximum_preview_pixel_difference,JS_syntax_valid=True,browser_render_verified=False,environment=contract['environment'],
  original_Markdown_preserved=len(protected),frozen_experiment_sources_preserved=27,zip_uploads=0,new_training=False,new_inference=False,new_independent_tyres=0,remote_writes_by_verifier=0)
 atomic(OUT/'VERIFICATION.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
