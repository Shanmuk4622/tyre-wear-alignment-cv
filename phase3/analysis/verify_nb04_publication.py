"""Read-only audit of executed NB04 and both pinned public HF namespaces."""
from PIL import Image
import ast,base64,csv,gzip,hashlib,json,math,os,shutil,sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
from collections import defaultdict
assert Path(sys.prefix).name=='cv_conda'
import nbformat,numpy as np,requests,torch
from huggingface_hub import HfApi
from verify_nb02_publication import retry
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'training'))
sys.path.append(str(ROOT/'runtime'))
from transactions import atomic,canonical,digest,read,sha
from telemetry import SCHEMAS
from common import delta
OUT=ROOT/'training_gates/nb04_publication';REPO='Shanmuk4622/tyre-wear-study'
SCIENTIFIC=('train_steps','train_samples','region_stats','geometry_points','latent_metrics','evidence_metrics','class_confusion','gradient_modules','activation_modules','validation_predictions','epoch_summaries')

def tables(folder,name):
 result=[]
 for p in sorted((folder/'csv').glob(f'epoch_*/{name}.csv')):
  with open(p,encoding='utf-8',newline='') as f:
   r=csv.DictReader(f);assert r.fieldnames==SCHEMAS[name];result.extend(r)
 return result

def same_state(a,b):
 keys=('model','optimizer','scheduler','scaler','rng','losses','next_epoch','next_cursor','updates')
 differences={k:delta(a[k],b[k]) for k in keys};assert max(differences.values())==0,differences
 assert all(a['partial'][k]==b['partial'][k] for k in ('batches','images','loss_sum'))
 return differences

def main():
 OUT.mkdir(parents=True,exist_ok=True);raw=(ROOT/'notebooks/phase3_NB04_TREAD_ONE_Evidence_And_Recovery_Gate.ipynb').read_bytes()
 archive=OUT/'executed.ipynb'
 if archive.exists():assert archive.read_bytes()==raw,'Preserve each independently supplied execution separately'
 else:archive.write_bytes(raw)
 nb=nbformat.reads(raw.decode(),as_version=4);nbformat.validate(nb);embedded={};streams=[];counts=[]
 for cell in nb.cells:
  if cell.cell_type!='code':continue
  assert cell.execution_count and not any(o.output_type=='error' for o in cell.outputs)
  counts.append(cell.execution_count)
  streams.extend(''.join(o.get('text',[])) if isinstance(o.get('text'),list) else o.get('text','') for o in cell.outputs if o.output_type=='stream')
  for c in ast.walk(ast.parse(cell.source)):
   if isinstance(c,ast.Call) and isinstance(c.func,ast.Attribute) and c.func.attr=='b64decode' and c.args and isinstance(c.args[0],ast.Constant):
    payload=json.loads(gzip.decompress(base64.b64decode(c.args[0].value)))
    for n,b64 in payload.items():
     b=base64.b64decode(b64);assert b==(ROOT/n).read_bytes(),n;embedded[n]=hashlib.sha256(b).hexdigest()
 assert len(counts)==3 and len(embedded)==17
 output='\n'.join(streams);(OUT/'notebook_streams.txt').write_text(output,encoding='utf-8')
 candidates=[];decoder=json.JSONDecoder()
 for pos,ch in enumerate(output):
  if ch!='{':continue
  try:o,_=decoder.raw_decode(output[pos:])
  except json.JSONDecodeError:continue
  if isinstance(o,dict) and o.get('status')=='passed' and o.get('live_hf_tested') and isinstance(o.get('devices'),list) and o['devices'] and isinstance(o['devices'][0],dict):candidates.append(o)
 result=candidates[-1];assert len(result['devices'])==2 and not result['new_study_started']
 atomic(OUT/'NOTEBOOK_COMPLETION.json',result)
 api=HfApi(token=False);objects=OUT/'objects';objects.mkdir(exist_ok=True);trees={}
 def inventory(dest,revision):
  key=(dest,revision)
  if key not in trees:
   entries=retry(lambda:list(api.list_repo_tree(REPO,path_in_repo=dest,revision=revision,recursive=True,repo_type='dataset')))
   trees[key]={e.path[len(dest)+1:]:e for e in entries if hasattr(e,'size')}
  return trees[key]
 def fetch(dest,revision,name,target):
  assert not Path(name).is_absolute() and '..' not in Path(name).parts
  item=inventory(dest,revision)[name];target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
  key=item.lfs.sha256 if item.lfs else 'git-'+item.blob_id;cached=objects/key
  def valid(p):
   if not p.exists() or p.stat().st_size!=item.size:return False
   if item.lfs:return sha(p)==item.lfs.sha256
   b=p.read_bytes();return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==item.blob_id
  if not valid(target):
   if not valid(cached):
    def transfer():
     session=requests.Session();session.trust_env=False
     with session.get(f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/{dest}/{name}',timeout=(20,150),stream=True) as r:
      r.raise_for_status()
      with open(cached.with_name(cached.name+'.part'),'wb') as f:
       for block in r.iter_content(4*1024**2):f.write(block)
     cached.with_name(cached.name+'.part').replace(cached)
    retry(transfer)
   assert valid(cached),name
   if target.exists():target.unlink() # Only this audit's individually named cached link.
   try:os.link(cached,target)
   except OSError:shutil.copy2(cached,target)
  assert valid(target),name;return target
 reports=[]
 for device in result['devices']:
  contract_id=device['protocol'];receipt=device['publication'];dest=receipt['destination'];revision=receipt['revision'];folder=OUT/'remote'/contract_id
  assert receipt['owner_released'] and receipt['zip_uploads']==0
  files=inventory(dest,revision);m=read(fetch(dest,revision,'MANIFEST.json',folder/'MANIFEST.json'))
  assert m['contract']==contract_id and len(m['files'])+1==receipt['files']
  assert not any(n.endswith(('.zip','.npz')) for n in m['files'])
  # Sequential cache population prevents duplicate writers for equal content hashes.
  for i,(name,spec) in enumerate(m['files'].items()):
   p=fetch(dest,revision,name,folder/name);assert sha(p)==spec['sha256'] and p.stat().st_size==spec['bytes']
   if (i+1)%50==0:print(device['device'],'verified files',i+1,'/',len(m['files']),flush=True)
  assert read(fetch(dest,revision,'OWNER.json',folder/'OWNER.json'))=={'owner':None,'expires':0}
  contract=read(folder/'CONTRACT.json');assert digest(contract)==contract_id
  assert contract['source_hashes']==embedded and contract['device']==device['device'] and contract['mode']=='engineering_gate'
  for n,h in contract['source_hashes'].items():assert sha(folder/'source'/n)==h
  local=read(folder/'LOCAL_RESULTS.json');live=read(folder/'LIVE_RESULTS.json')
  assert local==device['local'] and live==device['live'] and live['restored_completed_updates']==1 and live['continued_to_updates']==3
  cases={};all_rows=0
  for case in sorted((folder/'cases').iterdir()):
   inner=read(case/'MANIFEST.json');cursor=read(case/'CURSOR.json');assert inner['contract']==contract_id
   assert all(sha(case/n)==v['sha256'] for n,v in inner['files'].items())
   counts_case={t:len(tables(case,t)) for t in SCHEMAS};assert counts_case==cursor['row_cursors']
   state=torch.load(case/'state.pt',map_location='cpu',weights_only=False)
   assert sha(case/'state.pt')==cursor['state_sha256'] and state['protocol']==contract_id
   assert all(state[k]==cursor[k] for k in ('updates','next_epoch','next_cursor','row_cursors'))
   cases[case.name]=dict(state=state,folder=case,counts=counts_case);all_rows+=sum(counts_case.values())
  reference=cases['uninterrupted'];comparisons={}
  for name in ('math_reference','paused','abrupt','installed','live_resume'):
   comparisons[name]=same_state(reference['state'],cases[name]['state'])
   if name=='math_reference':continue
   for table in SCIENTIFIC:
    ignored={'checkpoint_before_sha256','checkpoint_after_sha256'}|({k for k in SCHEMAS[table] if k.endswith('_seconds')} if table=='train_steps' else set())
    clean=lambda rs:[{k:v for k,v in r.items() if k not in ignored} for r in rs]
    assert clean(tables(reference['folder'],table))==clean(tables(cases[name]['folder'],table)),(name,table)
   for p in (reference['folder']/'maps').glob('update_*/*.npy'):assert sha(p)==sha(cases[name]['folder']/'maps'/p.relative_to(reference['folder']/'maps'))
  assert sum(reference['counts'].values())==705 and reference['counts']['resources']==5
  samples=tables(reference['folder'],'train_samples');steps=tables(reference['folder'],'train_steps');reconstruction=0.
  for step in steps:
   group=[r for r in samples if r['update']==step['update']];assert len(group)==8
   total=lambda k:sum(float(r[k]) for r in group)
   loss=total('label_smoothed_ce')/8+.2*total('ordinal_loss')/8+total('seg_bce_numerator')/max(total('seg_bce_denominator'),1)+total('seg_dice_loss')/8
   loss+=2*total('geometry_numerator')/max(total('geometry_weight'),1)+.1*total('containment_numerator')/max(total('containment_denominator'),1)+.1*total('alignment_numerator')/max(total('alignment_denominator'),1)+.2*total('background_kl')/8
   reconstruction=max(reconstruction,abs(loss-float(step['loss'])));assert reconstruction<2e-5
  for p in (reference['folder']/'maps').glob('update_*/*.npy'):
   a=np.load(p,mmap_mode='r',allow_pickle=False);assert np.isfinite(a).all()
  source_revision=live['published_source_revision'];source=OUT/'source_fixture'/contract_id
  for name in ('CURSOR.json','state.pt','MANIFEST.json'):
   relative='cases/published_base_fixture/'+name;fetch(dest,source_revision,relative,source/name)
  source_cursor=read(source/'CURSOR.json');assert source_cursor['updates']==1
  assert sha(source/'state.pt')==source_cursor['state_sha256']==sha(cases['published_base_fixture']['folder']/'state.pt')
  with open(cases['live_resume']['folder']/'STATE_VERSIONS.csv',newline='',encoding='utf-8') as f:versions=list(csv.DictReader(f))
  assert next(r for r in versions if r['update']=='1')['state_sha256']==sha(source/'state.pt')
  reports.append(dict(device=device['device'],protocol=contract_id,revision=revision,source_revision=source_revision,published_files=receipt['files'],
   published_bytes=sum(v['bytes'] for v in m['files'].values())+(folder/'MANIFEST.json').stat().st_size,
   owner_released=True,all_manifest_Git_LFS_hashes_verified=True,case_counts={n:c['counts'] for n,c in cases.items()},
   total_core_case_rows=all_rows,comparisons=comparisons,scientific_tables_equal_per_recovery=11,native_arrays_equal_per_recovery=9,
   max_loss_reconstruction_error=reconstruction,environment=contract['environment'],
   resource_samples=tables(reference['folder'],'resources'),step_measurements=steps,remote_directory=str(folder)))
  print(device['device'],'complete: all files, native states, CSV and pinned source checked',flush=True)
 audit=dict(status='passed',verified_at_utc=datetime.now(timezone.utc).isoformat(),executed_notebook_sha256=sha(archive),executed_notebook_bytes=len(raw),
   code_cells=counts,notebook_errors=0,embedded_source_files=17,devices=reports,published_files=sum(d['published_files'] for d in reports),
   total_core_case_rows=sum(d['total_core_case_rows'] for d in reports),live_HF_recovery_independently_verified=True,
   notebook_preserved=True,new_measured_study_started=False,remote_writes_by_audit=0,training_updates_by_audit=0)
 atomic(OUT/'VERIFICATION.json',audit);print(json.dumps({k:v for k,v in audit.items() if k!='devices'},indent=2))

if __name__=='__main__':main()
