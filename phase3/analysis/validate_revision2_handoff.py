"""Bind preserved NB04, delivered/executed NB05, science, sources and notes."""
from PIL import Image
import ast,base64,csv,gzip,io,json,re,shutil,subprocess,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'revision2'))
import spec
import numpy as np,nbformat
from transactions import atomic,read,sha
from metrics_r2 import SCHEMAS
from report_r2 import rows

def main():
 publication=read(ROOT/'training_gates/nb04_publication/VERIFICATION.json');assert publication['status']=='passed' and publication['published_files']==572 and publication['live_HF_recovery_independently_verified']
 nb04=ROOT/'notebooks/phase3_NB04_TREAD_ONE_Evidence_And_Recovery_Gate.ipynb'
 assert sha(nb04)==publication['executed_notebook_sha256']==sha(ROOT/'training_gates/nb04_publication/executed.ipynb')
 assert all(d['owner_released'] and d['all_manifest_Git_LFS_hashes_verified'] for d in publication['devices'])
 delivery=read(ROOT/'revision2_validation/DELIVERY.json');tests=read(ROOT/'revision2_validation/VALIDATION.json')
 repair=read(ROOT/'revision2_validation/nb05_repair/REGRESSION.json')
 assert repair['status']=='passed' and repair['experiment_sources_unchanged']==27
 assert repair['notebook_sha256']==delivery['notebook_sha256']
 assert sha(ROOT/'analysis/nb05_launcher.py')==delivery['launcher_sha256']==repair['launcher_sha256']
 failed=ROOT/'revision2_validation/nb05_repair'/repair['failed_notebook_sha256'][:12]/'executed.ipynb'
 assert sha(failed)==repair['failed_notebook_sha256']
 assert tests['status']=='passed' and delivery['sources']==tests['source_hashes']
 assert all(sha(ROOT/n)==h for n,h in delivery['sources'].items()) and len(delivery['sources'])==27
 assert len(tests['offline_checks'])==16 and len(tests['actual_cases'])==2
 initial={r['initial_model_tensor_sha256'] for r in tests['actual_cases']};assert len(initial)==1
 assert not tests['measured_training_started'] and not tests['live_HF_tested']
 notebook=ROOT/'notebooks/phase3_NB05_TREAD_ONE_Shared_Gradient_Paired_Pilot.ipynb'
 executed=sha(notebook)!=delivery['notebook_sha256']
 if executed:
  publication5=ROOT/'revision2_validation/nb05_publication'
  assert sha(notebook)==sha(publication5/'executed.ipynb')==read(publication5/'NOTEBOOK_INSPECTION.json')['sha256']
  assert read(publication5/'DOWNLOAD.json')['published_files_including_manifests']==3515
  assert read(publication5/'SCIENTIFIC_VERIFICATION.json')['status']=='passed'
  assert read(publication5/'PREFLIGHT_VERIFICATION.json')['status']=='passed'
 nb=nbformat.read(notebook,as_version=4);nbformat.validate(nb);embedded={}
 for cell in nb.cells:
  if cell.cell_type!='code':continue
  if executed:assert cell.execution_count and not any(o.output_type=='error' for o in cell.outputs)
  else:assert not cell.outputs and cell.execution_count is None
  for c in ast.walk(ast.parse(cell.source)):
   if isinstance(c,ast.Call) and isinstance(c.func,ast.Attribute) and c.func.attr=='b64decode' and c.args and isinstance(c.args[0],ast.Constant):
    payload=json.loads(gzip.decompress(base64.b64decode(c.args[0].value)))
    for n,v in payload.items():assert base64.b64decode(v)==(ROOT/n).read_bytes();embedded[n]=sha(ROOT/n)
 assert embedded==delivery['sources']
 assert (ROOT/'analysis/nb05_launcher.py').read_text(encoding='utf-8') in nb.cells[1].source
 schema=read(ROOT/'revision2_validation/SCHEMA.json');assert schema['families']==SCHEMAS and schema['family_count']==23
 source={r['image_id']:r for r in __import__('data').plan(spec.locate())['rows']}
 out=ROOT/'revision2_validation/summary_exports';out.mkdir(exist_ok=True);cases=[];paired_samples={};paired_inputs={};paired_first={}
 for case in tests['actual_cases']:
  assert case['status']=='passed' and max(case['native_differences'].values())==0
  if case['condition']=='baseline':assert max(case['original_r1_update_math'].values())==0
  folder=Path(case['final_snapshot']);m=read(folder/'MANIFEST.json')
  assert all(sha(folder/n)==v['sha256'] for n,v in m['files'].items())
  train=list(rows(folder,'train_samples'));ev=list(rows(folder,'eval_predictions'));gradient=list(rows(folder,'task_gradients'))
  assert len(train)==32 and len(ev)==8 and len(gradient)==4
  assert all(source[r['image_id']]['role']=='train' and source[r['image_id']]['image_sha256']==r['image_sha256'] for r in train)
  assert all(source[r['image_id']]['role']=='validation' and r['split'] in ('validation','validation_fixture') for r in ev)
  assert all(float(r['head_gradient_max_change'])==0 for r in gradient)
  identity_fields=('update','epoch','cursor','slot','image_id','image_sha256','label','augmentation_seed','flip_horizontal','brightness','contrast','blur_applied','blur_radius','background_donor_image_id')
  paired_samples[case['condition']]=[{k:r[k] for k in identity_fields} for r in train]
  paired_inputs[case['condition']]={p.relative_to(folder/'artifacts').as_posix():sha(p) for p in (folder/'artifacts/train_maps').rglob('*.npy') if p.name in ('inputs.npy','targets.npy')}
  first=next(rows(folder,'train_steps'));paired_first[case['condition']]={k:first[k] for k in ('loss','classification','ordinal','segmentation','geometry','containment','alignment','background_kl')}
  for p in (folder/'artifacts').rglob('*.npy'):
   a=np.load(p,mmap_mode='r',allow_pickle=False);assert np.isfinite(a).all()
  lifecycle=folder/'WORKER_LIFECYCLE.csv';commit=folder/'COMMIT_TIMINGS.csv';assert lifecycle.exists() and commit.exists()
  with open(lifecycle,encoding='utf-8',newline='') as f:life=list(csv.DictReader(f))
  assert any(r['event']=='exited' and float(r['process_elapsed_seconds'])>0 for r in life)
  alias=out/f'TREAD_ONE_R2_{case["condition"]}_Engineering.html';shutil.copy2(folder/'TREAD_ONE_R2_Progress.html',alias)
  script=re.search(r'<script>(.*?)</script>',alias.read_text(encoding='utf-8'),re.S).group(1);p=out/(case['condition']+'_syntax.js');p.write_text(script,encoding='utf-8')
  r=subprocess.run([shutil.which('node'),'--check',str(p)],capture_output=True,text=True);assert r.returncode==0,r.stderr
  data=json.JSONDecoder().raw_decode(script.split('const DATA=',1)[1])[0];assert len(data['gallery'])==4
  for gallery in data['gallery']:
   with Image.open(io.BytesIO(base64.b64decode(gallery['image'].split(',',1)[1]))) as image:assert image.size==(384,512);image.verify()
  cases.append(dict(condition=case['condition'],training_occurrences=32,validation_fixture_observations=8,real_conflicts=case['real_conflicts'],applied_projections=case['projections'],scientific_tables_compared=len(case['scientific_tables_equal']),no_test_images_evaluated=True))
 assert paired_samples['baseline']==paired_samples['shared_projection'] and paired_inputs['baseline']==paired_inputs['shared_projection']
 assert paired_first['baseline']==paired_first['shared_projection']
 for n,h in read(ROOT/'revision2/BASE_SOURCES.json').items():assert sha(ROOT/n)==h
 protected=read(ROOT/'audit/markdown-baseline.json');assert all(sha(ROOT.parent/r['path'])==r['sha256'] for r in protected)
 for archived in (ROOT/'completion_20261028/executed_notebooks').glob('*.ipynb'):assert sha(archived)==sha(ROOT/'notebooks'/archived.name)
 for n,h in read(ROOT/'releases/DELIVERY.json')['files'].items():
  if not n.endswith('.ipynb'):assert sha(ROOT/n)==h
 atomic(ROOT/'revision2_validation/HANDOFF_VALIDATION.json',dict(status='validating'))
 links=0;broken=[]
 for doc in ROOT.glob('*.md'):
  for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
   if re.match(r'^(https?://|mailto:|#)',target):continue
   links+=1
   if not (doc.parent/target.split('#')[0]).exists():broken.append([doc.name,target])
 assert not broken,broken
 for p in (ROOT/'revision2').glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
 result=dict(status='passed',NB04_verified_HF_files=572,NB04_notebook_sha256=sha(nb04),NB04_two_T4_live_recovery_verified=True,
  NB05_notebook_sha256=sha(notebook),NB05_notebook_bytes=notebook.stat().st_size,source_files=27,core_CSV_families=23,
  schema_column_positions=schema['column_positions'],offline_checks=16,actual_cases=cases,paired_initial_model_sha256=next(iter(initial)),
  original_baseline_update_exact=True,paired_samples_augmentation_and_native_inputs_identical=True,paired_first_forward_losses_identical=True,all_native_optimizer_scheduler_scaler_RNG_cursor_recovery_exact=True,
  JS_syntax_and_embedded_images_valid=True,browser_render_verified=False,original_markdown_preserved=len(protected),original_r1_delivery_preserved=True,
  local_links_checked=links,broken_links=broken,measured_training_started=executed,r2_Kaggle_execution_tested=executed,r2_live_HF_tested=executed,remote_writes_by_this_turn=0,
  NB05_clean_delivery_sha256=delivery['notebook_sha256'],NB05_published_files=3515 if executed else None,
  notebook_only_lease_repair_checks=repair['check_count'],launcher_sha256=repair['launcher_sha256'],failed_NB05_notebook_preserved=True)
 atomic(ROOT/'revision2_validation/HANDOFF_VALIDATION.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
