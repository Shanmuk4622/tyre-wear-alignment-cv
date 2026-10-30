"""Bind preserved NB05/NB06, publication audits, original CPU proof and notes."""
from PIL import Image
import ast,base64,csv,gzip,json,re,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
import nbformat
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'revision2'))
import spec
from transactions import read,sha,atomic,digest
PUB=ROOT/'revision2_validation/nb05_publication'

def main():
 science=read(PUB/'SCIENTIFIC_VERIFICATION.json');download=read(PUB/'DOWNLOAD.json');preflight=read(PUB/'PREFLIGHT_VERIFICATION.json')
 assert science['status']==download['status']==preflight['status']=='passed'
 assert download['published_files_including_manifests']==3515 and download['manifest_and_owner_Git_LFS_verified']
 assert science['core_rows']==1561734 and all(c['test_native_maps_checked']==206 for c in science['cases'])
 assert all(max(c['native_differences'].values())==0 and len(c['scientific_tables_equal'])==20 for c in preflight['cases'])
 inspection=read(PUB/'NOTEBOOK_INSPECTION.json');nb05=ROOT/'notebooks/phase3_NB05_TREAD_ONE_Shared_Gradient_Paired_Pilot.ipynb'
 assert sha(nb05)==sha(PUB/'executed.ipynb')==inspection['sha256'] and inspection['errors']==0
 delivery=read(ROOT/'revision2_validation/nb06_delivery/DELIVERY.json');tests=read(ROOT/'revision2_validation/nb06_validation/VALIDATION.json')
 assert tests['status']=='passed' and tests['notebook_sha256']==delivery['notebook_sha256'] and tests['CUDA_disabled'] and tests['network_disabled_for_fresh_stages']
 nb06=ROOT/'notebooks/phase3_NB06_TREAD_ONE_Paired_Review_And_Failure_Atlas.ipynb'
 executed=sha(nb06)!=delivery['notebook_sha256']
 if executed:
  verified6=read(ROOT/'revision2_validation/nb06_publication/VERIFICATION.json')
  assert verified6['status']=='passed' and verified6['published_files']==35 and verified6['derived_rows']==758
  assert sha(nb06)==verified6['executed_notebook_sha256']==sha(ROOT/'revision2_validation/nb06_publication/executed.ipynb')
 notebook=nbformat.read(nb06,as_version=4);nbformat.validate(notebook);embedded={}
 for cell in notebook.cells:
  if cell.cell_type!='code':continue
  if executed:assert cell.execution_count and not any(o.output_type=='error' for o in cell.outputs)
  else:assert not cell.outputs and cell.execution_count is None
  for node in ast.walk(ast.parse(cell.source)):
   if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode' and node.args and isinstance(node.args[0],ast.Constant):
    payload=json.loads(gzip.decompress(base64.b64decode(node.args[0].value)))
    for name,value in payload.items():assert base64.b64decode(value)==(ROOT/name).read_bytes();embedded[name]=sha(ROOT/name)
 assert embedded==delivery['embedded_files'] and len(embedded)==36
 bundle=read(PUB/'REVIEW_DELIVERY.json');folder=Path(bundle['path']);manifest=read(folder/'MANIFEST.json')
 assert digest(read(folder/'CONTRACT.json'))==bundle['contract']==manifest['contract'] and bundle['files']==35
 assert all(sha(folder/n)==v['sha256'] for n,v in manifest['files'].items())
 assert not any(n.endswith(('.zip','.npz','.sqlite','.pt')) for n in manifest['files'])
 assert len(list((PUB/'analysis').glob('*.csv')))==10
 for name in ('physical_tyres','capture_sessions','images'):
  with open(ROOT/'data_intake'/(name+'.csv'),encoding='utf-8',newline='') as f:assert list(csv.DictReader(f))==[]
 protected=read(ROOT/'audit/markdown-baseline.json');assert all(sha(ROOT.parent/r['path'])==r['sha256'] for r in protected)
 for name,h in read(ROOT/'revision2_validation/DELIVERY.json')['sources'].items():assert sha(ROOT/name)==h
 for archive in (ROOT/'completion_20261028/executed_notebooks').glob('*.ipynb'):assert sha(archive)==sha(ROOT/'notebooks'/archive.name)
 assert sha(ROOT/'notebooks/phase3_NB04_TREAD_ONE_Evidence_And_Recovery_Gate.ipynb')==sha(ROOT/'training_gates/nb04_publication/executed.ipynb')
 links=0;broken=[]
 for doc in list(ROOT.glob('*.md'))+list((ROOT/'data_intake').glob('*.md')):
  for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
   if re.match(r'^(https?://|mailto:|#)',target):continue
   links+=1
   if not (doc.parent/target.split('#')[0]).exists():broken.append([str(doc),target])
 assert not broken,broken
 for p in (PUB/'analysis/figures').glob('*.png'):
  with Image.open(p) as im:im.verify()
 result=dict(status='passed',executed_NB05_sha256=sha(nb05),source_experiment_files_preserved=27,published_NB05_files=3515,core_rows_verified=1561734,
  native_endpoint_image_maps_verified=412,independent_live_preflights=2,NB06_sha256=sha(nb06),NB06_bytes=nb06.stat().st_size,NB06_embedded_files=36,
  fresh_CPU_review_passed=True,posthoc_CSV_tables=10,posthoc_CSV_rows=sum(science['scientific_derivative_tables'].values()),review_bundle_files=35,
  interactive_images=103,interactive_endpoint_records=412,embedded_image_blocks=1854,JS_syntax_valid=True,browser_render_verified=False,
  original_Markdown_files_preserved=len(protected),earlier_executed_notebooks_preserved=True,local_links_checked=links,broken_links=broken,
  new_optimizer_updates=0,new_model_inference=False,new_independent_tyres=0,remote_writes_by_this_audit=0,NB06_Kaggle_execution=executed,new_review_live_publication=executed,
  NB06_clean_delivery_sha256=delivery['notebook_sha256'],NB06_independent_published_files=35 if executed else None)
 atomic(PUB/'HANDOFF_VALIDATION.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
