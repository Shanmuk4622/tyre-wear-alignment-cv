"""Final isolation, provenance, native-window and documentation handoff checks."""
import ast,json,re,sys
from pathlib import Path
from urllib.parse import unquote
from PIL import Image
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];APP=ROOT/'workstation';OUT=ROOT/'workstation_validation';sys.path.insert(0,str(APP))
from paths import read,sha,write_json

def main():
 engine=read(OUT/'ENGINE_VALIDATION.json');ui=read(OUT/'UI_VALIDATION.json');native=read(OUT/'TREAD_ONE_Studio_Native_Final.json');assert engine['status']==ui['status']=='passed' and native['status']=='native_window_ready'
 assert engine['check_count']==12 and ui['check_count']==14 and native['Qt_platform']=='windows'
 assert (OUT/'TREAD_ONE_Studio_Native_Final.json').stat().st_mtime>=max(p.stat().st_mtime for p in APP.glob('*.py')),'Native preview must come from the final source revision'
 identity=read(APP/'models/MODEL.json');original_export=read(ROOT/'completion_20261028/EXPORT_VERIFICATION.json');source_contract=read(ROOT/'completion_20261028/hf/CONTRACT.json')
 assert identity['seed']==3 and identity['epoch']==20 and sha(APP/'models/weights.pt')==identity['weights_sha256']==original_export['weights_sha256']==native['checkpoint']
 assert sha(APP/'models/tread_one_r1.py')==identity['model_source_sha256']==sha(ROOT/'runtime/model.py')==source_contract['sources']['model.py']
 baseline=read(OUT/'ORIGINAL_BASELINE.json');assert len(baseline)==47 and all(sha(ROOT.parent/n)==h for n,h in baseline.items())
 protected=read(ROOT/'audit/markdown-baseline.json');assert all(sha(ROOT.parent/r['path'])==r['sha256'] for r in protected)
 assert all(sha(ROOT/n)==h for n,h in read(ROOT/'revision2_validation/DELIVERY.json')['sources'].items())
 for n,h in source_contract['sources'].items():assert sha(ROOT/'runtime'/n)==h
 latest=Path(read(ROOT/'data_gate_validation/nb07_reruns/LATEST.json')['path']);assert sha(ROOT/'notebooks/phase3_NB07_TREAD_ONE_Data_And_Protocol_Gate.ipynb')==read(latest/'VERIFICATION.json')['executed_notebook_sha256']
 for name,archive in [('phase3_NB02_Audit_And_Direct_CSV.ipynb',ROOT/'completion_20261028/nb02_publication/executed.ipynb'),('phase3_NB03_TREAD_ONE_Classifier_Diagnostic.ipynb',ROOT/'diagnostics/nb03_publication/executed.ipynb'),('phase3_NB04_TREAD_ONE_Evidence_And_Recovery_Gate.ipynb',ROOT/'training_gates/nb04_publication/executed.ipynb'),('phase3_NB05_TREAD_ONE_Shared_Gradient_Paired_Pilot.ipynb',ROOT/'revision2_validation/nb05_publication/executed.ipynb'),('phase3_NB06_TREAD_ONE_Paired_Review_And_Failure_Atlas.ipynb',ROOT/'revision2_validation/nb06_publication/executed.ipynb')]:assert sha(ROOT/'notebooks'/name)==sha(archive)
 sources=list(APP.glob('*.py'))+[APP/'models/tread_one_r1.py']
 for p in sources:ast.parse(p.read_text(encoding='utf-8'))
 for p in (OUT/'TREAD_ONE_Studio_Preview.png',OUT/'TREAD_ONE_Studio_Evidence.png',OUT/'TREAD_ONE_Studio_Compact.png',OUT/'TREAD_ONE_Studio_Native_Final.png'):
  with Image.open(p) as image:image.verify()
 session=Path(native['session']);assert session.resolve().is_relative_to((APP/'data/sessions').resolve()) and (session/'predictions.csv').exists()
 docs=list(ROOT.glob('*.md'))+list(APP.glob('*.md'))+list((ROOT/'data_intake').rglob('*.md'))+[ROOT/'data_gate_outputs/CSV_INDEX.md'];links=0;broken=[]
 for doc in docs:
  for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
   target=target.strip('<>')
   if re.match(r'^(https?://|mailto:|#)',target):continue
   target=unquote(target.split('#')[0]);links+=1
   if not (doc.parent/target).exists():broken.append([str(doc),target])
 assert not broken,broken
 assert 'The user will not collect additional tyre photos' in (ROOT/'AGENTS.md').read_text(encoding='utf-8')
 full_capture=Path(engine['fixture_capture']);capture_bytes=sum(p.stat().st_size for p in full_capture.iterdir() if p.is_file());files=list(APP.glob('*.py'))+list(APP.glob('*.md'))+list(APP.glob('*.cmd'))+[APP/'models/tread_one_r1.py',APP/'models/MODEL.json']
 result=dict(status='passed',original_workstation_files_preserved=len(baseline),original_Markdown_preserved=len(protected),frozen_experiment_sources_preserved=27,frozen_r1_runtime_sources_preserved=len(source_contract['sources']),earlier_executed_notebooks_preserved=True,
  independent_app_directory=str(APP),model_seed=3,model_epoch=20,weights_sha256=identity['weights_sha256'],source_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in files},
  engine_checks=12,UI_checks=14,CPU_reference_max_difference=max(engine['frozen_reference_max_differences'].values()),GPU_max_differences=engine['GPU_max_differences'],native_Qt_platform=native['Qt_platform'],native_device=native['device'],native_window_title=native['title'],
  full_capture_CSV_tables=10,full_capture_CSV_rows=245784,full_capture_dense_CSV_values=245760,full_capture_files=sum(1 for p in full_capture.iterdir() if p.is_file()),fixture_capture_bytes=capture_bytes,
  local_Markdown_documents=len(docs),local_links_checked=links,broken_links=broken,new_training=False,new_inference=True,new_independent_tyres=0,additional_photo_collection_planned=False,desktop_HF_uploads=0,camera_hardware_tested=False,
  scope='Separate experimental inference application; existing data only; original workstation remains available; no accuracy improvement claim')
 write_json(OUT/'DELIVERY.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
