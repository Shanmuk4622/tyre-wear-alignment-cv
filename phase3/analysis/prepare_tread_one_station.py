"""Record isolation baseline and prepare an offline, separately owned desktop model."""
import hashlib,importlib.util,json,shutil,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];APP=ROOT/'workstation';OUT=ROOT/'workstation_validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 APP.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
 baseline=OUT/'ORIGINAL_BASELINE.json'
 if not baseline.exists():
  paths=list((ROOT.parent/'prototype').rglob('*.py'))+list((ROOT.parent/'prototype').glob('*.md'))+list((ROOT.parent/'prototype').glob('*.cmd'))
  paths=[p for p in paths if not any(v in p.parts for v in ('.vendor','.cache','__pycache__'))]
  for p in (ROOT.parent/'phase2/workstation/registry.json',ROOT.parent/'phase2/workstation/ACTIVE'):
   if p.exists():paths.append(p)
  baseline.write_text(json.dumps({p.relative_to(ROOT.parent).as_posix():sha(p) for p in sorted(set(paths))},indent=2)+'\n',encoding='utf-8')
 report=json.loads((ROOT/'completion_20261028/EXPORT_VERIFICATION.json').read_text());assert report['status']=='passed' and report['seed']==3 and report['epoch']==20
 contract=json.loads((ROOT/'completion_20261028/hf/CONTRACT.json').read_text());model=ROOT/'runtime/model.py';weights=ROOT/'completion_20261028/hf/report/weights.pt'
 assert sha(model)==contract['sources']['model.py'] and sha(weights)==report['weights_sha256']
 package=APP/'models';package.mkdir(exist_ok=True)
 for source,name in ((model,'tread_one_r1.py'),(weights,'weights.pt')):
  target=package/name
  if target.exists():assert sha(target)==sha(source),'Preserve existing station model identity'
  else:shutil.copy2(source,target)
 identity=dict(model='TREAD-ONE r1',model_version='tread-one-r1',seed=3,epoch=20,selection='Declared original joint-validation winner; no test-based substitution',
  weights_sha256=sha(weights),model_source_sha256=sha(model),source_repository='Shanmuk4622/tyre-wear-study',source_revision='a0296f18e02b12d28f7e51377783d68538032209',
  protocol='371c40c0c6aaedaf5c5970f7721a284b042494db7bb330065f6cbde8fc3a0b8c',input_height=512,input_width=384,mean=[.485,.456,.406],std=[.229,.224,.225],
  guides=[384/1535,768/1535,1151/1535],mask_threshold=.5,parameters=1372159,labels=['Low mileage proxy','Mid mileage proxy','High mileage proxy'],
  selected_test_balanced_accuracy=.5,selected_test_mean_dice=.98419,selected_test_human_point_mae_width_fraction=.01196,
  scope='Existing 12-tyre development study; low/high test only, no held-out mid. All selected r1 seeds miss the low test tyre. Softmax is not measured wear, depth or safety.',
  station_role='Separate experimental desktop application; does not replace original installed models')
 (package/'MODEL.json').write_text(json.dumps(identity,indent=2)+'\n',encoding='utf-8')
 # Copy existing Qt bindings into the new app instead of writing/importing the
 # original prototype tree. No installation or changes to cv_conda packages.
 vendor=APP/'.vendor';vendor.mkdir(exist_ok=True)
 for name in ('PySide6','shiboken6'):
  source=ROOT.parent/'prototype/.vendor'/name;assert source.exists()
  target=vendor/name
  if not target.exists():shutil.copytree(source,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 import torch,numpy,PIL
 available={n:bool(importlib.util.find_spec(n)) for n in ('cv2','PySide6')}
 result=dict(status='prepared',interpreter=sys.executable,torch=torch.__version__,numpy=numpy.__version__,pillow=PIL.__version__,cuda_available=torch.cuda.is_available(),packages_before_private_Qt=available,
  original_files_recorded=len(json.loads(baseline.read_text())),private_Qt_files=sum(1 for p in vendor.rglob('*') if p.is_file()),private_Qt_bytes=sum(p.stat().st_size for p in vendor.rglob('*') if p.is_file()),model=identity)
 (OUT/'PREPARATION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='model'},indent=2))

if __name__=='__main__':main()
