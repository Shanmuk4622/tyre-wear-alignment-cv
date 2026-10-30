"""Prepare an additive direct-file review; publish only with explicit HF_TOKEN."""
from PIL import Image
import argparse,json,os,shutil,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda' or Path('/kaggle/input').exists()
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'revision2'))
from publisher_r2 import Publisher
from transactions import atomic,read,sha,file_meta,digest
from common import environment
import torch
PUB=ROOT/'revision2_validation/nb05_publication';OUT=PUB/'analysis'
SOURCES=('nb05_public_reads.py','verify_nb05_publication.py','verify_nb05_preflight.py','audit_nb05_results.py','render_nb05_review.py','publish_nb05_review.py')

def prepare():
 for n in ('DOWNLOAD.json','SCIENTIFIC_VERIFICATION.json','PREFLIGHT_VERIFICATION.json'):assert read(PUB/n)['status']=='passed'
 analysis_sources={'analysis/'+n:sha(ROOT/'analysis'/n) for n in SOURCES}
 receipt=read(PUB/'NOTEBOOK_COMPLETION.json')
 contract=dict(kind='TREAD-ONE r2 posthoc review; zero new training',version=1,source_publications=receipt['publications'],
  source_executed_notebook_sha256=sha(PUB/'executed.ipynb'),analysis_sources=analysis_sources,environment=environment(torch.device('cpu')),
  source_data_checksum='06c6c9d003aff808babf7ebb5786f441e95064eb000f5e5ed5b155d0ee9e7c6a',
  selection='Original validation selections only; no test-based checkpoint/seed reselection',independent_new_tyres=0)
 identity=digest(contract);folder=PUB/'review_bundle'/identity[:12];folder.mkdir(parents=True,exist_ok=True)
 # Keep fresh Windows workspaces below legacy path limits. The full identity
 # remains in the contract and remote namespace; detect any short-name clash.
 if (folder/'CONTRACT.json').exists():assert digest(read(folder/'CONTRACT.json'))==identity
 files={'analysis/'+p.name:p for p in OUT.glob('*.csv')}
 files.update({'figures/'+p.name:p for p in (OUT/'figures').glob('*.png')})
 files['TREAD_ONE_NB05_Comparison_Explorer.html']=OUT/'TREAD_ONE_NB05_Comparison_Explorer.html'
 for n in ('NOTEBOOK_COMPLETION.json','NOTEBOOK_INSPECTION.json','DOWNLOAD.json','SCIENTIFIC_VERIFICATION.json','PREFLIGHT_VERIFICATION.json','INVENTORY.json'):files['verification/'+n]=PUB/n
 for condition in ('baseline','shared_projection'):
  for n in ('REMOTE_MANIFEST.json','CONTRACT.json','CURSOR.json'):files['source_metadata/'+condition+'/'+n]=PUB/'remote'/condition/n
 for n in SOURCES:files['source/analysis/'+n]=ROOT/'analysis'/n
 for n,p in files.items():
  target=folder/n;target.parent.mkdir(parents=True,exist_ok=True)
  if target.exists():assert sha(target)==sha(p),'Immutable review bundle differs; regenerate under its actual source/environment identity'
  else:shutil.copy2(p,target)
 atomic(folder/'CONTRACT.json',contract)
 (folder/'README.md').write_text('''# TREAD-ONE NB05 completed paired development review

One seed, two matched 60-epoch conditions. Both complete 2,940 updates.
Baseline selected epoch 23; shared projection selected epoch 17, by the frozen
joint validation utility. Selected development-test balanced accuracy is 50%
versus 17.073%; mean Dice 97.852% versus 97.782%; human point error 1.067%
versus 1.885% image width. The candidate is not supported for promotion.

These files are post-hoc derivations of pinned saved CSV/native evidence.
No model is trained, no checkpoint reselected using test results, and no
additional independent tyre is added. The test cohort has 103 photos from two
already inspected tyres, low/high only. Human points exist on 25 of them.

Open TREAD_ONE_NB05_Comparison_Explorer.html after downloading it; all 103 source
images and all four endpoint predictions/maps are embedded for offline review.
Mask/point/evidence overlays and 60 validation epochs are inspectable. Image
heat colors are scaled within each class/image; numerical scales are retained.
Additive class evidence is not causal or physical wear/safety validation.

CSV units: probabilities, accuracy, Dice and error/width are fractions; durations
seconds, memory bytes. Blank means unavailable/not applicable. Derived rows and
repeated observations are not new independent samples. Confidence thresholds
are descriptive development diagnostics, not calibrated deployment rules.
The frozen per-point scalar errors can contain FP32 rounding; annotation MAE
and the frozen Torch selection arithmetic are kept separately and audited.

CONTRACT.json pins analysis sources/environment and original publications.
source_metadata/ retains original manifest hashes/cursors; original training
files remain in their own namespaces. verification/ states exactly which
native/CSV/maps were checked. No ZIP/NPZ or database is uploaded by this review.
''',encoding='utf-8')
 manifest=dict(contract=identity,files={p.relative_to(folder).as_posix():file_meta(p) for p in folder.rglob('*') if p.is_file() and p.name!='MANIFEST.json'})
 assert len(manifest['files'])<=79 and not any(n.endswith(('.zip','.npz','.sqlite','.pt')) for n in manifest['files'])
 atomic(folder/'MANIFEST.json',manifest)
 result=dict(status='prepared_not_published',contract=identity,path=str(folder),destination='phase3/tread-one/r2/review/paired-seed1/'+identity,
  files=len(manifest['files'])+1,direct_csv_tables=len(list(OUT.glob('*.csv'))),new_training=False)
 atomic(PUB/'REVIEW_DELIVERY.json',result);return result

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args();result=prepare()
 if args.prepare_only:print(json.dumps(result,indent=2));return
 token=os.environ.get('HF_TOKEN');assert token,'Enable Kaggle HF_TOKEN. No token-file or .env fallback is used.'
 hub=Publisher(result['destination'],token,PUB/'REVIEW_REMOTE_HISTORY.jsonl');hub.claim()
 try:
  publication=hub.publish(result['path'],release=True);atomic(PUB/'REVIEW_PUBLICATION.json',publication);print(json.dumps(publication,indent=2),flush=True)
 except BaseException:
  # Complete derived evidence is already local. Attempt an immediate consistent
  # flush on a catchable Stop/error; the frozen publisher respects server delay.
  try:
   publication=hub.publish(result['path'],release=True);atomic(PUB/'REVIEW_PUBLICATION.json',publication)
  except BaseException as exc:
   atomic(PUB/'REVIEW_PUBLICATION_PENDING.json',dict(status='pending_local_retained',error_type=type(exc).__name__,destination=result['destination']))
  raise

if __name__=='__main__':main()
