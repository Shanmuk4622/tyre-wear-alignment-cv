"""Additive direct CSV publication of a completed data audit, even if not ready to train."""
from PIL import Image
import argparse,json,os,shutil,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda' or Path('/kaggle/input').exists()
from common_gate import ROOT,read,sha,digest,atomic,file_meta,source_hashes
from publisher_r2 import Publisher

def prepare(base):
 folder=Path(read(base/'LATEST.json')['path']);summary=read(folder/'SUMMARY.json');contract=read(folder/'CONTRACT.json')
 assert contract['source_hashes']==source_hashes(),'Source changed; rerun collection before publication'
 assert summary['status']=='audit_completed' and summary['source_integrity_passed'];assert (folder/'TREAD_ONE_Data_Gate.html').exists()
 publication_contract=dict(data_contract=contract,sources={p.name:sha(p) for p in sorted((ROOT/'data_gate').glob('*')) if p.suffix in ('.py','.json')})
 identity=digest(publication_contract)
 target=base/'publication'/identity[:12];target.mkdir(exist_ok=True,parents=True)
 files={('DATA_CONTRACT.json' if p.name=='CONTRACT.json' else p.relative_to(folder).as_posix()):p for p in folder.rglob('*') if p.is_file() and p.name not in ('GALLERY.json','MANIFEST.json')}
 for p in (ROOT/'data_gate').glob('*'):
  if p.suffix in ('.py','.json'):files['source/data_gate/'+p.name]=p
 for name,path in files.items():
  out=target/name;out.parent.mkdir(parents=True,exist_ok=True)
  if out.exists():assert sha(out)==sha(path),'Immutable publication differs; preserve the existing source identity'
  else:shutil.copy2(path,out)
 atomic(target/'CONTRACT.json',publication_contract)
 (target/'README.md').write_text('''# TREAD-ONE data and protocol gate

This is an executed CPU data audit, not a new model fit. SUMMARY.json separates
source integrity, historical split consistency and prospective training
readiness. A completed audit can correctly report readiness=false when new
subjects, metadata, annotations or the prospective protocol are missing.

The 570 existing images belong to 12 recorded original physical tyres. Three
video clips show three distinct existing mid tyres; unresolved original-parent
mapping does not add subjects. Similarity candidates are not identity labels.
CSV metrics include source registries, fixed-grid image quality, native masks,
human/weak point provenance, leakage components, coverage, saved training
exposure and validated intake. No source labels or historical splits are edited.

CSV_SCHEMA.json and csv/csv_schema.csv describe table fields and counts. Blanks
mean unknown/not applicable; zero is observed zero. Quality descriptors are
audit outputs, not additional model training features. No physical wear/depth
or alignment truth is inferred. All saved training exposure comes from pinned
existing CSV; no neural inference or optimizer update is performed.

Open TREAD_ONE_Data_Gate.html locally for source previews, quality/coverage and
video-parent review. Complete the prospective intake and actual subject/split
policy before declaring training readiness. This report never launches training.
All tables are individual CSV with native JSON/PNG/HTML; no ZIP/NPZ/database.
''',encoding='utf-8')
 manifest=dict(contract=identity,files={p.relative_to(target).as_posix():file_meta(p) for p in target.rglob('*') if p.is_file() and p.name!='MANIFEST.json'})
 assert not any(n.endswith(('.zip','.npz','.sqlite','.pt')) for n in manifest['files'])
 atomic(target/'MANIFEST.json',manifest);result=dict(status='prepared_not_published',path=str(target),contract=identity,files=len(manifest['files'])+1,
  destination='phase3/tread-one/data-gate/v1/'+identity,prospective_training_ready=summary['prospective_training_ready'],CSV_tables=summary['csv_tables'])
 atomic(base/'PUBLICATION_DELIVERY.json',result);return result

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',default='');parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args();base=Path(args.output or ROOT/'data_gate_outputs');result=prepare(base)
 if args.prepare_only:print(json.dumps(result,indent=2));return
 token=os.environ.get('HF_TOKEN');assert token,'Enable the explicit Kaggle HF_TOKEN secret; no alternate credential files are used'
 hub=Publisher(result['destination'],token,base/'REMOTE_HISTORY.jsonl');hub.claim()
 try:
  receipt=hub.publish(result['path'],release=True);atomic(base/'PUBLICATION.json',receipt);print(json.dumps(receipt,indent=2),flush=True)
 except BaseException:
  try:atomic(base/'PUBLICATION.json',hub.publish(result['path'],release=True))
  except BaseException as error:atomic(base/'PENDING.json',dict(status='local_retained',error_type=type(error).__name__,destination=result['destination']))
  raise

if __name__=='__main__':main()
