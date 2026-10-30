"""Validate final user-facing CSV links, provenance and preserved study files."""
import csv,json,re,sys
from pathlib import Path
from PIL import Image
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'data_gate'))
from common_gate import read,sha,atomic,digest

def main():
 base=ROOT/'data_gate_outputs';latest=read(base/'LATEST.json');run=Path(latest['path']);schema=read(base/'CSV_SCHEMA.json')
 validated=read(ROOT/'data_gate_validation/VALIDATION.json');assert validated['status']=='passed' and validated['run_id']==latest['run_id']
 assert len(schema)==22 and sum(s['rows'] for s in schema.values())==7614
 for n,s in schema.items():
  p=base/'csv'/(n+'.csv');assert sha(p)==sha(run/'csv'/p.name)==s['sha256']
  with p.open(newline='',encoding='utf-8') as f:
   r=csv.DictReader(f);assert r.fieldnames==s['columns'] and len(list(r))==s['rows']
 for n in ('SUMMARY.json','CONTRACT.json','CSV_SCHEMA.json','POLICY.json'):assert sha(base/n)==sha(run/n)
 publication=read(base/'PUBLICATION_DELIVERY.json');snapshot=Path(publication['path']);manifest=read(snapshot/'MANIFEST.json')
 assert publication['status']=='prepared_not_published' and publication['contract']==digest(read(snapshot/'CONTRACT.json'))==manifest['contract']
 assert len(manifest['files'])+1==46
 assert all(sha(snapshot/n)==info['sha256'] for n,info in manifest['files'].items())
 assert sum(n.endswith('.csv') for n in manifest['files'])==26
 assert not any(n.endswith(('.zip','.npz','.sqlite','.pt')) for n in manifest['files'])
 assert publication['contract'] in (ROOT/'phase3NB07DataGate.md').read_text(encoding='utf-8')
 for p in (base/'figures').glob('*.png'):
  assert sha(p)==sha(run/'figures'/p.name)
  with Image.open(p) as image:image.verify()
 docs=list(ROOT.glob('*.md'))+list((ROOT/'data_intake').rglob('*.md'))+[base/'CSV_INDEX.md']
 links=0;broken=[]
 for doc in docs:
  for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
   if re.match(r'^(https?://|mailto:|#)',target):continue
   links+=1
   if not (doc.parent/target.split('#')[0]).exists():broken.append([str(doc),target])
 assert not broken,broken
 protected=read(ROOT/'audit/markdown-baseline.json');assert all(sha(ROOT.parent/r['path'])==r['sha256'] for r in protected)
 assert all(sha(ROOT/n)==h for n,h in read(ROOT/'revision2_validation/DELIVERY.json')['sources'].items())
 preserved=[]
 for archive in (ROOT/'completion_20261028/executed_notebooks').glob('*.ipynb'):
  assert sha(archive)==sha(ROOT/'notebooks'/archive.name);preserved.append(archive.name)
 for name,archive in [('phase3_NB03_TREAD_ONE_Classifier_Diagnostic.ipynb',ROOT/'diagnostics/nb03_publication/executed.ipynb'),('phase3_NB04_TREAD_ONE_Evidence_And_Recovery_Gate.ipynb',ROOT/'training_gates/nb04_publication/executed.ipynb'),('phase3_NB05_TREAD_ONE_Shared_Gradient_Paired_Pilot.ipynb',ROOT/'revision2_validation/nb05_publication/executed.ipynb'),('phase3_NB06_TREAD_ONE_Paired_Review_And_Failure_Atlas.ipynb',ROOT/'revision2_validation/nb06_publication/executed.ipynb')]:
  assert sha(ROOT/'notebooks'/name)==sha(archive);preserved.append(name)
 for name in ('physical_tyres','capture_sessions','images'):
  with (ROOT/'data_intake'/(name+'.csv')).open(newline='',encoding='utf-8') as f:assert list(csv.DictReader(f))==[]
 result=dict(status='passed',local_Markdown_documents=len(docs),local_links_checked=links,broken_links=broken,stable_CSV_tables=22,structured_rows=7614,
  direct_publication_files=46,direct_publication_CSV_files=26,publication_contract=publication['contract'],live_HF_published=False,
  original_Markdown_preserved=len(protected),frozen_experiment_sources_preserved=27,executed_notebooks_preserved=sorted(set(preserved)),
  prospective_training_ready=False,new_independent_tyres=0,new_training=False,new_inference=False)
 atomic(ROOT/'data_gate_validation/HANDOFF_VALIDATION.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
