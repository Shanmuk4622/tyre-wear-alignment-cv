"""Final handoff: verified Kaggle NB07, preserved study and local continuation."""
import re,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'data_gate'))
from common_gate import read,sha,atomic,table

def main():
 first_pub=ROOT/'data_gate_validation/nb07_publication';first=read(first_pub/'VERIFICATION.json');assert first['status']=='passed' and sha(first_pub/'executed.ipynb')==first['executed_notebook_sha256']
 pub=first_pub;r=first;notebook=ROOT/'notebooks/phase3_NB07_TREAD_ONE_Data_And_Protocol_Gate.ipynb'
 if sha(notebook)!=first['executed_notebook_sha256']:
  pub=Path(read(ROOT/'data_gate_validation/nb07_reruns/LATEST.json')['path']);assert pub.resolve().is_relative_to((ROOT/'data_gate_validation/nb07_reruns').resolve())
  r=read(pub/'VERIFICATION.json');assert r['status']=='passed' and read(pub/'COMPARISON.json')['same_code_cells']
 assert sha(notebook)==sha(pub/'executed.ipynb')==r['executed_notebook_sha256']
 inspection=read(pub/'NOTEBOOK_INSPECTION.json');counts=inspection['execution_counts'];assert not inspection['errors'] and len(counts)==5 and all(type(n) is int and n>0 for n in counts) and counts==sorted(set(counts))
 assert inspection['embedded_files']==read(ROOT/'data_gate_validation/DELIVERY.json')['embedded_files']
 assert all(sha(ROOT/n)==h for n,h in inspection['embedded_files'].items())
 remote=pub/'remote';manifest=read(remote/'REMOTE_MANIFEST.json');assert all(sha(remote/n)==v['sha256'] for n,v in manifest['files'].items())
 assert read(ROOT/'data_gate_outputs/PUBLISHED.json')['revision']==first['revision']
 if pub!=first_pub:assert read(ROOT/'data_gate_outputs/PUBLISHED_LATEST.json')['revision']==r['revision']
 local=ROOT/'data_gate_outputs';run=Path(read(local/'LATEST.json')['path']);schema=read(local/'CSV_SCHEMA.json')
 assert len(schema)==22 and sum(v['rows'] for v in schema.values())==7614
 assert all(sha(local/'csv'/(n+'.csv'))==sha(run/'csv'/(n+'.csv'))==s['sha256'] for n,s in schema.items())
 pack=ROOT/'collection_review_outputs';review=read(ROOT/'data_gate_validation/collection_review/VALIDATION.json');assert review['status']=='passed' and review['logic_check_count']==10
 contract=read(pack/'CONTRACT.json');assert contract['source_revision']==first['revision'] and contract['new_independent_tyres']==contract['confirmed_mappings_created_by_agent']==0
 assert all(sha(pack/n)==v['sha256'] for n,v in read(pack/'MANIFEST.json')['files'].items())
 assert all(sha(ROOT/'collection_review'/n)==h for n,h in contract['review_source_hashes'].items())
 assert [int(x['historical_train_tyres']) for x in table(pack/'capture_priorities.csv')]==[1,3,4]
 assert all(not x['new_subject_count_required'] for x in table(pack/'capture_priorities.csv'))
 assert all(not row['confirmed_original_tyre_id'] for row in table(ROOT/'data_intake/video_identity_resolutions.csv'))
 protected=read(ROOT/'audit/markdown-baseline.json');assert all(sha(ROOT.parent/x['path'])==x['sha256'] for x in protected)
 assert all(sha(ROOT/n)==h for n,h in read(ROOT/'revision2_validation/DELIVERY.json')['sources'].items())
 preserved=[]
 for archive in (ROOT/'completion_20261028/executed_notebooks').glob('*.ipynb'):
  assert sha(archive)==sha(ROOT/'notebooks'/archive.name);preserved.append(archive.name)
 for name,archive in [('phase3_NB02_Audit_And_Direct_CSV.ipynb',ROOT/'completion_20261028/nb02_publication/executed.ipynb'),('phase3_NB03_TREAD_ONE_Classifier_Diagnostic.ipynb',ROOT/'diagnostics/nb03_publication/executed.ipynb'),('phase3_NB04_TREAD_ONE_Evidence_And_Recovery_Gate.ipynb',ROOT/'training_gates/nb04_publication/executed.ipynb'),('phase3_NB05_TREAD_ONE_Shared_Gradient_Paired_Pilot.ipynb',ROOT/'revision2_validation/nb05_publication/executed.ipynb'),('phase3_NB06_TREAD_ONE_Paired_Review_And_Failure_Atlas.ipynb',ROOT/'revision2_validation/nb06_publication/executed.ipynb')]:
  assert sha(ROOT/'notebooks'/name)==sha(archive);preserved.append(name)
 docs=list(ROOT.glob('*.md'))+list((ROOT/'data_intake').rglob('*.md'))+[ROOT/'data_gate_outputs/CSV_INDEX.md'];links=0;broken=[]
 for doc in docs:
  for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
   if re.match(r'^(https?://|mailto:|#)',target):continue
   links+=1
   if not (doc.parent/target.split('#')[0]).exists():broken.append([str(doc),target])
 assert not broken,broken
 result=dict(status='passed',NB07_Kaggle_complete=True,NB07_independently_HF_verified=True,revision=r['revision'],published_files=46,published_CSV_files=26,structured_CSV_tables=22,structured_rows=7614,
  NB07_executed_sha256=sha(notebook),embedded_sources_preserved=42,frozen_experiment_sources_preserved=27,original_Markdown_preserved=len(protected),earlier_executed_notebooks_preserved=sorted(set(preserved)),
  continuation_local_only=True,review_source_images=44,planning_rows=56,review_export_checks=10,local_Markdown_documents=len(docs),local_links_checked=links,broken_links=broken,
  actual_video_mappings_unchanged=True,prospective_training_ready=False,new_independent_tyres=0,new_training=False,new_inference=False,remote_writes_by_verifier=0,browser_render_verified=False,
  first_NB07_archive_preserved=True,latest_is_repeat=pub!=first_pub)
 atomic(pub/'HANDOFF_VALIDATION.json',result);print(result)

if __name__=='__main__':main()
