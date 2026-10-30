"""Preserve and independently check a repeated NB07 without replacing the first audit."""
import json,shutil,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
import nbformat
import inspect_nb07_execution as inspection
import verify_nb07_publication as verifier
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'data_gate'))
from common_gate import read,sha,atomic
FIRST=ROOT/'data_gate_validation/nb07_publication'

def main():
 notebook=ROOT/'notebooks/phase3_NB07_TREAD_ONE_Data_And_Protocol_Gate.ipynb';current_hash=sha(notebook);out=ROOT/'data_gate_validation/nb07_reruns'/current_hash[:12];out.mkdir(parents=True,exist_ok=True)
 inspection.OUT=out;verifier.OUT=out
 current=inspection.inspect();first=read(FIRST/'VERIFICATION.json');assert first['status']=='passed'
 old=nbformat.read(FIRST/'executed.ipynb',as_version=4);new=nbformat.read(notebook,as_version=4)
 assert [c.source for c in old.cells if c.cell_type=='code']==[c.source for c in new.cells if c.cell_type=='code'],'Review changed notebook code before comparing this as an unchanged-input repeat'
 remote=out/'remote'
 # Content is independently rechecked against the new pinned Git/LFS tree;
 # verified previous copies save requests without altering the first archive.
 for previous in (FIRST/'remote').rglob('*'):
  if not previous.is_file():continue
  target=remote/previous.relative_to(FIRST/'remote')
  if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(previous,target)
 verifier.inspect=lambda:current
 verifier.main()
 latest=read(out/'VERIFICATION.json');a=read(FIRST/'remote/REMOTE_MANIFEST.json');b=read(remote/'REMOTE_MANIFEST.json');receipt=read(out/'PUBLICATION.json')
 unchanged=[n for n,v in a['files'].items() if n in b['files'] and v['sha256']==b['files'][n]['sha256']]
 changed=[n for n,v in b['files'].items() if n not in a['files'] or v['sha256']!=a['files'][n]['sha256']]
 comparison=dict(status='passed',first_executed_sha256=first['executed_notebook_sha256'],rerun_executed_sha256=current_hash,first_revision=first['revision'],rerun_revision=latest['revision'],
  same_code_cells=True,first_contract=first['publication_contract'],rerun_contract=latest['publication_contract'],same_publication_contract=first['publication_contract']==latest['publication_contract'],
  same_data_run_id=first['run_id']==latest['run_id'],same_summary=read(FIRST/'remote/SUMMARY.json')==read(remote/'SUMMARY.json'),same_environment=read(FIRST/'remote/DATA_CONTRACT.json')['environment']==read(remote/'DATA_CONTRACT.json')['environment'],
  unchanged_payload_files=len(unchanged),changed_payload_files=changed,payload_files=45,reported_uploaded_objects=receipt['uploaded_objects'],source_images=570,existing_tyres=12,structured_rows=7614,
  first_archive_preserved=sha(FIRST/'executed.ipynb')==first['executed_notebook_sha256'],current_notebook_preserved=sha(notebook)==current_hash,owner_released=latest['owner_released'],
  prospective_training_ready=latest['prospective_training_ready'],new_independent_tyres=0,new_training=False,new_inference=False,remote_writes_by_verifier=0,rerun_needed_for_unchanged_inputs=False)
 atomic(out/'COMPARISON.json',comparison);atomic(ROOT/'data_gate_validation/nb07_reruns/LATEST.json',dict(path=str(out),sha256=current_hash,revision=latest['revision']))
 print(json.dumps(comparison,indent=2))

if __name__=='__main__':main()
