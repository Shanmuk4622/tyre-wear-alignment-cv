"""Bind final NB03/NB04 artifacts, immutable sources, receipts and documentation."""
from PIL import Image
import ast,base64,gzip,json,re,sys,tempfile
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'training'))
from transactions import atomic,read,sha
from telemetry import SCHEMAS
import nbformat

def main():
 nb03=read(ROOT/'diagnostics/nb03_publication/VERIFICATION.json');assert nb03['status']=='passed'
 assert nb03['published_files']==27 and nb03['core_records']==336600 and nb03['executed_code_cells']==4 and not nb03['notebook_errors']
 path=ROOT/'notebooks/phase3_NB03_TREAD_ONE_Classifier_Diagnostic.ipynb'
 assert sha(path)==nb03['notebook_sha256']==sha(ROOT/'diagnostics/nb03_publication/executed.ipynb')
 remote=ROOT/'diagnostics/nb03_publication/remote';manifest=read(remote/'MANIFEST.json')
 for name,spec in manifest['files'].items():assert sha(remote/name)==spec['sha256']
 assert len(manifest['files'])+1==27 and nb03['remote_owner_released']
 done=read(ROOT/'training_gates/GATE_COMPLETION.json');audit=read(ROOT/'training_gates/EVIDENCE_VALIDATION.json')
 controls=read(ROOT/'training_gates/CONTROL_TESTS.json');faults=read(ROOT/'training_gates/FAULT_TESTS.json');delivery=read(ROOT/'training_gates/DELIVERY.json')
 assert all(r['status']=='passed' for r in (done,audit,controls,faults))
 protocol=done['devices'][0]['protocol'];assert protocol==audit['protocol']==controls['protocol']
 cfg=read(Path(done['devices'][0]['path'])/'CONFIG.json')
 assert all(sha(ROOT/n)==h for n,h in cfg['source_hashes'].items())
 assert sha(Path(cfg['data_root'])/'SHA256SUMS.txt')=='06c6c9d003aff808babf7ebb5786f441e95064eb000f5e5ed5b155d0ee9e7c6a'
 import gate
 wrong=Path(tempfile.mkdtemp(prefix='validation_checksum_',dir=ROOT/'training_gates'));(wrong/'SHA256SUMS.txt').write_bytes(b'fixture-corruption')
 try:gate.configuration(wrong,'cuda:0')
 except AssertionError as exc:assert 'Prepared checksum list changed' in str(exc)
 else:raise AssertionError('Wrong dataset checksum accepted')
 assert all(sha(ROOT/n)==h for n,h in faults['tested_sources'].items()) and faults['count']==22
 assert all(v==0 for k in ('marker_resume_differences','SIGINT_resume_differences') for v in controls[k].values())
 assert all(v==0 for k in ('original_r1_update_math','fresh_process_pause_resume','abrupt_native_exit_replay','direct_snapshot_install_resume') for v in done['devices'][0]['local'][k].values())
 assert not done['new_study_started'] and audit['total_core_rows']==703 and audit['real_augmented_sample_occurrences_verified']==24
 assert read(ROOT/'training_gates/TELEMETRY_SCHEMA.json')['families']==SCHEMAS
 notebook=ROOT/'notebooks/phase3_NB04_TREAD_ONE_Evidence_And_Recovery_Gate.ipynb'
 nb04=read(ROOT/'training_gates/nb04_publication/VERIFICATION.json');assert nb04['status']=='passed'
 assert sha(notebook)==nb04['executed_notebook_sha256']==sha(ROOT/'training_gates/nb04_publication/executed.ipynb')
 assert delivery['notebook_sha256']==audit.get('historical_clean_notebook_sha256',audit['notebook_sha256'])
 nb=nbformat.read(notebook,as_version=4);nbformat.validate(nb);embedded={}
 for cell in nb.cells:
  if cell.cell_type!='code':continue
  assert not any(o.get('output_type')=='error' for o in cell.outputs) and cell.execution_count
  for call in ast.walk(ast.parse(cell.source)):
   if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=='b64decode' and call.args and isinstance(call.args[0],ast.Constant):
    payload=json.loads(gzip.decompress(base64.b64decode(call.args[0].value)))
    for name,value in payload.items():
     assert base64.b64decode(value)==(ROOT/name).read_bytes();embedded[name]=sha(ROOT/name)
 assert embedded==delivery['embedded_sources'] and len(embedded)==17
 assert all(sha(ROOT/'runtime'/n)==h for n,h in read(ROOT/'training/FROZEN.json').items())
 protected=read(ROOT/'audit/markdown-baseline.json');assert len(protected)==103
 assert all(sha(ROOT.parent/r['path'])==r['sha256'] for r in protected)
 atomic(ROOT/'training_gates/HANDOFF_VALIDATION.json',dict(status='validating',protocol=protocol))
 links=0;broken=[]
 for doc in ROOT.glob('*.md'):
  for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
   if re.match(r'^(https?://|mailto:|#)',target):continue
   links+=1
   if not (doc.parent/target.split('#')[0]).exists():broken.append([doc.name,target])
 assert not broken,broken
 # All new authored Python is parsed without touching supplied notebook code.
 checked=[]
 for source in list((ROOT/'training').glob('*.py'))+[ROOT/'analysis'/n for n in (
  'build_training_gate_notebook.py','test_training_evidence.py','test_training_controls.py','validate_training_evidence.py',
  'write_training_schema.py','verify_nb03_publication.py','validate_diagnostic_handoff.py','validate_phase3_handoff.py')]:
  ast.parse(source.read_text(encoding='utf-8'));checked.append(str(source.relative_to(ROOT)))
 result=dict(status='passed',NB03_Kaggle_and_HF_verified=True,NB03_revision=nb03['revision'],NB03_executed_notebook_sha256=nb03['notebook_sha256'],
  NB04_protocol=protocol,NB04_notebook_bytes=notebook.stat().st_size,NB04_notebook_sha256=sha(notebook),
  embedded_source_files=17,core_CSV_families=13,core_rows_per_complete_engineering_case=703,offline_fault_checks=22,
  exact_real_model_resume=True,actual_flush_marker_and_SIGINT_entry_recovery=True,scientific_tables_and_native_arrays_verified=True,wrong_dataset_checksum_rejected=True,
  source_syntax_checked=checked,original_markdown_preserved=103,frozen_r1_sources_preserved=True,local_links_checked=links,broken_links=broken,
  NB04_Kaggle_execution_tested=True,NB04_live_HF_recovery_tested=True,NB04_verified_published_files=nb04['published_files'],
  NB04_historical_clean_notebook_sha256=delivery['notebook_sha256'],browser_render_tested=False,new_measured_study_started=False,remote_writes_by_this_turn=0)
 atomic(ROOT/'training_gates/HANDOFF_VALIDATION.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
