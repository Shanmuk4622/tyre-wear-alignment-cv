"""Real-model atomic boundary failures, prefix truncation/corruption and selected branch."""
import copy,json,shutil,sys,time
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
import torch
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'tread_two/trace_batch';sys.path.insert(0,str(SRC))
from common import atomic,read,digest
from engine import Engine
from store import semantic_hash
torch.set_num_threads(4)
valid=ROOT/'tread_two/trace_validation';base=Path(read(valid/'LOCAL_LIFECYCLE.json')['fixture']);cache=base/'cache'
out=valid/('fixture_transactions_'+str(time.time_ns()));out.mkdir()
def finish(e):
    while e.cursor['phase']!='completed':e.step()
    p=read(e.store.root/'POINTER.json');return torch.load(e.store.root/p['state'],map_location='cpu',weights_only=False)
continuous=Engine(out/'continuous',cache,'cpu',True);expected=finish(continuous)
checks=[]
for boundary in ('after_evidence','after_state','after_pointer'):
    root=out/boundary;e=Engine(root,cache,'cpu',True)
    while e.cursor['phase']!='train':e.step()
    original=e.store.commit
    def fail(*args,**kwargs):return original(*args,**kwargs,crash=boundary)
    e.store.commit=fail
    try:e.step()
    except InterruptedError:pass
    else:raise AssertionError('Fault did not fire')
    resumed=Engine(root,cache,'cpu',True);actual=finish(resumed)
    for key in ('model','optimizer','rng','cursor'):assert semantic_hash(actual[key])==semantic_hash(expected[key]),(boundary,key)
    checks.append(dict(boundary=boundary,status='passed',exact_model_optimizer_RNG_cursor=True))
# Appended uncommitted garbage is truncated to SHA-verified byte prefixes.
root=out/'appended_tails';shutil.copytree(base/'direct/seam',root/'run')
p=read(root/'run/POINTER.json');state=torch.load(root/'run'/p['state'],map_location='cpu',weights_only=False)
paths=[root/'run'/v['path'] for v in [*state['csv'].values(),*state['arrays'].values()]]
for path in paths:
    with path.open('ab') as stream:stream.write(b'UNCOMMITTED_FIXTURE_TAIL')
resumed=Engine(root,cache,'cpu',True);assert resumed.store.restored_tail_truncated
actual=finish(resumed)
for key in ('model','optimizer','rng','cursor'):assert semantic_hash(actual[key])==semantic_hash(expected[key]),key
checks.append(dict(boundary='appended_uncommitted_CSV_native_tails',status='passed',truncated=True))
# A changed committed prefix must fail, not be silently accepted as a repair.
root=out/'corrupt_prefix';shutil.copytree(base/'direct/seam',root/'run')
p=read(root/'run/POINTER.json');state=torch.load(root/'run'/p['state'],map_location='cpu',weights_only=False);entry=next(iter(state['csv'].values()));path=root/'run'/entry['path']
with path.open('r+b') as stream:head=stream.read(1);stream.seek(0);stream.write(bytes([head[0]^1]))
try:Engine(root,cache,'cpu',True)
except AssertionError:checks.append(dict(boundary='corrupt_committed_prefix',status='passed',rejected_before_updates=True))
else:raise AssertionError('Committed corruption accepted')
# Inject only into a never-published disposable fixture to exercise selected
# exports and metadata; this is not a scientific eligibility claim.
e=Engine(out/'selected_injection',cache,'cpu',True)
while e.cursor['phase']!='endpoint':e.step()
e.cursor['best']=dict(epoch=1,score=0.,path='epochs/epoch_001.pt');finish(e)
import pandas as pd
parts=[pd.read_csv(p) for p in e.store.root.glob('csv/sample_observations/*.csv')];rows=pd.concat(parts,ignore_index=True)
chosen=rows[rows.stage=='endpoint_selected'];assert len(chosen)==8 and chosen.parameter_update.eq(2).all()
assert (e.store.root/'exports/selected.pt').exists()
checks.append(dict(boundary='injected_selected_endpoint_only',status='passed',parameter_update_provenance=True,actual_selection_claim=False))
# Exercise the actual worker exception/last-complete-snapshot handler.
import worker
error_root=out/'worker_error';shutil.copytree(base/'direct/seam',error_root/'run');e=Engine(error_root,cache,'cpu',True)
before_model=semantic_hash(e.model.state_dict());before_optimizer=semantic_hash(e.optimizer.state_dict());before_update=e.cursor['update']
original_engine,original_argv=worker.Engine,sys.argv
def error_step():raise RuntimeError('injected_nonfinite_fixture_event_no_real_training')
e.step=error_step;worker.Engine=lambda *args,**kwargs:e
sys.argv=['worker.py','gate_direct','--output',str(error_root),'--cache',str(cache),'--offline','--no-ack']
try:
    try:worker.main()
    except RuntimeError as exc:assert 'injected_nonfinite_fixture' in str(exc)
    else:raise AssertionError('Worker numeric failure swallowed')
finally:worker.Engine=original_engine;sys.argv=original_argv
assert semantic_hash(e.model.state_dict())==before_model and semantic_hash(e.optimizer.state_dict())==before_optimizer and e.cursor['update']==before_update
assert read(error_root/'SNAPSHOT_READY.json')['reason']=='error_last_complete'
assert any(n.startswith('errors/') and n.endswith('_TRACEBACK.txt') for n in read(Path(read(error_root/'SNAPSHOT_READY.json')['path'])/'MANIFEST.json')['files'])
error_rows=pd.concat([pd.read_csv(p) for p in e.store.root.glob('csv/errors/*.csv')],ignore_index=True);assert len(error_rows)==1
checks.append(dict(boundary='worker_error_last_complete',status='passed',error_CSV_retained=True,model_optimizer_update_unchanged=True))
atomic(valid/'TRANSACTIONS.json',dict(status='passed',source_inventory=read(SRC/'FILES.json'),checks=checks,fixture=str(out),scope='disposable_CPU_actual_model_only',actual_dual_T4=False,remote_writes=0,credentials_read=False,new_measured_updates=0))
print(json.dumps(dict(status='passed',checks=checks),indent=2),flush=True)
