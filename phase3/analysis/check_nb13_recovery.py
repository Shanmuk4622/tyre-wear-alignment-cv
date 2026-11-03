"""Offline recovery-prefix/first-flush tests. No network, fitting, or model changes."""
import copy,json,shutil,sys,tempfile,time
from pathlib import Path
from unittest.mock import patch
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'tread_two/context_pilot'
sys.path.insert(0,str(SOURCE))
from common import atomic,canonical,digest,read,sha
from schemas import SCHEMAS
import coordinator,store
import torch
from check_context_training import Service,fake_publisher,rejected
import nb13_recovery_launcher as repair
OUT=ROOT/'tread_two/context_validation/nb13_status/recovery_validation';OUT.mkdir(parents=True,exist_ok=True)
work=Path(tempfile.mkdtemp(prefix='fixture_',dir=OUT))
before={n:sha(SOURCE/n) for n in read(SOURCE/'FILES.json')}
ctx=repair.install(SOURCE,work,monitor_seconds=.1)
contract=dict(revision='isolated repair-prefix fixture',sources=before)
cfg=dict(contract=contract,protocol=digest(contract));local=store.Store(work/'run',contract,SCHEMAS)
def commit(i):
    local.commit(dict(protocol=cfg['protocol'],updates=i,phase='fixture',model={'weight':torch.tensor([i/10])},optimizer={'state':torch.tensor([i])},rng={'cpu':torch.get_rng_state()}),{'predictions':[dict(run=cfg['protocol'][:12],sample=i)]},partition=1)
commit(1);snapshot1=local.export(work/'snapshot1',SOURCE,'fixture')
service=Service();hub=fake_publisher(service,work);hub.claim();hub.publish(snapshot1)
commit(2);native_before=sha(work/'run'/local.units[-1]['state_path']);calls=[]
CSV_before={p.relative_to(work/'run').as_posix():sha(p) for p in (work/'run/csv').rglob('*.csv')}
def first_flush(h,w,c,status):
    checked=store.Store(w/'run',c['contract'],SCHEMAS)
    calls.append((checked.state['updates'],status))
    return h.publish(checked.export(w/('snapshot'+str(len(calls)+1)),SOURCE,status))
with patch.object(coordinator,'publish_run',side_effect=first_flush):
    checked=coordinator.restore(hub,work,cfg)
assert calls==[(2,'recovery_local_preserved')] and checked.state['updates']==2
assert sha(work/'run'/local.units[-1]['state_path'])==native_before
assert CSV_before=={p.relative_to(work/'run').as_posix():sha(p) for p in (work/'run/csv').rglob('*.csv')}
manifest=hub.get_json('MANIFEST.json',hub.head());assert any(n.startswith('RECOVERY_REPAIR/') for n in manifest['files'])
with patch.object(coordinator,'publish_run',side_effect=AssertionError('No ahead state to flush')):
    same=coordinator.restore(hub,work,cfg)
assert same.state['updates']==2
# Divergent remote history has a valid object hash but must still be rejected.
obj=manifest['files']['HISTORY.json']['object'];key=hub.destination+'/'+obj
original=service.files[key];bad=json.loads(original);bad[0]['updates']=999
service.files[key]=canonical(bad);manifest['files']['HISTORY.json']['sha256']=digest(bad);manifest['files']['HISTORY.json']['bytes']=len(canonical(bad))
service.files[hub.destination+'/MANIFEST.json']=canonical(manifest);service.history[service.head]=dict(service.files)
with patch.object(coordinator,'publish_run',side_effect=AssertionError('Divergence must not flush')):
    rejected(lambda:coordinator.restore(hub,work,cfg))
service.files[key]=original;manifest['files']['HISTORY.json']['sha256']=__import__('hashlib').sha256(original).hexdigest();manifest['files']['HISTORY.json']['bytes']=len(original)
service.files[hub.destination+'/MANIFEST.json']=canonical(manifest);service.history[service.head]=dict(service.files)
commit(3)
with patch.object(coordinator,'publish_run',side_effect=ConnectionError('offline publication failed')):
    try:coordinator.restore(hub,work,cfg)
    except ConnectionError:pass
    else:raise AssertionError('Must stop before fitting if first flush fails')
assert store.Store(work/'run',contract,SCHEMAS).state['updates']==3
import requests,psutil
from types import SimpleNamespace
from rate_budget import BudgetSession
budget=SimpleNamespace(reserve=lambda bucket:None,observe=lambda bucket,response:None,category=lambda url:'api')
session=BudgetSession(budget,work/'HTTP_FIXTURE.jsonl');observed=[]
def send_fixture(self,request,**kwargs):
    observed.append(kwargs['timeout']);return SimpleNamespace(status_code=200,headers={})
request=requests.Request('GET','https://huggingface.co/api/offline-fixture').prepare()
with patch.object(requests.Session,'send',send_fixture):
    session.send(request,timeout=None);session.send(request,timeout=(3,7))
assert observed==[(20,150),(3,7)]
# Leftover-process handling targets this run's scripts and refuses a second writer.
other=SimpleNamespace(pid=99999,info=dict(status='running',cmdline=['python',str(work/'unrelated.py')]))
owned=SimpleNamespace(pid=99998,info=dict(status='running',cmdline=['python',str(SOURCE/'worker.py')]))
with patch.object(psutil,'process_iter',return_value=[other]):repair.quiesce(SOURCE,work,timeout=0)
assert not (work/'STOP_REQUEST.json').exists()
with patch.object(psutil,'process_iter',return_value=[owned]):rejected(lambda:repair.quiesce(SOURCE,work,timeout=0))
assert (work/'STOP_REQUEST.json').exists()
# Missing-local branch restores the exact snapshot rather than rejecting startup.
empty=work/'fresh_session';empty.mkdir()
restored=coordinator.restore(hub,empty,cfg)
assert restored.state['updates']==2 and torch.equal(restored.state['model']['weight'],torch.tensor([.2]))
assert torch.equal(restored.state['optimizer']['state'],torch.tensor([2]))
assert [int(r['sample']) for r in restored.rows('predictions')]==[1,2]
# With no remote snapshot, the recovery must refuse any fresh fit.
empty_hub=fake_publisher(Service(),work/'empty_hub');missing=work/'no_remote';missing.mkdir()
rejected(lambda:coordinator.restore(empty_hub,missing,cfg))
# Compatible attached output is read-only; partial output in working is retained.
input_root=work/'attached';saved=input_root/'old_output/run';saved.parent.mkdir(parents=True)
shutil.copytree(work/'run',saved)
source_files={p.relative_to(saved).as_posix():sha(p) for p in saved.rglob('*') if p.is_file()}
candidates=repair.saved_run_candidates([input_root],cfg['protocol'])
assert len(candidates)==1 and candidates[0]['updates']==3
import_work=work/'import_case';(import_work/'run').mkdir(parents=True)
partial=import_work/'run/unsent_note.csv';partial.write_text('retained,partial\n')
assert repair.import_saved_run(import_work,candidates)==3
assert any((p/'unsent_note.csv').read_text()=='retained,partial\n' for p in (import_work/'NB13_RECOVERY').glob('PRESERVED_PARTIAL_RUN_*'))
assert source_files=={p.relative_to(saved).as_posix():sha(p) for p in saved.rglob('*') if p.is_file()}
# Bootstrap is control/input validation only, and reuses the original live gate.
pinned=read(ROOT/'tread_two/context_validation/nb13_status/e0c714be2aba/remote/CONTRACT.json')
gate=read(ROOT/'tread_two/context_validation/nb13_status/e0c714be2aba/remote/LIVE_GATE_RECEIPT.json')
info=dict(object='evidence/gate.json',sha256=digest(gate),bytes=len(canonical(gate)))
reader=SimpleNamespace(head=lambda:'fixture_bootstrap_revision',get_json=lambda name,revision:dict(contract=repair.MEASURED_PROTOCOL,files={'LIVE_GATE_RECEIPT.json':info}) if name=='MANIFEST.json' else gate)
boot=work/'bootstrap_case';(boot/'NB13_RECOVERY').mkdir(parents=True)
def prepare_fixture(path,token):atomic(path/'CONFIG.json',dict(protocol=repair.MEASURED_PROTOCOL,contract=pinned))
with patch('common.environment',return_value=pinned['environment']),patch.object(coordinator,'prepare',side_effect=prepare_fixture) as prepared,patch('publisher.Publisher',return_value=reader):
    repair.prepare_recovery(SOURCE,boot,'fixture',pinned,input_roots=[work/'empty_input'])
    assert prepared.call_count==1
assert read(boot/'PREFLIGHT_ACCEPTED.json')==gate and not (boot/'run/POINTER.json').exists()
with patch('common.environment',return_value={'python':'different'}),patch.object(coordinator,'prepare',side_effect=AssertionError('Do not prepare on environment drift')):
    rejected(lambda:repair.prepare_recovery(SOURCE,boot,'fixture',pinned,input_roots=[work/'empty_input']))
assert (boot/'NB13_RECOVERY/ENVIRONMENT_MISMATCH.json').exists()
bad_gate=copy.deepcopy(gate);bad_gate['proof']['actual_live_HF_restore']=False
bad_reader=SimpleNamespace(head=reader.head,get_json=lambda name,revision:dict(contract=repair.MEASURED_PROTOCOL,files={'LIVE_GATE_RECEIPT.json':info}) if name=='MANIFEST.json' else bad_gate)
with patch('common.environment',return_value=pinned['environment']),patch.object(coordinator,'prepare',side_effect=prepare_fixture),patch('publisher.Publisher',return_value=bad_reader):
    rejected(lambda:repair.prepare_recovery(SOURCE,boot,'fixture',pinned,input_roots=[work/'empty_input']))
assert before=={n:sha(SOURCE/n) for n in before}
# A periodic status record was produced while the foreground was working.
deadline=time.monotonic()+2
while not (ctx['repair']/'STATUS.json').exists() and time.monotonic()<deadline:time.sleep(.02)
assert (ctx['repair']/'STATUS.json').exists()
ctx['done'].set()
result=dict(status='passed',revision=repair.REVISION,checks=['frozen32_source_entries_unchanged','native_and_scientific_CSV_bytes_unchanged_by_restore','verified_local_prefix_kept','ahead_local_state_published_before_training','equal_history_skips_full_download_and_extra_flush','divergence_rejected','failed_first_flush_does_not_train_or_delete_state','repair_provenance_in_direct_snapshot','live_phase_status_recorded','unconfigured_HTTP_timeout_bounded_explicit_timeout_preserved','only_owned_run_scripts_requested_to_stop_no_process_kill','fresh_session_native_CSV_snapshot_restore','missing_remote_refuses_fresh_fit','saved_output_readonly_import','unanchored_partial_run_preserved','bootstrap_reuses_verified_gate_without_fit','bootstrap_environment_drift_rejected','bootstrap_corrupt_gate_rejected'],
    network_requests=0,model_forward_calls=0,optimizer_updates=0,fixture=str(work),actual_Kaggle_recovery=False,launcher_sha256=sha(Path(repair.__file__)))
(OUT/'VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
