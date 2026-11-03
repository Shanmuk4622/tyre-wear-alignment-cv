"""External orchestration-only NB13 recovery. Never edits frozen model sources.

Verify local history against the pinned remote prefix, publish ahead local state
before fitting, and report long phases. No model/optimizer/scientific CSV change.
"""
import argparse,contextlib,faulthandler,functools,json,os,shutil,sys,threading,time,traceback
from pathlib import Path

REVISION='nb13-recovery-20261031-r2'
INVENTORY='f63d309ebed9dec516dba02ef8ed082b65b1cbbbfe2132368f47cc136f4193b8'
MEASURED_PROTOCOL='36f473b0886d6c2f49c416436d270198af3982230e3045da9969b0bf42169625'

def install(source,work,monitor_seconds=15):
    source,work=Path(source).resolve(),Path(work).resolve()
    sys.path.insert(0,str(source))
    from common import atomic,canonical,digest,event,file_meta,read,sha
    import coordinator,store,publisher,evidence_hub,rate_budget
    assert sha(source/'FILES.json')==INVENTORY,'Wrong source inventory; preserve the old run'
    inventory=read(source/'FILES.json');assert len(inventory)==32
    assert all(file_meta(source/n)==meta for n,meta in inventory.items()),'Frozen source/asset changed'
    repair=work/'NB13_RECOVERY';repair.mkdir(parents=True,exist_ok=True)
    metadata=dict(revision=REVISION,launcher_sha256=sha(Path(__file__)),source_inventory_sha256=INVENTORY,
        frozen_source_files=32,model_or_recipe_changed=False,contract_or_namespace_changed=False,
        scientific_CSV_changed=False,local_ahead_state_flushed_before_training=True,
        verified_local_prefix_skips_redundant_full_remote_download=True,empty_session_restores_verified_HF=True,
        progress_seconds=monitor_seconds,stack_capture_seconds=180,unconfigured_HTTP_timeout_seconds=[20,150],
        root_cause_of_reported_stop='Not established from saved output; no saved traceback')
    atomic(repair/'REPAIR.json',metadata)
    lock=threading.RLock();phase={'name':'installed','since':time.monotonic()};done=threading.Event()
    @contextlib.contextmanager
    def span(name):
        with lock:old=dict(phase);phase.update(name=name,since=time.monotonic())
        event(repair/'EVENTS.jsonl','phase_begin',phase=name)
        print('[NB13]',name,flush=True)
        try:yield
        finally:
            event(repair/'EVENTS.jsonl','phase_end',phase=name)
            with lock:phase.update(old)
    def monitor():
        while not done.wait(monitor_seconds):
            try:
                with lock:current=dict(phase)
                cursor={};pointer=work/'run/POINTER.json'
                if pointer.exists():
                    value=read(pointer);unit=read(work/'run'/value['unit']);cursor={k:unit[k] for k in ('seq','updates','phase')}
                usage=shutil.disk_usage(work)
                observation=dict(operation=current['name'],operation_seconds=round(time.monotonic()-current['since'],1),
                    local_cursor=cursor,free_bytes=usage.free)
                atomic(repair/'STATUS.json',observation)
                event(repair/'EVENTS.jsonl','heartbeat',**observation)
                print('[NB13 status]',json.dumps(observation),flush=True)
            except Exception as error:
                print('[NB13 status unavailable]',type(error).__name__,flush=True)
    threading.Thread(target=monitor,name='NB13-status',daemon=True).start()
    def wrap(cls,name,label):
        original=getattr(cls,name)
        @functools.wraps(original)
        def call(*args,**kwargs):
            with span(label):return original(*args,**kwargs)
        setattr(cls,name,call)
    for name in ('__init__','verify_and_rollback','export','install'):
        wrap(store.Store,name,'local '+name)
    wrap(publisher.Publisher,'publish','HF immutable snapshot publication')
    wrap(publisher.Publisher,'commit','HF bounded commit / verification')
    wrap(publisher.Publisher,'download','verified remote download')
    original_call=evidence_hub.Hub.call
    def observed_call(self,fn,operation):
        with span('HF '+operation):return original_call(self,fn,operation)
    evidence_hub.Hub.call=observed_call
    original_send=rate_budget.BudgetSession.send
    def bounded_send(self,request,**kwargs):
        if kwargs.get('timeout') is None:kwargs['timeout']=(20,150)
        return original_send(self,request,**kwargs)
    rate_budget.BudgetSession.send=bounded_send
    original_export=store.Store.export
    def exported(self,target,*args,**kwargs):
        target=original_export(self,target,*args,**kwargs)
        dest=target/'RECOVERY_REPAIR';dest.mkdir(exist_ok=True)
        shutil.copy2(Path(__file__).resolve(),dest/'launcher.py')
        for p in repair.iterdir():
            if p.is_file() and p.suffix in ('.json','.jsonl','.txt'):
                shutil.copy2(p,dest/p.name)
        manifest=read(target/'MANIFEST.json')
        manifest['files']={p.relative_to(target).as_posix():file_meta(p) for p in target.rglob('*') if p.is_file() and p.name!='MANIFEST.json'}
        atomic(target/'MANIFEST.json',manifest)
        return target
    store.Store.export=exported
    original_install=store.Store.install
    def installed(self,snapshot):
        result=original_install(self,snapshot)
        for p in (Path(snapshot)/'RECOVERY_REPAIR').glob('*'):
            if p.is_file():
                dest=self.root/'session_history'/(sha(p)[:20]+'-NB13-RECOVERY-'+p.name)
                dest.parent.mkdir(exist_ok=True);shutil.copy2(p,dest)
        return result
    store.Store.install=installed
    original_restore=coordinator.restore
    def restore(hub,work,cfg):
        with span('verify retained local state and remote history'):
            local=store.Store(Path(work)/'run',cfg['contract'],coordinator.SCHEMAS)
            if local.state is None:
                print('No local checkpoint is accessible. Restoring the complete verified HF snapshot; unsent local updates cannot be invented.',flush=True)
                restored=original_restore(hub,work,cfg)
                assert restored.state is not None,'HF recovery did not provide a checkpoint; refusing to start a fresh fit'
                print('HF recovery verified at update',restored.state['updates'],'stage',restored.state['phase'],flush=True)
                return restored
            parent=hub.head();manifest=hub.get_json('MANIFEST.json',parent)
            assert manifest and manifest['contract']==cfg['protocol'],'Remote identity mismatch'
            info=manifest['files']['HISTORY.json'];remote_history=hub.get_json(info['object'],parent)
            assert digest(remote_history)==info['sha256'] and len(canonical(remote_history))==info['bytes'],'Remote history integrity failure'
            if len(local.units)<len(remote_history):
                print('Remote history is ahead; invoking original full verified recovery.',flush=True)
                return original_restore(hub,work,cfg)
            assert local.units[:len(remote_history)]==remote_history,'Local/remote history diverges; preserve both'
            cursor_info=manifest['files']['CURSOR.json'];remote_cursor=hub.get_json(cursor_info['object'],parent)
            assert digest(remote_cursor)==cursor_info['sha256'] and remote_cursor['updates']==remote_history[-1]['updates']
            atomic(Path(work)/'LAST_REMOTE.json',dict(revision=parent,protocol=cfg['protocol'],restored_or_checked=True,
                verified_local_history_prefix=True,updates=remote_cursor['updates'],phase=remote_cursor['phase']))
            event(repair/'EVENTS.jsonl','verified_local_prefix',local_updates=local.state['updates'],remote_updates=remote_cursor['updates'],remote_revision=parent)
            print('Verified local update',local.state['updates'],'and remote update',remote_cursor['updates'],'with identical history prefix.',flush=True)
        # Preserve ahead local science remotely before any further optimizer work.
        if len(local.units)>len(remote_history):
            with span('publish retained local progress BEFORE resuming fitting'):
                coordinator.publish_run(hub,Path(work),cfg,'recovery_local_preserved')
        else:print('No ahead local units need an initial recovery publication.',flush=True)
        return local
    coordinator.restore=restore
    return dict(metadata=metadata,done=done,repair=repair,span=span)

def saved_run_candidates(roots,protocol,exclude=None):
    """Read metadata of compatible attached outputs; never mutate input files."""
    from common import read,sha,inside
    result=[];seen=set();exclude=Path(exclude).resolve() if exclude else None
    for root in roots:
        root=Path(root)
        if not root.exists():continue
        for pointer in root.rglob('POINTER.json'):
            run=pointer.parent.resolve()
            if run==exclude or run in seen or not run.is_relative_to(root.resolve()):continue
            seen.add(run)
            try:
                contract=read(run/'CONTRACT.json')
                from common import digest
                if digest(contract)!=protocol:continue
                p=read(pointer);unit_path=inside(run,p['unit'])
                if sha(unit_path)!=p['sha256']:continue
                unit=read(unit_path)
                if unit['protocol']!=protocol:continue
                state=inside(run,unit['state_path'])
                if not state.is_file() or state.stat().st_size!=unit['state_meta']['bytes'] or sha(state)!=unit['state_meta']['sha256']:continue
                result.append(dict(path=str(run),updates=unit['updates'],seq=unit['seq']))
            except (OSError,ValueError,KeyError,AssertionError):continue
    return sorted(result,key=lambda item:(item['updates'],item['seq']),reverse=True)

def preserve_unanchored_run(work):
    """Retain a partial tree before remote recovery; never erase it as a shortcut."""
    import uuid
    work=Path(work).resolve();run=work/'run'
    if not run.exists() or (run/'POINTER.json').exists():return
    preserved=(work/'NB13_RECOVERY'/('PRESERVED_PARTIAL_RUN_'+uuid.uuid4().hex)).resolve()
    assert preserved.is_relative_to(work) and run.parent==work and not preserved.exists()
    preserved.parent.mkdir(parents=True,exist_ok=True)
    os.replace(run,preserved)

def import_saved_run(work,candidates):
    """Copy the best compatible run into output; verify all science before use."""
    if not candidates:return None
    from common import read,atomic
    from schemas import SCHEMAS
    from store import Store
    work=Path(work).resolve();run=work/'run'
    assert not (run/'POINTER.json').exists(),'Retained current state takes priority'
    source=Path(candidates[0]['path']).resolve()
    assert source!=run and not source.is_relative_to(run),'Cannot import a run from itself'
    required=sum(p.stat().st_size for p in source.rglob('*') if p.is_file())
    assert shutil.disk_usage(work).free>required+2*1024**3,'Not enough free space to retain and verify saved output'
    preserve_unanchored_run(work)
    shutil.copytree(source,run)
    checked=Store(run,read(run/'CONTRACT.json'),SCHEMAS)
    assert checked.state is not None
    atomic(work/'NB13_RECOVERY'/'IMPORTED_RUN.json',dict(source=str(source),updates=checked.state['updates'],bytes=required,full_local_verification=True))
    return checked.state['updates']

def prepare_recovery(source,work,token,expected_contract,input_roots=None):
    """Rebuild paths from original inputs; reuse verified gate, never start a fit."""
    from common import atomic,canonical,digest,environment,read
    import coordinator
    from publisher import Publisher
    assert digest(expected_contract)==MEASURED_PROTOCOL,'Wrong pinned original contract'
    actual=environment();expected=expected_contract['environment']
    if actual!=expected:
        changed={k:dict(expected=expected.get(k),actual=actual.get(k)) for k in set(actual)|set(expected) if actual.get(k)!=expected.get(k)}
        atomic(work/'NB13_RECOVERY'/'ENVIRONMENT_MISMATCH.json',changed)
        raise RuntimeError('Environment differs from the saved experiment. Preserve files; inspect NB13_RECOVERY/ENVIRONMENT_MISMATCH.json. No new protocol or fit was created.')
    if (work/'CONFIG.json').exists():assert read(work/'CONFIG.json')['contract']==expected_contract,'Existing configuration differs'
    # The original verifier downloads the two exact upstream weights, reads the
    # dataset from Kaggle and checks the entire image/split/source manifest.
    coordinator.prepare(work,token)
    cfg=read(work/'CONFIG.json');assert cfg['protocol']==MEASURED_PROTOCOL and cfg['contract']==expected_contract
    if not (work/'run/POINTER.json').exists():
        candidates=saved_run_candidates(input_roots or [Path('/kaggle/input')],MEASURED_PROTOCOL,exclude=work/'run')
        atomic(work/'NB13_RECOVERY'/'SAVED_RUN_CANDIDATES.json',candidates)
        if candidates:
            print('Compatible saved output found; copying and fully verifying update',candidates[0]['updates'],flush=True)
            import_saved_run(work,candidates)
        else:
            preserve_unanchored_run(work)
            print('No compatible retained/attached run found. HF recovery will restore the last verified remote checkpoint.',flush=True)
    reader=Publisher(expected_contract['HF_prefix']+'/fit/'+MEASURED_PROTOCOL,token,work/'bootstrap/REMOTE_HISTORY.jsonl')
    revision=reader.head();manifest=reader.get_json('MANIFEST.json',revision)
    assert manifest and manifest['contract']==MEASURED_PROTOCOL,'No matching remote run; refusing a fresh fit'
    info=manifest['files']['LIVE_GATE_RECEIPT.json'];gate=reader.get_json(info['object'],revision)
    assert digest(gate)==info['sha256'] and len(canonical(gate))==info['bytes'],'Saved gate receipt changed'
    assert gate['protocol']==MEASURED_PROTOCOL and gate['proof']['status']=='passed' and gate['proof']['actual_live_HF_restore']
    assert gate['proof']['storage_projection']['projection_passed'],'Original storage preflight did not pass'
    atomic(work/'PREFLIGHT_ACCEPTED.json',gate)
    atomic(work/'NB13_RECOVERY'/'BOOTSTRAP.json',dict(protocol=MEASURED_PROTOCOL,verified_revision=revision,original_gate_reused=True,
        local_checkpoint_present=(work/'run/POINTER.json').exists(),new_fit_started=False))
    print('Reused the verified original live preflight. Saved-progress recovery is next; no repeated fit fixture.',flush=True)
    return cfg

def quiesce(source,work,timeout=180):
    """Request safe exit only from this run's owned scripts; never kill processes."""
    import psutil
    from common import atomic
    targets={str((source/name).resolve()) for name in ('worker.py','coordinator.py')}
    def active():
        found=[]
        for process in psutil.process_iter(['pid','cmdline','status']):
            try:
                if process.pid==os.getpid() or process.info['status']==psutil.STATUS_ZOMBIE:continue
                args=process.info['cmdline'] or []
                if any(str(Path(arg).resolve()) in targets for arg in args if arg.endswith('.py')):found.append(process.pid)
            except (psutil.Error,OSError):continue
        return found
    deadline=time.monotonic()+timeout;last=0
    while True:
        pids=active()
        if not pids:return
        atomic(work/'STOP_REQUEST.json',dict(reason='NB13 recovery requests safe exit of previous owned run'))
        if time.monotonic()>=deadline:raise RuntimeError('Previous NB13 processes still active; no second writer started. Preserve files and inspect NB13_RECOVERY diagnostics.')
        if time.monotonic()-last>=15:print('Waiting for previous NB13 processes to stop safely:',pids,flush=True);last=time.monotonic()
        time.sleep(2)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--work',required=True);parser.add_argument('--contract',required=True);args=parser.parse_args()
    source,work=Path(args.source).resolve(),Path(args.work).resolve()
    assert work==Path('/kaggle/working/tread_two_atlas_context_r3') and source==work/'source','Use the existing NB13 work directory'
    sys.path.insert(0,str(source));from common import read,event
    quiesce(source,work)
    context=install(source,work)
    stacks=(context['repair']/'THREAD_STACKS.txt').open('a',encoding='utf-8')
    faulthandler.dump_traceback_later(180,repeat=True,file=stacks)
    import coordinator
    old=sys.argv;sys.argv=[str(source/'coordinator.py'),'train','--work',str(work)]
    coordinator.ACTIVE_WORK=work
    import signal
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,coordinator.stop)
    try:
        from filelock import FileLock
        with FileLock(str(work/'CONTROLLER.lock'),timeout=1),context['span']('verify inputs and prepare local-or-HF recovery'):
            prepare_recovery(source,work,os.environ.get('HF_TOKEN'),read(Path(args.contract)))
        if coordinator.STOP:raise InterruptedError('Stop requested during preparation; no fitting started')
        with context['span']('resume original NB13 experiment'):
            coordinator.main()
        event(context['repair']/'EVENTS.jsonl','recovery_complete');return 0
    except BaseException as error:
        frames=[dict(file=Path(f.filename).name,line=f.lineno,function=f.name) for f in traceback.extract_tb(error.__traceback__)]
        event(context['repair']/'EVENTS.jsonl','recovery_error',error_type=type(error).__name__,frames=frames)
        print('NB13 recovery stopped:',type(error).__name__,str(error)[:400],flush=True)
        print('Preserve run/console.log and NB13_RECOVERY; no model or protocol was replaced.',flush=True)
        return 1
    finally:
        sys.argv=old;context['done'].set();faulthandler.cancel_dump_traceback_later();stacks.close()

if __name__=='__main__':sys.exit(main())
