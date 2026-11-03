"""Secret-free numerical worker; isolated gate and one measured resumable lifecycle."""
import argparse,os,time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
assert not os.environ.get('HF_TOKEN'),'The worker must not receive the HF token'
from common import atomic,cpu,digest,file_meta,read,sha
from engine_fit import Engine
from budget_fit import project
from store_fit import semantic_hash

def snapshot(engine,target,reason):
    receipt=engine.store.snapshot(target)
    atomic(engine.root/'SNAPSHOT_READY.json',dict(**receipt,reason=reason,unit=engine.store.unit,update=engine.cursor['update'],protocol=engine.protocol))
    return receipt

def service(engine):
    for path in sorted((engine.root/'service_queue').glob('*.json')):
        if path.stem not in engine.cursor.setdefault('service_applied',[]):
            records=read(path);engine.cursor['service_applied'].append(path.stem)
            engine.store.commit(engine.model,engine.optimizer,engine.cursor,records,kind='transport_metadata')
        # Durable pointer selects the typed service rows before acknowledgement.
        done=path.with_suffix('.done');path.replace(done)

def loop(engine,mode,allow_copy=False):
    last_phase=engine.cursor['phase'];last_notice=0.
    while engine.cursor['phase']!='completed':
        service(engine)
        if (engine.root/'STOP_REQUESTED').exists():
            snapshot(engine,engine.root/'snapshots'/str(time.time_ns()),'catchable_stop');return 'stopped'
        if (engine.root/'SNAPSHOT_REQUEST').exists():
            snapshot(engine,engine.root/'snapshots'/str(time.time_ns()),'periodic')
            (engine.root/'SNAPSHOT_REQUEST').unlink(missing_ok=True)
        if mode=='gate_direct' and engine.store.unit==18 and not (engine.root/'seam/MANIFEST.json').exists():
            engine.store.snapshot(engine.root/'seam',allow_copy=allow_copy)
        engine.step()
        phase=engine.cursor['phase']
        if time.monotonic()-last_notice>=20 or phase!=last_phase:
            print(mode,':',phase,'epoch',engine.cursor['epoch'],'update',engine.cursor['update'],'unit',engine.store.unit,flush=True);last_notice=time.monotonic()
        if mode=='train' and last_phase=='baseline' and phase=='train':snapshot(engine,engine.root/'snapshots'/str(time.time_ns()),'initial_controls_complete')
        last_phase=phase
    service(engine)
    if mode=='train':snapshot(engine,engine.root/'snapshots'/str(time.time_ns()),'completed')
    return 'completed'

def compare_gate(engine,reference):
    import torch
    reference=Path(reference);pointer=read(reference/'run/POINTER.json')
    state=torch.load(reference/'run'/pointer['state'],map_location='cpu',weights_only=False)
    actual=engine.store.root/read(engine.store.root/'POINTER.json')['state'];current=torch.load(actual,map_location='cpu',weights_only=False)
    hashes=[]
    for key in ('model','optimizer','rng','cursor'):
        assert semantic_hash(state[key])==semantic_hash(current[key]),'Gate restore changed '+key
        hashes.append(key)
    ignored={'pipeline_timings','resource_samples','unit_ledger','HF_requests','HF_publications','environment'}
    checked=0
    for name,meta in state['closed'].items():
        path=Path(name)
        scientific=path.suffix=='.npy' or (path.parts[0]=='csv' and path.parts[1] not in ignored)
        if scientific:
            assert current['closed'][name]==meta,(name,'scientific shard differs')
            checked+=1
    result=dict(status='passed',model_optimizer_RNG_cursor_exact=hashes,scientific_shards_exact=checked,
                unit1_within_epoch_restore=True,engineering_updates=engine.cursor['update'],protocol=engine.protocol,
                fixture_only=True)
    atomic(engine.root/'RESTORE_PROOF.json',result)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('gate_direct','gate_restore','init','train','snapshot'))
    parser.add_argument('--output',required=True);parser.add_argument('--weights',required=True);parser.add_argument('--data-root',default='')
    parser.add_argument('--reference');parser.add_argument('--target');parser.add_argument('--offline',action='store_true')
    args=parser.parse_args();root=Path(args.output)
    fixture=args.mode.startswith('gate')
    device='cpu' if args.offline else 'cuda:0';eval_device='cpu' if args.offline else 'cuda:1'
    engine=Engine(root,args.weights,args.data_root,device,eval_device,fixture)
    try:
        if args.mode=='snapshot':snapshot(engine,args.target,'explicit');return
        if args.mode=='init':snapshot(engine,root/'snapshots'/str(time.time_ns()),'initialized_before_measured_updates');return
        status=loop(engine,args.mode,args.offline)
        if status=='stopped':return
        if fixture:
            budget=project(engine)
            if not args.offline:assert budget['status']=='passed','20GB storage/staging gate failed before measured fitting'
            if args.mode=='gate_restore':compare_gate(engine,args.reference)
        atomic(root/'WORKER_RESULT.json',dict(status=status,protocol=engine.protocol,phase=engine.cursor['phase'],update=engine.cursor['update'],unit=engine.store.unit,fixture=fixture))
    except BaseException as exc:
        # Select the last native complete unit before snapshotting any state.
        try:
            engine.cursor=engine.store.restore(engine.model,engine.optimizer)
            is_numeric='nonfinite' in str(exc).lower() or 'nan' in str(exc).lower()
            engine.store.commit(engine.model,engine.optimizer,engine.cursor,{'numeric_events':[dict(event_id=engine.store.unit+1,unit=engine.store.unit+1,event_kind=type(exc).__name__,stage=engine.cursor['phase'],parameter_or_tensor=None,finite=False if is_numeric else None,action='stop_no_automatic_numeric_retry',replay_from_update=engine.cursor['update'],reason=str(exc))]},kind='error_event')
            snapshot(engine,root/'snapshots'/str(time.time_ns()),'error_last_complete')
        except BaseException:pass
        atomic(root/'WORKER_ERROR.json',dict(error_type=type(exc).__name__,message=str(exc),unit=engine.store.unit,last_complete_update=engine.cursor['update']))
        raise

if __name__=='__main__':main()
