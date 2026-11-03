"""Secret-free worker; chunk preparation, complete-update fitting and finite gate."""
import argparse,os,time,traceback
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
assert not os.environ.get('HF_TOKEN'),'Numeric worker must not receive token'
from common import atomic,read,digest,file_meta,seed_all
from config import configuration
from cache import prepare
from engine import Engine
from store import semantic_hash

def ready(output,folder,protocol,files,reason,unit=0,update=0):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    atomic(folder/'SNAPSHOT_STATUS.json',dict(reason=reason,unit=unit,last_completed_update=update,forced_death_cannot_flush=True));files['SNAPSHOT_STATUS.json']=file_meta(folder/'SNAPSHOT_STATUS.json')
    atomic(folder/'MANIFEST.json',dict(contract=protocol,files=files))
    atomic(Path(output)/'SNAPSHOT_READY.json',dict(path=str(folder),unit=unit,update=update,reason=reason))

def cache_snapshot(output,cache_root,protocol,index,reason):
    import os,shutil
    target=Path(output)/'snapshots'/str(time.time_ns());target.mkdir(parents=True)
    files={}
    for name,meta in index['files'].items():
        dest=target/'cache'/name;dest.parent.mkdir(parents=True,exist_ok=True);os.link(Path(cache_root)/name,dest);files['cache/'+name]=meta
    if (Path(cache_root)/'CACHE.json').exists():
        shutil.copyfile(Path(cache_root)/'CACHE.json',target/'cache/CACHE.json');files['cache/CACHE.json']=file_meta(target/'cache/CACHE.json')
    ready(output,target,protocol,files,reason,unit=len(index['chunks']))

def await_ack(output):
    output=Path(output)
    while (output/'SNAPSHOT_READY.json').exists():
        if (output/'STOP_REQUESTED').exists():return False
        time.sleep(.5)
    return True

def snapshot(engine,reason):
    atomic(engine.root/'STATUS.json',dict(phase='snapshot_staging',reason=reason,epoch=engine.cursor['epoch'],update=engine.cursor['update'],unit=engine.store.unit))
    print('TRACE staging immutable snapshot:',reason,'update',engine.cursor['update'],flush=True)
    target=engine.root/'snapshots'/str(time.time_ns());engine.store.snapshot(target,allow_copy=engine.fixture and engine.device=='cpu')
    m=read(target/'MANIFEST.json');atomic(target/'SNAPSHOT_STATUS.json',dict(reason=reason,unit=engine.store.unit,last_completed_update=engine.cursor['update']));m['files']['SNAPSHOT_STATUS.json']=file_meta(target/'SNAPSHOT_STATUS.json');atomic(target/'MANIFEST.json',m)
    atomic(engine.root/'SNAPSHOT_READY.json',dict(path=str(target),unit=engine.store.unit,update=engine.cursor['update'],reason=reason))
    engine.status()

def service(engine):
    for path in sorted((engine.root/'service_queue').glob('*.json')):
        if path.stem not in engine.cursor['services']:
            engine.cursor['services'].append(path.stem);engine.store.commit(engine.model,engine.optimizer,engine.cursor,read(path),kind='service')
        path.replace(path.with_suffix('.done'))

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','gate_direct','gate_restore','init','train']);p.add_argument('--output',required=True)
    p.add_argument('--cache',required=True);p.add_argument('--dataset',default='');p.add_argument('--geometry');p.add_argument('--reference');p.add_argument('--offline',action='store_true');p.add_argument('--no-ack',action='store_true')
    a=p.parse_args();output=Path(a.output);output.mkdir(parents=True,exist_ok=True);engine=None
    import torch
    torch.set_num_threads(4)
    try:
        if a.mode=='prepare':
            from data import locate
            cfg=configuration();protocol=digest(cfg)
            if not a.offline:assert torch.cuda.device_count()==2 and all('T4' in torch.cuda.get_device_name(i) for i in range(2)),'Use a Kaggle T4 x2 session'
            seed_all(1)
            def tick(index):
                if (output/'STOP_REQUESTED').exists():
                    cache_snapshot(output,a.cache,protocol,index,'catchable_prepare_stop');raise InterruptedError('Preparation stopped at complete chunk')
                if (output/'SNAPSHOT_REQUEST').exists() and (Path(a.cache)/'CACHE.json').exists():
                    cache_snapshot(output,a.cache,protocol,index,'periodic_prepare');(output/'SNAPSHOT_REQUEST').unlink(missing_ok=True)
                    if not a.no_ack and not await_ack(output):raise InterruptedError('Stop during preparation publication')
            cache=prepare(a.cache,locate(a.dataset),a.geometry,protocol,'cpu' if a.offline else 'cuda:1',service=tick)
            cache_snapshot(output,a.cache,protocol,cache.index,'fixed_cache_complete');return
        fixture=a.mode.startswith('gate')
        if not fixture:
            assert not a.offline and torch.cuda.device_count()==2 and all('T4' in torch.cuda.get_device_name(i) for i in range(2)),'Measured worker requires actual T4 x2; offline mode is engineering-only'
            cfg=configuration();certificate=read(output.parent/'preparation/ENGINEERING_READY.json')
            assert certificate['status']=='passed' and certificate['protocol']==digest(cfg) and certificate['sources']==cfg['sources'] and certificate['environment']==cfg['environment']
            assert certificate['actual_dual_T4'] and certificate['actual_HF_restore_continue'],'Actual engineering HF continuation certificate required before measured initialization'
        engine=Engine(output,a.cache,'cpu' if a.offline else 'cuda:0',fixture)
        if a.mode=='gate_direct':
            from engineering import check
            # Re-seeded model checks cannot advance the engine's saved RNG.
            from common import rng_state,restore_rng
            state=rng_state();check(engine.cache,engine.device,output/'NUMERICAL_GATE.json');restore_rng(state)
        if a.mode=='init':snapshot(engine,'before_new_optimizer_updates');return
        while engine.cursor['phase']!='completed':
            service(engine)
            if (output/'STOP_REQUESTED').exists():snapshot(engine,'catchable_stop');return
            if (output/'SNAPSHOT_REQUEST').exists():
                snapshot(engine,'periodic');(output/'SNAPSHOT_REQUEST').unlink(missing_ok=True)
                if not a.no_ack and not await_ack(output):return
            before=engine.cursor['phase'];engine.step()
            print('TRACE',a.mode,engine.cursor['phase'],'epoch',engine.cursor['epoch'],'update',engine.cursor['update'],'unit',engine.store.unit,flush=True)
            if a.mode=='gate_direct' and engine.cursor['update']==1 and not (output/'seam/MANIFEST.json').exists():engine.store.snapshot(output/'seam',allow_copy=a.offline)
            major=(before=='baseline' and engine.cursor['phase']=='train') or (before=='validate' and engine.cursor['phase']=='endpoint') or (before=='endpoint' and engine.cursor['phase']=='probe')
            # Epochs stay durable locally. Major *stage* boundaries and the
            # parent's30-minute request trigger HF; not two uploads every epoch.
            if a.mode=='train' and major:
                snapshot(engine,'major_phase_'+engine.cursor['phase'])
                if not a.no_ack and not await_ack(output):return
        service(engine)
        if fixture:
            from budget import project
            budget=project(engine)
            if a.mode=='gate_restore':
                reference=Path(a.reference)/'run';pointer=read(reference/'POINTER.json');state=torch.load(reference/pointer['state'],map_location='cpu',weights_only=False)
                actual=torch.load(engine.store.root/read(engine.store.root/'POINTER.json')['state'],map_location='cpu',weights_only=False)
                for key in ['model','optimizer','rng','cursor']:assert semantic_hash(state[key])==semantic_hash(actual[key]),'Continuation differs: '+key
                ignored={'timings','resource_measurements','state_ledger','resume_events','native_chunks_index','transport_events','transport_receipts','staging_inventory'}
                exact=0
                for name,meta in state['closed'].items():
                    parts=Path(name).parts
                    if name.startswith('native/') or parts[0]=='csv' and parts[1] not in ignored:
                        assert actual['closed'].get(name)==meta,('Scientific evidence differs',name);exact+=1
                atomic(output/'RESTORE_PROOF.json',dict(status='passed',exact_model_optimizer_RNG_cursor=True,scientific_files_exact=exact,engineering_updates=engine.cursor['update'],dual_CUDA_RNG_states=len(state['rng']['cuda']),actual_HF_download_asserted_by_parent=False))
        snapshot(engine,'completed')
    except BaseException as exc:
        trace=traceback.format_exc();(output/'TRACEBACK.txt').write_text(trace,encoding='utf-8')
        if engine is not None and (engine.store.root/'POINTER.json').exists():
            try:
                engine.cursor=engine.store.restore(engine.model,engine.optimizer)
                path=engine.store.root/'errors'/f'unit_{engine.store.unit+1:06d}_TRACEBACK.txt';path.parent.mkdir(exist_ok=True);path.write_text(trace,encoding='utf-8');engine.store.register(path)
                rec={'errors':[dict(stage='error',session='numeric_worker',phase=engine.cursor['phase'],exception_type=type(exc).__name__,sanitized_message=str(exc),traceback_file=path.relative_to(engine.store.root).as_posix(),last_committed_unit=engine.store.unit,flush_attempted=True)]}
                engine.store.commit(engine.model,engine.optimizer,engine.cursor,rec,seal=True,kind='error');snapshot(engine,'error_last_complete')
            except BaseException:pass
        atomic(output/'WORKER_ERROR.json',dict(error_type=type(exc).__name__,message=str(exc)))
        raise

if __name__=='__main__':main()
