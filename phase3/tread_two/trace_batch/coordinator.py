"""Kaggle-only parent: one writer, automatic dependencies, verified HF continuation."""
import csv,hashlib,io,json,os,queue,re,shutil,signal,subprocess,sys,threading,time,uuid
from pathlib import Path
from filelock import FileLock
from common import HERE,atomic,canonical,digest,file_meta,read,sha
from config import configuration,NAMESPACE,GEOMETRY
from publisher import Publisher
from store import Store

def visible(label,fn):
    stop=threading.Event()
    def beat():
        while not stop.wait(25):print('TRACE alive:',label,'— local completed work is retained.',flush=True)
    thread=threading.Thread(target=beat,daemon=True);thread.start()
    try:return fn()
    finally:stop.set();thread.join(timeout=1)

def selected(pub,name):
    rev=pub.head();manifest=pub.get_json('MANIFEST.json',rev)
    if not manifest or name not in manifest['files']:return None,rev,manifest
    meta=manifest['files'][name];body=pub.get_json(meta['object'],rev)
    assert hashlib.sha256(canonical(body)).hexdigest()==meta['sha256'],name
    return body,rev,manifest

def run_worker(mode,output,cache,dataset='',geometry=None,reference=None,tick=None,offline=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'STOP_REQUESTED').unlink(missing_ok=True)
    args=[sys.executable,'-u','-B',str(HERE/'worker.py'),mode,'--output',str(output),'--cache',str(cache),'--dataset',str(dataset)]
    if geometry:args+=['--geometry',str(geometry)]
    if reference:args+=['--reference',str(reference)]
    if offline:args+=['--offline']
    env=dict(os.environ);env.pop('HF_TOKEN',None)
    proc=subprocess.Popen(args,cwd=HERE,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,
        start_new_session=os.name!='nt',creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_NO_WINDOW) if os.name=='nt' else 0)
    messages=queue.Queue()
    def pump():
        for line in proc.stdout:messages.put(line)
    t=threading.Thread(target=pump,daemon=True);t.start();last=0
    try:
        while proc.poll() is None or t.is_alive() or not messages.empty():
            try:print(messages.get(timeout=.3),end='',flush=True)
            except queue.Empty:pass
            if tick:tick()
            if time.monotonic()-last>25:
                status=read(output/'STATUS.json') if (output/'STATUS.json').exists() else dict(phase=mode)
                print('TRACE heartbeat:',json.dumps(status),flush=True);last=time.monotonic()
    except BaseException:
        (output/'STOP_REQUESTED').write_text('catchable_parent_stop\n',encoding='utf-8')
        try:visible('finish current complete worker unit',lambda:proc.wait(timeout=90))
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=10)
        raise
    assert proc.returncode==0,'Worker stopped; keep files and inspect WORKER_ERROR.json / notebook output. No automatic numeric retry.'

def registry(pub,cfg=None):
    name=NAMESPACE+'/EXPERIMENT.json'
    def get():
        with pub.session.get(f'https://huggingface.co/datasets/{pub.REPO}/resolve/{pub.head()}/{name}',timeout=(20,150)) as response:
            if response.status_code==404:return None
            response.raise_for_status();return response.json()
    actual=pub.call(get,'read_single_TRACE_experiment')
    if cfg is None:return actual
    expected=dict(protocol=digest(cfg),config=cfg,mode='one_measured_candidate')
    if actual is not None:
        assert actual==expected,'Different TRACE source/environment is already bound. Preserve and resume the original; no automatic new fit.'
        return actual
    from huggingface_hub import CommitOperationAdd
    for attempt in range(6):
        parent=pub.head()
        try:
            pub.call(lambda:pub.api.create_commit(pub.REPO,repo_type='dataset',parent_commit=parent,operations=[CommitOperationAdd(path_in_repo=name,path_or_fileobj=canonical(expected))],commit_message='Bind one TRACE r1 candidate'),'bind_TRACE_experiment');return expected
        except Exception as exc:
            if getattr(getattr(exc,'response',None),'status_code',None) in (409,412):
                actual=pub.call(get,'retry_TRACE_registry')
                if actual is not None:assert actual==expected;return actual
                continue
            raise
    raise RuntimeError('Could not bind TRACE registry')

def chain(root,unit):
    for path in [*Path(root).glob('csv/state_ledger/*.csv'),*Path(root).glob('active/csv_state_ledger*.csv')]:
        with path.open(encoding='utf-8',newline='') as stream:
            for row in csv.DictReader(stream):
                if int(row['unit'])==unit:return row['chain_sha256']
    return None

def remote_chain(pub,rev,manifest,unit):
    for name,meta in manifest['files'].items():
        if not (name.startswith('csv/state_ledger/') or name.startswith('active/csv_state_ledger')):continue
        def get():
            with pub.session.get(f'https://huggingface.co/datasets/{pub.REPO}/resolve/{rev}/{pub.destination}/{meta["object"]}',timeout=(20,150)) as response:
                response.raise_for_status();body=response.content
                assert len(body)==meta['bytes'] and hashlib.sha256(body).hexdigest()==meta['sha256'];return body.decode()
        for row in csv.DictReader(io.StringIO(pub.call(get,'verify_history_prefix'))):
            if int(row['unit'])==unit:return row['chain_sha256']
    return None

def seed_cache_links(manifest,target,cache):
    for name,meta in manifest['files'].items():
        if not name.startswith('inputs/cache/'):continue
        original=Path(cache)/name[len('inputs/cache/'):];dest=Path(target)/name
        if original.exists() and file_meta(original)=={k:meta[k] for k in ('bytes','sha256')} and not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True);os.link(original,dest)

def restore(pub,target,protocol,cache=None):
    target=Path(target);rev=pub.head();manifest=pub.get_json('MANIFEST.json',rev)
    if not manifest:return None
    assert manifest['contract']==protocol
    if 'cache/CACHE.json' in manifest['files'] and (target/'cache/CACHE.json').exists():
        remote=pub.get_json(manifest['files']['cache/CACHE.json']['object'],rev);local=read(target/'cache/CACHE.json')
        assert remote['identity']==local['identity']
        if len(local['chunks'])>=len(remote['chunks']):
            assert all(local['files'].get(n)==m and file_meta(target/'cache'/n)==m for n,m in remote['files'].items()),'Divergent preparation prefix'
            return rev
        assert all(remote['files'].get(n)==m for n,m in local['files'].items()),'Divergent preparation prefix'
    pointer_meta=manifest['files'].get('POINTER.json')
    if pointer_meta:
        pointer=pub.get_json(pointer_meta['object'],rev);local=read(target/'POINTER.json') if (target/'POINTER.json').exists() else None
        if local:
            assert local['protocol']==protocol
            if local['unit']>=pointer['unit']:
                assert chain(target,pointer['unit'])==pointer['chain'],'Local history does not extend remote history'
                atomic(target/'RESTORE_CONTEXT.json',dict(revision=rev,remote_unit=pointer['unit'],action='retain_compatible_local_ahead_state'));return rev
            assert remote_chain(pub,rev,manifest,local['unit'])==local['chain'],'Remote history does not extend local history'
    if cache:seed_cache_links(manifest,target,cache)
    visible('restore verified direct HF files',lambda:pub.download(rev,target,protocol))
    if pointer_meta:atomic(target/'RESTORE_CONTEXT.json',dict(revision=rev,remote_unit=pointer['unit'],action='verified_HF_download_then_native_restore'))
    return rev

def service_rows(base,fit,publication=None,snapshot=None):
    base=Path(base);fit=Path(fit);cursor_path=base/'SERVICE_CURSOR.json';offsets=read(cursor_path) if cursor_path.exists() else {'session':uuid.uuid4().hex};records={}
    for filename in ['transport/REQUESTS.jsonl','transport/HTTP_REQUESTS.jsonl','RATE_EVENTS.jsonl']:
        path=base/filename
        if not path.exists():continue
        lines=path.read_text(encoding='utf-8').splitlines();start=offsets.get(filename,0)
        for i,line in enumerate(lines[start:],start):
            try:item=json.loads(line)
            except json.JSONDecodeError:break
            remaining=re.search(r'\br=(-?\d+)',item.get('rate_limit') or '');reset=re.search(r'\bt=(\d+)',item.get('rate_limit') or '')
            records.setdefault('transport_events',[]).append(dict(stage='transport',session=offsets['session'],event_id=offsets['session']+'-'+digest(filename)[:8]+'-'+str(i),category=item.get('category',item.get('operation',item.get('kind','event'))),
                status=item.get('status') if isinstance(item.get('status'),int) else item.get('http_status'),attempt=item.get('attempt'),bytes=item.get('bytes'),retry_after_seconds=item.get('retry_delay',item.get('seconds') if item.get('kind')=='quota_wait' else None),duration_seconds=item.get('seconds') if item.get('kind') not in ('quota_wait','server_policy') else None,quota_remaining=int(remaining[1]) if remaining else None,reset_seconds=float(reset[1]) if reset else None))
            offsets[filename]=i+1
    if publication:
        records['transport_receipts']=[dict(stage='transport',snapshot=str(snapshot),revision=publication['revision'],owner_released=publication['owner_released'],verified=True,last_update=publication.get('update'),pending_updates=0,event_coverage_through=offsets['session'],unsent_events=0)]
    if snapshot:
        folder=Path(snapshot);m=read(folder/'MANIFEST.json');logical=sum(v['bytes'] for v in m['files'].values());extra=sum(v['bytes'] for n,v in m['files'].items() if (folder/n).stat().st_nlink==1)
        records['staging_inventory']=[dict(stage='staging',snapshot=folder.name,file=n,bytes=v['bytes'],sha256=v['sha256'],hardlink=(folder/n).stat().st_nlink>1,staging_bytes=extra,logical_total_bytes=logical,actual_free_bytes=shutil.disk_usage(base).free) for n,v in m['files'].items()]
    if records:
        queue_dir=fit/'service_queue';queue_dir.mkdir(exist_ok=True,parents=True);atomic(queue_dir/(digest(records)+'.json'),records)
    atomic(cursor_path,offsets)
    return records

def publish_ready(pub,output,base,release=False,force=False,service_fit=None):
    output=Path(output);path=output/'SNAPSHOT_READY.json'
    if not path.exists():return None
    item=read(path);snapshot=Path(item['path'])
    receipt=visible('publish immutable direct CSV/native snapshot',lambda:pub.publish(snapshot,release=release));receipt['update']=item['update']
    atomic(Path(base)/'LAST_HF_PUBLICATION.json',dict(receipt,unit=item['unit']))
    if service_fit:service_rows(base,service_fit,receipt,snapshot)
    path.unlink(missing_ok=True)
    # Only redundant staged hardlinks/copies, never the original run/cache.
    resolved=snapshot.resolve();allowed=(output/'snapshots').resolve()
    if resolved.is_relative_to(allowed):shutil.rmtree(resolved)
    return receipt

def add_to_snapshot(folder,items):
    folder=Path(folder);m=read(folder/'MANIFEST.json')
    for name,path in items.items():
        target=folder/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target);m['files'][name]=file_meta(target)
    atomic(folder/'MANIFEST.json',m)

def preparation_snapshot(output,cache,protocol,index,reason):
    output=Path(output);target=output/'snapshots'/str(time.time_ns());target.mkdir(parents=True)
    files={}
    for name,meta in index['files'].items():
        path=target/'cache'/name;path.parent.mkdir(parents=True,exist_ok=True);os.link(Path(cache)/name,path);files['cache/'+name]=meta
    path=target/'cache/CACHE.json';path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(Path(cache)/'CACHE.json',path);files['cache/CACHE.json']=file_meta(path)
    atomic(target/'SNAPSHOT_STATUS.json',dict(reason=reason,completed_chunks=len(index['chunks'])));files['SNAPSHOT_STATUS.json']=file_meta(target/'SNAPSHOT_STATUS.json')
    atomic(target/'MANIFEST.json',dict(contract=protocol,files=files));atomic(output/'SNAPSHOT_READY.json',dict(path=str(target),unit=len(index['chunks']),update=0,reason=reason))

def transport_tail(token,base,fit,protocol,scientific_revision):
    """Publish pending administrative rows after the final selecting commit."""
    pub=Publisher(NAMESPACE+'/transport-tail/'+protocol,token,base/'transport/REQUESTS.jsonl')
    existing,rev,_=selected(pub,'COVERAGE.json')
    if existing:return dict(status='verified_tail_already_published',revision=rev)
    tail=base/'transport_tail';tail.mkdir(exist_ok=True);service_rows(base,fit)
    combined={}
    for path in sorted((fit/'service_queue').glob('*.json')):
        for family,rows in read(path).items():combined.setdefault(family,[]).extend(rows)
    # A fresh completed-fit-only session may have no local pending event queue.
    # Its known selecting revision still gets an explicit bounded provenance row.
    if not combined.get('transport_receipts'):
        combined['transport_receipts']=[dict(stage='transport_tail',snapshot='final_scientific_snapshot',revision=scientific_revision,owner_released=True,verified=True,pending_updates=0,event_coverage_through='known_final_scientific_revision',unsent_events=0)]
    from cache import table
    files={}
    for family,rows in combined.items():
        unique=list({digest(row):row for row in rows}.values());path=table(tail,family,unique,protocol);files[path.relative_to(tail).as_posix()]=file_meta(path)
    atomic(tail/'COVERAGE.json',dict(scientific_revision=scientific_revision,coverage='pending administrative rows through final scientific selecting publication; this tail upload and later review requests are outside the finite boundary',pending_numeric_updates=0));files['COVERAGE.json']=file_meta(tail/'COVERAGE.json')
    atomic(tail/'MANIFEST.json',dict(contract=protocol,files=files));pub.claim()
    try:
        receipt=visible('publish finite final transport receipt',lambda:pub.publish(tail,release=True));atomic(base/'TRANSPORT_TAIL.json',receipt);return receipt
    except BaseException as exc:
        atomic(base/'TRANSPORT_TAIL_PENDING.json',dict(error_type=type(exc).__name__,scientific_revision=scientific_revision,last_verified=pub.last_verified));raise

def run(token,mode='all',base='/kaggle/working/tread_two_trace',data_root=''):
    assert mode in ('all','prepare','fit','review') and token
    base=Path(base);base.mkdir(parents=True,exist_ok=True)
    assert base.resolve().is_relative_to(Path('/kaggle/working')),'Use notebook coordinator in Kaggle; local validation uses injected offline services only'
    with FileLock(str(base/'CONTROLLER.lock'),timeout=1):return _run(token,mode,base,data_root)

def _run(token,mode,base,data_root):
    cfg=configuration();protocol=digest(cfg);cache=base/'preparation/cache';prep=base/'preparation';fit=base/'fit'
    probe_pub=Publisher(NAMESPACE+'/registry-read',token,base/'transport/REQUESTS.jsonl');registered=registry(probe_pub)
    if registered:
        assert registered['config']['sources']==cfg['sources'],'Use the original frozen TRACE notebook/source to resume or review'
        protocol=registered['protocol']
    fit_pub=Publisher(NAMESPACE+'/fit/'+protocol,token,base/'transport/REQUESTS.jsonl')
    completed,fit_rev,_=selected(fit_pub,'RESULT.json')
    if completed and completed.get('status')=='completed':
        transport_tail(token,base,fit,protocol,fit_rev)
        if mode=='fit':return dict(status='completed_remote_skip_no_updates',revision=fit_rev,protocol=protocol)
        if mode=='prepare':return dict(status='fit_already_complete_no_preparation_rerun',revision=fit_rev,protocol=protocol)
        return review_stage(token,base,fit_pub,fit_rev,protocol)
    if mode=='review':raise RuntimeError('TRACE fit is not complete. Run NB24 master; NB23 never starts a fit by itself.')
    if registered:assert registered['config']==cfg,'Incomplete experiment environment differs; do not start a second fit'
    assert shutil.disk_usage(base).free>=19_000_000_000 or (fit/'run/POINTER.json').exists() or (cache/'CACHE.json').exists(),'Fresh run needs at least19GB free; preserve existing files to resume'
    prep_pub=Publisher(NAMESPACE+'/preparation/'+protocol,token,base/'transport/REQUESTS.jsonl')
    gate_cfg=configuration(True);gate_protocol=digest(gate_cfg);gate_pub=Publisher(NAMESPACE+'/engineering/'+gate_protocol,token,base/'transport/REQUESTS.jsonl')
    active=[];old_term=signal.getsignal(signal.SIGTERM)
    def terminate(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,terminate)
    try:
        preparation,prep_rev,_=selected(prep_pub,'ENGINEERING_READY.json')
        # Restore cache prefix before reusing a certificate or reconstructing a
        # partial preparation; downloads verify every selected size/SHA.
        if not (cache/'CACHE.json').exists() or not read(cache/'CACHE.json').get('completed'):
            restore(prep_pub,prep,protocol)
        if not preparation:
            prep_pub.claim();active.append((prep_pub,prep,protocol))
            export=base/'geometry.pt'
            if not export.exists() or sha(export)!=GEOMETRY['sha256']:
                def download():
                    with prep_pub.session.get(GEOMETRY['url'],timeout=(20,150),stream=True) as response:
                        response.raise_for_status()
                        with export.with_suffix('.part').open('wb') as stream:
                            for block in response.iter_content(4*1024**2):stream.write(block)
                    export.with_suffix('.part').replace(export)
                visible('download small pinned geometry export only',lambda:prep_pub.call(download,'download_pinned_geometry'))
            assert sha(export)==GEOMETRY['sha256']
            last=time.monotonic()
            def prep_tick():
                nonlocal last
                if (prep/'SNAPSHOT_READY.json').exists():publish_ready(prep_pub,prep,base);last=time.monotonic()
                if time.monotonic()-last>=1800:(prep/'SNAPSHOT_REQUEST').touch()
            run_worker('prepare',prep,cache,data_root,export,tick=prep_tick);publish_ready(prep_pub,prep,base)
            gate_pub.claim();active.append((gate_pub,base/'gate_recovered',gate_protocol))
            direct=base/'gate_direct';recovered=base/'gate_recovered'
            run_worker('gate_direct',direct,cache,data_root)
            assert read(direct/'STORAGE_PROJECTION.json')['status']=='passed','Full evidence storage projection failed before measured fitting'
            seam=direct/'seam';receipt=visible('publish actual complete-update engineering seam',lambda:gate_pub.publish(seam))
            # Fresh destination must obtain checkpoint/optimizer/RNG from HF.
            m=gate_pub.get_json('MANIFEST.json',receipt['revision']);seed_cache_links(m,recovered/'run',cache)
            visible('download actual HF checkpoint for fresh-worker continuation',lambda:gate_pub.download(receipt['revision'],recovered/'run',gate_protocol))
            atomic(recovered/'run/RESTORE_CONTEXT.json',dict(revision=receipt['revision'],remote_unit=read(recovered/'run/POINTER.json')['unit'],action='actual_engineering_HF_download_then_continue'))
            run_worker('gate_restore',recovered,cache,data_root,reference=direct)
            proof=read(recovered/'RESTORE_PROOF.json');budget=read(recovered/'STORAGE_PROJECTION.json')
            assert proof['status']=='passed' and proof['dual_CUDA_RNG_states']==2 and budget['status']=='passed'
            preparation=dict(status='passed',protocol=protocol,gate_protocol=gate_protocol,sources=cfg['sources'],environment=cfg['environment'],
                cache_hash=read(cache/'CACHE.json')['scientific_hash'],actual_dual_T4=True,actual_HF_restore_continue=True,restore=proof,storage=budget)
            atomic(prep/'ENGINEERING_READY.json',preparation)
            ready=read(recovered/'SNAPSHOT_READY.json');add_to_snapshot(ready['path'],{'ENGINEERING_READY.json':prep/'ENGINEERING_READY.json','STORAGE_PROJECTION.json':recovered/'STORAGE_PROJECTION.json'})
            publish_ready(gate_pub,recovered,base,release=True);active=[x for x in active if x[0] is not gate_pub]
            preparation_snapshot(prep,cache,protocol,read(cache/'CACHE.json'),'engineering_passed')
            ready=read(prep/'SNAPSHOT_READY.json');add_to_snapshot(ready['path'],{'ENGINEERING_READY.json':prep/'ENGINEERING_READY.json'})
            receipt=publish_ready(prep_pub,prep,base,release=True);active=[x for x in active if x[0] is not prep_pub]
        else:
            assert preparation['protocol']==protocol and preparation['sources']==cfg['sources'] and preparation['environment']==cfg['environment']
            assert preparation['actual_HF_restore_continue'] and preparation['actual_dual_T4']
            from cache import Cache
            verified=Cache(cache);verified.verify();assert verified.index['scientific_hash']==preparation['cache_hash']
        if mode=='prepare':return dict(status='preparation_complete',protocol=protocol,actual_HF_restore_continue=True,next='NB24 master or NB22; no need to rerun NB21')
        registry(fit_pub,cfg);fit_pub.claim();active.append((fit_pub,fit,protocol));restore(fit_pub,fit/'run',protocol,cache)
        # Safe local-ahead work is published before new measured optimizer steps.
        run_worker('init',fit,cache,data_root);publish_ready(fit_pub,fit,base,service_fit=fit)
        last=time.monotonic();completed_receipt=None
        def fit_tick():
            nonlocal last,completed_receipt
            if (fit/'SNAPSHOT_READY.json').exists():
                finished=read(fit/'SNAPSHOT_READY.json')['reason']=='completed'
                value=publish_ready(fit_pub,fit,base,release=finished,service_fit=fit)
                if finished:completed_receipt=value
                last=time.monotonic()
            if time.monotonic()-last>=1800:(fit/'SNAPSHOT_REQUEST').touch()
        service_rows(base,fit)
        run_worker('train',fit,cache,data_root,tick=fit_tick)
        assert read(fit/'run/RESULT.json')['status']=='completed'
        receipt=completed_receipt or publish_ready(fit_pub,fit,base,release=True,service_fit=fit)
        assert receipt and receipt['owner_released'],'Final selecting publication has not been verified/released'
        active=[]
        # Bounded administrative receipt captures the final selecting publication;
        # it does not recursively claim to include its own upload event forever.
        transport_tail(token,base,fit,protocol,receipt['revision'])
        if mode=='fit':return dict(status='fit_complete',protocol=protocol,revision=receipt['revision'],selected=read(fit/'run/RESULT.json')['selected'])
        return review_stage(token,base,fit_pub,receipt['revision'],protocol)
    except BaseException as exc:
        for pub,output,contract in reversed(active):
            try:
                if not (output/'SNAPSHOT_READY.json').exists():
                    if (output/'run/POINTER.json').exists():
                        folder=output/'snapshots'/str(time.time_ns());visible('stage last complete state for stop/error flush',lambda:Store(output/'run',contract).snapshot(folder))
                        atomic(output/'SNAPSHOT_READY.json',dict(path=str(folder),unit=read(output/'run/POINTER.json')['unit'],update=None,reason='catchable_stop'))
                    elif (cache/'CACHE.json').exists():
                        preparation_snapshot(output,cache,contract,read(cache/'CACHE.json'),'catchable_stop')
                publish_ready(pub,output,base,release=True)
            except BaseException as failure:atomic(base/'HF_PENDING.json',dict(error_type=type(failure).__name__,stop_type=type(exc).__name__,last_verified=pub.last_verified))
        raise
    finally:signal.signal(signal.SIGTERM,old_term)

def review_stage(token,base,fit_pub,fit_revision,protocol):
    from review import review_contract
    contract=review_contract(protocol);review_protocol=digest(contract);pub=Publisher(NAMESPACE+'/review/'+review_protocol,token,base/'transport/REQUESTS.jsonl')
    completed,revision,_=selected(pub,'RESULT.json')
    if completed and completed.get('status')=='completed':return dict(status='all_complete_remote_skip',fit_revision=fit_revision,review_revision=revision,scientific_status='research_only_unpromoted')
    fit=base/'fit/run';restore(fit_pub,fit,protocol,base/'preparation/cache')
    pub.claim();out=base/'review';out.mkdir(exist_ok=True);restore(pub,out,review_protocol)
    try:
        # Review is finite CPU saved-evidence analysis in a secret-free worker.
        env=dict(os.environ);env.pop('HF_TOKEN',None)
        args=[sys.executable,'-u','-B',str(HERE/'review.py'),'--input',str(fit),'--output',str(out),'--fit-protocol',protocol,'--fit-revision',fit_revision]
        with (out/'WORKER.log').open('w',encoding='utf-8') as log:
            proc=subprocess.Popen(args,env=env,cwd=HERE,stdout=log,stderr=subprocess.STDOUT,start_new_session=os.name!='nt',creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            last=0;last_push=time.monotonic();last_manifest=None
            while proc.poll() is None:
                if time.monotonic()-last>25:print('TRACE review alive: saved CSV/native audit; keep the cache.',flush=True);last=time.monotonic()
                if (out/'MANIFEST.json').exists():
                    current=sha(out/'MANIFEST.json')
                    if current!=last_manifest or time.monotonic()-last_push>=1800:
                        visible('publish completed review stages',lambda:pub.publish(out));last_manifest=current;last_push=time.monotonic()
                time.sleep(.5)
        assert proc.returncode==0,'Read review/WORKER.log; no fit is repeated'
        receipt=visible('publish full review and research passport',lambda:pub.publish(out,release=True))
        result=dict(status='all_complete',fit_revision=fit_revision,review_revision=receipt['revision'],scientific_status='research_only_unpromoted')
        atomic(base/'BATCH_RESULT.json',result);return result
    except BaseException as failure:
        if 'proc' in locals() and proc.poll() is None:proc.terminate();proc.wait(timeout=15)
        if not (out/'MANIFEST.json').exists():atomic(out/'MANIFEST.json',dict(contract=review_protocol,files={}))
        if (out/'MANIFEST.json').exists():
            try:
                error_dir=out/'errors';error_dir.mkdir(exist_ok=True);stamp=str(time.time_ns());files={}
                if (out/'WORKER.log').exists():
                    log=error_dir/(stamp+'.log');shutil.copyfile(out/'WORKER.log',log);files[log.relative_to(out).as_posix()]=file_meta(log)
                report=error_dir/(stamp+'.json');atomic(report,dict(error_type=type(failure).__name__,scope='finite_review_failed_no_new_fit',completed_stage_receipts=[p.name for p in out.glob('*_RECEIPT.json')]))
                files[report.relative_to(out).as_posix()]=file_meta(report)
                er=out/('ERROR_'+stamp+'_RECEIPT.json');atomic(er,dict(contract=review_protocol,files=files))
                m=read(out/'MANIFEST.json');m['files'].update(files);m['files'][er.name]=file_meta(er);atomic(out/'MANIFEST.json',m)
                pub.publish(out,release=True)
            except BaseException as exc:atomic(base/'REVIEW_HF_PENDING.json',dict(error_type=type(exc).__name__,last_verified=pub.last_verified))
        raise
