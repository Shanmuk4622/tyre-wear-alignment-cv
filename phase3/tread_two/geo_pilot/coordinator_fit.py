"""One Kaggle owner: actual new-storage gate, full HF restore and visible periodic publication."""
import csv,io,json,os,queue,re,shutil,signal,subprocess,sys,threading,time,uuid
from pathlib import Path
from filelock import FileLock
from common import HERE,atomic,digest,file_meta,read,sha
from engine_fit import configuration
from publisher import Publisher
from store_fit import Store


def visible(label,fn):
    stopped=threading.Event()
    def announce():
        while not stopped.wait(30):print('NB17 alive:',label,'— verified local progress is retained.',flush=True)
    worker=threading.Thread(target=announce,daemon=True);worker.start()
    try:return fn()
    finally:stopped.set();worker.join(timeout=1)


def selected_json(pub,name,revision=None):
    revision=revision or pub.head();manifest=pub.get_json('MANIFEST.json',revision)
    if not manifest or name not in manifest['files']:return None,revision,manifest
    value=pub.get_json(manifest['files'][name]['object'],revision)
    return value,revision,manifest


def pinned_json(pub,pin,name):
    """Resolve an immutable logical file through its selecting manifest."""
    import hashlib
    base=f'https://huggingface.co/datasets/{pub.REPO}/resolve/{pin["revision"]}/{pin["prefix"]}/'
    def fetch():
        with pub.session.get(base+'MANIFEST.json',timeout=(20,150)) as response:response.raise_for_status();manifest=response.json()
        assert manifest['contract']==pin['contract']
        meta=manifest['files'][name];assert '..' not in Path(meta['object']).parts
        with pub.session.get(base+meta['object'],timeout=(20,150)) as response:response.raise_for_status();body=response.content
        assert len(body)==meta['bytes'] and hashlib.sha256(body).hexdigest()==meta['sha256']
        return json.loads(body)
    return pub.call(fetch,'verify_pinned_'+name)


def run_worker(mode,output,weights,data_root='',reference=None,tick=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    args=[sys.executable,'-B',str(HERE/'worker_fit.py'),mode,'--output',str(output),'--weights',str(weights),'--data-root',data_root]
    if reference:args+=['--reference',str(reference)]
    env=dict(os.environ);env.pop('HF_TOKEN',None)
    proc=subprocess.Popen(args,cwd=HERE,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=os.name!='nt',creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name=='nt' else 0)
    messages=queue.Queue()
    def pump():
        for line in proc.stdout:messages.put(line)
    thread=threading.Thread(target=pump,daemon=True);thread.start();last=0.
    try:
        while proc.poll() is None or thread.is_alive() or not messages.empty():
            try:print(messages.get(timeout=.5),end='',flush=True)
            except queue.Empty:pass
            if time.monotonic()-last>=30:
                status=read(output/'STATUS.json') if (output/'STATUS.json').exists() else dict(phase='source verification / restore')
                print('NB17 alive:',mode,json.dumps(status),flush=True);last=time.monotonic()
                if tick:tick()
    except BaseException:
        (output/'STOP_REQUESTED').write_text('catchable_coordinator_stop\n',encoding='utf-8')
        visible('finishing current complete worker unit',lambda:proc.wait())
        raise
    assert proc.returncode==0,'Worker stopped; preserve output and WORKER_ERROR.json. No automatic numeric retry.'


def augment(snapshot,items):
    snapshot=Path(snapshot);manifest=read(snapshot/'MANIFEST.json')
    for name,path in items.items():
        if Path(path).exists():
            shutil.copyfile(path,snapshot/name);manifest['files'][name]=file_meta(snapshot/name)
    atomic(snapshot/'MANIFEST.json',manifest)


def current_snapshot(output,target,protocol,allow_copy=False):
    store=Store(Path(output)/'run',protocol);store.snapshot(target,allow_copy=allow_copy)
    return Path(target)


def service_rows(base,fit,publication=None):
    """Idempotent safe-event batches; worker commits these typed CSV rows atomically."""
    cursor_path=base/'SERVICE_CURSOR.json';offsets=read(cursor_path) if cursor_path.exists() else {};records={}
    if 'session_id' not in offsets:offsets['session_id']=uuid.uuid4().hex;atomic(cursor_path,offsets)
    for relative in ('transport/REQUESTS.jsonl','transport/HTTP_REQUESTS.jsonl','RATE_EVENTS.jsonl'):
        path=base/relative
        if not path.exists():continue
        lines=path.read_text(encoding='utf-8').splitlines();start=offsets.get(relative,0);done=start
        for line in lines[start:]:
            try:value=json.loads(line)
            except json.JSONDecodeError:break
            rate=re.search(r'\br=(-?\d+)',value.get('rate_limit') or '')
            reset=re.search(r'\bt=(\d+)',value.get('rate_limit') or '')
            quota=re.search(r'\bq=(\d+)',value.get('rate_policy') or '')
            window=re.search(r'\bw=(\d+)',value.get('rate_policy') or '')
            event_id=offsets['session_id']+'-'+digest(relative)[:12]+'-'+str(done)
            records.setdefault('HF_requests',[]).append(dict(event_id=event_id,category=value.get('category',value.get('operation',value.get('kind','event'))),method=value.get('method'),
                http_status=value.get('http_status') or (value.get('status') if isinstance(value.get('status'),int) else None),retry_delay_seconds=value.get('retry_delay',value.get('seconds') if value.get('kind')=='quota_wait' else None),
                rate_remaining=int(rate[1]) if rate else None,reset_seconds=int(reset[1]) if reset else None,bytes_transferred=value.get('bytes'),outcome=str(value.get('status',value.get('kind','event'))),
                server_quota=int(quota[1]) if quota else None,server_window_seconds=int(window[1]) if window else None,time_utc=value.get('time_utc')))
            done+=1
        offsets[relative]=done
    if publication:publication=dict(publication,attempt=digest(publication)[:32]);records['HF_publications']=[publication]
    if records:
        folder=fit/'service_queue';folder.mkdir(exist_ok=True)
        name=digest(records);path=folder/(name+'.json')
        if not path.exists() and not path.with_suffix('.done').exists():atomic(path,records)
    atomic(cursor_path,offsets)


def ledger_chain(root,unit):
    paths=list((Path(root)/'csv/unit_ledger').glob('*.csv'))+list((Path(root)/'active').glob('csv_unit_ledger*.csv'))
    for path in paths:
        with path.open(encoding='utf-8',newline='') as stream:
            for row in csv.DictReader(stream):
                if int(row['unit'])==unit:return row['current_unit_sha256']
    return None


def remote_chain(pub,revision,manifest,unit):
    for name,meta in manifest['files'].items():
        if name.startswith('csv/unit_ledger/') or name.startswith('active/csv_unit_ledger'):
            def fetch():
                with pub.session.get(f'https://huggingface.co/datasets/{pub.REPO}/resolve/{revision}/{pub.destination}/{meta["object"]}',timeout=(20,150)) as response:
                    response.raise_for_status()
                    import hashlib
                    assert len(response.content)==meta['bytes'] and hashlib.sha256(response.content).hexdigest()==meta['sha256']
                    return response.content.decode('utf-8')
            body=pub.call(fetch,'compare_remote_history')
            for row in csv.DictReader(io.StringIO(body)):
                if int(row['unit'])==unit:return row['current_unit_sha256']
    return None


def bind_single_experiment(pub,cfg,protocol):
    """A stable r1 registry prevents a changed source/environment silently refitting."""
    from huggingface_hub import CommitOperationAdd
    from common import canonical
    name='phase3/tread-two/geo-r1/EXPERIMENT.json'
    expected=dict(protocol=protocol,source_inventory_sha256=digest(cfg['sources']),
                  dataset_checksum=cfg['scientific_plan']['data']['checksum_file_sha256'],anchor_sha256=cfg['scientific_plan']['anchor']['weights_sha256'],
                  mode='one_measured_candidate',seed=1,epochs=60)
    for attempt in range(6):
        parent=pub.head()
        def existing():
            with pub.session.get(f'https://huggingface.co/datasets/{pub.REPO}/resolve/{parent}/{name}',timeout=(20,120)) as response:
                if response.status_code==404:return None
                response.raise_for_status();return response.json()
        actual=pub.call(existing,'resolve_single_experiment_registry')
        if actual is not None:
            assert actual==expected,'A different GEO r1 source/environment is already registered. Preserve/resume it; no automatic second fit.'
            return
        try:
            pub.call(lambda:pub.api.create_commit(pub.REPO,repo_type='dataset',parent_commit=parent,operations=[CommitOperationAdd(path_in_repo=name,path_or_fileobj=canonical(expected))],commit_message='Bind one GEO r1 measured experiment; prevent accidental refit'),'bind_single_experiment')
        except Exception as exc:
            if getattr(getattr(exc,'response',None),'status_code',None) in (409,412):continue
            raise
    raise RuntimeError('Unable to bind the single measured experiment atomically')


def run(token,base='/kaggle/working/tread_two_geo_nb17',data_root=''):
    base=Path(base);base.mkdir(parents=True,exist_ok=True)
    assert base.resolve().is_relative_to(Path('/kaggle/working')),'Use this coordinator in Kaggle'
    assert token,'Enable authorized Kaggle Secret HF_TOKEN'
    with FileLock(str(base/'COORDINATOR.lock'),timeout=1):return _run(token,base,data_root)


def _run(token,base,data_root):
    fit_cfg=configuration();protocol=digest(fit_cfg);fit=base/'fit';fit.mkdir(exist_ok=True)
    fit_pub=Publisher('phase3/tread-two/geo-r1/fit/'+protocol,token,base/'transport/REQUESTS.jsonl')
    gate_cfg=configuration(fixture=True);gate_protocol=digest(gate_cfg)
    gate_pub=Publisher('phase3/tread-two/geo-r1/pilot-gate/'+gate_protocol,token,base/'transport/REQUESTS.jsonl')
    gate_input=read(HERE/'NB16_GATE.json')
    assert fit_cfg['environment']==gate_input['environment'],'Kaggle numerical environment differs from verified NB16. Preserve any prior fit; no silent cross-release restart.'
    previous=pinned_json(fit_pub,gate_input,'FINAL_GATE.json')
    assert previous['actual_dual_T4'] and previous['actual_HF_restore_continue'] and previous['protocol']==gate_input['contract']
    completed,remote_revision,remote_manifest=selected_json(fit_pub,'RESULT.json')
    if completed and completed['status']=='completed':
        result=dict(status='completed_remote_no_optimizer_steps',protocol=protocol,revision=remote_revision,destination=fit_pub.destination,selected=completed['selected'])
        atomic(base/'RUN_RESULT.json',result);print(json.dumps(result,indent=2),flush=True);return result
    assert shutil.disk_usage(base).free>=19000000000 or (fit/'run/POINTER.json').exists(),'Start a fresh NB17 session with at least19GB free, or preserve/resume its existing files'
    weights=base/'anchor.pt';spec=fit_cfg['scientific_plan']['anchor']
    if not weights.exists() or sha(weights)!=spec['weights_sha256']:
        def download_anchor():
            with fit_pub.session.get(spec['url'],timeout=(20,150),stream=True) as response:
                response.raise_for_status()
                with weights.with_suffix('.part').open('wb') as stream:
                    for block in response.iter_content(1024**2):stream.write(block)
            weights.with_suffix('.part').replace(weights)
        visible('downloading pinned anchor only; images remain on Kaggle',lambda:fit_pub.call(download_anchor,'download_anchor'))
    assert sha(weights)==spec['weights_sha256']
    old_term=signal.getsignal(signal.SIGTERM)
    def terminate(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,terminate)
    claimed_fit=False;claimed_gate=False;last_published_unit=-1;last_push=time.monotonic()
    def publish_ready(force=False,release=False):
        nonlocal last_published_unit,last_push
        ready_path=fit/'SNAPSHOT_READY.json'
        if not ready_path.exists():return None
        ready=read(ready_path)
        if not force and ready['unit']<=last_published_unit:return None
        snapshot=Path(ready['path'])
        began=time.monotonic()
        receipt=visible('HF direct CSV/native publication',lambda:fit_pub.publish(snapshot,release=release))
        elapsed=time.monotonic()-began
        last_published_unit=ready['unit'];last_push=time.monotonic();atomic(base/'LAST_HF_PUBLICATION.json',dict(receipt,unit=ready['unit'],update=ready['update']))
        row=dict(attempt=ready['unit'],reason=ready['reason'],local_update=ready['update'],remote_update=ready['update'],revision=receipt['revision'],manifest_sha256=digest(read(snapshot/'MANIFEST.json')),
                 new_objects=receipt['uploaded_objects'],verified_objects=receipt['files']-1,elapsed_seconds=elapsed,owner_released=release,pending=False)
        service_rows(base,fit,row)
        # Only redundant, verified publication staging is removed; original run
        # evidence/native states are retained. Resolve/check the exact workspace.
        for prior in (fit/'snapshots').glob('*'):
            if prior.is_dir() and prior.resolve()!=snapshot.resolve():
                assert prior.resolve().is_relative_to((fit/'snapshots').resolve());shutil.rmtree(prior)
        return receipt
    def tick():
        publish_ready()
        if time.monotonic()-last_push>=1800:(fit/'SNAPSHOT_REQUEST').touch()
    try:
        live_gate,gate_revision,_=selected_json(gate_pub,'LIVE_GATE.json')
        if live_gate:
            assert live_gate['status']=='passed' and live_gate['sources']==fit_cfg['sources'] and live_gate['environment']==fit_cfg['environment']
            print('Reusing the matching verified live storage/restore gate at',gate_revision,flush=True)
        else:
            gate_pub.claim();claimed_gate=True;direct=base/'gate_direct';recovered=base/'gate_recovered'
            for output in (direct,recovered):(output/'STOP_REQUESTED').unlink(missing_ok=True)
            run_worker('gate_direct',direct,weights,data_root)
            seam=direct/'seam';assert read(seam/'MANIFEST.json')['contract']==gate_protocol
            gate_receipt=visible('publish actual new-store within-epoch gate seam',lambda:gate_pub.publish(seam))
            visible('download the actual gate snapshot',lambda:gate_pub.download(gate_receipt['revision'],recovered/'run',gate_protocol))
            run_worker('gate_restore',recovered,weights,data_root,reference=direct)
            proof=read(recovered/'RESTORE_PROOF.json');budget=read(recovered/'STORAGE_PROJECTION.json')
            assert proof['status']=='passed' and budget['status']=='passed'
            live_gate=dict(status='passed',protocol=gate_protocol,sources=fit_cfg['sources'],environment=fit_cfg['environment'],actual_HF_download_restore_continue=True,
                           actual_dual_T4=True,restore=proof,storage=budget,measured_updates=0)
            atomic(base/'LIVE_GATE.json',live_gate)
            gate_final=current_snapshot(recovered,base/'gate_final_snapshot',gate_protocol)
            augment(gate_final,{'LIVE_GATE.json':base/'LIVE_GATE.json','STORAGE_PROJECTION.json':recovered/'STORAGE_PROJECTION.json'})
            gate_receipt=visible('publish verified gate certificate',lambda:gate_pub.publish(gate_final,release=True));claimed_gate=False
            atomic(base/'LIVE_GATE_PUBLICATION.json',gate_receipt)
        bind_single_experiment(fit_pub,fit_cfg,protocol)
        fit_pub.claim();claimed_fit=True
        remote_manifest=fit_pub.get_json('MANIFEST.json',fit_pub.head())
        local_path=fit/'run/POINTER.json'
        if remote_manifest:
            revision=fit_pub.head();remote_manifest=fit_pub.get_json('MANIFEST.json',revision)
            assert remote_manifest['contract']==protocol
            pointer=fit_pub.get_json(remote_manifest['files']['POINTER.json']['object'],revision)
            if local_path.exists():
                local=read(local_path);assert local['protocol']==protocol
                if local['unit']>=pointer['unit']:
                    assert ledger_chain(fit/'run',pointer['unit'])==pointer['chain'],'Local history does not extend verified remote history'
                else:
                    assert visible('verify shared recovery history',lambda:remote_chain(fit_pub,revision,remote_manifest,local['unit']))==local['chain'],'Divergent recovery histories'
                    visible('restore full compatible HF state directly into working run',lambda:fit_pub.download(revision,fit/'run',protocol))
            else:visible('restore full compatible HF state directly into working run',lambda:fit_pub.download(revision,fit/'run',protocol))
        (fit/'STOP_REQUESTED').unlink(missing_ok=True)
        # Initializer restores/validates all native/CSV prefixes and publishes
        # local-ahead state before any further measured optimizer update.
        run_worker('init',fit,weights,data_root)
        publish_ready(force=True)
        service_rows(base,fit)
        run_worker('train',fit,weights,data_root,tick=tick)
        assert read(fit/'STATUS.json')['phase']=='completed' and read(fit/'run/RESULT.json')['status']=='completed','Worker stopped at a safe boundary; measured run is not complete'
        receipt=publish_ready(force=True,release=True);claimed_fit=False
        result=dict(status='completed',protocol=protocol,revision=receipt['revision'],destination=fit_pub.destination,
                    publication=receipt,live_gate=live_gate,measured_single_candidate=True)
        atomic(base/'RUN_RESULT.json',result);print(json.dumps({k:v for k,v in result.items() if k!='live_gate'},indent=2),flush=True);return result
    except BaseException as error:
        if claimed_fit and (fit/'run/POINTER.json').exists():
            try:
                snap=current_snapshot(fit,fit/'snapshots'/str(time.time_ns()),protocol)
                pointer=read(fit/'run/POINTER.json');atomic(fit/'SNAPSHOT_READY.json',dict(path=str(snap),reason='catchable_stop_or_error',unit=pointer['unit'],update=None,protocol=protocol))
                publish_ready(force=True,release=True);claimed_fit=False
            except BaseException as failure:atomic(base/'HF_PENDING.json',dict(error_type=type(failure).__name__,stop_type=type(error).__name__,last_verified=fit_pub.last_verified))
        if claimed_gate:
            try:
                candidate=base/'gate_recovered' if (base/'gate_recovered/run/POINTER.json').exists() else base/'gate_direct'
                if (candidate/'run/POINTER.json').exists():
                    # The finite disposable gate may copy a small snapshot even
                    # if its hardlink check failed. Measured staging never does.
                    seam=current_snapshot(candidate,base/'gate_stop_snapshot',gate_protocol,allow_copy=True)
                    gate_pub.publish(seam,release=True)
            except BaseException as failure:atomic(base/'GATE_HF_PENDING.json',dict(error_type=type(failure).__name__,last_verified=gate_pub.last_verified))
        raise
    finally:signal.signal(signal.SIGTERM,old_term)
