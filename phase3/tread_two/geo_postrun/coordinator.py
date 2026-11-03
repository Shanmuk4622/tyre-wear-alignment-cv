"""One owner, one verified cache, automatic stage handoff and catchable-stop flush."""
import hashlib,os,shutil,signal,subprocess,sys,threading,time,uuid
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
from filelock import FileLock
from common import HERE,atomic,canonical,digest,file_meta,read,sha,environment
from inputs import PIN,locate
from publisher import Publisher
from rate_budget import BudgetSession

def contract():
    sources=read(HERE/'FILES.json')
    return dict(version='GEO-postrun-1',input=PIN,sources=sources,stages=['review','passport'],
                scope='saved_data_CPU_no_forward_no_fit_no_checkpoint_reselection',float64_tolerance_policy='module-specific CPU-vs-CUDA tolerances; original scientific gates unchanged')

def selected_json(pub,name,revision):
    manifest=pub.get_json('MANIFEST.json',revision)
    if not manifest or name not in manifest['files']:return None,manifest
    return pub.get_json(manifest['files'][name]['object'],revision),manifest

def verified_prefix(folder,protocol):
    folder=Path(folder);count=0
    for name in ['files','core','losses','native','masked','dense','visuals']:
        path=folder/'stages'/(name+'.json')
        if not path.exists():break
        value=read(path);assert value['contract']==protocol,'Incompatible completed analysis stage'
        if not all((folder/n).exists() and file_meta(folder/n)==meta for n,meta in value['files'].items()):break
        count+=1
    return count

def snapshot(output,target,cfg,stage):
    """Only outputs bound by completed immutable stage receipts enter snapshots."""
    output=Path(output);target=Path(target);target.mkdir(parents=True,exist_ok=True)
    selected={};completed=False
    for receipt in sorted((output/'stages').glob('*.json')) if (output/'stages').exists() else []:
        data=read(receipt);assert data['contract']==digest(cfg)
        for name,meta in data['files'].items():
            assert file_meta(output/name)==meta
            selected[name]=meta
        selected[receipt.relative_to(output).as_posix()]=file_meta(receipt)
    # Passport is one finite operation: bind all finished outputs atomically by result.
    if stage=='passport':
        from passport import valid
        completed=valid(output,digest(cfg))
    if stage=='passport' and completed:
        for p in output.rglob('*'):
            if p.is_file() and not p.name.endswith(('.tmp','.partial.csv')):selected[p.relative_to(output).as_posix()]=file_meta(p)
    if stage=='review' and (output/'RESULT.json').exists():
        required={'files','core','losses','native','masked','dense','visuals'}
        completed=required.issubset({p.stem for p in (output/'stages').glob('*.json')})
        value=read(output/'RESULT.json')
        completed=completed and value.get('input')==PIN and value.get('findings')==read(output/'FINDINGS.json')
    for name in ['STATUS.json','ERROR.json','ENVIRONMENT.json','TABLES.json']+(['RESULT.json'] if completed else []):
        if (output/name).exists():
            # Atomic JSON bodies are copied below, never linked while mutable.
            selected[name]=None
    for name,meta in selected.items():
        source=output/name;dest=target/name;dest.parent.mkdir(parents=True,exist_ok=True)
        if name=='TABLES.json':atomic(dest,{k:v for k,v in read(source).items() if v['file'] in selected})
        elif meta is None:dest.write_bytes(source.read_bytes())
        else:
            try:os.link(source,dest)
            except OSError:shutil.copyfile(source,dest)
            assert file_meta(dest)==meta
    atomic(target/'CONTRACT.json',cfg)
    atomic(target/'STAGE.json',dict(stage=stage,protocol=digest(cfg),completed=completed))
    (target/'source').mkdir(exist_ok=True)
    for name in list(cfg['sources'])+['FILES.json']:shutil.copyfile(HERE/name,target/'source'/name)
    # Safe request categories/status only; vendored transport never logs tokens or URLs.
    import csv,json
    events=[]
    for path in sorted((output.parent/'transport').glob('*.jsonl')):
        for line in path.read_text(encoding='utf-8').splitlines():
            try:row=json.loads(line)
            except json.JSONDecodeError:continue # a concurrent trailing append is retried next publication
            events.append(dict(log_source=path.name,**{k:json.dumps(v,sort_keys=True) if isinstance(v,(list,dict)) else v for k,v in row.items()}))
    if events:
        fields=sorted(set().union(*(r.keys() for r in events)))
        with (target/'HF_transport_events.csv').open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(events)
    inventory={p.relative_to(target).as_posix():file_meta(p) for p in target.rglob('*') if p.is_file()}
    atomic(target/'MANIFEST.json',dict(contract=digest(cfg),files=inventory));return target

def download_input(pub,base,tick):
    reader=Publisher(PIN['prefix'],pub.session.headers['Authorization'].removeprefix('Bearer '),base/'transport/INPUT_HF.jsonl')
    manifest=reader.get_json('MANIFEST.json',PIN['revision']);assert manifest['contract']==PIN['protocol'] and len(manifest['files'])==PIN['selected_objects']
    root=base/'input';root.mkdir(exist_ok=True)
    atomic(root/'REMOTE_MANIFEST.json',manifest)
    reuse=Path('/kaggle/working/tread_two_geo_nb17/fit/run')
    # Reuse only exact immutable bytes; mutable rolling state is copied, not linked.
    for name,meta in manifest['files'].items():
        target=root/name;prior=reuse/name
        if not target.exists() and prior.is_file() and file_meta(prior)=={k:meta[k] for k in ('bytes','sha256')}:
            target.parent.mkdir(parents=True,exist_ok=True)
            if name.startswith(('native/','csv/','epochs/','source/','exports/','controls/')):
                try:os.link(prior,target)
                except OSError:shutil.copyfile(prior,target)
            else:shutil.copyfile(prior,target)
    missing=sum(m['bytes'] for n,m in manifest['files'].items() if not (root/n).exists())
    assert shutil.disk_usage(base).free>=missing+2_500_000_000,'Need space for the remaining ~10.31 GB input plus 2.5 GB reserve. Keep compatible cache; free unrelated output.'
    tls=threading.local();done=0;bytes_done=0;cancel=threading.Event()
    def fetch(item):
        if cancel.is_set():raise InterruptedError('Download stopped; verified cache retained')
        name,meta=item;assert not Path(name).is_absolute() and '..' not in Path(name).parts and '..' not in Path(meta['object']).parts
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and file_meta(target)=={k:meta[k] for k in ('bytes','sha256')}:return meta['bytes']
        if not hasattr(tls,'session'):
            tls.session=BudgetSession(reader.session.budget,base/'transport/INPUT_HTTP.jsonl');tls.session.headers.update(reader.session.headers)
        def transfer():
            part=target.with_name(target.name+'.part');h=hashlib.sha256();size=0
            with tls.session.get(f'https://huggingface.co/datasets/{reader.REPO}/resolve/{PIN["revision"]}/{PIN["prefix"]}/{meta["object"]}',timeout=(20,150),stream=True) as response:
                response.raise_for_status()
                with part.open('wb') as stream:
                    for block in response.iter_content(4*1024**2):
                        if cancel.is_set():raise InterruptedError('Download stopped; verified cache retained')
                        stream.write(block);h.update(block);size+=len(block)
            assert size==meta['bytes'] and h.hexdigest()==meta['sha256'],name
            os.replace(part,target)
        reader.call(transfer,'input:'+name);return meta['bytes']
    pool=ThreadPoolExecutor(max_workers=8);futures={pool.submit(fetch,item):item for item in manifest['files'].items()};pending=set(futures);notice=0.
    try:
        while pending:
            finished,pending=wait(pending,timeout=5,return_when=FIRST_COMPLETED)
            for future in finished:bytes_done+=future.result();done+=1
            if time.monotonic()-notice>=30 or not pending:
                atomic(base/'review/STATUS.json',dict(stage='input_download',files=done,total=len(futures),bytes=bytes_done,status='running'))
                print('Verified input',done,'/',len(futures),'files;',round(bytes_done/1e9,3),'GB; bounded retries/cache active.',flush=True);notice=time.monotonic()
            tick()
    except BaseException:
        cancel.set()
        for future in futures:future.cancel()
        raise
    finally:pool.shutdown(wait=False,cancel_futures=True)
    return root,manifest

def run(token,mode='all',base='/kaggle/working/tread_two_geo_postrun',dataset=''):
    assert mode in ('all','review','passport');base=Path(base);base.mkdir(parents=True,exist_ok=True)
    assert token,'Enable Kaggle Secret HF_TOKEN'
    data=locate(dataset) # exact existing masks are required by the full review
    cfg=contract();protocol=digest(cfg)
    if (base/'IDENTITY.json').exists():assert read(base/'IDENTITY.json')['protocol']==protocol,'Postrun source changed; keep the old cache and use a separate output folder'
    atomic(base/'CONTRACT.json',cfg)
    atomic(base/'IDENTITY.json',dict(protocol=protocol))
    root=base/'review';root.mkdir(exist_ok=True);atomic(root/'ENVIRONMENT.json',environment())
    old=signal.getsignal(signal.SIGTERM)
    def stop(sig,frame):raise KeyboardInterrupt('Catchable Kaggle stop')
    signal.signal(signal.SIGTERM,stop)
    receipt={};claimed=False;pub=None;last=0.;current='review';process=None
    lock=FileLock(str(base/'NOTEBOOK.lock'),timeout=0)
    try:
        with lock:
            pub=Publisher('phase3/tread-two/geo-r1/review/'+protocol,token,base/'transport/REVIEW_HF.jsonl')
            revision=pub.head();remote,manifest=selected_json(pub,'RESULT.json',revision)
            if remote and remote.get('status')=='completed':
                assert remote['input']==PIN
                assert pub.get_json('OWNER.json',revision)['owner'] is None
                print('Completed matching NB18 already published; restore and skip computation.',flush=True)
                pub.download(revision,root,protocol)
                receipt['review']=dict(status='restored_verified_completed',revision=revision,destination=pub.destination,owner_released=True,selected_objects=len(manifest['files']),bytes=sum(v['bytes'] for v in manifest['files'].values()))
            else:
                if manifest:
                    assert manifest['contract']==protocol,'Remote postrun contract mismatch; preserve existing evidence'
                    restored=base/'remote_resume';pub.download(revision,restored,protocol)
                    remote_count=verified_prefix(restored,protocol);local_count=verified_prefix(root,protocol)
                    if remote_count>local_count:
                        shutil.copytree(restored,root,dirs_exist_ok=True)
                        print('Restored newer published analysis:',remote_count,'completed stages; local had',local_count,flush=True)
                    else:print('Compatible local analysis is at least as complete as HF; keep',local_count,'verified stages.',flush=True)
                pub.claim();claimed=True
                def flush(force=False,release=False):
                    nonlocal last
                    if force or time.monotonic()-last>=1800:
                        staged=snapshot(root,base/'snapshots'/uuid.uuid4().hex,cfg,'review');value=pub.publish(staged,release=release);local=read(staged/'MANIFEST.json');value.update(selected_objects=len(local['files']),bytes=sum(v['bytes'] for v in local['files'].values()));last=time.monotonic();receipt['review']=value;atomic(base/'PUBLICATIONS.json',receipt)
                        # Only remove this self-created completed staging directory.
                        assert staged.resolve().is_relative_to((base/'snapshots').resolve());shutil.rmtree(staged)
                flush(True)
                input_root,_=download_input(pub,base,flush)
                flush(True)
                process=worker('review',input_root,root,protocol,str(data),base)
                await_worker(process,lambda:flush());process=None
                assert read(root/'RESULT.json')['status']=='completed'
                flush(True,True);claimed=False
            atomic(root/'HF_PUBLICATION.json',receipt['review']);atomic(base/'PUBLICATIONS.json',receipt)
            if mode=='review':return read(root/'RESULT.json')
            current='passport';out=base/'passport';out.mkdir(exist_ok=True)
            pub=Publisher('phase3/tread-two/geo-r1/passport/'+protocol,token,base/'transport/PASSPORT_HF.jsonl')
            revision=pub.head();remote,_=selected_json(pub,'RESULT.json',revision)
            if remote and remote.get('status')=='completed':
                assert remote['input']==PIN;pub.download(revision,out,protocol);receipt['passport']=dict(status='restored_verified_completed',revision=revision,destination=pub.destination,owner_released=True);print('Completed matching NB19 restored; skip.',flush=True)
            else:
                pub.claim();claimed=True
                from passport import valid
                if not valid(out,protocol):
                    process=worker('passport',root,out,protocol,'',base);await_worker(process,lambda:None);process=None
                assert read(out/'RESULT.json')['status']=='completed'
                staged=snapshot(out,base/'snapshots'/uuid.uuid4().hex,cfg,'passport');receipt['passport']=pub.publish(staged,release=True);claimed=False
                assert staged.resolve().is_relative_to((base/'snapshots').resolve());shutil.rmtree(staged)
                atomic(base/'PUBLICATIONS.json',receipt)
            result=dict(status='completed',protocol=protocol,stages=['review','passport'],publications=receipt,scientific_status='research_only_unselected',new_optimizer_updates=0)
            atomic(base/'PUBLICATIONS.json',receipt);atomic(base/'BATCH_RESULT.json',result);return result
    except BaseException as error:
        if process is not None and process.poll() is None:
            process.terminate()
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=15)
        out=base/current;out.mkdir(exist_ok=True);atomic(out/'ERROR.json',dict(type=type(error).__name__,stage=current,last_verified=pub.last_verified if pub else None))
        if claimed:
            try:
                staged=snapshot(out,base/'snapshots'/uuid.uuid4().hex,cfg,current);pub.publish(staged,release=True)
            except BaseException as failed:atomic(base/'HF_PENDING.json',dict(stop_type=type(error).__name__,publish_error=type(failed).__name__,last_verified=pub.last_verified))
        raise
    finally:signal.signal(signal.SIGTERM,old)

def worker(stage,input_root,out,protocol,dataset,base):
    env=dict(os.environ);env.pop('HF_TOKEN',None);env['OMP_NUM_THREADS']='4';env['MKL_NUM_THREADS']='4'
    log=(Path(base)/(stage+'_WORKER.log')).open('ab')
    process=subprocess.Popen([sys.executable,'-u','-B',str(HERE/'worker.py'),stage,'--input',str(input_root),'--output',str(out),'--contract',protocol,'--dataset',dataset],env=env,stdout=log,stderr=subprocess.STDOUT);log.close();process.geo_log=Path(base)/(stage+'_WORKER.log');return process

def await_worker(process,tick):
    last=0.
    while process.poll() is None:
        if time.monotonic()-last>=30:
            print('Postrun alive: CPU saved-data analysis; progress and input cache retained.',flush=True)
            if process.geo_log.exists():
                with process.geo_log.open('rb') as stream:stream.seek(max(0,process.geo_log.stat().st_size-700));print(stream.read().decode('utf-8',errors='replace'),flush=True)
            last=time.monotonic()
        tick();time.sleep(5)
    assert process.returncode==0,'Analysis stopped. Read review_WORKER.log or passport_WORKER.log; keep files and rerun Run All to resume completed stages.'
