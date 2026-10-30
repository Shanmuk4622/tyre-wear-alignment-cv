"""One-owner native/CSV recovery and separate HF publication for a frozen-model gate."""
import argparse
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

HERE=Path(__file__).resolve().parent
STOP=False


def initialize(source,launcher,work):
    sys.path.insert(0,str(Path(source).resolve()))
    if launcher:
        spec=importlib.util.spec_from_file_location('gate_transport',launcher)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.install(source,work)


def prepare(args):
    from common import atomic,capacity,digest,environment,read,sha
    from data_atlas import dataset,locate,DATA_CHECKSUM,OVERLAY_SHA
    from model_atlas import WEIGHTS_SHA
    import torch
    import importlib.metadata
    sys.path.insert(0,str(Path(args.review_source).resolve()))
    from review_atlas import Inputs
    work=Path(args.work);work.mkdir(parents=True,exist_ok=True)
    capacity(work,reserve=4*1024**3)
    assets=work/'assets';assets.mkdir(exist_ok=True)
    inputs=Inputs(assets/'input_cache',args.manifest)
    selected=inputs.file('artifacts/epochs/epoch_0018.pt')
    anchor=Path(args.anchor) if args.anchor else assets/'anchor.pt'
    if not anchor.exists() or sha(anchor)!=WEIGHTS_SHA:
        url='https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/resolve/a0296f18e02b12d28f7e51377783d68538032209/phase3/tread-one-r1/371c40c0c6aaedaf5c5970f7721a284b042494db7bb330065f6cbde8fc3a0b8c/report/weights.pt'
        body=inputs.request(url)
        import hashlib
        assert hashlib.sha256(body).hexdigest()==WEIGHTS_SHA
        anchor.parent.mkdir(parents=True,exist_ok=True);anchor.write_bytes(body)
    data=locate(args.data_root or '')
    plan,_=dataset(data)
    train=sorted([r for r in plan['rows'] if r['role']=='train'],key=lambda r:r['image_id'])
    rules=read(HERE/'RULES.json')
    inventory=read(Path(args.source)/'FILES.json')
    dense='cuda:0' if torch.cuda.is_available() else 'cpu'
    head='cuda:1' if torch.cuda.device_count()>=2 else dense
    sources={p.name:sha(p) for p in HERE.iterdir() if p.suffix=='.py' or p.name in ('RULES.json','UPSTREAM.json')}
    contract=dict(kind='ATLAS train-only appearance/reference gate',version=1,rules=rules,sources=sources,
        frozen_sources={name:v['sha256'] for name,v in inventory.items()},environment=environment(),
        source_checksum=DATA_CHECKSUM,overlay_sha256=OVERLAY_SHA,anchor_sha256=sha(anchor),selected_checkpoint_sha256=sha(selected),
        training_image_ids=[r['image_id'] for r in train],dense_device=dense,head_device=head,
        analysis_packages={name:importlib.metadata.version(name) for name in ('pandas','matplotlib','huggingface_hub')},
        source_fit_revision='82e7e1c5774f54c786571cdef7b3b60c1e96630f',input_manifest_sha256=sha(args.manifest))
    cfg=dict(contract=contract,protocol=digest(contract),frozen_source=str(Path(args.source).resolve()),anchor=str(anchor.resolve()),selected=str(selected.resolve()),data_root=str(data),control_root=str(work.resolve()))
    if (work/'CONFIG.json').exists(): assert read(work/'CONFIG.json')['contract']==contract,'Gate environment/source differs; preserve existing work for compatibility review'
    atomic(work/'CONFIG.json',cfg)
    print('Prepared train-only gate:',len(train),'images; dense',dense,'head',head,'; zero optimizer updates.',flush=True)
    return cfg


def make_snapshot(work,cfg,status):
    from common import atomic,file_meta,read
    from gate_worker import SCHEMAS
    from store import Store
    store=Store(work/'run',cfg['contract'],SCHEMAS)
    if store.state is None: return None
    root=work/'snapshots';root.mkdir(exist_ok=True)
    target=store.export(root/('snapshot-'+uuid.uuid4().hex),HERE,status)
    atomic(target/'GATE_STAGING.json',dict(protocol=cfg['protocol'],derived=True))
    atomic(target/'FROZEN_SOURCE_INVENTORY.json',read(Path(cfg['frozen_source'])/'FILES.json'))
    for src in (work/'control').glob('*.jsonl'):
        dst=target/'control_history'/src.name;dst.parent.mkdir(exist_ok=True);shutil.copy2(src,dst)
    if (work/'worker_console.log').exists():
        dst=target/'control_history/worker_console.log';dst.parent.mkdir(exist_ok=True);shutil.copy2(work/'worker_console.log',dst)
    for name in cfg['contract']['frozen_sources']:
        src=Path(cfg['frozen_source'])/name;dst=target/'frozen_source'/name
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    if store.state['phase']=='completed':
        from gate_report import build
        build(target,cfg)
    else: atomic(target/'GATE_PARTIAL.json',dict(next_image=store.state['next_image'],training_updates=0,classification_ready=False))
    manifest=read(target/'MANIFEST.json')
    manifest['files']={p.relative_to(target).as_posix():file_meta(p) for p in target.rglob('*') if p.is_file() and p.name!='MANIFEST.json'}
    atomic(target/'MANIFEST.json',manifest)
    return target


def execute(args,cfg):
    global STOP
    from common import atomic,read
    from gate_worker import SCHEMAS
    from store import Store
    work=Path(args.work).resolve()
    hub=None
    if args.stage=='run':
        from publisher import Publisher
        import coordinator
        token=os.environ.get('HF_TOKEN');assert token,'Enable Kaggle HF_TOKEN; no alternate credentials'
        hub=Publisher('phase3/tread-two/atlas-v1/appearance-gate/'+cfg['protocol'],token,work/'control/REMOTE_HISTORY.jsonl')
        coordinator.claim(hub)
        try:
            parent=hub.head();manifest=hub.get_json('MANIFEST.json',parent)
            if manifest:
                assert manifest['contract']==cfg['protocol']
                downloaded=hub.download(parent,work/'remote_resume',cfg['protocol'])
                store=Store(work/'run',cfg['contract'],SCHEMAS)
                history=read(downloaded/'HISTORY.json')
                if store.state is None or len(history)>len(store.units): store.install(downloaded)
                else: assert store.units[:len(history)]==history,'Divergent gate histories; preserve both'
                resolved=(work/'remote_resume').resolve()
                assert resolved.parent==work and resolved.name=='remote_resume'
                shutil.rmtree(resolved)
        except BaseException:
            try: coordinator.release(hub)
            except BaseException: pass
            raise
    def stopping(*_):
        global STOP
        STOP=True;atomic(work/'STOP_REQUEST.json',dict(reason='catchable_stop'))
        print('Stop requested; committing the current image before immediate publication attempt.',flush=True)
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,stopping)
    for name in ('STOP_REQUEST.json','FLUSH_REQUEST.json'): (work/name).unlink(missing_ok=True)
    environment=dict(os.environ);environment.pop('HF_TOKEN',None)
    command=[sys.executable,'-B','-u',str(HERE/'gate_worker.py'),'--config',str(work/'CONFIG.json'),'--folder',str(work/'run')]
    if args.limit_images is not None: command+=['--limit-images',str(args.limit_images)]
    log=(work/'worker_console.log').open('ab');child=None
    def publish(status,release=False):
        target=make_snapshot(work,cfg,status)
        if not target: return None
        receipt=hub.publish(target,release=release) if hub else dict(status='local_not_published',path=str(target))
        atomic(work/'LAST_SNAPSHOT.json',dict(path=str(target),receipt=receipt))
        if hub: atomic(work/'LAST_REMOTE.json',receipt)
        # Rotate only this controller's verified-published derived staging.
        if hub:
            for old in target.parent.iterdir():
                resolved=old.resolve()
                if old!=target and resolved.parent==target.parent.resolve() and old.name.startswith('snapshot-') and (old/'GATE_STAGING.json').exists(): shutil.rmtree(resolved)
        return receipt
    try:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=environment)
        last=heartbeat=notice=time.monotonic()
        while child.poll() is None:
            if time.monotonic()-notice>=20:
                pointer=work/'run/POINTER.json'
                if pointer.exists():
                    unit=read(work/'run'/read(pointer)['unit'])
                    print('Gate durable image cursor:',min(unit['seq'],386),'/ 386; training updates 0.',flush=True)
                notice=time.monotonic()
            if hub and time.monotonic()-heartbeat>=900:
                import coordinator
                coordinator.heartbeat(hub);heartbeat=time.monotonic()
            if hub and time.monotonic()-last>=cfg['contract']['rules']['publication_seconds']:
                atomic(work/'FLUSH_REQUEST.json',dict(reason='30_minute_gate_publication'))
                while child.poll() is None: time.sleep(1)
                if child.returncode==0:
                    publish('completed',True)
                    return 0
                assert child.returncode==91
                publish('periodic')
                if STOP: break
                (work/'FLUSH_REQUEST.json').unlink(missing_ok=True)
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=environment)
                last=heartbeat=time.monotonic()
            time.sleep(1)
        if child.returncode not in (0,91): raise RuntimeError('Gate worker failed; preserve worker_console.log and the committed cursor')
        receipt=publish('completed' if child.returncode==0 else 'paused',True)
        print(json.dumps(receipt,indent=2),flush=True)
        return child.returncode
    except BaseException:
        atomic(work/'STOP_REQUEST.json',dict(reason='controller_error'))
        if child and child.poll() is None:
            try: child.wait(timeout=180)
            except subprocess.TimeoutExpired: child.terminate();child.wait(timeout=30)
        try: publish('paused_or_error',True)
        except BaseException as error: atomic(work/'HF_PENDING.json',dict(error_type=type(error).__name__,last_complete_state_retained=True))
        raise
    finally: log.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','local','run'])
    for name in ('source','review-source','manifest','work'): parser.add_argument('--'+name,required=True)
    parser.add_argument('--launcher');parser.add_argument('--anchor');parser.add_argument('--data-root');parser.add_argument('--limit-images',type=int)
    args=parser.parse_args();work=Path(args.work).resolve();work.mkdir(parents=True,exist_ok=True)
    initialize(args.source,args.launcher,work)
    from filelock import FileLock
    with FileLock(str(work/'GATE_CONTROLLER.lock'),timeout=1):
        cfg=prepare(args)
        if args.stage!='prepare': sys.exit(execute(args,cfg))

if __name__=='__main__': main()
