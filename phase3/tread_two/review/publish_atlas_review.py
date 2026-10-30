"""Prepare/publish a separate immutable review, never modifying the fit namespace."""
import argparse
import hashlib
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

from review_atlas import PROTOCOL,REVISION,atomic,sha


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def prepare(out, source, launcher, manifest, work):
    out,source,launcher,manifest,work=map(Path,(out,source,launcher,manifest,work))
    analysis=json.loads((out/'ANALYSIS.json').read_text())
    assert analysis['status']=='passed' and analysis['source_protocol']==PROTOCOL
    assert analysis['training_updates_performed']==analysis['model_forward_calls']==0
    for name,meta in analysis['csv_tables'].items(): assert sha(out/(name+'.csv'))==meta['sha256']
    scripts={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
    files={p.relative_to(out).as_posix():p for p in out.rglob('*') if p.is_file() and p.suffix in ('.csv','.json','.md','.png','.html')}
    assert all(not n.startswith(('BUNDLE','publication')) for n in files)
    contract=dict(kind='ATLAS completed-fit post-hoc evidence review',version=1,source_revision=REVISION,source_protocol=PROTOCOL,
        original_manifest_sha256=sha(manifest),analysis_sources=scripts,
        transport_launcher_sha256=sha(launcher),frozen_inventory_sha256=sha(source/'FILES.json'),
        environment=analysis['environment'],artifacts={name:sha(path) for name,path in sorted(files.items())},
        new_training_updates=0,new_model_forward_calls=0,new_independent_tyres=0,checkpoint_selection='Original validation-selected epoch18; final60 retained')
    identity=digest(contract)
    folder=work/'bundle'/identity[:12]
    folder.mkdir(parents=True,exist_ok=True)
    files.update({'review_source/'+p.name:p for p in Path(__file__).parent.glob('*.py')})
    files.update({'publication_source/'+name:source/name for name in ('FILES.json','common.py','evidence_hub.py','publisher.py','rate_budget.py','store.py')})
    files['publication_source/nb08_transport_launcher.py']=launcher
    files['source_metadata/FIT_MANIFEST.json']=manifest
    for name,path in files.items():
        target=folder/name
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists(): assert sha(target)==sha(path),'Immutable prepared review differs'
        else: shutil.copy2(path,target)
    atomic(folder/'CONTRACT.json',contract)
    local=dict(contract=identity,files={p.relative_to(folder).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in folder.rglob('*') if p.is_file() and p.name!='MANIFEST.json'})
    atomic(folder/'MANIFEST.json',local)
    receipt=dict(status='prepared_not_published',contract=identity,path=str(folder.resolve()),
        destination='phase3/tread-two/atlas-v1/review/'+identity,files=len(local['files'])+1,
        csv_tables=len(analysis['csv_tables']),source_fit_unchanged=True,model_training=False)
    atomic(work/'REVIEW_DELIVERY.json',receipt)
    return receipt


def publish(delivery, source, launcher, work, token):
    # Install only the already audited transport overrides around the original
    # immutable publisher. This does not call a training coordinator or worker.
    sys.path.insert(0,str(Path(launcher).resolve().parent))
    import nb08_transport_launcher
    nb08_transport_launcher.install(source,work,sha(Path(source)/'FILES.json'))
    from publisher import Publisher
    from filelock import FileLock
    import coordinator
    def stopping(*_):
        coordinator.STOP=True
        print('Stop requested. Completed review is local; allowing the current HF flush to finish.',flush=True)
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,stopping)
    folder=Path(delivery['path'])
    local=json.loads((folder/'MANIFEST.json').read_text())
    hub=Publisher(delivery['destination'],token,Path(work)/'REVIEW_REMOTE_HISTORY.jsonl')
    with FileLock(str(Path(work)/'REVIEW_PUBLISH.lock'),timeout=1):
        head=hub.head()
        prior=hub.get_json('MANIFEST.json',head)
        if prior:
            assert prior['contract']==delivery['contract']
            assert {name:{k:v[k] for k in ('bytes','sha256')} for name,v in prior['files'].items()}==local['files']
            result=dict(status='already_published_same_manifest',revision=head,destination=delivery['destination'],remote_objects_unchanged=True)
            atomic(Path(work)/'REVIEW_PUBLICATION.json',result)
            return result
        coordinator.claim(hub)  # Existing live writer is never stolen/deleted.
        try:
            result=hub.publish(folder,release=True)
            atomic(Path(work)/'REVIEW_PUBLICATION.json',result)
            return result
        except BaseException:
            try:
                result=hub.publish(folder,release=True)
                atomic(Path(work)/'REVIEW_PUBLICATION.json',result)
            except BaseException as error:
                atomic(Path(work)/'REVIEW_PUBLICATION_PENDING.json',dict(status='local_complete_publication_pending',error_type=type(error).__name__,destination=delivery['destination']))
            raise


def main():
    parser=argparse.ArgumentParser()
    for name in ('out','source','launcher','manifest','work'): parser.add_argument('--'+name,required=True)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    result=prepare(args.out,args.source,args.launcher,args.manifest,Path(args.work))
    if not args.prepare_only:
        token=os.environ.get('HF_TOKEN')
        assert token,'Enable Kaggle HF_TOKEN; no credential file or alternate token lookup is used'
        result=publish(result,args.source,args.launcher,Path(args.work),token)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
