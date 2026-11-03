"""Additive direct-file gate publication using unchanged tested r3 transport."""
import argparse,hashlib,json,os,shutil,signal,sys
from pathlib import Path
from public_inputs import atomic,sha
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def prepare(out,source,work):
    out,source,work=map(Path,(out,source,work));analysis=json.loads((out/'ANALYSIS.json').read_text());assert analysis['status']=='passed'
    assert analysis['new_model_forwards']==analysis['optimizer_updates']==0
    for n,m in analysis['tables'].items():assert sha(out/(n+'.csv'))==m['sha256']
    inventory=json.loads((source/'FILES.json').read_text());assert all(sha(source/n)==m['sha256'] for n,m in inventory.items())
    here=Path(__file__).parent;source_files={p.name:sha(p) for p in here.iterdir() if p.suffix in ('.py','.json')}
    files={p.relative_to(out).as_posix():p for p in out.rglob('*') if p.is_file() and p.suffix in ('.csv','.json','.md','.png','.html')}
    contract=dict(kind='TREAD-TWO existing-data readiness and evidence passport',version=1,sources=source_files,frozen_transport_inventory_sha256=sha(source/'FILES.json'),
        input_spec=json.loads((here/'INPUTS.json').read_text()),artifacts={n:sha(p) for n,p in sorted(files.items())},new_model_forwards=0,optimizer_updates=0,new_independent_tyres=0)
    identity=digest(contract);folder=work/'bundle'/identity[:12];folder.mkdir(parents=True,exist_ok=True)
    files.update({'gate_source/'+p.name:p for p in here.iterdir() if p.suffix in ('.py','.json')})
    files.update({'publication_source/'+n:source/n for n in ('FILES.json','common.py','evidence_hub.py','publisher.py','rate_budget.py')})
    for n,p in files.items():
        target=folder/n;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert sha(target)==sha(p),'Prepared immutable gate differs'
        else:shutil.copy2(p,target)
    atomic(folder/'CONTRACT.json',contract)
    local=dict(contract=identity,files={p.relative_to(folder).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in folder.rglob('*') if p.is_file() and p.name!='MANIFEST.json'});atomic(folder/'MANIFEST.json',local)
    result=dict(status='prepared_not_published',contract=identity,path=str(folder.resolve()),destination='phase3/tread-two/readiness-gate/v1/'+identity,files=len(local['files'])+1,csv_tables=len(analysis['tables']),source_fit_unchanged=True,model_training=False)
    atomic(work/'GATE_DELIVERY.json',result);return result
def publish(delivery,source,work,token):
    sys.path.insert(0,str(Path(source).resolve()));from publisher import Publisher
    from filelock import FileLock
    hub=Publisher(delivery['destination'],token,Path(work)/'GATE_REMOTE_HISTORY.jsonl');folder=Path(delivery['path']);local=json.loads((folder/'MANIFEST.json').read_text())
    with FileLock(str(Path(work)/'GATE_PUBLISH.lock'),timeout=1):
        head=hub.head();prior=hub.get_json('MANIFEST.json',head)
        if prior:
            assert prior['contract']==delivery['contract'] and {n:{k:m[k] for k in ('bytes','sha256')} for n,m in prior['files'].items()}==local['files']
            assert hub.get_json('OWNER.json',head)==dict(owner=None,expires=0)
            result=dict(status='already_published_same_manifest',revision=head,destination=delivery['destination'],owner_released=True)
            atomic(Path(work)/'GATE_PUBLICATION.json',result);return result
        hub.claim()
        try:
            result=hub.publish(folder,release=True);atomic(Path(work)/'GATE_PUBLICATION.json',result);return result
        except BaseException:
            try:atomic(Path(work)/'GATE_PUBLICATION.json',hub.publish(folder,release=True))
            except BaseException as error:atomic(Path(work)/'GATE_PUBLICATION_PENDING.json',dict(status='local_complete_publication_pending',error_type=type(error).__name__,destination=delivery['destination']))
            raise
def main():
    parser=argparse.ArgumentParser()
    for n in ('out','source','work'):parser.add_argument('--'+n,required=True)
    parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args();delivery=prepare(args.out,args.source,args.work)
    if not args.prepare_only:
        token=os.environ.get('HF_TOKEN');assert token,'Enable Kaggle Secret HF_TOKEN; no alternate credential lookup'
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:print('Stop requested; allowing complete gate publication to finish. Preserve working files.',flush=True))
        delivery=publish(delivery,args.source,args.work,token)
    print(json.dumps(delivery,indent=2))
if __name__=='__main__':main()
