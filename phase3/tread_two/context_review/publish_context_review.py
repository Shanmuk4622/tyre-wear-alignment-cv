"""Direct-file review publication using frozen r3 transport; never runs a fit."""
import argparse,hashlib,json,os,shutil,signal,sys,time
from pathlib import Path
from review_context import PROTOCOL,REVISION,atomic,sha
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def prepare(out,source,manifest,work):
    out,source,manifest,work=map(Path,(out,source,manifest,work));analysis=json.loads((out/'ANALYSIS.json').read_text())
    assert analysis['status']=='passed' and analysis['source_protocol']==PROTOCOL
    assert analysis['training_updates_performed']==analysis['model_forward_calls']==0
    for name,m in analysis['csv_tables'].items():assert sha(out/(name+'.csv'))==m['sha256']
    inventory=json.loads((source/'FILES.json').read_text());assert all(sha(source/n)==m['sha256'] for n,m in inventory.items())
    scripts={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
    files={p.relative_to(out).as_posix():p for p in out.rglob('*') if p.is_file() and p.suffix in ('.csv','.json','.md','.png','.html')}
    contract=dict(kind='ATLAS-CONTEXT r3 completed-fit evidence, correction and attention review',version=1,source_revision=REVISION,source_protocol=PROTOCOL,
      original_manifest_sha256=sha(manifest),analysis_sources=scripts,frozen_inventory_sha256=sha(source/'FILES.json'),environment=analysis['environment'],
      peer_metadata=json.loads((Path(__file__).parent/'assets/PEER.json').read_text()),artifacts={n:sha(p) for n,p in sorted(files.items())},
      new_training_updates=0,new_model_forward_calls=0,new_independent_tyres=0,checkpoint_selection='Original validation rule selected epoch7; final60 retained; no re-selection')
    identity=digest(contract);folder=work/'bundle'/identity[:12];folder.mkdir(parents=True,exist_ok=True)
    files.update({'review_source/'+p.name:p for p in Path(__file__).parent.glob('*.py')})
    files.update({'review_source/assets/'+p.name:p for p in (Path(__file__).parent/'assets').iterdir() if p.is_file()})
    files.update({'publication_source/'+n:source/n for n in ('FILES.json','common.py','evidence_hub.py','publisher.py','rate_budget.py')})
    files['source_metadata/FIT_MANIFEST.json']=manifest
    for n,p in files.items():
        target=folder/n;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert sha(target)==sha(p),'Immutable prepared review differs'
        else:shutil.copy2(p,target)
    atomic(folder/'CONTRACT.json',contract)
    local=dict(contract=identity,files={p.relative_to(folder).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in folder.rglob('*') if p.is_file() and p.name!='MANIFEST.json'})
    atomic(folder/'MANIFEST.json',local)
    result=dict(status='prepared_not_published',contract=identity,path=str(folder.resolve()),destination='phase3/tread-two/atlas-context-r3/review/'+identity,
      files=len(local['files'])+1,csv_tables=len(analysis['csv_tables']),source_fit_unchanged=True,model_training=False)
    atomic(work/'REVIEW_DELIVERY.json',result);return result
def publish(delivery,source,work,token):
    sys.path.insert(0,str(Path(source).resolve()))
    from publisher import Publisher
    from filelock import FileLock
    hub=Publisher(delivery['destination'],token,Path(work)/'REVIEW_REMOTE_HISTORY.jsonl')
    folder=Path(delivery['path']);local=json.loads((folder/'MANIFEST.json').read_text())
    with FileLock(str(Path(work)/'REVIEW_PUBLISH.lock'),timeout=1):
        head=hub.head();prior=hub.get_json('MANIFEST.json',head)
        if prior:
            assert prior['contract']==delivery['contract']
            assert {n:{k:m[k] for k in ('bytes','sha256')} for n,m in prior['files'].items()}==local['files']
            owner=hub.get_json('OWNER.json',head);assert owner==dict(owner=None,expires=0),'Another active writer remains; preserve ownership'
            result=dict(status='already_published_same_manifest',revision=head,destination=delivery['destination'],owner_released=True,remote_objects_unchanged=True)
            atomic(Path(work)/'REVIEW_PUBLICATION.json',result);return result
        hub.claim()
        try:
            result=hub.publish(folder,release=True);atomic(Path(work)/'REVIEW_PUBLICATION.json',result);return result
        except BaseException:
            try:
                result=hub.publish(folder,release=True);atomic(Path(work)/'REVIEW_PUBLICATION.json',result)
            except BaseException as error:
                atomic(Path(work)/'REVIEW_PUBLICATION_PENDING.json',dict(status='local_complete_publication_pending',error_type=type(error).__name__,destination=delivery['destination']))
            raise
def main():
    parser=argparse.ArgumentParser()
    for n in ('out','source','manifest','work'):parser.add_argument('--'+n,required=True)
    parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    result=prepare(args.out,args.source,args.manifest,Path(args.work))
    if not args.prepare_only:
        token=os.environ.get('HF_TOKEN');assert token,'Enable Kaggle Secret HF_TOKEN; no alternate credential lookup'
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:print('Stop requested; completed review is local; allowing the current immutable HF publication to finish.',flush=True))
        result=publish(result,args.source,Path(args.work),token)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
