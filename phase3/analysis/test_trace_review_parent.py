"""Real CPU review subprocess/publication order, with an offline byte-checked Hub."""
import json,shutil,sys,time
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'tread_two/trace_batch';sys.path.insert(0,str(SRC))
from common import read,atomic,canonical,file_meta,digest
from config import configuration,NAMESPACE
from store import Store
import coordinator as c
valid=ROOT/'tread_two/trace_validation';ref=Path(read(valid/'LOCAL_LIFECYCLE.json')['fixture'])/'direct/run'
base=valid/('fixture_review_parent_'+str(time.time_ns()));base.mkdir();remote={}
class Fake:
    REPO='OFFLINE_BYTE_CHECKED_FIXTURE'
    def __init__(self,destination,token,log):self.destination=destination;self.last_verified=None
    def head(self):return 'offline_fixture_revision'
    def get_json(self,name,revision):
        v=remote.get(self.destination,{})
        if name=='MANIFEST.json':return v.get('manifest')
        return json.loads(v['bodies'][name]) if name in v.get('bodies',{}) else None
    def claim(self):remote.setdefault(self.destination,{})['owner']='fixture'
    def publish(self,folder,release=False):
        folder=Path(folder);m=read(folder/'MANIFEST.json');bodies={}
        for n,meta in m['files'].items():assert file_meta(folder/n)==meta;bodies['evidence/'+n]=(folder/n).read_bytes()
        remote[self.destination]=dict(manifest=dict(contract=m['contract'],files={n:dict(meta,object='evidence/'+n) for n,meta in m['files'].items()}),bodies=bodies,owner=None if release else 'fixture')
        self.last_verified='offline_fixture_revision';return dict(revision=self.last_verified,owner_released=release)
    def download(self,revision,folder,protocol):
        m=remote[self.destination]['manifest'];assert m['contract']==protocol
        for n,meta in m['files'].items():
            p=Path(folder)/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(remote[self.destination]['bodies'][meta['object']]);assert file_meta(p)=={k:meta[k] for k in ('bytes','sha256')}
        return folder
original=c.Publisher
try:
    c.Publisher=Fake;protocol=digest(configuration(True));fitpub=Fake('fixture_fit','OFFLINE',base/'transport/REQUESTS.jsonl')
    snap=base/'fit_snapshot';Store(ref,protocol).snapshot(snap,allow_copy=True);fitpub.claim();fitpub.publish(snap,release=True)
    good=base/'good';good.mkdir();first=c.review_stage('OFFLINE',good,fitpub,'offline_fixture_revision',protocol);assert first['status']=='all_complete'
    second=c.review_stage('OFFLINE',good,fitpub,'offline_fixture_revision',protocol);assert second['status']=='all_complete_remote_skip'
    remote.clear();fitpub=Fake('fixture_fit','OFFLINE',base/'transport/REQUESTS.jsonl');fitpub.claim();fitpub.publish(snap,release=True)
    bad=base/'bad';bad.mkdir();fitpub.download('offline_fixture_revision',bad/'fit/run',protocol)
    pointer=read(bad/'fit/run/POINTER.json');path=bad/'fit/run'/pointer['state']
    with path.open('r+b') as stream:b=stream.read(1);stream.seek(0);stream.write(bytes([b[0]^1]))
    try:c.review_stage('OFFLINE',bad,fitpub,'offline_fixture_revision',protocol)
    except AssertionError as exc:assert 'WORKER.log' in str(exc)
    else:raise AssertionError('Corrupt review input accepted')
    reviews=[v for k,v in remote.items() if k.startswith(NAMESPACE+'/review/')];assert len(reviews)==1
    review=reviews[0];assert review['owner'] is None and 'RESULT.json' not in review['manifest']['files']
    assert any(n.startswith('errors/') and n.endswith('.log') for n in review['manifest']['files'])
    atomic(valid/'REVIEW_PARENT.json',dict(status='passed',source_inventory=read(SRC/'FILES.json'),actual_CPU_review_worker=True,offline_HF_only=True,completed_review_remote_skip=True,
        corrupted_input_rejected=True,error_logs_selected_without_completed_RESULT=True,error_owner_released=True,new_optimizer_updates=0,remote_writes=0,credentials_read=False))
    print(dict(status='passed',scope='actual_CPU_review_and_offline_byte_checked_publisher'),flush=True)
finally:c.Publisher=original
