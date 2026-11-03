"""Offline parent integration: real orchestration, injected HF/numeric services.

No live HF, real CUDA certificate or numerical training is claimed by this test.
The actual CPU engine/continuation/audit tests are independent receipts.
"""
import copy,json,os,shutil,sys,time
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'tread_two/trace_batch';sys.path.insert(0,str(SRC))
from common import atomic,canonical,read,file_meta,digest
import coordinator as c
VALID=ROOT/'tread_two/trace_validation'
reference=Path(read(VALID/'LOCAL_LIFECYCLE.json')['fixture'])/'cache' if (VALID/'LOCAL_LIFECYCLE.json').exists() else max([p/'cache' for p in VALID.glob('fixture_[0-9]*') if (p/'cache/CACHE.json').exists()],key=lambda p:p.stat().st_mtime)
fixture=VALID/('fixture_controller_'+str(time.time_ns()));fixture.mkdir()
remote={};registered=None;calls=[];fail_mode=None

class FakePublisher:
    REPO='OFFLINE_FIXTURE_NO_NETWORK'
    def __init__(self,destination,token,log):
        self.destination=destination;self.last_verified=None;self.owner='fixture_writer'
        target=Path(log).parent/'HTTP_REQUESTS.jsonl';target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('ab') as stream:stream.write(canonical(dict(category='injected_HTTP_fixture',method='GET',status=200,seconds=.001,bytes=123,rate_limit='"api";r=400;t=10',scope='offline_transport_writer_exercise_not_a_network_measurement'))+b'\n')
    def head(self):return 'offline_fixture_revision'
    def get_json(self,name,revision):
        entry=remote.get(self.destination,{})
        if name=='MANIFEST.json':return entry.get('manifest')
        if name=='OWNER.json':return dict(owner=entry.get('owner'),expires=0)
        if name in entry.get('bodies',{}):return json.loads(entry['bodies'][name])
        return None
    def claim(self):remote.setdefault(self.destination,{})['owner']=self.owner;calls.append(('claim',self.destination))
    def publish(self,folder,release=False):
        folder=Path(folder);manifest=read(folder/'MANIFEST.json');bodies={}
        for name,meta in manifest['files'].items():
            assert file_meta(folder/name)==meta;bodies['evidence/'+name]=(folder/name).read_bytes()
        remote[self.destination]=dict(manifest=dict(contract=manifest['contract'],format='immutable-direct-files-v1',files={n:dict(m,object='evidence/'+n) for n,m in manifest['files'].items()}),bodies=bodies,owner=None if release else self.owner)
        calls.append(('publish',self.destination,release));self.last_verified='offline_fixture_revision'
        return dict(status='offline_fixture',revision=self.last_verified,files=len(bodies)+1,owner_released=release,uploaded_objects=len(bodies))
    def download(self,revision,folder,protocol):
        folder=Path(folder);entry=remote[self.destination];assert entry['manifest']['contract']==protocol
        for name,meta in entry['manifest']['files'].items():
            path=folder/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(entry['bodies'][meta['object']]);assert file_meta(path)=={k:meta[k] for k in ('bytes','sha256')}
        return folder

def fake_registry(pub,cfg=None):
    global registered
    if cfg is None:return registered
    expected=dict(protocol=digest(cfg),config=copy.deepcopy(cfg),mode='one_measured_candidate')
    if registered is not None:assert registered==expected
    registered=expected;return expected

def snapshot(output,protocol,files,reason):
    output=Path(output);target=output/'snapshots'/str(time.time_ns());target.mkdir(parents=True);inventory={}
    for name,body in files.items():
        path=target/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(canonical(body) if isinstance(body,dict) else body);inventory[name]=file_meta(path)
    atomic(target/'MANIFEST.json',dict(contract=protocol,files=inventory));atomic(output/'SNAPSHOT_READY.json',dict(path=str(target),unit=1,update=0,reason=reason));return target

def cache_snapshot(output,cache,protocol,index,reason):
    files={'cache/'+n:(Path(cache)/n).read_bytes() for n in index['files']};files['cache/CACHE.json']=canonical(index)
    return snapshot(output,protocol,files,reason)

def fake_worker(mode,output,cache,dataset='',geometry=None,reference=None,tick=None,offline=False):
    calls.append(('worker',mode));output=Path(output);output.mkdir(exist_ok=True,parents=True)
    if mode=='prepare':
        shutil.copytree(globals()['reference'],cache,dirs_exist_ok=True);index=read(Path(cache)/'CACHE.json');index['identity']['protocol']=digest(c.configuration());index['identity']['fixture']=False;atomic(Path(cache)/'CACHE.json',index)
        cache_snapshot(output,cache,digest(c.configuration()),index,'fixture_cache')
    elif mode=='gate_direct':
        gate=digest(c.configuration(True));seam=output/'seam';seam.mkdir(exist_ok=True)
        atomic(seam/'POINTER.json',dict(protocol=gate,unit=1,chain='offline_fixture_chain',state='STATE.json'));atomic(seam/'STATE.json',dict(scope='injected_orchestration_fixture_not_a_model'))
        atomic(seam/'MANIFEST.json',dict(contract=gate,files={n:file_meta(seam/n) for n in ('POINTER.json','STATE.json')}))
        atomic(output/'STORAGE_PROJECTION.json',dict(status='failed' if fail_mode=='budget' else 'passed',scope='injected_controller_fixture'))
    elif mode=='gate_restore':
        atomic(output/'RESTORE_PROOF.json',dict(status='passed',dual_CUDA_RNG_states=2,scope='injected_controller_fixture_not_live_proof'))
        atomic(output/'STORAGE_PROJECTION.json',dict(status='passed',scope='injected_controller_fixture'))
        snapshot(output,digest(c.configuration(True)),{'GATE_RESULT.json':dict(status='fixture')},'fixture_gate_complete')
    elif mode=='init':snapshot(output,digest(c.configuration()),{'INITIAL.json':dict(status='fixture_initialized')},'fixture_init')
    elif mode=='train':
        if fail_mode=='stop':
            snapshot(output,digest(c.configuration()),{'STOPPED.json':dict(status='stopped_last_complete_fixture_unit')},'fixture_catchable_stop');raise KeyboardInterrupt('injected_catchable_stop')
        (output/'run').mkdir(exist_ok=True);value=dict(status='completed',selected=None,scope='injected_controller_fixture_not_training');atomic(output/'run/RESULT.json',value)
        snapshot(output,digest(c.configuration()),{'RESULT.json':value},'completed')
    if tick:tick()

def fake_review(*args):calls.append(('review',));return dict(status='offline_fixture_all_complete')

original=(c.Publisher,c.registry,c.run_worker,c.review_stage,c.preparation_snapshot)
try:
    c.Publisher=FakePublisher;c.registry=fake_registry;c.run_worker=fake_worker;c.review_stage=fake_review;c.preparation_snapshot=cache_snapshot
    base=fixture/'all';base.mkdir();(base/'geometry.pt').write_bytes(b'fixture_only')
    # Avoid the real geometry network download in this orchestration-only test.
    old_geo=c.GEOMETRY;from common import sha
    c.GEOMETRY=dict(old_geo,sha256=sha(base/'geometry.pt'))
    result=c._run('OFFLINE_FIXTURE','all',base,'');assert result['status']=='offline_fixture_all_complete'
    sequence=[x[1] for x in calls if x[0]=='worker'];assert sequence==['prepare','gate_direct','gate_restore','init','train'],sequence
    before=len(sequence);c._run('OFFLINE_FIXTURE','all',base,'');assert len([x for x in calls if x[0]=='worker'])==before
    assert all(v['owner'] is None for k,v in remote.items() if '/registry-read' not in k)
    all_result=dict(dependencies_sequential=True,complete_remote_skips_all_numeric_workers=True,owners_released=True,finite_transport_tail=True)
    remote.clear();registered=None;calls.clear();fail_mode='stop';base=fixture/'stop';base.mkdir();(base/'geometry.pt').write_bytes(b'fixture_only')
    try:c._run('OFFLINE_FIXTURE','all',base,'')
    except KeyboardInterrupt:pass
    else:raise AssertionError('Stop swallowed')
    fits=[v for k,v in remote.items() if '/fit/' in k];assert len(fits)==1 and fits[0]['owner'] is None and 'STOPPED.json' in fits[0]['manifest']['files'] and 'RESULT.json' not in fits[0]['manifest']['files']
    remote.clear();registered=None;calls.clear();fail_mode='budget';base=fixture/'budget';base.mkdir();(base/'geometry.pt').write_bytes(b'fixture_only')
    try:c._run('OFFLINE_FIXTURE','all',base,'')
    except AssertionError as error:assert 'storage projection failed' in str(error).lower()
    else:raise AssertionError('Failed budget accepted')
    assert not any(x in [('worker','init'),('worker','train')] for x in calls)
    atomic(VALID/'COORDINATOR.json',dict(status='passed',source_inventory=read(SRC/'FILES.json'),scope='real_parent_code_with_injected_HF_workers_only',**all_result,catchable_stop_publishes_and_releases=True,failed_storage_blocks_measured_worker=True,actual_HF=False,actual_dual_T4=False,remote_writes=0,credentials_read=False))
    print(json.dumps(read(VALID/'COORDINATOR.json'),indent=2),flush=True)
finally:c.Publisher,c.registry,c.run_worker,c.review_stage,c.preparation_snapshot=original;c.GEOMETRY=old_geo
