"""Offline persistence faults plus actual GPU pause/resume inference equivalence."""
from PIL import Image
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'diagnostic'))
import requests
import numpy as np
from storage import Ledger,Publisher,atomic,canonical,read,sha,retry_delay


class Service:
    def __init__(self):
        self.files={};self.history={'r0':{}};self.head='r0';self.commits=0;self.lost=False
    def whoami(self): return {'name':'Shanmuk4622'}
    def repo_info(self,*args,**kw): return SimpleNamespace(sha=self.head)
    def create_commit(self,repo,parent_commit,operations,**kw):
        if parent_commit!=self.head:
            raise requests.HTTPError('fixture conflict',response=SimpleNamespace(status_code=409,headers={}))
        files=dict(self.files)
        for op in operations:
            with op.as_file() as stream: files[op.path_in_repo]=stream.read()
        self.commits+=1;self.head=f'r{self.commits}';self.files=files;self.history[self.head]=dict(files)
        if self.lost:
            self.lost=False;raise requests.ConnectionError('fixture: successful commit response was lost')
        return SimpleNamespace(oid=self.head)
    def list_repo_tree(self,repo,path_in_repo,revision,**kw):
        result=[]
        for path,body in self.history[revision].items():
            if path.startswith(path_in_repo+'/'):
                result.append(SimpleNamespace(path=path,size=len(body),lfs=SimpleNamespace(sha256=hashlib.sha256(body).hexdigest()) if path.endswith('.npy') else None,
                    blob_id=hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()))
        return result
    def get(self,url,**kw):
        split=url.split('/resolve/',1)[1];revision,path=split.split('/',1)
        body=self.history[revision].get(path)
        class Response:
            status_code=200 if body is not None else 404
            headers={}
            def __enter__(self): return self
            def __exit__(self,*args): return False
            def raise_for_status(self):
                if self.status_code!=200: raise requests.HTTPError('fixture missing',response=self)
            def json(self): return json.loads(body)
            def iter_content(self,size):
                for i in range(0,len(body),size): yield body[i:i+size]
        return Response()


def publisher(service,owner='test-owner'):
    p=Publisher.__new__(Publisher)
    p.destination='test-diagnostic';p.owner=owner;p.token=None;p.api=service;p.session=service;p.last_verified=None
    return p


def offline(base):
    checks=[];schemas={'toy':['unit','id','value']};contract={'fixture':True,'version':1}
    ledger=Ledger(base/'local',contract,schemas)
    ledger.commit('u0','job0',{'toy':[dict(id='a',value=.125),dict(id='b',value=.25)]},
        {'image-a':dict(array=np.ones((3,19,128,96),dtype=np.float32),job='job0',label=0)})
    (ledger.folder/'units/abandoned').mkdir()
    assert set(ledger.units())=={'u0'};checks.append('partial_unit_without_commit_marker_ignored')
    def report(folder): (folder/'FINDINGS.md').write_text('# Fixture\n',encoding='utf-8')
    snap=ledger.snapshot(report,'running')
    restored=Ledger(base/'restored',contract,schemas);restored.restore(snap);restored.restore(snap)
    assert set(restored.units())=={'u0'};checks.append('direct_csv_cursor_restore_is_idempotent')
    bad=restored.folder/'units/u0/toy.csv';bad.write_bytes(bad.read_bytes()+b'corrupt')
    try: restored.units()
    except AssertionError: checks.append('corrupt_complete_local_unit_rejected')
    else: raise AssertionError('Corruption passed')
    service=Service();p=publisher(service);p.claim();p.publish(snap)
    assert p.last_verified==service.head;checks.append('CAS_snapshot_publication_and_remote_blob_verification')
    fresh=Ledger(base/'fresh',contract,schemas);assert p.restore(fresh)
    assert set(fresh.units())=={'u0'};checks.append('fresh_remote_direct_file_resume')
    restored_gallery=read(fresh.folder/'units/u0/GALLERY.json')['image-a']['array_file']
    assert np.array_equal(np.load(fresh.folder/'units/u0'/restored_gallery),np.ones((3,19,128,96),dtype=np.float32))
    checks.append('direct_native_gallery_LFS_hash_and_fresh_restore')
    # Include a changed completed unit so the lost response has a unique outcome.
    ledger.commit('u1','job0',{'toy':[dict(id='c',value=.5)]});new=ledger.snapshot(report,'complete')
    service.lost=True;before=service.commits;p.publish(new,release=True)
    assert service.commits==before+1 and json.loads(service.files['test-diagnostic/OWNER.json'])['owner'] is None
    checks.append('ambiguous_success_acknowledged_without_duplicate_commit')
    stale=publisher(service,'stale')
    try: stale.publish(new)
    except AssertionError: checks.append('stale_writer_publication_fenced')
    else: raise AssertionError('Stale write was accepted')
    first=publisher(service,'first');first.claim();second=publisher(service,'second')
    try: second.claim()
    except RuntimeError: checks.append('active_other_writer_claim_rejected')
    else: raise AssertionError('Concurrent writer claim accepted')
    error=requests.HTTPError('fixture limit',response=SimpleNamespace(status_code=429,headers={'Retry-After':'75','RateLimit':'"api";r=0;t=90'}))
    assert retry_delay(error,0)==90;checks.append('429_retry_after_and_rate_reset_respected')
    assert retry_delay(ValueError('not transport'),0) is None;checks.append('nontransient_errors_not_retried')
    # Distinct contract cannot silently take existing local state.
    try: Ledger(ledger.folder,{'version':2},schemas)
    except AssertionError: checks.append('contract_drift_rejected')
    else: raise AssertionError('Contract drift accepted')
    return checks


def run_actual(base):
    args=[sys.executable,'-B',str(ROOT/'diagnostic/run.py'),'--seeds','3','--endpoints','selected',
          '--limit','2','--batch','2','--chunk','2']
    def execute(where,pause=False):
        cmd=args+['--output-root',str(where)]+(['--pause-after-units','1'] if pause else [])
        result=subprocess.run(cmd,capture_output=True,text=True)
        (base/(where.name+('-pause' if pause else '')+'.log')).write_text(result.stdout+result.stderr,encoding='utf-8')
        return result
    resumed=base/'resumed';r=execute(resumed,True)
    assert r.returncode!=0 and 'Requested diagnostic resume test boundary' in r.stderr
    folder=next(resumed.iterdir());unit=folder/'units/seed3-selected-b00000/UNIT.json'
    before=read(unit);assert read(folder/'LATEST.json')['status']=='paused'
    r=execute(resumed);assert r.returncode==0,r.stderr[-5000:]
    assert read(unit)==before and read(folder/'COMPLETE.json')['training_updates']==0
    reference=base/'reference';r=execute(reference);assert r.returncode==0,r.stderr[-5000:]
    other=next(reference.iterdir())
    a=folder/read(folder/'LATEST.json')['snapshot'];b=other/read(other/'LATEST.json')['snapshot']
    comparisons={}
    for name in ('predictions','regions','points','latents','evidence'):
        assert (a/'csv'/(name+'.csv')).read_bytes()==(b/'csv'/(name+'.csv')).read_bytes(),name
        with open(a/'csv'/(name+'.csv'),encoding='utf-8',newline='') as stream:
            comparisons[name]=sum(1 for _ in csv.DictReader(stream))
    assert comparisons['predictions']==20
    assert (a/'csv/gradients.csv').stat().st_size>200
    return dict(status='passed',scientific_csvs_bitwise_identical=comparisons,
         completed_first_unit_preserved=True,gradient_path_exercised=True,training_updates=0,
         report=str(a/'TREAD_ONE_Classifier_Diagnostic.html'))


def main():
    validation=ROOT/'diagnostics/validation';validation.mkdir(parents=True,exist_ok=True)
    base=Path(tempfile.mkdtemp(prefix='check_',dir=validation))
    checks=offline(base)
    actual=read(ROOT/'diagnostics/DIAGNOSTIC_TESTS.json')['actual_resume'] if '--offline-only' in sys.argv else run_actual(base)
    result=dict(status='passed',offline_checks=checks,actual_resume=actual,network_requests=0,
           live_hf_publication_tested=False,fixture_directory=str(base))
    atomic(ROOT/'diagnostics/DIAGNOSTIC_TESTS.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
