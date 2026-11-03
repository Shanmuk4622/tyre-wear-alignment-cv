"""Delivery, immutable-stage interruption and automatic-handoff contract checks."""
import ast,base64,json,os,subprocess,sys,time
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'tread_two/geo_postrun';sys.path.insert(0,str(SRC))
from common import atomic,digest,file_meta,read,sha
from pipeline import Stages,ARTIFACTS
import coordinator as c
valid=ROOT/'tread_two/geo_postrun_validation';fixture=valid/('fixture_'+str(time.time_ns()));fixture.mkdir(parents=True)
source=read(SRC/'FILES.json');assert all(file_meta(SRC/n)==m for n,m in source.items())
import nbformat
notes=read(valid/'NOTEBOOKS.json');notebooks=[]
for receipt in notes:
    path=ROOT/'notebooks'/receipt['notebook'];nb=read(path);nbformat.validate(nb);assert sha(path)==receipt['sha256']
    code=[cell for cell in nb['cells'] if cell['cell_type']=='code'];assert len(code)==3 and all(cell['execution_count'] is None and not cell['outputs'] for cell in code)
    for cell in code:ast.parse(''.join(cell['source']))
    tree=ast.parse(''.join(code[0]['source']));payload=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PAYLOAD' for t in n.targets))
    extract=fixture/receipt['mode'];extract.mkdir()
    for name,item in payload.items():
        body=base64.b64decode(item['base64'],validate=True);assert __import__('hashlib').sha256(body).hexdigest()==item['sha256']==sha(SRC/name);(extract/name).write_bytes(body)
    assert set(payload)==set(source)|{'FILES.json'}
    # Actual isolated extraction setup in a fresh interpreter, with only its
    # working-directory literal redirected to a disposable local folder.
    setup=''.join(code[0]['source']).replace('/kaggle/working',str(fixture/('fresh_'+receipt['mode'])).replace('\\','/'))
    private=fixture/('fresh_'+receipt['mode'])/'geo_private_hf_0362/huggingface_hub-0.36.2.dist-info';private.mkdir(parents=True)
    setup_path=fixture/(receipt['mode']+'_setup.py');setup_path.write_text(setup,encoding='utf-8')
    result=subprocess.run([sys.executable,'-B',str(setup_path)],capture_output=True,text=True);assert result.returncode==0,result.stderr
    notebooks.append(dict(name=path.name,mode=receipt['mode'],payload_files=len(payload),schema=True,fresh_setup_passed=True))
stage=Stages(fixture/'stages_test','fixture');runs=[]
def work():runs.append(1);(stage.out/'value.csv').write_text('value\n1\n');return dict(ok=True)
stage.run('accounting',work);stage.run('accounting',work);assert len(runs)==1
(stage.out/'value.csv').write_text('value\n2\n');stage.run('accounting',work);assert len(runs)==2
crash=Stages(fixture/'crash_test','fixture')
def interrupted():(crash.out/'value.csv').write_text('value\n1\n');raise InterruptedError('offline fixture')
try:crash.run('accounting',interrupted)
except InterruptedError:pass
assert not (crash.receipts/'accounting.json').exists()
def finish():(crash.out/'value.csv').write_text('value\n1\n');return dict(ok=True)
crash.run('accounting',finish);assert 'value.csv' in read(crash.receipts/'accounting.json')['files']
# The real coordinator is exercised offline; only the Hub, input transfer and
# expensive numeric worker are injected. Full numeric analysis is separate.
store={};calls=[];workers=[]
class FakePublisher:
    REPO=c.Publisher.REPO
    def __init__(self,destination,token,log):self.destination=destination;self.last_verified=None;self.session=type('S',(),dict(headers={'Authorization':'Bearer offline_fixture'}))()
    def head(self):return 'offline_fixture_revision'
    def get_json(self,name,revision):
        value=store.get(self.destination,{})
        if name=='OWNER.json':return dict(owner=None,expires=0)
        if name=='MANIFEST.json':return value.get('manifest')
        if name in value.get('bodies',{}):return json.loads(value['bodies'][name])
        return None
    def claim(self):calls.append(('claim',self.destination))
    def publish(self,folder,release=False):
        folder=Path(folder);m=read(folder/'MANIFEST.json');bodies={n:(folder/n).read_bytes() for n in m['files']}
        assert all(file_meta(folder/n)==meta for n,meta in m['files'].items())
        manifest=dict(contract=m['contract'],files={n:dict(meta,object=n) for n,meta in m['files'].items()});store[self.destination]=dict(manifest=manifest,bodies=bodies)
        calls.append(('publish',read(folder/'STAGE.json')['stage'],release,'RESULT.json' in bodies));self.last_verified='offline_fixture_revision'
        return dict(revision=self.last_verified,owner_released=release,status='offline_fixture_only')
    def download(self,revision,folder,protocol):
        value=store[self.destination];assert value['manifest']['contract']==protocol
        for name,body in value['bodies'].items():target=Path(folder)/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(body)
        return folder
old=(c.Publisher,c.locate,c.download_input,c.worker,c.await_worker)
from inputs import PIN
def fake_artifacts(out,name):
    for relative in ARTIFACTS[name]:
        path=Path(out)/relative;path.parent.mkdir(parents=True,exist_ok=True)
        if path.name=='FINDINGS.json':atomic(path,dict(fixture=True))
        elif path.suffix=='.json':atomic(path,dict(fixture=True))
        else:path.write_text('offline numeric-worker fixture only\n',encoding='utf-8')
    return dict(fixture=True)
def fake_worker(stage,input_root,out,protocol,dataset,base):
    workers.append(stage);out=Path(out)
    if stage=='review':
        from pipeline import Stages
        st=Stages(out,protocol);atomic(out/'FINDINGS.json',dict(fixture=True))
        for name in ['files','core','losses','native','masked','dense','visuals']:st.run(name,lambda name=name:fake_artifacts(out,name))
        atomic(out/'RESULT.json',dict(status='completed',input=PIN,findings=dict(fixture=True)))
    else:
        atomic(out/'OUTPUT_RECEIPT.json',dict(contract=protocol,files={}))
        atomic(out/'RESULT.json',dict(status='completed',input=PIN,review_contract=protocol))
    return type('P',(),dict(poll=lambda self:0))()
try:
    c.Publisher=FakePublisher;c.locate=lambda dataset:fixture;c.download_input=lambda pub,base,tick:(fixture,{})
    c.worker=fake_worker;c.await_worker=lambda process,tick:tick()
    first=c.run('offline_fixture',base=fixture/'handoff');assert first['status']=='completed' and workers==['review','passport']
    assert ('publish','review',True,True) in calls and ('publish','passport',True,True) in calls
    c.run('offline_fixture',base=fixture/'fresh_restore');assert workers==['review','passport']
    # A catchable stop must select only completed evidence/status and release
    # the owner immediately; it must not publish a completed RESULT.
    store.clear();calls.clear()
    def stopped(*args,**kwargs):raise KeyboardInterrupt('offline catchable stop')
    c.worker=stopped
    try:c.run('offline_fixture',base=fixture/'stopped')
    except KeyboardInterrupt:pass
    else:raise AssertionError('Stop was swallowed')
    assert ('publish','review',True,False) in calls
    # Exercise the actual 30-minute branch without sleeping or network calls.
    store.clear();calls.clear();c.worker=fake_worker
    instant=[10000.];monotonic=c.time.monotonic
    c.time.monotonic=lambda:instant[0]
    def after_half_hour(pub,base,tick):instant[0]+=1801;tick();return fixture,{}
    c.download_input=after_half_hour
    try:c.run('offline_fixture',mode='review',base=fixture/'periodic')
    finally:c.time.monotonic=monotonic
    assert len([v for v in calls if v[0]=='publish'])==4 # initial / periodic / input-complete / final
    store.clear();calls.clear();c.download_input=lambda pub,base,tick:(fixture,{})
    partial=fixture/'published_partial';partial.mkdir();protocol=digest(c.contract());atomic(partial/'FINDINGS.json',dict(fixture=True))
    partial_stages=Stages(partial,protocol)
    for name in ['files','core','losses']:partial_stages.run(name,lambda name=name:fake_artifacts(partial,name))
    fake=FakePublisher('phase3/tread-two/geo-r1/review/'+protocol,'offline',fixture/'partial.jsonl')
    fake.publish(c.snapshot(partial,fixture/'partial_snapshot',c.contract(),'review'),release=True)
    restored_prefix=[]
    def resumed(stage,input_root,out,protocol,dataset,base):
        restored_prefix.append(c.verified_prefix(out,protocol));return fake_worker(stage,input_root,out,protocol,dataset,base)
    c.worker=resumed;c.run('offline_fixture',mode='review',base=fixture/'partial_restore')
    assert restored_prefix==[3]
finally:c.Publisher,c.locate,c.download_input,c.worker,c.await_worker=old
viewer=subprocess.run(['node',str(ROOT/'analysis/test_geo_review_viewer.cjs'),str(ROOT/'tread_two/nb17_report/GEO_Review_Atlas.html')],capture_output=True,text=True);assert viewer.returncode==0,viewer.stderr
report=dict(status='passed',source_inventory_sha256=digest(source),notebooks=notebooks,stage_completed_skip=True,changed_output_detected=True,
            interrupted_identical_outputs_bound=True,automatic_review_to_passport_handoff=True,completed_remote_restore_skip=True,partial_remote_newer_stage_restore=True,catchable_stop_flush_and_release=True,periodic_30minute_branch=True,
            offline_transport_fixture=True,live_Kaggle_HF_publication_pending=True,viewer=json.loads(viewer.stdout),credentials_read=False,remote_writes=0)
atomic(valid/'DELIVERY_CHECKS.json',report);print(json.dumps(report,indent=2))
