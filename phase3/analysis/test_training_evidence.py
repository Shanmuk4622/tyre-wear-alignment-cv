"""Fault tests of the new training store and publisher, without network access."""
import csv,hashlib,json,os,shutil,subprocess,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'training'))
import numpy as np,torch,requests
from transactions import Store,atomic,canonical,digest,file_meta,read,sha
from evidence_hub import Hub,delay_for

SCHEMA={'toy':['update','epoch','value']};CONTRACT={'engineering_fixture':True,'revision':1}

def state(update):
 return dict(protocol=digest(CONTRACT),updates=update,next_epoch=1+update//2,next_cursor=update%2,
   row_cursors={'toy':0},model={'parameter':torch.tensor([float(update)])})

def commit(s,update,fault=None):
 return s.commit(state(update),{'toy':[] if update==0 else [dict(update=update,epoch=1+(update-1)//2,value=update*.25)]},
   None if not update else {'outputs.npy':np.full((2,3),update,dtype=np.float32)},fault=fault)

class Service:
 """Immutable Git/LFS fixture. This never contacts Hugging Face."""
 def __init__(self):self.files={};self.history={'r0':{}};self.head='r0';self.commits=0;self.lost=False
 def whoami(self):return {'name':'Shanmuk4622'}
 def repo_info(self,*a,**kw):return SimpleNamespace(sha=self.head)
 def create_commit(self,*a,parent_commit,operations,**kw):
  if parent_commit!=self.head:raise requests.HTTPError('fixture CAS conflict',response=SimpleNamespace(status_code=409,headers={}))
  files=dict(self.files)
  for op in operations:
   with op.as_file() as f:files[op.path_in_repo]=f.read()
  self.commits+=1;self.head=f'r{self.commits}';self.files=files;self.history[self.head]=dict(files)
  if self.lost:self.lost=False;raise requests.ConnectionError('Successful response lost')
  return SimpleNamespace(oid=self.head)
 def list_repo_tree(self,*a,path_in_repo,revision,**kw):
  return [SimpleNamespace(path=p,size=len(b),lfs=SimpleNamespace(sha256=hashlib.sha256(b).hexdigest()) if p.endswith(('.pt','.npy')) else None,
    blob_id=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()) for p,b in self.history[revision].items() if p.startswith(path_in_repo+'/')]
 def get(self,url,**kw):
  revision,path=url.split('/resolve/',1)[1].split('/',1);body=self.history[revision].get(path)
  class Response:
   status_code=200 if body is not None else 404;headers={}
   def __enter__(self):return self
   def __exit__(self,*a):return False
   def raise_for_status(self):
    if self.status_code!=200:raise requests.HTTPError('fixture missing',response=self)
   def json(self):return json.loads(body)
   def iter_content(self,size):
    for i in range(0,len(body),size):yield body[i:i+size]
  return Response()

def hub(service,base,owner='owner'):
 h=Hub.__new__(Hub);h.destination='offline-evidence';h.owner=owner;h.api=service;h.session=service;h.calls=0;h.last_verified=None;h.log=base/(owner+'.jsonl');return h

def rejected(fn):
 try:fn()
 except (AssertionError,FileNotFoundError):return
 raise AssertionError('Expected a verification failure')

def main():
 if '--fault-child' in sys.argv:
  folder=Path(sys.argv[sys.argv.index('--fault-child')+1]);stage=sys.argv[-1]
  s=Store(folder,CONTRACT,SCHEMA);commit(s,1,stage);raise AssertionError('Fault did not terminate')
 base=Path(tempfile.mkdtemp(prefix='fault_',dir=ROOT/'training_gates'));checks=[]
 # Kill a real child at each transaction boundary; recover only pointer-committed data.
 for stage in ('after_state','after_csv','after_manifest','before_pointer','after_pointer'):
  folder=base/stage;s=Store(folder,CONTRACT,SCHEMA);commit(s,0)
  r=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--fault-child',str(folder),stage],capture_output=True,text=True)
  assert r.returncode==86,(stage,r.stderr)
  recovered=Store(folder,CONTRACT,SCHEMA);expected=int(stage=='after_pointer')
  assert recovered.state()['updates']==expected and recovered.state()['row_cursors']['toy']==expected
  if not expected:commit(recovered,1)
  assert Store(folder,CONTRACT,SCHEMA).state()['updates']==1
  checks.append('native_exit_'+stage+'_consistent_cursor')
 s=Store(base/'rotation',CONTRACT,SCHEMA)
 for u in range(4):commit(s,u)
 assert len(list((s.root/'generations').glob('*/state.pt')))==2
 assert len(list((s.root/'generations').glob('*/outputs.npy')))==3
 assert len(read(s.root/'PRUNED_STATES.json'))==2
 checks.append('latest_two_native_states_all_csv_and_maps_retained')
 snap=s.export('passed');fresh=Store(base/'installed',CONTRACT,SCHEMA);fresh.install(snap)
 assert fresh.state()['updates']==3 and fresh.state()['row_cursors']=={'toy':3}
 again=fresh.export('restored')
 with open(again/'STATE_VERSIONS.csv',newline='',encoding='utf-8') as f:versions=list(csv.DictReader(f))
 assert [r['local_native_state_present'] for r in versions]==['0','0','0','1']
 checks.append('remote_base_historical_hashes_do_not_claim_missing_native_files')
 assert np.array_equal(np.load(again/'maps/update_00000001/outputs.npy'),np.full((2,3),1,dtype=np.float32))
 checks.append('direct_native_array_roundtrip')
 for target in ('state.pt','tables/toy.csv'):
  folder=base/('corrupt-'+Path(target).stem);shutil.copytree(s.root,folder)
  latest=Store(folder,CONTRACT,SCHEMA).generation()/target
  with open(latest,'ab') as f:f.write(b'corrupt')
  rejected(lambda:Store(folder,CONTRACT,SCHEMA));checks.append('corrupt_'+target+'_rejected')
 rejected(lambda:Store(s.root,{'different':True},SCHEMA));checks.append('contract_drift_rejected')
 broken=base/'bad-snapshot';shutil.copytree(snap,broken);cursor=read(broken/'CURSOR.json');cursor['row_cursors']['toy']=999;atomic(broken/'CURSOR.json',cursor)
 m=read(broken/'MANIFEST.json');m['files']['CURSOR.json']=file_meta(broken/'CURSOR.json');atomic(broken/'MANIFEST.json',m)
 rejected(lambda:Store(base/'bad-install',CONTRACT,SCHEMA).install(broken));checks.append('native_state_csv_cursor_disagreement_rejected')
 service=Service();h=hub(service,base)
 with patch('evidence_hub.wait',lambda d:None):
  service.lost=True;h.claim();assert service.commits==1
  checks.append('lost_successful_claim_response_no_duplicate_commit')
  h.publish(snap);assert h.last_verified==service.head;checks.append('CAS_direct_snapshot_Git_and_LFS_verification')
  downloaded=h.download(service.head,base/'download',digest(CONTRACT));remote=Store(base/'download-installed',CONTRACT,SCHEMA);remote.install(downloaded)
  assert remote.state()['updates']==3;checks.append('pinned_remote_native_state_CSV_array_restore')
  service.lost=True;before=service.commits;h.publish(snap,release=True)
  assert service.commits==before+1 and json.loads(service.files[h.destination+'/OWNER.json'])['owner'] is None
  checks.append('lost_successful_publication_acknowledged_and_owner_released')
  rejected(lambda:h.publish(snap));checks.append('stale_publication_writer_fenced')
  first=hub(service,base,'first');first.claim();second=hub(service,base,'second')
  try:second.claim()
  except RuntimeError:checks.append('second_live_writer_rejected')
  else:raise AssertionError('Concurrent writer accepted')
  error=requests.HTTPError('rate',response=SimpleNamespace(status_code=429,headers={'Retry-After':'75','RateLimit':'"api";r=0;t=90'}))
  assert delay_for(error,0)==90;checks.append('server_retry_and_reset_delay_respected')
  from email.utils import formatdate
  with patch('evidence_hub.time.time',return_value=1000):
   error.response.headers={'Retry-After':formatdate(1075,usegmt=True)};assert delay_for(error,0)==75
  checks.append('HTTP_date_retry_after_respected')
  import httpx
  assert delay_for(httpx.ConnectError('fixture'),0)==5 and delay_for(ValueError('fixture'),0) is None
  checks.append('requests_and_httpx_transports_nontransient_errors_distinguished')
  attempts=[0]
  def transient():
   attempts[0]+=1
   if attempts[0]==1:raise requests.Timeout('fixture')
   return 7
  assert first.call(transient,'retry_fixture')==7 and attempts[0]==2
  checks.append('transient_retry_logged_then_success')
 result=dict(status='passed',checks=checks,count=len(checks),fixture_directory=str(base),network_requests=0,live_hf_tested=False,
   tested_sources={n:sha(ROOT/n) for n in ('training/transactions.py','training/evidence_hub.py')})
 atomic(ROOT/'training_gates/FAULT_TESTS.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
