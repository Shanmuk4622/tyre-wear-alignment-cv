"""Offline failure injection: publisher fencing, checksums and checkpoint recovery."""
from PIL import Image
import json,tempfile,shutil,sys,types
from pathlib import Path
from unittest.mock import patch
import torch
import phase2_training_data as d
from phase2_hub import Hub
import phase2_run as run
import phase2_worker as worker

def main():
 results={}
 with tempfile.TemporaryDirectory(prefix='phase2-tests-') as tmp:
  root=Path(tmp);work=root/'work';work.mkdir();job=dict(id='segformer-combined-seed1',model='segformer',condition='combined',seed=1)
  folder=root/'runs'/job['id'];folder.mkdir(parents=True)
  state=dict(protocol='test',job=job,epoch=2,cursor=3,updates=123,history=[dict(loss=1)],validation=[],model={})
  torch.save(state,folder/'state.pt');d.write(folder/'LOCAL.json',dict(status='resumable'))
  for name in ('CONTRACT.json','OVERLAY.json','ENVIRONMENT.json','SESSION.json'):d.write(work/name,{})
  class Fake:
   prefix='phase2/test';owner='owner'
   def commit(self,files,message,release=False):
    manifest=d.read(files[self.prefix+'/runs/'+job['id']+'/MANIFEST.json'])
    assert manifest['updates']==123 and manifest['status']=='resumable'
    for name,h in manifest['files'].items():assert d.sha(files[self.prefix+'/runs/'+job['id']+'/'+name])==h
    self.manifest=manifest;return 'fake-revision'
  hub=Fake();run.snapshot(hub,work,root/'runs',[job],'test');results['immutable_snapshot_and_manifest']='PASS'
  class Failed(Fake):
   def commit(self,*a,**kw):raise ConnectionError('injected offline error')
  try:run.snapshot(Failed(),work,root/'runs',[job],'test')
  except ConnectionError:pass
  else:raise AssertionError('fault was ignored')
  assert (folder/'state.pt').exists() and (work/'snapshot'/job['id']/'state.pt').exists()
  results['failed_upload_retains_live_and_snapshot']='PASS'
  # A newer local checkpoint must not be overwritten by an older remote snapshot.
  manifest=dict(protocol='test',job=job,updates=100,status='resumable',files={'state.pt':'old'})
  mf=root/'remote_manifest.json';d.write(mf,manifest)
  obj=Hub.__new__(Hub);obj.prefix='phase2/test';obj.fetch=lambda *a,**kw:mf
  assert obj.restore(job,folder,'rev')['updates']==100
  assert torch.load(folder/'state.pt',weights_only=False)['updates']==123
  results['newer_local_checkpoint_preserved']='PASS'
  # Corrupt remote payload is rejected before replacing any destination.
  corrupt=root/'corrupt.pt';corrupt.write_bytes(b'corrupt');manifest['updates']=200
  manifest['files']['state.pt']='0'*64;d.write(mf,manifest)
  obj.fetch=lambda path,*a,**kw:mf if path.endswith('MANIFEST.json') else corrupt
  try:obj.restore(job,folder,'rev')
  except ValueError:pass
  else:raise AssertionError('corrupt checkpoint accepted')
  assert torch.load(folder/'state.pt',weights_only=False)['updates']==123
  results['corrupt_remote_checkpoint_rejected']='PASS'
  # A displaced lease owner cannot publish, even if still training offline.
  lease=root/'lease.json';d.write(lease,{job['id']:dict(owner='new-owner',expires=99999999999)})
  obj.leases={job['id']:'old-owner'};obj.revision=lambda:'rev';obj.fetch=lambda *a,**kw:lease
  try:obj.commit({},'must fail')
  except RuntimeError as e:assert 'fencing' in str(e)
  else:raise AssertionError('stale writer accepted')
  results['stale_writer_fenced']='PASS'
  # Catchable numerical errors do not advance optimizer/cursor (actual worker tested separately).
  obj=Hub.__new__(Hub);obj.calls=__import__('collections').deque();obj.limit=100;obj.budget_file=root/'budget.json'
  attempts=[0]
  def transient():
   attempts[0]+=1
   if attempts[0]<3:
    exc=RuntimeError('injected 429');exc.response=types.SimpleNamespace(status_code=429,headers={'Retry-After':'0'});raise exc
   return 42
  # Simulate time passage without a real 30-second wait.
  clock=[0.]
  with patch('phase2_hub.time.monotonic',side_effect=lambda:clock[0]),patch('phase2_hub.time.sleep',side_effect=lambda n:clock.__setitem__(0,clock[0]+n)):
   assert obj.retry(transient)==42
  assert attempts[0]==3;results['429_bounded_retry']='PASS'
  class Small(torch.nn.Module):
   def __init__(self):
    super().__init__();self.w=torch.nn.Parameter(torch.ones(1));self.register_buffer('calls',torch.zeros(1))
   def forward(self):self.calls.add_(1);return (self.w*torch.rand(1)).sum()
  model=Small();optimizer=torch.optim.AdamW(model.parameters(),lr=.01);scaler=torch.amp.GradScaler('cpu',init_scale=128.)
  remaining=[1]
  def overflow(g):
   if remaining[0]:remaining[0]-=1;return torch.full_like(g,float('inf'))
   return g
  handle=model.w.register_hook(overflow)
  with patch('phase2_worker.m.loss',side_effect=lambda model,*_:model()):
   value,events=worker.update(model,optimizer,scaler,{'img':torch.zeros(1)},dict(model='fault-test'),0)
  handle.remove()
  assert int(optimizer.state[model.w]['step'])==1 and int(model.calls)==1 and len(events)==1 and scaler.get_scale()==64
  results['overflow_retries_same_rng_and_buffers_without_skipping_update']='PASS'
 d.write(Path(__file__).parent/'manifests/phase2_persistence_local_tests.json',results)
 print(json.dumps(results,indent=2))
if __name__=='__main__':main()
