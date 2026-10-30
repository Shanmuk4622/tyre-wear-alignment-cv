"""Actual-model pilot acceptance plus new SQLite and immutable-publisher faults."""
from PIL import Image
import csv,json,os,shutil,subprocess,sys,tempfile
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'revision2'))
import spec
import numpy as np,torch
from common import deterministic,cpu,rng,delta
from transactions import atomic,read,digest,file_meta,sha
from journal import Journal
from optimizer_r2 import project,base
from metrics_r2 import SCHEMAS
from publisher_r2 import Publisher
import acceptance
from test_training_evidence import Service

TOY={'condition':'fixture','sources':{},'revision':2};FIELDS={'toy':['condition','epoch','update','value']}
def toy_state(update):return dict(protocol=digest(TOY),updates=update,epoch0=0,cursor=update,phase='train',eval_cursor=0,row_cursors={'toy':0},best=None,model={'x':torch.tensor([float(update)])})
def toy_commit(j,u,fault=None):j.commit(toy_state(u),{'toy':[] if u==0 else [dict(epoch=1,update=u,value=u*.25)]},fault=fault)
def reject(fn):
 try:fn()
 except (AssertionError,FileNotFoundError):return
 raise AssertionError('Expected verification failure')

def offline(folder):
 checks=[]
 for name in ('after_native','after_rows','after_commit'):
  j=Journal(folder/name,TOY,FIELDS);toy_commit(j,0);j.close()
  r=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--fault-child',str(folder/name),name],capture_output=True,text=True)
  assert r.returncode==86,(name,r.stderr)
  j=Journal(folder/name,TOY,FIELDS);assert j.load()['updates']==(1 if name=='after_commit' else 0);j.close();checks.append('native_exit_'+name+'_atomic_SQL_native_cursor')
 j=Journal(folder/'rotation',TOY,FIELDS)
 for u in range(4):toy_commit(j,u)
 assert len(list((j.root/'checkpoints').glob('*.pt')))==2 and len(list(j.rows('toy')))==3
 checks.append('native_rotation_all_observations_retained')
 bad=folder/'corrupt_rows';shutil.copytree(j.root,bad);b=Journal(bad,TOY,FIELDS)
 r=json.loads(b.db.execute('SELECT payload FROM records LIMIT 1').fetchone()[0]);r['value']=99;b.db.execute('UPDATE records SET payload=? WHERE idx=0',(json.dumps(r),))
 reject(b.load);b.close();checks.append('SQL_row_tamper_rejected_by_native_anchored_hash_chain')
 snap=j.export(ROOT,'fixture');j.close();fresh=Journal(folder/'installed',TOY,FIELDS);fresh.install(snap);assert fresh.load()['updates']==3;fresh.close();checks.append('direct_CSV_native_snapshot_install')
 service=Service();h=Publisher.__new__(Publisher);h.destination='r2-offline';h.owner='owner';h.api=service;h.session=service;h.calls=0;h.last_verified=None;h.log=folder/'remote.log'
 original=service.create_commit;operations=[]
 def counted(*args,**kwargs):operations.append(len(kwargs['operations']));return original(*args,**kwargs)
 service.create_commit=counted
 # More than one batch verifies bounded staging and a single selecting manifest.
 for i in range(165):
  p=snap/'fixture_csv'/f'table_{i:03d}.csv';p.parent.mkdir(exist_ok=True);p.write_text(f'index,value\n{i},{i/4}\n',encoding='utf-8')
 m=read(snap/'MANIFEST.json');m['files']={p.relative_to(snap).as_posix():file_meta(p) for p in snap.rglob('*') if p.is_file() and p!=snap/'MANIFEST.json'};atomic(snap/'MANIFEST.json',m)
 h.claim()
 from coordinator import heartbeat
 before_manifest=service.files.get(h.destination+'/MANIFEST.json');heartbeat(TOY,h)
 owner=json.loads(service.files[h.destination+'/OWNER.json']);assert owner['owner']==h.owner and owner['expires']>__import__('time').time()+2600
 assert service.files.get(h.destination+'/MANIFEST.json')==before_manifest
 checks.append('lease_heartbeat_renews_owner_without_publishing_data')
 service.lost=True;receipt=h.publish(snap);assert max(operations)<=82
 checks.append('immutable_staging_80_objects_then_atomic_manifest')
 current=read(snap/'MANIFEST.json');remote=json.loads(service.files[h.destination+'/MANIFEST.json']);assert all(v['object'].endswith(Path(n).suffix) for n,v in remote['files'].items())
 checks.append('direct_CSV_native_extensions_no_archives')
 downloaded=h.download(receipt['revision'],folder/'downloaded',digest(TOY));restored=Journal(folder/'remote_restored',TOY,FIELDS);restored.install(downloaded);assert restored.load()['updates']==3;restored.close()
 checks.append('pinned_immutable_remote_restore')
 before=service.commits;h.publish(snap,release=True);assert service.commits==before+1
 checks.append('unchanged_objects_deduplicated_and_owner_released')
 reject(lambda:h.publish(snap));checks.append('stale_owner_fenced')
 second=Publisher.__new__(Publisher);second.destination=h.destination;second.owner='new-owner';second.api=service;second.session=service;second.calls=0;second.last_verified=None;second.log=folder/'second.log'
 second.claim();second.publish(snap,release=True);assert json.loads(service.files[h.destination+'/OWNER.json'])['owner'] is None
 checks.append('new_writer_identical_snapshot_releases_its_own_lease')
 a=[torch.tensor([1.,0.])];b=[torch.tensor([-1.,1.])];x,y,info=project(a,b,True)
 assert info['projection_applied']==1 and float((x[0]*y[0]).sum())>=0 and torch.equal(x[0],torch.tensor([.5,.5])) and torch.equal(y[0],torch.tensor([0.,1.]))
 checks.append('analytic_conflict_projection_matches_closed_form')
 x,y,info=project(a,b,False);assert torch.equal(x[0],a[0]) and torch.equal(y[0],b[0]);checks.append('baseline_gradient_identity')
 for b in ([torch.tensor([2.,0.])],[torch.zeros(2)]):
  x,y,info=project(a,b,True);assert info['projection_applied']==0 and all(np.isfinite(v) for v in info.values())
 checks.append('aligned_and_zero_gradient_noop_finite')
 return checks

def actual(folder):
 root=spec.locate();reports=[];initial_hashes=[]
 for condition in spec.CONDITIONS:
  cfg=spec.configuration(root,'cuda:0',condition,True);case=folder/condition
  source=acceptance.prepare(cfg,case)
  if condition=='baseline':
   math_delta=read(case/'ORIGINAL_MATH_PROOF.json')['differences'];assert max(math_delta.values())==0
  final,proof=acceptance.finish(cfg,case,source);proof['final_snapshot']=str(final)
  initial_hashes.append(read(final/'artifacts/INITIALIZATION.json')['model_tensor_sha256'])
  if condition=='baseline':proof['original_r1_update_math']=math_delta
  j=Journal(case/'restored',spec.contract(cfg),SCHEMAS);g=list(j.rows('task_gradients'))
  assert len(g)==4 and all(float(r['head_gradient_max_change'])==0 for r in g)
  if condition=='shared_projection':assert all(int(r['projection_applied'])==int(float(r['cosine_before'])<0) for r in g)
  proof['real_conflicts']=sum(float(r['cosine_before'])<0 for r in g);proof['projections']=sum(int(r['projection_applied']) for r in g)
  proof['counts']=j.load()['row_cursors'];proof['stored_bytes']=sum(p.stat().st_size for p in case.rglob('*') if p.is_file());j.close();reports.append(proof)
 assert len(set(initial_hashes))==1,'Paired model initialization differs'
 for r in reports:r['initial_model_tensor_sha256']=initial_hashes[0]
 return reports

def main():
 if '--fault-child' in sys.argv:
  folder=Path(sys.argv[sys.argv.index('--fault-child')+1]);stage=sys.argv[-1];j=Journal(folder,TOY,FIELDS);toy_commit(j,1,stage);raise AssertionError('fault did not exit')
 out=ROOT/'revision2_validation';out.mkdir(exist_ok=True);folder=Path(tempfile.mkdtemp(prefix='run_',dir=out))
 checks=offline(folder);atomic(out/'OFFLINE_TESTS.json',dict(status='passed',checks=checks,network_requests=0,fixture_directory=str(folder)))
 reports=actual(folder)
 result=dict(status='passed',offline_checks=checks,actual_cases=reports,fixture_directory=str(folder),source_hashes=spec.configuration(spec.locate(),'cuda:0','baseline',True)['sources'],
  measured_training_started=False,live_HF_tested=False,local_device='GTX1650/cv_conda')
 atomic(out/'VALIDATION.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('source_hashes','actual_cases')},indent=2));print('Both actual optimizer/journal recovery cases passed',flush=True)

if __name__=='__main__':main()
