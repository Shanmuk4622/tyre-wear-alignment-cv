"""Local contract, transform, actual architecture and resume tests; no HF writes."""
import os,sys,json,tempfile,copy,gc
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'test_deps'))
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'local_validation/yolo_settings')
(Path(os.environ['YOLO_CONFIG_DIR'])/'Ultralytics').mkdir(parents=True,exist_ok=True)
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import numpy as np
import torch
import phase2_training_data as d
import phase2_models as m
import phase2_worker as w

def main():
 torch.set_num_threads(2);m.deterministic()
 root=ROOT/'data/phase2_dataset_v1';plan=d.overlay(root)
 assert plan['counts']=={'train':386,'validation':81,'test':103}
 groups=plan['groups'];assert not set(groups['train'])&set(groups['test']);assert not set(groups['train'])&set(groups['validation'])
 for permutation in __import__('itertools').permutations([g for g in groups['train'] if '040000' in g or '070000' in g or '090000' in g]):
  assert all(g in groups['train'] for g in permutation)
 d.write(ROOT/'manifests/phase2_training_overlay.json',plan)
 job=dict(model='hrnet',condition='combined',seed=1,id='hrnet-combined-seed1')
 rows,batches=d.schedule(plan,job,0);assert batches==d.schedule(plan,job,0)[1]
 assert len(batches)==len(d.schedule(plan,dict(job,condition='old_only'),0)[1])
 row=next(r for r in rows if r['domain']=='new_video_frame');x,a,p=d.sample(root,row,[64,64],123)
 x2,a2,p2=d.sample(root,row,[64,64],123);assert np.array_equal(x,x2) and np.array_equal(a,a2) and np.array_equal(p,p2)
 assert np.allclose(1-(1-p[[1,0,3,2,5,4]])[[1,0,3,2,5,4]],p)
 # Raw source remains unchanged; YOLO derived union guarantees containment.
 _,union,_=d.sample(root,row,[64,64],model='yolo26m');assert not (union[1]&~union[0]).any()
 assets=ROOT/'local_validation/assets';(assets/'segformer').mkdir(parents=True,exist_ok=True)
 import requests
 url='https://huggingface.co/nvidia/mit-b0/resolve/'+d.SPECS['segformer']['revision']+'/config.json'
 response=requests.get(url,timeout=30);response.raise_for_status();(assets/'segformer/config.json').write_text(response.text)
 results={}
 for name in d.MODELS:
  print('Testing actual architecture:',name,flush=True)
  d.SPECS[name]['hw']=[64,64];d.SPECS[name]['batch']=2
  job=dict(model=name,condition='combined',seed=1,id=name+'-combined-seed1');rows,_=d.schedule(plan,job,0)
  w.seed_all(42);model,opt,scaler,sched=w.components(job,assets,torch.device('cpu'),False)
  b=m.batch(root,rows[:2],name,seed=1,device='cpu')
  first,_=w.update(model,opt,scaler,b,job,0)
  saved=w.cpu(w.state(model,opt,scaler,sched,job,'local-test',0,1,[],[],None,-float('inf'),1))
  expected_loss,_=w.update(model,opt,scaler,b,job,0);expected=w.cpu(model.state_dict());expected_opt=w.cpu(opt.state_dict())
  model.eval()
  with torch.inference_mode():pred=m.predict(model,b,name)
  shape=list(pred.shape)
  del model,opt,scaler,sched;gc.collect()
  model,opt,scaler,sched=w.components(job,assets,torch.device('cpu'),False)
  w.restore(saved,model,opt,scaler,sched,'local-test',job);actual_loss,_=w.update(model,opt,scaler,b,job,0)
  dif=w.delta(expected,w.cpu(model.state_dict()));optdif=w.delta(expected_opt,w.cpu(opt.state_dict()))
  assert dif<=1e-5 and optdif<=1e-5 and abs(expected_loss-actual_loss)<=1e-5,(name,dif,optdif,expected_loss,actual_loss)
  results[name]=dict(status='PASS',first_loss=first,parameter_delta=dif,optimizer_delta=optdif,prediction_shape=shape,scope='actual architecture CPU 64x64 random initialization; GPU full-size pretrained smoke runs inside delivered notebook')
  d.write(ROOT/'manifests/phase2_training_local_tests.json',results)
  del model,opt,scaler,sched,b,saved,expected,expected_opt;gc.collect()
 print(json.dumps(results,indent=2))
if __name__=='__main__':main()
