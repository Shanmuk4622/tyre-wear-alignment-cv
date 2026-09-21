"""Small CPU model tests orchestration through pause/resume, selection and export."""
from pathlib import Path
import sys,tempfile,copy
from PIL import Image
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'test_deps'),str(ROOT)]
import torch
from unittest.mock import patch
import phase2_worker as w
import phase2_training_data as d

def main():
 torch.set_num_threads(2)
 root=ROOT/'data/phase2_dataset_v1';rows=d.read(root/'manifests/images.json')[:3]
 for r,role in zip(rows,['train','validation','test']):r.update(role=role,points=None,point_weight=0.,point_kind='none')
 plan=dict(rows=rows,decisions={'scope':'unit-test fixture, not real evaluation split'})
 job=dict(model='mobilenetv4',condition='combined',seed=1,id='lifecycle-test')
 cfg=dict(plan=plan,job=job,protocol='unit-test',root=str(root),assets='',sources={})
 class Tiny(torch.nn.Module):
  def __init__(self):super().__init__();self.head=torch.nn.Linear(3,3)
  def forward(self,x):return self.head(x.mean((2,3)))
 def components(job,assets,device,initialize):
  m=Tiny();o=torch.optim.AdamW(m.parameters(),lr=.001);s=torch.amp.GradScaler('cpu',enabled=False);sched=torch.optim.lr_scheduler.CosineAnnealingLR(o,60)
  return m,o,s,sched
 validations=[0];selected={};updates=[0];actual_update=w.update
 def update(*args):
  out=actual_update(*args);updates[0]+=1
  if updates[0]==4:w.STOP=True
  return out
 def evaluate(model,rows,*args):
  if w.STOP:raise InterruptedError('injected catchable stop')
  if rows[0]['role']=='validation':
   validations[0]+=1
   if validations[0]==3:selected.update(w.cpu(model.state_dict()))
   return dict(score=-abs(validations[0]-3),metric='fixture',records=[])
  return dict(score=0.,metric='fixture',records=[])
 with tempfile.TemporaryDirectory(prefix='phase2-lifecycle-') as folder,patch.object(w,'components',components),patch.object(w,'smoke',return_value={'status':'passed'}),patch.object(w,'evaluate',evaluate),patch.object(w,'update',update),patch.dict(d.SPECS['mobilenetv4'],hw=[64,64],batch=1):
  w.STOP=False;w.train(cfg,folder,torch.device('cpu'))
  state=torch.load(Path(folder)/'state.pt',weights_only=False);assert state['updates']==4 and state['cursor']==1 and not (Path(folder)/'FINAL.json').exists()
  w.STOP=False;w.train(cfg,folder,torch.device('cpu'))
  state=torch.load(Path(folder)/'state.pt',weights_only=False);assert state['epoch']==60 and state['updates']==60 and len(state['history'])==60
  exported=torch.load(Path(folder)/'weights.pt',weights_only=True);assert w.delta(selected,exported)==0
  assert d.read(Path(folder)/'LOCAL.json')['status']=='completed'
  assert d.read(Path(folder)/'EXPORT.json')['weights_sha256']==d.sha(Path(folder)/'weights.pt')
  assert d.read(Path(folder)/'FINAL.json')['best_score']==0
 result=dict(status='PASS',paused_updates=4,resumed_total_updates=60,history_preserved=True,validation_best_epoch=3,export_hash_verified=True,scope='CPU tiny-model control-flow test; actual architectures separately GPU-tested')
 d.write(ROOT/'manifests/phase2_lifecycle_local_tests.json',result);print(result)
if __name__=='__main__':main()
