"""Automatic short preflight for the new journal/projector/publication path."""
from PIL import Image
import argparse,csv,json,os,subprocess,sys
from pathlib import Path
import spec
from journal import Journal,link
from metrics_r2 import SCHEMAS
from report_r2 import report
from transactions import atomic,read,sha,file_meta
from common import delta,deterministic,cpu,rng

def run(config,folder,limit=None,fault=None):
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
 cmd=[sys.executable,'-B','-u',str(spec.ROOT/'revision2/train.py'),'--config',str(config),'--folder',str(folder)]
 if limit is not None:cmd+=['--max-train-updates',str(limit)]
 if fault:cmd+=['--fault',fault,'--fault-unit','2']
 env=dict(os.environ);env.pop('HF_TOKEN',None)
 with open(folder/'console.log','ab') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
 assert r.returncode==(86 if fault else 0),(r.returncode,(folder/'console.log').read_text(errors='replace')[-6000:])

def differences(a,b):
 keys=('model','optimizer','scaler','scheduler','rng','losses','updates','epoch0','cursor','phase','eval_cursor','best','partial','history','row_cursors')
 values={k:delta(a[k],b[k]) for k in keys};assert max(values.values())==0,values;return values

def prepare(cfg,folder):
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);cp=folder/'CONFIG.json';atomic(cp,cfg)
 run(cp,folder/'expected');run(cp,folder/'paused',1)
 j=Journal(folder/'paused',spec.contract(cfg),SCHEMAS);state=j.load();assert state['updates']==1 and state['phase']=='train' and state['cursor']==1
 if cfg['condition']=='baseline':
  import torch
  from optimizer_r2 import base
  from data import plan,schedule,batch
  deterministic(cfg['seed']);device=torch.device(cfg['device']);model,opt,scaler,scheduler=base.legacy.components(device)
  p=plan(Path(cfg['data_root']));source,order=schedule(p,cfg['seed'],0)
  b=batch(cfg['data_root'],[source[i] for i in order[0]],device,cfg['seed'],0,0);base.legacy.update(model,opt,scaler,b,0)
  expected=dict(model=cpu(model.state_dict()),optimizer=cpu(opt.state_dict()),scaler=scaler.state_dict(),scheduler=scheduler.state_dict(),rng=rng())
  values={k:delta(v,state[k]) for k,v in expected.items()};assert max(values.values())==0,values
  from journal import native
  native(folder/'FROZEN_UPDATE_REFERENCE.pt',expected);atomic(folder/'ORIGINAL_MATH_PROOF.json',dict(status='passed',differences=values))
  del model,opt,scaler,scheduler,b,expected;torch.cuda.empty_cache()
 snap=j.export(spec.ROOT,'preflight_paused',report);j.close()
 for name in ('ORIGINAL_MATH_PROOF.json','FROZEN_UPDATE_REFERENCE.pt'):
  if (folder/name).exists():link(folder/name,snap/name)
 m=read(snap/'MANIFEST.json');m['files']={p.relative_to(snap).as_posix():file_meta(p) for p in snap.rglob('*') if p.is_file() and p!=snap/'MANIFEST.json'};atomic(snap/'MANIFEST.json',m)
 return snap

def finish(cfg,folder,source,source_revision=None):
 folder=Path(folder);j=Journal(folder/'restored',spec.contract(cfg),SCHEMAS)
 if j.seq<0:j.install(source)
 j.close();run(folder/'CONFIG.json',folder/'restored')
 expected=Journal(folder/'expected',spec.contract(cfg),SCHEMAS);actual=Journal(folder/'restored',spec.contract(cfg),SCHEMAS)
 values=differences(expected.load(),actual.load());assert actual.load()['phase']=='completed' and actual.load()['updates']==4
 a=expected.export(spec.ROOT,'preflight_reference',report);b=actual.export(spec.ROOT,'preflight_passed',report)
 def rows(root,table):
  result=[]
  for p in sorted((Path(root)/'csv').glob(f'epoch_*/{table}.csv')):
   with open(p,encoding='utf-8',newline='') as f:result.extend(csv.DictReader(f))
  return result
 compared=[]
 for table in SCHEMAS:
  if table in ('resources','events','eval_timings'):continue
  omitted={'checkpoint_before_sha256','checkpoint_after_sha256'}|{n for n in SCHEMAS[table] if n.endswith('_seconds')}
  clean=lambda root:[{k:v for k,v in r.items() if k not in omitted} for r in rows(root,table)]
  assert clean(a)==clean(b),table;compared.append(table)
 gradients=rows(b,'task_gradients');assert len(gradients)==4
 assert all(int(r['finite'])==1 and float(r['head_gradient_max_change'])==0 for r in gradients)
 if cfg['condition']=='shared_projection':assert all(int(r['projection_applied'])==int(float(r['cosine_before'])<0) for r in gradients)
 proof=dict(status='passed',condition=cfg['condition'],protocol=spec.protocol(cfg),restored_updates=1,completed_updates=4,
  native_differences=values,scientific_tables_equal=compared,published_source_revision=source_revision,
  actual_live_HF_restore=source_revision is not None,measured_study_started=False)
 if cfg['condition']=='baseline':proof['original_r1_update_math']=read(folder/'ORIGINAL_MATH_PROOF.json')['differences']
 atomic(folder/'PREFLIGHT_PROOF.json',proof);atomic(b/'PREFLIGHT_PROOF.json',proof)
 for name in ('ORIGINAL_MATH_PROOF.json','FROZEN_UPDATE_REFERENCE.pt'):
  if (folder/name).exists():link(folder/name,b/name)
 # Preserve both cases' independent raw evidence directly in the preflight release.
 for p in a.rglob('*'):
  if p.is_file():link(p,b/'reference_case'/p.relative_to(a))
 m=read(b/'MANIFEST.json');m['files']={p.relative_to(b).as_posix():file_meta(p) for p in b.rglob('*') if p.is_file() and p!=b/'MANIFEST.json'};atomic(b/'MANIFEST.json',m)
 expected.close();actual.close();return b,proof

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--folder',required=True);args=parser.parse_args()
 cfg=read(args.config);source=prepare(cfg,args.folder);_,proof=finish(cfg,args.folder,source);print(json.dumps(proof,indent=2))

if __name__=='__main__':main()
