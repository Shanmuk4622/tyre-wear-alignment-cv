"""CPU audit of completed Phase 2 runs; validation selects exports, never test."""
from pathlib import Path
import argparse,os,time
import numpy as np
import phase2_training_data as d
from phase2_hub import Hub,REPO

def main(args):
 work=Path(args.work);work.mkdir(parents=True,exist_ok=True);hub=Hub(work)
 revision=hub.revision();prefix=args.prefix.strip('/')
 if not prefix:
  # Find a unique protocol with completed training; ambiguity is never guessed.
  from huggingface_hub.errors import EntryNotFoundError
  entries=hub.retry(lambda:list(hub.api.list_repo_tree(REPO,'phase2/training-r1',repo_type='dataset',revision=revision,recursive=False)))
  choices=[e.path for e in entries if type(e).__name__=='RepoFolder']
  if len(choices)!=1:raise ValueError('Set RUN_PREFIX to the intended phase2/training-r1/<protocol> printed by training. Found '+str(choices))
  prefix=choices[0]
 protocol=prefix.rsplit('/',1)[-1];hub.prefix=prefix;hub.leases={}
 root=d.locate(args.root);plan=d.overlay(root);test={r['image_id']:r for r in plan['rows'] if r['role']=='test'}
 results=[];pending=[];registry={};conditions=args.conditions.split(',')
 for job in d.jobs(conditions):
  base=prefix+'/runs/'+job['id'];mf=hub.fetch(base+'/MANIFEST.json',revision,True)
  if not mf or d.read(mf)['status']!='completed':pending.append(job['id']);continue
  manifest=d.read(mf);assert manifest['protocol']==protocol and manifest['job']==job and manifest['epoch']==60
  artifacts={}
  for name in ('FINAL.json','EXPORT.json'):
   path=hub.fetch(base+'/'+name,revision);assert d.sha(path)==manifest['files'][name];artifacts[name]=d.read(path)
  result=artifacts['FINAL.json'];export=artifacts['EXPORT.json'];records=result['test']['records']
  expected={k for k,r in test.items() if job['model']!='hrnet' or r['point_kind']=='human'}
  assert {r['image_id'] for r in records}==expected and len(records)==len(expected)
  assert all(r['tyre']==test[r['image_id']]['physical_tyre_id'] for r in records)
  if job['model'] in ('mobilenetv4','resnet50'):
   assert all(r['label']==test[r['image_id']]['class_index'] for r in records)
   metric=lambda rs:float(np.mean([np.mean([r['prediction']==c for r in rs if r['label']==c]) for c in sorted({r['label'] for r in rs})]))
  elif job['model']=='hrnet':
   assert all(np.allclose(r['target'],test[r['image_id']]['points']) for r in records)
   metric=lambda rs:-float(np.mean([np.abs(np.array(r['prediction'])-r['target']).mean() for r in rs]))
  else:
   metric=lambda rs:float(np.mean([x['dice'] for r in rs for x in r['regions']]))
  score=metric(records);assert abs(score-result['test']['score'])<1e-7
  per_tyre={t:metric([r for r in records if r['tyre']==t]) for t in sorted({r['tyre'] for r in records})}
  results.append(dict(job=job,score=score,metric=result['test']['metric'],per_tyre=per_tyre,validation_score=result['best_score']))
  key=job['model']+'-'+job['condition'];candidate=dict(job=job,validation_score=result['best_score'],weights=base+'/weights.pt',sha256=export['weights_sha256'],revision=revision,contract=export)
  if key not in registry or candidate['validation_score']>registry[key]['validation_score']:registry[key]=candidate
 aggregates={}
 for key in registry:
  values=[r['score'] for r in results if r['job']['model']+'-'+r['job']['condition']==key]
  aggregates[key]=dict(seeds=len(values),mean=float(np.mean(values)),std=float(np.std(values)),complete_three_seeds=len(values)==3)
 comparisons=[]
 for model in d.MODELS:
  for seed in (1,2,3):
   pair={r['job']['condition']:r for r in results if r['job']['model']==model and r['job']['seed']==seed}
   if set(pair)=={'old_only','combined'}:comparisons.append(dict(model=model,seed=seed,combined_minus_old=pair['combined']['score']-pair['old_only']['score']))
 report=dict(status='completed' if not pending else 'partial',revision=revision,protocol=protocol,results=results,aggregates=aggregates,paired_comparisons=comparisons,pending=pending,limitations=plan['decisions'],workstation_promotion='not performed; requires separate hardware/video checks')
 d.write(work/'REPORT.json',report);d.write(work/'EXPORT_REGISTRY.json',registry)
 manifest={name:d.sha(work/name) for name in ('REPORT.json','EXPORT_REGISTRY.json')};d.write(work/'MANIFEST.json',dict(files=manifest,status=report['status']))
 run=str(int(time.time()));hub.commit({prefix+'/reports/'+run+'/'+name:work/name for name in ('REPORT.json','EXPORT_REGISTRY.json','MANIFEST.json')},'Phase2 result audit')
 print('Report:',report['status'],'completed runs:',len(results),'pending:',len(pending))
 print('Saved REPORT.json and EXPORT_REGISTRY.json; no existing workstation changed.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--work',required=True);p.add_argument('--root',default='');p.add_argument('--prefix',default='');p.add_argument('--conditions',default='combined');main(p.parse_args())
