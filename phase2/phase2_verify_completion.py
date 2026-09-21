import json,pathlib,sys,numpy as np,hashlib
sys.path.insert(0,'phase2');import phase2_training_data as d
root=pathlib.Path('phase2/completion');audit=json.loads((root/'audit.json').read_text());plan=d.overlay(pathlib.Path('phase2/data/phase2_dataset_v1'));test={r['image_id']:r for r in plan['rows'] if r['role']=='test'};summary=[]
for row in audit['runs']:
 assert all(row['checks'].values()),row['run']
 folder=root/'runs'/row['run'];m=row['manifest'];smoke=d.read(folder/'SMOKE.json');assert smoke['status']=='passed'
 history=d.read(folder/'HISTORY.json');assert len(history)==m['updates'];assert all(np.isfinite(h['loss']) for h in history)
 if m['status']!='completed':continue
 assert m['epoch']==60
 f=d.read(folder/'FINAL.json');e=d.read(folder/'EXPORT.json');v=d.read(folder/'VALIDATION.json');records=f['test']['records'];model=m['job']['model']
 expected={k for k,r in test.items() if model!='hrnet' or r['point_kind']=='human'}
 assert len(records)==len(expected) and {r['image_id'] for r in records}==expected
 assert all(r['tyre']==test[r['image_id']]['physical_tyre_id'] for r in records)
 assert len(v)==60 and max(r['result']['score'] for r in v)==f['best_score']
 if model in ('mobilenetv4','resnet50'):
  assert all(r['label']==test[r['image_id']]['class_index'] for r in records)
  score=float(np.mean([np.mean([r['prediction']==c for r in records if r['label']==c]) for c in sorted({r['label'] for r in records})]))
 elif model=='hrnet':
  assert all(np.allclose(r['target'],test[r['image_id']]['points']) for r in records)
  score=-float(np.mean([np.abs(np.array(r['prediction'])-r['target']).mean() for r in records]))
 else:score=float(np.mean([x['dice'] for r in records for x in r['regions']]))
 assert abs(score-f['test']['score'])<1e-7
 summary.append(dict(run=row['run'],validation=f['best_score'],test=score,metric=f['test']['metric'],best_epoch=next(r['epoch'] for r in v if r['result']['score']==f['best_score']),test_count=len(records)))
reportdir=root/'reports/1789959917';mf=d.read(reportdir/'MANIFEST.json')
for name,sha in mf['files'].items():assert d.sha(reportdir/name)==sha
out=dict(revision=audit['revision'],verified_runs=summary,pending=[r['run'] for r in audit['runs'] if r['manifest']['status']!='completed'],checks='JSON SHA256; binary LFS SHA256; smoke; finite loss; update count; epoch count; test membership, labels and recomputed score; validation-best selection; report hashes')
d.write(root/'verification.json',out);print(json.dumps(out,indent=2))

