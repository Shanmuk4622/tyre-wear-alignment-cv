import sys,json,pathlib,hashlib,shutil
sys.path.insert(0,'prototype');from prepare_models import download,digest
root=pathlib.Path('phase2'); audit=json.loads((root/'completion/audit.json').read_text()); reg=json.loads((root/'completion/reports/1789959917/EXPORT_REGISTRY.json').read_text()); dest=root/'workstation';dest.mkdir(exist_ok=True)
for key,v in reg.items():
 name=v['job']['model'];p=download('Shanmuk4622/tyre-wear-study',v['weights'],audit['revision'],dest/'models'/name/'weights.pt');assert digest(p)==v['sha256']
 for f in ['EXPORT.json','MODEL_CONFIG.json']:
  source=root/'completion/runs'/v['job']['id']/f
  if source.exists():shutil.copy2(source,p.parent/f)
 v['revision']=audit['revision'];v['local_weights']=str(p.resolve());v['test_score']=json.loads((root/'completion/runs'/v['job']['id']/'FINAL.json').read_text())['test']['score']
 print(name,p.stat().st_size,flush=True)
(dest/'registry.json').write_text(json.dumps(reg,indent=2));print('downloaded verified five models')

