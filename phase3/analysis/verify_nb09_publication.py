"""Independent pinned public NB09 object, table, source and publication audit."""
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import local,Lock
import numpy as np
import pandas as pd
import requests
from PIL import Image
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
latest=json.loads((ROOT/'tread_two/nb09_completion/LATEST.json').read_text())
OUT=Path(latest['path'])
inspection=json.loads((OUT/'INSPECTION.json').read_text())
assert [c['execution_count'] for c in inspection['cells']]==[1,2,3,4]
assert not any(o['type']=='error' for c in inspection['cells'] for o in c['outputs'])
streams='\n'.join(o['text'] for c in inspection['cells'] for o in c['outputs'] if o['type']=='stream')
records=[]
decoder=json.JSONDecoder()
for m in re.finditer(r'\{\s*"status"',streams):
    try: value,_=decoder.raw_decode(streams[m.start():])
    except json.JSONDecodeError: continue
    if value.get('status')=='published_and_verified': records.append(value)
assert records and all(r==records[0] for r in records)
receipt=records[0];revision=receipt['revision'];prefix=receipt['destination']
assert prefix.startswith('phase3/tread-two/atlas-v1/review/')
base=f'https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/resolve/{revision}/'
sessions=local();lock=Lock();requests_count=0
def get(url):
    global requests_count
    with lock: requests_count+=1
    if not hasattr(sessions,'session'):
        sessions.session=requests.Session();sessions.session.trust_env=False
    response=sessions.session.get(url,timeout=(20,150));response.raise_for_status();return response
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
manifest=get(base+prefix+'/MANIFEST.json').json()
owner=get(base+prefix+'/OWNER.json').json()
assert owner==dict(owner=None,expires=0) and len(manifest['files'])==50
assert manifest['contract']==prefix.split('/')[-1]
(OUT/'REMOTE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
def fetch(item):
    name,meta=item
    assert '..' not in Path(name).parts and not Path(name).is_absolute()
    target=OUT/'remote'/name
    if not target.exists() or sha(target)!=meta['sha256']:
        target.parent.mkdir(parents=True,exist_ok=True)
        body=get(base+prefix+'/'+meta['object']).content
        assert len(body)==meta['bytes'] and hashlib.sha256(body).hexdigest()==meta['sha256'],name
        target.write_bytes(body)
    assert target.stat().st_size==meta['bytes'] and sha(target)==meta['sha256'],name
with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(fetch,manifest['files'].items()))
remote=OUT/'remote'
contract=json.loads((remote/'CONTRACT.json').read_text())
digest=hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
assert digest==manifest['contract']
assert contract['source_revision']=='82e7e1c5774f54c786571cdef7b3b60c1e96630f'
assert contract['new_training_updates']==contract['new_model_forward_calls']==contract['new_independent_tyres']==0
for name,expected in contract['analysis_sources'].items():
    assert sha(remote/'review_source'/name)==expected==sha(ROOT/'tread_two/review'/name)
analysis=json.loads((remote/'ANALYSIS.json').read_text())
comparisons=[]
for name,meta in analysis['csv_tables'].items():
    path=remote/(name+'.csv');assert sha(path)==meta['sha256']
    actual=pd.read_csv(path);reference=pd.read_csv(ROOT/'tread_two/analysis_report'/(name+'.csv'))
    assert len(actual)==meta['rows']==len(reference) and list(actual.columns)==list(reference.columns)
    max_difference=0.;numeric=0;string=0
    for field in actual.columns:
        a,b=actual[field],reference[field]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            assert np.array_equal(a.isna(),b.isna()),(name,field)
            mask=a.notna();difference=np.abs(a[mask].to_numpy()-b[mask].to_numpy())
            if len(difference): max_difference=max(max_difference,float(difference.max()))
            assert np.allclose(a[mask],b[mask],atol=2e-8,rtol=2e-9),(name,field,max_difference)
            numeric+=1
        else:
            # Inventory hashes legitimately differ for environment-dependent
            # source report outputs; the immutable original-input inventory
            # still pins the same NB08 objects and must agree exactly.
            assert a.fillna('').astype(str).equals(b.fillna('').astype(str)),(name,field)
            string+=1
    comparisons.append(dict(table=name,rows=len(actual),numeric_columns=numeric,text_columns=string,maximum_absolute_numeric_difference=max_difference))
assert len(comparisons)==30 and sum(r['rows'] for r in comparisons)==246157
metrics=pd.read_csv(remote/'cohort_metrics.csv')
for endpoint in ['selected','final']:
    rows=metrics[(metrics.endpoint==endpoint)&metrics.stage.isin(['selected_eval','final_eval'])]
    assert len(rows)==1 and rows.iloc[0].balanced_accuracy_observed==.5 and rows.iloc[0].predicted_mid==62 and rows.iloc[0].predicted_high==41
for path in (remote/'figures').glob('*.png'):
    with Image.open(path) as picture: picture.verify()
html=(remote/'ATLAS_Failure_Explorer.html').read_text(encoding='utf-8')
data=json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>',html,re.S)[1])
assert len(data['cases'])==1030 and len(data['bank'])==18
assert len({(r['endpoint'],r['variant'],r['image_id']) for r in data['cases']})==1030
original_fit='phase3/tread-two/atlas-v1/fit/fe7dc763fe6d4b64ce8ccf2f22c4222b6da28ca7975c925d56ba70c3e09d19dd'
fit_at_publication=get(base+original_fit+'/MANIFEST.json').json()
fit_before=json.loads((ROOT/'tread_two/pilot_validation/nb08_followup/82e7e1c5774f/fit_MANIFEST.json').read_text())
(OUT/'FIT_AT_PUBLICATION.json').write_text(json.dumps(fit_at_publication,indent=2)+'\n',encoding='utf-8')
differences={name:dict(before=fit_before['files'].get(name),after=fit_at_publication['files'].get(name)) for name in set(fit_before['files'])|set(fit_at_publication['files']) if fit_before['files'].get(name)!=fit_at_publication['files'].get(name)}
print('Fit manifest administrative/history differences:',len(differences),flush=True)
assert fit_at_publication['contract']==fit_before['contract']
scientific_names=[name for name in fit_before['files'] if name.startswith(('csv/','artifacts/','source/')) or name in ('state.pt','CONTRACT.json')]
assert all(fit_at_publication['files'].get(name)==fit_before['files'][name] for name in scientific_names)
result=dict(status='passed',executed_notebook_sha256=latest['notebook_sha256'],code_cells_completed=4,errors=0,
    revision=revision,destination=prefix,review_contract=digest,manifest_objects_verified=50,total_files_with_manifest=51,
    source_bytes_verified=sum(v['bytes'] for v in manifest['files'].values()),csv_tables=30,csv_rows=246157,
    tables=sorted(comparisons,key=lambda x:x['table']),actual_environment=analysis['environment'],
    local_environment=json.loads((ROOT/'tread_two/analysis_report/ANALYSIS.json').read_text())['environment'],
    source_code_matches_delivered=True,owner_released=True,original_fit_manifest_unchanged=not differences,
    original_fit_scientific_files_unchanged=len(scientific_names),original_fit_manifest_differences=differences,
    explorer_records=1030,reference_cases=18,zero_zip_npz_database=not any(n.endswith(('.zip','.npz','.sqlite')) for n in manifest['files']),
    public_read_requests=requests_count,local_credentials_read=False,remote_writes=0,model_forward_calls=0,training_updates=0)
(OUT/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ('tables','original_fit_manifest_differences')},indent=2))
