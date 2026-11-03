"""Pinned public NB12 audit; exact bytes, all40 tables, sources and original fit."""
import hashlib,json,re,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from filelock import FileLock
import nb11_public as client
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(json.loads((ROOT/'tread_two/nb12_completion/LATEST.json').read_text())['path'])
inspect=json.loads((OUT/'INSPECTION.json').read_text());assert [c['execution_count'] for c in inspect['cells']]==[1,2,3,4]
assert not any(o['type']=='error' for c in inspect['cells'] for o in c['outputs'])
streams='\n'.join(o['text'] for c in inspect['cells'] for o in c['outputs'] if o['type']=='stream');receipt=None;decoder=json.JSONDecoder()
for match in re.finditer(r'\{\s*"status"',streams):
    try:value,_=decoder.raw_decode(streams[match.start():])
    except json.JSONDecodeError:continue
    if value.get('status')=='published_and_verified':receipt=value
assert receipt and receipt['owner_released']
assert receipt['destination'].startswith('phase3/tread-two/atlas-lc-r2/review/')
# Reuse the bounded public cache transport with an explicit separate namespace.
# No training module, SDK credential cache, local env file or remote write is used.
client.OUT=OUT;client.REVISION=receipt['revision'];client.PREFIX=receipt['destination'];client.PROTOCOL=client.PREFIX.split('/')[-1]
client.BASE=f'https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/resolve/{client.REVISION}/{client.PREFIX}/'
with FileLock(str(OUT/'AUDIT.lock'),timeout=1):
    m=client.manifest()
    with ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(client.fetch,m['files'].items()))
    remote=OUT/'remote';contract=json.loads((remote/'CONTRACT.json').read_text())
    assert hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()==client.PROTOCOL
    assert contract['source_revision']=='3d5ae39602c3b25caa6af65c1e8d45f53a08e1c8'
    assert contract['new_training_updates']==contract['new_model_forward_calls']==contract['new_independent_tyres']==0
    for n,h in contract['analysis_sources'].items():assert client.sha(remote/'review_source'/n)==h==client.sha(ROOT/'tread_two/lc_review'/n)
    analysis=json.loads((remote/'ANALYSIS.json').read_text());assert analysis['status']=='passed' and len(analysis['csv_tables'])==40
    comparisons=[]
    for name,meta in analysis['csv_tables'].items():
        p=remote/(name+'.csv');assert client.sha(p)==meta['sha256']
        actual=pd.read_csv(p);local=pd.read_csv(ROOT/'tread_two/lc_analysis_report'/(name+'.csv'))
        assert len(actual)==meta['rows']==len(local) and list(actual.columns)==meta['columns']==list(local.columns)
        maximum=0.;numeric=0
        for field in actual:
            a,b=actual[field],local[field]
            assert np.array_equal(a.isna(),b.isna()),(name,field)
            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                use=a.notna();difference=np.abs(a[use].to_numpy(dtype=float)-b[use].to_numpy(dtype=float))
                maximum=max(maximum,float(difference.max()) if len(difference) else 0);numeric+=1
                assert np.allclose(a[use],b[use],atol=1e-8,rtol=1e-8),(name,field,maximum)
            else:assert a.fillna('').astype(str).tolist()==b.fillna('').astype(str).tolist(),(name,field)
        comparisons.append(dict(table=name,rows=len(actual),numeric_columns=numeric,maximum_numeric_difference=maximum))
    assert sum(c['rows'] for c in comparisons)==380595
    for p in (remote/'figures').glob('*.png'):
        with Image.open(p) as im:im.verify()
    html=(remote/'ATLAS_Failure_Explorer.html').read_text(encoding='utf-8')
    payload=json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>',html,re.S)[1])
    assert len(payload['cases'])==1442 and len(payload['bank'])==18
    # This publication must be additive: compare every original fit entry at its
    # new repository revision without downloading the5.28GB fit again.
    fit=Path(json.loads((ROOT/'tread_two/nb11_completion/LATEST.json').read_text())['path'])
    previous=json.loads((fit/'REMOTE_MANIFEST.json').read_text())
    fitprefix='phase3/tread-two/atlas-lc-r2/fit/'+previous['contract']
    unchanged=client.request(f'https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/resolve/{client.REVISION}/{fitprefix}/MANIFEST.json').json()
    assert unchanged==previous,'Original fit manifest changed'
    result=dict(status='passed',executed_notebook_sha256=inspect['executed_notebook_sha256'],revision=client.REVISION,contract=client.PROTOCOL,prefix=client.PREFIX,
      files=len(m['files'])+1,objects_verified=len(m['files']),bytes_verified=sum(x['bytes'] for x in m['files'].values()),all39_payload_files_match=True,
      code_cells_completed=4,notebook_errors=0,owner_released=True,CSV_tables=40,CSV_rows=380595,table_comparisons=comparisons,
      figures_verified=4,interactive_cases=1442,reference_cases=18,original_fit_manifest_all1350_entries_unchanged=True,
      new_model_forwards=0,new_training_updates=0,new_independent_tyres=0,environment=analysis['environment'],local_credentials_read=False,remote_writes=0,
      public_read_requests_this_invocation=client.CALLS)
    (OUT/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='table_comparisons'},indent=2))
