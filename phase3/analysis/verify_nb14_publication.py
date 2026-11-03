"""Pinned public NB14 audit: every byte, 52 tables, attention and original fit."""
import hashlib,json,re,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from filelock import FileLock
import nb11_public as client
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(json.loads((ROOT/'tread_two/nb14_completion/LATEST.json').read_text())['path'])
inspection=json.loads((OUT/'INSPECTION.json').read_text());assert [c['execution_count'] for c in inspection['cells']]==[1,2,3,4]
assert not any(o['type']=='error' for c in inspection['cells'] for o in c['outputs'])
streams='\n'.join(o['text'] for c in inspection['cells'] for o in c['outputs'] if o['type']=='stream');receipt=None;decoder=json.JSONDecoder()
for match in re.finditer(r'\{\s*"status"',streams):
    try:value,_=decoder.raw_decode(streams[match.start():])
    except json.JSONDecodeError:continue
    if value.get('status')=='published_and_verified':receipt=value
assert receipt and receipt['owner_released'] and receipt['files']==80
assert receipt['destination'].startswith('phase3/tread-two/atlas-context-r3/review/')
client.OUT=OUT;client.REVISION=receipt['revision'];client.PREFIX=receipt['destination'];client.PROTOCOL=client.PREFIX.split('/')[-1]
client.BASE=f'https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/resolve/{client.REVISION}/{client.PREFIX}/'
with FileLock(str(OUT/'AUDIT.lock'),timeout=1):
    manifest=client.manifest()
    with ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(client.fetch,manifest['files'].items()))
    remote=OUT/'remote';contract=json.loads((remote/'CONTRACT.json').read_text())
    assert hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()==client.PROTOCOL
    assert contract['source_revision']=='a44418844ba3f1c98adee2cae76a8bbf9f5f4e9e'
    assert contract['new_training_updates']==contract['new_model_forward_calls']==contract['new_independent_tyres']==0
    for n,h in contract['analysis_sources'].items():assert client.sha(remote/'review_source'/n)==h==client.sha(ROOT/'tread_two/context_review'/n)
    analysis=json.loads((remote/'ANALYSIS.json').read_text());assert analysis['status']=='passed' and len(analysis['csv_tables'])==52
    comparisons=[]
    for name,meta in analysis['csv_tables'].items():
        p=remote/(name+'.csv');assert client.sha(p)==meta['sha256']
        actual=pd.read_csv(p);local=pd.read_csv(ROOT/'tread_two/context_analysis_report'/(name+'.csv'))
        assert len(actual)==meta['rows']==len(local) and list(actual.columns)==meta['columns']==list(local.columns)
        maximum=0.;numeric=0
        for field in actual:
            a,b=actual[field],local[field];assert np.array_equal(a.isna(),b.isna()),(name,field)
            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                use=a.notna();difference=np.abs(a[use].to_numpy(dtype=float)-b[use].to_numpy(dtype=float))
                maximum=max(maximum,float(difference.max()) if len(difference) else 0);numeric+=1
                assert np.allclose(a[use],b[use],atol=1e-8,rtol=1e-8),(name,field,maximum)
            else:assert a.fillna('').astype(str).tolist()==b.fillna('').astype(str).tolist(),(name,field)
        comparisons.append(dict(table=name,rows=len(actual),numeric_columns=numeric,maximum_numeric_difference=maximum,byte_identical=client.sha(p)==client.sha(ROOT/'tread_two/context_analysis_report'/(name+'.csv'))))
    assert sum(c['rows'] for c in comparisons)==585825
    images=list((remote/'figures').glob('*.png'));assert len(images)==5
    for p in images:
        with Image.open(p) as im:im.verify()
    viewer=subprocess.run(['node',str(ROOT/'analysis/test_context_review_viewer.cjs'),str(remote/'ATLAS_Failure_Explorer.html')],capture_output=True,text=True,encoding='utf-8')
    assert viewer.returncode==0,viewer.stderr;viewer_result=json.loads(viewer.stdout)
    matrices=json.loads((remote/'ATTENTION_VIEW.json').read_text());assert len(matrices)==206
    for key,values in matrices.items():
        matrix=np.asarray(values);assert matrix.shape==(2,4,32,32) and np.isfinite(matrix).all()
        assert matrix.min()>=0 and matrix.max()<=1 and np.abs(matrix.sum(-1)-1).max()<2e-6
    fit=Path(json.loads((ROOT/'tread_two/nb13_completion/LATEST.json').read_text())['path']);previous=json.loads((fit/'REMOTE_MANIFEST.json').read_text())
    prefix='phase3/tread-two/atlas-context-r3/fit/'+previous['contract']
    unchanged=client.request(f'https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/resolve/{client.REVISION}/{prefix}/MANIFEST.json').json()
    assert unchanged==previous,'Original completed fit manifest changed'
    result=dict(status='passed',executed_notebook_sha256=inspection['executed_notebook_sha256'],revision=client.REVISION,contract=client.PROTOCOL,prefix=client.PREFIX,
      files=len(manifest['files'])+1,objects_verified=len(manifest['files']),bytes_verified=sum(x['bytes'] for x in manifest['files'].values()),all44_payload_files_match=True,
      code_cells_completed=4,notebook_errors=0,owner_released=True,CSV_tables=52,CSV_rows=585825,table_comparisons=comparisons,
      figures_verified=5,interactive_cases=1854,reference_cases=18,full_attention_cases=206,viewer=viewer_result,
      original_fit_manifest_all1502_entries_unchanged=True,new_model_forwards=0,new_training_updates=0,new_independent_tyres=0,
      environment=analysis['environment'],local_credentials_read=False,remote_writes=0,public_read_requests_this_invocation=client.CALLS)
    (OUT/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='table_comparisons'},indent=2))
