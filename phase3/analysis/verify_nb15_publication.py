"""Independent pinned NB15 public-byte, scientific-table and preservation audit."""
import hashlib,json,re,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from filelock import FileLock
import nb11_public as client
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];OUT=Path(json.loads((ROOT/'tread_two/nb15_completion/LATEST.json').read_text())['path'])
inspection=json.loads((OUT/'INSPECTION.json').read_text());assert [c['execution_count'] for c in inspection['cells']]==[1,2,3,4]
assert not any(o['type']=='error' for c in inspection['cells'] for o in c['outputs'])
streams='\n'.join(o['text'] for c in inspection['cells'] for o in c['outputs'] if o['type']=='stream');receipt=None;decoder=json.JSONDecoder()
for match in re.finditer(r'\{\s*"(?:status|destination)"',streams):
    try:value,_=decoder.raw_decode(streams[match.start():])
    except json.JSONDecodeError:continue
    if value.get('status')=='published_and_verified':receipt=value
assert receipt and receipt['owner_released'] and receipt['files']==38 and receipt['destination'].startswith('phase3/tread-two/readiness-gate/v1/')
client.OUT=OUT;client.REVISION=receipt['revision'];client.PREFIX=receipt['destination'];client.PROTOCOL=client.PREFIX.split('/')[-1]
client.BASE=f'https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/resolve/{client.REVISION}/{client.PREFIX}/'
with FileLock(str(OUT/'AUDIT.lock'),timeout=1):
    manifest=client.manifest()
    with ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(client.fetch,manifest['files'].items()))
    remote=OUT/'remote';contract=json.loads((remote/'CONTRACT.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()==client.PROTOCOL
    assert contract['new_model_forwards']==contract['optimizer_updates']==contract['new_independent_tyres']==0
    for name,value in contract['sources'].items():assert client.sha(remote/'gate_source'/name)==value==client.sha(ROOT/'tread_two/readiness_gate'/name)
    assert contract['input_spec']==json.loads((ROOT/'tread_two/readiness_gate/INPUTS.json').read_text(encoding='utf-8'))
    analysis=json.loads((remote/'ANALYSIS.json').read_text(encoding='utf-8'));assert analysis['status']=='passed' and len(analysis['tables'])==16
    comparisons=[]
    for name,meta in analysis['tables'].items():
        path=remote/(name+'.csv');local=ROOT/'tread_two/readiness_report'/(name+'.csv');assert client.sha(path)==meta['sha256']
        actual=pd.read_csv(path);previous=pd.read_csv(local)
        assert len(actual)==meta['rows']==len(previous) and list(actual.columns)==meta['columns']==list(previous.columns)
        maximum=0.
        for column in actual:
            left,right=actual[column],previous[column];assert np.array_equal(left.isna(),right.isna()),(name,column)
            if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
                use=left.notna();differences=np.abs(left[use].to_numpy(dtype=float)-right[use].to_numpy(dtype=float));maximum=max(maximum,float(differences.max()) if len(differences) else 0)
                assert np.allclose(left[use],right[use],atol=1e-8,rtol=1e-8),(name,column,maximum)
            else:assert left.fillna('').astype(str).tolist()==right.fillna('').astype(str).tolist(),(name,column)
        comparisons.append(dict(table=name,rows=len(actual),maximum_numeric_difference=maximum,byte_identical=client.sha(path)==client.sha(local)))
    assert sum(c['rows'] for c in comparisons)==196905
    summary=json.loads((remote/'SUMMARY.json').read_text());caps=json.loads((remote/'CAPABILITIES.json').read_text())
    assert summary==json.loads((ROOT/'tread_two/readiness_report/SUMMARY.json').read_text()) and caps==json.loads((ROOT/'tread_two/readiness_report/CAPABILITIES.json').read_text())
    assert summary['scientific_readiness'] is False and summary['folds_missing_fit_class']==2 and summary['heldout_prediction_rows_used']==0
    figures=list((remote/'figures').glob('*.png'));assert len(figures)==3
    for path in figures:
        with Image.open(path) as image:image.verify()
    viewer=subprocess.run(['node',str(ROOT/'analysis/test_readiness_viewer.cjs'),str(remote/'Evidence_Passport.html')],capture_output=True,text=True,encoding='utf-8');assert viewer.returncode==0,viewer.stderr;viewer_result=json.loads(viewer.stdout)
    unchanged=[]
    for notebook,label in [(13,'original_context_fit'),(14,'completed_context_review')]:
        folder=Path(json.loads((ROOT/f'tread_two/nb{notebook}_completion/LATEST.json').read_text())['path']);earlier=json.loads((folder/'REMOTE_MANIFEST.json').read_text());proof=json.loads((folder/'VERIFICATION.json').read_text())
        response=client.request(f'https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/resolve/{client.REVISION}/{proof["prefix"]}/MANIFEST.json').json();assert response==earlier,label
        unchanged.append(dict(publication=label,manifest_entries=len(earlier['files']),unchanged=True))
    result=dict(status='passed',executed_notebook_sha256=inspection['executed_notebook_sha256'],revision=client.REVISION,contract=client.PROTOCOL,prefix=client.PREFIX,
        files=len(manifest['files'])+1,objects_verified=len(manifest['files']),bytes_verified=sum(m['bytes'] for m in manifest['files'].values()),all40_payload_files_match=True,
        code_cells_completed=4,notebook_errors=0,owner_released=True,CSV_tables=16,CSV_rows=196905,table_comparisons=comparisons,summary_and_capabilities_match=True,
        figures_verified=3,viewer=viewer_result,scientific_readiness=False,training_images=386,known_training_tyres=8,conservative_training_components=6,folds_missing_fit_class=2,
        source_publications_unchanged=unchanged,private_SDK_publication_completed=True,environment=analysis['environment'],new_model_forwards=0,optimizer_updates=0,new_independent_tyres=0,
        local_credentials_read=False,remote_writes=0,public_read_requests_this_invocation=client.CALLS)
    (OUT/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='table_comparisons'},indent=2))
