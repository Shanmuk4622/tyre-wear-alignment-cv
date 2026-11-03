"""Credential-free pinned HF audit of completed NB20 review and passport."""
import ast,base64,hashlib,json,math,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import pandas as pd
import torch
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'tread_two/geo_postrun';sys.path.insert(0,str(SRC))
from common import atomic,read,sha,file_meta,digest
from coordinator import contract
import nb17_public as public
OUT=Path(read(ROOT/'tread_two/nb20_completion/LATEST.json')['path']);inspection=read(OUT/'INSPECTION.json');protocol=digest(contract())
streams='\n'.join(o['text'] for c in inspection['cells'] for o in c['outputs'] if o['type']=='stream')
needle='{\n  "status": "completed",\n  "protocol":'
start=streams.rfind(needle);assert start>=0,'No completed master result saved in notebook'
execution,_=json.JSONDecoder().raw_decode(streams[start:]);assert execution['protocol']==protocol and execution['stages']==['review','passport']
atomic(OUT/'EXECUTION_RESULT.json',execution)
assert inspection['embedded_files']==19 and inspection['all_payload_hashes_match']
assert [c['execution_count'] for c in inspection['cells']]==[1,2,3]
assert not any(o['type']=='error' for c in inspection['cells'] for o in c['outputs'])
REPO='Shanmuk4622/tyre-wear-study';latest=public.request(f'https://huggingface.co/api/datasets/{REPO}').json()['sha']
stages={};manifests={}
for stage,pin in execution['publications'].items():
    assert pin['owner_released'] and pin['status']=='published_and_verified' and pin['zip_uploads']==0
    revision,prefix=pin['revision'],pin['destination'];assert prefix=='phase3/tread-two/geo-r1/'+stage+'/'+protocol
    base=f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/{prefix}/'
    manifest=public.request(base+'MANIFEST.json').json();assert manifest['contract']==protocol and manifest['format']=='immutable-direct-files-v1'
    assert len(manifest['files'])+1==pin['files'] and public.request(base+'OWNER.json').json()==dict(owner=None,expires=0)
    assert public.request(f'https://huggingface.co/datasets/{REPO}/resolve/{latest}/{prefix}/MANIFEST.json').json()==manifest
    folder=OUT/'remote'/stage;folder.mkdir(parents=True,exist_ok=True);atomic(OUT/(stage.upper()+'_MANIFEST.json'),manifest)
    def fetch(item):
        name,meta=item
        assert not Path(name).is_absolute() and '..' not in Path(name).parts and '..' not in Path(meta['object']).parts
        path=folder/name;path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists() or file_meta(path)!={k:meta[k] for k in ('bytes','sha256')}:
            body=public.request(base+meta['object']).content
            assert len(body)==meta['bytes'] and hashlib.sha256(body).hexdigest()==meta['sha256'],(stage,name)
            part=path.with_name(path.name+'.part');part.write_bytes(body);part.replace(path)
        assert not name.endswith(('.zip','.npz','.sqlite','.raw'))
        return dict(file=name,**{k:meta[k] for k in ('bytes','sha256')})
    with ThreadPoolExecutor(max_workers=8) as pool:inventory=list(pool.map(fetch,manifest['files'].items()))
    pd.DataFrame(inventory).to_csv(OUT/(stage.upper()+'_FILE_INVENTORY.csv'),index=False)
    assert read(folder/'CONTRACT.json')==contract()
    for name,meta in read(SRC/'FILES.json').items():assert file_meta(folder/'source'/name)==meta
    assert sha(folder/'source/FILES.json')==sha(SRC/'FILES.json')
    assert read(folder/'STAGE.json')==dict(stage=stage,protocol=protocol,completed=True)
    result=read(folder/'RESULT.json');assert result['status']=='completed' and result['input']==read(SRC/'FIT_INPUT.json')
    csvs=[]
    for path in sorted(folder.rglob('*.csv')):
        count=0;columns=None
        for frame in pd.read_csv(path,chunksize=131072,low_memory=False):
            columns=list(frame.columns);count+=len(frame)
            for column in frame.select_dtypes(include='number'):assert np.isfinite(frame[column].dropna()).all(),(stage,path.name,column)
        csvs.append(dict(file=path.relative_to(folder).as_posix(),rows=count,columns=columns or []))
    pd.DataFrame([dict(file=r['file'],rows=r['rows'],columns=len(r['columns'])) for r in csvs]).to_csv(OUT/(stage.upper()+'_CSV_INVENTORY.csv'),index=False)
    stages[stage]=dict(revision=revision,prefix=prefix,files=len(inventory)+1,objects=len(inventory),bytes=sum(r['bytes'] for r in inventory),CSV_files=len(csvs),CSV_rows=sum(r['rows'] for r in csvs),owner_released=True,latest_manifest_unchanged=True)
    manifests[stage]=manifest
    print(stage,json.dumps(stages[stage]),flush=True)
atomic(OUT/'PUBLICATION_CHECK.json',dict(status='downloaded_and_byte_verified',protocol=protocol,latest_revision=latest,stages=stages))
# Independent scientific comparison uses the completed local actual-data review,
# not notebook text or just a remote success flag.
comparison=[];rounding=[]
for stage,reference in [('review',ROOT/'tread_two/nb17_report'),('passport',ROOT/'tread_two/geo_passport_report')]:
    folder=OUT/'remote'/stage
    names=read(folder/'TABLES.json')
    for name,spec in names.items():
        path=folder/spec['file'];assert file_meta(path)=={k:spec[k] for k in ('bytes','sha256')}
        actual=pd.read_csv(path);expected=pd.read_csv(reference/spec['file'])
        assert len(actual)==spec['rows'] and list(actual.columns)==spec['columns']
        assert list(actual.columns)==list(expected.columns) and len(actual)==len(expected),(stage,name,'shape')
        if name=='provenance':continue # actual NB18 HF receipt replaces local pending marker
        for column in actual:
            a,b=actual[column],expected[column]
            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                tolerance=dict(atol=5e-8,rtol=2e-7) if name=='epoch_parameter_audit' and column=='l2' else dict(atol=1e-8,rtol=1e-8)
                assert np.array_equal(a.isna(),b.isna()) and np.allclose(a,b,**tolerance,equal_nan=True),(stage,name,column)
                if name=='epoch_parameter_audit' and column=='l2':
                    a32=a.to_numpy(dtype='float32');b32=b.to_numpy(dtype='float32')
                    ulps=np.abs(a32.view('int32').astype('int64')-b32.view('int32').astype('int64'))
                    assert ulps.max()<=2
                    rounding.append(dict(table=name,column=column,maximum_absolute_difference=float(np.abs(a-b).max()),maximum_float32_ULP_difference=int(ulps.max()),tolerance=tolerance,reason='FP32 norm reduction differs between actual Torch 2.11 CPU and local Torch 2.5 CPU'))
            else:assert a.fillna('<NA>').astype(str).equals(b.fillna('<NA>').astype(str)),(stage,name,column)
        comparison.append(dict(stage=stage,table=name,rows=len(actual),matches_independent_local_reconstruction=True))
    if stage=='review':
        for name in ['source_file_inventory.csv','source_CSV_inventory.csv','source_family_inventory.csv']:
            a=pd.read_csv(folder/name);b=pd.read_csv(reference/name)
            pd.testing.assert_frame_equal(a,b,check_dtype=False)
            comparison.append(dict(stage=stage,table=name,rows=len(a),matches_independent_local_reconstruction=True))
        for name in ['projected.npy','global.npy']:
            a=np.load(folder/name,mmap_mode='r',allow_pickle=False);b=np.load(reference/name,mmap_mode='r',allow_pickle=False)
            assert a.shape==b.shape and a.dtype==b.dtype and np.isfinite(a).all() and np.array_equal(a,b)
        for receipt in (folder/'stages').glob('*.json'):
            value=read(receipt);assert value['contract']==protocol
            for name,meta in value['files'].items():assert file_meta(folder/name)==meta
        assert len(list((folder/'stages').glob('*.json')))==7
        assert read(folder/'FILE_VERIFICATION.json')==read(reference/'FILE_VERIFICATION.json')
        assert read(folder/'NATIVE_VERIFICATION.json')==read(reference/'NATIVE_VERIFICATION.json')
        visuals=read(folder/'VISUALS.json');local_visuals=read(reference/'VISUALS.json')
        assert {k:v for k,v in visuals.items() if k!='html_bytes'}=={k:v for k,v in local_visuals.items() if k!='html_bytes'}
        assert visuals['html_bytes']==(folder/'GEO_Review_Atlas.html').stat().st_size
        def html_parts(path):
            body=path.read_text(encoding='utf-8');marker='const cases=';offset=body.index(marker)+len(marker)
            data,length=json.JSONDecoder().raw_decode(body[offset:]);return body[:offset],data,body[offset+length:]
        hp,hdata,hs=html_parts(folder/'GEO_Review_Atlas.html');lp,ldata,ls=html_parts(reference/'GEO_Review_Atlas.html')
        assert hp==lp and hs==ls and len(hdata)==len(ldata)==570
        html_numeric_differences=[]
        def equivalent(a,b):
            if isinstance(a,dict):
                assert a.keys()==b.keys()
                for key in a:equivalent(a[key],b[key])
            elif isinstance(a,list):
                assert len(a)==len(b)
                for x,y in zip(a,b):equivalent(x,y)
            elif isinstance(a,(float,int)) and not isinstance(a,bool):
                assert math.isclose(a,b,abs_tol=1e-12,rel_tol=1e-12)
                if a!=b:html_numeric_differences.append(abs(a-b))
            else:assert a==b
        equivalent(hdata,ldata)
        html_comparison=dict(images=570,static_HTML_JS_exact=True,embedded_images_exact=True,HF_bytes=visuals['html_bytes'],local_bytes=local_visuals['html_bytes'],
                             numeric_values_different=len(html_numeric_differences),maximum_numeric_difference=max(html_numeric_differences,default=0.),comparison_tolerance=1e-12)
        atomic(OUT/'HTML_COMPARISON.json',html_comparison)
        # Resolve the only larger-than-1e-8 comparison against original epoch
        # tensors using an independent FP64 sum of squares, not a wider blanket tolerance.
        norm_check=subprocess.run([sys.executable,'-B',str(ROOT/'analysis/audit_nb20_weight_norms.py')],capture_output=True,text=True)
        assert norm_check.returncode==0,norm_check.stderr
        fp64_norms=read(OUT/'FP64_WEIGHT_NORM_SUMMARY.json');assert fp64_norms['status']=='passed' and fp64_norms['rows']==4020
        # Float64 reductions may differ at the last few bits across CPU libraries.
        findings=read(folder/'FINDINGS.json');assert findings['selected_checkpoint_available'] is False and findings['promotion']=='research_only'
        assert findings['geometry_eligible_epochs']==60 and findings['representation_eligible_epochs']==0
    else:
        assert read(folder/'RESULT.json')['new_optimizer_updates']==0
        pub=read(folder/'RESULT.json')['review_publication'];assert pub==execution['publications']['review']
        receipt=read(folder/'OUTPUT_RECEIPT.json');assert receipt['contract']==protocol
        for name,meta in receipt['files'].items():assert file_meta(folder/name)==meta
        provenance=pd.read_csv(folder/'provenance.csv').set_index('kind')
        assert provenance.loc['derived_review','revision']==stages['review']['revision'] and provenance.loc['derived_review','prefix']==stages['review']['prefix']
        copied=[]
        for path in folder.glob('*.csv'):
            if path.stem in names or path.name=='HF_transport_events.csv':continue
            assert sha(path)==sha(OUT/'remote/review'/path.name),path.name
            copied.append(path.name)
            comparison.append(dict(stage=stage,table=path.name,rows=len(pd.read_csv(path)),matches_independent_local_reconstruction=True))
        assert len(copied)==10
pd.DataFrame(comparison).to_csv(OUT/'SCIENTIFIC_COMPARISON.csv',index=False)
viewer=subprocess.run(['node',str(ROOT/'analysis/test_geo_review_viewer.cjs'),str(OUT/'remote/review/GEO_Review_Atlas.html')],capture_output=True,text=True);assert viewer.returncode==0,viewer.stderr
unchanged=[]
for n in (13,14,15,16,17):
    earlier=Path(read(ROOT/f'tread_two/nb{n}_completion/LATEST.json')['path']);v=read(earlier/'VERIFICATION.json');old=read(earlier/'REMOTE_MANIFEST.json')
    current=public.request(f'https://huggingface.co/datasets/{REPO}/resolve/{latest}/{v["prefix"]}/MANIFEST.json').json();assert current==old
    unchanged.append(dict(notebook=n,entries=len(old['files']),unchanged=True))
report=dict(status='passed',executed_notebook_sha256=inspection['executed_notebook_sha256'],protocol=protocol,latest_revision=latest,stages=stages,
            payload_files=19,all_payload_hashes_match=True,code_cells_completed=3,notebook_errors=0,completed_review_stages=7,
            scientific_tables_compared=len(comparison),derived_vectors_exact=True,original_27514418_row_audit_verified=True,original_8455_native_audit_verified=True,
            field_specific_FP32_rounding=rounding,epoch_weight_norms_reconstructed_FP64=4020,
            FP64_weight_norm_verification=fp64_norms,
            HTML_comparison=html_comparison,
            images=570,predicted_point_markers=3420,human_target_markers=720,static_figures=3,viewer=json.loads(viewer.stdout),
            original_publications_unchanged=unchanged,new_optimizer_updates=0,new_photos=0,image_model_forwards=0,promotion='research_only',selected_available=False,
            local_credentials_read=False,remote_writes=0,public_read_requests_this_invocation=public.CALLS)
atomic(OUT/'VERIFICATION.json',report);print(json.dumps(report,indent=2))
