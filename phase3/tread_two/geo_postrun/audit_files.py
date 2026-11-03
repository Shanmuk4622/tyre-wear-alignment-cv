"""Pinned content, executable CSV schemas and complete recovery row counters."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common import atomic,digest,file_meta,read,sha
from inputs import PIN

def run(saved,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);manifest=saved.manifest
    assert manifest['contract']==PIN['protocol'] and len(manifest['files'])==PIN['selected_objects']
    assert sum(m['bytes'] for m in manifest['files'].values())==PIN['bytes']
    inventory=[]
    for i,(name,meta) in enumerate(manifest['files'].items(),1):
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        assert not name.endswith(('.zip','.npz','.sqlite','.raw'))
        assert file_meta(saved.root/name)=={k:meta[k] for k in ('bytes','sha256')},name
        inventory.append(dict(file=name,bytes=meta['bytes'],sha256=meta['sha256'],verified=True))
        if i%1000==0:print('Input byte/SHA audit',i,'/',len(manifest['files']),flush=True)
    pd.DataFrame(inventory).to_csv(out/'source_file_inventory.csv',index=False)
    source=read(saved.root/'source/FILES.json')
    for name,meta in source.items():assert file_meta(saved.root/'source'/name)==meta,name
    # Protocol is the original canonical contract digest, independent of notebook outputs.
    assert digest(saved.contract)==PIN['protocol']
    pointer=read(saved.root/'POINTER.json');assert pointer['unit']==PIN['expected_unit']
    state=torch.load(saved.root/pointer['state'],map_location='cpu',weights_only=False)
    rows=[];families=[]
    for family,spec in saved.schema.items():
        names=[c['name'] for c in spec['columns']];total=0
        dtypes={c['name']:('str' if c['dtype']=='utf8' else 'UInt64' if c['dtype']=='uint64' else 'Int64' if c['dtype']=='int64' else 'Int8' if c['dtype']=='bool' else 'float64') for c in spec['columns']}
        for path in saved.paths(family):
            count=0
            for frame in pd.read_csv(path,chunksize=131072,dtype=dtypes):
                assert list(frame.columns)==names,(family,path)
                for col in spec['columns']:
                    if col['dtype']=='utf8':continue
                    values=frame[col['name']].dropna().to_numpy(dtype='float64')
                    assert np.isfinite(values).all(),(family,col['name'])
                    if col['dtype']=='bool':assert np.isin(values,[0,1]).all()
                    elif col['dtype'] in ('int64','uint64'):
                        assert np.equal(values,np.floor(values)).all()
                        if col['dtype']=='uint64':assert (values>=0).all()
                count+=len(frame)
            total+=count;rows.append(dict(file=path.relative_to(saved.root).as_posix(),family=family,rows=count,columns=len(names),schema_valid=True))
        assert total==state['rows'][family],(family,total,state['rows'][family])
        families.append(dict(family=family,rows=total,columns=len(names),files=len(saved.paths(family)),state_count_exact=True))
        print('CSV family checked:',family,total,'rows',flush=True)
    other=[name for name in manifest['files'] if name.endswith('.csv') and not name.startswith(('csv/','active/'))]
    for name in other:
        frame=pd.read_csv(saved.root/name);rows.append(dict(file=name,family='source_metadata',rows=len(frame),columns=len(frame.columns),schema_valid=True))
    pd.DataFrame(rows).to_csv(out/'source_CSV_inventory.csv',index=False)
    pd.DataFrame(families).to_csv(out/'source_family_inventory.csv',index=False)
    controls=[]
    indexed=set(saved.table('native_chunks_index').file)
    for name in manifest['files']:
        if name.endswith('.npy') and name not in indexed:
            a=np.load(saved.root/name,allow_pickle=False);assert np.isfinite(a).all()
            controls.append(dict(file=name,shape=list(a.shape),dtype=str(a.dtype),finite=True))
    assert len(controls)==2
    result=dict(status='passed',revision=PIN['revision'],protocol=PIN['protocol'],objects=len(inventory),bytes=sum(r['bytes'] for r in inventory),
                CSV_files=len(rows),CSV_rows=sum(r['rows'] for r in rows),CSV_families=len(families),CSV_fields=sum(r['columns'] for r in families),
                all_bytes_SHA_verified=True,all_CSV_schemas_and_saved_row_counters_verified=True,native_controls=controls,
                image_model_forwards=0,optimizer_updates=0)
    atomic(out/'FILE_VERIFICATION.json',result);return result
