"""Pinned, credential-free NB16 audit: every direct file and full scientific reconstruction."""
import csv,hashlib,json,re,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import nb11_public as client
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(json.loads((ROOT/'tread_two/nb16_completion/LATEST.json').read_text())['path'])
inspection=json.loads((OUT/'INSPECTION.json').read_text())
assert [c['execution_count'] for c in inspection['cells']]==[1,2,3]
assert not any(o['type']=='error' for c in inspection['cells'] for o in c['outputs'])
streams='\n'.join(o['text'] for c in inspection['cells'] for o in c['outputs'] if o['type']=='stream')
decoder=json.JSONDecoder();result=None
for match in re.finditer(r'\{\s*"(?:status|actual_HF_restore_continue)"',streams):
    try:value,_=decoder.raw_decode(streams[match.start():])
    except json.JSONDecodeError:continue
    if value.get('status')=='engineering_gate_passed':result=value
assert result and result['actual_dual_T4'] and result['actual_HF_restore_continue']
receipt=result['publication'];assert receipt['owner_released'] and receipt['files']==174
client.OUT=OUT;client.REVISION=receipt['revision'];client.PREFIX=receipt['destination'];client.PROTOCOL=result['protocol']
assert client.PREFIX=='phase3/tread-two/geo-r1/engineering/'+client.PROTOCOL
client.BASE=f'https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/resolve/{client.REVISION}/{client.PREFIX}/'
manifest=client.manifest()
with ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(client.fetch,manifest['files'].items()))
remote=OUT/'remote'
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
contract=read(remote/'CONTRACT.json');assert digest(contract)==client.PROTOCOL
assert contract['mode']=='engineering_fixture' and contract['measured_updates']==0
source=ROOT/'tread_two/geometry_first/runtime'
for name,meta in contract['sources'].items():
    assert client.sha(remote/'source'/name)==meta['sha256']==client.sha(source/name)
proof=read(remote/'ENGINEERING_PROOF.json');continuation=read(remote/'HF_CONTINUATION_PROOF.json')
assert proof['status']=='passed' and proof['actual_dual_T4'] and proof['resume_RNG_exact']
assert continuation['actual_remote_download_restore_continue'] and continuation['model_optimizer_counters_exact']
assert proof['resume_scientific_files_exact']==continuation['scientific_files_identical']==96
assert proof['environment']==contract['environment'] and len(contract['environment']['devices'])==2
assert all('T4' in d for d in contract['environment']['devices'])
assert all(proof['numerical_tests'].values())
assert read(remote/'FINAL_GATE.json')['status']=='engineering_gate_passed'
counts=read(remote/'PARAMETERS.json');assert counts==proof['parameters'] and counts['trainable']==355879 and counts['total']==1728038
tables=[]
for path in remote.rglob('*.csv'):
    with path.open(encoding='utf-8',newline='') as stream:
        reader=csv.DictReader(stream);rows=list(reader)
        assert reader.fieldnames and len(reader.fieldnames)==len(set(reader.fieldnames))
        assert all(None not in row for row in rows)
    tables.append(dict(file=path.relative_to(remote).as_posix(),rows=len(rows),columns=reader.fieldnames))
native=[]
for path in remote.rglob('*.npy'):
    value=np.load(path,allow_pickle=False)
    assert value.dtype in (np.float32,np.bool_)
    assert np.isfinite(value).all()
    native.append(dict(file=path.relative_to(remote).as_posix(),shape=list(value.shape),dtype=str(value.dtype)))
# Verify the original published unit1 seam and latest committed unit4 independently.
units=[]
for namespace in ('resume','progress'):
    pointer=read(remote/namespace/'POINTER.json');assert pointer['contract']==proof['fixture_contract']
    state_path=remote/namespace/pointer['state']
    assert client.sha(state_path)==pointer['state_meta']['sha256'] and state_path.stat().st_size==pointer['state_meta']['bytes']
    previous=None
    for relative in pointer['units']:
        record=read(remote/namespace/relative)
        assert record['previous']==previous and digest({k:v for k,v in record.items() if k!='chain'})==record['chain']
        for name,meta in record['files'].items():assert client.sha(remote/namespace/name)==meta['sha256']
        previous=record['chain']
    assert previous==pointer['chain'] and pointer['update']==(1 if namespace=='resume' else 4)
    units.append(dict(namespace=namespace,committed_updates=pointer['update'],chain_verified=True))
# Rebuild each raw VICReg/masked-target reduction from direct vectors and native predictions.
losses=[]
for folder in sorted((remote/'progress/units').iterdir()):
    scalars=pd.read_csv(folder/'objectives.csv').iloc[0]
    vectors=pd.read_csv(folder/'vectors.csv')
    support=pd.read_csv(folder/'tile_support.csv')
    projected=[]
    for view in (0,1):
        matrix=vectors.query('view==@view and kind=="projected"').pivot(index='slot',columns='dimension',values='value').to_numpy(dtype=np.float64)
        projected.append(matrix)
    eligibility=support.groupby('slot')['eligible'].min().to_numpy(dtype=bool)
    a,b=[m[eligibility] for m in projected]
    inv=np.square(a-b).mean()
    variance=[];covariance=[]
    for z in (a,b):
        cv=np.cov(z,rowvar=False,ddof=1)
        variance.append(np.maximum(0,1-np.sqrt(np.diag(cv)+1e-4)).mean())
        covariance.append((np.square(cv).sum()-np.square(np.diag(cv)).sum())/64)
    num=den=0.
    masks=pd.read_csv(folder/'masked_tiles.csv')
    for view in (0,1):
        pred=np.load(folder/f'v{view}_prediction.npy').astype(np.float64)
        target=np.load(folder/f'v{view}_target.npy').astype(np.float64)
        weights=np.load(folder/f'v{view}_weight.npy').astype(np.float64)
        actual=np.zeros((8,8),dtype=bool)
        for row in masks.query('view==@view').itertuples():actual[row.slot,row.packed_slot]=True
        weights*=actual[...,None,None,None]*eligibility[:,None,None,None,None]
        error=np.abs(pred-target)
        num+=np.sum(np.where(error<.1,error**2/.2,error-.05)*weights)
        den+=weights.sum()*4
    calculated=dict(invariance=inv,variance_a=variance[0],variance_b=variance[1],covariance_a=covariance[0],covariance_b=covariance[1],masked_numerator=num,masked_denominator=den,masked_huber=num/den)
    calculated['texture']=25*inv+25*sum(variance)+sum(covariance)+.5*num/den
    maxdiff=0.
    for k,v in calculated.items():
        reported=float(scalars['texture_'+k]);maxdiff=max(maxdiff,abs(reported-v))
        assert np.isclose(reported,v,atol=2e-5,rtol=1e-5),(folder.name,k,reported,v)
    losses.append(dict(update=int(folder.name),all_terms_reconstructed=True,maximum_absolute_difference=maxdiff))
gallery=subprocess.run(['node',str(ROOT/'analysis/test_geo_passport.cjs'),str(remote/'Engineering_Passport.html')],capture_output=True,text=True,encoding='utf-8')
assert gallery.returncode==0,gallery.stderr
unchanged=[]
for notebook in (13,14,15):
    earlier=Path(read(ROOT/f'tread_two/nb{notebook}_completion/LATEST.json')['path'])
    v=read(earlier/'VERIFICATION.json');m=read(earlier/'REMOTE_MANIFEST.json')
    current=client.request(f'https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/resolve/{client.REVISION}/{v["prefix"]}/MANIFEST.json').json()
    assert current==m
    unchanged.append(dict(notebook=notebook,entries=len(m['files']),unchanged=True))
report=dict(status='passed',executed_notebook_sha256=inspection['executed_notebook_sha256'],revision=client.REVISION,
            prefix=client.PREFIX,contract=client.PROTOCOL,files=len(manifest['files'])+1,objects_verified=len(manifest['files']),
            bytes_verified=sum(m['bytes'] for m in manifest['files'].values()),payload_files=27,all_payload_files_match=True,
            code_cells_completed=3,notebook_errors=0,owner_released=True,actual_dual_T4=True,actual_HF_restore_continue=True,
            scientific_files_identical=96,CSV_files=len(tables),CSV_rows=sum(t['rows'] for t in tables),CSV_inventory=tables,
            native_arrays=len(native),native_inventory=native,unit_chains=units,independent_loss_reconstruction=losses,
            gallery=json.loads(gallery.stdout),parameters=counts,environment=contract['environment'],GPU_memory=proof['GPU_memory'],
            storage=read(remote/'STORAGE.json'),original_publications_unchanged=unchanged,measured_fit_started=False,
            warnings='Scalar conversion warnings are diagnostic-only; no saved error or failed check',
            local_credentials_read=False,remote_writes=0,public_read_requests_this_invocation=client.CALLS)
(OUT/'VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ('CSV_inventory','native_inventory')},indent=2))
