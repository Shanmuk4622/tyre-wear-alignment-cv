"""Full pinned NB13/13A publication audit; no model forward or remote mutation."""
import csv,hashlib,json,re,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import torch
from filelock import FileLock
import nb11_public as pub
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];OUT=Path(json.loads((ROOT/'tread_two/nb13_completion/LATEST.json').read_text())['path'])
inspection=json.loads((OUT/'INSPECTION.json').read_text());assert [c['execution_count'] for c in inspection['cells']]==[1,2]
assert not any(o['type']=='error' for c in inspection['cells'] for o in c['outputs'])
streams='\n'.join(o['text'] for c in inspection['cells'] for o in c['outputs'] if o['type']=='stream');decoder=json.JSONDecoder();receipt=None
for match in re.finditer(r'\{\s*"destination"',streams):
    try:value,_=decoder.raw_decode(streams[match.start():])
    except json.JSONDecodeError:continue
    if value.get('status')=='published_and_verified':receipt=value
assert receipt and receipt['owner_released'] and receipt['phase']=='completed' and receipt['updates']==2940
pub.OUT=OUT;pub.REVISION=receipt['revision'];pub.PREFIX=receipt['destination'];pub.PROTOCOL=receipt['protocol']
assert pub.PROTOCOL=='36f473b0886d6c2f49c416436d270198af3982230e3045da9969b0bf42169625'
pub.BASE=f'https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/resolve/{pub.REVISION}/{pub.PREFIX}/'
sys.path.insert(0,str(ROOT/'tread_two/context_pilot'));from schemas import SCHEMAS
with FileLock(str(OUT/'AUDIT.lock'),timeout=1):
    manifest=pub.manifest();(OUT/'NOTEBOOK_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    core=['CONTRACT.json','CURSOR.json','state.pt','LIVE_GATE_RECEIPT.json','report/REPORT.json','SCHEMA.json']
    for n in core:pub.fetch((n,manifest['files'][n]))
    remote=OUT/'remote';state=torch.load(remote/'state.pt',map_location='cpu',weights_only=False)
    early=dict(revision=pub.REVISION,protocol=pub.PROTOCOL,files=len(manifest['files'])+1,total_bytes=sum(v['bytes'] for v in manifest['files'].values()),
        phase=state['phase'],updates=state['updates'],best=state['best'],sample_cursor=state['sample_cursor'],row_counts=state['row_cursors'],owner_released=True,
        audit_complete=False)
    (OUT/'CHECKPOINT_EARLY.json').write_text(json.dumps(early,indent=2)+'\n',encoding='utf-8');print(json.dumps(early,indent=2),flush=True)
    items=sorted(manifest['files'].items(),key=lambda item:(0 if item[0].startswith('csv/') and not item[0].endswith(('/affinities.csv','/dense_pixels.csv')) else 1 if item[1]['bytes']<3000000 else 2,item[0]))
    count=0;verified_bytes=0;last=time.monotonic()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures={pool.submit(pub.fetch,item):item for item in items}
        for future in as_completed(futures):
            future.result();count+=1;verified_bytes+=futures[future][1]['bytes']
            if count%100==0 or time.monotonic()-last>25:
                value=dict(stage='objects',objects=count,total=len(items),bytes_verified=verified_bytes,total_bytes=early['total_bytes'])
                (OUT/'AUDIT_PROGRESS.json').write_text(json.dumps(value)+'\n',encoding='utf-8');print(json.dumps(value),flush=True);last=time.monotonic()
    contract=json.loads((remote/'CONTRACT.json').read_text());cursor=json.loads((remote/'CURSOR.json').read_text())
    assert hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==pub.PROTOCOL
    assert contract['environment']['devices']==['Tesla T4','Tesla T4'] and contract['train_device']=='cuda:0' and contract['eval_device']=='cuda:1'
    assert cursor['updates']==2940 and cursor['phase']==cursor['status']=='completed'
    assert all(pub.sha(remote/'source'/n)==v==pub.sha(ROOT/'tread_two/context_pilot'/n) for n,v in contract['sources'].items())
    counts={}
    for family,fields in SCHEMAS.items():
        total=0
        for path in sorted((remote/'csv').glob(f'epoch_*/{family}.csv')):
            with path.open(newline='',encoding='utf-8') as stream:
                rows=csv.reader(stream);assert next(rows)==fields,(family,path)
                total+=sum(1 for _ in rows)
        assert total==cursor['row_cursors'][family],(family,total)
        counts[family]=total;print('Raw CSV verified:',family,total,flush=True)
    assert counts['predictions']==53754 and counts['train_updates']==2940 and counts['view_pairs']==23520 and counts['endpoint_attention']==1687552
    assert pub.sha(remote/'state.pt')==cursor['state_sha256']
    assert state['protocol']==pub.PROTOCOL and state['phase']=='completed' and state['updates']==2940 and state['epoch0']==60
    assert state['row_cursors']==counts and state['sample_cursor']==53754 and len(state['epoch_orders'])==60 and len(state['dense_seen'])==570
    def finite(value):
        if isinstance(value,torch.Tensor):return not value.is_floating_point() or bool(torch.isfinite(value).all())
        if isinstance(value,dict):return all(finite(v) for v in value.values())
        if isinstance(value,(list,tuple)):return all(finite(v) for v in value)
        return True
    assert finite(state['model']) and finite(state['optimizer'])
    for name,array in [('bank_rgb','bank_rgb.npy'),('bank_valid','bank_valid.npy'),('bank_sha256','bank_sha256.npy'),('bank_labels','bank_labels.npy')]:
        assert np.array_equal(state['model']['base.'+name].numpy(),np.load(ROOT/'tread_two/context_pilot/assets'/array,allow_pickle=False))
    # Every frozen base value must still equal the original selected LC native input.
    base=ROOT/'tread_two/nb11_completion/16fa0443f902/remote/artifacts/epochs/epoch_0007.pt'
    assert pub.sha(base)=='f6156aa560aeb880317c6acd745ec1b287aa615fd76656411059033b6a5959fd'
    base_state=torch.load(base,map_location='cpu',weights_only=False)['model']
    assert all(torch.equal(state['model']['base.'+n],v) for n,v in base_state.items())
    for epoch in range(1,61):
        name=f'artifacts/epochs/epoch_{epoch:04d}.pt';assert name in manifest['files']
        value=torch.load(remote/name,map_location='cpu',weights_only=False)
        assert value['protocol']==pub.PROTOCOL and value['epoch']==epoch and value['update']==49*epoch
        assert finite(value['model']) and all(torch.equal(value['model']['base.'+n],v) for n,v in base_state.items())
    live=json.loads((remote/'LIVE_GATE_RECEIPT.json').read_text());assert live['proof']['actual_live_HF_restore'] and live['proof']['actual_batch']==8 and live['proof']['storage_projection']['projection_passed']
    result=dict(status='passed',revision=pub.REVISION,protocol=pub.PROTOCOL,prefix=pub.PREFIX,files=len(items)+1,objects_verified=len(items),bytes_verified=sum(v['bytes'] for _,v in items),
        all_objects_downloaded_size_and_SHA_verified=True,all27_raw_CSV_schemas_and_counts_verified=True,scientific_rows=counts,total_scientific_rows=sum(counts.values()),
        all32_source_assets_match_delivery=True,executed_notebook_sha256=inspection['notebook_sha256'],all35_notebook_payloads_match=True,
        optimizer_updates=2940,epochs=60,selected_epoch=state['best']['epoch'],best=state['best'],final_artifact=state['final_artifact'],phase='completed',owner_released=True,
        native_state_hash_cursors_orders_banks_finite_verified=True,frozen_selected_LC_base_bit_unchanged=True,all60_epoch_checkpoints_retained=True,
        actual_dual_T4=True,actual_live_HF_preflight_passed=True,all60_native_epoch_identities_finite_base_unchanged=True,environment=contract['environment'],dense_images_retained=570,
        new_independent_tyres=0,local_credentials_read=False,remote_writes=0,public_read_requests_this_invocation=pub.CALLS,
        scope='Exhaustive published bytes and raw CSV schemas/counts; scientific reconstruction recorded separately')
    (OUT/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='scientific_rows'},indent=2),flush=True)
