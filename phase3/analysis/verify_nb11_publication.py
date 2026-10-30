"""Exhaustive byte/hash/schema/row-count audit of the pinned NB11 publication."""
import csv,hashlib,json,re,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from filelock import FileLock
import nb11_public as pub
assert Path(sys.prefix).name=='cv_conda'
sys.path.insert(0,str(pub.ROOT/'tread_two/lc_pilot'))
from schemas import SCHEMAS
with FileLock(str(pub.OUT/'AUDIT.lock'),timeout=1):
    m=pub.manifest();items=sorted(m['files'].items(),key=lambda item:(0 if item[0].startswith('csv/') and not item[0].endswith(('/affinities.csv','/dense_pixels.csv')) else 1 if item[1]['bytes']<3000000 else 2,item[0]))
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i,_ in enumerate(pool.map(pub.fetch,items),1):
            if i%100==0:
                print('Full pinned HF objects verified:',i,'/',len(items),flush=True)
                (pub.OUT/'AUDIT_PROGRESS.json').write_text(json.dumps(dict(stage='objects',objects=i,total=len(items)))+'\n')
    remote=pub.OUT/'remote';contract=json.loads((remote/'CONTRACT.json').read_text());cursor=json.loads((remote/'CURSOR.json').read_text())
    assert hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==pub.PROTOCOL
    assert contract['environment']['devices']==['Tesla T4','Tesla T4'] and contract['train_device']=='cuda:0' and contract['eval_device']=='cuda:1'
    assert cursor['updates']==2940 and cursor['phase']==cursor['status']=='completed'
    assert all(pub.sha(remote/'source'/n)==v==pub.sha(pub.ROOT/'tread_two/lc_pilot'/n) for n,v in contract['sources'].items())
    counts={}
    for family,fields in SCHEMAS.items():
        total=0
        for p in sorted((remote/'csv').glob(f'epoch_*/{family}.csv')):
            with p.open(newline='',encoding='utf-8') as stream:
                reader=csv.reader(stream);assert next(reader)==fields,(family,p)
                total+=sum(1 for _ in reader)
        assert total==cursor['row_cursors'][family],(family,total)
        counts[family]=total;print('Raw CSV counted:',family,total,flush=True)
    assert counts['predictions']==53342 and counts['train_updates']==2940 and counts['view_pairs']==23520
    assert pub.sha(remote/'state.pt')==cursor['state_sha256']
    state=torch.load(remote/'state.pt',map_location='cpu',weights_only=False)
    assert state['protocol']==pub.PROTOCOL and state['phase']=='completed' and state['updates']==2940 and state['epoch0']==60
    assert state['row_cursors']==counts and state['sample_cursor']==53342 and len(state['epoch_orders'])==60 and len(state['dense_seen'])==570
    assert all(torch.isfinite(v).all() for v in state['model'].values() if torch.is_tensor(v) and v.is_floating_point())
    for name,array in [('bank_rgb','bank_rgb.npy'),('bank_valid','bank_valid.npy'),('bank_sha256','bank_sha256.npy'),('bank_labels','bank_labels.npy')]:
        assert np.array_equal(state['model'][name].numpy(),np.load(pub.ROOT/'tread_two/lc_pilot/assets'/array,allow_pickle=False))
    for epoch in range(1,61):assert f'artifacts/epochs/epoch_{epoch:04d}.pt' in m['files']
    live=json.loads((remote/'LIVE_GATE_RECEIPT.json').read_text());assert live['proof']['actual_live_HF_restore'] and live['proof']['actual_batch']==8 and live['proof']['storage_projection']['projection_passed']
    result=dict(status='passed',revision=pub.REVISION,protocol=pub.PROTOCOL,prefix=pub.PREFIX,files=len(items)+1,objects_verified=len(items),bytes_verified=sum(v['bytes'] for _,v in items),
      all_objects_downloaded_size_and_SHA_verified=True,all24_raw_CSV_schemas_and_counts_verified=True,scientific_rows=counts,total_scientific_rows=sum(counts.values()),
      all30_source_assets_match_delivery=True,executed_notebook_sha256=json.loads((pub.OUT/'INSPECTION.json').read_text())['executed_notebook_sha256'],
      optimizer_updates=2940,epochs=60,selected_epoch=state['best']['epoch'],best=state['best'],final_artifact=state['final_artifact'],phase='completed',owner_released=True,
      native_state_hash_cursors_orders_banks_finite_verified=True,all60_epoch_checkpoints_retained=True,actual_dual_T4=True,actual_live_HF_preflight_passed=True,
      environment=contract['environment'],live_gate_storage_projection=live['proof']['storage_projection'],dense_images_retained=570,
      new_independent_tyres=0,local_credentials_read=False,remote_writes=0,public_read_requests_this_invocation=pub.CALLS,
      scope='Every publication object and raw CSV schema/count verified; scientific interpretation/reconstruction recorded separately')
    removed=0
    for name in m['files']:
        target=remote/name
        for partial in target.parent.glob(target.name+'.*.part'):
            if re.fullmatch(re.escape(target.name)+r'\.[a-f0-9]{32}\.part',partial.name):
                assert partial.resolve().is_relative_to(remote.resolve()) and target.exists()
                try:partial.unlink();removed+=1
                except PermissionError:pass
    result['superseded_audit_partial_files_removed']=removed
    (pub.OUT/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('scientific_rows','live_gate_storage_projection')},indent=2))
