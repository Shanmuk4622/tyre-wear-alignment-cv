"""Verify pinned diagnostic control/native state; no fitting or remote writes."""
import datetime,json,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import torch
import nb11_public as client
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(json.loads((ROOT/'tread_two/context_validation/NB13_STATUS_LATEST.json').read_text())['path'])
status=json.loads((OUT/'PUBLIC_STATUS.json').read_text());run=next(r for r in status['runs'] if '/fit/' in r['prefix'])
client.OUT=OUT;client.REVISION=status['revision'];client.PREFIX=run['prefix'];client.PROTOCOL=run['prefix'].split('/')[-1]
client.BASE=f'https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/resolve/{client.REVISION}/{client.PREFIX}/'
m=json.loads((OUT/('fit_'+client.PROTOCOL[:12]+'_MANIFEST.json')).read_text())
names=['state.pt','CONTRACT.json','CURSOR.json','POINTER.json','LIVE_GATE_RECEIPT.json','CAPACITY.json','REMOTE_HISTORY.jsonl','HTTP_REQUESTS.jsonl','LIFECYCLE.jsonl','STATE_VERSIONS.csv','COMMIT_TIMINGS.csv','report/REPORT.json']
train=sorted(n for n in m['files'] if n.endswith('/train_updates.csv'));names+=train[-3:]
with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(client.fetch,[(n,m['files'][n]) for n in names]))
remote=OUT/'remote';state=torch.load(remote/'state.pt',map_location='cpu',weights_only=False)
cfg=json.loads((remote/'CONTRACT.json').read_text());cursor=json.loads((remote/'CURSOR.json').read_text());gate=json.loads((remote/'LIVE_GATE_RECEIPT.json').read_text())
assert state['protocol']==client.PROTOCOL and len(state['row_cursors'])==27
assert __import__('hashlib').sha256(json.dumps(cfg,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()==client.PROTOCOL
assert cfg['sources']=={n:v['sha256'] for n,v in json.loads((ROOT/'tread_two/context_pilot/FILES.json').read_text()).items()}
owner=run['OWNER.json'];expires=datetime.datetime.fromtimestamp(owner['expires'],datetime.timezone.utc)
def finite(value):
    if isinstance(value,torch.Tensor):return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value,dict):return all(finite(v) for v in value.values())
    if isinstance(value,(list,tuple)):return all(finite(v) for v in value)
    return True
assert finite(state['model']) and finite(state['optimizer'])
result=dict(status='verified_partial_checkpoint',checked_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),revision=client.REVISION,protocol=client.PROTOCOL,
    verified_objects=len(names),verified_bytes=sum(m['files'][n]['bytes'] for n in names),native_SHA256=m['files']['state.pt']['sha256'],
    saved_updates=state['updates'],planned_updates=2940,phase=state['phase'],epoch0=state['epoch0'],within_epoch_cursor=state['cursor'],unit_seq=state['seq'],sample_cursor=state['sample_cursor'],
    source_inventory_matches=True,all_native_model_and_optimizer_tensors_finite=finite(state['model']) and finite(state['optimizer']),
    actual_gate=gate,cursor=cursor,row_counts=state['row_cursors'],owner_expiry_UTC=expires.isoformat(),owner_expired=expires<datetime.datetime.now(datetime.timezone.utc),
    saved_notebook_last_local_update=2467,full_snapshot_all798_objects_downloaded=False,remote_writes=0,credentials_read=False)
(OUT/'CHECKPOINT_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2),flush=True)
for name in ['REMOTE_HISTORY.jsonl','HTTP_REQUESTS.jsonl','LIFECYCLE.jsonl']:
    print(name, '\n'.join((remote/name).read_text().splitlines()[-6:]),flush=True)
