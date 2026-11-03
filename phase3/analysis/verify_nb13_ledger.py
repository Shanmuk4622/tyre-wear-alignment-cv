"""Native epoch identities and all committed CSV ledger chunks; no fitting."""
import hashlib,json,sys
from pathlib import Path
import torch
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];OUT=Path(json.loads((ROOT/'tread_two/nb13_completion/LATEST.json').read_text())['path'])
sys.path.insert(0,str(ROOT/'tread_two/context_pilot'));from common import canonical,digest,read,sha
v=read(OUT/'VERIFICATION.json');assert v['status']=='passed'
remote=OUT/'remote';state=torch.load(remote/'state.pt',map_location='cpu',weights_only=False);history=read(remote/'HISTORY.json');pointer=read(remote/'POINTER.json')
assert len(history)==state['seq']+1==3913 and digest(history[-1])==pointer['sha256']
prior_pointer=read(ROOT/'tread_two/context_validation/nb13_status/e0c714be2aba/remote/POINTER.json')
assert digest(history[1606])==prior_pointer['sha256'],'Completed ledger does not extend the original verified update1294'
offsets={};counts={name:0 for name in state['row_cursors']};chunks=0
for seq,unit in enumerate(history):
    assert unit['seq']==seq and unit['protocol']==v['protocol']
    if seq:assert unit['parent_sha256']==digest(history[seq-1])
    else:assert unit['parent'] is unit['parent_sha256'] is None
    for chunk in unit['chunks']:
        path=remote/chunk['path'];assert chunk['start']==offsets.get(chunk['path'],0)
        with path.open('rb') as stream:stream.seek(chunk['start']);body=stream.read(chunk['end']-chunk['start'])
        assert len(body)==chunk['end']-chunk['start'] and hashlib.sha256(body).hexdigest()==chunk['sha256']
        offsets[chunk['path']]=chunk['end'];counts[chunk['family']]+=chunk['rows'];chunks+=1
    assert counts==unit['row_cursors'] and offsets==unit['file_cursors']
    if seq%750==0:print('Ledger units and byte chunks checked:',seq,'/',len(history),flush=True)
assert counts==state['row_cursors'] and offsets==state['file_cursors']
assert all((remote/name).stat().st_size==end for name,end in offsets.items())
updates=[u['updates'] for u in history];assert sorted(set(updates))==list(range(2941)) and all(b-a in (0,1) for a,b in zip(updates,updates[1:]))
base=torch.load(ROOT/'tread_two/nb11_completion/16fa0443f902/remote/artifacts/epochs/epoch_0007.pt',map_location='cpu',weights_only=False)['model']
for epoch in range(1,61):
    x=torch.load(remote/f'artifacts/epochs/epoch_{epoch:04d}.pt',map_location='cpu',weights_only=False)
    assert x['protocol']==v['protocol'] and x['epoch']==epoch and x['update']==49*epoch
    assert all(torch.isfinite(t).all() for t in x['model'].values() if t.is_floating_point())
    assert all(torch.equal(x['model']['base.'+name],t) for name,t in base.items())
result=dict(status='passed',unit_chain=len(history),committed_CSV_chunks=chunks,all_chunk_hashes_and_cumulative_offsets=True,
    all60_native_epoch_protocols_updates_finite_base_unchanged=True,completed_update_sequence_0_to_2940_without_gaps=True,
    partial_original_remote_prefix_units=1607,model_forward_calls=0,optimizer_updates=0,remote_writes=0)
(OUT/'LEDGER_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
v.update(all_unit_and_CSV_chunk_hashes_verified=True,all60_native_epoch_identities_finite_base_unchanged=True)
(OUT/'VERIFICATION.json').write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
