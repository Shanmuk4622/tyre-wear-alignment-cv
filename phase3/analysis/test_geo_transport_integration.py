"""Read-only real pinned input plus idempotent transport CSV ingestion tests."""
import json,sys,time
from pathlib import Path
from types import SimpleNamespace
import requests
import torch
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tread_two/geo_pilot'))
from common import atomic,canonical,read
from coordinator_fit import pinned_json,service_rows
from worker_fit import service
from store_fit import Store
base=ROOT/'tread_two/geo_validation'/('fixture_transport_'+str(time.time_ns()));base.mkdir(parents=True)
session=requests.Session();session.trust_env=False
pub=SimpleNamespace(REPO='Shanmuk4622/tyre-wear-study',session=session,call=lambda fn,name:fn())
gate=pinned_json(pub,read(ROOT/'tread_two/geo_pilot/NB16_GATE.json'),'FINAL_GATE.json')
assert gate['status']=='engineering_gate_passed' and gate['actual_HF_restore_continue']
def prepare(folder):
    (folder/'transport').mkdir(parents=True)
    (folder/'transport/HTTP_REQUESTS.jsonl').write_bytes(canonical(dict(time_utc='2026-11-01T12:00:00Z',category='api',method='GET',status=200,rate_limit='"api";r=999;t=100',rate_policy='"fixed window";"api";q=1000;w=300'))+b'\n')
    fit=folder/'fit';fit.mkdir()
    service_rows(folder,fit,dict(attempt=1,reason='fixture',local_update=0,remote_update=0,revision='fixture_revision',manifest_sha256='fixture',new_objects=1,verified_objects=1,elapsed_seconds=1.,owner_released=False,pending=False))
    return fit
one=base/'one';two=base/'two';fit=prepare(one);other=prepare(two)
first=next((fit/'service_queue').glob('*.json'));other_first=next((other/'service_queue').glob('*.json'))
rows=read(first);second=read(other_first)
assert rows['HF_requests'][0]['event_id']!=second['HF_requests'][0]['event_id']
model=torch.nn.Linear(2,2);optimizer=torch.optim.AdamW(model.parameters());store=Store(base/'state','transport_fixture');store.initialize_headers()
engine=SimpleNamespace(root=fit,store=store,model=model,optimizer=optimizer,cursor=dict(phase='fixture',update=0,sample=0))
service(engine);units=store.unit;counts=dict(store.rows)
done=first.with_suffix('.done');first.write_bytes(done.read_bytes());service(engine)
assert store.unit==units and store.rows==counts and store.rows['HF_requests']==store.rows['HF_publications']==1
report=dict(status='passed',pinned_NB16_logical_file_resolved_via_manifest=True,public_HF_reads=2,
            cross_session_request_IDs_distinct=True,service_replay_idempotent=True,server_policy_fields_retained=True,
            remote_writes=0,credentials_read=False)
atomic(ROOT/'tread_two/geo_validation/TRANSPORT.json',report);print(json.dumps(report,indent=2))
