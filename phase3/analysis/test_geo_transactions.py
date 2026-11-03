"""Rotation/crash boundaries and observed quota policies, without any network write."""
import json,sys,time
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tread_two/geo_pilot'))
from common import atomic,cpu,digest,read,seed_all
from store_fit import Store,semantic_hash
from rate_budget import Budget
base=ROOT/'tread_two/geo_validation'/('fixture_transactions_'+str(time.time_ns()));base.mkdir(parents=True)
checks={}
for where in ('after_evidence','after_state','after_pointer'):
    seed_all(23);model=torch.nn.Linear(2,2);opt=torch.optim.AdamW(model.parameters(),lr=.01);store=Store(base/where,'transaction_fixture')
    store.initialize_headers()
    def step(index,crash=None):
        opt.zero_grad(set_to_none=True);model(torch.ones(1,2)).square().sum().backward();opt.step()
        cursor=dict(phase='train',update=index,sample=index*8,last_first_sample=(index-1)*8,epoch=1)
        return store.commit(model,opt,cursor,{'train_updates':[dict(update=index,epoch=1,update_committed=True)]},
                            {'test_values':np.full((8,2),index,dtype='float32')},crash=crash)
    for index in range(1,8):step(index)
    expected_model=semantic_hash(model.state_dict());expected_opt=semantic_hash(opt.state_dict())
    try:step(8,where);raise AssertionError('Crash not injected')
    except InterruptedError:pass
    cursor=store.restore(model,opt)
    if where=='after_pointer':assert cursor['update']==8
    else:
        assert cursor['update']==7 and semantic_hash(model.state_dict())==expected_model and semantic_hash(opt.state_dict())==expected_opt
        step(8)
    values=np.load(store.root/'native/test_values/00000.npy',allow_pickle=False)
    assert values.shape==(64,2) and all(np.all(values[8*(i-1):8*i]==i) for i in range(1,9))
    checks[where]='prior_or_new_complete_state_restored_across_64_sample_seal'
policy=Budget(base/'budget')
now=time.time()
policy.observe('api',SimpleNamespace(headers={'RateLimit-Policy':'"fixed window";"api";q=1000;w=300','RateLimit':'"api";r=999;t=120'},status_code=200))
state=read(policy.path);state['api']['times']=[now]*96;atomic(policy.path,state)
with patch('time.sleep',side_effect=AssertionError('Unexpected old96/API-window wait')):policy.reserve('api')
assert read(policy.path)['api']['effective_limit']==800
policy.observe('api',SimpleNamespace(headers={'RateLimit-Policy':'"fixed window";"api";q=100;w=300','RateLimit':'"api";r=99;t=120'},status_code=200))
state=read(policy.path);state['api']['times']=[];atomic(policy.path,state);policy.reserve('api')
assert read(policy.path)['api']['effective_limit']==80
checks['observed_API_headers']='defaults_then80percent_server_quota_under800_ceiling'
state=read(policy.path);state['repo_commit']={'times':[time.time()]*96,'blocked_until':0};atomic(policy.path,state)
class WaitRequired(Exception):pass
try:
    with patch('time.sleep',side_effect=WaitRequired):policy.reserve('repo_commit')
    raise AssertionError('96/hour commit guard bypassed')
except WaitRequired:pass
checks['commit_guard']='96_per_hour_unchanged'
report=dict(status='passed',checks=checks,remote_calls=0)
atomic(ROOT/'tread_two/geo_validation/TRANSACTIONS.json',report);print(json.dumps(report,indent=2))
