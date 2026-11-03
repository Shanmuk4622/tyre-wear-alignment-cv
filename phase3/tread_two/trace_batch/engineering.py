"""Real-architecture numerical and leakage gates; no eligible study checkpoint."""
import numpy as np
import torch
from common import atomic,seed_all
from model import Trace
from targets import tiles,choose_hidden
from objective import diversity

def check(cache,device,output):
    seed_all(1);ids=[r['slot'] for r in cache.rows if r['role']=='train'][:8]
    b=cache.batch(ids,device);rgb=tiles(b['rgb']);valid=tiles(b['valid'].float());hidden=choose_hidden(valid,range(8))
    model=Trace().to(device).eval();factors=rgb.new_tensor([[.9,1.1,.95]]).expand(8,-1)
    with torch.no_grad():
        first=model(rgb,valid,hidden,factors);changed=model(torch.where(hidden[...,None,None,None],1-rgb,rgb),valid,hidden,factors)
    assert torch.equal(first['masked_prediction'],changed['masked_prediction']),'Hidden raw pixels leaked into masked prediction'
    assert sum(p.numel() for p in model.parameters())==71022
    _,_,available=diversity(torch.randn(8,6,device=device),torch.tensor([True]*6+[False]*2,device=device));assert not available
    _,_,available=diversity(torch.randn(8,6,device=device),torch.tensor([True]*7+[False],device=device));assert available
    probe=torch.zeros(1,32,48,device=device);probe[0,0,0]=1;mass=torch.ones(1,32,device=device)/32
    assert not torch.equal(model.spatial_pool(probe,mass),model.spatial_pool(probe.reshape(1,8,4,48).flip(1).reshape_as(probe),mass))
    result=dict(status='passed',train_images=8,hidden_raw_pixel_prediction_exact=True,parameter_count=71022,rank_guard_six_disabled_seven_enabled=True,
                spatial_reversal_sensitive=True,optimizer_updates=0,device=device,actual_dual_T4=torch.cuda.device_count()==2 and all('T4' in torch.cuda.get_device_name(i) for i in range(2)))
    atomic(output,result);del model,first,changed
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return result
