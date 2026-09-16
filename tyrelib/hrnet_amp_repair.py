"""Checkpoint-compatible numerical repair. Original experiment source hashes stay frozen.

Only optimizer-step precision recovery changes. Each run publishes this file and
its hash explicitly; the old smoke test is not claimed to validate the new branch.
"""
import argparse
import math
import os
from pathlib import Path
import torch
import hrnet_runtime as h
import hrnet_protocol as p

REVISION='amp-same-batch-retry-2026-09-15-r1'
EVENTS=[]
def step(model,opt,scaler,batch):
    start_rng=h.rng()
    device=batch[0].device.type
    # Eight AMP attempts, then one FP32 forward/backward. No silent skipped batches.
    for attempt in range(9):
        full_precision=attempt==8
        h.set_rng(start_rng);model.training_mode();opt.zero_grad(set_to_none=True)
        with torch.autocast(device,dtype=torch.float16,enabled=not full_precision):
            logits=model(batch[0]);loss=h.coordinate_loss(logits,batch[1])
        scaled=scaler.scale(loss)  # Initialise scaler even if forward is nonfinite.
        finite=bool(torch.isfinite(loss))
        if finite:
            scaled.backward();scaler.unscale_(opt)
            finite=all(bool(torch.isfinite(v.grad).all()) for v in model.parameters() if v.grad is not None)
            if finite:
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                finite=bool(torch.isfinite(norm))
        if finite:
            scaler.step(opt);scaler.update()
            if attempt:print(f'[AMP repair] batch completed after {attempt} retry/retries; FP32={full_precision}',flush=True)
            return float(loss.detach())
        opt.zero_grad(set_to_none=True)
        before=float(scaler.get_scale())
        if full_precision:
            EVENTS.append(dict(status='failed_in_fp32',scale=before))
            h.set_rng(start_rng)
            raise RuntimeError('Nonfinite loss/gradient persists after bounded AMP retries and FP32 fallback; prior checkpoint retained')
        after=max(before*.5,1e-8)
        scaler.update(new_scale=after)
        EVENTS.append(dict(status='retry_same_batch',attempt=attempt+1,scale_before=before,scale_after=after))
        print(f'[AMP repair] nonfinite AMP computation: scale {before:g} -> {after:g}; retry SAME batch, no optimizer update',flush=True)
    raise AssertionError('unreachable')

def regression(device):
    """Real GradScaler test with injected overflow; no experiment model/data needed."""
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.w=torch.nn.Parameter(torch.zeros(1,6,16,device=device))
        def training_mode(self):self.train()
        def forward(self,x):return self.w.expand(x.shape[0],-1,-1)+x.sum()*0
    model=Toy();opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
    scaler=torch.amp.GradScaler(device,init_scale=128.)
    remaining=[1];before=model.w.detach().clone();optimizer_calls=[0]
    original=opt.step
    def counted(*a,**kw):optimizer_calls[0]+=1;return original(*a,**kw)
    opt.step=counted
    def corrupt(g):
        if remaining[0]:remaining[0]-=1;return torch.full_like(g,float('inf'))
        return g
    handle=model.w.register_hook(corrupt)
    loss=step(model,opt,scaler,(torch.zeros(2,1,device=device),torch.full((2,6),.4,device=device)))
    handle.remove()
    assert optimizer_calls[0]==1 and scaler.get_scale()==64 and math.isfinite(loss)
    assert not torch.equal(before,model.w) and int(opt.state[model.w]['step'])==1
    return dict(status='passed',device=device,optimizer_updates=1,retried_same_batch=True,scale_after=float(scaler.get_scale()))

def install(work):
    # Restrict the compatibility repair to the exact previously published contract.
    cfg=p.contract(work)
    assert p.sha(p.canonical(cfg))=='351c6638783996f46cf9efd99ec6689de726e045288054f240193cea61385712'
    original_publish=p.publish
    def publish(folder,path,token):
        folder=Path(folder)
        p.write(folder/'RUNTIME_REPAIR.json',dict(revision=REVISION,source_sha256=p.file_sha(__file__),
            original_protocol=p.sha(p.canonical(cfg)),policy='same batch AMP backoff; at most 8 AMP attempts then FP32; never count skipped update',
            events_this_process=EVENTS,source_smoke_scope='original model/resume test; repair regression recorded separately',
            regression=SELFTEST))
        (folder/'hrnet_amp_repair.py').write_bytes(Path(__file__).read_bytes())
        return original_publish(folder,path,token)
    h.step=step;p.publish=publish

SELFTEST={}
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--work',required=True);parser.add_argument('--root',default='')
    args=parser.parse_args()
    SELFTEST=regression('cuda')
    EVENTS.clear()  # Synthetic regression events are not real training overflows.
    print('AMP recovery regression passed on GPU; now resuming the original experiment.',flush=True)
    install(args.work)
    h.train(args.work,p.root_data(args.root),os.environ['HF_TOKEN'])
