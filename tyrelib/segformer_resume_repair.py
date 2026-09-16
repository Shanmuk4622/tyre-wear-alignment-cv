"""Deterministic bilinear backward adapter; original data/protocol remain frozen."""
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import segformer_matched as s

REVISION='segformer-bilinear-resume-r1'
ORIGINAL=F.interpolate

def weights(n,m,device):
    pos=((torch.arange(m,device=device,dtype=torch.float32)+.5)*n/m-.5).clamp(0,n-1)
    lo=pos.floor().long(); hi=(lo+1).clamp_max(n-1); frac=pos-lo
    # No scatter-add or atomic reduction: each output row is independent.
    cols=torch.arange(n,device=device)
    return (cols[None,:]==lo[:,None])*(1-frac[:,None])+(cols[None,:]==hi[:,None])*frac[:,None]

class Bilinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,size):
        ctx.input_hw=x.shape[-2:];ctx.dtype=x.dtype
        # Native forward, but FP32 avoids half-precision interpolation accumulation.
        return ORIGINAL(x.float(),size=size,mode='bilinear',align_corners=False).to(x.dtype)
    @staticmethod
    def backward(ctx,grad):
        ih,iw=ctx.input_hw;oh,ow=grad.shape[-2:]
        with torch.autocast(grad.device.type,enabled=False):
            wy=weights(ih,oh,grad.device);wx=weights(iw,ow,grad.device)
            result=torch.matmul(torch.matmul(wy.T,grad.float()),wx)
        return result.to(ctx.dtype),None

def interpolate(input,size=None,scale_factor=None,mode='nearest',align_corners=None,
                recompute_scale_factor=None,antialias=False):
    if (mode=='bilinear' and input.requires_grad and torch.is_grad_enabled()
            and align_corners is False and size is not None and scale_factor is None and not antialias):
        size=(size,size) if isinstance(size,int) else tuple(size)
        return Bilinear.apply(input,size)
    return ORIGINAL(input,size=size,scale_factor=scale_factor,mode=mode,align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,antialias=antialias)

def install():
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8', 'Set deterministic cuBLAS environment before launch'
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    F.interpolate=interpolate

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['train'])
    parser.add_argument('--work',required=True);parser.add_argument('--root',default='');a=parser.parse_args()
    token=os.environ['HF_TOKEN'];cfg=s.contract()
    assert s.p.sha(s.p.canonical(cfg))=='8013bf1e6418e6f884a353efa7ce892c31fbaac450f9d7caede636d9e61e2a18'
    root=s.p.root_data(a.root);s.bind(cfg,root);install()
    previous=s.p.publish
    def publish(folder,path,token):
        folder=Path(folder)
        s.p.write(folder/'RESUME_REPAIR.json',dict(revision=REVISION,source_sha256=s.p.file_sha(__file__),
            policy='FP32 bilinear forward; deterministic separable matrix backward; strict algorithms; original smoke tolerance unchanged',
            original_protocol=s.p.sha(s.p.canonical(cfg)),gpu_validation='see this snapshot smoke STATUS'))
        (folder/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
        return previous(folder,path,token)
    s.p.publish=publish
    print('Deterministic resize repair enabled; checking actual GPU resume before training.',flush=True)
    s.h.smoke(a.work,root,token)
    remaining=[]
    for seed in cfg['seeds']:
        st,_=s.h.pull_status(s.prefix(cfg)+f'/runs/seed{seed}/STATUS.json',token)
        if st and st['status']=='completed':
            assert st['protocol']==s.p.sha(s.p.canonical(cfg)) and st['completed_epochs']==60
            print(f'Seed {seed} already complete; skipping.',flush=True)
        else:remaining.append(seed)
    s.h.train(a.work,root,token,seeds=remaining)
