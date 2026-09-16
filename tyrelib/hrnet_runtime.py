"""Single-GPU HRNet geometry. Atomic step checkpoints; one synchronous HF writer."""
import copy
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import time
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
import hrnet_protocol as p

STOP=False
def request_stop(*_):
    global STOP
    STOP=True
    print('Stop requested; finishing current optimizer step and publishing durable state.',flush=True)
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def set_rng(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch']);torch.cuda.set_rng_state_all(r['cuda'])

class Geometry(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        import timm
        assert timm.__version__==cfg['packages']['timm']
        self.backbone=timm.create_model(cfg['model'],pretrained=False,features_only=True,
            feature_location='',out_indices=tuple(cfg['feature_indices']))
        channels=self.backbone.feature_info.channels()
        assert channels==[18,36,72,144],f'Not expected HRNet-W18: {channels}'
        self.projections=nn.ModuleList([nn.Conv2d(c,16,1) for c in channels])
        self.head=nn.Sequential(nn.Conv2d(64,64,3,padding=1),nn.ReLU(),nn.Conv2d(64,6,1))
        ref=cfg['rows'][0]
        self.levels=[y/(ref['height']-1) for y in ref['guide_y']]
        assert all([y/(r['height']-1) for y in r['guide_y']]==self.levels for r in cfg['rows'])
        assert sum(v.numel() for v in self.parameters())==9603962,'Unexpected HRNet-W18 adapter parameter count'
    def forward(self,x):
        maps=self.backbone(x);size=maps[0].shape[-2:]
        z=torch.cat([F.interpolate(proj(m),size=size,mode='nearest') for proj,m in zip(self.projections,maps)],1)
        heat=self.head(z)
        # Fixed rows; coordinate y is known, not a learned physical landmark.
        lines=[]
        for j in range(6):
            pos=(heat.shape[-2]-1)*self.levels[j//2];lo=int(pos);hi=min(lo+1,heat.shape[-2]-1)
            lines.append(heat[:,j,lo,:]*(1-(pos-lo))+heat[:,j,hi,:]*(pos-lo))
        return torch.stack(lines,1)
    def training_mode(self):
        self.train()
        for m in self.modules():
            if isinstance(m,nn.modules.batchnorm._BatchNorm):m.eval()

def pretrained(model,cfg,work,token):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    file=p.retry(lambda:hf_hub_download(cfg['pretrained_repo'],'model.safetensors',
        revision=cfg['pretrained_revision'],token=token,cache_dir=str(Path(work)/'weights')))
    state=load_file(file);needed=model.backbone.state_dict()
    missing=[k for k in needed if k not in state and not k.endswith('num_batches_tracked')]
    assert not missing,f'Pretrained tensor mismatch: {missing[:5]}'
    selected={k:state.get(k,v) for k,v in needed.items()}
    model.backbone.load_state_dict(selected,strict=True)
    return dict(repo=cfg['pretrained_repo'],revision=cfg['pretrained_revision'],sha256=p.file_sha(file),
        channels=model.backbone.feature_info.channels(),parameters=sum(v.numel() for v in model.parameters()),
        tensor_signature=p.sha(p.canonical({k:list(v.shape) for k,v in model.state_dict().items()})))

def load_batch(rows,root,cfg,device='cuda'):
    xx=[];yy=[]
    for r in rows:
        with Image.open(Path(root)/r['original_relative_path']) as image:
            image=image.convert('RGB').resize(tuple(reversed(cfg['input_hw'])),Image.Resampling.BILINEAR)
            x=np.asarray(image,dtype=np.float32).copy()/255
        xx.append(torch.from_numpy(x).permute(2,0,1));yy.append(r['x'])
    x=torch.stack(xx).to(device)
    x=(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
    return x,torch.tensor(yy,dtype=torch.float32,device=device)
def coordinate_loss(logits,targets):
    x=torch.arange(logits.shape[-1],device=logits.device).float()
    target=torch.exp(-.5*((x-targets[...,None]*(logits.shape[-1]-1))/1.5)**2)
    target=target/target.sum(-1,keepdim=True).clamp_min(1e-12)
    return -(target*logits.float().log_softmax(-1)).sum(-1).mean()
def positions(logits):
    return (logits.float().softmax(-1)*torch.linspace(0,1,logits.shape[-1],device=logits.device)).sum(-1)
def step(model,opt,scaler,batch):
    model.training_mode();opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.float16):logits=model(batch[0]);loss=coordinate_loss(logits,batch[1])
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss; prior durable checkpoint retained')
    scaler.scale(loss).backward();scaler.unscale_(opt)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient; prior durable checkpoint retained')
    scaler.step(opt);scaler.update();return float(loss.detach())
def atomic_save(path,state):
    path=Path(path);tmp=path.with_suffix('.tmp');torch.save(state,tmp);os.replace(tmp,path)
def make_state(model,opt,scaler,scheduler,protocol,seed,epoch,cursor,history,identity):
    return dict(model=model.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),
        scheduler=scheduler.state_dict(),rng=rng(),protocol=protocol,seed=seed,
        epoch=epoch,cursor=cursor,history=history,identity=identity)
def restore(state,model,opt,scaler,scheduler):
    model.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer'])
    scaler.load_state_dict(state['scaler']);scheduler.load_state_dict(state['scheduler']);set_rng(state['rng'])
def components(cfg):
    model=Geometry(cfg).cuda();opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    scaler=torch.amp.GradScaler('cuda');scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg['epochs'])
    return model,opt,scaler,scheduler
def metadata(folder,state,status):
    p.write(Path(folder)/'STATUS.json',dict(status=status,protocol=state['protocol'],seed=state['seed'],
        completed_epochs=state['epoch'],next_batch_cursor=state['cursor'],checkpoint_sha256=p.file_sha(Path(folder)/'state.pt'),
        endpoint='fixed_epoch_60',full_s9_complete=False))
    p.write(Path(folder)/'HISTORY.json',state['history'])
def pull_status(path,token):
    from huggingface_hub import HfApi,hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    rev=p.retry(lambda:HfApi(token=token).repo_info(p.REPO,repo_type='dataset').sha)
    try:
        f=p.retry(lambda:hf_hub_download(p.REPO,path,repo_type='dataset',revision=rev,token=token))
    except EntryNotFoundError:return None,rev
    return p.read_json(f),rev
def prerequisite(cfg,name,expected,token):
    result,_=pull_status(p.prefix(cfg)+'/'+name+'/STATUS.json',token)
    if not result or result.get('status')!=expected or result.get('protocol')!=p.sha(p.canonical(cfg)):
        raise RuntimeError(f'Run the matching {name} notebook first; no training started')
    return result
def recover(folder,remote,cfg,seed,token):
    path=Path(folder)/'state.pt';protocol=p.sha(p.canonical(cfg))
    if not path.exists():
        status,rev=pull_status(remote+'/STATUS.json',token)
        if status:
            assert status['protocol']==protocol and status['seed']==seed
            from huggingface_hub import hf_hub_download
            file=p.retry(lambda:hf_hub_download(p.REPO,remote+'/state.pt',repo_type='dataset',revision=rev,token=token,
                local_dir=str(Path(folder)/'restore')))
            assert p.file_sha(file)==status['checkpoint_sha256'],'Published checkpoint hash mismatch'
            shutil.copy2(file,path)
            # Remove only this run's verified temporary restore tree, not user data.
            restore_dir=(Path(folder)/'restore').resolve()
            assert restore_dir.parent==Path(folder).resolve()
            shutil.rmtree(restore_dir)
    if not path.exists():return None
    state=torch.load(path,map_location='cpu',weights_only=False)
    assert state['protocol']==protocol and state['seed']==seed
    assert 0<=state['epoch']<=cfg['epochs'] and state['cursor']>=0
    return state
def evaluate(model,rows,root,cfg):
    model.eval();records=[]
    with torch.inference_mode():
        for r in rows:
            x,y=load_batch([r],root,cfg)
            with torch.autocast('cuda',dtype=torch.float16):pred=positions(model(x))[0].cpu().tolist()
            for i,v in enumerate(pred):
                records.append(dict(pilot_id=r['pilot_id'],tyre=r['session'],point=p.POINTS[i],
                    prediction=v,label=r['x'][i],error_width_fraction=abs(v-r['x'][i]),
                    error_px=abs(v-r['x'][i])*(r['width']-1)))
    return dict(n_images=len(rows),n_points=len(records),coverage=1.0,
        note='Coordinate-only model; coverage is unconditional, not validated visibility detection',
        mean_width_error=float(np.mean([r['error_width_fraction'] for r in records])),
        median_px_error=float(np.median([r['error_px'] for r in records])),records=records)

def smoke(work,root,token):
    cfg=p.contract(work);p.validate_images(cfg,root);prerequisite(cfg,'preflight','preflight_passed',token)
    assert torch.cuda.is_available(),'Use Kaggle T4 GPU for NB27'
    seed_all(1);model,opt,scaler,scheduler=components(cfg);identity=pretrained(model,cfg,work,token)
    protocol=p.sha(p.canonical(cfg));folder=Path(work)/'smoke'/protocol;folder.mkdir(parents=True,exist_ok=True)
    p.write(folder/'IDENTITY.json',identity);p.write(folder/'CONTRACT.json',cfg)
    rows=[r for r in cfg['rows'] if r['role']=='train'][:2]
    batch=load_batch(rows,root,cfg);losses=[]
    for _ in range(2):losses.append(step(model,opt,scaler,batch))
    atomic_save(folder/'resume_test.pt',make_state(model,opt,scaler,scheduler,protocol,1,0,2,[],identity))
    for _ in range(2):losses.append(step(model,opt,scaler,batch))
    expected={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    del model,opt,scaler,scheduler;torch.cuda.empty_cache()
    model,opt,scaler,scheduler=components(cfg)
    saved=torch.load(folder/'resume_test.pt',map_location='cpu',weights_only=False)
    restore(saved,model,opt,scaler,scheduler);del saved
    resumed=[]
    for _ in range(2):resumed.append(step(model,opt,scaler,batch))
    delta=max(float((v.detach().cpu().float()-expected[k].float()).abs().max()) for k,v in model.state_dict().items())
    passed=delta<=1e-5 and np.allclose(losses[2:],resumed,rtol=1e-5,atol=1e-5)
    p.write(folder/'STATUS.json',dict(status='smoke_passed' if passed else 'smoke_failed',protocol=protocol,
        max_parameter_difference=delta,uninterrupted_losses=losses,resumed_losses=resumed,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
        note='4-step continuation test, not proof of convergence/generalisation'))
    # Only the temporary smoke checkpoint is discarded; no training progress.
    (folder/'resume_test.pt').unlink()
    p.publish(folder,p.prefix(cfg)+'/smoke',token)
    if not passed:raise RuntimeError('Resume equivalence failed; training blocked. Send outputs for review.')
    print('SMOKE PASSED: GPU forward/backward, checkpoint load and continuation agreement.',flush=True)

def train(work,root,token,seeds=(1,2,3)):
    global STOP
    STOP=False;signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
    assert torch.cuda.is_available(),'Use Kaggle T4 GPU'
    cfg=p.contract(work);p.validate_images(cfg,root);prerequisite(cfg,'preflight','preflight_passed',token)
    smoke_status=prerequisite(cfg,'smoke','smoke_passed',token)
    if smoke_status.get('torch')!=torch.__version__:
        raise RuntimeError('Torch version differs from the successful smoke run. Rerun NB27 in this environment first.')
    protocol=p.sha(p.canonical(cfg));train_rows=[r for r in cfg['rows'] if r['role']=='train']
    val_rows=[r for r in cfg['rows'] if r['role']=='validation'];test_rows=[r for r in cfg['rows'] if r['role']=='test']
    started=time.monotonic()
    for seed in seeds:
        assert seed in cfg['seeds']
        if STOP:return
        folder=Path(work)/'runs'/protocol/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
        remote=p.prefix(cfg)+f'/runs/seed{seed}'
        if shutil.disk_usage(work).free<2*1024**3:raise RuntimeError('Need at least 2 GiB free workspace before loading next run')
        seed_all(seed);model,opt,scaler,scheduler=components(cfg)
        saved=recover(folder,remote,cfg,seed,token)
        if saved:
            restore(saved,model,opt,scaler,scheduler)
            epoch,cursor,history,identity=saved['epoch'],saved['cursor'],saved['history'],saved['identity'];del saved
            print(f'Resume seed {seed}: {epoch} completed epochs, batch cursor {cursor}',flush=True)
        else:
            identity=pretrained(model,cfg,work,token);epoch=cursor=0;history=[]
            atomic_save(folder/'state.pt',make_state(model,opt,scaler,scheduler,protocol,seed,epoch,cursor,history,identity))
        p.write(folder/'CONTRACT.json',cfg);p.write(folder/'IDENTITY.json',identity)
        p.write(folder/'HARDWARE.json',dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda))
        last_push=time.monotonic()
        def flush(status):
            # Reload durable state only. Never pair a new status with an older interrupted checkpoint.
            durable=torch.load(folder/'state.pt',map_location='cpu',weights_only=False)
            metadata(folder,durable,status);del durable
            p.publish(folder,remote,token)
        try:
            while epoch<cfg['epochs']:
                if STOP or time.monotonic()-started>9*3600:
                    flush('resumable');return
                order=torch.randperm(len(train_rows),generator=torch.Generator().manual_seed(seed+10000*epoch)).tolist()
                batches=[order[i:i+cfg['batch_size']] for i in range(0,len(order),cfg['batch_size'])]
                assert cursor<=len(batches)
                while cursor<len(batches):
                    if STOP or time.monotonic()-started>9*3600:flush('resumable');return
                    batch=load_batch([train_rows[i] for i in batches[cursor]],root,cfg)
                    loss=step(model,opt,scaler,batch);cursor+=1
                    history.append(dict(epoch=epoch+1,batch=cursor,loss=loss,lr=opt.param_groups[0]['lr']))
                    atomic_save(folder/'state.pt',make_state(model,opt,scaler,scheduler,protocol,seed,epoch,cursor,history,identity))
                    del batch
                    if time.monotonic()-last_push>=1800:flush('resumable');last_push=time.monotonic()
                val=evaluate(model,val_rows,root,cfg)
                p.write(folder/f'validation_epoch{epoch+1:02d}.json',val)
                print(f'Seed {seed} epoch {epoch+1}/60 validation mean width error {val["mean_width_error"]:.5f}',flush=True)
                scheduler.step();epoch+=1;cursor=0
                atomic_save(folder/'state.pt',make_state(model,opt,scaler,scheduler,protocol,seed,epoch,cursor,history,identity))
            # Fixed endpoint: no early-stopping/model selection using test labels.
            test=evaluate(model,test_rows,root,cfg);p.write(folder/'TEST_FINAL.json',test)
            flush('completed')
            print(f'SEED {seed} COMPLETE — all 60 epochs and final test published.',flush=True)
        except BaseException:
            try:flush('resumable')
            except Exception as e:print('Emergency upload failed:',type(e).__name__,'Retain local files and retry.',flush=True)
            raise
        finally:
            del model,opt,scaler,scheduler;torch.cuda.empty_cache()
    print('All requested seeds completed. Verify HF; no full S9 or HRNet-superiority claim.',flush=True)

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['smoke','train']);a.add_argument('--work',required=True);a.add_argument('--root',default='')
    args=a.parse_args();root=p.root_data(args.root);token=os.environ['HF_TOKEN']
    if args.mode=='smoke':smoke(args.work,root,token)
    else:train(args.work,root,token)
