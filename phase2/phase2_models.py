"""Explicit Phase 2 model adapters. No old tyre-trained initialization."""
from PIL import Image  # Load imaging DLLs before torch on the local Windows test host.
import copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from phase2_training_data import SPECS, sample

_INTERPOLATE=F.interpolate
class DeterministicBilinear(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x,size):
  ctx.hw=x.shape[-2:];ctx.dtype=x.dtype
  return _INTERPOLATE(x.float(),size=size,mode='bilinear',align_corners=False).to(x.dtype)
 @staticmethod
 def backward(ctx,g):
  def weights(n,m):
   p=((torch.arange(m,device=g.device,dtype=torch.float32)+.5)*n/m-.5).clamp(0,n-1)
   lo=p.floor().long();hi=(lo+1).clamp_max(n-1);f=p-lo;c=torch.arange(n,device=g.device)
   return (c[None,:]==lo[:,None])*(1-f[:,None])+(c[None,:]==hi[:,None])*f[:,None]
  with torch.autocast(g.device.type,enabled=False):
   y=weights(ctx.hw[0],g.shape[-2]);x=weights(ctx.hw[1],g.shape[-1]);result=y.T@g.float()@x
  return result.to(ctx.dtype),None

def deterministic_resize(input,size=None,scale_factor=None,mode='nearest',align_corners=None,recompute_scale_factor=None,antialias=False):
 if mode=='bilinear' and input.requires_grad and torch.is_grad_enabled() and align_corners is False and size is not None and scale_factor is None and not antialias:
  return DeterministicBilinear.apply(input,(size,size) if isinstance(size,int) else tuple(size))
 return _INTERPOLATE(input,size=size,scale_factor=scale_factor,mode=mode,align_corners=align_corners,recompute_scale_factor=recompute_scale_factor,antialias=antialias)

def deterministic():
 torch.use_deterministic_algorithms(True)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 F.interpolate=deterministic_resize

class Geometry(nn.Module):
 def __init__(self):
  super().__init__();import timm
  self.backbone=timm.create_model(SPECS['hrnet']['model'],pretrained=False,features_only=True,feature_location='',out_indices=(1,2,3,4))
  ch=self.backbone.feature_info.channels();assert ch==[18,36,72,144]
  self.projections=nn.ModuleList(nn.Conv2d(c,16,1) for c in ch)
  self.head=nn.Sequential(nn.Conv2d(64,64,3,padding=1),nn.ReLU(),nn.Conv2d(64,6,1))
 def forward(self,x):
  maps=self.backbone(x);size=maps[0].shape[-2:]
  heat=self.head(torch.cat([F.interpolate(p(m),size=size,mode='nearest') for p,m in zip(self.projections,maps)],1))
  lines=[]
  for i in range(6):
   pos=(heat.shape[-2]-1)*([384/1535,768/1535,1151/1535][i//2]);lo=int(pos);f=pos-lo
   lines.append(heat[:,i,lo,:]*(1-f)+heat[:,i,min(lo+1,heat.shape[-2]-1),:]*f)
  return torch.stack(lines,1)

def build(name,assets,pretrained=True,device='cuda'):
 spec=SPECS[name];assets=Path(assets)
 if name in ('mobilenetv4','resnet50','hrnet'):
  import timm
  assert timm.__version__=='1.0.15'
  model=Geometry() if name=='hrnet' else timm.create_model(spec['model'],pretrained=False,num_classes=3)
  if pretrained:
   from safetensors.torch import load_file
   state=load_file(str(assets/name/'model.safetensors'))
   target=model.backbone if name=='hrnet' else model
   keys=target.state_dict(); selected={}
   for k,v in keys.items():
    if name!='hrnet' and k.startswith(('classifier.','fc.')):selected[k]=v
    elif k in state and state[k].shape==v.shape:selected[k]=state[k]
    elif k.endswith('num_batches_tracked'):selected[k]=v
    else:raise ValueError('Pretrained tensor mismatch: '+k)
   target.load_state_dict(selected,strict=True)
 elif name=='segformer':
  from transformers import SegformerConfig,SegformerForSemanticSegmentation,SegformerModel
  cfg=SegformerConfig.from_pretrained(str(assets/name),local_files_only=True);cfg.num_labels=2
  model=SegformerForSemanticSegmentation(cfg)
  if pretrained:
   encoder=SegformerModel.from_pretrained(str(assets/name),local_files_only=True,use_safetensors=False)
   model.segformer.load_state_dict(encoder.state_dict(),strict=True);del encoder
 elif name=='yolo26m':
  from ultralytics.nn.tasks import SegmentationModel
  from ultralytics.cfg import get_cfg
  model=SegmentationModel(spec['model'],ch=3,nc=2,verbose=False)
  model.args=get_cfg(overrides=dict(task='segment',overlap_mask=False,epochs=60,box=7.5,cls=.5,dfl=1.5))
  if pretrained:
   # Official upstream asset only; file hash is frozen in the run contract.
   ckpt=torch.load(assets/name/'yolo26m-seg.pt',map_location='cpu',weights_only=False)
   original=ckpt.get('ema') or ckpt['model'];model.load(original.float(),verbose=False);del ckpt,original
  # Native YOLO26 auxiliary semantic loss assumes exclusive classes. Our nested
  # tyre/tread instances remain overlapping; disable only that auxiliary branch.
  # Instance masks, both detection heads and their ordinary losses remain active.
  model.names={0:'tyre',1:'tread'}
 else:raise ValueError(name)
 model=model.to(device)
 if name=='yolo26m':model.criterion=model.init_criterion()
 return model

def training_mode(model):
 model.train()
 for layer in model.modules():
  if isinstance(layer,nn.modules.batchnorm._BatchNorm):layer.eval()

def batch(root,rows,name,epoch=0,step=0,seed=None,device='cuda'):
 arrays=[sample(root,r,SPECS[name]['hw'],None if seed is None else seed*10000019+epoch*100003+step*97+i,name) for i,r in enumerate(rows)]
 x=torch.tensor(np.stack([v[0] for v in arrays]),device=device)
 masks=torch.tensor(np.stack([v[1] for v in arrays]),device=device)
 points=torch.tensor(np.stack([v[2] for v in arrays]),device=device)
 if name!='yolo26m':x=(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
 result=dict(img=x,masks=masks,points=points,labels=torch.tensor([r['class_index'] for r in rows],device=device),point_weight=x.new_tensor([r['point_weight'] for r in rows]))
 if name=='yolo26m':
  boxes=[];indices=[];classes=[];instances=[];h,w=x.shape[-2:]
  for i in range(len(rows)):
   for c in (0,1):
    y,z=torch.where(masks[i,c]);assert len(z),'Empty resized YOLO target'
    x0,x1=z.min(),z.max()+1;y0,y1=y.min(),y.max()+1
    boxes.append(torch.stack(((x0+x1)/2/w,(y0+y1)/2/h,(x1-x0)/w,(y1-y0)/h)))
    indices.append(i);classes.append(c);instances.append(masks[i,c].float())
  result.update(bboxes=torch.stack(boxes),batch_idx=torch.tensor(indices,device=device),cls=x.new_tensor(classes).view(-1,1),masks=torch.stack(instances))
 return result

def without_semantic(pred):
 if isinstance(pred,dict):
  out={k:without_semantic(v) for k,v in pred.items()}
  if isinstance(out.get('proto'),tuple):out['proto']=out['proto'][0]
  return out
 return pred

def loss(model,b,name,epoch):
 if name in ('mobilenetv4','resnet50'):return F.cross_entropy(model(b['img']).float(),b['labels'],label_smoothing=.05)
 if name=='hrnet':
  z=model(b['img']).float();grid=torch.arange(z.shape[-1],device=z.device)
  target=torch.exp(-.5*((grid-b['points'][...,None]*(z.shape[-1]-1))/1.5)**2)
  target=target/target.sum(-1,keepdim=True).clamp_min(1e-12)
  values=-(target*z.log_softmax(-1)).sum(-1).mean(-1)
  return (values*b['point_weight']).mean()
 if name=='segformer':
  z=F.interpolate(model(pixel_values=b['img']).logits.float(),b['masks'].shape[-2:],mode='bilinear',align_corners=False)
  y=b['masks'][:,:2].float();valid=(~b['masks'][:,2:3]).float();p=z.sigmoid()
  bce=(F.binary_cross_entropy_with_logits(z,y,reduction='none')*valid).sum()/(valid.sum()*2).clamp_min(1)
  dims=(0,2,3);dice=(2*(p*y*valid).sum(dims)+1)/((p*valid).sum(dims)+(y*valid).sum(dims)+1)
  return bce+1-dice.mean()
 if name=='yolo26m':
  criterion=model.criterion
  # The upstream progressive loss has mutable epoch weighting. Set explicitly
  # from saved epoch rather than relying on framework callbacks.
  if hasattr(criterion,'o2m'):
   criterion.o2m=criterion.decay(epoch);criterion.o2o=1-criterion.o2m
  pred=without_semantic(model(b['img']))
  return criterion(pred,b)[0].sum()/len(b['img'])
 raise ValueError(name)

def predict(model,b,name):
 if name in ('mobilenetv4','resnet50'):return model(b['img']).float().softmax(-1)
 if name=='hrnet':
  z=model(b['img']).float();return (z.softmax(-1)*torch.linspace(0,1,z.shape[-1],device=z.device)).sum(-1)
 if name=='segformer':return F.interpolate(model(pixel_values=b['img']).logits.float(),b['img'].shape[-2:],mode='bilinear',align_corners=False).sigmoid()>=.5
 if name=='yolo26m':
  from ultralytics.utils.ops import process_mask
  out=model(b['img']);pair=out[0] if isinstance(out[0],tuple) else out
  detections,proto=pair
  result=torch.zeros((len(b['img']),2,*b['img'].shape[-2:]),dtype=torch.bool,device=b['img'].device)
  for i,det in enumerate(detections):
   det=det[det[:,4]>=.25]
   if not len(det):continue
   masks=process_mask(proto[i],det[:,6:],det[:,:4],b['img'].shape[-2:],upsample=True).bool()
   for c in (0,1):
    hit=det[:,5].long()==c
    if hit.any():result[i,c]=masks[hit].any(0)
  return result
