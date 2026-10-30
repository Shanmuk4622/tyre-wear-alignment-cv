"""Untrained TREAD-TWO/ATLAS architecture prototype, not a measured model release."""
import hashlib,importlib.util,json,math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from atlas import canonicalize,patches,band_weights
PHASE3=Path(__file__).resolve().parents[1]
WEIGHTS_SHA='a5b872ed2e4cbfb3b9782b0df563e2606631744cd0274fee2da42a832f985f65'
SOURCE_SHA='d16cd4eef75779044dd95900c0531fee474704cf88f523326ee069115347ba09'

def load_anchor():
 source=PHASE3/'runtime/model.py';weight=PHASE3/'completion_20261028/hf/report/weights.pt'
 assert hashlib.sha256(source.read_bytes()).hexdigest()==SOURCE_SHA and hashlib.sha256(weight.read_bytes()).hexdigest()==WEIGHTS_SHA
 spec=importlib.util.spec_from_file_location('tread_two_frozen_anchor',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);model=module.TreadOne();model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=True)
 model.eval()
 for p in model.parameters():p.requires_grad_(False)
 return model

class Groove(nn.Module):
 def __init__(self,cin,cout):
  super().__init__();self.vertical=nn.Conv2d(cin,cin,(7,3),padding=(3,1),groups=cin,bias=False);self.horizontal=nn.Conv2d(cin,cin,(3,7),padding=(1,3),groups=cin,bias=False);self.mix=nn.Sequential(nn.Conv2d(2*cin,cout,1,bias=False),nn.GroupNorm(8,cout),nn.SiLU())
 def forward(self,x):return self.mix(torch.cat((self.vertical(x),self.horizontal(x)),1))

class PatchEncoder(nn.Module):
 def __init__(self):
  super().__init__();self.body=nn.Sequential(nn.Conv2d(4,24,3,2,1,bias=False),nn.GroupNorm(8,24),nn.SiLU(),Groove(24,32),nn.Conv2d(32,48,3,2,1,bias=False),nn.GroupNorm(8,48),nn.SiLU(),Groove(48,64),nn.Conv2d(64,96,3,2,1,bias=False),nn.GroupNorm(8,96),nn.SiLU(),nn.AdaptiveAvgPool2d(1))
  self.register_buffer('sobel_x',torch.tensor([[-1.,0,1],[-2,0,2],[-1,0,1]])[None,None]/8)
 def forward(self,rgb):
  gray=rgb.mean(1,keepdim=True);dx=F.conv2d(gray,self.sobel_x,padding=1);dy=F.conv2d(gray,self.sobel_x.transpose(-1,-2),padding=1);detail=(dx.square()+dy.square()+1e-8).sqrt();x=torch.cat(((rgb-.5)/.25,detail),1)
  return F.normalize(self.body(x).flatten(1).float(),dim=1,eps=1e-6)

class TreadTwoAtlas(nn.Module):
 def __init__(self,bank_rgb,bank_labels,anchor=None):
  super().__init__();assert tuple(bank_rgb.shape)==(18,3,64,64) and tuple(bank_labels.shape)==(18,) and all(int((bank_labels==c).sum())==6 for c in range(3))
  self.anchor=anchor if anchor is not None else load_anchor()
  self.anchor.eval()
  for p in self.anchor.parameters():p.requires_grad_(False)
  self.encoder=PatchEncoder();self.register_buffer('bank_rgb',bank_rgb.detach().clone());self.register_buffer('bank_labels',bank_labels.long().clone());self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None]);self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])
  own=torch.arange(3)[:,None]==bank_labels[None];magnitude=torch.where(own,.5,.25);self.raw_weights=nn.Parameter(torch.logit(magnitude/2));self.bias=nn.Parameter(torch.zeros(3));self.raw_temperature=nn.Parameter(torch.tensor(math.log((.2-.05)/(1-.2))))
 def train(self,mode=True):
  super().train(mode);self.anchor.eval();return self
 def forward(self,rgb):
  assert rgb.dtype==torch.float32 and tuple(rgb.shape[1:])==(3,512,384)
  with torch.no_grad(),torch.autocast(rgb.device.type,enabled=False):dense=self.anchor((rgb-self.mean)/self.std);atlas=canonicalize(rgb,dense['masks'])
  tiles=patches(atlas['rgb']);alpha,coverage=band_weights(atlas['valid']);b=len(rgb);features=self.encoder(torch.cat((tiles.reshape(b*32,3,64,64),self.bank_rgb),0));query=features[:b*32].reshape(b,32,96);support=features[b*32:]
  with torch.autocast(rgb.device.type,enabled=False):
   cosine=torch.einsum('bid,pd->bip',query.float(),support.float()).clamp(-1,1);temperature=.05+.95*self.raw_temperature.sigmoid();affinity=((cosine-1)/temperature).exp();own=torch.arange(3,device=rgb.device)[:,None]==self.bank_labels[None];signed=torch.where(own,1.,-1.)*2*self.raw_weights.sigmoid()/6
   pair=affinity[:,:,None,:]*signed[None,None]*alpha[:,:,None,None];evidence=pair.sum(-1);logits=self.bias+evidence.sum(1)
  return dict(logits=logits,probabilities=logits.softmax(-1),patch_evidence=evidence,prototype_contributions=pair,affinity=affinity,cosine=cosine,patch_weights=alpha,patch_coverage=coverage,temperature=temperature,
   masks=dense['masks'],points=dense['points'],guides=dense['guides'],atlas=atlas,tread_one_probabilities=dense['probabilities'],model_status='UNTRAINED_ARCHITECTURE_PROTOTYPE')
