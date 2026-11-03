"""ATLAS-CONTEXT r3: frozen LC prior plus position-aware bounded correction."""
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from common import HERE,read,sha
from frozen_lc import TreadTwoAtlas as FrozenLC,PatchEncoder,WEIGHTS_SHA

class ContextBlock(nn.Module):
    """Explicit FP32 attention, no SDPA/flash/dropout/BatchNorm fast paths."""
    def __init__(self):
        super().__init__()
        self.norm1=nn.LayerNorm(96);self.qkv=nn.Linear(96,288);self.projection=nn.Linear(96,96)
        self.norm2=nn.LayerNorm(96);self.ff=nn.Sequential(nn.Linear(96,192),nn.GELU(),nn.Linear(192,96))
    def forward(self,x,valid):
        b,n,d=x.shape
        q,k,v=self.qkv(self.norm1(x)).reshape(b,n,3,4,24).permute(2,0,3,1,4).unbind(0)
        keys=valid.clone();keys[~keys.any(-1),0]=True
        score=(q@k.transpose(-1,-2))/math.sqrt(24)
        attention=score.masked_fill(~keys[:,None,None,:],-10000.).softmax(-1)
        x=x+self.projection((attention@v).transpose(1,2).reshape(b,n,d))
        return x+self.ff(self.norm2(x)),attention

class TreadTwoAtlas(nn.Module):
    def __init__(self,bank_rgb,bank_labels,weights,base_weights):
        super().__init__();meta=read(HERE/'BASE_INPUT.json')
        assert sha(Path(base_weights))==meta['sha256'],'Pinned LC base hash mismatch'
        self.base=FrozenLC(bank_rgb,bank_labels,weights)
        checkpoint=torch.load(base_weights,map_location='cpu',weights_only=False)
        assert checkpoint['protocol']==meta['protocol'] and checkpoint['epoch']==7
        self.base.load_state_dict(checkpoint['model'],strict=True);self.base.eval().requires_grad_(False)
        self.spatial_encoder=PatchEncoder()
        self.spatial_encoder.load_state_dict(self.base.encoder.state_dict(),strict=True)
        self.coverage=nn.Linear(2,96)
        self.blocks=nn.ModuleList([ContextBlock(),ContextBlock()]);self.norm=nn.LayerNorm(96)
        self.head=nn.Linear(96,3,bias=False);nn.init.zeros_(self.head.weight)
        self.context_bias_parameter=nn.Parameter(torch.zeros(3))
        row=torch.arange(8).repeat_interleave(4).float()/7
        col=torch.arange(4).repeat(8).float()/3
        frequency=2**torch.linspace(0,3,24)
        pos=torch.cat([torch.sin(2*math.pi*row[:,None]*frequency),torch.cos(2*math.pi*row[:,None]*frequency),
                       torch.sin(2*math.pi*col[:,None]*frequency),torch.cos(2*math.pi*col[:,None]*frequency)],1)
        self.register_buffer('position',pos)
    @property
    def anchor(self):return self.base.anchor
    @property
    def bank_rgb(self):return self.base.bank_rgb
    @property
    def bank_labels(self):return self.base.bank_labels
    @property
    def raw_temperature(self):return self.base.raw_temperature
    @property
    def bias(self):return self.base.bias+.5*self.context_bias_parameter.tanh()
    def coefficients(self):return self.base.coefficients()
    def train(self,mode=True):
        super().train(mode);self.base.eval();return self
    def correct(self,out,variant='identity'):
        out=dict(out);out.pop('_paired',None)
        if variant=='reverse_bands':order=[r*4+c for r in reversed(range(8)) for c in range(4)]
        elif variant=='reverse_columns':order=[r*4+c for r in range(8) for c in reversed(range(4))]
        else:order=list(range(32))
        index=torch.tensor(order,device=out['logits'].device)
        for key in ('affinity','cosine','patch_weights','patch_coverage','patch_evidence','query_embeddings','local_contrast_tiles'):
            out[key]=out[key][:,index]
        batch=len(out['logits']);features=out['local_contrast_tiles']
        query=self.spatial_encoder(features.reshape(batch*32,4,64,64)).reshape(batch,32,96)
        coverage=torch.stack((out['patch_coverage'],out['patch_weights']),-1)
        x=query+.1*self.position[None]+self.coverage(coverage)
        matrices=[]
        for block in self.blocks:
            x,a=block(x,out['patch_coverage']>0);matrices.append(a)
        tokens=self.norm(x);delta=2*self.head(tokens).tanh()*out['patch_weights'][:,:,None]
        bias_delta=.5*self.context_bias_parameter.tanh()
        total=delta.sum(1)+bias_delta
        base_logits=out['logits'];logits=base_logits+total
        return dict(out,base_logits=base_logits,base_query_embeddings=out['query_embeddings'],
            query_embeddings=F.normalize(tokens,dim=-1,eps=1e-6),logits=logits,probabilities=logits.softmax(-1),
            patch_evidence=out['patch_evidence']+delta,bias=out['bias']+bias_delta,
            context_evidence=delta,context_bias=bias_delta.clone(),context_total=total,
            attention=torch.stack(matrices,1),source_patch=index)
    def forward(self,rgb,augmentation=None,variant='identity',occluded_patch=None,*,source_hash):
        self.base.eval()
        base_variant='identity' if variant in ('reverse_bands','reverse_columns') else variant
        with torch.autocast(rgb.device.type,enabled=False):
            with torch.no_grad():base=self.base(rgb,augmentation,base_variant,occluded_patch,source_hash=source_hash)
            second=base.get('_paired')
            out=self.correct(base,variant)
            if second is not None:out['_paired']=self.correct(second,variant)
            return out
