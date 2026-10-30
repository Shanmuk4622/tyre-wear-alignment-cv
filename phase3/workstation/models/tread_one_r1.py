"""TREAD-ONE r1: custom train-from-scratch multi-task evidence network.

No downloaded backbone, detector, teacher or post-hoc CAM dependency.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

GUIDES = (384/1535, 768/1535, 1151/1535)
MODEL_VERSION = 'tread-one-r1'


class Conv(nn.Sequential):
    def __init__(self, cin, cout, kernel=3, stride=1):
        super().__init__(nn.Conv2d(cin, cout, kernel, stride, kernel//2, bias=False),
                         nn.GroupNorm(8, cout), nn.SiLU())


class TextureBlock(nn.Module):
    """Local grooves and wider texture context, with explicit residual scaling."""
    def __init__(self, channels, dilation):
        super().__init__()
        self.spatial = nn.Conv2d(channels, channels, 5, padding=2*dilation,
                                 dilation=dilation, groups=channels)
        self.mix = nn.Sequential(nn.GroupNorm(8, channels), nn.Conv2d(channels, 2*channels, 1),
                                 nn.SiLU(), nn.Conv2d(2*channels, channels, 1))
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, x):
        return x + self.scale*self.mix(self.spatial(x))


class ContextBlock(nn.Module):
    """Ordinary explicit attention math: no flash/SDPA backend ambiguity."""
    def __init__(self, channels=160, heads=4):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels*3)
        self.proj = nn.Linear(channels, channels)
        self.ff = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, channels*2),
                                nn.GELU(), nn.Linear(channels*2, channels))

    def forward(self, x):
        b, c, h, w = x.shape
        z = x.flatten(2).transpose(1, 2)
        q, k, v = self.qkv(self.norm(z)).reshape(b, h*w, 3, self.heads, c//self.heads).permute(2, 0, 3, 1, 4)
        with torch.autocast(x.device.type, enabled=False):
            weights = (q.float() @ k.float().transpose(-1, -2))/math.sqrt(c//self.heads)
            y = weights.softmax(-1) @ v.float()
        y = y.to(z.dtype).transpose(1, 2).reshape(b, h*w, c)
        z = z + self.proj(y)
        z = z + self.ff(z)
        return z.transpose(1, 2).reshape(b, c, h, w)


class TreadOne(nn.Module):
    def __init__(self, concepts=12):
        super().__init__()
        self.stem = nn.Sequential(Conv(3, 24, stride=2), Conv(24, 32, stride=2))
        self.s1 = nn.Sequential(TextureBlock(32, 1), TextureBlock(32, 2))
        self.s2 = nn.Sequential(Conv(32, 64, stride=2), TextureBlock(64, 1), TextureBlock(64, 2))
        self.s3 = nn.Sequential(Conv(64, 112, stride=2), TextureBlock(112, 1), TextureBlock(112, 2))
        self.s4 = nn.Sequential(Conv(112, 160, stride=2), TextureBlock(160, 1))
        self.position = nn.Conv2d(2, 160, 1)
        self.context = nn.Sequential(ContextBlock(), ContextBlock())
        self.d3, self.d2, self.d1 = Conv(272, 112), Conv(176, 64), Conv(96, 48)
        self.seg = nn.Conv2d(48, 2, 1)
        self.concepts = nn.Conv2d(48, concepts, 1)
        self.class_weights = nn.Parameter(torch.randn(3, concepts)*0.03)
        self.class_bias = nn.Parameter(torch.zeros(3))
        self.point_features = nn.Conv2d(48, 16, 1)
        self.points = nn.Sequential(nn.Linear(16*16, 96), nn.SiLU(), nn.Linear(96, 2))
        nn.init.zeros_(self.points[-1].weight)
        with torch.no_grad():
            self.points[-1].bias.copy_(torch.tensor([-1.95, 1.79]))

    @staticmethod
    def up(x, skip):
        # Integer repeat/crop keeps backward deterministic at all configured sizes.
        x = x.repeat_interleave(2, -2).repeat_interleave(2, -1)
        return torch.cat((x[..., :skip.shape[-2], :skip.shape[-1]], skip), 1)

    def forward(self, image, guides=None):
        if image.shape[-2] % 32 or image.shape[-1] % 64:
            raise ValueError('Input height must divide by 32 and width by 64.')
        a = self.s1(self.stem(image))
        b = self.s2(a)
        c = self.s3(b)
        z = self.s4(c)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, z.shape[-2], device=z.device),
                                 torch.linspace(-1, 1, z.shape[-1], device=z.device), indexing='ij')
        position = torch.stack((xx, yy))[None].to(z.dtype)
        z = self.context(z + self.position(position))
        features = self.d1(self.up(self.d2(self.up(self.d3(self.up(z, c)), b)), a))
        seg_logits = self.seg(features).float()
        masks = seg_logits.sigmoid()
        # Segmentation alone learns the gate. Classification cannot directly reshape it.
        gate = masks[:, 1:2].detach()
        gate = gate/gate.sum((-2, -1), keepdim=True).clamp_min(1e-6)
        concepts = F.softplus(self.concepts(features).float())
        with torch.autocast(image.device.type, enabled=False):
            contribution = torch.einsum('bkhw,ck->bchw', concepts, self.class_weights.float())*gate
        logits = contribution.sum((-2, -1)) + self.class_bias.float()
        if guides is None:
            guides = image.new_tensor(GUIDES)[None].expand(len(image), -1)
        indices = (guides*(features.shape[-2]-1)).round().long().clamp(0, features.shape[-2]-1)
        pf = self.point_features(features).float()
        rows = torch.stack([pf[i, :, indices[i], :].permute(1, 0, 2) for i in range(len(image))])
        rows = rows.reshape(len(image), 3, 16, 16, pf.shape[-1]//16).mean(-1).flatten(2)
        raw = self.points(rows)
        left = raw[..., 0].sigmoid()*0.99
        right = left + (1-left)*raw[..., 1].sigmoid()
        points = torch.stack((left, right), -1).flatten(1)
        return dict(logits=logits, probabilities=logits.softmax(-1), seg_logits=seg_logits,
                    masks=masks, points=points, concepts=concepts, gate=gate,
                    contributions=contribution, bias=self.class_bias.float(), guides=guides)


def loss_terms(out, batch):
    y, valid = batch['masks'][:, :2], 1-batch['masks'][:, 2:3]
    # Fixed 4x4 mean aggregation avoids nondeterministic interpolation backward.
    n, _, h, w = y.shape
    y = y.reshape(n, 2, h//4, 4, w//4, 4).mean((3, 5))
    valid = valid.reshape(n, 1, h//4, 4, w//4, 4).amin((3, 5))
    bce = (F.binary_cross_entropy_with_logits(out['seg_logits'], y, reduction='none')*valid).sum()/(2*valid.sum()).clamp_min(1)
    p = out['masks']
    dice = ((2*(p*y*valid).sum((-2, -1))+1)/((p*valid).sum((-2, -1))+(y*valid).sum((-2, -1))+1)).mean()
    segmentation = bce + 1-dice
    classification = F.cross_entropy(out['logits'], batch['labels'], label_smoothing=0.05)
    target = F.one_hot(batch['labels'], 3).float()
    probability_cdf = torch.stack((out['probabilities'][:, 0], out['probabilities'][:, 0]+out['probabilities'][:, 1]), -1)
    target_cdf = torch.stack((target[:, 0], target[:, 0]+target[:, 1]), -1)
    ordinal = (probability_cdf-target_cdf).square().mean()
    geometry = (F.smooth_l1_loss(out['points'], batch['points'], reduction='none', beta=.02).mean(-1)*batch['point_weight']).sum()/batch['point_weight'].sum().clamp_min(1)
    containment = (F.relu(p[:, 1:2]-p[:, :1])*valid).sum()/valid.sum().clamp_min(1)
    # Geometry and segmentation should describe the same guide-row interval.
    idx = (out['guides']*(p.shape[-2]-1)).round().long()
    rows = torch.stack([p[i, 1, idx[i]] for i in range(n)])
    row_valid = torch.stack([valid[i, 0, idx[i]] for i in range(n)])
    reference = torch.stack([y[i, 1, idx[i]] for i in range(n)])
    usable = ((reference[..., 0] < .1) & (reference[..., -1] < .1) & (reference.sum(-1) > 2)).float()
    x = torch.linspace(0, 1, p.shape[-1], device=p.device)[None, None]
    q = out['points'].reshape(n, 3, 2)
    interval = ((x-q[..., :1])/.015).sigmoid()*((q[..., 1:]-x)/.015).sigmoid()
    alignment = ((interval-rows.detach()).abs()*row_valid*usable[..., None]).sum()/(row_valid*usable[..., None]).sum().clamp_min(1)
    total = classification + .2*ordinal + segmentation + 2*geometry + .1*containment + .1*alignment
    return dict(total=total, classification=classification, ordinal=ordinal, segmentation=segmentation,
                geometry=geometry, containment=containment, alignment=alignment)


def boxes(binary):
    """Mask-derived box, not a separately trained object detector."""
    result = []
    for mask in binary:
        ys, xs = torch.where(mask)
        result.append(None if not len(xs) else [int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1])
    return result
