"""ATLAS-LC r2: frozen FP32 dense anchor and trainable literal-case classifier."""
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from common import HERE, read, sha
import numpy as np
from lc_features import contrast
from frozen_anchor import TreadOne
from geometry import canonicalize, patches, band_weights

WEIGHTS_SHA = 'a5b872ed2e4cbfb3b9782b0df563e2606631744cd0274fee2da42a832f985f65'


class Groove(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.vertical = nn.Conv2d(cin, cin, (7, 3), padding=(3, 1), groups=cin, bias=False)
        self.horizontal = nn.Conv2d(cin, cin, (3, 7), padding=(1, 3), groups=cin, bias=False)
        self.mix = nn.Sequential(nn.Conv2d(2 * cin, cout, 1, bias=False), nn.GroupNorm(8, cout), nn.SiLU())

    def forward(self, value):
        return self.mix(torch.cat((self.vertical(value), self.horizontal(value)), 1))


class PatchEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(4, 24, 3, 2, 1, bias=False), nn.GroupNorm(8, 24), nn.SiLU(),
                                  Groove(24, 32), nn.Conv2d(32, 48, 3, 2, 1, bias=False), nn.GroupNorm(8, 48), nn.SiLU(),
                                  Groove(48, 64), nn.Conv2d(64, 96, 3, 2, 1, bias=False), nn.GroupNorm(8, 96), nn.SiLU(),
                                  nn.AdaptiveAvgPool2d(1))

    def forward(self, features):
        return F.normalize(self.body(features).flatten(1).float(),dim=1,eps=1e-6)


class TreadTwoAtlas(nn.Module):
    def __init__(self, bank_rgb, bank_labels, weights):
        super().__init__()
        assert sha(Path(weights)) == WEIGHTS_SHA
        self.anchor = TreadOne()
        self.anchor.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
        self.anchor.eval()
        for parameter in self.anchor.parameters():
            parameter.requires_grad_(False)
        assert bank_rgb.shape == (18, 3, 64, 64) and bank_labels.shape == (18,)
        assert all(int((bank_labels == c).sum()) == 6 for c in range(3))
        self.encoder = PatchEncoder()
        self.register_buffer('bank_rgb', bank_rgb.detach().clone())
        self.register_buffer('bank_labels', bank_labels.long().clone())
        self.register_buffer('bank_valid',torch.from_numpy(np.load(HERE/'assets/bank_valid.npy',allow_pickle=False)))
        self.register_buffer('bank_sha256',torch.from_numpy(np.load(HERE/'assets/bank_sha256.npy',allow_pickle=False)))
        self.lc_rules=read(HERE/'LC_RULES.json')
        self.register_buffer('mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([.229, .224, .225])[None, :, None, None])
        own = torch.arange(3)[:, None] == bank_labels[None]
        self.raw_weights = nn.Parameter(torch.logit(torch.where(own, .5, .25) / 2))
        self.bias = nn.Parameter(torch.zeros(3))
        self.raw_temperature = nn.Parameter(torch.tensor(math.log((.2 - .05) / (1 - .2))))

    def train(self, mode=True):
        super().train(mode)
        self.anchor.eval()
        return self

    def coefficients(self):
        own = torch.arange(3, device=self.bank_labels.device)[:, None] == self.bank_labels[None]
        return torch.where(own, 1., -1.) * 2 * self.raw_weights.sigmoid() / 6

    def classify(self, atlas, source_hash, augmentation=None, variant='identity', occluded_patch=None):
        tiles = patches(atlas['rgb'])
        valid = patches(atlas['valid']).float()
        alpha, coverage = band_weights(atlas['valid'])
        batch = len(tiles)
        if augmentation is not None:
            factors = augmentation.to(tiles.device).reshape(batch, 3, 1, 1, 1, 1)
            # Per-image factors are shared across all its patches. Invalid RGB stays zero.
            brightness, contrast_factor, gamma = factors[:, 0], factors[:, 1], factors[:, 2]
            tiles = tiles * brightness
            support = valid.sum((1, 2, 3, 4), keepdim=True).clamp_min(1) * 3
            center = (tiles * valid).sum((1, 2, 3, 4), keepdim=True) / support
            tiles = ((tiles - center) * contrast_factor + center).clamp(0,1).pow(gamma)
        bank = self.bank_rgb
        if variant == 'grayscale_both':
            tiles = tiles.mean(2, keepdim=True).expand_as(tiles)
            bank = bank.mean(1, keepdim=True).expand_as(bank)
        elif variant == 'brightness_080':
            tiles = tiles * .8
        elif variant == 'brightness_120':
            tiles = tiles * 1.2
        elif variant == 'gamma_080':
            tiles = tiles.clamp(0,1).pow(.8)
        elif variant == 'gamma_120':
            tiles = tiles.clamp(0,1).pow(1.2)
        elif variant not in ('identity', 'occlude_top_patch'):
            raise ValueError('Unknown frozen diagnostic')
        tiles = tiles.clamp(0, 1) * valid
        replacement = None
        if occluded_patch is not None:
            replacement = (tiles * valid).sum((1, 3, 4)) / valid.sum((1, 3, 4)).clamp_min(1)
            tiles = tiles.clone()
            for i, index in enumerate(occluded_patch):
                tiles[i, int(index)] = replacement[i, :, None, None] * valid[i, int(index)]
        query_lc=contrast(tiles.reshape(batch*32,3,64,64),valid.reshape(batch*32,1,64,64),self.lc_rules)
        bank_lc=contrast(bank,self.bank_valid,self.lc_rules)
        features = self.encoder(torch.cat((query_lc, bank_lc), 0))
        query, reference = features[:batch * 32].reshape(batch, 32, 96), features[batch * 32:]
        cosine = torch.einsum('bid,pd->bip', query, reference).clamp(-1, 1)
        temperature = .05 + .95 * self.raw_temperature.sigmoid()
        affinity = ((cosine - 1) / temperature).exp()
        coefficients = self.coefficients()
        assert source_hash.dtype==torch.uint8 and source_hash.shape==(batch,32), 'Full source SHA required; no label input'
        allowed=~(source_hash[:,None,:]==self.bank_sha256[None]).all(-1)
        counts=torch.stack([allowed[:,self.bank_labels==c].sum(1) for c in range(3)],1)
        assert (counts>0).all(), 'No remaining reference support'
        scale=6/counts[:,self.bank_labels].float()
        effective=coefficients[None]*allowed[:,None,:]*scale[:,None,:]
        pair = affinity[:, :, None, :] * effective[:, None] * alpha[:, :, None, None]
        evidence = pair.sum(-1)
        logits = self.bias + evidence.sum(1)
        return dict(logits=logits, probabilities=logits.softmax(-1), patch_evidence=evidence,
                    affinity=affinity, cosine=cosine, patch_weights=alpha, patch_coverage=coverage,
                    reference_embeddings=reference, query_embeddings=query, coefficients=coefficients,
                    bias=self.bias.clone(), temperature=temperature, replacement=replacement,
                    reference_allowed=allowed, reference_scale=scale, allowed_counts=counts, effective_coefficients=effective)

    def forward(self, rgb, augmentation=None, variant='identity', occluded_patch=None, *, source_hash):
        assert rgb.dtype == torch.float32 and tuple(rgb.shape[1:]) == (3, 512, 384)
        # r2 uses FP32 throughout. Outer AMP cannot change anchor or evidence math.
        with torch.autocast(rgb.device.type, enabled=False):
            with torch.no_grad():
                # Fixed microbatch one makes a source image's dense geometry independent
                # of its training companions and supports exact deduplicated retention.
                parts = [self.anchor((image[None] - self.mean) / self.std) for image in rgb]
                dense = {key: torch.cat([part[key] for part in parts], 0)
                         for key in ('masks', 'points', 'guides', 'probabilities', 'logits')}
                atlas = canonicalize(rgb, dense['masks'])
            shared=dict(masks=dense['masks'],points=dense['points'],guides=dense['guides'],
                        tread_one_probabilities=dense['probabilities'],tread_one_logits=dense['logits'],atlas=atlas)
            if augmentation is None:
                return dict(self.classify(atlas,source_hash,None,variant,occluded_patch),**shared)
            assert augmentation.shape==(len(rgb),2,3)
            first=dict(self.classify(atlas,source_hash,augmentation[:,0]),**shared)
            second=dict(self.classify(atlas,source_hash,augmentation[:,1]),**shared)
            first['_paired']=second
            return first
