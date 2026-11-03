"""TREAD-TWO/GEO trainable modules; the frozen anchor is deliberately outside this module."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from atlas_geo import canonicalize, patches
from data_geo import photometric
from lc_features import contrast


def ordered_points(raw):
    left = .99 * raw[..., 0].sigmoid()
    right = left + (1 - left) * raw[..., 1].sigmoid()
    return torch.stack((left, right), -1).flatten(1)


class AxialBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = nn.ModuleList([nn.Conv2d(96, 96, k, padding=(k[0]//2, k[1]//2), groups=96)
                                      for k in ((3, 3), (5, 1), (1, 3))])
        self.mix = nn.Sequential(nn.GroupNorm(8, 96), nn.Conv2d(96, 192, 1), nn.SiLU(), nn.Conv2d(192, 96, 1))
        self.scale = nn.Parameter(torch.full((1, 96, 1, 1), .1))

    def forward(self, tokens):
        x = tokens.transpose(1, 2).reshape(-1, 96, 8, 4)
        y = x + self.scale * self.mix(sum(layer(x) for layer in self.spatial))
        return y.flatten(2).transpose(1, 2)


class Geo(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_adapter = nn.Sequential(nn.Conv2d(48, 24, 3, padding=1), nn.GroupNorm(8, 24), nn.SiLU(), nn.Conv2d(24, 2, 1))
        self.point_adapter = nn.Sequential(nn.Linear(256, 64), nn.SiLU(), nn.Linear(64, 2))
        for layer in (self.mask_adapter[-1], self.point_adapter[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        encoder = []
        for cin, cout in zip((4, 24, 32, 64), (24, 32, 64, 96)):
            encoder.extend([nn.Conv2d(cin, cout, 3, stride=2, padding=1), nn.GroupNorm(8, cout), nn.SiLU()])
        self.encoder = nn.Sequential(*encoder)
        self.metadata = nn.Linear(4, 96)
        self.context = nn.Sequential(AxialBlock(), AxialBlock())
        self.mask_token = nn.Parameter(torch.zeros(96))
        self.projector = nn.Sequential(nn.Linear(96, 128), nn.SiLU(), nn.Linear(128, 64))
        self.decoder = nn.Sequential(nn.Linear(96, 128), nn.SiLU(), nn.Linear(128, 1024))
        self.quality = nn.Sequential(nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 3), nn.Sigmoid())

    def groups(self):
        geometry = [*self.mask_adapter.parameters(), *self.point_adapter.parameters()]
        quality = list(self.quality.parameters())
        excluded = {id(p) for p in geometry + quality}
        texture = [p for p in self.parameters() if id(p) not in excluded]
        return {'geometry': geometry, 'texture': texture, 'quality': quality}

    def refine(self, base):
        logits = base['anchor_logits'] + self.mask_adapter(base['features']).tanh()
        raw = base['raw'] + .25 * self.point_adapter(base['point_features']).tanh()
        return dict(seg_logits=logits, masks=logits.sigmoid(), points=ordered_points(raw), raw=raw)

    @torch.no_grad()
    def quality_input(self, base, refined, atlas, guides):
        p = refined['masks'].detach()
        binary = p >= .5
        areas = p.mean((-2, -1))
        z = p.clamp(1e-6, 1-1e-6)
        entropy = -(z*z.log()+(1-z)*(1-z).log()).mean((-2, -1))
        empty = (~binary.flatten(2).any(-1)).float()
        edge = binary[..., 0] | binary[..., -1]
        clipped = edge.float().sum(-1) / binary.any(-1).float().sum(-1).clamp_min(1)
        index = (guides * 127).round().long().clamp(0, 127)
        rows = torch.stack([binary[i, 0, index[i]] & binary[i, 1, index[i]] for i in range(len(p))])
        xs = torch.arange(96, device=p.device)[None, None]
        available = rows.any(-1)
        left = torch.where(rows, xs, 96).amin(-1).float()/95
        right = torch.where(rows, xs, -1).amax(-1).float()/95
        limits = torch.stack((left, right), -1)
        gaps = (limits-refined['points'].detach().reshape(-1, 3, 2)).abs().mean(-1)
        gaps = torch.where(available, gaps, torch.zeros_like(gaps))
        cov = atlas['coverage']
        scalars = torch.cat((areas, entropy, empty, clipped, gaps, available.float(), cov.mean(-1, keepdim=True),
                             (cov.reshape(-1, 8, 4).sum(-1)>0).float().mean(-1, keepdim=True)), -1)
        assert scalars.shape[-1] == 16
        return torch.cat((base['features'].mean((-2, -1)), scalars), -1).detach()

    def contextualize(self, encoded, metadata, masked=None):
        tokens = encoded if masked is None else torch.where(masked[..., None], self.mask_token, encoded)
        return self.context(tokens + metadata)

    def texture_view(self, atlas, seeds):
        n = len(seeds)
        rngs = [np.random.default_rng(int(s)) for s in seeds]
        factors = atlas['rgb'].new_tensor(np.stack([rng.uniform(.8, 1.2, 3) for rng in rngs]))
        with torch.no_grad():
            rgb = atlas['rgb']
            valid_tiles = patches(atlas['valid'])
            # Contrast means are tile-local too; a global atlas mean would let
            # hidden pixel changes perturb visible inputs before local contrast.
            tiles = patches(rgb)
            tiles = photometric(tiles.flatten(0, 1), factors[:, None].expand(-1, 32, -1).reshape(-1, 3)).reshape(n, 32, 3, 64, 64)
            lc = contrast(tiles.flatten(0, 1), valid_tiles.flatten(0, 1),
                          {'local_contrast_window': 9, 'local_contrast_std_floor': .02, 'local_contrast_clip': 3})
            lc = lc.reshape(n, 32, 4, 64, 64).detach()
            targets = F.avg_pool2d(lc.flatten(0, 1), 4).reshape(n, 32, 4, 16, 16)
            loss_valid = patches(atlas['valid'] * (1-atlas['ignored']))
            weight = -F.max_pool2d(-loss_valid.flatten(0, 1), 4)
            weight = weight.reshape(n, 32, 1, 16, 16)
            supported = atlas['coverage'] >= .25
            chosen = torch.zeros((n, 32), dtype=torch.bool, device=rgb.device)
            for i, rng in enumerate(rngs):
                for band in range(8):
                    eligible = torch.where(supported[i, band*4:band*4+4])[0].cpu().tolist()
                    if len(eligible) >= 2:
                        chosen[i, band*4+int(rng.choice(eligible))] = True
            enough = (supported.sum(-1) >= 4) & (supported.reshape(n, 8, 4).any(-1).sum(-1) >= 2)
            enough &= (supported & ~chosen).sum(-1) >= 2
            enough &= (weight * chosen[..., None, None, None]).sum((1, 2, 3, 4)) > 0
            yy, xx = torch.meshgrid(torch.linspace(-1, 1, 8, device=rgb.device), torch.linspace(-1, 1, 4, device=rgb.device), indexing='ij')
            coordinates = torch.stack((yy, xx), -1).reshape(1, 32, 2).expand(n, -1, -1)
            meta_input = torch.cat((coordinates, atlas['coverage'][..., None], atlas['mass'][..., None]), -1)
        encoded = self.encoder(lc.flatten(0, 1)).mean((-2, -1)).reshape(n, 32, 96)
        metadata = self.metadata(meta_input)
        full = self.contextualize(encoded, metadata)
        masked = self.contextualize(encoded, metadata, chosen)
        embedding = (full * atlas['mass'][..., None]).sum(1)
        prediction = self.decoder(masked).reshape(n, 32, 4, 16, 16)
        return dict(encoded=encoded, full=full, masked_tokens=masked, embedding=embedding,
                    projected=self.projector(embedding), prediction=prediction, target=targets, weight=weight,
                    selected=chosen, eligible=enough, factors=factors, metadata=metadata, lc=lc)

    def forward(self, base, batch, seeds):
        refined = self.refine(base)
        atlas = canonicalize(batch['rgb'], refined['masks'], batch['masks'][:, 2:3])
        q = self.quality(self.quality_input(base, refined, atlas, batch['guides']))
        views = [self.texture_view(atlas, [int(s)*2+i for s in seeds]) for i in range(2)]
        return dict(refined=refined, atlas=atlas, quality=q, views=views)
