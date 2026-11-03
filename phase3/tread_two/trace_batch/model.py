"""Untrained TRACE architecture prototype, isolated from every frozen experiment."""
import torch
from torch import nn
from targets import input_view


class Axial(nn.Module):
    def __init__(self):
        super().__init__()
        self.row = nn.Conv2d(48, 48, (3, 1), padding=(1, 0), groups=48)
        self.column = nn.Conv2d(48, 48, (1, 3), padding=(0, 1), groups=48)
        self.mix = nn.Sequential(nn.GroupNorm(6, 48), nn.Conv2d(48, 96, 1), nn.SiLU(), nn.Conv2d(96, 48, 1))
        self.scale = nn.Parameter(torch.full((1, 48, 1, 1), .1))

    def forward(self, tokens):
        x = tokens.transpose(1, 2).reshape(-1, 48, 8, 4)
        return (x + self.scale * self.mix(self.row(x) + self.column(x))).flatten(2).transpose(1, 2)


class Trace(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for cin, cout in zip((4, 16, 24, 32), (16, 24, 32, 48)):
            layers.extend((nn.Conv2d(cin, cout, 3, 2, 1), nn.GroupNorm(4, cout), nn.SiLU()))
        self.encoder = nn.Sequential(*layers)
        self.metadata = nn.Linear(3, 48)  # row, column, predicted coverage only
        self.context = nn.Sequential(Axial(), Axial())
        self.mask_token = nn.Parameter(torch.zeros(48))
        self.decoder = nn.Sequential(nn.Linear(48, 96), nn.SiLU(), nn.Linear(96, 96))
        self.projector = nn.Sequential(nn.Linear(240, 48), nn.SiLU(), nn.Linear(48, 6))

    @staticmethod
    def spatial_pool(tokens, mass):
        """Five fixed spatial moments retain row/column arrangement explicitly."""
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, 8, device=tokens.device), torch.linspace(-1, 1, 4, device=tokens.device), indexing='ij')
        basis = torch.stack((torch.ones_like(yy), yy, xx, yy * xx, yy.square() - 1/3), -1).reshape(32, 5)
        return torch.einsum('ntc,nt,tk->nkc', tokens, mass, basis).flatten(1)

    def forward(self, rgb, valid, hidden, factors=None):
        if hidden.shape != valid.shape[:2] or hidden.dtype != torch.bool:
            raise ValueError('Expected boolean N,32 hidden plan')
        coverage = valid.mean((-3, -2, -1)).detach()
        supported = coverage >= .25
        if (hidden & ~supported).any():
            raise ValueError('Cannot mask unsupported tiles')
        full_input = input_view(rgb, valid, factors)
        masked_input = input_view(rgb, valid, factors, hidden)
        n = len(rgb)
        full = self.encoder(full_input.flatten(0, 1)).mean((-2, -1)).reshape(n, 32, 48)
        visible = self.encoder(masked_input.flatten(0, 1)).mean((-2, -1)).reshape(n, 32, 48)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, 8, device=rgb.device), torch.linspace(-1, 1, 4, device=rgb.device), indexing='ij')
        pos = torch.stack((yy, xx), -1).reshape(1, 32, 2).expand(n, -1, -1)
        meta = self.metadata(torch.cat((pos, coverage[..., None]), -1))
        visible = torch.where((supported & ~hidden)[..., None], visible, self.mask_token)
        contextual = self.context(visible + meta)
        full_context = self.context(torch.where(supported[..., None], full, self.mask_token) + meta)
        mass = coverage.reshape(n, 8, 4) * supported.reshape(n, 8, 4)
        mass = mass / mass.sum(-1, keepdim=True).clamp_min(1e-8)
        mass = mass / (mass.sum(-1, keepdim=True) > 0).sum(1, keepdim=True).clamp_min(1)
        mass = mass.reshape(n, 32)
        local_pooled = (full * mass[..., None]).sum(1)
        context_mean = (full_context * mass[..., None]).sum(1)
        pooled = self.spatial_pool(full_context, mass)
        return {'encoded': full, 'full_context': full_context, 'context': contextual,
                'local_pooled': local_pooled, 'context_mean': context_mean,
                'pooled': pooled, 'projected': self.projector(pooled),
                'local_prediction': self.decoder(full).reshape(n, 32, 6, 4, 4),
                'masked_prediction': self.decoder(contextual).reshape(n, 32, 6, 4, 4),
                'mass': mass, 'supported': supported, 'hidden': hidden,
                'visible_input': masked_input}
