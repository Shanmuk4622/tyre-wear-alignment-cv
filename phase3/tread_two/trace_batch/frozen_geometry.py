"""Explicitly reuse GEO's unselected epoch-60 geometry, never its texture branch.

Requires an already SHA-verified native export and the original TREAD-ONE source
class, supplied by the integration layer. No implicit import/download/token use.
"""
import hashlib
from pathlib import Path
import torch
from torch import nn

EXPORT_SHA256 = '8358e0595d46caee1160818ad975042f2031e14f7d0fa6f2ac105b9312325450'
SOURCE_SHA256 = 'd16cd4eef75779044dd95900c0531fee474704cf88f523326ee069115347ba09'
PROVENANCE = 'GEO_r1_epoch60_final_unselected_research_geometry_only'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for body in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(body)
    return h.hexdigest()


class FrozenGeometry(nn.Module):
    def __init__(self, export_path, source_path, anchor_class, device='cpu'):
        super().__init__()
        if sha(export_path) != EXPORT_SHA256 or sha(source_path) != SOURCE_SHA256:
            raise ValueError('GEO export or original anchor source differs from pinned identity')
        state = torch.load(export_path, map_location='cpu', weights_only=True)
        self.anchor = anchor_class()
        self.anchor.load_state_dict(state['anchor'], strict=True)
        self.mask_adapter = nn.Sequential(nn.Conv2d(48, 24, 3, padding=1), nn.GroupNorm(8, 24), nn.SiLU(), nn.Conv2d(24, 2, 1))
        self.point_adapter = nn.Sequential(nn.Linear(256, 64), nn.SiLU(), nn.Linear(64, 2))
        for name in ('mask_adapter', 'point_adapter'):
            prefix = name + '.'
            getattr(self, name).load_state_dict({k[len(prefix):]: v for k, v in state['trainable'].items() if k.startswith(prefix)}, strict=True)
        self.capture = {}
        self.anchor.d1.register_forward_hook(lambda m, a, y: self.capture.update(features=y))
        self.anchor.points.register_forward_pre_hook(lambda m, a: self.capture.update(point_features=a[0]))
        self.anchor.points.register_forward_hook(lambda m, a, y: self.capture.update(raw=y))
        self.to(device).eval().requires_grad_(False)

    def train(self, mode=True):
        # The enclosing future bundle may enter train mode; this stays frozen.
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, rgb, guides):
        device = next(self.parameters()).device
        if tuple(rgb.shape[1:]) != (3, 512, 384) or tuple(guides.shape) != (len(rgb), 3):
            raise ValueError('Expected N,3,512,384 RGB and N,3 guides')
        if not len(rgb) or not torch.isfinite(rgb).all() or not torch.isfinite(guides).all():
            raise ValueError('Empty/nonfinite geometry input')
        outputs = []
        for image, guide in zip(rgb, guides):
            image = image[None].to(device)
            normal = (image - image.new_tensor([.485, .456, .406])[None, :, None, None]) / image.new_tensor([.229, .224, .225])[None, :, None, None]
            self.capture.clear()
            result = self.anchor(normal, guide[None].to(device))
            outputs.append({**self.capture, 'anchor_logits': result['seg_logits']})
        self.capture.clear()
        base = {k: torch.cat([v[k] for v in outputs]) for k in outputs[0]}
        logits = base['anchor_logits'] + self.mask_adapter(base['features']).tanh()
        raw = base['raw'] + .25 * self.point_adapter(base['point_features']).tanh()
        left = .99 * raw[..., 0].sigmoid()
        right = left + (1 - left) * raw[..., 1].sigmoid()
        return {'seg_logits': logits, 'masks': logits.sigmoid(),
                'points': torch.stack((left, right), -1).flatten(1)}

