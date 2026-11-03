"""Hooks expose frozen original tensors without rewriting its forward or checkpoint keys."""
import torch
from common import read, HERE, sha
from data_geo import normalize
from frozen_anchor import TreadOne


class Anchor:
    def __init__(self, weights, device):
        assert sha(weights) == read(HERE / 'PLAN.json')['anchor']['weights_sha256'], 'Wrong anchor weights'
        self.device = torch.device(device)
        self.model = TreadOne().to(device).eval()
        state = torch.load(weights, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.requires_grad_(False)
        self.captured = {}
        self.model.d1.register_forward_hook(lambda m, a, y: self.captured.update(features=y))
        self.model.points.register_forward_pre_hook(lambda m, a: self.captured.update(point_features=a[0]))
        self.model.points.register_forward_hook(lambda m, a, y: self.captured.update(raw=y))

    @torch.no_grad()
    def __call__(self, rgb, guides, output_device):
        results = []
        for image, guide in zip(rgb, guides):
            self.captured.clear()
            out = self.model(normalize(image[None].to(self.device)), guide[None].to(self.device))
            results.append({**{k: v.detach().to(output_device) for k, v in self.captured.items()},
                            'anchor_logits': out['seg_logits'].detach().to(output_device),
                            'anchor_points': out['points'].detach().to(output_device)})
        self.captured.clear()
        return {name: torch.cat([r[name] for r in results]) for name in results[0]}
