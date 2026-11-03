"""TRACE design prototype: fixed, tile-local image targets; no fitted model input.

The observation mask is predicted geometry. Annotation ignore pixels only alter
target weights, never student features. All spatial operations stay inside one
64x64 tile. Nothing in this module reads a dataset role implicitly.
"""
import torch
from torch.nn import functional as F

CHANNELS = ('contrast', 'gradient_energy', 'orientation_axis', 'orientation_cross', 'relative_curvature', 'contrast_occupancy')
TARGET_REVISION = 'trace-fixed-target-v2'
RADIUS = 5  # local contrast radius 4, followed by Sobel radius 1


def tiles(atlas):
    if atlas.ndim != 4 or tuple(atlas.shape[-2:]) != (512, 256):
        raise ValueError('Expected N,C,512,256 atlas')
    n, c = atlas.shape[:2]
    return atlas.unfold(2, 64, 64).unfold(3, 64, 64).permute(0, 2, 3, 1, 4, 5).reshape(n, 32, c, 64, 64)


def contrast(rgb, valid):
    """Return four bounded input fields; tile-local means and GroupNorm later."""
    n, t = rgb.shape[:2]
    x = rgb.reshape(-1, 3, 64, 64)
    v = valid.reshape(-1, 1, 64, 64)
    gray = x.mean(1, keepdim=True)
    mass = F.avg_pool2d(v, 9, 1, 4).clamp_min(1e-8)
    mean = F.avg_pool2d(gray * v, 9, 1, 4) / mass
    var = (F.avg_pool2d(gray.square() * v, 9, 1, 4) / mass - mean.square()).clamp_min(0)
    z = ((gray - mean) / var.sqrt().clamp_min(.02)).clamp(-3, 3) / 3 * v
    k = x.new_tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]])[None, None] / 8
    dx = F.conv2d(z, k, padding=1) * v
    dy = F.conv2d(z, k.transpose(-1, -2), padding=1) * v
    return torch.cat((z, dx, dy, (dx.square() + dy.square() + 1e-12).sqrt() * v), 1).reshape(n, t, 4, 64, 64)


def eroded(valid):
    """Reject any target pixel whose radius-five dependency touches invalidity."""
    flat = valid.reshape(-1, 1, 64, 64)
    # Explicit invalid padding also rejects the outer five pixels of a tile.
    padded = F.pad(flat, (RADIUS,) * 4, value=0)
    return (-F.max_pool2d(-padded, 2 * RADIUS + 1, 1)).reshape_as(valid)


def descriptor_fields(c):
    """Six interpretable fields; orientation and curvature separate energy scale."""
    z, dx, dy = c[:, :, :1], c[:, :, 1:2], c[:, :, 2:3]
    energy2 = dx.square() + dy.square()
    axis = (dx.square() - dy.square()) / (energy2 + .0025)
    cross = 2 * dx * dy / (energy2 + .0025)
    laplacian = z.new_tensor([[0., 1, 0], [1, -4, 1], [0, 1, 0]])[None, None] / 4
    curvature = F.conv2d(z.flatten(0, 1), laplacian, padding=1).reshape_as(z).abs()
    curvature = (curvature / (dx.abs() + dy.abs() + .05)).clamp(0, 1)
    occupancy = (z.abs() > .15).to(z.dtype)
    return torch.cat((z, energy2.sqrt(), axis, cross, curvature, occupancy), 2)


@torch.no_grad()
def fixed_targets(rgb, valid, loss_valid=None):
    if tuple(rgb.shape[1:]) != (32, 3, 64, 64) or tuple(valid.shape[1:]) != (32, 1, 64, 64):
        raise ValueError('Expected N,32,3/1,64,64 tiles')
    if not torch.isfinite(rgb).all() or not torch.isfinite(valid).all():
        raise ValueError('Nonfinite target input')
    if not ((valid == 0) | (valid == 1)).all():
        raise ValueError('Support must be binary')
    if loss_valid is not None and (loss_valid.shape != valid.shape or not torch.isfinite(loss_valid).all() or not ((loss_valid == 0) | (loss_valid == 1)).all()):
        raise ValueError('Loss support must be matching, finite and binary')
    use = valid if loss_valid is None else valid * loss_valid
    weight = eroded(use)
    c = contrast(rgb.detach(), valid.detach())
    body = descriptor_fields(c)
    n = len(rgb)
    counts = F.avg_pool2d(weight.flatten(0, 1), 16) * 256
    sums = F.avg_pool2d((body * weight).flatten(0, 1), 16) * 256
    target = (sums / counts.clamp_min(1)).reshape(n, 32, 6, 4, 4)
    counts = counts.reshape(n, 32, 1, 4, 4)
    # Fixed count threshold; unavailable cells never become fabricated targets.
    available = counts >= 64
    return {'target': target * available, 'weight': counts * available,
            'input': c.detach(), 'available': available, 'pixel_weight': weight}


@torch.no_grad()
def fit_statistics(target, weight, roles):
    """Weighted channel statistics and position control, strictly training-only."""
    if len(roles) != len(target) or not roles or set(roles) != {'train'}:
        raise ValueError('Statistics require exclusively declared train rows')
    if target.ndim != 5 or tuple(target.shape[1:]) != (32, 6, 4, 4) or tuple(weight.shape) != (len(target), 32, 1, 4, 4):
        raise ValueError('Unexpected statistics shapes')
    if not torch.isfinite(target).all() or not torch.isfinite(weight).all() or (weight < 0).any():
        raise ValueError('Nonfinite or negative statistics input')
    x, w = target.double(), weight.double()
    total = w.sum((0, 1, 3, 4))
    if not (total > 0).all():
        raise ValueError('No supported training target cells')
    mean = (x * w).sum((0, 1, 3, 4)) / total
    variance = ((x - mean[None, None, :, None, None]).square() * w).sum((0, 1, 3, 4)) / total
    std = variance.sqrt().clamp_min(.001)
    position_count = w.sum(0)
    position_mean = (x * w).sum(0) / position_count.clamp_min(1)
    return {'mean': mean.float(), 'std': std.float(), 'count': total.float(),
            'position_mean': position_mean.float(), 'position_count': position_count.float()}


def standardized(target, statistics):
    return (target - statistics['mean'][None, None, :, None, None]) / statistics['std'][None, None, :, None, None]


def choose_hidden(valid, seeds):
    """At most one hidden supported tile per four-column band, never all context."""
    if len(seeds) != len(valid):
        raise ValueError('One explicit seed per image required')
    support = valid.mean((-3, -2, -1)) >= .25
    selected = torch.zeros_like(support)
    for i, seed in enumerate(seeds):
        gen = torch.Generator(device='cpu').manual_seed(int(seed))
        for band in range(8):
            eligible = torch.where(support[i, 4 * band:4 * band + 4].cpu())[0]
            if len(eligible) >= 2:
                selected[i, 4 * band + int(eligible[torch.randint(len(eligible), (), generator=gen)])] = True
    return selected


def input_view(rgb, valid, factors=None, hidden=None):
    """Mask raw pixels BEFORE any normalization/encoding. No global RGB statistics."""
    x, v = rgb.detach(), valid.detach()
    if hidden is not None:
        gate = (~hidden)[..., None, None, None]
        x, v = x * gate, v * gate
    if factors is not None:
        if tuple(factors.shape) != (len(x), 3):
            raise ValueError('Expected brightness/contrast/gamma per image')
        if not torch.isfinite(factors).all() or (factors <= 0).any():
            raise ValueError('Photometric factors must be positive and finite')
        b, c, g = [factors[:, k, None, None, None, None] for k in range(3)]
        x = (x * b).clamp(0, 1)
        mean = (x * v).sum((-2, -1), keepdim=True) / v.sum((-2, -1), keepdim=True).clamp_min(1)
        x = ((x - mean) * c + mean).clamp(0, 1).clamp_min(1e-8).pow(g).clamp(0, 1) * v
    return contrast(x, v)
