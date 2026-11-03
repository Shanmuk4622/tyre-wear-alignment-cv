"""Prototype loss arithmetic with explicit sufficient statistics and missingness."""
import torch
from torch.nn import functional as F


def reduction(prediction, target, weight, selected):
    w = weight * selected[..., None, None, None]
    error = F.smooth_l1_loss(prediction, target, reduction='none', beta=.1)
    sums = (error * w).sum((1, 2, 3, 4))
    counts = w.sum((1, 2, 3, 4)) * target.shape[2]
    available = counts > 0
    # Exact-zero empty results keep a gradient connection but never imply utility.
    loss = (sums / counts.clamp_min(1))[available].mean() if available.any() else prediction.sum() * 0
    return loss, {'sum': sums, 'count': counts, 'available': available}


def diversity(z, eligible):
    use = z[eligible]
    # Six nonzero decorrelated directions require at least seven eligible rows.
    # A small/invalid final batch must not quietly recreate GEO's rank conflict.
    if len(use) < z.shape[1] + 1:
        return z.sum() * 0, z.sum() * 0, False
    centered = use - use.mean(0)
    covariance = centered.T @ centered / (len(use) - 1)
    std = (covariance.diag() + 1e-4).sqrt()
    variance = F.relu(.1 - std).mean()
    off = covariance - torch.diag(covariance.diag())
    return variance, off.square().sum() / (z.shape[1] * (z.shape[1] - 1)), True


def objective(views, target, weight):
    if len(views) != 2:
        raise ValueError('Exactly two independently augmented views required')
    local, masked, var, cov, audit = [], [], [], [], []
    eligible = []
    for out in views:
        a, ar = reduction(out['local_prediction'], target, weight, out['supported'])
        b, br = reduction(out['masked_prediction'], target, weight, out['hidden'])
        band_count = out['supported'].reshape(-1, 8, 4).any(-1).sum(-1)
        ok = br['available'] & (out['supported'].sum(-1) >= 4) & (band_count >= 2) & ((out['supported'] & ~out['hidden']).sum(-1) >= 2)
        # Missing contextual support excludes a whole sample's masked objective.
        b = (br['sum'] / br['count'].clamp_min(1))[ok].mean() if ok.any() else b * 0
        c, d, available = diversity(out['projected'], ok)
        local.append(a); masked.append(b); var.append(c); cov.append(d); eligible.append(ok)
        audit.append({'local': ar, 'masked': br, 'eligible': ok, 'diversity_available': available})
    paired = eligible[0] & eligible[1]
    pair = (views[0]['projected'][paired] - views[1]['projected'][paired]).square().mean() if paired.any() else views[0]['projected'].sum() * 0
    terms = {'local': sum(local) / 2, 'masked': sum(masked) / 2, 'pair': pair, 'variance': sum(var) / 2, 'covariance': sum(cov) / 2}
    total = terms['local'] + terms['masked'] + .1 * terms['pair'] + .1 * terms['variance'] + .01 * terms['covariance']
    return total, terms, audit
