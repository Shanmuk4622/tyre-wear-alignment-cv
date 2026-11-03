"""Explicit reductions for geometry, human-only error targets and VICReg texture loss."""
import torch
from torch.nn import functional as F


def coarse_truth(masks):
    n, _, h, w = masks.shape
    target = masks[:, :2].reshape(n, 2, h//4, 4, w//4, 4).mean((3, 5))
    valid = (1-masks[:, 2:3]).reshape(n, 1, h//4, 4, w//4, 4).amin((3, 5))
    return target, valid


def full_geometry(refined, batch):
    pred = (refined['masks'].detach() >= .5).repeat_interleave(4, -2).repeat_interleave(4, -1)
    truth, valid = batch['masks'][:, :2] > .5, batch['masks'][:, 2:3] < .5
    tp = (pred & truth & valid).sum((-2, -1))
    fp = (pred & ~truth & valid).sum((-2, -1))
    fn = (~pred & truth & valid).sum((-2, -1))
    denominator = 2*tp+fp+fn
    dice = torch.where(denominator > 0, 2*tp.float()/denominator.clamp_min(1), torch.ones_like(tp).float())
    point_error = (refined['points'].detach()-batch['points']).abs().mean(-1)
    return dict(tp=tp, fp=fp, fn=fn, dice=dice, point_error=point_error,
                target_valid=valid.sum((-2, -1)).expand(-1, 2)>0)


def geometry_loss(refined, base, batch):
    p, logits = refined['masks'], refined['seg_logits']
    y, valid = coarse_truth(batch['masks'])
    bce_num = (F.binary_cross_entropy_with_logits(logits, y, reduction='none')*valid).sum()
    bce_den = 2*valid.sum()
    bce = bce_num/bce_den.clamp_min(1)
    dice = (2*(p*y*valid).sum((-2, -1))+1)/((p*valid).sum((-2, -1))+(y*valid).sum((-2, -1))+1)
    segmentation = bce+1-dice.mean()
    point_sum = (F.smooth_l1_loss(refined['points'], batch['points'], reduction='none', beta=.02).mean(-1)*batch['point_weight']).sum()
    point_den = batch['point_weight'].sum()
    point = point_sum/point_den.clamp_min(1e-12)
    containment_sum = (F.relu(p[:, 1:2]-p[:, :1])*valid).sum()
    containment_den = valid.sum()
    containment = containment_sum/containment_den.clamp_min(1)
    index = (batch['guides']*(p.shape[-2]-1)).round().long()
    rows = torch.stack([p[i, 1, index[i]] for i in range(len(p))])
    row_valid = torch.stack([valid[i, 0, index[i]] for i in range(len(p))])
    reference = torch.stack([y[i, 1, index[i]] for i in range(len(p))])
    usable = (reference[..., 0]<.1) & (reference[..., -1]<.1) & (reference.sum(-1)>2)
    x = torch.linspace(0, 1, p.shape[-1], device=p.device)[None, None]
    q = refined['points'].reshape(-1, 3, 2)
    interval = ((x-q[..., :1])/.015).sigmoid()*((q[..., 1:]-x)/.015).sigmoid()
    weights = row_valid*usable[..., None]
    row_sum = ((interval-rows.detach()).abs()*weights).sum()
    row_den = weights.sum()
    row_loss = row_sum/row_den.clamp_min(1)
    drift_mask = ((p-base['anchor_logits'].sigmoid()).square()*valid).sum()/bce_den.clamp_min(1)
    drift_points = (refined['points']-base['anchor_points']).square().mean()
    drift = drift_mask+drift_points
    total = segmentation+2*point+.1*containment+.1*row_loss+.1*drift
    return total, dict(bce_numerator=bce_num, bce_denominator=bce_den, soft_dice_loss=1-dice.mean(),
                      point_numerator=point_sum, point_denominator=point_den,
                      containment_numerator=containment_sum, containment_denominator=containment_den,
                      row_numerator=row_sum, row_denominator=row_den, anchor_drift=drift, geometry=total)


def quality_loss(prediction, refined, batch):
    metrics = full_geometry(refined, batch)
    targets = torch.cat((1-metrics['dice'], metrics['point_error'][:, None]), -1).detach()
    availability = torch.cat((metrics['target_valid'], batch['human'][:, None]), -1)
    numerator = (F.smooth_l1_loss(prediction, targets, reduction='none', beta=.02)*availability).sum()
    denominator = availability.sum()
    return numerator/denominator.clamp_min(1), dict(targets=targets, available=availability,
                                                  numerator=numerator, denominator=denominator)


def variance_covariance(z):
    centered = z-z.mean(0)
    covariance = centered.T@centered/(len(z)-1)
    std = (covariance.diag()+1e-4).sqrt()
    variance = F.relu(1-std).mean()
    off = covariance - torch.diag_embed(covariance.diag())
    return variance, off.square().sum()/z.shape[-1]


def texture_loss(views, components, epoch):
    a, b = views
    eligible = a['eligible'] & b['eligible']
    indices = torch.where(eligible)[0].cpu().tolist()
    count, component_count = len(indices), len({components[i] for i in indices})
    valid = count >= 4 and component_count >= 2
    zero = a['projected'].sum()*0
    values = dict(eligible_images=count, eligible_components=component_count, texture_step=valid,
                  ramp=min(1., epoch/5), reason='' if valid else 'fewer_than_4_images_or_2_components')
    if not valid:
        values.update(invariance=zero, variance_a=zero, variance_b=zero, covariance_a=zero,
                      covariance_b=zero, masked_numerator=zero, masked_denominator=zero, masked_huber=zero, texture=zero)
        return zero, values
    za, zb = a['projected'][eligible], b['projected'][eligible]
    inv = (za-zb).square().mean()
    va, ca = variance_covariance(za)
    vb, cb = variance_covariance(zb)
    numerators, denominators = [], []
    for view in views:
        weights = view['weight']*view['selected'][..., None, None, None]*eligible[:, None, None, None, None]
        numerator = (F.smooth_l1_loss(view['prediction'], view['target'], beta=.1, reduction='none')*weights).sum()
        denominator = weights.sum()*4
        numerators.append(numerator)
        denominators.append(denominator)
    numerator, denominator = sum(numerators), sum(denominators)
    assert float(denominator) > 0
    masked = numerator/denominator
    total = 25*inv+25*(va+vb)+ca+cb+.5*masked
    values.update(invariance=inv, variance_a=va, variance_b=vb, covariance_a=ca, covariance_b=cb,
                  masked_numerator=numerator, masked_denominator=denominator, masked_huber=masked, texture=total)
    return values['ramp']*total, values


def losses(out, base, batch, seeds, epoch):
    geo, gs = geometry_loss(out['refined'], base, batch)
    quality, qs = quality_loss(out['quality'], out['refined'], batch)
    texture, ts = texture_loss(out['views'], [r['component'] for r in batch['rows']], epoch)
    return geo+.25*quality+texture, {'geometry': gs, 'quality': qs, 'texture': ts}
