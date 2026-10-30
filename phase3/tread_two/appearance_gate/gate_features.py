"""Fixed train-only local-contrast diagnostics and deterministic case diversity."""
import math
import numpy as np
import torch
from torch.nn import functional as F


def contrast(tiles,valid,rules):
    """Masked local mean/std; four explicitly non-depth texture channels.

    Channels: clipped normalized luminance, Sobel x/y of that luminance,
    gradient magnitude. Invalid values stay zero. The std floor, clipping and
    input saturation limit exact multiplicative-brightness equivariance.
    """
    gray=tiles.mean(1,keepdim=True)
    window=rules['local_contrast_window'];pad=window//2
    mass=F.avg_pool2d(valid,window,1,pad).clamp_min(1e-8)
    mean=F.avg_pool2d(gray*valid,window,1,pad)/mass
    variance=(F.avg_pool2d(gray.square()*valid,window,1,pad)/mass-mean.square()).clamp_min(0)
    std=variance.sqrt().clamp_min(rules['local_contrast_std_floor'])
    z=((gray-mean)/std).clamp(-rules['local_contrast_clip'],rules['local_contrast_clip'])/rules['local_contrast_clip']*valid
    kernel=tiles.new_tensor([[-1.,0,1],[-2,0,2],[-1,0,1]])[None,None]/8
    dx=F.conv2d(z,kernel,padding=1)*valid
    dy=F.conv2d(z,kernel.transpose(-1,-2),padding=1)*valid
    return torch.cat((z,dx,dy,(dx.square()+dy.square()+1e-12).sqrt()*valid),1)


def descriptor(features,valid,grid=8):
    mass=F.adaptive_avg_pool2d(valid,grid).clamp_min(1e-8)
    return (F.adaptive_avg_pool2d(features*valid,grid)/mass).flatten(1)


def quality(rgb,valid):
    pixels=rgb.permute(1,2,0)[valid[0]>.5]
    if not len(pixels):
        return dict(valid_fraction=0.,gray_mean=None,gray_std=None,mean_gradient=None,glare_fraction=None,dark_fraction=None,chroma_fraction=None,low_texture=1)
    gray=rgb.mean(0)
    usable=valid[0]>.5
    values=gray[usable]
    horizontal=(gray[:,1:]-gray[:,:-1]).abs();vertical=(gray[1:]-gray[:-1]).abs()
    hv=usable[:,1:]&usable[:,:-1];vv=usable[1:]&usable[:-1]
    means=[horizontal[hv].mean() if hv.any() else gray.new_tensor(0),vertical[vv].mean() if vv.any() else gray.new_tensor(0)]
    grad=float(torch.stack(means).mean())
    maximum,minimum=pixels.max(1).values,pixels.min(1).values
    saturation=(maximum-minimum)/maximum.clamp_min(1e-6)
    return dict(valid_fraction=float(usable.float().mean()),gray_mean=float(values.mean()),gray_std=float(values.std(unbiased=False)),
        mean_gradient=grad,glare_fraction=float((values>=.98).float().mean()),dark_fraction=float((values<=.02).float().mean()),
        chroma_fraction=float(((saturation>.2)&(values>.3)).float().mean()),low_texture=int(grad<.008))


def choose_bank(records,rules):
    """No model predictions enter eligibility or the deterministic diversity score."""
    assert all(r.get('role')=='train' for r in records),'Reference proposal must never use validation/test records'
    selected=[]
    for cls in range(3):
        pool=[r for r in records if r['class_index']==cls and r['domain']=='original_photo'
              and r['valid_fraction']>=rules['candidate_min_valid_fraction']
              and r['glare_fraction']<=rules['candidate_max_glare_fraction']
              and r['gray_std']>=rules['candidate_min_gray_std']]
        groups=sorted({r['physical_tyre_id'] for r in pool})
        group_photos={g:len({r['image_id'] for r in records if r['domain']=='original_photo' and r['physical_tyre_id']==g}) for g in groups}
        chosen=[];counts={g:0 for g in groups};used=set()
        for _ in range(rules['candidate_cases_per_class']):
            candidates=[r for r in pool if r['image_id'] not in used and counts[r['physical_tyre_id']]<group_photos[r['physical_tyre_id']]-rules['minimum_remaining_original_query_photos_per_tyre']]
            if not candidates: break
            fewest=min(counts[r['physical_tyre_id']] for r in candidates)
            candidates=[r for r in candidates if counts[r['physical_tyre_id']]==fewest]
            def score(r):
                x=np.asarray(r['descriptor'],dtype=np.float64)
                x=x/max(np.linalg.norm(x),1e-12)
                distance=min(1-float(x@(np.asarray(s['descriptor'])/max(np.linalg.norm(s['descriptor']),1e-12))) for s in chosen) if chosen else 1.
                return distance+.1*r['valid_fraction']-.1*r['glare_fraction']
            winner=sorted(candidates,key=lambda r:(-score(r),r['image_id'],r['patch']))[0]
            chosen.append(winner);used.add(winner['image_id']);counts[winner['physical_tyre_id']]+=1
        selected.extend(chosen)
    return selected
