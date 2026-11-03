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

