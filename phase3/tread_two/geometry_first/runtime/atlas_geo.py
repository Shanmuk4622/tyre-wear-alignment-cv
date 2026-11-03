"""Detached predicted-support atlas and sufficient row parameters for exact replay."""
import torch
from torch.nn import functional as F
from geometry import patches, band_weights


def grid_from_rows(parameters, valid):
    # parameters: [N,512,3] = source-y, left, right in input pixels.
    yy, left, right = parameters.unbind(-1)
    u = torch.linspace(0, 1, 256, device=parameters.device)
    xx = left[..., None] + u * (right - left)[..., None]
    grid = torch.stack((2 * xx / 383 - 1, (2 * yy / 511 - 1)[..., None].expand(-1, -1, 256)), -1)
    return grid.masked_fill(~valid[..., None, None], 0)


@torch.no_grad()
def canonicalize(rgb, masks, ignore=None):
    rgb, masks = rgb.detach(), masks.detach()
    binary = (masks >= .5).repeat_interleave(4, -2).repeat_interleave(4, -1)
    support = binary[:, 0] & binary[:, 1]
    xs = torch.arange(384, device=rgb.device)[None, None]
    left = torch.where(support, xs, 384).amin(-1)
    right = torch.where(support, xs, -1).amax(-1)
    good = right - left + 1 >= 16
    parameters, flags = [], []
    for i in range(len(rgb)):
        rows = torch.where(good[i])[0]
        if not len(rows):
            parameters.append(rgb.new_zeros((512, 3)))
            flags.append(torch.zeros(512, device=rgb.device, dtype=torch.bool))
            continue
        y = torch.linspace(int(rows[0]), int(rows[-1]), 512, device=rgb.device)
        indices = y.round().long()
        parameters.append(torch.stack((y, left[i, indices].float(), right[i, indices].float()), -1))
        flags.append(good[i, indices])
    parameters, flags = torch.stack(parameters), torch.stack(flags)
    grid = grid_from_rows(parameters, flags)
    valid = F.grid_sample(support[:, None].float(), grid, mode='nearest', align_corners=True)
    valid *= flags[:, None, :, None]
    atlas = F.grid_sample(rgb, grid, mode='bilinear', align_corners=True) * valid
    ignored = torch.zeros_like(valid) if ignore is None else F.grid_sample(ignore.detach(), grid, mode='nearest', align_corners=True)
    mass, coverage = band_weights(valid)
    return dict(rgb=atlas.detach(), valid=valid.detach(), ignored=ignored.detach(),
                row_parameters=parameters, row_valid=flags, grid=grid,
                mass=mass, coverage=coverage)
