"""Visible 2D tread coordinates from frozen predicted masks; no truth-mask input."""
import torch
from torch.nn import functional as F
HEIGHT=512;WIDTH=256;PATCH=64

@torch.no_grad()
def canonicalize(rgb,masks):
 assert rgb.ndim==4 and tuple(rgb.shape[1:])==(3,512,384) and tuple(masks.shape[1:])==(2,128,96)
 rgb=rgb.detach();binary=(masks>=.5).repeat_interleave(4,-2).repeat_interleave(4,-1);support=binary[:,0]&binary[:,1]
 xs=torch.arange(384,device=rgb.device)[None,None];left=torch.where(support,xs,384).amin(-1);right=torch.where(support,xs,-1).amax(-1);row_valid=right-left+1>=16
 grids=[];diagnostics=[];sampled_valid=[]
 for i in range(len(rgb)):
  rows=torch.where(row_valid[i])[0]
  if not len(rows):
   grids.append(torch.zeros((HEIGHT,WIDTH,2),device=rgb.device));sampled_valid.append(torch.zeros(HEIGHT,device=rgb.device,dtype=torch.bool));diagnostics.append(dict(supported=False,source_first_row=None,source_last_row=None,clipped_row_fraction=None));continue
  yy=torch.linspace(int(rows[0]),int(rows[-1]),HEIGHT,device=rgb.device);indices=yy.round().long();ok=row_valid[i,indices];l=left[i,indices].float();r=right[i,indices].float();u=torch.linspace(0,1,WIDTH,device=rgb.device)
  xx=l[:,None]+u[None]*(r-l)[:,None];grid=torch.stack((2*xx/383-1,(2*yy/511-1)[:,None].expand(-1,WIDTH)),-1);grid[~ok]=0;grids.append(grid);sampled_valid.append(ok)
  clipped=(left[i,rows]==0)|(right[i,rows]==383);diagnostics.append(dict(supported=True,source_first_row=int(rows[0]),source_last_row=int(rows[-1]),clipped_row_fraction=float(clipped.float().mean())))
 grid=torch.stack(grids);valid=F.grid_sample(support[:,None].float(),grid,mode='nearest',padding_mode='zeros',align_corners=True)
 valid*=torch.stack(sampled_valid)[:,None,:,None]
 atlas=F.grid_sample(rgb,grid,mode='bilinear',padding_mode='zeros',align_corners=True)*valid
 return dict(rgb=atlas,valid=valid,grid=grid,diagnostics=diagnostics)

def patches(atlas):
 b,c,h,w=atlas.shape;assert (h,w)==(HEIGHT,WIDTH)
 return atlas.unfold(2,PATCH,PATCH).unfold(3,PATCH,PATCH).permute(0,2,3,1,4,5).reshape(b,32,c,PATCH,PATCH)

def band_weights(valid):
 coverage=patches(valid).mean((-3,-2,-1));rows=coverage.reshape(-1,8,4);row_sum=rows.sum(-1,keepdim=True);active=(row_sum>0).sum(1,keepdim=True).clamp_min(1)
 return (rows/row_sum.clamp_min(1e-9)/active).reshape(-1,32),coverage
