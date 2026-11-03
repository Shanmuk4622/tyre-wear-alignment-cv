"""Fixed controls, original-role diversity and predeclared validation selection."""
import numpy as np
import torch
from torch.nn import functional as F
from targets import standardized

def latent(z):
    z=np.asarray(z,dtype='float64')
    if len(z)<2:return dict(rank=0.,std=None,leading=None,eigenvalues=[])
    centered=z-z.mean(0);cov=centered.T@centered/(len(z)-1)
    eigen=np.linalg.eigvalsh(cov).clip(0)[::-1];total=eigen.sum()
    p=eigen[eigen>0]/total if total else np.array([])
    return dict(rank=float(np.exp(-(p*np.log(p)).sum())) if len(p) else 0.,std=float(np.median(np.sqrt(np.diag(cov).clip(0)))),
                leading=float(eigen[0]/total) if total else 0.,eigenvalues=eigen.tolist())

def scatter(z,components):
    z=np.asarray(z,dtype='float64');components=np.asarray(components);within=np.zeros_like(z)
    for name in np.unique(components):
        use=components==name;within[use]=z[use]-z[use].mean(0)
    centered=z-z.mean(0);between=centered-within
    t,w,b=[float((v*v).sum()) for v in (centered,within,between)]
    assert np.isclose(t,w+b,atol=1e-8,rtol=1e-10)
    return dict(total=t,within=w,between=b,fraction=b/t if t else 0.,error=t-w-b,within_vectors=within)

def controls(target,weight,hidden,stats):
    """Same cell intersection for model and all three controls."""
    position=stats['position_mean'][None].expand_as(target)
    neighbor=torch.zeros_like(target);ok=torch.zeros_like(weight)
    for band in range(8):
        sl=slice(4*band,4*band+4);w=weight[:,sl]*(~hidden[:,sl])[...,None,None,None]
        count=w.sum(1,keepdim=True)
        neighbor[:,sl]=(target[:,sl]*w).sum(1,keepdim=True)/count.clamp_min(1)
        ok[:,sl]=(count>0).expand(-1,4,-1,-1,-1)
    common=weight*hidden[...,None,None,None]*ok*(stats['position_count'][None]>0)
    values={'channel_mean':torch.zeros_like(target),'position_mean':standardized(position,stats),'visible_band_mean':standardized(neighbor,stats)}
    return values,common

def sums(pred,target,weight):
    return (F.smooth_l1_loss(pred,target,beta=.1,reduction='none')*weight).sum((1,2,3,4)),weight.sum((1,2,3,4))*6

def evaluate(samples,cfg,primary_fraction=1.):
    valid=[s for s in samples if s['eligible']]
    local=[s for s in samples if s['local_count']>0]
    def avg(key,rows=valid):
        a=[s[key] for s in rows if s.get(key) is not None];return float(np.mean(a)) if a else None
    ctr={name:avg('control_'+name) for name in ['channel_mean','position_mean','visible_band_mean']}
    model=avg('model_common');local_loss=avg('local_loss',local);local_control=avg('local_control',local)
    spectra={};within={}
    for name in ['projected','local_pooled','pooled','context_mean']:
        z=np.array([s[name] for s in valid],dtype='float64')
        spectra[name]=latent(z) if len(z) else dict(rank=0.,std=None,leading=None,eigenvalues=[])
        within[name]=latent(scatter(z,[s['component'] for s in valid])['within_vectors']) if len(z)>1 else dict(rank=0.)
    rule=cfg['selection'];coverage=avg('coverage')
    denom=all(v is not None and np.isfinite(v) and v>1e-8 for v in ctr.values())
    masked_gain={k:1-model/v if denom and model is not None else None for k,v in ctr.items()}
    local_gain=1-local_loss/local_control if local_loss is not None and local_control and local_control>1e-8 else None
    cosine=avg('cosine');fraction=len(valid)/len(samples) if samples else 0.
    gates=dict(geometry=True,exposure=fraction>=rule['eligible_primary_fraction'] and primary_fraction>=rule['eligible_primary_fraction'],
        control_coverage=coverage is not None and coverage>=rule['coverage'] and denom,
        masked_utility=denom and all(v is not None and v>=rule['masked_gain'] for v in masked_gain.values()),
        local_utility=local_gain is not None and local_gain>=rule['local_gain'],
        projector_diversity=spectra['projected']['rank']>=rule['projector_rank'] and (spectra['projected']['std'] or 0)>=rule['projector_std'],
        encoder_diversity=spectra['local_pooled']['rank']>=rule['local_rank'] and spectra['pooled']['rank']>=rule['spatial_rank'],
        within_component_diversity=within['local_pooled']['rank']>=rule['within_local_rank'] and within['pooled']['rank']>=rule['within_spatial_rank'],
        finite=model is not None and np.isfinite(model) and cosine is not None and np.isfinite(cosine))
    gates={k:bool(v) for k,v in gates.items()}
    eligible=all(gates.values());score=model/min(ctr.values())+.1*cosine if eligible else None
    return dict(gates=gates,eligible=eligible,score=score,rejection=';'.join(k for k,v in gates.items() if not v),images=len(samples),eligible_images=len(valid),
                primary_fraction=primary_fraction,coverage=coverage,model=model,controls=ctr,masked_gain=masked_gain,local=local_loss,local_control=local_control,local_gain=local_gain,cosine=cosine,spectra=spectra,within=within)
