"""Fixed symmetric appearance objective; no held-out/reference labels enter pairs."""
import torch
from torch.nn import functional as F
def objective(a,b,labels):
    ca=F.cross_entropy(a['logits'],labels,reduction='none')
    cb=F.cross_entropy(b['logits'],labels,reduction='none')
    p,q=a['probabilities'],b['probabilities'];m=(p+q)/2
    js=.5*((p*(p.clamp_min(1e-12).log()-m.clamp_min(1e-12).log())).sum(-1)+(q*(q.clamp_min(1e-12).log()-m.clamp_min(1e-12).log())).sum(-1))
    distance=1-F.cosine_similarity(a['query_embeddings'],b['query_embeddings'],dim=-1)
    texture=(distance*a['patch_weights']).sum(-1)
    total=.5*(ca+cb)+.1*js+.05*texture
    return total.mean(),dict(ce_a=ca,ce_b=cb,js=js,texture=texture,total=total,patch_distance=distance)
