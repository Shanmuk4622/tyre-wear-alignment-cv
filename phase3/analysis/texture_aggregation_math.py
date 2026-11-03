"""Pure algebra for TRACE saved-token aggregation; no trainable model or images."""
import numpy as np

def bases(seed=462225):
    y,x=np.meshgrid(np.linspace(-1,1,8),np.linspace(-1,1,4),indexing='ij')
    moments=np.stack([np.ones_like(y),y,x,y*x,y*y-1/3],-1).reshape(32,5)
    rng=np.random.default_rng(seed);random=rng.normal(size=(32,4));random-=random.mean(0);q=np.linalg.qr(random)[0]
    return dict(identity=np.eye(32),global_mean=np.ones((32,1)),band_means=np.repeat(np.eye(8),4,axis=0),
        column_means=np.tile(np.eye(4),(8,1)),moments5=moments,random5=np.column_stack([np.ones(32),q]))

def project(z,mass,basis):
    gram=np.einsum('nvt,tk,tl->nvkl',mass,basis,basis)
    rhs=np.einsum('nvt,tk,nvtd->nvkd',mass,basis,z)
    coef=np.linalg.pinv(gram,rcond=1e-10)@rhs
    return np.einsum('tk,nvkd->nvtd',basis,coef)

def aggregate(z,mass,variant,template,basis):
    if variant=='identity':return z.copy()
    if variant=='train_template':return np.broadcast_to(template[None,None],z.shape).copy()
    if variant.startswith('residual_'):
        name='global_mean' if variant=='residual_mean' else 'moments5'
        return template[None,None]+project(z-template[None,None],mass,basis[name])
    return project(z,mass,basis[variant])

def mlp(x,weights,prefix):
    z=x@weights[prefix+'.0.weight'].T+weights[prefix+'.0.bias']
    z=z*np.exp(-np.logaddexp(0.,-z))
    return z@weights[prefix+'.2.weight'].T+weights[prefix+'.2.bias']

def huber(pred,target):
    delta=np.abs(pred-target);return np.where(delta<.1,5*delta*delta,delta-.05)

def spectrum(x):
    x=np.asarray(x,dtype='float64');center=x-x.mean(0);n,d=x.shape
    cov=center@center.T/(n-1) if d>n else center.T@center/(n-1)
    eigen=np.linalg.eigvalsh(cov).clip(0)[::-1];total=eigen.sum()
    # Eigensolver round-off near the algebraic nullspace is not a signal.
    use=eigen>total*1e-14;p=eigen[use]/eigen[use].sum() if total else np.array([])
    return dict(effective_rank=float(np.exp(-(p*np.log(p)).sum())) if len(p) else 0.,leading_fraction=float(eigen[0]/total) if total else 0.,
        total_variance=float(total),std_median=float(np.median(np.std(x,axis=0,ddof=1))),dimensions=d,images=n),eigen

def within(x,components):
    x=x.copy();components=np.asarray(components)
    for c in np.unique(components):x[components==c]-=x[components==c].mean(0)
    return x

def fixed_mass_covariance(z,mass,components=None):
    z=within(z,components) if components is not None else z-z.mean(0)
    cov=np.einsum('ntd,nud->tu',z,z)/(len(z)-1)
    local=float(np.dot(mass,np.diag(cov)));mean=float(mass@cov@mass);diagonal=float(np.dot(mass*mass,np.diag(cov)))
    pooled=np.einsum('t,ntd->nd',mass,z);residual=z-pooled[:,None]
    spatial=float(np.einsum('t,ntd,ntd->',mass,residual,residual)/(len(z)-1))
    assert np.isclose(local,mean+spatial,atol=1e-9,rtol=1e-10)
    scale=np.sqrt(np.diag(cov).clip(0));den=scale[:,None]*scale[None,:]
    correlation=np.divide(cov,den,out=np.zeros_like(cov),where=den>1e-15)
    return dict(tile_weighted_variance=local,mean_mode_variance=mean,spatial_residual_variance=spatial,
        mean_mode_energy_fraction=mean/local if local else None,diagonal_pooled_variance=diagonal,cross_position_pooled_variance=mean-diagonal,
        cross_position_fraction=(mean-diagonal)/mean if mean else None,mean_off_diagonal_correlation=float(correlation[~np.eye(32,dtype=bool)].mean())),cov,correlation
