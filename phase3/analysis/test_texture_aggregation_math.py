"""Meaningful numerical identities for the saved-feature investigation."""
import sys
from pathlib import Path
import numpy as np
assert Path(sys.prefix).name=='cv_conda'
from texture_aggregation_math import bases,project,aggregate,spectrum,fixed_mass_covariance
b=bases();rng=np.random.default_rng(7);z=rng.normal(size=(9,2,32,48));m=rng.uniform(.1,1,size=(9,2,32));m/=m.sum(-1,keepdims=True)
template=z[:5,0].mean(0)
assert np.allclose(project(z,m,b['identity']),z,atol=1e-12)
mean=np.einsum('nvt,nvtd->nvd',m,z)
assert np.allclose(project(z,m,b['global_mean']),mean[:,:,None],atol=1e-12)
for name in ('band_means','column_means','moments5','random5'):
    p=project(z,m,b[name]);assert np.allclose(project(p,m,b[name]),p,atol=1e-12)
    assert np.allclose(np.einsum('nvt,tk,nvtd->nvkd',m,b[name],z-p),0,atol=1e-12)
assert np.allclose(aggregate(z,m,'residual_mean',template,b)-template[None,None],project(z-template[None,None],m,b['global_mean']),atol=1e-12)
fixed_mass_covariance(z[:,0],m[:,0].mean(0));fixed_mass_covariance(z[:,0],m[:,0].mean(0),['a']*4+['b']*5)
raw=rng.normal(size=(80,6));scaled=raw*np.array([100.,1,1,1,1,1]);a,_=spectrum(raw);c,_=spectrum(scaled)
assert a['effective_rank']>c['effective_rank']*2 and np.allclose(scaled/np.array([100.,1,1,1,1,1]),raw)
print('Weighted projections, mean identity, template residual, covariance decomposition and invertible-scaling control passed.')
