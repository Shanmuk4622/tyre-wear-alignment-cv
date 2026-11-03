"""Check raw arithmetic on actual first native shards during the bulk transfer."""
import json,sys
from pathlib import Path
import numpy as np
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tread_two/geo_postrun'))
from inputs import Saved,locate,ground_truth,sigmoid
from pipeline import load_core
from common import read
p=Path(read(ROOT/'tread_two/nb17_completion/LATEST.json')['path']);saved=Saved(p/'remote',read(p/'REMOTE_MANIFEST.json'));core=load_core(saved,ROOT/'tread_two/nb17_report');truth=ground_truth(locate(),core['source'])
obs=core['obs'].set_index('sample');gl=core['geometry'].set_index('sample');raw=np.load(saved.root/'native/geometry_logits/00000.npy')
for i,x0 in enumerate(raw):
    gt=truth[obs.loc[i,'image_id']];x=x0.astype('float64');pr=sigmoid(x);v=gt['valid'][None];target=gt['target'].astype('float64')
    bce=((np.maximum(x,0)-x*target+np.log1p(np.exp(-np.abs(x))))*v).sum();dl=(1-(2*(pr*target*v).sum((1,2))+1)/((pr*v).sum((1,2))+(target*v).sum((1,2))+1)).sum()
    assert np.isclose(bce,gl.loc[i,'bce_sum'],atol=.002,rtol=2e-5);assert np.isclose(dl,gl.loc[i,'dice_loss_sum'],atol=5e-7,rtol=2e-5)
mass=saved.table('patch_support').pivot(index='sample',columns='patch',values='pooling_mass').sort_index(axis=1)
for view in (0,1):
    a=np.load(saved.root/f'native/v{view}_full/00000.npy');pooled=(a.astype('float64')*mass.loc[:63].to_numpy(dtype='float64')[...,None]).sum(1)
    assert np.allclose(pooled,core['global_z'][:64,view],atol=2e-6,rtol=2e-6)
    pred=np.load(saved.root/f'native/v{view}_prediction/00000.npy').astype('float64');target=np.load(saved.root/f'native/v{view}_target/00000.npy').astype('float64');weight=np.load(saved.root/f'native/v{view}_weight/00000.npy').astype('float64')
    diff=np.abs(pred-target);actual=(np.where(diff<.1,diff**2/.2,diff-.05)*weight).sum((-3,-2,-1))
    plan=saved.table('masked_patch_plan');plan=plan[(plan['sample']<64)&(plan.view==view)];metrics=core['patches'].set_index(['sample','view','patch'])
    for row in plan.itertuples():
        slot=int(row.native_prediction_key.rsplit(':',1)[1]);expected=metrics.loc[(row.sample,row.view,row.patch),'huber_sum'];assert np.isclose(actual[row.sample,slot],expected,atol=1e-8,rtol=1e-9)
print('Actual first64 raw geometry, pooled vectors and both native masked views pass.')
