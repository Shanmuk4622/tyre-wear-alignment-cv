"""Descriptor-channel, tile and component reductions of saved counterfactuals."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];DEST=ROOT/'texture_aggregation_20261102';sys.path.insert(0,str(ROOT/'tread_two/trace_batch'))
from common import read,atomic,file_meta
from review import Saved
from cache import Cache
from targets import CHANNELS
from texture_aggregation_math import huber
old=Path(read(ROOT/'tread_two/nb24_completion/LATEST.json')['path']);saved=Saved(old/'remote/fit');cache=Cache(saved.root/'inputs/cache')
alignment=pd.read_csv(DEST/'IMAGE_ALIGNMENT.csv');ids=alignment.observation.to_numpy();slots=alignment.slot.to_numpy();stats=cache.statistics()
mu=stats['mean'].numpy();std=stats['std'].numpy();target=((cache.array('target',slots)-mu[None,None,:,None,None])/std[None,None,:,None,None]).astype('float64')
position=((stats['position_mean'].numpy()-mu[None,:,None,None])/std[None,:,None,None]).astype('float64')
w=cache.array('weight',slots).astype('float64');hidden=saved.get('hidden',ids);mass=saved.get('mass',ids)
channel_summary=[];tile_summary=[]
for layer in ('encoded','masked_context'):
    use=mass>0 if layer=='encoded' else hidden
    weight=w[:,None]*use[...,None,None,None]*(stats['position_count'].numpy()>0)[None,None]
    den=weight.sum((2,4,5));control=(huber(position[None,None],target[:,None])*weight).sum((2,4,5))/den
    c=pd.read_csv(DEST/('CHANNEL_LOSSES_'+layer+'.csv'))
    for keys,g in c.groupby(['role','view','variant','channel']):
        role,view,variant,channel=keys;rows=alignment.role.to_numpy()==role
        control_loss=float(control[rows,int(view),int(channel)].mean())
        reference=c[(c.role==role)&(c.view==view)&(c.variant=='identity')&(c.channel==channel)].loss.mean()
        value=float(g.loss.mean());advantage=control_loss-float(reference)
        channel_summary.append(dict(layer=layer,role=role,view=int(view),variant=variant,channel=int(channel),channel_name=CHANNELS[int(channel)],images=len(g),loss=value,identity_loss=float(reference),position_control_loss=control_loss,loss_delta=value-float(reference),
            original_has_positive_control_advantage=advantage>1e-8,retained_gain_fraction=(control_loss-value)/advantage if advantage>1e-8 else None))
    frames=[]
    for frame in pd.read_csv(DEST/'detail'/('TILE_LOSSES_'+layer+'.csv'),chunksize=131072):
        # A compact per-component sufficient-statistic table; no full-size
        # native prediction/cell data are duplicated into the interactive page.
        frame['available']=frame.loss.notna().astype(int);frame['loss_sum']=frame.loss.fillna(0.)
        frames.append(frame.groupby(['role','component','view','variant','tile'],as_index=False)[['huber_sum','weighted_channel_count','available','loss_sum']].sum())
    compact=pd.concat(frames).groupby(['role','component','view','variant','tile'],as_index=False)[['huber_sum','weighted_channel_count','available','loss_sum']].sum()
    compact['layer']=layer;compact['image_mean_tile_loss']=compact.loss_sum/compact.available.replace(0,np.nan)
    tile_summary.append(compact)
pd.DataFrame(channel_summary).to_csv(DEST/'CHANNEL_UTILITY_SUMMARY.csv',index=False)
tile=pd.concat(tile_summary,ignore_index=True);tile.to_csv(DEST/'TILE_UTILITY_SUMMARY.csv',index=False)
u=pd.read_csv(DEST/'UTILITY_SUMMARY.csv')
for keys,g in pd.DataFrame(channel_summary).groupby(['layer','role','view','variant']):
    r=u[(u.layer==keys[0])&(u.role==keys[1])&(u.view==keys[2])&(u.variant==keys[3])].iloc[0]
    assert np.isclose(g.loss.mean(),r.loss,atol=1e-12)
    assert np.isclose(g.position_control_loss.mean(),r.position_control_loss,atol=1e-12)
cov=pd.read_csv(DEST/'SPATIAL_COVARIANCE.csv');cov0=cov[(cov.view==0)&(cov.layer=='encoded')]
e=pd.read_csv(DEST/'SPATIAL_ENERGY.csv')
energy=e.groupby(['layer','role','view','basis'],as_index=False).agg(images=('image_id','count'),mean_retained_fraction=('retained_fraction','mean'),total_projected_energy=('projected_energy','sum'),total_residual_energy=('residual_energy','sum'))
energy['ratio_of_summed_energies']=energy.total_projected_energy/energy.total_residual_energy
energy.to_csv(DEST/'SPATIAL_ENERGY_SUMMARY.csv',index=False)
atomic(DEST/'SUMMARY_RECEIPT.json',dict(status='passed',source=file_meta(Path(__file__)),files={n:file_meta(DEST/n) for n in ['CHANNEL_UTILITY_SUMMARY.csv','TILE_UTILITY_SUMMARY.csv','SPATIAL_ENERGY_SUMMARY.csv']},channel_means_reconcile_to_image_losses=True,channel_control_means_reconcile=True,model_readouts=0,new_optimizer_updates=0,image_forwards=0))
print(pd.DataFrame(channel_summary).query('layer=="encoded" and role=="validation" and view==0 and variant in ["global_mean","residual_moments5"]').to_string(index=False))
