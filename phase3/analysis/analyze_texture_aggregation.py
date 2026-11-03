"""Trace retained local information through fixed aggregation and frozen decoders.

No image input, encoder execution, optimizer, gradient or network request.
Only existing endpoint features, targets, masks, statistics and head weights.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='4'
import json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];DEST=ROOT/'texture_aggregation_20261102'
sys.path.insert(0,str(ROOT/'tread_two/trace_batch'))
from common import read,atomic,file_meta,sha,digest,native
from review import Saved
from cache import Cache
from texture_aggregation_math import bases,aggregate,project,mlp,huber,spectrum,within,fixed_mass_covariance
torch.set_num_threads(4)
began=time.monotonic();cfg=read(DEST/'PROTOCOL.json');protocol=digest(cfg)
assert read(DEST/'INPUT_VERIFICATION.json')['status']=='passed'
old=Path(read(ROOT/'tread_two/nb24_completion/LATEST.json')['path']);root=old/'remote/fit';saved=Saved(root)
for r in pd.read_csv(DEST/'NATIVE_INPUTS.csv').itertuples():assert file_meta(root/r.file)==dict(bytes=r.bytes,sha256=r.sha256)
alignment=pd.read_csv(DEST/'IMAGE_ALIGNMENT.csv');ids=alignment.observation.to_numpy(dtype='int64');slots=alignment.slot.to_numpy(dtype='int64')
roles=alignment.role.to_numpy();components=alignment.component.to_numpy();train=roles=='train';assert train.sum()==386
cache=Cache(root/'inputs/cache');stats=cache.statistics();weights={k:v.numpy().astype('float64') for k,v in torch.load(root/'exports/final.pt',map_location='cpu',weights_only=False)['trace'].items()}
z={name:saved.get(name,ids).astype('float64') for name in cfg['feature_layers']}
mass=saved.get('mass',ids).astype('float64');hidden=saved.get('hidden',ids).astype(bool);support=mass>0
assert np.allclose(mass.sum(-1),1,atol=2e-6) and all(v.shape==(570,2,32,48) for v in z.values())
b=bases(cfg['random_seed']);raw=cache.array('target',slots);w=cache.array('weight',slots).astype('float64')
mu=stats['mean'].numpy();std=stats['std'].numpy()
target=((raw-mu[None,None,:,None,None])/std[None,None,:,None,None]).astype('float64')
position=((stats['position_mean'].numpy()-mu[None,:,None,None])/std[None,:,None,None]).astype('float64')
available=stats['position_count'].numpy()>0
(DEST/'native').mkdir(exist_ok=True);(DEST/'detail').mkdir(exist_ok=True)
templates={};template_rows=[]
for layer,x in z.items():
    counts=support[train,0].sum(0);template=(x[train,0]*support[train,0,:,None]).sum(0)/counts.clip(1)[:,None]
    templates[layer]=template;native(DEST/'native'/('template_'+layer+'.npy'),template)
    for t in range(32):
        for c in range(48):template_rows.append(dict(layer=layer,tile=t,channel=c,mean=template[t,c],train_supported_images=int(counts[t]),fit_role='train',fit_view=0))
pd.DataFrame(template_rows).to_csv(DEST/'TRAIN_TEMPLATES.csv',index=False)
bridges=[]
def check_bridge(name,actual,reference):
    error=float(np.max(np.abs(actual-reference)));assert np.allclose(actual,reference,atol=cfg['replay_tolerances']['bridge_atol'],rtol=2e-5),(name,error)
    bridges.append(dict(bridge=name,values=actual.size,max_absolute_difference=error,passed=True))
check_bridge('encoded_to_saved_local_pooled',np.einsum('nvt,nvtd->nvd',mass,z['encoded']),saved.get('local_pooled',ids))
check_bridge('full_context_to_saved_context_mean',np.einsum('nvt,nvtd->nvd',mass,z['full_context']),saved.get('context_mean',ids))
moments=np.einsum('nvt,nvtd,tk->nvkd',mass,z['full_context'],b['moments5']).reshape(570,2,240)
check_bridge('full_context_to_saved_spatial240',moments,saved.get('pooled',ids))
check_bridge('spatial240_to_saved_projected6',mlp(moments,weights,'projector'),saved.get('projected',ids))
predictions={key:saved.get(value,ids).astype('float64') for key,value in [('encoded','local_prediction'),('masked_context','masked_prediction')]}
for layer,pred in predictions.items():
    replay=mlp(z[layer],weights,'decoder').reshape(570,2,32,6,4,4)
    error=float(np.max(np.abs(replay-pred)))
    assert np.allclose(replay,pred,atol=cfg['replay_tolerances']['prediction_atol'],rtol=cfg['replay_tolerances']['prediction_rtol']),(layer,error)
    bridges.append(dict(bridge=layer+'_frozen_decoder_to_saved_prediction',values=pred.size,max_absolute_difference=error,passed=True))
pd.DataFrame(bridges).to_csv(DEST/'BRIDGE_VERIFICATION.csv',index=False)
print('All saved aggregation/head bridges and frozen-decoder prediction replay passed.',flush=True)
input_csv=[]
for family in ('image_losses','native_chunks_index'):
    for p in saved.paths(family):
        n=p.relative_to(root).as_posix();m=file_meta(p);assert m==saved.closed[n];input_csv.append(dict(family=family,file=n,**m))
pd.DataFrame(input_csv).to_csv(DEST/'CSV_INPUTS.csv',index=False)
original_losses=saved.frame('image_losses');original_losses=original_losses[original_losses.stage=='endpoint_final'].set_index(['observation','view','objective'])
implementation={p.name:file_meta(p) for p in [Path(__file__),Path(__file__).with_name('texture_aggregation_math.py')]}
atomic(DEST/'IMPLEMENTATION.json',dict(protocol=protocol,sources=implementation,python=sys.version,numpy=np.__version__,torch=torch.__version__,device='CPU',threads=4))

def finish_stage(name,paths,extra=None):
    atomic(DEST/(name+'_RECEIPT.json'),dict(protocol=protocol,sources=implementation,files={p.relative_to(DEST).as_posix():file_meta(p) for p in paths},**(extra or {})))
def completed(name):
    p=DEST/(name+'_RECEIPT.json')
    if not p.exists():return False
    r=read(p);assert r['protocol']==protocol and r['sources']==implementation
    assert all(file_meta(DEST/n)==m for n,m in r['files'].items());print('Verified complete analysis stage; reused',name,flush=True);return True

if not completed('SPACES'):
    spectra=[];eigenrows=[];covrows=[];covcells=[];energy=[];scale_rows=[]
    fixed_mass=mass[train,0].mean(0);fixed_mass/=fixed_mass.sum()
    native(DEST/'native/fixed_train_mass.npy',fixed_mass)
    for layer,x in z.items():
        spaces={'concat1536':(x*support[...,None]).reshape(570,2,-1),
            'sqrt_mass_concat1536':(x*np.sqrt(mass)[...,None]).reshape(570,2,-1),
            'mean48':np.einsum('nvt,nvtd->nvd',mass,x),
            'moments240':np.einsum('nvt,nvtd,tk->nvkd',mass,x,b['moments5']).reshape(570,2,-1)}
        for key in ('band_means','column_means'):
            basis=b[key];den=np.einsum('nvt,tk->nvk',mass,basis)
            spaces[key]=np.divide(np.einsum('nvt,tk,nvtd->nvkd',mass,basis,x),den[...,None],out=np.zeros((570,2,basis.shape[1],48)),where=den[...,None]>0).reshape(570,2,-1)
        for key in ('mean48','moments240'):
            mean=spaces[key][train,0].mean(0);scale=spaces[key][train,0].std(0,ddof=1).clip(.001)
            standardized=(spaces[key]-mean)/scale
            assert np.allclose(standardized*scale+mean,spaces[key],atol=1e-12,rtol=1e-12)
            spaces[key+'_train_scaled']=standardized
            for c in range(len(scale)):scale_rows.append(dict(layer=layer,space=key,coordinate=c,train_mean=mean[c],train_std_floor_1e3=scale[c],fit_role='train',fit_view=0))
        if layer=='full_context':spaces['actual_projected6']=saved.get('projected',ids).astype('float64')
        for key,body in spaces.items():
            native(DEST/'native'/('features_'+layer+'_'+key+'.npy'),body)
            for view in (0,1):
                for role in ('train','validation','test'):
                    use=roles==role;value=body[use,view];plain,_=spectrum(value)
                    for centered in (False,True):
                        actual=within(value,components[use]) if centered else value
                        result,eigen=spectrum(actual);result['between_component_fraction']=1-result['total_variance']/plain['total_variance'] if centered and plain['total_variance'] else None
                        spectra.append(dict(layer=layer,space=key,role=role,view=view,component_centered=centered,**result))
                        eigenrows.extend(dict(layer=layer,space=key,role=role,view=view,component_centered=centered,dimension=i,eigenvalue=float(v)) for i,v in enumerate(eigen))
        for view in (0,1):
            for role in ('train','validation','test'):
                use=roles==role
                for center in (False,True):
                    r,cov,corr=fixed_mass_covariance(x[use,view],fixed_mass,components[use] if center else None)
                    covrows.append(dict(layer=layer,role=role,view=view,component_centered=center,images=int(use.sum()),**r))
                    covcells.extend(dict(layer=layer,role=role,view=view,component_centered=center,tile_i=i,tile_j=j,covariance=cov[i,j],correlation=corr[i,j]) for i in range(32) for j in range(32))
        residual=x-templates[layer][None,None];den=np.einsum('nvt,nvtd,nvtd->nv',mass,residual,residual)
        for name,basis in b.items():
            projected=residual if name=='identity' else project(residual,mass,basis)
            num=np.einsum('nvt,nvtd,nvtd->nv',mass,projected,projected)
            assert (num<=den+1e-8).all()
            for i in range(570):
                for view in (0,1):energy.append(dict(image_id=alignment.image_id.iloc[i],role=roles[i],component=components[i],view=view,layer=layer,basis=name,residual_energy=den[i,view],projected_energy=num[i,view],retained_fraction=num[i,view]/den[i,view] if den[i,view] else None))
        print('Feature spaces, covariance and spatial energies complete:',layer,flush=True)
    tables={'FEATURE_SPACES':spectra,'SPATIAL_COVARIANCE':covrows,'TRAIN_SCALES':scale_rows,'SPATIAL_ENERGY':energy}
    paths=[]
    for name,rows in tables.items():
        p=DEST/(name+'.csv');pd.DataFrame(rows).to_csv(p,index=False);paths.append(p)
    for name,rows in [('SPECTRA',eigenrows),('SPATIAL_COVARIANCE_CELLS',covcells)]:
        p=DEST/'detail'/(name+'.csv');pd.DataFrame(rows).to_csv(p,index=False);paths.append(p)
    paths.extend((DEST/'native').glob('features_*.npy'));paths.append(DEST/'native/fixed_train_mass.npy')
    finish_stage('SPACES',paths)

replay_rows=[]
for layer,branch in [('encoded','local'),('masked_context','masked')]:
    stage='UTILITY_'+layer
    if completed(stage):continue
    use=support if layer=='encoded' else hidden
    full_weight=w[:,None]*use[...,None,None,None];weight=full_weight*available[None,None]
    counts=weight.sum((2,3,4,5))*6;full_counts=full_weight.sum((2,3,4,5))*6
    tile_counts=weight.sum((3,4,5))*6;channel_counts=weight.sum((2,4,5))
    assert (counts>0).all()
    position_loss=(huber(position[None,None],target[:,None])*weight).sum((2,3,4,5))/counts
    native_loss=(huber(predictions[layer],target[:,None])*weight).sum((2,3,4,5))/counts
    for i,row in enumerate(alignment.itertuples()):
        for view in (0,1):
            r=original_losses.loc[(row.observation,view,branch)]
            assert np.isclose(native_loss[i,view],float(r.image_mean),atol=cfg['replay_tolerances']['loss_atol'],rtol=2e-5)
    image_rows=[];channel_rows=[];native_paths=[]
    tilepath=DEST/'detail'/('TILE_LOSSES_'+layer+'.csv');first=True
    for variant in cfg['variants']:
        altered=aggregate(z[layer],mass,variant,templates[layer],b)
        prediction=mlp(altered,weights,'decoder').reshape(570,2,32,6,4,4)
        path=DEST/'native'/('prediction_'+layer+'_'+variant+'.npy');native(path,prediction);native_paths.append(path)
        weighted=huber(prediction,target[:,None])*weight;tile_num=weighted.sum((3,4,5));image_num=tile_num.sum(2);value=image_num/counts
        channel_num=weighted.sum((2,4,5));channel_value=channel_num/channel_counts
        err=np.einsum('nvt,nvtd,nvtd->nv',mass,altered-z[layer],altered-z[layer]);norm=np.einsum('nvt,nvtd,nvtd->nv',mass,z[layer],z[layer])
        detail=[]
        for i,row in enumerate(alignment.itertuples()):
            for view in (0,1):
                fields=dict(image_id=row.image_id,role=row.role,component=row.component,view=view,layer=layer,branch=branch,variant=variant,epoch=60,parameter_update=2940)
                image_rows.append(dict(**fields,huber_sum=image_num[i,view],weighted_channel_count=counts[i,view],coverage=counts[i,view]/full_counts[i,view],loss=value[i,view],native_original_loss=native_loss[i,view],position_control_loss=position_loss[i,view],loss_delta=value[i,view]-native_loss[i,view],
                    gain_vs_position_control=1-value[i,view]/position_loss[i,view],retained_gain_fraction=(position_loss[i,view]-value[i,view])/(position_loss[i,view]-native_loss[i,view]),feature_error_squared=err[i,view],feature_relative_error=err[i,view]/norm[i,view]))
                for channel in range(6):channel_rows.append(dict(**fields,channel=channel,huber_sum=channel_num[i,view,channel],weighted_count=channel_counts[i,view,0],loss=channel_value[i,view,channel]))
                for tile in range(32):detail.append(dict(**fields,tile=tile,hidden=bool(hidden[i,view,tile]),supported=bool(support[i,view,tile]),huber_sum=tile_num[i,view,tile],weighted_channel_count=tile_counts[i,view,tile],loss=tile_num[i,view,tile]/tile_counts[i,view,tile] if tile_counts[i,view,tile] else None))
        pd.DataFrame(detail).to_csv(tilepath,index=False,mode='w' if first else 'a',header=first);first=False
        print('Frozen decoder counterfactual complete:',layer,variant,flush=True)
    paths=[tilepath,*native_paths]
    for name,rows in [('IMAGE_LOSSES',image_rows),('CHANNEL_LOSSES',channel_rows)]:
        p=DEST/(name+'_'+layer+'.csv');pd.DataFrame(rows).to_csv(p,index=False);paths.append(p)
    finish_stage(stage,paths,dict(frozen_decoder_token_readouts=570*2*32*len(cfg['variants']),image_encoder_forwards=0,optimizer_updates=0))

summary=[];components_out=[]
for layer in ('encoded','masked_context'):
    frame=pd.read_csv(DEST/('IMAGE_LOSSES_'+layer+'.csv'))
    for keys,g in frame.groupby(['role','view','variant']):
        c=float(g.position_control_loss.mean());original=float(g.native_original_loss.mean());loss=float(g.loss.mean())
        summary.append(dict(layer=layer,role=keys[0],view=int(keys[1]),variant=keys[2],images=len(g),loss=loss,native_original_loss=original,position_control_loss=c,loss_delta=loss-original,gain_vs_position_control=1-loss/c,retained_gain_fraction=(c-loss)/(c-original),image_wins_vs_original=int((g.loss<g.native_original_loss).sum()),mean_feature_relative_error=float(g.feature_relative_error.mean()),minimum_coverage=float(g.coverage.min())))
    for keys,g in frame.groupby(['role','component','view','variant']):
        c=float(g.position_control_loss.mean());original=float(g.native_original_loss.mean());loss=float(g.loss.mean())
        components_out.append(dict(layer=layer,role=keys[0],component=keys[1],view=int(keys[2]),variant=keys[3],images=len(g),loss=loss,native_original_loss=original,position_control_loss=c,retained_gain_fraction=(c-loss)/(c-original)))
pd.DataFrame(summary).to_csv(DEST/'UTILITY_SUMMARY.csv',index=False);pd.DataFrame(components_out).to_csv(DEST/'COMPONENT_UTILITY.csv',index=False)
output_files={p.relative_to(DEST).as_posix():file_meta(p) for p in DEST.rglob('*') if p.is_file() and p.suffix in ('.csv','.npy')}
atomic(DEST/'ARTIFACTS.json',dict(protocol=protocol,files=output_files))
result=dict(status='passed',protocol=protocol,images=570,views=2,feature_layers=3,variants=9,all_saved_bridges_pass=True,
    primary_view=0,train_only_statistic_images=386,template_neural_training=False,image_encoder_forwards=0,geometry_forwards=0,new_optimizer_updates=0,new_photos=0,
    frozen_decoder_token_readouts=570*2*32*9*2,frozen_projector_vector_readouts=570*2,original_selected=None,original_gate_unchanged=True,
    direct_derivative_files=len(output_files),direct_derivative_bytes=sum(m['bytes'] for m in output_files.values()),elapsed_seconds=time.monotonic()-began,
    local_credentials_read=False,network_calls=0,remote_writes=0,new_analysis_HF_published=False)
atomic(DEST/'ANALYSIS.json',result);print(json.dumps(result,indent=2))
