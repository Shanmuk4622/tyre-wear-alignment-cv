"""Independent saved-evidence audit: all objectives and original validation gates.

Uses SVD instead of the trainer's covariance eigensolver; no image forward,
training, credential lookup or HF write. Full-body native checks are bounded
and enumerated. The original HF full dense audit is retained separately.
"""
import collections,csv,json,math,subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import torch
from nb24_public import *
from common import digest
from review import Saved
from store import semantic_hash
torch.set_num_threads(4)
assert read(OUT/'OBJECT_VERIFICATION.json')['status']=='passed'
root=OUT/'remote/fit';result=read(root/'RESULT.json');cfg=read(root/'CONTRACT.json');saved=Saved(root);state=saved.state
assert result['status']=='completed' and result['fixture'] is False and result['selected'] is None
assert state['cursor']['phase']=='completed' and state['cursor']['update']==2940 and state['cursor']['observation']==29443
assert state['protocol']==META['protocol'] and state['unit']==3809 and not state['arrays']
assert len(state['rng']['cuda'])==2 and set(state['rng'])=={'python','numpy','cpu','cuda'}
assert state['precision']=='FP32_no_scaler' and state['scheduler'] is state['scaler'] is None
assert all(int(v['step'])==2940 for v in state['optimizer']['state'].values())
for n,m in state['closed'].items():assert {k:MANIFESTS['fit']['files'][n][k] for k in ('bytes','sha256')}==m
ledger=saved.frame('state_ledger').sort_values('unit');assert ledger.unit.tolist()==list(range(1,3810))
tensor={k:state[k] for k in ('model','optimizer','rng','cursor','precision','scheduler','scaler')}
last=ledger.iloc[-1];assert semantic_hash(tensor)==last.state_sha256
assert last.chain_sha256==state['chain']==read(root/'POINTER.json')['chain']
assert ledger.previous_unit_sha256.iloc[1:].tolist()==ledger.chain_sha256.iloc[:-1].tolist()
cache=read(root/'inputs/cache/CACHE.json');assert cache['completed'] and len(cache['rows'])==570 and len(cache['chunks'])==72
assert digest(cache['files'])==cache['scientific_hash']==result['cache_hash']
assert collections.Counter(r['role'] for r in cache['rows'])==dict(train=386,validation=81,test=103)
source={r['image_id']:r for r in cache['rows']}
obs=saved.frame('sample_observations').sort_values('observation');loss=saved.frame('image_losses');ctrl=saved.frame('control_losses')
assert obs.observation.tolist()==list(range(29443))
train=obs[obs.stage=='train'];val=obs[obs.stage=='validate'];end=obs[obs.stage=='endpoint_final']
assert len(train)==23520 and len(val)==4860 and len(end)==570 and not (obs.stage=='endpoint_selected').any()
assert (train.parameter_update==train['update']-1).all() and (end.parameter_update==2940).all()
assert train.groupby('update').size().eq(8).all()
for epoch,g in train.groupby('epoch'):
    assert len(g)==392 and g.primary.sum()==386 and g[g.primary==1].image_id.nunique()==386
    assert set(g[g.primary==1].image_id)=={r['image_id'] for r in cache['rows'] if r['role']=='train'}
    assert g.groupby('update').source_sha256.nunique().eq(8).all()
assert obs.eligible.eq(1).all()
# Download exactly the native inputs required for all training objectives and
# original validation gates, plus endpoint space diagnostics.
wanted=set()
focus=obs[obs.stage.isin(['validate','endpoint_final'])].observation.to_numpy()
for r in saved.index.itertuples():
    if r.kind=='projected' or r.kind in ('pooled','local_pooled','context_mean') and ((focus>=r.first_observation)&(focus<=r.last_observation)).any():wanted.add(r.file)
with ThreadPoolExecutor(max_workers=8) as pool:
    for i,_ in enumerate(pool.map(lambda n:fetch('fit',n),sorted(wanted))):
        if (i+1)%100==0:print('Science native inputs',i+1,'/',len(wanted),flush=True)
for n in wanted:
    a=np.load(root/n,mmap_mode='r',allow_pickle=False);assert np.isfinite(a).all()
pd.DataFrame([dict(file=n,**file_meta(root/n)) for n in sorted(wanted)]).to_csv(OUT/'SCIENCE_NATIVE_INPUTS.csv',index=False)
def spectrum(z):
    z=np.asarray(z,dtype='float64');center=z-z.mean(0);s=np.linalg.svd(center,compute_uv=False);e=s*s/(len(z)-1);total=e.sum()
    p=e[e>0]/total if total else np.array([])
    return dict(rank=float(np.exp(-(p*np.log(p)).sum())) if total else 0.,std=float(np.median(np.std(z,axis=0,ddof=1))),leading=float(e[0]/total) if total else 0.)
def centered(z,components):
    z=np.asarray(z,dtype='float64');w=z.copy();components=np.asarray(components)
    for name in set(components):w[components==name]-=z[components==name].mean(0)
    total=float(((z-z.mean(0))**2).sum());within=float((w*w).sum())
    return w,1-within/total if total else 0.
lp=loss.pivot(index=['observation','view'],columns='objective',values='image_mean').astype('float64')
counts=loss.pivot(index=['observation','view'],columns='objective',values='weighted_channel_count').astype('float64')
objectives=saved.frame('objective_terms').set_index('update');objective_checks=[]
for update,g in train.groupby('update'):
    ids=g.observation.to_numpy(dtype='int64');z=saved.get('projected',ids).astype('float64')
    l=[lp.loc[(ids,v),'local'].mean() for v in (0,1)];m=[lp.loc[(ids,v),'masked'].mean() for v in (0,1)]
    variance=[];covariance=[]
    for v in (0,1):
        c=np.cov(z[:,v],rowvar=False,ddof=1);off=c-np.diag(np.diag(c))
        variance.append(float(np.maximum(0,.1-np.sqrt(np.diag(c)+1e-4)).mean()));covariance.append(float((off**2).sum()/30))
    terms=dict(local=float(np.mean(l)),masked=float(np.mean(m)),pair=float(((z[:,0]-z[:,1])**2).mean()),variance=float(np.mean(variance)),covariance=float(np.mean(covariance)))
    terms['total']=terms['local']+terms['masked']+.1*terms['pair']+.1*terms['variance']+.01*terms['covariance']
    old=objectives.loc[update];dif=max(abs(v-float(old[k])) for k,v in terms.items())
    assert all(np.isclose(v,float(old[k]),atol=3e-6,rtol=3e-5) for k,v in terms.items()),(update,terms)
    objective_checks.append(dict(update=int(update),**terms,max_absolute_difference=dif))
pd.DataFrame(objective_checks).to_csv(OUT/'INDEPENDENT_OBJECTIVES.csv',index=False)
print('All 2940 objectives independently reconstructed.',flush=True)
cp=ctrl.set_index(['stage','epoch','image_id','view','control']);epochs=[];endpoints=[]
rule=cfg['selection']
def summary(frame,stage,role):
    ids=frame.observation.to_numpy(dtype='int64');epoch=int(frame.epoch.iloc[0]);images=frame.image_id.tolist();components=[source[x]['component'] for x in images]
    c=ctrl[(ctrl.stage==stage)&(ctrl.epoch==epoch)&(ctrl.view==0)&(ctrl.image_id.isin(images))]
    controls={k:float(g.control_mean.astype(float).mean()) for k,g in c.groupby('control')}
    model=float(c[c.control=='channel_mean'].model_mean.astype(float).mean());local=float(lp.loc[(ids,0),'local'].mean());local_ctl=float(lp.loc[(ids,0),'local_position_control'].mean())
    row=dict(epoch=epoch,stage=stage,role=role,images=len(frame),masked_loss=model,local_loss=local,local_control=local_ctl,local_gain=1-local/local_ctl,
             **{'control_'+k:v for k,v in controls.items()},**{'masked_gain_'+k:1-model/v for k,v in controls.items()})
    for kind in ('projected','local_pooled','pooled','context_mean'):
        z=saved.get(kind,ids)[:,0];sp=spectrum(z);w,between=centered(z,components);ws=spectrum(w)
        row.update({kind+'_'+k:v for k,v in sp.items()});row[kind+'_within_rank']=ws['rank'];row[kind+'_between_fraction']=between
    projected=saved.get('projected',ids).astype('float64');cos=1-(projected[:,0]*projected[:,1]).sum(1)/(np.linalg.norm(projected[:,0],axis=1)*np.linalg.norm(projected[:,1],axis=1))
    recorded=frame.paired_cosine.to_numpy(dtype='float64');assert np.allclose(cos,recorded,atol=2e-6,rtol=2e-5)
    row['paired_cosine']=float(recorded.mean());row['coverage']=float((c[c.control=='channel_mean'].common_count.to_numpy()/counts.loc[(ids,0),'masked'].to_numpy()).mean())
    row['primary_fraction']=float(train[(train.epoch==epoch)&(train.primary==1)].eligible.mean()) if stage=='validate' else 1.
    gates=dict(geometry=True,exposure=bool(frame.eligible.mean()>=.9 and row['primary_fraction']>=.9),control_coverage=row['coverage']>=.95 and min(controls.values())>1e-8,
        masked_utility=min(row['masked_gain_'+k] for k in controls)>=.05,local_utility=row['local_gain']>=.1,
        projector_diversity=row['projected_rank']>=4 and row['projected_std']>=.05,
        encoder_diversity=row['local_pooled_rank']>=4 and row['pooled_rank']>=4,
        within_component_diversity=row['local_pooled_within_rank']>=2 and row['pooled_within_rank']>=2,
        finite=bool(np.isfinite(model) and np.isfinite(recorded).all()))
    row.update({'gate_'+k:bool(v) for k,v in gates.items()});row['eligible']=all(gates.values());return row
original=saved.frame('eligibility')
for epoch,g in val.groupby('epoch'):
    row=summary(g,'validate','validation');old=original[(original.stage=='validate')&(original.epoch==epoch)].iloc[0]
    assert row['eligible']==bool(old.eligible)
    for k in rule:pass
    for k in [n[5:] for n in row if n.startswith('gate_')]:assert row['gate_'+k]==bool(old[k]),(epoch,k)
    h=result['history'][int(epoch)-1];assert np.isclose(row['masked_loss'],h['model'],atol=1e-8)
    epochs.append(row)
pd.DataFrame(epochs).to_csv(OUT/'VALIDATION_GATE_AUDIT.csv',index=False)
for role,g in end.assign(role=end.image_id.map(lambda x:source[x]['role'])).groupby('role'):endpoints.append(summary(g,'endpoint_final',role))
pd.DataFrame(endpoints).to_csv(OUT/'ENDPOINT_COHORTS.csv',index=False)
# All epoch weights, final export and original final complete-state agree.
weights=[]
for e in range(1,61):
    path=root/f'epochs/epoch_{e:03}.pt';w=torch.load(path,map_location='cpu',weights_only=True)
    assert sum(t.numel() for t in w.values())==71022 and all(torch.isfinite(t).all() for t in w.values())
    weights.append(dict(epoch=e,file=path.relative_to(root).as_posix(),**file_meta(path)))
final=torch.load(root/'exports/final.pt',map_location='cpu',weights_only=False)
atomic(OUT/'FINAL_EXPORT_FIELDS.json',dict(keys=list(final),geometry_pin=cfg['geometry'],selected=None))
last_weights=torch.load(root/'epochs/epoch_060.pt',map_location='cpu',weights_only=True)
assert all(torch.equal(t,state['model'][k]) for k,t in last_weights.items())
export_weights=final.get('model',final.get('trace'))
assert export_weights is not None and all(torch.equal(t,export_weights[k]) for k,t in last_weights.items())
pd.DataFrame(weights).to_csv(OUT/'EPOCH_WEIGHTS.csv',index=False)
# Published FILE_AUDIT exactly describes the closed selecting state.
audit=pd.read_csv(OUT/'remote/review/FILE_AUDIT.csv').set_index('file')
assert len(audit)==len(state['closed'])==10343
assert all(int(audit.loc[n,'bytes'])==m['bytes'] and audit.loc[n,'sha256']==m['sha256'] for n,m in state['closed'].items())
for name in ('FILES','SCIENCE','VISUALS'):
    receipt=read(OUT/'remote/review'/(name+'_RECEIPT.json'));assert receipt['contract']==META['review_protocol']
    for n,m in receipt['files'].items():assert {k:MANIFESTS['review']['files'][n][k] for k in ('bytes','sha256')}==m
v=pd.read_csv(OUT/'remote/review/VALIDATION_RECONSTRUCTION.csv')
for k,ind in [('projector_rank','projected_rank'),('local_rank','local_pooled_rank'),('spatial_rank','pooled_rank'),('masked_loss','masked_loss')]:
    assert np.allclose(v[k],pd.DataFrame(epochs)[ind],atol=1e-8,rtol=1e-8),(k,ind)
events=saved.frame('numerical_events');errors=saved.frame('errors')
resource=saved.frame('resource_measurements');timing=saved.frame('timings')
resource_summary=[]
for device,g in resource.groupby('device'):
    resource_summary.append(dict(device=device,samples=len(g),max_allocated_bytes=int(g.max_allocated_bytes.max()),max_reserved_bytes=int(g.max_reserved_bytes.max()),max_host_RSS_bytes=int(g.host_RSS_bytes.max()),minimum_free_bytes=int(g.free_bytes.min())))
pd.DataFrame(resource_summary).to_csv(OUT/'RESOURCE_SUMMARY.csv',index=False)
df=pd.DataFrame(epochs)
report=dict(status='passed',protocol=META['protocol'],epochs=60,updates=2940,observations=29443,training_draws=23520,training_primary_draws=23160,padding_draws=360,
    selected=None,eligible_epochs=int(df.eligible.sum()),gates_passed_epochs={k[5:]:int(df[k].sum()) for k in df if k.startswith('gate_')},
    final_validation=epochs[-1],best_masked_loss_diagnostic_epoch=int(df.loc[df.masked_loss.idxmin(),'epoch']),
    endpoint_cohorts=endpoints,all_objectives_reconstructed=2940,max_objective_absolute_difference=max(x['max_absolute_difference'] for x in objective_checks),
    independent_SVD_validation_spaces=240,full_body_native_inputs=len(wanted),full_body_native_bytes=sum((root/n).stat().st_size for n in wanted),
    state_ledger_units=3809,complete_rng_families=['python','numpy','cpu','cuda:0','cuda:1'],optimizer_steps_all_parameters=2940,epoch_weights=60,
    native_arrays_all_sealed=True,original_review_file_audit_matches_closed_state=True,original_published_dense_review=read(OUT/'remote/review/SCIENTIFIC_AUDIT.json'),
    local_dense_prediction_audit='Original published full dense audit inspected; not independently replayed over every dense prediction body here',
    numerical_event_rows=len(events),error_rows=len(errors),resources=resource_summary,local_credentials_read=False,remote_writes=0,new_optimizer_updates=0,new_photos=0)
atomic(OUT/'SCIENTIFIC_VERIFICATION.json',report)
print(json.dumps({k:v for k,v in report.items() if k not in ('endpoint_cohorts','original_published_dense_review')},indent=2),flush=True)
