"""Independent saved-data accounting, selection, geometry/error and representation review."""
import json,math
import numpy as np
import pandas as pd
from common import atomic,read
from inputs import PIN,latent
from tables import Tables

def run(saved,out):
    tables=Tables(out);result=read(saved.root/'RESULT.json');g0=read(saved.root/'controls/G0_validation.json');median=read(saved.root/'controls/quality_median.json')
    obs=saved.table('sample_observations').sort_values('sample').reset_index(drop=True)
    assert len(obs)==PIN['expected_samples'] and obs['sample'].to_numpy(dtype='int64').tolist()==list(range(len(obs)))
    source=saved.table('source_images');assert len(source)==570 and source.image_id.nunique()==570
    assert source.groupby('role').size().to_dict()=={'test':103,'train':386,'validation':81}
    expected={'mean':386,'baseline':81,'train':23520,'validate':4860,'endpoint':570,'intervention':1030}
    assert obs.groupby('stage').size().to_dict()==expected
    tables.write('observation_coverage',obs.groupby(['stage','endpoint','epoch'],dropna=False).agg(observations=('sample','size'),unique_images=('image_id','nunique'),SSL_valid=('SSL_valid','sum')).reset_index())
    tables.write('source_registry',source)
    regions=saved.table('region_metrics');points=saved.table('point_values');quality=saved.table('quality_predictions')
    geometry=saved.table('geometry_losses');patches=saved.table('masked_patch_metrics');pairs=saved.table('pair_losses')
    projected=saved.vector('projected_embeddings',obs,64);global_z=saved.vector('global_embeddings',obs,96)
    np.save(tables.root/'projected.npy',projected,allow_pickle=False);np.save(tables.root/'global.npy',global_z,allow_pickle=False)
    # All observed targets retain original provenance and exact availability.
    point=points.assign(delta=(points.prediction-points.target).abs())
    valid=point['target'].notna()
    assert np.allclose(point.loc[valid,'delta'],point.loc[valid,'absolute_error_width_fraction'],atol=1e-7,rtol=1e-6)
    q=quality.merge(obs[['sample','stage','epoch','endpoint','image_id']],on='sample',validate='many_to_one')
    assert not ((q.quantity=='human_point_error')&(q.target_available==1)&~q['sample'].isin(points.query('human_metric_available==1')['sample'].unique())).any()
    q['error']=(q.predicted_error-q.observed_error).abs();q['squared_error']=(q.predicted_error-q.observed_error)**2
    med=dict(zip(['tyre_error','tread_error','human_point_error'],median['median']))
    q['baseline_error']=(q.quantity.map(med)-q.observed_error).abs();q['baseline_squared']=q['baseline_error']**2
    qvalid=q.query('target_available==1')
    quality_summary=qvalid.groupby(['stage','endpoint','epoch','quantity'],dropna=False).agg(count=('sample','size'),MAE=('error','mean'),MSE=('squared_error','mean'),median_MAE=('baseline_error','mean'),median_MSE=('baseline_squared','mean')).reset_index()
    quality_summary['RMSE']=np.sqrt(quality_summary.MSE);quality_summary['median_RMSE']=np.sqrt(quality_summary.median_MSE)
    tables.write('quality_comparison',quality_summary)
    final_q=qvalid[qvalid.stage=='endpoint'].copy();final_q['bin']=np.minimum(9,(final_q.predicted_error*10).astype('int64'))
    tables.write('quality_calibration',final_q.groupby(['quantity','bin']).agg(count=('sample','size'),predicted=('predicted_error','mean'),observed=('observed_error','mean'),MAE=('error','mean'),median_MAE=('baseline_error','mean')).reset_index())
    point_human=point[point.human_metric_available==1].groupby('sample').delta.mean()
    for name,region in [('tyre_error',0),('tread_error',1)]:
        expected_error=1-regions[regions.region==region].set_index('sample').dice
        values=quality[(quality.quantity==name)&(quality.target_available==1)]
        assert np.allclose(values.observed_error,values['sample'].map(expected_error),atol=1e-7,rtol=1e-6)
    values=quality[(quality.quantity=='human_point_error')&(quality.target_available==1)]
    assert np.allclose(values.observed_error,values['sample'].map(point_human),atol=1e-7,rtol=1e-6)
    region_mean=regions.groupby('sample').dice.mean()
    masked=patches.groupby('sample')[['huber_sum','huber_weight_count','mean_predictor_huber_sum','mean_predictor_weight_count']].sum()
    case=obs.merge(region_mean.rename('mean_dice'),on='sample',validate='one_to_one').merge(point_human.rename('human_point_MAE'),on='sample',how='left',validate='one_to_one')
    case=case.merge(masked,on='sample',how='left',validate='one_to_one').merge(pairs[['sample','pair_cosine_distance','invariance']],on='sample',validate='one_to_one')
    case['masked_huber']=case.huber_sum/case.huber_weight_count.replace(0,np.nan)
    case['mean_huber']=case.mean_predictor_huber_sum/case.mean_predictor_weight_count.replace(0,np.nan)
    case['masked_ratio']=case.masked_huber/case.mean_huber.replace(0,np.nan)
    case=case.merge(source[['image_id','leakage_component','domain','label_proxy']],on='image_id',validate='many_to_one')
    tables.write('case_metrics',case)
    # Every validation criterion is independently reconstructed on the declared role.
    epoch_rows=[];spectra=[];latent_rows=[]
    for epoch,frame in case[case.stage=='validate'].groupby('epoch'):
        ids=frame.loc[frame.SSL_valid==1,'sample'].to_numpy(dtype='int64');p=projected[ids,0];z=global_z[ids,0]
        lp=latent(p);lz=latent(z)
        for name,values in [('projected64',p),('global96',z)]:
            cov=np.cov(values.astype('float64'),rowvar=False,ddof=1);ev=np.linalg.eigvalsh(cov).clip(0)[::-1];total=ev.sum()
            spectra.extend(dict(epoch=int(epoch),space=name,dimension=i,eigenvalue=float(value),fraction=float(value/total) if total else 0.) for i,value in enumerate(ev))
        mask_loss=frame.loc[frame.SSL_valid==1,'huber_sum'].sum()/frame.loc[frame.SSL_valid==1,'huber_weight_count'].sum()
        mean_loss=frame.loc[frame.SSL_valid==1,'mean_predictor_huber_sum'].sum()/frame.loc[frame.SSL_valid==1,'mean_predictor_weight_count'].sum()
        d=float(frame.mean_dice.mean());pmae=float(frame.human_point_MAE.mean());invalid=float((frame.geometry_valid==0).mean());cosine=float(frame.loc[frame.SSL_valid==1,'pair_cosine_distance'].mean())
        geom=d>=g0['mean_dice']-.002 and pmae<=g0['human_point_MAE']+.001 and invalid<=g0['invalid_atlas_fraction']+.01
        collapse=lp['std_median']>=.1 and lp['effective_rank']>=4
        eligible=geom and collapse and mean_loss>=1e-6
        objective=float(mask_loss/mean_loss+.1*cosine) if eligible else None
        epoch_rows.append(dict(epoch=int(epoch),mean_dice=d,human_point_MAE=pmae,invalid_atlas_fraction=invalid,masked_huber=float(mask_loss),mean_huber=float(mean_loss),masked_gain=1-float(mask_loss/mean_loss),pair_cosine=cosine,
                               projector_std_median=lp['std_median'],projector_rank=lp['effective_rank'],projector_leading_fraction=lp['leading_fraction'],global_rank=lz['effective_rank'],global_leading_fraction=lz['leading_fraction'],
                               geometry_eligible=geom,collapse_eligible=collapse,eligible=eligible,objective=objective,rejection='' if eligible else ';'.join(([] if geom else ['geometry_floor'])+([] if collapse else ['collapse']))))
    epochs=tables.write('validation_reconstruction',epoch_rows);tables.write('validation_spectra',spectra)
    selection=saved.table('selection').sort_values('epoch');assert len(selection)==len(epochs)==60
    assert np.array_equal(selection.geometry_eligible.to_numpy(dtype=bool),epochs.geometry_eligible.to_numpy())
    assert np.array_equal(selection.collapse_eligible.to_numpy(dtype=bool),epochs.collapse_eligible.to_numpy())
    assert not epochs.eligible.any() and result['selected'] is None
    assert np.allclose(selection.effective_rank,epochs.projector_rank,atol=1e-8,rtol=1e-8)
    assert np.allclose(selection.projector_std_median,epochs.projector_std_median,atol=1e-8,rtol=1e-8)
    tables.write('selection_source_comparison',selection)
    for stage,endpoint,frame in [(a,b,c) for (a,b),c in case.groupby(['stage','endpoint'],dropna=False) if a in ('baseline','endpoint')]:
        for role,group in [('all',frame)]+[(r,g) for r,g in frame.merge(source[['image_id','role']],on='image_id').groupby('role')]:
            ids=group.loc[group.SSL_valid==1,'sample'].to_numpy(dtype='int64')
            for name,values in [('projected64',projected[ids,0]),('global96',global_z[ids,0])]:latent_rows.append(dict(stage=stage,endpoint=endpoint,role=role,space=name,**latent(values)))
    tables.write('endpoint_latent_spaces',latent_rows)
    endpoint=case[case.stage=='endpoint'].merge(source[['image_id','role']],on='image_id')
    cohort=[]
    for role,group in [('all',endpoint)]+[(r,g) for r,g in endpoint.groupby('role')]+[('component:'+r,g) for r,g in endpoint.groupby('leakage_component')]:
        valid=group.SSL_valid==1;num=group.loc[valid,'huber_sum'].sum();den=group.loc[valid,'huber_weight_count'].sum();mn=group.loc[valid,'mean_predictor_huber_sum'].sum()
        cohort.append(dict(group=role,images=len(group),human_images=int(group.human_point_MAE.notna().sum()),mean_dice=group.mean_dice.mean(),human_point_MAE=group.human_point_MAE.mean(),masked_huber=num/den if den else None,mean_huber=mn/den if den else None,masked_gain=1-num/mn if mn else None,pair_cosine=group.loc[valid,'pair_cosine_distance'].mean()))
    tables.write('endpoint_cohorts',cohort)
    # Schedule/source exposure are validated against actual observed training rows.
    schedule=saved.table('epoch_schedule');exposure=saved.table('source_exposure');train=case[case.stage=='train']
    assert len(schedule)==23520 and len(train)==23520 and len(exposure)==386*60
    for epoch,frame in schedule.groupby('epoch'):
        assert len(frame)==392 and frame.query('is_padding==0').image_id.nunique()==386 and int(frame.is_padding.sum())==6
        for _,batch in frame.groupby('batch'):assert batch.image_id.nunique()==8 and batch.leakage_component.nunique()>=2
    assert int(exposure.geometry_draws.sum())==23520 and int(exposure.padding_draws.sum())==360 and int(exposure.SSL_valid_draws.sum())==23520
    actual=train.groupby(['epoch','image_id']).size().sort_index();recorded=exposure.set_index(['epoch','image_id']).geometry_draws.sort_index();assert actual.equals(recorded.astype('int64'))
    tables.write('training_exposure_audit',exposure)
    tables.write('training_task_gradients',saved.table('group_gradient_summary'))
    gradients=saved.table('parameter_gradients');assert len(gradients)==2940*67 and gradients.finite.dropna().eq(1).all()
    tables.write('tensor_gradient_summary',gradients.groupby(['group','parameter']).agg(updates=('update','size'),norm_mean=('l2_norm','mean'),norm_max=('l2_norm','max'),max_absolute=('max_absolute','max'),zero_fraction_mean=('zero_fraction','mean')).reset_index())
    resources=saved.table('resource_samples');resource_rows=[]
    for name in resources.columns:
        if name in ('run','unit'):continue
        values=resources[name].dropna()
        resource_rows.append(dict(quantity=name,recorded_samples=len(values),missing_samples=int(resources[name].isna().sum()),minimum=float(values.min()) if len(values) else None,maximum_sampled=float(values.max()) if len(values) else None,last_recorded=float(values.iloc[-1]) if len(values) else None,scope='Sampled at completed observation units; not a true within-step peak'))
    tables.write('resource_summary',resource_rows)
    timing=saved.table('pipeline_timings');timing_rows=[]
    for stage,group in timing.groupby('stage'):
        for name in [c for c in group if c.endswith('_seconds')]:
            values=group[name].dropna();timing_rows.append(dict(stage=stage,quantity=name,recorded_samples=len(values),missing_samples=int(group[name].isna().sum()),total_component_seconds=float(values.sum()),median_seconds=float(values.median()) if len(values) else None,p95_seconds=float(values.quantile(.95)) if len(values) else None,maximum_seconds=float(values.max()) if len(values) else None))
    tables.write('timing_summary',timing_rows)
    interventions=saved.table('interventions');assert len(interventions)==1030
    assert interventions.probe_available.eq(1).all()
    for value in interventions.source_patch_mapping:assert sorted(json.loads(value))==list(range(32))
    tables.write('intervention_summary',interventions.groupby('variant').agg(images=('sample','size'),mean_dice_change=('mask_dice_change','mean'),point_MAE_change=('point_MAE_change','mean'),embedding_cosine_change=('embedding_cosine_change','mean'),quality_error_change=('quality_error_change','mean'),valid_fraction_change=('valid_fraction_change','mean')).reset_index())
    tables.write('intervention_cases',interventions)
    neighbours=saved.table('endpoint_neighbours').merge(obs[['sample','image_id']],on='sample').merge(source[['image_id','image_sha256','leakage_component']],on='image_id')
    assert (neighbours.reference_sha256!=neighbours.image_sha256).all() and (neighbours.reference_component!=neighbours.leakage_component).all()
    refs=endpoint[(endpoint.role=='train')&(endpoint.SSL_valid==1)].merge(source[['image_id','image_sha256']],on='image_id')
    ref_vectors=global_z[refs['sample'].to_numpy(dtype='int64'),0].astype('float64');ref_vectors/=np.maximum(np.linalg.norm(ref_vectors,axis=1,keepdims=True),1e-12)
    references=refs.set_index('image_id');reconstructed=[]
    for query,group in neighbours.groupby('sample'):
        current=group.iloc[0];z=global_z[int(query),0].astype('float64');z/=max(np.linalg.norm(z),1e-12)
        scores=ref_vectors@z;allowed=(refs.image_sha256!=current.image_sha256)&(refs.leakage_component!=current.leakage_component)
        ranked=sorted([(float(scores[i]),refs.iloc[i].image_id) for i in np.where(allowed)[0]],key=lambda v:(-v[0],v[1]))[:5]
        actual=group.sort_values('rank');assert actual.reference_id.tolist()==[r[1] for r in ranked]
        assert np.allclose(actual.cosine_similarity,[r[0] for r in ranked],atol=1e-12,rtol=1e-12)
        for row in actual.itertuples():
            ref=references.loc[row.reference_id];assert row.reference_sha256==ref.image_sha256 and row.reference_component==ref.leakage_component
        reconstructed.append(dict(sample=int(query),neighbours=len(actual),exact_top5_IDs=True,cosines_match=True,self_source_and_component_excluded=True))
    tables.write('retrieval_reconstruction',reconstructed)
    tables.write('retrieval_audit',neighbours)
    final=epochs.iloc[-1];last=result['history'][-1]
    assert np.isclose(final.mean_dice,last['mean_dice'],atol=1e-7) and np.isclose(final.human_point_MAE,last['human_point_MAE'],atol=1e-7)
    findings=dict(execution_completed=True,selected_checkpoint_available=False,promotion='research_only',geometry_eligible_epochs=int(epochs.geometry_eligible.sum()),representation_eligible_epochs=int(epochs.collapse_eligible.sum()),
                  rank_min=float(epochs.projector_rank.min()),rank_max=float(epochs.projector_rank.max()),final_validation=final.where(pd.notna(final),None).to_dict(),G0=g0,
                  training_SSL_coverage=1.,point_relative_gain=1-float(final.human_point_MAE)/g0['human_point_MAE'],Dice_delta=float(final.mean_dice)-g0['mean_dice'],
                  new_photos=0,new_optimizer_updates=0,image_model_forwards=0,physical_depth_validated=False,safety_validated=False,classification_trained=False)
    # Pandas row may contain NumPy scalars; make the public JSON explicit and finite.
    findings=json.loads(json.dumps(findings,default=lambda v:v.item() if hasattr(v,'item') else str(v),allow_nan=False))
    atomic(tables.root/'FINDINGS.json',findings)
    return dict(tables=tables,obs=obs,source=source,case=case,regions=regions,points=points,quality=quality,geometry=geometry,patches=patches,projected=projected,global_z=global_z,findings=findings)
