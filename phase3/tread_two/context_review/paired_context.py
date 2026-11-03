"""R2 paired-loss, full-source reference permissions and exposure accounting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

def extend(inputs,out,p,source,exposures,updates,coef,save,checks):
    paired=inputs.table('view_pairs');patches=inputs.table('view_patch_consistency');rules=inputs.table('case_rules')
    assert len(paired)==23520 and len(patches)==752640 and len(rules)==967572
    assert paired.sample_a.is_unique and paired.sample_b.is_unique
    pred=p.set_index('sample');a=pred.loc[paired.sample_a];b=pred.loc[paired.sample_b]
    assert np.array_equal(a.image_index,b.image_index) and a.stage.eq('train').all() and b.stage.eq('train').all()
    assert np.array_equal(a.label,b.label) and np.array_equal(a.label,paired.label)
    assert a.variant.eq('appearance_view_a').all() and b.variant.eq('appearance_view_b').all()
    assert np.array_equal(paired.disagree,(a.prediction.to_numpy()!=b.prediction.to_numpy()).astype(int))
    pa=a[[f'probability_{c}' for c in range(3)]].to_numpy();pb=b[[f'probability_{c}' for c in range(3)]].to_numpy();mean=(pa+pb)/2
    js=.5*((pa*np.log(pa.clip(1e-12)/mean.clip(1e-12))).sum(1)+(pb*np.log(pb.clip(1e-12)/mean.clip(1e-12))).sum(1))
    patches['weighted_distance']=patches.pooling_mass*patches.cosine_distance
    texture=patches.groupby('sample_a').weighted_distance.sum().loc[paired.sample_a].to_numpy()
    assert patches.groupby('sample_a').size().eq(32).all()
    ce_a=a.nll.to_numpy();ce_b=b.nll.to_numpy();residual=inputs.table("context_summary").set_index("sample").correction_l2
    regularizer=.5*(residual.loc[paired.sample_a].to_numpy()+residual.loc[paired.sample_b].to_numpy())
    assert np.allclose(regularizer,paired.residual_l2,atol=1e-6,rtol=2e-6)
    total=.5*(ce_a+ce_b)+.1*js+.05*texture+.02*regularizer
    for expected,actual in [(js,paired.js),(texture,paired.texture),(ce_a,paired.ce_a),(ce_b,paired.ce_b),(total,paired.total)]:assert np.allclose(expected,actual,atol=4e-7,rtol=3e-6)
    pair=paired.copy();pair['reconstructed_total']=total;pair['loss_residual']=abs(total-pair.total)
    save('paired_loss_audit',pair)
    avg=paired.groupby('update')[['ce_a','ce_b','js','texture','residual_l2','total']].mean().sort_index()
    u=updates.set_index('update').sort_index()
    for f in ('ce_a','ce_b','js','texture','residual_l2'):assert np.allclose(avg[f],u[f],atol=4e-7,rtol=3e-6)
    assert np.allclose(avg.total,u.loss,atol=4e-7,rtol=3e-6)
    epochs=paired.groupby('epoch').agg(draws=('sample_a','size'),ce_a=('ce_a','mean'),ce_b=('ce_b','mean'),js=('js','mean'),texture=('texture','mean'),residual_l2=('residual_l2','mean'),total=('total','mean'),view_disagreement=('disagree','mean')).reset_index()
    save('paired_epoch_metrics',epochs)
    save('paired_class_metrics',paired.groupby(['epoch','label']).agg(draws=('sample_a','size'),ce_a=('ce_a','mean'),ce_b=('ce_b','mean'),js=('js','mean'),texture=('texture','mean'),view_disagreement=('disagree','mean')).reset_index())
    save('paired_patch_metrics',patches.groupby('patch').agg(observations=('sample_a','size'),mean_pooling_mass=('pooling_mass','mean'),mean_cosine_distance=('cosine_distance','mean'),maximum_cosine_distance=('cosine_distance','max')).reset_index())
    # Exact deterministic six factors per draw; A/B are not independent draws.
    ex=exposures.sort_values(['update','slot','view'])
    assert ex.groupby(['update','slot']).size().eq(2).all() and set(ex.view)=={0,1}
    expected=np.stack([np.random.default_rng(int(s)).uniform(.8,1.2,(2,3))[int(v)] for s,v in zip(ex.augmentation_seed,ex.view)]).astype(np.float32)
    assert np.allclose(expected,ex[['brightness','contrast','gamma']].to_numpy(),atol=1e-7,rtol=0)
    assert ex.groupby(['update','slot']).augmentation_seed.nunique().eq(1).all()
    save('appearance_factor_epochs',ex.groupby(['epoch','view'])[['brightness','contrast','gamma']].agg(['mean','min','max']).reset_index().pipe(lambda x:x.set_axis(['_'.join(str(v) for v in c if str(v)) if isinstance(c,tuple) else c for c in x.columns],axis=1)))
    # Permissions are checked against full original SHA, independent of target.
    bank=inputs.table('bank_cases').sort_values('prototype_index')
    registry=source.set_index('image_id');bank_sha=registry.loc[bank.source_image_id,'image_sha256'].to_numpy()
    assert len(np.unique(bank_sha))==18 and len(bank)==18
    assert np.array_equal(p.source_sha256,registry.loc[p.image_id,'image_sha256'].to_numpy())
    expected_allowed=(p.source_sha256.to_numpy()[:,None]!=bank_sha[None]).astype(int)
    expected_counts=np.stack([expected_allowed[:,c*6:(c+1)*6].sum(1) for c in range(3)],1)
    assert np.array_equal(expected_counts,p[['allowed_low','allowed_mid','allowed_high']].to_numpy())
    rule=rules.sort_values(['sample','prototype']);ordered=p.sort_values('sample')
    assert rules.groupby('sample').size().eq(18).all() and not rules.duplicated(['sample','prototype']).any()
    idx=pd.Index(p['sample']).get_indexer(ordered['sample']);allow=expected_allowed[idx]
    assert np.array_equal(rule.allowed.to_numpy(),allow.reshape(-1))
    assert np.array_equal(rule.same_source_sha256,1-rule.allowed)
    assert np.array_equal(rule.reference_class,np.tile(np.repeat(np.arange(3),6),len(p)))
    scales=6/expected_counts[idx][:,np.repeat(np.arange(3),6)]
    assert np.allclose(rule.normalization_scale.to_numpy(),scales.reshape(-1),atol=1e-7)
    assert expected_allowed[p.stage.isin(['selected_eval','final_eval','diagnostics'])].all()
    rules['blocked']=1-rules.allowed
    permission_summary=rules.merge(p[['sample','stage','image_id','image_index','label']],on='sample',validate='many_to_one')
    save('reference_permission_summary',permission_summary.groupby(['stage','label','reference_class']).agg(observations=('sample','nunique'),reference_rows=('sample','size'),blocked_rows=('blocked','sum'),min_scale=('normalization_scale','min'),max_scale=('normalization_scale','max')).reset_index())
    # Retain draw-level exposure separately from both-view online telemetry.
    one=exposures[exposures.view==0].copy();queries=source[source.is_query.astype(int)==1]
    assert len(queries)==386 and len(one)==23520
    pool=queries.groupby(['class_index','domain','source_group']).size()
    probabilities={}
    for r in queries.itertuples():
        groups=pool.loc[(r.class_index,r.domain)]
        domain_n=queries[queries.class_index==r.class_index].domain.nunique()
        n=groups.loc[r.source_group]
        probabilities[r.image_index]=float(np.sqrt(n)/np.sqrt(groups).sum()/n/3/domain_n)
    assert abs(sum(probabilities.values())-1)<1e-12
    assert np.allclose(one.draw_probability,one.image_index.map(probabilities),atol=1e-14,rtol=1e-12)
    draws=one.groupby('image_index').size()
    exposure=queries[['image_id','image_index','class_index','domain','source_group','physical_tyre_id','is_bank']].copy()
    exposure['actual_draws']=exposure.image_index.map(draws).fillna(0).astype(int)
    exposure['view_observations']=2*exposure.actual_draws
    exposure['draw_probability']=exposure.image_index.map(probabilities)
    exposure['expected_draws']=23520*exposure.draw_probability
    exposure['actual_minus_expected']=exposure.actual_draws-exposure.expected_draws
    save('query_draw_exposure',exposure)
    save('source_group_draws',exposure.groupby(['class_index','domain','source_group']).agg(images=('image_index','size'),draws=('actual_draws','sum'),view_observations=('view_observations','sum'),expected_draws=('expected_draws','sum'),maximum_image_draws=('actual_draws','max')).reset_index())
    save('resource_epochs',updates.groupby('epoch').agg(compute_seconds=('compute_seconds','sum'),record_seconds=('record_seconds','sum'),peak_cuda_allocated=('cuda_peak_allocated_bytes','max'),peak_cuda_reserved=('cuda_peak_reserved_bytes','max'),peak_host_rss=('host_peak_rss_bytes','max'),minimum_free_disk=('free_disk_bytes','min')).reset_index())
    checks.append('All23520 paired CE/JS/texture losses,752640 patch distances,2940 update losses and47040 seeded appearance factors checked')
    checks.append('All967572 source-SHA permissions/scales verified independently of labels; all386 query draw probabilities checked; endpoint references all allowed')
    import hashlib
    peer=Path(__file__).parent/'assets/v1_cohort_metrics.csv';peer_meta=json.loads((peer.parent/'PEER.json').read_text())
    assert hashlib.sha256(peer.read_bytes()).hexdigest()==peer_meta['sha256']
    current=pd.read_csv(Path(out)/'cohort_metrics.csv');old=pd.read_csv(peer)
    keep=['stage','endpoint','variant','n','accuracy','balanced_accuracy_observed','nll','brier','ece_10_equal_width']
    left=old[old.stage.isin(['selected_eval','final_eval','diagnostics'])][keep]
    right=current[current.stage.isin(['selected_eval','final_eval','diagnostics'])][keep]
    comparison=left.merge(right,on=['stage','endpoint','variant'],how='outer',suffixes=('_v1','_r3'),validate='one_to_one')
    for f in ('accuracy','balanced_accuracy_observed','nll','brier','ece_10_equal_width'):comparison['delta_'+f]=comparison[f+'_r3']-comparison[f+'_v1']
    save('v1_r3_comparison',comparison)
    # Additional figure explains the terms actually optimized, not inferred causes.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    axes[0].plot(epochs.epoch,.5*(epochs.ce_a+epochs.ce_b),label='Mean paired CE');axes[0].plot(epochs.epoch,epochs.total,label='Joint loss');axes[0].legend()
    axes[1].plot(epochs.epoch,epochs.js,label='Probability JS');axes[1].plot(epochs.epoch,epochs.texture,label='Embedding distance');axes[1].legend()
    axes[2].plot(epochs.epoch,epochs.view_disagreement);axes[2].set_ylabel('Fraction of paired predictions differing')
    for ax in axes:ax.set_xlabel('Epoch');ax.grid(alpha=.2)
    fig.suptitle('ATLAS-CONTEXT · saved online paired observations · not fixed-checkpoint cohort accuracy')
    folder=Path(out)/'figures';folder.mkdir(exist_ok=True);fig.savefig(folder/'paired_learning.png',dpi=160);plt.close(fig)
