"""CSV-only correction, attention and frozen-prior accounting; no model imports."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

def extend(inputs,out,p,coef,save,checks):
    from review_context import scores
    n=len(p);ordered=p.sort_values('sample');assert ordered['sample'].tolist()==list(range(n))
    summary=inputs.table('context_summary').sort_values('sample').set_index('sample')
    delta=inputs.table('context_evidence').sort_values(['sample','patch']).reset_index(drop=True)
    support=inputs.table('patch_support').sort_values(['sample','patch']).reset_index(drop=True)
    assert len(summary)==n and len(delta)==len(support)==n*32
    assert np.array_equal(delta['sample'],np.repeat(np.arange(n),32)) and np.array_equal(delta.patch,np.tile(np.arange(32),n))
    assert np.array_equal(support['sample'],delta['sample']) and np.array_equal(support.patch,delta.patch)
    d=delta[['delta_0','delta_1','delta_2']].to_numpy().reshape(n,32,3)
    alpha=support.pooling_mass.to_numpy().reshape(n,32)
    assert np.isfinite(d).all() and np.isfinite(summary.select_dtypes(include=np.number).to_numpy()).all()
    assert np.isfinite(alpha).all() and (alpha>=0).all()
    assert np.allclose(alpha.sum(1),1-ordered.empty_support.to_numpy(),atol=2e-7)
    assert (np.abs(d)<=2*alpha[:,:,None]+2e-7).all()
    expected=np.tile(np.arange(32),(n,1))
    for variant,order in [('reverse_bands',[r*4+c for r in reversed(range(8)) for c in range(4)]),('reverse_columns',[r*4+c for r in range(8) for c in reversed(range(4))])]:
        expected[ordered.variant.eq(variant).to_numpy()]=order
    assert np.array_equal(delta.source_patch.to_numpy().reshape(n,32),expected)
    bias_delta=summary[[f'bias_delta_{c}' for c in range(3)]].to_numpy();assert (np.abs(bias_delta)<=.5+1e-7).all()
    spatial=d.sum(1);total=spatial+bias_delta;base=summary[[f'base_logit_{c}' for c in range(3)]].to_numpy()
    logits=ordered[[f'logit_{c}' for c in range(3)]].to_numpy();error=np.abs(base+total-logits).max(1)
    assert error.max()<2e-6,error.max()
    penalty=(total**2).mean(1);assert np.allclose(penalty,summary.correction_l2,atol=1e-6,rtol=2e-6)
    weights=coef[coef.kind=='signed_weight'];assert weights.groupby(['class_index','prototype']).value.nunique().eq(1).all()
    assert coef[coef.kind=='temperature'].value.nunique()==1
    base_bias=coef[(coef.parameter_update==0)&(coef.kind=='bias')].sort_values('class_index').value.to_numpy();assert len(base_bias)==3
    biases=coef[coef.kind=='bias'].pivot(index='parameter_update',columns='class_index',values='value')
    assert np.allclose(biases.loc[ordered.parameter_update].to_numpy(),base_bias+bias_delta,atol=3e-7)
    bp=np.exp(base-base.max(1,keepdims=True));bp/=bp.sum(1,keepdims=True)
    assert np.array_equal(bp.argmax(1),summary.base_prediction)
    assert np.allclose(-np.log(bp[np.arange(n),ordered.label.to_numpy(int)].clip(1e-12)),summary.base_nll,atol=2e-6,rtol=2e-6)
    observations=summary.reset_index().merge(ordered[['sample','stage','endpoint','variant','epoch','parameter_update','image_id','label','prediction','nll']],on='sample',validate='one_to_one')
    for c in range(3):observations[f'spatial_delta_{c}']=spatial[:,c];observations[f'total_delta_{c}']=total[:,c]
    observations['logit_reconstruction_residual']=error;observations['penalty_reconstruction_residual']=np.abs(penalty-summary.correction_l2.to_numpy())
    observations['prediction_changed_from_base']=(observations.prediction!=observations.base_prediction).astype(int)
    observations['delta_nll_from_base']=observations.nll-observations.base_nll
    save('context_observations',observations)
    save('context_epochs',observations.groupby(['stage','epoch']).agg(observations=('sample','size'),prediction_flips=('prediction_changed_from_base','sum'),mean_base_nll=('base_nll','mean'),mean_nll=('nll','mean'),mean_correction_l2=('correction_l2','mean'),max_accounting_residual=('logit_reconstruction_residual','max')).reset_index())
    save('context_by_variant',observations.groupby(['stage','endpoint','variant','label']).agg(images=('sample','size'),prediction_flips=('prediction_changed_from_base','sum'),mean_base_nll=('base_nll','mean'),mean_nll=('nll','mean'),mean_correction_l2=('correction_l2','mean'),mean_delta_low=('total_delta_0','mean'),mean_delta_mid=('total_delta_1','mean'),mean_delta_high=('total_delta_2','mean')).reset_index())
    endpoint=ordered[ordered.stage.isin(['selected_eval','final_eval','diagnostics'])]
    ids=endpoint['sample'].to_numpy();saved=delta[delta['sample'].isin(ids)].copy();save('endpoint_context_evidence',saved)
    baseline=ordered.copy();baseline['prediction']=bp.argmax(1)
    for c in range(3):baseline[f'probability_{c}']=bp[:,c]
    base_scores=[]
    for (stage,ep,var,epoch),g in baseline.groupby(['stage','endpoint','variant','epoch']):base_scores.append(dict(stage=stage,endpoint=ep,variant=var,epoch=epoch,**scores(g)))
    save('frozen_LC_prior_metrics',base_scores)
    paired=observations[observations['sample'].isin(ids)].copy();save('base_context_pairs',paired)
    original_observations=observations[observations.stage.isin(['selected_eval','final_eval'])].set_index(['endpoint','image_id'])
    probe_pairs=pd.read_csv(Path(out)/'probe_pairs.csv').set_index('probe_sample');layout=[]
    for row in observations[observations.variant.isin(['reverse_bands','reverse_columns'])].itertuples():
        original=original_observations.loc[(row.endpoint,row.image_id)]
        base_change=max(abs(getattr(row,f'base_logit_{c}')-original[f'base_logit_{c}']) for c in range(3))
        correction_change=max(abs(getattr(row,f'total_delta_{c}')-original[f'total_delta_{c}']) for c in range(3))
        assert base_change<2e-6
        pair=probe_pairs.loc[row.sample]
        layout.append(dict(sample=row.sample,endpoint=row.endpoint,variant=row.variant,image_id=row.image_id,label=row.label,
            max_frozen_logit_change=base_change,max_context_logit_change=correction_change,max_probability_change=pair.max_absolute_probability_change,prediction_flip=int(pair.flip)))
    save('layout_prior_context_delta',layout)
    checks.append('All412 layout probes preserve the frozen prior; learned correction and probability response measured separately')
    checks.append('All53754 base+spatial+bias logits,correction bounds/penalty/source mappings and frozen reference coefficients checked')

    attention=inputs.table('endpoint_attention').sort_values(['sample','layer','head','query','key'])
    original=ordered[ordered.stage.isin(['selected_eval','final_eval'])].sort_values('sample');aid=original['sample'].to_numpy()
    assert len(attention)==len(aid)*8192 and np.array_equal(attention['sample'],np.repeat(aid,8192))
    for field,pattern in [('layer',np.repeat(np.arange(2),4096)),('head',np.tile(np.repeat(np.arange(4),1024),2)),('query',np.tile(np.repeat(np.arange(32),32),8)),('key',np.tile(np.arange(32),256))]:
        assert np.array_equal(attention[field],np.tile(pattern,len(aid))),field
    a=attention.value.to_numpy().reshape(len(aid),2,4,32,32)
    assert np.isfinite(a).all() and a.min()>=0 and a.max()<=1
    row_error=np.abs(a.sum(-1)-1).max(axis=(1,2,3));assert row_error.max()<3e-7
    coverage=support.valid_fraction.to_numpy().reshape(n,32)[aid];mass=alpha[aid]
    key_mask=coverage>0;key_mask[~key_mask.any(1),0]=True
    assert np.where(key_mask[:,None,None,None,:],0,a).max()<1e-20
    ent=-(a*np.log(a.clip(1e-12))).sum(-1)/np.log(np.maximum(2,(coverage>0).sum(1)))[:,None,None,None]
    ent=(ent*mass[:,None,None,:]).sum(-1);maximum=(a.max(-1)*mass[:,None,None,:]).sum(-1)
    rows=[]
    for i,sample in enumerate(aid):
        for layer in range(2):
            for head in range(4):
                sr=summary.loc[sample]
                ee=abs(ent[i,layer,head]-sr[f'entropy_{layer}_{head}']);me=abs(maximum[i,layer,head]-sr[f'max_attention_{layer}_{head}'])
                assert ee<7e-7 and me<7e-7
                rows.append(dict(sample=sample,endpoint=original.iloc[i].endpoint,image_id=original.iloc[i].image_id,label=int(original.iloc[i].label),layer=layer,head=head,normalized_entropy=ent[i,layer,head],weighted_max_attention=maximum[i,layer,head],entropy_residual=ee,max_residual=me,row_sum_max_residual=row_error[i]))
    att_summary=save('attention_head_audit',rows)
    save('attention_by_endpoint',att_summary.groupby(['endpoint','layer','head']).agg(images=('sample','size'),mean_entropy=('normalized_entropy','mean'),mean_max_attention=('weighted_max_attention','mean'),maximum_entropy_error=('entropy_residual','max'),maximum_row_sum_error=('row_sum_max_residual','max')).reset_index())
    # Full matrices only belong to saved original endpoints, never fabricate them for probes.
    (Path(out)/'ATTENTION_VIEW.json').write_text(json.dumps({str(int(s)):a[i].round(7).tolist() for i,s in enumerate(aid)},separators=(',',':'))+'\n',encoding='utf-8')
    checks.append('All1687552 endpoint attention values/keys/row sums and1648 head entropy/max summaries checked')

    root=Path(__file__).parent/'assets';meta=json.loads((root/'PEER.json').read_text())
    for name,key in [('lc_cohort_metrics.csv','lc_metrics_sha256'),('lc_selected_identity.csv','lc_selected_identity_sha256')]:assert hashlib.sha256((root/name).read_bytes()).hexdigest()==meta[key]
    peer=pd.read_csv(root/'lc_selected_identity.csv').set_index('image_id')
    identity=original.merge(observations[['sample','base_logit_0','base_logit_1','base_logit_2']],on='sample',validate='one_to_one')
    compared=[]
    for row in identity.itertuples():
        previous=peer.loc[row.image_id];err=max(abs(getattr(row,f'base_logit_{c}')-previous[f'logit_{c}']) for c in range(3))
        assert err<2e-6
        compared.append(dict(endpoint=row.endpoint,image_id=row.image_id,label=row.label,frozen_prior_max_logit_difference=err))
    save('frozen_prior_peer_check',compared)
    old=pd.read_csv(root/'lc_cohort_metrics.csv');current=pd.read_csv(Path(out)/'cohort_metrics.csv')
    keep=['stage','endpoint','variant','n','accuracy','balanced_accuracy_observed','nll','brier','ece_10_equal_width']
    left=old[old.stage.isin(['selected_eval','final_eval','diagnostics'])][keep];right=current[current.stage.isin(['selected_eval','final_eval','diagnostics'])][keep]
    comparison=left.merge(right,on=['stage','endpoint','variant'],how='outer',suffixes=('_LC','_context'),validate='one_to_one')
    for f in ('accuracy','balanced_accuracy_observed','nll','brier','ece_10_equal_width'):comparison['delta_'+f]=comparison[f+'_context']-comparison[f+'_LC']
    save('LC_context_comparison',comparison)
    checks.append('All206 identity frozen-prior logits match pinned selected LC epoch7; peer comparisons are descriptive')
    return dict(summary=summary,deltas=delta,base_bias=base_bias)

def patch_accounting(a,support,patch,coef,ctx,save,checks):
    samples=np.sort(a['sample'].unique());n=len(samples)
    affinity=a.sort_values(['sample','patch','prototype']).affinity.to_numpy().reshape(n,32,18)
    mass=support.sort_values(['sample','patch']).pooling_mass.to_numpy().reshape(n,32)
    weights=coef[(coef.parameter_update==0)&(coef.kind=='signed_weight')].pivot(index='class_index',columns='prototype',values='value').to_numpy()
    base=np.einsum('npr,cr,np->npc',affinity,weights,mass)
    d=ctx['deltas'][ctx['deltas']['sample'].isin(samples)].sort_values(['sample','patch'])[['delta_0','delta_1','delta_2']].to_numpy().reshape(n,32,3)
    actual=patch.sort_values(['sample','patch','class_index']).contribution.to_numpy().reshape(n,32,3)
    error=np.abs(base+d-actual).max(axis=(1,2));assert error.max()<2e-6
    save('endpoint_patch_accounting',pd.DataFrame(dict(sample=samples,maximum_patch_residual=error)))
    checks.append('All1854 endpoint/probe combined patch decompositions checked against frozen reference branch plus spatial correction')

def write_notes(out):
    analysis=json.loads((out/'ANALYSIS.json').read_text());metrics=pd.read_csv(out/'cohort_metrics.csv');rows=metrics[metrics.stage.isin(['selected_eval','final_eval'])]
    lines=['# ATLAS-CONTEXT r3 — completed saved-evidence review','', '**No fitting, new inference, checkpoint reselection or new tyre acquisition.**','', '| Endpoint | Accuracy | Observed-class BA | NLL | Brier | ECE |','|---|---:|---:|---:|---:|---:|']
    for r in rows.itertuples():lines.append(f'| {r.endpoint} | {r.accuracy:.6f} | {r.balanced_accuracy_observed:.6f} | {r.nll:.6f} | {r.brier:.6f} | {r.ece_10_equal_width:.6f} |')
    lines+=['','Use [interactive evidence](ATLAS_Failure_Explorer.html), [CSV index](CSV_INDEX.md) and [methods](METHODS.md). Both workstations remain unchanged. These are previously inspected low/high historical endpoints, with no mid targets or independent confirmation. Classes are mileage proxies, not physical depth/wear/safety truth.','', 'Reference cards explain only the frozen LC branch. Context corrections and combined signed patch accounting are separate. Attention probabilities are internal diagnostics, not causal physical explanations. Structural probes rearrange saved tokens; all thumbnails show the original image. No new probe image or independent tyre is claimed.']
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with (out/'METHODS.md').open('a',encoding='utf-8') as f:f.write('''\n## Context and attention accounting

Every observation retains a frozen LC logit, 32 weighted spatial correction vectors and three bounded global bias corrections. Their sum must equal the combined saved logits; corrections are bounded by 2×pooling mass per tile and 0.5 globally. The paired objective additionally includes 0.02 times the mean two-view squared total correction. Per-pair CE/JS/contextual cosine loss and per-update means are independently reconstructed.

All 53,754 correction records and 1,720,128 source-patch mappings/bounds are checked. Endpoint/probe reference affinities, weights and combined patch values are audited for all 1,854 cases. Full training affinities are retained in raw HF CSV; this review does not independently recalculate every training affinity. The separate full publication audit verifies all raw bytes/schemas/row counts.

Full attention matrices exist only for 206 selected/final identity endpoints: two layers, four heads, 32 query rows and 32 keys. Every value and masked key is checked; rows sum to one. Normalized entropy divides by log(max(2, valid keys)); query rows are averaged with geometric pooling mass. Saved per-head maxima use the same weights. The explorer displays seven-decimal copies for interaction; raw CSV retains original precision. No full training/probe matrices are invented from summary statistics.

The frozen selected LC epoch7 prior is checked against pinned peer logits on all206 identity endpoint images. It is not LC's final checkpoint. LC/context/v1 comparisons describe changed models on an already inspected cohort, not an independent single-factor causal experiment or a deployment gate.
''')
    lines=['# Derived context-review CSV inventory','','| Table | Rows | Columns |','|---|---:|---:|']
    for name,m in analysis['csv_tables'].items():lines.append(f"| [{name}.csv]({name}.csv) | {m['rows']:,} | {len(m['columns'])} |")
    lines+=['','[Column dictionary](table_dictionary.csv) · [Methods and limits](METHODS.md). Missing/absent-class metrics stay blank; observed zeros stay zero. Larger telemetry does not create independent tyres.']
    (out/'CSV_INDEX.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # A separate figure makes prior/correction accounting visible.
    import matplotlib.pyplot as plt
    context=pd.read_csv(out/'context_by_variant.csv').query("stage in ['selected_eval','final_eval']")
    attention=pd.read_csv(out/'attention_by_endpoint.csv');comparison=pd.read_csv(out/'LC_context_comparison.csv').query("stage in ['selected_eval','final_eval']")
    fig,axes=plt.subplots(1,3,figsize=(14,4.3),layout='constrained')
    for ep,color,off in [('selected','#158473',-.18),('final','#a34e9a',.18)]:
        subset=context[context.endpoint==ep];values=[np.average(subset[f'mean_delta_{c}'],weights=subset.images) for c in ('low','mid','high')]
        axes[0].bar(np.arange(3)+off,values,width=.35,label=ep,color=color)
        a=attention[attention.endpoint==ep].sort_values(['layer','head']);axes[1].plot(range(8),a.mean_entropy,marker='o',label=ep,color=color)
    axes[0].set(xticks=range(3),xticklabels=['low','mid','high'],ylabel='Mean total context correction (logit units)');axes[0].axhline(0,color='#555',lw=.7);axes[0].legend()
    axes[1].set(xticks=range(8),xticklabels=['L1H1','L1H2','L1H3','L1H4','L2H1','L2H2','L2H3','L2H4'],ylabel='Normalized attention entropy');axes[1].tick_params(axis='x',rotation=45);axes[1].legend()
    for i,ep in enumerate(['selected','final']):
        row=comparison[comparison.endpoint==ep].iloc[0];axes[2].bar(i-.18,row.nll_LC,width=.35,color='#647f9b',label='LC' if i==0 else None);axes[2].bar(i+.18,row.nll_context,width=.35,color='#d8864b',label='Context' if i==0 else None)
    axes[2].set(xticks=range(2),xticklabels=['selected','final'],ylabel='Historical endpoint NLL');axes[2].legend()
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.suptitle('Saved spatial correction and attention · additive accounting is not causal truth');fig.savefig(out/'figures/context_and_attention.png',dpi=160);plt.close(fig)
