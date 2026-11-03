"""NB15: group-support feasibility and saved-online correction utility, no fitting."""
import argparse,json,os,platform
from pathlib import Path
import numpy as np
import pandas as pd
from public_inputs import Inputs,atomic,sha
CLASSES=['low','mid','high']
def probability(logits):
    x=logits-logits.max(1,keepdims=True);p=np.exp(x);return p/p.sum(1,keepdims=True)
def frames(source,registry,components,tyres,pred,context,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);tables={}
    def save(name,data):
        data=data if isinstance(data,pd.DataFrame) else pd.DataFrame(data)
        path=out/(name+'.csv');temp=path.with_suffix('.part');data.to_csv(temp,index=False,float_format='%.12g',lineterminator='\n');os.replace(temp,path)
        tables[name]=dict(rows=len(data),columns=list(data.columns),sha256=sha(path));return data
    # Filter before consulting numeric labels, losses, logits or corrections.
    source=source[source.role.eq('train')].copy();registry=registry[registry.historical_study_split.eq('train')].copy()
    pred=pred[pred.stage.eq('train')].copy();context=context[context.stage.eq('train')].copy()
    components=components[components.historical_split.eq('train')].copy();tyres=tyres[tyres.historical_study_split.eq('train')].copy()
    assert len(source)==len(registry)==386 and len(pred)==len(context)==47040
    assert set(source.image_id)==set(registry.image_id) and source.image_id.is_unique and registry.image_id.is_unique
    roster=source.merge(registry[['image_id','image_sha256','class_index','confirmed_original_tyre_id','provisional_video_tyre_id','leakage_component']],on='image_id',validate='one_to_one',suffixes=('','_gate')).fillna('')
    assert roster.image_sha256.eq(roster.image_sha256_gate).all() and roster.class_index.eq(roster.class_index_gate).all()
    assert roster.groupby('leakage_component').class_index.nunique().eq(1).all()
    assert set(roster.leakage_component)==set(components.leakage_component)
    counts=roster.groupby('leakage_component').size();assert components.set_index('leakage_component').images.eq(counts).all()
    assert roster.image_sha256.is_unique and not components.crosses_historical_split.astype(str).str.lower().eq('true').any()
    roster=save('training_roster',roster.drop(columns=['image_sha256_gate','class_index_gate']))
    cs=[]
    for c in range(3):
        g=roster[roster.class_index.eq(c)];known=tyres[tyres.proxy_class.eq(CLASSES[c])]
        cs.append(dict(class_index=c,class_name=CLASSES[c],images=len(g),original_images=int(g.domain.eq('original_photo').sum()),video_frames=int(g.domain.ne('original_photo').sum()),known_original_tyres=len(known),conservative_components=g.leakage_component.nunique(),source_groups=g.source_group.nunique()))
    class_support=save('training_class_support',cs)
    component_support=save('training_component_support',roster.groupby(['leakage_component','class_index']).agg(images=('image_id','size'),known_original_tyres=('confirmed_original_tyre_id',lambda s:len(set(s)-{''})),provisional_video_groups=('provisional_video_tyre_id',lambda s:len(set(s)-{''})),bank_images=('is_bank','sum')).reset_index())
    bank=save('frozen_bank_registry',roster[roster.is_bank.eq(1)].copy());assert len(bank)==18 and bank.groupby('class_index').size().eq(6).all()
    folds=[];fold_classes=[];assign=[];refs=[]
    for index,held in enumerate(sorted(roster.leakage_component.unique())):
        selected=roster.leakage_component.eq(held);fit=roster[~selected];held_rows=roster[selected];missing=[]
        for c in range(3):
            train=fit[fit.class_index.eq(c)];test=held_rows[held_rows.class_index.eq(c)]
            if not len(train):missing.append(CLASSES[c])
            fold_classes.append(dict(fold=index,held_component=held,class_index=c,fit_images=len(train),fit_components=train.leakage_component.nunique(),held_images=len(test),held_components=test.leakage_component.nunique(),frozen_bank_fit_references=int((bank.class_index.eq(c)&bank.leakage_component.ne(held)).sum())))
        folds.append(dict(fold=index,held_component=held,fit_images=len(fit),held_images=len(held_rows),fit_classes=fit.class_index.nunique(),held_classes=held_rows.class_index.nunique(),missing_fit_classes=';'.join(missing),all_three_fit_classes=not missing,model_fitted=False))
        for row in roster.itertuples():assign.append(dict(fold=index,image_id=row.image_id,class_index=row.class_index,leakage_component=row.leakage_component,conditional_role='held_component' if row.leakage_component==held else 'fit_components',original_role='train'))
        for row in held_rows.itertuples():
            allowed=bank[bank.leakage_component.ne(held)&bank.image_sha256.ne(row.image_sha256)]
            for c in range(3):
                n=int(allowed.class_index.eq(c).sum())
                refs.append(dict(fold=index,image_id=row.image_id,query_class=row.class_index,reference_class=c,allowed_references=n,normalization_if_supported=6/n if n else None,unsupported=int(n==0),bank_refit=False))
    fold_summary=save('conditional_folds',folds);save('conditional_fold_class_support',fold_classes);save('conditional_fold_assignments',assign);save('held_component_bank_support',refs)
    # Fixed saved observations, not a new model or a fixed final checkpoint cohort.
    assert pred['sample'].is_unique and context['sample'].is_unique
    keep=['sample',*[f'base_logit_{c}' for c in range(3)],*[f'bias_delta_{c}' for c in range(3)],*[f'spatial_delta_{c}' for c in range(3)]]
    rows=pred.merge(context[keep],on='sample',validate='one_to_one').merge(roster[['image_id','leakage_component']],on='image_id',validate='many_to_one').sort_values('sample')
    assert len(rows)==47040 and rows.role.eq('train').all() and rows['sample'].is_unique
    base=rows[[f'base_logit_{c}' for c in range(3)]].to_numpy();bias=rows[[f'bias_delta_{c}' for c in range(3)]].to_numpy();spatial=rows[[f'spatial_delta_{c}' for c in range(3)]].to_numpy()
    actual=rows[[f'logit_{c}' for c in range(3)]].to_numpy();assert np.abs(actual-base-bias-spatial).max()<2e-6
    labels=rows.label.to_numpy(int);bp=probability(base);bpred=bp.argmax(1);bnll=-np.log(bp[np.arange(len(rows)),labels].clip(1e-12));bc=bpred==labels
    variants=[]
    for name,logits in [('frozen_prior',base),('bias_only',base+bias),('spatial_only',base+spatial),('full_context',base+bias+spatial)]:
        pr=probability(logits);prediction=pr.argmax(1);correct=prediction==labels;nll=-np.log(pr[np.arange(len(rows)),labels].clip(1e-12))
        frame=rows[['sample','epoch','parameter_update','image_id','label','domain','source_group','leakage_component']].copy();frame['variant']=name
        for c in range(3):frame[f'logit_{c}']=logits[:,c];frame[f'probability_{c}']=pr[:,c]
        frame['prediction']=prediction;frame['correct']=correct.astype(int);frame['nll']=nll;frame['confidence']=pr.max(1);frame['prior_correct']=bc.astype(int);frame['prior_nll']=bnll;frame['delta_nll']=nll-bnll
        frame['rescued_prior_error']=(~bc&correct).astype(int);frame['spoiled_prior_correct']=(bc&~correct).astype(int);frame['prediction_flip']=(prediction!=bpred).astype(int)
        if name=='full_context':assert np.array_equal(prediction,rows.prediction) and np.allclose(nll,rows.nll,atol=2e-6,rtol=2e-6)
        variants.append(frame)
    values=save('online_component_counterfactuals',pd.concat(variants,ignore_index=True))
    def aggregate(keys):
        return values.groupby(keys).agg(view_observations=('sample','size'),unique_images=('image_id','nunique'),online_accuracy=('correct','mean'),online_nll=('nll','mean'),online_mean_confidence=('confidence','mean'),mean_delta_nll=('delta_nll','mean'),rescued_prior_errors=('rescued_prior_error','sum'),spoiled_prior_corrects=('spoiled_prior_correct','sum'),prediction_flips=('prediction_flip','sum')).reset_index()
    save('online_epoch_variants',aggregate(['epoch','variant']));save('online_class_variants',aggregate(['epoch','label','variant']));save('online_component_variants',aggregate(['epoch','leakage_component','label','variant']));save('online_image_variants',aggregate(['image_id','label','variant']))
    save('online_prior_status_variants',aggregate(['epoch','label','prior_correct','variant']))
    all_folds_have_support=bool(fold_summary.all_three_fit_classes.all())
    gates=save('readiness_checks',[
        dict(check='all_training_images_have_unchanged_hash_identity',passed=True,meaning='386 original training records agree with independently published NB07'),
        dict(check='all_leave_component_out_fits_retain_three_classes',passed=all_folds_have_support,meaning='A complete group-held-out comparison across low/mid/high requires class support in every fit fold'),
        dict(check='at_least_two_conservative_training_components_per_class',passed=bool(class_support.conservative_components.ge(2).all()),meaning='Minimum support for withholding one independent conservative component while retaining that class'),
        dict(check='new_fit_or_workstation_promotion_justified',passed=False,meaning='Existing context result fails low recognition; this metadata/post-hoc gate does not establish useful unseen-tyre classification'),
    ])
    summary=dict(status='passed',scientific_readiness=all_folds_have_support,classifier_status='research_only_unpromoted',next_full_classification_fit_authorized_by_gate=False,
        training_images=386,known_training_original_tyres=int(class_support.known_original_tyres.sum()),training_conservative_components=len(component_support),class_support=class_support.to_dict('records'),
        conditional_folds=len(folds),folds_missing_fit_class=int((~fold_summary.all_three_fit_classes).sum()),online_view_observations=47040,counterfactual_rows=len(values),
        new_model_forwards=0,optimizer_updates=0,new_independent_tyres=0,heldout_prediction_rows_used=0,
        limitation='Frozen LC prior was selected historically using validation. Training-only here describes rows analyzed, not an untouched pipeline or independent generalization experiment.',
        counterfactual_scope='Algebraic parts of saved changing online model states on identical paired views; no independently trained ablations, no component-held-out model fit, no new checkpoint selection.',
        coassignment_scope='Unresolved mid video-parent component is a conservative leakage constraint, not proof that three known mid tyres are the same tyre.')
    atomic(out/'SUMMARY.json',summary)
    atomic(out/'CAPABILITIES.json',dict(version=1,model_family='TREAD-TWO ATLAS through context-r3',classification='research_only_unpromoted',geometry='retained_frozen_TREAD_ONE_outputs_not_newly_certified',explanation='conserved_internal_signed_evidence_and_attention_not_causal_wear_truth',physical_depth_mm='unavailable',safety_assessment='unavailable',new_full_classification_fit='not_supported_by_this_gate',workstation_installation=False,new_photos_required_for_this_audit=False))
    definitions={
        'sample':'Original NB14 observation join key; training IDs contain validation gaps and are never renumbered',
        'parameter_update':'Already-applied optimizer updates for this saved pre-update model state',
        'epoch':'One-based epoch of the saved changing online state',
        'image_id':'Original source image identifier; joins independently pinned NB07/NB14 registries',
        'image_sha256':'SHA-256 of exact original image bytes; both registries must match',
        'leakage_component':'Conservative NB07 co-assignment group; unresolved mid group is not physical identity equivalence',
        'conditional_role':'Metadata-only hypothetical fold role; original role remains train and no fold model is fitted',
        'all_three_fit_classes':'True only if withholding the entire component leaves at least one fit image in every class',
        'missing_fit_classes':'Semicolon-separated classes with no remaining fit images; empty when all three remain',
        'known_original_tyres':'Distinct confirmed original physical tyres; provisional video groups do not add confirmed tyres',
        'conservative_components':'Distinct conservative co-assignment groups, separately counted from confirmed tyres',
        'allowed_references':'Frozen bank references remaining after full held-component and same-source-SHA removal',
        'normalization_if_supported':'6 divided by positive allowed-reference count; blank for zero support',
        'unsupported':'One when no reference remains for the named query/reference class in this conditional fold',
        'variant':'Algebraic frozen_prior, bias_only, spatial_only or full_context using the same saved online state/view',
        'prediction':'Argmax of reconstructed three-class logits; low=0, mid=1, high=2',
        'correct':'One when reconstructed argmax equals this mileage-proxy target',
        'nll':'Negative natural log of reconstructed true-class probability, clipped at 1e-12',
        'prior_nll':'Negative natural log of frozen prior true-class probability for the identical saved view',
        'delta_nll':'Variant NLL minus prior NLL on the same saved view; negative means lower loss',
        'mean_delta_nll':'Arithmetic view-weighted group mean of variant-minus-prior NLL; no new inference',
        'confidence':'Maximum reconstructed class probability; not a safety/calibrated deployment score',
        'rescued_prior_error':'One when prior argmax was wrong and this variant argmax is correct',
        'spoiled_prior_correct':'One when prior argmax was correct and this variant argmax is wrong',
        'prediction_flip':'One when variant and frozen-prior argmax differ on this identical saved view',
        'view_observations':'Number of correlated saved view rows; two per paired query draw',
        'unique_images':'Distinct original image IDs in this row group; not independent tyre count',
        'online_accuracy':'Mean correct over sampled changing pre-update view states; not a fixed final-cohort score',
        'online_nll':'Mean NLL over sampled changing pre-update view states; not an independently fitted ablation',
        'online_mean_confidence':'Mean maximum probability over sampled changing pre-update view states',
        'model_fitted':'Always false: conditional folds are enumerated metadata, not trained models',
        'bank_refit':'Always false: reference support is counted without retraining or applying a replacement bank',
    }
    def units(column):
        if 'nll' in column:return 'natural-log units'
        if 'logit_' in column:return 'unnormalized class-logit units'
        if 'probability_' in column or column in ('confidence','online_accuracy','online_mean_confidence'):return 'dimensionless fraction [0,1]'
        if column in ('normalization_if_supported',):return 'dimensionless scale; blank for unsupported'
        if column in ('correct','prior_correct','rescued_prior_error','spoiled_prior_correct','prediction_flip','unsupported','is_bank','is_query','all_three_fit_classes','model_fitted','bank_refit','passed'):return 'boolean / 0-or-1 indicator'
        if column.endswith(('_id','sha256')) or column in ('run','role','domain','source_group','point_kind','leakage_component','held_component','variant','conditional_role','original_role','image_path','missing_fit_classes','class_name','check','meaning'):return 'text identifier/category/path/hash'
        return 'integer index or count; class encodings are low=0, mid=1, high=2'
    save('table_dictionary',[dict(table=name,column=column,definition=definitions.get(column,column.replace('_',' ')+' within '+name.replace('_',' ')+'; counts retain named image/group/view units; see METHODS.md'),units=units(column),missing='Blank means unsupported/unavailable; never substitute zero for missing support') for name,meta in tables.items() for column in meta['columns']])
    return tables,summary
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cache',required=True);parser.add_argument('--out',required=True);parser.add_argument('--spec',default=str(Path(__file__).with_name('INPUTS.json')));args=parser.parse_args()
    inputs=Inputs(args.spec,args.cache)
    data={key:pd.read_csv(inputs.file(key),keep_default_na=False) for key in inputs.spec['inputs']}
    tables,summary=frames(data['review_source.csv'],data['identity_images.csv'],data['identity_components.csv'],data['identity_tyres.csv'],data['review_predictions.csv'],data['review_context.csv'],args.out)
    out=Path(args.out);atomic(out/'ANALYSIS.json',dict(status='passed',tables=tables,summary=summary,inputs=inputs.used,public_read_requests=inputs.calls,environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__),new_model_forwards=0,optimizer_updates=0,remote_writes=0))
    from render_gate import render
    render(out);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
