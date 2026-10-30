"""Post-hoc ATLAS evidence audit. No Torch, model forward pass or fitting.

Inputs are immutable NB11 objects at a pinned HF commit. Local caches are
rechecked, partial downloads are atomic, and derived CSVs never alter NB11.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REVISION = '3d5ae39602c3b25caa6af65c1e8d45f53a08e1c8'
PROTOCOL = 'ae918ce787775192ce38031936cbfd8285b51b72fbbedb1da570c9271018eb28'
PREFIX = 'phase3/tread-two/atlas-lc-r2/fit/' + PROTOCOL
BASE = f'https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/resolve/{REVISION}/{PREFIX}/'
CLASSES = ['low','mid','high']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,path)


class Inputs:
    def __init__(self, cache, manifest=None):
        self.cache = Path(cache)
        self.cache.mkdir(parents=True,exist_ok=True)
        self.session = requests.Session()
        self.session.trust_env = False  # Public reads; no alternate token lookup.
        self.reads = 0
        path = Path(manifest) if manifest else self.cache/'REMOTE_MANIFEST.json'
        if not path.exists():
            body = self.request(BASE+'MANIFEST.json')
            path.write_bytes(body)
        self.manifest = json.loads(path.read_text(encoding='utf-8'))
        assert self.manifest['contract'] == PROTOCOL and self.manifest['format']=='immutable-direct-files-v1'
        self.manifest_sha = sha(path)
        self.used = {}

    def request(self, url):
        for attempt in range(8):
            self.reads += 1
            try:
                response = self.session.get(url,timeout=(20,150))
                if response.status_code == 429 or response.status_code >= 500:
                    value = response.headers.get('Retry-After','0')
                    try: delay = float(value)
                    except ValueError:
                        from email.utils import parsedate_to_datetime
                        delay = max(0,parsedate_to_datetime(value).timestamp()-time.time())
                    reset = re.search(r';t=(\d+)',response.headers.get('RateLimit',''))
                    delay = max(delay,float(reset[1]) if reset and response.status_code==429 else 0,min(60,2**attempt))
                    if attempt==7: response.raise_for_status()
                    print('Public evidence read delayed by HF; verified local files retained.',flush=True)
                    deadline=time.monotonic()+delay
                    while time.monotonic()<deadline: time.sleep(min(10,max(0,deadline-time.monotonic())))
                    continue
                response.raise_for_status()
                return response.content
            except (requests.Timeout,requests.ConnectionError):
                if attempt==7: raise
                time.sleep(min(30,2**attempt))
        raise RuntimeError('Bounded read attempts exhausted')

    def file(self, name):
        meta = self.manifest['files'][name]
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        assert '..' not in Path(meta['object']).parts
        target = self.cache/name
        if not target.exists() or target.stat().st_size!=meta['bytes'] or sha(target)!=meta['sha256']:
            body = self.request(BASE+meta['object'])
            assert len(body)==meta['bytes'] and hashlib.sha256(body).hexdigest()==meta['sha256'],name
            target.parent.mkdir(parents=True,exist_ok=True)
            tmp=target.with_name(target.name+'.part')
            tmp.write_bytes(body)
            os.replace(tmp,target)
        self.used[name]=dict(bytes=meta['bytes'],sha256=meta['sha256'])
        if len(self.used)%50==0:
            print('Verified review input files:',len(self.used),flush=True)
        return target

    def table(self,family):
        names=sorted(n for n in self.manifest['files'] if re.fullmatch(r'csv/epoch_\d+/'+re.escape(family)+r'\.csv',n))
        assert names,family
        return pd.concat([pd.read_csv(self.file(n)) for n in names],ignore_index=True)


def scores(rows):
    y=rows.label.to_numpy(dtype=int)
    pred=rows.prediction.to_numpy(dtype=int)
    prob=rows[[f'probability_{c}' for c in range(3)]].to_numpy()
    confusion=np.zeros((3,3),dtype=int)
    np.add.at(confusion,(y,pred),1)
    n=len(rows)
    recalls=[confusion[c,c]/confusion[c].sum() for c in range(3) if confusion[c].sum()]
    confidence=prob.max(1)
    ece=0.
    for bucket in range(10):
        mask=np.minimum((confidence*10).astype(int),9)==bucket
        if mask.any(): ece+=mask.mean()*abs(float((pred[mask]==y[mask]).mean())-confidence[mask].mean())
    return dict(n=n,accuracy=float((pred==y).mean()),balanced_accuracy_observed=float(np.mean(recalls)),
                nll=float(-np.log(prob[np.arange(n),y].clip(1e-12)).mean()),
                brier=float(((prob-np.eye(3)[y])**2).sum(1).mean()),ece_10_equal_width=float(ece),
                mean_confidence=float(confidence.mean()),low_to_mid=int(confusion[0,1]),
                predicted_low=int((pred==0).sum()),predicted_mid=int((pred==1).sum()),predicted_high=int((pred==2).sum()))


def audit(inputs, out):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    checks=[]
    tables={}
    inputs.file('report/ATLAS_Evidence_Explorer.html')  # Images are part of input provenance too.
    def save(name,frame):
        frame=frame if isinstance(frame,pd.DataFrame) else pd.DataFrame(frame)
        assert not frame.empty,name
        target=out/(name+'.csv')
        temp=target.with_suffix('.part')
        frame.to_csv(temp,index=False,float_format='%.12g',lineterminator='\n')
        os.replace(temp,target)
        tables[name]=dict(rows=len(frame),columns=list(frame.columns),sha256=sha(target))
        return frame

    cursor=json.loads(inputs.file('CURSOR.json').read_text())
    assert cursor['phase']==cursor['status']=='completed' and cursor['updates']==2940
    p=inputs.table('predictions')
    source=inputs.table('source_images').fillna('')
    assert len(p)==53342 and len(source)==570 and p['sample'].is_unique
    assert sorted(p['sample'])==list(range(53342))
    assert p.groupby('stage').size().to_dict()==dict(train=47040,validate=4860,selected_eval=103,final_eval=103,diagnostics=1236)
    assert source.image_id.is_unique and source.image_index.is_unique
    p=p.merge(source[['image_id','image_index','role','domain','physical_tyre_id','source_group','is_bank','is_query']],on=['image_id','image_index'],validate='many_to_one')
    assert not p['role'].isna().any()
    assert set(p.loc[p.stage=='train','role'])=={'train'} and set(p.loc[p.stage=='validate','role'])=={'validation'}
    assert set(p.loc[p.stage.isin(['selected_eval','final_eval','diagnostics']),'role'])=={'test'}
    assert source.loc[source.role=='train','is_query'].astype(int).eq(1).all()
    prob=p[[f'probability_{c}' for c in range(3)]].to_numpy()
    logits=p[[f'logit_{c}' for c in range(3)]].to_numpy()
    expected=np.exp(logits-logits.max(1,keepdims=True));expected/=expected.sum(1,keepdims=True)
    assert np.allclose(expected,prob,atol=2e-7,rtol=2e-6)
    assert (logits.argmax(1)==p.prediction).all() and (p.correct==p.prediction.eq(p.label).astype(int)).all()
    actual=-np.log(prob[np.arange(len(p)),p.label.to_numpy(int)].clip(1e-12))
    assert np.allclose(actual,p.nll,atol=5e-7,rtol=2e-6)
    assert np.allclose(((prob-np.eye(3)[p.label.to_numpy(int)])**2).sum(1),p.brier,atol=5e-7)
    p['true_probability']=prob[np.arange(len(p)),p.label.to_numpy(int)]
    p['confidence']=prob.max(1)
    p['true_margin']=np.array([v[c]-max(v[d] for d in range(3) if d!=c) for v,c in zip(logits,p.label.to_numpy(int))])
    save('prediction_audit',p)
    checks.append('All 53342 prediction IDs, roles, probabilities, raw argmax, correctness, NLL and Brier checked')

    rows=[]; class_rows=[]; confusion_rows=[]
    for key,g in p.groupby(['stage','endpoint','variant','epoch'],sort=True):
        base=dict(zip(['stage','endpoint','variant','epoch'],key))
        rows.append(dict(base,**scores(g)))
        for c in range(3):
            group=g[g.label==c]
            class_rows.append(dict(base,class_index=c,target_n=len(group),recall=float(group.correct.mean()) if len(group) else None,
                mean_true_probability=float(group.true_probability.mean()) if len(group) else None,
                mean_confidence=float(group.confidence.mean()) if len(group) else None,
                mean_true_margin=float(group.true_margin.mean()) if len(group) else None,mean_nll=float(group.nll.mean()) if len(group) else None))
            for predicted in range(3):
                confusion_rows.append(dict(base,target_class=c,predicted_class=predicted,count=int((group.prediction==predicted).sum())))
    # Same original-frame dense anchor on the same saved endpoint images.
    # These two context rows describe one fixed seed3/epoch20 control, not
    # two new fits or transformed-control forwards.
    for endpoint,g in p[p.stage.isin(['selected_eval','final_eval'])].groupby('endpoint'):
        baseline=g.copy()
        baseline['prediction']=baseline.anchor_prediction
        for c in range(3): baseline[f'probability_{c}']=baseline[f'anchor_probability_{c}']
        assert (baseline[[f'probability_{c}' for c in range(3)]].to_numpy().argmax(1)==baseline.prediction).all()
        rows.append(dict(stage='frozen_control',endpoint=endpoint,variant='original_frame',epoch=20,**scores(baseline)))
    rows += [dict(stage='train',endpoint='paired_online_combined',variant='paired_views_combined',epoch=e,**scores(g)) for e,g in p[p.stage=='train'].groupby('epoch')]
    summary=save('cohort_metrics',rows)
    save('class_metrics',class_rows);save('confusion',confusion_rows)
    metrics=inputs.table('metrics')
    cohort=metrics[(metrics.level=='cohort') & metrics.metric.isin(['balanced_accuracy_observed','nll','brier','accuracy','ece_10_equal_width'])]
    for row in cohort.itertuples():
        endpoint,variant=(row.endpoint.split(':',1) if row.stage=='diagnostics' else (row.endpoint,'identity'))
        if row.stage=='train': endpoint,variant='paired_online_combined','paired_views_combined'
        observed=summary[(summary.stage==row.stage)&(summary.endpoint==endpoint)&(summary.variant==variant)&(summary.epoch==row.epoch)]
        assert len(observed)==1 and math.isclose(observed.iloc[0][row.metric],row.value,abs_tol=5e-7),(row.stage,row.metric)
    selection=inputs.table('selection')
    best=selection.sort_values(['validation_ba_observed','validation_nll','epoch'],ascending=[False,True,True]).iloc[0]
    assert int(best.epoch)==int(selection.iloc[-1].selected_epoch)==7
    validation=p[p.stage=='validate']
    assert math.isclose(selection.validation_ba_observed.max(),.5625)
    assert validation[validation.label==0].correct.any()
    save('selection_audit',selection)
    checks.append('All saved cohort accuracy, BA, NLL, Brier and ECE rows recomputed; validation selection epoch7 checked')

    group_rows=[]
    for key,g in p[p.stage.isin(['selected_eval','final_eval','diagnostics'])].groupby(['stage','endpoint','variant','label','physical_tyre_id']):
        group_rows.append(dict(zip(['stage','endpoint','variant','label','physical_tyre_id'],key),**scores(g)))
    save('endpoint_tyre_metrics',group_rows)
    probes=[]
    identity=p[p.stage.isin(['selected_eval','final_eval'])]
    assert identity[identity.label.eq(0)].prediction.eq(1).all()
    assert identity[identity.label.eq(2)].prediction.eq(2).all()
    for (endpoint,variant),g in p[p.stage=='diagnostics'].groupby(['endpoint','variant']):
        baseline=identity[identity.endpoint==endpoint].set_index('image_id')
        altered=g.set_index('image_id').loc[baseline.index]
        assert len(altered)==103 and altered.label.equals(baseline.label)
        for identifier,row in altered.iterrows():
            ref=baseline.loc[identifier]
            probes.append(dict(endpoint=endpoint,variant=variant,image_id=identifier,label=int(row.label),
                base_sample=int(ref['sample']),probe_sample=int(row['sample']),base_prediction=int(ref.prediction),probe_prediction=int(row.prediction),
                flip=int(ref.prediction!=row.prediction),base_correct=int(ref.correct),probe_correct=int(row.correct),
                delta_true_probability=row.true_probability-ref.true_probability,delta_confidence=row.confidence-ref.confidence,
                delta_true_margin=row.true_margin-ref.true_margin,delta_nll=row.nll-ref.nll,
                max_absolute_probability_change=max(abs(row[f'probability_{c}']-ref[f'probability_{c}']) for c in range(3)),
                occluded_patch=row.occluded_patch,replacement_r=row.replacement_r,replacement_g=row.replacement_g,replacement_b=row.replacement_b))
    pairs=save('probe_pairs',probes)
    probe_summary=[]
    for (endpoint,variant),g in pairs.groupby(['endpoint','variant']):
        probe_summary.append(dict(endpoint=endpoint,variant=variant,n=len(g),flips=int(g.flip.sum()),
            mean_delta_true_probability=float(g.delta_true_probability.mean()),mean_delta_nll=float(g.delta_nll.mean()),
            maximum_probability_change=float(g.max_absolute_probability_change.max())))
    save('probe_summary',probe_summary)
    checks.append('All1236 diagnostic rows paired by image/checkpoint against their identity endpoint')

    exposures=inputs.table('sample_exposures')
    train=p[p.stage=='train']
    ex=exposures.merge(train[['sample','image_index','label','domain','physical_tyre_id','source_group','correct','nll']],
        on=['sample','image_index'],validate='one_to_one',suffixes=('_draw','_registry'))
    assert len(ex)==47040 and (ex.class_draw==ex.label).all()
    assert (ex.source_group_draw==ex.source_group_registry).all()
    assert exposures.groupby('update').size().eq(16).all() and sorted(exposures['update'].unique())==list(range(1,2941))
    assert exposures[['brightness','contrast','gamma']].ge(.8-1e-6).all().all() and exposures[['brightness','contrast','gamma']].le(1.2+1e-6).all().all()
    queries=source[source.is_query.astype(int)==1]
    exposure_rows=[]
    for row in queries.itertuples():
        selected=ex[ex.image_index==row.image_index]
        exposure_rows.append(dict(image_id=row.image_id,image_index=row.image_index,class_index=row.class_index,
            domain=row.domain,physical_tyre_id=row.physical_tyre_id,source_group=row.source_group,
            training_view_observations=len(selected),epochs_seen=selected.epoch.nunique(),first_update=int(selected['update'].min()) if len(selected) else None,
            online_accuracy=float(selected.correct.mean()) if len(selected) else None,online_nll=float(selected.nll.mean()) if len(selected) else None))
    coverage=save('query_exposure',exposure_rows)
    assert len(coverage)==386 and coverage.training_view_observations.gt(0).all()
    save('sampling_groups',ex.groupby(['class_draw','domain','source_group_draw']).agg(view_observations=('sample','size'),unique_images=('image_index','nunique'),online_accuracy=('correct','mean')).reset_index())
    save('source_registry',source)
    updates=inputs.table('train_updates')
    assert len(updates)==2940 and updates['update'].tolist()==list(range(1,2941))
    assert updates.anchor_gradient_count.eq(0).all() and updates.gradient_finite.eq(1).all()
    assert updates.grad_norm_after_clip.le(1.00001).all()
    gradients=inputs.table('parameter_gradients')
    assert len(gradients)==64680 and gradients.finite.eq(1).all() and gradients.groupby('update').size().eq(22).all()
    save('gradient_parameters',gradients.groupby('parameter').agg(updates=('update','size'),mean_norm=('norm','mean'),maximum_norm=('norm','max'),zero_norm_updates=('norm',lambda x:int(x.eq(0).sum()))).reset_index())
    save('optimizer_epochs',updates.groupby('epoch').agg(updates=('update','size'),mean_loss=('loss','mean'),first_lr=('lr','first'),last_lr=('lr','last'),mean_gradient_norm=('grad_norm_before_clip','mean'),maximum_gradient_norm=('grad_norm_before_clip','max'),clipped_updates=('grad_norm_before_clip',lambda x:int(x.gt(1).sum())),mean_compute_seconds=('compute_seconds','mean'),mean_record_seconds=('record_seconds','mean')).reset_index())
    checks.append('All386 query images exposed; all23520 paired draws and2940 finite-gradient updates checked; anchor gradients alwayszero')

    coef=inputs.table('coefficients')
    embeddings=inputs.table('reference_embeddings')
    support=[];patch=[];affinities=[]
    endpoint_rows=p[p.stage.isin(['selected_eval','final_eval','diagnostics'])]
    needed=set(endpoint_rows['sample'])
    for epoch in [7,60]:
        for family,collector in [('patch_support',support),('patch_evidence',patch),('affinities',affinities)]:
            for chunk in pd.read_csv(inputs.file(f'csv/epoch_{epoch:04d}/{family}.csv'),chunksize=200000):
                selected=chunk[chunk['sample'].isin(needed)]
                if len(selected): collector.append(selected)
    a=pd.concat(affinities,ignore_index=True);alpha=pd.concat(support,ignore_index=True);patch=pd.concat(patch,ignore_index=True)
    assert len(a)==1442*32*18 and len(alpha)==1442*32 and len(patch)==1442*32*3
    assert not a.duplicated(['sample','patch','prototype']).any() and alpha.groupby('sample').pooling_mass.sum().between(.999999,1.000001).all()
    joined=a.merge(alpha[['sample','patch','pooling_mass']],on=['sample','patch'],validate='many_to_one')
    joined['pooled_affinity']=joined.affinity*joined.pooling_mass
    pooled=joined.groupby(['sample','prototype']).pooled_affinity.sum().reset_index()
    sample_meta=endpoint_rows[['sample','endpoint','variant','label','image_id','parameter_update','prediction']]
    bank=inputs.table('bank_cases')
    assert len(bank)==18
    pooled=pooled.merge(sample_meta,on='sample',validate='many_to_one').merge(bank[['prototype_index','class_index','source_image_id','source_tyre_id']],left_on='prototype',right_on='prototype_index',validate='many_to_one').rename(columns={'class_index':'reference_class'})
    save('pooled_reference_affinities',pooled)
    contribution=[]; logit_check=[];weight_rows=[];embedding_rows=[];concentration=[]
    for endpoint,epoch,update in [('selected',7,343),('final',60,2940)]:
        weights=coef[(coef.parameter_update==update)&(coef.kind=='signed_weight')].pivot(index='class_index',columns='prototype',values='value').to_numpy()
        bias=coef[(coef.parameter_update==update)&(coef.kind=='bias')].sort_values('class_index').value.to_numpy()
        temperature=float(coef[(coef.parameter_update==update)&(coef.kind=='temperature')].value.iloc[0])
        assert weights.shape==(3,18) and bias.shape==(3,)
        end=endpoint_rows[endpoint_rows.endpoint==endpoint].sort_values('sample')
        sample_ids=end['sample'].to_numpy()
        selected=pooled[pooled.endpoint==endpoint].pivot(index='sample',columns='prototype',values='pooled_affinity').loc[sample_ids].to_numpy()
        calculated=selected@weights.T+bias
        actual_logits=end[[f'logit_{c}' for c in range(3)]].to_numpy()
        residual=np.abs(calculated-actual_logits).max()
        assert residual<2e-6,residual
        for row,calc in zip(end.itertuples(index=False),calculated):
            for c in range(3):
                logit_check.append(dict(sample=row.sample,endpoint=endpoint,variant=row.variant,image_id=row.image_id,class_index=c,bias=float(bias[c]),reference_sum=float(calc[c]-bias[c]),reconstructed_logit=float(calc[c]),saved_logit=float(getattr(row,f'logit_{c}')),residual=abs(float(calc[c]-getattr(row,f'logit_{c}')))))
        for c in range(3):
            for ref in range(18):
                weight_rows.append(dict(endpoint=endpoint,epoch=epoch,class_index=c,prototype=ref,reference_class=int(bank.set_index('prototype_index').loc[ref,'class_index']),coefficient=float(weights[c,ref]),bias=float(bias[c]),temperature=temperature))
        for c in range(3):
            values=pooled[pooled.endpoint==endpoint].copy()
            values['class_index']=c
            values['signed_contribution']=values.pooled_affinity*values.prototype.map(dict(enumerate(weights[c])))
            contribution.append(values[['sample','endpoint','variant','image_id','label','prediction','prototype','reference_class','source_image_id','source_tyre_id','class_index','pooled_affinity','signed_contribution']])
        emb=embeddings[(embeddings.epoch==epoch)&(embeddings.endpoint==endpoint)].pivot(index='prototype',columns='dimension',values='value').to_numpy()
        assert emb.shape==(18,96) and np.allclose(np.linalg.norm(emb,axis=1),1,atol=2e-6)
        cos=emb@emb.T
        for i in range(18):
            for j in range(18):
                embedding_rows.append(dict(endpoint=endpoint,prototype_a=i,prototype_b=j,class_a=int(bank.set_index('prototype_index').loc[i,'class_index']),class_b=int(bank.set_index('prototype_index').loc[j,'class_index']),cosine=float(cos[i,j])))
        # Verify every retained affinity against cosine and the saved temperature.
        end_a=a[a['sample'].isin(sample_ids)]
        assert np.allclose(np.exp((end_a.cosine-1)/temperature),end_a.affinity,atol=3e-6,rtol=2e-5)
        end_patch=patch[patch['sample'].isin(sample_ids)]
        assert np.allclose(end_patch.groupby(['sample','class_index']).contribution.sum().unstack().loc[sample_ids].to_numpy()+bias,actual_logits,atol=2e-6)
        # Pooling mass is an observed geometric weight, not the contribution magnitude.
        for row in end.itertuples(index=False):
            tiles=end_patch[(end_patch['sample']==row.sample)&(end_patch.class_index==row.prediction)].sort_values('patch')
            v=tiles.contribution.to_numpy();mass=np.abs(v).sum()
            positive=v.clip(0).sum()
            concentration.append(dict(sample=row.sample,endpoint=endpoint,variant=row.variant,image_id=row.image_id,label=row.label,prediction=row.prediction,
                top_signed_patch=int(v.argmax()),top_absolute_share=float(np.abs(v).max()/mass) if mass else 0,
                top_positive_share=float(v.max()/positive) if positive else None,signed_sum=float(v.sum()),absolute_mass=float(mass),
                positive_patches=int((v>0).sum()),negative_patches=int((v<0).sum())))
    contributions=save('reference_contributions',pd.concat(contribution,ignore_index=True))
    save('logit_reconstruction',logit_check);save('endpoint_coefficients',weight_rows)
    save('bank_embedding_cosines',embedding_rows);save('patch_concentration',concentration)
    save('endpoint_patch_evidence',patch.merge(sample_meta,on='sample',validate='many_to_one'))
    save('endpoint_patch_support',alpha.merge(sample_meta,on='sample',validate='many_to_one'))
    nearest=pooled.sort_values(['sample','pooled_affinity','prototype'],ascending=[True,False,True]).groupby('sample').head(1)
    save('nearest_reference',nearest)
    affinity_groups=pooled.groupby(['endpoint','variant','label','reference_class']).agg(images=('sample','nunique'),mean_affinity=('pooled_affinity','mean'),min_affinity=('pooled_affinity','min'),max_affinity=('pooled_affinity','max')).reset_index()
    save('reference_class_affinities',affinity_groups)
    saved_refs=pd.read_csv(inputs.file('report/REFERENCE_CASES.csv'))
    compare=contributions[contributions.variant=='identity'].merge(saved_refs,on=['sample','endpoint','image_id','class_index','prototype'],validate='one_to_one')
    assert len(compare)==206*3*18 and np.allclose(compare.signed_contribution,compare.contribution,atol=2e-6)
    occlusion=pairs[pairs.variant=='occlude_top_patch'].merge(pd.DataFrame(concentration).query("variant == 'identity'")[['sample','top_signed_patch']],left_on='base_sample',right_on='sample',validate='one_to_one')
    assert occlusion.occluded_patch.eq(occlusion.top_signed_patch).all()
    save('occlusion_selection_check',occlusion)
    checks.append('All1442 endpoint/probe logits reconstructed from saved576 affinities, geometric pooling,18 reference weights andbias; all32patch sums andocclusion indices checked')

    regions=inputs.table('dense_regions').merge(source[['image_index','role']],on='image_index',validate='many_to_one')
    points=inputs.table('dense_points').merge(source[['image_index','role']],on='image_index',validate='many_to_one')
    save('dense_region_by_role',regions.groupby(['role','region']).agg(images=('image_index','nunique'),mean_dice=('dice','mean'),mean_iou=('iou','mean'),mean_box_iou=('box_iou','mean')).reset_index())
    save('dense_points_by_role',points.groupby(['role','point_kind','metric_available'],dropna=False).agg(points=('image_index','size'),images=('image_index','nunique'),mae_width_fraction=('absolute_error_width_fraction','mean')).reset_index())
    calibration=[];risk=[]
    for endpoint,g in identity.groupby('endpoint'):
        for bucket in range(10):
            group=g[np.minimum((g.confidence*10).astype(int),9)==bucket]
            calibration.append(dict(endpoint=endpoint,bin=bucket,n=len(group),accuracy=float(group.correct.mean()) if len(group) else None,mean_confidence=float(group.confidence.mean()) if len(group) else None))
        for threshold in np.arange(.0,1.001,.05):
            group=g[g.confidence>=threshold]
            risk.append(dict(endpoint=endpoint,threshold=float(threshold),retained_images=len(group),coverage=len(group)/len(g),error_rate=float(1-group.correct.mean()) if len(group) else None,scope='Descriptive already-inspected endpoint; not a deployment rule'))
    save('endpoint_calibration',calibration);save('confidence_risk',risk)
    from paired_review import extend
    extend(inputs,out,p,source,exposures,updates,coef,save,checks)
    save('input_inventory',[dict(name=name,**meta) for name,meta in sorted(inputs.used.items())])
    def units(column):
        if 'seconds' in column: return 'seconds'
        if 'bytes' in column: return 'bytes'
        if any(x in column for x in ('logit','margin','bias','signed_contribution','reference_sum','residual','signed_sum','absolute_mass')): return 'class-logit units'
        if 'cosine' in column: return 'dimensionless [-1,1]'
        if 'nll' in column or 'entropy' in column: return 'natural-log units'
        if any(x in column for x in ('probability','accuracy','recall','confidence','affinity','coverage','share','mass','dice','iou','mae_width_fraction')): return 'dimensionless fraction; see METHODS.md for denominator'
        return 'identifier/count/scalar as named; see METHODS.md and original schema'
    save('table_dictionary',[dict(table=table,column=column,definition='Post-hoc '+column.replace('_',' '),units=units(column),missing='Blank is unavailable/not applicable; absent class recall is not zero') for table,meta in tables.items() for column in meta['columns']])
    result=dict(status='passed',source_revision=REVISION,source_protocol=PROTOCOL,source_manifest_sha256=inputs.manifest_sha,
        source_prediction_rows=53342,completed_optimizer_updates=2940,selected_epoch=7,final_epoch=60,
        model_forward_calls=0,training_updates_performed=0,new_independent_tyres=0,remote_writes=0,
        all_large_source_families_recounted=False,
        input_files=len(inputs.used),input_bytes=sum(v['bytes'] for v in inputs.used.values()),
        csv_tables=tables,checks=checks,environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__),
        scope='Post-hoc existing-cohort development analysis; no new checkpoint selection or causal wear/safety claim')
    atomic(out/'ANALYSIS.json',result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',required=True)
    parser.add_argument('--manifest')
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    inputs=Inputs(args.cache,args.manifest)
    result=audit(inputs,args.out)
    from render_lc_review import render
    render(Path(args.out),inputs)
    atomic(Path(args.cache)/'REVIEW_READ_STATUS.json',dict(public_read_requests_this_invocation=inputs.reads,remote_writes=0))
    print(json.dumps({k:v for k,v in result.items() if k!='csv_tables'},indent=2))


if __name__=='__main__': main()
