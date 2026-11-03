"""Post-hoc ATLAS evidence audit. No Torch, model forward pass or fitting.

Inputs are immutable NB13 objects at a pinned HF commit. Local caches are
rechecked, partial downloads are atomic, and derived CSVs never alter NB13.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REVISION = 'a44418844ba3f1c98adee2cae76a8bbf9f5f4e9e'
PROTOCOL = '36f473b0886d6c2f49c416436d270198af3982230e3045da9969b0bf42169625'
PREFIX = 'phase3/tread-two/atlas-context-r3/fit/' + PROTOCOL
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
        self.lock=threading.Lock();self.tls=threading.local()
        self.reads = 0
        path = Path(manifest) if manifest else self.cache/'REMOTE_MANIFEST.json'
        if not path.exists():
            body = self.request(BASE+'MANIFEST.json')
            path.write_bytes(body)
        self.manifest = json.loads(path.read_text(encoding='utf-8'))
        assert self.manifest['contract'] == PROTOCOL and self.manifest['format']=='immutable-direct-files-v1'
        self.manifest_sha = sha(path)
        self.used = {}

    def request(self, url, byte_range=None):
        for attempt in range(8):
            with self.lock:self.reads += 1
            try:
                if not hasattr(self.tls,'session'):
                    self.tls.session=requests.Session();self.tls.session.trust_env=False
                headers={'Range':f'bytes={byte_range[0]}-{byte_range[1]}'} if byte_range else None
                response = self.tls.session.get(url,timeout=(20,150),headers=headers,stream=True)
                if response.status_code == 429 or response.status_code >= 500:
                    value = response.headers.get('Retry-After','0')
                    try: delay = float(value)
                    except ValueError:
                        from email.utils import parsedate_to_datetime
                        delay = max(0,parsedate_to_datetime(value).timestamp()-time.time())
                    reset = re.search(r';t=(\d+)',response.headers.get('RateLimit',''))
                    delay = max(delay,float(reset[1]) if reset and response.status_code==429 else 0,min(60,2**attempt))
                    if attempt==7: response.raise_for_status()
                    response.close()
                    print('Public evidence read delayed by HF; verified local files retained.',flush=True)
                    deadline=time.monotonic()+delay
                    while time.monotonic()<deadline: time.sleep(min(10,max(0,deadline-time.monotonic())))
                    continue
                response.raise_for_status()
                try:
                    if byte_range:
                        start,end,total=byte_range
                        assert response.status_code==206 and response.headers.get('Content-Range')==f'bytes {start}-{end}/{total}','Invalid HF byte-range response'
                    return response.content
                finally:response.close()
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
            target.parent.mkdir(parents=True,exist_ok=True)
            tmp=target.with_name(target.name+'.'+uuid.uuid4().hex+'.part')
            if meta['bytes']>1024**2:
                print('Reading bounded byte ranges:',name,'bytes:',meta['bytes'],flush=True)
                def block(start):
                    end=min(meta['bytes']-1,start+1024**2-1)
                    body=self.request(BASE+meta['object']+f'?download=true&range_start={start}&range_end={end}',(start,end,meta['bytes']))
                    assert len(body)==end-start+1
                    return body
                with tmp.open('wb') as stream,ThreadPoolExecutor(max_workers=4) as pool:
                    for body in pool.map(block,range(0,meta['bytes'],1024**2)):stream.write(body)
            else:tmp.write_bytes(self.request(BASE+meta['object']))
            assert tmp.stat().st_size==meta['bytes'] and sha(tmp)==meta['sha256'],name
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
        print('Saved review table:',name,'rows:',len(frame),flush=True)
        return frame

    cursor=json.loads(inputs.file('CURSOR.json').read_text())
    assert cursor['phase']==cursor['status']=='completed' and cursor['updates']==2940
    p=inputs.table('predictions')
    source=inputs.table('source_images').fillna('')
    assert len(p)==53754 and len(source)==570 and p['sample'].is_unique
    assert sorted(p['sample'])==list(range(53754))
    assert p.groupby('stage').size().to_dict()==dict(train=47040,validate=4860,selected_eval=103,final_eval=103,diagnostics=1648)
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
    checks.append('All 53754 prediction IDs, roles, probabilities, raw argmax, correctness, NLL and Brier checked')

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
    assert math.isclose(float(best.validation_ba_observed),float(selection.validation_ba_observed.max()))
    save('selection_audit',selection)
    checks.append('All saved cohort accuracy, BA, NLL, Brier and ECE rows recomputed; validation selection epoch7 checked')

    group_rows=[]
    for key,g in p[p.stage.isin(['selected_eval','final_eval','diagnostics'])].groupby(['stage','endpoint','variant','label','physical_tyre_id']):
        group_rows.append(dict(zip(['stage','endpoint','variant','label','physical_tyre_id'],key),**scores(g)))
    save('endpoint_tyre_metrics',group_rows)
    probes=[]
    identity=p[p.stage.isin(['selected_eval','final_eval'])]
    assert identity.groupby('endpoint').size().eq(103).all()
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
    checks.append('All1648 diagnostic rows paired by image/checkpoint against their identity endpoint')

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
    assert len(gradients)==144060 and gradients.finite.eq(1).all() and gradients.groupby('update').size().eq(49).all()
    save('gradient_parameters',gradients.groupby('parameter').agg(updates=('update','size'),mean_norm=('norm','mean'),maximum_norm=('norm','max'),zero_norm_updates=('norm',lambda x:int(x.eq(0).sum()))).reset_index())
    save('optimizer_epochs',updates.groupby('epoch').agg(updates=('update','size'),mean_loss=('loss','mean'),first_lr=('lr','first'),last_lr=('lr','last'),mean_gradient_norm=('grad_norm_before_clip','mean'),maximum_gradient_norm=('grad_norm_before_clip','max'),clipped_updates=('grad_norm_before_clip',lambda x:int(x.gt(1).sum())),mean_compute_seconds=('compute_seconds','mean'),mean_record_seconds=('record_seconds','mean')).reset_index())
    checks.append('All386 query images exposed; all23520 paired draws and2940 finite-gradient updates checked; anchor gradients alwayszero')

    coef=inputs.table('coefficients')
    embeddings=inputs.table('reference_embeddings')
    from context_checks import extend as context_audit
    ctx=context_audit(inputs,out,p,coef,save,checks)
    support=[];patch=[];affinities=[]
    endpoint_rows=p[p.stage.isin(['selected_eval','final_eval','diagnostics'])]
    needed=set(endpoint_rows['sample'])
    for epoch in [7,60]:
        for family,collector in [('patch_support',support),('patch_evidence',patch),('affinities',affinities)]:
            for chunk in pd.read_csv(inputs.file(f'csv/epoch_{epoch:04d}/{family}.csv'),chunksize=200000):
                selected=chunk[chunk['sample'].isin(needed)]
                if len(selected): collector.append(selected)
    a=pd.concat(affinities,ignore_index=True);alpha=pd.concat(support,ignore_index=True);patch=pd.concat(patch,ignore_index=True)
    assert len(a)==1854*32*18 and len(alpha)==1854*32 and len(patch)==1854*32*3
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
        context_delta=ctx['deltas'].groupby('sample')[[f'delta_{c}' for c in range(3)]].sum().loc[sample_ids].to_numpy()
        calculated=selected@weights.T+bias+context_delta
        actual_logits=end[[f'logit_{c}' for c in range(3)]].to_numpy()
        residual=np.abs(calculated-actual_logits).max()
        assert residual<2e-6,residual
        for row,calc,delta in zip(end.itertuples(index=False),calculated,context_delta):
            for c in range(3):
                logit_check.append(dict(sample=row.sample,endpoint=endpoint,variant=row.variant,image_id=row.image_id,class_index=c,bias=float(bias[c]),reference_sum=float(calc[c]-bias[c]-delta[c]),context_spatial_sum=float(delta[c]),reconstructed_logit=float(calc[c]),saved_logit=float(getattr(row,f'logit_{c}')),residual=abs(float(calc[c]-getattr(row,f'logit_{c}')))))
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
    from context_checks import patch_accounting
    patch_accounting(a,alpha,patch,coef,ctx,save,checks)
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
    checks.append('All1854 endpoint/probe logits reconstructed from saved576 affinities, geometric pooling,18 frozen reference weights,context correction andcombined bias; all32patch sums andocclusion indices checked')

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
    from paired_context import extend
    extend(inputs,out,p,source,exposures,updates,coef,save,checks)
    save('input_inventory',[dict(name=name,**meta) for name,meta in sorted(inputs.used.items())])
    def units(column):
        if 'seconds' in column: return 'seconds'
        if 'bytes' in column or column in ('peak_cuda_allocated','peak_cuda_reserved','peak_host_rss','minimum_free_disk'): return 'bytes'
        if 'correction_l2' in column or column in ('residual_l2','penalty_reconstruction_residual'): return 'squared class-logit units'
        if column.startswith('entropy_') or 'attention' in column or column in ('normalized_entropy','mean_entropy','maximum_entropy_error','max_residual','maximum_row_sum_error','row_sum_max_residual'): return 'dimensionless normalized attention statistic'
        if column in ('loss_residual','ce_a','ce_b','js','total','reconstructed_total','mean_loss'): return 'joint objective scalar; CE/JS use natural logs; see METHODS.md'
        if 'gradient_norm' in column or column in ('mean_norm','maximum_norm'): return 'L2 norm of parameter gradient in model coordinates'
        if 'cosine_distance' in column or column=='texture': return 'dimensionless cosine distance [0,2]'
        if 'lr' in column: return 'optimizer learning-rate scalar'
        if 'nll' in column or column=='entropy': return 'natural-log units'
        if re.fullmatch(r'(spatial_delta_|total_delta_|delta_)\d',column) or column in ('mean_delta_low','mean_delta_mid','mean_delta_high'): return 'class-logit units'
        if any(x in column for x in ('logit','margin','bias','contribution','reference_sum','residual','signed_sum','absolute_mass','context_spatial_sum','coefficient')): return 'class-logit units'
        if 'cosine' in column: return 'dimensionless [-1,1]'
        if 'brier' in column: return 'dimensionless three-class squared probability error [0,2]'
        if any(x in column for x in ('probability','accuracy','recall','confidence','affinity','coverage','share','mass','dice','iou','ece','error_rate','disagreement')) or column in ('valid_fraction','clipped_fraction'): return 'dimensionless fraction; see METHODS.md for denominator'
        if column in ('correct','base_correct','probe_correct','flip','prediction_flip','disagree','selected','empty_support','metric_available','is_bank','is_query','prediction_changed_from_base'): return 'binary 0/1'
        if any(x in column for x in ('brightness','contrast','gamma','scale','temperature')): return 'dimensionless scalar'
        if column.startswith('replacement_'): return 'RGB input intensity [0,1]'
        if column=='mae_width_fraction': return 'absolute error / original image width'
        if column in ('run','stage','endpoint','variant','domain','role','point_kind','scope','tie_break','name','parameter','checkpoint_artifact','image_path') or column.endswith(('_id','sha256')): return 'text identifier/category/path/hash'
        return 'integer index/count; expected_draws and actual_minus_expected may be fractional'
    definitions={
        'sample':'Unique saved observation key, 0..53753; join raw families by sample, not by image alone',
        'epoch':'Training epoch, one-based; stage distinguishes sampled training from fixed evaluation',
        'update':'Saved completed-update cursor; parameter_update identifies the weights used for this observation',
        'parameter_update':'Optimizer updates already applied to the weights used for the prediction; online rows precede their current batch update',
        'stage':'Saved phase: train, validate, selected_eval, final_eval or diagnostics',
        'endpoint':'Checkpoint role: selected epoch7, final epoch60, or the original saved phase role',
        'variant':'Identity or saved appearance/occlusion/token-layout intervention; same image/checkpoint pairing',
        'label':'Ground-truth mileage-proxy class: 0 low, 1 mid, 2 high; not physical wear/depth/safety',
        'prediction':'Argmax of combined saved three-class logits; 0 low, 1 mid, 2 high',
        'class_index':'Explained or grouped class: 0 low, 1 mid, 2 high',
        'physical_tyre_id':'Known original tyre identity where available; missing/other source groups are not invented identities',
        'source_group':'Original source/acquisition grouping used by the frozen sampler',
        'source_patch':'Original row-major tile index after the saved token-layout mapping',
        'patch':'Row-major tile index 0..31 in eight bands by four columns',
        'layer':'Attention layer index 0 or 1', 'head':'Attention head index 0..3',
        'prototype':'Reference-bank entry index 0..17; reference class does not authorize target-conditioned filtering',
        'pooling_mass':'Geometric valid-support mass; nonnegative and sums to one over a nonempty observation',
        'valid_fraction':'Valid tread support fraction of this tile before normalized geometric pooling',
        'nll':'Mean negative natural log of true-class saved probability; clips probability at 1e-12',
        'brier':'Mean sum over all three classes of (saved probability minus one-hot target)^2',
        'balanced_accuracy_observed':'Mean recall over target classes actually present; missing mid class is omitted, never assigned zero recall',
        'accuracy':'Fraction of saved argmax predictions equal to targets in this row group',
        'ece_10_equal_width':'Ten equal-width max-confidence bins: sum of bin mass times absolute accuracy-confidence gap',
        'confidence':'Maximum of the three saved class probabilities',
        'true_margin':'True-class logit minus maximum competing-class logit',
        'draw_probability':'Frozen class/domain/sqrt-group/uniform-image probability of a paired query draw',
        'actual_draws':'Observed paired query draws, counting two correlated views once',
        'expected_draws':'23520 times frozen draw_probability; expectation, not observed count',
        'view_observations':'Two correlated observations per paired query draw',
        'correction_l2':'Mean over three classes of squared total spatial-plus-bias context correction',
        'residual_l2':'Mean of the two view correction_l2 values in this paired draw',
        'ce_a':'True-class cross entropy for view a from saved logits',
        'ce_b':'True-class cross entropy for view b from saved logits',
        'js':'Symmetric Jensen-Shannon divergence between the two saved view distributions',
        'texture':'Valid-pooling weighted cosine distance between paired contextual tile embeddings',
        'total':'Mean paired CE + 0.1*JS + 0.05*texture + 0.02*residual_l2',
        'reconstructed_total':'Independent recomputation of the declared paired joint objective',
        'normalized_entropy':'Attention row entropy/log(max(2,valid keys)), averaged over query rows by pooling mass',
        'weighted_max_attention':'Maximum key probability per query row, averaged by query pooling mass',
        'max_frozen_logit_change':'Maximum absolute prior logit change against same-image/checkpoint identity',
        'max_context_logit_change':'Maximum absolute total correction change against same-image/checkpoint identity',
        'region':'Frozen dense target: 0 tyre, 1 tread',
        'mae_width_fraction':'Mean absolute guide-point coordinate error divided by original image width; only metric_available targets',
        'threshold':'Descriptive confidence retention threshold; no deployment rule is selected',
        'coverage':'Retained endpoint images / all images at this endpoint',
        'error_rate':'Fraction incorrect among retained images; blank when none retained',
        'count':'Confusion cell count for the row target_class and predicted_class',
    }
    def definition(table,column):
        if column in definitions:return definitions[column]
        for suffix in ('_LC','_context','_v1','_r3'):
            if column.endswith(suffix):return definition(table,column[:-len(suffix)])+'; pinned '+suffix[1:]+' model'
        if column.startswith('delta_') and column[6:] in definitions:return definitions[column[6:]]+'; signed new-minus-reference difference (probe−identity, context−LC or r3−v1 as table declares)'
        if re.fullmatch(r'(base_|anchor_)?logit_\d',column):return 'Saved unnormalized class logit; base is frozen selected LC, anchor is the separate frozen geometry-runtime classifier, no prefix is combined context model'
        if column.startswith(('total_delta_','spatial_delta_','bias_delta_','delta_')):return 'Signed context contribution to the named class; total sums spatial and global bias, spatial sums 32 weighted tiles'
        if column.startswith(('entropy_','max_attention_')):return 'Saved normalized attention entropy or weighted maximum for the named zero-based layer/head; query rows weighted by geometric mass'
        if 'residual' in column or column.endswith('_error'):return 'Absolute independent reconstruction discrepancy for the named quantity, aggregated as stated; see this table calculation in METHODS.md'
        if column.startswith('mean_'):return 'Arithmetic group mean of '+definition(table,column[5:])
        if column.startswith('maximum_'):return 'Maximum within this table row group of '+column[8:].replace('_',' ')
        if column.startswith(('first_','last_','minimum_','peak_','min_','max_')):return column.replace('_',' ')+' within the saved row group; resources are observed measurements, not capacity guarantees'
        if 'sha256' in column:return 'SHA-256 of the named source/artifact bytes; retained provenance, never a credential'
        if column.endswith('_id') or column in ('image_index','source_group_draw','run','name','parameter','image_path','checkpoint_artifact','scope','role','domain','point_kind','tie_break'):return 'Original '+column.replace('_',' ')+' grouping/join value; preserve text and missingness exactly'
        return column.replace('_',' ')+' for the '+table.replace('_',' ')+' row group; named counts are observation/draw/image counts and remain distinct'
    save('table_dictionary',[dict(table=table,column=column,definition=definition(table,column),units=units(column),missing='Blank is unavailable/not applicable; absent class recall is not zero') for table,meta in tables.items() for column in meta['columns']])
    result=dict(status='passed',source_revision=REVISION,source_protocol=PROTOCOL,source_manifest_sha256=inputs.manifest_sha,
        source_prediction_rows=53754,completed_optimizer_updates=2940,selected_epoch=7,final_epoch=60,
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
    from render_context import render
    render(Path(args.out),inputs)
    atomic(Path(args.cache)/'REVIEW_READ_STATUS.json',dict(public_read_requests_this_invocation=inputs.reads,remote_writes=0))
    print(json.dumps({k:v for k,v in result.items() if k!='csv_tables'},indent=2))


if __name__=='__main__': main()
