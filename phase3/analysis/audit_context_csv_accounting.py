"""Independently reconstruct context logits, losses and attention from retained CSV."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'tread_two/context_validation'
proof=json.loads((OUT/'LOCAL_VALIDATION.json').read_text());SNAP=Path(proof['final_snapshot'])
def table(name):return pd.concat([pd.read_csv(p) for p in sorted((SNAP/'csv').glob(f'epoch_*/{name}.csv'))],ignore_index=True)
pred=table('predictions').set_index('sample');aff=table('affinities');support=table('patch_support');rules=table('case_rules');coef=table('coefficients')
context=table('context_evidence');summary=table('context_summary').set_index('sample');evidence=table('patch_evidence')
sources=table('source_images').set_index('image_index');bank=pd.read_csv(ROOT/'tread_two/context_pilot/assets/prototype_bank.csv')
maximum_logit=maximum_patch=maximum_correction=0.;blocked=0
for sample,r in pred.iterrows():
    a=aff[aff['sample']==sample].pivot(index='patch',columns='prototype',values='affinity').to_numpy()
    alpha=support[support['sample']==sample].sort_values('patch').pooling_mass.to_numpy()
    rule=rules[rules['sample']==sample].sort_values('prototype');allowed=rule.allowed.to_numpy();scale=rule.normalization_scale.to_numpy()
    same=(bank.source_image_sha256.to_numpy()==sources.loc[int(r.image_index)].image_sha256)
    assert np.array_equal(allowed,1-same.astype(int));blocked+=int(same.sum())
    counts=[int(allowed[c*6:(c+1)*6].sum()) for c in range(3)]
    assert counts==[int(r[f'allowed_{c}']) for c in ('low','mid','high')]
    assert np.allclose(scale,[6/counts[j//6] for j in range(18)])
    c=coef[coef.parameter_update==r.parameter_update]
    weights=c[c.kind=='signed_weight'].pivot(index='class_index',columns='prototype',values='value').to_numpy()
    bias=c[c.kind=='bias'].sort_values('class_index').value.to_numpy()
    delta=context[context['sample']==sample].sort_values('patch')
    assert sorted(delta.source_patch.tolist())==list(range(32))
    d=delta[[f'delta_{i}' for i in range(3)]].to_numpy()
    assert np.all(np.abs(d)<=2*alpha[:,None]+1e-7)
    sr=summary.loc[sample];bd=sr[[f'bias_delta_{c}' for c in range(3)]].to_numpy(dtype=float)
    assert np.all(np.abs(bd)<=.5+1e-7)
    patch=np.einsum('pr,cr,p,r,r->pc',a,weights,alpha,allowed,scale)+d
    retained=evidence[evidence['sample']==sample].pivot(index='patch',columns='class_index',values='contribution').to_numpy()
    pe=float(np.max(np.abs(patch-retained)));maximum_patch=max(maximum_patch,pe);assert pe<2e-6
    logits=patch.sum(0)+bias
    le=float(np.max(np.abs(logits-r[[f'logit_{i}' for i in range(3)]].to_numpy(dtype=float))))
    maximum_logit=max(maximum_logit,le);assert le<2e-6
    base=logits-d.sum(0)-bd
    assert np.max(np.abs(base-sr[[f'base_logit_{c}' for c in range(3)]].to_numpy(dtype=float)))<2e-6
    correction=d.sum(0)+bd
    ce=abs(float(np.mean(correction**2))-sr.correction_l2);maximum_correction=max(maximum_correction,ce);assert ce<2e-7
pairs=table('view_pairs');patches=table('view_patch_consistency');updates=table('train_updates');maximum_loss=0.
for r in pairs.itertuples():
    pa,pb=pred.loc[r.sample_a],pred.loc[r.sample_b]
    p=pa[[f'probability_{c}' for c in range(3)]].to_numpy(dtype=float);q=pb[[f'probability_{c}' for c in range(3)]].to_numpy(dtype=float);m=(p+q)/2
    js=.5*((p*np.log(p/m)).sum()+(q*np.log(q/m)).sum())
    rows=patches[patches.sample_a==r.sample_a];texture=float((rows.pooling_mass*rows.cosine_distance).sum())
    residual=.5*(summary.loc[r.sample_a].correction_l2+summary.loc[r.sample_b].correction_l2)
    total=.5*(pa.nll+pb.nll)+.1*js+.05*texture+.02*residual
    maximum_loss=max(maximum_loss,abs(total-r.total));assert abs(total-r.total)<4e-7
    assert abs(residual-r.residual_l2)<1e-7
for r in updates.itertuples():
    subset=pairs[pairs['update']==r.update];assert len(subset)==8 and abs(subset.total.mean()-r.loss)<4e-7
    assert r.trainable_parameters==231555 and r.anchor_gradient_count==0
attention=table('endpoint_attention');max_rowsum=0.;max_entropy=0.
for (sample,layer,head),rows in attention.groupby(['sample','layer','head']):
    matrix=rows.pivot(index='query',columns='key',values='value').to_numpy()
    assert matrix.shape==(32,32) and np.isfinite(matrix).all() and matrix.min()>=0
    error=float(np.max(np.abs(matrix.sum(-1)-1)));max_rowsum=max(max_rowsum,error);assert error<2e-7
    alpha=support[support['sample']==sample].sort_values('patch').pooling_mass.to_numpy()
    valid=support[support['sample']==sample].sort_values('patch').valid_fraction.to_numpy()>0
    normalizer=np.log(max(int(valid.sum()),2))
    entropy=-(matrix*np.log(np.maximum(matrix,1e-12))).sum(-1)/normalizer
    actual=float((entropy*alpha).sum());expected=summary.loc[sample,f'entropy_{layer}_{head}']
    error=abs(actual-expected);max_entropy=max(max_entropy,error);assert error<5e-7
result=dict(status='passed',model_forward_calls=0,optimizer_updates=0,observations=len(pred),logits_reconstructed=3*len(pred),
    maximum_logit_residual=maximum_logit,maximum_patch_residual=maximum_patch,maximum_correction_penalty_residual=maximum_correction,
    paired_draw_losses_reconstructed=len(pairs),update_losses_reconstructed=len(updates),maximum_pair_loss_residual=maximum_loss,
    attention_values_checked=len(attention),maximum_attention_rowsum_residual=max_rowsum,maximum_attention_entropy_residual=max_entropy,
    blocked_reference_observations_checked=blocked,all_SHA_permissions_and_6_over_n_scales_checked=True,
    scope='CSV-only disposable engineering evidence; no measured r3 accuracy claim')
(OUT/'CSV_ACCOUNTING.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
