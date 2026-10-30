"""Independent retained-row loss/logit reconstruction; no model forward or fit."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'tread_two/lc_validation'
proof=json.loads((OUT/'LOCAL_VALIDATION.json').read_text());SNAP=Path(proof['final_snapshot'])
def table(name):return pd.concat([pd.read_csv(p) for p in sorted((SNAP/'csv').glob(f'epoch_*/{name}.csv'))],ignore_index=True)
pred=table('predictions').set_index('sample');aff=table('affinities');support=table('patch_support');rules=table('case_rules');coef=table('coefficients')
sources=table('source_images').set_index('image_index');bank=pd.read_csv(ROOT/'tread_two/lc_pilot/assets/prototype_bank.csv')
max_error=0.;blocked=0
for sample,r in pred.iterrows():
    a=aff[aff['sample']==sample].pivot(index='patch',columns='prototype',values='affinity').to_numpy()
    alpha=support[support['sample']==sample].sort_values('patch').pooling_mass.to_numpy()
    rule=rules[rules['sample']==sample].sort_values('prototype');allowed=rule.allowed.to_numpy();scale=rule.normalization_scale.to_numpy()
    same=(bank.source_image_sha256.to_numpy()==sources.loc[int(r.image_index)].image_sha256)
    assert np.array_equal(allowed,1-same.astype(int));blocked+=int(same.sum())
    actual_counts=[int(allowed[c*6:(c+1)*6].sum()) for c in range(3)]
    assert actual_counts==[int(r[f'allowed_{c}']) for c in ('low','mid','high')]
    assert np.allclose(scale,[6/actual_counts[j//6] for j in range(18)])
    c=coef[coef.parameter_update==r.parameter_update]
    weights=c[c.kind=='signed_weight'].pivot(index='class_index',columns='prototype',values='value').to_numpy()
    bias=c[c.kind=='bias'].sort_values('class_index').value.to_numpy()
    logits=np.einsum('pr,cr,p,r,r->c',a,weights,alpha,allowed,scale)+bias
    error=float(np.max(np.abs(logits-r[[f'logit_{i}' for i in range(3)]].to_numpy(dtype=float))))
    max_error=max(max_error,error);assert error<2e-6
pairs=table('view_pairs');patch=table('view_patch_consistency');updates=table('train_updates');max_loss=0.
for r in pairs.itertuples():
    pa=pred.loc[r.sample_a];pb=pred.loc[r.sample_b]
    p=pa[[f'probability_{c}' for c in range(3)]].to_numpy(dtype=float);q=pb[[f'probability_{c}' for c in range(3)]].to_numpy(dtype=float);m=(p+q)/2
    js=.5*((p*np.log(p/m)).sum()+(q*np.log(q/m)).sum())
    rows=patch[patch.sample_a==r.sample_a];texture=float((rows.pooling_mass*rows.cosine_distance).sum())
    total=.5*(pa.nll+pb.nll)+.1*js+.05*texture
    max_loss=max(max_loss,abs(total-r.total));assert abs(total-r.total)<3e-7
    assert abs(r.ce_a-pa.nll)<1e-7 and abs(r.ce_b-pb.nll)<1e-7 and abs(js-r.js)<1e-7 and abs(texture-r.texture)<1e-7
for r in updates.itertuples():
    subset=pairs[pairs['update']==r.update];assert len(subset)==8 and abs(subset.total.mean()-r.loss)<3e-7
result=dict(status='passed',model_forward_calls=0,optimizer_updates=0,observations=len(pred),logits_reconstructed=len(pred)*3,
  maximum_logit_residual=max_error,paired_draw_losses_reconstructed=len(pairs),update_losses_reconstructed=len(updates),
  maximum_pair_loss_residual=max_loss,blocked_reference_observations_checked=blocked,all_SHA_permissions_and_6_over_n_scales_checked=True,
  scope='Independent CSV-only engineering accounting; no measured r2 accuracy claim')
(OUT/'CSV_ACCOUNTING.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
