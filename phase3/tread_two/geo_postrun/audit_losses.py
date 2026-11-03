"""Independent per-update objective and raw masked-target accounting."""
import numpy as np
from common import atomic

def run(saved,core):
    tables=core['tables'];obs=core['obs'];geometry=core['geometry'].set_index('sample');quality=core['quality'];patches=core['patches'];z=core['projected'];batches=saved.table('batch_objectives')
    q=quality[quality.target_available==1].groupby('sample')[['loss_sum','loss_count']].sum()
    patch=patches.groupby('sample')[['huber_sum','huber_weight_count']].sum()
    rows=[]
    train=obs[obs.stage=='train']
    by_update={int(update):frame for update,frame in train.groupby('completed_update')}
    for row in batches.itertuples():
        frame=by_update[int(row.update)];ids=frame['sample'].to_numpy(dtype='int64');assert len(ids)==8
        g=geometry.loc[ids]
        def ratio(a,b):return float(g[a].sum()/max(g[b].sum(),1e-12))
        geo=ratio('bce_sum','bce_count')+ratio('dice_loss_sum','dice_region_count')+2*ratio('point_loss_sum','point_weight_count')+.1*ratio('containment_sum','containment_count')+.1*ratio('row_agreement_sum','row_agreement_count')+.1*(ratio('anchor_mask_sum','anchor_mask_count')+ratio('anchor_point_sum','anchor_point_count'))
        qs=q.reindex(ids).fillna(0);qloss=float(qs.loss_sum.sum()/max(qs.loss_count.sum(),1))
        valid=frame[frame.SSL_valid==1]['sample'].to_numpy(dtype='int64')
        if bool(row.texture_step_skipped):texture=0.;inv=var=cov=0.
        else:
            a,b=z[valid,0].astype('float64'),z[valid,1].astype('float64');inv=float(((a-b)**2).mean());var=cov=0.
            for values in (a,b):
                cv=np.cov(values,rowvar=False,ddof=1);var+=float(np.maximum(0,1-np.sqrt(np.diag(cv)+1e-4)).mean());cov+=float(((cv**2).sum()-(np.diag(cv)**2).sum())/64)
            p=patch.loc[valid];masked=float(p.huber_sum.sum()/p.huber_weight_count.sum());texture=25*inv+25*var+cov+.5*masked
        total=geo+.25*qloss+row.texture_ramp*texture
        checks=dict(geometry=abs(geo-row.geometry_objective),quality=abs(qloss-row.quality_objective),SSL=abs(texture-row.SSL_objective),total=abs(total-row.total_loss))
        for actual,expected in [(geo,row.geometry_objective),(qloss,row.quality_objective),(texture,row.SSL_objective),(total,row.total_loss)]:assert np.isclose(actual,expected,atol=5e-5,rtol=2e-5),(row.update,actual,expected)
        rows.append(dict(update=int(row.update),epoch=int(row.epoch),geometry_recomputed=geo,quality_recomputed=qloss,SSL_recomputed=texture,total_recomputed=total,geometry_difference=checks['geometry'],quality_difference=checks['quality'],SSL_difference=checks['SSL'],total_difference=checks['total'],passed=True))
    assert len(rows)==2940
    tables.write('loss_reconstruction',rows)
    return dict(updates=2940,maximum_total_difference=max(r['total_difference'] for r in rows),all_passed=True)

def raw_masked(saved,core,index):
    """Recompute every retained masked patch directly from saved native bodies."""
    tables=core['tables'];plan=saved.table('masked_patch_plan');metrics=core['patches'];count=0;maximum=0.;records=[]
    # Logical native keys preserve packed-slot identity; probes explicitly have no raw texture.
    retained=plan[plan.native_prediction_key.notna()].copy()
    retained['packed_slot']=retained.native_prediction_key.str.rsplit(':',n=1).str[-1].astype('int64')
    mapping=retained.set_index(['sample','view','packed_slot'])['patch'].to_dict()
    reported=metrics.set_index(['sample','view','patch']);mean=np.load(saved.root/'controls/mean.npy',allow_pickle=False).astype('float64')
    for view in (0,1):
        predictions=index[index.kind==f'v{view}_prediction']
        targets=index[index.kind==f'v{view}_target'].set_index('first_sample');weights=index[index.kind==f'v{view}_weight'].set_index('first_sample')
        for row in predictions.itertuples():
            first,last=int(row.first_sample),int(row.last_sample);pred=np.load(saved.root/row.file,allow_pickle=False).astype('float64')
            target=np.load(saved.root/targets.loc[first,'file'],allow_pickle=False).astype('float64');weight=np.load(saved.root/weights.loc[first,'file'],allow_pickle=False).astype('float64')
            error=np.abs(pred-target);huber=np.where(error<.1,error**2/.2,error-.05);sums=(huber*weight).sum((-3,-2,-1));counts=weight.sum((-3,-2,-1))*4
            abs_sums=(error*weight).sum((-3,-2,-1));maxima=error.max((-3,-2,-1));assert len(pred)==last-first+1
            localmax=0.;localcount=0
            for i in range(len(pred)):
                for slot in range(8):
                    key=(first+i,view,slot)
                    if key not in mapping:assert not np.any(pred[i,slot]) and not np.any(weight[i,slot]);continue
                    patch=int(mapping[key]);metric=reported.loc[(first+i,view,patch)];difference=abs(sums[i,slot]-metric.huber_sum)
                    assert np.isclose(sums[i,slot],metric.huber_sum,atol=1e-8,rtol=1e-9)
                    assert counts[i,slot]==metric.huber_weight_count
                    assert np.isclose(abs_sums[i,slot],metric.absolute_error_sum,atol=1e-8,rtol=1e-9)
                    assert np.isclose(maxima[i,slot],metric.absolute_error_max,atol=1e-8,rtol=1e-9)
                    if np.isfinite(metric.mean_predictor_huber_sum):
                        e=np.abs(mean[patch]-target[i,slot]);mean_sum=(np.where(e<.1,e**2/.2,e-.05)*weight[i,slot]).sum()
                        assert np.isclose(mean_sum,metric.mean_predictor_huber_sum,atol=1e-8,rtol=1e-9)
                    localmax=max(localmax,difference);localcount+=1
            count+=localcount;maximum=max(maximum,localmax);records.append(dict(file=row.file,view=view,first_sample=first,last_sample=last,masked_patches=localcount,maximum_huber_difference=localmax,passed=True))
            if len(records)%100==0:print('Raw masked-tensor audit:',len(records),'shards,',count,'patches',flush=True)
    assert count==len(retained)
    tables.write('raw_masked_reconstruction',records)
    return dict(masked_patches=count,shards=len(records),maximum_difference=maximum,all_passed=True,probe_texture_not_retained_by_declared_policy=True)
