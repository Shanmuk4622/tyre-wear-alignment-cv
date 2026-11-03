"""All observation families with exact sufficient counts and full raw native outputs."""
import math,os,shutil
from collections import defaultdict
import numpy as np
import torch
from torch.nn import functional as F
from common import cpu,digest,environment,sha
from objective_geo import coarse_truth,full_geometry,geometry_loss,quality_loss
from geometry import patches

def number(x):return float(x.detach()) if torch.is_tensor(x) else float(x)
def array(x):return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
def add(rows,family,**record):rows.setdefault(family,[]).append(record)

def source_records(root,overlay):
    rows={}
    for r in overlay['rows']:
        add(rows,'source_images',image_id=r['image_id'],image_sha256=r['image_sha256'],role=r['role'],domain=r['domain'],
            confirmed_tyre_id=r['physical_tyre_id'] or None,source_group=r['physical_tyre_id'] or r['source_video'],
            leakage_component=r['component'],label_proxy=['low','mid','high'][r['class_index']],point_kind=r['point_kind'],
            human_target_available=r['point_kind']=='human',weak_target_available=r['point_kind']=='polygon_weak',
            mask_sha256=digest([sha(root/r['tyre_mask']),sha(root/r['tread_mask'])]),ignore_sha256=sha(root/r['ignore_mask']))
        for j,x in enumerate(r['points'] or [None]*6):
            add(rows,'annotation_targets',image_id=r['image_id'],guide=j//2,side=j%2,x_width_fraction=x,y_height_fraction=r['guide_fractions'][j//2],
                target_kind=r['point_kind'],optimization_weight=r['point_weight'],human_eligible=r['point_kind']=='human',metric_available=x is not None)
    return rows

def box(mask):
    ys,xs=np.where(mask)
    return None if not len(xs) else [int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1]
def box_iou(a,b):
    if a is None or b is None:return None
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union else None

def latent(z):
    z=np.asarray(z,dtype=np.float64)
    if len(z)<2:return dict(std_min=None,std_median=None,below=None,covariance=None,rank=0.)
    cv=np.cov(z,rowvar=False,ddof=1);ev=np.linalg.eigvalsh(cv).clip(0);total=ev.sum();p=ev[ev>0]/total if total>0 else []
    rank=float(np.exp(-np.sum(p*np.log(p)))) if len(p) else 0.
    std=np.sqrt(np.maximum(0,np.diag(cv)))
    return dict(std_min=float(std.min()),std_median=float(np.median(std)),below=int((std<.1).sum()),
                covariance=float((cv**2).sum()-(np.diag(cv)**2).sum()),rank=rank)

@torch.no_grad()
def observe(out,base,batch,seeds,first,stage,endpoint,epoch,update,mean=None,dense=False,median=None):
    records={};n=len(batch['rows']);ref=out['refined'];metrics=full_geometry(ref,batch)
    y,valid=coarse_truth(batch['masks']);p=ref['masks'];q=ref['points']
    arr={k:array(v) for k,v in ref.items()};truth=array(batch['masks']);valid0=truth[:,2]<.5
    target=array(batch['points']);weights=array(batch['point_weight']);guides=array(batch['guides'])
    atlas={k:array(out['atlas'][k]) for k in ('row_parameters','row_valid','coverage','mass')}
    native=dict(geometry_logits=arr['seg_logits'],row_parameters=atlas['row_parameters'],row_valid=atlas['row_valid'],points=arr['points'],quality=array(out['quality']))
    _,quality=quality_loss(out['quality'],ref,batch);qt=array(quality['targets']);qa=array(quality['available']);qp=array(out['quality'])
    m={k:array(v) for k,v in metrics.items()};views=out['views'];eligible=array(views[0]['eligible']&views[1]['eligible']).astype(bool)
    ignored=patches(out['atlas']['ignored']);target_fraction=array((patches(out['atlas']['valid'])*(1-ignored)).mean((-3,-2,-1)))
    info=[]
    for i,row in enumerate(batch['rows']):
        sample=first+i;geometry_valid=bool(atlas['row_valid'][i].any())
        add(records,'sample_observations',sample=sample,stage=stage,endpoint=endpoint,epoch=epoch,completed_update=update+1 if stage=='train' else update,
            parameter_update=update,image_id=row['image_id'],source_sha256=row['image_sha256'],geometry_valid=geometry_valid,
            SSL_valid=bool(eligible[i]),SSL_skip_reason='' if eligible[i] else 'predicted_support_or_masked_target_unavailable',
            guide_count=3,mask_valid_pixels=int(valid0[i].sum()),point_target_kind=row['point_kind'])
        subbatch={k:(v[i:i+1] if torch.is_tensor(v) else v) for k,v in batch.items()}
        subref={k:v[i:i+1] for k,v in ref.items()};subbase={k:v[i:i+1] for k,v in base.items()}
        _,gs=geometry_loss(subref,subbase,subbatch)
        mask_sum=number(((p[i:i+1]-base['anchor_logits'][i:i+1].sigmoid()).square()*valid[i:i+1]).sum())
        point_sum=number((q[i]-base['anchor_points'][i]).square().sum())
        add(records,'geometry_losses',sample=sample,bce_sum=number(gs['bce_numerator']),bce_count=number(gs['bce_denominator']),
            dice_loss_sum=2*number(gs['soft_dice_loss']),dice_region_count=2,point_loss_sum=number(gs['point_numerator']),point_weight_count=float(weights[i]),
            containment_sum=number(gs['containment_numerator']),containment_count=number(gs['containment_denominator']),
            row_agreement_sum=number(gs['row_numerator']),row_agreement_count=number(gs['row_denominator']),
            anchor_drift_sum=number(gs['anchor_drift']),anchor_drift_count=1,anchor_mask_sum=mask_sum,
            anchor_mask_count=int(number(gs['bce_denominator'])),anchor_point_sum=point_sum,anchor_point_count=6)
        pred_full=np.repeat(np.repeat(arr['masks'][i]>=.5,4,axis=1),4,axis=2)
        pred_boxes=[]
        for region in (0,1):
            pb=box(pred_full[region]);tb=box((truth[i,region]>.5)&valid0[i]);pred_boxes.append(pb)
            tp,fp,fn=[int(m[k][i,region]) for k in ('tp','fp','fn')]
            record=dict(sample=sample,region=region,tp=tp,fp=fp,fn=fn,valid_pixels=int(valid0[i].sum()),target_pixels=tp+fn,predicted_pixels=tp+fp,
                        dice=float(m['dice'][i,region]),iou=tp/(tp+fp+fn) if tp+fp+fn else 1.,box_iou=box_iou(pb,tb),
                        target_box_available=tb is not None,predicted_box_available=pb is not None,
                        bce_sum=number((F.binary_cross_entropy_with_logits(ref['seg_logits'][i,region],y[i,region],reduction='none')*valid[i,0]).sum()),
                        bce_count=int(number(valid[i,0].sum())))
            for label,value in [('predicted',pb),('target',tb)]:
                for axis,v in zip(('left','top','right','bottom'),value or [None]*4):record[label+'_box_'+axis]=v
            add(records,'region_metrics',**record)
        for j,x in enumerate(arr['points'][i]):
            available=weights[i]>0;delta=abs(float(x-target[i,j])) if available else None
            huber=(delta**2/.04 if delta<.02 else delta-.01) if available else None
            add(records,'point_values',sample=sample,guide=j//2,side=j%2,prediction=float(x),target=float(target[i,j]) if available else None,
                absolute_error_width_fraction=delta,smooth_l1_sum=huber,target_kind=row['point_kind'],optimization_weight=float(weights[i]),human_metric_available=bool(batch['human'][i]))
        for j,name in enumerate(('tyre_error','tread_error','human_point_error')):
            add(records,'quality_predictions',sample=sample,quantity=name,predicted_error=float(qp[i,j]),observed_error=float(qt[i,j]) if qa[i,j] else None,
                target_available=bool(qa[i,j]),human_only_target=j==2,loss_sum=number(F.smooth_l1_loss(out['quality'][i,j],quality['targets'][i,j],beta=.02)) if qa[i,j] else None,
                loss_count=int(qa[i,j]),validated_for_deployment=False)
        anchor_binary=array(base['anchor_logits'][i].sigmoid())>=.5
        anchor_boxes=[box(np.repeat(np.repeat(anchor_binary[j],4,0),4,1)) for j in (0,1)]
        box_deltas=[]
        for a,b in zip(pred_boxes,anchor_boxes):box_deltas.append(float(np.mean(np.abs(np.asarray(a)-np.asarray(b))/np.array([384,512,384,512]))) if a is not None and b is not None else None)
        add(records,'geometry_drift',sample=sample,mask_logit_delta_max=number((ref['seg_logits'][i]-base['anchor_logits'][i]).abs().max()),
            mask_probability_delta_mean=number((p[i]-base['anchor_logits'][i].sigmoid()).abs().mean()),
            point_delta_max_width_fraction=number((q[i]-base['anchor_points'][i]).abs().max()),tyre_box_delta=box_deltas[0],tread_box_delta=box_deltas[1],anchor_weights_unchanged=True,
            anchor_tyre_box_available=anchor_boxes[0] is not None,anchor_tread_box_available=anchor_boxes[1] is not None)
        params=atlas['row_parameters'][i];flags=atlas['row_valid'][i]
        band_gaps=defaultdict(list)
        for gi,guide in enumerate(guides[i]):
            yindex=int(np.rint(guide*127));xs=np.where((arr['masks'][i,0,yindex]>=.5)&(arr['masks'][i,1,yindex]>=.5))[0]
            sy=guide*511
            if len(xs) and flags.any() and params[:,0].min()<=sy<=params[:,0].max():
                atlas_row=int(np.argmin(np.abs(params[:,0]-sy)))
                if flags[atlas_row]:band_gaps[atlas_row//64].append(float(np.abs(arr['points'][i].reshape(3,2)[gi]-np.array([xs[0],xs[-1]])/95).mean()))
        for band in range(8):
            sl=slice(64*band,64*(band+1));ok=flags[sl];width=(params[sl,2]-params[sl,1]+1)/384
            add(records,'atlas_bands',sample=sample,band=band,valid_rows=int(ok.sum()),valid_fraction=float(ok.mean()),mean_width_width_fraction=float(width[ok].mean()) if ok.any() else None,
                clipped_fraction=float(((params[sl,1]==0)|(params[sl,2]==383))[ok].mean()) if ok.any() else None,point_mask_gap_width_fraction=float(np.mean(band_gaps[band])) if band_gaps[band] else None,unsupported=not ok.any(),
                point_mask_gap_count=len(band_gaps[band]),point_mask_gap_exceeds_flag=bool(max(band_gaps[band])>.05) if band_gaps[band] else None)
        for patch in range(32):
            band,col=divmod(patch,4)
            add(records,'patch_support',sample=sample,patch=patch,band=band,column=col,valid_fraction=float(atlas['coverage'][i,patch]),pooling_mass=float(atlas['mass'][i,patch]),
                valid_target_fraction=float(target_fraction[i,patch]),source_row_start=int(round(params[band*64,0])),source_row_end=int(round(params[band*64+63,0])))
        if dense:
            for r in range(512):add(records,'atlas_rows',sample=sample,atlas_row=r,source_y=float(params[r,0]),left_x=float(params[r,1]),right_x=float(params[r,2]),row_valid=bool(flags[r]),source_clipped=bool(params[r,1]==0 or params[r,2]==383),grid_parameter_native_key=f'row_parameters:{sample}:{r}')
            foreground=(truth[i,:2]*valid0[i]).reshape(2,128,4,96,4).sum((2,4)).astype(int)
            available=valid0[i].reshape(128,4,96,4).sum((1,3)).astype(int)
            for region in (0,1):
                for yy in range(128):
                    for xx in range(96):add(records,'dense_pixels',sample=sample,region=region,row=yy,column=xx,probability=np.float32(arr['masks'][i,region,yy,xx]),predicted=bool(arr['masks'][i,region,yy,xx]>=.5),target_foreground_subpixels=int(foreground[region,yy,xx]),valid_subpixels=int(available[yy,xx]))
        info.append(dict(sample=sample,image_id=row['image_id'],role=row['role'],component=row['component'],sha256=row['image_sha256'],dice=m['dice'][i].tolist(),
                         human=bool(batch['human'][i]),point_error=float(m['point_error'][i]),geometry_valid=geometry_valid,SSL_valid=bool(eligible[i]),
                         valid_fraction=float(atlas['coverage'][i].mean()),
                         quality=qp[i].tolist(),quality_target=qt[i].tolist(),quality_available=qa[i].tolist(),
                         embedding=array(views[0]['embedding'][i]).tolist(),projected=array(views[0]['projected'][i]).tolist()))
    masked_totals=[]
    for vi,view in enumerate(views):
        projected=array(view['projected']);embedding=array(view['embedding']);pred=array(view['prediction']);target0=array(view['target']);weight=array(view['weight']);selected=array(view['selected'])
        for field in ('encoded','full'):native[f'v{vi}_{field}']=array(view[field])
        packs={name:np.zeros((n,8,channels,16,16),dtype='float32') for name,channels in [('prediction',4),('target',4),('weight',1)]}
        per_image=[]
        for i,row in enumerate(batch['rows']):
            sample=first+i;chosen=np.where(selected[i])[0];factors=array(view['factors'][i]);texseed=int(seeds[i])*2+vi
            add(records,'view_plans',sample=sample,view=vi,brightness=float(factors[0]),contrast=float(factors[1]),gamma=float(factors[2]),
                geometry_augmentation_seed=int(seeds[i]) if stage=='train' else None,texture_seed=texseed,mask_seed=texseed,
                masked_tiles=len(chosen),visible_tiles=int(((atlas['coverage'][i]>=.25)&~selected[i]).sum()),
                masked_target_pixels=int((weight[i]*selected[i,:,None,None,None]).sum()),masking_policy='one_per_supported_band_after_3_photometric_draws')
            for name,vector in [('global_embeddings',embedding[i]),('projected_embeddings',projected[i])]:
                for dimension,value in enumerate(vector):add(records,name,sample=sample,view=vi,dimension=dimension,value=np.float32(value))
            num=den=mean_num=0.
            for slot,patch in enumerate(chosen):
                for name,value in [('prediction',pred),('target',target0),('weight',weight)]:packs[name][i,slot]=value[i,patch]
                w=weight[i,patch].astype('float64');error=np.abs(pred[i,patch].astype('float64')-target0[i,patch].astype('float64'))
                hn=float((np.where(error<.1,error**2/.2,error-.05)*w).sum());hd=float(w.sum()*4)
                baseline=None
                if mean is not None:
                    e=np.abs(mean[patch].astype('float64')-target0[i,patch].astype('float64'));baseline=float((np.where(e<.1,e**2/.2,e-.05)*w).sum())
                add(records,'masked_patch_metrics',sample=sample,view=vi,patch=int(patch),huber_sum=hn,huber_weight_count=hd,absolute_error_sum=float((error*w).sum()),absolute_error_max=float(error.max()),mean_predictor_huber_sum=baseline,mean_predictor_weight_count=hd if baseline is not None else None)
                add(records,'masked_patch_plan',sample=sample,view=vi,patch=int(patch),masked=True,valid_loss_pixels=int(w.sum()),mask_weight_sum=hd,
                    target_mean=float(target0[i,patch].mean()),target_std=float(target0[i,patch].std()),native_prediction_key=f'v{vi}_prediction:{sample}:{slot}',native_target_key=f'v{vi}_target:{sample}:{slot}',native_weight_key=f'v{vi}_weight:{sample}:{slot}')
                if eligible[i]:num+=hn;den+=hd;mean_num+=baseline or 0.
            per_image.append((num,den,mean_num))
        for name,value in packs.items():native[f'v{vi}_{name}']=value
        masked_totals.append(per_image)
    for i,item in enumerate(info):
        za=array(views[0]['projected'][i]);zb=array(views[1]['projected'][i]);ea=array(views[0]['embedding'][i]);eb=array(views[1]['embedding'][i])
        cosine=1-float(np.dot(ea,eb)/(max(np.linalg.norm(ea)*np.linalg.norm(eb),1e-12)))
        item.update(masked_num=sum(v[i][0] for v in masked_totals),masked_den=sum(v[i][1] for v in masked_totals),mean_num=sum(v[i][2] for v in masked_totals),pair_cosine=cosine)
        add(records,'pair_losses',sample=first+i,invariance=float(np.mean((za.astype('float64')-zb)**2)),masked_huber_a=masked_totals[0][i][0]/masked_totals[0][i][1] if masked_totals[0][i][1] else None,
            masked_huber_b=masked_totals[1][i][0]/masked_totals[1][i][1] if masked_totals[1][i][1] else None,pair_cosine_distance=cosine,
            valid_images_in_batch=int(eligible.sum()),valid_components_in_batch=len({batch['rows'][j]['component'] for j in np.where(eligible)[0]}))
    for i in range(n):
        for name,value in native.items():add(records,'spatial_maps_index',sample=first+i,quantity=name,native_file=name,native_key=str(first+i),shape=str(list(value.shape[1:])),dtype=str(value.dtype),coordinate_system='original_512x384_or_atlas_512x256_by_quantity',sha256=None,precision_note='logical_stream_join_native_chunks_index; active prefix in selected state')
    return records,native,info

def summary(info,baseline=None,median=None):
    valid=[r for r in info if r['SSL_valid']];human=[r for r in info if r['human']]
    z=[r['projected'] for r in valid];stats=latent(z)
    den=sum(r['masked_den'] for r in valid);num=sum(r['masked_num'] for r in valid);mn=sum(r['mean_num'] for r in valid)
    return dict(mean_dice=float(np.mean([r['dice'] for r in info])) if info else None,
                human_point_MAE=float(np.mean([r['point_error'] for r in human])) if human else None,
                invalid_atlas_fraction=float(np.mean([not r['geometry_valid'] for r in info])) if info else None,
                masked_huber=num/den if den else None,mean_huber=mn/den if den else None,
                masked_num=num,masked_den=den,mean_num=mn,
                pair_cosine=float(np.mean([r['pair_cosine'] for r in valid])) if valid else None,
                eligible_images=len(valid),components=len({r['component'] for r in valid}),**stats)

def selection(value,baseline):
    reasons=[]
    geometry=(value['mean_dice'] is not None and value['human_point_MAE'] is not None and
              value['mean_dice']>=baseline['mean_dice']-.002 and value['human_point_MAE']<=baseline['human_point_MAE']+.001 and value['invalid_atlas_fraction']<=baseline['invalid_atlas_fraction']+.01)
    collapse=value['std_median'] is not None and value['std_median']>=.1 and value['rank']>=4
    if not geometry:reasons.append('geometry_floor')
    if not collapse:reasons.append('collapse')
    if value['mean_huber'] is None or value['mean_huber']<1e-6:reasons.append('mean_predictor_denominator')
    if value['masked_huber'] is None or value['pair_cosine'] is None:reasons.append('missing_representation_metric')
    objective=value['masked_huber']/value['mean_huber']+.1*value['pair_cosine'] if not reasons else None
    return objective,geometry,collapse,reasons
