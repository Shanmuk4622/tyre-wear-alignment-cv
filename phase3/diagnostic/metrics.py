"""Explicit schemas and post-hoc diagnostic metrics for frozen TREAD-ONE."""
import math
import numpy as np
import torch

IDENTITY = ['unit','job','seed','endpoint','epoch','split','group','tyre','clip','domain','image_id','image_sha256','label','variant']
SCHEMAS = {
 'predictions': IDENTITY+['prediction','correct','confidence','class_margin','true_class_probability','nll','brier','entropy_nats']+
       [f'probability_{c}' for c in range(3)]+[f'logit_{c}' for c in range(3)]+
       ['normal_prediction','probability_l1_delta','true_probability_delta','normal_predicted_class_probability_delta','prediction_changed',
        'mean_two_region_dice','mean_box_iou','point_kind','annotation_point_mae','human_point_mae','human_point_available',
        'gate_entropy_nats','gate_mass_inside_true_tread','gate_support_fraction','evidence_residual'],
 'regions': IDENTITY+['region','tp','fp','fn','tn','valid_pixels','ignored_pixels','dice','iou','box_iou']+
       [f'{prefix}_{c}' for prefix in ('pred_box','target_box') for c in ('x0','y0','x1_exclusive','y1_exclusive')],
 'points': IDENTITY+['point_index','guide_fraction','side','prediction','target','target_kind','human_eligible',
       'signed_error_width_fraction','absolute_error_width_fraction','absolute_error_native_pixels','native_width','native_height'],
 'latents': IDENTITY+['latent','minimum','maximum','mean','std','gated_mass'],
 'evidence': IDENTITY+['class_index','logit','bias','sum_contribution','positive_sum','negative_sum',
       'inside_true_tread_sum','outside_true_tread_sum','evidence_max','evidence_min'],
 'timings': ['unit','job','images','load_seconds','normal_forward_seconds','background_forward_seconds','occlusion_forward_seconds',
       'metrics_and_gallery_seconds','unit_compute_seconds','host_rss_bytes','cuda_allocated_bytes','cuda_peak_bytes','working_free_bytes'],
 'gradients': ['unit','job','row_type','task','other_task','module_group','task_loss','gradient_norm','cosine','missing_reason','image_ids','seconds']
}
VARIANTS = ['normal','uniform_gate','oracle_tread_gate','background_neutralized','top10_image_occluded']


def group_key(row):
    return 'clip:'+str(row.get('source_video') or row.get('physical_tyre_id')) if row['domain'] == 'new_video_frame' else 'tyre:'+str(row['physical_tyre_id'])


def identity(row, job, variant):
    return dict(job=job['key'], seed=job['seed'], endpoint=job['endpoint'], epoch=job['epoch'],
          split=row['role'], group=group_key(row), tyre=row.get('physical_tyre_id'), clip=row.get('source_video'),
          domain=row['domain'], image_id=row['image_id'], image_sha256=row['image_sha256'], label=row['class_index'], variant=variant)


def box(mask):
    ys, xs = np.where(mask)
    return [int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1] if len(xs) else None


def box_iou(a,b):
    if a is None or b is None: return 0.
    inter = max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union else 0.


def cpu_output(out):
    return {k:v.detach().float().cpu().numpy() for k,v in out.items() if k in ('masks','points','concepts','gate','contributions','logits','probabilities','bias')}


def gate_output(out, gate, weights, bias):
    gate = gate/gate.sum((-2,-1),keepdim=True).clamp_min(1e-6)
    contribution = torch.einsum('bkhw,ck->bchw',out['concepts'].float(),weights.float())*gate
    logits = contribution.sum((-2,-1))+bias.float()
    return dict(out, gate=gate, contributions=contribution, logits=logits, probabilities=logits.softmax(-1))


def records(row, job, variant, out, index, target, normal):
    """All scalar calculations use the already computed FP32 model output."""
    ident = identity(row, job, variant)
    p = out['probabilities'][index].astype(np.float64)
    y = row['class_index']; pred = int(p.argmax()); n = normal['probabilities'][index]
    nclass = int(n.argmax()); confidence = float(p[pred])
    logits = out['logits'][index]; points = out['points'][index]
    valid = target[2] < .5; truth = target[:2] > .5
    binary = np.repeat(np.repeat(out['masks'][index]>=.5,4,1),4,2)
    region_rows = []
    for c,name in enumerate(('tyre','tread')):
        a,b = binary[c]&valid,truth[c]&valid
        tp,fp,fn,tn = [int(v.sum()) for v in (a&b,a&~b&valid,~a&b&valid,~a&~b&valid)]
        ab,bb = box(a),box(b)
        rec = dict(ident, region=name, tp=tp,fp=fp,fn=fn,tn=tn,valid_pixels=int(valid.sum()),ignored_pixels=int((~valid).sum()),
                   dice=(2*tp+1)/(2*tp+fp+fn+1),iou=(tp+1)/(tp+fp+fn+1),box_iou=box_iou(ab,bb))
        for prefix,value in (('pred_box',ab),('target_box',bb)):
            rec.update({prefix+'_'+name: value[k] if value else None for k,name in enumerate(('x0','y0','x1_exclusive','y1_exclusive'))})
        region_rows.append(rec)
    point_rows=[]
    target_points=row['points']
    annotation_error=float(np.abs(points-np.array(target_points)).mean()) if target_points is not None else None
    for k,value in enumerate(points):
        error=float(value-target_points[k]) if target_points is not None else None
        point_rows.append(dict(ident, point_index=k,guide_fraction=row['guide_fractions'][k//2],side='left' if k%2==0 else 'right',
             prediction=float(value),target=target_points[k] if target_points is not None else None,target_kind=row['point_kind'],human_eligible=int(row['point_kind']=='human'),
             signed_error_width_fraction=error,absolute_error_width_fraction=abs(error) if error is not None else None,
             absolute_error_native_pixels=abs(error)*(row['width']-1) if error is not None else None,native_width=row['width'],native_height=row['height']))
    gate=out['gate'][index,0]; concepts=out['concepts'][index]
    # Match the gate grid with the same 4x4 valid/tread aggregation as the loss.
    gt=(truth[1]&valid).reshape(128,4,96,4).mean((1,3))
    evidence=out['contributions'][index]
    rec=dict(ident,prediction=pred,correct=int(pred==y),confidence=confidence,class_margin=float(np.sort(p)[-1]-np.sort(p)[-2]),
        true_class_probability=float(p[y]),nll=float(-np.log(max(p[y],1e-12))),brier=float(np.square(p-np.eye(3)[y]).sum()),
        entropy_nats=float(-(p*np.log(np.maximum(p,1e-12))).sum()),normal_prediction=nclass,probability_l1_delta=float(np.abs(p-n).sum()),
        true_probability_delta=float(p[y]-n[y]),normal_predicted_class_probability_delta=float(p[nclass]-n[nclass]),prediction_changed=int(pred!=nclass),
        mean_two_region_dice=float(np.mean([v['dice'] for v in region_rows])),mean_box_iou=float(np.mean([v['box_iou'] for v in region_rows])),
        point_kind=row['point_kind'],annotation_point_mae=annotation_error,human_point_mae=annotation_error if row['point_kind']=='human' else None,
        human_point_available=int(row['point_kind']=='human'),gate_entropy_nats=float(-(gate*np.log(np.maximum(gate,1e-12))).sum()),
        gate_mass_inside_true_tread=float((gate*gt).sum()),gate_support_fraction=float((gate>1e-6).mean()),
        evidence_residual=float(np.abs(evidence.sum((-2,-1))+out['bias']-logits).max()))
    rec.update({f'probability_{c}':float(p[c]) for c in range(3)});rec.update({f'logit_{c}':float(logits[c]) for c in range(3)})
    latent_rows=[dict(ident,latent=k,minimum=float(v.min()),maximum=float(v.max()),mean=float(v.mean()),std=float(v.std()),gated_mass=float((v*gate).sum())) for k,v in enumerate(concepts)]
    evidence_rows=[dict(ident,class_index=c,logit=float(logits[c]),bias=float(out['bias'][c]),sum_contribution=float(v.sum()),positive_sum=float(np.maximum(v,0).sum()),
             negative_sum=float(np.minimum(v,0).sum()),inside_true_tread_sum=float((v*gt).sum()),outside_true_tread_sum=float((v*(1-gt)).sum()),
             evidence_max=float(v.max()),evidence_min=float(v.min())) for c,v in enumerate(evidence)]
    return rec,region_rows,point_rows,latent_rows,evidence_rows


def classification(rows):
    matrix=np.zeros((3,3),dtype=int)
    for r in rows: matrix[int(r['label']),int(r['prediction'])]+=1
    support=matrix.sum(1); observed=support>0
    result=dict(images=len(rows),accuracy=float(np.trace(matrix)/len(rows)),balanced_accuracy_observed=float(np.mean(np.diag(matrix)[observed]/support[observed])),
                nll=float(np.mean([float(r['nll']) for r in rows])),brier=float(np.mean([float(r['brier']) for r in rows])),
                mean_two_region_dice=float(np.mean([float(r['mean_two_region_dice']) for r in rows])))
    human=[float(r['human_point_mae']) for r in rows if r['human_point_mae'] not in ('',None)]
    result.update(human_point_images=len(human),human_point_mae=float(np.mean(human)) if human else None)
    classes=[]
    for c in range(3):
        tp=int(matrix[c,c]);fp=int(matrix[:,c].sum()-tp);fn=int(matrix[c].sum()-tp)
        classes.append(dict(class_index=c,support=int(support[c]),predicted=int(matrix[:,c].sum()),tp=tp,fp=fp,fn=fn,
            recall=tp/(tp+fn) if tp+fn else None,precision=tp/(tp+fp) if tp+fp else None,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.))
    return result,classes,matrix
