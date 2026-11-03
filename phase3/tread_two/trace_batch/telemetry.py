"""Direct per-view reconstruction, masks, losses and native evidence for every draw."""
import json
import numpy as np
import torch
from torch.nn import functional as F
from common import HERE,cpu,canonical,read
from schema import add
from targets import tiles,choose_hidden,standardized
from metrics import controls,sums

def view_plan(batch,ids,epoch,training,variant='identity'):
    views=[];plans=[]
    for view in range(2):
        seeds=[1000003+epoch*100003+int(slot)*97+view*31 for slot in ids]
        factors=np.stack([np.ones(3) if not training and view==0 else np.random.default_rng(seed).uniform(.9,1.1,3) for seed in seeds]).astype('float32')
        hidden=choose_hidden(tiles(batch['valid'].float()),[seed*2+view for seed in seeds])
        plans.append(dict(seeds=seeds,factors=factors,hidden=hidden))
    return plans

def run_views(model,batch,plans):
    rgb=tiles(batch['rgb']);valid=tiles(batch['valid'].float())
    return [model(rgb,valid,p['hidden'],torch.from_numpy(p['factors']).to(rgb.device)) for p in plans]

def observe(store,cache,batch,outputs,plans,ids,flags,stats,cursor,stage,variant='identity'):
    records={};arrays={};samples=[];epoch=cursor['epoch'];update=cursor['update'];first=cursor['observation']
    target=standardized(batch['target'],stats);weight=batch['weight']
    native_keys={'local_prediction':'local_prediction','masked_prediction':'masked_prediction','encoded':'encoded','full_context':'full_context','masked_context':'context',
                 'local_pooled':'local_pooled','context_mean':'context_mean','pooled':'pooled','projected':'projected','hidden':'hidden','mass':'mass'}
    for name,key in native_keys.items():arrays[name]=np.stack([o[key].detach().cpu().numpy() for o in outputs],axis=1)
    cp=[{k:v.detach().cpu().numpy() for k,v in o.items() if torch.is_tensor(v) and k!='visible_input'} for o in outputs]
    result={}
    for v,out in enumerate(outputs):
        control,common=controls(batch['target'],weight,out['hidden'],stats)
        model_sum,common_count=sums(out['masked_prediction'],target,common)
        local_sum,local_count=sums(out['local_prediction'],target,weight*out['supported'][...,None,None,None])
        masked_sum,masked_count=sums(out['masked_prediction'],target,weight*out['hidden'][...,None,None,None])
        local_control_sum,_=sums(control['position_mean'],target,weight*out['supported'][...,None,None,None])
        control_sum={n:sums(value,target,common)[0] for n,value in control.items()}
        eligible=(masked_count>0)&(out['supported'].sum(-1)>=4)&(out['supported'].reshape(-1,8,4).any(-1).sum(-1)>=2)&((out['supported']&~out['hidden']).sum(-1)>=2)
        result[v]=dict(model_sum=model_sum,common_count=common_count,local_sum=local_sum,local_count=local_count,masked_sum=masked_sum,masked_count=masked_count,local_control_sum=local_control_sum,control_sum=control_sum,eligible=eligible)
    for i,slot in enumerate(ids):
        row=cache.rows[slot];image=row['image_id'];observation=first+i
        common_fields=dict(epoch=epoch,update=update,image_id=image)
        add(records,'sample_schedule',stage,**common_fields,draw=observation,source_sha256=row['image_sha256'],component=row['component'],primary=flags[i],padding=not flags[i],ordinal_in_epoch=cursor.get('batch',0)*8+i,batch_position=i)
        eligible=bool(result[0]['eligible'][i] and result[1]['eligible'][i])
        support=cp[0]['supported'][i];hidden=cp[0]['hidden'][i]
        parameter_update=update-1 if stage=='train' else cursor['best']['epoch']*cursor['steps_per_epoch'] if stage=='endpoint_selected' else update
        add(records,'sample_observations',stage,**common_fields,parameter_update=parameter_update,observation=observation,source_sha256=row['image_sha256'],primary=flags[i],eligible=eligible,
            skip_reason='' if eligible else 'insufficient_predicted_support_or_hidden_target',supported_tiles=int(support.sum()),supported_bands=int(support.reshape(8,4).any(-1).sum()),visible_tiles=int((support&~hidden).sum()),hidden_cells=int((weight[i]*outputs[0]['hidden'][i,:,None,None,None]>0).sum()))
        norms=[float(o['projected'][i].detach().norm()) for o in outputs]
        cosine=float((1-F.cosine_similarity(outputs[0]['projected'][i:i+1],outputs[1]['projected'][i:i+1],dim=1)).detach()) if min(norms)>=1e-8 else None
        records['sample_observations'][-1]['paired_cosine']=cosine
        sample=dict(image_id=image,component=row['component'],role=row['role'],observation=observation,eligible=eligible,primary=flags[i],cosine=cosine,
            projected=cp[0]['projected'][i].tolist(),local_pooled=cp[0]['local_pooled'][i].tolist(),pooled=cp[0]['pooled'][i].tolist(),context_mean=cp[0]['context_mean'][i].tolist(),
            encoded=cp[0]['encoded'][i].tolist(),full_context=cp[0]['full_context'][i].tolist(),support=support.tolist())
        for v,out in enumerate(outputs):
            q=result[v];vf=dict(common_fields,view=v);p=plans[v]
            add(records,'view_plans',stage,**vf,observation=observation,source_sha256=row['image_sha256'],appearance_seed=p['seeds'][i],brightness=float(p['factors'][i,0]),contrast=float(p['factors'][i,1]),gamma=float(p['factors'][i,2]),target_id=f'{slot}:{variant}',geometry_id=str(slot))
            for t in range(32):
                add(records,'mask_plans',stage,**vf,observation=observation,tile=t,band=t//4,column=t%4,mask_seed=p['seeds'][i]*2+v,supported=bool(cp[v]['supported'][i,t]),hidden=bool(cp[v]['hidden'][i,t]),valid_fraction=float(tiles(batch['valid'][i:i+1].float()).mean((-3,-2,-1))[0,t]))
                for y in range(4):
                    for x in range(4):
                        add(records,'reconstruction_cells',stage,**vf,observation=observation,tile=t,cell_y=y,cell_x=x,target_id=f'{slot}:{variant}',valid_pixels=int(weight[i,t,0,y,x]),hidden=bool(cp[v]['hidden'][i,t]),
                            **{f'local_{k}':float(cp[v]['local_prediction'][i,t,k,y,x]) for k in range(6)},**{f'masked_{k}':float(cp[v]['masked_prediction'][i,t,k,y,x]) for k in range(6)})
            for objective,num,den in [('local',q['local_sum'],q['local_count']),('masked',q['masked_sum'],q['masked_count']),('local_position_control',q['local_control_sum'],q['local_count'])]:
                count=float(den[i]);add(records,'image_losses',stage,**vf,observation=observation,objective=objective,huber_sum=float(num[i].detach()),weighted_channel_count=count,available=count>0,eligible=bool(q['eligible'][i]),image_mean=float((num[i]/den[i]).detach()) if count else None)
            for name,value in q['control_sum'].items():
                count=float(q['common_count'][i]);native_count=float(q['masked_count'][i]);model=float(q['model_sum'][i].detach())
                add(records,'control_losses',stage,**vf,control=name,role=row['role'],model_sum=model,control_sum=float(value[i].detach()),common_count=count,model_mean=model/count if count else None,control_mean=float(value[i].detach())/count if count else None,available=count>0)
                add(records,'control_coverage',stage,**vf,control=name,role=row['role'],model_native_count=native_count,control_native_count=count,common_count=count,coverage=count/native_count if native_count else None,missing_reason='' if native_count==count else 'unavailable_train_position_or_visible_neighbor')
            for moment in range(5):
                for channel in range(48):
                    gen=store.generations.get('native:pooled',0)
                    add(records,'spatial_moment_vectors',stage,**vf,observation=observation,moment=['mean','row','column','row_column','row_squared'][moment],channel=channel,value=float(cp[v]['pooled'][i,moment*48+channel]),space='contextual240',native_file=f'native/pooled/{gen:05d}.npy')
        q=result[0];den=float(q['common_count'][i]);local_den=float(q['local_count'][i]);masked_den=float(q['masked_count'][i])
        sample.update(local_count=local_den,local_loss=float(q['local_sum'][i].detach())/local_den if local_den else None,local_control=float(q['local_control_sum'][i].detach())/local_den if local_den else None,
                      model_common=float(q['model_sum'][i].detach())/den if den else None,coverage=den/masked_den if masked_den else None)
        sample.update({'control_'+k:float(v[i].detach())/den if den else None for k,v in q['control_sum'].items()})
        samples.append(sample)
    return records,arrays,samples

def metadata(cfg):
    records={}
    for key,value in cfg['environment'].items():add(records,'environment','initialization',session='environment_identity',key=key,value=canonical(value).decode(),available=True,identity_relevant=True)
    for key,value in cfg.items():
        if key not in ('sources','schema'):add(records,'run_configuration','initialization',key=key,value=canonical(value).decode(),type=type(value).__name__,protocol_revision=cfg['version'])
    for name,meta in cfg['sources'].items():add(records,'source_hashes','initialization',logical_file=name,bytes=meta['bytes'],sha256=meta['sha256'],origin='embedded_trace_package',frozen=True)
    add(records,'geometry_provenance','initialization',artifact='GEO_final_geometry_only',revision=cfg['geometry']['revision'],sha256=cfg['geometry']['sha256'],selected=False,inherited_status=cfg['geometry']['selection'],source_sha256=cfg['sources']['frozen_anchor.py']['sha256'])
    for artifact,reason in [('per_view_input_contrast','98.65GB training-only; reconstruct from immutable clean cache, exact seeds and preprocessing'),('every_historical_optimizer_snapshot','Two rolling complete states plus all epoch weights and ledger; overwritten optimizer history is not retained'),('physical_depth_safety','No measured physical labels'),('learned_confidence_head','Omitted after prior model lost to median controls')]:
        add(records,'artifact_gaps','initialization',artifact=artifact,requested_grain='declared_scope',available=False,reason=reason,reconstruction_inputs='pinned cache,plans,source,epoch weights where applicable',interpretation_limit='No invented unsaved evidence or safety claim')
    return records
