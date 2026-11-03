"""Measured geometry-first lifecycle, complete-unit cursor and train-only controls."""
import copy,hashlib,json,math,os,shutil,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from common import HERE,atomic,cpu,digest,environment,file_meta,native,read,restore_rng,rng_state,seed_all,sha
from anchor_geo import Anchor
from atlas_geo import canonicalize
from data_geo import dataset,load_batch,locate,photometric
from geometry import patches
from lc_features import contrast
from model_geo import Geo
from objective_geo import full_geometry,losses
from schedule_geo import schedule
from schema_fit import specification
from store_fit import Store
from telemetry_fit import add,array,latent,number,observe,selection,source_records,summary


def optimizer_for(model):
    return torch.optim.AdamW([dict(params=p,name=n,lr=lr,initial_lr=lr) for (n,p),lr in zip(model.groups().items(),(1e-4,1e-3,3e-4))],
                            weight_decay=1e-4,betas=(.9,.999),eps=1e-8,foreach=False,fused=False)

def view_seed(image_id):return int.from_bytes(hashlib.sha256(image_id.encode()).digest()[:8],'little')&((1<<61)-1)

def configuration(device='cuda:0',eval_device='cuda:1',fixture=False):
    inv=read(HERE/'FILES.json');assert all(file_meta(HERE/n)==m for n,m in inv.items()),'Frozen source mismatch'
    value=dict(revision='geo-r1-pilot-v1',mode='engineering_fixture' if fixture else 'measured_single_candidate',
                sources=inv,environment=environment(),devices=[device,eval_device],seed=1,epochs=2 if fixture else 60,
                batches_per_epoch=2 if fixture else 49,batch=8,FP32=True,fixture=fixture,
                scientific_plan=read(HERE/'PLAN.json'),schema=specification(),NB16_gate=read(HERE/'NB16_GATE.json'),
                selection='geometry_and_collapse_eligibility_then_masked_ratio_plus_point1_pair_cosine_earliest_tie',
                native_shard_samples=64,maximum_train_updates_per_shard=8,CSV_active_limit=8*1024**2,
                initial_control='386_identity_training_images_anchor_atlas_mean; unobserved mean coordinates predict0 with counts',
                mean_fallback_zero_is_not_truth=True,probe_native_policy='mask_logits_points_quality_row_parameters_and_flags_only; full vectors/statistics_CSV; no probe_texture_raw',
                storage=dict(logical=15000000000,peak=18000000000,reserve=2000000000,output=20000000000),
                probe_variants=['brightness_0.8','brightness_1.2','gamma_0.8','gamma_1.2','grayscale','blur','background_neutral',
                                'reverse_bands','reverse_columns','top_residual_tile'])
    return json.loads(json.dumps(value))

def make_eval(model,device):
    state=rng_state();result=Geo().to(device);result.load_state_dict(model.state_dict());result.eval();result.requires_grad_(False);restore_rng(state)
    return result

def identity_target(atlas):
    with torch.no_grad():
        tiles=patches(atlas['rgb']);valid=patches(atlas['valid']);n=len(tiles)
        lc=contrast(tiles.flatten(0,1),valid.flatten(0,1),{'local_contrast_window':9,'local_contrast_std_floor':.02,'local_contrast_clip':3})
        target=F.avg_pool2d(lc,4).reshape(n,32,4,16,16)
        w=-F.max_pool2d(-(patches(atlas['valid']*(1-atlas['ignored']))).flatten(0,1),4).reshape(n,32,1,16,16)
    return array(target),array(w)

def gradient_step(model,optimizer,out,base,batch,seeds,epoch,update,counters):
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:group['lr']=group['initial_lr']*.5*(1+math.cos(math.pi*update/2940))
    total,parts=losses(out,base,batch,seeds,epoch);assert torch.isfinite(total),'Nonfinite loss; last complete state remains authoritative'
    total.backward()
    if not parts['texture']['texture_step']:
        for p in model.groups()['texture']:p.grad=None
    records={}
    membership={id(p):name for name,group in model.groups().items() for p in group}
    for name,p in model.named_parameters():
        grad=p.grad
        add(records,'parameter_gradients',update=update+1,group=membership[id(p)],parameter=name,gradient_available=grad is not None,
            numel=p.numel(),l2_norm=number(grad.norm()) if grad is not None else None,max_absolute=number(grad.abs().max()) if grad is not None else None,
            mean_absolute=number(grad.abs().mean()) if grad is not None else None,zero_fraction=number((grad==0).float().mean()) if grad is not None else None,finite=bool(torch.isfinite(grad).all()) if grad is not None else None)
    for name,group in model.groups().items():
        pre=number(torch.nn.utils.clip_grad_norm_(group,1.,error_if_nonfinite=True));applied=any(p.grad is not None for p in group)
        post=math.sqrt(sum(number(p.grad.square().sum()) for p in group if p.grad is not None))
        if applied:counters[name]+=1
        add(records,'group_gradient_summary',update=update+1,group=name,pre_clip_norm=pre,post_clip_norm=post,clip_coefficient=min(1.,1/(pre+1e-6)),
            gradient_count=sum(p.grad.numel() for p in group if p.grad is not None),optimizer_step_applied=applied,frozen_anchor_gradient_count=0)
    optimizer.step();assert all(torch.isfinite(p).all() for p in model.parameters()),'Nonfinite state; replay is not committed'
    gs,qs,ts=parts['geometry'],parts['quality'],parts['texture']
    add(records,'batch_objectives',update=update+1,epoch=epoch,geometry_objective=number(gs['geometry']),quality_objective=number(qs['numerator']/qs['denominator'].clamp_min(1)),
        SSL_objective=number(ts['texture']),variance_a=number(ts['variance_a']),variance_b=number(ts['variance_b']),covariance_a=number(ts['covariance_a']),covariance_b=number(ts['covariance_b']),
        texture_ramp=ts['ramp'],total_loss=number(total),geometry_samples=len(batch['rows']),quality_mask_targets=int(qs['available'][:,:2].sum()),quality_human_point_targets=int(qs['available'][:,2].sum()),
        SSL_pairs=ts['eligible_images'],texture_step_skipped=not ts['texture_step'],invariance=number(ts['invariance']),masked_huber_sum=number(ts['masked_numerator']),masked_huber_count=number(ts['masked_denominator']),
        mean_predictor_huber_sum=None,mean_predictor_huber_count=None)
    return records

def metric_records(info,stage,endpoint,epoch,median):
    records={}
    groups={'overall':info}
    for component in sorted({r['component'] for r in info}):groups['component:'+component]=[r for r in info if r['component']==component]
    for role in sorted({r['role'] for r in info}):groups['role:'+role]=[r for r in info if r['role']==role]
    for group,items in groups.items():
        for task,values in [('tyre_dice',[r['dice'][0] for r in items]),('tread_dice',[r['dice'][1] for r in items]),
                            ('human_point_MAE',[r['point_error'] for r in items if r['human']]),('SSL_support',[int(r['SSL_valid']) for r in items])]:
            add(records,'epoch_metrics',stage=stage,endpoint=endpoint,epoch=epoch,task=task,group=group,numerator=sum(values),denominator=len(values),value=float(np.mean(values)) if values else None,
                available=bool(values),unit='image_width_fraction' if task=='human_point_MAE' else 'fraction',aggregation_definition='arithmetic_mean_over_available_images')
    for j,quantity in enumerate(('tyre_error','tread_error','human_point_error')):
        items=[r for r in info if r['quality_available'][j]]
        for k in range(10):
            block=[r for r in items if min(9,int(r['quality'][j]*10))==k]
            add(records,'quality_bins',stage=stage,endpoint=endpoint,epoch=epoch,quantity=quantity,bin=k,count=len(block),
                predicted_error_sum=sum(r['quality'][j] for r in block),observed_error_sum=sum(r['quality_target'][j] for r in block),
                absolute_error_sum=sum(abs(r['quality'][j]-r['quality_target'][j]) for r in block),median_baseline_error_sum=sum(abs(median[j]-r['quality_target'][j]) for r in block) if median is not None else None)
        metrics=[('error_head_MAE',[abs(r['quality'][j]-r['quality_target'][j]) for r in items]),
                 ('error_head_MSE',[(r['quality'][j]-r['quality_target'][j])**2 for r in items]),
                 ('error_head_RMSE',[(r['quality'][j]-r['quality_target'][j])**2 for r in items])]
        if median is not None and median[j] is not None:
            metrics.extend([('median_baseline_MAE',[abs(median[j]-r['quality_target'][j]) for r in items]),
                            ('median_baseline_MSE',[(median[j]-r['quality_target'][j])**2 for r in items]),
                            ('median_baseline_RMSE',[(median[j]-r['quality_target'][j])**2 for r in items])])
        for task,values in metrics:
            mean=float(np.mean(values)) if values else None;root=task.endswith('RMSE')
            add(records,'epoch_metrics',stage=stage,endpoint=endpoint,epoch=epoch,task=task,group=quantity,numerator=sum(values),denominator=len(values),value=math.sqrt(mean) if mean is not None and root else mean,available=bool(values),unit='error_fraction_squared' if task.endswith('MSE') and not root else 'error_fraction',aggregation_definition='root_of_mean_squared_error' if root else 'arithmetic_mean')
    value=summary(info)
    norms=[np.linalg.norm(r['embedding']) for r in info if r['SSL_valid']]
    add(records,'latent_diagnostics',update=None,stage=stage,eligible_images=value['eligible_images'],components=value['components'],projector_std_min=value['std_min'],projector_std_median=value['std_median'],
        dimensions_below_floor=value['below'],covariance_offdiag_energy=value['covariance'],effective_rank=value['rank'],embedding_norm_min=min(norms) if norms else None,embedding_norm_max=max(norms) if norms else None,finite=True)
    return records,value


class Engine:
    def __init__(self,output,weights,data_root='',device='cuda:0',eval_device='cuda:1',fixture=False):
        self.root=Path(output);self.root.mkdir(parents=True,exist_ok=True)
        self.cfg=configuration(device,eval_device,fixture);self.protocol=digest(self.cfg);self.fixture=fixture
        self.data=locate(data_root);print('Verifying exact existing input and annotation checksums...',flush=True)
        self.overlay,self.train=dataset(self.data,verify=True);self.lookup={r['image_id']:r for r in self.overlay['rows']}
        self.device,self.eval_device=device,eval_device
        if not fixture:
            assert torch.cuda.device_count()==2 and all('T4' in torch.cuda.get_device_name(i) for i in range(2)),'GPU T4 x2 required'
        seed_all(1);torch.set_num_threads(2);self.anchor=Anchor(weights,eval_device);seed_all(1);self.model=Geo().to(device);self.optimizer=optimizer_for(self.model)
        self.store=Store(self.root/'run',self.protocol)
        self.cursor=dict(phase='mean',index=0,epoch=1,batch=0,update=0,sample=0,counters=dict(geometry=0,texture=0,quality=0),
                         mean_sum=np.zeros((32,4,16,16),dtype='float64'),mean_count=np.zeros((32,1,16,16),dtype='float64'),median_targets=[[],[],[]],
                         baseline=[],evaluation=[],history=[],best=None,exposure={},endpoint=None,endpoint_index=0,endpoint_info=[],probe_index=0,
                         training_SSL_valid=0,training_draws=0)
        if (self.root/'CONTRACT.json').exists():assert read(self.root/'CONTRACT.json')==self.cfg
        else:atomic(self.root/'CONTRACT.json',self.cfg)
        if (self.store.root/'POINTER.json').exists():self.cursor=self.store.restore(self.model,self.optimizer)
        else:
            self.store.initialize_headers();self.store.register(self.root/'CONTRACT.json','CONTRACT.json')
            shutil.copyfile(weights,self.store.root/'anchor.pt');self.store.register(self.store.root/'anchor.pt')
            for name in read(HERE/'FILES.json'):
                self.store.register(HERE/name,'source/'+name)
            self.store.register(HERE/'FILES.json','source/FILES.json')
            rows=source_records(self.data,self.overlay)
            for epoch in range(1,61):
                order,hashed=schedule(self.train,1,epoch)
                for draws in order:
                    for draw in draws:
                        row=self.lookup[draw['image_id']];seed=draw['seed']
                        add(rows,'epoch_schedule',epoch=epoch,draw=draw['batch']*8+draw['slot'],batch=draw['batch'],batch_slot=draw['slot'],image_id=row['image_id'],source_sha256=row['image_sha256'],leakage_component=row['component'],is_padding=draw['padding'],
                            base_augmentation_seed=seed,view_a_seed=seed*2,view_b_seed=seed*2+1,mask_a_seed=seed*2,mask_b_seed=seed*2+1,schedule_sha256=hashed)
            env=environment()
            add(rows,'environment',record_id=0,python=env['python'],torch=env['torch'],cuda=env['cuda'],cudnn=str(env['cudnn']),numpy=env['numpy'],pillow=env['pillow'],HF_SDK='0.36.2',devices=str(env['devices']),free_output_bytes=shutil.disk_usage(self.root).free,
                source_inventory_sha256=digest(self.cfg['sources']),dataset_checksum=self.cfg['scientific_plan']['data']['checksum_file_sha256'],anchor_sha256=self.cfg['scientific_plan']['anchor']['weights_sha256'])
            self.store.commit(self.model,self.optimizer,self.cursor,rows,seal=True,kind='initialize')
        self.eval_model=None

    def control_rows(self):
        if not self.fixture:return self.train
        order,_=schedule(self.train,1,1)
        return [self.lookup[d['image_id']] for d in order[0]]

    def validation_rows(self):
        if self.fixture:return self.control_rows()
        return sorted([r for r in self.overlay['rows'] if r['role']=='validation'],key=lambda r:r['image_id'])

    def predict(self,rows,stage,seeds=None):
        begin=time.monotonic()
        device=self.device if stage=='train' else self.eval_device
        batch=load_batch(self.data,rows,device,seeds=seeds if stage=='train' else None)
        self.timings={'data_loading_seconds':time.monotonic()-begin};begin=time.monotonic()
        base=self.anchor(batch['rgb'],batch['guides'],device)
        if str(device).startswith('cuda'):torch.cuda.synchronize(device)
        self.timings['feature_seconds']=time.monotonic()-begin
        model=self.model if stage=='train' else self.eval_model
        if model is None:self.eval_model=make_eval(self.model,self.eval_device);model=self.eval_model
        seeds=seeds or [view_seed(r['image_id']) for r in rows]
        begin=time.monotonic();refined=model.refine(base)
        atlas=canonicalize(batch['rgb'],refined['masks'],batch['masks'][:,2:3])
        quality=model.quality(model.quality_input(base,refined,atlas,batch['guides']))
        if str(device).startswith('cuda'):torch.cuda.synchronize(device)
        self.timings['geometry_seconds']=time.monotonic()-begin;begin=time.monotonic()
        views=[model.texture_view(atlas,[int(s)*2+i for s in seeds]) for i in (0,1)]
        if str(device).startswith('cuda'):torch.cuda.synchronize(device)
        self.timings['texture_seconds']=time.monotonic()-begin
        out=dict(refined=refined,atlas=atlas,quality=quality,views=views)
        return batch,base,out,seeds

    def record_resources(self,records,started):
        timings=getattr(self,'timings',{})
        add(records,'pipeline_timings',unit=self.store.unit+1,stage=self.cursor.get('record_stage',self.cursor['phase']),feature_seconds=timings.get('feature_seconds'),transfer_seconds=None,
            geometry_seconds=timings.get('geometry_seconds'),texture_seconds=timings.get('texture_seconds'),record_seconds=time.monotonic()-started,
            publish_wait_seconds=0,evaluation_seconds=None,data_loading_seconds=timings.get('data_loading_seconds'),backward_optimizer_seconds=timings.get('backward_optimizer_seconds'),previous_commit_seconds=getattr(self,'last_commit_seconds',None))
        mem={}
        for i in (0,1):
            available=torch.cuda.is_available() and torch.cuda.device_count()>i
            mem[f'gpu{i}_allocated_bytes']=torch.cuda.memory_allocated(i) if available else 0
            mem[f'gpu{i}_reserved_bytes']=torch.cuda.memory_reserved(i) if available else 0
        try:
            import psutil
            rss=psutil.Process().memory_info().rss
        except ImportError:rss=None
        logical=sum(m['bytes'] for m in self.store.closed.values())+sum(v['bytes'] for v in self.store.csv.values())+sum(v['bytes'] for v in self.store.arrays.values())
        add(records,'resource_samples',unit=self.store.unit+1,**mem,host_rss_bytes=rss,working_free_bytes=shutil.disk_usage(self.root).free,logical_artifact_bytes=logical,staging_bytes=None)
        assert logical<15000000000 and shutil.disk_usage(self.root).free>2000000000,'Storage reserve/15 GB logical limit reached; preserve and publish'

    def commit_observation(self,records,arrays,first,count,started,seal=False):
        self.cursor['last_first_sample']=first;self.cursor['sample']=first+count
        self.record_resources(records,started)
        commit_start=time.monotonic();self.store.commit(self.model,self.optimizer,self.cursor,records,arrays,seal=seal);self.last_commit_seconds=time.monotonic()-commit_start
        atomic(self.root/'STATUS.json',dict(phase=self.cursor['phase'],epoch=self.cursor['epoch'],update=self.cursor['update'],unit=self.store.unit,samples=self.cursor['sample'],protocol=self.protocol))

    def step(self):
        start=time.monotonic();c=self.cursor;phase=c['phase'];first=c['sample'];mean=getattr(self,'mean',None)
        c['record_stage']=phase;c['record_epoch']=0 if phase in ('mean','baseline') else c['epoch']
        if mean is None and (self.store.root/'controls/mean.npy').exists():self.mean=np.load(self.store.root/'controls/mean.npy',allow_pickle=False);mean=self.mean
        if phase in ('mean','baseline','validate'):
            rows=self.control_rows() if phase=='mean' else self.validation_rows();index=c['index']
            if self.eval_model is None:self.eval_model=make_eval(self.model,self.eval_device)
            with torch.no_grad():
                batch,base,out,seeds=self.predict([rows[index]],phase)
                records,arrays,info=observe(out,base,batch,seeds,first,phase,None,c['record_epoch'],c['update'],mean=mean,median=c.get('median'))
            c['index']+=1
            if phase=='mean':
                target,weight=identity_target(out['atlas']);c['mean_sum']+=(target.astype('float64')*weight).sum(0);c['mean_count']+=weight.sum(0)
                for j in range(3):
                    if info[0]['quality_available'][j]:c['median_targets'][j].append(info[0]['quality_target'][j])
            else:c['baseline' if phase=='baseline' else 'evaluation'].extend(info)
            last=c['index']==len(rows)
            if last:
                if phase=='mean':
                    self.mean=np.divide(c['mean_sum'],c['mean_count'],out=np.zeros_like(c['mean_sum']),where=c['mean_count']>0).astype('float32')
                    for name,value in [('mean',self.mean),('mean_count',c['mean_count'].astype('float32'))]:
                        path=self.store.root/f'controls/{name}.npy';native(path,value);self.store.register(path)
                    c['median']=[float(np.median(v)) if v else None for v in c['median_targets']]
                    path=self.store.root/'controls/quality_median.json';atomic(path,dict(median=c['median'],counts=[len(v) for v in c['median_targets']],point_targets='human_training_only'));self.store.register(path)
                    c['mean_sum']=c['mean_count']=None;c['phase']='baseline';c['index']=0
                elif phase=='baseline':
                    c['G0']=summary(c['baseline']);c['baseline']=[];c['phase']='train';c['index']=0;self.eval_model=None
                    path=self.store.root/'controls/G0_validation.json';atomic(path,c['G0']);self.store.register(path)
                else:
                    extra,value=metric_records(c['evaluation'],'validation',None,c['epoch'],c['median'])
                    for name,items in extra.items():records.setdefault(name,[]).extend(items)
                    score,geometry,collapse,reasons=selection(value,c['G0']);chosen=score is not None and (c['best'] is None or score<c['best']['score'])
                    add(records,'selection',epoch=c['epoch'],validation_mean_dice=value['mean_dice'],baseline_mean_dice=c['G0']['mean_dice'],validation_human_point_MAE=value['human_point_MAE'],baseline_human_point_MAE=c['G0']['human_point_MAE'],
                        invalid_atlas_fraction=value['invalid_atlas_fraction'],baseline_invalid_atlas_fraction=c['G0']['invalid_atlas_fraction'],projector_std_median=value['std_median'],effective_rank=value['rank'],geometry_eligible=geometry,collapse_eligible=collapse,
                        selection_objective=score,selected_so_far=chosen,rejection_reasons=';'.join(reasons))
                    path=self.store.root/f'epochs/epoch_{c["epoch"]:03d}.pt';native(path,cpu(self.model.state_dict()));self.store.register(path)
                    if chosen:c['best']=dict(epoch=c['epoch'],score=score,path=path.relative_to(self.store.root).as_posix())
                    c['history'].append(dict(epoch=c['epoch'],score=score,**value));c['evaluation']=[];c['index']=0
                    for image_id,v in c['exposure'].items():add(records,'source_exposure',epoch=c['epoch'],image_id=image_id,**v)
                    c['exposure']={}
                    if c['epoch']==self.cfg['epochs']:c['phase']='endpoint_setup';c['endpoint']='selected' if c['best'] else 'final';c['endpoint_index']=0
                    else:c['epoch']+=1;c['phase']='train';c['batch']=0
                    self.eval_model=None
            self.commit_observation(records,arrays,first,1,start,seal=last)
            return
        if phase=='train':
            order,_=schedule(self.train,1,c['epoch']);draws=order[c['batch']];rows=[self.lookup[d['image_id']] for d in draws];seeds=[d['seed'] for d in draws]
            batch,base,out,seeds=self.predict(rows,'train',seeds);previous=c['update']
            backward_start=time.monotonic();records=gradient_step(self.model,self.optimizer,out,base,batch,seeds,c['epoch'],previous,c['counters']);self.timings['backward_optimizer_seconds']=time.monotonic()-backward_start
            obs,arrays,info=observe(out,base,batch,seeds,first,'train',None,c['epoch'],previous,mean=self.mean,median=c['median'])
            if records['batch_objectives'][0]['texture_step_skipped']:
                for item in info:item['SSL_valid']=False
                for item in obs['sample_observations']:item['SSL_valid']=False;item['SSL_skip_reason']='batch_requires_4_images_2_components'
            for name,items in obs.items():records.setdefault(name,[]).extend(items)
            records['batch_objectives'][0]['mean_predictor_huber_sum']=sum(r['mean_num'] for r in info)
            records['batch_objectives'][0]['mean_predictor_huber_count']=sum(r['masked_den'] for r in info)
            for draw,item in zip(draws,info):
                v=c['exposure'].setdefault(draw['image_id'],dict(geometry_draws=0,texture_views=0,padding_draws=0,SSL_valid_draws=0,SSL_skipped_draws=0,first_update=previous+1,last_update=previous+1))
                v['geometry_draws']+=1;v['texture_views']+=2;v['padding_draws']+=int(draw['padding']);v['SSL_valid_draws']+=int(item['SSL_valid']);v['SSL_skipped_draws']+=int(not item['SSL_valid']);v['last_update']=previous+1
            c['training_draws']+=len(info);c['training_SSL_valid']+=sum(r['SSL_valid'] for r in info)
            diagnostic=summary(info);norms=[float(np.linalg.norm(r['embedding'])) for r in info if r['SSL_valid']]
            add(records,'latent_diagnostics',update=previous+1,stage='train',eligible_images=diagnostic['eligible_images'],components=diagnostic['components'],projector_std_min=diagnostic['std_min'],projector_std_median=diagnostic['std_median'],dimensions_below_floor=diagnostic['below'],covariance_offdiag_energy=diagnostic['covariance'],effective_rank=diagnostic['rank'],embedding_norm_min=min(norms) if norms else None,embedding_norm_max=max(norms) if norms else None,finite=True)
            c['update']+=1;c['batch']+=1
            add(records,'train_updates',update=c['update'],epoch=c['epoch'],next_draw=c['batch']*8,geometry_optimizer_steps=c['counters']['geometry'],texture_optimizer_steps=c['counters']['texture'],quality_optimizer_steps=c['counters']['quality'],
                geometry_lr=self.optimizer.param_groups[0]['lr'],texture_lr=self.optimizer.param_groups[1]['lr'],quality_lr=self.optimizer.param_groups[2]['lr'],gradient_finite=True,update_committed=True,unit_sha256=None)
            last=c['batch']==self.cfg['batches_per_epoch']
            if last:c['phase']='validate';c['index']=0;self.eval_model=None
            self.commit_observation(records,arrays,first,len(rows),start,seal=last)
            return
        return self.endpoint_step()

    def endpoint_model(self):
        c=self.cursor
        if self.eval_model is None:
            self.eval_model=make_eval(self.model,self.eval_device)
            chosen=c['best']['path'] if c['endpoint']=='selected' else f'epochs/epoch_{self.cfg["epochs"]:03d}.pt'
            self.eval_model.load_state_dict(torch.load(self.store.root/chosen,map_location=self.eval_device,weights_only=True))
        return self.eval_model

    def endpoint_rows(self):
        if self.fixture:return self.control_rows()
        order={'train':0,'validation':1,'test':2}
        return sorted(self.overlay['rows'],key=lambda r:(order[r['role']],r['image_id']))

    def endpoint_step(self):
        from endpoint_fit import finish_endpoint,probe,thumbnail
        start=time.monotonic();c=self.cursor;phase=c['phase'];records={}
        checkpoint_epoch=c['best']['epoch'] if c['endpoint']=='selected' and c['best'] else self.cfg['epochs']
        checkpoint_update=checkpoint_epoch*self.cfg['batches_per_epoch']
        c['record_epoch']=checkpoint_epoch
        if phase=='endpoint_setup':
            self.eval_model=None;self.endpoint_model();c['phase']='endpoint';c['endpoint_index']=0;c['endpoint_info']=[]
            add(records,'gate_checks',check='endpoint_selection',stage=c['endpoint'],passed=True,observed=c['best']['epoch'] if c['endpoint']=='selected' else self.cfg['epochs'],required='validation_rule_only_no_test_reselection',evidence_path='csv/selection',fixture_or_measured='fixture' if self.fixture else 'measured')
            self.store.commit(self.model,self.optimizer,c,records,seal=True)
            return
        if phase=='endpoint':
            rows=self.endpoint_rows();row=rows[c['endpoint_index']];self.endpoint_model()
            with torch.no_grad():
                batch,base,out,seeds=self.predict([row],'endpoint')
                obs,arrays,info=observe(out,base,batch,seeds,c['sample'],'endpoint',c['endpoint'],checkpoint_epoch,checkpoint_update,mean=self.mean,dense=True,median=c['median'])
                for observed in obs['sample_observations']:observed['completed_update']=c['update']
                info[0]['thumbnail']=thumbnail(self.root,c['endpoint'],row,batch,out)
            c['endpoint_info'].extend(info);c['endpoint_index']+=1
            last=c['endpoint_index']==len(rows)
            if last:
                extra,value=metric_records(c['endpoint_info'],'endpoint',c['endpoint'],checkpoint_epoch,c['median'])
                for name,items in extra.items():obs.setdefault(name,[]).extend(items)
                finish_endpoint(self,obs,value)
                c['phase']='probes';c['probe_index']=0
            self.commit_observation(obs,arrays,c['sample'],1,start,seal=last)
            return
        if phase=='probes':
            cohort=self.control_rows()[:2] if self.fixture else sorted([r for r in self.overlay['rows'] if r['role']=='test'],key=lambda r:r['image_id'])
            variants=self.cfg['probe_variants'] if not self.fixture else ['grayscale','reverse_bands']
            total=len(cohort)*len(variants)
            if c['probe_index']<total:
                row=cohort[c['probe_index']//len(variants)];variant=variants[c['probe_index']%len(variants)]
                self.endpoint_model()
                with torch.no_grad():batch,base,out,seeds=self.predict([row],'probe');batch,base,out=probe(self,batch,base,out,seeds,variant)
                obs,arrays,info=observe(out,base,batch,seeds,c['sample'],'intervention',c['endpoint'],checkpoint_epoch,checkpoint_update,mean=self.mean,median=c['median'])
                for observed in obs['sample_observations']:observed['completed_update']=c['update']
                # The approved probe allowance preserves geometry native + full
                # scalar/vector records; raw texture bodies are fit/val/endpoints only.
                keep={'geometry_logits','points','quality','row_parameters','row_valid'}
                arrays={k:v for k,v in arrays.items() if k in keep}
                obs['spatial_maps_index']=[r for r in obs['spatial_maps_index'] if r['quantity'] in keep]
                for r in obs.get('masked_patch_plan',[]):r['native_prediction_key']=r['native_target_key']=r['native_weight_key']=None
                original=next(r for r in c['endpoint_info'] if r['image_id']==row['image_id']);item=info[0]
                ea,eb=np.asarray(original['embedding']),np.asarray(item['embedding']);distance=1-float(ea@eb/max(np.linalg.norm(ea)*np.linalg.norm(eb),1e-12))
                available=np.array(item['quality_available'])&np.array(original['quality_available'])
                quality_delta=float(np.mean((np.abs(np.array(item['quality'])-np.array(item['quality_target']))-np.abs(np.array(original['quality'])-np.array(original['quality_target'])))[available])) if available.any() else None
                probe_meta=out['probe_meta']
                add(obs,'interventions',sample=c['sample'],variant=variant,base_sample=original['sample'],mask_dice_change=float(np.mean(item['dice'])-np.mean(original['dice'])),point_MAE_change=item['point_error']-original['point_error'] if item['human'] else None,
                    embedding_cosine_change=distance,quality_error_change=quality_delta,
                    valid_fraction_change=item['valid_fraction']-original['valid_fraction'],scope='synthetic_structural_diagnostic_not_new_tyres_or_damage_truth',
                    probe_available=probe_meta['available'],source_patch=probe_meta['source_patch'],source_patch_mapping=json.dumps(probe_meta['mapping']),unavailable_reason=probe_meta['reason'])
                c['probe_index']+=1
                self.commit_observation(obs,arrays,c['sample'],1,start,seal=c['probe_index']==total)
                return
            if c['endpoint']=='selected':c['endpoint']='final';c['phase']='endpoint_setup';self.eval_model=None
            else:c['phase']='finish'
            self.store.commit(self.model,self.optimizer,c,{},seal=True)
            return
        if phase=='finish':
            from endpoint_fit import final_report
            final_report(self,records)
            c['phase']='completed';self.store.commit(self.model,self.optimizer,c,records,seal=True)
            atomic(self.root/'STATUS.json',dict(phase='completed',update=c['update'],unit=self.store.unit,protocol=self.protocol,samples=c['sample']))
            return
        assert phase=='completed',phase
