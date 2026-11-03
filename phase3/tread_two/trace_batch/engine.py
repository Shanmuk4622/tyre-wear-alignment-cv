"""One resumable TRACE lifecycle; every forward has direct CSV/native evidence."""
import json,time
from pathlib import Path
import numpy as np
import torch
from common import atomic,native,read,digest,file_meta,seed_all,cpu,sha
from config import configuration
from cache import Cache
from data import schedule
from model import Trace
from objective import objective
from targets import standardized
from store import Store,semantic_hash
from schema import add
from telemetry import metadata,view_plan,run_views,observe
from metrics import evaluate,latent,scatter
from resources import peaks_reset,records as resource_records,guard

def transform(batch,variant):
    if variant=='identity':return batch
    b=dict(batch)
    if variant=='brightness_0p85':b['rgb']=(b['rgb']*.85).clamp(0,1)
    elif variant=='grayscale':b['rgb']=b['rgb'].mean(1,keepdim=True).expand(-1,3,-1,-1)
    elif variant in ('reverse_bands','reverse_columns'):
        # Apply the same permutation to image/valid and targets, keeping each
        # descriptor aligned; retain explicit variant in every target pointer.
        axis=1 if variant=='reverse_bands' else 2
        for name in ('rgb','valid'):
            a=b[name];n,c=a.shape[:2];a=a.reshape(n,c,8,64,4,64).permute(0,2,4,1,3,5).flip(axis)
            b[name]=a.permute(0,3,1,4,2,5).reshape(n,c,512,256)
        for name in ('target','weight'):
            a=b[name];b[name]=a.reshape(len(a),8,4,*a.shape[2:]).flip(axis).reshape_as(a)
    else:raise ValueError(variant)
    return b

class Engine:
    def __init__(self,root,cache_root,device='cuda:0',fixture=False):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.cfg=configuration(fixture);self.protocol=digest(self.cfg)
        self.cache=Cache(cache_root);self.cache.verify();self.device=device;self.fixture=fixture
        assert fixture or not self.cache.index['identity']['fixture']
        assert self.cache.index['identity']['protocol']==digest(configuration()),'Prepared target/source/environment identity differs'
        seed_all(1);self.model=Trace().to(device);self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=.001,weight_decay=.0001)
        self.stats=self.cache.statistics(device);self.store=Store(self.root/'run',self.protocol)
        self.previous_commit_seconds=0.
        if (self.store.root/'POINTER.json').exists():
            self.cursor=self.store.restore(self.model,self.optimizer)
            context=read(self.store.root/'RESTORE_CONTEXT.json') if (self.store.root/'RESTORE_CONTEXT.json').exists() else {}
            event=dict(stage='resume',session='fresh_numeric_worker',local_unit=self.store.unit,chosen_unit=self.store.unit,remote_unit=context.get('remote_unit'),revision=context.get('revision'),compatibility=True,action=context.get('action','verified_complete_state_and_prefix_restore'),replayed_incomplete_update=True if self.store.restored_tail_truncated else None)
            self.store.commit(self.model,self.optimizer,self.cursor,{'resume_events':[event]},kind='resume')
        else:
            self.store.initialize_headers();atomic(self.store.root/'CONTRACT.json',self.cfg);self.store.register(self.store.root/'CONTRACT.json')
            self.initial=cpu(self.model.state_dict());native(self.store.root/'controls/initial_trace.pt',self.initial);self.store.register(self.store.root/'controls/initial_trace.pt')
            for name in self.cache.index['files']:self.store.register(self.cache.root/name,'inputs/cache/'+name)
            self.store.register(self.cache.root/'CACHE.json','inputs/cache/CACHE.json')
            self.cursor=dict(phase='baseline',epoch=0,update=0,batch=0,index=0,observation=0,last_first_sample=0,evaluation=[],history=[],best=None,
                primary_seen=0,primary_eligible=0,exposure={},endpoints={},endpoint='final',variant_index=1,services=[],steps_per_epoch=2 if fixture else 49)
            records=metadata(self.cfg)
            add(records,'preflight_gates','initialization',gate='fixed_cache_and_geometry',passed=True,evidence_file='inputs/cache/CACHE.json',evidence_sha256=sha(self.cache.root/'CACHE.json'),actual_Kaggle=Path('/kaggle').exists(),actual_HF=False,detail='Content hashes checked; actual HF continuation certificate is separately bound by coordinator')
            self.store.commit(self.model,self.optimizer,self.cursor,records,seal=True)
        self.initial=torch.load(self.store.root/'controls/initial_trace.pt',map_location='cpu',weights_only=True)
        self.status()

    def status(self):
        c=self.cursor;atomic(self.root/'STATUS.json',dict(phase=c['phase'],epoch=c['epoch'],update=c['update'],unit=self.store.unit,observation=c['observation'],fixture=self.fixture,protocol=self.protocol))

    def role_ids(self,role):
        if self.fixture:return [r['slot'] for r in self.cache.rows if r['role']=='train'][:8]
        return [r['slot'] for r in self.cache.rows if r['role']==role]

    def commit(self,records,arrays=None,seal=False):
        began=time.monotonic();self.store.commit(self.model,self.optimizer,self.cursor,records,arrays,seal=seal)
        self.previous_commit_seconds=time.monotonic()-began;self.status()

    def step(self):
        c=self.cursor;phase=c['phase'];epoch=c['epoch'];began=time.monotonic();records={};arrays=None;seal=False
        assert phase!='completed'
        peaks_reset()
        if phase=='train':
            batches=schedule(self.cache.rows,1,epoch,self.fixture);ids,flags=batches[c['batch']]
        elif phase in ('baseline','validate'):
            ids=self.role_ids('validation')[c['index']:c['index']+8];flags=[False]*len(ids)
        elif phase=='endpoint':
            pool=self.role_ids('train') if self.fixture else list(range(len(self.cache.rows)))
            ids=pool[c['index']:c['index']+8];flags=[False]*len(ids)
        elif phase=='probe':
            ids=self.role_ids('test')[c['index']:c['index']+8];flags=[False]*len(ids)
        else:raise ValueError(phase)
        if ids:
            variant=self.cfg['probes'][c['variant_index']] if phase=='probe' else 'identity'
            batch=transform(self.cache.batch(ids,self.device),variant)
            plans=view_plan(batch,ids,epoch if phase=='train' else 0,phase=='train',variant)
            self.model.train(phase=='train');self.optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(phase=='train'):
                outputs=run_views(self.model,batch,plans)
                loss,parts,audit=objective(outputs,standardized(batch['target'],self.stats),batch['weight'])
                assert torch.isfinite(loss),'nonfinite TRACE loss'
                if phase=='train':
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.,error_if_nonfinite=True)
                    post=torch.linalg.vector_norm(torch.stack([p.grad.detach().norm() for p in self.model.parameters() if p.grad is not None]))
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.parameters()),'nonfinite/disconnected gradient'
                    self.optimizer.step();c['update']+=1
                    add(records,'update_gradients',phase,epoch=epoch,update=c['update'],group='texture_only',pre_clip_norm=float(norm),post_clip_norm=float(post),clip_limit=1.,finite=True,learning_rate=.001,optimizer_step=c['update'])
                    add(records,'objective_terms',phase,epoch=epoch,update=c['update'],**{k:float(v.detach()) for k,v in parts.items()},total=float(loss.detach()),local_weight=1.,masked_weight=1.,pair_weight=.1,variance_weight=.1,covariance_weight=.01,eligible_rows=int(audit[0]['eligible'].sum()),diversity_available=audit[0]['diversity_available'])
            # The arrays/predictions describe the pre-update forward. Explicit
            # parameter update = completed update - 1 for training; review binds
            # this convention and never labels them post-update predictions.
            write_stage='probe_'+variant if phase=='probe' else 'endpoint_'+c['endpoint'] if phase=='endpoint' else phase
            rec,arrays,samples=observe(self.store,self.cache,batch,outputs,plans,ids,flags,self.stats,c,write_stage,variant)
            for f,rr in rec.items():records.setdefault(f,[]).extend(rr)
            c['last_first_sample']=c['observation'];c['observation']+=len(ids)
            if phase=='train':
                for s in samples:
                    e=c['exposure'].setdefault(s['image_id'],dict(primary_draws=0,padding_draws=0,eligible_draws=0,skipped_draws=0,views=0))
                    e['primary_draws' if s['primary'] else 'padding_draws']+=1;e['eligible_draws' if s['eligible'] else 'skipped_draws']+=1;e['views']+=2
                    if s['primary']:c['primary_seen']+=1;c['primary_eligible']+=int(s['eligible'])
                c['batch']+=1
                if c['batch']==len(batches):c['phase']='validate';c['index']=0;c['evaluation']=[];seal=True
            elif phase=='probe':
                for s in samples:
                    base=c['final_lookup'][s['image_id']]
                    for space in ['projected','local_pooled','pooled']:
                        a=np.asarray(s[space]);b=np.asarray(base[space]);den=np.linalg.norm(a)*np.linalg.norm(b)
                        add(records,'intervention_metrics','probe',epoch=epoch,update=c['update'],image_id=s['image_id'],variant=variant,plan_sha256=digest(dict(variant=variant,slot=next(r['slot'] for r in self.cache.rows if r['image_id']==s['image_id']))),space=space,
                            cosine_change=float(1-a@b/den) if den>=1e-8 else None,l2_change=float(np.linalg.norm(a-b)),local_loss=s['local_loss'],masked_loss=s['model_common'],support_changed=s['support']!=base['support'])
                c['index']+=len(ids)
            else:c['evaluation'].extend(samples);c['index']+=len(ids)
            add(records,'numerical_events',phase,epoch=epoch,update=c['update'],event='forward_and_optimizer_finite',tensor='loss_all_new_parameters',finite=True,minimum=float(loss.detach()),maximum=float(loss.detach()),action='committed_complete_unit',available=True)
            del outputs,batch,loss
        else:
            if phase=='baseline':
                self.summarize(records,'baseline',c['evaluation']);c['phase']='train';c['epoch']=1;c['index']=0;c['evaluation']=[]
            elif phase=='validate':
                summary=self.summarize(records,'validate',c['evaluation'])
                path=self.store.root/f'epochs/epoch_{epoch:03d}.pt';native(path,cpu(self.model.state_dict()));self.store.register(path)
                if summary['eligible'] and (c['best'] is None or summary['score']<c['best']['score']):c['best']=dict(epoch=epoch,score=summary['score'],path=path.relative_to(self.store.root).as_posix())
                c['history'].append(dict(epoch=epoch,**{k:v for k,v in summary.items() if k not in ('spectra','within')}))
                for row in self.cache.rows:
                    if row['role']=='train':
                        e=c['exposure'].get(row['image_id'],dict(primary_draws=0,padding_draws=0,eligible_draws=0,skipped_draws=0,views=0))
                        add(records,'source_exposure','train',epoch=epoch,update=c['update'],image_id=row['image_id'],source_sha256=row['image_sha256'],**e)
                self.parameters(records)
                c['evaluation']=[];c['index']=0;c['batch']=0;c['primary_seen']=0;c['primary_eligible']=0;c['exposure']={}
                if epoch<self.cfg['epochs']:c['epoch']+=1;c['phase']='train'
                else:c['phase']='endpoint';c['endpoint']='final';self.export('final',records)
            elif phase=='endpoint':
                endpoint=c['endpoint'];summary=self.summarize(records,'endpoint_'+endpoint,c['evaluation'])
                # Compact endpoint cases retain spectrum/retrieval inputs; token
                # arrays remain native, not duplicated indefinitely in cursor.
                compact=[{k:v for k,v in s.items() if k not in ('encoded','full_context')} for s in c['evaluation']]
                c['endpoints'][endpoint]=compact
                if endpoint=='final':c['final_lookup']={s['image_id']:s for s in compact}
                c['evaluation']=[];c['index']=0
                if endpoint=='final' and c['best']:
                    c['endpoint']='selected';self.model.load_state_dict(torch.load(self.store.root/c['best']['path'],map_location=self.device,weights_only=True));self.export('selected',records)
                else:
                    if not c['best']:add(records,'endpoint_records','endpoint',endpoint='selected',available=False,provenance='No checkpoint passed all original TRACE gates',scientific_status='unavailable')
                    self.model.load_state_dict(torch.load(self.store.root/f'epochs/epoch_{self.cfg["epochs"]:03d}.pt',map_location=self.device,weights_only=True))
                    c['phase']='probe';c['variant_index']=1
            elif phase=='probe':
                c['variant_index']+=1;c['index']=0
                if c['variant_index']==len(self.cfg['probes']):
                    self.retrieval(records)
                    c['phase']='completed'
                    result=dict(status='completed',protocol=self.protocol,fixture=self.fixture,epochs=self.cfg['epochs'],updates=c['update'],selected=c['best'],scientific_status='research_only_unpromoted',
                                original_GEO_selected=None,existing_photos_only=True,cache_hash=self.cache.index['scientific_hash'],observations=c['observation'],history=c['history'])
                    atomic(self.store.root/'RESULT.json',result);self.store.register(self.store.root/'RESULT.json')
            seal=True
        for row in resource_records(self.root,phase,self.store.unit+1):records.setdefault('resource_measurements',[]).append(row)
        add(records,'timings',phase,epoch=epoch,update=c['update'],session='numeric_worker',phase='step_excluding_current_commit',start_monotonic=began,duration_seconds=time.monotonic()-began,images=len(ids),complete=True)
        self.commit(records,arrays,seal=seal)
        if self.store.unit%16==0:guard(self.root.parent,self.cfg)

    def summarize(self,records,stage,samples):
        c=self.cursor;fraction=c['primary_eligible']/c['primary_seen'] if c['primary_seen'] else 1.
        summary=evaluate(samples,self.cfg,fraction)
        add(records,'eligibility',stage,epoch=c['epoch'],update=c['update'],**summary['gates'],eligible=summary['eligible'],objective=summary['score'],rejection_reasons=summary['rejection'])
        roles=sorted({s['role'] for s in samples})
        for role in roles:
            subset=[s for s in samples if s['role']==role];value=evaluate(subset,self.cfg,fraction)
            for metric,num in [('masked_model_huber',value['model']),('local_huber',value['local']),('paired_cosine',value['cosine']),('common_coverage',value['coverage']),('eligible_fraction',value['eligible_images']/len(subset))]:
                add(records,'epoch_metrics',stage,epoch=c['epoch'],update=c['update'],role=role,task='texture',metric=metric,value=num,images=len(subset),available=num is not None)
            eligible=[s for s in subset if s['eligible']]
            for component in sorted({s['component'] for s in subset}):
                group=[s for s in subset if s['component']==component];g=evaluate(group,self.cfg,fraction)
                add(records,'component_metrics',stage,epoch=c['epoch'],update=c['update'],role=role,component=component,task='texture',metric='masked_model_huber',value=g['model'],images=len(group),available=g['model'] is not None)
            for space in ['projected','local_pooled','pooled','context_mean']:
                z=np.array([s[space] for s in eligible]);st=latent(z)
                if len(z)<2:continue
                decomp=scatter(z,[s['component'] for s in eligible])
                add(records,'scatter_decomposition',stage,epoch=c['epoch'],update=c['update'],role=role,space=space,components=len(set(s['component'] for s in eligible)),total_scatter=decomp['total'],within_scatter=decomp['within'],between_scatter=decomp['between'],between_fraction=decomp['fraction'],decomposition_error=decomp['error'])
                for centered,body in [(False,z),(True,decomp['within_vectors'])]:
                    ss=latent(body)
                    for dim,eigen in enumerate(ss['eigenvalues']):
                        total=sum(ss['eigenvalues']);add(records,'representation_spectra',stage,epoch=c['epoch'],update=c['update'],role=role,space=space,component_centered=centered,dimension=dim,eigenvalue=eigen,fraction=eigen/total if total else 0,image_count=len(body),std_median=ss['std'],effective_rank=ss['rank'])
            for layer in ['encoded','full_context']:
                for tile in range(32):
                    a=np.array([s[layer][tile] for s in eligible if s['support'][tile]]);st=latent(a)
                    add(records,'local_token_statistics',stage,epoch=c['epoch'],update=c['update'],role=role,layer=layer,tile=tile,image_count=len(a),feature_count=48,effective_rank=st['rank'],std_median=st['std'],leading_fraction=st['leading'])
        return summary

    def parameters(self,records):
        for name,value in self.model.state_dict().items():
            a=value.detach().double();delta=a.cpu()-self.initial[name].double()
            add(records,'parameter_statistics','epoch',epoch=self.cursor['epoch'],update=self.cursor['update'],parameter=name,elements=a.numel(),l2=float(a.norm()),max_absolute=float(a.abs().max()),mean=float(a.mean()),std=float(a.std(unbiased=False)),finite=bool(torch.isfinite(a).all()),observation_frequency='every_epoch')
            add(records,'parameter_deltas','epoch',epoch=self.cursor['epoch'],update=self.cursor['update'],group=name,reference='initial_trace',l2_delta=float(delta.norm()),max_absolute_delta=float(delta.abs().max()),unchanged=bool((delta==0).all()),elements=delta.numel())

    def export(self,endpoint,records):
        c=self.cursor;epoch=c['best']['epoch'] if endpoint=='selected' else self.cfg['epochs'];path=self.store.root/f'exports/{endpoint}.pt'
        native(path,dict(trace=cpu(self.model.state_dict()),geometry=self.cfg['geometry'],statistics=cpu(self.stats),cache_hash=self.cache.index['scientific_hash'],epoch=epoch,parameter_update=epoch*(2 if self.fixture else 49),status='research_only',selected=endpoint=='selected',protocol=self.protocol))
        self.store.register(path)
        add(records,'endpoint_records','endpoint',endpoint=endpoint,available=True,epoch_selected=epoch,parameter_update=epoch*(2 if self.fixture else 49),file=path.relative_to(self.store.root).as_posix(),sha256=sha(path),provenance='TRACE validation rule; inherited geometry remains GEO-unselected',scientific_status='research_only')

    def retrieval(self,records):
        for endpoint,cases in self.cursor['endpoints'].items():
            train=[s for s in cases if s['role']=='train'];bank=np.array([s['pooled'] for s in train],dtype='float64');norm=np.linalg.norm(bank,axis=1);bank=bank/norm.clip(1e-8)[:,None]
            byid={r['image_id']:r for r in self.cache.rows}
            for s in cases:
                q=np.array(s['pooled']);qnorm=np.linalg.norm(q);dist=1-bank@(q/max(qnorm,1e-8));source=byid[s['image_id']]['image_sha256']
                allowed=np.array([byid[t['image_id']]['image_sha256']!=source for t in train]);dist[~allowed]=np.inf
                for j in np.where(~allowed)[0]:add(records,'retrieval_neighbors','retrieval_'+endpoint,image_id=s['image_id'],query_sha256=source,neighbor_id=train[j]['image_id'],neighbor_sha256=source,neighbor_component=train[j]['component'],excluded_self=True,permitted=False)
                for rank,j in enumerate(np.argsort(dist)[:5],1):
                    if not np.isfinite(dist[j]) or qnorm<1e-8 or norm[j]<1e-8:continue
                    add(records,'retrieval_neighbors','retrieval_'+endpoint,image_id=s['image_id'],query_sha256=source,neighbor_id=train[j]['image_id'],neighbor_sha256=byid[train[j]['image_id']]['image_sha256'],neighbor_component=train[j]['component'],rank=rank,distance=float(dist[j]),excluded_self=False,permitted=True)
