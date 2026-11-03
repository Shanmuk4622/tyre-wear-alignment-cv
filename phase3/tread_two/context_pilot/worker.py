"""One writer, one measured head; GPU 0 fits and GPU 1 evaluates at barriers."""
import argparse
import copy
import math
import os
import signal
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from common import HERE, atomic, capacity, cpu, digest, event, native, read, restore_rng, rng_state, seed_all, sha
from data_atlas import bank_cases, batch, dataset, load_bank, schedule, source_hashes
from objective import objective
from model_atlas import TreadTwoAtlas
from schemas import SCHEMAS
from spec import evaluation_rows
from store import Store
from telemetry import coefficients, dense_once, learner_sha, observations, splat, summarize

STOP = False


def stop(*_):
    global STOP
    STOP = True


def merge(target, source):
    for key, rows in source.items():
        target.setdefault(key, []).extend(rows)


def host_memory():
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in
                        ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
        value = Counters()
        value.cb = ctypes.sizeof(value)
        current = ctypes.windll.kernel32.GetCurrentProcess
        current.restype = wintypes.HANDLE
        info = ctypes.windll.psapi.GetProcessMemoryInfo
        info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        info.restype = wintypes.BOOL
        try:
            return int(value.PeakWorkingSetSize) if info(current(), ctypes.byref(value), value.cb) else None
        except OSError:
            return None  # Optional host resource counter, never a reason to lose training progress.
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


class Engine:
    def __init__(self, cfg, folder, fault_stage=None):
        self.cfg, self.folder = cfg, Path(folder)
        self.recipe = cfg['contract']
        self.fault_stage = fault_stage
        assert fault_stage is None or self.recipe['mode'] == 'engineering_fixture'
        self.store = Store(folder, cfg['contract'], SCHEMAS)
        self.plan, self.queries = dataset(cfg['data_root'])
        ordered = sorted(self.plan['rows'], key=lambda row: row['image_id'])
        self.registry = {row['image_id']: i for i, row in enumerate(ordered)}
        seed_all(self.recipe['seed'])
        torch.set_num_threads(4)
        bank, labels = load_bank()
        self.model = TreadTwoAtlas(bank, labels, cfg['weights'],cfg['base_weights']).to(self.recipe['train_device'])
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        assert sum(p.numel() for p in parameters) == self.recipe['trainable_parameters']
        self.optimizer = torch.optim.AdamW(parameters, lr=self.recipe['lr'], weight_decay=self.recipe['weight_decay'],
                                            betas=(.9, .999), eps=1e-8, foreach=False, fused=False)
        total = self.recipe['epochs'] * self.recipe['batches_per_epoch']
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: .5 * (1 + math.cos(math.pi * min(step, total) / total)))
        self.scaler = torch.amp.GradScaler('cuda', enabled=False)
        self.eval_model = None
        self.anchor_start = {key: value.detach().cpu().clone() for key, value in self.model.base.state_dict().items()}
        if self.store.state:
            self.state = copy.deepcopy(self.store.state)
            assert torch.equal(self.state['model']['base.bank_rgb'], bank) and torch.equal(self.state['model']['base.bank_labels'], labels)
            self.model.load_state_dict(self.state['model'], strict=True)
            self.optimizer.load_state_dict(self.state['optimizer'])
            self.scheduler.load_state_dict(self.state['scheduler'])
            self.scaler.load_state_dict(self.state['scaler'])
            restore_rng(self.state['rng'])
        else:
            self.state = dict(protocol=cfg['protocol'], updates=0, epoch0=0, cursor=0, phase='train',
                              eval_cursor=0, sample_cursor=0, train_records=[], eval_records=[], dense_seen=[], best=None,
                              diagnostic_endpoint=0, diagnostic_variant=0, last_validation=None,
                              phase_weights_sha256=None, epoch_orders={}, fp32=True, scaler_enabled=False)
            self.capture_training_state()
            cases = bank_cases()
            ids = {case['source_image_id'] for case in cases}
            query_ids = {row['image_id'] for row in self.queries}
            tables = {'source_images': [], 'bank_cases': [], 'bank_pixels': [], 'coefficients': coefficients(self.model, cfg['run'], 0)}
            for row in ordered:
                tables['source_images'].append(dict(run=cfg['run'], image_index=self.registry[row['image_id']],
                    image_id=row['image_id'], image_sha256=row['image_sha256'], class_index=row['class_index'], role=row['role'],
                    domain=row['domain'], physical_tyre_id=row['physical_tyre_id'], source_group=row['physical_tyre_id'] or row['source_video'],
                    image_path=row['image_path'], point_kind=row['point_kind'], is_bank=int(row['image_id'] in ids), is_query=int(row['image_id'] in query_ids)))
            for case in cases:
                tables['bank_cases'].append(dict(run=cfg['run'], prototype_index=int(case['prototype_index']), class_index=int(case['class_index']),
                    image_index=self.registry[case['source_image_id']], source_image_id=case['source_image_id'], source_tyre_id=case['source_tyre_id'],
                    atlas_patch_index=int(case['atlas_patch_index']), valid_fraction=float(case['valid_fraction']), clipped_fraction=float(case['clipped_source_rows_fraction']),
                    review_policy='NB10 training-only diversity proposal; source-SHA self exclusion; all386 queries',
                    risk_note='marker/paint/glare/weak detail may be shortcuts; no physical groove annotations'))
            values = bank.numpy()
            tables['bank_pixels'] = [dict(run=cfg['run'], prototype_index=p, channel=c, row=y, column=x, value=float(values[p, c, y, x]))
                                     for p in range(18) for c in range(3) for y in range(64) for x in range(64)]
            self.store.commit(self.state, tables, artifacts={'INITIALIZATION.json': dict(learner_sha256=learner_sha(self.model),
                trainable_parameters=self.recipe['trainable_parameters'], mode=self.recipe['mode'], seed=self.recipe['seed'], anchor_sha256=self.recipe['anchor_weights_sha256'],
                exact_bank_sha256=sha(HERE / 'assets/bank_rgb.npy'), class_support_tyres=[1,3,4])})
            self.state = copy.deepcopy(self.store.state)
        self.assert_anchor()

    def assert_anchor(self):
        assert not self.model.base.training and all(not p.requires_grad and p.grad is None for p in self.model.base.parameters())
        assert all(torch.equal(value.detach().cpu(), self.anchor_start[key]) for key, value in self.model.base.state_dict().items())

    def capture_training_state(self):
        self.state.update(model=cpu(self.model.state_dict()), optimizer=cpu(self.optimizer.state_dict()),
                          scheduler=self.scheduler.state_dict(), scaler=self.scaler.state_dict(), rng=rng_state())

    def commit(self, tables, artifacts=None, partition=None):
        fault = self.fault_stage if self.state['updates'] == 1 and self.store.units[-1]['updates'] == 0 else None
        self.store.commit(self.state, tables, partition=self.state['epoch0'] + 1 if partition is None else partition, artifacts=artifacts, fault=fault)
        self.state = copy.deepcopy(self.store.state)

    def checkpoint(self, epoch):
        return dict(model=cpu(self.model.state_dict()), epoch=epoch, update=self.state['updates'],
                    learner_sha256=learner_sha(self.model), protocol=self.cfg['protocol'], mode=self.recipe['mode'])

    def train_update(self):
        restore_rng(self.state['rng'])
        self.model.to(self.recipe['train_device']).train()
        epoch0, cursor = self.state['epoch0'], self.state['cursor']
        order = schedule(self.queries, self.recipe['seed'], epoch0, self.recipe['batches_per_epoch'], self.recipe['batch'])
        key = str(epoch0)
        if key not in self.state['epoch_orders']:
            self.state['epoch_orders'][key] = order
        assert self.state['epoch_orders'][key] == order
        draws = order[cursor]
        rows = [self.queries[draw['index']] for draw in draws]
        images, labels, factors = batch(self.cfg['data_root'], rows, self.recipe['train_device'], draws)
        before = learner_sha(self.model)
        if images.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(images.device)
            torch.cuda.synchronize(images.device)
        begin = time.monotonic()
        self.optimizer.zero_grad(set_to_none=True)
        out = self.model(images, factors, source_hash=source_hashes(rows,images.device))
        paired=out['_paired']
        loss,parts=objective(out,paired,labels)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss; last completed update remains the recovery state')
        loss.backward()
        gradients = []
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), 'Nonfinite/missing trainable gradient'
                value = parameter.grad.detach()
                gradients.append(dict(run=self.cfg['run'], epoch=epoch0 + 1, update=self.state['updates'] + 1, parameter=name,
                    elements=value.numel(), norm=float(value.norm()), mean=float(value.mean()), minimum=float(value.min()), maximum=float(value.max()), finite=1))
        self.assert_anchor()
        norm = float(torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], self.recipe['clip_norm'], error_if_nonfinite=True))
        clipped = math.sqrt(sum(float(p.grad.norm())**2 for p in self.model.parameters() if p.requires_grad))
        lr = self.optimizer.param_groups[0]['lr']
        self.optimizer.step()
        assert all(torch.isfinite(p).all() for p in self.model.parameters() if p.requires_grad)
        assert all(torch.isfinite(v).all() for item in self.optimizer.state.values() for v in item.values() if torch.is_tensor(v))
        self.scheduler.step()
        if images.device.type == 'cuda':
            torch.cuda.synchronize(images.device)
        elapsed = time.monotonic() - begin
        record_start = time.monotonic()
        update = self.state['updates'] + 1
        tables, records = observations(out, rows, self.registry, self.cfg, 'train', 'online_before_update_view_a', epoch0 + 1,
                                       update, self.state['sample_cursor'], update - 1, before, variant='appearance_view_a')
        second,records_b=observations(paired,rows,self.registry,self.cfg,'train','online_before_update_view_b',epoch0+1,
            update,self.state['sample_cursor']+len(rows),update-1,before,variant='appearance_view_b')
        merge(tables,second);records.extend(records_b)
        tables['view_pairs']=[];tables['view_patch_consistency']=[]
        values={k:v.detach().cpu().numpy() for k,v in parts.items()}
        for i,row in enumerate(rows):
            sa=self.state['sample_cursor']+i;sb=sa+len(rows)
            pa=int(out['probabilities'][i].argmax());pb=int(paired['probabilities'][i].argmax())
            tables['view_pairs'].append(dict(run=self.cfg['run'],epoch=epoch0+1,update=update,image_index=self.registry[row['image_id']],sample_a=sa,sample_b=sb,label=row['class_index'],prediction_a=pa,prediction_b=pb,disagree=int(pa!=pb),**{k:float(values[k][i]) for k in ('ce_a','ce_b','js','texture','residual_l2','total')}))
            for p in range(32): tables['view_patch_consistency'].append(dict(run=self.cfg['run'],sample_a=sa,sample_b=sb,patch=p,pooling_mass=float(out['patch_weights'][i,p]),cosine_distance=float(values['patch_distance'][i,p])))
        tables['parameter_gradients'] = gradients
        tables['sample_exposures'] = []
        factor_values = factors.cpu().numpy()
        for view in range(2):
            for slot,(draw,row) in enumerate(zip(draws,rows)):
                tables['sample_exposures'].append(dict(run=self.cfg['run'],sample=self.state['sample_cursor']+view*len(rows)+slot,
                    epoch=epoch0+1,update=update,slot=slot,occurrence=draw['occurrence'],image_index=self.registry[row['image_id']],
                    class_draw=draw['class_draw'],domain_draw=draw['domain_draw'],source_group=draw['source_group'],draw_probability=draw['probability'],
                    augmentation_seed=draw['augmentation_seed'],view=view,brightness=float(factor_values[slot,view,0]),contrast=float(factor_values[slot,view,1]),gamma=float(factor_values[slot,view,2])))
        dense, packed, seen = dense_once(out, rows, self.registry, self.cfg, self.state['dense_seen'])
        merge(tables, dense)
        artifacts = {f'dense/u{update:06d}.pt': packed} if packed else {}
        self.state['dense_seen'].extend(seen)
        self.state['train_records'].extend(records)
        self.state['sample_cursor'] += 2*len(rows)
        self.state['updates'] = update
        self.state['cursor'] += 1
        if self.state['cursor'] == len(order):
            self.state['phase'] = 'validate'
            self.state['eval_cursor'] = 0
            self.state['eval_records'] = []
        after = learner_sha(self.model)
        tables['coefficients'] = coefficients(self.model, self.cfg['run'], update)
        tables['train_updates'] = [dict(run=self.cfg['run'], epoch=epoch0 + 1, update=update, cursor=cursor,
            loss=float(loss.detach()),prediction_views=2,**{k:float(v.detach().mean()) for k,v in parts.items() if k in ('ce_a','ce_b','js','texture','residual_l2')}, lr=lr, grad_norm_before_clip=norm, grad_norm_after_clip=clipped, gradient_finite=1,
            trainable_parameters=self.recipe['trainable_parameters'], anchor_gradient_count=0, before_learner_sha256=before, after_learner_sha256=after,
            compute_seconds=elapsed, record_seconds=time.monotonic() - record_start, device=str(images.device),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(images.device) if images.device.type == 'cuda' else None,
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(images.device) if images.device.type == 'cuda' else None,
            host_peak_rss_bytes=host_memory(), free_disk_bytes=__import__('shutil').disk_usage(self.folder).free)]
        self.optimizer.zero_grad(set_to_none=True)
        self.capture_training_state()
        self.commit(tables, artifacts)
        event(self.folder / 'LIFECYCLE.jsonl', 'completed_update_committed', update=update, epoch=epoch0 + 1, cursor=self.state['cursor'])
        print(f"{self.recipe['mode']}: epoch {epoch0+1}/{self.recipe['epochs']} update {update} loss {float(loss.detach()):.6f}", flush=True)

    def setup_evaluation(self):
        stage = self.state['phase']
        if self.eval_model is None:
            bank, labels = load_bank()
            self.eval_model = TreadTwoAtlas(bank, labels, self.cfg['weights'],self.cfg['base_weights']).to(self.recipe['eval_device'])
            restore_rng(self.state['rng'])
        if stage in ('selected_eval','final_eval','diagnostics'):
            if stage == 'diagnostics':
                endpoint = ('selected','final')[self.state['diagnostic_endpoint']]
            else:
                endpoint = 'selected' if stage == 'selected_eval' else 'final'
            artifact = self.state['best']['artifact'] if endpoint == 'selected' else self.state['final_artifact']
            saved = torch.load(self.folder / 'artifacts' / artifact, map_location='cpu', weights_only=False)
            assert saved['protocol'] == self.cfg['protocol']
            self.eval_model.load_state_dict(saved['model'], strict=True)
            checkpoint_epoch = saved['epoch']
        else:
            self.eval_model.load_state_dict(self.state['model'], strict=True)
            endpoint, checkpoint_epoch = 'validation', self.state['epoch0'] + 1
        self.eval_model.eval()
        return endpoint, checkpoint_epoch

    def dense_summaries(self, records, stage, endpoint, epoch):
        ids = {r['image_index'] for r in records}
        regions = [r for r in self.store.rows('dense_regions') if int(r['image_index']) in ids]
        points = [r for r in self.store.rows('dense_points') if int(r['image_index']) in ids and int(r['metric_available'])]
        base = dict(run=self.cfg['run'], stage=stage, endpoint=endpoint, epoch=epoch, update=self.state['updates'], level='cohort', group='all', class_index=None)
        metrics = []
        for field, name in (('dice','mean_two_region_dice'), ('iou','mean_two_region_iou'), ('box_iou','mean_two_region_box_iou')):
            metrics.append(dict(base, metric=name, value=float(np.mean([float(r[field]) for r in regions])) if regions else None,
                                n=len(regions), unavailable_reason='' if regions else 'no_dense_records'))
        for kind in ('human','polygon_weak'):
            selected = [r for r in points if r['point_kind'] == kind]
            metrics.append(dict(base, metric=kind + '_point_mae_width_fraction',
                value=float(np.mean([float(r['absolute_error_width_fraction']) for r in selected])) if selected else None,
                n=len(selected), unavailable_reason='' if selected else 'no_comparable_guide_targets'))
        return metrics

    def eval_chunk(self):
        stage = self.state['phase']
        endpoint, checkpoint_epoch = self.setup_evaluation()
        rows_all = evaluation_rows(self.plan, self.cfg, stage)
        start = self.state['eval_cursor']
        rows = rows_all[start:start + self.recipe['evaluation_batch']]
        assert rows
        images, _, _ = batch(self.cfg['data_root'], rows, self.recipe['eval_device'])
        variant = self.recipe['diagnostics'][self.state['diagnostic_variant']] if stage == 'diagnostics' else 'identity'
        occluded = None
        with torch.inference_mode():
            base = self.eval_model(images,source_hash=source_hashes(rows,images.device))
            out = base
            if stage == 'diagnostics':
                if variant == 'occlude_top_patch':
                    chosen = base['probabilities'].argmax(-1)
                    evidence = torch.stack([base['patch_evidence'][i,:,int(cls)] for i,cls in enumerate(chosen)])
                    occluded = evidence.argmax(1).cpu().tolist()
                out = self.eval_model(images, variant=variant, occluded_patch=occluded,source_hash=source_hashes(rows,images.device))
        fingerprint = learner_sha(self.eval_model)
        tables, records = observations(out, rows, self.registry, self.cfg, stage, endpoint, checkpoint_epoch,
            self.state['updates'], self.state['sample_cursor'], self.state['updates'] if stage == 'validate' else
            (self.state['best']['update'] if endpoint == 'selected' else self.state['updates']), fingerprint, variant, occluded)
        artifacts = {}
        dense, packed, seen = dense_once(base, rows, self.registry, self.cfg, self.state['dense_seen'])
        merge(tables, dense)
        if packed:
            artifacts[f'dense/{stage}-{endpoint}-{variant}-{start:05d}.pt'] = packed
        self.state['dense_seen'].extend(seen)
        if stage in ('selected_eval','final_eval'):
            captures = []
            tables['explanation_checks'] = []
            for slot, row in enumerate(rows):
                atlas = {key: base['atlas'][key][slot].cpu().numpy() for key in ('valid','grid','rgb')}
                evidence = base['patch_evidence'][slot].cpu().numpy()
                projection, unsupported = splat(atlas, evidence)
                projection = projection.astype(np.float32)
                bias = base['bias'].cpu().numpy()
                logits = base['logits'][slot].cpu().numpy()
                for cls in range(3):
                    mass = float(projection[cls].sum(dtype=np.float64))
                    residual = abs(mass + float(bias[cls]) - float(logits[cls]))
                    assert residual < 2e-6 and abs(float(unsupported[cls])) < 1e-10
                    tables['explanation_checks'].append(dict(run=self.cfg['run'], sample=self.state['sample_cursor'] + slot,
                        epoch=checkpoint_epoch, update=self.state['updates'], endpoint=endpoint, class_index=cls,
                        mass_input=float(evidence[:,cls].sum(dtype=np.float64)), mass_projected=mass, bias=float(bias[cls]),
                        logit=float(logits[cls]), residual=residual, supported_pixels=int(atlas['valid'].sum()), unsupported_mass=float(unsupported[cls])))
                captures.append(dict(image_id=row['image_id'], sample=self.state['sample_cursor'] + slot, atlas=atlas,
                    projection=projection, patch_evidence=evidence, logits=logits, probabilities=base['probabilities'][slot].cpu().numpy(),
                    masks=base['masks'][slot].cpu().numpy(), points=base['points'][slot].cpu().numpy(), bias=bias,
                    coefficients=base['effective_coefficients'][slot].cpu().numpy(), reference_embeddings=base['reference_embeddings'].cpu().numpy(),
                    reference_evidence=((base['affinity'][slot] * base['patch_weights'][slot,:,None]).sum(0)[None] * base['effective_coefficients'][slot]).cpu().numpy(),
                    base_logits=base['base_logits'][slot].cpu().numpy(),context_evidence=base['context_evidence'][slot].cpu().numpy(),
                    context_bias=base['context_bias'].cpu().numpy(),attention=base['attention'][slot].cpu().numpy(),
                    context_embeddings=base['query_embeddings'][slot].cpu().numpy(),base_query_embeddings=base['base_query_embeddings'][slot].cpu().numpy()))
                attention_capture=captures[-1]['attention']
                for layer in range(2):
                    for head in range(4):
                        for q in range(32):
                            tables.setdefault('endpoint_attention',[]).extend(dict(run=self.cfg['run'],sample=self.state['sample_cursor']+slot,layer=layer,head=head,query=q,key=k,value=float(attention_capture[layer,head,q,k])) for k in range(32))
            artifacts[f'endpoint_maps/{endpoint}/chunk_{start:05d}.pt'] = captures
        if start == 0 and stage != 'diagnostics':
            embeddings = out['reference_embeddings'].detach().cpu().numpy()
            tables['reference_embeddings'] = [dict(run=self.cfg['run'], epoch=checkpoint_epoch, update=self.state['updates'], endpoint=endpoint,
                prototype=p, dimension=d, value=float(embeddings[p,d]), learner_sha256=fingerprint) for p in range(18) for d in range(96)]
        self.state['sample_cursor'] += len(rows)
        self.state['eval_records'].extend(records)
        self.state['eval_cursor'] += len(rows)
        if self.state['eval_cursor'] == len(rows_all):
            summaries, score = summarize(self.state['eval_records'], self.cfg, stage, endpoint + (':' + variant if stage == 'diagnostics' else ''),
                                          checkpoint_epoch, self.state['updates'])
            merge(tables, summaries)
            # New dense rows in this chunk are not yet in Store; add dense summaries in
            # a later zero-inference completion unit to include their full denominators.
            self.state['phase'] = stage + '_finish'
            self.state['last_score'] = score
        self.commit(tables, artifacts, partition=checkpoint_epoch)

    def finish_evaluation(self):
        stage = self.state['phase'].removesuffix('_finish')
        endpoint = 'validation' if stage == 'validate' else ('selected' if stage == 'selected_eval' else 'final')
        if stage == 'diagnostics':
            endpoint = ('selected','final')[self.state['diagnostic_endpoint']]
        epoch = self.state['epoch0'] + 1 if stage == 'validate' else (self.state['best']['epoch'] if endpoint == 'selected' else self.recipe['epochs'])
        tables = {'metrics': self.dense_summaries(self.state['eval_records'], stage, endpoint, epoch)} if stage != 'diagnostics' else {}
        artifacts = {}
        if stage == 'validate':
            score = self.state['last_score']
            best = self.state['best']
            better = best is None or score['balanced_accuracy_observed'] > best['ba'] or (score['balanced_accuracy_observed'] == best['ba'] and score['nll'] < best['nll'])
            artifact = f'epochs/epoch_{epoch:04d}.pt'
            artifacts[artifact] = self.checkpoint(epoch)
            if better:
                self.state['best'] = dict(epoch=epoch, update=self.state['updates'], ba=score['balanced_accuracy_observed'], nll=score['nll'], artifact=artifact)
            tables['selection'] = [dict(run=self.cfg['run'], epoch=epoch, update=self.state['updates'], validation_ba_observed=score['balanced_accuracy_observed'],
                validation_nll=score['nll'], selected=int(better), selected_epoch=self.state['best']['epoch'], tie_break='BA descending, NLL ascending, earliest epoch',
                checkpoint_artifact=artifact, learner_sha256=learner_sha(self.model))]
            train_summary, _ = summarize(self.state['train_records'], self.cfg, 'train', 'online_mixed_update_predictions', epoch, self.state['updates'])
            merge(tables, train_summary)
            self.state['last_validation'] = dict(epoch=epoch, **score)
            self.state['epoch0'] += 1
            self.state['cursor'] = 0
            self.state['train_records'] = []
            self.state['phase'] = 'train' if self.state['epoch0'] < self.recipe['epochs'] else 'selected_eval'
            if self.state['phase'] == 'selected_eval':
                self.state['final_artifact'] = artifact
        elif stage == 'selected_eval':
            self.state['phase'] = 'final_eval'
        elif stage == 'final_eval':
            self.state['phase'] = 'diagnostics'
        else:
            self.state['diagnostic_variant'] += 1
            if self.state['diagnostic_variant'] == len(self.recipe['diagnostics']):
                self.state['diagnostic_variant'] = 0
                self.state['diagnostic_endpoint'] += 1
            self.state['phase'] = 'completed' if self.state['diagnostic_endpoint'] == 2 else 'diagnostics'
        self.state['eval_cursor'] = 0
        self.state['eval_records'] = []
        self.commit(tables, artifacts, partition=epoch)
        event(self.folder / 'LIFECYCLE.jsonl', 'evaluation_finished', stage=stage, epoch=epoch, next_phase=self.state['phase'])

    def run(self, limit_updates=None):
        try:
            while self.state['phase'] != 'completed':
                control = Path(self.cfg.get('control_root', str(self.folder.parent)))
                if STOP or (control / 'STOP_REQUEST.json').exists() or (control / 'FLUSH_REQUEST.json').exists():
                    return 'paused'
                if limit_updates is not None and self.state['updates'] >= limit_updates:
                    return 'paused'
                if self.state['phase'] == 'train':
                    capacity(self.folder)
                    self.train_update()
                elif self.state['phase'].endswith('_finish'):
                    self.finish_evaluation()
                else:
                    self.eval_chunk()
            self.assert_anchor()
            return 'completed'
        except BaseException as error:
            event(self.folder / 'NUMERICAL_EVENTS.jsonl', 'worker_failure', error_type=type(error).__name__,
                  committed_updates=self.store.units[-1]['updates'] if self.store.units else 0)
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--folder', required=True)
    parser.add_argument('--limit-updates', type=int)
    parser.add_argument('--fault-stage', choices=['after_csv','after_native','before_pointer','after_pointer'])
    args = parser.parse_args()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    cfg = read(args.config)
    event(Path(args.folder) / 'LIFECYCLE.jsonl', 'worker_start', mode=cfg['contract']['mode'])
    engine = Engine(cfg, args.folder, args.fault_stage)
    result = engine.run(args.limit_updates)
    atomic(Path(args.folder) / 'WORKER_RESULT.json', dict(status=result, updates=engine.state['updates'], phase=engine.state['phase']))
    sys.exit(0 if result == 'completed' else 91)


if __name__ == '__main__':
    main()
