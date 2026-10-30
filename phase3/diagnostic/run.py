"""TREAD-ONE NB03: frozen-checkpoint diagnosis; never performs an optimizer step."""
from PIL import Image
import argparse
import base64
import hashlib
import io
import json
import os
import shutil
import signal
import sys
import time
import traceback
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT=Path(__file__).resolve().parents[1]
assert Path(sys.prefix).name == 'cv_conda' or Path('/kaggle/input').exists()
sys.path.insert(0,str(ROOT/'runtime'))
import numpy as np
import torch
from filelock import FileLock
from common import deterministic, environment, memory
from data import batch, plan
from model import TreadOne, loss_terms
from storage import Ledger, Publisher, atomic, canonical, digest, read, sha
from metrics import SCHEMAS, VARIANTS, cpu_output, gate_output, group_key, records
from report import build_report

SOURCE='a0296f18e02b12d28f7e51377783d68538032209'
PROTOCOL='371c40c0c6aaedaf5c5970f7721a284b042494db7bb330065f6cbde8fc3a0b8c'
STOP=False


def stop(*_):
    global STOP
    STOP=True


def state_hash(weights):
    h=hashlib.sha256()
    for name,tensor in sorted(weights.items()):
        h.update(canonical([name,str(tensor.dtype),list(tensor.shape)]))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def source_data(data_root):
    if data_root:
        root=Path(data_root).resolve()
    elif Path('/kaggle/input').exists():
        candidates=[]
        for p in Path('/kaggle/input').rglob('VERSION.json'):
            if read(p).get('dataset_title')=='Tire Dataset Prepared phase2': candidates.append(p.parent)
        assert len(candidates)==1, 'Attach exactly one prepared Phase 2 dataset.'
        root=candidates[0]
    else:
        root=ROOT.parent/'phase2/data/phase2_dataset_v1'
    return root


def prepare(data_root,seeds,endpoints,device,limit,batch_size,chunk,gradients):
    cache=ROOT/'completion_20261028'
    contract=read(cache/'hf/CONTRACT.json'); overlay=read(cache/'hf/OVERLAY.json')
    assert digest(contract)==PROTOCOL and digest(overlay)==contract['overlay_sha256']
    for name,expected in contract['sources'].items():
        assert sha(ROOT/'runtime'/name)==expected, 'Frozen runtime identity changed: '+name
    root=source_data(data_root)
    assert sha(root/'SHA256SUMS.txt')==overlay['source_checksum']
    source_hashes=dict(line.split('  ',1)[::-1] for line in (root/'SHA256SUMS.txt').read_text().splitlines())
    for name in ('manifests/images.json','geometry/original_human_points.json','geometry/new_point_proposals.json'):
        assert sha(root/name)==source_hashes[name]
    assert digest(plan(root))==digest(overlay), 'Data/point/guide overlay differs from measured r1.'
    rows=sorted([r for r in overlay['rows'] if r['role'] in ('train','validation')],key=lambda r:(r['role'],r['image_id']))
    if limit:
        rows=[r for role in ('train','validation') for r in [v for v in rows if v['role']==role][:limit]]
    assert rows and all(r['role']!='test' for r in rows)
    jobs=[]; checkpoint_cache=cache/'diagnostic_weights';checkpoint_cache.mkdir(exist_ok=True)
    remote=read(cache/'remote_tree.json')
    entries={r['path']:r for r in remote['files']}
    for seed in seeds:
        archive=cache/f'hf/runs/seed{seed}/latest.zip'
        assert sha(archive)==entries[f'runs/seed{seed}/latest.zip']['lfs_sha256']
        with zipfile.ZipFile(archive) as z:
            manifest=json.loads(z.read('MANIFEST.json'));meta=manifest['files']['state.pt'];body=z.read('state.pt')
            assert len(body)==meta['bytes'] and hashlib.sha256(body).hexdigest()==meta['sha256']
            # Only these pinned, whole-file/member-hash-verified own-study states are deserialized.
            state=torch.load(io.BytesIO(body),map_location='cpu',weights_only=False)
        assert state['completed'] and state['epoch']==60 and state['updates']==2940 and state['protocol']==PROTOCOL
        for endpoint in endpoints:
            weights=state['best'] if endpoint=='selected' else state['model']
            key=f'seed{seed}-{endpoint}'
            target=checkpoint_cache/(key+'.pt')
            if target.exists():
                assert state_hash(torch.load(target,map_location='cpu',weights_only=True))==state_hash(weights)
            else:
                torch.save(weights,target)
            jobs.append(dict(key=key,seed=seed,endpoint=endpoint,epoch=state['best_epoch'] if endpoint=='selected' else 60,
                        tensor_sha256=state_hash(weights),source_bundle_sha256=sha(archive),weights_path=str(target)))
        del state,body
    visible=dict(environment(device),visible_gpus=torch.cuda.device_count() if torch.cuda.is_available() else 0)
    sources={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sorted((ROOT/'diagnostic').glob('*')) if p.suffix in ('.py','.html')}
    study=dict(family='TREAD-ONE',revision='classifier-diagnostic-r1',source_revision=SOURCE,source_protocol=PROTOCOL,
       overlay_sha256=digest(overlay),dataset_sha256=overlay['source_checksum'],
       sources=sources,frozen_sources=contract['sources'],environment=visible,precision='FP32; no autocast/TF32; deterministic eval',
       input_hw=[512,384],batch=batch_size,chunk=chunk,seeds=seeds,endpoints=endpoints,
       jobs=[{k:v for k,v in j.items() if k!='weights_path'} for j in jobs],
       rows=[r['image_id'] for r in rows],limit_per_split=limit,fixture=bool(limit),gradient_diagnostic=gradients,
       variants=VARIANTS,selection='No new checkpoint/seed selection; existing selected and fixed-final endpoints.',
       scope='Post-hoc nominal train/validation inference only; no optimizer updates, new test evaluation, new tyres or historical training-accuracy reconstruction.',
       interventions=dict(uniform_gate='Fixed features with uniform spatial gate; no head refit.',
         oracle_tread_gate='Evaluation-only manual tread/valid mask aggregated 4x4; fixed features and weights; not deployable inference.',
         background_neutralized='Evaluation-only manual tyre/tread/ignore union protects foreground; outside becomes normalized zero (ImageNet RGB mean).',
         top10_image_occluded='Top floor(0.10*128*96) signed normal-predicted-class evidence cells, stable descending index order; input blocks become normalized zero.'),
       gallery='First fixed image per split and tyre-or-video-clip per checkpoint; not an unbiased random failure sample.',
       telemetry=SCHEMAS)
    return root,rows,jobs,study,source_hashes


def jpeg(tensor):
    im=tensor.detach().cpu().numpy().transpose(1,2,0)
    im=np.clip((im*np.array([.229,.224,.225])+np.array([.485,.456,.406]))*255,0,255).astype(np.uint8)
    stream=io.BytesIO();Image.fromarray(im).save(stream,format='JPEG',quality=72)
    return 'data:image/jpeg;base64,'+base64.b64encode(stream.getvalue()).decode()


def sync(device):
    if device.type=='cuda': torch.cuda.synchronize(device)


def infer_unit(model,job,rows,root,device,batch_size,gallery_ids,source_hashes,verified):
    start=time.perf_counter();result={t:[] for t in SCHEMAS};gallery={}
    loads=normal_secs=bg_secs=occ_secs=metrics_secs=0.
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    for offset in range(0,len(rows),batch_size):
        if STOP: raise InterruptedError('Catchable stop: the last complete diagnostic unit will be published.')
        chosen=rows[offset:offset+batch_size];t=time.perf_counter()
        for row in chosen:
            for key in ('image_path','tyre_mask','tread_mask','ignore_mask'):
                name=row[key]
                if name not in verified:
                    assert sha(root/name)==source_hashes[name],name
                    verified.add(name)
        b=batch(root,chosen,device);sync(device);loads+=time.perf_counter()-t
        with torch.inference_mode():
            t=time.perf_counter();out=model(b['img'],b['guides']);sync(device);normal_secs+=time.perf_counter()-t
            uniform=gate_output(out,torch.ones_like(out['gate']),model.class_weights,model.class_bias)
            valid=1-b['masks'][:,2:3]
            oracle=(b['masks'][:,1:2]*valid).reshape(len(chosen),1,128,4,96,4).mean((3,5))
            oracle=gate_output(out,oracle,model.class_weights,model.class_bias)
            fg=(b['masks'][:,:1]>.5)|(b['masks'][:,1:2]>.5)|(b['masks'][:,2:3]>.5)
            bg_image=torch.where(fg,b['img'],torch.zeros_like(b['img']))
            t=time.perf_counter();bg=model(bg_image,b['guides']);sync(device);bg_secs+=time.perf_counter()-t
            scores=out['contributions'][torch.arange(len(chosen),device=device),out['logits'].argmax(1)]
            top=scores.flatten(1).argsort(dim=1,descending=True,stable=True)[:,:scores[0].numel()//10]
            selected=torch.zeros_like(scores).flatten(1).bool();selected.scatter_(1,top,True)
            selected=selected.reshape_as(scores).repeat_interleave(4,-2).repeat_interleave(4,-1)
            occ_image=torch.where(selected[:,None],torch.zeros_like(b['img']),b['img'])
            t=time.perf_counter();occ=model(occ_image,b['guides']);sync(device);occ_secs+=time.perf_counter()-t
            outputs={name:cpu_output(v) for name,v in zip(VARIANTS,(out,uniform,oracle,bg,occ))}
        t=time.perf_counter();targets=b['masks'].cpu().numpy()
        for i,row in enumerate(chosen):
            for variant,values in outputs.items():
                pred,regions,points,latents,evidence=records(row,job,variant,values,i,targets[i],outputs['normal'])
                result['predictions'].append(pred);result['regions'].extend(regions);result['points'].extend(points)
                result['latents'].extend(latents);result['evidence'].extend(evidence)
            if row['image_id'] in gallery_ids:
                raw=[]
                for variant in ('normal','background_neutralized','top10_image_occluded'):
                    v=outputs[variant]
                    raw.append(np.concatenate([v['masks'][i],v['concepts'][i],v['contributions'][i],v['gate'][i],outputs['oracle_tread_gate']['gate'][i]],axis=0))
                gallery[row['image_id']]=dict(array=np.stack(raw),job=job['key'],seed=job['seed'],endpoint=job['endpoint'],epoch=job['epoch'],
                    split=row['role'],group=group_key(row),label=row['class_index'],point_kind=row['point_kind'],
                    target_points=row['points'],guides=row['guide_fractions'],
                    class_weights=model.class_weights.detach().cpu().tolist(),class_bias=model.class_bias.detach().cpu().tolist(),
                    predicted_points={name:values['points'][i].tolist() for name,values in outputs.items()},
                    images=[jpeg(x[i]) for x in (b['img'],bg_image,occ_image)])
        metrics_secs+=time.perf_counter()-t
        del b,out,uniform,oracle,bg,occ,outputs
    try:
        import psutil
        rss=psutil.Process().memory_info().rss
    except ImportError: rss=None
    result['timings'].append(dict(job=job['key'],images=len(rows),load_seconds=loads,normal_forward_seconds=normal_secs,
        background_forward_seconds=bg_secs,occlusion_forward_seconds=occ_secs,metrics_and_gallery_seconds=metrics_secs,
        unit_compute_seconds=time.perf_counter()-start,host_rss_bytes=rss,
        cuda_allocated_bytes=torch.cuda.memory_allocated(device) if device.type=='cuda' else None,
        cuda_peak_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,working_free_bytes=shutil.disk_usage(ROOT).free))
    return result,gallery


def gradient_unit(model,job,rows,root,device):
    # One fixed nominal train image per class, human-labelled preferred.
    chosen=[]
    for c in range(3):
        candidates=sorted([r for r in rows if r['role']=='train' and r['class_index']==c],key=lambda r:(r['point_kind']!='human',r['image_id']))
        if candidates: chosen.append(candidates[0])
    start=time.perf_counter();b=batch(root,chosen,device)
    out=model(b['img'],b['guides']);terms=loss_terms(out,b)
    tasks={'classification_plus_ordinal':terms['classification']+.2*terms['ordinal'],
           'segmentation':terms['segmentation'],'geometry_times2':2*terms['geometry'],
           'row_agreement_times0.1':.1*terms['alignment'],'containment_times0.1':.1*terms['containment']}
    params=list(model.named_parameters())
    def module(name):
        return 'shared' if name.split('.')[0] in ('stem','s1','s2','s3','s4','position','context','d1','d2','d3') else name.split('.')[0]
    groups={name:[i for i,(key,p) in enumerate(params) if module(key)==name] for name in sorted({module(key) for key,p in params})}
    shared={};records_out=[]
    for name,value in tasks.items():
        grads=torch.autograd.grad(value,[p for _,p in params],retain_graph=True,allow_unused=True)
        shared[name]=torch.cat([grads[i].detach().flatten() if grads[i] is not None else torch.zeros_like(params[i][1]).flatten() for i in groups['shared']])
        for group,indices in groups.items():
            present=[grads[i] for i in indices if grads[i] is not None]
            norm=float(torch.sqrt(sum((g.detach().float().square().sum() for g in present),start=value.new_tensor(0)))) if present else 0.
            records_out.append(dict(job=job['key'],row_type='norm',task=name,other_task=None,module_group=group,task_loss=float(value.detach()),gradient_norm=norm,cosine=None,
                 missing_reason='no_gradient_path' if not present else None,image_ids='|'.join(r['image_id'] for r in chosen),seconds=None))
    for i,a in enumerate(tasks):
        for bname in list(tasks)[i+1:]:
            aa,bb=shared[a],shared[bname];denom=aa.norm()*bb.norm()
            cosine=float(torch.dot(aa,bb)/denom) if float(denom)>0 else None
            records_out.append(dict(job=job['key'],row_type='cosine',task=a,other_task=bname,module_group='shared',task_loss=None,gradient_norm=None,cosine=cosine,
                  missing_reason='zero_norm' if cosine is None else None,image_ids='|'.join(r['image_id'] for r in chosen),seconds=time.perf_counter()-start))
    assert all(p.grad is None for _,p in params)
    return dict(gradients=records_out)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',default='');parser.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seeds',default='1,2,3');parser.add_argument('--endpoints',default='selected,fixed_final')
    parser.add_argument('--batch',type=int,default=4);parser.add_argument('--chunk',type=int,default=32)
    parser.add_argument('--limit',type=int,default=0);parser.add_argument('--no-gradients',action='store_true')
    parser.add_argument('--publish',action='store_true');parser.add_argument('--pause-after-units',type=int,default=0)
    parser.add_argument('--output-root',default='')
    args=parser.parse_args();device=torch.device(args.device)
    assert args.batch>0 and args.chunk>=args.batch
    deterministic(4622)
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,stop)
    assert shutil.disk_usage(ROOT).free>6*1024**3,'Need 6 GiB free for immutable diagnostic generations and pinned source evidence.'
    seeds=[int(s) for s in args.seeds.split(',')];endpoints=args.endpoints.split(',')
    assert set(seeds)<=set((1,2,3)) and set(endpoints)<=set(('selected','fixed_final'))
    root,rows,jobs,contract,source_hashes=prepare(args.data_root,seeds,endpoints,device,args.limit,args.batch,args.chunk,not args.no_gradients)
    tag=digest(contract);folder=(Path(args.output_root) if args.output_root else ROOT/'diagnostics/classifier_r1')/tag
    ledger=Ledger(folder,contract,SCHEMAS)
    destination='phase3/tread-one-r1/'+PROTOCOL+'/diagnostics/classifier-r1/'+tag
    publisher=Publisher(destination,os.environ.get('HF_TOKEN')) if args.publish else None
    print('TREAD-ONE diagnostic:',tag,'images/checkpoint:',len(rows),'jobs:',len(jobs),'device:',device,'publishing:',bool(publisher),flush=True)
    def report(snapshot):
        build_report(snapshot,ROOT)
        source=snapshot/'source';source.mkdir(exist_ok=True)
        for name in contract['sources']:
            path=ROOT/name;target=source/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    acquired=False;last_push=time.monotonic();completed_now=0;verified=set()
    def event(kind,unit=None,job=None,seconds=None,message=None):
        value=dict(event_id=uuid.uuid4().hex,time_utc=datetime.now(timezone.utc).isoformat(),kind=kind,unit=unit,job=job,
                   seconds=seconds,message=message,last_verified_remote_revision=publisher.last_verified if publisher else None)
        with open(folder/'EVENTS.jsonl','ab') as stream:
            stream.write(canonical(value)+b'\n');stream.flush();os.fsync(stream.fileno())
    with FileLock(str(folder/'RUN.lock'),timeout=1):
        try:
            if publisher:
                publisher.claim();acquired=True;publisher.restore(ledger)
            event('session_start',message='Fixed-weight FP32 diagnostic; no optimizer update.')
            for job in jobs:
                model=TreadOne().to(device)
                weights=torch.load(job['weights_path'],map_location='cpu',weights_only=True)
                model.load_state_dict(weights,strict=True);model.eval()
                assert state_hash(model.state_dict())==job['tensor_sha256']
                before=job['tensor_sha256'];del weights
                groups={}
                for row in rows: groups.setdefault((row['role'],group_key(row)),row['image_id'])
                gallery_ids=set(groups.values())
                completed_units=ledger.units()
                for start in range(0,len(rows),args.chunk):
                    unit=f'{job["key"]}-b{start:05d}'
                    if unit in completed_units: continue
                    if STOP: raise InterruptedError('Catchable stop before next complete inference unit.')
                    data,gallery=infer_unit(model,job,rows[start:start+args.chunk],root,device,args.batch,gallery_ids,source_hashes,verified)
                    t=time.perf_counter();ledger.commit(unit,job['key'],data,gallery)
                    atomic(folder/'LAST_UNIT.json',dict(unit=unit,local_commit_seconds=time.perf_counter()-t))
                    event('unit_committed',unit,job['key'],time.perf_counter()-t)
                    completed_now+=1
                    print(job['key'],'images',min(start+args.chunk,len(rows)),'/',len(rows),'complete; direct local records retained.',flush=True)
                    if args.pause_after_units and completed_now>=args.pause_after_units: raise InterruptedError('Requested diagnostic resume test boundary.')
                    if publisher and time.monotonic()-last_push>=30*60:
                        publisher.publish(ledger.snapshot(report,'running'));last_push=time.monotonic()
                unit=job['key']+'-gradient'
                if not args.no_gradients and unit not in completed_units:
                    ledger.commit(unit,job['key'],gradient_unit(model,job,rows,root,device))
                    event('gradient_diagnostic_committed',unit,job['key'])
                assert state_hash(model.state_dict())==before, 'Diagnostic altered a learned parameter.'
                atomic(folder/(job['key']+'-IDENTITY.json'),dict(job=job['key'],tensor_sha256=before,unchanged_after_inference_and_gradients=True))
                if publisher and job != jobs[-1]:
                    publisher.publish(ledger.snapshot(report,'running'));last_push=time.monotonic()
                del model
                if device.type=='cuda': torch.cuda.empty_cache()
            snapshot=ledger.snapshot(report,'complete')
            if publisher: publisher.publish(snapshot,release=True);acquired=False
            atomic(folder/'COMPLETE.json',dict(status='complete',contract=tag,images_per_checkpoint=len(rows),jobs=len(jobs),
                 expected_prediction_rows=len(rows)*len(jobs)*len(VARIANTS),snapshot=str(snapshot),remote_publication='verified' if publisher else 'not_attempted_local',training_updates=0))
            print('DIAGNOSTIC COMPLETE:',snapshot,flush=True)
        except BaseException as exc:
            event('catchable_stop' if isinstance(exc,(InterruptedError,KeyboardInterrupt)) else 'error',message=traceback.format_exc())
            atomic(folder/'ERROR.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),
                   catchable_stop=isinstance(exc,(InterruptedError,KeyboardInterrupt)),last_verified_remote_revision=publisher.last_verified if publisher else None))
            try:
                snapshot=ledger.snapshot(report,'paused' if isinstance(exc,(InterruptedError,KeyboardInterrupt)) else 'error')
                if publisher and acquired: publisher.publish(snapshot,release=True)
            except BaseException as flush_error:
                print('Final publication remains pending:',type(flush_error).__name__,'Local verified units remain:',folder,flush=True)
            raise


if __name__=='__main__':
    main()
