"""Resumable per-image train-only probes. No optimizer or validation/test forward."""
import argparse
import hashlib
import json
import os
import signal
import sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent


def setup(source):
    sys.path.insert(0,str(Path(source).resolve()))


SCHEMAS={
 'source_train':['image_index','image_id','image_sha256','class_index','domain','physical_tyre_id','source_group','is_v1_bank','is_v1_query'],
 'patch_quality':['image_index','image_id','class_index','domain','physical_tyre_id','patch','valid_fraction','gray_mean','gray_std','mean_gradient','glare_fraction','dark_fraction','chroma_fraction','low_texture'],
 'predictions':['image_index','image_id','class_index','cohort','variant','prediction','correct','logit_0','logit_1','logit_2','probability_0','probability_1','probability_2','nll','supported_patches','evidence_residual'],
 'patch_evidence':['image_index','variant','patch','class_index','contribution','pooling_mass'],
 'affinities':['image_index','variant','patch','prototype','cosine','affinity'],
 'feature_consistency':['image_index','variant','patch','valid_fraction','new_clip_fraction','rgb_relative_l2','local_contrast_relative_l2','local_contrast_cosine','local_std_floor_scope'],
 'execution':['image_index','image_id','dense_device','head_device','optimizer_updates','parameters_with_grad','model_frozen','compute_seconds','cuda_peak_allocated_dense_bytes','cuda_peak_allocated_head_bytes'],
 'candidate_bank':['prototype','class_index','image_index','image_id','image_sha256','physical_tyre_id','patch','valid_fraction','glare_fraction','chroma_fraction','mean_gradient','low_texture','role','status'],
}


def weights_digest(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run(cfg,folder,limit=None):
    setup(cfg['frozen_source'])
    from common import capacity,read,seed_all,sha
    from data_atlas import dataset,input_rgb,load_bank
    from geometry import canonicalize,patches
    from model_atlas import TreadTwoAtlas
    from store import Store
    from gate_features import contrast,descriptor,quality,choose_bank
    import time
    seed_all(1)
    assert all(sha(HERE/name)==value for name,value in cfg['contract']['sources'].items()),'Gate source drift'
    assert all(sha(Path(cfg['frozen_source'])/name)==value for name,value in cfg['contract']['frozen_sources'].items()),'Frozen v1 source drift'
    rules=cfg['contract']['rules']
    plan,queries=dataset(Path(cfg['data_root']))
    all_rows=sorted([r for r in plan['rows'] if r['role']=='train'],key=lambda r:r['image_id'])
    assert len(all_rows)==386 and {r['image_id'] for r in all_rows}==set(cfg['contract']['training_image_ids'])
    query_ids={r['image_id'] for r in queries}
    store=Store(folder,cfg['contract'],SCHEMAS)
    if store.state and store.state['phase']=='completed': return 0
    dense_device,head_device=cfg['contract']['dense_device'],cfg['contract']['head_device']
    bank,labels=load_bank()
    model=TreadTwoAtlas(bank,labels,cfg['anchor'])
    saved=torch.load(cfg['selected'],map_location='cpu',weights_only=True)
    assert saved['protocol']==rules['source_fit_protocol'] and saved['epoch']==18
    model.load_state_dict(saved['model'],strict=True)
    model.to(head_device).eval().requires_grad_(False)
    model.anchor.to(dense_device)
    original_sha=weights_digest(model)
    if store.state:
        assert store.state['model_sha256']==original_sha
    else:
        from common import digest
        rows=[dict(image_index=i,image_id=r['image_id'],image_sha256=r['image_sha256'],class_index=r['class_index'],domain=r['domain'],physical_tyre_id=r['physical_tyre_id'],source_group=r['physical_tyre_id'] or r['source_video'],is_v1_bank=int(r['image_id'] not in query_ids),is_v1_query=int(r['image_id'] in query_ids)) for i,r in enumerate(all_rows)]
        store.commit(dict(protocol=digest(cfg['contract']),updates=0,phase='probe',next_image=0,model_sha256=original_sha),{'source_train':rows})
    dense_mean=torch.tensor([.485,.456,.406],device=dense_device)[None,:,None,None]
    dense_std=torch.tensor([.229,.224,.225],device=dense_device)[None,:,None,None]
    completed_this_process=0
    for index in range(store.state['next_image'],len(all_rows)):
        if limit is not None and completed_this_process>=limit: return 91
        if any((Path(cfg['control_root'])/name).exists() for name in ('STOP_REQUEST.json','FLUSH_REQUEST.json')): return 91
        capacity(folder)
        row=all_rows[index];assert row['role']=='train'
        assert sha(Path(cfg['data_root'])/row['image_path'])==row['image_sha256']
        begun=time.monotonic()
        rgb=torch.tensor(input_rgb(cfg['data_root'],row)[None],device=dense_device)
        tables={name:[] for name in SCHEMAS if name!='candidate_bank'}
        with torch.inference_mode(),torch.autocast(torch.device(dense_device).type,enabled=False):
            dense=model.anchor((rgb-dense_mean)/dense_std)
            atlas=canonicalize(rgb,dense['masks'])
        atlas={k:(v.to(head_device) if torch.is_tensor(v) else v) for k,v in atlas.items()}
        base_tiles=patches(atlas['rgb'])[0]
        valid=patches(atlas['valid'])[0]
        base_contrast=contrast(base_tiles,valid,rules)
        descriptors=descriptor(base_contrast,valid,rules['descriptor_grid']).cpu().numpy()
        qualities=[];probe_records=[]
        for patch in range(32):
            q=quality(base_tiles[patch],valid[patch]);qualities.append(q)
            tables['patch_quality'].append(dict(image_index=index,image_id=row['image_id'],class_index=row['class_index'],domain=row['domain'],physical_tyre_id=row['physical_tyre_id'],patch=patch,**q))
        with torch.inference_mode(),torch.autocast(torch.device(head_device).type,enabled=False):
            for variant in rules['variants']:
                altered=dict(atlas)
                rgb_variant=atlas['rgb']
                if variant in ('grayscale_both','query_grayscale'): rgb_variant=rgb_variant.mean(1,keepdim=True).expand_as(rgb_variant)
                elif variant=='brightness_080': rgb_variant=rgb_variant*.8
                elif variant=='brightness_120': rgb_variant=rgb_variant*1.2
                elif variant=='gamma_080': rgb_variant=rgb_variant.pow(.8)
                elif variant=='gamma_120': rgb_variant=rgb_variant.pow(1.2)
                altered['rgb']=rgb_variant.clamp(0,1)*atlas['valid']
                # grayscale_both is delegated to frozen v1 so reference RGB
                # is transformed too. Other transforms affect query only.
                out=model.classify(atlas if variant=='grayscale_both' else altered,variant='grayscale_both' if variant=='grayscale_both' else 'identity')
                logits=out['logits'][0].cpu().numpy();prob=out['probabilities'][0].cpu().numpy()
                evidence=out['patch_evidence'][0].cpu().numpy();alpha=out['patch_weights'][0].cpu().numpy()
                cosine=out['cosine'][0].cpu().numpy();affinity=out['affinity'][0].cpu().numpy()
                bias=out['bias'].cpu().numpy()
                residual=float(np.abs(evidence.sum(0,dtype=np.float64)+bias-logits).max())
                assert residual<2e-6 and np.isfinite(prob).all()
                record=dict(image_index=index,image_id=row['image_id'],class_index=row['class_index'],cohort='support_photo' if row['image_id'] not in query_ids else 'v1_query',variant=variant,prediction=int(prob.argmax()),correct=int(prob.argmax()==row['class_index']),nll=float(-np.log(max(float(prob[row['class_index']]),1e-12))),supported_patches=int((alpha>0).sum()),evidence_residual=residual)
                record.update({f'logit_{c}':float(logits[c]) for c in range(3)});record.update({f'probability_{c}':float(prob[c]) for c in range(3)})
                tables['predictions'].append(record);probe_records.append(record)
                tiles=patches(altered['rgb'])[0]
                unclipped=patches(rgb_variant)[0]
                normalized=contrast(tiles,valid,rules)
                for patch in range(32):
                    mask=valid[patch].expand_as(tiles[patch])>.5
                    new_clip=float(((unclipped[patch]>1)&mask).float().sum()/mask.float().sum().clamp_min(1))
                    x,y=base_tiles[patch]*valid[patch],tiles[patch]*valid[patch]
                    a,b=base_contrast[patch],normalized[patch]
                    tables['feature_consistency'].append(dict(image_index=index,variant=variant,patch=patch,valid_fraction=qualities[patch]['valid_fraction'],new_clip_fraction=new_clip,
                        rgb_relative_l2=float((y-x).norm()/x.norm().clamp_min(1e-6)),local_contrast_relative_l2=float((b-a).norm()/a.norm().clamp_min(1e-6)),
                        local_contrast_cosine=float(F.cosine_similarity(a.flatten()[None],b.flatten()[None],eps=1e-8)),local_std_floor_scope='floor=.02; clipped z/3; not exact in flat/saturated regions'))
                    for c in range(3): tables['patch_evidence'].append(dict(image_index=index,variant=variant,patch=patch,class_index=c,contribution=float(evidence[patch,c]),pooling_mass=float(alpha[patch])))
                    for j in range(18): tables['affinities'].append(dict(image_index=index,variant=variant,patch=patch,prototype=j,cosine=float(cosine[patch,j]),affinity=float(affinity[patch,j])))
        assert not any(p.grad is not None or p.requires_grad for p in model.parameters())
        tables['execution'].append(dict(image_index=index,image_id=row['image_id'],dense_device=dense_device,head_device=head_device,optimizer_updates=0,parameters_with_grad=0,model_frozen=1,compute_seconds=time.monotonic()-begun,cuda_peak_allocated_dense_bytes=torch.cuda.max_memory_allocated(dense_device) if torch.device(dense_device).type=='cuda' else 0,cuda_peak_allocated_head_bytes=torch.cuda.max_memory_allocated(head_device) if torch.device(head_device).type=='cuda' else 0))
        packet=dict(image_id=row['image_id'],image_index=index,class_index=row['class_index'],domain=row['domain'],physical_tyre_id=row['physical_tyre_id'],role='train',image_sha256=row['image_sha256'],descriptors=descriptors,
            atlas={k:atlas[k][0].cpu().numpy() for k in ('rgb','valid','grid')},masks=dense['masks'][0].cpu().numpy(),points=dense['points'][0].cpu().numpy(),quality=qualities,probes=probe_records)
        state=dict(store.state,next_image=index+1,phase='probe')
        store.commit(state,tables,partition=index//50,artifacts={f'train/{index:04d}.pt':packet})
        completed_this_process+=1
        print('Committed train-only image',index+1,'/ 386; optimizer updates 0',flush=True)
    if any((Path(cfg['control_root'])/name).exists() for name in ('STOP_REQUEST.json','FLUSH_REQUEST.json')): return 91
    candidates=[]
    for index,row in enumerate(all_rows):
        packet=torch.load(Path(folder)/f'artifacts/train/{index:04d}.pt',map_location='cpu',weights_only=False)
        for patch,q in enumerate(packet['quality']):
            if q['gray_std'] is not None:
                candidates.append(dict(image_index=index,image_id=row['image_id'],image_sha256=row['image_sha256'],class_index=row['class_index'],domain=row['domain'],physical_tyre_id=row['physical_tyre_id'],role='train',patch=patch,descriptor=packet['descriptors'][patch],**q))
    chosen=choose_bank(candidates,rules)
    bank_rows=[];rgb=[];normalized=[]
    for j,r in enumerate(chosen):
        packet=torch.load(Path(folder)/f"artifacts/train/{r['image_index']:04d}.pt",map_location='cpu',weights_only=False)
        picture=torch.tensor(packet['atlas']['rgb'][None]);mask=torch.tensor(packet['atlas']['valid'][None])
        tile=patches(picture)[0,r['patch']];valid_tile=patches(mask)[0,r['patch']]
        rgb.append(tile.numpy());normalized.append(contrast(tile[None],valid_tile[None],rules)[0].numpy())
        bank_rows.append(dict(prototype=j,class_index=r['class_index'],image_index=r['image_index'],image_id=r['image_id'],image_sha256=r['image_sha256'],physical_tyre_id=r['physical_tyre_id'],patch=r['patch'],valid_fraction=r['valid_fraction'],glare_fraction=r['glare_fraction'],chroma_fraction=r['chroma_fraction'],mean_gradient=r['mean_gradient'],low_texture=r['low_texture'],role='train',status='UNTRAINED_REFERENCE_PROPOSAL_DO_NOT_INSTALL'))
    assert weights_digest(model)==original_sha
    artifacts={}
    if rgb: artifacts={'candidate_rgb.npy':np.stack(rgb),'candidate_local_contrast.npy':np.stack(normalized)}
    store.commit(dict(store.state,phase='completed',next_image=386),{'candidate_bank':bank_rows},partition=8,artifacts=artifacts)
    return 0


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--folder',required=True);parser.add_argument('--limit-images',type=int)
    args=parser.parse_args();cfg=json.loads(Path(args.config).read_text(encoding='utf-8'))
    def stopping(*_):
        target=Path(cfg['control_root'])/'STOP_REQUEST.json'
        target.write_text('{"reason":"catchable_worker_signal"}',encoding='utf-8')
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,stopping)
    sys.exit(run(cfg,Path(args.folder),args.limit_images))

if __name__=='__main__': main()
