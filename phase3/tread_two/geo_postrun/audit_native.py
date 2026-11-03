"""Every native body, final recovery state, and original-mask evidence checks; zero forwards."""
import csv,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common import canonical,read,sha,atomic,digest
from inputs import PIN,sigmoid

def semantic(value):
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            y=x.detach().cpu().contiguous();h.update(canonical([str(y.dtype),list(y.shape)]));h.update(y.numpy().tobytes())
        elif isinstance(x,np.ndarray):h.update(canonical([str(x.dtype),list(x.shape)]));h.update(x.tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(list,tuple)):
            for item in x:visit(item)
        elif isinstance(x,(bytes,bytearray)):h.update(x)
        else:h.update(canonical(x))
    visit(value);return h.hexdigest()

def run(saved,core,truth):
    tables=core['tables'];index=saved.table('native_chunks_index');assert index.file.nunique()==len(index)
    assert len(index)==8455  # excludes two immutable mean/count control arrays
    obs=core['obs'].set_index('sample');regions=core['regions'].set_index(['sample','region']);gl=core['geometry'].set_index('sample')
    inventory=[];geometry_checks=[];points_checks=[];vector_checks=[]
    point_values=core['points'].pivot(index='sample',columns=['guide','side'],values='prediction').sort_index(axis=1)
    quality_values=core['quality'].pivot(index='sample',columns='quantity',values='predicted_error').reindex(columns=['tyre_error','tread_error','human_point_error'])
    support=saved.table('patch_support').pivot(index='sample',columns='patch',values='pooling_mass').sort_index(axis=1)
    done_path=tables.root/'NATIVE_PROGRESS.json'
    # Progress is durable in a direct CSV. Resume skips only already checked
    # immutable bodies with the same source manifest and matching table prefix.
    partial=tables.root/'native_audit.partial.csv';completed={}
    columns=['file','kind','first_sample','last_sample','samples','shape','dtype','cells','minimum','maximum','finite','sha256']
    if partial.exists():
        prior=pd.read_csv(partial);completed={r.file:r._asdict() for r in prior.itertuples(index=False)}
    stream=partial.open('a',encoding='utf-8',newline='');writer=csv.DictWriter(stream,fieldnames=columns,lineterminator='\n')
    if partial.stat().st_size==0:writer.writeheader()
    try:
        for position,row in enumerate(index.itertuples(),1):
            path=saved.root/row.file;assert sha(path)==row.sha256
            value=np.load(path,mmap_mode='r',allow_pickle=False);assert len(value)==int(row.last_sample-row.first_sample+1)
            assert value.dtype in (np.dtype('float32'),np.dtype('bool')) and np.isfinite(value).all()
            record=dict(file=row.file,kind=row.kind,first_sample=int(row.first_sample),last_sample=int(row.last_sample),samples=len(value),shape=str(list(value.shape)),dtype=str(value.dtype),cells=value.size,minimum=float(value.min()),maximum=float(value.max()),finite=True,sha256=row.sha256)
            if row.file not in completed:writer.writerow(record)
            inventory.append(record)
            if row.kind=='points':
                a=np.asarray(value).reshape(-1,3,2);assert (a>=0).all() and (a<=1).all() and (a[...,1]>a[...,0]).all()
                assert np.array_equal(np.asarray(value),point_values.loc[row.first_sample:row.last_sample].to_numpy(dtype='float32'))
                points_checks.append(dict(file=row.file,samples=len(value),ordered=True,bounded=True))
            if row.kind=='quality':assert np.array_equal(np.asarray(value),quality_values.loc[row.first_sample:row.last_sample].to_numpy(dtype='float32'))
            if row.kind in ('v0_full','v1_full'):
                weights=support.loc[row.first_sample:row.last_sample].to_numpy(dtype='float64')
                pooled=(np.asarray(value,dtype='float64')*weights[...,None]).sum(1)
                view=int(row.kind[1]);reported=core['global_z'][int(row.first_sample):int(row.last_sample)+1,view]
                assert np.allclose(pooled,reported,atol=2e-6,rtol=2e-6),(row.file,'native_pooled_vector')
                vector_checks.append(dict(file=row.file,view=view,samples=len(value),maximum_absolute_difference=float(np.abs(pooled-reported).max()),passed=True))
            if row.kind=='geometry_logits':
                differences=[]
                for offset,raw in enumerate(value):
                    sample=int(row.first_sample)+offset;case=obs.loc[sample];gt=truth[case.image_id]
                    x=raw.astype('float64');p=sigmoid(x);valid=gt['valid'][None];target=gt['target'].astype('float64')
                    bce=((np.maximum(x,0)-x*target+np.log1p(np.exp(-np.abs(x))))*valid).sum()
                    den=float(2*valid.sum());soft=(2*(p*target*valid).sum((1,2))+1)/((p*valid).sum((1,2))+(target*valid).sum((1,2))+1)
                    dice_loss=float((1-soft).sum());recorded=gl.loc[sample]
                    assert np.isclose(bce,recorded.bce_sum,atol=2e-3,rtol=2e-5),(sample,'raw_BCE')
                    assert den==recorded.bce_count
                    assert np.isclose(dice_loss,recorded.dice_loss_sum,atol=5e-7,rtol=2e-5),(sample,'raw_soft_Dice')
                    # At the rounding boundary, CPU FP64 sigmoid may not reproduce
                    # CUDA FP32's exact.5 tie. Bound only those explicit cells.
                    binary=p>=.5;fg=gt['foreground'].astype('int64');available=gt['available'].astype('int64')
                    near=np.abs(p-.5)<2e-7
                    for region in (0,1):
                        r=regions.loc[(sample,region)];tp=int((binary[region]*fg[region]).sum());fp=int((binary[region]*(available-fg[region])).sum());fn=int(((~binary[region])*fg[region]).sum())
                        ambiguity=int((near[region]*available).sum());assert max(abs(tp-r.tp),abs(fp-r.fp),abs(fn-r.fn))<=ambiguity,(sample,'raw_confusion')
                    differences.append(abs(bce-recorded.bce_sum))
                geometry_checks.append(dict(file=row.file,samples=len(value),maximum_BCE_sum_difference=max(differences),passed=True))
            del value
            if position%32==0:
                stream.flush();__import__('os').fsync(stream.fileno());atomic(done_path,dict(files=position,total=len(index),phase='native_values_and_existing_mask_truth'))
            if position%256==0:print('Native audit:',position,'/',len(index),flush=True)
    finally:stream.flush();stream.close()
    tables.write('native_inventory',inventory);tables.write('raw_geometry_audit',geometry_checks);tables.write('point_order_audit',points_checks);tables.write('native_vector_pooling_audit',vector_checks)
    pointer=read(saved.root/'POINTER.json');assert pointer['protocol']==PIN['protocol'] and pointer['unit']==9877
    state_path=saved.root/pointer['state'];assert sha(state_path)==pointer['state_meta']['sha256']
    state=torch.load(state_path,map_location='cpu',weights_only=False);assert state['unit']==9877 and state['cursor']['phase']=='completed' and state['cursor']['update']==2940
    assert state['cursor']['sample']==30447 and state['cursor']['best'] is None
    assert state['cursor']['counters']==dict(geometry=2940,texture=2940,quality=2940)
    for name,meta in state['closed'].items():assert name in saved.manifest['files'] and all(saved.manifest['files'][name][k]==meta[k] for k in ('bytes','sha256'))
    ledger=saved.table('unit_ledger').sort_values('unit');assert len(ledger)==9877 and ledger.unit.tolist()==list(range(1,9878))
    assert ledger.current_unit_sha256.iloc[-1]==pointer['chain']
    assert ledger.previous_unit_sha256.iloc[1:].tolist()==ledger.current_unit_sha256.iloc[:-1].tolist()
    state_keys={k:state[k] for k in ('model','optimizer','rng','cursor','precision','scheduler')}
    assert semantic(state_keys)==ledger.state_semantic_sha256.iloc[-1]
    last=ledger.iloc[-1];assert last.csv_prefix_sha256==digest({}) and not state['native_prefix']
    assert digest(dict(previous=last.previous_unit_sha256,unit=int(last.unit),state=last.state_semantic_sha256,csv={},native={}))==pointer['chain']
    assert set(state['rng'])=={'python','numpy','cpu','cuda'} and len(state['rng']['cuda'])==2
    def finite_tree(value):
        if torch.is_tensor(value):assert torch.isfinite(value).all()
        elif isinstance(value,dict):
            for child in value.values():finite_tree(child)
        elif isinstance(value,(tuple,list)):
            for child in value:finite_tree(child)
    finite_tree(state['optimizer'])
    epoch_rows=[]
    for epoch in range(1,61):
        weights=torch.load(saved.root/f'epochs/epoch_{epoch:03d}.pt',map_location='cpu',weights_only=True)
        assert all(torch.isfinite(v).all() for v in weights.values()) and sum(v.numel() for v in weights.values())==355879
        for name,value in weights.items():epoch_rows.append(dict(epoch=epoch,parameter=name,numel=value.numel(),l2=float(value.norm()),max_absolute=float(value.abs().max()),finite=True))
    exported=torch.load(saved.root/'exports/final.pt',map_location='cpu',weights_only=True)
    final=torch.load(saved.root/'epochs/epoch_060.pt',map_location='cpu',weights_only=True)
    assert semantic(final)==semantic(state['model'])==semantic(exported['trainable'])
    anchor=torch.load(saved.root/'anchor.pt',map_location='cpu',weights_only=True);assert semantic(anchor)==semantic(exported['anchor'])
    assert sha(saved.root/'anchor.pt')==saved.contract['scientific_plan']['anchor']['weights_sha256']
    tables.write('epoch_parameter_audit',epoch_rows)
    from latent_diagnosis import run as diagnose
    token_diagnosis=diagnose(saved,core,index)
    counts=saved.table('source_exposure');assert int(counts.SSL_skipped_draws.sum())==0
    result=dict(status='passed',native_shards=len(index),native_control_arrays=2,raw_geometry_samples=sum(r['samples'] for r in geometry_checks),
                point_samples=sum(r['samples'] for r in points_checks),complete_unit_links=9877,last_state_semantic_hash_verified=True,
                epoch_weights_finite=60,final_export_matches_epoch60_and_last_state=True,frozen_anchor_unchanged=True,native_points_quality_match_CSV=True,native_global_pooling_verified=True,
                ground_truth_masks_reloaded=True,image_model_forwards=0,optimizer_updates=0,
                final_unit_digest_verified=True,optimizer_finite=True,two_CUDA_RNG_states_retained=True,
                token_diagnosis=token_diagnosis,
                limitation='Historical ledger link continuity verified; historical complete prefix maps/optimizer states are not all permanently retained, so every historical unit digest is not independently re-derived. Anchor drift uses retained sums/counts; no new anchor forwards.')
    atomic(tables.root/'NATIVE_VERIFICATION.json',result)
    return index,result

def dense(saved,core,truth):
    tables=core['tables'];obs=core['obs'].set_index('sample');expected=core['regions'].set_index(['sample','region']);sums={};checked=0
    endpoint_ids=core['obs'][core['obs'].stage=='endpoint']['sample'].sort_values().to_numpy(dtype='int64');assert len(endpoint_ids)==570
    slots={int(sample):i for i,sample in enumerate(endpoint_ids)};seen=np.zeros(570*2*128*96,dtype=bool)
    for path in saved.paths('dense_pixels'):
        for frame in pd.read_csv(path,chunksize=131072):
            if not len(frame):continue
            for (sample,region),block in frame.groupby(['sample','region']):
                sample,region=int(sample),int(region);gt=truth[obs.loc[sample,'image_id']]
                y=block['row'].to_numpy(dtype='int64');x=block['column'].to_numpy(dtype='int64');fg=gt['foreground'][region,y,x].astype('int64');valid=gt['available'][y,x].astype('int64')
                assert region in (0,1) and ((0<=y)&(y<128)).all() and ((0<=x)&(x<96)).all()
                positions=((slots[sample]*2+region)*128+y)*96+x
                assert len(np.unique(positions))==len(positions) and not seen[positions].any();seen[positions]=True
                assert np.array_equal(fg,block.target_foreground_subpixels.to_numpy()) and np.array_equal(valid,block.valid_subpixels.to_numpy())
                pred=block.predicted.to_numpy(dtype='int64');assert np.isfinite(block.probability).all() and ((block.probability>=0)&(block.probability<=1)).all()
                assert np.array_equal(pred,(block.probability.to_numpy()>=.5).astype('int64'))
                add=np.array([(pred*fg).sum(),(pred*(valid-fg)).sum(),((1-pred)*fg).sum(),len(block)],dtype='int64')
                sums[(sample,region)]=sums.get((sample,region),np.zeros(4,dtype='int64'))+add
            checked+=len(frame)
    assert checked==570*2*128*96 and seen.all()
    rows=[]
    for (sample,region),v in sums.items():
        actual=expected.loc[(sample,region)];assert tuple(v[:3])==(actual.tp,actual.fp,actual.fn) and v[3]==128*96
        rows.append(dict(sample=sample,region=region,TP=int(v[0]),FP=int(v[1]),FN=int(v[2]),cells=int(v[3]),counts_match=True))
    tables.write('dense_confusion_audit',rows)
    return dict(rows=checked,source_images=570,regions=1140,all_GT_counts_and_confusion_exact=True)
