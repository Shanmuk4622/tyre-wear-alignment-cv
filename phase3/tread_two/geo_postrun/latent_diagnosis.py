"""Descriptive component/within-component scatter and pre/post-pooling token spaces."""
import numpy as np
from inputs import latent

def run(saved,core,index):
    tables=core['tables'];endpoint=core['case'][core['case'].stage=='endpoint'].merge(core['source'][['image_id','role']],on='image_id').sort_values('sample')
    ids=endpoint['sample'].to_numpy(dtype='int64');assert len(ids)==570
    roles=[('all',np.ones(570,dtype=bool))]+[(r,(endpoint.role==r).to_numpy()) for r in ['train','validation','test']]
    components=endpoint.leakage_component.to_numpy();variance=[]
    for space,body in [('projected64',core['projected']),('global96',core['global_z'])]:
        z=np.asarray(body[ids,0],dtype='float64')
        for role,mask in roles:
            a=z[mask];labels=components[mask];within=np.empty_like(a)
            for name in np.unique(labels):which=labels==name;within[which]=a[which]-a[which].mean(0)
            centered=a-a.mean(0);between=centered-within
            total=float((centered**2).sum());b=float((between**2).sum());w=float((within**2).sum());assert np.isclose(total,b+w,atol=1e-10,rtol=1e-12)
            variance.append(dict(role=role,space=space,images=len(a),conservative_components=len(np.unique(labels)),total_scatter=total,between_component_scatter=b,within_component_scatter=w,
                                 between_fraction=b/total if total else 0.,within_fraction=w/total if total else 0.,original_rank=latent(a)['effective_rank'],within_component_centered_rank=latent(within)['effective_rank'],
                                 interpretation='Descriptive conservative-group decomposition; grouping is not proven physical identity or a causal explanation'))
    tables.write('component_variance_decomposition',variance)
    support=saved.table('patch_support');mass=support.pivot(index='sample',columns='patch',values='pooling_mass').sort_index(axis=1).loc[ids].to_numpy(dtype='float64')
    coverage=support.pivot(index='sample',columns='patch',values='valid_fraction').sort_index(axis=1).loc[ids].to_numpy(dtype='float64');supported=coverage>=.25
    tokens_summary=[];patch_rows=[]
    for layer in ['encoded','full']:
        values=np.zeros((570,32,96),dtype='float32');seen=np.zeros(570,dtype=bool)
        for row in index[(index.kind=='v0_'+layer)&(index.last_sample>=ids.min())&(index.first_sample<=ids.max())].itertuples():
            choose=(ids>=row.first_sample)&(ids<=row.last_sample)
            if not choose.any():continue
            a=np.load(saved.root/row.file,mmap_mode='r',allow_pickle=False);assert not seen[choose].any()
            values[choose]=a[ids[choose]-int(row.first_sample)];seen[choose]=True
        assert seen.all()
        for role,mask in roles:
            a=values[mask].astype('float64');ok=supported[mask];centered=np.zeros_like(a);ranks=[]
            for patch in range(32):
                valid=ok[:,patch];subset=a[valid,patch];stats=latent(subset);ranks.append(stats['effective_rank']);centered[valid,patch]=subset-subset.mean(0) if len(subset) else subset
                patch_rows.append(dict(role=role,layer=layer,patch=patch,band=patch//4,column=patch%4,**stats))
            pooled=(a*mass[mask,:,None]).sum(1)
            tokens_summary.append(dict(role=role,layer=layer,images=len(a),supported_tile_observations=int(ok.sum()),flat_token_rank=latent(a[ok])['effective_rank'],
                                       per_position_mean_removed_rank=latent(centered[ok])['effective_rank'],median_position_rank=float(np.median(ranks)),pooled_rank=latent(pooled)['effective_rank'],
                                       interpretation='Saved view0 at final epoch; flattening includes position variation; centered values remove each position mean; no causal/promotion claim'))
    tables.write('endpoint_token_spaces',tokens_summary);tables.write('endpoint_patch_latent',patch_rows)
    return dict(component_rows=len(variance),token_space_rows=len(tokens_summary),position_rows=len(patch_rows),new_model_forwards=0)
