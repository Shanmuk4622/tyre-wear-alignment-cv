"""Exploratory addendum: distinguish mask-placement variation from texture signal.

Added after the main aggregation results exposed high masked-context rank.
No new aggregation candidate, parameter learning, image forward or selection.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='4'
import sys
from pathlib import Path
import numpy as np
import pandas as pd
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];DEST=ROOT/'texture_aggregation_20261102';sys.path.insert(0,str(ROOT/'tread_two/trace_batch'))
from common import read,atomic,file_meta,digest
from review import Saved
from texture_aggregation_math import spectrum,within
addendum=dict(version='TRACE-mask-confounds-v1',relationship='exploratory follow-up after main results; not predeclared in primary aggregation protocol',
    inputs='same570 final images /two retained views; full context, masked context, hidden plans and saved projected vectors',
    calculations='role and component-centered linear centered-kernel alignment of feature Gram matrices with hidden-tile-plan Gram; mask rank/counts',
    interpretation='association only; recorded masks are deterministic image-addressed plans, not a new randomized trial',
    fitting=False,training=False,image_forwards=0,optimizer_updates=0,new_photos=0,selection_changes=False)
path=DEST/'MASK_ADDENDUM_PROTOCOL.json'
if path.exists():assert read(path)==addendum
else:atomic(path,addendum)
old=Path(read(ROOT/'tread_two/nb24_completion/LATEST.json')['path']);saved=Saved(old/'remote/fit');alignment=pd.read_csv(DEST/'IMAGE_ALIGNMENT.csv');ids=alignment.observation.to_numpy()
hidden=saved.get('hidden',ids).astype('float64');full=saved.get('full_context',ids).astype('float64');masked=saved.get('masked_context',ids).astype('float64')
features={'full_context_concat':full.reshape(570,2,-1),'masked_context_concat':masked.reshape(570,2,-1),'masked_minus_full_context':(masked-full).reshape(570,2,-1),'actual_projected6':saved.get('projected',ids).astype('float64')}
rows=[];summary=[]
for role in ('train','validation','test'):
    use=alignment.role.to_numpy()==role;comp=alignment.component.to_numpy()[use]
    for view in (0,1):
        h=hidden[use,view];rank,_=spectrum(h);summary.append(dict(role=role,view=view,images=len(h),unique_plans=len(np.unique(h,axis=0)),hidden_tiles_min=int(h.sum(1).min()),hidden_tiles_max=int(h.sum(1).max()),uniform_pooled_hidden_plan_std=float((h.mean(1)).std()),mask_effective_rank=rank['effective_rank']))
        for centered in (False,True):
            hh=within(h,comp) if centered else h-h.mean(0);kh=hh@hh.T
            for name,body in features.items():
                x=body[use,view];x=within(x,comp) if centered else x-x.mean(0);kx=x@x.T
                den=np.linalg.norm(kx)*np.linalg.norm(kh);cka=float((kx*kh).sum()/den) if den else None
                rows.append(dict(role=role,view=view,component_centered=centered,feature=name,images=len(x),linear_CKA_with_mask_plan=cka))
pd.DataFrame(rows).to_csv(DEST/'MASK_PLAN_ALIGNMENT.csv',index=False);pd.DataFrame(summary).to_csv(DEST/'MASK_PLAN_SUMMARY.csv',index=False)
mass=saved.get('mass',ids);support=[]
for role in ('train','validation','test'):
    use=alignment.role.to_numpy()==role
    for view in (0,1):
        count=(mass[use,view]>0).sum(1)
        support.append(dict(role=role,view=view,images=int(use.sum()),minimum_supported_tiles=int(count.min()),maximum_supported_tiles=int(count.max()),images_with_unsupported_tiles=int((count<32).sum()),minimum_mass_sum=float(mass[use,view].sum(1).min()),maximum_mass_sum=float(mass[use,view].sum(1).max())))
pd.DataFrame(support).to_csv(DEST/'SUPPORT_SUMMARY.csv',index=False)
atomic(DEST/'MASK_ADDENDUM_RECEIPT.json',dict(status='passed',protocol=digest(addendum),source=file_meta(Path(__file__)),files={n:file_meta(DEST/n) for n in ['MASK_PLAN_ALIGNMENT.csv','MASK_PLAN_SUMMARY.csv','SUPPORT_SUMMARY.csv']},new_optimizer_updates=0,image_forwards=0,network_calls=0,remote_writes=0))
print(pd.DataFrame(rows).query('view==0 and role!="train"').to_string(index=False))
