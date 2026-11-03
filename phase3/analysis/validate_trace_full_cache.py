"""All570 existing images through the new cache, without training or remote writes."""
import sys,time
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'tread_two/trace_batch';sys.path.insert(0,str(SRC))
from common import atomic,read,digest,seed_all
from config import configuration
from cache import prepare
from data import schedule
import torch
torch.set_num_threads(4);seed_all(1)
valid=ROOT/'tread_two/trace_validation';out=valid/('fixture_full_cache_'+str(time.time_ns()))
folder=Path(read(ROOT/'tread_two/nb17_completion/LATEST.json')['path'])
cfg=configuration();cache=prepare(out,ROOT.parent/'phase2/data/phase2_dataset_v1',folder/'remote/exports/final.pt',digest(cfg),'cuda:0' if torch.cuda.is_available() else 'cpu')
cache.verify()
assert len(cache.rows)==570 and {r:sum(s['role']==r for s in cache.rows) for r in ('train','validation','test')}==dict(train=386,validation=81,test=103)
for epoch in range(1,61):
    batches=schedule(cache.rows,1,epoch);assert len(batches)==49
    assert all(len(set(ids))==8 for ids,_ in batches)
    assert sum(sum(flags) for _,flags in batches)==386 and sum(sum(not x for x in flags) for _,flags in batches)==6
atomic(valid/'FULL_CACHE.json',dict(status='passed',cache=str(out),images=570,roles=dict(train=386,validation=81,test=103),chunks=len(cache.index['chunks']),
       source_inventory=cfg['sources'],cache_hash=cache.index['scientific_hash'],cache_bytes=sum(m['bytes'] for m in cache.index['files'].values()),
       all60_schedules_checked=True,unique_images_per_update=8,primary_per_epoch=386,padding_per_epoch=6,
       frozen_geometry_forwards=570,new_optimizer_updates=0,heldout_fitted_statistics=False,remote_writes=0,credentials_read=False,
       device='GTX1650_single_GPU_local' if torch.cuda.is_available() else 'CPU',actual_dual_T4=False))
print(read(valid/'FULL_CACHE.json'),flush=True)
