"""Measured allocator peaks, RSS, inode-aware working/staging accounting."""
import os,shutil,time
from pathlib import Path
import torch
import psutil

def peaks_reset():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            # A visible but never-used secondary device may not yet have an
            # allocator. A non-random four-byte warm-up creates its context.
            with torch.cuda.device(i):
                warm=torch.empty(1,device='cuda:'+str(i));del warm
                torch.cuda.synchronize(i);torch.cuda.reset_peak_memory_stats(i)

def records(root,stage,unit):
    host=psutil.Process().memory_info().rss;free=shutil.disk_usage(root).free;rows=[]
    for i in range(torch.cuda.device_count()):
        rows.append(dict(stage=stage,device='cuda:'+str(i),allocated_bytes=torch.cuda.memory_allocated(i),reserved_bytes=torch.cuda.memory_reserved(i),
            max_allocated_bytes=torch.cuda.max_memory_allocated(i),max_reserved_bytes=torch.cuda.max_memory_reserved(i),peak_reset_unit=unit,host_RSS_bytes=host,free_bytes=free,measurement_scope='allocator_peak_since_explicit_update_reset'))
    if not rows:rows=[dict(stage=stage,device='cpu',host_RSS_bytes=host,free_bytes=free,measurement_scope='CPU_RSS_sample_no_CUDA_peak')]
    return rows

def disk(root):
    root=Path(root);seen=set();physical=0;logical=0
    for path in root.rglob('*'):
        if not path.is_file():continue
        st=path.stat();logical+=st.st_size;key=(st.st_dev,st.st_ino)
        if key not in seen:physical+=st.st_size;seen.add(key)
    return dict(logical_bytes=logical,unique_inode_bytes=physical,free_bytes=shutil.disk_usage(root).free)

def guard(root,cfg):
    status=disk(root)
    assert status['free_bytes']>=cfg['storage']['reserve'],'Storage reserve breached; preserve and publish last complete state'
    assert status['unique_inode_bytes']<=cfg['storage']['peak_ceiling'],'Working/staging storage ceiling exceeded'
    return status
