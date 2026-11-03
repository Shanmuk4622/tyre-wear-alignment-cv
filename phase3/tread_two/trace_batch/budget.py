"""Extrapolate ALL actual formatted gate families/native bodies before long fitting."""
import csv
from collections import defaultdict
from pathlib import Path
from common import read,atomic,file_meta

def project(engine):
    store=engine.store;root=store.root;groups=defaultdict(int);counts=defaultdict(int)
    for name,meta in store.closed.items():
        if not name.startswith('csv/') or name.endswith('empty.csv'):continue
        family=Path(name).parts[1]
        with (root/name).open(encoding='utf-8',newline='') as stream:
            reader=csv.DictReader(stream)
            for row in reader:
                # The exact nine/seventeen-digit serializer has already run.
                import io
                out=io.StringIO(newline='');writer=csv.writer(out,lineterminator='\n');writer.writerow([row[k] for k in reader.fieldnames])
                stage=row['stage'];groups[(family,stage)]+=len(out.getvalue().encode());counts[(family,stage)]+=1
    ratios={'train':23520/16,'baseline':81/8,'validate':4860/8,'endpoint_final':570/8,'endpoint_selected':570/8,
            'epoch':60.,'retrieval_final':570/8,'retrieval_selected':570/8,'initialization':1.}
    for variant in engine.cfg['probes'][1:]:ratios['probe_'+variant]=103/8
    rows=[];csv_total=0
    for (family,stage),value in groups.items():
        scale=ratios.get(stage,1.)
        if family=='source_exposure':scale=386*60/8
        if family=='state_ledger':scale=5000/max(1,counts[(family,stage)])
        if family in ('resource_measurements','timings','numerical_events') and stage in ('endpoint','probe'):scale=200.
        expected=int(value*scale)+1024
        rows.append(dict(family=family,stage=stage,fixture_rows=counts[(family,stage)],fixture_bytes=value,scale=scale,projected_bytes=expected));csv_total+=expected
        if stage=='endpoint_final' and not any(k[1]=='endpoint_selected' and k[0]==family for k in groups):csv_total+=expected # possible selected endpoint
    native_bytes=sum(m['bytes'] for n,m in store.closed.items() if n.startswith('native/'))
    fixture_obs=engine.cursor['observation'];max_observations=81+23520+4860+2*570+4*103
    native_projection=int(native_bytes*max_observations/max(1,fixture_obs))
    cache_bytes=sum(m['bytes'] for m in engine.cache.index['files'].values())
    epoch_size=(root/'epochs/epoch_001.pt').stat().st_size
    state_size=(root/read(root/'POINTER.json')['state']).stat().st_size
    logical=csv_total+native_projection+cache_bytes+epoch_size*62+state_size*160+100_000_000
    peak=int(logical*1.2+128*1024**2)
    from resources import disk
    usage=disk(engine.root.parent);required=max(0,peak-usage['unique_inode_bytes'])+engine.cfg['storage']['reserve']
    cfg=engine.cfg['storage'];passed=logical<=cfg['logical_ceiling'] and peak<=cfg['peak_ceiling'] and usage['free_bytes']>=required
    result=dict(status='passed' if passed else 'failed',actual_formatted_families_measured=True,projection_rows=rows,
        CSV_bytes=csv_total,native_bytes=native_projection,cache_bytes=cache_bytes,logical_bytes=logical,projected_peak_bytes=peak,
        logical_ceiling=cfg['logical_ceiling'],peak_ceiling=cfg['peak_ceiling'],uncertainty_factor=1.2,
        measured_current_unique_bytes=usage['unique_inode_bytes'],actual_free_bytes=usage['free_bytes'],required_free_bytes=required,
        coverage='Both potential endpoints, all training/validation predictions, all token arrays, all epoch weights, rolling state, transport reserve and active staging',
        hardlink_staging_required=True,scratch_1TB_assumed=False)
    atomic(engine.root/'STORAGE_PROJECTION.json',result);return result
