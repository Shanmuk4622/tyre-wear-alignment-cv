"""Independently recompute every train objective from the disposable direct CSVs."""
import csv,json,sys
from pathlib import Path
import numpy as np
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tread_two/geo_pilot'))
from common import atomic,read
receipt=read(ROOT/'tread_two/geo_validation/RESUME.json');root=Path(receipt['fixture'])/'direct/run'
def table(family):
    records=[]
    for path in (root/'csv'/family).glob('*.csv'):
        with path.open(encoding='utf-8',newline='') as stream:records+=list(csv.DictReader(stream))
    for path in (root/'active').glob('csv_'+family+'_*.csv'):
        # Only the selected active prefix is part of the public scientific table;
        # completed fixtures seal every data family except the final unit ledger.
        if family=='unit_ledger':continue
    return records
def f(row,key):return float(row[key]) if row[key] else None
observations=table('sample_observations');geometry={int(r['sample']):r for r in table('geometry_losses')}
qualities=table('quality_predictions');vectors=table('projected_embeddings');masked=table('masked_patch_metrics');batches=table('batch_objectives');results=[]
for row in batches:
    update=int(row['update']);ids={int(r['sample']) for r in observations if r['stage']=='train' and int(r['parameter_update'])==update-1}
    assert len(ids)==8
    values=[geometry[i] for i in ids]
    def ratio(n,d):return sum(f(v,n) for v in values)/max(sum(f(v,d) for v in values),1e-12)
    geom=ratio('bce_sum','bce_count')+ratio('dice_loss_sum','dice_region_count')+2*ratio('point_loss_sum','point_weight_count')+.1*ratio('containment_sum','containment_count')+.1*ratio('row_agreement_sum','row_agreement_count')+.1*(ratio('anchor_mask_sum','anchor_mask_count')+ratio('anchor_point_sum','anchor_point_count'))
    quality=[q for q in qualities if int(q['sample']) in ids and q['target_available']=='1']
    qloss=sum(f(q,'loss_sum') for q in quality)/max(sum(f(q,'loss_count') for q in quality),1)
    eligible=sorted(int(r['sample']) for r in observations if int(r['sample']) in ids and r['SSL_valid']=='1')
    if row['texture_step_skipped']=='1':texture=0.
    else:
        projected=[]
        for view in (0,1):
            lookup={(int(v['sample']),int(v['dimension'])):f(v,'value') for v in vectors if int(v['sample']) in ids and int(v['view'])==view}
            projected.append(np.array([[lookup[(sample,d)] for d in range(64)] for sample in eligible]))
        inv=float(np.square(projected[0]-projected[1]).mean());var=cov=0.
        for z in projected:
            matrix=np.cov(z,rowvar=False,ddof=1);var+=float(np.maximum(0,1-np.sqrt(np.diag(matrix)+1e-4)).mean());cov+=float(((matrix**2).sum()-(np.diag(matrix)**2).sum())/64)
        patch=[m for m in masked if int(m['sample']) in eligible];num=sum(f(m,'huber_sum') for m in patch);den=sum(f(m,'huber_weight_count') for m in patch)
        texture=25*inv+25*var+cov+.5*num/den
    calculated=dict(geometry_objective=geom,quality_objective=qloss,SSL_objective=texture,total_loss=geom+.25*qloss+f(row,'texture_ramp')*texture)
    errors={k:abs(f(row,k)-value) for k,value in calculated.items()}
    assert all(np.isclose(f(row,k),value,atol=3e-5,rtol=1e-5) for k,value in calculated.items()),(update,errors)
    results.append(dict(update=update,all_losses_reconstructed=True,absolute_differences=errors))
report=dict(status='passed',scope='independent_CSV_only_geometry_quality_SSL_and_total_reductions',source_inventory_sha256=receipt['source_inventory_sha256'],updates=results)
atomic(ROOT/'tread_two/geo_validation/LOSS_ACCOUNTING.json',report);print(json.dumps(report,indent=2))
