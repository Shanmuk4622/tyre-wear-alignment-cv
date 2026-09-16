"""Verify final HRNet metadata and recompute all published held-out point errors."""
import json,math,statistics,sys
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tyrelib'))
import hrnet_protocol as p
REV='a92c0f9c5c1b78c6a06e13d51e18722195230658'
KEY='351c6638783996f46cf9efd99ec6689de726e045288054f240193cea61385712'
BASE=f's9/{p.VERSION}/{KEY}';OUT=ROOT/'outputs/hrnet_final_audit';OUT.mkdir(exist_ok=True);total=0
def get(path,tree=False):
    global total
    url=(f'https://huggingface.co/api/datasets/{p.REPO}/tree/{REV}/{BASE}/{path}' if tree else
         f'https://huggingface.co/datasets/{p.REPO}/resolve/{REV}/{BASE}/{path}')
    with requests.get(url,timeout=45,stream=True) as r:
        r.raise_for_status();raw=bytearray()
        for chunk in r.iter_content(65536):
            raw.extend(chunk);total+=len(chunk);assert len(raw)<2*1024**2 and total<5*1024**2
    obj=json.loads(raw)
    p.write(OUT/(path.replace('/','_')+('_tree.json' if tree else '')),obj)
    return obj
cfg=get('report/CONTRACT.json');assert p.sha(p.canonical(cfg))==KEY
report=get('report/REPORT.json');assert report['protocol']==KEY
train=[r for r in cfg['rows'] if r['role']=='train'];test={r['pilot_id']:r for r in cfg['rows'] if r['role']=='test'}
constant=[statistics.mean(r['x'][i] for r in train) for i in range(6)]
baseline=statistics.mean(abs(constant[i]-r['x'][i]) for r in test.values() for i in range(6))
results=[]
for seed in (1,2,3):
    base=f'runs/seed{seed}';status=get(base+'/STATUS.json');hist=get(base+'/HISTORY.json')
    assert status['status']=='completed' and status['completed_epochs']==60 and status['next_batch_cursor']==0
    assert status['protocol']==KEY and status['seed']==seed
    assert len(hist)==2160 and [(r['epoch'],r['batch']) for r in hist]==[(e,b) for e in range(1,61) for b in range(1,37)]
    assert all(math.isfinite(r['loss']) and math.isfinite(r['lr']) for r in hist)
    tree=get(base,True);ck=next(x for x in tree if x['path'].endswith('/state.pt'))
    assert ck['lfs']['oid']==status['checkpoint_sha256']
    assert len([x for x in tree if '/validation_epoch' in x['path'] and x['path'].endswith('.json')])==60
    metric=get(base+'/TEST_FINAL.json');repair=get(base+'/RUNTIME_REPAIR.json')
    assert repair['regression']['status']=='passed'
    rr=metric['records'];assert len(rr)==144
    assert {(r['pilot_id'],r['point']) for r in rr}=={(pid,n) for pid in test for n in p.POINTS}
    for r in rr:
        ref=test[r['pilot_id']];target=ref['x'][p.POINTS.index(r['point'])]
        assert r['label']==target and r['tyre']==ref['session'] and 0<=r['prediction']<=1
        err=abs(r['prediction']-target)
        assert math.isclose(err,r['error_width_fraction'],abs_tol=1e-12)
        assert math.isclose(err*(ref['width']-1),r['error_px'],abs_tol=1e-9)
    score=statistics.mean(r['error_width_fraction'] for r in rr)
    published=next(x for x in report['results'] if x['seed']==seed)
    assert math.isclose(score,published['mean_width_error'],abs_tol=1e-12)
    assert math.isclose(score,metric['mean_width_error'],abs_tol=1e-12)
    assert math.isclose(statistics.median(r['error_px'] for r in rr),metric['median_px_error'],abs_tol=1e-9)
    assert math.isclose(baseline,published['train_mean_baseline'],abs_tol=1e-12)
    assert published['runtime_repair_sha256']==repair['source_sha256']
    for tyre,value in published['per_tyre'].items():assert math.isclose(value,statistics.mean(r['error_width_fraction'] for r in rr if r['tyre']==tyre),abs_tol=1e-12)
    results.append(dict(seed=seed,mean_width_error=score,mean_px_error=statistics.mean(r['error_px'] for r in rr),
        median_px_error=metric['median_px_error'],per_tyre=published['per_tyre']))
avg=statistics.mean(r['mean_width_error'] for r in results)
audit=dict(status='verified',revision=REV,completed_seeds=3,epochs_per_seed=60,optimizer_steps=6480,
    test_images=24,test_tyres=2,point_records_recomputed=432,results=results,
    mean_seed_width_error=avg,mean_seed_px_error=statistics.mean(r['mean_px_error'] for r in results),
    constant_baseline_width_error=baseline,relative_error_reduction=1-avg/baseline,downloaded_bytes=total,
    checkpoint_check='LFS metadata hashes match statuses; tensor payloads not downloaded',
    comparison='constant training-label mean only; no matched SegFormer comparison',full_s9_complete=False)
p.write(OUT/'AUDIT.json',audit);print(json.dumps(audit,indent=2))
