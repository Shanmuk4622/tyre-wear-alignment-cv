"""Bounded public metadata audit; no checkpoint/image downloads or HF writes."""
import json,math,statistics,sys
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tyrelib'))
import hrnet_protocol as p
REV='bbe586c6f00cf12ae4cac8b2e9cb4f875b272abb'
KEY='8013bf1e6418e6f884a353efa7ce892c31fbaac450f9d7caede636d9e61e2a18'
BASE=f's9/segformer-matched-2026-09-15-r1/{KEY}'
OUT=ROOT/'outputs/segformer_completion_audit';OUT.mkdir(exist_ok=True);total=0
def get(path,tree=False,base=BASE,rev=REV):
    global total
    url=(f'https://huggingface.co/api/datasets/{p.REPO}/tree/{rev}/{base}/{path}' if tree else
         f'https://huggingface.co/datasets/{p.REPO}/resolve/{rev}/{base}/{path}')
    with requests.get(url,timeout=45,stream=True) as r:
        r.raise_for_status();raw=bytearray()
        for part in r.iter_content(65536):raw.extend(part);total+=len(part);assert len(raw)<2*1024**2 and total<5*1024**2
        if tree:assert 'rel="next"' not in r.headers.get('Link',''), 'Handle tree pagination before counting'
    obj=json.loads(raw);p.write(OUT/((('hrnet_' if base!=BASE else '')+path.replace('/','_'))+('_tree.json' if tree else '')),obj)
    return obj
cfg=get('comparison/CONTRACT.json');assert p.sha(p.canonical(cfg))==KEY
report=get('comparison/REPORT.json');assert report['protocol']==KEY
paired=get('comparison/PAIRED_POINTS.json');assert len(paired)==432
smoke=get('smoke/STATUS.json');assert smoke['status']=='smoke_passed' and smoke['protocol']==KEY
assert smoke['max_parameter_difference']<=1e-5
hrbase='s9/hrnet-geometry-2026-09-15-r1/'+cfg['hrnet_protocol'];hrrev=cfg['hrnet_revision']
hrcfg=get('report/CONTRACT.json',base=hrbase,rev=hrrev)
assert cfg['rows']==hrcfg['rows'] and cfg['groups']==hrcfg['groups']
test={r['pilot_id']:r for r in cfg['rows'] if r['role']=='test'}
expected={(pid,n) for pid in test for n in p.POINTS};results=[]
assert {(r['seed'],r['pilot_id'],r['point']) for r in paired}=={(s,pid,n) for s in (1,2,3) for pid,n in expected}
for seed in (1,2,3):
    base=f'runs/seed{seed}';st=get(base+'/STATUS.json');hist=get(base+'/HISTORY.json');tree=get(base,True)
    assert st['status']=='completed' and st['completed_epochs']==60 and st['next_batch_cursor']==0 and st['protocol']==KEY and st['seed']==seed
    assert [(r['epoch'],r['batch']) for r in hist]==[(e,b) for e in range(1,61) for b in range(1,37)]
    assert all(math.isfinite(r['loss']) and math.isfinite(r['lr']) for r in hist)
    assert next(x for x in tree if x['path'].endswith('/state.pt'))['lfs']['oid']==st['checkpoint_sha256']
    assert sum('/validation_epoch' in x['path'] and x['path'].endswith('.json') for x in tree)==60
    repair=get(base+'/RESUME_REPAIR.json');assert repair['source_sha256']==p.file_sha(ROOT/'tyrelib/segformer_resume_repair.py')
    metric=get(base+'/TEST_FINAL.json');hr=get(base+'/TEST_FINAL.json',base=hrbase,rev=hrrev)
    rows=metric['records'];hrrows={(r['pilot_id'],r['point']):r for r in hr['records']}
    assert len(rows)==144 and {(r['pilot_id'],r['point']) for r in rows}==expected
    assert metric['coverage']==1 and all(not r['fallback_used'] and r['raw_prediction']==r['prediction'] for r in rows)
    for r in rows:
        target=test[r['pilot_id']];label=target['x'][p.POINTS.index(r['point'])];err=abs(r['prediction']-label)
        assert r['label']==label and r['tyre']==target['session'] and 0<=r['prediction']<=1
        assert math.isclose(err,r['error_width_fraction'],abs_tol=1e-12) and math.isclose(err*(target['width']-1),r['error_px'],abs_tol=1e-9)
        h=hrrows[r['pilot_id'],r['point']];assert h['label']==label and h['tyre']==r['tyre']
        assert math.isclose(abs(h['prediction']-label),h['error_width_fraction'],abs_tol=1e-12)
        pair=next(x for x in paired if (x['seed'],x['pilot_id'],x['point'])==(seed,r['pilot_id'],r['point']))
        assert math.isclose(pair['hrnet_error'],h['error_width_fraction'],abs_tol=1e-12)
        assert math.isclose(pair['segformer_error'],err,abs_tol=1e-12)
        assert math.isclose(pair['difference_segformer_minus_hrnet'],err-h['error_width_fraction'],abs_tol=1e-12)
    sg=statistics.mean(r['error_width_fraction'] for r in rows);hh=statistics.mean(r['error_width_fraction'] for r in hr['records'])
    pub=next(r for r in report['results'] if r['seed']==seed)
    assert math.isclose(sg,metric['mean_width_error'],abs_tol=1e-12)
    assert math.isclose(sg,pub['segformer_fallback_mean'],abs_tol=1e-12) and math.isclose(hh,pub['hrnet_mean'],abs_tol=1e-12)
    for tyre,values in pub['per_tyre'].items():
        assert math.isclose(values['hrnet'],statistics.mean(r['error_width_fraction'] for r in hr['records'] if r['tyre']==tyre),abs_tol=1e-12)
        assert math.isclose(values['segformer_with_fallback'],statistics.mean(r['error_width_fraction'] for r in rows if r['tyre']==tyre),abs_tol=1e-12)
    results.append(dict(seed=seed,hrnet=hh,segformer=sg,hrnet_px=statistics.mean(r['error_px'] for r in hr['records']),
        segformer_px=statistics.mean(r['error_px'] for r in rows),coverage=1,per_tyre=pub['per_tyre']))
hrmean=statistics.mean(r['hrnet'] for r in results);sgmean=statistics.mean(r['segformer'] for r in results)
audit=dict(status='verified',revision=REV,training_revision=report['source_revision'],protocol=KEY,
    completed_seeds=3,epochs_per_seed=60,optimizer_step_records=6480,paired_points_recomputed=432,
    test_images=24,test_tyres=2,smoke_parameter_delta=smoke['max_parameter_difference'],results=results,
    hrnet_mean=hrmean,segformer_mean=sgmean,hrnet_relative_error_reduction=1-hrmean/sgmean,
    hrnet_mean_px=statistics.mean(r['hrnet_px'] for r in results),segformer_mean_px=statistics.mean(r['segformer_px'] for r in results),
    downloaded_bytes=total,checkpoint_check='LFS hashes checked; weight tensors not downloaded',full_s9_complete=False)
p.write(OUT/'AUDIT.json',audit);print(json.dumps(audit,indent=2))
