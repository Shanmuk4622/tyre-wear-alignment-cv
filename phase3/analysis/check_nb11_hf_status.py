"""Prioritize completion/selection/results before the exhaustive publication audit."""
import hashlib,json,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from filelock import FileLock
import nb11_public as pub
assert Path(sys.prefix).name=='cv_conda'
with FileLock(str(pub.OUT/'AUDIT.lock'),timeout=1):
    m=pub.manifest();names=['CONTRACT.json','CURSOR.json','HISTORY.json','LIVE_GATE_RECEIPT.json','report/REPORT.json']
    items=[(n,x) for n,x in m['files'].items() if n in names or n.endswith('/selection.csv') or n.endswith('/metrics.csv') or n.endswith('/confusion.csv') or n.endswith('/predictions.csv')]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i,_ in enumerate(pool.map(pub.fetch,items),1):
            if i%100==0:print('Priority HF files verified:',i,'/',len(items),flush=True)
    remote=pub.OUT/'remote';contract=json.loads((remote/'CONTRACT.json').read_text())
    assert hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==pub.PROTOCOL
    cursor=json.loads((remote/'CURSOR.json').read_text());assert cursor['phase']=='completed' and cursor['updates']==2940
    def table(f):return pd.concat([pd.read_csv(p) for p in sorted((remote/'csv').glob(f'epoch_*/{f}.csv'))],ignore_index=True)
    selection=table('selection');pred=table('predictions');metric=table('metrics')
    assert len(selection)==60 and len(pred)==53342
    selected=selection[selection.selected==1].iloc[-1]
    result=dict(status='completed_priority_checks_passed',revision=pub.REVISION,protocol=pub.PROTOCOL,files=len(m['files'])+1,
      objects=len(m['files']),bytes=sum(v['bytes'] for v in m['files'].values()),owner_released=True,updates=cursor['updates'],phase=cursor['phase'],
      source_hashes_match=all(pub.sha(pub.ROOT/'tread_two/lc_pilot'/n)==v for n,v in contract['sources'].items()),environment=contract['environment'],
      selected_epoch=int(selected.selected_epoch),validation_ba_max=float(selection.validation_ba_observed.max()),selected_validation_nll=float(selected.validation_nll),
      endpoints=metric[(metric.stage.isin(['selected_eval','final_eval','diagnostics']))&(metric.level=='cohort')&(metric.metric.isin(['accuracy','balanced_accuracy_observed','nll','mean_two_region_dice','human_point_mae_width_fraction']))].fillna('').to_dict('records'),
      predictions_by_stage=pred.groupby('stage').size().to_dict(),endpoint_confusion={},priority_objects_verified=len(items),public_read_requests_this_invocation=pub.CALLS,full_object_audit_pending=True)
    for endpoint in ('selected_eval','final_eval'):
        sub=pred[pred.stage==endpoint]
        result['endpoint_confusion'][endpoint]=pd.crosstab(sub.label,sub.prediction).reindex(index=range(3),columns=range(3),fill_value=0).values.tolist()
    (pub.OUT/'STATUS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
