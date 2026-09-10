"""Read-only public HF recovery audit. Prints evidence; never uploads."""
import io
import json
import concurrent.futures
import pandas as pd
import numpy as np
import requests
from PIL import Image
from huggingface_hub import HfApi, hf_hub_url

REPO = 'Shanmuk4622/tyre-wear-study'
api = HfApi()
rev = api.repo_info(REPO, repo_type='dataset').sha
files = set(api.list_repo_files(REPO, repo_type='dataset', revision=rev))
print('REVISION', rev, flush=True)
def raw(p):
    r = requests.get(hf_hub_url(REPO, p, repo_type='dataset', revision=rev), timeout=60)
    r.raise_for_status()
    return r.content
def table(p):
    return pd.read_csv(io.BytesIO(raw(p)))
prefix = 'tables/closure_2026-09-09/'
paths = sorted(p for p in files if p.startswith(prefix))
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    contents = dict(zip(paths, pool.map(raw, paths)))
for p, value in contents.items():
    if p.endswith('.json'):
        print(p, value.decode())
    elif p.endswith('.csv'):
        d = pd.read_csv(io.BytesIO(value))
        print(p, 'ROWS', len(d), 'COLUMNS', list(d))
        if p.endswith(('baseline_summary.csv', 'hypothesis_outcomes.csv')):
            print(d.to_string(index=False))
def local(name):
    return pd.read_csv(io.BytesIO(contents[prefix+name+'.csv']))
b = local('baselines_by_fold')
assert len(b) == 15
assert not b.duplicated(['baseline', 'fold']).any()
pred = local('baseline_predictions')
assert len(pred) == 1672 and not pred.duplicated(['baseline', 'image_id']).any()
from sklearn.metrics import f1_score
for (baseline, fold), group in pred.groupby(['baseline','fold']):
    expected = b.loc[b.baseline.eq(baseline) & b.fold.eq(fold), 'f1_macro'].iloc[0]
    assert np.isclose(f1_score(group['true'], group.pred, average='macro'), expected)
print('BASELINE_PREDICTIONS verified against all 12 CPU fold metrics')
coverage = local('intervention_region_coverage')
print('MASK_POSITIVES', coverage.assign(marking=coverage.marking_pixels.gt(0), damage=coverage.damage_pixels.gt(0)).groupby('fold')[['marking','damage']].sum().to_dict())
panel = local('saliency_panel_manifest')
print('SALIENCY_PANELS', len(panel), 'RUNS', panel.run_id.nunique(), 'ZERO_MAPS', int(panel.zero_map.sum()))
a = local('stage_a_coverage')
assert len(a) == 162 and a.budget_complete.all() and a.artifacts_present.all()
assert a.quarantined.sum() == 9
ids = [f's1-resnet50-randmatched_r1-f{f}-s{s}' for f in range(3) for s in (1,2,3)]
def audit_run(rid):
    status = json.loads(raw(f'runs/{rid}/STATUS.json'))
    hist = table(f'runs/{rid}/metrics/epochs.csv')
    final = table(f'runs/{rid}/metrics/final.csv')
    assert status['status'] == 'completed', (rid, status)
    assert list(hist.epoch) == list(range(1,61)), rid
    assert final.iloc[0].epochs_trained == 60 and final.iloc[0].epochs_planned == 60
    assert np.isclose(hist.iloc[-1].val_f1_macro, final.iloc[0].final_val_f1_macro)
    for ck in ('ckpt_best.pt','ckpt_last.pt'):
        assert f'runs/{rid}/checkpoints/{ck}' in files
    return dict(run_id=rid, f1=float(hist.iloc[-1].val_f1_macro), status=status,
                final=final.iloc[0].to_dict())
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    runs = list(pool.map(audit_run, ids))
for r in runs:
    print('MATCHED_RUN', r['run_id'], r['f1'], 'epochs=60', 'nan_batches=', r['final']['nan_or_inf_batches_total'])
print('MATCHED_FINAL_MEAN', np.mean([r['f1'] for r in runs]))
pretrained = [table(f'runs/a-resnet50-base-f{f}-s{s}/metrics/final.csv').iloc[0] for f in range(3) for s in (1,2,3)]
print('PRETRAINED_FINAL_MEAN', np.mean([r.final_val_f1_macro for r in pretrained]))
print('FOLD1_RANDOM', np.mean([r['f1'] for r in runs if '-f1-' in r['run_id']]))
print('FOLD1_PRETRAINED', np.mean([r.final_val_f1_macro for r in pretrained if r['fold'] == 1]))
figs = sorted(p for p in files if p.startswith('analysis/closure_2026-09-09/fig') and p.endswith('.png'))
assert len(figs) == 10
for p in figs:
    with Image.open(io.BytesIO(raw(p))) as im:
        im.verify()
print('PASS: 15 baseline rows; 162 audited Stage-A rows / 9 quarantined; 9 matched runs x 60 epochs, final metrics and checkpoint paths; 10 decodable figures')
print('OTHER_RUN_STAGES', sorted(set(p.split('/')[1].split('-')[0] for p in files if p.startswith('runs/') and p.endswith('/STATUS.json'))))
