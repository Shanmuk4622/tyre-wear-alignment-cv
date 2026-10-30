"""Verify final public ATLAS publication and saved report, without model execution."""
import csv
import hashlib
import json
import math
import re
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock, local

import pandas as pd
import requests
from PIL import Image

assert Path(sys.prefix).name == 'cv_conda'
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'tread_two/pilot_validation/nb08_followup/82e7e1c5774f'
audit = json.loads((OUT / 'PUBLIC_AUDIT.json').read_text(encoding='utf-8'))
revision = audit['revision']
fit = audit['destinations']['fit']
prefix = fit['prefix']
manifest = json.loads((OUT / 'fit_MANIFEST.json').read_text(encoding='utf-8'))
sessions = local()
count_lock = Lock()
calls = 0
downloaded = []


def get(url):
    global calls
    with count_lock:
        calls += 1
    if not hasattr(sessions, 'session'):
        sessions.session = requests.Session()
        sessions.session.trust_env = False
    response = sessions.session.get(url, timeout=(20, 120))
    response.raise_for_status()
    return response


# Verify every manifest object using HF Git/LFS metadata at the pinned commit.
url = f'https://huggingface.co/api/datasets/Shanmuk4622/tyre-wear-study/tree/{revision}/{prefix}?recursive=true&expand=false&limit=1000'
entries = []
while url:
    response = get(url)
    entries.extend(response.json())
    url = response.links.get('next', {}).get('url')
lookup = {entry['path'][len(prefix)+1:]:entry for entry in entries if entry['type'] == 'file'}
(OUT / 'PINNED_TREE.json').write_text(json.dumps(entries, indent=2)+'\n', encoding='utf-8')
verified_lfs = verified_git = 0
for name, meta in manifest['files'].items():
    obj = lookup[meta['object']]
    assert obj['size'] == meta['bytes'], name
    if obj.get('lfs'):
        assert obj['lfs']['oid'] == meta['sha256'], name
        verified_lfs += 1


def fetch(name):
    global verified_git
    meta = manifest['files'][name]
    target = OUT / 'fit' / name
    if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != meta['sha256']:
        response = get(f'https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/resolve/{revision}/{prefix}/{meta["object"]}')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(response.content)
    body = target.read_bytes()
    assert len(body) == meta['bytes'] and hashlib.sha256(body).hexdigest() == meta['sha256'], name
    obj = lookup[meta['object']]
    if not obj.get('lfs'):
        assert obj['oid'] == hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest(), name
        verified_git += 1
    downloaded.append(name)
    return target


# Fetch all Git objects for independent SHA-256 + Git blob confirmation. Large
# LFS tensors/tables stay remote, except the final state hash checked below.
git_names = [name for name, meta in manifest['files'].items() if not lookup[meta['object']].get('lfs')]
with ThreadPoolExecutor(max_workers=4) as pool:
    # Independent immutable reads; at most four requests in flight. No HF API
    # client/token is initialized, and cached byte-verified objects are reused.
    for index, _ in enumerate(pool.map(fetch, git_names), 1):
        if index % 200 == 0:
            print('Verified direct Git objects:', index, '/', len(git_names), flush=True)
print('All manifest objects found; checking final cursor and reports.', flush=True)
cursor = json.loads(fetch('CURSOR.json').read_text())
assert cursor['phase'] == cursor['status'] == 'completed' and cursor['updates'] == 2940
assert fit['owner'] == dict(owner=None, expires=0)
state = fetch('state.pt')
assert hashlib.sha256(state.read_bytes()).hexdigest() == cursor['state_sha256']
assert not any(name.endswith(('.zip', '.npz', '.sqlite')) for name in manifest['files'])
families = Counter(Path(name).stem for name in manifest['files'] if name.startswith('csv/') and name.endswith('.csv'))
assert len(families) == len(cursor['row_cursors']) == 21

# Reproduce NB08's last display cell against byte-verified retained metrics.
metric_paths = sorted(name for name in manifest['files'] if re.fullmatch(r'csv/epoch_\d+/metrics.csv',name))
metrics = pd.concat([pd.read_csv(fetch(name)) for name in metric_paths], ignore_index=True)
endpoint_metrics = metrics[(metrics['level']=='cohort') & metrics['stage'].isin(['selected_eval','final_eval'])]
assert len(metric_paths) == 60 and len(metrics) == cursor['row_cursors']['metrics']
assert len(endpoint_metrics) > 0
selections = pd.concat([pd.read_csv(fetch(name)) for name in sorted(manifest['files']) if re.fullmatch(r'csv/epoch_\d+/selection.csv',name)],ignore_index=True)
assert len(selections) == 60 and list(selections['epoch']) == list(range(1,61))
best = selections.sort_values(['validation_ba_observed','validation_nll','epoch'], ascending=[False,True,True]).iloc[0]
assert int(best['epoch']) == int(selections.iloc[-1]['selected_epoch']) == 18
endpoints = {}
for epoch in [18,60]:
    frame = pd.read_csv(fetch(f'csv/epoch_{epoch:04d}/predictions.csv'))
    for stage in ['selected_eval','final_eval']:
        rows = frame[frame['stage'] == stage]
        if rows.empty:
            continue
        assert len(rows) == rows['image_id'].nunique() == 103
        counts = rows['label'].value_counts().to_dict()
        assert counts == {0:62,2:41}
        assert ((rows[['logit_0','logit_1','logit_2']].values.argmax(1)) == rows['prediction']).all()
        assert (rows['prediction'].eq(rows['label']).astype(int) == rows['correct']).all()
        recalls = rows.groupby('label')['correct'].mean()
        ba = float(recalls.mean())
        baseline = float(rows.assign(anchor_correct=rows['anchor_prediction'].eq(rows['label'])).groupby('label')['anchor_correct'].mean().mean())
        summary = endpoint_metrics[endpoint_metrics['stage']==stage].set_index('metric')['value']
        assert math.isclose(ba, summary['balanced_accuracy_observed'],abs_tol=1e-12)
        assert math.isclose(baseline, summary['frozen_anchor_on_original_balanced_accuracy_observed'],abs_tol=1e-12)
        checks = pd.read_csv(fetch(f'csv/epoch_{epoch:04d}/explanation_checks.csv'))
        assert len(checks) == 103*3
        endpoints[stage] = dict(epoch=epoch, images=103, balanced_accuracy_observed=ba,
                                frozen_anchor_balanced_accuracy_observed=baseline,
                                recalls={str(key):float(value) for key,value in recalls.items()},
                                explanation_rows=len(checks), maximum_projection_residual=float(checks['residual'].abs().max()))

png = fetch('report/training_curves.png')
with Image.open(png) as picture:
    picture.verify()
html = fetch('report/ATLAS_Evidence_Explorer.html').read_text(encoding='utf-8')
match = re.search(r'<script type="application/json" id="data">(.*?)</script>',html,re.S)
data = json.loads(match[1])
assert len(data['cases']) == 206 and len(data['bank']) == 18
assert Counter(row['endpoint'] for row in data['cases']) == {'selected':103,'final':103}
assert len({(row['endpoint'], row['image_id']) for row in data['cases']}) == 206
report = json.loads(fetch('report/REPORT.json').read_text())
assert report['endpoint_images'] == 206 and report['model_forward_calls'] == 0
delivery = ROOT / 'tread_two/completion_report'
delivery.mkdir(exist_ok=True)
for name in ('training_curves.png','ATLAS_Evidence_Explorer.html','REPORT.json'):
    origin = fetch('report/' + name)
    shutil.copy2(origin, delivery / name)
    assert hashlib.sha256((delivery/name).read_bytes()).hexdigest() == manifest['files']['report/'+name]['sha256']
checkpoints = [name for name in manifest['files'] if re.search(r'/epoch_\d+\.pt$',name)]
assert len(checkpoints) == 60, checkpoints
record = dict(status='passed', revision=revision, fit_protocol=manifest['contract'],
              training_completed=True, epochs=60, optimizer_updates=2940, owner_released=True,
              manifest_objects_verified=len(manifest['files']), lfs_objects_sha256_verified=verified_lfs,
              git_objects_byte_and_sha256_verified=len(set(downloaded))-sum(bool(lookup[manifest['files'][name]['object']].get('lfs')) for name in set(downloaded)),
              manifest_total_bytes=sum(meta['bytes'] for meta in manifest['files'].values()),
              state_bytes_hash_verified=True, state_deserialized=False, epoch_checkpoints=60,
              csv_families=21, declared_scientific_rows=sum(cursor['row_cursors'].values()),
              all_large_CSV_rows_independently_recounted=False, metrics_rows_independently_counted=len(metrics),
              selection_rows_checked=60, selected_epoch=18, endpoints=endpoints,
              report_display_dataframe_passed=True, explorer_endpoint_images=206, explorer_reference_cases=18,
              interactive_browser_render_tested=False, latest_error_output_available_locally=False,
              user_clarification='No error was displayed; notebook stopped running after some time.',
              new_error_reproduced=False, notebook_source_changed=False, retraining_needed=False,
              public_read_requests=calls, prior_public_audit_reads=audit['public_read_requests'],
              local_credentials_read=False,remote_writes=0,model_forward_or_training_calls=0,
              downloaded_files=sorted(set(downloaded)))
(OUT/'VERIFICATION.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps({key:value for key,value in record.items() if key!='downloaded_files'},indent=2))
