"""Additive direct-CSV publication; never uploads a ZIP or alters training artifacts."""
import csv
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

assert Path(sys.prefix).name == 'cv_conda' or Path('/kaggle/input').exists()
from huggingface_hub import HfApi, CommitOperationAdd, get_token

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'completion_20261028'
CSV = OUT/'csv'
REPO = 'Shanmuk4622/tyre-wear-study'
SOURCE_REVISION = 'a0296f18e02b12d28f7e51377783d68538032209'
PROTOCOL = '371c40c0c6aaedaf5c5970f7721a284b042494db7bb330065f6cbde8fc3a0b8c'
DEST = 'phase3/tread-one-r1/'+PROTOCOL+'/posthoc-csv/2026-10-28'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    audit = json.loads((OUT/'AUDIT.json').read_text())
    assert audit['status'] == 'passed' and audit['source_revision'] == SOURCE_REVISION
    manifest_rows = [dict(path='csv/'+p.name, bytes=p.stat().st_size, sha256=sha(p), source_revision=SOURCE_REVISION)
                     for p in sorted(CSV.glob('*.csv')) if p.name != 'artifact_manifest.csv']
    with open(CSV/'artifact_manifest.csv', 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['path', 'bytes', 'sha256', 'source_revision'])
        writer.writeheader(); writer.writerows(manifest_rows)
    counts = {}
    for path in sorted(CSV.glob('*.csv')):
        with open(path, newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            counts[path.name] = dict(rows=sum(1 for _ in reader), columns=fields,
                                     sha256=sha(path), bytes=path.stat().st_size)
    (OUT/'CSV_SCHEMA.json').write_text(json.dumps(dict(version='r1-posthoc-csv-1', files=counts,
        missing='Blank CSV value means unavailable or not applicable; zero is an observed numeric zero.',
        units='Probabilities/accuracy/Dice/IoU are fractions; points are normalized width fractions; native-pixel errors are explicitly named; durations are seconds; memory is bytes.',
        geometry='Targets missing where no original human point label exists. Predictions can still be present.',
        metrics='ECE uses ten equal confidence bins; Brier sums three class errors; missing classes contribute zero to the explicitly named three-class macro-F1.',
        selection='Only validation scores selected checkpoints. An inherited joint_validation_utility column on test/group rows is a diagnostic formula value, never an additional selection rule.',
        scope='Low/high-only two-tyre holdout. Images, points, seeds and newly derived rows are not independent subjects.'), indent=2)+'\n')
    (OUT/'CSV_README.md').write_text('''# TREAD-ONE completed run — direct CSV recovery

These directly readable CSV files recover the original r1 training evidence and
add explicitly post-hoc metrics without retraining, changing weights, selecting
on test scores or overwriting the frozen experiment.

Source: `Shanmuk4622/tyre-wear-study`, immutable revision
`a0296f18e02b12d28f7e51377783d68538032209`.

Start with `csv/run_summary.csv`, `csv/validation_epochs.csv` and
`csv/completion_checks.csv`. `CSV_SCHEMA.json` defines columns/units/missing values.
`AUDIT.json` records cryptographic and independent recalculation checks.

Recovered: 8,820 training updates, 70,560 sampled-image exposures, 180 validation
epochs, 14,580 validation-image predictions and 618 test-image endpoint records.
Point predictions, class/calibration/confusion tables, raw-test segmentation
counts, numerical events, GPU recovery checks and provenance are separate CSVs.

Additional NLL/Brier/entropy/ECE/QWK/ordinal/class metrics and segmentation
confusion counts were derived from already saved predictions. They are not
measurements newly collected during training. Missing per-image training
predictions, continuous utilization/energy and complete validation mask maps
cannot be invented retrospectively.

All three runs completed 60 epochs and 2,940 updates. The declared export remains
seed 3 / validation-selected epoch 20. Mean selected test balanced accuracy is
34.1463%, mean Dice is 97.9777% and human boundary error is 1.18937% image width.
All selected checkpoints miss the held-out low tyre. These are two-tyre internal
results, not a deployment, measured-wear or unseen-mid/video claim.

Future HF progress/results must use directly browsable CSV plus native tensor
state/JSON manifests, never external ZIP bundles. Original historical evidence
is retained; this additive export does not rewrite it.
''', encoding='utf-8')
    files = {'csv/'+p.name: p for p in sorted(CSV.glob('*.csv'))}
    files.update({name: OUT/name for name in ('AUDIT.json', 'CSV_SCHEMA.json')})
    files['README.md'] = OUT/'CSV_README.md'
    if (OUT/'PHASE2_INSTALLED_REGISTRY.json').exists():
        files['PHASE2_INSTALLED_REGISTRY.json'] = OUT/'PHASE2_INSTALLED_REGISTRY.json'
    if (OUT/'figures').exists():
        files.update({'figures/'+p.name: p for p in (OUT/'figures').glob('*.png')})
    for path in (ROOT/'analysis').glob('*.py'):
        if path.name in ('fetch_completed_results.py', 'audit_and_export_csv.py', 'publish_recovered_csv.py'):
            files['analysis/'+path.name] = path
    manifest = dict(source_revision=SOURCE_REVISION, source_protocol=PROTOCOL, derived_metrics=True,
                    files={k: dict(sha256=sha(v), bytes=v.stat().st_size) for k, v in files.items()})
    (OUT/'CSV_MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
    files['MANIFEST.json'] = OUT/'CSV_MANIFEST.json'
    assert not any(p.suffix.lower() in ('.zip', '.npz', '.pt') for p in files.values())
    return files


def request(fn):
    for attempt in range(6):
        try:
            return fn()
        except Exception as exc:
            response = getattr(exc, 'response', None)
            code = getattr(response, 'status_code', None)
            if code not in (429, 500, 502, 503, 504) or attempt == 5:
                raise
            headers = getattr(response, 'headers', {}) or {}
            try:
                delay = float(headers.get('Retry-After', '0'))
            except ValueError:
                delay = 0
            match = re.search(r'\bt=(\d+)', headers.get('RateLimit', ''))
            delay = max(delay, float(match.group(1)) if match else 0, min(120, 5*2**attempt))
            print('Publication paused for server delay', delay, flush=True)
            time.sleep(delay)


def main():
    files = prepare()
    if '--prepare-only' in sys.argv:
        print('Prepared', len(files), 'direct files; no remote write. Destination:', DEST)
        return
    token = get_token()
    if not token:
        raise RuntimeError('No HF token available. Use the CPU CSV recovery notebook with Kaggle HF_TOKEN; local CSVs are ready.')
    api = HfApi(token=token)
    who = request(lambda: api.whoami())
    assert who['name'].lower() == 'shanmuk4622', 'Use the intended HF account'
    for attempt in range(6):
        parent = request(lambda: api.repo_info(REPO, repo_type='dataset').sha)
        operations = [CommitOperationAdd(path_in_repo=DEST+'/'+name, path_or_fileobj=str(path)) for name, path in files.items()]
        try:
            commit = request(lambda: api.create_commit(REPO, repo_type='dataset', parent_commit=parent,
                              operations=operations, commit_message='TREAD-ONE verified direct CSV recovery; no training or weight changes'))
            infos = request(lambda: api.get_paths_info(REPO, [DEST+'/'+n for n in files], revision=commit.oid, repo_type='dataset'))
            assert len(infos) == len(files)
            for info in infos:
                relative = info.path[len(DEST)+1:]
                path = files[relative]
                if info.lfs:
                    assert info.lfs.sha256 == sha(path), relative
                else:
                    body = path.read_bytes()
                    assert info.blob_id == hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest(), relative
            result = dict(status='published_and_verified', revision=commit.oid, destination=DEST,
                          source_revision=SOURCE_REVISION, files=len(files), zip_files_uploaded=0)
            (OUT/'CSV_PUBLICATION.json').write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps(result), flush=True)
            return
        except Exception as exc:
            if getattr(getattr(exc, 'response', None), 'status_code', None) not in (409, 412) or attempt == 5:
                raise


if __name__ == '__main__':
    main()
