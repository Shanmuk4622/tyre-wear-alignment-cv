"""Final source-label, point-score, CSV-plan and delivery/preservation checks."""
import ast
import base64
import csv
import gzip
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

assert Path(sys.prefix).name == 'cv_conda'
import numpy as np
import nbformat
from publish_recovered_csv import prepare

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'completion_20261028'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    overlay = read(OUT/'hf/OVERLAY.json')
    source = ROOT.parent/'phase2/data/phase2_dataset_v1'
    hashes = dict(line.split('  ', 1)[::-1] for line in (source/'SHA256SUMS.txt').read_text().splitlines())
    for name in ('manifests/images.json', 'geometry/original_human_points.json'):
        assert sha(source/name) == hashes[name]
    originals = {r['image_id']: r for r in read(source/'manifests/images.json')}
    points = {r['image_id']: r for r in read(source/'geometry/original_human_points.json')}
    rows = {r['image_id']: r for r in overlay['rows']}
    for image_id, r in rows.items():
        for key in ('class_index', 'physical_tyre_id', 'domain', 'image_sha256', 'image_path', 'tyre_mask', 'tread_mask', 'ignore_mask'):
            assert originals[image_id][key] == r[key]
        if r['point_kind'] == 'human':
            reference = [p['x']/(r['width']-1) for p in points[image_id]['points']]
            assert np.allclose(reference, r['points'], atol=1e-12)
    checked = 0
    for name in ('validation_predictions.csv', 'test_predictions.csv'):
        with open(OUT/'csv'/name, newline='', encoding='utf-8') as f:
            for rec in csv.DictReader(f):
                r = rows[rec['image_id']]
                assert int(rec['label']) == r['class_index'] and rec['tyre'] == r['physical_tyre_id']
                assert r['role'] == ('validation' if name.startswith('validation') else 'test')
                checked += 1
    grouped = defaultdict(list)
    with open(OUT/'csv/point_predictions.csv', newline='', encoding='utf-8') as f:
        for rec in csv.DictReader(f):
            r = rows[rec['image_id']]
            if rec['target']:
                target = float(rec['target'])
                close = r['points'][int(rec['point_index'])]
                assert abs(target-close) < 1e-12 and r['point_kind'] == 'human'
                error = abs(float(rec['prediction'])-target)
                assert abs(error-float(rec['abs_error_width_fraction'])) < 1e-12
                assert abs(error*(r['width']-1)-float(rec['abs_error_native_pixels'])) < 1e-9
                grouped[(rec['seed'], rec['split'], rec['epoch'], rec['endpoint'], rec['image_id'])].append(error)
    scored = 0
    for name in ('validation_predictions.csv', 'test_predictions.csv'):
        with open(OUT/'csv'/name, newline='', encoding='utf-8') as f:
            for rec in csv.DictReader(f):
                if rec['human_point_mae']:
                    values = grouped[(rec['seed'], rec['split'], rec['epoch'], rec.get('endpoint', ''), rec['image_id'])]
                    assert len(values) == 6 and abs(np.mean(values)-float(rec['human_point_mae'])) < 1e-7
                    scored += 1
    files = prepare()
    assert all(not name.endswith(('.zip', '.npz', '.pt')) for name in files)
    manifest = read(OUT/'CSV_MANIFEST.json')
    assert all(sha(files[name]) == entry['sha256'] for name, entry in manifest['files'].items())
    notebook = ROOT/'notebooks/phase3_NB02_Audit_And_Direct_CSV.ipynb'
    nb = nbformat.read(notebook, as_version=4)
    nbformat.validate(nb)
    payload_files = 0
    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        tree = ast.parse(cell.source)
        for call in ast.walk(tree):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == 'b64decode' and call.args and isinstance(call.args[0], ast.Constant):
                payload = json.loads(gzip.decompress(base64.b64decode(call.args[0].value)))
                for name, value in payload.items():
                    assert base64.b64decode(value) == (ROOT/'analysis'/name).read_bytes()
                    payload_files += 1
    assert payload_files == 3
    publication_path = OUT/'nb02_publication/VERIFICATION.json'
    publication = read(publication_path) if publication_path.exists() else None
    if publication:
        assert publication['status'] == 'passed' and publication['live_publication_verified']
        assert sha(notebook) == publication['notebook_sha256']
        assert sha(OUT/'nb02_publication/executed.ipynb') == publication['notebook_sha256']
        assert publication['published_files'] == 25 and publication['csv_table_count'] == 18
        assert publication['zip_files_uploaded'] == 0 and publication['all_remote_hashes_verified']
        for record in publication['files']:
            assert sha(OUT/'nb02_publication/remote'/record['path']) == record['sha256']
    protected = read(ROOT/'audit/markdown-baseline.json')
    assert all(sha(ROOT.parent/r['path']) == r['sha256'] for r in protected)
    for archived in (OUT/'executed_notebooks').glob('*.ipynb'):
        assert sha(archived) == sha(ROOT/'notebooks'/archived.name)
    original_delivery = read(ROOT/'releases/DELIVERY.json')
    for name, expected in original_delivery['files'].items():
        if not name.endswith('.ipynb'):
            assert sha(ROOT/name) == expected
    links = 0
    broken = []
    for doc in ROOT.glob('*.md'):
        for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)', doc.read_text(encoding='utf-8')):
            if re.match(r'^(https?://|mailto:|#)', target):
                continue
            links += 1
            if not (doc.parent/target.split('#')[0]).exists():
                broken.append([doc.name, target])
    assert not broken, broken
    result = dict(status='passed', prediction_labels_and_cohorts_checked=checked, human_point_image_scores_recalculated=scored,
        local_prepared_files=len(files), zip_uploads_planned=0, local_csv_files=len(list((OUT/'csv').glob('*.csv'))),
        cpu_notebook_schema_and_payloads_valid=True, protected_original_markdown=len(protected),
        executed_notebook_archives_unchanged=True, frozen_r1_non_notebook_delivery_unchanged=True,
        links_checked=links, broken_links=broken, remote_writes_performed=False,
        publication_status='Kaggle publication complete; immutable remote files independently verified.' if publication else 'Local direct-CSV package ready; Kaggle publication pending.',
        cpu_notebook_kaggle_execution_verified=bool(publication),
        verified_hf_publication_revision=publication['revision'] if publication else None,
        published_files=publication['published_files'] if publication else 0,
        published_csv_files=publication['csv_table_count'] if publication else 0,
        verified_published_scientific_rows=publication['verified_scientific_rows'] if publication else 0)
    (OUT/'REVIEW_VALIDATION.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
