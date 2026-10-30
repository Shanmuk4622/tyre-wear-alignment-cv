"""Verify the untrained concept's direct artifacts and prior-work isolation."""
import ast
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

import numpy as np
from PIL import Image

assert Path(sys.prefix).name == 'cv_conda'
ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'tread_two'
REPORT = HERE / 'prototype_report'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def checked_hashes(base, hashes):
    assert all(sha(base / name) == digest for name, digest in hashes.items())
    return len(hashes)


def main():
    proof = read(REPORT / 'PROOF.json')
    assert proof['check_count'] == len(proof['checks']) == 15
    assert proof['model_status'] == 'UNTRAINED_ARCHITECTURE_PROTOTYPE'
    assert proof['optimizer_updates'] == 0 and not proof['measured_training']
    assert proof['trained_classifier_accuracy'] is None
    assert not proof['Kaggle_training_notebook_delivered'] and not proof['full_training_recipe_frozen']
    assert proof['masks_max_difference'] == proof['points_max_difference'] == proof['logit_accounting_residual'] == 0
    assert proof['total_parameters'] == proof['frozen_parameters'] + proof['trainable_parameters'] == 1453433
    assert proof['trainable_parameters'] == 81274
    checked_hashes(HERE, proof['source_hashes'])
    for path in list(HERE.glob('*.py')) + [Path(__file__), ROOT / 'analysis/build_tread_two_explorer.py']:
        ast.parse(path.read_text(encoding='utf-8'))

    manifest = read(REPORT / 'MANIFEST.json')['files']
    assert set(manifest) == {p.name for p in REPORT.iterdir() if p.is_file() and p.name != 'MANIFEST.json'}
    for name, item in manifest.items():
        path = REPORT / name
        assert sha(path) == item['sha256'] and path.stat().st_size == item['bytes']
    schema = read(REPORT / 'CSV_SCHEMA.json')
    counts = {}
    for name, item in schema.items():
        path = REPORT / (name + '.csv')
        with path.open(encoding='utf-8', newline='') as stream:
            reader = csv.DictReader(stream)
            assert reader.fieldnames == item['columns']
            count = sum(1 for _ in reader)
        assert count == item['rows'] and sha(path) == item['sha256']
        counts[name] = count
    assert len(counts) == 8 and sum(counts.values()) == 224787

    bank = np.load(REPORT / 'bank_rgb.npy', allow_pickle=False)
    labels = np.load(REPORT / 'bank_labels.npy', allow_pickle=False)
    assert bank.shape == (18, 3, 64, 64) and bank.dtype == np.float32
    assert labels.shape == (18,) and labels.dtype == np.int64
    assert [int((labels == i).sum()) for i in range(3)] == [6, 6, 6]
    assert sha(REPORT / 'bank_rgb.npy') == proof['bank_RGB_SHA256']
    reconstructed = np.empty_like(bank)
    seen = np.zeros(bank.shape, dtype=bool)
    with (REPORT / 'bank_pixels.csv').open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            index = tuple(int(row[key]) for key in ('prototype_index', 'channel', 'row', 'column'))
            assert all(0 <= value < limit for value, limit in zip(index, bank.shape)) and not seen[index]
            seen[index] = True
            reconstructed[index] = float(row['value'])
    assert seen.all() and np.array_equal(bank.view(np.uint32), reconstructed.view(np.uint32))

    source = {r['image_id']: r for r in records(ROOT / 'data_gate_outputs/csv/source_images.csv')}
    cases = records(REPORT / 'prototype_bank.csv')
    bank_images = records(REPORT / 'bank_images.csv')
    queries = records(REPORT / 'query_manifest.csv')
    ids = {row['source_image_id'] for row in cases}
    assert len(ids) == len(bank_images) == 8
    assert ids == {r['image_id'] for r in bank_images}
    assert not ids.intersection(row['image_id'] for row in queries)
    data = ROOT.parent / 'phase2/data/phase2_dataset_v1'
    for row in cases:
        original = source[row['source_image_id']]
        assert row['source_split'] == original['historical_study_split'] == 'train'
        assert original['domain'] == 'original_photo'
        assert row['source_tyre_id'] == original['confirmed_original_tyre_id']
        assert row['source_image_sha256'] == original['image_sha256'] == sha(data / original['image_path'])
        assert int(row['class_index']) == int(original['class_index']) == labels[int(row['prototype_index'])]
        assert int(row['atlas_patch_index']) == int(row['atlas_row']) * 4 + int(row['atlas_column'])
    assert [len({r['source_tyre_id'] for r in cases if int(r['class_index']) == i}) for i in range(3)] == [1, 3, 4]
    assert len({r['source_tyre_id'] for r in cases}) == 8
    query_pool = [r for r in source.values() if r['historical_study_split'] == 'train' and r['image_id'] not in ids]
    assert len(query_pool) == 378
    assert [sum(int(r['class_index']) == i for r in query_pool) for i in range(3)] == [66, 246, 66]
    for query in queries:
        original = source[query['image_id']]
        assert query['source_split'] == original['historical_study_split'] == 'train'
        assert query['source_image_sha256'] == original['image_sha256'] == sha(data / original['image_path'])

    original_count = checked_hashes(ROOT.parent, read(ROOT / 'workstation_validation/ORIGINAL_BASELINE.json'))
    app_count = checked_hashes(ROOT, read(ROOT / 'workstation_validation/DELIVERY.json')['source_hashes'])
    checked_hashes(ROOT, read(REPORT / 'WORKSTATION_SNAPSHOT.json'))
    md_baseline = read(ROOT / 'audit/markdown-baseline.json')
    assert all(sha(ROOT.parent / r['path']) == r['sha256'] for r in md_baseline)
    experiment_count = checked_hashes(ROOT, read(ROOT / 'revision2_validation/DELIVERY.json')['sources'])
    contract = read(ROOT / 'completion_20261028/hf/CONTRACT.json')
    r1_count = checked_hashes(ROOT / 'runtime', contract['sources'])
    weights_sha = 'a5b872ed2e4cbfb3b9782b0df563e2606631744cd0274fee2da42a832f985f65'
    assert sha(ROOT / 'completion_20261028/hf/report/weights.pt') == sha(ROOT / 'workstation/models/weights.pt') == weights_sha
    for name, archive in [
        ('phase3_NB02_Audit_And_Direct_CSV.ipynb', 'completion_20261028/nb02_publication/executed.ipynb'),
        ('phase3_NB03_TREAD_ONE_Classifier_Diagnostic.ipynb', 'diagnostics/nb03_publication/executed.ipynb'),
        ('phase3_NB04_TREAD_ONE_Evidence_And_Recovery_Gate.ipynb', 'training_gates/nb04_publication/executed.ipynb'),
        ('phase3_NB05_TREAD_ONE_Shared_Gradient_Paired_Pilot.ipynb', 'revision2_validation/nb05_publication/executed.ipynb'),
        ('phase3_NB06_TREAD_ONE_Paired_Review_And_Failure_Atlas.ipynb', 'revision2_validation/nb06_publication/executed.ipynb'),
    ]:
        assert sha(ROOT / 'notebooks' / name) == sha(ROOT / archive)
    latest = Path(read(ROOT / 'data_gate_validation/nb07_reruns/LATEST.json')['path'])
    assert sha(ROOT / 'notebooks/phase3_NB07_TREAD_ONE_Data_And_Protocol_Gate.ipynb') == read(latest / 'VERIFICATION.json')['executed_notebook_sha256']

    docs = list(ROOT.glob('*.md')) + list(HERE.glob('*.md')) + list((ROOT / 'workstation').glob('*.md'))
    docs += list((ROOT / 'data_intake').rglob('*.md')) + [ROOT / 'data_gate_outputs/CSV_INDEX.md']
    links = 0
    for path in docs:
        for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            target = target.strip('<>')
            if re.match(r'^(https?://|mailto:|#)', target):
                continue
            target = unquote(target.split('#')[0])
            assert (path.parent / target).exists(), (str(path), target)
            links += 1
    for path in REPORT.glob('*.png'):
        with Image.open(path) as image:
            image.verify()

    html = (HERE / 'ATLAS_EXPLORER.html').read_text(encoding='utf-8')
    payload = json.loads(re.search(r'<script id="payload" type="application/json">(.*?)</script>', html, re.S)[1])
    assert len(payload['queries']) == 3 and len(payload['bank']) == 18 and len(payload['rows']) == 1536
    assert all(p['source_split'] == 'train' for p in payload['queries'])
    behavior = re.findall(r'<script>(.*?)</script>', html, re.S)
    node = shutil.which('node')
    assert node and len(behavior) == 1
    check = subprocess.run([node, '--check'], input=behavior[0], text=True, capture_output=True, check=False)
    assert check.returncode == 0, check.stderr
    assert 'fetch(' not in behavior[0] and 'https://' not in behavior[0]
    status = read(REPORT / 'PROTOTYPE_STATUS.json')
    assert status['HF_uploads'] == 0 and not status['classifier_weights_published']
    result = dict(
        status='passed', scope='Untrained TREAD-TWO architecture/design delivery; not a measured model release',
        architecture_check_count=15, optimizer_updates=0, trained_accuracy=None,
        direct_CSV_tables=len(counts), direct_CSV_rows=sum(counts.values()), CSV_rows_by_table=counts,
        bank_pixels_bit_exact=True, bank_cases=18, bank_source_photos=8, bank_independent_tyres_by_class=[1, 3, 4],
        classifier_query_images=378, engineering_query_images=3,
        artifact_manifest_files=len(manifest), artifact_manifest_bytes=sum(p['bytes'] for p in manifest.values()),
        original_workstation_files_preserved=original_count, TREAD_ONE_Studio_source_files_preserved=app_count,
        original_Markdown_preserved=len(md_baseline), frozen_experiment_sources_preserved=experiment_count,
        frozen_r1_runtime_sources_preserved=r1_count, earlier_executed_notebooks_preserved=True,
        selected_anchor_weights_sha256=weights_sha, workstation_changes=False, new_independent_tyres=0,
        HF_uploads=0, Kaggle_training_notebook_delivered=False,
        local_Markdown_documents=len(docs), local_links_checked=links, broken_links=[],
        interactive_illustration=dict(payload_and_JS_syntax_checked=True, browser_render_tested=False,
                                      bytes=(HERE / 'ATLAS_EXPLORER.html').stat().st_size),
        code_hashes={p.relative_to(ROOT).as_posix(): sha(p) for p in list(HERE.glob('*.py')) + [Path(__file__), ROOT / 'analysis/build_tread_two_explorer.py']},
        documentation_hashes={p.relative_to(ROOT).as_posix(): sha(p) for p in docs},
        generated_explorer_sha256=sha(HERE / 'ATLAS_EXPLORER.html'),
    )
    (HERE / 'DELIVERY.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if not k.endswith('hashes')}, indent=2))


if __name__ == '__main__':
    main()
