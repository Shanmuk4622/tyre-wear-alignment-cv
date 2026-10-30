"""Final NB08 source, retained-evidence, packaged-worker and isolation checks."""
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

import nbformat
from PIL import Image
assert Path(sys.prefix).name == 'cv_conda'
ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'tread_two'
OUT = HERE / 'pilot_validation'
sys.path.insert(0, str(HERE / 'pilot'))
from common import atomic, file_meta, read, sha
from data_atlas import dataset, locate
from schemas import SCHEMAS
from spec import sources


def check_hashes(base, mapping):
    assert all(sha(base / name) == expected for name, expected in mapping.items())
    return len(mapping)


def main():
    local = read(OUT / 'LOCAL_VALIDATION.json')
    stop = read(OUT / 'STOP_VALIDATION.json')
    notebook = read(OUT / 'NOTEBOOK_DELIVERY.json')
    extracted = read(OUT / 'NOTEBOOK_VALIDATION.json')
    assert local['status'] == stop['status'] == extracted['status'] == 'passed'
    source = sources()
    assert local['source_hashes'] == source and len(source) == 24
    assert local['actual_model_proof']['scientific_CSV_families_equal'] == len(SCHEMAS) == 21
    assert sum(local['actual_model_proof']['scientific_rows_by_family'].values()) == 900380
    assert local['actual_model_proof']['native_model_optimizer_scheduler_scaler_RNG_equal']
    assert len(local['offline_checks']) == 17 and not local['full_measured_training_started']
    assert stop['paused_at_completed_update'] == 1 and stop['stop_marker_path_tested']
    marker = read(Path(local['fixture_directory']) / 'marker_pause/CONFIG.json')
    assert marker['protocol'] == local['contract'] and marker['contract']['sources'] == source
    assert extracted['source_files'] == 24 and extracted['embedded_files'] == 25 and extracted['actual_batch'] == 8
    assert extracted['extracted_worker_completed_updates'] == 1 and extracted['extracted_model_optimizer_RNG_and_scientific_CSV_match']
    assert extracted['notebook_sha256'] == notebook['notebook_sha256'] == sha(Path(notebook['notebook']))
    assert extracted['network_requests'] == local['network_requests'] == 0
    assert not local['actual_live_HF_tested'] and not local['actual_dual_T4_tested']
    nb = nbformat.read(notebook['notebook'], as_version=4)
    nbformat.validate(nb)
    code = [cell for cell in nb.cells if cell.cell_type == 'code']
    assert len(code) == 6 and all(not cell.outputs and cell.execution_count is None for cell in code)
    for cell in code:
        ast.parse(cell.source)
    for name in source:
        if name.endswith('.py'):
            ast.parse((HERE / 'pilot' / name).read_text(encoding='utf-8'))
    with (HERE / 'pilot/assets/COLUMN_DICTIONARY.csv').open(encoding='utf-8', newline='') as stream:
        dictionary = list(csv.DictReader(stream))
    assert len(dictionary) == sum(map(len, SCHEMAS.values())) == 247
    assert all(row['unit'] and row['dtype'] and row['definition'] and row['missing'] for row in dictionary)

    snapshot = Path(local['final_snapshot'])
    manifest = read(snapshot / 'MANIFEST.json')
    assert all(file_meta(snapshot / name) == meta for name,meta in manifest['files'].items())
    schema = read(snapshot / 'SCHEMA.json')
    assert schema['families'] == SCHEMAS
    for family, fields in SCHEMAS.items():
        count = 0
        for path in (snapshot / 'csv').glob(f'epoch_*/{family}.csv'):
            with path.open(encoding='utf-8', newline='') as stream:
                reader = csv.DictReader(stream)
                assert reader.fieldnames == fields
                count += sum(1 for _ in reader)
        assert count == schema['row_cursors'][family] == local['actual_model_proof']['scientific_rows_by_family'][family]
    with (snapshot / 'report/REFERENCE_CASES.csv').open(encoding='utf-8', newline='') as stream:
        cases = list(csv.DictReader(stream))
    assert len(cases) == 4 * 3 * 18
    checks = []
    for path in (snapshot / 'csv').glob('epoch_*/explanation_checks.csv'):
        with path.open(encoding='utf-8', newline='') as stream:
            checks.extend(csv.DictReader(stream))
    assert len(checks) == 12
    max_projection_residual = max(float(row['residual']) for row in checks)
    assert max_projection_residual < 2e-6
    with (snapshot / 'COLUMN_DICTIONARY.csv').open(encoding='utf-8', newline='') as stream:
        assert len(list(csv.DictReader(stream))) == 247
    assert (snapshot / 'COMMIT_TIMINGS.csv').exists() and (snapshot / 'STATE_VERSIONS.csv').exists()
    # Real source checksum audit; metadata registration is not an inference on test.
    plan, queries = dataset(locate(), verify=True)
    assert len(queries) == 378 and len(plan['rows']) == 570
    original_count = check_hashes(ROOT.parent, read(ROOT / 'workstation_validation/ORIGINAL_BASELINE.json'))
    app_count = check_hashes(ROOT, read(ROOT / 'workstation_validation/DELIVERY.json')['source_hashes'])
    original_md = read(ROOT / 'audit/markdown-baseline.json')
    assert len(original_md) == 103 and all(sha(ROOT.parent / row['path']) == row['sha256'] for row in original_md)
    experiment_count = check_hashes(ROOT, read(ROOT / 'revision2_validation/DELIVERY.json')['sources'])
    r1_count = check_hashes(ROOT / 'runtime', read(ROOT / 'completion_20261028/hf/CONTRACT.json')['sources'])
    weights_sha = 'a5b872ed2e4cbfb3b9782b0df563e2606631744cd0274fee2da42a832f985f65'
    assert sha(ROOT / 'workstation/models/weights.pt') == sha(ROOT / 'completion_20261028/hf/report/weights.pt') == weights_sha
    for name, archive in [
        ('phase3_NB02_Audit_And_Direct_CSV.ipynb','completion_20261028/nb02_publication/executed.ipynb'),
        ('phase3_NB03_TREAD_ONE_Classifier_Diagnostic.ipynb','diagnostics/nb03_publication/executed.ipynb'),
        ('phase3_NB04_TREAD_ONE_Evidence_And_Recovery_Gate.ipynb','training_gates/nb04_publication/executed.ipynb'),
        ('phase3_NB05_TREAD_ONE_Shared_Gradient_Paired_Pilot.ipynb','revision2_validation/nb05_publication/executed.ipynb'),
        ('phase3_NB06_TREAD_ONE_Paired_Review_And_Failure_Atlas.ipynb','revision2_validation/nb06_publication/executed.ipynb')]:
        assert sha(ROOT / 'notebooks' / name) == sha(ROOT / archive)
    latest = Path(read(ROOT / 'data_gate_validation/nb07_reruns/LATEST.json')['path'])
    assert sha(ROOT / 'notebooks/phase3_NB07_TREAD_ONE_Data_And_Protocol_Gate.ipynb') == read(latest / 'VERIFICATION.json')['executed_notebook_sha256']
    # Original three-file prototype remains separate, including its zero-update proof.
    check_hashes(HERE, read(HERE / 'prototype_report/PROOF.json')['source_hashes'])
    assert read(HERE / 'prototype_report/PROOF.json')['optimizer_updates'] == 0
    html = (snapshot / 'report/ATLAS_Evidence_Explorer.html').read_text(encoding='utf-8')
    behavior = re.findall(r'<script>(.*?)</script>',html,re.S)
    assert len(behavior) == 1 and shutil.which('node')
    checked = subprocess.run([shutil.which('node'),'--check'],input=behavior[0],text=True,capture_output=True,check=False)
    assert checked.returncode == 0, checked.stderr
    payload = json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>',html,re.S)[1])
    assert len(payload['cases']) == 4 and len(payload['bank']) == 18 and payload['mode'] == 'engineering_fixture'
    assert all(len(item['references']) == 3 and len(item['references'][0]) == 18 and item['atlas_picture'].startswith('data:image/') for item in payload['cases'])
    for path in (snapshot / 'report').glob('*.png'):
        with Image.open(path) as picture:
            picture.verify()
    docs = list(ROOT.glob('*.md')) + list(HERE.glob('*.md')) + list((ROOT / 'workstation').glob('*.md'))
    docs += list((ROOT / 'data_intake').rglob('*.md')) + [ROOT / 'data_gate_outputs/CSV_INDEX.md']
    links = 0
    receipt = OUT / 'HANDOFF.json'
    for doc in docs:
        for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
            target = target.strip('<>')
            if re.match(r'^(https?://|mailto:|#)',target):
                continue
            resolved = (doc.parent / unquote(target.split('#')[0])).resolve()
            assert resolved.exists() or resolved == receipt.resolve(), (doc, target)
            links += 1
    result = dict(status='passed', notebook_sha256=notebook['notebook_sha256'], notebook_bytes=Path(notebook['notebook']).stat().st_size,
        embedded_files=25, source_asset_files=24, code_cells=6, scientific_CSV_families=21, column_definitions=247,
        local_fixture_rows=900380, local_fixture_updates=4, local_batch=8, offline_fault_checks=17,
        stop_marker_and_fresh_extracted_worker_passed=True, original_image_projection_max_residual=max_projection_residual,
        conservative_peak_output_bytes=local['actual_model_proof']['storage_projection']['conservative_peak_output_bytes'],
        original_workstation_files_preserved=original_count, separate_workstation_sources_preserved=app_count,
        original_Markdown_preserved=103, frozen_experiment_sources_preserved=experiment_count, frozen_r1_runtime_sources_preserved=r1_count,
        executed_NB02_to_NB07_preserved=True, original_prototype_preserved=True, data_release_hashes_verified=True,
        local_Markdown_documents=len(docs), local_links_checked=links, broken_links=[],
        local_network_requests=0, measured_fit_started=False, actual_Kaggle_dual_T4_tested=False, actual_live_HF_tested=False,
        interactive_HTML_payload_and_JS_syntax_checked=True, browser_render_tested=False,
        source_hashes=source, documentation_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in docs},
        check_script_sha256=sha(Path(__file__)), scope='Implemented NB08; local engineering acceptance only; live gates precede one measured Kaggle fit')
    atomic(receipt, result)
    print(json.dumps({k:v for k,v in result.items() if not k.endswith('hashes')},indent=2))


if __name__ == '__main__':
    main()
