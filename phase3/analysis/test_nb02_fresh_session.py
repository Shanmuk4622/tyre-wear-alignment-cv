"""Run the actual embedded fetch script in a pristine notebook-like workspace.

Pinned local response fixtures replace HTTP. No credentials, network or training.
"""
import ast
import base64
import contextlib
import gzip
import hashlib
import io
import json
import os
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

assert Path(sys.prefix).name == 'cv_conda'
import huggingface_hub
import requests
import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'completion_20261028'
FIXTURES = OUT/'hf'
TREE = json.loads((OUT/'remote_tree.json').read_text())
FAILED = OUT/'nb02_repair/c6c6b287249a/executed.ipynb'
CURRENT = ROOT/'notebooks/phase3_NB02_Audit_And_Direct_CSV.ipynb'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def embedded(path):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        for call in ast.walk(ast.parse(cell.source)):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == 'b64decode' and call.args and isinstance(call.args[0], ast.Constant):
                payload = json.loads(gzip.decompress(base64.b64decode(call.args[0].value)))
                return {k: base64.b64decode(v) for k, v in payload.items()}
    raise AssertionError('No embedded runtime')


class Service:
    def __init__(self, flaky=False):
        self.download_calls = 0
        self.list_calls = 0
        self.flaky = flaky
        self.interruptions = 0

    def api(self, token):
        assert token is False
        service = self
        class Api:
            def list_repo_tree(self, repo, path_in_repo, revision, repo_type, recursive):
                assert repo == TREE['repo'] and revision == TREE['revision'] and path_in_repo == TREE['prefix']
                service.list_calls += 1
                return [SimpleNamespace(path=TREE['prefix']+'/'+r['path'], size=r['bytes'],
                         lfs=SimpleNamespace(sha256=r['lfs_sha256']) if r['lfs_sha256'] else None,
                         blob_id=r['blob_id']) for r in TREE['files']]
        return Api()

    def get(self, url, headers, timeout, stream):
        assert headers == {} and stream
        prefix = f'https://huggingface.co/datasets/{TREE["repo"]}/resolve/{TREE["revision"]}/{TREE["prefix"]}/'
        assert url.startswith(prefix)
        path = FIXTURES/url[len(prefix):]
        assert path.is_file()
        self.download_calls += 1
        service = self
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def raise_for_status(self): pass
            def iter_content(self, block):
                with open(path, 'rb') as stream:
                    if service.flaky and service.interruptions == 0:
                        service.interruptions += 1
                        yield stream.read(100)
                        raise requests.exceptions.ConnectionError('fixture: interrupted stream')
                    for data in iter(lambda: stream.read(block), b''):
                        yield data
        return Response()


def execute(payload, folder, service, *arguments):
    scripts = folder/'analysis'
    scripts.mkdir(parents=True, exist_ok=True)
    for name, data in payload.items():
        (scripts/name).write_bytes(data)
    with patch.object(huggingface_hub, 'HfApi', side_effect=service.api), \
         patch.object(huggingface_hub, 'get_token', return_value=None), \
         patch.object(requests, 'get', side_effect=service.get), \
         patch('time.sleep'), patch.object(sys, 'argv', [str(scripts/'fetch_completed_results.py'), *arguments]), \
         contextlib.redirect_stdout(io.StringIO()):
        runpy.run_path(str(scripts/'fetch_completed_results.py'), run_name='__main__')


def main():
    base = OUT/'nb02_repair'
    work = Path(tempfile.mkdtemp(prefix='fresh_', dir=base)).resolve()
    assert work.is_relative_to(base.resolve())
    old = embedded(FAILED)
    reproduced = False
    try:
        execute(old, work/'old', Service())
    except FileNotFoundError as exc:
        assert Path(exc.filename).name == 'remote_tree.json'
        reproduced = True
    assert reproduced, 'Original fresh-session error was not reproduced'
    payload = embedded(CURRENT)
    for name, data in payload.items():
        assert data == (ROOT/'analysis'/name).read_bytes()
    service = Service(flaky=True)
    folder = work/'fixed'
    assert not (folder/'completion_20261028').exists()
    execute(payload, folder, service, '--bundles')
    local = folder/'completion_20261028'
    assert (local/'remote_tree.json').is_file() and (local/'fetched_metadata.json').is_file()
    files = list((local/'hf').rglob('*.json'))
    assert len(files) == 11 and service.interruptions == 1
    assert all(sha(p) == sha(FIXTURES/p.relative_to(local/'hf')) for p in files)
    bundles = list((local/'hf/runs').glob('seed*/latest.zip'))
    assert len(bundles) == 3
    assert all(sha(p) == sha(FIXTURES/p.relative_to(local/'hf')) for p in bundles)
    assert not list(local.rglob('*.part'))
    first_hashes = {str(p.relative_to(local)): sha(p) for p in files}
    cached = Service()
    execute(payload, folder, cached, '--bundles')
    assert cached.download_calls == 0 and cached.list_calls == 0
    assert first_hashes == {str(p.relative_to(local)): sha(p) for p in files}
    # Direct invocation of the CSV stage also initializes missing parent paths.
    fresh_audit = work/'standalone/analysis'
    fresh_audit.mkdir(parents=True)
    path = fresh_audit/'audit_and_export_csv.py'
    path.write_bytes(payload[path.name])
    runpy.run_path(str(path), run_name='directory_initialization_test')
    assert (fresh_audit.parent/'completion_20261028/csv').is_dir()
    # Exercise the remaining real CPU stages in this newly created workspace.
    # The publisher's prepare-only path never calls a service or loads a token.
    environment = dict(os.environ, PHASE3_SOURCE_ROOT=str(ROOT.parent/'phase2/data/phase2_dataset_v1'))
    for name, arguments in (('audit_and_export_csv.py', []),
                            ('publish_recovered_csv.py', ['--prepare-only'])):
        completed = subprocess.run([sys.executable, '-B', str(folder/'analysis'/name), *arguments],
                                   env=environment, capture_output=True, text=True)
        (work/(name+'.log')).write_text(completed.stdout+completed.stderr, encoding='utf-8')
        assert completed.returncode == 0, (name, completed.stderr[-4000:])
    audit = json.loads((local/'AUDIT.json').read_text())
    assert audit['status'] == 'passed'
    schema = json.loads((local/'CSV_SCHEMA.json').read_text())
    assert schema['files']['training_steps.csv']['rows'] == 8820
    assert schema['files']['validation_predictions.csv']['rows'] == 14580
    assert schema['files']['test_predictions.csv']['rows'] == 618
    manifest = json.loads((local/'CSV_MANIFEST.json').read_text())
    assert all(not name.endswith(('.zip', '.npz', '.pt')) for name in manifest['files'])
    assert not (local/'CSV_PUBLICATION.json').exists()
    result = dict(status='passed', original_failure_reproduced=True,
        fresh_embedded_fetch_completed=True, verified_metadata_files=11,
        verified_legacy_bundles=len(bundles), full_fresh_workspace_csv_audit_passed=True,
        csv_tables=len(schema['files']), prepared_manifest_files=len(manifest['files'])+1,
        training_update_rows=8820, validation_prediction_rows=14580, test_prediction_rows=618,
        zip_uploads_planned=0,
        interrupted_download_retried_without_corruption=True, cached_restart_download_calls=0,
        standalone_csv_parent_directories_created=True, embedded_sources_match_current_files=True,
        fixed_notebook_sha256=sha(CURRENT), failed_notebook_sha256=sha(FAILED),
        network_calls_performed=False, credential_access_performed=False,
        actual_kaggle_reexecution_tested=False, live_hf_publication_tested=False,
        fixture_directory=str(work.relative_to(ROOT)))
    (base/'REGRESSION.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
