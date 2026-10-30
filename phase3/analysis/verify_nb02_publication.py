"""Read-only verification of the executed NB02 and its pinned direct-file commit.

Notebook contents are parsed as data. They are never executed. Remote reads use
explicit short paths and no token; this script cannot publish or train a model.
"""
import ast
import base64
import csv
import gzip
import hashlib
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

assert Path(sys.prefix).name == 'cv_conda'
import nbformat
import requests
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'completion_20261028/nb02_publication'
NOTEBOOK = ROOT/'notebooks/phase3_NB02_Audit_And_Direct_CSV.ipynb'
REPO = 'Shanmuk4622/tyre-wear-study'
SOURCE = 'a0296f18e02b12d28f7e51377783d68538032209'
REVISION = '91ccd3315bd296a76a8b583a6d0f2515ccb3d370'
PROTOCOL = '371c40c0c6aaedaf5c5970f7721a284b042494db7bb330065f6cbde8fc3a0b8c'
DEST = 'phase3/tread-one-r1/'+PROTOCOL+'/posthoc-csv/2026-10-28'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def text(value):
    return ''.join(value) if isinstance(value, list) else value or ''


def retry(fn):
    for attempt in range(6):
        try:
            return fn()
        except Exception as exc:
            response = getattr(exc, 'response', None)
            status = getattr(response, 'status_code', None)
            if (status not in (429, 500, 502, 503, 504)
                    and not (status is None and isinstance(exc, requests.RequestException))) or attempt == 5:
                raise
            headers = getattr(response, 'headers', {}) or {}
            raw = headers.get('Retry-After', '0')
            try:
                delay = float(raw)
            except ValueError:
                from email.utils import parsedate_to_datetime
                delay = max(0., (parsedate_to_datetime(raw)-datetime.now(timezone.utc)).total_seconds())
            rate = re.search(r'\bt=(\d+)', headers.get('RateLimit', ''))
            delay = max(delay, float(rate.group(1)) if rate else 0., min(60, 5*2**attempt))
            print('Public verification read retry:', status, 'server/backoff seconds:', delay, flush=True)
            while delay > 0:
                chunk = min(delay, 30.)
                time.sleep(chunk)
                delay -= chunk


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = NOTEBOOK.read_bytes()
    archive = OUT/'executed.ipynb'
    if archive.exists():
        assert archive.read_bytes() == raw, 'A different executed notebook needs a separate evidence folder'
    else:
        archive.write_bytes(raw)
    nb = nbformat.reads(raw.decode('utf-8'), as_version=4)
    nbformat.validate(nb)
    cells, publications, embedded = [], [], {}
    decoder = json.JSONDecoder()
    all_streams = []
    for index, cell in enumerate(nb.cells):
        if cell.cell_type != 'code':
            continue
        assert isinstance(cell.execution_count, int) and cell.execution_count > 0, index
        ast.parse(cell.source)
        assert not any(o.output_type == 'error' for o in cell.outputs), index
        streams = ''.join(text(o.get('text')) for o in cell.outputs if o.output_type == 'stream')
        all_streams.append(streams)
        cells.append(dict(index=index, execution_count=cell.execution_count,
                          output_types=[o.output_type for o in cell.outputs],
                          source_sha256=hashlib.sha256(cell.source.encode()).hexdigest()))
        for offset, ch in enumerate(streams):
            if ch != '{':
                continue
            try:
                obj, _ = decoder.raw_decode(streams[offset:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get('status') == 'published_and_verified':
                publications.append(obj)
        for call in ast.walk(ast.parse(cell.source)):
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and call.func.attr == 'b64decode' and call.args
                    and isinstance(call.args[0], ast.Constant)):
                payload = json.loads(gzip.decompress(base64.b64decode(call.args[0].value)))
                for name, value in payload.items():
                    body = base64.b64decode(value)
                    assert body == (ROOT/'analysis'/name).read_bytes(), name
                    embedded[name] = hashlib.sha256(body).hexdigest()
    assert len(cells) == 5 and [c['execution_count'] for c in cells] == [1, 2, 3, 4, 5]
    assert len(embedded) == 3 and publications and all(p == publications[0] for p in publications)
    publication = publications[0]
    assert publication == dict(status='published_and_verified', revision=REVISION, destination=DEST,
                               source_revision=SOURCE, files=25, zip_files_uploaded=0)
    (OUT/'CSV_PUBLICATION.json').write_text(json.dumps(publication, indent=2)+'\n', encoding='utf-8')
    (OUT/'notebook_streams.txt').write_text('\n'.join(all_streams), encoding='utf-8')
    print('Executed notebook: all five code cells complete; source payloads match r2.', flush=True)

    api = HfApi(token=False)
    entries = retry(lambda: list(api.list_repo_tree(REPO, path_in_repo=DEST, revision=REVISION,
                                                   repo_type='dataset', recursive=True)))
    files = {v.path[len(DEST)+1:]: v for v in entries if hasattr(v, 'size')}
    assert len(files) == 25 and not any(Path(n).suffix.lower() in ('.zip', '.npz', '.pt') for n in files)
    session = requests.Session()
    session.trust_env = False  # No alternate credential files, auth or proxy discovery.

    def download(name):
        assert name in files and not Path(name).is_absolute() and '..' not in Path(name).parts
        target = OUT/'remote'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        entry = files[name]

        def valid():
            if not target.is_file() or target.stat().st_size != entry.size:
                return False
            lfs = getattr(entry, 'lfs', None)
            if lfs:
                return sha(target) == lfs.sha256
            body = target.read_bytes()
            return hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest() == entry.blob_id

        if not valid():
            def transfer():
                url = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{DEST}/{name}'
                part = target.with_suffix(target.suffix+'.part')
                with session.get(url, timeout=(20, 120), stream=True) as response:
                    response.raise_for_status()
                    with open(part, 'wb') as stream:
                        for block in response.iter_content(1024**2):
                            stream.write(block)
                part.replace(target)
            retry(transfer)
        assert valid(), name
        return target

    manifest = read(download('MANIFEST.json'))
    assert manifest['source_revision'] == SOURCE and manifest['source_protocol'] == PROTOCOL
    assert set(files) == set(manifest['files']) | {'MANIFEST.json'}
    records = []
    for name in sorted(files):
        path = download(name)
        if name != 'MANIFEST.json':
            assert path.stat().st_size == manifest['files'][name]['bytes']
            assert sha(path) == manifest['files'][name]['sha256'], name
        records.append(dict(path=name, bytes=path.stat().st_size, sha256=sha(path),
                            blob_id=files[name].blob_id))
    schema = read(OUT/'remote/CSV_SCHEMA.json')
    audit = read(OUT/'remote/AUDIT.json')
    assert audit['status'] == 'passed' and audit['source_revision'] == SOURCE
    tables, compared, rounding = [], 0, []
    for name, spec in sorted(schema['files'].items()):
        path = OUT/'remote/csv'/name
        assert path.stat().st_size == spec['bytes'] and sha(path) == spec['sha256'], name
        with open(path, encoding='utf-8', newline='') as stream:
            reader = csv.DictReader(stream)
            columns = reader.fieldnames
            rows = list(reader)
        assert columns == spec['columns'] and len(rows) == spec['rows'], name
        local = ROOT/'completion_20261028/csv'/name
        # The artifact manifest reflects Kaggle file bytes, whose line endings
        # can differ from Windows. Compare every scientific table semantically.
        if name != 'artifact_manifest.csv':
            with open(local, encoding='utf-8', newline='') as stream:
                reader = csv.DictReader(stream)
                local_rows = list(reader)
                assert reader.fieldnames == columns and len(local_rows) == len(rows), name
            for index, (actual, reference) in enumerate(zip(rows, local_rows)):
                for column in columns:
                    a, b = actual[column], reference[column]
                    if a == b:
                        continue
                    # Float arithmetic across NumPy versions may differ at the
                    # final decimal. Identity/text/integer changes still fail.
                    numeric = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
                    assert (re.fullmatch(numeric, a) and re.fullmatch(numeric, b)
                            and any(c in a+b for c in '.eE')), (name, index, column, a, b)
                    af, bf = float(a), float(b)
                    assert math.isfinite(af) and math.isfinite(bf) and math.isclose(af, bf, rel_tol=1e-12, abs_tol=1e-12), (name, index, column, a, b)
                    rounding.append(dict(file=name, data_row=index+1, column=column,
                                         remote_value=a, local_value=b, absolute_difference=abs(af-bf)))
            compared += len(rows)
        tables.append(dict(file=name, rows=len(rows), columns=len(columns), bytes=path.stat().st_size,
                           sha256=sha(path), local_semantic_match=name != 'artifact_manifest.csv'))
    assert len(tables) == 18
    for name, expected in (('training_steps.csv', 8820), ('validation_predictions.csv', 14580),
                           ('test_predictions.csv', 618), ('training_samples.csv', 70560)):
        assert schema['files'][name]['rows'] == expected
    for name, expected in embedded.items():
        assert sha(OUT/'remote/analysis'/name) == expected
    result = dict(status='passed', verified_at_utc=datetime.now(timezone.utc).isoformat(),
                  repository=REPO, revision=REVISION, source_revision=SOURCE, protocol=PROTOCOL,
                  destination=DEST, notebook_sha256=sha(archive), completed_code_cells=cells,
                  notebook_errors=0, embedded_sources=embedded, published_files=len(files), csv_tables=tables,
                  csv_table_count=18, verified_scientific_rows=compared,
                  total_published_bytes=sum(r['bytes'] for r in records),
                  zip_files_uploaded=0, all_remote_hashes_verified=True,
                  all_scientific_tables_match_local=True, remote_write_performed_by_this_audit=False,
                  comparison_float_tolerance=dict(absolute=1e-12, relative=1e-12),
                  rounding_differences=rounding,
                  training_performed_by_this_audit=False, public_reads_without_token=True,
                  live_publication_verified=True, files=records)
    (OUT/'VERIFICATION.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('csv_tables', 'files', 'completed_code_cells')}, indent=2))


if __name__ == '__main__':
    main()
