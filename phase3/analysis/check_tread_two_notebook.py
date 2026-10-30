"""Extract the delivered notebook, verify all bytes, and run its actual worker."""
import ast
import base64
import csv
import itertools
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import nbformat
import torch
assert Path(sys.prefix).name == 'cv_conda'
ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / 'tread_two/pilot'
sys.path.insert(0, str(PILOT))
from common import atomic, file_meta, read, sha
from schemas import SCHEMAS
from store import Store
import acceptance

delivery = read(ROOT / 'tread_two/pilot_validation/NOTEBOOK_DELIVERY.json')
notebook = Path(delivery['notebook'])
assert sha(notebook) == delivery['notebook_sha256']
nb = nbformat.read(notebook, as_version=4)
nbformat.validate(nb)
payload = None
for cell in nb.cells:
    if cell.cell_type != 'code':
        continue
    assert cell.execution_count is None and not cell.outputs
    tree = ast.parse(cell.source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'b64decode' and node.args and isinstance(node.args[0], ast.Constant):
            payload = json.loads(base64.b64decode(node.args[0].value))
assert payload is not None and set(payload) == set(delivery['sources'])
out = ROOT / 'tread_two/pilot_validation'
folder = Path(tempfile.mkdtemp(prefix='fresh_notebook_', dir=out))
source = folder / 'source'
for name, encoded in payload.items():
    target = (source / name).resolve()
    assert target.is_relative_to(source.resolve())
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(base64.b64decode(encoded))
    assert target.read_bytes() == (PILOT / name).read_bytes() and sha(target) == delivery['sources'][name]
    if target.suffix == '.py':
        compile(target.read_text(encoding='utf-8'), str(target), 'exec')
inventory = read(source / 'FILES.json')
assert all(file_meta(source / name) == spec for name, spec in inventory.items())
with (source / 'assets/COLUMN_DICTIONARY.csv').open(encoding='utf-8', newline='') as stream:
    dictionary = list(csv.DictReader(stream))
assert {(row['table'],row['column']) for row in dictionary} == {(table,column) for table,columns in SCHEMAS.items() for column in columns}
local = read(out / 'LOCAL_VALIDATION.json')
cfg = read(Path(local['fixture_directory']) / 'CONFIG.json')
assert cfg['contract']['sources'] == {name:item['sha256'] for name,item in inventory.items()}
case = folder / 'actual_worker'
case.mkdir()
atomic(case / 'CONFIG.json', cfg)
environment = dict(os.environ)
environment.pop('HF_TOKEN', None)
with (case / 'console.log').open('ab') as stream:
    result = subprocess.run([sys.executable,'-B','-u',str(source/'worker.py'),'--config',str(case/'CONFIG.json'),
                             '--folder',str(case),'--limit-updates','1'], stdout=stream, stderr=subprocess.STDOUT, env=environment, check=False)
assert result.returncode == 91, 'Extracted notebook worker failed; inspect fresh fixture console.log'
actual = Store(case, cfg['contract'], SCHEMAS)
expected = Store(Path(local['fixture_directory']) / 'actual/paused', cfg['contract'], SCHEMAS)
assert actual.state['updates'] == expected.state['updates'] == 1
for key in ('model','optimizer','scheduler','scaler','rng','row_cursors','epoch_orders','train_records'):
    acceptance.same(actual.state[key], expected.state[key], key)
omit = {'compute_seconds','record_seconds','cuda_peak_allocated_bytes','cuda_peak_reserved_bytes','host_peak_rss_bytes','free_disk_bytes'}
for table in SCHEMAS:
    for a,b in itertools.zip_longest(actual.rows(table), expected.rows(table)):
        assert a is not None and b is not None
        if table == 'train_updates':
            a,b = ({key:value for key,value in row.items() if key not in omit} for row in (a,b))
        assert a == b, table
record = dict(status='passed', notebook_sha256=sha(notebook), embedded_files=len(payload), source_files=len(inventory),
              dictionary_definitions=len(dictionary), code_cells=delivery['code_cells'], actual_batch=8,
              extracted_worker_completed_updates=1, extracted_model_optimizer_RNG_and_scientific_CSV_match=True,
              network_requests=0, Kaggle_executed=False, fixture_directory=str(folder))
atomic(out / 'NOTEBOOK_VALIDATION.json', record)
print(json.dumps(record, indent=2))
