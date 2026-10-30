"""Inspect notebook errors as untrusted data; preserve failed execution bytes."""
import hashlib
import json
import re
import sys
from pathlib import Path

assert Path(sys.prefix).name == 'cv_conda'
ROOT = Path(__file__).resolve().parents[1]
path = ROOT/'notebooks/phase3_NB02_Audit_And_Direct_CSV.ipynb'
raw = path.read_bytes()
digest = hashlib.sha256(raw).hexdigest()
folder = ROOT/'completion_20261028/nb02_repair'/digest[:12]
folder.mkdir(parents=True, exist_ok=True)
archive = folder/'executed.ipynb'
if archive.exists():
    assert archive.read_bytes() == raw
else:
    archive.write_bytes(raw)
nb = json.loads(raw)
records = []
def join(value):
    return ''.join(value) if isinstance(value, list) else value or ''
def safe(value):
    return re.sub(r'hf_[A-Za-z0-9]{20,}', '[REDACTED]', value)
for i, cell in enumerate(nb['cells']):
    lines, errors = [], []
    for output in cell.get('outputs', []):
        if output.get('output_type') == 'error':
            errors.append(dict(name=output.get('ename'), value=output.get('evalue'), traceback=output.get('traceback')))
            lines += output.get('traceback', [])
        elif output.get('output_type') == 'stream':
            lines += join(output.get('text')).splitlines()
        else:
            lines += join(output.get('data', {}).get('text/plain')).splitlines()
    rec = dict(cell=i, type=cell['cell_type'], execution_count=cell.get('execution_count'), errors=errors,
               output_tail=lines[-30:])
    records.append(rec)
    print(safe(json.dumps(rec, ensure_ascii=False)))
(folder/'inspection.json').write_text(safe(json.dumps(dict(notebook_sha256=digest, cells=records), indent=2))+'\n', encoding='utf-8')
print('ARCHIVE', str(archive), 'SHA256', digest)
