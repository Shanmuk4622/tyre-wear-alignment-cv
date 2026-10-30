"""Preserve the supplied executed NB08 and safely summarize its saved errors."""
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

assert Path(sys.prefix).name == 'cv_conda'
ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'notebooks/phase3_NB08_TREAD_TWO_ATLAS_Resumable_Pilot.ipynb'
raw = path.read_bytes()
identity = hashlib.sha256(raw).hexdigest()
archive = ROOT / 'tread_two/pilot_validation/nb08_repair' / identity[:12]
archive.mkdir(parents=True, exist_ok=True)
target = archive / 'executed.ipynb'
if target.exists():
    assert target.read_bytes() == raw
else:
    target.write_bytes(raw)
for name in ('NOTEBOOK_DELIVERY.json','HANDOFF.json','LOCAL_VALIDATION.json'):
    saved = archive / ('ORIGINAL_' + name)
    if not saved.exists():
        shutil.copy2(ROOT / 'tread_two/pilot_validation' / name, saved)


def safe(value):
    text = ''.join(value) if isinstance(value, list) else str(value)
    text = re.sub(r'\x1b\[[0-9;]*m', '', text)
    text = re.sub(r'hf_[A-Za-z0-9]+', '[HF_TOKEN_REDACTED]', text)
    text = re.sub(r'(?i)(Bearer\s+)[^\s\"\']+', r'\1[REDACTED]', text)
    return text


nb = json.loads(raw)
records = []
for index, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    outputs = []
    for output in cell.get('outputs', []):
        if output['output_type'] == 'error':
            outputs.append(dict(type='error', name=output.get('ename'), value=safe(output.get('evalue','')),
                                traceback=safe(output.get('traceback',[]))))
        elif output['output_type'] == 'stream':
            value = safe(output.get('text', ''))
            outputs.append(dict(type='stream', text=value[-18000:]))
    record = dict(cell_index=index, execution_count=cell.get('execution_count'), outputs=outputs)
    records.append(record)
report = dict(notebook_sha256=identity, bytes=len(raw), archived=str(target), code_cells=records)
(archive / 'INSPECTION.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
print(json.dumps(report, indent=2))
