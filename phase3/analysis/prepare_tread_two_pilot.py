"""Initialize isolated source copies; never rewrite an edited pilot module."""
import json
import sys
import hashlib
from pathlib import Path

assert Path(sys.prefix).name == 'cv_conda'
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'tread_two/pilot'
OUT.mkdir(parents=True, exist_ok=True)
copies = {
    'frozen_anchor.py': 'runtime/model.py',
    'phase2_training_data.py': 'runtime/phase2_training_data.py',
    'legacy_data.py': 'runtime/data.py',
    'geometry.py': 'tread_two/atlas.py',
    'evidence_hub.py': 'training/evidence_hub.py',
    'publisher.py': 'revision2/publisher_r2.py',
}
provenance = {}
for target, source in copies.items():
    value = (ROOT / source).read_bytes()
    if not (OUT / target).exists():
        text = value.decode('utf-8')
        if target == 'publisher.py':
            text = text.replace('import spec\n', '').replace('from transactions import', 'from common import')
            text = text.replace('TREAD-ONE r2', 'TREAD-TWO ATLAS')
        if target == 'evidence_hub.py':
            text = text.replace('from transactions import', 'from common import').replace('TREAD-ONE', 'TREAD-TWO ATLAS')
        if target in ('publisher.py', 'evidence_hub.py'):
            (OUT / target).write_bytes(text.replace('\r\n', '\n').encode('utf-8'))
        else:
            (OUT / target).write_bytes(value)
    provenance[target] = dict(source=source, original_sha256=hashlib.sha256(value).hexdigest(),
                              copied_or_adapted=True)
(OUT / 'VENDORED.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
print('Isolated pilot dependencies prepared; previous experiment sources unchanged.')
