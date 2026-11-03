"""Copy immutable upstream dependencies into the new engineering package only."""
import hashlib
import json
import sys
from pathlib import Path

assert Path(sys.prefix).name == 'cv_conda'
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'tread_two/geometry_first/runtime'
OUT.mkdir(parents=True, exist_ok=True)
names = ('common.py', 'frozen_anchor.py', 'phase2_training_data.py', 'legacy_data.py',
         'geometry.py', 'lc_features.py', 'evidence_hub.py', 'publisher.py', 'rate_budget.py')
inventory = {}
for name in names:
    src = ROOT / 'tread_two/context_pilot' / name
    body = src.read_bytes()
    target = OUT / name
    if target.exists() and target.read_bytes() != body:
        raise RuntimeError('Refusing to overwrite modified vendored dependency: ' + name)
    target.write_bytes(body)
    inventory[name] = {'source': 'phase3/tread_two/context_pilot/' + name,
                       'sha256': hashlib.sha256(body).hexdigest()}
(OUT / 'VENDORED.json').write_text(json.dumps(inventory, indent=2) + '\n', encoding='utf-8')
for name in ('PLAN.json', 'SCHEMAS.json', 'BUDGET.json'):
    (OUT / name).write_bytes((OUT.parent / 'plan' / name).read_bytes())
source = ROOT / 'data_gate_outputs/csv/source_images.csv'
(OUT / 'SOURCE_COMPONENTS.csv').write_bytes(source.read_bytes())
print(json.dumps({'copied_frozen_dependencies': len(names), 'original_sources_changed': False}))
