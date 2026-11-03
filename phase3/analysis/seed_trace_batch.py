"""Vendor immutable, already-tested helpers into a NEW, unmeasured TRACE package."""
import json,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'tread_two/trace_batch';OUT.mkdir(exist_ok=True)
source=ROOT/'tread_two/geo_pilot'
files=['common.py','evidence_hub.py','publisher.py','rate_budget.py','frozen_anchor.py','geometry.py','atlas_geo.py','legacy_data.py','phase2_training_data.py','SOURCE_COMPONENTS.csv']
files2=['targets.py','model.py','objective.py','frozen_geometry.py']
import hashlib
inventory={}
for base,names in [(source,files),(ROOT/'tread_two/trace_design/runtime',files2)]:
    for name in names:
        body=(base/name).read_bytes();path=OUT/name
        if path.exists():assert path.read_bytes()==body,('Do not reset edited new source',name)
        else:path.write_bytes(body)
        inventory[name]={'origin':str((base/name).relative_to(ROOT)),'sha256':hashlib.sha256(body).hexdigest()}
(OUT/'VENDORED.json').write_text(json.dumps(inventory,indent=2)+'\n',encoding='utf-8')
(OUT/'SCHEMAS.json').write_bytes((ROOT/'tread_two/trace_design/SCHEMAS.json').read_bytes())
print('Seeded new TRACE package only:',len(inventory),'immutable helper/core copies')
