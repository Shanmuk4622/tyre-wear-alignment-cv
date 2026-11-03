"""Start an isolated measured-pilot implementation, preserving executed NB16."""
import json,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'tread_two/geometry_first/runtime';OUT=ROOT/'tread_two/geo_pilot'
OUT.mkdir(exist_ok=True)
names=('common.py','frozen_anchor.py','phase2_training_data.py','legacy_data.py','geometry.py','lc_features.py',
       'evidence_hub.py','publisher.py','rate_budget.py','data_geo.py','schedule_geo.py','anchor_geo.py',
       'atlas_geo.py','model_geo.py','objective_geo.py','PLAN.json','SCHEMAS.json','SOURCE_COMPONENTS.csv')
import hashlib
inventory={}
for name in names:
    body=(SRC/name).read_bytes();target=OUT/name
    if target.exists():assert target.read_bytes()==body,'Refusing to replace an edited copied source: '+name
    else:target.write_bytes(body)
    inventory[name]={'source':'phase3/tread_two/geometry_first/runtime/'+name,'sha256':hashlib.sha256(body).hexdigest()}
(OUT/'UPSTREAM.json').write_text(json.dumps(inventory,indent=2)+'\n',encoding='utf-8')
completion=Path(json.loads((ROOT/'tread_two/nb16_completion/LATEST.json').read_text())['path'])
verified=json.loads((completion/'VERIFICATION.json').read_text())
assert verified['status']=='passed' and verified['actual_dual_T4'] and verified['actual_HF_restore_continue']
(OUT/'NB16_GATE.json').write_text(json.dumps({k:verified[k] for k in ('revision','prefix','contract','executed_notebook_sha256','parameters','environment')},indent=2)+'\n',encoding='utf-8')
print(json.dumps({'isolated_copies':len(inventory),'NB16_preserved':True}))
