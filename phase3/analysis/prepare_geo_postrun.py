"""Isolate finite, zero-fit postrun analysis/transport from the completed pilot."""
import hashlib,json,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'tread_two/geo_postrun';OUT.mkdir(exist_ok=True)
inventory={}
for name in ('common.py','evidence_hub.py','publisher.py','rate_budget.py'):
    src=ROOT/'tread_two/geo_pilot'/name;body=src.read_bytes();target=OUT/name
    if target.exists():assert target.read_bytes()==body
    else:target.write_bytes(body)
    inventory[name]=dict(source='phase3/tread_two/geo_pilot/'+name,sha256=hashlib.sha256(body).hexdigest())
(OUT/'VENDORED.json').write_text(json.dumps(inventory,indent=2)+'\n',encoding='utf-8')
pin=dict(repository='Shanmuk4622/tyre-wear-study',revision='1113bf8a9dd69687b23570fe985ed3a979a5f0e1',protocol='7d1bc8554813ae16ba3d73d5d4af2d8c05c977e9282d1bf324ff9dbef6e89bdb',prefix='phase3/tread-two/geo-r1/fit/7d1bc8554813ae16ba3d73d5d4af2d8c05c977e9282d1bf324ff9dbef6e89bdb',logical_files=11637,selected_objects=11636,bytes=10307839796,expected_epochs=60,expected_updates=2940,expected_samples=30447,expected_unit=9877,source_checksum='06c6c9d003aff808babf7ebb5786f441e95064eb000f5e5ed5b155d0ee9e7c6a')
(OUT/'FIT_INPUT.json').write_text(json.dumps(pin,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'copied_dependencies':4,'separate_postrun_package':str(OUT)}))
