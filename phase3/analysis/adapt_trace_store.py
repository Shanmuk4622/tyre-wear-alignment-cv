"""Create a separately versioned TRACE store; never edit the GEO source."""
import hashlib,json,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'tread_two/trace_batch'
source=(ROOT/'tread_two/geo_pilot/store_fit.py').read_text(encoding='utf-8')
source=source.replace('from schema_fit import SCHEMA,encoded','from schema import SCHEMA,encoded')
source=source.replace("precision='FP32_no_scaler',scheduler='cosine_at_previous_completed_global_update'","precision='FP32_no_scaler',scheduler=None,scaler=None")
old="self.append_csv('unit_ledger',[dict(unit=unit,kind=kind or cursor.get('record_stage',cursor['phase']),update=cursor.get('update',0),csv_prefix_sha256=digest(csv_prefix),native_state_sha256=state_hash,previous_unit_sha256=self.chain,current_unit_sha256=chain,state_semantic_sha256=state_hash)])"
new="self.append_csv('state_ledger',[dict(stage=kind or cursor['phase'],unit=unit,update=cursor.get('update',0),csv_prefix_sha256=digest(csv_prefix),native_prefix_sha256=digest(native_prefix),state_sha256=state_hash,previous_unit_sha256=self.chain,chain_sha256=chain,next_cursor=canonical(cursor).decode())])"
assert old in source;source=source.replace(old,new)
old='target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)'
new='target.parent.mkdir(parents=True,exist_ok=True)\n            if not target.exists():\n                try:os.link(path,target)\n                except OSError:shutil.copyfile(path,target)\n            assert file_meta(target)==file_meta(path)'
assert old in source;source=source.replace(old,new)
assert not (OUT/'store.py').exists(),'One-time bootstrap only: never overwrite the developed TRACE store'
(OUT/'store.py').write_text(source,encoding='utf-8')
(OUT/'DATA_PIN.json').write_text(json.dumps(json.loads((ROOT/'tread_two/geo_pilot/PLAN.json').read_text())['data'],indent=2)+'\n')
(OUT/'STORE_LINEAGE.json').write_text(json.dumps(dict(origin='geo_pilot/store_fit.py',source_sha256=hashlib.sha256((ROOT/'tread_two/geo_pilot/store_fit.py').read_bytes()).hexdigest(),changes=['executable TRACE schema','state_ledger fields','explicit no scheduler/scaler','immutable input hardlink reuse']),indent=2)+'\n')
print('New TRACE transactional store and exact data pin created.')
