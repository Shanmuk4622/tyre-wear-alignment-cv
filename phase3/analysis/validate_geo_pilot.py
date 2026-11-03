"""Disposable CPU lifecycle, streamed storage/recovery and frozen NB16 preservation."""
import hashlib,json,sys,time
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'tread_two/geo_pilot'
sys.path.insert(0,str(SOURCE))
from common import atomic,digest,file_meta,read,sha
inventory={p.name:file_meta(p) for p in SOURCE.iterdir() if p.is_file() and p.name!='FILES.json'}
atomic(SOURCE/'FILES.json',inventory)
from engine_fit import Engine
folder=ROOT/('tread_two/geo_validation/fixture_lifecycle_'+digest(inventory)[:12]);folder.mkdir(parents=True,exist_ok=True)
start=time.monotonic()
engine=Engine(folder,ROOT/'workstation/models/weights.pt',data_root=str(ROOT.parent/'phase2/data/phase2_dataset_v1'),device='cpu',eval_device='cpu',fixture=True)
for step in range(200):
    if engine.cursor['phase']=='completed':break
    print('local engineering:',engine.cursor['phase'],'unit',engine.store.unit,'update',engine.cursor['update'],flush=True)
    engine.step()
    if engine.store.unit==21:
        # Real new-model/optimizer/native/CSV reconstruction in a new Engine.
        del engine
        engine=Engine(folder,ROOT/'workstation/models/weights.pt',data_root=str(ROOT.parent/'phase2/data/phase2_dataset_v1'),device='cpu',eval_device='cpu',fixture=True)
assert engine.cursor['phase']=='completed'
snapshot=engine.store.snapshot(folder/'snapshot',allow_copy=True)
result=read(engine.store.root/'RESULT.json')
assert result['updates']==4 and result['scope']=='engineering_fixture'
nb16=Path(read(ROOT/'tread_two/nb16_completion/LATEST.json')['path']);proof=read(nb16/'VERIFICATION.json')
assert sha(ROOT/'notebooks/phase3_NB16_TREAD_TWO_GEO_Preparation_And_Contract_Gate.ipynb')==proof['executed_notebook_sha256']
for name,meta in read(ROOT/'tread_two/geometry_first/runtime/FILES.json').items():assert sha(ROOT/'tread_two/geometry_first/runtime'/name)==meta['sha256']
out=ROOT/'tread_two/geo_validation';out.mkdir(exist_ok=True)
receipt=dict(status='passed',scope='local_CPU_disposable_lifecycle',source_inventory_sha256=digest(inventory),
             fixture=str(folder),units=engine.store.unit,updates=engine.cursor['update'],row_counts=engine.store.rows,
             snapshot=snapshot,result=result,seconds=time.monotonic()-start,NB16_preserved=True,actual_new_live_gate_pending=True)
atomic(out/'LIFECYCLE.json',receipt)
print(json.dumps({k:v for k,v in receipt.items() if k not in ('result','row_counts')},indent=2))
