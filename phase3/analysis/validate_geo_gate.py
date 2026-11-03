"""New store full-model seam in fresh processes; no HF writes or measured fit."""
import json,shutil,subprocess,sys,time
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'tread_two/geo_pilot'
sys.path.insert(0,str(SOURCE))
from common import atomic,digest,file_meta,read,sha
inventory={p.name:file_meta(p) for p in SOURCE.iterdir() if p.suffix in ('.py','.json','.csv') and p.name!='FILES.json'}
atomic(SOURCE/'FILES.json',inventory)
base=ROOT/('tread_two/geo_validation/fixture_gate_'+digest(inventory)[:12]);base.mkdir(parents=True,exist_ok=True)
start=time.monotonic();worker=SOURCE/'worker_fit.py';weights=ROOT/'workstation/models/weights.pt';data=ROOT.parent/'phase2/data/phase2_dataset_v1'
def run(mode,output,reference=None):
    args=[sys.executable,'-B',str(worker),mode,'--output',str(output),'--weights',str(weights),'--data-root',str(data),'--offline']
    if reference:args+=['--reference',str(reference)]
    process=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (base/(mode+'.log')).write_text(process.stdout,encoding='utf-8')
    assert process.returncode==0,process.stdout[-7000:]
    print(process.stdout[-2500:],flush=True)
run('gate_direct',base/'direct')
source=base/'direct/seam';target=base/'restored/run'
if not target.exists():shutil.copytree(source,target)
manifest=read(source/'MANIFEST.json');assert all(file_meta(target/n)==m for n,m in manifest['files'].items())
run('gate_restore',base/'restored',base/'direct')
proof=read(base/'restored/RESTORE_PROOF.json');budget=read(base/'restored/STORAGE_PROJECTION.json')
assert proof['status']=='passed'
assert budget['adjusted_measured_projection_bytes']<=15000000000 and budget['typed_upper_logical_bytes']<=15000000000 and budget['projected_peak_bytes']<=18000000000
# Windows workspace hardlinks may be restricted; the actual Kaggle gate must
# establish them. The offline copies do not claim that live staging passed.
report=dict(status='passed',scope='offline_CPU_fresh_process_storage_gate',source_inventory_sha256=digest(inventory),
            fixture=str(base),restore=proof,storage=budget,seconds=time.monotonic()-start,
            actual_Kaggle_hardlink_and_HF_gate_pending=True,remote_writes=0,measured_updates=0)
atomic(ROOT/'tread_two/geo_validation/RESUME.json',report)
print(json.dumps({k:v for k,v in report.items() if k!='storage'},indent=2))
