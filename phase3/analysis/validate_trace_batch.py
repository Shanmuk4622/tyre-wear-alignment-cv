"""Actual-data CPU lifecycle, complete-update faults, CSV/native audit and payload checks."""
import ast,base64,json,os,shutil,subprocess,sys,time
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'tread_two/trace_batch';VALID=ROOT/'tread_two/trace_validation';VALID.mkdir(exist_ok=True)
sys.path.insert(0,str(SRC))
from common import atomic,read,digest,file_meta,sha,seed_all
from cache import prepare
from config import configuration
import torch
torch.set_num_threads(4)
fixture=VALID/('fixture_'+str(time.time_ns()));fixture.mkdir()
dataset=ROOT.parent/'phase2/data/phase2_dataset_v1'
folder=Path(read(ROOT/'tread_two/nb17_completion/LATEST.json')['path']);geometry=folder/'remote/exports/final.pt'
cfg=configuration();cache=prepare(fixture/'cache',dataset,geometry,digest(cfg),'cpu',fixture=True)
env=dict(os.environ);env.pop('HF_TOKEN',None)
direct=fixture/'direct'
args=[sys.executable,'-u','-B',str(SRC/'worker.py'),'gate_direct','--output',str(direct),'--cache',str(cache.root),'--offline','--no-ack']
assert subprocess.run(args,env=env).returncode==0,'Direct lifecycle failed'
recovered=fixture/'recovered';shutil.copytree(direct/'seam',recovered/'run')
args=[sys.executable,'-u','-B',str(SRC/'worker.py'),'gate_restore','--output',str(recovered),'--cache',str(cache.root),'--reference',str(direct),'--offline','--no-ack']
assert subprocess.run(args,env=env).returncode==0,'Restored lifecycle failed'
assert read(recovered/'RESTORE_PROOF.json')['status']=='passed'
from review import run
proof=run(direct/'run',fixture/'review',digest(configuration(True)),'offline_fixture_not_HF')
assert proof['status']=='completed'
again=run(direct/'run',fixture/'review',digest(configuration(True)),'offline_fixture_not_HF');assert again==proof
import nbformat
notebooks=[]
for item in read(VALID/'NOTEBOOKS.json'):
    path=ROOT/'notebooks'/item['notebook'];nb=read(path);nbformat.validate(nb);assert sha(path)==item['sha256']
    cells=[c for c in nb['cells'] if c['cell_type']=='code'];assert len(cells)==3 and all(c['execution_count'] is None and not c['outputs'] for c in cells)
    for cell in cells:ast.parse(''.join(cell['source']))
    tree=ast.parse(''.join(cells[0]['source']));payload=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PAYLOAD' for t in n.targets))
    for name,body in payload.items():assert __import__('hashlib').sha256(base64.b64decode(body['base64'])).hexdigest()==sha(SRC/name)==body['sha256']
    notebooks.append(dict(name=path.name,payload_files=len(payload),schema_valid=True))
atomic(VALID/'LOCAL_LIFECYCLE.json',dict(status='passed',fixture=str(fixture),real_existing_train_images=8,full_lifecycle=True,CPU_complete_update_restore=True,review=proof,
       source_inventory=read(SRC/'FILES.json'),notebooks=notebooks,actual_dual_T4=False,actual_HF_restore=False,measured_training_started=False,HF_writes=0,credentials_read=False))
print(json.dumps(dict(status='passed',fixture=str(fixture),storage_projection=read(direct/'STORAGE_PROJECTION.json'),notebooks=notebooks),indent=2),flush=True)
