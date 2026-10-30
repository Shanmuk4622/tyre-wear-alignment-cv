"""Self-contained dual-T4 frozen-model gate, direct CSV/native/HF recovery."""
import ast,base64,hashlib,json,sys
from pathlib import Path
import nbformat
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
pilot=ROOT/'tread_two/pilot';gate=ROOT/'tread_two/appearance_gate'
inventory=json.loads((pilot/'FILES.json').read_text())
files={'frozen_source/'+name:(pilot/name).read_bytes() for name in list(inventory)+['FILES.json']}
files.update({'appearance_gate/'+p.name:p.read_bytes() for p in gate.iterdir() if p.suffix=='.py' or p.name in ('RULES.json','UPSTREAM.json')})
files.update({'review/'+p.name:p.read_bytes() for p in (ROOT/'tread_two/review').glob('*.py')})
files['transport/nb08_transport_launcher.py']=(ROOT/'analysis/nb08_transport_launcher.py').read_bytes()
files['FIT_MANIFEST.json']=(ROOT/'tread_two/pilot_validation/nb08_followup/82e7e1c5774f/fit_MANIFEST.json').read_bytes()
for name,body in files.items():
    if name.endswith('.py'): ast.parse(body.decode('utf-8'))
for name,meta in inventory.items(): assert hashlib.sha256(files['frozen_source/'+name]).hexdigest()==meta['sha256']
encoded=base64.b64encode(json.dumps({n:base64.b64encode(b).decode() for n,b in files.items()},separators=(',',':')).encode()).decode()
setup='''import base64,hashlib,importlib.metadata,json,os,signal,subprocess,sys,time,shutil
from pathlib import Path
from filelock import FileLock
assert Path('/kaggle/working').exists(),'Run NB10 on Kaggle.'
WORK=Path('/kaggle/working/tread_two_nb10_appearance_gate')
WORK.mkdir(parents=True,exist_ok=True)
assert shutil.disk_usage(WORK).free>=4*1024**3,'Keep at least 4 GiB free for native/CSV gate recovery and staging.'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
try: hub=importlib.metadata.version('huggingface_hub')
except importlib.metadata.PackageNotFoundError: hub=None
if hub!='0.36.2': subprocess.check_call([sys.executable,'-m','pip','install','--disable-pip-version-check','huggingface_hub==0.36.2'])
payload=json.loads(base64.b64decode('__PAYLOAD__'))
with FileLock(str(WORK/'NOTEBOOK.lock'),timeout=1):
    for name,value in payload.items():
        target=(WORK/name).resolve();assert target.is_relative_to(WORK.resolve()) and '..' not in Path(name).parts
        body=base64.b64decode(value)
        if target.exists(): assert target.read_bytes()==body,'Preserve existing gate sources/progress for compatibility review.'
        else:
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(body)
import torch
assert torch.cuda.device_count()==2 and all('T4' in torch.cuda.get_device_name(i) for i in range(2)),'Select GPU T4 x2 for this gate.'
RUNTIME=WORK/'runtime'
arguments=['--source',str(WORK/'frozen_source'),'--review-source',str(WORK/'review'),'--manifest',str(WORK/'FIT_MANIFEST.json'),'--launcher',str(WORK/'transport/nb08_transport_launcher.py'),'--work',str(RUNTIME)]
def launch(stage,token=None):
    environment=dict(os.environ);environment.pop('HF_TOKEN',None)
    if token is not None: environment['HF_TOKEN']=token
    with FileLock(str(WORK/'NOTEBOOK.lock'),timeout=1):
        child=subprocess.Popen([sys.executable,'-B','-u',str(WORK/'appearance_gate/gate_controller.py'),stage,*arguments],env=environment,start_new_session=True)
        environment.pop('HF_TOKEN',None)
        try:
            while True:
                try: status=child.wait(timeout=15);break
                except subprocess.TimeoutExpired: pass
        except KeyboardInterrupt:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGINT)
                for i in range(180):
                    try: child.wait(timeout=5);break
                    except subprocess.TimeoutExpired:
                        if i%6==0: print('Waiting for image commit / immediate HF flush attempt. Keep this session/files.',flush=True)
            raise
        assert status==0,'Gate paused/stopped. Preserve runtime files and worker_console.log; rerun the same notebook unchanged to recover the next complete image.'
print('Exact gate sources ready. GPU0: frozen dense anchor; GPU1: frozen patch head. No fitting, DDP or extra training seed.')
'''.replace('__PAYLOAD__',encoded)
cells=[]
def md(s): cells.append(nbformat.v4.new_markdown_cell(s))
def py(s): ast.parse(s);cells.append(nbformat.v4.new_code_cell(s))
md('''# Phase 3 · NB10 · TREAD-TWO train-only appearance/reference gate

**386 existing training images · frozen selected v1 · seven probes · zero optimizer updates.**

NB09 is complete and HF-verified. This next gate examines appearance stability,
a fixed local-contrast feature proposal and a reference bank drawn from more
distinct existing training photos. It never queries validation/test pixels.
The candidate bank is **UNTRAINED** and is not installed in NB08 or a workstation.

## Kaggle settings and order

1. Select **GPU T4 ×2**, **Internet ON**.
2. Attach exactly one **expanded** `shanmuk4622/tire-dataset-prepared-phase2`
   dataset directly from Kaggle. It stays read-only; no new photos are needed.
3. Keep at least **4 GiB free** for native atlas/CSV state and publication staging.
   The total output guard is 16 GB within the user's 20 GB working-output budget.
4. Enable Kaggle Secret **HF_TOKEN** with write access to
   `Shanmuk4622/tyre-wear-study` for the execution/publication cell.
5. **Run All.** The tested Hub client is pinned; Torch/CUDA are not changed.

GPU0 performs the frozen original-frame dense pass once per training image.
GPU1 runs the frozen selected epoch18 patch classifier and feature diagnostics.
Both remain eval/FP32 with gradients disabled. This is a probe, not another fit.
The one-GPU local recovery check is distinct from actual Kaggle dual-T4 execution.
''')
py(setup)
md('''## 1 · Prepare the exact input/weights/contract

The historical role overlay, source inventory, selected native checkpoint and
anchor hashes are verified. Only role=train enters the worker. All 386 images
include the original 378-query pool plus eight original support photos, which
are reported separately. The contract includes actual source/environment and
device identities; incompatible existing state is never silently overwritten.
''')
py("launch('prepare')")
md('''## 2 · Run/resume the gate and publish complete direct evidence

Every completed image atomically commits native original RGB/valid/grid and
dense outputs with all seven probe score, affinity, patch and feature CSV rows.
An interrupted incomplete image is replayed; the next completed-image cursor
and immutable native/CSV hashes determine recovery, not an epoch counter.

Seven fixed probes: identity, brightness ×0.8/×1.2, grayscale both query/bank,
query-only grayscale and gamma ×0.8/×1.2. The local-contrast proposal has masked
9×9 mean/std, std floor .02, clipped normalized luminance, Sobel x/y and gradient
magnitude. These are texture channels, not depth. Input clipping and flat-region
floors limit exact equivariance; subset counts are retained.

The proposed bank is six cases per class, one per distinct original training
photo. Least-represented available tyres are considered first, then fixed
descriptor diversity/coverage/glare. At least one original query photo per
known training tyre must remain. Low texture is flagged, not discarded as a
presumed wear label. All rules were fixed before this gate; held-out scores
never choose bank entries, thresholds or features.

Normal HF snapshots occur every30 minutes, at major completion, and on catchable
Stop/error. The publisher first checks remote state, verifies immutable objects
and releases ownership. Server quota/backoff can delay a Stop flush. Forced
kernel death can recover only the last verified remote boundary; keep local
files when possible. Do not delete leases or run duplicate writers.
''')
py("from kaggle_secrets import UserSecretsClient\ntoken=UserSecretsClient().get_secret('HF_TOKEN')\ntry:\n    launch('run',token=token)\nfinally:\n    token=None")
md('''## 3 · Inspect all training-only records and the untrained bank proposal

Source PNG/JPEG previews and fixed feature views are illustrative derivatives.
Original float RGB, masks, grid, descriptors and hashes remain direct native
artifacts; all scalar measurements remain CSV. The interactive report covers
all 386 training images and separates support photos from v1 queries.

The new namespace is `phase3/tread-two/atlas-v1/appearance-gate/<contract>`.
All data remain direct CSV/PT/NPY/JSON/source/PNG/HTML, never ZIP/NPZ/database.
Directory names `epoch_0000…` come from the reusable store and mean blocks of50
images here, not training epochs. Candidate pixel arrays are proposals for a
future versioned recipe. More views do not increase independent tyre support
beyond the original 1/3/4 training tyres per class; no v2 classifier is measured.
''')
py("from IPython.display import display,FileLink,Image\nimport pandas as pd\nlatest=json.loads((RUNTIME/'LAST_SNAPSHOT.json').read_text())\nSNAPSHOT=Path(latest['path'])\nprint(json.dumps(latest['receipt'],indent=2))\nprint((SNAPSHOT/'report/GATE_RESULT.json').read_text())\ndisplay(pd.read_csv(SNAPSHOT/'report/train_probe_metrics.csv'))\ndisplay(Image(filename=str(SNAPSHOT/'report/candidate_bank.png')))\ndisplay(FileLink(str(SNAPSHOT/'report/Training_Gate_Explorer.html')))\nprint('Keep this executed NB10 for independent HF verification. No model has been trained or installed.')")
for i,cell in enumerate(cells): cell.id=f'atlas-nb10-{i:02d}'
nb=nbformat.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'),kaggle=dict(isInternetEnabled=True,isGpuEnabled=True,datasetSources=['shanmuk4622/tire-dataset-prepared-phase2'])))
nbformat.validate(nb)
path=ROOT/'notebooks/phase3_NB10_TREAD_TWO_Train_Only_Appearance_And_Reference_Gate.ipynb'
if path.exists(): assert not any(c.get('execution_count') or c.get('outputs') for c in nbformat.read(path,as_version=4).cells if c.cell_type=='code'),'Preserve executed NB10 before regeneration'
nbformat.write(nb,path)
out=ROOT/'tread_two/gate_validation';out.mkdir(exist_ok=True)
receipt=dict(status='packaged_not_executed_on_Kaggle',notebook=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,code_cells=4,embedded_files=len(files),sources={n:hashlib.sha256(v).hexdigest() for n,v in files.items()},optimizer_updates=0,HF_gate_publication_performed_locally=False)
(out/'NOTEBOOK.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in receipt.items() if k!='sources'},indent=2))
