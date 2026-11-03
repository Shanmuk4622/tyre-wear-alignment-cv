"""Generate only the new self-contained measured-pilot notebook; preserve executed NB16."""
import ast,base64,hashlib,json,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'tread_two/geo_pilot';OUT=ROOT/'notebooks/phase3_NB17_TREAD_TWO_GEO_Resumable_Pilot.ipynb'
VALID=ROOT/'tread_two/geo_validation';VALID.mkdir(exist_ok=True)
def sha(body):return hashlib.sha256(body).hexdigest()
files={p.name:dict(sha256=sha(p.read_bytes()),bytes=p.stat().st_size) for p in SRC.iterdir() if p.suffix in ('.py','.json','.csv') and p.name!='FILES.json'}
for path in SRC.glob('*.py'):ast.parse(path.read_text(encoding='utf-8'),filename=path.name)
(SRC/'FILES.json').write_text(json.dumps(files,sort_keys=True,separators=(',',':')),encoding='utf-8')
payload={p.name:dict(base64=base64.b64encode(p.read_bytes()).decode(),sha256=sha(p.read_bytes())) for p in SRC.iterdir() if p.name in set(files)|{'FILES.json'}}
package=sha(json.dumps({k:v['sha256'] for k,v in payload.items()},sort_keys=True).encode())
cells=[]
def md(value):cells.append(dict(cell_type='markdown',metadata={},source=value.splitlines(True)))
def code(value):
    ast.parse(value);cells.append(dict(cell_type='code',metadata={},source=value.splitlines(True),execution_count=None,outputs=[]))
md('''# NB17 · TREAD-TWO / GEO — one resumable measured pilot

**NB16 is completed and independently HF-verified. Do not rerun it.**
This notebook first checks its new streamed storage/CSV lifecycle and an actual
HF-restored disposable continuation, then runs **one seed / 60 epochs / 2,940
optimizer updates**. No mileage-classification loss is fitted. All photos already
exist; nothing is collected or downloaded from another image source.

| Kaggle setting | Required value |
|---|---|
| Accelerator | **GPU T4 ×2** |
| Internet | **ON** |
| Input | Expanded **shanmuk4622/tire-dataset-prepared-phase2** |
| Secret | Enable **HF_TOKEN** with write access to **Shanmuk4622/tyre-wear-study** |
| Working space | Fresh start: **at least19 GB free**; resume keeps/uses existing compatible files |
| Numeric environment | Same verified NB16 Python/Torch/NumPy/Pillow/CUDA versions; no global package replacements |
| Action | Fresh kernel, **Run All**, in order |

GPU1 runs the frozen selected TREAD-ONE anchor and checkpoint-matched evaluation.
GPU0 trains the isolated geometry, texture and experimental error modules:
**355,879 trainable / 1,728,038 total parameters**. The new gate is disposable,
training-role only and never a selected measured model. Any failed math, storage,
ownership or restore gate stops before measured fitting.
''')
code('''# 1 · Private tested HF transport + exact self-contained source extraction.
import base64,hashlib,json,os,subprocess,sys
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
private=Path('/kaggle/working/geo_private_hf_0362')
loaded=sys.modules.get('huggingface_hub')
if loaded is not None and loaded.__version__!='0.36.2':
    raise RuntimeError('Restart the kernel and run NB17 first: a different HF SDK is already loaded.')
if not (private/'huggingface_hub-0.36.2.dist-info').is_dir():
    subprocess.check_call([sys.executable,'-m','pip','install','--disable-pip-version-check','--no-deps',
                           '--target',str(private),'huggingface_hub==0.36.2'])
sys.path.insert(0,str(private))
import huggingface_hub
assert huggingface_hub.__version__=='0.36.2'
PAYLOAD='''+repr(payload)+'''
source=Path('/kaggle/working/geo17_source_'''+package[:12]+'''')
source.mkdir(parents=True,exist_ok=True)
for name,item in PAYLOAD.items():
    assert Path(name).name==name
    body=base64.b64decode(item['base64'],validate=True)
    assert hashlib.sha256(body).hexdigest()==item['sha256']
    path=source/name
    if path.exists() and path.read_bytes()!=body:
        raise RuntimeError('Frozen extracted source changed: '+name+'. Keep existing progress; do not bypass its hash.')
    path.write_bytes(body)
if 'common' in sys.modules and Path(sys.modules['common'].__file__).resolve().parent!=source.resolve():
    raise RuntimeError('Restart the kernel: another experiment runtime is already imported.')
sys.path.insert(0,str(source))
print('Verified',len(PAYLOAD),'source/config files. Kaggle Torch/CUDA remain unchanged.')
''')
md('''## 2 · Check, restore, train, validate and publish

Run this cell once. On a compatible restart it checks HF and the local unit
chain, restores the newest compatible complete state, publishes local-ahead work
before another update, and continues the next draw. A completed remote run exits
without another optimizer step. A stable experiment registry prevents a changed
source/environment from silently starting a second GEO r1 fit.

The first run performs a new disposable two-epoch/two-batch integration gate,
publishes its checkpoint after optimizer update1 within the epoch, restores it
from HF in a fresh worker, and compares model/moments/RNG/cursor and scientific
shards. It measures all40 CSV families' capacity forecast and actual same-
filesystem hardlinks before the measured run. Existing matching live gate
certificates are reused. The measured run begins only if these checks pass.

Normal HF publication is every30 minutes, plus initialization/major completion
and catchable stop/error. Worker and transfer heartbeats identify the current
phase at least every30 seconds. API/resolver quotas use real server headers;
the separate96 commit-attempts/hour guard remains. Keep the notebook open during
visible server waits.

**Stop:** a catchable interrupt asks the worker for a complete-unit boundary,
then immediately attempts a consistent HF flush. Forced kernel/session death
cannot upload unsent bytes. Preserve the working output if possible; rerun this
unchanged notebook to recover from compatible local state or verified HF state.
''')
code('''# 2 · The numeric worker never receives HF_TOKEN.
from kaggle_secrets import UserSecretsClient
from coordinator_fit import run
token=UserSecretsClient().get_secret('HF_TOKEN')
try:
    result=run(token)
finally:
    token=None
''')
md('''## 3 · Result, selected/final exports and interactive evidence

All60 epoch weights and complete rolling optimizer/RNG state are retained.
Validation preserves geometry floors and rejects collapsed representations before
ranking masked loss relative to the training-only mean predictor. If no epoch
qualifies, **selected is unavailable**; final weights and every rejection remain.
The historical test cohort never selects a checkpoint.

The full executable contract has40 CSV families with pre-fit corrections and
additional sufficient-statistic/provenance fields. Raw training/validation/
endpoint logits, tokens, masked predictions/targets/weights and row parameters
stay directly native, with CSV metrics/vectors/points/boxes/gradients/exposure.
No ZIP/NPZ/database envelope. Selected/final inference covers all570 images with
original roles and source/component-excluded training-reference diagnostics.
Ten predeclared synthetic/structural historical-cohort probes are marked as
diagnostics; their compact-native retention policy is explicit in the contract.

Open the standalone Explorer links for cohort filters and actual prediction
overlays. Error estimates remain experimental annotation-agreement values,
not calibrated safety, physical depth, damage labels or wear measurements.
''')
code('''from IPython.display import FileLink,display
from pathlib import Path
import json
base=Path('/kaggle/working/tread_two_geo_nb17')
print(json.dumps({k:v for k,v in result.items() if k!='live_gate'},indent=2))
for path in [base/'RUN_RESULT.json',base/'LIVE_GATE.json',base/'LIVE_GATE_PUBLICATION.json',
             base/'LAST_HF_PUBLICATION.json',base/'fit/run/RESULT.json']:
    if path.exists():display(FileLink(str(path)))
reports=base/'fit/run/reports'
if reports.exists():
    for path in sorted(reports.glob('*Explorer.html')):display(FileLink(str(path)))
    for path in sorted(reports.glob('*metrics.json')):display(FileLink(str(path)))
exports=base/'fit/run/exports'
if exports.exists():
    for path in sorted(exports.glob('*.pt')):display(FileLink(str(path)))
''')
md('''## Troubleshooting and next action

| Message | Action |
|---|---|
| Dataset/secret/T4 check | Correct the settings, then rerun unchanged |
| Different numerical environment | Preserve any existing fit; do not change its hashes or silently restart |
| Source/registry/history conflict | Keep files and exact notebook; do not create a second writer or bypass the check |
| Insufficient space / hardlinks unavailable | The gate stops before measured updates; retain its report |
| Budget wait / HTTP429 | Leave the session open; respect the visible server delay |
| Catchable stop/error | Keep output and rerun unchanged; inspect HF_PENDING.json / WORKER_ERROR.json if present |
| Forced death with lost local output | Restore the latest verified HF snapshot; unsent bytes are not guaranteed |
| No selected checkpoint | This is a retained scientific rejection, not permission to pick the best test epoch |

HF measured namespace: **phase3/tread-two/geo-r1/fit/&lt;contract&gt;**.
The disposable integration gate has its own **pilot-gate** namespace. Earlier
NB16/ATLAS/TREAD-ONE publications are preserved. Both workstations remain deferred.

After completion, save the executed NB17 with outputs. We will independently
verify its HF bytes/CSV/native history and original selection, update the
documents, then create NB18 for the completed-run evidence review. No automatic
second seed, new photography or production promotion follows this pilot.
''')
for index,cell in enumerate(cells):cell['id']='geo17-'+str(index)
nb=dict(cells=cells,nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'),language_info=dict(name='python',version='3.13'),kaggle=dict(isGpuEnabled=True,isInternetEnabled=True),geo=dict(package_sha256=package,mode='one_measured_candidate_after_automatic_live_gates')))
if OUT.exists():
    old=json.loads(OUT.read_text(encoding='utf-8'));assert not any(c.get('outputs') or c.get('execution_count') is not None for c in old['cells'] if c['cell_type']=='code'),'Never overwrite an executed notebook'
OUT.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
receipt=dict(status='built_unexecuted',notebook=str(OUT),sha256=sha(OUT.read_bytes()),bytes=OUT.stat().st_size,package_sha256=package,embedded_files=len(payload),source_inventory_sha256=sha((SRC/'FILES.json').read_bytes()),measured_run_not_performed_locally=True)
(VALID/'NOTEBOOK.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt,indent=2))
