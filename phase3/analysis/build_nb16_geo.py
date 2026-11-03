"""Self-contained Kaggle engineering notebook. Never regenerates a completed notebook."""
import ast
import base64
import hashlib
import json
import sys
from pathlib import Path

assert Path(sys.prefix).name == 'cv_conda'
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'tread_two/geometry_first/runtime'
OUT = ROOT/'notebooks/phase3_NB16_TREAD_TWO_GEO_Preparation_And_Contract_Gate.ipynb'
VALIDATION = ROOT/'tread_two/geometry_first/validation'
VALIDATION.mkdir(exist_ok=True)
def digest(body): return hashlib.sha256(body).hexdigest()
inventory = {p.name:dict(sha256=digest(p.read_bytes()),bytes=p.stat().st_size)
             for p in SOURCE.iterdir() if p.is_file() and p.name!='FILES.json'}
for path in SOURCE.glob('*.py'):
    ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
(SOURCE/'FILES.json').write_text(json.dumps(inventory, sort_keys=True, separators=(',', ':')), encoding='utf-8')
payload = {p.name:dict(base64=base64.b64encode(p.read_bytes()).decode(),sha256=digest(p.read_bytes()))
           for p in SOURCE.iterdir() if p.is_file()}
package = digest(json.dumps({k:v['sha256'] for k,v in payload.items()},sort_keys=True).encode())
cells=[]
def md(value): cells.append(dict(cell_type='markdown',metadata={},source=value.splitlines(True)))
def code(value):
    ast.parse(value)
    cells.append(dict(cell_type='code',metadata={},source=value.splitlines(True),outputs=[],execution_count=None))
md('''# NB16 · TREAD-TWO / GEO preparation and contract gate

**Run this notebook first. This is engineering preparation, not the 60-epoch fit.**
It tests the real geometry/texture/error model using existing training images,
then publishes direct CSV/native evidence and actually restores a published
checkpoint in a fresh worker. It never edits either workstation or earlier runs.

| Kaggle setting | Required value |
|---|---|
| Accelerator | **GPU T4 ×2** |
| Internet | **ON** |
| Input | Add the existing **shanmuk4622/tire-dataset-prepared-phase2** expanded dataset |
| Secret | Enable **HF_TOKEN** with write access to **Shanmuk4622/tyre-wear-study** |
| Space | Start with at least **4 GB free** in working output; retain stopped-session files |
| Action | **Run All**, in order; no helper pasting or old session variables |

GPU1 runs the immutable selected TREAD-ONE anchor one image at a time. GPU0 runs
the small new modules at batch8. All tests use training-role images only; no
measured fit or held-out checkpoint selection takes place. The finite checks
compare four uninterrupted updates, a local restored path and a true HF-restored
continuation. Disposable checks are never eligible model checkpoints.

The approved next notebooks are NB17 (one resumable fit) and NB18 (review).
They follow actual NB16 evidence and completion of the measured-run CSV/shard
integration; this preparation notebook does not claim those remaining gates pass.
''')
code('''# 1. Install only the tested private HF transport; preserve Kaggle Torch/CUDA.
import base64, hashlib, json, os, subprocess, sys
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
private = Path('/kaggle/working/geo_private_hf_0362')
loaded = sys.modules.get('huggingface_hub')
if loaded is not None and loaded.__version__ != '0.36.2':
    raise RuntimeError('Restart the Kaggle kernel and run NB16 first; a different HF SDK is already loaded.')
if not (private/'huggingface_hub-0.36.2.dist-info').is_dir():
    subprocess.check_call([sys.executable,'-m','pip','install','--disable-pip-version-check',
                           '--no-deps','--target',str(private),'huggingface_hub==0.36.2'])
sys.path.insert(0,str(private))
import huggingface_hub
assert huggingface_hub.__version__=='0.36.2'
PAYLOAD = '''+repr(payload)+'''
source = Path('/kaggle/working/geo_source_'''+package[:12]+'''')
source.mkdir(parents=True,exist_ok=True)
for name,item in PAYLOAD.items():
    assert Path(name).name==name
    data=base64.b64decode(item['base64'],validate=True)
    assert hashlib.sha256(data).hexdigest()==item['sha256']
    path=source/name
    if path.exists() and path.read_bytes()!=data:
        raise RuntimeError('Extracted GEO source changed: '+name+'. Preserve it and restart from the unchanged notebook.')
    path.write_bytes(data)
if 'common' in sys.modules and Path(sys.modules['common'].__file__).resolve().parent!=source.resolve():
    raise RuntimeError('Restart the kernel: another experiment runtime is already imported.')
sys.path.insert(0,str(source))
print('Verified',len(PAYLOAD),'embedded source/config files. Private HF SDK',huggingface_hub.__version__)
''')
md('''## 2 · Run the actual gate

Expect visible phases for checksum verification, each engineering update,
CSV/native saving, HF publication and downloaded-state continuation. A heartbeat
appears at least every30 seconds while the coordinator is waiting. HF quota waits
may legitimately be longer; keep the session and its files.

Catchable **Stop** asks the separate worker to finish its current complete update,
then immediately attempts publication of its last committed state. A forced
kernel/session kill cannot upload unsent bytes. Restart with the same notebook;
compatible local state is verified and reused. In a fresh empty session the
disposable preparation checks can repeat; they are not the measured experiment.
''')
code('''# 2. HF_TOKEN is read only from the authorized Kaggle Secret, never printed.
from kaggle_secrets import UserSecretsClient
from coordinator_geo import run
token = UserSecretsClient().get_secret('HF_TOKEN')
try:
    result = run(token)
finally:
    token = None
''')
md('''## 3 · Read the receipt and interactive engineering gallery

Success means **engineering_gate_passed** with actual dual-T4 and HF-restored
continuation proofs. **measured_fit_ready stays false**: the full 40-family fit
telemetry/shard/staging integration still needs its own acceptance before NB17.
The passport shows predicted geometry for disposable training examples. It is not
a physical wear, depth, safety or improved-accuracy result.
''')
code('''from IPython.display import HTML, FileLink, display
from pathlib import Path
import json
output=Path('/kaggle/working/tread_two_geo_nb16')
receipt=json.loads((output/'RUN_RESULT.json').read_text())
print(json.dumps(receipt,indent=2))
display(HTML((output/'Engineering_Passport.html').read_text()))
for name in ('RUN_RESULT.json','ENGINEERING_PROOF.json','HF_CONTINUATION_PROOF.json',
             'STORAGE.json','PARAMETERS.json','Engineering_Passport.html'):
    display(FileLink(str(output/name)))
''')
md('''## Stop, recovery and troubleshooting

| Symptom | Action |
|---|---|
| GPU/input/secret check fails | Correct the Kaggle settings above, then rerun unchanged |
| Source or numerical-environment mismatch | Preserve output; do not edit hashes to bypass the check |
| A prior HF owner is active | Stop the other copy cleanly or wait for the displayed lease; never run concurrent writers |
| HTTP429 / budget wait | Leave the session open; server-directed waiting preserves local state |
| Catchable stop/error | Keep output and rerun unchanged; inspect HF_PENDING.json if transfer failed |
| Forced session death | Retain saved Kaggle output if available; unsent bytes cannot be guaranteed |
| No geometry support / deterministic-op failure | Keep the saved transcript; do not disable determinism or change the model mid-check |
| Complete rerun | Shows the existing compatible receipt without new optimizer steps or HF writes |

HF records live under **phase3/tread-two/geo-r1/engineering/&lt;contract&gt;** in
**Shanmuk4622/tyre-wear-study**, using directly browsable CSV, JSON, HTML, NPY and
native PT. No ZIP/NPZ/database envelope. The coordinator alone receives the token.

After completion, save this executed notebook with outputs and return its result.
We will independently verify the HF revision, inspect the measured storage and
device evidence, update the documents, and finish/freeze NB17. Existing photos only.
''')
nb=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
                language_info=dict(name='python',version='3.13'),
                kaggle=dict(isGpuEnabled=True,isInternetEnabled=True),
                geo=dict(package_sha256=package,scope='engineering_only',measured_training=False)),nbformat=4,nbformat_minor=5)
for i,cell in enumerate(cells): cell['id']='geo16-'+str(i)
if OUT.exists():
    old=json.loads(OUT.read_text(encoding='utf-8'))
    assert not any(c.get('outputs') or c.get('execution_count') is not None for c in old['cells'] if c['cell_type']=='code'), 'Never overwrite an executed notebook'
OUT.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
receipt=dict(status='built_unexecuted',notebook=str(OUT),sha256=digest(OUT.read_bytes()),bytes=OUT.stat().st_size,
             package_sha256=package,embedded_files=len(payload),code_cells=sum(c['cell_type']=='code' for c in cells),
             measured_fit_notebook_delivered=False)
(VALIDATION/'NOTEBOOK.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,indent=2))
