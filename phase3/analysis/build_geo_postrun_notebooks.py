"""Deliver all justified postrun stages plus one automatic Run All master."""
import ast,base64,hashlib,json,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'tread_two/geo_postrun';VALID=ROOT/'tread_two/geo_postrun_validation';VALID.mkdir(exist_ok=True)
def sha(data):return hashlib.sha256(data).hexdigest()
files={p.name:dict(sha256=sha(p.read_bytes()),bytes=p.stat().st_size) for p in SRC.iterdir() if p.suffix in ('.py','.json') and p.name!='FILES.json'}
for p in SRC.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=p.name)
(SRC/'FILES.json').write_text(json.dumps(files,sort_keys=True,separators=(',',':')),encoding='utf-8')
payload={p.name:dict(base64=base64.b64encode(p.read_bytes()).decode(),sha256=sha(p.read_bytes())) for p in SRC.iterdir() if p.name in set(files)|{'FILES.json'}}
package=sha(json.dumps({k:v['sha256'] for k,v in payload.items()},sort_keys=True).encode())
setup='''# 1 · Private tested HF transport and exact source extraction.
import base64,hashlib,json,os,subprocess,sys
from pathlib import Path
private=Path('/kaggle/working/geo_private_hf_0362')
loaded=sys.modules.get('huggingface_hub')
if loaded is not None and loaded.__version__!='0.36.2':
    raise RuntimeError('Restart the kernel: another HF SDK is already imported. Keep all saved files.')
if not (private/'huggingface_hub-0.36.2.dist-info').is_dir():
    subprocess.check_call([sys.executable,'-m','pip','install','--disable-pip-version-check','--no-deps','--target',str(private),'huggingface_hub==0.36.2'])
sys.path.insert(0,str(private))
import huggingface_hub
assert huggingface_hub.__version__=='0.36.2'
PAYLOAD='''+repr(payload)+'''
source=Path('/kaggle/working/geo_postrun_source_'''+package[:12]+'''')
source.mkdir(parents=True,exist_ok=True)
for name,item in PAYLOAD.items():
    assert Path(name).name==name
    body=base64.b64decode(item['base64'],validate=True)
    assert hashlib.sha256(body).hexdigest()==item['sha256']
    path=source/name
    if path.exists() and path.read_bytes()!=body:raise RuntimeError('Extracted source changed: '+name+'. Preserve saved files and use the original notebook.')
    path.write_bytes(body)
if 'common' in sys.modules and Path(sys.modules['common'].__file__).resolve().parent!=source.resolve():
    raise RuntimeError('Restart the kernel: a previous experiment runtime is imported. Keep existing files.')
sys.path.insert(0,str(source))
print('Verified',len(PAYLOAD),'source/config files. CPU saved-data review; Torch/CUDA unchanged.')
'''
entries=[(18,'review','phase3_NB18_TREAD_TWO_GEO_Completed_Review.ipynb','Completed evidence review'),(19,'passport','phase3_NB19_TREAD_TWO_GEO_Evidence_Passport_And_Research_Handoff.ipynb','Evidence passport and research handoff'),(20,'all','phase3_NB20_TREAD_TWO_GEO_All_In_One_Postrun.ipynb','All-in-one postrun batch — recommended')]
receipts=[]
for number,mode,name,title in entries:
    cells=[]
    def md(value):cells.append(dict(cell_type='markdown',metadata={},source=value.splitlines(True)))
    def code(value):ast.parse(value);cells.append(dict(cell_type='code',metadata={},source=value.splitlines(True),execution_count=None,outputs=[]))
    md(f'''# NB{number} · TREAD-TWO / GEO — {title}

**NB17 finished: 60 epochs / 2,940 updates. Do not retrain it.**
No original validation checkpoint passed the representation-rank gate. That is
a scientific result, not an execution error. This batch verifies all saved
evidence and closes the pilot as unpromoted research.

**Recommended: run NB20 once with Run All.** It completes the NB18 audit,
publishes it, then completes/publishes NB19 automatically. NB18 and NB19 are
also supplied for separate use; NB19 automatically satisfies its NB18 dependency.
Do not launch these notebooks in parallel against the same namespace.

| Kaggle setting | Value |
|---|---|
| Accelerator | **None / CPU**; GPU time is unnecessary |
| Internet | **ON** |
| Input | Existing expanded **shanmuk4622/tire-dataset-prepared-phase2** |
| Secret | Enable **HF_TOKEN**, write access to **Shanmuk4622/tyre-wear-study** |
| Fresh working space | **At least 13 GB free**; pinned source is 10.31 GB, shared once |
| RAM | About **8 GB** recommended; tables stream in bounded chunks |
| Run | Fresh kernel → **Run All**; no intermediate choices |

Saved NB17 files in the same Kaggle session are reused only after exact hashes
match. A new session restores the pinned HF publication automatically. No image
copy/download is made outside the existing Kaggle attachment. No model forward,
new optimizer step, checkpoint reselection or extra fit is performed.

```mermaid
flowchart LR
  A[Completed NB17, pinned HF bytes] --> B[NB18: complete saved-evidence audit]
  B --> C[Verified direct CSV, native accounting and interactive atlas]
  C --> D[Publish NB18]
  D --> E[NB19: capability passport and research handoff]
  E --> F[Publish NB19 and finish]
```
''')
    code(setup)
    md('''## Run and recovery

The parent alone reads Kaggle Secret HF_TOKEN. A separate numeric worker has
no token. Publication uses a single owner and bounded server-aware retries.
Completed stage receipts, per-file hash checks and matching remote results
prevent repeat work. An interrupted CPU stage replays from its preceding completed
stage; this is not training resumption and never changes the completed model.

Publish after major boundaries, every 30 minutes while active, and immediately
attempt a flush on a catchable Stop/error. Respect Retry-After even during a stop.
A forced kernel/session death cannot upload unsent local files. Keep the
compatible output directory; rerun Run All to restore/skip verified work.
''')
    code('''# 2 · Run the whole requested stage/batch without manual handoffs.
from kaggle_secrets import UserSecretsClient
from coordinator import run
try:
    batch_result=run(UserSecretsClient().get_secret('HF_TOKEN'),mode='''+repr(mode)+''')
except BaseException:
    for log in Path('/kaggle/working/tread_two_geo_postrun').glob('*_WORKER.log'):
        print(log.name,':\\n',log.read_text(encoding='utf-8',errors='replace')[-6000:])
    raise
print(json.dumps(batch_result,indent=2))
''')
    code('''# 3 · Inspect/download the durable interactive reports and direct tables.
from IPython.display import display,HTML,FileLink,Image
base=Path('/kaggle/working/tread_two_geo_postrun')
for folder,page in [('review','GEO_Review_Atlas.html'),('passport','GEO_Evidence_Passport.html')]:
    path=base/folder/page
    if path.exists():
        display(FileLink(str(path)))
        # Full 570-image atlas remains a standalone artifact; avoid duplicating
        # all its embedded images into saved notebook output.
        if folder=='passport':display(HTML(path.read_text(encoding='utf-8')))
for path in sorted((base/'review/figures').glob('*.png')):display(Image(filename=str(path)))
for path in [base/'PUBLICATIONS.json',base/'review/RESULT.json',base/'passport/RESULT.json']:
    if path.exists():display(FileLink(str(path)))
print('Direct CSV files:',sum(1 for f in ['review','passport'] for _ in (base/f).glob('*.csv')))
print('No new training or photo collection. GEO remains research-only and unselected.')
''')
    md('''## Troubleshooting and evidence limits

| Situation | Action |
|---|---|
| Long download | The pinned input has 11,636 objects. Watch verified-file progress; keep the cache. |
| Rate-limit/backoff | Wait for the displayed server delay; do not restart a second writer. |
| Wrong/missing dataset | Attach the expanded prepared Phase 2 Kaggle input, keep files, rerun. |
| Another live owner | Stop the other notebook cleanly or wait for its 45-minute lease; never clear ownership manually. |
| CPU stage interrupted | Keep outputs and rerun Run All. Exact completed stages and verified downloads are skipped. |
| Technical audit mismatch | Stop at the gate, retain logs/cache, report the error; do not bypass hashes/tolerances. |
| No selected checkpoint | Expected completed scientific result. Final epoch remains an unselected research artifact. |

Outputs are direct CSV, JSON/Markdown, PNG/HTML and native NPY where appropriate.
There is no ZIP/NPZ envelope. All 60 epoch weights and detailed original telemetry
already remain on HF. This review adds mask-truth checks, objective reconstruction,
per-role latent spectra, calibration controls, probes, retrieval/source exclusions,
source exposure and capability limits. Richer telemetry does not create independent
tyres. The previously inspected test cohort cannot support a new blind evaluation.
''')
    for i,cell in enumerate(cells):cell['id']=f'geo{number}-{i:02d}'
    nb=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python'),kaggle=dict(accelerator='none',isInternetEnabled=True),geo_postrun=dict(mode=mode,package=package,files=len(payload))),nbformat=4,nbformat_minor=5)
    path=ROOT/'notebooks'/name
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'));assert not any(c.get('execution_count') is not None or c.get('outputs') for c in old['cells'] if c['cell_type']=='code'),'Never overwrite an executed delivery'
    path.write_text(json.dumps(nb,indent=1)+'\n',encoding='utf-8');receipts.append(dict(notebook=name,mode=mode,bytes=path.stat().st_size,sha256=sha(path.read_bytes()),payload_files=len(payload),package=package))
(VALID/'NOTEBOOKS.json').write_text(json.dumps(receipts,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipts,indent=2))
