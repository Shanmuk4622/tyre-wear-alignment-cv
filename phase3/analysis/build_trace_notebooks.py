"""Build all TRACE notebooks together from new editable, unmeasured source."""
import ast,base64,hashlib,json,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'tread_two/trace_batch';VALID=ROOT/'tread_two/trace_validation';VALID.mkdir(exist_ok=True)
def sha(body):return hashlib.sha256(body).hexdigest()
schema=json.loads((SRC/'SCHEMAS.json').read_text(encoding='utf-8'))
for name,f in schema['families'].items():f['writer_status']='implemented; exercised writer coverage required by delivery checks'
columns=schema['families']['sample_observations']['columns']
if 'parameter_update' not in [c['name'] for c in columns]:columns.append(dict(name='parameter_update',dtype='int64',nullable=False,definition='Forward weights update: completed-1 during training; explicit selected endpoint update otherwise'))
if 'paired_cosine' not in [c['name'] for c in columns]:columns.append(dict(name='paired_cosine',dtype='float64',nullable=True,definition='Original FP32 paired cosine for exact selection; independently checked from native vectors'))
schema['status']='executable_training_contract';schema['field_declarations']=sum(len(f['columns']) for f in schema['families'].values())
(SRC/'SCHEMAS.json').write_text(json.dumps(schema,indent=2)+'\n',encoding='utf-8')
files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for p in SRC.iterdir() if p.suffix in ('.py','.json','.csv') and p.name!='FILES.json'}
for path in SRC.glob('*.py'):ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
(SRC/'FILES.json').write_text(json.dumps(files,sort_keys=True,separators=(',',':')),encoding='utf-8')
payload={n:dict(base64=base64.b64encode((SRC/n).read_bytes()).decode(),sha256=sha((SRC/n).read_bytes())) for n in [*files,'FILES.json']}
package=sha(json.dumps({k:v['sha256'] for k,v in payload.items()},sort_keys=True).encode())
setup='''# 1 · Verify exact embedded source; install only private tested HF transport.
import base64,hashlib,json,os,subprocess,sys
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
private=Path('/kaggle/working/trace_private_hf_0362')
loaded=sys.modules.get('huggingface_hub')
if loaded is not None and loaded.__version__!='0.36.2':
    raise RuntimeError('Restart the kernel and Run All; preserve saved files. Another HF SDK is already imported.')
if not (private/'huggingface_hub-0.36.2.dist-info').is_dir():
    subprocess.check_call([sys.executable,'-m','pip','install','--disable-pip-version-check','--no-deps','--target',str(private),'huggingface_hub==0.36.2'])
sys.path.insert(0,str(private))
import huggingface_hub
assert huggingface_hub.__version__=='0.36.2'
PAYLOAD='''+repr(payload)+'''
source=Path('/kaggle/working/trace_source_'''+package[:12]+'''')
source.mkdir(parents=True,exist_ok=True)
for name,item in PAYLOAD.items():
    assert Path(name).name==name
    body=base64.b64decode(item['base64'],validate=True)
    assert hashlib.sha256(body).hexdigest()==item['sha256']
    path=source/name
    if path.exists() and path.read_bytes()!=body:raise RuntimeError('Preserve the original extracted source and notebook: '+name)
    path.write_bytes(body)
if 'common' in sys.modules and Path(sys.modules['common'].__file__).resolve().parent!=source.resolve():
    raise RuntimeError('Restart the kernel; a different experiment source is imported. Keep all files.')
sys.path.insert(0,str(source))
print('Verified TRACE source files:',len(PAYLOAD),'; Torch/CUDA unchanged.')
'''
entries=[(21,'prepare','Preparation_And_Contract_Gate','Preparation and real HF recovery gate'),(22,'fit','Resumable_Pilot','One resumable measured pilot'),(23,'review','Completed_Review_And_Passport','Saved-evidence review and research passport'),(24,'all','All_In_One_Batch','All-in-one batch — RECOMMENDED')]
receipts=[]
for number,mode,suffix,title in entries:
    cells=[]
    def md(s):
        for a,b in {'at least19':'at least 19','a2GBreserve':'a 2 GB reserve','the20GBplan':'the 20 GB plan','assume20':'assume 20','no1TB':'no 1 TB','has71,022new':'has 71,022 new','guards6D':'guards 6D','than7rows':'than 7 rows','existing570photos':'existing 570 photos','GPU1':'GPU 1','GPU0':'GPU 0','batch8':'batch 8','batch4':'batch 4','every30minutes':'every 30 minutes','HF429':'HF 429','the25-second':'the 25-second','Roles386train/81validation/':'Roles: 386 train / 81 validation / ','103test':'103 test','within the20':'within the 20'}.items():s=s.replace(a,b)
        cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(True)))
    def code(s):ast.parse(s);cells.append(dict(cell_type='code',metadata={},source=s.splitlines(True),execution_count=None,outputs=[]))
    md(f'''# NB{number} · TREAD-TWO / TRACE — {title}

**Recommended: run NB24 once with Run All.** It automatically prepares the fixed
targets, proves storage/numerics and actual HF continuation, runs one seed-1 /
60-epoch candidate, then audits and publishes the review. NB21/NB22/NB23 are
included for individual stage use. **Do not run these four notebooks in parallel.**

| Kaggle setting | Value |
|---|---|
| Accelerator | {'None / CPU for NB23 saved-evidence review' if mode=='review' else '**GPU T4 ×2**'} |
| Internet | **ON** |
| Input | Existing expanded **shanmuk4622/tire-dataset-prepared-phase2** |
| Secret | Enable **HF_TOKEN** for **Shanmuk4622/tyre-wear-study** |
| Fresh working space | Start a dedicated session with **at least19 GB free**; live projection must also leave a2GBreserve within the20GBplan |
| Action | Fresh kernel → **Run All**, no intermediate manual handoffs |

TRACE has71,022new texture parameters. It freezes the original selected TREAD-ONE
anchor and GEO's **unselected final** geometry adapters. It predicts fixed local
image descriptors, retains spatial information and guards6D covariance when
fewer than7rows are eligible. Only the existing570photos are used. Neither
workstation nor any completed source/checkpoint is changed.

NB22 automatically satisfies preparation if needed. NB23 requires an already
completed TRACE fit and **never** starts training. Completed work restores and
skips; a rerun is not permission to create a second fit or change the recipe.

```mermaid
flowchart LR
 A[NB21 fixed targets and gates] --> B[Actual HF restore and continuation]
 B --> C[NB22 one resumable pilot]
 C --> D[NB23 CSV/native audit and passport]
 E[NB24 Run All master] --> A
```
''')
    code(setup)
    md('''## Execution, recovery and publication

One parent reads the Kaggle Secret. Separate numeric/review workers receive no
token. GPU1 builds frozen geometry/cache; GPU0 trains batch8. No DDP/batch4
covariance split. Allocator peaks, staging bytes and the complete formatted
evidence projection are checked before the long fit.

Every complete optimizer update saves model, optimizer, Python/NumPy/CPU/both
CUDA RNG states, schedule/cursor, partial epoch and committed CSV/native prefixes.
An incomplete update replays; it does not restart the epoch. Keep stopped-session
files and rerun Run All. A fresh session restores verified HF progress.

Publish direct CSV/native files every30minutes, after major stages and immediately
attempt a flush on a catchable Stop/error. Honour server backoff and one-owner
fencing. A forced kernel death cannot upload unsent work. No ZIP/NPZ metric
envelopes, alternate credential lookup or extra model sweep.
''')
    code('''# 2 · Automatic sequential stage/batch. DATA_ROOT is normally inferred from Kaggle input.
from kaggle_secrets import UserSecretsClient
from coordinator import run
DATA_ROOT=''
batch_result=run(UserSecretsClient().get_secret('HF_TOKEN'),mode='''+repr(mode)+''',data_root=DATA_ROOT)
print(json.dumps(batch_result,indent=2))
''')
    code('''# 3 · Inspect durable reports and direct evidence after automatic completion.
from IPython.display import display,FileLink,Image
base=Path('/kaggle/working/tread_two_trace')
for path in [base/'BATCH_RESULT.json',base/'review/TRACE_Evidence_Passport.html',base/'review/MODEL_CARD.md',base/'review/VALIDATION_RECONSTRUCTION.csv',base/'preparation/ENGINEERING_READY.json',base/'LAST_HF_PUBLICATION.json']:
    if path.exists():display(FileLink(str(path)))
if (base/'review/TRACE_Validation.png').exists():display(Image(filename=str(base/'review/TRACE_Validation.png')))
print('Full evidence: direct CSV/NPY/PT on HF. Existing photos only. Research status depends on original validation gates; selected may be unavailable.')
''')
    md('''## Troubleshooting and interpretation

| Situation | Action |
|---|---|
| Long preparation/restore/upload | Watch the25-second heartbeat; keep cache and completed files. |
| HF429 or quota wait | Respect displayed delay; do not start another writer. |
| Wrong input, source or environment | Stop at the explicit gate. Do not bypass checks or edit an active notebook. |
| Storage projection fails | Keep gate outputs. Long fitting is blocked; no promised data is silently dropped. |
| Session stops | Keep files; rerun this batch to restore the latest complete update. |
| Native/CSV audit mismatch | Preserve output/logs and report the traceback; do not rerun old completed models. |
| Selected unavailable | Valid scientific rejection. Final/all epoch weights and every rejection reason are retained. |

Image-derived descriptors/residuals are not physical depth, damage or calibrated
safety/confidence. Mileage labels remain proxies. Roles386train/81validation/
103test were already inspected and do not constitute a new blind evaluation.
The inherited GEO selected=None remains unchanged even if TRACE later passes
its own rule. Read the generated model card and passport before interpreting it.
''')
    for i,cell in enumerate(cells):cell['id']=f'trace{number}-{i:02d}'
    notebook=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python'),kaggle=dict(accelerator='none' if mode=='review' else 'gpu',isInternetEnabled=True),trace=dict(mode=mode,package=package,payload_files=len(payload))))
    path=ROOT/'notebooks'/f'phase3_NB{number}_TREAD_TWO_TRACE_{suffix}.ipynb'
    if path.exists():
        previous=json.loads(path.read_text(encoding='utf-8'));assert not any(c.get('outputs') or c.get('execution_count') is not None for c in previous['cells'] if c['cell_type']=='code'),'Never overwrite an executed notebook'
    path.write_text(json.dumps(notebook,indent=1)+'\n',encoding='utf-8')
    receipts.append(dict(notebook=path.name,mode=mode,sha256=sha(path.read_bytes()),bytes=path.stat().st_size,payload_files=len(payload),package=package))
(VALID/'NOTEBOOKS.json').write_text(json.dumps(receipts,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status='built',source_files=len(files),payload_files=len(payload),schema_families=48,schema_fields=schema['field_declarations'],notebooks=receipts),indent=2))
